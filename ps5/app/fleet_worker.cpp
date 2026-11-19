// SPDX-License-Identifier: GPL-3.0-or-later
#include "fleet_worker.hpp"

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>

namespace ps5lm {

namespace {

bool hex16(const std::string & s, size_t at) {
    return s.size() >= at + 16 && std::all_of(s.begin() + at, s.begin() + at + 16, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

// The names llama.cpp's RPC cache writes (patches/fleet/0005): a tensor's FNV-1a
// hash as 16 hex digits, and <hash>.<16 hex digits>.part while one is written.
bool cache_file_name(const std::string & name) {
    return name.size() == 16 && hex16(name, 0);
}

bool part_file_name(const std::string & name) {
    return name.size() == 38 && hex16(name, 0) && name[16] == '.' && hex16(name, 17) && name.compare(33, 5, ".part") == 0;
}

}  // namespace

RpcCacheUse prune_rpc_cache(const RpcCache & cache, bool at_start) {
    namespace fs = std::filesystem;
    RpcCacheUse use;
    if (cache.dir.empty()) {
        return use;
    }
    std::error_code ec;
    if (at_start) {
        fs::create_directories(cache.dir, ec);
    }
    struct File {
        fs::file_time_type time;
        uint64_t           size;
        fs::path           path;
    };
    std::vector<File> files, parts;
    for (fs::directory_iterator it(cache.dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        const bool part = part_file_name(name);
        std::error_code e;
        if ((!part && !cache_file_name(name)) || !it->is_regular_file(e)) {
            continue;
        }
        const uint64_t size = it->file_size(e);
        const fs::file_time_type time = e ? fs::file_time_type{} : it->last_write_time(e);
        if (!e) {
            (part ? parts : files).push_back({ time, size, it->path() });
        }
    }
    // A .part is a write in progress, or one cut off: only before the worker runs is it the latter.
    for (const File & f : parts) {
        std::error_code e;
        if (at_start && fs::remove(f.path, e)) {
            ++use.dropped;
            use.dropped_bytes += f.size;
        }
    }
    std::sort(files.begin(), files.end(), [](const File & a, const File & b) { return a.time < b.time; });
    use.files = files.size();
    for (const File & f : files) {
        use.bytes += f.size;
    }
    // Oldest first: patches/fleet/0005 gives a file the time of its last use.
    for (const File & f : files) {
        if (use.bytes <= cache.limit) {
            break;
        }
        std::error_code e;
        if (fs::remove(f.path, e)) {
            use.bytes -= f.size;
            --use.files;
            ++use.dropped;
            use.dropped_bytes += f.size;
        }
    }
    return use;
}

FleetWorkerConfig read_fleet_worker_config(const char * path) {
    // No lstat: a PS5 title gets EPERM from it (seen on a console). Opening with
    // O_NOFOLLOW refuses a link (ELOOP), O_NONBLOCK keeps a FIFO from hanging the
    // title before the status/quit loop starts, and fstat checks the opened file.
    int flags = O_RDONLY | O_NONBLOCK;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int fd = open(path, flags);
    if (fd < 0) {
        if (errno == ENOENT) {
            return {};  // Ordinary PS5LM until explicitly opted in.
        }
        if (errno == ELOOP) {
            return { FleetWorkerConfig::invalid, "", "fleet-worker.txt must be a regular file, not a link or device" };
        }
        return { FleetWorkerConfig::invalid, "", std::string("Cannot open fleet-worker.txt: ") + std::strerror(errno) };
    }
    struct stat info = {};
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
        close(fd);
        return { FleetWorkerConfig::invalid, "", "fleet-worker.txt changed or is not a regular file" };
    }
    char data[1025];
    size_t size = 0;
    bool failed = false;
    while (size < sizeof(data)) {
        const ssize_t n = read(fd, data + size, sizeof(data) - size);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            failed = n < 0;
            break;
        }
        size += static_cast<size_t>(n);
    }
    close(fd);
    if (failed || size == sizeof(data)) {
        return { FleetWorkerConfig::invalid, "", "fleet-worker.txt is unreadable or exceeds 1024 bytes" };
    }
    return parse_fleet_worker_config(std::string(data, size));
}

std::string run_fleet_worker(const FleetWorkerConfig & config, const RpcCache & cache) {
    // Revalidate at the listener boundary, even if a caller bypassed the file reader.
    if (config.state != FleetWorkerConfig::enabled ||
        parse_fleet_worker_config(config.endpoint + "\nI_UNDERSTAND_RPC_IS_UNAUTHENTICATED\n").state != FleetWorkerConfig::enabled) {
        return "Refusing invalid fleet worker configuration";
    }
    try {
        // Layer splitting does not need peer collectives. The matching server
        // patch rejects COMM_INIT, which would otherwise open another wildcard
        // listener at a port chosen by the controller. Override env.txt too.
        if (setenv("GGML_RPC_NO_COMM", "1", 1) != 0) {
            return "Cannot disable extra RPC listeners; refusing to start";
        }
        // The title's stderr does not reach app.log; ggml's log (RPC and Vulkan
        // errors included) goes to stdout, which does.
        ggml_log_set([](ggml_log_level, const char * text, void *) { std::fputs(text, stdout); }, nullptr);
        ggml_backend_load_all();
        ggml_backend_reg_t vulkan = ggml_backend_reg_by_name("Vulkan");
        if (!vulkan) {
            return "No Vulkan backend; refusing CPU fallback";
        }
        ggml_backend_dev_t device = nullptr;
        for (size_t i = 0; i < ggml_backend_reg_dev_count(vulkan); ++i) {
            auto candidate = ggml_backend_reg_dev_get(vulkan, i);
            const auto type = ggml_backend_dev_type(candidate);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                if (device) {
                    return "Expected exactly one PS5 Vulkan device";
                }
                device = candidate;
            }
        }
        if (!device) {
            return "No Vulkan GPU device; refusing CPU fallback";
        }
        std::fprintf(stdout,
            "ps5lm-fleet: WARNING: unauthenticated, unencrypted RPC v%d.%d.%d on %s\n"
            "ps5lm-fleet: trusted isolated LAN ONLY; no public forwarding; one controller at a time\n"
            "ps5lm-fleet: exporting only %s; local model server/downloads are disabled\n",
            RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, RPC_PROTO_PATCH_VERSION,
            config.endpoint.c_str(), ggml_backend_dev_name(device));
        // TCP only (GGML_RPC_RDMA=OFF). The host sends the assigned weights over
        // this socket, or, with a cache, the hashes of those the console kept:
        // in a fixed folder (no HOME), and kept to its size by prune_rpc_cache.
        ggml_backend_rpc_start_server(config.endpoint.c_str(), cache.dir.empty() ? nullptr : cache.dir.c_str(),
                                      6, 1, &device);
        return "RPC stopped or failed to bind; inspect /data/PS5LM/app.log and restart the title";
    } catch (const std::exception & ex) {
        return std::string("RPC worker failed: ") + ex.what();
    } catch (...) {
        return "RPC worker failed with an unknown exception";
    }
}

}  // namespace ps5lm
