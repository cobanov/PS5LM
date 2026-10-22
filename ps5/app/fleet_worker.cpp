// SPDX-License-Identifier: GPL-3.0-or-later
#include "fleet_worker.hpp"

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ps5lm {

FleetWorkerConfig read_fleet_worker_config(const char * path) {
    struct stat info = {};
    if (lstat(path, &info) != 0) {
        if (errno == ENOENT) {
            return {};  // Ordinary PS5LM until explicitly opted in.
        }
        return { FleetWorkerConfig::invalid, "", std::string("Cannot read fleet-worker.txt: ") + std::strerror(errno) };
    }
    if (!S_ISREG(info.st_mode)) {
        return { FleetWorkerConfig::invalid, "", "fleet-worker.txt must be a regular file, not a link or device" };
    }
    // Nonblocking open prevents a replaced FIFO from hanging the title before
    // the status/quit loop starts. Check the opened file, not just its pathname.
    int flags = O_RDONLY | O_NONBLOCK;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int fd = open(path, flags);
    if (fd < 0) {
        return { FleetWorkerConfig::invalid, "", std::string("Cannot open fleet-worker.txt: ") + std::strerror(errno) };
    }
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

std::string run_fleet_worker(const FleetWorkerConfig & config) {
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
        std::fprintf(stderr,
            "ps5lm-fleet: WARNING: unauthenticated, unencrypted RPC v%d.%d.%d on %s\n"
            "ps5lm-fleet: trusted isolated LAN ONLY; no public forwarding; one controller at a time\n"
            "ps5lm-fleet: exporting only %s; local model server/downloads are disabled\n",
            RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, RPC_PROTO_PATCH_VERSION,
            config.endpoint.c_str(), ggml_backend_dev_name(device));
        // No cache: avoids unbounded disk growth and HOME dependence. TCP only
        // (GGML_RPC_RDMA=OFF). The host sends the assigned weights over this socket.
        ggml_backend_rpc_start_server(config.endpoint.c_str(), nullptr, 6, 1, &device);
        return "RPC stopped or failed to bind; inspect /data/PS5LM/app.log and restart the title";
    } catch (const std::exception & ex) {
        return std::string("RPC worker failed: ") + ex.what();
    } catch (...) {
        return "RPC worker failed with an unknown exception";
    }
}

}  // namespace ps5lm
