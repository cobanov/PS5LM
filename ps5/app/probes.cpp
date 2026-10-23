// SPDX-License-Identifier: GPL-3.0-or-later
#include "probes.hpp"

#include "fetch.hpp"
#include "storage.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifdef __PROSPERO__
#include <ps5platform/kernel.h>
#endif

namespace ps5lm {

// Probe: download a URL over HTTPS to fetch.out in the data folder, logging the
// status, size and speed (scripts/ps5lm-app.sh fetch <url>).
void probe_fetch(std::string url) {
    FILE * out = std::fopen(data_path("fetch.out").c_str(), "wb");
    const FetchResult r = https_get(url, "", [&](const char * p, size_t n) { return out && std::fwrite(p, 1, n, out) == n; });
    if (out) {
        std::fclose(out);
    }
    std::printf("ps5lm-app: fetch %s: status %d, length %lld, %llu bytes in %.1f s (%.2f MB/s)%s%s\n", url.c_str(),
                r.status, (long long) r.length, (unsigned long long) r.bytes, r.seconds,
                r.seconds > 0 ? r.bytes / 1e6 / r.seconds : 0.0, r.error.empty() ? "" : ", ", r.error.c_str());
}

// Probe: how fast the app can write, by method and place, fsync included
// (scripts/ps5lm-app.sh bench-write). 128 MiB each.
void probe_write() {
    const size_t total = 128u << 20;
    std::vector<char> buf(8u << 20, 'x');
    for (const std::string & dir : { data_root(), std::string("/mnt/usb0") }) {
        struct statvfs v = {};
        if (statvfs(dir.c_str(), &v) != 0) {
            continue;
        }
        const std::string path = std::string(dir) + "/write-bench.tmp";
        struct Method {
            const char * name;
            size_t       chunk;
            bool         stdio;
            int          flags;
        } methods[] = {
            { "stdio 8M", 8u << 20, true, 0 },
            { "write 1M", 1u << 20, false, 0 },
            { "write 8M", 8u << 20, false, 0 },
            { "write 8M direct", 8u << 20, false, O_DIRECT },
        };
        for (const auto & m : methods) {
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = true;
            if (m.stdio) {
                FILE * f = std::fopen(path.c_str(), "wb");
                for (size_t done = 0; f && ok && done < total; done += m.chunk) {
                    ok = std::fwrite(buf.data(), 1, m.chunk, f) == m.chunk;
                }
                ok = f && ok && std::fflush(f) == 0 && fsync(fileno(f)) == 0;
                if (f) {
                    std::fclose(f);
                }
            } else {
                const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | m.flags, 0666);
                for (size_t done = 0; fd >= 0 && ok && done < total; done += m.chunk) {
                    ok = write(fd, buf.data(), m.chunk) == (ssize_t) m.chunk;
                }
                ok = fd >= 0 && ok && fsync(fd) == 0;
                if (fd >= 0) {
                    close(fd);
                }
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("ps5lm-app: write-bench %s %s: %s, %.1f MB/s\n", dir.c_str(), m.name, ok ? "ok" : strerror(errno),
                        ok ? total / 1e6 / s : 0.0);
            unlink(path.c_str());
        }
    }
    std::printf("ps5lm-app: write-bench done\n");
}

// The read probe's pieces, on one thread or several: random offsets in the
// file, aligned to the piece, each read timed.
namespace {

struct ReadResult {
    double   seconds = 0;
    uint64_t bytes = 0;
    std::vector<double> ms;  // each read
    int      err = 0;
};

ReadResult read_random(int fd, uint64_t size, size_t piece, int threads, double seconds) {
    ReadResult total;
    std::vector<ReadResult> part((size_t) threads);
    std::vector<std::thread> pool;
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            void * buf = nullptr;
            if (posix_memalign(&buf, 1u << 16, piece) != 0) {
                part[(size_t) t].err = ENOMEM;
                return;
            }
            uint64_t x = 0x9e3779b97f4a7c15ull * (uint64_t) (t + 1);
            const uint64_t slots = size / piece;
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
                x ^= x << 13, x ^= x >> 7, x ^= x << 17;
                const auto r0 = std::chrono::steady_clock::now();
                const ssize_t got = pread(fd, buf, piece, (off_t) ((x % slots) * piece));
                if (got != (ssize_t) piece) {
                    part[(size_t) t].err = got < 0 ? errno : EIO;
                    break;
                }
                part[(size_t) t].ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count());
                part[(size_t) t].bytes += piece;
            }
            std::free(buf);
        });
    }
    for (auto & th : pool) {
        th.join();
    }
    total.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (auto & p : part) {
        total.bytes += p.bytes;
        total.ms.insert(total.ms.end(), p.ms.begin(), p.ms.end());
        total.err = total.err ? total.err : p.err;
    }
    std::sort(total.ms.begin(), total.ms.end());
    return total;
}

}  // namespace

void probe_read(std::string path) {
    struct stat st = {};
    if (stat(path.c_str(), &st) != 0 || st.st_size < (256ll << 20)) {
        std::printf("ps5lm-app: read-bench %s: %s\n", path.c_str(),
                    st.st_size ? "under 256 MiB; give it a model of several GiB" : strerror(errno));
        return;
    }
    const uint64_t size = (uint64_t) st.st_size;
    std::printf("ps5lm-app: read-bench %s, %.2f GiB (%s)\n", path.c_str(), size / 1073741824.0, drive_name(path).c_str());
    // Sequentially, 8 MiB at a time, up to 2 GiB from the start.
    {
        const int fd = open(path.c_str(), O_RDONLY);
        std::vector<char> buf(8u << 20);
        const uint64_t want = std::min<uint64_t>(size, 2ull << 30);
        uint64_t done = 0;
        const auto t0 = std::chrono::steady_clock::now();
        while (fd >= 0 && done < want) {
            const ssize_t got = read(fd, buf.data(), buf.size());
            if (got <= 0) {
                break;
            }
            done += (uint64_t) got;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("ps5lm-app: read-bench sequential 8 MiB: %.2f GiB in %.1f s, %.0f MB/s\n", done / 1073741824.0, s,
                    s > 0 ? done / 1e6 / s : 0.0);
        if (fd >= 0) {
            close(fd);
        }
    }
    // At random, as experts would be fetched: a few seconds for each piece size,
    // one thread and four, through the cache and then around it.
    for (const bool direct : { false, true }) {
        const int fd = open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
        if (fd < 0) {
            std::printf("ps5lm-app: read-bench %s: cannot open (%s)\n", direct ? "O_DIRECT" : "cached", strerror(errno));
            continue;
        }
        for (const size_t piece : { (size_t) 64 << 10, (size_t) 256 << 10, (size_t) 1 << 20, (size_t) 4 << 20 }) {
            for (const int threads : { 1, 4 }) {
                const ReadResult r = read_random(fd, size, piece, threads, 3.0);
                const size_t n = r.ms.size();
                std::printf("ps5lm-app: read-bench random %s %4zu KiB x%d: %.0f MB/s, %.0f reads/s, %.2f ms median, %.2f ms p99%s%s\n",
                            direct ? "O_DIRECT" : "cached  ", piece >> 10, threads, r.seconds > 0 ? r.bytes / 1e6 / r.seconds : 0.0,
                            r.seconds > 0 ? n / r.seconds : 0.0, n ? r.ms[n / 2] : 0.0, n ? r.ms[std::min(n - 1, n * 99 / 100)] : 0.0,
                            r.err ? ", stopped: " : "", r.err ? strerror(r.err) : "");
            }
        }
        close(fd);
    }
    std::printf("ps5lm-app: read-bench done\n");
}

void probe_memory(bool model_loaded) {
#ifdef __PROSPERO__
    const int64_t pool = sceKernelGetDirectMemorySize();
    int64_t start = 0;
    size_t largest = 0, flex = 0, flex_free = 0;
    sceKernelAvailableDirectMemorySize(0, pool, 2u << 20, &start, &largest);
    sceKernelConfiguredFlexibleMemorySize(&flex);
    sceKernelAvailableFlexibleMemorySize(&flex_free);
    std::printf("ps5lm-app: memory-probe: direct pool %.4f GiB (the system keeps %.4f GiB of a nominal 12), largest free "
                "block %.2f GiB at 0x%llx; flexible %zu MiB, %zu MiB free\n",
                pool / 1073741824.0, 12.0 - pool / 1073741824.0, largest / 1073741824.0, (unsigned long long) start,
                flex >> 20, flex_free >> 20);
    if (model_loaded) {
        std::printf("ps5lm-app: memory-probe: a model is loaded; not allocating (unload it to measure the usable block)\n");
        return;
    }
    // The largest block that really allocates and maps, in 64 MiB steps down
    // from the largest free one, given back at once.
    constexpr size_t kStep = 64u << 20;
    for (size_t len = largest / kStep * kStep; len >= kStep; len -= kStep) {
        int64_t phys = 0;
        if (sceKernelAllocateDirectMemory(0, pool, len, 2u << 20, PS5_KERNEL_DIRECT_TYPE_CPU, &phys) != 0) {
            continue;
        }
        void * addr = nullptr;
        const int mapped = sceKernelMapDirectMemory(&addr, len, PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE, 0,
                                                    phys, 2u << 20);
        if (mapped == 0) {
            static_cast<volatile char *>(addr)[0] = 1;  // touched, at both ends
            static_cast<volatile char *>(addr)[len - 1] = 1;
            sceKernelMunmap(addr, len);
        }
        sceKernelReleaseDirectMemory(phys, len);
        std::printf("ps5lm-app: memory-probe: allocated %.2f GiB in one block%s\n", len / 1073741824.0,
                    mapped == 0 ? ", mapped and touched" : ", but mapping it failed");
        return;
    }
    std::printf("ps5lm-app: memory-probe: no 64 MiB block allocates\n");
#else
    (void) model_loaded;
#endif
}

}  // namespace ps5lm
