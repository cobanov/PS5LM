// Prompt cache spillover files (patches/0020), written through ftpsrv for the
// same reason as weight downloads and parked fleet states (rpc_park_writer.cpp):
// PS5 slows a title's own writes to /data to ~1.2 MB/s after the first 2 GB
// (#41, docs/CONSOLE.md). A dedicated FtpWriter + mutex avoids contending with
// downloads or fleet parking on the same socket.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#include "ftp_writer.hpp"
#include "log.hpp"

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

std::mutex        g_mu;
ps5lm::FtpWriter  g_ftp;
bool              g_connected = false;
bool              g_said_used = false;
Clock::time_point g_said_none{};
Clock::time_point g_began{};
size_t            g_bytes = 0;

void drop() {
    g_ftp.close();
    g_connected = false;
}

}  // namespace

// op 0 begin (path), 1 data, 2 end, 3 abort. 1 done, 0 failed, -1 no ftpsrv.
extern "C" int ps5lm_prompt_cache_write(int op, const char * path, const void * data, size_t size) {
    std::lock_guard<std::mutex> lock(g_mu);
    std::string error;
    switch (op) {
        case 0: {
            for (int attempt = 0; attempt < 2; ++attempt) {
                if (!g_connected) {
                    g_connected = ps5lm::connect_ftp_writer(g_ftp, error);
                    if (!g_connected) {
                        if (Clock::now() - g_said_none > std::chrono::minutes(1)) {
                            g_said_none = Clock::now();
                            ps5lm::log_line("prompt-cache: no ftpsrv (%s); using title POSIX write\n", error.c_str());
                        }
                        return -1;
                    }
                }
                if (!g_said_used) {
                    g_said_used = true;
                    ps5lm::log_line("prompt-cache: writing through ftpsrv on %d\n", g_ftp.port());
                }
                if (g_ftp.begin(path, 0, error)) {
                    g_began = Clock::now();
                    g_bytes = 0;
                    return 1;
                }
                drop();
            }
            ps5lm::log_line("prompt-cache: could not begin %s through ftpsrv: %s\n", path, error.c_str());
            return 0;
        }
        case 1:
            if (g_ftp.send(static_cast<const char *>(data), size, error)) {
                g_bytes += size;
                return 1;
            }
            drop();
            ps5lm::log_line("prompt-cache: writing %s through ftpsrv failed: %s%s\n", path, error.c_str(),
                            ps5lm::ftp_disk_full(error) ? " (the disk is full)" : "");
            return 0;
        case 2: {
            if (!g_ftp.end(error)) {
                drop();
                ps5lm::log_line("prompt-cache: ftpsrv did not confirm %s: %s%s\n", path, error.c_str(),
                                ps5lm::ftp_disk_full(error) ? " (the disk is full)" : "");
                return 0;
            }
            const double s = std::chrono::duration<double>(Clock::now() - g_began).count();
            if (g_bytes >= (16u << 20)) {
                ps5lm::log_line("prompt-cache: wrote %.1f MiB in %.2f s (%.0f MB/s)\n", g_bytes / 1048576.0, s,
                                s > 0 ? g_bytes / 1e6 / s : 0.0);
            }
            return 1;
        }
        default:
            drop();
            return 1;
    }
}
