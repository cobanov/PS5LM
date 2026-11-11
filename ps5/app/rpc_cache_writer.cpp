// The fleet worker's weight cache files (patches/fleet/0005), written
// through ftpsrv. The console slows a title's own writes to /data to about
// 1.2 MB/s after the first 2 GB (#41, docs/CONSOLE.md): the first load of a
// model through the cache would have taken about half an hour that way. ftpsrv, a payload,
// writes at 80 MB/s and more, as it does for downloads (ftp_writer.hpp).
// Only the RPC server's thread calls this, one tensor at a time.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#include "ftp_writer.hpp"
#include "log.hpp"

#include <chrono>
#include <cstddef>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

ps5lm::FtpWriter  g_ftp;
bool              g_connected = false;
bool              g_full      = false;  // the disk is full: nothing more is kept this run
bool              g_said_used = false, g_said_none = false;
Clock::time_point g_retry{};            // no ftpsrv answered: when to look again

bool connect(std::string & error) {
    if (g_connected) {
        return true;
    }
    if (Clock::now() < g_retry) {
        error = "none answered a moment ago";
        return false;
    }
    g_connected = ps5lm::connect_ftp_writer(g_ftp, error);
    if (!g_connected) {
        g_retry = Clock::now() + std::chrono::minutes(1);
    }
    return g_connected;
}

}  // namespace

// 1 written, 0 failed, -1 not kept this time (said once, here).
extern "C" int ps5lm_rpc_cache_write(const char * path, const void * data, size_t size) {
    if (g_full) {
        return -1;
    }
    std::string error;
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!connect(error)) {
            if (!g_said_none) {
                g_said_none = true;
                ps5lm::log_line("fleet: no ftpsrv for the weight cache (%s), so it keeps nothing for now: "
                                "the app's own writes slow to about 1.2 MB/s after 2 GB\n", error.c_str());
            }
            return -1;
        }
        if (!g_said_used) {
            g_said_used = true;
            ps5lm::log_line("fleet: the weight cache writes through ftpsrv on %d\n", g_ftp.port());
        }
        if (g_ftp.write_at(path, 0, static_cast<const char *>(data), size, error)) {
            return 1;
        }
        // A dropped connection: once more, on a new one. Not for a full disk.
        g_ftp.close();
        g_connected = false;
        if (ps5lm::ftp_disk_full(error)) {
            g_full = true;
            ps5lm::log_line("fleet: the disk is full, so the weight cache keeps nothing more this run (%s)\n",
                            error.c_str());
            return -1;
        }
    }
    ps5lm::log_line("fleet: could not write %s through ftpsrv: %s\n", path, error.c_str());
    return 0;
}
