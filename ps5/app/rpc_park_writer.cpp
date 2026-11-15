// The fleet worker's parked state files (patches/fleet/0009), written through
// ftpsrv for the reason the weight cache's are (rpc_cache_writer.cpp): the
// console slows a title's own writes to /data to about 1.2 MB/s after the first
// 2 GB (#41, docs/CONSOLE.md), and one save of a long conversation is more than
// that. A file is one transfer: begin, 8 MiB pieces, end. Only the RPC server's
// thread calls this, one file at a time.
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
bool              g_said_used = false;
Clock::time_point g_said_none{};  // when "no ftpsrv" was last logged
Clock::time_point g_began{};
size_t            g_bytes = 0;

void drop() {
    g_ftp.close();
    g_connected = false;
}

}  // namespace

// op 0 begin (path), 1 data, 2 end, 3 abort. 1 done, 0 failed, -1 no ftpsrv
// (the worker then refuses the save: parked state is never written the slow way).
extern "C" int ps5lm_rpc_park_write(int op, const char * path, const void * data, size_t size) {
    std::string error;
    switch (op) {
        case 0: {
            for (int attempt = 0; attempt < 2; ++attempt) {
                if (!g_connected) {
                    g_connected = ps5lm::connect_ftp_writer(g_ftp, error);
                    if (!g_connected) {
                        if (Clock::now() - g_said_none > std::chrono::minutes(1)) {
                            g_said_none = Clock::now();
                            ps5lm::log_line("fleet: no ftpsrv for parked state (%s): the save is refused\n", error.c_str());
                        }
                        return -1;
                    }
                }
                if (!g_said_used) {
                    g_said_used = true;
                    ps5lm::log_line("fleet: parked state is written through ftpsrv on %d\n", g_ftp.port());
                }
                if (g_ftp.begin(path, 0, error)) {
                    g_began = Clock::now();
                    g_bytes = 0;
                    return 1;
                }
                drop();  // a dropped connection: once more, on a new one
            }
            ps5lm::log_line("fleet: could not begin %s through ftpsrv: %s\n", path, error.c_str());
            return 0;
        }
        case 1:
            if (g_ftp.send(static_cast<const char *>(data), size, error)) {
                g_bytes += size;
                return 1;
            }
            g_connected = false;
            ps5lm::log_line("fleet: writing %s through ftpsrv failed: %s%s\n", path, error.c_str(),
                            ps5lm::ftp_disk_full(error) ? " (the disk is full)" : "");
            return 0;
        case 2: {
            if (!g_ftp.end(error)) {
                g_connected = false;
                ps5lm::log_line("fleet: ftpsrv did not confirm %s: %s%s\n", path, error.c_str(),
                                ps5lm::ftp_disk_full(error) ? " (the disk is full)" : "");
                return 0;
            }
            const double s = std::chrono::duration<double>(Clock::now() - g_began).count();
            if (g_bytes >= (64u << 20)) {
                ps5lm::log_line("fleet: parked %.1f MiB in %.1f s (%.0f MB/s)\n", g_bytes / 1048576.0, s,
                                s > 0 ? g_bytes / 1e6 / s : 0.0);
            }
            return 1;
        }
        default:
            drop();  // the transfer ends with the connection; the worker removes the piece
            return 1;
    }
}
