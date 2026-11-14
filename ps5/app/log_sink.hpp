// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ps5lm {

// Prints to stdout and stderr through a memory ring, and one thread of the
// sink's own that writes them to the log file.
//
// Stock PS5LM points both streams at app.log on /data, stderr unbuffered. The
// console sometimes makes a title's write to /data wait for seconds (#41;
// thejswaggart measured 9-26 s in #69), and every thread that prints waits
// with it: the GPU driver prints with its submit lock held, the GL library
// from the render thread, llama.cpp from its log thread. That stopped the
// whole app, screen included, until the write returned (cobanov/PS5LM#70).
//
// Here printf, fprintf, vprintf, vfprintf, puts, fputs, fputc, putc, putchar
// and fwrite are wrapped at link time (--wrap, scripts/build-app.sh). Once the
// sink runs, a print to stdout or stderr copies its text into the ring and
// returns; a writer thread empties the ring into the file in large writes,
// and only it waits for storage. When the ring is full, new text is dropped
// and counted, and a line in its place says how much.
//
// Truncates `path` and points both streams at it, appending, as stock PS5LM
// does: they still carry what the wrappers do not see. False when the writer
// cannot start; prints then go to the file directly, and `why` says why.
bool log_sink_start(const std::string & path, std::string * why);

// Waits, at most `ms`, until the writer has written everything it holds.
// For a quit or a restart.
void log_sink_flush(int ms);

struct LogSinkStats {
    bool     running       = false;
    uint64_t bytes_in      = 0;  // printed to stdout and stderr
    uint64_t bytes_written = 0;  // written to the file
    uint64_t bytes_failed  = 0;  // refused by storage (disk full, I/O error)
    uint64_t bytes_dropped = 0;  // the ring was full
    uint64_t slowest_write_ms = 0;  // the longest single write to the file
};
LogSinkStats log_sink_stats();

// For the host test: the ring's size, set before log_sink_start.
void log_sink_set_ring_bytes(size_t bytes);

}  // namespace ps5lm
