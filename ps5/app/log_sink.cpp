// SPDX-License-Identifier: GPL-3.0-or-later
// Prints that never wait for storage (log_sink.hpp).
//
// The console's C library has no funopen, and a title may not dup2 a pipe
// over its stdout or stderr (EPERM, measured 2026-11-11). So the print
// functions themselves are wrapped at link time (scripts/build-app.sh passes
// --wrap for each), the way the title heap wraps malloc: a print to stdout or
// stderr from anything linked into the title (the app, llama.cpp, ggml, the
// GPU driver, the GL library, libc++) lands in a memory ring here, and a
// thread of the sink's own writes the ring to the file. Prints to any other
// stream, and every print before the sink starts, go to the real function.
#include "log_sink.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C" {
int    __real_vfprintf(FILE * f, const char * fmt, va_list ap);
int    __real_fputs(const char * s, FILE * f);
int    __real_puts(const char * s);
int    __real_fputc(int c, FILE * f);
int    __real_putc(int c, FILE * f);
int    __real_putchar(int c);
size_t __real_fwrite(const void * p, size_t size, size_t n, FILE * f);
}

namespace ps5lm {

namespace {

using Clock = std::chrono::steady_clock;

struct Sink {
    std::mutex              mtx;
    std::condition_variable has_text;  // the writer waits on it
    std::condition_variable written;   // log_sink_flush waits on it
    std::vector<char>       ring;
    size_t                  head = 0;  // the oldest byte held
    size_t                  held = 0;
    bool                    writing = false;  // the writer holds a chunk outside the lock
    uint64_t                unreported = 0;   // dropped and not yet noted in the file
    LogSinkStats            stats;
    int                     out_fd = -1;
};

Sink              g;
std::atomic<bool> g_active{ false };
size_t            g_ring_bytes = 2u << 20;

bool ours(FILE * f) {
    return g_active.load(std::memory_order_acquire) && (f == stdout || f == stderr);
}

// Appends to the ring; the caller holds the lock and checked the room.
void put_locked(const char * p, size_t n) {
    const size_t cap  = g.ring.size();
    size_t       tail = (g.head + g.held) % cap;
    while (n > 0) {
        const size_t run = std::min(n, cap - tail);
        std::memcpy(g.ring.data() + tail, p, run);
        p += run;
        n -= run;
        g.held += run;
        tail = (tail + run) % cap;
    }
}

// One line where text went missing, in its place in the file.
void note_drops_locked(size_t room_needed_after) {
    if (g.unreported == 0) {
        return;
    }
    char      note[96];
    const int len = std::snprintf(note, sizeof(note), "\n[log] %llu bytes dropped here: the log file fell behind\n",
                                  (unsigned long long) g.unreported);
    if (len > 0 && g.ring.size() - g.held >= (size_t) len + room_needed_after) {
        put_locked(note, (size_t) len);
        g.unreported = 0;
    }
}

// Text into the ring. Waits only for the ring's lock, held for a copy.
void push(const char * p, size_t n) {
    if (n == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g.mtx);
    g.stats.bytes_in += n;
    note_drops_locked(n);
    const size_t take = std::min(n, g.ring.size() - g.held);
    put_locked(p, take);
    if (take < n) {
        g.stats.bytes_dropped += n - take;
        g.unreported += n - take;
    }
    g.has_text.notify_one();
}

int push_vformat(const char * fmt, va_list ap) {
    char    small[512];
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(small, sizeof(small), fmt, ap);
    if (n < 0) {
        va_end(ap2);
        return n;
    }
    if ((size_t) n < sizeof(small)) {
        push(small, (size_t) n);
    } else {
        std::string big((size_t) n + 1, '\0');
        std::vsnprintf(&big[0], big.size(), fmt, ap2);
        push(big.data(), (size_t) n);
    }
    va_end(ap2);
    return n;
}

void write_all(int fd, const char * p, size_t n) {
    while (n > 0) {
        const ssize_t w = write(fd, p, n);
        if (w < 0 && errno == EINTR) {
            continue;
        }
        if (w <= 0) {
            return;  // storage refused it: the text is lost, the app goes on
        }
        p += w;
        n -= (size_t) w;
    }
}

// Ring to file, in large writes. The only thread that waits for storage.
void writer() {
    std::vector<char>            chunk(1u << 20);
    std::unique_lock<std::mutex> lock(g.mtx);
    for (;;) {
        g.has_text.wait(lock, [] { return g.held > 0; });
        const size_t n   = std::min(g.held, chunk.size());
        const size_t cap = g.ring.size();
        const size_t run = std::min(n, cap - g.head);
        std::memcpy(chunk.data(), g.ring.data() + g.head, run);
        std::memcpy(chunk.data() + run, g.ring.data(), n - run);
        g.head = (g.head + n) % cap;
        g.held -= n;
        g.writing = true;
        lock.unlock();

        const Clock::time_point t0 = Clock::now();
        write_all(g.out_fd, chunk.data(), n);
        const uint64_t ms =
            (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();

        lock.lock();
        g.writing = false;
        g.stats.bytes_written += n;
        g.stats.slowest_write_ms = std::max(g.stats.slowest_write_ms, ms);
        if (g.held == 0) {
            g.written.notify_all();
        }
    }
}

}  // namespace

void log_sink_set_ring_bytes(size_t bytes) {
    g_ring_bytes = std::max<size_t>(bytes, 4096);
}

bool log_sink_start(const std::string & path, std::string * why) {
    // Stock PS5LM's streams first: both append to the file, stderr unbuffered.
    // They still carry whatever the wrappers do not see (the C library's own
    // prints), and everything if the sink cannot start.
    if (std::FILE * f = std::fopen(path.c_str(), "w")) {
        std::fclose(f);
    }
    if (std::freopen(path.c_str(), "a", stdout) != nullptr) {
        setvbuf(stdout, nullptr, _IOLBF, 0);
    }
    if (std::freopen(path.c_str(), "a", stderr) != nullptr) {
        setvbuf(stderr, nullptr, _IONBF, 0);
    }
    const int out = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (out < 0) {
        if (why) {
            *why = std::string("open: ") + std::strerror(errno);
        }
        return false;
    }
    g.ring.assign(g_ring_bytes, '\0');
    g.out_fd = out;
    try {
        std::thread(writer).detach();
    } catch (...) {
        close(out);
        if (why) {
            *why = "no thread for the writer";
        }
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g.mtx);
        g.stats.running = true;
    }
    std::fflush(stdout);
    g_active.store(true, std::memory_order_release);
    return true;
}

void log_sink_flush(int ms) {
    if (!g_active.load(std::memory_order_acquire)) {
        std::fflush(stdout);
        std::fflush(stderr);
        return;
    }
    const Clock::time_point      deadline = Clock::now() + std::chrono::milliseconds(ms);
    std::unique_lock<std::mutex> lock(g.mtx);
    note_drops_locked(0);
    if (g.held > 0) {
        g.has_text.notify_one();
    }
    g.written.wait_until(lock, deadline, [] { return g.held == 0 && !g.writing; });
}

LogSinkStats log_sink_stats() {
    std::lock_guard<std::mutex> lock(g.mtx);
    return g.stats;
}

}  // namespace ps5lm

// The wrapped print functions (scripts/build-app.sh: --wrap=<name> for each).
extern "C" {

int __wrap_vfprintf(FILE * f, const char * fmt, va_list ap) {
    return ps5lm::ours(f) ? ps5lm::push_vformat(fmt, ap) : __real_vfprintf(f, fmt, ap);
}

int __wrap_fprintf(FILE * f, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int n = __wrap_vfprintf(f, fmt, ap);
    va_end(ap);
    return n;
}

int __wrap_vprintf(const char * fmt, va_list ap) {
    return __wrap_vfprintf(stdout, fmt, ap);
}

int __wrap_printf(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int n = __wrap_vfprintf(stdout, fmt, ap);
    va_end(ap);
    return n;
}

int __wrap_fputs(const char * s, FILE * f) {
    if (!ps5lm::ours(f)) {
        return __real_fputs(s, f);
    }
    ps5lm::push(s, std::strlen(s));
    return 1;
}

int __wrap_puts(const char * s) {
    if (!ps5lm::ours(stdout)) {
        return __real_puts(s);
    }
    ps5lm::push(s, std::strlen(s));
    ps5lm::push("\n", 1);
    return 1;
}

int __wrap_fputc(int c, FILE * f) {
    if (!ps5lm::ours(f)) {
        return __real_fputc(c, f);
    }
    const char ch = (char) c;
    ps5lm::push(&ch, 1);
    return (unsigned char) c;
}

int __wrap_putc(int c, FILE * f) {
    if (!ps5lm::ours(f)) {
        return __real_putc(c, f);
    }
    return __wrap_fputc(c, f);
}

int __wrap_putchar(int c) {
    if (!ps5lm::ours(stdout)) {
        return __real_putchar(c);
    }
    return __wrap_fputc(c, stdout);
}

size_t __wrap_fwrite(const void * p, size_t size, size_t n, FILE * f) {
    if (!ps5lm::ours(f)) {
        return __real_fwrite(p, size, n, f);
    }
    ps5lm::push(static_cast<const char *>(p), size * n);
    return n;
}

}  // extern "C"
