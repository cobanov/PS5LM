// SPDX-License-Identifier: GPL-3.0-or-later
// Host test for log_sink, no console needed:
//   c++ -std=c++17 -O1 -U_FORTIFY_SOURCE -pthread ps5/app/log_sink_test.cpp ps5/app/log_sink.cpp
//       -Wl,--wrap=printf,--wrap=fprintf,--wrap=vprintf,--wrap=vfprintf,--wrap=puts,--wrap=fputs,--wrap=fputc,--wrap=putc,--wrap=putchar,--wrap=fwrite -o build/log_sink_test
//   build/log_sink_test
// Each case runs in a child process of its own, with a sink of its own.
// The log file is a FIFO the test reads when it chooses, so a write to it
// blocks as a slow /data does, or /dev/full, which refuses every write.
#include "log_sink.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

int g_report = 2;  // the test's own output, past the sink
int g_failed = 0;

void check(bool ok, const char * what) {
    dprintf(g_report, "%s %s\n", ok ? "ok  " : "FAIL", what);
    g_failed += ok ? 0 : 1;
}

// Everything the FIFO yields until it has been quiet for `quiet_ms`.
std::string read_all(int fd, int quiet_ms) {
    std::string out;
    char        buf[65536];
    for (;;) {
        pollfd p = { fd, POLLIN, 0 };
        if (poll(&p, 1, quiet_ms) <= 0) {
            return out;
        }
        const ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0) {
            return out;
        }
        out.append(buf, (size_t) n);
    }
}

// A FIFO in a fresh folder, with its read end open (the sink's open needs one).
struct Fifo {
    std::string dir, path;
    int         reader = -1;
    Fifo() {
        char d[] = "/tmp/log_sink_test.XXXXXX";
        dir  = mkdtemp(d) ? d : "";
        path = dir + "/app.log";
        mkfifo(path.c_str(), 0600);
        reader = open(path.c_str(), O_RDONLY | O_NONBLOCK);
    }
    ~Fifo() {
        unlink(path.c_str());
        rmdir(dir.c_str());
    }
};

// Reads the FIFO on a thread of its own from `start_ms` on, until stopped.
struct Drain {
    std::atomic<bool> stop{ false };
    std::string       text;
    std::thread       thread;
    Drain(int fd, int start_ms) {
        thread = std::thread([this, fd, start_ms] {
            std::this_thread::sleep_for(std::chrono::milliseconds(start_ms));
            while (!stop.load()) {
                text += read_all(fd, 50);
            }
            text += read_all(fd, 200);
        });
    }
    std::string finish() {
        stop = true;
        thread.join();
        return text;
    }
};

bool start(const std::string & path, size_t ring) {
    ps5lm::log_sink_set_ring_bytes(ring);
    std::string why;
    const bool  on = ps5lm::log_sink_start(path, &why);
    check(on, ("the sink starts" + (on ? std::string() : " (" + why + ")")).c_str());
    return on;
}

// Storage stops taking writes: prints still return at once, a full ring
// drops text and says so in place, what arrives is in order.
void blocked_file() {
    Fifo fifo;
    if (!start(fifo.path, 256u << 10)) {
        return;
    }
    // 3 MB of numbered lines while nothing reads the file: the FIFO fills
    // (64 KiB), the writer blocks in write(), then the 256 KiB ring fills.
    constexpr int kLines = 30000;
    double        slowest_ms = 0;
    for (int i = 0; i < kLines; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        if (i % 2) {
            std::fprintf(stderr, "line %06d %s|\n", i, std::string(80, 'e').c_str());
        } else {
            std::printf("line %06d %s|\n", i, std::string(80, 'o').c_str());
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        slowest_ms = ms > slowest_ms ? ms : slowest_ms;
    }
    dprintf(g_report, "     slowest print %.2f ms\n", slowest_ms);
    check(slowest_ms < 100.0, "no print waited for the blocked file");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check(ps5lm::log_sink_stats().bytes_dropped > 0, "a full ring drops text and counts it");

    // Storage comes back: read it all, print one last line, flush.
    std::string text = read_all(fifo.reader, 300);
    std::printf("the end|\n");
    ps5lm::log_sink_flush(2000);
    text += read_all(fifo.reader, 300);

    check(text.rfind("line 000000 ", 0) == 0, "the file starts with the first line");
    check(text.find("bytes dropped here") != std::string::npos, "the drop is noted in the file");
    check(text.size() >= 9 && text.compare(text.size() - 9, 9, "the end|\n") == 0, "the last line arrives after the drop");
    int  last = -1, whole = 0;
    bool ordered = true;
    for (size_t at = 0; at < text.size();) {
        const size_t      nl   = text.find('\n', at);
        const size_t      end  = nl == std::string::npos ? text.size() : nl;
        const std::string line = text.substr(at, end - at);
        at = end + 1;
        if (line.size() == 5 + 6 + 1 + 80 + 1 && line.rfind("line ", 0) == 0 && line.back() == '|') {
            const int i = std::atoi(line.c_str() + 5);
            ordered = ordered && i > last;
            last = i;
            whole++;
        }
    }
    dprintf(g_report, "     %d whole lines of %d arrived\n", whole, kLines);
    check(whole > 1000, "most of the ring's worth arrives");
    check(ordered, "lines arrive in the order printed");
}

// Eight threads print with puts at once: each line arrives whole (the text
// and its newline go into the ring together).
void threads_puts() {
    Fifo fifo;
    if (!start(fifo.path, 8u << 20)) {
        return;
    }
    Drain                    drain(fifo.reader, 0);
    constexpr int            kThreads = 8, kEach = 2000;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; t++) {
        threads.emplace_back([t] {
            char line[64];
            for (int i = 0; i < kEach; i++) {
                std::snprintf(line, sizeof(line), "T%d %06d %s", t, i, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
                std::puts(line);
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }
    ps5lm::log_sink_flush(3000);
    const std::string text = drain.finish();
    int               whole = 0, broken = 0;
    for (size_t at = 0; at < text.size();) {
        const size_t      nl   = text.find('\n', at);
        const size_t      end  = nl == std::string::npos ? text.size() : nl;
        const std::string line = text.substr(at, end - at);
        at = end + 1;
        const bool ok = line.size() == 2 + 1 + 6 + 1 + 40 && line[0] == 'T' && line[2] == ' ' && line[9] == ' ' &&
                        line.find_first_not_of('x', 10) == std::string::npos;
        whole += ok;
        broken += !ok;
    }
    dprintf(g_report, "     %d whole lines, %d broken\n", whole, broken);
    check(ps5lm::log_sink_stats().bytes_dropped == 0, "nothing dropped with storage keeping up");
    check(whole == kThreads * kEach && broken == 0, "every puts line arrives whole");
}

// The ring is full when the flush starts and nothing is printed after: the
// flush still waits for the drop note, and the note reaches the file.
void flush_notes_drops() {
    Fifo fifo;
    if (!start(fifo.path, 64u << 10)) {
        return;
    }
    for (int i = 0; i < 6000; i++) {
        std::printf("fill %06d %s|\n", i, std::string(60, 'f').c_str());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check(ps5lm::log_sink_stats().bytes_dropped > 0, "the ring filled and dropped text");
    Drain drain(fifo.reader, 200);  // storage comes back during the flush
    ps5lm::log_sink_flush(3000);
    const std::string text = drain.finish();
    // Every dropped byte is accounted for by a note in the file, the last
    // drop's included (it happened with the ring full and nothing after it).
    unsigned long long noted = 0;
    for (size_t at = text.find("[log] "); at != std::string::npos; at = text.find("[log] ", at + 1)) {
        noted += std::strtoull(text.c_str() + at + 6, nullptr, 10);
    }
    const unsigned long long dropped = ps5lm::log_sink_stats().bytes_dropped;
    dprintf(g_report, "     dropped %llu, noted in the file %llu\n", dropped, noted);
    check(noted == dropped, "the flush waits for the last drop note, and every drop is noted");
}

// Storage refuses every write: what it refused is not counted as written.
void storage_refuses() {
    if (access("/dev/full", W_OK) != 0) {
        dprintf(g_report, "     no /dev/full here, skipped\n");
        return;
    }
    if (!start("/dev/full", 64u << 10)) {
        return;
    }
    std::printf("one line the disk will not take\n");
    ps5lm::log_sink_flush(2000);
    const ps5lm::LogSinkStats s = ps5lm::log_sink_stats();
    dprintf(g_report, "     written %llu, refused %llu\n", (unsigned long long) s.bytes_written,
            (unsigned long long) s.bytes_failed);
    check(s.bytes_written == 0 && s.bytes_failed > 0, "bytes storage refused are counted as refused, not written");
}

}  // namespace

int main() {
    const int report = dup(2);
    const struct {
        const char * name;
        void (*run)();
    } cases[] = { { "blocked file", blocked_file },
                  { "eight threads with puts", threads_puts },
                  { "flush with a full ring", flush_notes_drops },
                  { "storage refuses", storage_refuses } };
    int failed = 0;
    for (const auto & c : cases) {
        dprintf(report, "-- %s\n", c.name);
        const pid_t pid = fork();
        if (pid == 0) {
            g_report = report;
            c.run();
            _exit(g_failed ? 1 : 0);  // the sink's threads run to the end of the process
        }
        int status = 0;
        waitpid(pid, &status, 0);
        failed += !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    dprintf(report, failed ? "log_sink_test: %d case(s) failed\n" : "log_sink_test: all passed\n", failed);
    return failed ? 1 : 0;
}
