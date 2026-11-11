// SPDX-License-Identifier: GPL-3.0-or-later
// Host test for log_sink, no console needed:
//   c++ -std=c++17 -O1 -U_FORTIFY_SOURCE -pthread ps5/app/log_sink_test.cpp ps5/app/log_sink.cpp
//       -Wl,--wrap=printf,--wrap=fprintf,--wrap=vprintf,--wrap=vfprintf,--wrap=puts,--wrap=fputs,--wrap=fputc,--wrap=putc,--wrap=putchar,--wrap=fwrite -o build/log_sink_test
//   build/log_sink_test
// The log file is a FIFO that nothing reads while the test prints, so every
// write to it blocks, as a slow /data does. Prints must still return at once;
// what the ring cannot hold is dropped and noted; what arrives is in order.
#include "log_sink.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

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

}  // namespace

int main() {
    g_report = dup(2);
    char dir[] = "/tmp/log_sink_test.XXXXXX";
    if (!mkdtemp(dir)) {
        dprintf(g_report, "FAIL mkdtemp\n");
        return 1;
    }
    const std::string fifo = std::string(dir) + "/app.log";
    if (mkfifo(fifo.c_str(), 0600) != 0) {
        dprintf(g_report, "FAIL mkfifo: %s\n", std::strerror(errno));
        return 1;
    }
    const int reader = open(fifo.c_str(), O_RDONLY | O_NONBLOCK);  // the sink's open needs a reader

    ps5lm::log_sink_set_ring_bytes(256u << 10);
    std::string why;
    const bool  on = ps5lm::log_sink_start(fifo, &why);
    check(on, ("the sink starts" + (on ? std::string() : " (" + why + ")")).c_str());
    if (!on) {
        return 1;
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
    const ps5lm::LogSinkStats mid = ps5lm::log_sink_stats();
    check(mid.bytes_dropped > 0, "a full ring drops text and counts it");

    // Storage comes back: read it all, print one last line, flush.
    std::string text = read_all(reader, 300);
    std::printf("the end|\n");
    ps5lm::log_sink_flush(2000);
    text += read_all(reader, 300);

    check(text.rfind("line 000000 ", 0) == 0, "the file starts with the first line");
    check(text.find("bytes dropped here") != std::string::npos, "the drop is noted in the file");
    check(text.size() >= 9 && text.compare(text.size() - 9, 9, "the end|\n") == 0, "the last line arrives after the drop");
    int  last    = -1;
    int  whole   = 0;
    bool ordered = true;
    for (size_t at = 0; at < text.size();) {
        const size_t nl   = text.find('\n', at);
        const size_t end  = nl == std::string::npos ? text.size() : nl;
        const std::string line = text.substr(at, end - at);
        at = end + 1;
        if (line.size() == 5 + 6 + 1 + 80 + 1 && line.rfind("line ", 0) == 0 && line.back() == '|') {
            const int i = std::atoi(line.c_str() + 5);
            ordered = ordered && i > last;
            last = i;
            whole++;
        }
    }
    dprintf(g_report, "     %d whole lines of %d arrived, %llu bytes dropped\n", whole, kLines,
            (unsigned long long) ps5lm::log_sink_stats().bytes_dropped);
    check(whole > 1000, "most of the ring's worth arrives");
    check(ordered, "lines arrive in the order printed");
    dprintf(g_report, g_failed ? "log_sink_test: %d failed\n" : "log_sink_test: all passed\n", g_failed);
    unlink(fifo.c_str());
    rmdir(dir);
    _exit(g_failed ? 1 : 0);  // the sink's threads run to the end of the process
}
