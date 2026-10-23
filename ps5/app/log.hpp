// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <string>

namespace ps5lm {

// One app.log line: the console's local time, "ps5lm-app: " and the text, as
// "22:56:01 ps5lm-app: start" (#40). Formatted first and written with one call,
// so another thread's line cannot land between the time and the text.
inline void log_line(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
inline void log_line(const char * fmt, ...) {
    const std::time_t now = std::time(nullptr);
    std::tm tm = {};
    char when[16] = "--:--:--";
    if (localtime_r(&now, &tm)) {
        std::strftime(when, sizeof(when), "%H:%M:%S", &tm);
    }
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string text(n > 0 ? (size_t) n : 0, '\0');
    if (n > 0) {
        std::vsnprintf(&text[0], text.size() + 1, fmt, ap2);
    }
    va_end(ap2);
    std::printf("%s ps5lm-app: %s", when, text.c_str());
}

}  // namespace ps5lm
