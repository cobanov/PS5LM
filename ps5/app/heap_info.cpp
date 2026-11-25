// SPDX-License-Identifier: GPL-3.0-or-later
#include "heap_info.hpp"

#include <atomic>
#include <chrono>
#include <mutex>

#ifdef __PROSPERO__
#    include <ps5platform/heap.h>
// build-app.sh sets PS5LM_PATCHED_HEAP when it links the heap from patches/sdk:
// its ps5_heap_stats fills a larger struct than the stock header declares.
#    if defined(PS5LM_PATCHED_HEAP) && !defined(PS5_HEAP_TRIM)
#        error "the heap comes from patches/sdk but this is the stock ps5platform/heap.h"
#    endif
#endif

namespace ps5lm {
namespace {

std::atomic<uint64_t> g_ended{ 0 };
std::atomic<int>      g_in_flight{ 0 };
std::mutex            g_last_mutex;
HeapFigures           g_last;
bool                  g_measured = false;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

bool heap_can_trim() {
#if defined(__PROSPERO__) && defined(PS5_HEAP_TRIM)
    return true;
#else
    return false;
#endif
}

HeapFigures heap_figures() {
    HeapFigures f;
#ifdef __PROSPERO__
    const auto            t0 = std::chrono::steady_clock::now();
    struct ps5_heap_stats s = {};
    ps5_heap_stats(&s);
    f.walk_ms = ms_since(t0);
    f.mapped = s.mapped_bytes;
    f.peak = s.peak_bytes;
    f.segments = s.segments;
    f.arenas = s.arenas;
    f.libc_fallbacks = s.libc_fallbacks;
#    ifdef PS5_HEAP_TRIM
    f.in_use = s.in_use_bytes;
    f.has_in_use = true;
#    endif
#endif
    std::lock_guard<std::mutex> lock(g_last_mutex);
    g_last = f;
    g_measured = true;
    return f;
}

HeapFigures heap_cheap() {
    if (heap_can_trim()) {
        std::lock_guard<std::mutex> lock(g_last_mutex);
        if (g_measured) {
            return g_last;
        }
    }
    return heap_figures();  // no walk without patches/sdk; or not asked yet
}

double heap_trim() {
#if defined(__PROSPERO__) && defined(PS5_HEAP_TRIM)
    const auto t0 = std::chrono::steady_clock::now();
    ps5_heap_trim();
    return ms_since(t0);
#else
    return -1;
#endif
}

void heap_request_begun() {
    g_in_flight.fetch_add(1, std::memory_order_relaxed);
}

void heap_request_ended() {
    g_in_flight.fetch_sub(1, std::memory_order_relaxed);
    g_ended.fetch_add(1, std::memory_order_release);
}

uint64_t heap_requests_ended() {
    return g_ended.load(std::memory_order_acquire);
}

int heap_requests_in_flight() {
    return g_in_flight.load(std::memory_order_relaxed);
}

}  // namespace ps5lm
