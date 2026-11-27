// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// The title heap (the payload SDK's ps5platform/heap.h) as the app
// sees it, and its trim. Only heap_info.cpp includes the SDK's header, so the
// struct the heap fills is always the one it was built with: patches/sdk adds
// a field to it.
//
// With patches/sdk the heap maps its segments apart, so a segment nothing is
// left in can go back to the pool the GPU shares, and ps5_heap_trim does that.
// Without them (a stock SDK) there is no trim, and the figures have no in-use.

#include <cstddef>
#include <cstdint>

namespace ps5lm {

struct HeapFigures {
    size_t             mapped = 0;          // direct memory mapped for the heap
    size_t             peak = 0;            // the most it has held
    size_t             in_use = 0;          // in blocks handed out (has_in_use only)
    bool               has_in_use = false;  // this heap can say (patches/sdk)
    unsigned           segments = 0;
    unsigned           arenas = 0;
    unsigned long long libc_fallbacks = 0;  // allocations libc served instead
    double             walk_ms = 0;         // what asking took
};

// Whether this build's heap can hand its empty segments back (patches/sdk).
bool heap_can_trim();

// Asks the heap now. With patches/sdk the in-use figure walks every block,
// one arena's lock at a time, so the main loop asks at trims and memory lines,
// not the stats thread once a second.
HeapFigures heap_figures();

// Figures without walking the blocks: the heap's own when that is cheap (a
// stock SDK), else those of the last heap_figures().
HeapFigures heap_cheap();

// Hands each arena's empty segments back to the pool. Milliseconds it took, or
// -1 when this heap cannot.
double heap_trim();

// llama-server's POST requests, from the hooks in activity.cpp: the trim
// follows each request, and the memory line says how many are in flight.
void     heap_request_begun();
void     heap_request_ended();
uint64_t heap_requests_ended();
int      heap_requests_in_flight();


}  // namespace ps5lm
