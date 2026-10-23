// Measurements the app makes only when the PC asks (scripts/ps5lm-app.sh),
// each on a thread of its own, with its results in app.log. None of them is
// part of serving a model.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <string>

namespace ps5lm {

// Downloads a URL over HTTPS to fetch.out in the data folder: status, size, speed.
void probe_fetch(std::string url);

// Write speed by method (stdio, write, O_DIRECT) to the data folder and a USB drive.
void probe_write();

// Read speed of a file, the way streaming MoE experts from a drive would read
// it: sequentially, then at random offsets in pieces of 64 KiB to 4 MiB with
// one and four threads, through the cache and with O_DIRECT. Point it at a
// model of several GiB, so the cache can hold little of it.
void probe_read(std::string path);

// The title's memory as the system gave it: the direct pool and its largest
// free block, the flexible pool, and the largest block the app can allocate
// and map now, given back at once. `model_loaded`: skip that last part, so it
// never takes memory from a running model.
void probe_memory(bool model_loaded);

}  // namespace ps5lm
