# Title heap patches

Four later commits of the payload SDK fork PS5LM pins (`mihawk-99/PS5_PayloadSDK` at `BASE`), by its author,
path-filtered to the title heap (`platform/src/heap.c`, `platform/src/dlmalloc/mspace.c`,
`platform/include/ps5platform/heap.h`) and its host tests. Each patch's message names the commit it comes from and
what of it is left out. They are carried here rather than moving the pin, which would bring in many unrelated
commits.

| Patch | SDK commit | What it does |
| --- | --- | --- |
| 0001 | `8278ca5e` | calloc clears a mapped chunk: the console hands released direct memory out with its old bytes (heap part) |
| 0002 | `3752be7e` | arena locks block instead of spinning; arena set-up sleeps while it waits (heap part) |
| 0003 | `9d048784` | the heap grows in 2 MiB segments instead of 16 MiB; a segment table for 8,192 |
| 0004 | `55a2de04` | a 64 KiB gap between mappings so dlmalloc never joins segments, and a segment with nothing left in it goes back wherever it lies; `ps5_heap_trim()`; the heap's in-use figure; 32 MiB trim threshold |

At the pin the heap maps its segments back to back and dlmalloc joins neighbours into one segment, which can only
go back to the pool whole: one block left near its end keeps everything below it. With these, every empty segment
can go back, and the app calls `ps5_heap_trim()` 0.25 s after each request and every 10 s (`ps5/app/heap_info.cpp`).

`scripts/build-app.sh` exports `BASE` from the pinned checkout (`git archive`), applies these with
`patch --fuzz=0`, compiles `heap.o` and `mspace.o` with the SDK's own Makefile, and links them as
`libps5lm_heap.a` ahead of `libps5platform.a`; lld's `--why-extract` report is checked so the stock members never
come in. The pinned checkout and the SDK cache built from it are not modified, so `scripts/test-native-build.sh`'s
pin checks hold, and that script runs the SDK's host tests (`make -C platform test`) on the patched copy first.

Delete this folder once PS5LM pins the SDK at or past `55a2de04`.
