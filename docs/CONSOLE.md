# Console facts

Measured on the project's test console. Payload numbers come from `probes/memprobe` and `probes/threadprobe`, run through shsrv.

## Setup

- PS5 Slim (CFI-2015), firmware 13.60
- Relapse through WebKit Autoloader v0.5.2 and Payload Manager (PLK)
- Autoload: kstuff-lite 1.11, ShadowMountPlus 1.7beta2, etaHEN 2.5B, ftpsrv 0.21.1, shsrv 0.20, klogsrv 0.9
- FTP upload from a Mac on the same network: about 61 MB/s

The Relapse kernel stage sometimes hangs and the console then shuts down, or sticks on "preparing to turn off" until the power button is held. That is the exploit's race, not anything PS5LM runs.

## Memory, for one payload process

| What | Value |
|---|---|
| Flexible memory configured / free | 8.00 / 7.97 GiB |
| malloc ceiling (256 MiB steps, pages touched) | 6.25 GiB |
| Largest single block (`posix_memalign`) | 6.00 GiB |
| Direct memory | none: size 0, allocation fails with `0x80020023` |

The flexible pool is shared with the system: holding about 6 GiB froze the home screen until the process ended. Plan on 4 to 5 GiB for model, context and buffers in a payload. Bigger models (Qwen 3.8 27B) need a process with direct memory, that is a native app.

Why a payload has no direct memory: elfldr starts every payload as a new process with budget 0 (`sys_budget_set(0)` before it executes SceSpZeroConf, `elfldr.c` in ps5-payload-dev/elfldr), the budget of a system app. `0x80020023` is `0x80020000 + EAGAIN`: the pool is empty, the call is not refused. etaHEN's jailbreak request changes a process's rights (uid, sandbox, authid), not its budget, and kstuff patches no budgets; every homebrew project using more than 6 GiB is a native app.

## CPU, for one payload process

- AVX, AVX2, FMA, F16C, BMI2; no AVX-512. 16 logical CPUs online.
- **Affinity mask `0xea00`: only CPUs 9, 11, 13, 14 and 15**, shared with system services. Sony priority 700 (normal) for the main thread and new threads.
- 1, 2 and 4 threads busy for a second each: fine.
- Busy-waiting on all five (llama.cpp's defaults: one thread per physical core, `--poll 50`) starved the system: services stopped answering and the console shut down twice. `patches/0001-ps5-safe-cpu-defaults.patch` makes the default thread count the allowed CPUs minus two and turns polling off.
- `--poll 0` only ends the spinning between graphs. ggml's barrier spins at every operation inside a graph whatever the poll value, so during generation every worker runs flat out: the thread count is the real load setting.
- Only CPUs 14 and 15 are the two halves of one core (SMT siblings are 2k and 2k+1, as PS5SX2 and PS5CEMU-HAR lay them out). 9, 11 and 13 share their cores with threads outside the payload.
- llama.cpp's `-C`, `--cpu-strict` and `--prio` did nothing here: ggml took the PS5 for an unsupported platform. `patches/0002-ps5-thread-affinity-and-priority.patch` maps them to `scePthreadSetaffinity` and `scePthreadSetprio`, only ever lowers priority, and leaves the main thread unpinned. Not measured yet (#7).

## llama.cpp

First run, Qwen3.5 0.8B Q4_K_M, `-t 2 --poll 0 -lm none`, 512 context:

| | |
|---|---|
| Load | 4.4 s |
| Prompt | 38.5 tok/s |
| Generation | 14.2 tok/s |

`llama-server` loads Granite 4.2 3B (2.1 GiB) with mmap, so mapping a model file works in a payload.

Chat through `llama-server` and `ps5/webui` in the PS5's browser (browser open, `-c 4096 -np 1 -lm none`, 3 threads from the safe defaults):

| Model | Prompt | Generation |
|---|---|---|
| Qwen3.5 0.8B Q4_K_M, thinking off | 67 tok/s | 13.1 tok/s |
| Granite 4.2 3B Q4_K_M | 0.6 tok/s | 0.2 tok/s |

llama.cpp's own web UI renders blank in the PS5 browser; `ps5/webui` (plain ES5) works.

## Payload memory gets paged out

Models over ~1 GB ran 10 to 65 times slower than their size suggests (Granite 4.2 3B at 0.2 tok/s, Qwen3.5 2B at 0.9 tok/s). The reason is paging, not the architecture: while the PS5 browser is in front, the system pages a background payload's memory out to disk. An idle ps5lm was found holding 6 MB of a 600 MB model; the next request paged it back in at 0.3 tok/s.

Locking the weights (`-lm mlock`) fixes it: Qwen3.5 2B went from 0.9 to **8.5 to 9.2 tok/s** and stayed at 1.4 GB resident. Locked pages come out of the pool the home screen needs, though: 1.4 GB locked was fine, **2.7 GB (Qwen3.5 4B) froze the console**. ps5lm locks every model and refuses files over 1.6 GB.

## Downloads

- Sony's `libSceHttp2` cannot send a request from a payload (`sceHttp2SendRequest` fails); plain sockets, DNS and TCP 443 work. ps5lm uses cpp-httplib over mbedTLS 3.6.7 with the Mozilla CA list.
- The kernel caps `SO_RCVBUF` at 64 KB, whatever is asked for. With the Hugging Face CDN ~150 ms away, one connection tops out at 0.9 MB/s.
- 16 connections fetching 8 MiB pieces in parallel: **about 12.5 MB/s** (Qwen3.5 4B, 2.6 GB, in 217 s).

At exit, Sony's libc prints `[SceLibc] A heap error is detected` (SceLibcInternalHeap) after all work is done. After `main`, the SDK's start code unloads the modules the payload loaded and calls Sony's `exit()`, which runs the C++ static destructors while other threads may still run; the tools now leave through `_Exit` instead (`ps5/compat/exit.cpp`). Not yet checked on the console (#3).

## shsrv quirks

- Arguments are split on spaces with no quoting: put prompts in a file (`-f`).
- Programs need an absolute path; a relative one, and very long command lines, end with exit code 3 ("command not found").
- `kill` takes no `-9`; Payload Manager's `/process_kill?pid=N` works.
- `browse <URL>` opens the PS5's web browser.

## The native app on the GPU

PS5LM's app (`ps5/app`, PPSA99575) is a title, not a payload: it gets the console's direct memory and the GPU, through ggml-vulkan on PS5_Vulkan's RADV.

| What | Value |
|---|---|
| Direct memory for the title | 11.4375 GiB up to v0.9.1, 12 GiB expected after; RADV reports it all as one device heap with `radv_enable_unified_heap_on_apu` (2/3 of it without) |
| Device | PlayStation 5 GPU (RADV NAVI21), Vulkan 1.4 |
| Qwen3.8-27B UD-IQ2_XXS, `-ngl 999 -lm none -c 4096 -ub 64` | loads in 82 s |
| Prompt | 35 to 50 tok/s |
| Generation | 21 tok/s on a short prompt, 9 tok/s after the web UI's 371-token prompt |

The pool is what `sceKernelGetDirectMemorySize()` returns, and `sce_sys/param.json` sets it. Up to v0.9.1 the app's file carried the `amm` block (`pagetableMemorySizeInMib` 256, 512 GiB address ranges) and the `kernel` block (256 MiB CPU and 256 MiB GPU page tables) of ps5-homebrew-ui's template, and the pool was 11.4375 GiB (12,280,922,112 bytes, from a user's log in #23). PS5_Vulkan's and PS5_RetroArch's titles declare no `amm` block and no page-table sizes and get 12 GiB (PS5_Vulkan measured on system software 6.02 to 12.70, PS5_RetroArch's probe on 12.09); that probe mapped an 11.875 GiB block from it and touched every word, so the default page tables hold a pool-sized mapping. Neither the GL stack nor RADV calls an AMM function (only `sceKernelAllocateDirectMemory` and `sceKernelMapDirectMemory`). The blocks are gone since; the app logs the pool and the flexible memory at start (`memory at start` in `app.log`). Not yet measured on 13.60. The flexible pool (448 MiB, the system's default) is the other part of the title's budget: ProsperoEden declares 320 MiB in `kernel.flexibleMemorySize` and its direct pool grows by the 128 MiB; PS5LM keeps the default until `app.log` shows how much of it the app leaves free.

What a title has to do differently from a payload, each found by a crash:

- **Never exit.** `exit()` and `_Exit` end in SIGSYS; the title waits for the shell to close it, as ps5-native-app-boilerplate's examples do. It hides the splash screen itself and posts notifications to show its state.
- **Imports that resolve to nothing.** Functions only `libScePosixForWebKit` or `libkernel_sys` define in the SDK's stubs are null in a title: `isatty` (llama.cpp's log setup jumped to address 0), `getnameinfo`, `mkstemp`, `readlink`, `link`, `symlink`, `pathconf`, `fork`. `ps5/app/compat_app.c` defines them, in an archive so the title exports nothing.
- **The SDK fork's `getaddrinfo` always fails**, and cpp-httplib resolves even the address it binds; `compat_app.c` has a numeric one. Its `accept4` is the title's own too, and logs `errno` when `accept` fails: with the payload's version the listener stopped 20 s into the first run.
- **No `HOME`**: llama.cpp throws looking for its cache directory without one.
- C++ exceptions and RTTI are off by default for the PS5 target and must be turned on for llama.cpp.

### Prompt batch

llama-server's `-ub` is how many prompt tokens go through the GPU at once. The planner gives language
models 64 (`kPromptUbatch` in `ps5/app/model_plan.hpp`), what the console has run since the first
release; decision models and image encoders get 2048. A larger batch needs larger compute buffers.
Measured on a PC (RTX 3090, ggml-vulkan with cooperative matrices, integer dot products and bf16 off,
so it takes the console's shader paths): Qwen3.8-27B UD-Q2_K_XL at 64k with a q4_0 cache, flash
attention on, a 2,048-token prompt and 128 tokens generated, the faster of two runs.

| `-ub` | Device compute buffer | Host compute buffer | Prompt, PC | Generation, PC |
|---|---|---|---|---|
| 64 | 27 MiB | 13 MiB | 270 tok/s | 35.1 tok/s |
| 128 | 54 MiB | 26 MiB | 307 tok/s | 34.9 tok/s |
| 256 | 108 MiB | 52 MiB | 332 tok/s | 34.4 tok/s |
| 512 | 216 MiB | 104 MiB | 355 tok/s | 34.2 tok/s |

The speeds are the PC's, not the console's. The buffers carry over: on the console both come from the
pool, about 38 bytes a token for each unit of feed-forward width, which the planner now counts. With
the 11.04 GiB a v0.9.1 console had free before a model (#23), the 27B gets 32k at any of these sizes;
with the 12 GiB pool it keeps 64k at 64 to 256 (11.4 of 11.6 GiB expected) and only just fits at 512.
`scripts/bench-ubatch.py` measures speed and memory on the console, size by size.

### Probes

Measurements the app makes only when the PC asks, each on a thread of its own, with the results in
`app.log` (`ps5/app/probes.cpp`). They are for decisions not taken yet, such as reading MoE experts
from a drive instead of holding them in memory (Qwen3.8-Flash-Next on the roadmap).

| `scripts/ps5lm-app.sh` | Logs |
|---|---|
| `bench-read <model file>` | Sequential reads (8 MiB, up to 2 GiB), then three seconds each of random reads of 64 KiB, 256 KiB, 1 MiB and 4 MiB with one and four threads, through the cache and with `O_DIRECT`: MB/s, reads a second, median and p99 latency |
| `bench-memory` | The direct pool and what the system keeps of a nominal 12 GiB, its largest free block, the flexible pool; with no model loaded, also the largest block that allocates and maps, given back at once |
| `bench-write` | Write speed by method to the data folder and a USB drive |
| `memory` | One `memory asked` line: pools, CPU heap and ggml's devices |

Point `bench-read` at a model of several GiB on the drive in question (internal storage, the M.2
drive or USB), so the cache holds little of it; the `O_DIRECT` lines are the drive itself.

### Switching models

Unloading a model in the running app leaves part of its GPU memory held: with Qwen3.8-27B UD-IQ2_XXS at 64k, the pool went from 11.01 GiB free before loading to 5.77 GiB free after `llama_server` returned, and ggml's Vulkan figure fell the same way, so the next model could not get its KV cache. The app therefore changes model by restarting itself: it writes the choice to `/data/PS5LM/model.txt` (read once by the new process, then deleted, so a plain launch loads nothing and opens the library) and calls `sceSystemServiceLoadExec("/app0/eboot.bin", NULL)`. The system replaces the process (`Kill for LoadExec(0x11d) => 0`, then a new `EXEC`) without the crash reporter, and the new process starts from the whole pool (11.01 GiB free). Where the memory is held is still open.

### The dashboard on the console

ps5-homebrew-ui draws through ps5-opengl in the same title as RADV: the GL stack is linked as one object with only its GL/EGL API global (4,399 symbol names collide with RADV's Mesa otherwise), `sceAgcInit` runs once for both drivers (`--wrap`), and the GL runtime's thread-local initialiser `_ZTH23_mesa_glapi_tls_Context` is an empty function, as ProsperoAI has it. The screen costs about 0.35 GiB of the pool (11.38 to 11.03 GiB free). The app can be driven from a PC without a controller: `scripts/ps5lm-app.sh close|deploy|launch|screenshot|load|unload`.

ShadowMountPlus registers a title once and copies its icon and backgrounds then; replacing them later needs a re-registration. Uninstalling to force one hit "Register failed: TitleDir bridge unavailable" after ShadowMountPlus had logged `stale cave bridge detached`; reloading ShadowMountPlus fixed it.
