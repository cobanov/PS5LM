# Experimental fleet: one model, several PS5 workers

**Status: host controller, and a PS5 worker that is part of the app and starts
only when you opt in; CPU RPC inference and the title's cross-build are tested
in CI. On hardware, one PS5 worker has served one model together with a PC's
CPU, a Mac's GPU (Metal) and an RTX 3090 (CUDA and Vulkan) (2026-11-01 and
11-02, below); more consoles at once are not validated by the maintainers.** This is not a performance claim. Up to v0.10.0 the worker was a
separate build, `ps5lm-app-fleet-v0.10.0.zip`; from v0.11.0 every PS5LM has it.

What it is for: a model that does not fit in one console's memory runs across
several, each holding a run of its layers. Every token passes through every
worker over the network, so it is not faster than one console; it makes larger
models possible. Workers are PS5s; a PC can join too, with its CPU or, with a
GPU host build (`PS5LM_FLEET_GPU`), its GPU.

Any single-file model the pinned llama.cpp loads as one stack of decoder blocks
can be split: Llama, Qwen 2.5, Qwen 3.5/3.6/3.8 (hybrid Gated DeltaNet, dense and
MoE), Gemma 3/4, gpt-oss and the other attention, sliding-window and Mamba-style
hybrid models ([Which models](#which-models)). The control server
(`python3 -m fleet control`, and `python3 -m fleet ctl` for shells and agents)
runs many devices as pools of replicas behind one OpenAI- and Anthropic-style
API. A replica is a layer split, one console serving a model by itself, or a
GPU. The server moves consoles in and out of worker mode over the network and
restarts what fails ([Control server](#control-server)).

## Quick start

1. **On each PS5:** PS5LM v0.11.0 or later (for v0.10.0, its
   `ps5lm-app-fleet-v0.10.0.zip` from the
   [releases](https://github.com/cobanov/PS5LM/releases)), then put `/data/PS5LM/fleet-worker.txt` on it over FTP: the console's own IP
   and a port below 40000, then the acknowledgement line
   ([Configure workers](#configure-workers-only-after-building-and-validating-the-title)).
   Start PS5LM: it shows the worker screen instead of the dashboard.
2. **On a Linux or macOS PC on the same network:** `git submodule update --init
   --depth 1 third_party/llama.cpp`, then `scripts/build-fleet-host.sh`
   ([Host](#host-cpu--rpc)). To add the PC's GPU as a worker, build with
   `PS5LM_FLEET_GPU=metal` (or `cuda`, `vulkan`) and start
   `GGML_RPC_NO_COMM=1 build/fleet-host/bin/ggml-rpc-server -H <PC IP> -p 30053 -d MTL0`
   (`CUDA0`, `Vulkan0`), then list it as one more node. `GGML_RPC_NO_COMM=1`
   makes the server refuse the peer-to-peer requests a layer split never uses
   (`patches/fleet/0002`); the control server sets it for the workers it starts.
3. **Nodes:** copy `fleet/nodes.example.json` and list each console's
   `IP:port` with its memory budget.
4. **Plan, then serve:** `python3 -m fleet plan --model model.gguf --nodes
   nodes.json` shows which layers go where; `python3 -m fleet serve ...
   --ack-insecure-rpc` starts one llama-server on `127.0.0.1:8081` of the PC
   ([Plan and serve](#plan-and-serve-from-the-controller)).
5. **Back to normal:** delete `fleet-worker.txt` and restart PS5LM, or put a
   `/data/PS5LM/leave-worker` file on it: a worker takes it within a second and
   restarts as the app (PS5LM after v0.11.0).

Or, instead of steps 1 and 3 to 5: `python3 -m fleet control` and the page at
http://127.0.0.1:8090 ([Control server](#control-server)).

RPC has no authentication: keep workers on a network you trust. On macOS, the
app that runs the controller (Terminal, iTerm, a multiplexer) needs Local
Network permission, or connections to the consoles fail with "No route to host".

## Architecture

```
GGUF file / public HTTPS URL
            |
 Linux/macOS controller: bounded download + SHA256 + GGUF inspection
            |             contiguous layer/KV capacity plan
       one llama-server (API bound to 127.0.0.1)
         /                         \
   llama.cpp RPC               llama.cpp RPC
 PS5 A: Vulkan layers 0..k     PS5 B: Vulkan layers k+1..L + output
```

This is **model parallelism, not load balancing**. A request needs every selected
worker. The host holds the complete GGUF on disk, loads input/shared tensors
locally, and upstream llama.cpp transfers the assigned tensors to each RPC
backend. There is no separate shard-file format, full-model replica per worker,
custom activation protocol, or independent model server on each console. RPC
also manages the distributed computation and per-layer KV cache. Transfers may
return through the controller; this does not promise direct PS5-to-PS5 pipelines.

The pinned dependency already supplies this machinery. We keep its wire format
(RPC **8.0.0**) and use `--split-mode layer`, an explicit RPC-only device list,
and `--tensor-split`. Midpoint boundaries reproduce the pinned loader's `L+1`
logical slots (the extra slot is output), avoiding floating-point boundary drift.
Host accelerators are not silently added. The planner minimises the maximum
estimated memory utilisation for the **given node order**, not latency or tokens/s.

### Which models

- 2–15 selected workers, exactly one device per endpoint, at least one block per
  worker. The pinned CLI rejects 16 tensor-split entries.
- Complete, single-file, little-endian GGUF v3, at most 512 blocks, in any of the
  35 tensor types the pinned ggml defines (IQ, TQ, MXFP4 and Q2_0 included).
- One sequence, FP16 K/V cache, flash attention off, fixed context no greater than
  training context. Automatic fit, lazy loading and opportunistic host-operation
  offload are disabled.

The estimator follows where the pinned llama.cpp puts each tensor and what cache
it keeps for each block, for any architecture:

- **Weights:** exact file spans per block. `token_embd` (and Gemma's per-layer
  embeddings) stay on the controller. RoPE factor tensors (Llama 3.1+) get one
  copy on every worker. Gemma's `per_layer_model_proj` lives with the first
  worker. The output head, final norm and a tied output copy of `token_embd` live
  with the last worker.
- **NextN (MTP) blocks** (Qwen 3.8 27B's block 64): they take a tensor-split slot,
  but llama-server loads neither their weights (`load_mtp` is off) nor a cache.
- **Attention cache:** per-block K, and V rows as wide as the model's widest V,
  since without flash attention llama.cpp pads them (`n_embd_v_gqa_max()`).
  Sliding-window blocks keep `min(context, window + 128)` tokens padded to 256, from
  the file's per-block list or llama.cpp's own pattern for the architecture
  (gpt-oss, Gemma 2/3, Cohere 2, ...). Gemma's shared-KV blocks keep none.
- **Recurrent state:** Qwen 3.5/3.6/3.8 keep attention every fourth block and a
  Gated DeltaNet state in the others. Mamba keeps a state in every block.
  Granite 4, Nemotron-H, Jamba and LFM2 keep one where `head_count_kv` is 0. The
  state is `n_embd_r + n_embd_s` floats per block.
- **Refused:** RWKV, KDA, linear and latent (MLA) attention caches, parallel
  attention+SSM (Falcon-H1), encoder-decoder and audio-codec layouts, tensors
  llama.cpp would not place, and split GGUF files.

Checked against llama.cpp itself: `llama-fit-params --fit-print on` loads a model
without allocating and prints each device's weights, cache and compute buffers.
For the planner's own two-worker split, over two local RPC workers, it matched the
estimate within 1 MiB, weights and cache on every device, at 4k and 32k tokens,
for Llama 3.1 8B, Qwen 2.5 0.5B, Qwen 3.5 0.8B and 9B, Qwen 3.6 35B-A3B,
Qwen 3.8 27B (UD-Q2_K_XL and UD-Q4_K_XL), Gemma 4 E4B and gpt-oss 20B (2026-11-09).
Qwen 3.5 0.8B and Gemma 3 270M, split over two CPU RPC workers by
`python3 -m fleet serve`, gave the same greedy tokens as the whole model on one RPC
worker.

The compute buffers are not in the estimate; the reserve covers them. They grow
with context without flash attention (0.5 to 1.8 GiB per worker at 32k for the
models above). So before llama-server starts, `serve` and the control server run
`llama-fit-params --fit-print on` with the plan's exact split against the live
workers. It refuses to start when llama.cpp projects any worker within 256 MiB of
its budget, or an eighth of a smaller budget (`llama_check` in the printed plan). A successful plan is still **not
an allocation guarantee or reservation**; live memory can change.

## Security boundary — read before enabling workers

Upstream explicitly calls RPC experimental and insecure. It has **no
authentication or encryption** and is not an ordinary safe inference API.
Anyone who can reach it must be treated as trusted with the worker process and
memory. IP validation is not authentication.

- Use an isolated, trusted wired LAN/VLAN with access limited to the controller.
  No guest/shared LAN, port forwarding, public interface, public proxy or tunnel.
- The worker requires an explicit numeric private IPv4 bind, high port and a
  written acknowledgement. No wildcard `0.0.0.0` or hostname is accepted.
- `GGML_RPC_NO_COMM=1` is forced by this integration. `patches/fleet/0002` makes the server
  reject `COMM_INIT` before opening the extra wildcard peer listener in upstream
  tensor-parallel code. This fleet uses layer splitting, not those collectives.
- `patches/fleet/0001` initializes BSD socket lengths and suppresses SIGPIPE portably.
  Neither patch makes arbitrary RPC requests safe. Keep the network isolated.
- Workers have no disk tensor cache; normal PS5LM API, model auto-load and
  downloads are disabled in worker mode. There is one controller per worker.
- The host API is hard-bound to loopback; the CLI starts a foreground process,
  cleans it up on Ctrl+C/SIGTERM, and does not deploy consoles, modify firewall
  rules, create credentials, enroll nodes or persist a background service.

## Revisions and build

Developed on PS5LM `c4595784ce75c1d58777b11ef812d3e087e74296`, with llama.cpp
`970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc` and UI
`bf44d80384d5b8f8f4f213cf98d748887211328d`. Build host and workers from the same
checkout and patch set. The host manifest checks the local binary checksum and
llama revision. A remote handshake checks **protocol compatibility**, not a remote
Git revision or source attestation; the operator must deploy matching builds.

### Host (CPU + RPC)

Python 3.10+, a C++17 compiler, CMake, Ninja and Git are required. NumPy and
PyYAML are needed only to generate the tiny test fixture, not for the controller.

```sh
git submodule update --init --depth 1 third_party/llama.cpp
scripts/build-fleet-host.sh
# build/fleet-host/bin/{llama-server,ggml-rpc-server,...}
# build/fleet-host/fleet-build.json (revision, patches, binary SHA256)
```

The host script applies the repository's numbered llama.cpp patches and
`patches/fleet`, disables RDMA, and builds the RPC tools, `llama-fit-params` (the
memory check before every start) and the upstream multi-server regression. It is CPU only unless `PS5LM_FLEET_GPU` is `metal`,
`cuda` or `vulkan`; then `ggml-rpc-server -d MTL0` (`CUDA0`, `Vulkan0`) serves
that GPU as a worker. The controller always names its RPC devices explicitly,
so a GPU in the host build is used only through a worker you start and list. It does not fetch model weights or start a listener. Set `CMAKE`,
`NINJA` or `CMAKE_BUILD_PARALLEL_LEVEL` for local tool locations/resources.

### PS5 worker — part of the app

Follow [BUILDING.md](BUILDING.md) to prepare PS5_Vulkan, its pinned PS5_Mesa and
PS5_PayloadSDK forks, patched RADV, native packaging tool, `libc.prx` and the
checksum-pinned ps5-opengl SDK. Then:

```sh
git submodule update --init --depth 1
scripts/build-app.sh
```

Every build has the worker: llama.cpp with `GGML_RPC=ON` (`GGML_RPC_RDMA=OFF`)
and `patches/fleet` on top of the numbered patches, and the title with
`fleet_worker.cpp` and the worker's startup mode in `main.cpp`. It stays idle,
with no listener, unless `/data/PS5LM/fleet-worker.txt` exists. Up to v0.10.0
this was a separate `PS5LM_FLEET=1` build and zip.

CI builds the title on every pull request (`native-build.yml`,
`scripts/test-native-build.sh`, with the structural ELF/FSELF checks,
[CI.md](CI.md)). That proves the worker links, not that it boots, that its
Vulkan results are right, or that a
real multi-console run works. QEMU is not a substitute: its
[x86 system models](https://www.qemu.org/docs/master/system/target-i386.html)
and [Linux/BSD user modes](https://www.qemu.org/docs/master/user/main.html)
provide no PS5 loader, AGC or direct memory.

On a console (2026-11-01): the fleet build on one PS5 (worker on
port 8082) and `ggml-rpc-server` on a Linux PC's CPU, with
`python3 -m fleet serve` on that PC, served `qwen2.5-0.5b-instruct-q4_k_m.gguf`
with blocks 0–22 on the PS5 and block 23 plus the output on the PC, and
answered a chat request correctly. This run found the 40000 port limit below;
the worker writes its own and ggml's messages to `/data/PS5LM/app.log`.

The same day, with the worker in the normal app (this build) and a Mac's GPU:
`ggml-rpc-server -d MTL0` from a `PS5LM_FLEET_GPU=metal` host build on an Apple
M-series Mac, the controller on the Linux PC. Blocks 0–22 on the PS5, block 23
and the output on the Mac's GPU; it answered correctly at 13 tokens/s (7.4 with
the PC's CPU in that place). One small model, so not a performance figure.

On 2026-11-02, the same PS5 worker with an RTX 3090 on the Linux PC, from
`PS5LM_FLEET_GPU=cuda` and `vulkan` host builds (`ggml-rpc-server -d CUDA0`,
`-d Vulkan0`), the controller on the same PC: blocks 0–19 on the PS5, blocks
20–23 and the output on the 3090 (`--tensor-split 39,11`). Both answered
correctly, at 28 to 30 tokens/s on a longer reply. The controller's binary must
run on the PC it is started on: a `build/fleet-host` made inside a newer Linux
container stops at `GLIBC_2.43 not found` on an older host.

## Configure workers (only after building and validating the title)

On each console, explicitly create `/data/PS5LM/fleet-worker.txt` containing
**that console's own private IP**, port 1024–39999, and the exact acknowledgement:

```text
192.168.50.21:30052
I_UNDERSTAND_RPC_IS_UNAUTHENTICATED
```

The port must be below 40000: a PS5 title gets "Permission denied" for 40000
and up (measured on a console), so upstream's default 50052 cannot work there.
Workers on a PC or Mac (`ggml-rpc-server`) can keep 50052.

Restart PS5LM. It displays an experimental worker screen, not “Ready”: upstream
has no ready callback, so the host must successfully handshake. A malformed,
unreadable, non-regular or oversized config fails closed without starting the
normal model server. No config keeps ordinary PS5LM behaviour. Remove this file
and restart to return to single-console serving.

A worker takes three request files, checked once a second:

- `/data/PS5LM/quit` closes the title.
- `/data/PS5LM/leave-worker` (PS5LM after v0.11.0) deletes `fleet-worker.txt` and
  restarts the title the way a model change does, so it comes back as the app
  without anyone at the console.
- `/data/PS5LM/unload` (after v0.11.0 too), which restarts the app, restarts a
  worker as a worker. It reads `env.txt` again, so its settings change without
  anyone at the console.

On the 2026-11-09 console run (one PS5 on firmware 11.60), worker mode took 3 s to
start, and the app was back and taking requests 6 s after `leave-worker`. On
2026-11-11 the same console's worker answered again 4 s after `unload`.

### A worker without its screen

`PS5LM_WORKER_SCREEN=off` in `/data/PS5LM/env.txt` (PS5LM after v0.11.0) starts a
worker without its screen. The screen and the dashboard take about 350 MiB of the
title's memory pool, which the model can have instead. The TV keeps the system's
splash screen, the notification about unauthenticated RPC still shows, and
`app.log` says what the worker does. A worker logs its memory once it has started
(`memory worker started`), and again when `/data/PS5LM/memory` is there.

The free memory the worker reports to the host, on one PS5 on 11.60:

| Worker | Free |
|---|---|
| With its screen | 11,854 MiB |
| Without it | 12,217 MiB |

### The weight cache

`PS5LM_RPC_CACHE_GB=N` in env.txt (whole GiB, 1 to 4096; off when 0 or not set,
PS5LM after v0.11.0) keeps up to N GiB of the weights hosts send, in
`/data/PS5LM/rpc-cache`. llama.cpp's RPC client offers a tensor of more than 10
MiB by its hash first; when the worker has it, it reads the tensor from the
console's storage instead of getting it over the network. That helps when a
model loads again on the same console: at another context, after a restart or a
crash, or in another split that gives the console the same blocks. It helps
most where the network is the limit: many consoles loading from one host at
once, or a console on Wi-Fi.

- Each file is one tensor under its FNV-1a hash. `patches/fleet/0005` writes it
  under a temporary name and renames it once whole, and uses it only when it
  hashes to what the host sent and fits the tensor's buffer. Anything else is
  dropped, and the host sends the tensor again.
- The files are written through ftpsrv, as downloads are: the console slows a
  title's own writes to /data to about 1.2 MB/s after 2 GB. Without an ftpsrv
  the worker keeps nothing and says so once in `app.log`. They are read in 8 MiB
  reads: every read call of a title takes about 1.4 ms, so small reads are slow.
- The worker keeps the folder to N GiB: at start and every 30 s, the least
  recently used files go first. At start it also removes what a cut-off write
  left.
- The fleet control server sets it per console (`cache_gib`, 32 by default) and
  deletes the files with its `clear-cache` action.

On 2026-11-11, one PS5 on 11.60 over gigabit Ethernet, loading Ornith 1.5 9B
Q4_K_M (5.4 GiB; 4.2 GiB of it in tensors the cache keeps) from a PC's CPU:

| Load | Seconds |
|---|---|
| No cache | 48.9 |
| First load, saving the cache | 113 |
| From the cache | 17.6 |

Every load gave the same answer at temperature 0.

### Parked state

A host that serves many conversations on one slot (a swarm of agents taking
turns) has to put the state of the one leaving somewhere. llama-server's own
answers keep it on the host: the prompt cache in memory, or a slot file. For
Qwen3.8-27B at 153K tokens that is 2.8 GiB per conversation, fetched over the
network at about 94 MiB/s and sent back the same way: 50 s out and 65 s back,
and a 12 GB host holds two.

With device parking the state stays on the console. The worker keeps it in
files in `/data/PS5LM/park`, the host keeps the token list and a manifest, a few
MB per conversation, and a save writes only what was added since the last one.

On the console: `PS5LM_PARK_GB=N` in env.txt (whole GiB, 1 to 4096; 128 when
not set, 0 turns it off; PS5LM after v0.11.0). The worker tells hosts in its
handshake that it takes park commands, so an older host or an older worker
serves as before.

On the host: `--slot-park-path DIR` (a folder for the bookkeeping, not in
`/tmp`) and `--cache-ram 0`: the automatic prompt cache would still fetch the
state of every conversation that loses the slot. The client parks and resumes
by name:

| Request | Does |
|---|---|
| `POST /slots/0?action=save` `{"filename": "n17"}` | Parks the slot under the key `n17` |
| `POST /slots/0?action=restore` `{"filename": "n17"}` | Puts that state back in the slot |
| `POST /slots/0?action=park-list` | The keys, their sizes and state, the room left |
| `POST /slots/0?action=park-delete` `{"filename": "n17"}` | Drops a key, on the console and on the host |
| `POST /slots/0?action=park-status` | Free space, bytes parked, the cap |

A key is 1 to 64 letters, digits, `.`, `_` or `-`. With `--slot-save-path` also
set, `"device": false` in the body makes the same two actions use a host file
as before.

A save or a restore is never refused because the slot is busy, and never runs
beside a request: it waits for the slot, and the server does one thing on a
slot at a time.

- Sent while a request generates, a save parks what that request leaves when
  it ends, and a restore replaces it then. The request itself is not disturbed.
- Saves, restores and requests that name the slot (`"id_slot": 0`, as the
  swarm's do) run in the order they were sent. A request that names no slot is
  taken up after the park actions that wait, whenever it came.
- Two saves of one key at once: the first parks, the second answers
  `"mode": "unchanged"`. A save and a restore of one key at once run one after
  the other in the order sent; either way the key is whole.
- A client that gives up on a save or a restore that is still waiting cannot
  count on it being dropped: the server notices a closed connection about
  once a second and drops the action then, but one whose turn comes first is
  run all the same, and once started it runs to its end. After giving up on a
  restore, look at the slot (`GET /slots`) or send the restore again.
- `park-list`, `park-delete` and `park-status` do not wait for the slot; they
  run between the batches of a request in progress.

A save answers with what it wrote (Qwen3.5-0.8B on a CPU worker, the fourth
firing of a conversation):

```json
{"id_slot": 0, "filename": "n17", "device": true, "mode": "append",
 "n_saved": 10063, "n_appended": 2516,
 "n_written": 28897228, "n_written_kv": 8695296, "n_written_state": 20201932, "n_written_checkpoint": 0,
 "n_parked_bytes": 75181592, "n_segments": 4,
 "checkpoint_n_tokens": 7547, "checkpoint_kind": "previous_save",
 "pruned": [], "timings": {"save_ms": 104.2}}
```

- `mode` is `new` (the key's first save, or one that could not be added to:
  `reason` says why), `append`, `rewrite` or `unchanged`.
- **Append only.** The attention cells of the tokens added since the key's last
  save go to a new file (18 KiB a token for Qwen3.8-27B with a q4_0 cache). The
  recurrent state has a fixed size (150 MiB for that model) and is written
  whole each time: that is the fixed cost of a save.
- **A changed history is never resumed as it was.** The host compares the
  slot's tokens with the parked ones. If they differ before the parked end (the
  client compacted or edited the history), the save keeps the files that end
  before the difference and writes the rest again: `"mode": "rewrite"`,
  `"rewritten_from": N`.
- **Going back.** A recurrent state cannot be rewound, so a request that
  returns and differs from the parked tokens before their end needs an earlier
  state. That is the recurrent state of the save before, which stays on the
  console (`"checkpoint_kind": "previous_save"`): such a request reads again
  what the last save added. A client whose requests usually stop short of the
  parked end (it does not send the last reply back as it was written) sends
  `"checkpoint": true` with the save: the slot's own checkpoint, 4 tokens before
  the end of its last prompt, is parked too, and the return reads a handful of
  tokens, at the price of a second 150 MiB block per save. A key's first save
  always does this.
- `pruned` names the keys the console dropped to make room for this save.

A restore answers `{"n_restored": tokens, "n_read": bytes, "n_segments": files,
"checkpoint_n_tokens": ..., "timings": {"restore_ms": ...}}`. When it fails, the
slot is left empty, never half restored, and the body's `error.park_error` says
why:

| Status | `park_error` | Means |
|---|---|---|
| 404 | (llama-server's own "File Not Found" body) | No such key |
| 410 | `park_pruned` | The console dropped the key to make room, or its storage was cleared. The history has to be read again |
| 409 | `park_torn` | The files and the bookkeeping do not agree (a size or a checksum). Delete the key |
| 409 | `park_incompatible` | Parked by another model, or it does not fit this context |
| 507 | `park_no_space` | (save) Over the cap or the disk is full, with nothing left to drop |
| 501 | `park_unsupported` | The worker has no parking (older, or `PS5LM_PARK_GB=0`), there is more than one worker, or the model has a draft model or media in the slot |
| 400 | `park_empty`, `park_bad_request` | Nothing in the slot; a bad key |
| 500 | `park_io` | Anything else, with the worker's message |

How it holds together:

- A key is a folder on the console. Every file in it is written once, in order,
  under a temporary name, and renamed when whole: `NNNNNN.kv` (the cells one
  save added), `NNNNNN.rs` (the recurrent state at the end of that save; the
  newest two are kept), `NNNNNN.ck` (a parked checkpoint). The host's manifest
  names the files that make up the state, with their sizes and checksums, and is
  replaced only after the console has the new files whole. A save that dies
  half way (the worker killed in it) leaves the previous state loadable; what it
  left behind is removed by the next save.
- The worker moves the data between the GPU and the file through one 8 MiB
  buffer, whatever the size of the state. The files are written through ftpsrv,
  as the weight cache's are, and read in 8 MiB reads. Without an ftpsrv a save
  is refused (`park_io`): it is never written the slow way.
- The worker keeps the folder to N GiB: before a save it drops whole keys, the
  one used longest ago first, never one with a file open and never the one
  restored last. It keeps the order of use itself (`/data/PS5LM/park/.used`,
  rewritten at every save and restore), because the console's file times move
  when a key is saved but not when it is read. 128 GiB holds forty
  conversations at a full 155K window (3.1 GiB each) or a hundred at about 60K.
  There is no free-space floor on a console: a title's free-space call answers
  a fixed 16 GiB, so only N bounds the folder.
- On the host a key is `manifest.json`, `tokens.NNNNNN.bin` (4 bytes a token)
  and `meta.NNNNNN.bin` (24 bytes a token): 4.3 MB at 153K tokens. They survive
  a server restart and a reboot; a parked key is resumed by a new server process
  as by the old one. A save that adds appends to the two files past what the
  manifest covers; a save that rewrites starts a new pair and drops the old one
  after the manifest is replaced. So if the server or the PC dies in a save,
  append or rewrite, the key is as its last finished save left it.
- One worker only: with several, a slot's state is spread over them, and
  parking is refused.

Measured on 2026-11-15 on one PS5 on 11.60 serving Qwen3.8-27B (GSQ-RCO
IQ3_XXS, every layer on the console, one slot of 155,648 tokens, q4_0 cache)
from a PC over gigabit Ethernet. Neurons of real source code, each firing 200
new tokens and a 16-token reply; times are the server's own for the whole
action:

| | 20K tokens | 100K | 153K |
|---|---|---|---|
| First save | 651 MiB, 10.4 s | 2,057 MiB, 30.6 s | |
| Save after growing to this size | | 1,552 MiB for 79,784 new tokens, 22.1 s | 1,059 MiB for 51,704 new tokens, 15.0 s |
| Save after a firing (216 new tokens) | 153.4 MiB, 2.2 s | 153.4 MiB, 2.2 s | 153.4 MiB, 2.2 s |
| Restore | 2.2 s | 3.6 s | 4.6 s |
| The same return through the host prompt cache | 14 s | 46 s | 65 s |

- A save writes 149.6 MiB (the recurrent state) plus 18.0 KiB per new token,
  at 72 to 74 MB/s; with `"checkpoint": true` 303 MiB in 5.7 s. A first save
  includes the checkpoint.
- Four neurons at 100K took turns on the slot, twelve hand-overs: each reply
  was the one the same neuron gave without leaving the slot, 201 tokens were
  read for the 200 new ones, and a hand-over (restore, firing, save) took about
  18 s. The host's llama-server stayed at 1.1 GiB.
- A request that differed from the parked tokens 3 tokens before their end
  read 414 tokens after a default save (217 with the neuron still in the slot,
  and 217 after a save with `"checkpoint": true`), at 101K and at 153K alike.
- The console's free memory did not move during saves and restores (never
  under 71.7 MiB in 306 samples), and the keys were restored by a new server
  process and a restarted worker with the same replies.
- At 5 hand-overs an hour that is 18 GiB written to the console's SSD a day,
  at 48 an hour 173 GiB, for firings this small; a firing that adds 3,000
  tokens writes 202 MiB (24 and 228 GiB a day).

`scripts/ps5lm-app.sh bench-park [GiB]` measures the write path first: that
many GiB (4) in 8 MiB pieces through ftpsrv to the data folder and to
`/mnt/usb0` and `/mnt/ext0` when they are there, with the speed of every GiB,
then the file read back and checked. It works beside a running worker.

**Restart the worker title between different model runs and after errors.**
PS5LM documents retained Vulkan memory after in-process unload; this increment
does not promise safe hot reload. Stop through the normal title-close / existing
quit-file mechanism. A worker failure fails the whole model; no failover/resume
is implemented.

## Plan and serve from the controller

Copy `fleet/nodes.example.json`, replace endpoints, and set conservative host and
worker memory/reserve values from measurements. The example's 10 GiB worker
budget with 2 GiB reserve is **an operator example, not measured fleet capacity**.
`memory_mib` is the total allowed budget before subtracting `reserve_mib`.
Do not run the normal local PS5 model alongside a worker.

```sh
# Offline capacity estimate from an already downloaded model:
python3 -m fleet plan --model /models/model.gguf \
  --nodes my-nodes.json --context 4096 --plan-out plan.json

# The model argument also accepts a public HTTPS URL:
python3 -m fleet plan --model 'https://example.org/model.gguf' \
  --sha256 <expected-full-file-sha256> --nodes my-nodes.json --context 4096

# Explicitly opt in after isolating/trusting the network:
python3 -m fleet serve --model /models/model.gguf \
  --nodes my-nodes.json --context 4096 --ack-insecure-rpc --plan-out run-plan.json
```

`plan` never connects to workers and reports `operator_budgets_only`. A URL is
fully downloaded before inspection; the downloader validates public HTTPS and
redirect destinations, pins DNS to the TLS connection, bounds bytes/time/disk,
streams to a partial file and hashes the complete file. No gated model token,
remote code, pickle or arbitrary extraction is supported. Prefer an expected
`--sha256`; without it the digest records identity but not publisher authenticity.
Default download limit is 32 GiB, with `--max-download-gib` available up to 128.
Model URLs are not automatically downloaded in CI.

`serve` validates the local binary manifest, performs the upstream RPC handshake,
requires exactly the requested device order/topology, caps every operator budget
by reported live free memory, and repeats preflight after acquisition. It then
recomputes the plan and launches a single llama-server. It does **not** execute
an old JSON plan as trusted instructions. Wait for llama-server readiness:

```sh
curl --fail http://127.0.0.1:8081/health
curl --fail http://127.0.0.1:8081/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is", "n_predict":16, "temperature":0}'
```

Review server logs for actual device/layer allocation. Model-load failure,
unsupported operations, worker disconnect or OOM is not converted into a
successful fleet deployment. A snapshot/printed plan does not mean inference
has started. Input and unsupported-operation work can still occur on the host;
backend behaviour and memory must be checked on real consoles.

### Serving many agents

`scripts/bench-swarm.py` loads a running llama-server the way an agent swarm
does. Agents share a system prompt, histories grow each turn and compact, and a
few hot agents come back more often than the rest. The default sizes are half
of a measured swarm task. In the runs below that came to about 11k prompt
tokens a call, 4k of them not cached, and 360 generated. It reports turns per
hour, time per turn, and the share of the prompt served from cache.

```sh
scripts/bench-swarm.py --url http://127.0.0.1:18330 --concurrency 2 --tag np2 --out np2.json
```

The table shows one PS5 (11.60) on RPC from a CPU host, with q4_0 KV and
`--reasoning-budget 256`. The values are turns per hour, with the mean turn
time in brackets.

| Server | Ornith-1.5-9B Q4_K_M | Qwen3.8-27B IQ3_XXS |
|---|---|---|
| `--parallel 1` | 123 (29 s) | 48.5 (74 s) |
| `--parallel 2 -no-kvu` | **140 (51 s)** | **53.9 (134 s)** |
| `--parallel 2 -kvu` | 110 (66 s) | 49.1 (147 s) |
| `--parallel 3 -no-kvu` | 144 (75 s) | worker out of memory |
| `--parallel 2 -no-kvu`, MTP n=2 | 118 (61 s) | worker out of memory |
| `--parallel 1`, MTP n=2 | 110 (33 s) | 44.8 (80 s) |

For a swarm on one console, use `--parallel 2 -no-kvu` and let two requests in
at a time:
- A unified KV cache (`-kvu`, which the server also turns on when `--parallel` is
  left on auto) makes every ubatch's attention mask cover every slot's cells.
  That mask goes over RPC on every ubatch, and costs 9–22% of turns here (the
  Ornith `-kvu` run had a 131k pool, the rest 32k a slot).
- More slots than requests in flight doesn't help. Only slots with consecutive
  ids decode in one batch.
- When an agent lands on a slot that holds another agent's history, the server
  saves and restores the prompt cache over RPC. That stalls everything for
  about 4 s.
- MTP doesn't help this load. The drafts were accepted about 90% of the time,
  and it was still slower.
- `-ub` 256 to 1024 are within 2% of each other.

### A reply sent back

A chat client sends each reply back as a message with its next request. The
server writes that message out through the chat template and tokenises it
again, and compares the result with the tokens the slot holds. They are not
always the same tokens although the client changed nothing:

- The model spells text with whatever tokens it likes (` book` `marks`); the
  tokeniser has one spelling (` bookmarks`).
- The whitespace around the end of the reasoning is the template's, not the
  model's. The template trims the reasoning and writes its own separator
  before the end-of-reasoning marker, and after it: the server hands the
  content over without its leading whitespace. A reasoning budget that ends
  the thinking in the middle of a line leaves other whitespace in the slot,
  before the marker (after a space or a blank line) or after it (the model
  goes on with its line).

A hybrid model's recurrent state cannot step back, so the slot used to return
to its checkpoint and read the whole reply again: 10 of 227 follow-up turns of
Qwen3.8-27B, 84 to 10,508 tokens each, 16 s at 30K tokens of depth and 48 s at
150K for a reply of ordinary length. After a restore from a parked save it read
the whole last firing, and the next save rewrote the key from there.

`patches/0017` keeps the reply as it was generated: where the slot's tokens and
the prompt stop matching, it compares the two as text, marker tokens by id,
and when they say the same the prompt continues from the slot's tokens.
Whitespace is text like any other, except in one place: right before, and
right after, the marker that closes the thinking block the template opened in
an assistant turn. A `</think>` anywhere else (one the model wrote into its
answer, one in a user or a tool message) is compared with its whitespace.
Nothing changes in the API. For a client:

- After a kept reply the model continues from the tokens it wrote, not from a
  re-tokenised copy, so the next reply can differ from what an unpatched
  server gives after reading the reply again.
- `prompt_tokens` counts the kept tokens: a token or two away from tokenising
  the rendered prompt (`/apply-template` then `/tokenize`), for the rest of
  the conversation.
- A parked key stays in append mode after such a turn, and its token list is
  the slot's, not the rendered one.
- A message that was really changed is read again from the change, as before.
- The slot's tokens are kept as far as they say the same as the prompt, which
  need not be the end of either. A reply cut off by `max_tokens` (the slot
  holds it less its last token), a reply that was cancelled, or a later
  message that was changed, is read again from where it differs, not from
  the kept reply.

- Keep a conversation's reasoning settings the same from turn to turn. A
  request the comparison does not run for (`cache_prompt` false, thinking
  switched off, the off switch) stops matching at the first kept reply and
  reads on from the checkpoint or the parked save before it.

It is off unless `LLAMA_REPLY_AS_GENERATED` in the server's environment is `1`,
`true`, `on` or `yes` (in any case), because it changes how every PS5LM server
matches a request against its slot, the app's own included. The fleet
controller sets it to `1` for the llama-server it starts; set it yourself for
one you start by hand, or set it to `0` before starting the controller to
compare. The log line is `previous reply kept as generated: the prompt differs at
N, ...`; `previous reply not kept: ...` is a warning that the patch's own last
check refused a prompt it had built, and the request was served without it.

## Control server

```sh
python3 -m fleet control            # page and JSON API on http://127.0.0.1:8090
python3 -m fleet control --build ~/PS5LM/build/fleet-host --port 8090 --config fleet/control.json
python3 -m fleet ctl status         # the same, as JSON, from a shell or an agent
```

The control server runs a fleet of any size from one JSON document,
`fleet/control.json` (ignored by git). It holds three things:

- **Devices** hold layers:
  - `ps5`: a console with PS5LM and ftpsrv.
  - `gpu`: one of this computer's GPUs.
  - `rpc`: a `ggml-rpc-server` elsewhere.

  Each device has a memory budget (or `null` for the free memory it reports), a
  reserve, and tags for picking groups. The reserve is what the planner leaves
  for llama.cpp's compute buffers: 512 MiB on a console by default (a worker
  reports its free memory with the app's own use taken off, and llama.cpp
  measured 192 to 256 MiB of compute buffers on one at up to 192k tokens), 1024
  MiB on a GPU or an RPC server.

  A console also has its worker's settings, which go into its
  `/data/PS5LM/env.txt` (PS5LM after v0.11.0 reads them; older builds ignore them):

  | Setting | Default | In env.txt |
  |---|---|---|
  | `screen` | `false`: about 350 MiB more for the model, and the TV shows the system's splash | `PS5LM_WORKER_SCREEN=on` or `off` |
  | `cache_gib` | `0`: no weight cache, as in the app. Set, up to that many GiB of the weights hosts send are kept on the console, so loading them again reads them there instead of sending them; `32` holds a 27B model's share | `PS5LM_RPC_CACHE_GB=0` |

  They are written whenever the control server makes the console a worker. A
  change restarts the console's worker when nothing uses it; a worker in use
  takes it at its next restart, and a pool's start restarts a worker whose
  env.txt it had to change first. The page shows each console's settings and
  what its cache holds.
- **Pools** are one model and how to serve it:
  - `context`: tokens per request.
  - `slots`: requests at once per replica. With `shared_context` the slots share
    one cache of `context` tokens (llama-server's `--kv-unified`).
  - `cache_type` and `flash_attn`: the cache (default `q8_0` with flash attention).

  A pool's replicas each serve the whole model:

  | Replica | Runs as |
  |---|---|
  | `{"devices": ["ps5-a", "pc-gpu"]}` | a layer split over RPC, 1 to 15 devices in layer order (`"via": "rpc"`), planned and checked like `serve` |
  | `{"devices": ["ps5-b"]}` | one console serving the model with PS5LM itself (`"via": "app"`): the model is copied to `/data/PS5LM/models` over FTP once, and `app-args.txt` sets the pool's context, slots and cache |
  | `{"devices": ["pc-gpu"]}` | one of this computer's GPUs, directly (`"via": "direct"`), checked with `llama-fit-params` first |
  | `{"url": "http://10.0.0.9:8080"}` | a server run elsewhere (another control server, a plain llama-server), only routed to |

  `"split": [40, 25]` sets a split's block counts by hand. Pools may share
  devices, so the same consoles can be laid out several ways (two-console splits
  of a big model, or one console each for a small one). Only one of those runs
  at a time: within a pool, and among running pools, a device serves one
  replica. A pool started on another's devices waits until that pool has let go
  of them.
- **Settings:**
  - `api_port` (8081) and `api_lan`: the API listens on 127.0.0.1 unless `api_lan` is set.
    `api_lan` needs an `api_key`: a configuration that shares the API on the
    network without one is refused, when it is loaded and when it is changed.
  - `api_key`: once set, every request needs it as `Authorization: Bearer` or `x-api-key`.
  - The controller's own memory, and `lan`, the network to search.
  - `launcher`: another launch payload for **launch** (below), a `.elf` in the
    launcher folder.

  Web pages on other sites can use the API only with a key. With one set, the
  API answers any origin (CORS), and each request has to carry the key. Without
  one it sends no CORS headers (a replica's own are dropped too), and refuses a
  request that carries another site's `Origin`, so a page open in your browser
  cannot drive a keyless API on 127.0.0.1. llama-server's chat page, opened
  through the API on loopback, works either way.

Raising throughput means more replicas. Fitting a bigger model or a longer
context means more devices per replica. Both work at once: 100 consoles can be
50 two-console replicas of one model. `"auto": {"count": 50, "per": 2,
"kinds": ["ps5"], "tags": ["rack1"]}` in a new pool (or `pool add --auto 50
--per 2`) picks free devices in the order they are listed.

**The API** (fleet/router.py) is one endpoint for every pool, on `api_port`:
- It takes `/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
  `/v1/embeddings`, `/completion` and the rest of llama-server's API.
- The request's `model` picks the pool: its id, an alias, or its model file's
  name. A name no pool has goes to the only ready pool, when exactly one is.
- Within a pool, the replica with the fewest requests in flight for its slots
  gets the request.
- A replica that refuses the connection is skipped, and the request goes to another.
- Responses stream through (`"stream": true`).
- `/v1/models` lists the pools; `/health` says whether any can take a request.
- Other GETs, `/` (llama-server's chat page) included, go to the first ready pool.

**Running pools are kept running.** Starting a pool (`"running": true`) starts
its replicas:
- A split makes its consoles workers when they are not (over FTP, as below),
  and restarts a console that already is one: the console's driver keeps much
  of what an earlier host allocated after it disconnects (12.12 GiB free at
  start, 7.08 GiB after one 27B session), so a second load could run out of
  memory. The restart takes 4 s, and the weights come back from the console's
  cache. It starts the GPU's RPC worker, probes the free memory, plans, checks
  with llama.cpp, and starts llama-server on a local port. The watcher leaves a
  console's port alone while a split on it starts: its RPC server takes one
  client at a time, and a probe between those connections made llama-server's
  one attempt fail.
- llama-server samples on the worker that has the output layer
  (`--backend-sampling`), so a token comes back instead of the logits, about
  1 MB a token for a 248k vocabulary. On a PS5 alone, Qwen3.8-27B went from 16.0
  to 18.4 tokens/s with the same greedy output. Requests that ask for
  log-probabilities are sampled on the host, as before.
- The control server watches every device and replica. A replica that fails,
  loses a device (a console that stops answering or leaves worker mode) or stops
  answering is marked failed. It starts again after 15 s, then 30 s, and so on
  up to 10 minutes.
- Changing what a replica serves (the model, context, cache, its devices)
  restarts only that replica. Aliases or other pools leave it alone.

**Consoles over the network.** All of this goes through ftpsrv (2121) and
PS5LM's request files, so no one has to be at the console:
- **Search** sends the PlayStation discovery query (UDP 9302) to every address of
  the LAN, one by one, so it also works from WSL's NAT. Under WSL it asks Windows
  which network that is.
- **worker** writes `/data/PS5LM/fleet-worker.txt` (its own address, port 30052
  and the acknowledgement) and sends `unload`: a running PS5LM restarts as a
  worker. When PS5LM is closed, the request is not taken within 20 s and is
  removed. The console then becomes a worker when PS5LM is next opened.
- **leave** sends `leave-worker`. PS5LM restarts as the app
  ([Configure workers](#configure-workers-only-after-building-and-validating-the-title)).
  An older PS5LM can only quit; the worker file is then removed, and it opens as
  the app from the home screen.
- **restart** sets the console's env.txt lines and sends `unload`: a worker
  restarts as a worker and reads them again; the app restarts with the model it
  serves. A worker of an older PS5LM does not take the request; it leaves worker
  mode and is made a worker again (25 s on a console).
- **clear-cache** deletes the weights the worker keeps
  (`/data/PS5LM/rpc-cache`).
- **launch** opens a closed (or crashed) PS5LM: it sends PS5LM's launch payload
  (`ps5/app/launch_app.c`, built as `build/app-ps5/launch-PPSA99575.elf`) to an
  ELF loader on the console's port 9021, which runs it. The payload comes from
  one folder, the launcher folder: `build/app-ps5`, or the one
  `--launcher-dir DIR` names when the app is built elsewhere. The `launcher`
  setting can pick another `.elf` in that folder and nothing outside it, and
  only a file that starts as an ELF is sent, so the page's API cannot send
  another file of this computer's to an address. The title opens as a worker when `fleet-worker.txt`
  is there, as the app otherwise. A split, `worker` and `restart` do the same
  when they find PS5LM closed, so a console whose title crashed comes back by
  itself on the replica's next retry. Without a loader on 9021 they say so, as
  before. On a PS5 on 11.60 the worker answered 2 s after the payload was sent.

  OnionHEN does not expose an ELF loader of its own (it keeps a private one on
  9020). One way to have one on 9021 every boot is ps5-payload-dev's elfldr as
  an OnionHEN plugin: OnionHEN starts a plugin only when it starts, and only
  with a `<id>.elf.auto_start` marker next to it, or from Toolbox, Plugins.
- **quit** closes PS5LM, and **unload** stops a model it serves.
- A device's mode comes from the network:
  - `worker`: its RPC port answers.
  - `serving` (with the model file), or `loading`: PS5LM's server answers.
  - `idle`: ftpsrv answers, but neither of those does.
  - `offline`: nothing answers.

  At most three short connection attempts, every 4 s, 32 at a time.
- A device of a running pool belongs to it: stop the pool before acting on it.

From a shell or an agent, `python3 -m fleet ctl` does all of it and answers in
JSON. It exits 1 with the refusal in `"error"`:

```sh
ctl() { python3 -m fleet ctl "$@"; }          # --url or $PS5LM_FLEET_URL for another server,
                                              # --token, $PS5LM_FLEET_TOKEN or the token file when it asks for one
ctl search --wait                             # consoles on the LAN
ctl device add ps5-a --kind ps5 --address 192.168.50.21
ctl device add pc-gpu --kind gpu --device CUDA0
ctl pool add qwen27 --model ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --context 131072 \
    --cache-type q4_0 --replica ps5-a,pc-gpu
ctl pool plan qwen27 --wait                   # blocks and memory per device, with llama.cpp's check
ctl pool start qwen27 --wait 900              # exits 0 when every replica is ready
ctl pool set qwen27 context=81920 cache_type=q8_0
ctl device leave ps5-a --wait               # after `ctl pool stop qwen27`
ctl device set ps5-a screen=true cache_gib=32 # restarts its worker if nothing uses it
ctl config > fleet.json; ctl apply fleet.json # the whole configuration; keys stay "set"
```

On 2026-11-09 this ran Qwen3.8-27B-UD-Q4_K_XL (16.6 GB) as one replica split
over one PS5 (firmware 11.60) and an RTX 3080 (10 GB) under WSL.
- **128k context:** fits with a q4_0 cache. llama.cpp projected 10,547 of 11,854
  MiB on the console (blocks 0–39) and 8,497 of 9,069 MiB on the 3080 (blocks
  40–64 and the output). The planner was within 1 MiB of it on the cache.
- **q8_0 cache:** 80k fits; 96k does not.
- **Start:** 2 minutes 21 seconds, with both devices already workers.
- **Through the API:** 14.4 tokens/s generating, and the first streamed token
  after 1.35 s.

The page listens on loopback only. It can write to consoles and start processes,
so it answers only requests whose Host header is its own (no DNS rebinding) and
changes that carry its own request header (no other site can drive it from your
browser). Under WSL, Windows reaches it at http://localhost:8090. Behind a proxy
that only your own devices reach, such as `tailscale serve`, add
`--allow-host NAME:PORT`. The workers' RPC is unauthenticated whatever the page
does: keep them on a network you trust.

With `--allow-host`, changes need a token: every POST carries it as
`Authorization: Bearer`, under the other name and on loopback alike. Reading the
state does not need it.

```sh
# A token of your own (16 to 256 characters) in the token file: the same at every
# start, and `ctl` reads it there too, from any shell or script.
mkdir -p ~/.config/ps5lm-fleet && (umask 077; openssl rand -hex 24 > ~/.config/ps5lm-fleet/token)
python3 -m fleet control --allow-host pc.tailnet.ts.net:8090
python3 -m fleet ctl pool start qwen27
# Or given: --token or $PS5LM_FLEET_TOKEN, to the control server and to ctl.
PS5LM_FLEET_TOKEN="$(cat ~/.ps5lm-fleet-token)" python3 -m fleet control --allow-host pc.tailnet.ts.net:8090
# Or none: one is made at each start and printed once.
python3 -m fleet control --allow-host pc.tailnet.ts.net:8090
#   ps5lm-fleet control: changes need this token: Zk3...
PS5LM_FLEET_TOKEN=Zk3... python3 -m fleet ctl pool start qwen27    # or ctl --token
```

The page asks for the token at the first change and keeps it in that browser.
The token is `--token`, else `$PS5LM_FLEET_TOKEN`, else the first line of the
token file: `~/.config/ps5lm-fleet/token` (under `$XDG_CONFIG_HOME` when set),
or `--token-file FILE`. The file is refused when other users can read it
(`chmod 600`). A token without `--allow-host` is asked for on loopback too.
Without a token or `--allow-host`, the page is as before: loopback only, no token.
A model given as a path is copied to a console only when the file is a GGUF.

## Tests / CI

```sh
scripts/test-fleet.sh              # offline Python, C++ config/runtime, existing planner
python3 -m pip install -r tests/requirements-fleet.txt # fixture only; use a venv
scripts/test-fleet.sh --rpc        # build + real localhost RPC regression/inference
```

The smoke creates a **231 KB, deterministic, untrained** four-layer GGUF. It runs
a local CPU reference, two CPU RPC processes, and the actual `python -m fleet
serve` CLI. It checks token IDs, assigned layers and per-worker computation.
This proves a real distributed execution path on host CPUs, not useful language
quality, PS5 GPU operation, large-model scaling or network speed. Tests use
loopback only and clean up their child processes.

`tests/fleet_rpc_park_test.py` covers parked state the same way (a CPU worker
with a park folder): a swap, five appending firings, several keys, restarts, a
worker killed in a save, a changed history, the cap, a worker without parking,
and the worker's memory. With `PS5LM_PARK_TEST_MODEL` set to a GGUF it runs a
second time with that model: Qwen3.5-0.8B has the attention cells plus
recurrent state of the models the consoles serve, which the untrained fixture
(attention only) has not.

`tests/server_reply_as_generated_test.py` gives a slot a reply token by token
and sends it back as a chat message. Kept: split into other tokens, other
whitespace before and after the marker that closes the thinking, with a tool
call, sent back by a streaming request, with two more turns on top, a later
reply that was cut off; and what the model writes after a kept reply is what
it writes from the same tokens read afresh. Read again: whitespace at a
second marker in the content, two spaces for one in the content and in a tool
argument, the marker's characters as plain text, a later reply that comes
back changed or was cancelled, three real edits, a prompt that would not fit
the context, the patch's last check refusing, and the off switch's four
spellings. It uses an untrained recurrent fixture with one token per letter,
and the named model as well when `PS5LM_PARK_TEST_MODEL` is set. The park
test's `kept` case does the same through a parked save and restore.

`.github/workflows/fleet-host.yml` runs these checks on pull requests and pushes
on `main` or `fleet/**`, using ordinary Ubuntu hosted runners, read-only
repository permissions and per-ref cancellation. The host job has two parallel
compiler processes and a 25-minute timeout; the separate native-object/runtime
job has a 15-minute timeout. Full title CI uses a digest-pinned official Debian
container on an ordinary Ubuntu runner, a two-CPU quota and a 45-minute timeout.
Actions are pinned to commits; fixture dependencies are version-pinned.
Jobs check out the pinned submodules they need and install public build tools.
The host job downloads no trained model, PS5 SDK or driver. The native job does
fetch the pinned public homebrew SDK and source dependencies, but no proprietary
firmware, trained model, OpenGL SDK or RADV driver. The full-title job also
builds patched RADV, downloads the checksum-verified OpenGL SDK and public
ps5-exporter payload for packaging, and builds presentation assets. No secrets,
`pull_request_target`, private runners or console are required.

The inspected upstream revision had no checked-in workflow to reuse. This new
workflow separates host inference, native-object compilation and full title
cross-building. All three remain separate from the hardware gate. See the
validation report for local evidence and the PR checks for hosted results.
