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

A worker takes two request files, checked once a second: `/data/PS5LM/quit`
closes the title, and `/data/PS5LM/leave-worker` (PS5LM after v0.11.0) deletes
`fleet-worker.txt` and restarts the title the way a model change does, so it comes
back as the app without anyone at the console. On the 2026-11-09 console run
(one PS5 on firmware 11.60), worker mode took 3 s to start, and the app was back and
taking requests 6 s after `leave-worker`.

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
  | `cache_gib` | `32`: up to 32 GiB of the weights hosts send are kept on the console, so loading them again reads them there instead of sending them; `0` for none | `PS5LM_RPC_CACHE_GB=32` |

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
  - `api_key`: once set, every request needs it as `Authorization: Bearer` or `x-api-key`.
  - The controller's own memory, and `lan`, the network to search.

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
- A split makes its consoles workers when they are not (over FTP, as below) and
  starts the GPU's RPC worker. It probes their free memory, plans, checks with
  llama.cpp, and starts llama-server on a local port.
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
ctl() { python3 -m fleet ctl "$@"; }          # --url or $PS5LM_FLEET_URL for another server
ctl search --wait                             # consoles on the LAN
ctl device add ps5-a --kind ps5 --address 192.168.50.21
ctl device add pc-gpu --kind gpu --device CUDA0
ctl pool add qwen27 --model ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --context 131072 \
    --cache-type q4_0 --replica ps5-a,pc-gpu
ctl pool plan qwen27 --wait                   # blocks and memory per device, with llama.cpp's check
ctl pool start qwen27 --wait 900              # exits 0 when every replica is ready
ctl pool set qwen27 context=81920 cache_type=q8_0
ctl device leave ps5-a --wait               # after `ctl pool stop qwen27`
ctl device set ps5-a screen=true cache_gib=64 # restarts its worker if nothing uses it
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
