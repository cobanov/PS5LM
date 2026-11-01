# Experimental fleet: one model, several PS5 workers

**Status: host controller and an opt-in PS5 worker in a separate fleet build of
the title; CPU RPC inference and the fleet title's cross-build are tested in CI.
On hardware, one PS5 worker has served one model together with a Linux CPU
worker (2026-11-01, below); more consoles at once are not validated by the
maintainers.** This is not a performance claim. The default
PS5LM title has no RPC code; releases ship the fleet build as its own zip,
`ps5lm-app-fleet-vX.Y.Z.zip`.

What it is for: a model that does not fit in one console's memory runs across
several, each holding a run of its layers. Every token passes through every
worker over the network, so it is not faster than one console; it makes larger
models possible. Workers are PS5s; a PC can also join with its CPU (the host
build has no GPU backend, so a PC's GPU cannot be a worker yet).

## Quick start

1. **On each PS5:** install `ps5lm-app-fleet-vX.Y.Z.zip` from the
   [releases](https://github.com/cobanov/PS5LM/releases) like the normal app,
   then put `/data/PS5LM/fleet-worker.txt` on it over FTP: the console's own IP
   and a port below 40000, then the acknowledgement line
   ([Configure workers](#configure-workers-only-after-building-and-validating-the-title)).
   Start PS5LM: it shows the worker screen instead of the dashboard.
2. **On a Linux or macOS PC on the same network:** `git submodule update --init
   --depth 1 third_party/llama.cpp`, then `scripts/build-fleet-host.sh`
   ([Host](#host-cpu--rpc)).
3. **Nodes:** copy `fleet/nodes.example.json` and list each console's
   `IP:port` with its memory budget.
4. **Plan, then serve:** `python3 -m fleet plan --model model.gguf --nodes
   nodes.json` shows which layers go where; `python3 -m fleet serve ...
   --ack-insecure-rpc` starts one llama-server on `127.0.0.1:8081` of the PC
   ([Plan and serve](#plan-and-serve-from-the-controller)).
5. **Back to normal:** delete `fleet-worker.txt` and restart PS5LM, or install
   the normal zip again.

Dense `llama` and `qwen2` GGUFs only; Qwen 3.5/3.8 (`qwen35`) is refused. RPC
has no authentication: keep workers on a network you trust. On macOS, the app
that runs the controller (Terminal, iTerm, a multiplexer) needs Local Network
permission, or connections to the consoles fail with "No route to host".

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

### Deliberately narrow model admission

- 2–15 selected workers, exactly one device per endpoint, at least one repeating
  layer per worker. The pinned CLI rejects 16 tensor-split entries.
- Complete, single-file, little-endian GGUF v3, **dense `llama` or `qwen2`**
  layouts, at most 512 layers, known tensor encodings and validated shapes.
- One sequence, FP16 K/V cache, fixed context no greater than training context.
  KV is rounded up to 256 tokens conservatively. Flash attention, automatic fit,
  lazy loading and opportunistic host-operation offload are disabled.
- Unknown architectures, hybrid/recurrent state, MoE, split GGUF, missing tensors,
  unsupported quantizations and replicated RoPE weight tensors fail closed.
- Budgets include exact inspected weight spans, tied-output duplication, estimated
  per-layer KV and operator-selected reserves. Reserves must cover allocations,
  dequantization, activation/compute buffers and backend overhead. A successful
  plan **is not an allocation guarantee or reservation**. Live memory can change.

**Why not the README's Qwen 3.8 27B yet?** `docs/MODELS.md` identifies that model's
architecture as `qwen35`. PS5LM's normal single-console planner has broader
architecture support than this first fleet estimator. The fleet does not yet
model Qwen35's hybrid/recurrent state and placement, so it rejects that model
rather than extrapolating a dense-transformer KV formula. Next step is a
Qwen35-specific inspector/placement test against the pinned loader, followed by
small real RPC inference and two-console validation. This is not a claim that
upstream RPC itself cannot run it.

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
`04f7226a2c12c7f382c545a6db5a034266aa53cb` and UI
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
`patches/fleet`, disables
GPU backends and RDMA, and builds the RPC tools plus the upstream multi-server
regression. It does not fetch model weights or start a listener. Set `CMAKE`,
`NINJA` or `CMAKE_BUILD_PARALLEL_LEVEL` for local tool locations/resources.

### PS5 worker title — the fleet build

Follow [BUILDING.md](BUILDING.md) to prepare PS5_Vulkan, its pinned PS5_Mesa and
PS5_PayloadSDK forks, patched RADV, native packaging tool, `libc.prx` and the
checksum-pinned ps5-opengl SDK. Then:

```sh
git submodule update --init --depth 1
PS5LM_FLEET=1 scripts/build-app.sh
```

`PS5LM_FLEET=1` is what makes it the fleet build: llama.cpp with `GGML_RPC=ON`
(`GGML_RPC_RDMA=OFF`) and `patches/fleet` on top of the numbered patches, and
the title with `fleet_worker.cpp` and the worker's startup mode in `main.cpp`.
Without it, `scripts/build-app.sh` builds the default title with none of that,
and takes `patches/fleet` back out of the llama.cpp checkout if a fleet build
left it there. Both variants have the same title ID, so installing one replaces
the other; settings and models in `/data/PS5LM` stay. `scripts/release-app.sh`
names the zip after the variant the last build made.

CI builds both variants on every pull request: `native-build.yml` runs
`scripts/test-native-build.sh` once as is and once with `PS5LM_FLEET=1`, each
with the structural ELF/FSELF checks ([CI.md](CI.md)). That proves the fleet
title links, not that it boots, that its Vulkan results are right, or that a
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
