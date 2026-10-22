# Experimental fleet: one model, several PS5 workers

**Status: implemented host controller and opt-in PS5 worker source; CPU RPC
inference tested, PS5 title cross-build and real consoles not validated.** This
is not a ready-to-install fleet release or a performance claim.

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
- `GGML_RPC_NO_COMM=1` is forced by this integration. Patch 0010 makes the server
  reject `COMM_INIT` before opening the extra wildcard peer listener in upstream
  tensor-parallel code. This fleet uses layer splitting, not those collectives.
- Patch 0009 initializes BSD socket lengths and suppresses SIGPIPE portably.
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

Python 3.10+, a C++17 compiler, CMake, Ninja and Git are required. NumPy is needed
only to generate the tiny inference test fixture, not for the controller.

```sh
git submodule update --init --depth 1 third_party/llama.cpp
scripts/build-fleet-host.sh
# build/fleet-host/bin/{llama-server,ggml-rpc-server,...}
# build/fleet-host/fleet-build.json (revision, patches, binary SHA256)
```

The host script applies the repository's numbered llama.cpp patches, disables
GPU backends and RDMA, and builds the RPC tools plus the upstream multi-server
regression. It does not fetch model weights or start a listener. Set `CMAKE`,
`NINJA` or `CMAKE_BUILD_PARALLEL_LEVEL` for local tool locations/resources.

### PS5 worker title — source integration, uncompiled here

Follow [BUILDING.md](BUILDING.md) to prepare PS5_Vulkan, its pinned PS5_Mesa and
PS5_PayloadSDK forks, patched RADV, native packaging tool, `libc.prx` and the
checksum-pinned ps5-opengl SDK. Then:

```sh
git submodule update --init --depth 1
scripts/build-app.sh
```

The existing title build now links `GGML_RPC=ON`, `GGML_RPC_RDMA=OFF` and
`fleet_worker.cpp`. No change to the legacy payload build is implied. A payload
is not the same memory environment as the native Vulkan title.

The ordinary build currently points at a removed ps5-opengl release. The
repository already documents the archive mirror and checksum check in
[BUILDING.md](BUILDING.md); see [FLEET-VALIDATION.md](FLEET-VALIDATION.md) for the
actual build attempt and missing prerequisite evidence. Host CI cannot recreate
Sony's runtime, direct-memory pool, RADV driver or console networking.

## Configure workers (only after building and validating the title)

On each console, explicitly create `/data/PS5LM/fleet-worker.txt` containing
**that console's own private IP**, port 1024–65535, and the exact acknowledgement:

```text
192.168.50.21:50052
I_UNDERSTAND_RPC_IS_UNAUTHENTICATED
```

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
python3 -m pip install numpy==2.2.6 # test fixture only, preferably in a venv
scripts/test-fleet.sh --rpc        # build + real localhost RPC regression/inference
```

The smoke creates a **231 KB, deterministic, untrained** four-layer GGUF. It runs
a local CPU reference, two CPU RPC processes, and the actual `python -m fleet
serve` CLI. It checks token IDs, assigned layers and per-worker computation.
This proves a real distributed execution path on host CPUs, not useful language
quality, PS5 GPU operation, large-model scaling or network speed. Tests use
loopback only and clean up their child processes.

`.github/workflows/fleet-host.yml` runs these checks on pull requests and pushes
on `main` or `fleet/**`, using an ordinary Ubuntu hosted runner, two parallel compiler processes,
a 25-minute timeout, read-only repository permissions and per-ref cancellation.
Actions are pinned to commits; the NumPy fixture dependency is version-pinned.
The workflow checks out the pinned llama.cpp submodule and installs build
prerequisites, but downloads no trained model, PS5 SDK or driver. No secrets,
`pull_request_target`, private runners or console are required.

The inspected upstream revision had no checked-in workflow to reuse. This new
job deliberately gives host confidence only; the PS5 title still requires the
separate build and hardware gate. See the validation report for local evidence
and the PR checks for the current hosted result.
