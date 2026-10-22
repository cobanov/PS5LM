# Fleet validation — 2026-10-21–22

## Result

A working host-side increment: **one model split over two real CPU RPC processes**,
including the actual `python3 -m fleet serve` path. Experimental PS5 worker source
is integrated. **The complete RPC-enabled native PS5 title now cross-builds and
passes static ELF/FSELF validation. No console inference has been tested.**
At the time of the initial local validation below, no public listener or console
deployment was performed. Hosted CI results are reported separately in the PR
checks. This is not a hardware-validated PS5 release.

## Revisions

| Component | Commit |
|---|---|
| PS5LM base | `c4595784ce75c1d58777b11ef812d3e087e74296` |
| llama.cpp | `04f7226a2c12c7f382c545a6db5a034266aa53cb` |
| ps5-homebrew-ui | `bf44d80384d5b8f8f4f213cf98d748887211328d` |

All existing llama.cpp patches 0001–0008 and new patches 0009–0011 were applied.
`fleet-build.json` records the exact patch hashes and local host binary checksum.
The wire protocol remains upstream RPC 8.0.0; a handshake is not revision
attestation. Both ends must be built from the same sources.

## Passed

Final aggregate command (Linux sandbox, no system-wide package installation):

```sh
CMAKE="$PWD/.deps/host-tools/cmake/data/bin/cmake" \
NINJA="$PWD/.deps/host-tools/bin/ninja" \
scripts/test-fleet.sh --rpc
```

Public CMake 4.4.4 and Ninja 1.13.2 wheels were installed under `.deps/host-tools`
because those host tools were absent. On a normal configured system the command
is simply `scripts/test-fleet.sh --rpc`. NumPy and PyYAML supply fixture generation only.

| Check | Outcome / scope |
|---|---|
| Python unit/security suite | **52 passed**: GGUF parsing, unsafe/unknown layouts, download policy/framing/cleanup (mocked HTTPS), budgets, KV, split mapping, preflight failures, binary verification, SIGTERM child cleanup |
| Worker configuration | **43 checks passed**, no listener |
| Worker runtime/file tests | **19 checks passed**, fake ggml backend, no GPU/socket |
| Worker sanitizers | ASan + UBSan passed; LeakSanitizer disabled because sandbox ptrace prevents it |
| Worker compile | Linux C++17 compile against real pinned ggml headers with `-Wall -Wextra -Werror` |
| Existing model planner | Original `model_plan_test` synthetic-header test passed |
| CPU/RPC build | llama-server, llama-cli, ggml-rpc-server, test-rpc-multi-server built |
| Upstream RPC regression | `test-rpc-multi-server` **passed** |
| Server-side NO_COMM regression | Rank 0 and rank 1 peer commands refused; ordinary device-count RPC remained usable; test ports loopback/reserved |
| Real two-worker inference | **Passed**, not mocked; actual tensor allocation and graph execution on both processes |
| Actual fleet CLI | **Passed** manifest → preflight → plan → llama-server → inference with new workers |
| Cleanliness | `git diff --check`, shell/Python syntax checks passed; no leftover llama-server/RPC test processes |

### Inference evidence

Fixture: 231,136-byte, four-block dense `llama` GGUF with deterministic random
**untrained** weights, 32-dimensional embeddings, 4 query heads, 2 KV heads,
256-token model context. It is generated locally by `tests/make_fleet_fixture.py`;
no trained model was downloaded. Tests used context 128, one sequence, FP16 KV,
two CPU RPC workers, and tensor-split `3,7`.

Verbose direct-RPC logs assign:

| Logical layer | Device |
|---|---|
| Blocks 0, 1 | RPC0 (worker A) |
| Blocks 2, 3 | RPC1 (worker B) |
| Output slot 4 | RPC1 |

Each direct worker made 3 buffer allocations and 9 graph compute/recompute calls.
The local reference, direct two-worker execution, and actual fleet CLI all
returned token IDs:

```text
282, 131, 193, 131, 193, 131, 193, 131
```

All 40 top-five log-probability values per run matched the local baseline, with
maximum absolute error **0.0** here (test tolerance `1e-5`). This proves the tested
execution path, not useful text quality, larger-model correctness or performance.

Evidence is in `build/fleet-final-validation.log`, `build/fleet-smoke/result.json`,
its worker/controller logs and JSON responses, plus
`build/fleet-host/fleet-build.json`. Re-running the scripts regenerates this
evidence locally; generated models, binaries, logs and SDK archives are not
committed to the repository.

## CI preparation — 2026-10-22

Re-ran `scripts/test-fleet.sh --rpc` successfully after adding the bounded
GitHub Actions workflow: the same 52 Python, 43 configuration and 19 runtime
checks passed, as did both RPC regressions and the actual fleet CLI inference
comparison (maximum log-probability error 0.0). The workflow passed actionlint
v1.7.7 and YAML permission/event checks. Actions use verified commit pins;
the fixture dependencies are pinned in `tests/requirements-fleet.txt`. This is local pre-publication evidence;
refer to the PR checks for the hosted run.

### Hosted dependency correction — 2026-10-22

The first fork run built all C++ targets and passed the safety/RPC regression
checks, but fixture generation exposed a missing transitive `gguf` import:
PyYAML. The workflow now installs a pinned fixture requirements file, and the
test script checks imports before compiling. The complete suite was re-run
successfully in a fresh Python virtual environment containing only those
requirements; both inference comparisons again had maximum error 0.0. See the
PR/fork Actions run for the hosted result on the current commit.

## Initial full PS5 title build attempt — 2026-10-21

The upstream tree had no checked-in CI workflow. Its documented route is a Linux
cross-build using PS5_Vulkan, its pinned forks, RADV and homebrew SDK tooling.
The workflow now separates host inference, native-object compilation and the
full-title cross-build. The first two jobs alone do not prove a title build. The evidence below records the initial local
full-title build attempt; the SDK/runtime gaps were subsequently resolved for
the native-object gate below.

1. Ran `bash scripts/build-app.sh`: stopped because UI submodule was uninitialized.
2. Initialized its pinned commit and read its `AGENTS.md`; reran the official build.
3. The upstream OpenGL SDK endpoint returned **HTTP 404**:
   `https://github.com/blackbearreloaded/ps5-opengl/releases/download/v1.0.0/ps5-opengl-sdk-1.0.0.tar.gz`
4. Followed the alternative already documented in `docs/BUILDING.md`:
   `https://github.com/MyLuxy/overdrive-deps/releases/download/ps5-opengl-v1.0.0/ps5-opengl-sdk-1.0.0.tar.gz`
5. Downloaded exactly 226,263,775 bytes, bounded by 250 MB / 120 seconds. Computed
   archive SHA-256 matched the submodule's pinned value:
   `f93643c04c843d56143b00951df1f8042ea7706ae9f19e4e9abf158e1ead77c5`.
6. Ran official `fetch-opengl-sdk.sh`: extraction and pinned manifest verification
   passed (manifest SHA-256
   `f4b91f672be037fbac3f82494f1225deaf4c227a03f37ac3ffa56abb213b943f`).
7. Reran official `scripts/build-app.sh`: **exit 2**, exact next blocker:
   `.deps/src/PS5_Vulkan/.deps/native/radv-release/lib/libvulkan_radeon.ps5.a` missing.

At that initial attempt, also absent under `.deps/src/PS5_Vulkan`:
`build/host/ps5-native-tool`,
`runtime/libc.prx`, `.deps/native/ps5-payload-sdk/bin/prospero-lld` and its C++
headers. The compiler did not run in that initial full-title attempt. Downloading
the standalone legacy payload SDK
would not replace the native Vulkan title dependencies. No unverified PS5 binary
is included.

Reproduction: follow `docs/BUILDING.md` to prepare the pinned PS5_Vulkan/Mesa/SDK
sources and host packages, apply `patches/mesa`, run PS5_Vulkan's
`tools/setup-native-dependencies.sh`, `tools/build-radv.sh release` and `make app`,
then run this repository's `scripts/build-app.sh`. The verified OpenGL mirror
resolves the removed archive, not the remaining RADV setup.

Build evidence: `build/fleet-ps5-build-attempt*.log`,
`build/fleet-opengl-mirror-download.log`, `build/fleet-opengl-mirror-verify.log`.

## Native-target follow-up — 2026-10-22

Using the public documented bootstrap (not a proprietary SDK or firmware dump):

| Component | Pinned input |
|---|---|
| PS5_Vulkan | `f410e67a0241b3e8edcdbcd2a9b285dfc4a412ed` |
| PS5_PayloadSDK fork | `b83202be73e930050e00de0f1b4d9ec46c0391bf` |
| Base homebrew SDK v0.42 archive SHA-256 | `8cfbc7cd5811e719eb4f0c47eea668d3dc7b40bc8ab11c4a5031d40c23ec02da` |
| zlib 1.3.2 archive SHA-256 | `bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16` |

`fleet_worker.cpp`, patched `ggml-rpc.cpp`, and patched `transport.cpp` compile
successfully through PS5LM's actual `ps5cxx` wrapper. Target macros include
`__PROSPERO__`, `__SCE__` and `__FreeBSD__=9`; outputs are x86-64 relocatable ELF
objects, not executable PS5 titles. The worker's compile uses `-Werror`.

The clean-room runtime reproducer builds two copies and verifies upstream's
recorded hashes, including signed `libc.prx`:
`e6ff45d16adf687855cc3b33b0c8a4132b6504360b221e0a34c7e99fb3ba0036`.
No Sony implementation or console dump is downloaded. Local Clang 19 validation
passed; the repeatable `scripts/test-fleet-native.sh` also passed from a fresh
dependency cache using Clang 18.1.8, which is the object-only CI compiler family.
The separate native job's status is in the fork Actions checks.

The object-only check must not be described as a full build, console boot,
Vulkan validation or PS5 emulation. The full-title follow-up below is separate. [QEMU's
documented system/user modes](FLEET.md#ps5-target-compile-and-runtime-check-no-console)
do not provide the PS5-specific AGC/loader/direct-memory environment, so a generic
Linux/FreeBSD VM is not an equivalent hardware gate.

## Full native title cross-build — 2026-10-22

`scripts/test-fleet-title.sh` passed locally using public Debian 13/Clang 19
packages and the documented upstream build scripts:

- Mesa base `1ba9874f478657ccf6da72ce55f0260164cc0bd0` plus all three `patches/mesa`
  commits yields deterministic source revision
  `9e0a7c63bc6d4b6bcb60831163bb3fe9f46e91ea`.
- All 389 host shader-tool tasks and 794 PS5 RADV tasks completed; the archive
  includes the actual PS5 winsys, not a stub or host Vulkan replacement.
- The real title CMake build exposed `strnlen` hidden by `_XOPEN_SOURCE=600`.
  Patch 0011 replaces it with the equivalent bounded C++ scan. The faster native
  check now uses those actual CMake feature-test macros, so it catches regressions.
- The patched Vulkan/RPC/server libraries and native console dashboard compiled
  and linked into a complete title folder. Optional downloaded browser assets
  are omitted, not the console UI. No trained model or proprietary firmware
  was downloaded.
- `eboot.elf` and `eboot.bin` both pass upstream `tools/inspect.sh` with zero
  static errors: FreeBSD OSABI 9, ABI 2, type `0xfe10`, aligned LOADs and required
  process metadata. The native tool's FSELF check also reports valid integrity.

The first local passing `eboot.bin` was 100,329,398 bytes, SHA-256
`58a2bd3aec133b8d1bffe7a63f534fb3ff8271857ca3bf74b595055b263a0317`.
That identifies this local artifact, not a promise of bit-identical builds across
paths/compiler versions. Only the upstream clean-room runtime has a fixed
reproduction checksum. Package binaries, SDKs and logs are not committed.

Evidence: `build/fleet-native-full/title-validation.log` and
`build/fleet-title/result.json`. Reproduce using the package list in the
workflow's digest-pinned Debian `ps5-title` job and `scripts/test-fleet-title.sh`.
The fresh hosted run, rather than this local cache-backed aggregate, establishes
whether the entire CI bootstrap works from scratch; see fork Actions results.

This is a real cross-build/package check, **not a console boot or hardware test**.

## Review fixes and remaining boundaries

Independent review plus real host execution found and corrected:
- Missing binary checksum in the host manifest.
- Wrapper SIGTERM orphaning the server.
- 16-node plans exceeding the pinned CLI's accepted split count (now max 15).
- Unmodeled shared RoPE weight duplication (those layouts now rejected).
- Confusing RPC buffer names with CLI device names (actual names RPC0, RPC1).
- Server peer-communication commands opening an additional wildcard listener
  (server-side NO_COMM rejection plus forced worker/controller setting).
- PS5/BSD socket `sin_len` initialization and SIGPIPE portability concerns.

Not verified: actual console boot/runtime import resolution, driver/device memory allocations,
Vulkan kernels, actual worker-screen rendering, real network drops on PS5,
thermal behaviour, sustained inference, large GGUF models or multi-console
performance. Fresh HTTPS GGUF acquisition is covered by offline downloader
tests, not a live download/inference integration run. The OpenGL SDK download is
separate and does not validate the model acquisition path.

Qwen 3.8 27B is **not supported by this fleet estimator**: PS5LM lists its GGUF
architecture as `qwen35`, whose hybrid/recurrent memory model is intentionally not
guessed. See `FLEET.md` for the narrow supported layouts and extension path.

## Next hardware gate

With a title built as above: use an isolated wired network, two consoles,
one controller and a small admitted dense model. Verify both RPC handshakes,
actual per-device allocation logs, identical greedy output against a local
reference, memory high-water marks, disconnect/close recovery and title restarts
between models. Only then test a model too large for one console; use measured
memory reserves and benchmark prefill/decode separately. Nothing in the current
host results substitutes for that gate.
