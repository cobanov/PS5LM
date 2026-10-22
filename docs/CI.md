# CI validation

[`native-build.yml`](../.github/workflows/native-build.yml) checks the existing app
on pull requests and pushes to `main` or `ci/**`. It uses ordinary Ubuntu hosted
runners, a read-only token, no secrets, and no console connection. Superseded runs
are cancelled. There are two independent jobs:

- **Host checks** (5-minute limit): shell/Python syntax, the existing synthetic
  GGUF `model_plan_test.cpp`, and regressions for dashboard compiler job limits
  and propagation of failed background compilers. No submodule or model download
  is needed for these checks.
- **Full PS5 title cross-build** (45-minute limit, two compiler jobs): bootstrap
  the public SDK, reproduce clean-room `libc.prx`, build patched RADV and the
  existing llama-server/native dashboard, package the title folder, and inspect
  its ELF/FSELF structure and integrity. It does not execute the resulting title.

The second job downloads build tools, source dependencies and checksum-verified
public SDK archives. It does **not** download a trained model or proprietary
console firmware. Only the optional browser UI is omitted: its upstream fallback
fetches a floating `latest` release. The native console dashboard is built.
Normal `scripts/build-app.sh` builds keep the prebuilt browser UI enabled;
`PS5LM_PREBUILT_UI=OFF` explicitly opts out and clears legacy CMake aliases.

## Reproduce the checks

Host checks need Bash, Python 3 and a C++17 compiler:

```sh
for script in scripts/*.sh ps5/app/ps5cc ps5/app/ps5cxx; do bash -n "$script"; done
PYTHONPYCACHEPREFIX=build/pycache python3 -m py_compile scripts/*.py
PYTHONPYCACHEPREFIX=build/pycache python3 -m unittest discover -s tests -p 'test_build_app.py' -v
mkdir -p build
c++ -std=c++17 -O1 ps5/app/model_plan_test.cpp ps5/app/model_plan.cpp -o build/model_plan_test
build/model_plan_test
```

For the full build, use the digest-pinned Debian trixie container and exact apt
package list in the workflow, or an x86-64 Linux host with equivalent tools:

```sh
git submodule update --init --depth 1
export PATH=/usr/lib/llvm-19/bin:$PATH
LLVM_CONFIG=/usr/bin/llvm-config-19 BUILD_JOBS=2 scripts/test-native-build.sh
```

The script keeps dependencies in `.deps/native-ci-src` (override with
`PS5LM_NATIVE_SOURCES`) and validates existing dependency revisions before reuse.
It uses upstream's checked SDK setup and `make libc`, which rebuilds the runtime
twice and compares the recorded raw/signed hashes. That target also produces the
same host packaging tool required by the app, without building an unrelated demo
title. It applies this repository's three Mesa patches with fixed commit identity
and dates. The removed OpenGL release is fetched from the mirror already
documented in [BUILDING.md](BUILDING.md), checked against the original archive
hash, then checked by the pinned UI fetcher's file manifest.

| Input | Pin |
|---|---|
| Debian container | `trixie-20261020-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad` |
| PS5_Vulkan | `f410e67a0241b3e8edcdbcd2a9b285dfc4a412ed` |
| PS5_PayloadSDK fork | `b83202be73e930050e00de0f1b4d9ec46c0391bf` |
| Mesa base | `1ba9874f478657ccf6da72ce55f0260164cc0bd0` |
| Mesa after this repository's patches | `9e0a7c63bc6d4b6bcb60831163bb3fe9f46e91ea` |
| llama.cpp / UI | The checkout's gitlink revisions (verified before building) |
| OpenGL SDK archive SHA-256 | `f93643c04c843d56143b00951df1f8042ea7706ae9f19e4e9abf158e1ead77c5` |

`build/native-ci/result.json` records the source pins and SHA-256 hashes of the
packaged files after successful inspections. The title is in
`build/app-ps5/title/PPSA99575`. A failed check exits nonzero; there is no automatic
deployment, release publication or binary execution. CI publishes logs, not a
release package.

## What a green result means

It proves the checked source can build/link/package with these public tools and
pass static ELF/FSELF checks, including the native UI and Vulkan driver build.
It does **not** prove the title boots on a PS5, the GPU executes inference, models
fit memory, or networking/performance works on hardware. A generic QEMU x86-64
guest cannot supply the PS5 loader, system services or AGC GPU interface; it would
not close that gap. Hardware smoke tests remain separate.

Source pins and archive checksums make the build inputs explicit, but apt package
updates and build paths/compiler versions can change title bytes. Only the
upstream clean-room runtime has a fixed byte-reproduction check; the title hashes
identify a particular build, not a promised reproducible release binary.
