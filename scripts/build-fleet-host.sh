#!/usr/bin/env bash
# Build the CPU-only host/controller and localhost RPC test workers.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
llama="$root/third_party/llama.cpp"
out="${PS5LM_FLEET_BUILD:-$root/build/fleet-host}"
cmake_bin="${CMAKE:-cmake}"
ninja_bin="${NINJA:-ninja}"
for tool in "$cmake_bin" "$ninja_bin" python3 git; do
    command -v "$tool" >/dev/null || { echo "build-fleet-host: missing $tool" >&2; exit 2; }
done
if [ ! -f "$llama/CMakeLists.txt" ]; then
    echo 'build-fleet-host: run git submodule update --init third_party/llama.cpp' >&2
    exit 2
fi
expected=$(git -C "$root" ls-tree HEAD third_party/llama.cpp | awk '{print $3}')
actual=$(git -C "$llama" rev-parse HEAD)
if [ "$actual" != "$expected" ]; then
    echo "build-fleet-host: llama.cpp revision $actual differs from pinned $expected" >&2
    exit 2
fi
# Keep the same patch set on the controller and the fleet title (PS5LM_FLEET=1
# scripts/build-app.sh), with no protocol fork.
for patch in "$root"/patches/*.patch "$root"/patches/fleet/*.patch; do
    git -C "$llama" apply --reverse --check "$patch" 2>/dev/null || git -C "$llama" apply "$patch"
done
# patches/0007's request hooks are weak and only the PS5 title defines them; ELF
# linkers leave them null, Apple's ld must be told they may stay undefined.
link_flags=()
if [ "$(uname -s)" = Darwin ]; then
    link_flags=("-DCMAKE_EXE_LINKER_FLAGS=-Wl,-U,_ps5lm_activity_begin -Wl,-U,_ps5lm_activity_data -Wl,-U,_ps5lm_activity_end")
fi
"$cmake_bin" -S "$llama" -B "$out" -G Ninja "${link_flags[@]}" \
    -DCMAKE_MAKE_PROGRAM="$(command -v "$ninja_bin")" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF -DGGML_BACKEND_DL=OFF \
    -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF \
    -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
"$cmake_bin" --build "$out" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}" \
    --target llama-server llama-cli ggml-rpc-server test-rpc-multi-server
python3 - "$out" "$actual" "$(git -C "$root" rev-parse HEAD)" "$root" <<'PY_MANIFEST'
import hashlib
import json
from pathlib import Path
import sys
out = Path(sys.argv[1]).resolve()
(out / "fleet-build.json").write_text(json.dumps({
    "llama_revision": sys.argv[2],
    "ps5lm_revision": sys.argv[3],
    "binary": str(out / "bin/llama-server"),
    "server_sha256": hashlib.sha256((out / "bin/llama-server").read_bytes()).hexdigest(),
    "rpc_binary": str(out / "bin/ggml-rpc-server"),
    "backends": ["CPU", "RPC"],
    "applied_patches": {str(p.relative_to(Path(sys.argv[4]) / "patches")): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((Path(sys.argv[4]) / "patches").glob("*.patch"))
                        + sorted((Path(sys.argv[4]) / "patches/fleet").glob("*.patch"))},
}, indent=2) + "\n")
PY_MANIFEST
printf 'build-fleet-host: %s (llama.cpp %s)\n' "$out/bin" "$actual"
