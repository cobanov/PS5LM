#!/usr/bin/env bash
# PS5-target object compilation + clean-room runtime reproduction, NOT a title
# link, console execution, Vulkan validation, or PS5 emulation.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
sources="${PS5LM_NATIVE_SOURCES:-$root/.deps/fleet-native-src}"
out="$root/build/fleet-native"
vk="$sources/PS5_Vulkan"
sdk_fork="$sources/PS5_PayloadSDK"
vk_revision=f410e67a0241b3e8edcdbcd2a9b285dfc4a412ed
sdk_revision=b83202be73e930050e00de0f1b4d9ec46c0391bf
llama="$root/third_party/llama.cpp"
export BUILD_JOBS="${BUILD_JOBS:-2}"
export LLVM_CONFIG="${LLVM_CONFIG:-llvm-config}"
mkdir -p "$out"
rm -f "$out/result.json"

for tool in git clang clang++ "$LLVM_CONFIG" make wget unzip sha256sum timeout python3; do
    command -v "$tool" >/dev/null || { echo "native-check: missing $tool" >&2; exit 2; }
done
[[ -f "$llama/CMakeLists.txt" ]] || {
    echo 'native-check: initialize the pinned llama.cpp submodule first' >&2; exit 2;
}
expected=$(git ls-tree HEAD third_party/llama.cpp | awk '{print $3}')
[[ $(git -C "$llama" rev-parse HEAD) == "$expected" ]] || {
    echo 'native-check: llama.cpp revision does not match the submodule pin' >&2; exit 2;
}

fetch_pinned() {
    local url=$1 directory=$2 revision=$3
    if [[ ! -d "$directory/.git" ]]; then
        [[ ! -e "$directory" ]] || { echo "native-check: refusing existing non-repo $directory" >&2; exit 2; }
        mkdir -p "$directory"
        git -C "$directory" init -q
        timeout 120 git -C "$directory" fetch --depth 1 "$url" "$revision"
        git -C "$directory" checkout -q --detach FETCH_HEAD
    fi
    [[ $(git -C "$directory" rev-parse HEAD) == "$revision" ]] || {
        echo "native-check: unexpected revision in $directory" >&2; exit 2;
    }
    git -C "$directory" diff --quiet HEAD -- || {
        echo "native-check: tracked dependency source is modified: $directory" >&2; exit 2;
    }
}
fetch_pinned https://github.com/mihawk-99/PS5_Vulkan.git "$vk" "$vk_revision"
fetch_pinned https://github.com/mihawk-99/PS5_PayloadSDK.git "$sdk_fork" "$sdk_revision"

# Upstream scripts verify the v0.42 SDK release and zlib archive SHA-256s.
# These are public homebrew sources/tools, not proprietary firmware dumps.
export PS5_PAYLOAD_SDK_FORK="$sdk_fork"
timeout 300 bash "$vk/tools/setup-native-dependencies.sh"
export PS5_VULKAN_ROOT="$vk"
export PS5_PAYLOAD_SDK="$vk/.deps/native/ps5-payload-sdk"
[[ $(cat "$PS5_PAYLOAD_SDK/.ps5-sdk-revision") == "$sdk_revision" ]]
for patch in "$root"/patches/*.patch; do
    git -C "$llama" apply --reverse --check "$patch" 2>/dev/null || git -C "$llama" apply "$patch"
done

cxx=("$root/ps5/app/ps5cxx" -std=c++17 -O2 -fexceptions -fcxx-exceptions -frtti)
"${cxx[@]}" -dM -E -x c++ /dev/null > "$out/target-macros.txt"
grep -qx '#define __PROSPERO__ 1' "$out/target-macros.txt"
grep -qx '#define __SCE__ 1' "$out/target-macros.txt"
grep -qx '#define __FreeBSD__ 9' "$out/target-macros.txt"
"${cxx[@]}" -Wall -Wextra -Werror -I "$llama/ggml/include" \
    -c "$root/ps5/app/fleet_worker.cpp" -o "$out/fleet_worker.o"
for source in ggml-rpc transport; do
    # Match the feature-test macros set by ggml CMake for the actual PS5 build.
    "${cxx[@]}" -DGGML_BACKEND_BUILD -DNDEBUG -D_XOPEN_SOURCE=600 -D__BSD_VISIBLE \
        -I "$llama/ggml/include" -I "$llama/ggml/src" \
        -c "$llama/ggml/src/ggml-rpc/$source.cpp" -o "$out/$source.o"
done

# Builds twice and compares against upstream's recorded raw/signed hashes.
# The generated libc.prx is clean-room; no console firmware is downloaded.
timeout 180 make -C "$vk" libc
python3 - "$out" "$vk_revision" "$sdk_revision" "$expected" "$vk/runtime/libc.prx" <<'PY'
import hashlib
import json
from pathlib import Path
import struct
import sys

out = Path(sys.argv[1])
objects = {}
for name in ('fleet_worker', 'ggml-rpc', 'transport'):
    data = (out / f'{name}.o').read_bytes()
    # PS5 relocatable objects use generic ELF OSABI; this is not a title check.
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError(f'{name}: expected ELF64 little-endian object')
    if struct.unpack_from('<HH', data, 16) != (1, 62):  # ET_REL, EM_X86_64
        raise ValueError(f'{name}: expected x86-64 relocatable object')
    objects[name] = hashlib.sha256(data).hexdigest()
result = {
    'status': 'passed',
    'scope': 'PS5-target objects and clean-room runtime only; no title link, console or Vulkan execution',
    'target': 'x86_64-sie-ps5',
    'ps5_vulkan_revision': sys.argv[2],
    'ps5_payload_sdk_revision': sys.argv[3],
    'llama_revision': sys.argv[4],
    'objects_sha256': objects,
    'clean_room_runtime_sha256': hashlib.sha256(Path(sys.argv[5]).read_bytes()).hexdigest(),
}
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
