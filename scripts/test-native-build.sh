#!/usr/bin/env bash
# Full native cross-build and static package validation, NOT console execution.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
output="$root/build/native-ci"
mkdir -p "$output/tools"
# Clear stale success evidence even if a preflight check below fails.
rm -f "$output/result.json"
export PS5LM_NATIVE_SOURCES="${PS5LM_NATIVE_SOURCES:-$root/.deps/native-ci-src}"
export BUILD_JOBS="${BUILD_JOBS:-2}"
[[ $BUILD_JOBS =~ ^[1-9][0-9]*$ ]] || { echo 'BUILD_JOBS must be a positive integer' >&2; exit 2; }
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
# SDK archive wrappers resolve this as a path, not a PATH command name.
llvm_config=$(command -v "${LLVM_CONFIG:-llvm-config}") || {
    echo "title-check: missing LLVM_CONFIG tool: ${LLVM_CONFIG:-llvm-config}" >&2; exit 2;
}
export LLVM_CONFIG="$llvm_config"
for tool in git clang clang++ make wget unzip sha256sum timeout python3 cmake ninja meson rsync glslangValidator glslc pkg-config curl ffmpeg; do
    command -v "$tool" >/dev/null || { echo "title-check: missing $tool" >&2; exit 2; }
done
ui="$root/third_party/ps5-homebrew-ui"
ui_revision=$(git ls-tree HEAD third_party/ps5-homebrew-ui | awk '{print $3}')
[[ -f "$ui/tools/fetch-opengl-sdk.sh" && $(git -C "$ui" rev-parse HEAD) == "$ui_revision" ]] || {
    echo 'title-check: initialize the pinned UI submodule first' >&2; exit 2;
}
llama="$root/third_party/llama.cpp"
llama_revision=$(git ls-tree HEAD third_party/llama.cpp | awk '{print $3}')
[[ -f "$llama/CMakeLists.txt" && $(git -C "$llama" rev-parse HEAD) == "$llama_revision" ]] || {
    echo 'title-check: initialize the pinned llama.cpp submodule first' >&2; exit 2;
}
vk="$PS5LM_NATIVE_SOURCES/PS5_Vulkan"
sdk_fork="$PS5LM_NATIVE_SOURCES/PS5_PayloadSDK"
vk_revision=f410e67a0241b3e8edcdbcd2a9b285dfc4a412ed
sdk_revision=b83202be73e930050e00de0f1b4d9ec46c0391bf
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
# Rebuild twice and verify upstream's recorded clean-room runtime checksums.
# This also builds the host packaging tool without linking an unrelated demo.
timeout 180 make -C "$vk" libc

mesa="$PS5LM_NATIVE_SOURCES/PS5_Mesa"
mesa_base=1ba9874f478657ccf6da72ce55f0260164cc0bd0
mesa_patched=9e0a7c63bc6d4b6bcb60831163bb3fe9f46e91ea
if [[ ! -d "$mesa/.git" ]]; then
    [[ ! -e "$mesa" ]] || { echo "title-check: refusing existing non-repo $mesa" >&2; exit 2; }
    mkdir -p "$mesa"
    git -C "$mesa" init -q
    timeout 180 git -C "$mesa" fetch --depth 1 https://github.com/mihawk-99/PS5_Mesa.git "$mesa_base"
    git -C "$mesa" checkout -q --detach FETCH_HEAD
fi
git -C "$mesa" diff --quiet HEAD -- || { echo 'title-check: modified Mesa sources' >&2; exit 2; }
if [[ $(git -C "$mesa" rev-parse HEAD) == "$mesa_base" ]]; then
    git -C "$mesa" -c user.name='PS5LM CI' -c user.email='ci@ps5lm.invalid' \
        -c commit.gpgsign=false am --committer-date-is-author-date "$root"/patches/mesa/*.patch
fi
[[ $(git -C "$mesa" rev-parse HEAD) == "$mesa_patched" ]] || {
    echo 'title-check: unexpected patched Mesa revision' >&2; exit 2;
}
for patch in "$root"/patches/mesa/*.patch; do
    git -C "$mesa" apply --reverse --check "$patch"
done
export PS5_MESA_FORK="$mesa" PS5_MESA_REVISION="$mesa_patched"
export PS5_VULKAN_ROOT="$vk" PS5_PAYLOAD_SDK="$vk/.deps/native/ps5-payload-sdk"
# Upstream build-radv invokes Ninja without a job limit; bound both host and target stages.
ninja_real="$(command -v ninja)"
printf '#!/bin/sh\nexec "%s" -j "%s" "$@"\n' "$ninja_real" "$BUILD_JOBS" > "$output/tools/ninja"
chmod +x "$output/tools/ninja"
export NINJA="$output/tools/ninja"
timeout 1800 bash "$vk/tools/build-radv.sh" release

# make libc already built this exact host tool, without the unrelated demo title.
mkdir -p "$vk/build/host"
cp "$vk/build/runtime-shim/ps5-native-tool" "$vk/build/host/ps5-native-tool"
# The original OpenGL release and its mirror were removed. PS5LM's deps-ps5-opengl-1.0.0
# release holds the extracted SDK, repacked (see BUILDING.md); the pinned UI fetcher
# below checks the SDK manifest's hash and every file against it.
glcache="$ui/.deps/ps5-opengl"
glarchive="$glcache/ps5-opengl-sdk-1.0.0-extracted.tar.gz"
if [[ ! -f "$glcache/ps5-opengl-sdk-1.0.0/sdk/manifest.sha256" ]]; then
    mkdir -p "$glcache"
    curl -fL --retry 2 --max-time 300 \
        https://github.com/cobanov/PS5LM/releases/download/deps-ps5-opengl-1.0.0/ps5-opengl-sdk-1.0.0.tar.gz \
        -o "$glarchive.part"
    echo "a3ce70a1d3278a2367c3135f034d4925dc85cc4bda30916f44aa474d37241053  $glarchive.part" | sha256sum --check --status
    tar -xzf "$glarchive.part" -C "$glcache"
    rm -f "$glarchive.part"
fi
glsdk=$(bash "$ui/tools/fetch-opengl-sdk.sh")
export PS5_OPENGL_SDK="$glsdk"
# Cross-build validation omits llama.cpp's optional browser UI (which otherwise
# falls back to a floating Hugging Face "latest" download). Keep the console UI.
[[ ! -e "$root/third_party/llama.cpp/tools/ui/dist/index.html" ]] || {
    echo 'title-check: remove custom source-tree browser assets or use a clean checkout' >&2; exit 2;
}
rm -rf -- "$root/build/app-ps5/llama/tools/ui/dist"
export PS5LM_PREBUILT_UI=OFF
timeout 1800 bash scripts/build-app.sh
bash "$vk/tools/inspect.sh" "$root/build/app-ps5/eboot.elf"
title_id=$(python3 -c 'import json; print(json.load(open("ps5/app/sce_sys/param.json"))["titleId"])')
app="$root/build/app-ps5/title/$title_id"
bash "$vk/tools/inspect.sh" "$app/eboot.bin"
"$vk/build/host/ps5-native-tool" self --inspect --file "$app/eboot.bin"
python3 - "$output" "$app" "$mesa_base" "$mesa_patched" "$ui_revision" "$llama_revision" "$vk_revision" "$sdk_revision" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output, app = map(Path, sys.argv[1:3])
files = ('eboot.bin', 'sce_module/libc.prx', 'sce_sys/param.json', 'sce_sys/icon0.png', 'sce_sys/pic0.dds', 'sce_sys/pic1.dds')
result = {
    'status': 'passed',
    'scope': 'Complete native title cross-build and static ELF/FSELF validation; NOT console/Vulkan execution',
    'mesa_base_revision': sys.argv[3],
    'mesa_patched_revision': sys.argv[4],
    'ui_revision': sys.argv[5],
    'llama_revision': sys.argv[6],
    'ps5_vulkan_revision': sys.argv[7],
    'ps5_payload_sdk_revision': sys.argv[8],
    'optional_browser_ui': 'omitted; native console dashboard is built',
    'files_sha256': {name: hashlib.sha256((app / name).read_bytes()).hexdigest() for name in files},
}
(output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
