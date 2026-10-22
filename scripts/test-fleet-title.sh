#!/usr/bin/env bash
# Full native cross-build and static package validation, NOT console execution.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PS5LM_NATIVE_SOURCES="${PS5LM_NATIVE_SOURCES:-$root/.deps/fleet-native-src}"
export BUILD_JOBS="${BUILD_JOBS:-2}"
[[ $BUILD_JOBS =~ ^[1-9][0-9]*$ ]] || { echo 'BUILD_JOBS must be a positive integer' >&2; exit 2; }
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
# SDK archive wrappers resolve this as a path, not a PATH command name.
llvm_config=$(command -v "${LLVM_CONFIG:-llvm-config}")
export LLVM_CONFIG="$llvm_config"
output="$root/build/fleet-title"
mkdir -p "$output/tools"
rm -f "$output/result.json"
for tool in cmake ninja meson rsync glslangValidator glslc pkg-config curl ffmpeg; do
    command -v "$tool" >/dev/null || { echo "title-check: missing $tool" >&2; exit 2; }
done
ui="$root/third_party/ps5-homebrew-ui"
ui_revision=$(git ls-tree HEAD third_party/ps5-homebrew-ui | awk '{print $3}')
[[ -f "$ui/tools/fetch-opengl-sdk.sh" && $(git -C "$ui" rev-parse HEAD) == "$ui_revision" ]] || {
    echo 'title-check: initialize the pinned UI submodule first' >&2; exit 2;
}
scripts/test-fleet-native.sh
vk="$PS5LM_NATIVE_SOURCES/PS5_Vulkan"
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
# The original OpenGL release was removed. Use the mirror documented in BUILDING.md,
# with the original archive hash; the pinned UI fetcher also verifies every SDK file.
glcache="$ui/.deps/ps5-opengl"
glarchive="$glcache/ps5-opengl-sdk-1.0.0.tar.gz"
if [[ ! -f "$glcache/ps5-opengl-sdk-1.0.0/sdk/manifest.sha256" ]]; then
    mkdir -p "$glcache"
    curl -fL --retry 2 --max-time 300 \
        https://github.com/MyLuxy/overdrive-deps/releases/download/ps5-opengl-v1.0.0/ps5-opengl-sdk-1.0.0.tar.gz \
        -o "$glarchive.part"
    echo "f93643c04c843d56143b00951df1f8042ea7706ae9f19e4e9abf158e1ead77c5  $glarchive.part" | sha256sum --check --status
    mv "$glarchive.part" "$glarchive"
fi
glsdk=$(bash "$ui/tools/fetch-opengl-sdk.sh")
export PS5_OPENGL_SDK="$glsdk"
# Fleet validation does not need llama.cpp's optional browser UI (which otherwise
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
python3 - "$output" "$app" "$mesa_base" "$mesa_patched" "$ui_revision" <<'PY'
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
    'optional_browser_ui': 'omitted; native console dashboard is built',
    'files_sha256': {name: hashlib.sha256((app / name).read_bytes()).hexdigest() for name in files},
}
(output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
