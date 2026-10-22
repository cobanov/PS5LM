# Building PS5LM

The app is a PS5 title (`ps5/app`, PPSA99575): llama-server linked with ggml-vulkan against
[PS5_Vulkan](https://github.com/mihawk-99/PS5_Vulkan)'s RADV build, a dashboard drawn with
[ps5-homebrew-ui](https://github.com/blackbearreloaded/ps5-homebrew-ui), packed as a title folder for
ShadowMountPlus. The build runs on x86-64 Linux; PS5_Vulkan supports Arch (CachyOS), and Fedora 44
works too.

## How it works

- **A title, not a payload.** llama-server is linked into an fSELF `eboot.bin` with a clean-room
  `libc.prx` and registered through ShadowMountPlus. Titles may not `exit()`, some libc imports
  resolve to nothing, and `getaddrinfo`, `accept4`, `HOME` and `localeconv` need replacing
  (`ps5/app/compat_app.c`); each was found by a crash.
- **The GPU.** ggml-vulkan on RADV with no Vulkan loader, and RADV's unified heap for 11.44 GiB in one
  heap. A GPU submission is costly here (its words copied, a cache flush, a state reset); measured,
  fewer of them did not make generation faster.
- **The screen.** ps5-homebrew-ui through ps5-opengl, in the same process as RADV. The two Mesa builds
  share 4,399 symbol names, so the GL stack is linked with only its API visible.
- **Switching models.** Unloading in place leaves about 5 GiB of the pool held, so the app restarts
  itself (LoadExec) for each switch.
- **The requests on Home.** llama-server calls three weak hooks (`patches/0007`) with each POST's body,
  the answer as it is sent and the status; `ps5/app/activity.cpp` turns them into text.
- **HTTPS.** Downloads use the system's libSceHttp2 and libSceSsl, so the app carries no TLS library.
- **What a title cannot do.** Read sensors, see real free space, or start processes. ps5-exporter, a
  payload the app sends to elfldr, covers the first two.

## Build

For the automated, pinned-source Debian build and its validation limits, see
[CI validation](CI.md). `scripts/test-native-build.sh` builds and statically checks
the existing app without a console, model download, or proprietary firmware.

On Arch, the host packages (or run them in an `archlinux` container with the repository mounted):

```sh
pacman -S base-devel clang llvm lld cmake ninja meson glslang python-mako python-yaml python-packaging \
  python-setuptools spirv-tools spirv-headers spirv-llvm-translator libclc vulkan-headers shaderc \
  imagemagick ffmpeg rsync libxml2 elfutils xxd python-numpy python-fonttools python-brotli librsvg
```

PS5_Vulkan and its forks go in `.deps/src`, at the revisions PS5_Vulkan pins (its
`tools/setup-native-dependencies.sh` and `tools/build-radv.sh` name them); RADV gets the three
patches in [patches/mesa](../patches/mesa/README.md):

```sh
git clone --recursive https://github.com/cobanov/PS5LM && cd PS5LM
mkdir -p .deps/src && cd .deps/src
git clone https://github.com/mihawk-99/PS5_Vulkan
# PS5_Mesa and PS5_PayloadSDK at the pinned commits, then:
(cd PS5_Mesa && git am ../../../patches/mesa/*.patch)
(cd PS5_Vulkan && bash tools/setup-native-dependencies.sh &&
  PS5_MESA_REVISION=$(git -C ../PS5_Mesa rev-parse HEAD) bash tools/build-radv.sh release && make app)
cd ../..
scripts/build-app.sh        # build/app-ps5/title/PPSA99575
scripts/release-app.sh      # build/release/ps5lm-app-vX.Y.Z.zip and its .sha256
```

BlackBearReloaded's repositories, ps5-homebrew-ui and ps5-opengl, are no longer on GitHub. The
submodule comes from a mirror at the same commit. `build-app.sh` gets the ps5-opengl SDK through
ps5-homebrew-ui's `tools/fetch-opengl-sdk.sh`, whose download now fails: put
`ps5-opengl-sdk-1.0.0.tar.gz` from a mirror (one is
[here](https://github.com/MyLuxy/overdrive-deps/releases/tag/ps5-opengl-v1.0.0)) in
`third_party/ps5-homebrew-ui/.deps/ps5-opengl/`, and the script checks it against its pinned
sha256 before using it.

`make app` builds PS5_Vulkan's host tool and `libc.prx`, which `build-app.sh` uses; its own demo title
may fail to link at PS5_Vulkan's HEAD, which does not matter here.

## On the PC

```sh
scripts/preview-app-ui.sh                          # every screen to build/app-preview/*.png (Mesa, no console)
PS5_HOST=<console IP> scripts/ps5lm-app.sh cycle   # close, deploy, launch; also close|deploy|launch
PS5_HOST=<console IP> scripts/ps5lm-app.sh load /data/PS5LM/models/<file>.gguf
PS5_HOST=<console IP> scripts/ps5lm-app.sh screenshot out.bmp
```

- The app closes itself on request (`/data/PS5LM/quit`), so nothing is killed while it draws.
  Launching uses ps5-homebrew-dev-protocol's launch payload, cloned into `.deps/src`. A title started
  any other way without a user (shsrv's `launch`) is ended by the system after about 20 seconds.
- `/data/PS5LM/app-args.txt` (one argument a line) overrides the planner. `/data/PS5LM/env.txt` (one
  `NAME=value` a line) sets `RADV_`, `MESA_`, `ACO_`, `GGML_` and `LLAMA_` switches before the GPU opens.
- The artwork comes from SVG: `scripts/render-artwork.sh` renders the icon and the home screen
  background with the site's Recursive fonts. `scripts/bake-fonts.sh` bakes Manrope for the dashboard.

| Path | What |
|---|---|
| `ps5/app` | the app: dashboard, planner, suggestions, downloads, the request feed, llama-server on the GPU |
| `third_party/llama.cpp` | upstream llama.cpp, pinned as a submodule |
| `third_party/ps5-homebrew-ui` | the UI toolkit |
| `third_party/fonts` | Manrope (OFL) |
| `patches/` | PS5 changes to llama.cpp, kept small; `patches/mesa` the driver's |
| `ps5/compat`, `probes/` | the payload build's libc additions and memory probes ([PAYLOAD.md](PAYLOAD.md)) |
| `scripts/` | build, release, artwork, deploy and console tools |
