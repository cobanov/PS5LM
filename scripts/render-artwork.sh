#!/usr/bin/env bash
# Renders the app's artwork from its SVG sources in ps5/app/sce_sys/source:
# icon0.png (512x512) and background.png (3840x2160), which scripts/build-app.sh
# turns into the home screen's DDS backgrounds. The SVGs come from brand/build.py
# with their letters outlined, so no fonts are needed. Needs rsvg-convert.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/ps5/app/sce_sys/source"
rsvg-convert -w 512 -h 512 "$src/icon.svg" -o "$root/ps5/app/sce_sys/icon0.png"
rsvg-convert -w 3840 -h 2160 "$src/background.svg" -o "$src/background.png"
echo "render-artwork: icon0.png and source/background.png"
