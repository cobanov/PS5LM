#!/usr/bin/env bash
# Drives the PS5LM app on the console from the PC, with no controller:
#
#   scripts/ps5lm-app.sh close     ask the running app to end itself, and wait
#   scripts/ps5lm-app.sh deploy    upload build/app-ps5/title/<TITLE_ID>
#   scripts/ps5lm-app.sh launch    start it (build/app-ps5/launch-<TITLE_ID>.elf, from build-app.sh)
#   scripts/ps5lm-app.sh cycle     close, deploy, launch
#   scripts/ps5lm-app.sh screenshot [file]   what the TV shows, as a BMP
#   scripts/ps5lm-app.sh load <model path>   switch the model, as the library does
#   scripts/ps5lm-app.sh unload              unload the model
#   scripts/ps5lm-app.sh press <button>      press a button (up, down, cross, l1, r1, ...)
#   scripts/ps5lm-app.sh fetch <url>         probe: download over HTTPS (system TLS), see app.log
#   scripts/ps5lm-app.sh bench-write         probe: write speed by method, see app.log
#   scripts/ps5lm-app.sh memory              log the memory pools now, see app.log
#   scripts/ps5lm-app.sh search <words>      search Get models from the PC
#
# The app is never killed: it polls /data/PS5LM/quit and leaves through
# sceSystemServiceLoadExec("exit"), since killing a title that is rendering
# has been followed by lost consoles. Needs ftpsrv (2121) and elfldr (9021).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
host=${PS5_HOST:?set PS5_HOST to the console IP}
ftp="ftp://$host:${PS5_FTP_PORT:-2121}"
title=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["titleId"])' "$root/ps5/app/sce_sys/param.json")
app="$root/build/app-ps5/title/$title"

close() {
    printf 'quit\n' | curl -sS -T - "$ftp/data/PS5LM/quit"
    for _ in $(seq 1 30); do
        sleep 2
        if ! curl -sf "$ftp/data/PS5LM/quit" -o /dev/null; then
            # The app took the request; give the system a moment to end it.
            sleep 5
            echo "ps5lm-app: closed"
            return 0
        fi
    done
    curl -s -Q "DELE /data/PS5LM/quit" "$ftp/" -o /dev/null || true
    echo "ps5lm-app: the app did not take the quit request (not running?)"
}

deploy() {
    [ -f "$app/eboot.bin" ] || { echo "ps5lm-app: build it first (scripts/build-app.sh)" >&2; exit 1; }
    (cd "$app" && find . -type f | while read -r f; do
        curl -sS --ftp-create-dirs -T "$f" "$ftp/data/homebrew/$title/${f#./}" ||
            { echo "ps5lm-app: upload of $f failed (is the app still running?)" >&2; exit 1; }
    done)
    echo "ps5lm-app: deployed $title"
}

launch() {
    # ps5/app/launch_app.c, built next to the app; elfldr runs it, it starts the title.
    local elf="$root/build/app-ps5/launch-$title.elf"
    [ -f "$elf" ] || { echo "ps5lm-app: build it first (scripts/build-app.sh makes $elf)" >&2; exit 1; }
    socat -u "FILE:$elf" "TCP:$host:${PS5_PORT:-9021}"
    echo "ps5lm-app: launch sent"
}

# Requests the running app takes once a second.
request() {
    printf '%s\n' "${2:--}" | curl -sS -T - "$ftp/data/PS5LM/$1"
}

screenshot() {
    request screenshot
    for _ in $(seq 1 20); do
        sleep 1
        curl -sf "$ftp/data/PS5LM/screenshot" -o /dev/null || break
    done
    sleep 2
    curl -sS "$ftp/data/PS5LM/screen.bmp" -o "${1:-build/app-ps5/screen.bmp}"
    echo "ps5lm-app: screenshot in ${1:-build/app-ps5/screen.bmp}"
}

case ${1:-} in
    load) request load "${2:?model path on the console}" ;;
    unload) request unload ;;
    fetch) request fetch "${2:?https URL}" ;;
    bench-write) request bench-write ;;
    memory) request memory ;;
    search) request search "${2:?search words}" ;;
    press) request press "${2:?up|down|left|right|cross|circle|triangle|square|l1|r1|l2|r2}" ;;
    screenshot) screenshot "${2:-}" ;;
    close) close ;;
    deploy) deploy ;;
    launch) launch ;;
    cycle) close; deploy; launch ;;
    *) echo "usage: $0 close|deploy|launch|cycle|screenshot|load|unload|press|fetch|bench-write|memory|search" >&2; exit 2 ;;
esac
