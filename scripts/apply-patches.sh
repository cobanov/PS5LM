#!/usr/bin/env bash
# Applies to a git tree each patch that is not in it yet, in the order given.
#
#   scripts/apply-patches.sh TREE PATCH...
#
# A patch is in the tree when it reverses cleanly. But a later patch can change
# the lines beside an earlier one's (patches/fleet/0009 adds its RPC command
# right after the one 0006 added), and then the earlier patch no longer reverses
# although it is applied, and applying it again fails. So this remembers, in the
# tree's git directory, which patches it applied and the state it left the tree
# in. While the tree is still in that state those patches are skipped; a tree
# that was reset or edited since is checked patch by patch, as before.
set -euo pipefail
tree=${1:?usage: apply-patches.sh TREE PATCH...}
shift
stamp="$(git -C "$tree" rev-parse --absolute-git-dir)/ps5lm-patches"
state() {
    { git -C "$tree" status --porcelain; git -C "$tree" diff; } | git hash-object --stdin
}
known=""
if [ -f "$stamp" ] && [ "$(head -n 1 "$stamp")" = "$(state)" ]; then
    known=$(tail -n +2 "$stamp")
fi
applied=""
for patch in "$@"; do
    [ -e "$patch" ] || continue
    sum=$(git hash-object "$patch")
    if ! grep -qx "$sum" <<<"$known"; then
        git -C "$tree" apply --reverse --check "$patch" 2>/dev/null || git -C "$tree" apply "$patch"
    fi
    applied+="$sum"$'\n'
done
{ state; printf '%s' "$applied"; } > "$stamp"
