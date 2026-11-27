#!/usr/bin/env python3
"""Checks the Suggested models (ps5/app/suggested.hpp) against Hugging Face.

Each entry is fetched from resolve/<revision>/, so the revision must serve every pinned file with
the pinned size and sha256. This asks Hugging Face for each file's size and LFS hash at that
revision (HEAD requests, nothing is downloaded) and says whether the branch has moved on.

    scripts/check-suggested.py           check every entry
    scripts/check-suggested.py --find    also name, for an entry that fails, the newest commit
                                         whose files match the pinned size and sha256

Moving an entry to a new upload is a deliberate change of its revision, sizes and hashes together.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HUB = "https://huggingface.co"
AGENT = {"User-Agent": "ps5lm-check-suggested"}


def entries():
    """(name, repo, revision, [(file, bytes, sha256)]) for each Suggested entry."""
    source = (ROOT / "ps5/app/suggested.hpp").read_text()
    body = source[source.index("kSuggestions[] = {"):]
    files = r'\{\s*"([^"]+)",\s*(\d+)ull,\s*"([0-9a-f]{64})"[^}]*\}'
    pattern = re.compile(r'\{\s*"([^"]+)",\s*(?:true|false),[^{]*?"([^"]+/[^"]+)",\s*"([^"]*)",\s*'
                         + files + r',\s*(?:\{\s*\}|' + files + r')', re.S)
    for m in pattern.finditer(body):
        pinned = [(m.group(4), int(m.group(5)), m.group(6))]
        if m.group(7):
            pinned.append((m.group(7), int(m.group(8)), m.group(9)))
        yield m.group(1), m.group(2), m.group(3), pinned


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def served(repo, revision, name):
    """The size and sha256 Hugging Face serves for a file at a revision, from its LFS headers."""
    request = urllib.request.Request(f"{HUB}/{repo}/resolve/{revision}/{name}", method="HEAD", headers=AGENT)
    try:
        headers = urllib.request.build_opener(NoRedirect).open(request, timeout=60).headers
    except urllib.error.HTTPError as error:
        headers = error.headers
    size = headers.get("x-linked-size")
    return (int(size) if size else None), (headers.get("x-linked-etag") or "").strip('"')


def get_json(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = dict(AGENT, **({"Content-Type": "application/json"} if data else {}))
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=60) as response:
        return json.load(response)


def find(repo, pinned):
    """The newest commit on main whose files match the pinned sizes and hashes, or None."""
    for commit in get_json(f"{HUB}/api/models/{repo}/commits/main"):
        info = {p["path"]: p for p in get_json(f"{HUB}/api/models/{repo}/paths-info/{commit['id']}",
                                               {"paths": [name for name, _, _ in pinned]})}
        if all((info.get(name, {}).get("lfs") or {}).get("oid") == sha
               and (info[name]["lfs"].get("size") == size) for name, size, sha in pinned):
            return commit["id"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--find", action="store_true", help="search the history of an entry that fails")
    args = parser.parse_args()
    failed = 0
    for name, repo, revision, pinned in entries():
        bad = [f for f, size, sha in pinned if served(repo, revision, f) != (size, sha)]
        moved = [f for f, size, sha in pinned if served(repo, "main", f) != (size, sha)]
        if bad:
            failed += 1
            print(f"FAIL  {name}: {', '.join(bad)} at {revision[:12]} differ from the pins")
            if args.find:
                print(f"      newest matching commit: {find(repo, pinned) or 'none'}")
        else:
            note = f"; main has moved on ({', '.join(moved)})" if moved else ""
            print(f"ok    {name}: {len(pinned)} file(s) at {revision[:12]}{note}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
