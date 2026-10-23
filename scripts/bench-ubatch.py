#!/usr/bin/env python3
# Prompt and generation speed, and memory, of one model at several prompt
# batch sizes (llama-server's -ub), on the console, to choose the planner's
# kPromptUbatch (ps5/app/model_plan.hpp).
#
#   PS5_HOST=192.168.1.19 scripts/bench-ubatch.py /data/PS5LM/models/Qwen3.8-27B-UD-Q2_K_XL.gguf
#   PS5_HOST=192.168.1.19 scripts/bench-ubatch.py <model on the console> 64 128 256 512
#
# The app loads the model with the planner first, and its arguments (from
# app.log) are kept for every size; only -ub, -b and the log verbosity change.
# Each size restarts the app with them in app-args.txt, sends the same prompt
# (2,048 tokens of the LICENSE) twice with 128 tokens to generate, keeps the
# faster run, and reads what the app logged about memory: the pool free once
# the model serves and again after the prompt, and llama.cpp's buffers. A size
# that does not load is reported as such. At the end app-args.txt is removed
# and the model loads with the planner again.
#
# Needs the app running with /data open, ftpsrv on 2121, Python 3.8+.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import ftplib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("PS5_HOST") or sys.exit("bench-ubatch: set PS5_HOST to the console's address")
FTP_PORT = int(os.environ.get("PS5_FTP_PORT", "2121"))
API = "http://%s:8081" % HOST
DATA = "/data/PS5LM"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_TOKENS, GENERATE = 2048, 128


def ftp():
    f = ftplib.FTP()
    f.connect(HOST, FTP_PORT, timeout=30)
    f.login()
    return f


def put(name, text):
    with ftp() as f:
        f.storbinary("STOR %s/%s" % (DATA, name), io.BytesIO(text.encode()))


def get(name):
    out = io.BytesIO()
    try:
        with ftp() as f:
            f.retrbinary("RETR %s/%s" % (DATA, name), out.write)
    except ftplib.error_perm:
        return ""
    return out.getvalue().decode("utf-8", "replace")


def remove(name):
    try:
        with ftp() as f:
            f.delete("%s/%s" % (DATA, name))
    except ftplib.error_perm:
        pass


def post(path, body, timeout=3600):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def this_run(log):
    """app.log from the app's latest start."""
    at = log.rfind("ps5lm-app: version")
    return log[at:] if at >= 0 else log


def wait_ready(started_after, timeout=1200):
    """Until the server answers /health, or the new process logs that it stopped."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(5)
        log = get("app.log")
        if "ps5lm-app: version" not in log:
            continue
        run = this_run(log)
        if run == started_after:
            continue  # the old process is still there
        if "llama_server returned" in run:
            return False, run
        try:
            with urllib.request.urlopen(API + "/health", timeout=5) as r:
                if r.status == 200:
                    return True, run
        except (urllib.error.URLError, OSError):
            pass
    return False, this_run(get("app.log"))


def planner_args(run):
    args = [l.split("ps5lm-app: arg ", 1)[1] for l in run.splitlines() if l.startswith("ps5lm-app: arg ")]
    out, skip = [], False
    for a in args:  # what the app adds itself
        if skip:
            skip = False
        elif a in ("--log-file", "--tools"):
            skip = True
        elif a != "--metrics":
            out.append(a)
    return out


def with_ubatch(args, ub):
    out = list(args)
    for flag, value in (("-ub", ub), ("-b", max(256, ub))):
        if flag in out:
            out[out.index(flag) + 1] = str(value)
        else:
            out += [flag, str(value)]
    return out + ["-lv", "4"]  # llama.cpp's buffer sizes in llama.log


def pool_free(run, when):
    m = re.findall(r"memory %s: pool [0-9.]+ GiB, pool free ([0-9.]+) GiB" % re.escape(when), run)
    return float(m[-1]) if m else None


def buffers(llama_log):
    sizes = {}
    for line in llama_log.splitlines():
        m = re.search(r"(\S+) (model|KV|RS|compute) buffer size = +([0-9.]+) MiB", line)
        if m:
            sizes["%s %s" % (m.group(1), m.group(2))] = float(m.group(3))
    return sizes


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: bench-ubatch.py <model path on the console> [ubatch ...]")
    model = sys.argv[1]
    sizes = [int(x) for x in sys.argv[2:]] or [64, 128, 256, 512]
    prompt_text = open(os.path.join(ROOT, "LICENSE"), encoding="utf-8").read()

    print("bench-ubatch: loading %s with the planner" % model)
    before = this_run(get("app.log"))
    remove("app-args.txt")
    put("load", model + "\n")
    ok, run = wait_ready(before)
    if not ok:
        sys.exit("bench-ubatch: the model did not load with the planner; see app.log and llama.log")
    base = planner_args(run)
    print("bench-ubatch: the planner's arguments: %s" % " ".join(base))

    rows = []
    for ub in sizes:
        args = with_ubatch(base, ub)
        before = run
        put("app-args.txt", "\n".join(args) + "\n")
        put("unload", "-\n")  # restarts the app, which then reads app-args.txt
        ok, run = wait_ready(before)
        if not ok:
            rows.append((ub, None, None, None, None, "did not load (out of memory? see llama.log)"))
            print("bench-ubatch: -ub %d did not load" % ub)
            continue
        tokens = post("/tokenize", {"content": prompt_text})["tokens"][:PROMPT_TOKENS]
        post("/completion", {"prompt": tokens[:64], "n_predict": 4, "cache_prompt": False})  # shaders, buffers
        best = None
        for _ in range(2):
            t = post("/completion", {"prompt": tokens, "n_predict": GENERATE, "cache_prompt": False,
                                     "temperature": 0, "ignore_eos": True})["timings"]
            best = t if best is None or t["prompt_per_second"] > best["prompt_per_second"] else best
        put("memory", "-\n")
        time.sleep(4)
        run = this_run(get("app.log"))
        buf = buffers(get("llama.log"))
        compute = sum(v for k, v in buf.items() if k.endswith("compute"))
        rows.append((ub, best["prompt_per_second"], best["predicted_per_second"], pool_free(run, "when ready"),
                     pool_free(run, "asked"), "compute buffers %.0f MiB" % compute))
        print("bench-ubatch: -ub %d: prompt %.1f tok/s, generation %.1f tok/s" % (ub, rows[-1][1], rows[-1][2]))

    remove("app-args.txt")
    put("load", model + "\n")
    print()
    print("| -ub | prompt tok/s (%d tokens) | generation tok/s | pool free when ready | pool free after the prompt | |" % PROMPT_TOKENS)
    print("|---|---|---|---|---|---|")
    for ub, pp, tg, ready, after, note in rows:
        f = lambda v, fmt: fmt % v if v is not None else "-"
        print("| %d | %s | %s | %s | %s | %s |" % (ub, f(pp, "%.1f"), f(tg, "%.1f"), f(ready, "%.2f GiB"),
                                                  f(after, "%.2f GiB"), note))
    print("\nbench-ubatch: the model loads with the planner again")


if __name__ == "__main__":
    main()
