#!/usr/bin/env python3
# Swarm-shaped load against a running llama-server: many agents with growing
# chat histories share a few server slots, like neuron-swarm's neurons do.
#
# Every agent starts from the same system prompt (shared prefix) plus its own
# seed history, then takes turns: a new user/tool message of --new-tokens, a
# reply capped at --max-tokens, and the reply appended to its history. Once a
# history passes --compact-at tokens it drops back to its seed, like the
# swarm's compaction. Workers (--concurrency) pick the next idle agent with
# Zipf weights (--zipf), so a few agents are hot and the rest come back after
# other agents have used the slots, which is what makes the server's prompt
# cache, --cache-ram and context checkpoints matter.
#
# neuron-swarm's recorded fix-git task averaged about 21k prompt tokens per
# call, 74% of them cached, 5.4k new, 720 generated, with 1-2 calls in flight.
# The defaults are that shape at half size so one configuration finishes in a
# few minutes on a console; --full uses the measured sizes.
#
#   scripts/bench-swarm.py --url http://127.0.0.1:18330 --tag np2 --out np2.json
#
# Reports turns per hour, wall time per turn, the share of prompt tokens
# served from cache, and prompt/generation speed, from llama-server's timings.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

WORDS = ("error warning file path line test pass fail build commit branch merge diff patch module function return value "
         "config parser token cache slot request reply neighbor message plan review check result status output input "
         "array index offset length buffer stream thread lock queue worker task step retry timeout limit budget").split()


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def text(rng, n_words):
    lines, line = [], []
    for _ in range(n_words):
        line.append(rng.choice(WORDS) if rng.random() > 0.15 else str(rng.randrange(10000)))
        if len(line) >= rng.randrange(6, 16):
            lines.append(" ".join(line))
            line = []
    return "\n".join(lines + [" ".join(line)])


def words_per_token(base):
    sample = text(random.Random(0), 2000)
    n = len(post(base + "/tokenize", {"content": sample})["tokens"])
    return 2000 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18330")
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=2, help="requests in flight (the swarm's admission gate)")
    ap.add_argument("--turns", type=int, default=24, help="total turns")
    ap.add_argument("--zipf", type=float, default=1.0, help="skew of which agent fires next; 0 is uniform")
    ap.add_argument("--compact-at", type=int, default=24000, help="history tokens at which an agent compacts")
    ap.add_argument("--system-tokens", type=int, default=2000)
    ap.add_argument("--seed-tokens", type=int, default=3000)
    ap.add_argument("--new-tokens", type=int, default=2700)
    ap.add_argument("--max-tokens", type=int, default=360)
    ap.add_argument("--full", action="store_true", help="measured sizes: 4k system, 6k seed, 5.4k new, 720 out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.full:
        a.system_tokens, a.seed_tokens, a.new_tokens, a.max_tokens, a.compact_at = 4000, 6000, 5400, 720, 48000
    base = a.url.rstrip("/")
    wpt = words_per_token(base)
    system = "You are a neuron in a swarm of agents. Work only on your part and reply briefly.\n" + \
        text(random.Random(1), int(a.system_tokens * wpt))
    agents = []
    for i in range(a.agents):
        rng = random.Random(100 + i)
        agents.append({"rng": rng, "turn": 0, "tokens": a.system_tokens + a.seed_tokens, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Earlier context:\n" + text(rng, int(a.seed_tokens * wpt))},
            {"role": "assistant", "content": "Noted. Waiting for the next message."},
        ]})
    # Zipf-weighted choice among idle agents: a few (coordinators) fire often, most rarely, as in the swarm.
    weights = [1 / (i + 1) ** a.zipf for i in range(a.agents)]
    choose = random.Random(3)
    left = [a.turns]
    busy = set()
    pick = threading.Condition()
    samples, errors = [], []
    t0 = time.time()

    def worker():
        while True:
            with pick:
                while left[0] > 0 and len(busy) == len(agents):
                    pick.wait()
                if left[0] <= 0:
                    return
                left[0] -= 1
                idle = [i for i in range(len(agents)) if i not in busy]
                i = choose.choices(idle, [weights[j] for j in idle])[0]
                busy.add(i)
                ag = agents[i]
            try:
                turn(ag)
            finally:
                with pick:
                    busy.discard(i)
                    pick.notify_all()

    def turn(ag):
        ag["turn"] += 1
        if ag["tokens"] + a.new_tokens + a.max_tokens > a.compact_at:  # the swarm compacts long histories
            ag["messages"], ag["tokens"] = ag["messages"][:3], a.system_tokens + a.seed_tokens
        msg = "Tool output and messages since your last turn:\n" + text(ag["rng"], int(a.new_tokens * wpt))
        body = {"messages": ag["messages"] + [{"role": "user", "content": msg}], "max_tokens": a.max_tokens,
                "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7, "ignore_eos": True,
                "cache_prompt": True}
        s = time.time()
        try:
            r = post(base + "/v1/chat/completions", body)
        except Exception as e:  # keep the run going; report it
            errors.append(repr(e))
            return
        e = time.time()
        t = r.get("timings", {})
        reply = r["choices"][0]["message"].get("content") or ""
        ag["messages"] += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        ag["tokens"] += a.new_tokens + a.max_tokens
        samples.append({"agent": agents.index(ag), "turn": ag["turn"], "start": s - t0, "end": e - t0,
                        "wall": e - s, "prompt_n": t.get("prompt_n", 0), "cache_n": t.get("cache_n", 0),
                        "prompt_ms": t.get("prompt_ms", 0), "predicted_n": t.get("predicted_n", 0),
                        "predicted_ms": t.get("predicted_ms", 0)})
        print("agent %2d turn %d: %5.1fs  new %5d  cached %5d  gen %4d" % (
            agents.index(ag), ag["turn"], e - s, t.get("prompt_n", 0), t.get("cache_n", 0),
            t.get("predicted_n", 0)), flush=True)

    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0
    pn = sum(x["prompt_n"] for x in samples)
    cn = sum(x["cache_n"] for x in samples)
    gn = sum(x["predicted_n"] for x in samples)
    walls = sorted(x["wall"] for x in samples) or [0]
    summary = {
        "tag": a.tag, "agents": a.agents, "concurrency": a.concurrency, "turns": len(samples), "errors": errors,
        "sizes": {"system": a.system_tokens, "seed": a.seed_tokens, "new": a.new_tokens, "max_tokens": a.max_tokens},
        "wall_s": round(wall, 1),
        "turns_per_hour": round(len(samples) / wall * 3600, 1),
        "turn_wall_mean_s": round(statistics.mean(walls), 1),
        "turn_wall_p95_s": round(walls[min(len(walls) - 1, int(len(walls) * 0.95))], 1),
        "prompt_tokens_evaluated": pn, "prompt_tokens_cached": cn,
        "cache_share": round(cn / (pn + cn), 3) if pn + cn else 0,
        "prefill_tps": round(pn / (sum(x["prompt_ms"] for x in samples) / 1000), 1) if pn else 0,
        "decode_tps_per_stream": round(gn / (sum(x["predicted_ms"] for x in samples) / 1000), 2) if gn else 0,
        "aggregate_tps": round((pn + gn) / wall, 1),
    }
    print(json.dumps(summary, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "samples": samples}, f, indent=1)


if __name__ == "__main__":
    main()
