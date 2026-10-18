#!/usr/bin/env python3
"""Which neuron in a layer moves the answer? Ask the running PS5LM.

    python3 examples/interpret/neuron_ablation.py --host <console> \
        --prompt "The capital of France is" --layer 20

The loop this closes is the whole point of the endpoint:

1. ask for the model's **logits** at the last prompt position, with the whole
   vocabulary row, and the **ffn_out** activation of one layer at that position;
2. rank that layer's channels by how strongly they fire (the FFN neurons that are
   doing something for this token);
3. for each candidate, send the same request again with that one channel **zeroed**
   in place, and measure what changed: the target token's probability, its rank,
   and the exact KL divergence of the next-token distribution over the whole
   vocabulary.

Everything except the ranking comes from one request per candidate. A layer whose
ablation barely moves anything is a layer that is not carrying this decision; the
channels at the top of the last column are the ones worth looking at next.

`--full-row` is on by default because the KL is then exact (over all `n_vocab`
values); pass `--top-only` to trade that for a tenth of the traffic.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
import urllib.error
import urllib.request


def post(host: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://{host}:8081{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{e.code} from the console: {e.read().decode('utf-8', 'replace')[:800]}")


def floats(b64: str):
    import array

    a = array.array("f")
    a.frombytes(base64.b64decode(b64))
    return a


def softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    s = sum(exps)
    return [e / s for e in exps]


def request(host: str, args, interventions=None, want_activation=True):
    spec: dict = {
        "logits": {"top_k": args.top_k, "probs": True, "full": not args.top_only, "max_full_rows": 1},
        "positions": "last",
        "generated": False,
        "capture_generated": False,
    }
    if want_activation:
        spec["activations"] = {"tensors": [args.tensor], "layers": [args.layer]}
    if interventions:
        spec["interventions"] = interventions
    body = {"prompt": args.prompt, "n_predict": 0, "temperature": 0.0, "introspect": spec}
    t0 = time.time()
    doc = post(host, "/v1/introspect", body)
    return doc, time.time() - t0


def intro(doc):
    it = doc.get("introspection")
    if not it:
        raise SystemExit("no introspection in the answer: " + json.dumps(doc)[:400])
    entry = it["positions"][-1]
    return it, entry


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--tensor", default="ffn_out", help="ffn_out (one neuron per channel) or l_out, attn_out, ...")
    p.add_argument("--top", type=int, default=12, help="how many channels to rank and ablate")
    p.add_argument("--top-k", type=int, default=20, help="candidates printed from the answer")
    p.add_argument("--target", help="token piece to track (default: the baseline argmax)")
    p.add_argument("--top-only", action="store_true", help="skip the full vocabulary row (approximate KL)")
    args = p.parse_args()

    print(f"{args.prompt!r} -> layer {args.layer}, tensor {args.tensor}")
    t0 = time.time()
    doc, dt = request(args.host, args)
    it, entry = intro(doc)
    acts = entry.get("activations") or {}
    key = [k for k in acts if k.startswith(args.tensor)]
    if not key:
        raise SystemExit(f"the answer carried no {args.tensor}-<layer>: {sorted(acts)}")
    key = key[0]
    row = floats(acts[key]["data"])
    logits = entry["logits"]
    target = args.target or logits["argmax"]["piece"]
    n_vocab = logits["n_vocab"]

    print(f"  {it['meta']['n_layer']} layers, vocab {n_vocab}, prompt {it['meta']['n_prompt']} tokens; answer in {dt:.2f} s")
    print(f"  next token: {logits['argmax']['piece']!r} (id {logits['argmax']['id']}, logit {logits['argmax']['logit']:.2f}, "
          f"p {logits['argmax'].get('prob', float('nan')):.4f}), entropy {logits.get('entropy', float('nan')):.3f} nats")
    print(f"  {key}: {len(row)} channels, mean {sum(row)/len(row):.3f}, "
          f"largest |activation| {max(abs(v) for v in row):.3f}")

    # the baseline distribution, exactly if we have the full row
    if "full" in logits:
        base_row = floats(logits["full"]["data"])
        base_probs = softmax(base_row)
    else:
        base_row = None
        base_probs = None

    def prob_of(doc_logits, piece):
        if "full" in doc_logits:
            row_i = floats(doc_logits["full"]["data"])
            probs = softmax(row_i)
            # find the piece's id among the printed candidates
            for t in [doc_logits["argmax"]] + (doc_logits.get("top") or []):
                if t["piece"] == piece:
                    return probs[t["id"]], sum(1 for v in row_i if v > row_i[t["id"]]) + 1, len(row_i)
            return None, None, len(row_i)
        for t in [doc_logits["argmax"]] + (doc_logits.get("top") or []):
            if t["piece"] == piece:
                return t.get("prob"), None, None
        return None, None, None

    base_p, base_rank, _ = prob_of(logits, target)
    print(f"  tracking {target!r}: baseline p={base_p:.6f}" + (f", rank {base_rank}" if base_rank else ""))

    # candidate neurons: the channels that fire hardest at this position
    order = sorted(range(len(row)), key=lambda i: -abs(row[i]))[: args.top]
    print(f"\n  ablating the {len(order)} loudest channels one at a time (zeroing {args.tensor}-{args.layer} channel c):\n")
    print(f"  {'channel':>7}  {'activation':>11}  {'d logit':>8}  {'p(target)':>12}  {'rank':>7}  {'KL(base||abl)':>13}  argmax")
    results = []
    for c in order:
        doc2, dt2 = request(args.host, args, interventions=[
            {"tensor": args.tensor, "layers": [args.layer], "channels": [c], "op": "zero"},
        ], want_activation=False)
        it2, e2 = intro(doc2)
        lg2 = e2["logits"]
        mod = (it2.get("interventions") or [{}])[0].get("modified")
        p2, rank2, _ = prob_of(lg2, target)
        target_logit = None
        if "full" in lg2:
            r2 = floats(lg2["full"]["data"])
            for t in [lg2["argmax"]] + (lg2.get("top") or []):
                if t["piece"] == target:
                    target_logit = r2[t["id"]]
        if base_row is not None and "full" in lg2:
            r2 = floats(lg2["full"]["data"])
            ab = softmax(r2)
            kl = sum(p * math.log(p / q) for p, q in zip(base_probs, ab) if p > 0 and q > 0)
        else:
            kl = float("nan")
        dlogit = None
        if target_logit is not None:
            for t in [logits["argmax"]] + (logits.get("top") or []):
                if t["piece"] == target:
                    dlogit = target_logit - t["logit"]
        results.append((abs(dlogit) if dlogit is not None else 0.0, c, row[c], dlogit, p2, rank2, kl, lg2["argmax"]["piece"], mod))
        print(f"  {c:>7}  {row[c]:>11.4f}  {dlogit if dlogit is not None else float('nan'):>8.3f}  "
              f"{p2 if p2 is not None else float('nan'):>12.6f}  {rank2 if rank2 else '':>7}  {kl:>13.5f}  {lg2['argmax']['piece']!r}")

    results.sort(reverse=True)
    if results:
        _, c, a, dlogit, p2, rank2, kl, am, mod = results[0]
        print(f"\n  biggest single-neuron effect: {args.tensor}-{args.layer} channel {c} (activation {a:.3f}) "
              f"-> {target!r} logit {dlogit:+.3f}, p {p2:.6f}, argmax {am!r}, KL {kl:.5f} ({mod} elements changed)")
    print(f"  total {time.time() - t0:.1f} s for {1 + len(order)} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
