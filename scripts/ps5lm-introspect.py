#!/usr/bin/env python3
"""Talk to PS5LM's introspection endpoint and turn the answer into numbers.

    # what does the model think the next token is, and what is layer 12's residual stream?
    python3 tools/ps5lm-introspect.py --host 192.168.1.50 --prompt "Once upon a time" \
        --positions last --top-k 10 --tensors l_out --layers 12

    # every prompt position, logits + the residual stream of every layer, saved for poking at
    python3 tools/ps5lm-introspect.py --host 192.168.1.50 --prompt "Once upon a time" \
        --positions all --tensors l_out ffn_out --top-k 0 --save run1.npz

    # where does the token "cat" rank, and what are the model's top candidates?
    python3 tools/ps5lm-introspect.py --host 192.168.1.50 --prompt "The animal sat on the" \
        --positions last --top-k 20 --rank-of cat

    # chat: the prompt is rendered with the model's own chat template
    python3 tools/ps5lm-introspect.py --host 192.168.1.50 --messages user=hello --positions last \
        --n-predict 8 --tensors l_out --layers 20 --save chat.npz

The endpoint is POST /v1/introspect on the console's llama-server (port 8081). Everything
that arrives as base64 float32 is decoded here; --save writes one .npz with arrays named
"logits_<index>", "act_<tensor>_<index>" plus the metadata as JSON under "meta".
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request

try:
    import numpy as np
except ImportError:  # numpy is only needed for --save and array prints
    np = None


def post(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SystemExit(f"{e.code} from the server: {detail[:2000]}")


def floats(b64: str):
    raw = base64.b64decode(b64)
    if np is not None:
        return np.frombuffer(raw, dtype="<f4")
    import array

    a = array.array("f")
    a.frombytes(raw)
    return a


def build_request(args) -> dict:
    spec: dict = {}
    if args.top_k or args.full or args.entropy:
        spec["logits"] = {
            "top_k": args.top_k,
            "entropy": args.entropy,
            "probs": args.probs,
            "full": args.full,
            "max_full_rows": args.max_full_rows,
        }
    spec["positions"] = (
        "all" if args.positions == "all" else "last" if args.positions == "last" else [int(x) for x in args.positions.split(",")]
    )
    if args.tensors:
        layers = [int(x) for x in args.layers] if args.layers else "all"
        spec["activations"] = {"tensors": args.tensors, "layers": layers}
    spec["generated"] = not args.no_generated

    if args.intervention:
        ivs = []
        for raw in args.intervention:
            iv = json.loads(raw)
            if "tensor" not in iv:
                raise SystemExit(f"an --intervention needs a tensor: {raw!r}")
            iv.setdefault("layers", [int(x) for x in args.layers] if args.layers else [])
            if not iv["layers"]:
                del iv["layers"]
            ivs.append(iv)
        spec["interventions"] = ivs

    body: dict = {"n_predict": args.n_predict, "introspect": spec}
    if args.messages:
        msgs = []
        for m in args.messages:
            if "=" not in m:
                raise SystemExit(f"--messages needs role=text, got {m!r}")
            role, text = m.split("=", 1)
            msgs.append({"role": role, "content": text})
        body["messages"] = msgs
    else:
        body["prompt"] = args.prompt
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if args.raw_json:
        body.update(json.loads(args.raw_json))
    return body


def show_logits(e: dict, args) -> None:
    lg = e.get("logits")
    if not lg:
        return
    am = lg["argmax"]
    line = f"  argmax {am['piece']!r} (id {am['id']}, logit {am['logit']:.3f}"
    line += f", p={am['prob']:.3f})" if "prob" in am else ")"
    if "entropy" in lg:
        line += f"  entropy {lg['entropy']:.3f} nats"
    print(line)
    top = lg.get("top") or []
    if top:
        shown = ", ".join(f"{t['piece']!r}:{t['logit']:.2f}" for t in top[: args.print_top])
        print(f"  top: {shown}")
    if args.rank_of:
        if "full" not in lg:
            print("  --rank-of needs --full (the whole vocabulary row)")
        else:
            row = floats(lg["full"]["data"])
            target = None
            for t in lg.get("top") or []:
                if t["piece"] == args.rank_of:
                    target = t["id"]
            if target is None:
                print(f"  rank of {args.rank_of!r}: token not among the top; use --print-top to widen")
            else:
                rank = int((row > row[target]).sum()) + 1
                print(f"  rank of {args.rank_of!r}: {rank} of {len(row)} (logit {row[target]:.3f})")


def save_npz(path: str, doc: dict) -> None:
    if np is None:
        raise SystemExit("--save needs numpy (pip install numpy)")
    it = doc["introspection"]
    out = {"meta": json.dumps(it["meta"]), "inputs": json.dumps(it["inputs"])}
    for e in it["positions"]:
        idx = e["index"]
        lg = e.get("logits") or {}
        if "full" in lg:
            out[f"logits_{idx}"] = floats(lg["full"]["data"])
        for name, t in (e.get("activations") or {}).items():
            out[f"act_{name}_{idx}"] = floats(t["data"])
    np.savez_compressed(path, **out)
    print(f"saved {path}: {', '.join(sorted(out)[:6])}{' …' if len(out) > 6 else ''}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="console address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8081, help="llama-server port (default 8081)")
    p.add_argument("--path", default="/v1/introspect")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="raw prompt text")
    src.add_argument("--messages", nargs="+", help="chat turns as role=text, rendered with the model's template")
    p.add_argument("--n-predict", type=int, default=0, help="tokens to generate and capture (default 0)")
    p.add_argument("--positions", default="last", help='"last", "all", or a comma-separated index list')
    p.add_argument("--top-k", type=int, default=10, help="top candidates per position (0 = none)")
    p.add_argument("--print-top", type=int, default=8, help="how many candidates to print")
    p.add_argument("--entropy", action="store_true", help="include the entropy of the distribution")
    p.add_argument("--probs", action="store_true", default=True, help="include probabilities (default)")
    p.add_argument("--full", action="store_true", help="include the whole vocabulary row per position")
    p.add_argument("--max-full-rows", type=int, default=8, help="cap on full rows in one answer")
    p.add_argument("--rank-of", help="report the rank of this token piece when --full is used")
    p.add_argument("--tensors", nargs="*", default=[], help="activation tensors, e.g. l_out ffn_out attn_out attn_norm result_norm")
    p.add_argument("--layers", nargs="*", default=[], help="layer indices for the activations (default: all)")
    p.add_argument("--no-generated", action="store_true", help="do not capture the generated tokens")
    p.add_argument("--intervention", action="append", metavar="JSON",
                   help='change an activation in place, e.g. \'{"tensor":"ffn_out","channels":[3456],"op":"zero"}\'; '
                        'repeatable, and --layers fills in the layer list when the object has none')
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0 = greedy)")
    p.add_argument("--raw-json", help="extra request fields as JSON, merged last")
    p.add_argument("--save", help="write the arrays to this .npz")
    p.add_argument("--json", action="store_true", help="print the whole response as JSON")
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    url = f"http://{args.host}:{args.port}{args.path}"
    body = build_request(args)
    doc = post(url, body, args.timeout)

    if args.json:
        json.dump(doc, sys.stdout, indent=2)
        print()
        return 0

    it = doc.get("introspection")
    if it is None:
        print("no introspection in the answer; keys:", list(doc.keys()))
        return 1

    meta = it["meta"]
    print(
        f"{meta['model'].rsplit('/', 1)[-1]}: {meta['n_layer']} layers, vocab {meta['n_vocab']}, "
        f"context {meta['n_ctx']}, prompt {meta['n_prompt']} tokens, build {meta['build']}"
    )
    for e in it["positions"]:
        kind = "generated" if e["generated"] else "prompt"
        acts = ", ".join(f"{n}{t['shape'][:1]}" for n, t in (e.get("activations") or {}).items())
        print(f"position {e['index']} ({kind}) {e['piece']!r}" + (f"  activations: {acts}" if acts else ""))
        show_logits(e, args)
    if it.get("truncated"):
        print("note: the answer hit the server's activation budget and dropped some tensors")
    for iv in (it.get("interventions") or []):
        print(f"intervention {iv['op']} {iv['tensor']}"
              + (f" layers {iv['layers']}" if iv.get("layers") not in (None, "all") else "")
              + (f" channels {iv['channels']}" if iv.get("channels") else "")
              + f": modified {iv['modified']}" + ("  (nothing matched!)" if iv["modified"] == 0 else ""))
    if "content" in doc and doc["content"]:
        print(f"generated: {doc['content']!r}")
    if args.save:
        save_npz(args.save, doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
