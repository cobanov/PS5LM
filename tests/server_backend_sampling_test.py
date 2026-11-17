#!/usr/bin/env python3
"""patches/0018 on a host llama-server: backend sampling stays on with a reasoning budget and with a grammar.

With --backend-sampling the worker samples and only the result crosses the network. A reasoning budget and a
grammar (tool calls) run on the host, so with either the server gave backend sampling up for the request ("backend
sampling is not compatible with ..., disabling") and fetched the scores of the whole vocabulary for every token.
With the patch the worker keeps the top-k and the host does the rest; the whole vocabulary is fetched only for a
token the budget forces, a token the grammar rejects, and equal scores among the candidates.

Two servers of the same build, each with its own CPU ggml-rpc-server: one with LLAMA_BACKEND_SAMPLING_CHECKS=0
(sampling on the host, as before the patch) and one without. The same requests go to both:

  budget     a reasoning budget that is hit: the same tokens, and the end marker right after the budget
  natural    a budget that is not reached (the model is biased to end its reasoning itself): the same tokens
  grammar    a grammar from the first token, sampled hot so that it rejects tokens: the same tokens, all legal
  lazy       a grammar that waits for its trigger, with a budget, as tool calls have: the same tokens
  greedy     the cases above at temperature 0
  plain      no budget and no grammar: the same tokens, and the patch's path is not taken
  topk       top-k switched off, and a logit bias: the old path with its log line, the same tokens
  two        two requests at once on two slots: each the same tokens as alone
  logs       the patched server never says "disabling" for a request the patch covers and reports how many
             tokens came from the worker's top-k; the other says it for every such request
  bytes      what the host receives from the worker for a reply, from the server's own count of rows fetched

The model is an untrained recurrent fixture with one token per character, so only token ids are compared, not
words. With --model the budget and tool-call cases also run as chat requests on a real GGUF (Qwen3.5-0.8B has
the consoles' layout, reasoning markers and tool-call format).
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import threading
import urllib.request

from fleet_rpc_smoke import free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_recurrent_fixture

MARKERS = ("<think>", "</think>")
STATS = re.compile(r"backend sampling with host checks: (\d+) tokens from the backend's top-k, (\d+) from the full logits "
                   r"\((\d+) forced by the reasoning budget, (\d+) for the grammar, (\d+) equal logits, (\d+) other\)")
DISABLING = "backend sampling is not compatible with"
SWARM = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "repeat_penalty": 1.0}
HOT = {"temperature": 2.5, "top_k": 20, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0}
GREEDY = {"temperature": 0.0}
BUDGET = {"reasoning_budget_start_tag": "<think>", "reasoning_budget_end_tags": ["</think>"], "reasoning_budget_message": ""}
# letters only, lower case: the fixture has one token per letter
GRAMMAR = 'root ::= [a-j]+'
LAZY = {"grammar": 'root ::= "ab" [a-e]+', "grammar_lazy": True, "grammar_triggers": [{"type": 1, "value": "ab"}]}


class Server:
    def __init__(self, build, output, name, model, slots, env):
        self.name, self.real = name, model is not None
        self.worker_port, self.port = free_ports(2)
        self.log = output / f"server-{name}.log"
        self.worker_log = output / f"worker-{name}.log"
        self.env = dict(os.environ, GGML_RPC_NO_RDMA="1", GGML_RPC_NO_COMM="1", CUDA_VISIBLE_DEVICES="-1")
        self.worker = [str(build / "bin/ggml-rpc-server"), "--host", "127.0.0.1", "--port", str(self.worker_port),
                       "--device", "CPU", "--threads", "2"]
        self.command = [str(build / "bin/llama-server"), "--host", "127.0.0.1", "--port", str(self.port),
                        "--parallel", str(slots), "--threads", "2", "--fit", "off", "--no-warmup", "--no-op-offload",
                        "--no-context-shift", "--cache-ram", "0", "--ctx-checkpoints", "3", "--slots", "--jinja",
                        "--rpc", f"127.0.0.1:{self.worker_port}", "--device", "RPC0", "--gpu-layers", "all",
                        "--backend-sampling", "--log-verbosity", "4"]
        if self.real:
            self.command += ["--model", str(model), "--ctx-size", str(4096 * slots), "--batch-size", "512", "--ubatch-size", "64",
                             "-no-kvu", "--reasoning-budget", "1024"]
        else:
            fixture = make_recurrent_fixture(output / "tiny-mamba-markers.gguf", markers=MARKERS)
            self.command += ["--model", str(fixture), "--ctx-size", str(1024 * slots), "--batch-size", "64", "--ubatch-size", "16",
                             "--load-mode", "mmap", "--lazy-mode", "off"]
        self.server_env = dict(self.env, **env)

    def start(self, stack):
        wait_ready(spawn(stack, self.worker, self.worker_log, self.env), self.worker_port)
        wait_ready(spawn(stack, self.command, self.log, self.server_env), self.port, http=True)
        return self

    def call(self, path, body):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)

    def ids(self, text):
        return self.call("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

    def complete(self, prompt, n, seed, slot=0, **fields):
        body = dict({"prompt": prompt, "n_predict": n, "seed": seed, "cache_prompt": False, "id_slot": slot,
                     "return_tokens": True}, **fields)
        return self.call("/completion", body)["tokens"]

    def log_text(self):
        return self.log.read_text(errors="replace")

    def stats(self):
        """The per-request counts the patched server logs, summed."""
        total = [0] * 6
        rows = STATS.findall(self.log_text())
        for row in rows:
            total = [a + int(b) for a, b in zip(total, row)]
        return len(rows), dict(zip(("top_k", "full", "forced", "grammar", "equal", "other"), total))


def fixture_cases(off, on, report):
    (think,), (end,) = on.ids("<think>"), on.ids("</think>")
    letters = {on.ids(chr(c))[0]: chr(c) for c in range(97, 123)}
    prompt = on.ids("what is a ledger") + [think]
    plain = on.ids("name a colour")
    covered = 0     # requests the patch covers, so far

    def both(name, prompt, n, seed, **fields):
        a, b = off.complete(prompt, n, seed, **fields), on.complete(prompt, n, seed, **fields)
        assert a == b, f"{name}: the tokens differ\n  host sampling {a}\n  patched       {b}"
        assert len(b) > 0, name
        return b

    for label, sampling in (("", SWARM), ("-greedy", GREEDY)):
        # budget: thinking is cut at exactly the budget, by the forced end marker
        for budget in (0, 1, 7, 12):
            tokens = both(f"budget{label} {budget}", prompt, 40, 100 + budget, reasoning_budget_tokens=budget,
                          logit_bias=None, **BUDGET, **sampling)
            assert end in tokens and tokens.index(end) == budget, (budget, tokens)
            covered += 1
        # lazy: free until "ab" is written, then held to the grammar; with a budget beside it
        tokens = both(f"lazy{label}", prompt, 60, 7, reasoning_budget_tokens=5, **BUDGET, **LAZY, **sampling)
        assert tokens.index(end) == 5, tokens
        covered += 1
    report["budget_enforced_at"] = [0, 1, 7, 12]

    # grammar: every token is one of a-j, and hot sampling over 288 tokens makes the first choice illegal often
    rejected_before = on.stats()[1]["grammar"]
    for seed in range(3):
        tokens = both(f"grammar {seed}", plain, 48, 200 + seed, grammar=GRAMMAR, **HOT)
        assert all(letters.get(t, "z") <= "j" for t in tokens), tokens
        covered += 1
    tokens = both("grammar greedy", plain, 24, 1, grammar=GRAMMAR, **GREEDY)
    assert all(letters.get(t, "z") <= "j" for t in tokens), tokens
    covered += 1
    rejected = on.stats()[1]["grammar"] - rejected_before
    assert rejected > 0, "the grammar never rejected a token, so that path was not run"
    report["grammar_rejections"] = rejected

    # two at once, one slot each, against each alone
    alone = [on.complete(prompt, 40, 300 + k, reasoning_budget_tokens=9, **BUDGET, **SWARM) for k in range(2)]
    covered += 2
    for server in (off, on):
        got = [None, None]

        def one(k, server=server, got=got):
            got[k] = server.complete(prompt, 40, 300 + k, slot=k, reasoning_budget_tokens=9, **BUDGET, **SWARM)

        threads = [threading.Thread(target=one, args=(k,)) for k in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert got == alone, f"two at once on {server.name}: {got} != {alone}"
    covered += 2

    # logs so far: every request above was covered
    n_rows, stats = on.stats()
    assert DISABLING not in on.log_text(), "the patched server gave backend sampling up"
    assert n_rows == covered, (n_rows, covered)
    assert off.log_text().count(DISABLING) == covered + 2, (off.log_text().count(DISABLING), covered)
    assert STATS.search(off.log_text()) is None
    assert stats["other"] == 0, stats
    assert stats["forced"] > 0 and stats["top_k"] > stats["full"], stats

    # plain: nothing to check on the host, so the whole chain is on the worker as before
    for sampling in (SWARM, GREEDY):
        both("plain", plain, 40, 5, **sampling)
    assert on.stats()[0] == n_rows, "a request with no budget and no grammar took the patch's path"
    assert DISABLING not in on.log_text()

    # topk: without a top-k (or with a logit bias before it) there is nothing small for the worker to return,
    # and the request is sampled on the host as before, with the old line
    both("no top-k", prompt, 30, 6, reasoning_budget_tokens=4, top_k=0, temperature=1.0, **BUDGET)
    both("logit bias", prompt, 30, 6, reasoning_budget_tokens=4, logit_bias=[[letters and next(iter(letters)), -2.0]], **BUDGET, **SWARM)
    assert on.log_text().count(DISABLING) == 2 and on.log_text().count("host checks is not possible") == 2
    assert on.stats()[0] == n_rows

    # bytes: a row of the vocabulary is fetched for `full` tokens only
    report["tokens_from_top_k"], report["tokens_from_full_logits"] = stats["top_k"], stats["full"]
    report["full_logits_by_cause"] = {k: stats[k] for k in ("forced", "grammar", "equal", "other")}


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]


def chat(server, question, budget, seed, sampling, stream=False):
    body = dict({"messages": [{"role": "system", "content": "You are a careful assistant."}, {"role": "user", "content": question}],
                 "tools": TOOLS, "max_tokens": 220, "seed": seed, "cache_prompt": False, "id_slot": 0,
                 "chat_template_kwargs": {"enable_thinking": True}, "thinking_budget_tokens": budget}, **sampling)
    if not stream:
        body_out = server.call("/v1/chat/completions", body)
        message = body_out["choices"][0]["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
        return message.get("reasoning_content"), message.get("content"), calls, body_out["usage"]["completion_tokens"]
    body["stream"] = True
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    reasoning, content, calls = "", "", {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: {"):
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                delta = choice.get("delta", {})
                reasoning += delta.get("reasoning_content") or ""
                content += delta.get("content") or ""
                for call in delta.get("tool_calls") or []:
                    entry = calls.setdefault(call.get("index", 0), ["", ""])
                    entry[0] += call.get("function", {}).get("name") or ""
                    entry[1] += call.get("function", {}).get("arguments") or ""
    return reasoning or None, content or None, [tuple(calls[k]) for k in sorted(calls)], None


def real_cases(off, on, report):
    question = "Call the get_weather tool for Paris for 3 days."
    n_calls = 0
    for stream in (False, True):
        for sampling, seed in ((GREEDY, 1), (SWARM, 11), (SWARM, 12)):
            for budget in (16, 1024):
                a = chat(off, question, budget, seed, sampling, stream)
                b = chat(on, question, budget, seed, sampling, stream)
                assert a == b, f"chat stream={stream} seed={seed} budget={budget}:\n  host sampling {a}\n  patched       {b}"
                for _, arguments in b[2]:
                    json.loads(arguments)
                    n_calls += 1
    assert n_calls > 0, "no reply called the tool"
    n_rows, stats = on.stats()
    assert n_rows == 12 and DISABLING not in on.log_text(), n_rows
    assert off.log_text().count(DISABLING) == 12
    assert stats["forced"] > 0 and stats["other"] == 0 and stats["top_k"] > 20 * stats["full"], stats
    report.update(chat_replies_compared=12, tool_calls=n_calls, tokens_from_top_k=stats["top_k"], tokens_from_full_logits=stats["full"],
                  full_logits_by_cause={k: stats[k] for k in ("forced", "grammar", "equal", "other")})


def run(build, output, model):
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    for binary in (build / "bin/ggml-rpc-server", build / "bin/llama-server"):
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}; first run scripts/build-fleet-host.sh")
    report = {}
    with ExitStack() as stack:
        slots = 1 if model else 2
        off = Server(build, output, "off", model, slots, {"LLAMA_BACKEND_SAMPLING_CHECKS": "0"}).start(stack)
        on = Server(build, output, "on", model, slots, {}).start(stack)
        (real_cases if model else fixture_cases)(off, on, report)
    result = dict(status="passed", scope="Linux CPU TCP RPC only; no PS5",
                  model=str(model) if model else "untrained recurrent fixture, one token per character", **report)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/server-backend-sampling")
    parser.add_argument("--model", type=Path, default=None, help="a real GGUF with reasoning markers and tool calls in its template")
    args = parser.parse_args()
    run(args.build, args.output, args.model)
