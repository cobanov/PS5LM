#!/usr/bin/env python3
"""patches/0010 on a host llama-server: with --ctx-checkpoints 1, as the app and the
fleet run it, a prompt makes only the context checkpoint it keeps (4 tokens before
its end), and the next turn still rolls back to that one.

Run after scripts/build-fleet-host.sh. The model is an untrained recurrent fixture,
so only the server's log and timings are checked, not the words.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import os
import time
import urllib.request

from fleet_rpc_smoke import free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_recurrent_fixture

TEMPLATE = (
    "{%- for m in messages -%}<{{ m.role }}>{{ m.content }}{%- endfor -%}"
    "{%- if add_generation_prompt -%}<assistant>{%- endif -%}"
)
EVICTED = "erasing old context checkpoint"


def chat(port, messages):
    body = {"messages": messages, "max_tokens": 2, "temperature": 0}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    return body["timings"], body["choices"][0]["message"].get("content") or ""


def evictions(log, requests):
    # The server logs from its own thread: wait for the request's timings, the
    # last lines it logs for it.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        text = log.read_text(errors="replace")
        if text.count("total time =") >= requests:
            return text.count(EVICTED)
        time.sleep(0.05)
    raise TimeoutError(f"{log}: no timings for request {requests}")


def run(build, output):
    output.mkdir(parents=True, exist_ok=True)
    fixture = make_recurrent_fixture(output / "tiny-mamba.gguf")
    template = output / "roles.jinja"
    template.write_text(TEMPLATE)
    server = build / "bin/llama-server"
    if not server.is_file():
        raise FileNotFoundError(f"{server}; first run scripts/build-fleet-host.sh")
    (port,) = free_ports(1)
    log = output / "server.log"
    # Batches of 16 tokens put the places llama-server checkpoints a prompt (the last
    # user message, n_ubatch before the end, 4 before it) in different batches.
    command = [str(server), "--model", str(fixture), "--host", "127.0.0.1", "--port", str(port),
               "--ctx-size", "1024", "--parallel", "1", "--threads", "2", "--device", "none",
               "--gpu-layers", "0", "--fit", "off", "--no-warmup", "--jinja",
               "--chat-template-file", str(template), "--batch-size", "16", "--ubatch-size", "16",
               "--ctx-checkpoints", "1", "--cache-ram", "0"]
    system = {"role": "system", "content": "answer in one word " * 4}
    user = {"role": "user", "content": "name a colour that is not red or blue"}
    with ExitStack() as stack:
        process = spawn(stack, command, log, dict(os.environ))
        wait_ready(process, port, http=True)

        first, reply = chat(port, [system, user])
        assert first["cache_n"] == 0, first
        evicted = evictions(log, 1)
        assert evicted == 0, f"turn 1 made {evicted + 1} checkpoints to keep one"

        # Another reply than the one generated, so the recurrent state has to go back
        # to the checkpoint, 4 tokens before the end of turn 1's prompt.
        other = "qq" if reply.startswith("z") else "zz"
        second, _ = chat(port, [system, user, {"role": "assistant", "content": other},
                                {"role": "user", "content": "and another"}])
        assert second["cache_n"] == first["prompt_n"] - 4, (first, second)
        evicted = evictions(log, 2)
        assert evicted == 1, f"turn 2 evicted {evicted} checkpoints, not just turn 1's"
    print(f"server checkpoints: one a prompt; turn 2 reused {second['cache_n']} of turn 1's {first['prompt_n']} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/server-checkpoints")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve())
