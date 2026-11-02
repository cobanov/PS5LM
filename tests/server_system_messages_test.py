#!/usr/bin/env python3
"""patches/0004 on a host llama-server: system and developer messages after the
first one reach a template that refuses them (Qwen's rule), on every endpoint.

Run after scripts/build-fleet-host.sh. The model is the untrained fixture, so
only the rendered prompt and the HTTP status are checked, not the words.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import os
import urllib.error
import urllib.request

from fleet_rpc_smoke import free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_fixture

# Qwen's rule, cut down: a system message anywhere but first is an error.
TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' and not loop.first -%}"
    "{{- raise_exception('System message must be at the beginning.') -}}"
    "{%- endif -%}"
    "<{{ m.role }}>{{ m.content }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<assistant>{%- endif -%}"
)


def post(port, path, body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def run(build, output):
    output.mkdir(parents=True, exist_ok=True)
    fixture = make_fixture(output / "tiny-llama.gguf")
    template = output / "system-first.jinja"
    template.write_text(TEMPLATE)
    server = build / "bin/llama-server"
    if not server.is_file():
        raise FileNotFoundError(f"{server}; first run scripts/build-fleet-host.sh")
    (port,) = free_ports(1)
    command = [str(server), "--model", str(fixture), "--host", "127.0.0.1", "--port", str(port),
               "--ctx-size", "256", "--parallel", "1", "--threads", "2", "--device", "none",
               "--gpu-layers", "0", "--fit", "off", "--no-warmup", "--jinja",
               "--chat-template-file", str(template)]
    with ExitStack() as stack:
        process = spawn(stack, command, output / "server.log", dict(os.environ))
        wait_ready(process, port, http=True)

        # The order and the joins, on the rendered prompt.
        status, body = post(port, "/apply-template", {"messages": [
            {"role": "system", "content": "a"}, {"role": "user", "content": "b"},
            {"role": "system", "content": "c"}, {"role": "developer", "content": "d"}]})
        assert status == 200, body
        assert body["prompt"] == "<system>a\n\nc\n\nd<user>b<assistant>", body["prompt"]

        # No leading system message: the late ones become one in front.
        status, body = post(port, "/apply-template", {"messages": [
            {"role": "user", "content": "b"}, {"role": "system", "content": "c"}]})
        assert status == 200, body
        assert body["prompt"] == "<system>c<user>b<assistant>", body["prompt"]

        cases = {
            "/v1/chat/completions": {"max_tokens": 1, "messages": [
                {"role": "system", "content": "a"}, {"role": "user", "content": "b"},
                {"role": "system", "content": "c"}]},
            # What codex sends (#55): instructions, then two developer messages.
            "/v1/responses": {"max_output_tokens": 1, "instructions": "a", "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "c"}]},
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "d"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "b"}]}]},
            # What Claude Code sends: a system message inside `messages`.
            "/v1/messages": {"max_tokens": 1, "system": "a", "messages": [
                {"role": "user", "content": "b"}, {"role": "system", "content": "c"},
                {"role": "user", "content": "e"}]},
        }
        for path, request in cases.items():
            status, body = post(port, path, request)
            assert status == 200, f"{path}: {status} {body}"
    print("server system messages: folded into the first on chat, responses and messages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/server-system-messages")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve())
