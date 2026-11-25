#!/usr/bin/env python3
"""patches/0022 on a host llama-server: the media marker that /props publishes never reaches the
prompt as text. A tool that read /props put it into a chat, and mtmd then counted one image more
than the request had: every later request of that chat failed with "Failed to tokenize prompt".

Run after scripts/build-fleet-host.sh. The model is the untrained fixture without a projector, so
only the rendered prompt is checked; on the console the same requests were answered.
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

# Every role and the tool definitions, as plain text.
TEMPLATE = (
    "{%- if tools -%}<tools>{{ tools | tojson }}{%- endif -%}"
    "{%- for m in messages -%}<{{ m.role }}>{{ m.content }}{%- endfor -%}"
    "{%- if add_generation_prompt -%}<assistant>{%- endif -%}"
)


def request(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def run(build, output):
    output.mkdir(parents=True, exist_ok=True)
    fixture = make_fixture(output / "tiny-llama.gguf")
    template = output / "plain.jinja"
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

        status, props = request(port, "/props")
        assert status == 200, props
        marker = props["media_marker"]
        assert marker, props

        # A user message, a tool result and a tool definition that quote /props.
        status, body = request(port, "/apply-template", {
            "messages": [
                {"role": "user", "content": f"what is {marker}?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "props", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": json.dumps(props)},
                {"role": "user", "content": [{"type": "text", "text": f"and {marker}"}]}],
            "tools": [{"type": "function", "function": {"name": "props", "parameters": {"type": "object"},
                "description": f"returns /props, e.g. {marker}"}}]})
        assert status == 200, body
        prompt = body["prompt"]
        assert marker not in prompt, prompt
        assert prompt.count("[media marker]") >= 4, prompt

        # A chat that holds the marker is still answered on every endpoint.
        cases = {
            "/v1/chat/completions": {"max_tokens": 1, "messages": [
                {"role": "user", "content": f"what is {marker}?"}]},
            "/v1/messages": {"max_tokens": 1, "messages": [
                {"role": "user", "content": f"what is {marker}?"}]},
            "/v1/responses": {"max_output_tokens": 1, "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"what is {marker}?"}]}]},
        }
        for path, body in cases.items():
            status, reply = request(port, path, body)
            assert status == 200, f"{path}: {status} {reply}"
    print("server media marker: written as [media marker] in messages, tool results and tool definitions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/server-media-marker")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve())
