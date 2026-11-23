#!/usr/bin/env python3
"""patches/0023 on a host llama-server: a streamed reply leaves its keep-alive connection open.

The chunk provider called done() and then returned false, and cpp-httplib took that as a cancelled
response: it closed the connection right after one that advertised Keep-Alive. A client that
reuses it at once (Open WebUI sends each tool-call continuation within a fraction of a second) got
a disconnect. Here two streamed requests and a plain one go over one connection, on each endpoint.

Run after scripts/build-fleet-host.sh. The model is the untrained fixture, so only the HTTP framing
is checked, not the words.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import os
import re
import socket

from fleet_rpc_smoke import free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_fixture


def post(path, body):
    data = json.dumps(body).encode()
    return (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n\r\n").encode() + data


def read_response(sock):
    """One response; a chunked body ends at the zero chunk. EOFError if the server closed."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise EOFError("connection closed before a response")
        data += chunk
    head, body = data.split(b"\r\n\r\n", 1)
    if b"transfer-encoding: chunked" in head.lower():
        while not body.endswith(b"0\r\n\r\n"):
            chunk = sock.recv(4096)
            if not chunk:
                raise EOFError("connection closed inside the body")
            body += chunk
    else:
        length = int(re.search(rb"(?i)content-length:\s*(\d+)", head).group(1))
        while len(body) < length:
            chunk = sock.recv(4096)
            if not chunk:
                raise EOFError("connection closed inside the body")
            body += chunk
    return head.split(b"\r\n", 1)[0].decode(), body


def run(build, output):
    output.mkdir(parents=True, exist_ok=True)
    fixture = make_fixture(output / "tiny-llama.gguf")
    server = build / "bin/llama-server"
    if not server.is_file():
        raise FileNotFoundError(f"{server}; first run scripts/build-fleet-host.sh")
    (port,) = free_ports(1)
    command = [str(server), "--model", str(fixture), "--host", "127.0.0.1", "--port", str(port),
               "--ctx-size", "256", "--parallel", "1", "--threads", "2", "--device", "none",
               "--gpu-layers", "0", "--fit", "off", "--no-warmup"]
    streams = {
        "/v1/chat/completions": {"stream": True, "max_tokens": 2,
                                 "messages": [{"role": "user", "content": "hi"}]},
        "/v1/messages": {"stream": True, "max_tokens": 2,
                         "messages": [{"role": "user", "content": "hi"}]},
        "/v1/responses": {"stream": True, "max_output_tokens": 2, "input": "hi"},
        "/completion": {"stream": True, "n_predict": 2, "prompt": "hi"},
    }
    with ExitStack() as stack:
        process = spawn(stack, command, output / "server.log", dict(os.environ))
        wait_ready(process, port, http=True)
        for path, body in streams.items():
            with socket.create_connection(("127.0.0.1", port), timeout=60) as sock:
                for i in range(2):
                    sock.sendall(post(path, body))
                    status, data = read_response(sock)
                    assert status == "HTTP/1.1 200 OK", f"{path} #{i + 1}: {status} {data[:200]!r}"
                # A plain request at once on the same connection, as a client's next call.
                sock.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
                status, data = read_response(sock)
                assert status == "HTTP/1.1 200 OK", f"{path} then /health: {status} {data[:200]!r}"
    print("server stream keep-alive: two streams and a plain request on one connection, every endpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/server-stream-keepalive")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve())
