#!/usr/bin/env python3
"""The RPC worker's weight cache (patches/fleet/0005), with real CPU RPC.

One loopback ggml-rpc-server with its cache on (-c; LLAMA_CACHE puts it in
build/fleet-cache) and llama-server loading a fixture whose 12 feed-forward
tensors are 12 MiB each, over the 10 MiB from which the client offers a tensor's
hash before sending it. The first load saves the 12 tensors, whole and under
their hashes; the second takes all 12 from the cache; a third, after one file is
cut short and another has a byte changed, drops both, gets them sent again and
saves them again. All three answer with the same tokens and log-probabilities.

This is the host's CPU, not a PS5. Run after scripts/build-fleet-host.sh.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import sys

from fleet_rpc_smoke import complete, free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_fixture

BIG = 12      # 4 blocks of ffn_gate, ffn_down and ffn_up
WIDE = 98304  # 98304 x 32 f32 = 12 MiB


def run(build, output):
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    fixture = make_fixture(output / "wide-llama.gguf", ffn=WIDE)
    cache = output / "cache" / "rpc"
    rpc_server, server = build / "bin/ggml-rpc-server", build / "bin/llama-server"
    for binary in (rpc_server, server):
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}; first run scripts/build-fleet-host.sh")
    worker_port, api = free_ports(2)
    env = dict(os.environ, GGML_RPC_DEBUG="2", GGML_RPC_NO_RDMA="1", GGML_RPC_NO_COMM="1",
               LLAMA_CACHE=str(output / "cache"))
    host = [str(server), "--model", str(fixture), "--host", "127.0.0.1", "--port", str(api),
            "--ctx-size", "128", "--parallel", "1", "--threads", "2", "--threads-batch", "2",
            "--batch-size", "128", "--ubatch-size", "128", "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--flash-attn", "off", "--fit", "off", "--no-warmup", "--load-mode", "mmap", "--lazy-mode", "off",
            "--no-op-offload", "--rpc", f"127.0.0.1:{worker_port}", "--device", "RPC0", "--gpu-layers", "all"]
    worker_log = output / "worker.log"
    seen = 0

    def load(name):
        """One llama-server start against the same worker: its answer, and what the worker logged meanwhile."""
        nonlocal seen
        with ExitStack() as stack:
            process = spawn(stack, host, output / f"host-{name}.log", env)
            wait_ready(process, api, http=True)
            answer = complete(api)
        text = worker_log.read_text(errors="replace")
        part, seen = text[seen:], len(text)
        return answer, dict(saved=len(re.findall(r"\[set_tensor\] saved to", part)),
                            hits=len(re.findall(r"\[set_tensor_hash\].*cache hit", part)),
                            misses=len(re.findall(r"\[set_tensor_hash\].*cache miss", part)),
                            dropped=len(re.findall(r"does not hold the tensor it names; dropped", part)))

    def files():
        return {f.name: f.stat().st_size for f in cache.iterdir()}

    with ExitStack() as stack:
        worker = spawn(stack, [str(rpc_server), "--host", "127.0.0.1", "--port", str(worker_port),
                               "--device", "CPU", "--threads", "2", "--cache"], worker_log, env)
        wait_ready(worker, worker_port)

        first, counts = load("first")
        saved = files()
        assert counts == dict(saved=BIG, hits=0, misses=BIG, dropped=0), counts
        assert len(saved) == BIG and all(re.fullmatch(r"[0-9a-f]{16}", n) for n in saved), saved
        assert set(saved.values()) == {WIDE * 32 * 4}, saved   # whole tensors; no .part left

        second, counts = load("second")
        assert counts == dict(saved=0, hits=BIG, misses=0, dropped=0), counts
        assert second["tokens"] == first["tokens"], (first["tokens"], second["tokens"])
        assert second["completion_probabilities"] == first["completion_probabilities"]

        names = sorted(saved)
        cut, changed = cache / names[0], cache / names[1]
        with cut.open("r+b") as stream:
            stream.truncate(saved[names[0]] // 2)
        with changed.open("r+b") as stream:
            stream.seek(4096)
            byte = stream.read(1)
            stream.seek(4096)
            stream.write(bytes([byte[0] ^ 0x01]))
        third, counts = load("third")
        assert counts == dict(saved=2, hits=BIG - 2, misses=2, dropped=2), counts
        assert third["tokens"] == first["tokens"], (first["tokens"], third["tokens"])
        assert third["completion_probabilities"] == first["completion_probabilities"]
        assert files() == saved, files()   # both saved again, whole
    result = dict(status="passed", scope="Linux CPU TCP RPC only; no PS5", cached_tensors=BIG,
                  tensor_bytes=WIDE * 32 * 4, tokens=first["tokens"])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/fleet-cache")
    args = parser.parse_args()
    try:
        run(args.build.resolve(), args.output.resolve())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
