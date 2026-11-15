#!/usr/bin/env python3
"""Large tensor transfers go to a worker in pieces (patches/fleet/0007), with real CPU RPC.

An RPC worker takes each tensor write into one buffer of the message's size. The
client now splits a transfer of more than GGML_RPC_CHUNK_MB MiB. Here a fixture
whose 12 feed-forward tensors are 8 MiB each (under the 10 MiB from which a
tensor goes by its hash instead) is loaded twice through one loopback
ggml-rpc-server: with pieces of 1 MiB, where the worker must see each of those
tensors arrive as 8 writes at rising offsets and nothing above 1 MiB, and with
splitting off, where it sees each whole. Both answer with the same tokens and
log-probabilities.

This is the host's CPU, not a PS5, and it shows the split, not the memory it
saves: that was measured on a console. Run after scripts/build-fleet-host.sh.
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
WIDE = 65536  # 65536 x 32 f32 = 8 MiB
MIB = 1 << 20


def run(build, output):
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    fixture = make_fixture(output / "wide-llama.gguf", ffn=WIDE)
    rpc_server, server = build / "bin/ggml-rpc-server", build / "bin/llama-server"
    for binary in (rpc_server, server):
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}; first run scripts/build-fleet-host.sh")
    worker_port, api = free_ports(2)
    env = dict(os.environ, GGML_RPC_DEBUG="2", GGML_RPC_NO_RDMA="1", GGML_RPC_NO_COMM="1")
    host = [str(server), "--model", str(fixture), "--host", "127.0.0.1", "--port", str(api),
            "--ctx-size", "128", "--parallel", "1", "--threads", "2", "--threads-batch", "2",
            "--batch-size", "128", "--ubatch-size", "128", "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--flash-attn", "off", "--fit", "off", "--no-warmup", "--load-mode", "mmap", "--lazy-mode", "off",
            "--no-op-offload", "--rpc", f"127.0.0.1:{worker_port}", "--device", "RPC0", "--gpu-layers", "all"]
    worker_log = output / "worker.log"
    seen = 0

    def load(name, chunk_mb):
        """One llama-server start against the same worker: its answer, and the offsets of the writes the worker saw."""
        nonlocal seen
        with ExitStack() as stack:
            process = spawn(stack, host, output / f"host-{name}.log", dict(env, GGML_RPC_CHUNK_MB=str(chunk_mb)))
            wait_ready(process, api, http=True)
            answer = complete(api)
        text = worker_log.read_text(errors="replace")
        part, seen = text[seen:], len(text)
        offsets = [int(m) for m in re.findall(r"\[set_tensor\] tensor: .*?, offset: (\d+), size:", part)]
        return answer, offsets

    with ExitStack() as stack:
        worker = spawn(stack, [str(rpc_server), "--host", "127.0.0.1", "--port", str(worker_port),
                               "--device", "CPU", "--threads", "2"], worker_log, env)
        wait_ready(worker, worker_port)

        split, offsets = load("split", 1)
        later = [o for o in offsets if o > 0]
        # each 8 MiB tensor in 8 writes: 7 of them past its start, a MiB apart
        assert len(later) == BIG * 7 and all(o % MIB == 0 and o < WIDE * 32 * 4 for o in later), (len(later), later[:10])
        assert sorted(set(later)) == [i * MIB for i in range(1, 8)], sorted(set(later))

        whole, offsets = load("whole", 0)
        assert not [o for o in offsets if o > 0], "a write past a tensor's start although splitting is off"
        assert whole["tokens"] == split["tokens"], (whole["tokens"], split["tokens"])
        assert whole["completion_probabilities"] == split["completion_probabilities"]

        default, offsets = load("default", 8)    # the default piece size: an 8 MiB tensor is not over it
        assert not [o for o in offsets if o > 0]
        assert default["tokens"] == split["tokens"]
    result = dict(status="passed", scope="Linux CPU TCP RPC only; no PS5", tensors=BIG, tensor_bytes=WIDE * 32 * 4,
                  writes_past_the_start_with_1_MiB_pieces=len(later), tokens=split["tokens"])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/fleet-chunk")
    args = parser.parse_args()
    try:
        run(args.build.resolve(), args.output.resolve())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
