#!/usr/bin/env python3
"""patches/0011 on a host llama-server: with a quantized KV cache on one RPC worker, the
cache's Hadamard rotation matrices are kept on the worker instead of being uploaded as
inputs to every graph, and the answer matches the same model run without RPC.

Run after scripts/build-fleet-host.sh. The model is an untrained fixture with 64-wide
heads (the narrowest that llama.cpp rotates), so only tokens and logprobs are compared.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import math
import os

from fleet_rpc_smoke import complete, free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_fixture

KEPT = "init_attn_rot_dev: rotation matrices kept on RPC0"
INPUT = "input tensor '"


def run(build, output):
    output.mkdir(parents=True, exist_ok=True)
    fixture = make_fixture(output / "tiny-llama-hd64.gguf", head_dim=64)
    rpc_server = build / "bin/ggml-rpc-server"
    server = build / "bin/llama-server"
    for binary in (rpc_server, server):
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}; first run scripts/build-fleet-host.sh")
    rpc, api = free_ports(2)
    env = dict(os.environ, GGML_RPC_NO_RDMA="1", GGML_RPC_NO_COMM="1")
    common = [str(server), "--model", str(fixture), "--host", "127.0.0.1", "--port", str(api),
              "--ctx-size", "128", "--parallel", "1", "--threads", "2", "--threads-batch", "2",
              "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--flash-attn", "on",
              "--fit", "off", "--no-warmup", "--load-mode", "mmap", "--lazy-mode", "off",
              "--no-op-offload", "--log-verbosity", "5"]
    with ExitStack() as stack:
        local = spawn(stack, common + ["--device", "none", "--gpu-layers", "0"], output / "local.log", env)
        wait_ready(local, api, http=True)
        expected = complete(api)
    with ExitStack() as stack:
        worker = spawn(stack, [str(rpc_server), "--host", "127.0.0.1", "--port", str(rpc),
                               "--device", "CPU", "--threads", "2"], output / "worker.log", env)
        wait_ready(worker, rpc)
        remote = spawn(stack, common + ["--rpc", f"127.0.0.1:{rpc}", "--device", "RPC0",
                                        "--gpu-layers", "all"], output / "controller.log", env)
        wait_ready(remote, api, http=True)
        actual = complete(api)
    local_log = (output / "local.log").read_text(errors="replace")
    controller_log = (output / "controller.log").read_text(errors="replace")
    # On the CPU the cache is in host memory and the matrices stay graph inputs.
    assert "n_rot_k = 64" in local_log and KEPT not in local_log, "the fixture's cache is not rotated"
    assert "attn_inp_k_rot'" in local_log, "the local run did not use the rotation inputs"
    assert KEPT in controller_log, "the rotation matrices were not kept on the RPC worker"
    uploads = [line for line in controller_log.splitlines() if INPUT in line and "_rot'" in line]
    assert not uploads, f"rotation matrices still graph inputs: {uploads[:2]}"
    assert expected["tokens"] == actual["tokens"], (expected["tokens"], actual["tokens"])
    # Byte tokens that form one character share an entry, so there can be fewer than 8.
    left, right = expected["completion_probabilities"], actual["completion_probabilities"]
    assert left and len(left) == len(right), (len(left), len(right))
    error = 0.0
    for a, b in zip(left, right):
        assert [x["id"] for x in a["top_logprobs"]] == [y["id"] for y in b["top_logprobs"]], (a, b)
        error = max([error] + [abs(x["logprob"] - y["logprob"])
                               for x, y in zip(a["top_logprobs"], b["top_logprobs"])])
    assert math.isfinite(error) and error <= 1e-5, error
    print(json.dumps({"status": "passed", "tokens": actual["tokens"], "max_logprob_error": error}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/fleet-kv-rotation")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve())
