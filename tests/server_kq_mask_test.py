#!/usr/bin/env python3
"""patches/0012 on a host llama-server: with the KV cache on one RPC worker, the causal KQ
mask is built on the worker from cell and token positions instead of being uploaded as a
full [n_kv, n_tokens] graph input, and the answer matches the same model run without RPC.

Run after scripts/build-fleet-host.sh. The model is an untrained fixture, so only tokens and
logprobs are compared. Both the f16 flash-attention mask and the f32 soft_max mask are run.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import math
import os

from fleet_rpc_smoke import complete, free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_fixture

DEVICE = "init_kq_mask_dev: KQ mask built on the device"
INPUT = "input tensor '"


def compare(expected, actual):
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
    return error


def run_case(build, output, fixture, flash_attn):
    rpc_server = build / "bin/ggml-rpc-server"
    server = build / "bin/llama-server"
    rpc, api = free_ports(2)
    env = dict(os.environ, GGML_RPC_NO_RDMA="1", GGML_RPC_NO_COMM="1")
    common = [str(server), "--model", str(fixture), "--host", "127.0.0.1", "--port", str(api),
              "--ctx-size", "256", "--parallel", "2", "-no-kvu", "--threads", "2",
              "--threads-batch", "2", "--cache-type-k", "f16", "--cache-type-v", "f16",
              "--flash-attn", flash_attn, "--fit", "off", "--no-warmup", "--load-mode", "mmap",
              "--lazy-mode", "off", "--no-op-offload", "--log-verbosity", "5"]
    local_log, controller_log = output / f"local-fa-{flash_attn}.log", output / f"controller-fa-{flash_attn}.log"
    with ExitStack() as stack:
        local = spawn(stack, common + ["--device", "none", "--gpu-layers", "0"], local_log, env)
        wait_ready(local, api, http=True)
        expected = complete(api)
    with ExitStack() as stack:
        worker = spawn(stack, [str(rpc_server), "--host", "127.0.0.1", "--port", str(rpc),
                               "--device", "CPU", "--threads", "2"], output / f"worker-fa-{flash_attn}.log", env)
        wait_ready(worker, rpc)
        remote = spawn(stack, common + ["--rpc", f"127.0.0.1:{rpc}", "--device", "RPC0",
                                        "--gpu-layers", "all"], controller_log, env)
        wait_ready(remote, api, http=True)
        actual = complete(api)
    local_text = local_log.read_text(errors="replace")
    controller_text = controller_log.read_text(errors="replace")
    # On the CPU the cache is in host memory and the mask stays a graph input.
    assert DEVICE not in local_text and "attn_inp_kq_mask'" in local_text, "the local run did not use the host mask"
    assert DEVICE in controller_text, "the KQ mask was not built on the RPC worker"
    uploads = [line for line in controller_text.splitlines() if INPUT in line and "attn_inp_kq_mask'" in line]
    assert not uploads, f"the KQ mask is still a graph input: {uploads[:2]}"
    assert "attn_inp_kq_pos_kv'" in controller_text, "the cell positions were not graph inputs"
    return actual["tokens"], compare(expected, actual)


def run(build, output):
    output.mkdir(parents=True, exist_ok=True)
    for binary in (build / "bin/ggml-rpc-server", build / "bin/llama-server"):
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}; first run scripts/build-fleet-host.sh")
    fixture = make_fixture(output / "tiny-llama.gguf")
    result = {"status": "passed"}
    for flash_attn in ("on", "off"):
        tokens, error = run_case(build, output, fixture, flash_attn)
        result[f"fa_{flash_attn}"] = {"tokens": tokens, "max_logprob_error": error}
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/fleet-kq-mask")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve())
