#!/usr/bin/env python3
"""Real CPU RPC smoke: one tiny model, two localhost workers, one API host.

This proves host TCP RPC graph execution, not PS5 hardware or Vulkan support.
Run after scripts/build-fleet-host.sh. Logs/results remain in build/fleet-smoke.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import math
import os
import re
import signal
import sys
import socket
import subprocess
import time
import urllib.error
import urllib.request

from make_fleet_fixture import ROOT, make_fixture


def free_ports(count):
    # Hold all sockets until we have distinct ports. Workers take them next.
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]


def stop(process):
    # Own a new process group so the CLI and its llama-server child are reaped.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def spawn(stack, command, logfile, env):
    stream = stack.enter_context(logfile.open("w"))
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    stack.callback(stop, process)
    return process


def wait_ready(process, port, http=False):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited with {process.returncode} before port {port} was ready")
        try:
            if http:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200:
                        return
            else:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise TimeoutError(f"port {port} was not ready in 60 seconds")


def complete(port):
    payload = {"prompt": [1, 259, 260, 261], "n_predict": 8, "temperature": 0,
               "seed": 42, "ignore_eos": True, "return_tokens": True,
               "n_probs": 5, "cache_prompt": False}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    assert len(result.get("tokens", [])) == 8, result
    return result


def compare_logprobs(expected, actual):
    left = expected["completion_probabilities"]
    right = actual["completion_probabilities"]
    assert len(left) == len(right) == 8
    differences = []
    for a, b in zip(left, right):
        assert len(a["top_logprobs"]) == len(b["top_logprobs"]) == 5
        for x, y in zip(a["top_logprobs"], b["top_logprobs"]):
            assert x["id"] == y["id"], (x, y)
            difference = abs(x["logprob"] - y["logprob"])
            assert math.isfinite(difference) and difference <= 1e-5, difference
            differences.append(difference)
    return max(differences)


def run(build, output):
    output.mkdir(parents=True, exist_ok=True)
    fixture = make_fixture(output / "tiny-llama.gguf")
    rpc_server = build / "bin/ggml-rpc-server"
    server = build / "bin/llama-server"
    for binary in (rpc_server, server):
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}; first run scripts/build-fleet-host.sh")
    rpc_a, rpc_b, api = free_ports(3)
    endpoints = [f"127.0.0.1:{port}" for port in (rpc_a, rpc_b)]
    devices = ["RPC0", "RPC1"]
    env = dict(os.environ, GGML_RPC_DEBUG="1", GGML_RPC_NO_RDMA="1", GGML_RPC_NO_COMM="1")
    common = [str(server), "--model", str(fixture), "--host", "127.0.0.1", "--port", str(api),
              "--ctx-size", "128", "--parallel", "1", "--threads", "2", "--threads-batch", "2",
              "--batch-size", "128", "--ubatch-size", "128", "--cache-type-k", "f16",
              "--cache-type-v", "f16", "--flash-attn", "off", "--fit", "off", "--no-warmup",
              "--load-mode", "mmap", "--lazy-mode", "off", "--no-op-offload", "--log-verbosity", "5"]
    commands = []
    with ExitStack() as local_stack:
        local_command = common + ["--device", "none", "--gpu-layers", "0"]
        commands.append(local_command)
        local = spawn(local_stack, local_command, output / "local.log", env)
        wait_ready(local, api, http=True)
        expected = complete(api)
    with ExitStack() as stack:
        for suffix, port in (("a", rpc_a), ("b", rpc_b)):
            command = [str(rpc_server), "--host", "127.0.0.1", "--port", str(port),
                       "--device", "CPU", "--threads", "2"]
            commands.append(command)
            worker = spawn(stack, command, output / f"worker-{suffix}.log", env)
            wait_ready(worker, port)
        # Five offloadable units: block 0-3 plus output. Midpoint 3/10 puts
        # blocks 0-1 on A and blocks 2-3 plus output on B, away from boundaries.
        command = common + ["--rpc", ",".join(endpoints), "--device", ",".join(devices),
                            "--gpu-layers", "all", "--split-mode", "layer", "--tensor-split", "3,7"]
        commands.append(command)
        remote = spawn(stack, command, output / "controller.log", env)
        wait_ready(remote, api, http=True)
        actual = complete(api)
        assert expected["tokens"] == actual["tokens"], (expected["tokens"], actual["tokens"])
        rpc_logprob_error = compare_logprobs(expected, actual)
    controller_log = (output / "controller.log").read_text(errors="replace")
    placements = {}
    for layer, device in re.findall(r"load_tensors: layer\s+(\d+) assigned to device ([^\s,]+)", controller_log):
        placements[int(layer)] = device
    assert placements == {0: devices[0], 1: devices[0], 2: devices[1], 3: devices[1], 4: devices[1]}, placements
    worker_evidence = {}
    for suffix in ("a", "b"):
        log = (output / f"worker-{suffix}.log").read_text(errors="replace")
        allocations = len(re.findall(r"\[alloc_buffer\].*remote_size: [1-9]", log))
        computes = len(re.findall(r"\[graph_(?:re)?compute\]", log))
        assert allocations > 0 and computes > 0, (suffix, allocations, computes)
        assert "invalid data ptr" not in log, suffix
        worker_evidence[suffix] = {"allocations": allocations, "compute_calls": computes}
    # Exercise the actual user-facing CLI after restarting both workers.
    cli_a, cli_b, cli_api = free_ports(3)
    cli_endpoints = [f"127.0.0.1:{port}" for port in (cli_a, cli_b)]
    inventory = {"host_memory_mib": 128, "host_reserve_mib": 16,
                 "nodes": [{"id": f"worker-{i}", "endpoint": endpoint,
                            "memory_mib": 128, "reserve_mib": 16}
                           for i, endpoint in enumerate(cli_endpoints)]}
    inventory_path = output / "nodes.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    cli_plan = output / "cli-plan.json"
    with ExitStack() as stack:
        for suffix, port in (("a", cli_a), ("b", cli_b)):
            command = [str(rpc_server), "--host", "127.0.0.1", "--port", str(port),
                       "--device", "CPU", "--threads", "2"]
            commands.append(command)
            worker = spawn(stack, command, output / f"cli-worker-{suffix}.log", env)
            wait_ready(worker, port)
        command = [sys.executable, "-m", "fleet", "serve", "--model", str(fixture),
                   "--nodes", str(inventory_path), "--context", "128", "--binary", str(server),
                   "--port", str(cli_api), "--plan-out", str(cli_plan), "--ack-insecure-rpc"]
        commands.append(command)
        cli_env = dict(env, PYTHONPATH=str(ROOT))
        controller = spawn(stack, command, output / "cli-controller.log", cli_env)
        wait_ready(controller, cli_api, http=True)
        cli_response = complete(cli_api)
        assert cli_response["tokens"] == expected["tokens"], cli_response
        cli_logprob_error = compare_logprobs(expected, cli_response)
        checked_plan = json.loads(cli_plan.read_text(errors="replace"))
        assert checked_plan["memory_source"] == "min(operator_budget,live_RPC_snapshot)", checked_plan
        assert checked_plan["tensor_split"] == [3, 7], checked_plan
    for suffix in ("a", "b"):
        log = (output / f"cli-worker-{suffix}.log").read_text(errors="replace")
        assert re.search(r"\[alloc_buffer\].*remote_size: [1-9]", log), suffix
        assert "[graph_compute]" in log, suffix
    (output / "cli-response.json").write_text(json.dumps(cli_response, indent=2) + "\n")
    result = {"status": "passed", "scope": "Linux CPU TCP RPC only; no PS5 or Vulkan validation",
              "model": str(fixture), "model_bytes": fixture.stat().st_size,
              "tokens": actual["tokens"], "local_tokens": expected["tokens"],
              "fleet_cli_tokens": cli_response["tokens"], "fleet_cli_preflight": "passed",
              "top5_logprobs_compared_per_run": 40,
              "rpc_max_logprob_error": rpc_logprob_error, "cli_max_logprob_error": cli_logprob_error,
              "placements": placements, "workers": worker_evidence, "commands": commands}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "local-response.json").write_text(json.dumps(expected, indent=2) + "\n")
    (output / "rpc-response.json").write_text(json.dumps(actual, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "commands"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/fleet-smoke")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve())
