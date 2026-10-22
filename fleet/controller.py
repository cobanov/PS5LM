"""Host CLI: acquire one GGUF, estimate a split, and serve through real RPC.

No node enrollment, payload deployment or network listener changes are performed.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from .models import ModelError, fetch_model, inspect_file
from .planner import FleetError, MIB, inventory, plan

LLAMA_REVISION = '04f7226a2c12c7f382c545a6db5a034266aa53cb'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / 'build/fleet-host/bin/llama-server'


def child_environment():
    # Environment CLI overrides could enable unrelated endpoints, routers, models
    # or caches. Do not silently let them invalidate the checked placement.
    env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
    env['GGML_RPC_NO_RDMA'] = '1'
    env['GGML_RPC_NO_COMM'] = '1'
    return env


def command(binary, model_path, placement, port=8081):
    nodes = placement['nodes']
    return [str(Path(binary).resolve()), '--model', str(Path(model_path).resolve()),
            '--rpc', ','.join(n['endpoint'] for n in nodes),
            '--device', ','.join(n['device'] for n in nodes),
            '--gpu-layers', 'all', '--split-mode', 'layer',
            '--tensor-split', ','.join(str(v) for v in placement['tensor_split']),
            '--fit', 'off', '--ctx-size', str(placement['context_tokens']), '--parallel', '1',
            '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--flash-attn', 'off',
            '--load-mode', 'mmap', '--lazy-mode', 'off', '--no-op-offload',
            '--batch-size', '128', '--ubatch-size', '128', '--no-warmup',
            '--host', '127.0.0.1', '--port', str(port)]


def probe(binary, config, timeout=20):
    """Use the pinned client's own RPC handshake, never implement a second protocol.

    An offline node, incompatible server, unexpected device topology or timeout
    is an error. Free memory is a snapshot, not a reservation.
    """
    config = copy.deepcopy(inventory(config))
    args = [str(Path(binary).resolve()), '--rpc', ','.join(n['endpoint'] for n in config['nodes']), '--list-devices']
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=child_environment())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FleetError(f'RPC preflight failed ({type(exc).__name__}); no model server started') from exc
    if result.returncode:
        raise FleetError(f'RPC preflight exited {result.returncode}; check worker/revision/network.\n' + result.stderr[-2000:])
    matches = re.findall(r'^\s*(RPC[0-9]+): ([0-9.]+:[0-9]+) \(([0-9]+) MiB, ([0-9]+) MiB free\)\s*$',
                         result.stdout, re.MULTILINE)
    if len(matches) != len(config['nodes']):
        raise FleetError('Expected exactly one RPC device per endpoint; refusing to guess topology')
    for index, (node, match) in enumerate(zip(config['nodes'], matches)):
        name, address, total_text, free_text = match
        if name != f'RPC{index}' or address != node['endpoint']:
            raise FleetError(f'{node["id"]}: RPC device order/endpoint differs from the requested inventory')
        total, free = int(total_text), int(free_text)
        if not 0 < free <= total:
            raise FleetError(f'{node["id"]}: worker did not report valid free memory')
        node['memory_mib'] = min(node['memory_mib'], free)
    return config


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(MIB), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    if Path(path).stat().st_size > MIB:
        raise FleetError('JSON configuration exceeds 1 MiB')
    return json.loads(Path(path).read_text())


def verify_binary(binary):
    binary = Path(binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FleetError(f'No executable at {binary}; run scripts/build-fleet-host.sh first')
    manifest = binary.parent.parent / 'fleet-build.json'
    if not manifest.is_file():
        raise FleetError(f'Missing build manifest {manifest}; use scripts/build-fleet-host.sh')
    data = read_json(manifest)
    if data.get('llama_revision') != LLAMA_REVISION:
        raise FleetError('Host build revision differs from the pinned PS5 worker; rebuild both together')
    if data.get('server_sha256') != checksum(binary):
        raise FleetError('Host binary checksum differs from fleet-build.json; rebuild the host')


class _StopSignal(Exception):
    def __init__(self, signum):
        self.signum = signum


def run_foreground(argv):
    """Own the server until exit; wrapper SIGTERM must not orphan RPC users."""
    child = None

    def stop(signum, _frame):
        raise _StopSignal(signum)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        child = subprocess.Popen(argv, env=child_environment())
        return child.wait()
    except KeyboardInterrupt:
        return 130
    except _StopSignal as exc:
        return 128 + exc.signum
    finally:
        signal.signal(signal.SIGTERM, previous)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Experimental PS5LM single-model layer fleet (not a load balancer)')
    parser.add_argument('action', choices=('plan', 'serve'))
    parser.add_argument('--model', required=True, help='single-file local GGUF or public HTTPS URL')
    parser.add_argument('--nodes', required=True, help='inventory JSON; see fleet/nodes.example.json')
    parser.add_argument('--context', type=int, default=4096, help='total context for one sequence, default 4096')
    parser.add_argument('--binary', default=str(DEFAULT_BINARY))
    parser.add_argument('--cache-dir', default=str(ROOT / 'models/fleet-cache'))
    parser.add_argument('--sha256', help='optional expected full-file SHA256')
    parser.add_argument('--max-download-gib', type=int, default=32)
    parser.add_argument('--port', type=int, default=8081, help='controller API port, localhost only')
    parser.add_argument('--plan-out', help='write reviewed JSON plan (not executable state)')
    parser.add_argument('--ack-insecure-rpc', action='store_true',
                        help='acknowledge RPC is unauthenticated and all endpoints are on an isolated trusted network')
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535 or not 1 <= args.max_download_gib <= 128:
            raise FleetError('Port must be 1..65535 and download limit 1..128 GiB')
        if args.sha256 and not re.fullmatch(r'[0-9a-fA-F]{64}', args.sha256):
            raise FleetError('--sha256 must contain exactly 64 hexadecimal characters')
        config = inventory(read_json(args.nodes))
        if args.action == 'serve':
            if not args.ack_insecure_rpc:
                raise FleetError('Serving requires --ack-insecure-rpc; isolate and trust every RPC peer first')
            verify_binary(args.binary)
            # Fail early, before downloading a large model. Probe again immediately
            # before execution because downloading/hashing may take a long time.
            config = probe(args.binary, config)
        if args.model.startswith('https://'):
            path, digest = fetch_model(args.model, args.cache_dir, max_bytes=args.max_download_gib * 1024**3,
                                       max_cache_bytes=max(64, args.max_download_gib) * 1024**3, max_seconds=3600)
        elif '://' in args.model:
            raise FleetError('Only local paths and public HTTPS model URLs are accepted')
        else:
            path = Path(args.model).resolve()
            # Reject devices/directories before hashing, and inspect metadata first.
            inspect_file(path)
            digest = checksum(path)
        if args.sha256 and digest.lower() != args.sha256.lower():
            raise FleetError('Full model SHA256 does not match --sha256; no server started')
        model = inspect_file(path)
        if args.action == 'serve':
            config = probe(args.binary, config)
        placement = plan(model, config, args.context)
        placement.update(model_path=str(Path(path).resolve()), model_sha256=digest,
                         llama_revision=LLAMA_REVISION,
                         memory_source='min(operator_budget,live_RPC_snapshot)' if args.action == 'serve' else 'operator_budgets_only')
        placement['argv'] = command(args.binary, path, placement, args.port)
        encoded = json.dumps(placement, indent=2) + '\n'
        if args.plan_out:
            Path(args.plan_out).write_text(encoded)
        print(encoded, end='', flush=True)
        if args.action == 'plan':
            return 0
        print('Starting host llama-server; wait for its ready/health response. A printed plan is not successful inference.', file=sys.stderr)
        # Foreground process: no daemon, enrollment, automatic console deployment,
        # or insecure public API. Ctrl+C reaches both and stops this session.
        return run_foreground(placement['argv'])
    except (FleetError, ModelError, OSError, ValueError) as exc:
        print(f'ps5lm-fleet: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
