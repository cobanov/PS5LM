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
from .planner import CACHE_TYPES, FleetError, MIB, UBATCH, inventory, plan

LLAMA_REVISION = '970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / 'build/fleet-host/bin/llama-server'


def child_environment():
    # Environment CLI overrides could enable unrelated endpoints, routers, models
    # or caches. Do not silently let them invalidate the checked placement.
    env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
    env['GGML_RPC_NO_RDMA'] = '1'
    env['GGML_RPC_NO_COMM'] = '1'
    return env


def cache_args(placement):
    """The context, slots, cache type and flash attention the plan was priced for."""
    cache = placement.get('cache_dtype', 'f16')
    return ['--ctx-size', str(placement['context_tokens']),
            '--parallel', str(placement.get('parallel_sequences', 1)),
            *(['--kv-unified'] if placement.get('kv_unified') else []),
            '--cache-type-k', cache, '--cache-type-v', cache,
            '--flash-attn', 'on' if placement.get('flash_attn') else 'off']


def command(binary, model_path, placement, port=8081, host='127.0.0.1', api_key=None):
    """llama-server arguments for a placement.

    The API listens on loopback unless host is '0.0.0.0', which the control
    server sets only when asked to share the API on the network; api_key then
    makes llama-server require that key on every request.
    """
    if host not in ('127.0.0.1', '0.0.0.0'):
        raise FleetError('The API listens on 127.0.0.1, or on 0.0.0.0 when shared on purpose')
    nodes = placement['nodes']
    return [str(Path(binary).resolve()), '--model', str(Path(model_path).resolve()),
            '--rpc', ','.join(n['endpoint'] for n in nodes),
            '--device', ','.join(n['device'] for n in nodes),
            '--gpu-layers', 'all', '--split-mode', 'layer',
            '--tensor-split', ','.join(str(v) for v in placement['tensor_split']),
            '--fit', 'off', *cache_args(placement),
            '--load-mode', 'mmap', '--lazy-mode', 'off', '--no-op-offload',
            '--batch-size', str(UBATCH), '--ubatch-size', str(UBATCH), '--no-warmup',
            # Sampling where the logits are, on the worker with the output layer:
            # one token id comes back instead of the logits (about 1 MB a token
            # for a 248k vocabulary). On a PS5 alone, 16.0 -> 18.4 tok/s for
            # Qwen3.8-27B, with the same greedy output (2026-11-11). Requests
            # that ask for log-probabilities are sampled on the host as before.
            '--backend-sampling',
            # Saving a prompt cache or a checkpoint reads the KV cache or
            # recurrent state back from every worker over the network; keep one
            # checkpoint (what hybrid models need to reuse a prompt) and no RAM
            # prompt cache, as the console app does.
            '--ctx-checkpoints', '1', '--cache-ram', '0',
            '--host', host, '--port', str(port)] + (['--api-key', api_key] if api_key else [])


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


# Headroom llama.cpp's projection must leave on every worker, for driver
# allocation granularity and what it does not count: 256 MiB, or an eighth of
# a smaller budget (a test worker of 128 MiB keeps 16).
CHECK_MARGIN_MIB = 256


def check_margin(budget_mib):
    return min(CHECK_MARGIN_MIB, max(1, budget_mib // 8))


def llama_check(binary, model_path, placement, timeout=300):
    """llama.cpp's own projection of every worker's memory for this placement.

    llama-fit-params (built next to llama-server) loads the model without
    allocating, against the live workers and with the same split and cache
    settings, and prints each device's weights, cache and compute buffers. It
    counts what the estimator leaves to the reserve (compute buffers) and checks
    the estimator's layout. Returns {worker id: MiB figures}; raises FleetError
    when a worker would not keep check_margin() of its budget free.
    """
    fit = Path(binary).resolve().parent / 'llama-fit-params'
    if not fit.is_file():
        raise FleetError(f'No {fit}; rebuild the host with scripts/build-fleet-host.sh')
    nodes = placement['nodes']
    args = [str(fit), '--model', str(Path(model_path).resolve()),
            '--rpc', ','.join(n['endpoint'] for n in nodes),
            '--device', ','.join(n['device'] for n in nodes),
            '--gpu-layers', 'all', '--split-mode', 'layer',
            '--tensor-split', ','.join(str(v) for v in placement['tensor_split']), *cache_args(placement),
            '--batch-size', str(UBATCH), '--ubatch-size', str(UBATCH), '--fit-print', 'on']
    rows = fit_rows(args, timeout)
    report, over = {}, []
    for node in nodes:
        if node['device'] not in rows:
            raise FleetError(f"llama-fit-params reported no figures for {node['device']}")
        weights, cache, compute = rows[node['device']]
        total = weights + cache + compute
        report[node['id']] = dict(weights_mib=weights, cache_mib=cache, compute_mib=compute, total_mib=total,
                                  budget_mib=node['memory_mib'])
        margin = check_margin(node['memory_mib'])
        if total + margin > node['memory_mib']:
            over.append(f"{node['id']} needs {total} MiB (weights {weights}, cache {cache}, compute {compute}) "
                        f"and {margin} MiB to spare, of {node['memory_mib']} MiB")
    if over:
        raise FleetError('llama.cpp projects more memory than the workers have: ' + '; '.join(over))
    return report


def fit_rows(args, timeout):
    """Run llama-fit-params and read its table: {device: [model, context, compute] MiB}."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=child_environment())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FleetError(f'llama-fit-params did not finish ({type(exc).__name__})') from exc
    if result.returncode:
        raise FleetError(f'llama-fit-params exited {result.returncode}:\n' + (result.stderr or result.stdout)[-1500:])
    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and re.fullmatch(r'[A-Za-z]{2,16}[0-9]{0,3}', parts[0]) and all(p.isdigit() for p in parts[1:]):
            rows[parts[0]] = [int(p) for p in parts[1:]]
    return rows


# A model on one of this computer's GPUs, with no RPC: llama.cpp's own batch
# sizes, which a local GPU handles (the fleet's small ubatch is for consoles).
DIRECT_BATCH, DIRECT_UBATCH = 2048, 512


def direct_args(model_path, device, context, slots=1, unified=False, cache_type='f16', flash_attn=False):
    placement = dict(context_tokens=context, parallel_sequences=slots, kv_unified=unified,
                     cache_dtype=cache_type, flash_attn=flash_attn)
    return ['--model', str(Path(model_path).resolve()), '--device', device, '--gpu-layers', 'all',
            *cache_args(placement), '--batch-size', str(DIRECT_BATCH), '--ubatch-size', str(DIRECT_UBATCH)]


def direct_command(binary, model_path, device, context, port, slots=1, unified=False, cache_type='f16',
                   flash_attn=False):
    """llama-server on one local GPU, on loopback (the control server's router is the way in)."""
    # llama-server's prompt cache in RAM, 8 GiB by default, shares the
    # controller with the splits' host processes: a smaller one.
    return [str(Path(binary).resolve()), *direct_args(model_path, device, context, slots, unified, cache_type, flash_attn),
            '--fit', 'off', '--cache-ram', '2048', '--host', '127.0.0.1', '--port', str(port)]


def direct_check(binary, model_path, device, free_mib, context, slots=1, unified=False, cache_type='f16',
                 flash_attn=False, timeout=300):
    """llama.cpp's projection for a model on one local GPU; raises when it would
    not leave check_margin() of the free memory."""
    if CACHE_TYPES.get(cache_type, (1,))[0] > 1 and not flash_attn:
        raise FleetError('A quantized cache needs flash attention: llama.cpp quantizes V only with it')
    fit = Path(binary).resolve().parent / 'llama-fit-params'
    if not fit.is_file():
        raise FleetError(f'No {fit}; rebuild the host with scripts/build-fleet-host.sh')
    rows = fit_rows([str(fit), *direct_args(model_path, device, context, slots, unified, cache_type, flash_attn),
                     '--fit-print', 'on'], timeout)
    if device not in rows:
        raise FleetError(f'llama-fit-params reported no figures for {device}')
    weights, cache, compute = rows[device]
    total = weights + cache + compute
    report = dict(weights_mib=weights, cache_mib=cache, compute_mib=compute, total_mib=total, budget_mib=free_mib)
    margin = check_margin(free_mib)
    if total + margin > free_mib:
        raise FleetError(f'{device} needs {total} MiB (weights {weights}, cache {cache}, compute {compute}) '
                         f'and {margin} MiB to spare, with {free_mib} MiB free')
    return report


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
    parser.add_argument('--cache-type', default='f16', choices=sorted(CACHE_TYPES),
                        help='KV cache type (llama-server -ctk/-ctv); quantized types need --flash-attn on')
    parser.add_argument('--flash-attn', default='off', choices=('on', 'off'),
                        help='flash attention; on keeps each layer\'s V at its own width and the compute buffers small')
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
        placement = plan(model, config, args.context, cache_type=args.cache_type, flash_attn=args.flash_attn == 'on')
        placement.update(model_path=str(Path(path).resolve()), model_sha256=digest,
                         llama_revision=LLAMA_REVISION,
                         memory_source='min(operator_budget,live_RPC_snapshot)' if args.action == 'serve' else 'operator_budgets_only')
        if args.action == 'serve':
            placement['llama_check'] = llama_check(args.binary, path, placement)
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
