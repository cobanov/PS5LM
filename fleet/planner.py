"""Contiguous layer placement for the pinned llama.cpp RPC layer split.

This is a capacity estimate, not a reservation or a throughput prediction. The
host retains input/shared tensors; the last remote gets the output projection.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import ipaddress
import math
import re

MIB = 1024**2
PRIVATE = tuple(ipaddress.ip_network(c) for c in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '127.0.0.0/8'))


class FleetError(ValueError):
    pass


def integer(value, label, low=1, high=2**53):
    if type(value) is not int or not low <= value <= high:
        raise FleetError(f'{label} must be an integer in {low}..{high}')
    return value


def endpoint(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9.]+:[0-9]{1,5}', value):
        raise FleetError('Node endpoint must be numeric private IPv4:port (no DNS, URL or IPv6)')
    host, port = value.split(':')
    try:
        ip = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError as exc:
        raise FleetError('Invalid IPv4 node endpoint') from exc
    if not any(ip in net for net in PRIVATE) or ip == ipaddress.IPv4Address('127.0.0.0'):
        raise FleetError('RPC is unauthenticated: only explicit RFC1918 or loopback addresses are allowed')
    integer(int(port), 'RPC port', 1, 65535)
    return f'{ip}:{int(port)}'


def inventory(raw):
    if not isinstance(raw, dict):
        raise FleetError('Inventory must be a JSON object')
    integer(raw.get('host_memory_mib'), 'host_memory_mib')
    integer(raw.get('host_reserve_mib'), 'host_reserve_mib', 1)
    nodes = raw.get('nodes')
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= 15:
        raise FleetError('A fleet needs 2..15 selected nodes; every node receives at least one layer')
    result, ids, endpoints = [], set(), set()
    for node in nodes:
        if not isinstance(node, dict):
            raise FleetError('Every node must be an object')
        name = node.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', name) or name in ids:
            raise FleetError('Node IDs must be unique simple names, at most 64 characters')
        address = endpoint(node.get('endpoint'))
        if address in endpoints:
            raise FleetError('Duplicate RPC endpoint')
        memory = integer(node.get('memory_mib'), f'{name}.memory_mib')
        reserve = integer(node.get('reserve_mib'), f'{name}.reserve_mib', 1)
        if reserve >= memory:
            raise FleetError(f'{name}: reserve must be below memory budget')
        result.append(dict(id=name, endpoint=address, memory_mib=memory, reserve_mib=reserve))
        ids.add(name)
        endpoints.add(address)
    return dict(host_memory_mib=raw['host_memory_mib'], host_reserve_mib=raw['host_reserve_mib'], nodes=result)


def plan(model, config, context=4096):
    config = inventory(config)
    context = integer(context, 'context', 32, 1048576)
    if model.get('supported') is not True:
        raise FleetError('Unsupported model: ' + '; '.join(model.get('unsupported_reasons', [])))
    if model.get('architecture') not in ('llama', 'qwen2'):
        raise FleetError('Only dense llama and qwen2 are admitted by the fleet estimator')
    limit = integer(model.get('context_length'), 'model context_length')
    if context > limit:
        raise FleetError(f'Requested context {context} exceeds model training context {limit}')
    count = integer(model.get('layer_count'), 'layer_count', 2, 512)
    nodes = config['nodes']
    if len(nodes) > count:
        raise FleetError('More nodes than repeating layers')
    weights = model.get('layer_bytes')
    kv = model.get('kv_bytes_per_layer_per_token')
    if not isinstance(weights, list) or not isinstance(kv, list) or len(weights) != count or len(kv) != count:
        raise FleetError('Missing complete layer and KV memory accounting')
    if model.get('kv_cache_dtype') != 'fp16' or model.get('kv_element_bytes') != 2:
        raise FleetError('Fleet currently supports only FP16 KV cache')
    for value in weights + kv:
        integer(value, 'layer weight/KV bytes')
    # Round up conservatively for backend padding. One sequence only, no sliding
    # window/recurrent state, no quantized cache, no flash-attention assumptions.
    padded_context = (context + 255) // 256 * 256
    resident = [w + k * padded_context for w, k in zip(weights, kv)]
    output = integer(model.get('output_bytes'), 'output_bytes')
    input_bytes = integer(model.get('input_bytes'), 'input_bytes')
    # The entire file stays on host disk. Only the input weights are retained on
    # host for compute; reserve covers model I/O staging, tokenizer and buffers.
    host_bytes = input_bytes + config['host_reserve_mib'] * MIB
    if host_bytes > config['host_memory_mib'] * MIB:
        raise FleetError('Host input/shared tensors plus reserve exceed host memory budget')
    prefix = [0]
    for value in resident:
        prefix.append(prefix[-1] + value)
    # Exact minimax *memory utilisation* for the chosen node order, not fastest
    # inference. All weights in a block stay together. Ties are deterministic.
    previous = {0: 0.0}
    parents = []
    for i, node in enumerate(nodes):
        current, parent = {}, {}
        capacity, reserve = node['memory_mib'] * MIB, node['reserve_mib'] * MIB
        for end in range(i + 1, count - (len(nodes) - i - 1) + 1):
            for start, score in previous.items():
                if start >= end:
                    continue
                memory = prefix[end] - prefix[start] + reserve + (output if i == len(nodes) - 1 else 0)
                if memory > capacity:
                    continue
                candidate = max(score, memory / capacity)
                if candidate < current.get(end, math.inf):
                    current[end], parent[end] = candidate, start
        previous = current
        parents.append(parent)
    if count not in previous:
        raise FleetError('No contiguous layer split fits every selected node including FP16 KV, output and reserves')
    spans, end = [], count
    for i in range(len(nodes) - 1, -1, -1):
        start = parents[i][end]
        spans.append((start, end))
        end = start
    spans.reverse()
    # llama.cpp assigns slot i using upper_bound(cumulative splits, i/(L+1)).
    # Boundaries halfway between slots avoid rounding a layer onto the wrong
    # device. The output is the extra slot L, assigned to the last worker.
    cumulative = [2 * end - 1 for _, end in spans[:-1]] + [2 * (count + 1)]
    ratios = [a - b for a, b in zip(cumulative, [0] + cumulative[:-1])]
    assignments = []
    for i, (node, (start, end)) in enumerate(zip(nodes, spans)):
        weights_bytes = sum(weights[start:end]) + (output if i == len(nodes) - 1 else 0)
        kv_bytes = sum(kv[start:end]) * padded_context
        reserve = node['reserve_mib'] * MIB
        assignments.append(dict(**node, device=f'RPC{i}', layer_start=start,
                                layer_end=end, output_layer=i == len(nodes) - 1,
                                weights_bytes=weights_bytes, kv_bytes=kv_bytes,
                                reserve_bytes=reserve, estimated_bytes=weights_bytes + kv_bytes + reserve))
    return dict(schema_version=1, status='estimated_not_reserved', architecture=model['architecture'],
                layer_count=count, context_tokens=context, padded_context_tokens=padded_context,
                parallel_sequences=1, cache_dtype='f16', tensor_split=ratios, nodes=assignments,
                host_estimated_bytes=host_bytes, host_model_disk_bytes=model['file_size_bytes'],
                max_estimated_memory_utilisation=previous[count],
                warnings=[
                    'Experimental unauthenticated RPC: isolated trusted network only, no public/guest LAN exposure.',
                    'Memory estimates are not reservations; backend allocation/compute buffers can exceed the configured reserves.',
                    'Input/shared tensors and model file stay on the controller; only layer/output tensors are distributed by RPC.',
                    'Every node is required for each request. No failover, pipeline throughput or PS5 performance is claimed.',
                    'Only dense llama/qwen2 single-file GGUF and one sequence with FP16 KV are admitted; hybrids/MoE are refused.',
                    'A successful host test does not validate PS5 title linkage, RADV kernels, network behaviour or memory stability.',
                ])
