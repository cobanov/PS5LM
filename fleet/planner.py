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
# llama-server's --ubatch-size as the controller runs it. Sliding-window layers
# keep the window plus one ubatch, padded to 256 (llama-kv-cache-iswa.cpp).
UBATCH = 128
# Cache types llama-server takes for -ctk/-ctv: (values per block, bytes per block).
CACHE_TYPES = {'f32': (1, 4), 'f16': (1, 2), 'bf16': (1, 2), 'q8_0': (32, 34), 'q5_1': (32, 24),
               'q5_0': (32, 22), 'q4_1': (32, 20), 'q4_0': (32, 18), 'iq4_nl': (32, 18)}


def _pad256(tokens):
    return (tokens + 255) // 256 * 256


def cache_row_bytes(values, cache_type):
    """Bytes of one cache row (one layer, one token) of K or V in a cache type."""
    block, size = CACHE_TYPES[cache_type]
    if values % block:
        raise FleetError(f'A {cache_type} cache needs rows in multiples of {block} values; this model has {values}')
    return values // block * size


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
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 15:
        raise FleetError('A split takes 1..15 nodes; every node receives at least one layer')
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


def cache_cells(context, slots=1, unified=False, window=0):
    """Cache cells llama.cpp allocates per full-attention layer and per sliding-window
    layer for llama-server's --ctx-size, --parallel and --kv-unified.

    Without --kv-unified every slot is a stream of its own with ctx-size/slots
    cells, padded to 256 (llama-context.cpp), and window layers keep, per stream,
    min(those cells, window + ubatch) padded to 256. With it, one stream of
    ctx-size cells, and window layers keep window * slots + ubatch
    (llama-kv-cache-iswa.cpp). Returns (full cells, window cells, cells per slot).
    """
    if unified:
        per_slot = _pad256(context)
        full = per_slot
        windowed = _pad256(min(full, window * slots + UBATCH)) if window else 0
    else:
        per_slot = _pad256(context // slots)
        full = per_slot * slots
        windowed = _pad256(min(per_slot, window + UBATCH)) * slots if window else 0
    return full, windowed, per_slot


def plan(model, config, context=4096, layer_counts=None, cache_type='f16', flash_attn=False, slots=1, unified=False):
    """Place the blocks on the nodes in order; the last node also holds the output.

    By default the split minimises the largest estimated memory utilisation.
    layer_counts, one positive count per node summing to the block count, sets
    the split by hand instead; it must still fit every node's budget.
    cache_type is llama-server's -ctk/-ctv. A quantized cache needs flash
    attention, which also keeps each layer's V at its own width.
    context is llama-server's --ctx-size: with slots (--parallel) above one, the
    slots share it out, or with unified (--kv-unified) all draw on the whole.
    """
    config = inventory(config)
    if cache_type not in CACHE_TYPES:
        raise FleetError('cache_type is one of ' + ', '.join(CACHE_TYPES))
    if CACHE_TYPES[cache_type][0] > 1 and not flash_attn:
        raise FleetError('A quantized cache needs flash attention: llama.cpp quantizes V only with it')
    context = integer(context, 'context', 32, 16777216)
    slots = integer(slots, 'slots', 1, 256)
    if context // slots < 32:
        raise FleetError('Each slot needs at least 32 tokens of context')
    if model.get('supported') is not True:
        raise FleetError('Unsupported model: ' + '; '.join(model.get('unsupported_reasons', [])))
    limit = integer(model.get('context_length'), 'model context_length')
    per_request = context if unified else context // slots
    if per_request > limit:
        raise FleetError(f'Requested context {per_request} per request exceeds model training context {limit}')
    count = integer(model.get('layer_count'), 'layer_count', 2, 512)
    nodes = config['nodes']
    if len(nodes) > count:
        raise FleetError('More nodes than repeating layers')
    weights = model.get('layer_bytes')
    kv = model.get('kv_bytes_per_layer_per_token')
    # Models inspected before sliding windows and recurrent state were modeled
    # carry neither field: every layer is full attention.
    kv_window = model.get('swa_kv_bytes_per_layer_per_token') or [0] * count
    state = model.get('state_bytes_per_layer') or [0] * count
    columns = (weights, kv, kv_window, state)
    if not all(isinstance(c, list) and len(c) == count for c in columns):
        raise FleetError('Missing complete layer and KV memory accounting')
    if model.get('kv_cache_dtype') != 'fp16' or model.get('kv_element_bytes') != 2:
        raise FleetError('The model inspection has no cache accounting')
    k_values, v_values = model.get('k_elements_per_layer'), model.get('v_elements_per_layer')
    window_flags = model.get('window_layer_flags')
    if all(isinstance(c, list) and len(c) == count for c in (k_values, v_values, window_flags)):
        # Price each layer's K and V rows in the cache type. Without flash
        # attention, V rows take the model's widest V (llama-kv-cache.cpp).
        v_widest = integer(model.get('v_widest_elements', 0), 'v_widest_elements', 0)
        kv, kv_window = [0] * count, [0] * count
        for il in range(count):
            if k_values[il]:
                row = cache_row_bytes(k_values[il], cache_type) + \
                    cache_row_bytes(v_values[il] if flash_attn else v_widest, cache_type)
                (kv_window if window_flags[il] else kv)[il] = row
    elif cache_type != 'f16' or flash_attn:
        raise FleetError('This inspection only has FP16 cache sizes without flash attention; inspect the model again')
    for value in weights + kv + kv_window + state:
        integer(value, 'layer weight/cache bytes', 0)
    window = integer(model.get('sliding_window', 0), 'sliding_window', 0)
    if any(kv_window) and not window:
        raise FleetError('Sliding-window layers without a window size')
    # Cells as llama.cpp pads them (cache_cells); recurrent layers keep one
    # fixed state per slot.
    padded_context, window_tokens, per_slot = cache_cells(context, slots, unified, window)
    state = [s * slots for s in state]
    resident = [w + k * padded_context + s * window_tokens + r
                for w, k, s, r in zip(weights, kv, kv_window, state)]
    output = integer(model.get('output_bytes'), 'output_bytes')
    # RoPE factor tensors: one copy on every worker that holds layers. Tensors
    # llama.cpp creates as layer 0's sit with the first worker.
    per_worker = integer(model.get('per_worker_bytes', 0), 'per_worker_bytes', 0)
    first_worker = integer(model.get('first_worker_bytes', 0), 'first_worker_bytes', 0)
    input_bytes = integer(model.get('input_bytes'), 'input_bytes')
    # The entire file stays on host disk. Only the input weights are retained on
    # host for compute; reserve covers model I/O staging, tokenizer and buffers.
    host_bytes = input_bytes + config['host_reserve_mib'] * MIB
    if host_bytes > config['host_memory_mib'] * MIB:
        raise FleetError('Host input/shared tensors plus reserve exceed host memory budget')
    prefix = [0]
    for value in resident:
        prefix.append(prefix[-1] + value)

    def fixed(i):
        return per_worker + (first_worker if i == 0 else 0) + (output if i == len(nodes) - 1 else 0)

    def memory_of(i, start, end):
        return prefix[end] - prefix[start] + nodes[i]['reserve_mib'] * MIB + fixed(i)

    if layer_counts is not None:
        if (not isinstance(layer_counts, list) or len(layer_counts) != len(nodes)
                or any(type(c) is not int or c < 1 for c in layer_counts) or sum(layer_counts) != count):
            raise FleetError(f'A manual split gives each of the {len(nodes)} nodes at least one block, '
                             f'{count} blocks in all')
        spans, start = [], 0
        for c in layer_counts:
            spans.append((start, start + c))
            start += c
        over = [f"{node['id']} needs {memory_of(i, s, e) / MIB:.0f} of {node['memory_mib']} MiB"
                for i, (node, (s, e)) in enumerate(zip(nodes, spans))
                if memory_of(i, s, e) > node['memory_mib'] * MIB]
        if over:
            raise FleetError('The manual split does not fit: ' + '; '.join(over))
        utilisation = max(memory_of(i, s, e) / (node['memory_mib'] * MIB)
                          for i, (node, (s, e)) in enumerate(zip(nodes, spans)))
    else:
        # Exact minimax *memory utilisation* for the chosen node order, not
        # fastest inference. All weights in a block stay together. Ties are
        # deterministic.
        previous = {0: 0.0}
        parents = []
        for i, node in enumerate(nodes):
            current, parent = {}, {}
            capacity = node['memory_mib'] * MIB
            for end in range(i + 1, count - (len(nodes) - i - 1) + 1):
                for start, score in previous.items():
                    if start >= end:
                        continue
                    memory = memory_of(i, start, end)
                    if memory > capacity:
                        continue
                    candidate = max(score, memory / capacity)
                    if candidate < current.get(end, math.inf):
                        current[end], parent[end] = candidate, start
            previous = current
            parents.append(parent)
        if count not in previous:
            raise FleetError(f'No contiguous layer split fits every node with a {context}-token {cache_type} cache, '
                             'the output and the reserves')
        spans, end = [], count
        for i in range(len(nodes) - 1, -1, -1):
            start = parents[i][end]
            spans.append((start, end))
            end = start
        spans.reverse()
        utilisation = previous[count]
    # llama.cpp assigns slot i using upper_bound(cumulative splits, i/(L+1)).
    # Boundaries halfway between slots avoid rounding a layer onto the wrong
    # device. The output is the extra slot L, assigned to the last worker.
    cumulative = [2 * end - 1 for _, end in spans[:-1]] + [2 * (count + 1)]
    ratios = [a - b for a, b in zip(cumulative, [0] + cumulative[:-1])]
    assignments = []
    for i, (node, (start, end)) in enumerate(zip(nodes, spans)):
        weights_bytes = sum(weights[start:end]) + fixed(i)
        kv_bytes = sum(kv[start:end]) * padded_context + sum(kv_window[start:end]) * window_tokens
        state_bytes = sum(state[start:end])
        reserve = node['reserve_mib'] * MIB
        assignments.append(dict(**node, device=f'RPC{i}', layer_start=start,
                                layer_end=end, output_layer=i == len(nodes) - 1,
                                weights_bytes=weights_bytes, kv_bytes=kv_bytes, state_bytes=state_bytes,
                                reserve_bytes=reserve,
                                estimated_bytes=weights_bytes + kv_bytes + state_bytes + reserve))
    return dict(schema_version=1, status='estimated_not_reserved', architecture=model['architecture'],
                layer_count=count, context_tokens=context, padded_context_tokens=padded_context,
                window_tokens=window_tokens, parallel_sequences=slots, kv_unified=bool(unified),
                context_per_slot=per_slot, cache_dtype=cache_type,
                flash_attn=bool(flash_attn), tensor_split=ratios,
                nodes=assignments, host_estimated_bytes=host_bytes, host_model_disk_bytes=model['file_size_bytes'],
                max_estimated_memory_utilisation=utilisation, split_mode='manual' if layer_counts else 'auto',
                notes=list(model.get('notes', [])),
                warnings=[
                    'Experimental unauthenticated RPC: isolated trusted network only, no public/guest LAN exposure.',
                    'Memory estimates are not reservations; backend allocation/compute buffers can exceed the configured reserves.',
                    'Input/shared tensors and model file stay on the controller; only layer/output tensors are distributed by RPC.',
                    'Every node is required for each request. No failover, pipeline throughput or PS5 performance is claimed.',
                    'Single-file GGUF only; RWKV, linear/latent-attention and parallel-SSM caches are refused.',
                    'A successful host test does not validate PS5 title linkage, RADV kernels, network behaviour or memory stability.',
                ])
