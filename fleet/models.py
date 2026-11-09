"""Bounded, metadata-only GGUF v3 inspection and public HTTPS acquisition.

No model code, pickle, tensor execution, or third-party packages are used. Metadata
identity is deliberately NOT a checksum of model weights. Downloaded files have a
separate complete-file SHA256. Size layouts follow ggml's GGUF specification and
src/ggml-common.h; unsupported encodings/architectures fail closed.
"""
# Metadata inspection and bounded acquisition; no model or simulation execution.
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import math
import os
from pathlib import Path
import re
import queue
import shutil
import socket
import ssl
import stat
import struct
import tempfile
import threading
import time
from urllib.parse import parse_qsl, quote, urljoin, urlsplit


class ModelError(ValueError):
    """Invalid model file or unsupported structural encoding."""


class DownloadError(ModelError):
    """Model download failed or violated public-HTTPS/resource policy."""


MAX_HEADER_BYTES = 64 * 1024**2
MAX_STRING_BYTES = 1024**2
MAX_METADATA = 16_384
MAX_ARRAY_ELEMENTS = 1_000_000
MAX_TENSORS = 200_000
MAX_LAYERS = 4096
DEFAULT_MAX_BYTES = 16 * 1024**3
DEFAULT_MAX_CACHE_BYTES = 64 * 1024**3
DISK_RESERVE_BYTES = 16 * 1024**2
# GGML type id -> (elements per block, bytes per block): every type the pinned
# ggml defines (ggml/include/ggml.h), with block sizes compiled from its
# ggml-common.h. Unknown types must not be estimated from a filename or
# bits-per-weight label.
GGML_TYPES = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 36),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    34: (256, 54),   # TQ1_0
    35: (256, 66),   # TQ2_0
    39: (32, 17),    # MXFP4
    40: (64, 36),    # NVFP4
    41: (128, 18),   # Q1_0
    42: (64, 18),    # Q2_0
}
_SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
            6: 'f', 7: 'B', 10: 'Q', 11: 'q', 12: 'd'}
_BLOCK = re.compile(r'^blk\.(0|[1-9][0-9]*)\.(.+)$')

# Tensors outside the blocks, by where the pinned llama.cpp keeps them
# (LLM_TENSOR_INFOS in src/llama-arch.cpp). Input tensors stay on the
# controller. RoPE factor tensors are created for every layer and duplicated,
# so each worker that holds layers keeps one copy. Everything else that
# llama.cpp knows outside the blocks (output head, final norms, projections)
# lives on the output device, the last worker.
INPUT_TENSORS = frozenset({'token_embd', 'per_layer_token_embd', 'position_embd', 'token_types',
                           'masked_embd_centroids', 'masked_embd_ordering'})
PER_WORKER_TENSORS = frozenset({'rope_freqs', 'rope_factors_long', 'rope_factors_short'})
# Created as layer 0's (tn(..., "weight", 0) in src/models/gemma3n.cpp and
# gemma4.cpp), so they live with the first worker.
FIRST_WORKER_TENSORS = frozenset({'per_layer_model_proj', 'per_layer_proj_norm'})
OUTPUT_TENSORS = frozenset({'output', 'output_norm', 'token_embd_norm', 'cls', 'cls.norm', 'cls.output',
                            'dense_2', 'dense_3', 'altup_proj', 'altup_unembd_proj',
                            'conf_proj', 'd2t', 'fc', 'markov_w1', 'markov_w2',
                            'nextn.pre_projection', 'nextn.post_projection', 'output_res_score', 'hrm.z_l_init'})
# llama.cpp's own sliding-window pattern for architectures whose files carry a
# window but no per-layer list (load_swa_pattern() in src/models/*.cpp):
# (period, dense_first). A layer keeps only the window unless it is the last of
# its period, or the first when dense_first is set; NextN blocks never do.
DEFAULT_SWA_PATTERNS = {'gpt-oss': (2, False), 'gemma2': (2, False), 'gemma3': (6, False), 'gemma3n': (5, False),
                        'gemma-embedding': (6, False), 'cohere2': (4, False), 'cohere2moe': (4, True),
                        'exaone4': (4, False), 'exaone-moe': (4, False), 'olmo2': (4, False),
                        'muse-glimmer': (4, False), 'afmoe': (4, False), 'mellum': (4, False),
                        'plamo3': (8, False), 'smallthinker': (4, True), 'laguna': (4, True)}
OUTPUT_PREFIXES = ('decision.', 'output_hc_', 'selector_')
# Layouts that are not one stack of decoder blocks (encoder-decoder, audio
# codecs): a layer split cannot be reasoned about from the block tensors.
UNSPLITTABLE_PREFIXES = ('enc.', 'dec.', 'posnet.', 'convnext.', 'lilicorr.', 'conv1d')
# Recurrent layers keep a fixed f32 state instead of a KV cache. Which layers
# are recurrent, per the pinned llama.cpp:
#   - qwen3next/qwen35/qwen35moe: attention at a fixed interval (4 unless the
#     file says otherwise), Gated DeltaNet in between (src/models/qwen35.cpp);
#   - mamba/mamba2: every layer;
#   - the hybrids below: layers whose attention.head_count_kv is 0.
# Other recurrent and hybrid layouts (RWKV, KDA, linear attention, MLA caches,
# parallel attention+SSM) are not modeled, so those models fail closed.
INTERVAL_HYBRIDS = frozenset({'qwen3next', 'qwen35', 'qwen35moe'})
ALL_RECURRENT = frozenset({'mamba', 'mamba2'})
KV_ZERO_HYBRIDS = frozenset({'granitehybrid', 'nemotron_h', 'nemotron_h_moe', 'jamba', 'lfm2', 'lfm2moe'})
UNMODELED_ARCHITECTURES = frozenset({'rwkv6', 'rwkv6qwen2', 'rwkv7', 'arwkv7', 'kimi-linear', 'kimi-k3',
                                     'minimax-01', 'falcon-h1', 'plamo2', 'qwen4exp', 'deepseek4',
                                     'glm5-next', 'bailingmoe3'})
_UNMODELED_KEYS = ('.wkv.', '.kda.', '.ple.', '.attention.kv_lora_rank')


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


class _Reader:
    def __init__(self, stream, file_size):
        self.stream = stream
        self.file_size = file_size
        self.pos = 0
        self.identity = hashlib.sha256()
        self.endian = '<'

    def read(self, size):
        if size < 0 or self.pos + size > min(self.file_size, MAX_HEADER_BYTES):
            raise ModelError('Truncated GGUF header or header exceeds 64 MiB inspection limit')
        data = self.stream.read(size)
        if len(data) != size:
            raise ModelError('Truncated GGUF header')
        self.pos += size
        self.identity.update(data)
        return data

    def unpack(self, fmt):
        return struct.unpack(self.endian + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, *, limit=MAX_STRING_BYTES):
        size = self.unpack('Q')
        if size > limit:
            raise ModelError(f'GGUF string exceeds {limit} byte limit')
        try:
            return self.read(size).decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ModelError('Invalid UTF-8 in GGUF header') from exc

    def value(self, kind, *, retain=True):
        if kind in _SCALARS:
            result = self.unpack(_SCALARS[kind])
            if kind == 7:
                if result not in (0, 1):
                    raise ModelError('Invalid GGUF boolean')
                return bool(result)
            if isinstance(result, float) and not math.isfinite(result):
                raise ModelError('Non-finite GGUF metadata')
            return result
        if kind == 8:
            return self.string()
        if kind != 9:
            raise ModelError(f'Unknown GGUF metadata value type {kind}')
        item_type = self.unpack('I')
        count = self.unpack('Q')
        if count > MAX_ARRAY_ELEMENTS:
            raise ModelError('GGUF metadata array exceeds element limit')
        if item_type == 9 or item_type not in {*_SCALARS, 8}:
            raise ModelError('Nested/unknown GGUF metadata arrays are unsupported')
        # Tokenizer vocabularies can be large; validate and hash their contents
        # without retaining a million Python objects. Relevant numeric arrays
        # are bounded by MAX_LAYERS and are needed for per-layer GQA.
        keep = retain and count <= MAX_LAYERS
        values = [] if keep else None
        for _ in range(count):
            item = self.value(item_type, retain=False)
            if keep:
                values.append(item)
        return values if keep else {'array_length': count, 'array_type': item_type}


def _positive_int(value, name, maximum=2**31 - 1):
    if type(value) is not int or value <= 0 or value > maximum:
        raise ModelError(f'{name} must be a positive bounded integer')
    return value


def _per_layer(metadata, key, layers, default=None):
    """A metadata value given once or once per block, as one entry per block.

    Entries are non-negative integers (booleans count as 0/1), or None when the
    key is absent and there is no default.
    """
    value = metadata.get(key, default)
    if isinstance(value, dict):
        raise ModelError(f'{key} is an array too long to hold one entry per layer')
    values = value if isinstance(value, list) else [value] * layers
    if len(values) != layers:
        raise ModelError(f'{key} must have exactly {layers} layer entries')
    result = []
    for item in values:
        if isinstance(item, bool):
            item = int(item)
        if item is not None and (type(item) is not int or not 0 <= item <= 2**31 - 1):
            raise ModelError(f'{key} must hold non-negative bounded integers')
        result.append(item)
    return result


def _count(metadata, key, default=0, maximum=2**31 - 1):
    value = metadata.get(key, default)
    if type(value) is not int or not 0 <= value <= maximum:
        raise ModelError(f'{key} must be a non-negative bounded integer')
    return value


def _layer_cache(metadata, arch, layers, main_layers, hidden, unsupported):
    """Per-layer cache memory the pinned llama.cpp allocates for one sequence.

    Returns full-context KV bytes per token, sliding-window KV bytes per token
    (those layers keep only the window), and fixed recurrent-state bytes, for
    every block, with an FP16 KV cache. Returns None after recording why the
    layout is not modeled. Attention layers listed in llama.cpp's has_kv() but
    sharing another layer's cache (Gemma's shared_kv_layers) keep none.
    """
    prefix = arch + '.'
    if arch in UNMODELED_ARCHITECTURES or any(
            marker in key for key in metadata if key.startswith(prefix) for marker in _UNMODELED_KEYS):
        unsupported.append(f'The {arch!r} cache layout (RWKV, KDA, linear or latent attention) is not modeled')
        return None
    heads = _per_layer(metadata, prefix + 'attention.head_count', layers, 0)
    kv_heads = [h if k is None else k for h, k in
                zip(heads, _per_layer(metadata, prefix + 'attention.head_count_kv', layers))]
    key_len = _per_layer(metadata, prefix + 'attention.key_length', layers)
    value_len = _per_layer(metadata, prefix + 'attention.value_length', layers)
    key_swa = _per_layer(metadata, prefix + 'attention.key_length_swa', layers)
    value_swa = _per_layer(metadata, prefix + 'attention.value_length_swa', layers)
    ssm = {name: _count(metadata, f'{prefix}ssm.{name}') for name in
           ('conv_kernel', 'inner_size', 'state_size', 'group_count')}
    shortconv = _count(metadata, prefix + 'shortconv.l_cache')
    notes = []

    if arch in INTERVAL_HYBRIDS:
        if prefix + 'attention.recurrent_layers' in metadata:
            recurrent = [bool(v) for v in _per_layer(metadata, prefix + 'attention.recurrent_layers', layers)]
        else:
            interval = _count(metadata, prefix + 'full_attention_interval', 4)
            if not interval:
                raise ModelError('full_attention_interval must be positive')
            # NextN blocks are full-attention layers (src/models/qwen35.cpp).
            recurrent = [il < main_layers and (il + 1) % interval != 0 for il in range(layers)]
    elif arch in ALL_RECURRENT:
        recurrent = [il < main_layers for il in range(layers)]
    elif arch in KV_ZERO_HYBRIDS:
        recurrent = [il < main_layers and kv_heads[il] == 0 for il in range(layers)]
    elif ssm['inner_size'] or shortconv or prefix + 'attention.recurrent_layers' in metadata:
        unsupported.append(f'The {arch!r} recurrent layer layout is not modeled')
        return None
    else:
        recurrent = [False] * layers

    state_bytes = 0
    if any(recurrent):
        # llama_hparams::n_embd_r() and n_embd_s(), stored as f32, one row each.
        if shortconv:
            conv_state = hidden * (shortconv - 1)
        else:
            conv_state = (ssm['conv_kernel'] - 1) * (ssm['inner_size'] + 2 * ssm['group_count'] * ssm['state_size']) \
                if ssm['conv_kernel'] else 0
        state_bytes = (conv_state + ssm['state_size'] * ssm['inner_size']) * 4
        if not state_bytes:
            unsupported.append('Recurrent layers without ssm/shortconv state sizes in the metadata')
            return None

    window = _count(metadata, prefix + 'attention.sliding_window')
    swa = [False] * layers
    if window:
        pattern = metadata.get(prefix + 'attention.sliding_window_pattern')
        if isinstance(pattern, list):
            swa = [bool(v) for v in _per_layer(metadata, prefix + 'attention.sliding_window_pattern', layers)]
        elif arch in DEFAULT_SWA_PATTERNS:
            period, dense_first = DEFAULT_SWA_PATTERNS[arch]
            if type(pattern) is int:   # a period in the file replaces the default one
                period = pattern
            swa = [il < main_layers and (period == 0 or (il % period != 0 if dense_first else il % period < period - 1))
                   for il in range(layers)]
        else:
            # Count every attention layer at full context, which over-reserves.
            notes.append('sliding-window layers are counted at full context: the file does not list them')
    shared = _count(metadata, prefix + 'attention.shared_kv_layers', 0, layers - 1)
    kv_from = layers - shared if shared else layers

    def head_sizes(il):
        implicit = hidden // heads[il] if heads[il] and not hidden % heads[il] else None
        if swa[il]:
            return key_swa[il] or key_len[il] or implicit, value_swa[il] or value_len[il] or implicit
        return key_len[il] or implicit, value_len[il] or implicit

    # The fleet runs with flash attention off, so the V cache is transposed and
    # every layer's V rows take the model's widest V (n_embd_v_gqa_max(), over
    # all blocks; llama-kv-cache.cpp).
    v_widest = max((kv_heads[il] * (head_sizes(il)[1] or 0) for il in range(layers)), default=0)
    kv_full, kv_window, state = [0] * layers, [0] * layers, [0] * layers
    # Element counts per token, for any cache type: K and V per layer, and
    # whether the layer keeps only the window. FP16 bytes above are kept for
    # older callers; the planner prices these for the cache type it runs.
    k_elements, v_elements, window_layer = [0] * layers, [0] * layers, [False] * layers
    for il in range(layers):
        if recurrent[il]:
            state[il] = state_bytes
            continue
        # NextN blocks are outside the main context's cache (il < n_layer()).
        if il >= main_layers or il >= kv_from or not kv_heads[il]:
            continue
        if heads[il] and heads[il] % kv_heads[il]:
            raise ModelError('attention.head_count must be divisible by head_count_kv')
        key, value = head_sizes(il)
        if not key or not value:
            raise ModelError(f'Layer {il} has no attention head size (key_length/value_length, '
                             'or embedding_length divisible by head_count)')
        (kv_window if swa[il] else kv_full)[il] = (kv_heads[il] * key + v_widest) * 2
        k_elements[il], v_elements[il], window_layer[il] = kv_heads[il] * key, kv_heads[il] * value, swa[il]
    return dict(kv_full=kv_full, kv_window=kv_window, state=state,
                window=window if any(kv_window) else 0, recurrent=sum(recurrent), notes=notes,
                k_elements=k_elements, v_elements=v_elements, window_layer=window_layer, v_widest=v_widest)


def inspect_file(path) -> dict:
    """Inspect a regular GGUF v3 file without reading tensor payloads.

    Invalid structure raises ModelError. Unsupported, well-formed models return
    supported=False and explanatory unsupported_reasons; never plan those models.
    layer_bytes/input_bytes/output_bytes/per_worker_bytes include actual tensor
    storage spans and padding, for any architecture: memory follows where the
    pinned llama.cpp places each tensor, not its name. Tied embeddings are
    additionally charged to the output stage; NextN blocks' weights (mtp_bytes)
    are not resident. Per block, the cache is full-context KV bytes per token,
    sliding-window KV bytes per token, or a fixed recurrent state. The file
    header is reported separately, not presented as resident tensor memory.
    """
    path = Path(path)
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0)), 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ModelError('GGUF input must be a regular file')
            size = info.st_size
            reader = _Reader(stream, size)
            if reader.read(4) != b'GGUF':
                raise ModelError('Not a GGUF file (expected GGUF magic)')
            version_bytes = reader.read(4)
            if version_bytes == struct.pack('>I', 3):
                raise ModelError('Big-endian GGUF is not supported; use little-endian GGUF v3')
            if struct.unpack('<I', version_bytes)[0] != 3:
                raise ModelError('Only GGUF version 3 is supported')
            tensor_count = reader.unpack('Q')
            metadata_count = reader.unpack('Q')
            if not 0 < tensor_count <= MAX_TENSORS:
                raise ModelError('GGUF tensor count is zero or exceeds inspection limit')
            if metadata_count > MAX_METADATA:
                raise ModelError('GGUF metadata count exceeds inspection limit')
            metadata = {}
            for _ in range(metadata_count):
                key = reader.string(limit=1024)
                if not key or key in metadata:
                    raise ModelError('Empty or duplicate GGUF metadata key')
                kind = reader.unpack('I')
                # Keep per-layer arrays (head counts, window and recurrent
                # flags); vocabularies are validated but not retained.
                metadata[key] = reader.value(kind, retain=not key.startswith('tokenizer.'))
            alignment = metadata.get('general.alignment', 32)
            if type(alignment) is not int or alignment < 1 or alignment > 1024**2 or alignment & (alignment - 1):
                raise ModelError('GGUF alignment must be a bounded power of two')
            tensors = []
            names = set()
            unsupported = []
            unknown_types = set()
            for _ in range(tensor_count):
                name = reader.string(limit=1024)
                if not name or name in names:
                    raise ModelError('Empty or duplicate GGUF tensor name')
                names.add(name)
                dimensions = reader.unpack('I')
                if not 1 <= dimensions <= 4:
                    raise ModelError('GGUF tensors must have one to four dimensions')
                shape = [_positive_int(reader.unpack('Q'), 'tensor dimension') for _ in range(dimensions)]
                elements = math.prod(shape)
                if elements > 2**63 - 1:
                    raise ModelError('GGUF tensor element count exceeds signed 64-bit bound')
                dtype = reader.unpack('I')
                offset = reader.unpack('Q')
                if offset % alignment:
                    raise ModelError(f'Tensor {name} has an unaligned offset')
                nbytes = None
                if dtype in GGML_TYPES:
                    block, block_size = GGML_TYPES[dtype]
                    if shape[0] % block:
                        raise ModelError(f'Tensor {name} row is not a whole quantization block')
                    nbytes = elements // block * block_size
                else:
                    unknown_types.add(dtype)
                tensors.append({'name': name, 'shape': shape, 'type': dtype,
                                'offset': offset, 'nbytes': nbytes})
            data_offset = _align(reader.pos, alignment)
            if data_offset >= size:
                raise ModelError('GGUF file has no tensor data')
            header_hash = reader.identity.hexdigest()
    except OSError as exc:
        raise ModelError(f'Could not read GGUF file: {exc.strerror or type(exc).__name__}') from exc

    tensors.sort(key=lambda t: t['offset'])
    for index, tensor in enumerate(tensors):
        start = tensor['offset']
        end = tensors[index + 1]['offset'] if index + 1 < len(tensors) else size - data_offset
        if start >= end or data_offset + end > size:
            raise ModelError('GGUF tensor offsets overlap or lie outside the file')
        if tensor['nbytes'] is not None and tensor['nbytes'] > end - start:
            raise ModelError(f'Tensor {tensor["name"]} is truncated or overlaps its successor')
        # Assign every payload byte, including file padding/gaps, exactly once.
        tensor['span_bytes'] = end - start
    if unknown_types:
        unsupported.append('Unsupported GGML tensor type IDs: ' + ', '.join(map(str, sorted(unknown_types))))
    architecture = metadata.get('general.architecture', '')
    if not isinstance(architecture, str) or not architecture:
        raise ModelError('Missing general.architecture string')
    if any(key.startswith('split.') for key in metadata) or re.search(r'-\d{5}-of-\d{5}\.gguf$', path.name):
        unsupported.append('Split GGUF files are unsupported; supply a complete merged single-file model')
    if metadata.get('general.type', 'model') != 'model':
        unsupported.append('Only complete model GGUF files are supported (not adapters or vocabularies)')
    layers = _positive_int(metadata.get(f'{architecture}.block_count'), 'block_count', MAX_LAYERS)
    hidden = _positive_int(metadata.get(f'{architecture}.embedding_length'), 'embedding_length')
    # Multi-token-prediction (NextN) blocks follow the main stack. llama-server
    # loads their weights only with load_mtp, off by default, so they are not
    # resident; the blocks still take a tensor-split slot and a KV cache.
    nextn = _count(metadata, f'{architecture}.nextn_predict_layers', 0, layers - 1)
    main_layers = layers - nextn
    layer_bytes = [0] * layers
    input_bytes = tensors[0]['offset']
    output_bytes = per_worker_bytes = first_worker_bytes = mtp_bytes = 0
    by_name = {t['name']: t for t in tensors}
    unplaced, unsplittable = [], []
    for tensor in tensors:
        name, span = tensor['name'], tensor['span_bytes']
        match = _BLOCK.fullmatch(name)
        if match:
            layer = int(match.group(1))
            if layer >= layers:
                raise ModelError('Tensor block index exceeds declared block_count')
            if layer >= main_layers:
                mtp_bytes += span
            else:
                layer_bytes[layer] += span
            continue
        if name.startswith('blk.'):
            raise ModelError(f'Malformed block tensor name {name}')
        base = name.rsplit('.', 1)[0] if name.endswith(('.weight', '.bias')) else name
        if base in INPUT_TENSORS:
            input_bytes += span
        elif base in PER_WORKER_TENSORS:
            per_worker_bytes += span
        elif base in FIRST_WORKER_TENSORS:
            first_worker_bytes += span
        else:
            # Charged to the output worker in every case; unknown names are
            # also refused, since llama.cpp would not know where to put them.
            output_bytes += span
            if name.startswith(UNSPLITTABLE_PREFIXES):
                unsplittable.append(name)
            elif base not in OUTPUT_TENSORS and not base.startswith(OUTPUT_PREFIXES):
                unplaced.append(name)
    if unsplittable:
        unsupported.append('Encoder-decoder and audio-codec layouts are not split by layer: ' + ', '.join(unsplittable[:4]))
    if unplaced:
        unsupported.append('Tensors outside the blocks that the pinned llama.cpp does not place: ' + ', '.join(unplaced[:8]))
    embedding = by_name.get('token_embd.weight')
    if embedding is None:
        unsupported.append('Missing token_embd.weight')
    elif len(embedding['shape']) != 2 or embedding['shape'][0] != hidden:
        raise ModelError('token_embd.weight shape disagrees with embedding_length')
    if 'output_norm.weight' in by_name and by_name['output_norm.weight']['shape'] != [hidden]:
        raise ModelError('output_norm.weight shape disagrees with embedding_length')
    if 'output.weight' in by_name and by_name['output.weight']['shape'][0] != hidden:
        raise ModelError('output.weight shape disagrees with embedding_length')
    # A model without output.weight reuses token_embd as its output head;
    # llama.cpp duplicates it onto the output device.
    tied = 'output.weight' not in by_name and embedding is not None
    replicated = embedding['span_bytes'] if tied else 0
    output_bytes += replicated
    cache = None
    if not unsupported:
        cache = _layer_cache(metadata, architecture, layers, main_layers, hidden, unsupported)
    if cache is None:
        cache = dict(kv_full=[], kv_window=[], state=[], window=0, recurrent=0, notes=[],
                     k_elements=[], v_elements=[], window_layer=[], v_widest=0)
    unsupported = list(dict.fromkeys(unsupported))
    identity = hashlib.sha256((header_hash + ':' + str(size)).encode('ascii')).hexdigest()
    has_cache = not unsupported
    return {
        'model_id': 'gguf-meta-sha256:' + identity,
        'identity_kind': 'metadata-only-sha256',
        'format': 'gguf', 'gguf_version': 3, 'architecture': architecture,
        'name': metadata.get('general.name', path.name),
        'supported': not unsupported, 'unsupported_reasons': unsupported,
        'notes': cache['notes'],
        'layer_count': layers, 'layer_bytes': layer_bytes,
        'mtp_layers': nextn, 'mtp_bytes': mtp_bytes,
        'input_bytes': input_bytes, 'output_bytes': output_bytes, 'per_worker_bytes': per_worker_bytes,
        'first_worker_bytes': first_worker_bytes,
        'kv_bytes_per_layer_per_token': cache['kv_full'],
        'swa_kv_bytes_per_layer_per_token': cache['kv_window'],
        'sliding_window': cache['window'],
        'state_bytes_per_layer': cache['state'],
        'recurrent_layer_count': cache['recurrent'],
        'kv_cache_dtype': 'fp16' if has_cache else None,
        'kv_element_bytes': 2 if has_cache else None,
        'kv_elements_per_layer_per_token': [value // 2 for value in cache['kv_full']],
        'k_elements_per_layer': cache['k_elements'], 'v_elements_per_layer': cache['v_elements'],
        'window_layer_flags': cache['window_layer'], 'v_widest_elements': cache['v_widest'],
        'expert_count': metadata.get(f'{architecture}.expert_count', 0),
        'context_length': _positive_int(metadata[f'{architecture}.context_length'], 'context_length') if f'{architecture}.context_length' in metadata else None,
        'hidden_size': hidden, 'file_size_bytes': size, 'tensor_count': len(tensors),
        'header_bytes': data_offset, 'tensor_storage_bytes': size - data_offset,
        'tied_embeddings': tied, 'replicated_output_bytes': replicated,
        'alignment_bytes': alignment,
    }


_CREDENTIAL_KEYS = {'token', 'access_token', 'auth', 'authorization', 'api_key',
                    'apikey', 'password', 'secret', 'bearer'}
_TRANSITION_RANGES = (ipaddress.ip_network('64:ff9b::/96'),
                      ipaddress.ip_network('64:ff9b:1::/48'),
                      ipaddress.ip_network('2002::/16'),
                      ipaddress.ip_network('2001::/32'))


def _public_ip(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if not address.is_global or address.is_multicast or address.is_unspecified or address.is_reserved:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped and not _public_ip(str(address.ipv4_mapped)):
            return False
        if any(address in network for network in _TRANSITION_RANGES):
            return False
    return True


def _validate_url(url):
    if not isinstance(url, str) or len(url) > 8192 or any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise DownloadError('URL must be a bounded HTTPS URL without whitespace/control characters')
    try:
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.port not in (None, 443):
            raise DownloadError('Only public HTTPS URLs on port 443 are allowed')
        if parsed.username is not None or parsed.password is not None:
            raise DownloadError('Embedded URL credentials are not allowed')
        if parsed.fragment or '\\' in url or '%' in parsed.hostname:
            raise DownloadError('URL fragments, backslashes, and encoded hostnames are not allowed')
        if any(key.lower() in _CREDENTIAL_KEYS for key, _ in parse_qsl(parsed.query, max_num_fields=128)):
            raise DownloadError('Bearer tokens and authentication secrets in model URLs are not allowed')
        hostname = parsed.hostname.encode('idna').decode('ascii').rstrip('.')
        if not hostname or len(hostname) > 253:
            raise DownloadError('Invalid URL hostname')
        target = quote(parsed.path or '/', safe="/%:@!$&'()*+,;=-._~")
        if parsed.query:
            target += '?' + quote(parsed.query, safe="%/?@!$&'()*+,;=:-._~")
        return hostname, target
    except (ValueError, UnicodeError) as exc:
        raise DownloadError('Invalid public HTTPS URL') from exc


def _resolve_public(host, timeout=30):
    # getaddrinfo has no native timeout. A daemon resolver bounds the caller's
    # wait; at most four lookups occur for a download. No address is used before
    # this result has passed every public-address check.
    result = queue.Queue(maxsize=1)
    def resolve():
        try:
            result.put(socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM))
        except Exception as exc:
            result.put(exc)
    threading.Thread(target=resolve, daemon=True).start()
    try:
        addresses = result.get(timeout=timeout)
    except queue.Empty as exc:
        raise DownloadError('Model DNS resolution exceeded time limit') from exc
    if isinstance(addresses, Exception):
        raise DownloadError('Model host could not be resolved') from addresses
    if not addresses or len(addresses) > 64:
        raise DownloadError('Host has no usable addresses or too many DNS results')
    pins = []
    for family, socktype, protocol, _canonical, sockaddr in addresses:
        if family not in (socket.AF_INET, socket.AF_INET6) or not _public_ip(sockaddr[0]):
            raise DownloadError('Model host resolves to a non-public address; download blocked')
        if family == socket.AF_INET6 and (sockaddr[2] or sockaddr[3]):
            raise DownloadError('Scoped IPv6 addresses are not allowed')
        pin = (family, socktype, protocol, sockaddr)
        if pin not in pins:
            pins.append(pin)
    return pins


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """TCP connects to checked numeric IP; TLS SNI/cert checks use original host.

    No second DNS lookup, proxy environment, cookies, auth headers, or connection
    reuse across redirects. Checking getpeername adds a post-connect invariant.
    """
    def __init__(self, host, pin, timeout):
        super().__init__(host, port=443, timeout=timeout, context=ssl.create_default_context())
        self.pin = pin
        self.transport = None

    def abort(self):
        transport = self.transport
        if transport is not None:
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            transport.close()

    def connect(self):
        family, socktype, protocol, sockaddr = self.pin
        raw = socket.socket(family, socktype, protocol)
        self.transport = raw
        try:
            raw.settimeout(self.timeout)
            raw.connect(sockaddr)
            if ipaddress.ip_address(raw.getpeername()[0]) != ipaddress.ip_address(sockaddr[0]):
                raise DownloadError('Connected model server does not match the validated IP')
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host, do_handshake_on_connect=False)
            self.transport = self.sock
            self.sock.do_handshake()
        except BaseException:
            raw.close()
            raise


def _cache_usage(directory):
    total = 0
    for index, entry in enumerate(directory.iterdir()):
        if index >= 10_000:
            raise DownloadError('Model cache has too many files')
        info = entry.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise DownloadError('Model cache must contain only regular files')
        total += info.st_size
    return total


def fetch_model(url, cache_dir, max_bytes=DEFAULT_MAX_BYTES, *, timeout=30,
                max_seconds=600, max_cache_bytes=DEFAULT_MAX_CACHE_BYTES):
    """Download bounded public HTTPS bytes into a content-addressed cache.

    Returns (Path, full_file_sha256). At most three redirects, revalidated and
    numerically pinned at every hop. Does not load/extract/execute model code.
    Call inspect_file on the result before admitting it for model planning.
    """
    if type(max_bytes) is not int or max_bytes < 1 or max_bytes > 2**63 - 1:
        raise DownloadError('max_bytes must be a positive signed 64-bit integer')
    if type(max_cache_bytes) is not int or max_cache_bytes < 1:
        raise DownloadError('max_cache_bytes must be a positive integer')
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 120 or not isinstance(max_seconds, (int, float)) or not 0 < max_seconds <= 3600:
        raise DownloadError('timeout must be 0..120 seconds and max_seconds 0..3600 seconds')
    _validate_url(url)
    directory = Path(cache_dir)
    partial = None
    connection = None
    timer = None
    deadline = time.monotonic() + max_seconds
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise DownloadError('Model cache must be a real directory')
        used = _cache_usage(directory)
        current = url
        for redirects in range(4):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DownloadError('Model download exceeded total time limit')
            host, target = _validate_url(current)
            pins = _resolve_public(host, min(timeout, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DownloadError('Model download exceeded total time limit')
            # Deterministic first validated endpoint; no unbounded connection retries.
            connection = _PinnedHTTPSConnection(host, pins[0], min(timeout, remaining))
            # Interrupt a slow-drip header/body or TLS peer at the absolute
            # deadline, rather than merely applying an idle socket timeout.
            timer = threading.Timer(remaining, connection.abort)
            timer.daemon = True
            timer.start()
            connection.request('GET', target, headers={'Accept': 'application/octet-stream',
                               'Accept-Encoding': 'identity', 'User-Agent': 'ps5fleet/0.1',
                               'Connection': 'close'})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                timer.cancel()
                connection.close()
                connection = None
                if not location or redirects == 3:
                    raise DownloadError('Model download redirect limit exceeded or missing Location')
                current = urljoin(current, location)
                continue
            if response.status != 200:
                raise DownloadError(f'Model server returned HTTP {response.status}')
            if response.getheader('Content-Encoding', 'identity').lower() not in ('identity', ''):
                raise DownloadError('Compressed HTTP transfer content is unsupported')
            lengths = response.headers.get_all('Content-Length', [])
            if len(lengths) > 1:
                raise DownloadError('Ambiguous Content-Length headers')
            advertised = None
            if lengths:
                if not re.fullmatch(r'[0-9]+', lengths[0]) or len(lengths[0]) > 20:
                    raise DownloadError('Invalid Content-Length header')
                advertised = int(lengths[0])
                if advertised > max_bytes:
                    raise DownloadError('Model exceeds download byte limit')
            transfer = response.getheader('Transfer-Encoding', '').lower()
            if transfer and (transfer != 'chunked' or lengths):
                raise DownloadError('Ambiguous or unsupported HTTP transfer encoding')
            allowance = min(max_bytes, max_cache_bytes - used)
            if allowance <= 0 or (advertised is not None and advertised > allowance):
                raise DownloadError('Model cache disk budget would be exceeded')
            if shutil.disk_usage(directory).free < (advertised or 1) + DISK_RESERVE_BYTES:
                raise DownloadError('Insufficient free disk space for model')
            descriptor, temporary = tempfile.mkstemp(prefix='.download-', suffix='.part', dir=directory)
            partial = Path(temporary)
            digest = hashlib.sha256()
            received = 0
            with os.fdopen(descriptor, 'wb') as output:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise DownloadError('Model download exceeded total time limit')
                    if connection.sock is not None:
                        connection.sock.settimeout(min(timeout, remaining))
                    chunk = response.read1(min(1024**2, allowance - received + 1))
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > allowance:
                        raise DownloadError('Model exceeds download byte or cache disk budget')
                    if shutil.disk_usage(directory).free < len(chunk) + DISK_RESERVE_BYTES:
                        raise DownloadError('Insufficient free disk space for model')
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if received == 0 or (advertised is not None and received != advertised):
                raise DownloadError('Empty or truncated model download')
            # Enforce framing completion even when HTTPResponse.read1 returned
            # EOF quietly for a truncated chunked stream (normally it raises).
            if response.length not in (None, 0):
                raise DownloadError('Truncated model download')
            checksum = digest.hexdigest()
            final = directory / (checksum + '.gguf')
            os.replace(partial, final)
            partial = None
            return final, checksum
        raise DownloadError('Model download redirect limit exceeded')
    except DownloadError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise DownloadError(f'Model download failed: {type(exc).__name__}') from exc
    finally:
        if timer is not None:
            timer.cancel()
        if connection is not None:
            connection.close()
        if partial is not None:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass


def inspect_url(url, cache_dir, max_bytes=DEFAULT_MAX_BYTES, **download_options):
    """Convenience wrapper adding full file checksum and cached path."""
    path, digest = fetch_model(url, cache_dir, max_bytes, **download_options)
    result = inspect_file(path)
    result.update({'sha256': digest, 'path': str(path)})
    return result
