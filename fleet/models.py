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
# GGML type id -> (elements per block, bytes per block). Unknown types must not
# be estimated from a filename or bits-per-weight label.
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
    30: (1, 2),      # BF16
}
_SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
            6: 'f', 7: 'B', 10: 'Q', 11: 'q', 12: 'd'}
_BLOCK = re.compile(r'^blk\.(0|[1-9][0-9]*)\.(.+)$')
_ARRAY_KEYS = {f'{arch}.attention.{suffix}' for arch in ('llama', 'qwen2')
               for suffix in ('head_count', 'head_count_kv', 'key_length', 'value_length')}
_REQUIRED = ('attn_norm.weight', 'attn_q.weight', 'attn_k.weight',
             'attn_v.weight', 'attn_output.weight', 'ffn_norm.weight',
             'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight')


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
    value = metadata.get(key, default)
    values = value if isinstance(value, list) else [value] * layers
    if len(values) != layers:
        raise ModelError(f'{key} must have exactly {layers} layer entries')
    return [_positive_int(v, key) for v in values]


def inspect_file(path) -> dict:
    """Inspect a regular GGUF v3 file without reading tensor payloads.

    Invalid structure raises ModelError. Unsupported, well-formed models return
    supported=False and explanatory unsupported_reasons; never plan those models.
    layer_bytes/input_bytes/output_bytes include actual tensor storage spans and
    padding. Tied embeddings are additionally charged to the output stage. The
    file header is reported separately, not presented as resident tensor memory.
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
                metadata[key] = reader.value(kind, retain=key in _ARRAY_KEYS)
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
    if architecture not in ('llama', 'qwen2'):
        unsupported.append(f'Unsupported architecture {architecture!r}; only standard dense llama/qwen2 are supported. Hybrid/recurrent KV layouts are not inferred.')
    if any(key.startswith('split.') for key in metadata) or re.search(r'-\d{5}-of-\d{5}\.gguf$', path.name):
        unsupported.append('Split GGUF files are unsupported; supply a complete merged single-file model')
    if metadata.get('general.type', 'model') != 'model':
        unsupported.append('Only complete model GGUF files are supported (not adapters or vocabularies)')
    layers = _positive_int(metadata.get(f'{architecture}.block_count'), 'block_count', MAX_LAYERS)
    hidden = _positive_int(metadata.get(f'{architecture}.embedding_length'), 'embedding_length')
    for suffix in ('expert_count', 'expert_used_count'):
        if metadata.get(f'{architecture}.{suffix}', 0) != 0:
            unsupported.append('Mixture-of-experts models are unsupported')
            break
    if any('.ssm.' in k or '.recurrent.' in k for k in metadata):
        unsupported.append('Hybrid/recurrent model state is unsupported')
    layer_bytes = [0] * layers
    input_bytes = tensors[0]['offset']
    output_bytes = 0
    block_names = [set() for _ in range(layers)]
    by_name = {t['name']: t for t in tensors}
    shared = []
    for tensor in tensors:
        name = tensor['name']
        match = _BLOCK.fullmatch(name)
        if match:
            layer = int(match.group(1))
            if layer >= layers:
                raise ModelError('Tensor block index exceeds declared block_count')
            layer_bytes[layer] += tensor['span_bytes']
            block_names[layer].add(match.group(2))
            if any(marker in name for marker in ('_exps.', '_shexp.', 'ssm_', 'recurrent')):
                unsupported.append('Expert/hybrid tensor layout is unsupported')
        elif name.startswith('blk.'):
            raise ModelError(f'Malformed block tensor name {name}')
        elif name.startswith(('output.', 'output_norm.')):
            output_bytes += tensor['span_bytes']
        else:
            input_bytes += tensor['span_bytes']
            shared.append(name)
    # Unknown shared tensors still consume memory. They also prevent declaring
    # an unfamiliar graph to be a supported dense transformer.
    # The pinned llama loader may duplicate RoPE weight tensors across devices.
    # Refuse those layouts until replication is modeled, rather than charge a
    # single host copy and undercount each remote stage.
    if any('rope_freqs' in name or 'rope_factors' in name for name in by_name):
        unsupported.append('RoPE weight-tensor replication across RPC devices is not yet modeled')
    known_shared = {'token_embd.weight'}
    unfamiliar = sorted(set(shared) - known_shared)
    if unfamiliar:
        unsupported.append('Unrecognized shared tensors: ' + ', '.join(unfamiliar[:8]))
    for required in ('token_embd.weight', 'output_norm.weight'):
        if required not in by_name:
            unsupported.append(f'Missing required shared tensor {required}')
    allowed_block = set(_REQUIRED) | {name.replace('.weight', '.bias') for name in _REQUIRED}
    allowed_block |= {'rope_freqs.weight'}
    for layer, available in enumerate(block_names):
        unfamiliar_block = available - allowed_block
        if unfamiliar_block:
            unsupported.append(f'Block {layer} has unrecognized tensors: ' + ', '.join(sorted(unfamiliar_block)[:8]))
        missing = set(_REQUIRED) - available
        if missing:
            unsupported.append(f'Block {layer} is incomplete: missing ' + ', '.join(sorted(missing)))
    tied = 'output.weight' not in by_name and 'token_embd.weight' in by_name
    replicated = by_name['token_embd.weight']['span_bytes'] if tied else 0
    output_bytes += replicated
    kv = []
    # Only compute KV for a supported architecture with a complete known graph.
    if not unsupported:
        heads = _per_layer(metadata, f'{architecture}.attention.head_count', layers)
        kv_heads = _per_layer(metadata, f'{architecture}.attention.head_count_kv', layers, heads)
        key_lengths = metadata.get(f'{architecture}.attention.key_length')
        value_lengths = metadata.get(f'{architecture}.attention.value_length')
        for layer in range(layers):
            if heads[layer] % kv_heads[layer]:
                raise ModelError('attention.head_count must be divisible by head_count_kv')
            if hidden % heads[layer] and (key_lengths is None or value_lengths is None):
                raise ModelError('embedding_length must be divisible by head_count for implicit head sizes')
        default_dims = [hidden // count for count in heads]
        key_dims = _per_layer(metadata, f'{architecture}.attention.key_length', layers, default_dims)
        value_dims = _per_layer(metadata, f'{architecture}.attention.value_length', layers, default_dims)
        embedding = by_name['token_embd.weight']['shape']
        if len(embedding) != 2 or embedding[0] != hidden:
            raise ModelError('token_embd.weight shape disagrees with embedding_length')
        if by_name['output_norm.weight']['shape'] != [hidden]:
            raise ModelError('output_norm.weight shape disagrees with embedding_length')
        if not tied and by_name['output.weight']['shape'] != embedding:
            raise ModelError('output.weight shape disagrees with token embedding')
        for layer in range(layers):
            prefix = f'blk.{layer}.'
            expected = {'attn_norm.weight': [hidden], 'ffn_norm.weight': [hidden],
                        'attn_q.weight': [hidden, heads[layer] * key_dims[layer]],
                        'attn_k.weight': [hidden, kv_heads[layer] * key_dims[layer]],
                        'attn_v.weight': [hidden, kv_heads[layer] * value_dims[layer]],
                        'attn_output.weight': [heads[layer] * value_dims[layer], hidden]}
            for suffix, shape in expected.items():
                if by_name[prefix + suffix]['shape'] != shape:
                    raise ModelError(f'{prefix + suffix} shape disagrees with attention metadata')
            gate = by_name[prefix + 'ffn_gate.weight']['shape']
            if len(gate) != 2 or gate[0] != hidden or by_name[prefix + 'ffn_up.weight']['shape'] != gate or by_name[prefix + 'ffn_down.weight']['shape'] != [gate[1], hidden]:
                raise ModelError(f'Block {layer} feed-forward shapes are inconsistent')
            kv.append(kv_heads[layer] * (key_dims[layer] + value_dims[layer]) * 2)
    unsupported = list(dict.fromkeys(unsupported))
    identity = hashlib.sha256((header_hash + ':' + str(size)).encode('ascii')).hexdigest()
    return {
        'model_id': 'gguf-meta-sha256:' + identity,
        'identity_kind': 'metadata-only-sha256',
        'format': 'gguf', 'gguf_version': 3, 'architecture': architecture,
        'name': metadata.get('general.name', path.name),
        'supported': not unsupported, 'unsupported_reasons': unsupported,
        'layer_count': layers, 'layer_bytes': layer_bytes,
        'input_bytes': input_bytes, 'output_bytes': output_bytes,
        'kv_bytes_per_layer_per_token': kv, 'kv_cache_dtype': 'fp16' if kv else None,
        'kv_element_bytes': 2 if kv else None,
        'kv_elements_per_layer_per_token': [value // 2 for value in kv],
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
