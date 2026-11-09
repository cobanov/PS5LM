"""Synthetic GGUF fixtures and offline download policy/security tests."""
import hashlib
from http.client import IncompleteRead
from email.message import Message
from pathlib import Path
import socket
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch

from fleet import models


def _string(value):
    encoded = value.encode('utf-8')
    return struct.pack('<Q', len(encoded)) + encoded


def _value(value):
    if isinstance(value, str):
        return struct.pack('<I', 8) + _string(value)
    if isinstance(value, list):
        return struct.pack('<IIQ', 9, 4, len(value)) + b''.join(struct.pack('<I', item) for item in value)
    return struct.pack('<II', 4, value)


def write_gguf(path, layers=2, arch='llama', tied=True, *, metadata=None,
               omit=(), extra=(), offsets=None, dtype=0, hidden=4,
               heads=2, kv_heads=1, fill=0):
    """Write a complete tiny, structurally valid dense transformer GGUF."""
    meta = {'general.architecture': arch, 'general.name': 'Tiny offline fixture',
            f'{arch}.block_count': layers, f'{arch}.embedding_length': hidden,
            f'{arch}.context_length': 4096,
            f'{arch}.attention.head_count': heads,
            f'{arch}.attention.head_count_kv': kv_heads}
    meta.update(metadata or {})
    specs = [('token_embd.weight', [hidden, 8], dtype)]
    key_dims = meta.get(f'{arch}.attention.key_length', hidden // heads)
    value_dims = meta.get(f'{arch}.attention.value_length', hidden // heads)
    for layer in range(layers):
        layer_kv = kv_heads[layer] if isinstance(kv_heads, list) else kv_heads
        layer_key = key_dims[layer] if isinstance(key_dims, list) else key_dims
        layer_value = value_dims[layer] if isinstance(value_dims, list) else value_dims
        parts = [('attn_norm.weight', [hidden]),
                 ('attn_q.weight', [hidden, heads * layer_key]),
                 ('attn_k.weight', [hidden, layer_kv * layer_key]),
                 ('attn_v.weight', [hidden, layer_kv * layer_value]),
                 ('attn_output.weight', [heads * layer_value, hidden]),
                 ('ffn_norm.weight', [hidden]),
                 ('ffn_gate.weight', [hidden, 8]),
                 ('ffn_up.weight', [hidden, 8]),
                 ('ffn_down.weight', [8, hidden])]
        specs += [(f'blk.{layer}.{name}', shape, 0) for name, shape in parts]
    specs += [('output_norm.weight', [hidden], 0)]
    if not tied:
        specs += [('output.weight', [hidden, 8], dtype)]
    specs += list(extra)
    specs = [item for item in specs if item[0] not in omit]
    header = b'GGUF' + struct.pack('<IQQ', 3, len(specs), len(meta))
    for key, value in meta.items():
        header += _string(key) + _value(value)
    data = bytearray()
    for index, (name, shape, encoding) in enumerate(specs):
        elements = 1
        for dimension in shape:
            elements *= dimension
        block, block_size = models.GGML_TYPES.get(encoding, (1, 4))
        nbytes = elements // block * block_size
        offset = offsets[index] if offsets is not None else len(data)
        header += _string(name) + struct.pack('<I', len(shape))
        header += b''.join(struct.pack('<Q', dim) for dim in shape)
        header += struct.pack('<IQ', encoding, offset)
        data.extend(bytes([fill]) * models._align(nbytes, 32))
    header += bytes(models._align(len(header), 32) - len(header))
    Path(path).write_bytes(header + data)
    return Path(path)


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'tiny.gguf'

    def test_dense_llama_gqa_exact_accounting(self):
        model = models.inspect_file(write_gguf(self.path))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        self.assertEqual(model['layer_bytes'], [640, 640])
        self.assertEqual(model['input_bytes'], 128)
        self.assertEqual(model['output_bytes'], 160)
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [8, 8])
        self.assertEqual(model['kv_cache_dtype'], 'fp16')
        self.assertEqual(model['kv_element_bytes'], 2)
        self.assertEqual(model['context_length'], 4096)
        self.assertEqual(model['file_size_bytes'], self.path.stat().st_size)
        self.assertEqual(sum(model['layer_bytes']) + model['input_bytes'] + model['output_bytes']
                         + model['per_worker_bytes'] + model['first_worker_bytes'] + model['mtp_bytes'],
                         model['tensor_storage_bytes'] + model['replicated_output_bytes'])
        self.assertEqual(model['state_bytes_per_layer'], [0, 0])
        self.assertEqual(model['swa_kv_bytes_per_layer_per_token'], [0, 0])

    def test_rope_freqs_one_copy_per_worker(self):
        baseline = models.inspect_file(write_gguf(self.path))
        model = models.inspect_file(write_gguf(self.path, extra=[('rope_freqs.weight', [2], 0)]))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        self.assertEqual(model['per_worker_bytes'], 32)
        self.assertEqual(model['input_bytes'], baseline['input_bytes'])
        self.assertEqual(model['output_bytes'], baseline['output_bytes'])

    def test_qwen2_untied_output(self):
        model = models.inspect_file(write_gguf(self.path, arch='qwen2', tied=False))
        self.assertTrue(model['supported'])
        self.assertFalse(model['tied_embeddings'])
        self.assertEqual(model['replicated_output_bytes'], 0)
        self.assertEqual(model['output_bytes'], 160)

    def test_per_layer_gqa_and_head_size(self):
        model = models.inspect_file(write_gguf(self.path, kv_heads=[1, 2], metadata={
            'llama.attention.key_length': 3, 'llama.attention.value_length': 2}))
        # K per layer (3 and 6 values); without flash attention both layers' V
        # rows take the widest V (2 heads * 2 = 4 values).
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [14, 20])
        self.assertEqual(model['kv_elements_per_layer_per_token'], [7, 10])

    def test_identity_is_explicitly_not_full_weight_hash(self):
        write_gguf(self.path, fill=1)
        first = models.inspect_file(self.path)
        sha1 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        write_gguf(self.path, fill=2)
        second = models.inspect_file(self.path)
        sha2 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(first['model_id'], second['model_id'])
        self.assertNotEqual(sha1, sha2)
        self.assertEqual(first['identity_kind'], 'metadata-only-sha256')

    def test_hybrid_without_state_sizes_no_fake_kv(self):
        model = models.inspect_file(write_gguf(self.path, arch='qwen35'))
        self.assertFalse(model['supported'])
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [])
        self.assertIn('state sizes', ' '.join(model['unsupported_reasons']))

    QWEN35_SSM = {'qwen35.ssm.conv_kernel': 4, 'qwen35.ssm.inner_size': 8,
                  'qwen35.ssm.state_size': 2, 'qwen35.ssm.group_count': 1}

    def test_qwen35_attention_every_fourth_layer_rest_recurrent(self):
        model = models.inspect_file(write_gguf(self.path, arch='qwen35', layers=8, metadata=self.QWEN35_SSM))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        # n_embd_r = (4-1)*(8+2*1*2) = 36, n_embd_s = 2*8 = 16, f32: 208 bytes.
        self.assertEqual(model['state_bytes_per_layer'], [208, 208, 208, 0, 208, 208, 208, 0])
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [0, 0, 0, 8, 0, 0, 0, 8])
        self.assertEqual(model['recurrent_layer_count'], 6)

    def test_qwen35_interval_and_explicit_recurrent_flags(self):
        model = models.inspect_file(write_gguf(self.path, arch='qwen35', layers=4, metadata=dict(
            self.QWEN35_SSM, **{'qwen35.full_attention_interval': 2})))
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [0, 8, 0, 8])
        model = models.inspect_file(write_gguf(self.path, arch='qwen35', layers=4, metadata=dict(
            self.QWEN35_SSM, **{'qwen35.attention.recurrent_layers': [0, 1, 1, 0]})))
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [8, 0, 0, 8])

    def test_nextn_block_neither_resident_nor_cached(self):
        model = models.inspect_file(write_gguf(self.path, arch='qwen35', layers=5, metadata=dict(
            self.QWEN35_SSM, **{'qwen35.nextn_predict_layers': 1})))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        # The NextN block keeps its tensor-split slot, but llama-server loads
        # neither its weights (load_mtp is off) nor a cache for it.
        self.assertEqual(model['layer_count'], 5)
        self.assertEqual(model['mtp_layers'], 1)
        self.assertEqual(model['layer_bytes'][4], 0)
        self.assertEqual(model['mtp_bytes'], 640)
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [0, 0, 0, 8, 0])
        self.assertEqual(model['state_bytes_per_layer'][4], 0)

    def test_sliding_window_pattern_head_sizes_and_shared_cache(self):
        model = models.inspect_file(write_gguf(self.path, arch='gemma4', layers=4, metadata={
            'gemma4.attention.sliding_window': 16, 'gemma4.attention.sliding_window_pattern': [1, 0, 1, 0],
            'gemma4.attention.key_length_swa': 1, 'gemma4.attention.value_length_swa': 1,
            'gemma4.attention.shared_kv_layers': 1}))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        # Without flash attention every layer's V takes the widest V (2 here):
        # window layers keep K 1 + V 2, full layers K 2 + V 2, at 2 bytes each.
        self.assertEqual(model['swa_kv_bytes_per_layer_per_token'], [6, 0, 6, 0])
        # The last layer reuses an earlier layer's cache.
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [0, 8, 0, 0])
        self.assertEqual(model['sliding_window'], 16)

    def test_default_window_pattern_from_llama_cpp(self):
        model = models.inspect_file(write_gguf(self.path, arch='gpt-oss', layers=4,
                                               metadata={'gpt-oss.attention.sliding_window': 8}))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        # gpt-oss: period 2, the last layer of each pair keeps the full context.
        self.assertEqual(model['swa_kv_bytes_per_layer_per_token'], [8, 0, 8, 0])
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [0, 8, 0, 8])
        model = models.inspect_file(write_gguf(self.path, arch='gpt-oss', layers=4, metadata={
            'gpt-oss.attention.sliding_window': 8, 'gpt-oss.attention.sliding_window_pattern': 4}))
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [0, 0, 0, 8])

    def test_sliding_window_without_pattern_counts_full_context(self):
        model = models.inspect_file(write_gguf(self.path, metadata={'llama.attention.sliding_window': 8}))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [8, 8])
        self.assertEqual(model['swa_kv_bytes_per_layer_per_token'], [0, 0])
        self.assertEqual(model['sliding_window'], 0)
        self.assertIn('full context', ' '.join(model['notes']))

    def test_layer_zero_tensors_go_with_the_first_worker(self):
        model = models.inspect_file(write_gguf(self.path, extra=[('per_layer_model_proj.weight', [4, 2], 0)]))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        self.assertEqual(model['first_worker_bytes'], 32)

    def test_zero_kv_head_layers_are_recurrent_in_known_hybrids(self):
        model = models.inspect_file(write_gguf(
            self.path, arch='nemotron_h', layers=4, kv_heads=[1, 1, 1, 1],
            metadata={'nemotron_h.attention.head_count_kv': [1, 0, 1, 0], 'nemotron_h.ssm.conv_kernel': 4,
                      'nemotron_h.ssm.inner_size': 8, 'nemotron_h.ssm.state_size': 2, 'nemotron_h.ssm.group_count': 1}))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        self.assertEqual(model['kv_bytes_per_layer_per_token'], [8, 0, 8, 0])
        self.assertEqual(model['state_bytes_per_layer'], [0, 208, 0, 208])

    def test_unmodeled_cache_layouts_fail_closed(self):
        cases = [('rwkv7', {}), ('falcon-h1', {}), ('llama', {'llama.attention.kv_lora_rank': 4}),
                 ('newhybrid', {'newhybrid.ssm.inner_size': 8, 'newhybrid.ssm.state_size': 2})]
        for arch, extra in cases:
            with self.subTest(arch=arch):
                model = models.inspect_file(write_gguf(self.path, arch=arch, metadata=extra))
                self.assertFalse(model['supported'])
                self.assertIn('not modeled', ' '.join(model['unsupported_reasons']))

    def test_moe_and_iq_types_are_counted(self):
        model = models.inspect_file(write_gguf(
            self.path, hidden=32, dtype=20, metadata={'llama.expert_count': 4, 'llama.expert_used_count': 2},
            extra=[('blk.0.ffn_up_exps.weight', [32, 8, 4], 0)]))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        self.assertEqual(model['expert_count'], 4)
        self.assertEqual(model['input_bytes'], 160)  # IQ4_NL: 256/32*18 = 144, aligned to 32

    def test_missing_token_embedding_not_supported(self):
        model = models.inspect_file(write_gguf(self.path, omit=['token_embd.weight'], tied=False))
        self.assertFalse(model['supported'])
        self.assertIn('token_embd', ' '.join(model['unsupported_reasons']))

    def test_encoder_decoder_layout_not_split(self):
        model = models.inspect_file(write_gguf(self.path, extra=[('enc.blk.0.attn_q.weight', [4, 4], 0)]))
        self.assertFalse(model['supported'])
        self.assertIn('Encoder-decoder', ' '.join(model['unsupported_reasons']))

    def test_shared_unknown_tensor_is_counted_and_rejected(self):
        baseline = models.inspect_file(write_gguf(self.path))
        model = models.inspect_file(write_gguf(self.path, extra=[('surprise.weight', [4], 0)]))
        self.assertFalse(model['supported'])
        self.assertEqual(model['output_bytes'], baseline['output_bytes'] + 32)
        self.assertIn('surprise.weight', ' '.join(model['unsupported_reasons']))

    def test_split_metadata_is_not_supported(self):
        model = models.inspect_file(write_gguf(self.path, metadata={'split.count': 2}))
        self.assertFalse(model['supported'])
        self.assertIn('Split GGUF', ' '.join(model['unsupported_reasons']))

    def test_split_filename_is_not_supported(self):
        path = self.path.with_name('tiny-00001-of-00002.gguf')
        self.assertFalse(models.inspect_file(write_gguf(path))['supported'])

    def test_unknown_type_is_explicitly_unsupported(self):
        model = models.inspect_file(write_gguf(self.path, dtype=999))
        self.assertFalse(model['supported'])
        self.assertIn('999', ' '.join(model['unsupported_reasons']))

    def test_any_block_tensor_is_counted_with_its_layer(self):
        # Block contents differ per architecture; llama.cpp itself checks them at
        # load. Their memory belongs to the layer, whatever they are called.
        model = models.inspect_file(write_gguf(self.path, extra=[('blk.0.unrecognized.weight', [4], 0)]))
        self.assertTrue(model['supported'], model['unsupported_reasons'])
        self.assertEqual(model['layer_bytes'][0], 672)

    def test_directory_and_fifo_are_rejected_without_blocking(self):
        with self.assertRaises(models.ModelError):
            models.inspect_file(self.tmp.name)
        if hasattr(models.os, 'mkfifo'):
            fifo = Path(self.tmp.name) / 'named-pipe'
            models.os.mkfifo(fifo)
            with self.assertRaisesRegex(models.ModelError, 'regular file'):
                models.inspect_file(fifo)

    def test_q4_block_bytes(self):
        model = models.inspect_file(write_gguf(self.path, hidden=32, dtype=2))
        self.assertTrue(model['supported'])
        self.assertEqual(model['input_bytes'], 160)  # 8*18 bytes rounded to 32

    def test_quantized_row_must_have_complete_blocks(self):
        write_gguf(self.path, hidden=4, dtype=2)
        with self.assertRaisesRegex(models.ModelError, 'quantization block'):
            models.inspect_file(self.path)

    def test_truncated_tensor(self):
        write_gguf(self.path)
        self.path.write_bytes(self.path.read_bytes()[:-20])
        with self.assertRaisesRegex(models.ModelError, 'truncated'):
            models.inspect_file(self.path)

    def test_overlapping_tensor_offsets(self):
        write_gguf(self.path, offsets=[0] * 20)
        with self.assertRaisesRegex(models.ModelError, 'overlap'):
            models.inspect_file(self.path)

    def test_offsets_outside_file(self):
        write_gguf(self.path, offsets=[2**50 + n * 1024 for n in range(20)])
        with self.assertRaisesRegex(models.ModelError, 'outside'):
            models.inspect_file(self.path)

    def test_metadata_shape_mismatch(self):
        write_gguf(self.path, metadata={'llama.embedding_length': 8})
        with self.assertRaisesRegex(models.ModelError, 'shape disagrees'):
            models.inspect_file(self.path)

    def test_invalid_magic_version_alignment(self):
        for content in (b'nope' + bytes(32), b'GGUF' + struct.pack('<IQQ', 2, 1, 0)):
            self.path.write_bytes(content)
            with self.assertRaises(models.ModelError):
                models.inspect_file(self.path)
        write_gguf(self.path, metadata={'general.alignment': 3})
        with self.assertRaisesRegex(models.ModelError, 'power of two'):
            models.inspect_file(self.path)

    def test_oversized_string_and_array_are_bounded(self):
        base = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
        self.path.write_bytes(base + struct.pack('<Q', 2**63))
        with self.assertRaisesRegex(models.ModelError, 'string exceeds'):
            models.inspect_file(self.path)
        self.path.write_bytes(base + _string('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 2**63))
        with self.assertRaisesRegex(models.ModelError, 'array exceeds'):
            models.inspect_file(self.path)

    def test_huge_tensor_and_metadata_count(self):
        for tensor_count, metadata_count in ((2**63, 0), (1, 2**63)):
            self.path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, tensor_count, metadata_count))
            with self.assertRaises(models.ModelError):
                models.inspect_file(self.path)


class FakeResponse:
    def __init__(self, body=b'GGUF-example', status=200, headers=None, fail=False):
        self.status = status
        self.body = body
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = str(value)
        self.length = None
        self.fail = fail

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, amount):
        if self.fail:
            raise IncompleteRead(b'partial')
        part, self.body = self.body[:amount], self.body[amount:]
        return part


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / 'cache'
        self.connections = []
        self.responses = []
        owner = self

        class FakeConnection:
            def __init__(self, host, pin, timeout):
                self.host, self.pin = host, pin
                self.sock = None
                owner.connections.append(self)
            def request(self, method, target, headers):
                self.method, self.target, self.headers = method, target, headers
            def getresponse(self):
                return owner.responses.pop(0)
            def close(self):
                pass
            def abort(self):
                pass

        self.connection = patch.object(models, '_PinnedHTTPSConnection', FakeConnection)
        self.connection.start()
        self.addCleanup(self.connection.stop)
        self.dns = patch.object(models.socket, 'getaddrinfo', return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))])
        self.dns_mock = self.dns.start()
        self.addCleanup(self.dns.stop)

    def fetch(self, **kwargs):
        return models.fetch_model('https://example.org/model.gguf', self.cache, **kwargs)

    def test_stream_hash_atomic_name(self):
        body = b'GGUF-tiny-model'
        self.responses = [FakeResponse(body, headers={'Content-Length': len(body)})]
        path, digest = self.fetch(max_bytes=100)
        self.assertEqual(path.name, hashlib.sha256(body).hexdigest() + '.gguf')
        self.assertEqual(path.read_bytes(), body)
        self.assertEqual(digest, hashlib.sha256(body).hexdigest())
        self.assertEqual(len(list(self.cache.iterdir())), 1)
        self.assertNotIn('Authorization', self.connections[0].headers)
        self.assertEqual(self.connections[0].pin[3], ('93.184.216.34', 443))

    def test_forbidden_url_forms(self):
        invalid = ['http://example.org/m', 'https://name:pass@example.org/m',
                   'https://example.org:8080/m', 'file:///etc/passwd',
                   'https://example.org/m?access_token=secret',
                   'https://example.org/m#fragment', 'https://example.org/\nX:1',
                   'https://%31%32%37.0.0.1/m', 'https://example.org\\@localhost/m']
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(models.DownloadError):
                models.fetch_model(url, self.cache)
        self.assertEqual(self.connections, [])

    def test_private_and_mixed_dns_blocked(self):
        for address in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '192.168.1.1',
                        '100.64.0.1', '0.0.0.0', '224.0.0.1'):
            self.dns_mock.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443))]
            with self.subTest(address=address), self.assertRaisesRegex(models.DownloadError, 'non-public'):
                self.fetch()
        self.dns_mock.return_value += [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]
        with self.assertRaises(models.DownloadError):
            self.fetch()
        self.assertEqual(self.connections, [])

    def test_ipv6_mapped_private_and_transition_blocked(self):
        for address in ('::1', 'fe80::1', 'fc00::1', '::ffff:127.0.0.1',
                        '64:ff9b::7f00:1', '2002:7f00:1::', 'ff02::1'):
            self.assertFalse(models._public_ip(address), address)
        self.assertTrue(models._public_ip('2606:4700:4700::1111'))

    def test_redirect_validated_no_private_connection(self):
        self.responses = [FakeResponse(status=302, headers={'Location': 'https://localhost/model'})]
        self.dns_mock.side_effect = [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))]]
        with self.assertRaisesRegex(models.DownloadError, 'non-public'):
            self.fetch()
        self.assertEqual(len(self.connections), 1)

    def test_relative_redirect_and_cross_origin_no_auth(self):
        self.responses = [FakeResponse(status=302, headers={'Location': '/next'}),
                          FakeResponse(status=307, headers={'Location': 'https://cdn.example.net/m'}),
                          FakeResponse(b'ok')]
        path, _ = self.fetch()
        self.assertEqual(path.read_bytes(), b'ok')
        self.assertEqual([c.target for c in self.connections], ['/model.gguf', '/next', '/m'])
        self.assertEqual(self.dns_mock.call_count, 3)
        self.assertTrue(all(not {'Authorization', 'Cookie'} & set(c.headers) for c in self.connections))

    def test_dns_resolution_deadline(self):
        release = threading.Event()
        completed = threading.Event()
        def blocked_lookup(*args, **kwargs):
            release.wait(timeout=5)
            completed.set()
            return []
        self.dns_mock.side_effect = blocked_lookup
        try:
            with self.assertRaisesRegex(models.DownloadError, 'DNS resolution'):
                models._resolve_public('example.org', timeout=0.01)
            # A deadline must return while DNS is still blocked. Avoid a tight
            # wall-clock speed assertion that flakes on busy hosted runners.
            self.assertFalse(completed.is_set())
        finally:
            release.set()
            completed.wait(timeout=5)

    def test_numeric_ip_connect_uses_pin_and_original_tls_sni(self):
        # Use the real connection class beneath the mock layer to verify there
        # is no second DNS lookup, and that certificate/SNI hostname is retained.
        self.connection.stop()
        pin = (socket.AF_INET, socket.SOCK_STREAM, 6, ('93.184.216.34', 443))
        with patch.object(models.socket, 'socket') as socket_factory, patch.object(models.ssl, 'create_default_context') as context:
            raw = socket_factory.return_value
            raw.getpeername.return_value = ('93.184.216.34', 443)
            connection = models._PinnedHTTPSConnection('example.org', pin, 1)
            connection.connect()
            raw.connect.assert_called_once_with(('93.184.216.34', 443))
            context.return_value.wrap_socket.assert_called_once_with(raw, server_hostname='example.org', do_handshake_on_connect=False)
            context.return_value.wrap_socket.return_value.do_handshake.assert_called_once()
        self.dns_mock.assert_not_called()

    def test_redirect_limit(self):
        self.responses = [FakeResponse(status=302, headers={'Location': '/next'}) for _ in range(4)]
        with self.assertRaisesRegex(models.DownloadError, 'redirect limit'):
            self.fetch()
        self.assertEqual(len(self.connections), 4)

    def test_known_and_unknown_lengths_bounded_and_cleaned(self):
        for headers in ({'Content-Length': 10}, {}):
            self.responses = [FakeResponse(b'0123456789', headers=headers)]
            with self.assertRaisesRegex(models.DownloadError, 'limit|budget'):
                self.fetch(max_bytes=4)
            self.assertEqual(list(self.cache.iterdir()), [])

    def test_truncation_and_network_error_cleaned(self):
        for response in (FakeResponse(b'ab', headers={'Content-Length': 4}), FakeResponse(fail=True)):
            self.responses = [response]
            with self.assertRaises(models.DownloadError):
                self.fetch()
            self.assertEqual(list(self.cache.iterdir()), [])

    def test_bad_status_and_encoding(self):
        for response in (FakeResponse(status=404), FakeResponse(headers={'Content-Encoding': 'gzip'}),
                         FakeResponse(headers={'Transfer-Encoding': 'chunked', 'Content-Length': 12})):
            self.responses = [response]
            with self.assertRaises(models.DownloadError):
                self.fetch()

    def test_cache_budget_and_space(self):
        self.cache.mkdir()
        (self.cache / 'old.gguf').write_bytes(b'old')
        self.responses = [FakeResponse(b'abc')]
        with self.assertRaisesRegex(models.DownloadError, 'budget'):
            self.fetch(max_cache_bytes=4)
        self.responses = [FakeResponse(b'abc')]
        with patch.object(models.shutil, 'disk_usage') as disk:
            disk.return_value.free = 0
            with self.assertRaisesRegex(models.DownloadError, 'disk space'):
                self.fetch()
        self.assertEqual([p.name for p in self.cache.iterdir()], ['old.gguf'])

    def test_inspect_url_includes_full_hash(self):
        file = Path(self.tmp.name) / 'fixture.gguf'
        write_gguf(file)
        self.responses = [FakeResponse(file.read_bytes())]
        result = models.inspect_url('https://example.org/tiny', self.cache)
        self.assertTrue(result['supported'])
        self.assertEqual(result['sha256'], hashlib.sha256(file.read_bytes()).hexdigest())
        self.assertTrue(Path(result['path']).exists())


if __name__ == '__main__':
    unittest.main()
