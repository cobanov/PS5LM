import bisect
import itertools
import json
import os
from pathlib import Path
import struct
import signal
import time
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fleet import controller, models, planner
from test_fleet_models import write_gguf


def inventory(count=2, memory=32, reserve=1):
    return dict(host_memory_mib=64, host_reserve_mib=1,
                nodes=[dict(id=f'n{i}', endpoint=f'127.0.0.1:{50052+i}', memory_mib=memory, reserve_mib=reserve)
                       for i in range(count)])


def model(count=6):
    return dict(supported=True, architecture='llama', context_length=4096, layer_count=count,
                layer_bytes=[2*planner.MIB]*count, kv_bytes_per_layer_per_token=[64]*count,
                kv_cache_dtype='fp16', kv_element_bytes=2, input_bytes=planner.MIB,
                output_bytes=3*planner.MIB, file_size_bytes=(count*2+4)*planner.MIB)


class PlannerTests(unittest.TestCase):
    def test_complete_coverage_and_kv(self):
        result = planner.plan(model(), inventory(), 300)
        self.assertEqual(result['padded_context_tokens'], 512)
        self.assertEqual(sum(n['layer_end']-n['layer_start'] for n in result['nodes']), 6)
        self.assertEqual(result['nodes'][0]['layer_start'], 0)
        self.assertEqual(result['nodes'][-1]['layer_end'], 6)
        self.assertFalse(result['nodes'][0]['output_layer'])
        self.assertTrue(result['nodes'][-1]['output_layer'])
        self.assertEqual(sum(n['kv_bytes'] for n in result['nodes']), 6*64*512)

    def test_midpoint_ratios_match_llama_float32_upper_bound(self):
        f32 = lambda v: struct.unpack('f', struct.pack('f', v))[0]
        for count in range(2, 40):
            for workers in range(2, min(count, 8)+1):
                result = planner.plan(model(count), inventory(workers, memory=200), 32)
                cumulative = list(itertools.accumulate(result['tensor_split']))
                splits = [f32(v/cumulative[-1]) for v in cumulative]
                for i, node in enumerate(result['nodes']):
                    for layer in range(node['layer_start'], node['layer_end']):
                        self.assertEqual(bisect.bisect_right(splits, f32(layer/(count+1))), i)
                self.assertEqual(bisect.bisect_right(splits, f32(count/(count+1))), workers-1)

    def test_unequal_layers_choose_feasible_partition(self):
        info = model(4)
        info['layer_bytes'] = [8*planner.MIB, 2*planner.MIB, planner.MIB, planner.MIB]
        conf = inventory(memory=12)
        conf['nodes'][1]['memory_mib'] = 8
        result = planner.plan(info, conf, 32)
        self.assertEqual(result['nodes'][0]['layer_end'], 2)
        for node in result['nodes']:
            self.assertLessEqual(node['estimated_bytes'], node['memory_mib'] * planner.MIB)

    def test_infeasible_not_silently_cpu_spill(self):
        with self.assertRaisesRegex(planner.FleetError, 'No contiguous'):
            planner.plan(model(), inventory(memory=5), 4096)

    def test_maximum_15_nodes_executes_cli_limit(self):
        result = planner.plan(model(16), inventory(15, memory=64), 32)
        self.assertEqual(len(result['tensor_split']), 15)

    def test_host_budget(self):
        conf = inventory()
        conf['host_memory_mib'] = 1
        with self.assertRaisesRegex(planner.FleetError, 'Host'):
            planner.plan(model(), conf)

    def test_window_state_and_per_worker_memory(self):
        info = dict(model(), swa_kv_bytes_per_layer_per_token=[32]*6, sliding_window=100,
                    state_bytes_per_layer=[1000]*6, per_worker_bytes=4096)
        result = planner.plan(info, inventory(), 300)
        # 512-token cache; window layers keep min(512, 100 + 128) padded to 256.
        self.assertEqual(result['window_tokens'], 256)
        self.assertEqual(sum(n['kv_bytes'] for n in result['nodes']), 6*64*512 + 6*32*256)
        self.assertEqual(sum(n['state_bytes'] for n in result['nodes']), 6000)
        plain = planner.plan(model(), inventory(), 300)
        for with_extras, without in zip(result['nodes'], plain['nodes']):
            self.assertEqual(with_extras['weights_bytes'] - without['weights_bytes'], 4096)
        first = planner.plan(dict(model(), first_worker_bytes=8192), inventory(), 300)
        self.assertEqual([n['weights_bytes'] - p['weights_bytes'] for n, p in zip(first['nodes'], plain['nodes'])],
                         [8192, 0])
        with self.assertRaisesRegex(planner.FleetError, 'window size'):
            planner.plan(dict(info, sliding_window=0), inventory(), 300)

    def test_manual_split(self):
        result = planner.plan(model(), inventory(), 300, layer_counts=[2, 4])
        self.assertEqual([(n['layer_start'], n['layer_end']) for n in result['nodes']], [(0, 2), (2, 6)])
        self.assertEqual(result['split_mode'], 'manual')
        for counts in ([3, 3, 0], [1, 4], [0, 6], [6]):
            with self.subTest(counts=counts), self.assertRaisesRegex(planner.FleetError, 'manual split'):
                planner.plan(model(), inventory(), 300, layer_counts=counts)
        with self.assertRaisesRegex(planner.FleetError, 'does not fit'):
            planner.plan(model(), inventory(memory=12), 300, layer_counts=[1, 5])

    def test_any_supported_architecture_is_planned(self):
        result = planner.plan(dict(model(), architecture='qwen35'), inventory(), 300)
        self.assertEqual(result['architecture'], 'qwen35')

    def test_unsupported_and_long_context(self):
        for change in [dict(supported=False, unsupported_reasons=['hybrid']),
                       dict(kv_cache_dtype='q8_0'), dict(kv_bytes_per_layer_per_token=[]), dict(layer_count=513),
                       dict(state_bytes_per_layer=[1]*5)]:
            with self.subTest(change=change), self.assertRaises(planner.FleetError):
                planner.plan(dict(model(), **change), inventory())
        with self.assertRaisesRegex(planner.FleetError, 'training context'):
            planner.plan(model(), inventory(), 8192)

    def test_endpoint_validation(self):
        for value in ('0.0.0.0:50052', '8.8.8.8:50052', '100.64.0.1:1', '169.254.1.2:1',
                      'ps5:50052', 'http://127.0.0.1:1', '[::1]:50', '127.0.0.1:0', '127.0.0.1:65536',
                      '127.0.0.1:1,--host', '192.168.001.1:50', None):
            with self.subTest(value=value), self.assertRaises(planner.FleetError):
                planner.endpoint(value)
        for value in ('192.168.1.21:50052', '10.1.1.1:50052', '172.31.1.1:1', '127.0.0.1:50052'):
            self.assertEqual(planner.endpoint(value), value)

    def test_parallel_slots(self):
        # Each slot its own stream: ctx-size / slots cells each, padded to 256.
        self.assertEqual(planner.cache_cells(1000, 4), (1024, 0, 256))
        self.assertEqual(planner.cache_cells(4096, 4, window=100), (4096, 4 * 256, 1024))
        # One shared stream; window layers keep window * slots + ubatch.
        self.assertEqual(planner.cache_cells(4096, 4, unified=True, window=100), (4096, 768, 4096))
        info = dict(model(), state_bytes_per_layer=[1000]*6)
        one = planner.plan(info, inventory(memory=64), 1024)
        four = planner.plan(info, inventory(memory=64), 4096, slots=4)
        self.assertEqual(sum(n['state_bytes'] for n in four['nodes']), 4 * 6000)
        self.assertEqual(sum(n['kv_bytes'] for n in four['nodes']), 4 * sum(n['kv_bytes'] for n in one['nodes']))
        self.assertEqual((four['parallel_sequences'], four['kv_unified'], four['context_per_slot']), (4, False, 1024))
        # The training context limits each request, not the whole cache.
        planner.plan(info, inventory(memory=64), 16384, slots=4)
        with self.assertRaisesRegex(planner.FleetError, 'training context'):
            planner.plan(info, inventory(memory=64), 8192, slots=4, unified=True)
        args = controller.command('/tmp/server', '/tmp/m.gguf', planner.plan(info, inventory(memory=64), 4096,
                                                                           slots=4, unified=True))
        self.assertEqual(args[args.index('--parallel')+1], '4')
        self.assertIn('--kv-unified', args)

    def test_one_node_is_a_split_of_one(self):
        result = planner.plan(model(), inventory(1, memory=64), 300)
        self.assertEqual((result['nodes'][0]['layer_start'], result['nodes'][0]['layer_end']), (0, 6))
        self.assertTrue(result['nodes'][0]['output_layer'])

    def test_inventory_invalid(self):
        for conf in ({}, dict(inventory(), nodes=[]), inventory(16), inventory(17), inventory(reserve=32)):
            with self.subTest(config=conf), self.assertRaises(planner.FleetError):
                planner.inventory(conf)
        conf = inventory()
        conf['nodes'][1]['endpoint'] = conf['nodes'][0]['endpoint']
        with self.assertRaisesRegex(planner.FleetError, 'Duplicate'):
            planner.inventory(conf)
        conf = inventory()
        conf['nodes'][0]['memory_mib'] = True
        with self.assertRaises(planner.FleetError):
            planner.inventory(conf)


class ControllerTests(unittest.TestCase):
    def test_generated_command_pins_devices_and_runtime(self):
        result = planner.plan(model(), inventory())
        args = controller.command('/tmp/server', '/tmp/a model.gguf', result)
        self.assertEqual(args[args.index('--device')+1], 'RPC0,RPC1')
        self.assertEqual(args[args.index('--host')+1], '127.0.0.1')
        self.assertEqual(args[args.index('--fit')+1], 'off')
        self.assertEqual(args[args.index('--model')+1], str(Path('/tmp/a model.gguf').resolve()))
        self.assertEqual(args[args.index('--parallel')+1], '1')
        self.assertEqual(args[args.index('--ubatch-size')+1], str(planner.UBATCH))
        self.assertEqual(args[args.index('--ctx-checkpoints')+1], '1')
        self.assertEqual(args[args.index('--cache-ram')+1], '0')
        self.assertIn('--backend-sampling', args)   # the token comes back from the worker, not the logits

    def test_check_margin_scales_down_for_small_budgets(self):
        self.assertEqual(controller.check_margin(12000), controller.CHECK_MARGIN_MIB)
        self.assertEqual(controller.check_margin(128), 16)      # the RPC smoke test's workers
        self.assertEqual(controller.check_margin(4), 1)

    def test_direct_command_uses_one_local_gpu(self):
        args = controller.direct_command('/tmp/server', '/tmp/m.gguf', 'CUDA0', 65536, 18100, slots=2,
                                         cache_type='q8_0', flash_attn=True)
        for flag, value in (('--device', 'CUDA0'), ('--ctx-size', '65536'), ('--parallel', '2'),
                            ('--cache-type-k', 'q8_0'), ('--flash-attn', 'on'), ('--host', '127.0.0.1'),
                            ('--port', '18100'), ('--ubatch-size', str(controller.DIRECT_UBATCH))):
            self.assertEqual(args[args.index(flag)+1], value)
        self.assertNotIn('--rpc', args)
        with self.assertRaisesRegex(planner.FleetError, 'flash attention'):
            controller.direct_check('/tmp/server', '/tmp/m.gguf', 'CUDA0', 8000, 4096, cache_type='q8_0')

    def test_environment_cannot_override_plan(self):
        with patch.dict(os.environ, {'LLAMA_ARG_MODEL': 'bad', 'LLAMA_ARG_RPC': '8.8.8.8:1'}):
            env = controller.child_environment()
            self.assertNotIn('LLAMA_ARG_MODEL', env)
            self.assertNotIn('LLAMA_ARG_RPC', env)
            self.assertEqual(env['GGML_RPC_NO_RDMA'], '1')

    def test_preflight_caps_declared_budget(self):
        text = ('Available devices:\n  RPC0: 127.0.0.1:50052 (64 MiB, 24 MiB free)\n'
                '  RPC1: 127.0.0.1:50053 (64 MiB, 48 MiB free)\n')
        with patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, text, '')):
            conf = controller.probe('/tmp/server', inventory())
            self.assertEqual([n['memory_mib'] for n in conf['nodes']], [24, 32])

    def test_preflight_fail_closed(self):
        texts = ['', 'Available devices:\n  (none)\n',
                 'RPC0: 127.0.0.1:50052 (32 MiB, 0 MiB free)\n',
                 'RPC0: 127.0.0.1:50052 (32 MiB, 40 MiB free)\n',
                 'RPC0: 127.0.0.1:50052 (32 MiB, 20 MiB free)\nRPC1: 127.0.0.1:50052 (32 MiB, 20 MiB free)\n']
        for text in texts:
            with self.subTest(text=text), patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, text, '')):
                with self.assertRaises(planner.FleetError):
                    controller.probe('/tmp/server', inventory())
        with patch.object(subprocess, 'run', side_effect=subprocess.TimeoutExpired('server', 1)):
            with self.assertRaisesRegex(planner.FleetError, 'no model server'):
                controller.probe('/tmp/server', inventory())

    def test_cli_plan_real_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            path = write_gguf(d/'tiny.gguf', layers=4)
            config = d/'nodes.json'
            config.write_text(json.dumps(inventory()))
            output = d/'plan.json'
            result = subprocess.run([os.sys.executable, '-m', 'fleet', 'plan', '--model', str(path),
                                     '--nodes', str(config), '--context', '256', '--plan-out', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            decoded = json.loads(output.read_text())
            self.assertEqual(decoded['model_sha256'], controller.checksum(path))
            self.assertEqual(decoded['status'], 'estimated_not_reserved')
            self.assertEqual(decoded['memory_source'], 'operator_budgets_only')
            for action in ('serve', 'plan'):
                argv = [action, '--model', str(path), '--nodes', str(config)]
                if action == 'plan':
                    argv += ['--sha256', '0'*64]
                self.assertEqual(controller.main(argv), 2)

    @unittest.skipUnless(os.name == 'posix', 'POSIX wrapper signal ownership')
    def test_wrapper_sigterm_reaps_server(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/'server.pid'
            child_code = ("import os,time; from pathlib import Path; "
                          f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(120)")
            wrapper_code = ("from fleet.controller import run_foreground; import sys; "
                            f"sys.exit(run_foreground({[os.sys.executable, '-c', child_code]!r}))")
            wrapper = subprocess.Popen([os.sys.executable, '-c', wrapper_code])
            try:
                deadline = time.monotonic()+5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pidfile.exists())
                child_pid = int(pidfile.read_text())
                wrapper.send_signal(signal.SIGTERM)
                self.assertEqual(wrapper.wait(timeout=5), 143)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.wait()

    def test_manifest_hash_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d/'bin').mkdir()
            binary = d/'bin/llama-server'
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o700)
            with self.assertRaisesRegex(planner.FleetError, 'Missing build manifest'):
                controller.verify_binary(binary)
            manifest = d/'fleet-build.json'
            data = dict(llama_revision=controller.LLAMA_REVISION, server_sha256=controller.checksum(binary))
            manifest.write_text(json.dumps(data))
            controller.verify_binary(binary)
            binary.write_text('#!/bin/sh\nexit 1\n')
            with self.assertRaisesRegex(planner.FleetError, 'checksum'):
                controller.verify_binary(binary)


if __name__ == '__main__':
    unittest.main()
