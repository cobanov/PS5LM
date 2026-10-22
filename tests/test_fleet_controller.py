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

    def test_unsupported_and_long_context(self):
        for change in [dict(supported=False, unsupported_reasons=['hybrid']), dict(architecture='qwen35'),
                       dict(kv_cache_dtype='q8_0'), dict(kv_bytes_per_layer_per_token=[]), dict(layer_count=513)]:
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

    def test_inventory_invalid(self):
        for conf in ({}, dict(inventory(), nodes=[]), inventory(1), inventory(16), inventory(17), inventory(reserve=32)):
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
        self.assertEqual(args[args.index('--model')+1], '/tmp/a model.gguf')
        self.assertEqual(args[args.index('--parallel')+1], '1')

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
