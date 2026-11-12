"""Control server: the configuration, consoles over FTP, discovery replies, replicas,
the page's API and its guards, and the ctl client."""
import contextlib
import ftplib
import http.client
import http.server
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fleet import control, ctl, devices
from test_fleet_models import write_gguf


def ps5(name='ps5-a', address='192.168.50.21', **extra):
    return dict(dict(id=name, kind='ps5', address=address, port=30052, reserve_mib=1024, memory_mib=None), **extra)


def gpu(name='pc-gpu', **extra):
    return dict(dict(id=name, kind='gpu', port=50052, device='CUDA0', reserve_mib=1024, memory_mib=None), **extra)


def pool(name='qwen', replicas=(), **extra):
    return dict(dict(id=name, model='/models/qwen.gguf', replicas=list(replicas)), **extra)


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = control.validate_config({})
        self.assertEqual((config['devices'], config['pools'], config['api_port']), ([], [], 8081))

    def test_a_fleet_and_how_each_replica_runs(self):
        config = control.validate_config(dict(
            devices=[ps5(), ps5('ps5-b', '192.168.50.22'), gpu(), dict(id='box', kind='rpc', address='10.0.0.5')],
            pools=[pool(replicas=[dict(devices=['ps5-a', 'pc-gpu']), dict(devices=['ps5-b']),
                                  dict(url='http://10.0.0.9:8080/')]),
                   pool('small', model='https://example.org/s.gguf', replicas=[dict(devices=['box'])])]))
        replicas = config['pools'][0]['replicas']
        self.assertEqual([r.get('via') for r in replicas], ['rpc', 'app', None])
        self.assertEqual(replicas[2], dict(url='http://10.0.0.9:8080', api_key='', slots=1))
        self.assertEqual(config['pools'][1]['replicas'][0]['via'], 'rpc')
        defaults = config['pools'][0]
        self.assertEqual((defaults['cache_type'], defaults['flash_attn'], defaults['context'], defaults['running']),
                         ('q8_0', True, 32768, False))
        self.assertEqual(config['devices'][1]['port'], devices.WORKER_PORT)
        self.assertEqual(config['devices'][3]['port'], 50052)

    def test_refusals(self):
        def fleet(*pools, devices_=None):
            return dict(devices=devices_ or [ps5(), gpu()], pools=list(pools))
        bad = [
            dict(devices=[ps5(port=40000)]),                       # a PS5 title cannot bind 40000+
            dict(devices=[ps5(address='127.0.0.1')]),
            dict(devices=[ps5(address='8.8.8.8')]),                 # public: RPC is unauthenticated
            dict(devices=[gpu(device='CUDA0; rm -rf /')]),
            dict(devices=[ps5(), ps5()]),
            dict(devices=[ps5(), ps5(name='other')]),               # same address and port twice
            dict(devices=[gpu(), gpu('gpu-2', port=50053)]),        # the same GPU twice
            dict(devices=[ps5(memory_mib=0)]),
            dict(devices=[ps5(tags=['has space'])]),
            dict(lan='8.8.8.0/24'), dict(lan='10.0.0.0/8'), dict(lan='nonsense'), dict(api_key='has space'),
            fleet(pool(sha256='abc')), fleet(pool(model='http://example.org/m.gguf')), fleet(pool(context=100)),
            fleet(pool(cache_type='q8_0', flash_attn=False)),       # a quantized V needs flash attention
            fleet(pool(cache_type='q3_k')),
            fleet(pool(replicas=[dict(devices=['ps5-a', 'pc-gpu'], via='app')])),
            fleet(pool(replicas=[dict(devices=['ps5-a'], via='direct')])),
            fleet(pool(replicas=[dict(devices=['ps5-a', 'pc-gpu'], split=[3])])),
            fleet(pool(replicas=[dict(devices=['ps5-a'], via='app', split=[3])])),
            fleet(pool(replicas=[dict(devices=['missing'])])),
            fleet(pool(replicas=[dict(devices=['ps5-a', 'ps5-a'])])),
            fleet(pool(replicas=[dict(devices=['ps5-a']), dict(devices=['ps5-a', 'pc-gpu'])])),   # one replica each
            fleet(pool(running=True, replicas=[dict(devices=['ps5-a'])]),          # two running pools, one device
                  pool('other', running=True, replicas=[dict(devices=['ps5-a', 'pc-gpu'])])),
            fleet(pool(aliases=['shared']), pool('other', aliases=['SHARED'])),
            fleet(pool(), pool()),
            fleet(pool(replicas=[dict(url='ftp://10.0.0.9')])),
            fleet(pool(replicas=[dict(url='http://10.0.0.9/v1; x')])),
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(control.ControlError):
                control.validate_config(raw)

    def test_version_1_configuration_becomes_devices_and_a_stopped_pool(self):
        old = dict(model='/models/Qwen3.8-27B-UD-Q4_K_XL.gguf', sha256='', context=32768, api_port=8081,
                   api_lan=False, api_key='', host_memory_mib=8192, host_reserve_mib=2048, lan='',
                   workers=[dict(ps5(), enabled=True), dict(gpu(), address='127.0.0.1', enabled=True)], split=[])
        config = control.validate_config(old)
        self.assertEqual([d['id'] for d in config['devices']], ['ps5-a', 'pc-gpu'])
        (only,) = config['pools']
        self.assertEqual(only['id'], 'Qwen3.8-27B-UD-Q4_K_XL')
        self.assertEqual(only['replicas'], [dict(devices=['ps5-a', 'pc-gpu'], via='rpc', split=[])])
        self.assertEqual((only['context'], only['running'], only['cache_type']), (32768, False, 'q8_0'))

    def test_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control.json'
            control.save_config(path, control.validate_config(dict(devices=[ps5()])))
            self.assertEqual(control.load_config(path)['devices'][0]['id'], 'ps5-a')

    def test_keys_are_masked_and_kept(self):
        config = control.validate_config(dict(api_key='secret', devices=[ps5()], pools=[
            pool(replicas=[dict(url='http://10.0.0.9:8080', api_key='remote-key')])]))
        shown = control.masked(config)
        self.assertEqual((shown['api_key'], shown['pools'][0]['replicas'][0]['api_key']), ('set', 'set'))
        back = control.validate_config(control.unmask(shown, config))
        self.assertEqual(back, config)

    def test_assign_takes_free_enabled_devices_in_order(self):
        config = control.validate_config(dict(devices=[
            ps5('a', '192.168.50.1', tags=['rack1']), ps5('b', '192.168.50.2', tags=['rack1']),
            ps5('c', '192.168.50.3', enabled=False), ps5('d', '192.168.50.4'), gpu()],
            pools=[pool(replicas=[dict(devices=['b'])])]))
        self.assertEqual(control.assign(config, 2, kinds=['ps5']), [dict(devices=['a']), dict(devices=['d'])])
        self.assertEqual(control.assign(config, 1, per=2), [dict(devices=['a', 'd'])])
        self.assertEqual(control.assign(config, 1, tags=['rack1']), [dict(devices=['a'])])
        with self.assertRaisesRegex(control.ControlError, 'need 4 free devices; 3 match'):
            control.assign(config, 2, per=2)

    def test_console_settings_and_a_reserve_per_kind(self):
        config = control.validate_config(dict(devices=[
            dict(id='ps5-a', kind='ps5', address='192.168.50.21'), dict(id='pc-gpu', kind='gpu'),
            dict(id='box', kind='rpc', address='10.0.0.5')]))
        console, card, box = config['devices']
        self.assertEqual((console['reserve_mib'], card['reserve_mib'], box['reserve_mib']), (512, 1024, 1024))
        self.assertEqual((console['screen'], console['cache_gib']), (False, 32))   # a fleet console is only a worker
        self.assertNotIn('screen', card)
        for bad in (dict(cache_gib=-1), dict(cache_gib=4097), dict(cache_gib='32'), dict(screen='off')):
            with self.assertRaises(control.ControlError):
                control.validate_device(dict(id='ps5-a', kind='ps5', address='192.168.50.21', **bad))

    def test_cache_tokens(self):
        self.assertEqual(control.cache_tokens(dict(context=32768, slots=4, shared_context=False)), 131072)
        self.assertEqual(control.cache_tokens(dict(context=32768, slots=4, shared_context=True)), 32768)


class DiscoveryTests(unittest.TestCase):
    def test_reply_parsing(self):
        reply = (b'HTTP/1.1 200 Ok\nhost-id:ABC\nhost-type:PS5\nhost-name:PS5-123\n'
                 b'host-request-port:997\nsystem-version:11600005\n')
        self.assertEqual(devices.parse_ddp(reply), dict(name='PS5-123', type='PS5', state='on', firmware='11.60'))
        standby = reply.replace(b'200 Ok', b'620 Server Standby')
        self.assertEqual(devices.parse_ddp(standby)['state'], 'rest mode')
        self.assertIsNone(devices.parse_ddp(b'HTTP/1.1 200 Ok\nhost-type:Toaster\n'))
        self.assertIsNone(devices.parse_ddp(b'garbage'))
        self.assertEqual(devices.firmware('09050001'), '9.05')
        self.assertEqual(devices.firmware('bad'), '')


class FakeFTP:
    """Records what the console is told. Files listed in `waiting` stay put when
    the app is asked to take them; anything else the app takes at once. `files`
    are what RETR reads (STOR writes them), `folders` what LIST shows of a folder."""
    log = []
    waiting = set()
    files = {}
    folders = {}

    def connect(self, host, port, timeout=None):
        self.log.append(('connect', host, port))

    def login(self):
        pass

    def storbinary(self, command, stream, blocksize=8192, callback=None):
        path, text = command.split(' ', 1)[1], stream.read().decode()
        self.log.append(('STOR', path, text))
        self.files[path] = text

    def retrbinary(self, command, callback):
        path = command.split(' ', 1)[1]
        if path not in self.files:
            raise ftplib.error_perm('550 No such file')
        callback(self.files[path].encode())

    def retrlines(self, command, callback):
        folder = command.split(' ', 1)[1]
        if folder in self.folders:
            for name, size in self.folders[folder].items():
                callback(f'-rw-r--r-- 1 root root {size} Nov 9 12:00 {name}')
            return
        if folder == devices.CACHE_DIR:
            raise ftplib.error_perm('550 No such folder')
        for name in self.waiting:
            callback(f'-rw-r--r-- 1 root root 2 Nov 9 12:00 {name}')

    def sendcmd(self, command):
        self.log.append(tuple(command.split(' ', 1)))
        return '226 Done'

    def quit(self):
        pass

    def close(self):
        pass


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        FakeFTP.log, FakeFTP.waiting, FakeFTP.files, FakeFTP.folders = [], set(), {}, {}
        patcher = patch.object(devices.ftplib, 'FTP', FakeFTP)
        patcher.start()
        self.addCleanup(patcher.stop)
        sleep = patch.object(devices.time, 'sleep', lambda seconds: None)
        sleep.start()
        self.addCleanup(sleep.stop)

    def stored(self):
        return [entry for entry in FakeFTP.log if entry[0] == 'STOR']

    def test_make_worker_writes_config_then_restarts_the_app(self):
        self.assertTrue(devices.make_worker('192.168.50.21'))
        self.assertEqual(self.stored(), [
            ('STOR', '/data/PS5LM/fleet-worker.txt', '192.168.50.21:30052\nI_UNDERSTAND_RPC_IS_UNAUTHENTICATED\n'),
            ('STOR', '/data/PS5LM/unload', '-\n')])
        deleted = [entry[1] for entry in FakeFTP.log if entry[0] == 'DELE']
        self.assertIn('/data/PS5LM/app-args.txt', deleted)
        self.assertIn('/data/PS5LM/leave-worker', deleted)    # a stale one would end worker mode at once

    def test_a_busy_worker_is_still_a_worker(self):
        probes = {30052: 'silent', 2121: 'open'}
        with patch.object(devices, 'port_probe', lambda address, port, timeout=0.6: probes.get(port, 'refused')):
            # Serving a split: its one-client RPC server leaves new connections unanswered.
            self.assertEqual(devices.ps5_mode('192.168.50.21'), dict(mode='worker', busy=True))
            probes[30052] = 'refused'                     # no worker listening: the app
            with patch.object(devices, 'server_info', return_value=dict(state='down')):
                self.assertEqual(devices.ps5_mode('192.168.50.21'), dict(mode='idle'))
            probes.update({30052: 'silent', 2121: 'silent'})   # nothing answers: off or in rest mode
            self.assertEqual(devices.ps5_mode('192.168.50.21'), dict(mode='offline'))

    def test_port_probe(self):
        import socket
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            self.assertEqual(devices.port_probe('127.0.0.1', port), 'open')
        self.assertEqual(devices.port_probe('127.0.0.1', port), 'refused')

    def test_request_consumed_reads_the_folder(self):
        with devices.console_ftp('192.168.50.21') as ftp:
            self.assertTrue(devices.request_consumed(ftp, 'unload', 5))
            FakeFTP.waiting = {'unload'}
            with patch.object(devices.time, 'monotonic', side_effect=[0, 0, 10]):
                self.assertFalse(devices.request_consumed(ftp, 'unload', 5))

    def test_make_worker_with_the_app_closed_leaves_no_request_behind(self):
        with patch.object(devices, 'request_consumed', return_value=False):
            self.assertFalse(devices.make_worker('192.168.50.21'))
        self.assertEqual(FakeFTP.log[-1], ('DELE', '/data/PS5LM/unload'))

    def test_worker_settings_are_set_in_env_txt_and_the_rest_kept(self):
        text = 'RADV_DEBUG=nocache\nPS5LM_WORKER_SCREEN=on\nnot a setting\n'
        merged = devices.merged_env(text, devices.worker_env(screen=False, cache_gib=32))
        self.assertEqual(merged, 'RADV_DEBUG=nocache\nnot a setting\nPS5LM_WORKER_SCREEN=off\nPS5LM_RPC_CACHE_GB=32\n')
        self.assertEqual(devices.merged_env(merged, devices.worker_env(False, 32)), merged)
        self.assertEqual(devices.merged_env('', devices.worker_env(True, 0)),
                         'PS5LM_WORKER_SCREEN=on\nPS5LM_RPC_CACHE_GB=0\n')

    def test_make_worker_sets_its_settings_first(self):
        FakeFTP.files[devices.ENV_FILE] = 'GGML_VK_X=1\n'
        self.assertTrue(devices.make_worker('192.168.50.21', env=devices.worker_env(False, 16)))
        self.assertEqual([p for _, p, _ in self.stored()], [devices.ENV_FILE, devices.WORKER_FILE, '/data/PS5LM/unload'])
        self.assertEqual(FakeFTP.files[devices.ENV_FILE],
                         'GGML_VK_X=1\nPS5LM_WORKER_SCREEN=off\nPS5LM_RPC_CACHE_GB=16\n')

    def test_restart_with_settings(self):
        env = devices.worker_env(True, 0)
        self.assertTrue(devices.restart('192.168.50.21', env))
        self.assertEqual([p for _, p, _ in self.stored()], [devices.ENV_FILE, '/data/PS5LM/unload'])
        FakeFTP.log = []
        with patch.object(devices, 'request_consumed', return_value=False):   # closed, or an older worker
            self.assertFalse(devices.restart('192.168.50.21', env))
        self.assertEqual([p for _, p, _ in self.stored()], ['/data/PS5LM/unload'])   # env.txt was already so
        self.assertEqual(FakeFTP.log[-1], ('DELE', '/data/PS5LM/unload'))

    def test_launch_sends_the_payload_to_the_elf_loader(self):
        import socket
        with tempfile.TemporaryDirectory() as folder:
            launcher = Path(folder) / 'launch-PPSA99575.elf'
            launcher.write_bytes(b'\x7fELF payload')
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                listener.listen(1)
                port = listener.getsockname()[1]
                received = []

                def accept():
                    # Like elfldr: one connection after another (launch_app probes
                    # the port first, then sends the payload on a new connection).
                    while not received or not received[-1]:
                        conn, _ = listener.accept()
                        with conn:
                            chunks = []
                            while chunk := conn.recv(1 << 16):
                                chunks.append(chunk)
                            received.append(b''.join(chunks))
                server = threading.Thread(target=accept)
                server.start()
                with patch.object(devices, 'ELF_LOADER_PORT', port):
                    self.assertTrue(devices.launch_app('127.0.0.1', launcher))
                server.join(5)
            self.assertEqual(received[-1], b'\x7fELF payload')
            self.assertFalse(devices.launch_app('127.0.0.1', Path(folder) / 'missing.elf'))
        with patch.object(devices, 'port_probe', return_value='refused'):
            self.assertFalse(devices.launch_app('192.168.50.21', launcher))   # no loader on the console

    def test_the_weight_cache_is_listed_and_cleared(self):
        self.assertIsNone(devices.cache_use('192.168.50.21'))       # no folder: no cache yet
        FakeFTP.folders[devices.CACHE_DIR] = {'00000000000000a1': 3 << 20, '00000000000000a2': 1 << 20,
                                               '00000000000000a3.00000000000000ff.part': 5, 'notes.txt': 9}
        self.assertEqual(devices.cache_use('192.168.50.21'), dict(files=3, mib=4))
        self.assertEqual(devices.clear_cache('192.168.50.21'), dict(files=3, mib=4))
        deleted = sorted(entry[1] for entry in FakeFTP.log if entry[0] == 'DELE')
        self.assertEqual(deleted, [f'{devices.CACHE_DIR}/{name}' for name in
                                   ('00000000000000a1', '00000000000000a2', '00000000000000a3.00000000000000ff.part')])

    def test_leave_worker_restarts_as_the_app(self):
        self.assertEqual(devices.leave_worker('192.168.50.21'), 'restarted')
        self.assertEqual(self.stored(), [('STOR', '/data/PS5LM/leave-worker', '-\n')])

    def test_leave_worker_on_an_older_build_quits(self):
        with patch.object(devices, 'request_consumed', return_value=False):
            self.assertEqual(devices.leave_worker('192.168.50.21'), 'quit')
        self.assertEqual(FakeFTP.log[-3:], [('DELE', '/data/PS5LM/leave-worker'),
                                            ('DELE', '/data/PS5LM/fleet-worker.txt'),
                                            ('STOR', '/data/PS5LM/quit', 'quit\n')])

    def test_serving_a_model(self):
        args = devices.replica_args('/data/PS5LM/models/m.gguf', 8192, 2, 'q8_0', True)
        self.assertTrue(devices.serve_model('192.168.50.21', args))
        stored = dict((path, text) for _, path, text in self.stored())
        self.assertEqual(stored['/data/PS5LM/app-args.txt'].splitlines(), args)
        self.assertIn('/data/PS5LM/unload', stored)
        line = ' '.join(args)
        for part in ('-c 16384', '-np 2', '-fa on', '-ctk q8_0 -ctv q8_0', '--host 0.0.0.0 --port 8081'):
            self.assertIn(part, line)
        self.assertIn('-kvu', devices.replica_args('/m.gguf', 8192, 2, 'q8_0', True, unified=True))
        self.assertIn('-c 8192', ' '.join(devices.replica_args('/m.gguf', 8192, 2, 'q8_0', True, unified=True)))

    def test_a_worker_serves_a_model_by_leaving_worker_mode(self):
        self.assertTrue(devices.serve_model('192.168.50.21', ['-m', 'x'], worker=True))
        self.assertEqual([p for _, p, _ in self.stored()], ['/data/PS5LM/app-args.txt', '/data/PS5LM/leave-worker'])


class FleetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.fleet = control.Fleet(self.dir / 'control.json', binary_dir=self.dir / 'no-build', watch=False)

    def test_plan_from_budgets_without_a_build(self):
        model = write_gguf(self.dir / 'tiny.gguf', layers=4)
        self.fleet.add_device(ps5(memory_mib=64, reserve_mib=1))
        self.fleet.add_device(gpu(memory_mib=64, reserve_mib=1))
        self.fleet.add_pool(pool(model=str(model), context=256, cache_type='f16', flash_attn=False,
                                 replicas=[dict(devices=['ps5-a', 'pc-gpu'])]))
        replica = self.fleet.replicas[0]
        result = self.fleet._plan_replica(replica, self.fleet.pool('qwen'), model)
        self.assertIn('ps5-a blocks', result)
        self.assertEqual([n['id'] for n in replica.placement['nodes']], ['ps5-a', 'pc-gpu'])
        self.assertTrue(replica.placement['nodes'][-1]['output_layer'])
        self.fleet.update_pool('qwen', dict(replicas=[dict(devices=['ps5-a', 'pc-gpu'], split=[3, 1])]))
        replica = self.fleet.replicas[0]
        self.fleet._plan_replica(replica, self.fleet.pool('qwen'), model)
        self.assertEqual([n['layer_end'] - n['layer_start'] for n in replica.placement['nodes']], [3, 1])

    def test_unbudgeted_devices_need_to_be_up_to_plan(self):
        model = write_gguf(self.dir / 'tiny.gguf', layers=4)
        self.fleet.apply(dict(devices=[ps5(), gpu()], pools=[pool(model=str(model), context=256,
                                                                  replicas=[dict(devices=['ps5-a', 'pc-gpu'])])]))
        with self.assertRaisesRegex(control.ControlError, 'memory_mib'):
            self.fleet._plan_replica(self.fleet.replicas[0], self.fleet.pool('qwen'), model)

    def test_replicas_survive_changes_that_do_not_touch_them(self):
        self.fleet.apply(dict(devices=[ps5(), gpu()], pools=[pool(replicas=[dict(devices=['ps5-a']),
                                                                            dict(devices=['pc-gpu'])])]))
        first, second = self.fleet.replicas
        self.assertEqual((first.key, second.key, first.via, second.via), ('qwen/0', 'qwen/1', 'app', 'direct'))
        self.fleet.update_pool('qwen', dict(running=True, aliases=['q']))
        self.assertEqual(self.fleet.replicas, [first, second])
        self.fleet.update_pool('qwen', dict(replicas=[dict(devices=['pc-gpu'])]))
        self.assertEqual(self.fleet.replicas, [second])
        self.assertEqual(second.key, 'qwen/0')
        self.fleet.update_pool('qwen', dict(context=65536))       # what it serves changed: start over
        self.assertIsNot(self.fleet.replicas[0], second)
        self.assertEqual(json.loads(self.fleet.config_path.read_text())['pools'][0]['context'], 65536)

    def test_auto_adds_replicas_from_free_devices(self):
        self.fleet.apply(dict(devices=[ps5('a', '192.168.50.1'), ps5('b', '192.168.50.2'), gpu()]))
        self.fleet.add_pool(dict(pool(), auto=dict(count=2, kinds=['ps5'])))
        self.assertEqual([r['devices'] for r in self.fleet.pool('qwen')['replicas']], [['a'], ['b']])
        with self.assertRaisesRegex(control.ControlError, 'free devices'):
            self.fleet.update_pool('qwen', dict(auto=dict(count=1, kinds=['ps5'])))

    def test_a_running_pool_keeps_its_devices(self):
        self.fleet.apply(dict(devices=[ps5()], pools=[pool(running=True, replicas=[dict(devices=['ps5-a'])])]))
        with self.assertRaisesRegex(control.ControlError, 'stop the pool first'):
            self.fleet.device_action('ps5-a', 'leave')
        with self.assertRaisesRegex(control.ControlError, 'take it out of the pool'):
            self.fleet.remove_device('ps5-a')
        with self.assertRaisesRegex(control.ControlError, 'takes worker, leave'):
            self.fleet.device_action('ps5-a', 'stop')

    def test_a_console_already_serving_the_model_is_taken_over(self):
        model = self.dir / 'm.gguf'
        model.write_bytes(b'gguf')
        self.fleet.apply(dict(devices=[ps5()], pools=[pool(model=str(model), running=True,
                                                           replicas=[dict(devices=['ps5-a'])])]))
        (replica,) = self.fleet.replicas
        settings = self.fleet.pool('qwen')
        args = devices.replica_args('/data/PS5LM/models/m.gguf', settings['context'], settings['slots'],
                                    settings['cache_type'], settings['flash_attn'], settings['shared_context'])
        serving = dict(mode='serving', model='/data/PS5LM/models/m.gguf')
        with patch.object(devices, 'ps5_mode', return_value=serving), \
                patch.object(devices, 'read_text', return_value='\n'.join(args) + '\n'), \
                patch.object(devices, 'serve_model') as serve, patch.object(devices, 'push_file') as push:
            self.fleet._start_replica(replica, settings)
        self.assertEqual((replica.state, replica.url), ('starting', 'http://192.168.50.21:8081'))
        serve.assert_not_called()
        push.assert_not_called()
        # Other settings on the console: load the pool's.
        with patch.object(devices, 'ps5_mode', return_value=serving), \
                patch.object(devices, 'read_text', return_value='-m\n/data/PS5LM/models/m.gguf\n-c\n4096\n'), \
                patch.object(devices, 'console_has', return_value=True), \
                patch.object(devices, 'serve_model', return_value=True) as serve:
            self.fleet._start_replica(replica, settings)
        serve.assert_called_once_with('192.168.50.21', args, worker=False)

    def test_stopping_a_console_replica_unloads_its_model(self):
        self.fleet.apply(dict(devices=[ps5()], pools=[pool(running=True, replicas=[dict(devices=['ps5-a'])])]))
        (replica,) = self.fleet.replicas
        replica.state = 'ready'
        with patch.object(devices, 'stop_serving') as unload:
            self.fleet.update_pool('qwen', dict(running=False))
            self.fleet._reconcile(control.concurrent.futures.ThreadPoolExecutor(2))
            for _ in range(250):
                if replica.state == 'stopped':
                    break
                time.sleep(0.02)
        unload.assert_called_once_with('192.168.50.21')
        self.assertEqual(replica.state, 'stopped')

    def test_a_start_under_way_is_not_failed_by_the_watcher(self):
        # A console replica copies its model before it sets a load deadline; the
        # watcher must not read the missing deadline as a passed one.
        self.fleet.apply(dict(devices=[ps5()], pools=[pool(running=True, replicas=[dict(devices=['ps5-a'])])]))
        (replica,) = self.fleet.replicas
        idle = {'ps5-a': dict(mode='idle', checked=time.time() + 1)}
        copying = threading.Event()
        with patch.object(self.fleet, '_start_replica', lambda r, p: copying.wait(5)):
            self.fleet._launch(replica, self.fleet.pool('qwen'))
            self.fleet._check(replica, idle)
            self.assertEqual(replica.state, 'starting')
            copying.set()
            replica.thread.join(5)
        self.fleet._check(replica, idle)
        self.assertEqual(replica.state, 'starting')
        replica.deadline = time.time() - 1
        self.fleet._check(replica, idle)
        self.assertEqual(replica.state, 'failed')

    def finished(self):
        """Wait for the device jobs started so far."""
        for _ in range(250):
            if all(job['state'] != 'running' for job in self.fleet.jobs):
                return list(self.fleet.jobs)
            threading.Event().wait(0.02)
        raise AssertionError(f'jobs still running: {list(self.fleet.jobs)}')

    def test_new_settings_restart_an_idle_worker(self):
        self.fleet.add_device(ps5())
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='worker')), \
                patch.object(devices, 'restart', return_value=True) as restart, \
                patch.object(self.fleet, '_wait_port', return_value=True), patch.object(control.time, 'sleep'):
            self.fleet.update_device('ps5-a', dict(screen=True))
            (job,) = self.finished()
        self.assertEqual(job['state'], 'done', job)
        restart.assert_called_once_with('192.168.50.21', devices.worker_env(True, 32))
        self.assertNotIn('ps5-a', self.fleet.env_pending)

    def test_new_settings_wait_for_a_worker_in_use(self):
        self.fleet.apply(dict(devices=[ps5()], pools=[pool(running=True, replicas=[dict(devices=['ps5-a'])])]))
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='worker', busy=True)), \
                patch.object(devices, 'write_env', return_value=True) as write, \
                patch.object(devices, 'restart') as restart:
            self.fleet.update_device('ps5-a', dict(cache_gib=64))
            self.finished()
            self.fleet.update_device('ps5-a', dict(enabled=True))     # not a worker setting: nothing to do
            self.assertEqual(len(self.finished()), 1)
        restart.assert_not_called()
        write.assert_called_once_with('192.168.50.21', devices.worker_env(False, 64))
        (view,) = self.fleet.snapshot()['devices']
        self.assertTrue(view['settings_pending'])

    def test_a_split_starts_on_a_freshly_started_worker(self):
        # The console's driver keeps memory an earlier host allocated: every split
        # restarts its consoles' workers first, with their settings.
        self.fleet.apply(dict(devices=[ps5(), gpu()], pools=[pool(replicas=[dict(devices=['ps5-a', 'pc-gpu'])])]))
        (replica,) = self.fleet.replicas
        device = self.fleet.device('ps5-a')
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='worker')), \
                patch.object(devices, 'restart', return_value=True) as restart, \
                patch.object(self.fleet, '_wait_port', return_value=True), patch.object(control.time, 'sleep'):
            self.fleet._ensure_worker(device, replica)
        restart.assert_called_once_with('192.168.50.21', devices.worker_env(False, 32))
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='worker', busy=True)), \
                patch.object(devices, 'restart') as restart:
            self.fleet._ensure_worker(device, replica)                 # in use by someone: left alone
        restart.assert_not_called()

    def test_a_starting_split_is_not_probed(self):
        # Probing a console's one-client RPC server while a split connects to it
        # several times in a row could take the slot llama-server needs.
        self.fleet.apply(dict(devices=[ps5(), gpu()], pools=[pool(replicas=[dict(devices=['ps5-a', 'pc-gpu'])])]))
        (replica,) = self.fleet.replicas
        self.fleet.status['ps5-a'] = dict(mode='worker', checked=0)
        replica.state = 'starting'
        with patch.object(devices, 'ps5_mode') as mode:
            self.fleet._probe(self.fleet.device('ps5-a'))
        mode.assert_not_called()
        self.assertEqual(self.fleet.status['ps5-a']['mode'], 'worker')
        replica.state = 'ready'
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='worker', busy=True)) as mode, \
                patch.object(devices, 'cache_use', return_value=None):
            self.fleet._probe(self.fleet.device('ps5-a'))
        mode.assert_called_once()

    def test_a_closed_console_is_opened_through_its_elf_loader(self):
        self.fleet.add_device(ps5())
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='idle')), \
                patch.object(devices, 'launch_app', return_value=True) as launch:
            self.fleet.device_action('ps5-a', 'launch')
            (job,) = self.finished()
        self.assertEqual(job['result'], 'asked the console to open PS5LM')
        launch.assert_called_once_with('192.168.50.21', control.DEFAULT_LAUNCHER)
        # Without a loader it says so, instead of claiming it opened.
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='idle')), \
                patch.object(devices, 'launch_app', return_value=False):
            self.fleet.device_action('ps5-a', 'launch')
            job = self.finished()[-1]
        self.assertEqual(job['state'], 'failed')
        self.assertIn('no ELF loader on port 9021', job['result'])

    def test_a_split_reopens_a_crashed_console_as_a_worker(self):
        # PS5LM gone (ftpsrv still answers): fleet-worker.txt is written, then the
        # console is opened through its ELF loader and comes back as a worker.
        self.fleet.apply(dict(devices=[ps5(), gpu()], pools=[pool(replicas=[dict(devices=['ps5-a', 'pc-gpu'])])],
                              launcher='/opt/launch.elf'))
        (replica,) = self.fleet.replicas
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='idle')), \
                patch.object(devices, 'make_worker', return_value=False) as make, \
                patch.object(devices, 'launch_app', return_value=True) as launch, \
                patch.object(self.fleet, '_wait_port', return_value=True):
            self.fleet._ensure_worker(self.fleet.device('ps5-a'), replica)
        make.assert_called_once()
        launch.assert_called_once_with('192.168.50.21', Path('/opt/launch.elf'))

    def test_an_older_worker_is_restarted_by_leaving_worker_mode(self):
        self.fleet.add_device(ps5())
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='worker')), \
                patch.object(devices, 'restart', return_value=False), \
                patch.object(devices, 'leave_worker', return_value='restarted') as leave, \
                patch.object(devices, 'make_worker', return_value=True) as make, \
                patch.object(self.fleet, '_wait_port', return_value=True), patch.object(control.time, 'sleep'):
            answer = self.fleet.device_action('ps5-a', 'restart')
            (job,) = self.finished()
        self.assertEqual((job['id'], job['state']), (answer['job']['id'], 'done'))
        leave.assert_called_once_with('192.168.50.21')
        make.assert_called_once_with('192.168.50.21', 30052, devices.worker_env(False, 32))

    def test_clear_cache(self):
        self.fleet.add_device(ps5())
        with patch.object(devices, 'ps5_mode', return_value=dict(mode='worker')), \
                patch.object(devices, 'clear_cache', return_value=dict(files=280, mib=10240)):
            self.fleet.device_action('ps5-a', 'clear-cache')
            (job,) = self.finished()
        self.assertEqual(job['result'], 'deleted 280 cached tensors, 10.0 GiB')
        (view,) = self.fleet.snapshot()['devices']
        self.assertEqual(view['cache'], dict(files=0, mib=0))

    def test_pools_share_devices_but_run_one_at_a_time(self):
        self.fleet.apply(dict(devices=[ps5(), gpu()], pools=[
            pool('big', replicas=[dict(devices=['ps5-a', 'pc-gpu'])]),
            pool('small', model='/m/small.gguf', replicas=[dict(devices=['ps5-a']), dict(devices=['pc-gpu'])])]))
        self.fleet.update_pool('big', dict(running=True))
        with self.assertRaisesRegex(control.ControlError, 'stop big first'):
            self.fleet.update_pool('small', dict(running=True))
        # Taking over: big still holds its devices while it stops, so small waits for them.
        big, small_a, small_b = self.fleet.replicas
        big.state = 'stopping'
        self.fleet.update_pool('big', dict(running=False))
        self.fleet.update_pool('small', dict(running=True))
        launched = []
        with patch.object(self.fleet, '_launch', lambda replica, pool: launched.append(replica.key)), \
                patch.object(self.fleet, '_stop_soon', lambda replica, reason: None):
            self.fleet._reconcile(control.concurrent.futures.ThreadPoolExecutor(2))
            self.assertEqual(launched, [])
            big.state = 'stopped'
            self.fleet._reconcile(control.concurrent.futures.ThreadPoolExecutor(2))
        self.assertEqual(launched, ['small/0', 'small/1'])

    def test_routing(self):
        self.fleet.apply(dict(devices=[], pools=[
            pool('qwen', model='/m/Qwen-27B.gguf', aliases=['gpt-4o'], replicas=[
                dict(url='http://10.0.0.1:8080'), dict(url='http://10.0.0.2:8080', slots=4)]),
            pool('small', model='/m/small.gguf', replicas=[dict(url='http://10.0.0.3:8080')])]))
        one, two, small = self.fleet.replicas
        with self.assertRaises(control.router.NoRoute) as caught:
            self.fleet.acquire('qwen')
        self.assertEqual(caught.exception.status, 503)
        one.state = two.state = 'ready'
        for name in ('qwen', 'GPT-4o', 'Qwen-27B.gguf', 'qwen-27b', 'anything'):   # the only ready pool takes the rest
            self.assertIn(self.fleet.acquire(name), (one, two))
        small.state = 'ready'
        with self.assertRaises(control.router.NoRoute) as caught:
            self.fleet.acquire('anything')
        self.assertEqual(caught.exception.status, 404)
        self.assertIs(self.fleet.acquire('small'), small)
        # The least busy for its slots: two (4 slots) before one (1 slot).
        for replica in (one, two):
            replica.in_flight = 0
        picks = [self.fleet.acquire('qwen') for _ in range(5)]
        self.assertEqual(picks.count(two), 4)
        with self.assertRaises(control.router.NoRoute):
            self.fleet.acquire('qwen', exclude={one.key, two.key})

    def test_a_server_replica_is_ready_when_it_answers(self):
        backend = http.server.ThreadingHTTPServer(('127.0.0.1', 0), HealthHandler)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        self.addCleanup(backend.server_close)
        self.addCleanup(backend.shutdown)
        url = f'http://127.0.0.1:{backend.server_address[1]}'
        self.fleet.apply(dict(pools=[pool(running=True, replicas=[dict(url=url)])]))
        (replica,) = self.fleet.replicas
        self.fleet._launch(replica, self.fleet.pool('qwen'))
        replica.thread.join(5)
        self.assertEqual(replica.state, 'starting')
        self.fleet._check(replica, {})
        self.assertEqual(replica.state, 'ready')
        self.fleet.unreachable(replica, 'refused')
        self.assertEqual(replica.state, 'starting')
        backend.shutdown()
        backend.server_close()
        replica.state = 'ready'
        self.fleet._check(replica, {})
        self.assertEqual(replica.state, 'failed')
        self.assertGreater(replica.retry_at, 0)


class HealthHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200 if self.path == '/health' else 404)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class PageTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fleet = control.Fleet(Path(tmp.name) / 'control.json', binary_dir=Path(tmp.name), watch=False)
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), control.handler_for(self.fleet, 0))
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = control.handler_for(self.fleet, self.port)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f'http://127.0.0.1:{self.port}'

    def request(self, method, path, body=None, host=None, header=True):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        headers = {'Host': host or f'127.0.0.1:{self.port}', 'Content-Type': 'application/json'}
        if header:
            headers['X-Fleet-Control'] = '1'
        connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, (json.loads(data) if response.getheader('Content-Type', '').startswith('application/json') else data)

    def ctl(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ctl.main(['--url', self.url, *argv])
        return code, json.loads(out.getvalue())

    def test_page_and_state(self):
        status, page = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'PS5LM Fleet', page)
        status, state = self.request('GET', '/api/state')
        self.assertEqual(status, 200)
        self.assertEqual((state['devices'], state['pools'], state['api']['listening']), ([], [], False))

    def test_foreign_host_and_missing_header_are_refused(self):
        self.assertEqual(self.request('GET', '/api/state', host='evil.example')[0], 403)
        self.assertEqual(self.request('POST', '/api/settings', {'api_port': 9000}, header=False)[0], 403)
        self.assertEqual(self.fleet.config['api_port'], 8081)

    def test_changes_are_validated_and_saved(self):
        status, reply = self.request('POST', '/api/settings', {'api_port': 9000})
        self.assertEqual(status, 200)
        self.assertEqual(reply['state']['settings']['api_port'], 9000)
        self.assertEqual(json.loads(self.fleet.config_path.read_text())['api_port'], 9000)
        status, reply = self.request('POST', '/api/devices/add', ps5(port=45000))
        self.assertEqual(status, 400)
        self.assertIn('39999', reply['error'])

    def test_api_key_is_never_shown_or_overwritten_with_the_mask(self):
        self.request('POST', '/api/settings', {'api_key': 'secret-key'})
        status, state = self.request('GET', '/api/state')
        self.assertEqual(state['settings']['api_key'], 'set')
        self.request('POST', '/api/settings', {'api_key': 'set', 'api_port': 9001})
        self.assertEqual(self.fleet.config['api_key'], 'secret-key')
        status, exported = self.request('GET', '/api/config')
        self.assertEqual(exported['api_key'], 'set')
        status, reply = self.request('POST', '/api/apply', {'config': exported})
        self.assertEqual(status, 200)
        self.assertEqual(self.fleet.config['api_key'], 'secret-key')

    def test_ctl_drives_the_same_api(self):
        code, answer = self.ctl('device', 'add', 'ps5-a', '--kind', 'ps5', '--address', '192.168.50.21', '--tag', 'lounge')
        self.assertEqual((code, answer['ok']), (0, True))
        self.assertNotIn('state', answer)
        code, answer = self.ctl('device', 'add', 'pc-gpu', '--kind', 'gpu')
        code, answer = self.ctl('pool', 'add', 'qwen', '--model', '/m/q.gguf', '--context', '65536',
                                '--replica', 'ps5-a,pc-gpu', '--alias', 'q')
        self.assertEqual(code, 0, answer)
        code, answer = self.ctl('pools')
        (only,) = answer['pools']
        self.assertEqual((only['context'], only['cache_type'], only['replicas'][0]['devices'], only['replicas'][0]['via']),
                         (65536, 'q8_0', ['ps5-a', 'pc-gpu'], 'rpc'))
        code, answer = self.ctl('pool', 'set', 'qwen', 'context=131072', 'cache_type=q4_0', 'slots=2')
        self.assertEqual(code, 0, answer)
        self.assertEqual(self.fleet.pool('qwen')['context'], 131072)
        code, answer = self.ctl('device', 'set', 'ps5-a', 'memory_mib=12000', 'enabled=false')
        self.assertEqual((self.fleet.device('ps5-a')['memory_mib'], self.fleet.device('ps5-a')['enabled']), (12000, False))
        code, answer = self.ctl('pool', 'set', 'qwen', 'context=100')
        self.assertEqual(code, 1)
        self.assertIn('context', answer['error'])
        code, answer = self.ctl('device', 'stop', 'ps5-a')
        self.assertEqual(code, 1)
        code, answer = self.ctl('config')
        self.assertEqual(answer['pools'][0]['aliases'], ['q'])

    def test_ctl_says_when_the_server_is_not_there(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ctl.main(['--url', 'http://127.0.0.1:9', 'status'])
        self.assertEqual(code, 1)
        self.assertIn('cannot reach', json.loads(out.getvalue())['error'])

    def test_ctl_arguments(self):
        self.assertEqual(ctl.pairs(['context=8192', 'model=/a/b.gguf', 'sha256=0123', 'memory_mib=null',
                                    'tags=["a"]', 'cache_type=q8_0']),
                         dict(context=8192, model='/a/b.gguf', sha256='0123', memory_mib=None, tags=['a'],
                              cache_type='q8_0'))
        args = ctl.parser().parse_args(['pool', 'add', 'p', '--model', 'm', '--replica', 'a,b', '--replica', 'c:rpc',
                                        '--server', 'http://10.0.0.9:8080'])
        self.assertEqual(ctl.replicas(args), [dict(devices=['a', 'b']), dict(devices=['c'], via='rpc'),
                                              dict(url='http://10.0.0.9:8080')])


if __name__ == '__main__':
    unittest.main()
