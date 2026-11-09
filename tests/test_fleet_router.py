"""The fleet's API router: pool by "model", least-busy replica, streaming, keys, failover."""
import http.client
import http.server
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from fleet import control, router


def backend(name):
    """A stand-in llama-server that says who it is and echoes what it got."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def _json(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._json(dict(backend=name, path=self.path))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if not body.get('stream'):
                self._json(dict(backend=name, path=self.path, model=body.get('model'),
                                authorization=self.headers.get('Authorization'), api_key=self.headers.get('x-api-key')))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            for piece in ('Hel', 'lo', ' there'):
                event = b'data: ' + json.dumps({'choices': [{'delta': {'content': piece}}]}).encode() + b'\n\n'
                self.wfile.write(b'%x\r\n%s\r\n' % (len(event), event))
                self.wfile.flush()
                time.sleep(0.05)
            done = b'data: [DONE]\n\n'
            self.wfile.write(b'%x\r\n%s\r\n0\r\n\r\n' % (len(done), done))

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def closed_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class RouterTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.backends = {name: backend(name) for name in ('a1', 'a2', 'b1')}
        for server in self.backends.values():
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        url = {name: f'http://127.0.0.1:{s.server_address[1]}' for name, s in self.backends.items()}
        self.fleet = control.Fleet(Path(tmp.name) / 'control.json', binary_dir=Path(tmp.name), watch=False)
        self.fleet.apply(dict(pools=[
            dict(id='alpha', model='/m/Alpha-9B.gguf', aliases=['gpt-4o'],
                 replicas=[dict(url=url['a1']), dict(url=url['a2'])]),
            dict(id='beta', model='/m/beta.gguf', replicas=[dict(url=url['b1'], api_key='backend-key')])]))
        self.api = router.make_server(self.fleet, '127.0.0.1', 0)
        threading.Thread(target=self.api.serve_forever, daemon=True).start()
        self.addCleanup(self.api.server_close)
        self.addCleanup(self.api.shutdown)

    def ready(self, *keys):
        for replica in self.fleet.replicas:
            replica.state = 'ready' if replica.key in keys else 'stopped'

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.api.server_address[1], timeout=10)
        connection.request(method, path, body=None if body is None else json.dumps(body),
                           headers=dict({'Content-Type': 'application/json'}, **(headers or {})))
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response, data

    def test_requests_go_to_the_pool_their_model_names(self):
        self.ready('alpha/0', 'beta/0')
        for model, pool in (('alpha', 'alpha'), ('GPT-4o', 'alpha'), ('Alpha-9B', 'alpha'), ('beta', 'beta')):
            response, data = self.request('POST', '/v1/chat/completions', dict(model=model, messages=[]))
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader('X-Fleet-Pool'), pool)
            self.assertEqual(json.loads(data)['path'], '/v1/chat/completions')
        response, data = self.request('POST', '/v1/chat/completions', dict(model='nobody', messages=[]))
        self.assertEqual(response.status, 404)
        self.assertIn('alpha, beta', json.loads(data)['error']['message'])

    def test_replicas_share_the_requests(self):
        self.ready('alpha/0', 'alpha/1')
        seen = set()
        for _ in range(4):
            response, data = self.request('POST', '/v1/completions', dict(model='alpha', prompt='x'))
            seen.add(json.loads(data)['backend'])
        self.assertEqual(seen, {'a1', 'a2'})

    def test_streaming_passes_through_as_it_comes(self):
        self.ready('alpha/0')
        connection = http.client.HTTPConnection('127.0.0.1', self.api.server_address[1], timeout=10)
        connection.request('POST', '/v1/chat/completions', body=json.dumps(dict(model='alpha', stream=True)),
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        self.assertEqual(response.getheader('Content-Type'), 'text/event-stream')
        first = response.read1(65536)
        self.assertIn(b'Hel', first)                 # the first piece arrives before the last is sent
        rest = response.read()
        connection.close()
        text = ''.join(json.loads(line[6:])['choices'][0]['delta']['content']
                       for line in (first + rest).decode().split('\n\n') if line.startswith('data: {'))
        self.assertEqual(text, 'Hello there')
        self.assertIn(b'[DONE]', first + rest)

    def test_api_key(self):
        self.ready('beta/0')
        self.fleet.set_settings(dict(api_key='fleet-key'))
        response, _ = self.request('POST', '/v1/chat/completions', dict(model='beta'))
        self.assertEqual(response.status, 401)
        response, _ = self.request('GET', '/v1/models')
        self.assertEqual(response.status, 401)
        response, data = self.request('POST', '/v1/chat/completions', dict(model='beta'),
                                      {'Authorization': 'Bearer fleet-key'})
        self.assertEqual(response.status, 200)
        # The fleet's key stays here; the replica gets its own.
        self.assertEqual(json.loads(data)['authorization'], 'Bearer backend-key')
        response, data = self.request('POST', '/v1/messages', dict(model='beta'), {'x-api-key': 'fleet-key'})
        self.assertEqual(response.status, 200)
        self.assertIsNone(json.loads(data)['api_key'])
        response, _ = self.request('GET', '/health')
        self.assertEqual(response.status, 200)

    def test_models_and_health(self):
        response, _ = self.request('GET', '/health')
        self.assertEqual(response.status, 503)
        self.ready('alpha/1')
        response, _ = self.request('GET', '/health')
        self.assertEqual(response.status, 200)
        response, data = self.request('GET', '/v1/models')
        models = {m['id']: m for m in json.loads(data)['data']}
        self.assertEqual((models['alpha']['ready_replicas'], models['beta']['ready_replicas']), (1, 0))
        self.assertEqual(models['alpha']['aliases'], ['gpt-4o'])

    def test_other_requests_go_to_the_first_ready_pool(self):
        self.ready('beta/0')
        response, data = self.request('GET', '/props')
        self.assertEqual(json.loads(data), dict(backend='b1', path='/props'))
        response, data = self.request('GET', '/props?model=alpha')
        self.assertEqual(response.status, 503)

    def test_a_replica_that_refuses_is_skipped(self):
        self.fleet.update_pool('alpha', dict(replicas=[dict(url=f'http://127.0.0.1:{closed_port()}'),
                                                       dict(url=self.fleet.pool('alpha')['replicas'][1]['url'])]))
        self.ready('alpha/0', 'alpha/1')
        for _ in range(2):
            response, data = self.request('POST', '/v1/chat/completions', dict(model='alpha'))
            self.assertEqual((response.status, json.loads(data)['backend']), (200, 'a2'))
        dead = self.fleet.replicas[0]
        self.assertEqual(dead.state, 'starting')
        self.assertIn('not answering', dead.detail)
        # Released once the reply is written, a moment after the client has it.
        for _ in range(50):
            if sum(r.in_flight for r in self.fleet.replicas) == 0:
                break
            time.sleep(0.02)
        self.assertEqual(sum(r.in_flight for r in self.fleet.replicas), 0)

    def test_nothing_ready(self):
        response, data = self.request('POST', '/v1/chat/completions', dict(model='alpha'))
        self.assertEqual(response.status, 503)
        self.assertIn('No replica of alpha', json.loads(data)['error']['message'])


if __name__ == '__main__':
    unittest.main()
