"""One API in front of the fleet: OpenAI- and Anthropic-style requests go to a
ready replica of the pool their "model" names.

    POST /v1/chat/completions, /v1/completions, /v1/messages, /v1/embeddings,
         /completion, /tokenize, ... (anything llama-server takes)
    GET  /v1/models   the pools, with how many replicas are ready
    GET  /health      200 while any pool can take a request
    GET  anything else, such as / (llama-server's chat page), goes to the first
         pool with a ready replica, or the one ?model= names

The pool is the request's "model": a pool's id, one of its aliases or its model
file's name. A name no pool has goes to the only pool that is ready, when
exactly one is. Within a pool the replica with the fewest requests in flight
for its slots gets the request; llama-server queues what its slots cannot
take yet. A replica that refuses the connection is skipped and the request
tried on another one (nothing has been sent back by then). Responses stream
through as they arrive, so "stream": true works end to end.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import contextlib
import hmac
import http.client
import http.server
import json
import socket
import urllib.parse

# Hop-by-hop headers, and the ones the router sets itself.
_SKIP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailer', 'trailers',
         'transfer-encoding', 'upgrade', 'host', 'content-length', 'authorization', 'x-api-key'}
MAX_BODY = 256 * 1024 * 1024
# A reply without streaming sends nothing until it ends: a 128k prompt on a split
# reads at about 100 tokens/s, over 20 minutes before the first token.
READ_TIMEOUT = 3600
ATTEMPTS = 3


class NoRoute(Exception):
    """No pool or replica for a request: (HTTP status, message)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class _Unreachable(Exception):
    pass


def handler_for(fleet):
    """fleet: acquire(model, exclude) -> replica (raises NoRoute), release(replica),
    unreachable(replica, reason), models() -> list, healthy() -> bool, api_key() -> str."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = 'ps5lm-fleet'

        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            # Streamed tokens are small writes: send each at once.
            with contextlib.suppress(OSError):
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def _json(self, code, body):
            data = json.dumps(body).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(data)

        def _error(self, code, message, kind):
            self._json(code, {'error': {'message': message, 'type': kind, 'code': code}})

        def _authorized(self):
            key = fleet.api_key()
            if not key:
                return True
            given = self.headers.get('x-api-key', '')
            bearer = self.headers.get('Authorization', '')
            if not given and bearer[:7].lower() == 'bearer ':
                given = bearer[7:].strip()
            if hmac.compare_digest(given.encode('utf-8'), key.encode('utf-8')):
                return True
            self._error(401, 'Invalid API key', 'authentication_error')
            return False

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path in ('/health', '/v1/health'):
                healthy = fleet.healthy()
                self._json(200 if healthy else 503, {'status': 'ok' if healthy else 'no pool is ready'})
                return
            if not self._authorized():
                return
            if path in ('/v1/models', '/models'):
                self._json(200, {'object': 'list', 'data': fleet.models()})
                return
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            self._forward(None, (query.get('model') or [None])[0])

        def do_POST(self):
            if not self._authorized():
                return
            if 'chunked' in self.headers.get('Transfer-Encoding', '').lower():
                self._error(411, 'Send the request with a Content-Length', 'invalid_request_error')
                self.close_connection = True
                return
            try:
                length = int(self.headers.get('Content-Length') or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                self._error(413, 'Request body too large', 'invalid_request_error')
                self.close_connection = True
                return
            body = self.rfile.read(length)
            model = None
            if body[:1] in (b'{', b' ', b'\n', b'\t', b'\r'):
                try:
                    data = json.loads(body)
                    model = data.get('model') if isinstance(data, dict) else None
                except ValueError:
                    pass
            self._forward(body, model if isinstance(model, str) else None)

        def _forward(self, body, model):
            tried = set()
            for _ in range(ATTEMPTS):
                try:
                    replica = fleet.acquire(model, tried)
                except NoRoute as exc:
                    if tried:
                        self._error(502, f'No replica answered: {exc}', 'api_error')
                    else:
                        self._error(exc.status, str(exc), 'not_found_error' if exc.status == 404 else 'overloaded_error')
                    return
                tried.add(replica.key)
                try:
                    self._relay(replica, body)
                    return
                except _Unreachable as exc:
                    fleet.unreachable(replica, str(exc))
                finally:
                    fleet.release(replica)
            self._error(502, 'No replica answered', 'api_error')

        def _relay(self, replica, body):
            base = urllib.parse.urlsplit(replica.url)
            secure = base.scheme == 'https'
            connection = (http.client.HTTPSConnection if secure else http.client.HTTPConnection)(
                base.hostname, base.port or (443 if secure else 80), timeout=READ_TIMEOUT)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _SKIP}
            if replica.api_key:
                headers['Authorization'] = f'Bearer {replica.api_key}'
            if body is not None:
                headers['Content-Length'] = str(len(body))
            try:
                connection.request(self.command, base.path.rstrip('/') + self.path, body=body, headers=headers)
                response = connection.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                connection.close()
                raise _Unreachable(f'{replica.key}: {exc}') from exc
            try:
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in _SKIP:
                        self.send_header(key, value)
                self.send_header('X-Fleet-Pool', replica.pool_id)
                self.send_header('X-Fleet-Replica', replica.key)
                if self.command == 'HEAD' or response.status in (204, 304) or 100 <= response.status < 200:
                    self.end_headers()
                elif response.getheader('Content-Length') is not None:
                    data = response.read()
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    # Streaming (server-sent events): pass each piece on as it comes.
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.end_headers()
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            except (OSError, http.client.HTTPException):
                # The client went away, or the replica did mid-response; closing
                # the replica's connection makes llama-server stop generating.
                self.close_connection = True
            finally:
                connection.close()

    return Handler


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


def make_server(fleet, host, port):
    return Server((host, port), handler_for(fleet))
