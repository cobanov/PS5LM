"""Drive the fleet control server from a shell or an agent; every answer is JSON.

    python3 -m fleet ctl [--url URL] [--state] COMMAND ...

  status                       everything: devices, pools and replicas, API, jobs
  devices | pools | jobs       one part of it
  config                       the configuration, keys shown as "set"; edit it and `apply` it back
  log --replica KEY | --device ID
                               a replica's llama-server output, or a GPU's RPC worker's
  search [--wait]              find consoles on the network (PlayStation discovery)
  apply FILE|-                 replace the whole configuration (keys left as "set" are kept)

  device add ID --kind ps5|gpu|rpc [--address IP] [--port N] [--device CUDA0]
             [--memory-mib N] [--reserve-mib N] [--tag T]... [--disabled]
             [--screen] [--cache-gib N]
  device set ID KEY=VALUE...   e.g. memory_mib=null enabled=false tags='["rack1"]'
                               a console's worker: screen=true (default false: about
                               350 MiB more for the model), cache_gib=64 (default 32,
                               0 for none: the weights it keeps, so a reload is local)
  device rm ID
  device worker|leave|restart|launch|quit|unload|clear-cache ID [--wait]
                               a console: into worker mode, out of it (PS5LM restarts as
                               the app), restart it as it is (with its settings), open a
                               closed PS5LM (needs an ELF loader on the console's port
                               9021), quit PS5LM, unload its model, or delete its cached
                               weights
  device worker|stop ID        a local GPU: start or stop its RPC worker

  pool add ID --model PATH|URL [--sha256 HEX] [--context N] [--slots N] [--shared-context]
           [--cache-type q8_0] [--flash-attn on|off] [--alias NAME]...
           [--replica DEV[,DEV...][:via]]... [--server URL]...
           [--auto COUNT [--per K] [--kind KIND]... [--tag TAG]...] [--start]
  pool set ID KEY=VALUE...     e.g. context=131072 cache_type=q4_0 replicas='[...]'
  pool rm ID
  pool start ID [--wait SECONDS] | pool stop ID | pool plan ID [--wait]
  replica restart KEY          KEY is pool/index, as `pools` shows it

A replica is a layer split over its devices (:rpc), one console serving the
model with PS5LM (:app, the default for one console), or one local GPU (:direct,
the default for one GPU). --auto adds COUNT replicas of K free devices each.

The control server is --url, else $PS5LM_FLEET_URL, else http://127.0.0.1:8090.
Exit status: 0 done, 1 refused or failed (the JSON's "error" says why), 2 usage.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Fields that stay strings in KEY=VALUE, even when they would parse as JSON.
STRINGS = {'id', 'model', 'sha256', 'address', 'device', 'cache_type', 'api_key', 'lan', 'url'}


class Failed(Exception):
    def __init__(self, answer):
        super().__init__(answer.get('error', 'failed'))
        self.answer = answer


def call(url, path, body=None, timeout=120):
    """GET (body None) or POST a JSON body; the control server's JSON answer."""
    data = None if body is None else json.dumps(body).encode('utf-8')
    headers = {'Accept': 'application/json'}
    if data is not None:
        headers.update({'Content-Type': 'application/json', 'X-Fleet-Control': '1'})
    request = urllib.request.Request(url.rstrip('/') + path, data=data, headers=headers,
                                     method='GET' if data is None else 'POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.loads(response.read() or b'{}')
    except urllib.error.HTTPError as exc:
        try:
            answer = json.loads(exc.read() or b'{}')
        except ValueError:
            answer = {}
        answer = dict(answer, ok=False, error=answer.get('error') or f'HTTP {exc.code}')
    except (urllib.error.URLError, OSError, ValueError) as exc:
        answer = dict(ok=False, error=f'cannot reach the control server at {url}: {exc}')
    if answer.get('ok') is False:
        raise Failed(answer)
    return answer


def pairs(items):
    """KEY=VALUE arguments: values parsed as JSON (numbers, true, null, lists), else text."""
    result = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or not key:
            raise SystemExit(f'ctl: {item!r} is not KEY=VALUE')
        if key in STRINGS:
            result[key] = value
            continue
        try:
            result[key] = json.loads(value)
        except ValueError:
            result[key] = value
    return result


def replicas(args):
    found = []
    for spec in args.replica or []:
        ids, _, via = spec.partition(':')
        replica = dict(devices=[i for i in ids.split(',') if i])
        if via:
            replica['via'] = via
        found.append(replica)
    found += [dict(url=url) for url in args.server or []]
    return found


def wait_job(url, job, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = call(url, '/api/state')
        current = next((j for j in state['jobs'] if j['id'] == job['id']), None)
        if current is None or current['state'] != 'running':
            return current or job
        time.sleep(1)
    return dict(job, state='running', result='still running when ctl stopped waiting')


def wait_pool(url, pool_id, seconds):
    deadline = time.monotonic() + seconds
    while True:
        pool = next((p for p in call(url, '/api/state')['pools'] if p['id'] == pool_id), None)
        if pool is None:
            return dict(ok=False, error=f'No pool called {pool_id}')
        if pool['state'] == 'ready' or time.monotonic() >= deadline:
            return dict(ok=pool['state'] == 'ready', pool=pool,
                        **({} if pool['state'] == 'ready' else dict(error=f"{pool_id} is {pool['state']} after {seconds} s")))
        time.sleep(3)


def parser():
    top = argparse.ArgumentParser(prog='python3 -m fleet ctl', description=__doc__.split('\n')[0],
                                  epilog='Commands: see python3 -m fleet ctl help', add_help=True)
    top.add_argument('--url', default=os.environ.get('PS5LM_FLEET_URL', 'http://127.0.0.1:8090'),
                     help='the control server (default $PS5LM_FLEET_URL or http://127.0.0.1:8090)')
    top.add_argument('--state', action='store_true', help='after a change, print the whole state too')
    sub = top.add_subparsers(dest='command', required=True)
    sub.add_parser('help')
    for name in ('status', 'devices', 'pools', 'jobs', 'config'):
        sub.add_parser(name)
    log = sub.add_parser('log').add_mutually_exclusive_group(required=True)
    log.add_argument('--replica')
    log.add_argument('--device')
    sub.add_parser('search').add_argument('--wait', action='store_true')
    sub.add_parser('apply').add_argument('file', help='a JSON file, or - for standard input')

    device = sub.add_parser('device').add_subparsers(dest='action', required=True)
    add = device.add_parser('add')
    add.add_argument('id')
    add.add_argument('--kind', required=True, choices=('ps5', 'gpu', 'rpc'))
    add.add_argument('--address')
    add.add_argument('--port', type=int)
    add.add_argument('--device', help='a GPU: its ggml name, such as CUDA0')
    add.add_argument('--memory-mib', type=int)
    add.add_argument('--reserve-mib', type=int)
    add.add_argument('--tag', action='append')
    add.add_argument('--disabled', action='store_true')
    add.add_argument('--screen', action='store_true', help='a console: keep its screen as a worker')
    add.add_argument('--cache-gib', type=int, help="a console: GiB of weights its worker keeps (default 32, 0: none)")
    edit = device.add_parser('set')
    edit.add_argument('id')
    edit.add_argument('pairs', nargs='+', metavar='KEY=VALUE')
    device.add_parser('rm').add_argument('id')
    for action in ('worker', 'leave', 'restart', 'launch', 'quit', 'unload', 'clear-cache', 'stop'):
        command = device.add_parser(action)
        command.add_argument('id')
        command.add_argument('--wait', action='store_true', help='wait for the action to finish')

    pool = sub.add_parser('pool').add_subparsers(dest='action', required=True)
    add = pool.add_parser('add')
    add.add_argument('id')
    add.add_argument('--model', required=True)
    add.add_argument('--sha256')
    add.add_argument('--context', type=int, help='tokens per request (default 32768)')
    add.add_argument('--slots', type=int, help='requests at once per replica (default 1)')
    add.add_argument('--shared-context', action='store_true', help='the slots share one cache of --context tokens')
    add.add_argument('--cache-type', help='q8_0 (default), q4_0, f16, ...')
    add.add_argument('--flash-attn', choices=('on', 'off'))
    add.add_argument('--alias', action='append')
    add.add_argument('--replica', action='append', metavar='DEV[,DEV...][:via]')
    add.add_argument('--server', action='append', metavar='URL', help='a server run elsewhere, as a replica')
    add.add_argument('--auto', type=int, metavar='COUNT', help='add COUNT replicas from free devices')
    add.add_argument('--per', type=int, default=1, metavar='K', help='devices per --auto replica (default 1)')
    add.add_argument('--kind', action='append', choices=('ps5', 'gpu', 'rpc'), help='--auto: only these kinds')
    add.add_argument('--tag', action='append', help='--auto: only devices with these tags')
    add.add_argument('--start', action='store_true', help='start serving at once')
    edit = pool.add_parser('set')
    edit.add_argument('id')
    edit.add_argument('pairs', nargs='+', metavar='KEY=VALUE')
    pool.add_parser('rm').add_argument('id')
    start = pool.add_parser('start')
    start.add_argument('id')
    start.add_argument('--wait', type=int, default=0, metavar='SECONDS', help='wait until every replica is ready')
    pool.add_parser('stop').add_argument('id')
    plan = pool.add_parser('plan')
    plan.add_argument('id')
    plan.add_argument('--wait', action='store_true')

    replica = sub.add_parser('replica').add_subparsers(dest='action', required=True)
    replica.add_parser('restart').add_argument('key')
    return top


def run(args):
    url = args.url
    command, action = args.command, getattr(args, 'action', None)
    if command == 'help':
        return dict(ok=True, help=__doc__)
    if command in ('status', 'devices', 'pools', 'jobs'):
        state = call(url, '/api/state')
        return state if command == 'status' else {command: state[command]}
    if command == 'config':
        return call(url, '/api/config')
    if command == 'log':
        query = f'replica={args.replica}' if args.replica else f'device={args.device}'
        return call(url, '/api/log?' + urllib.parse.quote(query, safe='=/'))
    if command == 'search':
        answer = call(url, '/api/search', {})
        return dict(answer, job=wait_job(url, answer['job']), consoles=call(url, '/api/state')['consoles']) \
            if args.wait else answer
    if command == 'apply':
        text = sys.stdin.read() if args.file == '-' else Path(args.file).read_text()
        try:
            config = json.loads(text)
        except ValueError as exc:
            raise Failed(dict(ok=False, error=f'{args.file} is not JSON: {exc}')) from exc
        return call(url, '/api/apply', dict(config=config))
    if command == 'device':
        if action == 'add':
            body = dict(id=args.id, kind=args.kind)
            for key in ('address', 'port', 'device', 'memory_mib', 'reserve_mib', 'cache_gib'):
                if getattr(args, key) is not None:
                    body[key] = getattr(args, key)
            if args.tag:
                body['tags'] = args.tag
            if args.disabled:
                body['enabled'] = False
            if args.screen:
                body['screen'] = True
            return call(url, '/api/devices/add', body)
        if action == 'set':
            return call(url, '/api/devices/update', dict(pairs(args.pairs), id=args.id))
        if action == 'rm':
            return call(url, '/api/devices/remove', dict(id=args.id))
        answer = call(url, '/api/devices/action', dict(id=args.id, action=action))
        if args.wait and 'job' in answer:
            answer['job'] = wait_job(url, answer['job'])
            answer['ok'] = answer['job']['state'] == 'done'
        return answer
    if command == 'pool':
        if action == 'add':
            body = dict(id=args.id, model=args.model)
            for key in ('sha256', 'context', 'slots', 'cache_type'):
                if getattr(args, key) is not None:
                    body[key] = getattr(args, key)
            if args.shared_context:
                body['shared_context'] = True
            if args.flash_attn:
                body['flash_attn'] = args.flash_attn == 'on'
            if args.alias:
                body['aliases'] = args.alias
            body['replicas'] = replicas(args)
            if args.auto:
                body['auto'] = dict(count=args.auto, per=args.per, kinds=args.kind or [], tags=args.tag or [])
            if args.start:
                body['running'] = True
            return call(url, '/api/pools/add', body)
        if action == 'set':
            return call(url, '/api/pools/update', dict(pairs(args.pairs), id=args.id))
        if action in ('rm', 'stop'):
            return call(url, f"/api/pools/{'remove' if action == 'rm' else 'stop'}", dict(id=args.id))
        if action == 'start':
            answer = call(url, '/api/pools/start', dict(id=args.id))
            return wait_pool(url, args.id, args.wait) if args.wait else answer
        answer = call(url, '/api/pools/plan', dict(id=args.id))
        if args.wait:
            answer['job'] = wait_job(url, answer['job'], timeout=900)
            pool = next(p for p in call(url, '/api/state')['pools'] if p['id'] == args.id)
            answer['replicas'] = pool['replicas']
        return answer
    return call(url, '/api/replicas/restart', dict(key=args.key))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        answer = run(args)
        code = 0 if answer.get('ok', True) is not False else 1
    except Failed as exc:
        answer, code = exc.answer, 1
    if not args.state and args.command not in ('status',):
        answer = {k: v for k, v in answer.items() if k != 'state'}
    print(json.dumps(answer, indent=2))
    return code
