"""Fleet control server: the devices, the pools that serve models on them, and
one API in front of it all.

    python3 -m fleet control           # page and JSON API on http://127.0.0.1:8090
    python3 -m fleet ctl status        # the same from a shell or an agent (fleet/ctl.py)

Devices hold model layers:
  ps5   a console with PS5LM and ftpsrv (2121). The control server moves it
        between worker mode and the app over FTP (fleet/devices.py).
  gpu   one of this computer's GPUs: used directly by a replica of its own, or
        through a ggml-rpc-server the control server runs for it in a split.
  rpc   a ggml-rpc-server somewhere else, by address.

A pool is one model and how to serve it (context per request, slots, cache
type, flash attention), with replicas that each serve all of it:
  {"devices": ["ps5-a", "pc-gpu"]}   a layer split over RPC, 1 to 15 devices
  {"devices": ["ps5-b"]}             one console: PS5LM serves the model itself
  {"devices": ["pc-gpu"]}              one of this computer's GPUs, directly
  {"url": "http://10.0.0.9:8080"}      a server run elsewhere, only routed to
The API (fleet/router.py) sends each request to a ready replica of the pool its
"model" names. A running pool's replicas are started, watched, and started
again with back-off when they fail or lose a device.

Everything is one JSON document (fleet/control.json, docs/FLEET.md): the page
and `ctl` edit it a piece at a time, and `ctl apply` replaces it whole.

RPC has no authentication, so devices belong on a network you trust. The page
can write to consoles and start processes: it listens on loopback only, and
refuses requests with a foreign Host header or without its own request header,
so another web site cannot drive it from your browser.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import copy
import ftplib
import http.server
import ipaddress
import itertools
import json
from pathlib import Path
import re
import signal
import socket
import threading
import time
import urllib.parse

from . import controller, devices, router
from .models import ModelError, fetch_model, inspect_file
from .planner import CACHE_TYPES, FleetError, MIB, endpoint, plan as make_plan

CONFIG_PATH = controller.ROOT / 'fleet' / 'control.json'
PAGE = Path(__file__).with_name('control.html')
BIN = controller.ROOT / 'build' / 'fleet-host' / 'bin'
UNBOUNDED_MIB = 1 << 20    # a device without a budget: capped by the free memory it reports
# launcher: PS5LM's launch payload (ps5/app/launch_app.c), which an ELF loader on the
# console's port 9021 runs to open PS5LM, so a closed or crashed console comes back
# without anyone there. Empty: the app build's own copy, when there is one.
SETTINGS = dict(api_port=8081, api_lan=False, api_key='', host_memory_mib=8192, host_reserve_mib=2048, lan='',
                launcher='')
DEFAULT_LAUNCHER = controller.ROOT / 'build' / 'app-ps5' / 'launch-PPSA99575.elf'
POOL_FIELDS = dict(model='', sha256='', context=32768, slots=1, shared_context=False, cache_type='q8_0',
                   flash_attn=True, running=False, aliases=[], replicas=[])
SERVING = ('model', 'sha256', 'context', 'slots', 'shared_context', 'cache_type', 'flash_attn')
DEFAULT_PORT = dict(ps5=devices.WORKER_PORT, gpu=50052, rpc=50052)
# What a device keeps back from the planner for llama.cpp's compute buffers and
# headroom. A console's worker reports its free memory with the app's own use
# taken off, and llama.cpp measured 192 to 256 MiB of compute buffers on it at
# up to 192k tokens; a PC's GPU shares its memory with the desktop.
DEFAULT_RESERVE = dict(ps5=512, gpu=1024, rpc=1024)
# A console's worker settings, written to its env.txt (devices.worker_env). No
# screen by default: a console in a fleet is only a worker, and its screen takes
# about 350 MiB the model can have. 32 GiB of the weights hosts send are kept.
CONSOLE_FIELDS = dict(screen=False, cache_gib=32)
CACHE_GIB_MAX = 4096
ACTIONS = dict(ps5=('worker', 'leave', 'restart', 'launch', 'quit', 'unload', 'clear-cache'), gpu=('worker', 'stop'),
               rpc=())
CACHE_CHECK_SECONDS = 60   # how often a console's weight cache is listed (FTP)
LIMITS = dict(devices=1024, pools=64, replicas=256)
REPLICA_PORTS = range(18100, 19100)
LOAD_TIMEOUT = dict(rpc=1800, direct=900, app=900, server=0)
MASK = 'set'               # how the page and `ctl export` show a key that is set
_ID = re.compile(r'[A-Za-z0-9_.-]{1,64}')
_DEVICE = re.compile(r'[A-Za-z]{2,16}[0-9]{0,2}')
_TAG = re.compile(r'[A-Za-z0-9_.:-]{1,32}')
_ALIAS = re.compile(r'\S{1,128}')
_URL = re.compile(r'https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?/?')


class ControlError(ValueError):
    """A request the control server refuses, with a reason to show."""


# What a failed start, action or check can raise: reported, never fatal.
ERRORS = (ControlError, FleetError, ModelError, OSError, ftplib.Error, EOFError)


# --- configuration ---------------------------------------------------------------

def _int(raw, key, low, high, default=None, label=None):
    value = raw.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ControlError(f'{label or key} must be a whole number from {low} to {high}')
    return value


def _bool(raw, key, default, label=None):
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ControlError(f'{label or key} is true or false')
    return value


def _secret(value, label):
    if not isinstance(value, str) or len(value) > 256 or any(c.isspace() for c in value):
        raise ControlError(f'{label} is up to 256 characters without spaces')
    return value


def validate_device(raw):
    if not isinstance(raw, dict):
        raise ControlError('Each device is an object')
    name = raw.get('id')
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise ControlError('Device names are 1 to 64 letters, digits, dots, dashes or underscores')
    kind = raw.get('kind')
    if kind not in ACTIONS:
        raise ControlError(f'{name}: a device is a ps5, a gpu (on this computer) or an rpc server')
    port = _int(raw, 'port', 1024, 39999 if kind == 'ps5' else 65535, DEFAULT_PORT[kind], f'{name}: port')
    if kind == 'gpu':
        address = '127.0.0.1'
    else:
        try:
            address = endpoint(f"{raw.get('address')}:{port}").rsplit(':', 1)[0]
        except FleetError as exc:
            raise ControlError(f'{name}: {exc}') from exc
        if kind == 'ps5' and address.startswith('127.'):
            raise ControlError(f'{name}: a console has its own network address')
    memory = raw.get('memory_mib')
    if memory is not None and (type(memory) is not int or not 1 <= memory <= UNBOUNDED_MIB):
        raise ControlError(f'{name}: memory_mib is a budget in MiB, or null for what the device reports free')
    tags = raw.get('tags', [])
    if not isinstance(tags, list) or len(tags) > 16 or not all(isinstance(t, str) and _TAG.fullmatch(t) for t in tags):
        raise ControlError(f'{name}: tags are up to 16 words of letters, digits and . _ : -')
    device = dict(id=name, kind=kind, address=address, port=port,
                  enabled=_bool(raw, 'enabled', True, f'{name}: enabled'), memory_mib=memory,
                  reserve_mib=_int(raw, 'reserve_mib', 1, UNBOUNDED_MIB, DEFAULT_RESERVE[kind], f'{name}: reserve_mib'),
                  tags=sorted(set(tags)))
    if kind == 'ps5':
        # As a worker: with its screen or without (about 350 MiB more for the
        # model), and how many GiB of the weights it keeps (0: none).
        device['screen'] = _bool(raw, 'screen', CONSOLE_FIELDS['screen'], f'{name}: screen')
        device['cache_gib'] = _int(raw, 'cache_gib', 0, CACHE_GIB_MAX, CONSOLE_FIELDS['cache_gib'], f'{name}: cache_gib')
    if kind == 'gpu':
        gpu = raw.get('device', 'CUDA0')
        if not isinstance(gpu, str) or not _DEVICE.fullmatch(gpu):
            raise ControlError(f'{name}: the GPU is a ggml device name such as CUDA0, Vulkan0 or MTL0')
        device['device'] = gpu
    return device


def validate_replica(raw, by_id, pool):
    if not isinstance(raw, dict):
        raise ControlError(f'{pool}: each replica is an object')
    if 'url' in raw:
        url = raw['url']
        if not isinstance(url, str) or not _URL.fullmatch(url):
            raise ControlError(f'{pool}: a server replica is a base URL such as http://10.0.0.9:8080')
        return dict(url=url.rstrip('/'), api_key=_secret(raw.get('api_key', ''), f'{pool}: api_key'),
                    slots=_int(raw, 'slots', 1, 256, 1, f'{pool}: slots'))
    ids = raw.get('devices')
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 15 or not all(isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ControlError(f'{pool}: a replica runs on 1 to 15 different devices, or is a server "url"')
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ControlError(f'{pool}: no device called {missing[0]}')
    kinds = [by_id[i]['kind'] for i in ids]
    via = raw.get('via') or ('app' if kinds == ['ps5'] else 'direct' if kinds == ['gpu'] else 'rpc')
    if via not in ('rpc', 'app', 'direct'):
        raise ControlError(f'{pool}: via is rpc (a layer split), app (one console serves it) or direct (one local GPU)')
    if via == 'app' and kinds != ['ps5']:
        raise ControlError(f'{pool}: via app is one console serving the model with PS5LM')
    if via == 'direct' and kinds != ['gpu']:
        raise ControlError(f"{pool}: via direct is one of this computer's GPUs")
    split = raw.get('split') or []
    if split and (via != 'rpc' or not isinstance(split, list) or len(split) != len(ids)
                  or any(type(c) is not int or c < 1 for c in split)):
        raise ControlError(f'{pool}: a manual split is one positive block count per device of a split')
    return dict(devices=list(ids), via=via, split=list(split))


def validate_pool(raw, by_id):
    if not isinstance(raw, dict):
        raise ControlError('Each pool is an object')
    name = raw.get('id')
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise ControlError('Pool names are 1 to 64 letters, digits, dots, dashes or underscores')
    pool = dict(copy.deepcopy(POOL_FIELDS), id=name)
    pool.update({k: v for k, v in raw.items() if k in POOL_FIELDS})
    model = pool['model']
    if not isinstance(model, str) or len(model) > 4096:
        raise ControlError(f'{name}: the model is a file path or an https:// URL')
    if model.startswith(('http:', 'ftp:')) or ('://' in model and not model.startswith('https://')):
        raise ControlError(f'{name}: model URLs must be public https:// URLs')
    if not isinstance(pool['sha256'], str) or (pool['sha256'] and not re.fullmatch(r'[0-9a-fA-F]{64}', pool['sha256'])):
        raise ControlError(f'{name}: the SHA-256 is 64 hexadecimal characters, or empty')
    _int(pool, 'context', 256, 1048576, label=f'{name}: context')
    _int(pool, 'slots', 1, 64, label=f'{name}: slots')
    for key in ('shared_context', 'flash_attn', 'running'):
        _bool(pool, key, False, f'{name}: {key}')
    if pool['cache_type'] not in CACHE_TYPES:
        raise ControlError(f'{name}: cache_type is one of ' + ', '.join(CACHE_TYPES))
    if CACHE_TYPES[pool['cache_type']][0] > 1 and not pool['flash_attn']:
        raise ControlError(f"{name}: a {pool['cache_type']} cache needs flash_attn; llama.cpp quantizes V only with it")
    aliases = pool['aliases']
    if (not isinstance(aliases, list) or len(aliases) > 16
            or not all(isinstance(a, str) and _ALIAS.fullmatch(a) for a in aliases)):
        raise ControlError(f'{name}: aliases are up to 16 names without spaces')
    replicas = pool['replicas']
    if not isinstance(replicas, list) or len(replicas) > LIMITS['replicas']:
        raise ControlError(f"{name}: up to {LIMITS['replicas']} replicas")
    pool['replicas'] = [validate_replica(r, by_id, name) for r in replicas]
    return pool


def pool_name(model):
    """A pool name from a model path or URL: the file name without .gguf."""
    stem = model.split('?', 1)[0].rstrip('/').rsplit('/', 1)[-1]
    stem = re.sub(r'\.gguf$', '', stem, flags=re.IGNORECASE)
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', stem).strip('-.')[:64] or 'model'


def migrate(raw):
    """A version 1 configuration (one fleet of workers) as devices and one pool,
    stopped, with the new defaults for the cache (q8_0 with flash attention)."""
    workers = [w for w in raw.get('workers') or [] if isinstance(w, dict)]
    found = [{k: w[k] for k in ('id', 'kind', 'address', 'port', 'enabled', 'memory_mib', 'reserve_mib', 'device')
              if k in w} for w in workers]
    pools = []
    if isinstance(raw.get('model'), str) and raw['model']:
        members = [w.get('id') for w in workers if w.get('enabled', True)]
        split = raw.get('split') if isinstance(raw.get('split'), list) and len(raw['split']) == len(members) else []
        pools.append(dict(id=pool_name(raw['model']), model=raw['model'], sha256=raw.get('sha256', ''),
                          context=raw.get('context', POOL_FIELDS['context']),
                          replicas=[dict(devices=members, via='rpc', split=split)] if members else []))
    return dict({k: raw[k] for k in SETTINGS if k in raw}, devices=found, pools=pools)


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ControlError('The configuration is an object')
    if 'workers' in raw and 'devices' not in raw:
        raw = migrate(raw)
    config = dict(copy.deepcopy(SETTINGS), devices=[], pools=[])
    config.update({k: copy.deepcopy(v) for k, v in raw.items() if k in config})
    _int(config, 'api_port', 1024, 65535)
    _bool(config, 'api_lan', False)
    _secret(config['api_key'], 'The API key')
    _int(config, 'host_memory_mib', 256, UNBOUNDED_MIB)
    _int(config, 'host_reserve_mib', 1, UNBOUNDED_MIB)
    if not isinstance(config['launcher'], str) or len(config['launcher']) > 4096:
        raise ControlError('launcher is the path of launch-PPSA99575.elf on this computer, or empty')
    if not isinstance(config['lan'], str):
        raise ControlError('lan is a network such as 192.168.1.0/24, or empty')
    if config['lan']:
        try:
            network = ipaddress.ip_network(config['lan'], strict=False)
        except ValueError as exc:
            raise ControlError('The network to search is written like 192.168.1.0/24') from exc
        if network.version != 4 or not network.is_private or network.prefixlen < 22:
            raise ControlError('Search a private IPv4 network of at most 1024 addresses (a /22 or smaller)')
        config['lan'] = str(network)
    if not isinstance(config['devices'], list) or len(config['devices']) > LIMITS['devices']:
        raise ControlError(f"Up to {LIMITS['devices']} devices")
    config['devices'] = [validate_device(d) for d in config['devices']]
    ids = [d['id'] for d in config['devices']]
    if len(set(ids)) != len(ids):
        raise ControlError('Device names must be unique')
    ends = [(d['address'], d['port']) for d in config['devices']]
    if len(set(ends)) != len(ends):
        raise ControlError('Two devices have the same address and port')
    gpus = [d['device'] for d in config['devices'] if d['kind'] == 'gpu']
    if len(set(gpus)) != len(gpus):
        raise ControlError('Each of this computer\'s GPUs is one device')
    by_id = {d['id']: d for d in config['devices']}
    if not isinstance(config['pools'], list) or len(config['pools']) > LIMITS['pools']:
        raise ControlError(f"Up to {LIMITS['pools']} pools")
    config['pools'] = [validate_pool(p, by_id) for p in config['pools']]
    pool_ids = [p['id'] for p in config['pools']]
    if len(set(pool_ids)) != len(pool_ids):
        raise ControlError('Pool names must be unique')
    # Pools may share devices, so the same consoles can be laid out several ways
    # and switched between. Within a pool, and among running pools, a device
    # serves one replica.
    names, running = {}, {}
    for pool in config['pools']:
        for name in {pool['id'].lower(), *(a.lower() for a in pool['aliases'])}:
            if names.get(name, pool['id']) != pool['id']:
                raise ControlError(f"{name} names two pools ({names[name]} and {pool['id']})")
            names[name] = pool['id']
        users = {}
        for index, replica in enumerate(pool['replicas']):
            for device in replica.get('devices', []):
                if device in users:
                    raise ControlError(f"{device} is in two replicas of {pool['id']} ({users[device]} and {index}); "
                                       'a device serves one replica')
                users[device] = index
                if pool['running']:
                    if device in running:
                        raise ControlError(f"{device} is in running pools {running[device]} and {pool['id']}; "
                                           f"stop {running[device]} first")
                    running[device] = pool['id']
    return config


def load_config(path):
    path = Path(path)
    if not path.exists():
        return validate_config({})
    if path.stat().st_size > 4 * MIB:
        raise ControlError(f'{path} is over 4 MiB')
    return validate_config(json.loads(path.read_text()))


def save_config(path, config):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config, indent=2) + '\n')
    temporary.replace(path)


def masked(config):
    """The configuration with its keys shown only as set."""
    config = copy.deepcopy(config)
    if config.get('api_key'):
        config['api_key'] = MASK
    for pool in config.get('pools', []):
        for replica in pool.get('replicas', []):
            if replica.get('api_key'):
                replica['api_key'] = MASK
    return config


def unmask(raw, current):
    """Keys sent back as 'set' keep their current value."""
    if not isinstance(raw, dict):
        return raw
    raw = copy.deepcopy(raw)
    if raw.get('api_key') == MASK:
        raw['api_key'] = current['api_key']
    known = {(p['id'], r['url']): r['api_key'] for p in current['pools'] for r in p['replicas'] if 'url' in r}
    for pool in raw.get('pools') if isinstance(raw.get('pools'), list) else []:
        for replica in pool.get('replicas') or [] if isinstance(pool, dict) else []:
            if isinstance(replica, dict) and replica.get('api_key') == MASK:
                replica['api_key'] = known.get((pool.get('id'), replica.get('url')), '')
    return raw


def assign(config, count, per=1, kinds=(), tags=()):
    """Replicas of `per` devices each, from the enabled devices no replica has yet,
    in the order they are listed; only of `kinds`, and only with all of `tags`."""
    if type(count) is not int or not 1 <= count <= LIMITS['replicas'] or type(per) is not int or not 1 <= per <= 15:
        raise ControlError('auto takes a count of 1 to 256 replicas of 1 to 15 devices each ("per")')
    if (not isinstance(kinds, (list, tuple)) or not set(kinds) <= set(ACTIONS)
            or not isinstance(tags, (list, tuple)) or not all(isinstance(t, str) for t in tags)):
        raise ControlError('auto "kinds" is a list of ps5, gpu and rpc, and "tags" a list of tags')
    used = {d for p in config['pools'] for r in p['replicas'] if isinstance(r, dict) for d in r.get('devices', [])}
    free = [d['id'] for d in config['devices'] if d['enabled'] and d['id'] not in used
            and (not kinds or d['kind'] in kinds) and set(tags) <= set(d['tags'])]
    if len(free) < count * per:
        raise ControlError(f'{count} replicas of {per} need {count * per} free devices; {len(free)} match')
    return [dict(devices=free[i * per:(i + 1) * per]) for i in range(count)]


def signature(pool, replica):
    """What a replica serves with: when it changes, the replica starts over."""
    return json.dumps([{k: pool[k] for k in SERVING}, replica], sort_keys=True)


def cache_tokens(pool):
    """llama-server's --ctx-size for a pool: each slot its own context, or one shared."""
    return pool['context'] if pool['shared_context'] else pool['context'] * pool['slots']


# --- replicas as they run ---------------------------------------------------------------

class Replica:
    """One replica's process or address, state and load."""

    def __init__(self, pool_id, spec, sig):
        self.pool_id = pool_id
        self.spec = spec
        self.signature = sig
        self.key = ''
        self.via = spec.get('via', 'server')
        self.state = 'stopped'      # stopped, starting, waiting, ready, failed, stopping
        self.detail = ''
        self.since = time.time()
        self.process = None
        self.port = None
        self.url = spec.get('url', '')
        self.api_key = spec.get('api_key', '')
        self.slots = spec.get('slots', 1)
        self.expect = ''            # the model file a console should report serving
        self.in_flight = 0
        self.served = 0
        self.failures = 0
        self.retry_at = 0.0
        self.deadline = 0.0         # when a start still loading counts as failed
        self.placement = None
        self.check = None
        self.thread = None
        self.cancel = threading.Event()

    def view(self, lines=8):
        view = dict(key=self.key, pool=self.pool_id, via=self.via, devices=self.spec.get('devices', []),
                    split=self.spec.get('split', []), url=self.url, state=self.state, detail=self.detail,
                    since=round(self.since), slots=self.slots, in_flight=self.in_flight, served=self.served,
                    failures=self.failures, check=self.check,
                    retry_in=max(0, round(self.retry_at - time.time())) if self.state == 'failed' else 0)
        if self.placement:
            view['layers'] = [dict(id=n['id'], first=n['layer_start'], last=n['layer_end'] - 1,
                                   output=n['output_layer'], weights_mib=round(n['weights_bytes'] / MIB),
                                   cache_mib=round((n['kv_bytes'] + n['state_bytes']) / MIB))
                              for n in self.placement['nodes']]
        if self.process is not None and lines:
            view['log'] = list(self.process.lines)[-lines:]
        return view


class Fleet:
    """The configuration and everything running for it, behind one lock."""

    def __init__(self, config_path, binary_dir=BIN, cache_dir=None, watch=True):
        self.config_path = Path(config_path)
        self.binary_dir = Path(binary_dir)
        # Downloaded models. Not inside the checkout by default: under WSL that
        # is often /mnt/c, a slow disk to mmap a model from.
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'ps5lm-fleet'
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.config = load_config(self.config_path)
        self.replicas = []            # Replica, in configuration order
        self.status = {}              # device id -> live status
        self.probing = set()
        self.rpc_servers = {}         # device id -> Process: a local GPU's ggml-rpc-server
        self.manual = set()           # GPU workers started by hand, kept without a replica
        self.busy = {}                # device id -> the action under way
        self.env_pending = set()      # consoles whose env.txt changed since their worker started
        self.cache_seen = {}          # console id -> dict(files, mib, checked) of its weight cache
        self.consoles, self.networks = {}, []
        self.jobs = collections.deque(maxlen=50)
        self.job_ids = itertools.count(1)
        self.messages = collections.deque(maxlen=200)
        self.ports = set()
        self.digests = {}             # (path, size, mtime) -> SHA-256
        self.model_locks = collections.defaultdict(threading.Lock)
        self.gpu_memory = {}          # ggml device -> (total, free) MiB, as last read
        self.api, self.api_bind, self.api_error = None, None, ''
        self.api_wanted = False       # set by start_api(); tests run without the API
        self.requests = 0
        self.closing = False
        self._sync()
        if watch:
            threading.Thread(target=self._watch, daemon=True).start()

    # -- lookups and notes
    def note(self, text):
        with self.lock:
            self.messages.append(f"{time.strftime('%H:%M:%S')} {text}")

    def device(self, device_id):
        for device in self.config['devices']:
            if device['id'] == device_id:
                return device
        raise ControlError(f'No device called {device_id}')

    def pool(self, pool_id):
        for pool in self.config['pools']:
            if pool['id'] == pool_id:
                return pool
        raise ControlError(f'No pool called {pool_id}')

    def _binary(self, name='llama-server'):
        binary = self.binary_dir / name
        if not binary.is_file():
            raise ControlError(f'No {binary}; build the host first (scripts/build-fleet-host.sh)')
        return binary

    # -- the configuration
    def _sync(self):
        """Match replicas to the configuration: keep each whose pool settings and
        replica are unchanged, add new ones, and return the ones left over."""
        with self.lock:
            old, kept = list(self.replicas), []
            for pool in self.config['pools']:
                for index, spec in enumerate(pool['replicas']):
                    sig = signature(pool, spec)
                    match = next((r for r in old if r.pool_id == pool['id'] and r.signature == sig), None)
                    if match is None:
                        match = Replica(pool['id'], spec, sig)
                    else:
                        old.remove(match)
                    match.key = f"{pool['id']}/{index}"
                    kept.append(match)
            self.replicas = kept
            return old

    def apply(self, raw, label='Applied a new configuration'):
        """Replace the whole configuration. Replicas whose pool settings or devices
        changed start over; a running pool's new replicas start."""
        with self.lock:
            config = validate_config(unmask(raw, self.config))
            save_config(self.config_path, config)
            self.config = config
            stale = self._sync()
        for replica in stale:
            self._stop_soon(replica, 'removed or changed')
        self.note(label)
        self.wake.set()
        if self.api_wanted:
            self.start_api()      # the port or the network may have changed
        return config

    def _edit(self, change, label):
        with self.lock:
            config = copy.deepcopy(self.config)
            change(config)
            return self.apply(config, label)

    def set_settings(self, changes):
        allowed = {k: v for k, v in changes.items() if k in SETTINGS and not (k == 'api_key' and v == MASK)}
        return self._edit(lambda config: config.update(allowed), 'Changed the settings')

    def add_device(self, raw):
        name = raw.get('id') if isinstance(raw, dict) else None
        return self._edit(lambda config: config['devices'].append(raw), f'Added device {name}')

    def update_device(self, device_id, changes):
        before = dict(self.device(device_id))
        allowed = {k: v for k, v in changes.items()
                   if k in ('address', 'port', 'memory_mib', 'reserve_mib', 'enabled', 'device', 'tags', *CONSOLE_FIELDS)}

        def change(config):
            config['devices'] = [dict(d, **allowed) if d['id'] == device_id else d for d in config['devices']]
        config = self._edit(change, f'Changed device {device_id}')
        after = self.device(device_id)
        if after['kind'] == 'ps5' and any(after[k] != before.get(k) for k in CONSOLE_FIELDS):
            self._apply_console_settings(after)
        return config

    # -- a console's worker settings: its env.txt, read when PS5LM starts
    @staticmethod
    def console_env(device):
        return devices.worker_env(device['screen'], device['cache_gib'])

    def _in_running_pool(self, device_id):
        with self.lock:
            return any(p['running'] and any(device_id in r.get('devices', []) for r in p['replicas'])
                       for p in self.config['pools'])

    def _apply_console_settings(self, device):
        """A console's screen or cache setting changed: write its env.txt, and restart
        a worker no running pool uses, so the change applies now. A worker in use
        takes it at its next restart (a pool's next start restarts it first)."""
        device_id, address, port = device['id'], device['address'], device['port']
        env = self.console_env(device)

        def work():
            mode = devices.ps5_mode(address, port)
            if mode['mode'] == 'offline':
                with self.lock:
                    self.env_pending.add(device_id)
                return 'the console does not answer; the settings are written when it is next made a worker'
            if mode['mode'] != 'worker':
                devices.write_env(address, env)
                return 'written to env.txt; they apply when it is next a worker'
            if mode.get('busy') or self._in_running_pool(device_id):
                devices.write_env(address, env)
                with self.lock:
                    self.env_pending.add(device_id)
                return 'written to env.txt; its worker is in use, so they apply when it next restarts'
            return self._restart_worker(device, env)
        try:
            self.job(f'Settings for {device_id}', work, device=device_id)
        except ControlError as exc:   # another action on it is under way
            with self.lock:
                self.env_pending.add(device_id)
            self.note(f'{device_id}: {exc}; the settings apply at its next restart')

    def launcher(self):
        with self.lock:
            return Path(self.config['launcher']) if self.config['launcher'] else DEFAULT_LAUNCHER

    def _open_console(self, device, replica=None):
        """Open PS5LM on a console where it is closed (or crashed): send the launch
        payload to its ELF loader on 9021. False when it cannot be opened from here
        (no loader, or no launcher file). It opens as a worker when fleet-worker.txt
        is there, as the app otherwise; the caller waits for what it expects."""
        if not devices.launch_app(device['address'], self.launcher()):
            return False
        self.note(f"{device['id']}: opening PS5LM through the ELF loader on {devices.ELF_LOADER_PORT}")
        return True

    def _restart_worker(self, device, env, replica=None):
        """Restart a console's worker after setting `env` in env.txt. A worker of an
        older PS5LM does not take the restart request: it leaves worker mode, and
        is made a worker again."""
        name, address, port = device['id'], device['address'], device['port']
        if not devices.restart(address, env):
            if devices.leave_worker(address) != 'restarted':
                # An older build can only quit, and leave_worker removed its
                # fleet-worker.txt. Write it again (the app is closed, so nothing
                # takes the request), then open PS5LM, which starts as a worker.
                devices.make_worker(address, port, env)
                if not self._open_console(device, replica):
                    raise ControlError(f'{name}: this PS5LM cannot restart by itself, so it quit; open it on the console')
            elif not devices.make_worker(address, port, env):
                raise ControlError(f'{name}: PS5LM did not take the request to become a worker again')
        with self.lock:
            self.env_pending.discard(name)
        time.sleep(2)   # the old worker can answer for a moment after it took the request
        if not self._wait_port(address, port, 150, replica):
            raise ControlError(f'{name} did not come back as a worker in 150 s')
        return f'worker restarted on {address}:{port}'

    def remove_device(self, device_id):
        self.device(device_id)
        for pool in self.config['pools']:
            if any(device_id in r.get('devices', []) for r in pool['replicas']):
                raise ControlError(f"{device_id} is in pool {pool['id']}; take it out of the pool first")
        self._stop_rpc_server(device_id)

        def change(config):
            config['devices'] = [d for d in config['devices'] if d['id'] != device_id]
        return self._edit(change, f'Removed device {device_id}')

    def add_pool(self, raw):
        if not isinstance(raw, dict):
            raise ControlError('A pool is an object')
        raw = dict(raw)
        auto = raw.pop('auto', None)

        def change(config):
            if auto is not None:
                raw['replicas'] = list(raw.get('replicas') or []) + self._auto(config, auto)
            config['pools'].append(raw)
        return self._edit(change, f"Added pool {raw.get('id')}")

    def update_pool(self, pool_id, changes):
        self.pool(pool_id)
        changes = dict(changes)
        auto = changes.pop('auto', None)
        allowed = {k: v for k, v in changes.items() if k in POOL_FIELDS}

        def change(config):
            pool = next(p for p in config['pools'] if p['id'] == pool_id)
            pool.update(allowed)
            if auto is not None:
                pool['replicas'] = list(pool['replicas']) + self._auto(config, auto)
        what = 'Started' if allowed == {'running': True} else 'Stopped' if allowed == {'running': False} else 'Changed'
        return self._edit(change, f'{what} pool {pool_id}')

    @staticmethod
    def _auto(config, auto):
        if not isinstance(auto, dict):
            raise ControlError('auto is {"count": N, "per": K, "kinds": [...], "tags": [...]}')
        return assign(config, auto.get('count'), auto.get('per', 1), auto.get('kinds') or (), auto.get('tags') or ())

    def remove_pool(self, pool_id):
        self.pool(pool_id)

        def change(config):
            config['pools'] = [p for p in config['pools'] if p['id'] != pool_id]
        return self._edit(change, f'Removed pool {pool_id}')

    def restart_replica(self, key):
        with self.lock:
            replica = next((r for r in self.replicas if r.key == key), None)
            if replica is None:
                raise ControlError(f'No replica {key}')
        self._stop_soon(replica, 'restarting')
        self.wake.set()

    def export(self):
        with self.lock:
            return masked(self.config)

    # -- jobs: device actions, searches and plans, each on a thread of its own
    def job(self, label, work, device=None):
        with self.lock:
            if device is not None:
                if device in self.busy:
                    raise ControlError(f'{device} is busy: {self.busy[device]}')
                self.busy[device] = label
            job = dict(id=next(self.job_ids), label=label, state='running', result='',
                       started=round(time.time()), finished=None)
            self.jobs.append(job)

        def run():
            try:
                outcome = dict(state='done', result=work() or 'done')
            except ERRORS as exc:
                outcome = dict(state='failed', result=str(exc))
            except Exception as exc:   # a bug: keep serving, and say what happened
                outcome = dict(state='failed', result=f'{type(exc).__name__}: {exc}')
            with self.lock:
                job.update(outcome, finished=round(time.time()))
                if device is not None:
                    self.busy.pop(device, None)
            self.note(f"{label}: {job['result']}")
            self.wake.set()
        threading.Thread(target=run, daemon=True).start()
        return dict(job)

    def search(self):
        def work():
            networks = ([ipaddress.ip_network(self.config['lan'])] if self.config['lan']
                        else devices.detect_networks())
            if not networks:
                raise ControlError('Could not tell which network to search; set "lan" in the settings')
            found = devices.discover(networks)
            with self.lock:
                self.networks, self.consoles = networks, found
            return (f"searched {', '.join(map(str, networks))}: "
                    + (', '.join(f"{i['name'] or a} at {a}" for a, i in found.items()) or 'no consoles answered'))
        return self.job('Searching the network', work, device='(search)')

    def device_action(self, device_id, action):
        """worker / leave / restart / quit / unload / clear-cache for a console;
        worker / stop for a local GPU."""
        device = self.device(device_id)
        if action not in ACTIONS[device['kind']]:
            raise ControlError(f"A {device['kind']} device takes " + (', '.join(ACTIONS[device['kind']]) or 'no actions'))
        with self.lock:
            for pool in self.config['pools']:
                if pool['running'] and any(device_id in r.get('devices', []) for r in pool['replicas']):
                    raise ControlError(f"{device_id} serves pool {pool['id']}, which is running; stop the pool first")
        if device['kind'] == 'gpu':
            if action == 'worker':
                self._start_rpc_server(device, manual=True)
                return dict(result=f"{device_id}: RPC worker on 127.0.0.1:{device['port']} ({device['device']})")
            self._stop_rpc_server(device_id)
            return dict(result=f'{device_id}: RPC worker stopped')
        address, port = device['address'], device['port']
        env = self.console_env(device)

        def work():
            status = devices.ps5_mode(address, port)
            mode = status['mode']
            if action == 'clear-cache':
                cleared = devices.clear_cache(address)
                with self.lock:
                    self.cache_seen[device_id] = dict(files=0, mib=0, checked=time.time())
                return f"deleted {cleared['files']} cached tensors, {cleared['mib'] / 1024:.1f} GiB"
            if mode == 'offline':
                raise ControlError('the console does not answer: off, in rest mode, or not jailbroken since it started')
            if action == 'launch':
                if mode != 'idle':
                    return f'PS5LM is already open ({mode})'
                if not self._open_console(device):
                    raise ControlError('cannot open PS5LM from here: no ELF loader on port 9021, or no launcher '
                                       f'at {self.launcher()}')
                return 'asked the console to open PS5LM'
            if action == 'restart':
                if mode == 'worker':
                    if status.get('busy'):
                        raise ControlError('its worker is serving a split; stop that first')
                    return self._restart_worker(device, env)
                if not devices.restart(address):
                    if not self._open_console(device):
                        raise ControlError('PS5LM is not open; open it on the console')
                    return 'PS5LM was closed; asked the console to open it'
                return 'asked PS5LM to restart' + (' with its model' if mode in ('serving', 'loading') else '')
            if action == 'worker':
                if mode == 'worker':
                    return 'already a worker'
                if not devices.make_worker(address, port, env):
                    # Closed: fleet-worker.txt is written, so it opens as a worker.
                    if not self._open_console(device):
                        return 'PS5LM is not open; it starts as a worker when it is opened'
                with self.lock:
                    self.env_pending.discard(device_id)
                if self._wait_port(address, port, 150):
                    return f'worker on {address}:{port}'
                raise ControlError('PS5LM took the request, but no worker answers yet; look at the console')
            if action == 'leave':
                if mode != 'worker':
                    with devices.console_ftp(address) as ftp:
                        devices.delete(ftp, devices.WORKER_FILE)
                    return f'not a worker ({mode}); it will not start as one either'
                if devices.leave_worker(address) == 'restarted':
                    return 'left worker mode; PS5LM restarted as the app'
                return ('this PS5LM cannot leave worker mode by itself, so it quit; it opens as the app '
                        'from the home screen (install a build with leave-worker to skip that)')
            if action == 'quit':
                devices.quit_app(address)
                return 'asked PS5LM to quit'
            devices.stop_serving(address)
            return 'asked PS5LM to unload its model'
        return dict(job=self.job(f'{action} {device_id}', work, device=device_id))

    def plan(self, pool_id):
        """Check each replica of a pool against its devices without starting it."""
        pool = copy.deepcopy(self.pool(pool_id))

        def work():
            with self.lock:
                replicas = [r for r in self.replicas if r.pool_id == pool_id]
            if not any(r.via != 'server' for r in replicas):
                return 'no replicas on devices to plan'
            path = self._model(pool, None)
            results = []
            for replica in replicas:
                if replica.via == 'server':
                    continue
                if replica.state in ('starting', 'ready') or replica.thread is not None:
                    results.append(f'{replica.key}: running with its plan')
                    continue
                try:
                    results.append(f'{replica.key}: ' + self._plan_replica(replica, pool, path))
                except ERRORS as exc:
                    results.append(f'{replica.key}: {exc}')
            return '; '.join(results)
        return dict(job=self.job(f'Planning {pool_id}', work))

    def _plan_replica(self, replica, pool, path):
        members = [self.device(d) for d in replica.spec['devices']]
        if replica.via == 'app':
            return 'PS5LM checks it when it loads the model'
        if replica.via == 'direct':
            binary = self._binary()
            gpu = members[0]
            found = devices.local_devices(binary)
            if gpu['device'] not in found:
                raise ControlError(f"llama.cpp sees no {gpu['device']} here")
            free = min(found[gpu['device']][1], gpu['memory_mib'] or UNBOUNDED_MIB)
            replica.check = {gpu['id']: controller.direct_check(
                binary, path, gpu['device'], free, cache_tokens(pool), pool['slots'], pool['shared_context'],
                pool['cache_type'], pool['flash_attn'])}
            figures = replica.check[gpu['id']]
            return f"fits: {figures['total_mib']} of {figures['budget_mib']} MiB free on {gpu['device']}"
        # With the build and every device up as a worker: their free memory and
        # llama.cpp's own check. Otherwise an estimate from the budgets.
        binary = self.binary_dir / 'llama-server'
        up = binary.is_file() and all(self.status.get(d['id'], {}).get('mode') == 'worker' for d in members)
        inventory = self._inventory(members)
        if up:
            inventory = controller.probe(binary, inventory)
        elif any(d['memory_mib'] is None for d in members):
            raise ControlError('needs its devices up as workers to read their free memory, or a memory_mib for each')
        placement = self._make_plan(pool, path, inventory, replica.spec['split'])
        if up:
            placement['llama_check'] = controller.llama_check(binary, path, placement)
        replica.placement, replica.check = placement, placement.get('llama_check')
        return ', '.join(f"{n['id']} blocks {n['layer_start']}-{n['layer_end'] - 1}" for n in placement['nodes']) + (
            '' if up else ' (estimate; llama.cpp checks it when the workers are up)')

    def _inventory(self, members):
        return dict(host_memory_mib=self.config['host_memory_mib'], host_reserve_mib=self.config['host_reserve_mib'],
                    nodes=[dict(id=d['id'], endpoint=f"{d['address']}:{d['port']}",
                                memory_mib=d['memory_mib'] or UNBOUNDED_MIB, reserve_mib=d['reserve_mib'])
                           for d in members])

    @staticmethod
    def _make_plan(pool, path, inventory, split):
        info = inspect_file(path)
        placement = make_plan(info, inventory, cache_tokens(pool), split or None, cache_type=pool['cache_type'],
                              flash_attn=pool['flash_attn'], slots=pool['slots'], unified=pool['shared_context'])
        placement['model_name'] = info['name']
        placement['model_path'] = str(path)
        return placement

    # -- models
    def _model(self, pool, replica):
        model = pool['model']
        if not model:
            raise ControlError(f"Pool {pool['id']} has no model")
        with self.model_locks[model]:
            if model.startswith('https://'):
                if replica is not None:
                    replica.detail = 'downloading the model'
                path, digest = fetch_model(model, self.cache_dir, max_bytes=128 * 1024**3,
                                           max_cache_bytes=256 * 1024**3, max_seconds=3600)
                path = Path(path)
            else:
                path = Path(model).expanduser().resolve()
                if not path.is_file():
                    raise ControlError(f'No model file {path}')
                digest = None
                if pool['sha256']:
                    stat = path.stat()
                    key = (str(path), stat.st_size, stat.st_mtime_ns)
                    digest = self.digests.get(key)
                    if digest is None:
                        if replica is not None:
                            replica.detail = "checking the model's SHA-256"
                        digest = self.digests[key] = controller.checksum(path)
            if pool['sha256'] and digest.lower() != pool['sha256'].lower():
                raise ControlError(f"{path.name}: its SHA-256 is {digest}, not the pool's {pool['sha256']}")
        return path

    # -- starting and stopping replicas
    def _set(self, replica, state, detail=''):
        with self.lock:
            if replica.state != state:
                replica.since = time.time()
            replica.state, replica.detail = state, detail

    def _launch(self, replica, pool):
        with self.lock:
            if replica.thread is not None:
                return
            replica.cancel = threading.Event()
            replica.placement = replica.check = None
            replica.deadline = 0.0        # set once the start has handed over to loading
            self._set(replica, 'starting', 'starting')
            replica.thread = threading.Thread(target=self._start, args=(replica, copy.deepcopy(pool)), daemon=True)
            replica.thread.start()

    def _start(self, replica, pool):
        try:
            self._start_replica(replica, pool)
        except ERRORS as exc:
            self._fail(replica, str(exc))
        except Exception as exc:   # a bug: fail the replica, keep the server
            self._fail(replica, f'{type(exc).__name__}: {exc}')
        finally:
            with self.lock:
                replica.thread = None
            self.wake.set()

    def _fail(self, replica, message):
        with self.lock:
            process, replica.process = replica.process, None
            self._free_port(replica)
            if replica.cancel.is_set():
                self._set(replica, 'stopped')
                stopped = True
            else:
                replica.failures += 1
                delay = min(600, 15 * 2 ** min(replica.failures - 1, 6))
                replica.retry_at = time.time() + delay
                self._set(replica, 'failed', message)
                stopped = False
        if process is not None:
            process.stop()
        if not stopped:
            self.note(f'{replica.key}: {message} (trying again in {delay} s)')

    def _ready(self, replica):
        with self.lock:
            replica.failures = 0
            self._set(replica, 'ready')
        self.note(f'{replica.key}: ready ({replica.url})')

    def _checkpoint(self, replica):
        if replica.cancel.is_set():
            raise ControlError('stopped')

    def _take_port(self):
        with self.lock:
            for port in REPLICA_PORTS:
                if port in self.ports:
                    continue
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    try:
                        sock.bind(('127.0.0.1', port))
                    except OSError:
                        continue
                self.ports.add(port)
                return port
        raise ControlError('No free local port for a replica')

    def _free_port(self, replica):
        with self.lock:
            self.ports.discard(replica.port)
            replica.port = None

    def _run(self, replica, argv, port, detail):
        with self.lock:
            if replica.cancel.is_set():
                self.ports.discard(port)
                raise ControlError('stopped')
            replica.process = devices.Process(argv, replica.key)
            replica.port, replica.url = port, f'http://127.0.0.1:{port}'
            replica.deadline = time.time() + LOAD_TIMEOUT[replica.via]
            self._set(replica, 'starting', detail)

    def _wait_port(self, address, port, timeout, replica=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if devices.port_open(address, port, 1):
                return True
            if replica is not None and replica.cancel.is_set():
                return False
            time.sleep(2)
        return False

    def _start_replica(self, replica, pool):
        if replica.via == 'server':
            with self.lock:
                replica.deadline = 0
            self._set(replica, 'starting', 'waiting for the server to answer')
            return
        members = [self.device(d) for d in replica.spec['devices']]
        off = [d['id'] for d in members if not d['enabled']]
        if off:
            raise ControlError(f'{off[0]} is disabled')
        path = self._model(pool, replica)
        self._checkpoint(replica)
        replica.slots = pool['slots']
        if replica.via == 'app':
            return self._start_app(replica, pool, members[0], path)
        binary = self._binary()
        if replica.via == 'direct':
            gpu = members[0]
            self._stop_rpc_server(gpu['id'])      # the GPU directly, not through its RPC worker
            replica.detail = "reading the GPU's free memory"
            found = devices.local_devices(binary)
            with self.lock:
                self.gpu_memory.update(found)
            if gpu['device'] not in found:
                raise ControlError(f"{gpu['id']}: llama.cpp sees no {gpu['device']} here "
                                   f"(it sees {', '.join(found) or 'no GPU'})")
            free = min(found[gpu['device']][1], gpu['memory_mib'] or UNBOUNDED_MIB)
            replica.detail = 'llama.cpp checking that the model fits'
            replica.check = {gpu['id']: controller.direct_check(
                binary, path, gpu['device'], free, cache_tokens(pool), pool['slots'], pool['shared_context'],
                pool['cache_type'], pool['flash_attn'])}
            self._checkpoint(replica)
            port = self._take_port()
            argv = controller.direct_command(binary, path, gpu['device'], cache_tokens(pool), port, pool['slots'],
                                             pool['shared_context'], pool['cache_type'], pool['flash_attn'])
            return self._run(replica, argv, port, f"loading the model onto {gpu['device']}")
        for device in members:
            self._ensure_worker(device, replica)
            self._checkpoint(replica)
        replica.detail = "reading the workers' free memory"
        inventory = controller.probe(binary, self._inventory(members))
        placement = self._make_plan(pool, path, inventory, replica.spec['split'])
        replica.placement = placement
        replica.detail = 'llama.cpp checking the plan'
        placement['llama_check'] = replica.check = controller.llama_check(binary, path, placement)
        self._checkpoint(replica)
        port = self._take_port()
        self._run(replica, controller.command(binary, path, placement, port), port,
                  'loading the layers onto the workers')

    def _ensure_worker(self, device, replica):
        name = device['id']
        if device['kind'] == 'gpu':
            self._start_rpc_server(device)
            if not self._wait_port('127.0.0.1', device['port'], 30, replica):
                raise ControlError(f'{name}: its RPC worker did not start (see its log)')
            return
        if device['kind'] == 'rpc':
            if not devices.port_open(device['address'], device['port'], 2):
                raise ControlError(f"{name}: nothing answers on {device['address']}:{device['port']}")
            return
        status = devices.ps5_mode(device['address'], device['port'])
        mode, env = status['mode'], self.console_env(device)
        if mode == 'worker':
            if status.get('busy'):
                return
            # Every split starts on a freshly started worker. The console's driver
            # keeps much of what an earlier host allocated after it disconnects
            # (12.12 GiB free at start, 7.08 GiB after one 27B session, measured
            # 2026-11-11), so a second load can run out of memory. The restart also
            # applies the console's settings; it takes 4 s, and the weights come
            # back from the console's cache.
            replica.detail = f'restarting {name} for a clean start'
            self._restart_worker(device, env, replica)
            return
        if mode == 'offline':
            raise ControlError(f'{name} does not answer: off, in rest mode, or not jailbroken since it started')
        replica.detail = f'making {name} a worker'
        if not devices.make_worker(device['address'], device['port'], env):
            # Closed or crashed: fleet-worker.txt is written, so it opens as a worker.
            replica.detail = f'opening PS5LM on {name}'
            if not self._open_console(device, replica):
                raise ControlError(f'PS5LM is not open on {name}; open it and it starts as a worker')
        with self.lock:
            self.env_pending.discard(name)
        replica.detail = f'waiting for {name} to restart as a worker'
        if not self._wait_port(device['address'], device['port'], 150, replica):
            raise ControlError(f'{name} did not come back as a worker in 150 s')

    def _start_app(self, replica, pool, device, path):
        name, address = device['id'], device['address']
        mode = devices.ps5_mode(address, device['port'])
        if mode['mode'] == 'offline':
            raise ControlError(f'{name} does not answer: off, in rest mode, or not jailbroken since it started')
        size = path.stat().st_size
        remote = f'{devices.PS5_MODELS}/{path.name}'
        args = devices.replica_args(remote, pool['context'], pool['slots'], pool['cache_type'], pool['flash_attn'],
                                    pool['shared_context'])
        # Already serving this model with these settings (the control server
        # restarted, say): take it over as it is instead of reloading it.
        if (mode['mode'] == 'serving' and Path(mode.get('model') or '').name == path.name
                and devices.read_text(address, devices.ARGS_FILE).splitlines() == args):
            with self.lock:
                replica.url, replica.expect = f'http://{address}:{devices.APP_PORT}', path.name
                replica.deadline = time.time() + LOAD_TIMEOUT['app']
            self._set(replica, 'starting', f'{name} already serves the model')
            return
        replica.detail = f'looking for the model on {name}'
        if not devices.console_has(address, remote, size):
            def progress(sent, total):
                if replica.cancel.is_set():
                    raise ControlError('stopped')
                replica.detail = f'copying the model to {name}: {sent * 100 // total}% of {total / 2**30:.1f} GiB'
            devices.push_file(address, path, remote, progress)
        self._checkpoint(replica)
        with self.lock:
            replica.url, replica.expect = f'http://{address}:{devices.APP_PORT}', path.name
            replica.deadline = time.time() + LOAD_TIMEOUT['app']
        worker = devices.ps5_mode(address, device['port'])['mode'] == 'worker'
        if devices.serve_model(address, args, worker=worker):
            self._set(replica, 'starting', f'PS5LM on {name} is restarting with the model')
            return
        if worker:
            # A PS5LM without leave-worker can only quit; the model loads when it is opened.
            with devices.console_ftp(address) as ftp:
                devices.delete(ftp, devices.WORKER_FILE)
            devices.quit_app(address)
            self._set(replica, 'waiting', f'open PS5LM on {name}: this build cannot leave worker mode by itself')
        else:
            self._set(replica, 'waiting', f'open PS5LM on {name}; it loads the model when it opens')

    def _stop(self, replica, reason='stopped', was=None):
        """Stop a replica. `was` is its state before stopping began, when the
        caller has already marked it stopping (_stop_soon)."""
        with self.lock:
            replica.cancel.set()
            process, replica.process = replica.process, None
            was = was or replica.state
            self._set(replica, 'stopping', reason)
        if process is not None:
            process.stop()
        if replica.via == 'app' and was in ('starting', 'ready', 'waiting'):
            device = next((d for d in self.config['devices'] if d['id'] == replica.spec['devices'][0]), None)
            if device is not None:
                try:
                    devices.stop_serving(device['address'])
                except ERRORS as exc:
                    self.note(f"{replica.key}: could not ask {device['id']} to unload the model: {exc}")
        with self.lock:
            self._free_port(replica)
            replica.failures, replica.retry_at = 0, 0.0
            self._set(replica, 'stopped')

    def _stop_soon(self, replica, reason):
        with self.lock:
            # Already stopping, or stopped with nothing starting: nothing to do.
            if replica.state == 'stopping' or (replica.state == 'stopped' and replica.thread is None):
                return
            was = replica.state
            replica.cancel.set()
            self._set(replica, 'stopping', reason)
        threading.Thread(target=self._stop, args=(replica, reason, was), daemon=True).start()

    def _start_rpc_server(self, device, manual=False):
        with self.lock:
            if manual:
                self.manual.add(device['id'])
            process = self.rpc_servers.get(device['id'])
            if process is not None and process.running():
                return process
            binary = self._binary('ggml-rpc-server')
            process = devices.Process([str(binary), '-H', '127.0.0.1', '-p', str(device['port']),
                                       '-d', device['device']], device['id'])
            self.rpc_servers[device['id']] = process
        self.note(f"{device['id']}: RPC worker started on 127.0.0.1:{device['port']} ({device['device']})")
        return process

    def _stop_rpc_server(self, device_id):
        with self.lock:
            process = self.rpc_servers.pop(device_id, None)
            self.manual.discard(device_id)
        if process is not None:
            process.stop()
            self.note(f'{device_id}: RPC worker stopped')

    # -- watching: device probes, replica health, and the running pools
    def _watch(self):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix='fleet-probe')
        while not self.closing:
            try:
                self._probe_devices(executor)
                self._reconcile(executor)
            except Exception as exc:   # keep watching whatever happens
                self.note(f'watch: {type(exc).__name__}: {exc}')
            self.wake.wait(3)
            self.wake.clear()

    def _probe_devices(self, executor):
        now = time.time()
        with self.lock:
            due = [dict(d) for d in self.config['devices'] if d['id'] not in self.probing
                   and now - self.status.get(d['id'], {}).get('checked', 0) >= 4]
            self.probing.update(d['id'] for d in due)
        for device in due:
            executor.submit(self._probe, device)

    def _probe(self, device):
        # An RPC worker a split is using serves only that split: other connections
        # wait unanswered ('silent'), which is busy, not gone (see ps5_mode).
        with self.lock:
            in_use = any(device['id'] in r.spec.get('devices', []) and r.state in ('starting', 'ready')
                         for r in self.replicas)
            process = self.rpc_servers.get(device['id'])
            direct = any(r.via == 'direct' and device['id'] in r.spec['devices']
                         and r.state in ('starting', 'ready') for r in self.replicas)
            # A split being started connects to its consoles' RPC servers several
            # times in a row (memory probe, llama.cpp's check, llama-server). Each
            # serves one client with a listen queue of one, and a probe between
            # two of those connections made llama-server's single attempt fail.
            starting = device['kind'] == 'ps5' and any(
                device['id'] in r.spec.get('devices', []) and r.via == 'rpc' and r.state == 'starting'
                for r in self.replicas)
            previous = dict(self.status.get(device['id'], {}))
        try:
            if starting:
                status = {k: v for k, v in previous.items() if k != 'checked'} or dict(mode='worker')
            elif device['kind'] == 'ps5':
                status = devices.ps5_mode(device['address'], device['port'])
                with self.lock:
                    seen = self.cache_seen.get(device['id'], {})
                if status['mode'] != 'offline' and time.time() - seen.get('checked', 0) >= CACHE_CHECK_SECONDS:
                    use = devices.cache_use(device['address'])
                    with self.lock:
                        self.cache_seen[device['id']] = dict(use or dict(files=0, mib=0), checked=time.time())
            elif device['kind'] == 'rpc':
                port = devices.port_probe(device['address'], device['port'])
                status = (dict(mode='worker') if port == 'open' else dict(mode='worker', busy=True)
                          if port == 'silent' and in_use else dict(mode='offline'))
            elif direct:
                status = dict(mode='direct')
            elif process is not None and process.running():
                port = devices.port_probe('127.0.0.1', device['port'])
                status = (dict(mode='worker') if port == 'open' else dict(mode='worker', busy=True)
                          if port == 'silent' and in_use else dict(mode='starting'))
            else:
                status = dict(mode='idle')
        except Exception as exc:   # a probe never stops the watch
            status = dict(mode='unknown', error=str(exc))
        status['checked'] = time.time()
        with self.lock:
            self.status[device['id']] = status
            self.probing.discard(device['id'])

    def _reconcile(self, executor):
        with self.lock:
            pools = {p['id']: p for p in self.config['pools']}
            replicas = [r for r in self.replicas if r.pool_id in pools]
            status = copy.deepcopy(self.status)
        checks = [executor.submit(self._check, r, status) for r in replicas]
        concurrent.futures.wait(checks, timeout=20)
        now = time.time()
        # Stop first, then start: a pool that takes over another's devices waits
        # until that pool's replicas have let go of them.
        for replica in replicas:
            with self.lock:
                idle, state = replica.thread is None, replica.state
            if not pools[replica.pool_id]['running'] and (state not in ('stopped', 'stopping') or not idle):
                self._stop_soon(replica, 'pool stopped')
        for replica in replicas:
            pool = pools[replica.pool_id]
            with self.lock:
                idle, state = replica.thread is None, replica.state
                held = {d for r in self.replicas if r is not replica and (r.state != 'stopped' or r.thread is not None)
                        and r.state != 'failed' for d in r.spec.get('devices', [])}
            if (pool['running'] and idle and state in ('stopped', 'failed') and now >= replica.retry_at
                    and not held & set(replica.spec.get('devices', []))):
                self._launch(replica, pool)
        with self.lock:
            active = {d for r in self.replicas if r.via == 'rpc' and (r.state in ('starting', 'ready')
                                                                       or r.thread is not None)
                      for d in r.spec['devices']}
            unused = [d for d in self.rpc_servers if d not in active and d not in self.manual]
        for device_id in unused:
            self._stop_rpc_server(device_id)

    def _check(self, replica, status):
        """Move a starting replica to ready, and a ready one that lost its process,
        server or a device to failed."""
        with self.lock:
            # A start under way (copying a model, making workers, checking the
            # plan) owns the replica until it hands over.
            if replica.thread is not None:
                return
            state, process, url, deadline = replica.state, replica.process, replica.url, replica.deadline
        if replica.via in ('rpc', 'direct'):
            if state not in ('starting', 'ready') or process is None:
                return
            if not process.running():
                errors = [line for line in process.lines if re.search(r'error|failed|abort', line, re.IGNORECASE)]
                self._fail(replica, errors[-1].strip()[:300] if errors
                           else f'llama-server exited with code {process.proc.returncode}')
                return
            if state == 'starting':
                code, _ = devices.http_json(url, '/health', timeout=2)
                if code == 200:
                    self._ready(replica)
                elif time.time() > deadline:
                    self._fail(replica, f'did not finish loading in {LOAD_TIMEOUT[replica.via] // 60} minutes')
                return
            lost = self._lost(replica, status)
            if lost:
                self._fail(replica, lost)
            return
        if replica.via == 'app':
            if state not in ('starting', 'waiting', 'ready'):
                return
            device_status = status.get(replica.spec['devices'][0], {})
            serving = device_status.get('mode') == 'serving' and Path(device_status.get('model') or '').name == replica.expect
            if serving and state != 'ready':
                self._ready(replica)
            elif state == 'starting' and deadline and time.time() > deadline:
                self._fail(replica, 'PS5LM did not serve the model in 15 minutes; see llama.log in /data/PS5LM')
            elif state == 'ready' and not serving and device_status.get('checked', 0) > replica.since:
                self._fail(replica, f"{replica.spec['devices'][0]} stopped serving the model "
                                    f"(it is {device_status.get('mode')})")
            return
        if state in ('starting', 'ready'):
            code, _ = devices.http_json(url, '/health', timeout=3)
            if code == 200 and state == 'starting':
                self._ready(replica)
            elif code != 200 and state == 'ready':
                self._fail(replica, f'{url} does not answer /health' + (f' ({code})' if code else ''))

    def _lost(self, replica, status):
        for device_id in replica.spec['devices']:
            device = next((d for d in self.config['devices'] if d['id'] == device_id), None)
            if device is None:
                return f'{device_id} is no longer configured'
            if device['kind'] == 'gpu':
                process = self.rpc_servers.get(device_id)
                if replica.via == 'rpc' and (process is None or not process.running()):
                    return f"lost {device_id}: its RPC worker stopped"
                continue
            seen = status.get(device_id, {})
            if seen.get('checked', 0) > replica.since and seen.get('mode') not in ('worker', 'unknown'):
                return f"lost {device_id}: it is {seen.get('mode')}"
        return None

    # -- the API in front (fleet/router.py calls these)
    def api_key(self):
        return self.config['api_key']

    def healthy(self):
        with self.lock:
            return any(r.state == 'ready' for r in self.replicas)

    def models(self):
        with self.lock:
            return [dict(id=p['id'], object='model', created=0, owned_by='ps5lm-fleet', aliases=p['aliases'],
                         context=p['context'], slots=p['slots'],
                         ready_replicas=sum(r.pool_id == p['id'] and r.state == 'ready' for r in self.replicas),
                         replicas=len(p['replicas']), file=Path(p['model'].split('?', 1)[0]).name if p['model'] else '')
                    for p in self.config['pools']]

    def _names(self, pool):
        names = {pool['id'].lower(), *(a.lower() for a in pool['aliases'])}
        if pool['model']:
            file = Path(pool['model'].split('?', 1)[0]).name.lower()
            names |= {file, re.sub(r'\.gguf$', '', file)}
        return names

    def acquire(self, model, exclude=()):
        with self.lock:
            ready = [p for p in self.config['pools'] if any(r.pool_id == p['id'] and r.state == 'ready'
                                                          for r in self.replicas)]
            if model:
                wanted = model.lower()
                pool = next((p for p in self.config['pools'] if wanted in self._names(p)), None)
                if pool is None:
                    if len(ready) != 1:
                        raise router.NoRoute(404, f'No pool serves "{model}"; the pools are '
                                                  + (', '.join(p['id'] for p in self.config['pools']) or 'none'))
                    pool = ready[0]
            elif ready:
                pool = ready[0]
            else:
                raise router.NoRoute(503, 'No pool has a ready replica')
            choices = [r for r in self.replicas if r.pool_id == pool['id'] and r.state == 'ready'
                       and r.key not in exclude]
            if not choices:
                raise router.NoRoute(503, f"No replica of {pool['id']} is ready")
            replica = min(choices, key=lambda r: (r.in_flight / r.slots, r.served))
            replica.in_flight += 1
            replica.served += 1
            self.requests += 1
            return replica

    def release(self, replica):
        with self.lock:
            replica.in_flight = max(0, replica.in_flight - 1)

    def unreachable(self, replica, reason):
        """The router could not connect: no more requests until a check finds it again."""
        with self.lock:
            if replica.state == 'ready':
                replica.deadline = time.time() + 300
                self._set(replica, 'starting', f'not answering ({reason}); waiting for it')
        self.note(f'{replica.key}: not answering the router: {reason}')
        self.wake.set()

    def start_api(self):
        with self.lock:
            self.api_wanted = True
            bind = ('0.0.0.0' if self.config['api_lan'] else '127.0.0.1', self.config['api_port'])
            if self.closing or (self.api is not None and self.api_bind == bind):
                return
            old, self.api = self.api, None
        if old is not None:
            old.shutdown()
            old.server_close()
        try:
            server = router.make_server(self, *bind)
        except OSError as exc:
            with self.lock:
                self.api_bind, self.api_error = None, f'Cannot listen on {bind[0]}:{bind[1]}: {exc}'
            self.note(self.api_error)
            return
        with self.lock:
            self.api, self.api_bind, self.api_error = server, bind, ''
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.note(f'API listening on {bind[0]}:{bind[1]}')

    # -- what the page and ctl show
    def snapshot(self):
        with self.lock:
            config = masked(self.config)
            # A device's replica: the one of a running pool, else the first that lists it.
            running = {p['id'] for p in self.config['pools'] if p['running']}
            used = {}
            for replica in sorted(self.replicas, key=lambda r: r.pool_id not in running):
                for device_id in replica.spec.get('devices', []):
                    used.setdefault(device_id, replica.key)
            device_views = []
            for device in config['devices']:
                status = dict(self.status.get(device['id'], {'mode': 'checking'}))
                status['age'] = round(time.time() - status.pop('checked', time.time()))
                view = dict(device, status=status, replica=used.get(device['id']),
                            busy=self.busy.get(device['id']), actions=list(ACTIONS[device['kind']]),
                            console=self.consoles.get(device['address']) if device['kind'] == 'ps5' else None,
                            rpc_worker=device['id'] in self.rpc_servers)
                if device['kind'] == 'ps5':
                    cache = self.cache_seen.get(device['id'])
                    view.update(cache={k: cache[k] for k in ('files', 'mib')} if cache else None,
                                settings_pending=device['id'] in self.env_pending)
                device_views.append(view)
            pool_views = []
            for pool in config['pools']:
                views = [r.view(0 if r.state == 'ready' else 8) for r in self.replicas if r.pool_id == pool['id']]
                ready = sum(v['state'] == 'ready' for v in views)
                states = {v['state'] for v in views}
                state = ('stopped' if not pool['running'] and states <= {'stopped'} else
                         'stopping' if not pool['running'] else
                         'ready' if views and ready == len(views) else 'partial' if ready else
                         'failed' if views and states <= {'failed'} else 'starting' if views else 'empty')
                pool_views.append(dict(pool, replicas=views, ready=ready, state=state,
                                       cache_tokens=cache_tokens(pool)))
            host, port = self.api_bind or ('127.0.0.1', self.config['api_port'])
            return dict(
                settings={k: config[k] for k in SETTINGS},
                api=dict(url=f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/v1", lan=host == '0.0.0.0',
                         listening=self.api is not None, error=self.api_error, requests=self.requests,
                         in_flight=sum(r.in_flight for r in self.replicas)),
                devices=device_views, pools=pool_views,
                consoles=[dict(address=a, **i, added=any(d['address'] == a for d in config['devices']))
                          for a, i in sorted(self.consoles.items())],
                networks=[str(n) for n in self.networks],
                jobs=list(self.jobs)[-20:], messages=list(self.messages)[-60:],
                build=dict(llama_server=(self.binary_dir / 'llama-server').is_file(),
                           rpc_server=(self.binary_dir / 'ggml-rpc-server').is_file(),
                           fit_params=(self.binary_dir / 'llama-fit-params').is_file()),
                gpus={k: dict(total_mib=v[0], free_mib=v[1]) for k, v in self.gpu_memory.items()})

    def log(self, replica_key='', device_id=''):
        with self.lock:
            if replica_key:
                replica = next((r for r in self.replicas if r.key == replica_key), None)
                if replica is None:
                    raise ControlError(f'No replica {replica_key}')
                return dict(key=replica_key, lines=list(replica.process.lines) if replica.process else [])
            process = self.rpc_servers.get(device_id)
            if process is None:
                raise ControlError(f'No RPC worker runs for {device_id}')
            return dict(device=device_id, lines=list(process.lines))

    def shutdown(self):
        """Stop what runs on this computer. Consoles serving a model by themselves keep serving."""
        with self.lock:
            self.closing = True
            replicas = list(self.replicas)
            api, self.api = self.api, None
        if api is not None:
            api.shutdown()
            api.server_close()
        for replica in replicas:
            with self.lock:
                replica.cancel.set()
                process, replica.process = replica.process, None
            if process is not None:
                process.stop()
        for device_id in list(self.rpc_servers):
            self._stop_rpc_server(device_id)


# --- the page and its API ----------------------------------------------------------

def handler_for(fleet, port, extra_hosts=()):
    # extra_hosts: names a trusted proxy puts the page under, such as
    # `tailscale serve` (tailnet only), which keeps the visitor's Host header.
    allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}', f'[::1]:{port}', *extra_hosts}

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = 'ps5lm-fleet-control'

        def log_message(self, *args):
            pass

        def _send(self, code, body, kind='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _trusted(self):
            # A foreign Host header means DNS rebinding; refuse it.
            if self.headers.get('Host') not in allowed_hosts:
                self._send(403, {'error': 'Open this page as http://127.0.0.1:%d/' % port})
                return False
            return True

        def do_GET(self):
            if not self._trusted():
                return
            url = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(url.query)
            try:
                if url.path == '/':
                    self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
                elif url.path == '/api/state':
                    self._send(200, fleet.snapshot())
                elif url.path == '/api/config':
                    self._send(200, fleet.export())
                elif url.path == '/api/log':
                    self._send(200, fleet.log((query.get('replica') or [''])[0], (query.get('device') or [''])[0]))
                else:
                    self._send(404, {'error': 'Not found'})
            except ControlError as exc:
                self._send(400, {'ok': False, 'error': str(exc)})

        def do_POST(self):
            if not self._trusted():
                return
            # Browsers send a custom header cross-site only after a CORS
            # preflight, which this server never grants.
            if self.headers.get('X-Fleet-Control') != '1':
                self._send(403, {'error': 'Missing X-Fleet-Control header'})
                return
            length = int(self.headers.get('Content-Length') or 0)
            if length > 4 * MIB:
                self._send(413, {'error': 'Request too large'})
                return
            try:
                body = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(body, dict):
                    raise ControlError('Send a JSON object')
                result = self._route(body) or {}
                self._send(200, dict(ok=True, **result, state=fleet.snapshot()))
            except (ControlError, FleetError, ModelError, json.JSONDecodeError) as exc:
                self._send(400, {'ok': False, 'error': str(exc), 'state': fleet.snapshot()})

        def _route(self, body):
            name = body.get('id')
            routes = {
                '/api/apply': lambda: dict(config=masked(fleet.apply(body.get('config', body)))),
                '/api/settings': lambda: fleet.set_settings(body) and None,
                '/api/search': lambda: dict(job=fleet.search()),
                '/api/devices/add': lambda: fleet.add_device(body) and None,
                '/api/devices/update': lambda: fleet.update_device(name, body) and None,
                '/api/devices/remove': lambda: fleet.remove_device(name) and None,
                '/api/devices/action': lambda: fleet.device_action(name, body.get('action')),
                '/api/pools/add': lambda: fleet.add_pool(body) and None,
                '/api/pools/update': lambda: fleet.update_pool(name, body) and None,
                '/api/pools/remove': lambda: fleet.remove_pool(name) and None,
                '/api/pools/start': lambda: fleet.update_pool(name, dict(running=True)) and None,
                '/api/pools/stop': lambda: fleet.update_pool(name, dict(running=False)) and None,
                '/api/pools/plan': lambda: fleet.plan(name),
                '/api/replicas/restart': lambda: fleet.restart_replica(body.get('key')),
            }
            if self.path not in routes:
                raise ControlError('Unknown action')
            return routes[self.path]()

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python3 -m fleet control',
                                     description='PS5LM fleet control: devices, pools of replicas, and one API')
    parser.add_argument('--port', type=int, default=8090, help='page port on 127.0.0.1 (default 8090)')
    parser.add_argument('--config', default=str(CONFIG_PATH), help='where the configuration is kept')
    parser.add_argument('--build', default=str(BIN.parent), help='the host build (scripts/build-fleet-host.sh output)')
    parser.add_argument('--cache-dir', default=str(Path.home() / '.cache' / 'ps5lm-fleet'),
                        help='where models given as https:// URLs are downloaded')
    parser.add_argument('--allow-host', action='append', default=[], metavar='NAME[:PORT]',
                        help='also answer requests for this Host header: the name a proxy that only your own devices '
                             'reach puts the page under, e.g. `tailscale serve --https=8090 http://127.0.0.1:8090` '
                             'and --allow-host pc.tailnet.ts.net:8090. Never a public proxy.')
    args = parser.parse_args(argv)
    for host in args.allow_host:
        if not re.fullmatch(r'[A-Za-z0-9.-]{1,253}(:[0-9]{1,5})?', host):
            print(f'ps5lm-fleet control: --allow-host takes a host name with an optional :port, not {host!r}')
            return 2
    try:
        fleet = Fleet(args.config, binary_dir=Path(args.build).expanduser() / 'bin',
                      cache_dir=Path(args.cache_dir).expanduser())
    except (ControlError, json.JSONDecodeError, OSError) as exc:
        print(f'ps5lm-fleet control: {exc}')
        return 2
    server = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(fleet, args.port, args.allow_host))
    server.daemon_threads = True
    fleet.start_api()

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    print(f'ps5lm-fleet control: http://127.0.0.1:{args.port}/  (configuration in {args.config})'
          + ''.join(f', also as {host}' for host in args.allow_host), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        fleet.shutdown()
    return 0
