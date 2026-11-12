"""Devices the fleet control server drives: PS5 consoles (over FTP and PS5LM's
request files), this computer's GPUs and CPU, ggml-rpc-servers and
OpenAI-compatible servers.

A console running PS5LM is in one of these modes, as seen from the network:

    worker      its fleet RPC worker answers (fleet-worker.txt was there at start)
    serving     PS5LM's llama-server answers on 8081 with a model loaded
    loading     PS5LM's llama-server is up but still loading
    idle        ftpsrv answers, but neither of the above: PS5LM is closed, or
                open with no model
    offline     nothing answers: off, in rest mode, or not jailbroken since it started

Moving between them, all over FTP:

    idle/serving -> worker   write fleet-worker.txt, request `unload`: the app
                             restarts (LoadExec) and starts as a worker
    worker -> idle           request `leave-worker`: the app forgets
                             fleet-worker.txt and restarts as the app (PS5LM
                             after this change); older builds only `quit`
    idle/serving -> serving  write app-args.txt (or request `load`), request
                             `unload`: the app restarts with that model
    serving -> idle          remove app-args.txt, request `unload`
    worker -> worker         request `unload`: the worker restarts and reads env.txt
                             again, its screen and weight cache settings (PS5LM
                             after v0.11.0)

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import collections
import contextlib
import ftplib
import http.client
import io
import ipaddress
import json
from pathlib import Path
import posixpath
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse

from . import controller

DDP_PORT = 9302
DDP_SEARCH = b'SRCH * HTTP/1.1\ndevice-discovery-protocol-version:00030010\n'
FTP_PORT = 2121
APP_PORT = 8081              # PS5LM's own llama-server
WORKER_PORT = 30052          # a PS5 title cannot bind 40000 and up
DATA = '/data/PS5LM'
WORKER_FILE = f'{DATA}/fleet-worker.txt'
ARGS_FILE = f'{DATA}/app-args.txt'
ENV_FILE = f'{DATA}/env.txt'       # switches PS5LM sets at start, one NAME=value a line
CACHE_DIR = f'{DATA}/rpc-cache'    # the worker's weight cache (PS5LM_RPC_CACHE_GB)
ACK = 'I_UNDERSTAND_RPC_IS_UNAUTHENTICATED'
PS5_MODELS = f'{DATA}/models'
MIB = 1 << 20
_CACHE_NAME = re.compile(r'[0-9a-f]{16}(\.[0-9a-f]{16}\.part)?')
# How long a running PS5LM gets to take a request file. It looks once a second,
# but an app short of memory under load was measured taking longer than 6 s.
TAKE_SECONDS = 20


# --- discovery ----------------------------------------------------------------------

def firmware(version):
    """'11600005' (system-version in a discovery reply) -> '11.60'."""
    if not isinstance(version, str) or not re.fullmatch(r'\d{8}', version):
        return ''
    return f'{int(version[:2])}.{version[2:4]}'


def parse_ddp(data):
    """A PlayStation discovery reply: HTTP-like status line and key:value lines."""
    lines = data.decode('utf-8', 'replace').splitlines()
    if not lines or not lines[0].startswith('HTTP/1.1 '):
        return None
    status = lines[0][9:].strip()
    fields = {}
    for line in lines[1:]:
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key.strip().lower()] = value.strip()
    if not fields.get('host-type', '').upper().startswith('PS'):
        return None
    return dict(name=fields.get('host-name', ''), type=fields.get('host-type', ''),
                state='on' if status.startswith('200') else 'rest mode' if status.startswith('620') else status,
                firmware=firmware(fields.get('system-version')))


def detect_networks():
    """The LAN to search. Under WSL's NAT the Linux side only sees a virtual
    network, so Windows is asked for its own; elsewhere, the default route's
    address with a /24."""
    networks = []
    try:
        wsl = 'microsoft' in Path('/proc/version').read_text().lower()
    except OSError:
        wsl = False
    powershell = shutil.which('powershell.exe') or '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    if wsl and Path(powershell).exists():
        script = ("Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } | "
                  "ForEach-Object { $_.IPv4Address | ForEach-Object { '{0}/{1}' -f $_.IPAddress, $_.PrefixLength } }")
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            output = subprocess.run([powershell, '-NoProfile', '-Command', script],
                                    capture_output=True, text=True, timeout=20).stdout
            for line in output.split():
                with contextlib.suppress(ValueError):
                    networks.append(ipaddress.ip_interface(line.strip()).network)
    if not networks:
        with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('192.0.2.1', 9))   # no packet is sent for a UDP connect
            networks.append(ipaddress.ip_interface(probe.getsockname()[0] + '/24').network)
    usable = []
    for network in networks:
        if network.is_private and not network.is_loopback:
            if network.prefixlen < 22:   # search at most 1024 addresses per network
                network = ipaddress.ip_network(f'{network.network_address}/22')
            if network not in usable:
                usable.append(network)
    return usable


def discover(networks, timeout=1.5):
    """Ask every address once (unicast, so it also works through NAT) and
    collect the consoles that answer: {address: {name, type, state, firmware}}."""
    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(0.2)
        for network in networks:
            for host in network.hosts():
                with contextlib.suppress(OSError):
                    sock.sendto(DDP_SEARCH, (str(host), DDP_PORT))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data, (address, _) = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            info = parse_ddp(data)
            if info:
                found[address] = info
    return found


# --- probes ------------------------------------------------------------------------------

def port_probe(address, port, timeout=0.6):
    """'open' (it accepted), 'refused' (nothing listens there) or 'silent' (no answer:
    the host is down, or a listener whose queue is full)."""
    try:
        with socket.create_connection((address, port), timeout=timeout):
            return 'open'
    except ConnectionRefusedError:
        return 'refused'
    except OSError:
        return 'silent'


def port_open(address, port, timeout=0.6):
    return port_probe(address, port, timeout) == 'open'


def http_json(url, path, timeout=3):
    """GET a JSON document from a llama-server: (status, data); (0, None) when unreachable."""
    base = urllib.parse.urlsplit(url)
    if base.scheme not in ('http', 'https') or not base.hostname:
        return 0, None
    secure = base.scheme == 'https'
    connection = (http.client.HTTPSConnection if secure else http.client.HTTPConnection)(
        base.hostname, base.port or (443 if secure else 80), timeout=timeout)
    try:
        connection.request('GET', base.path.rstrip('/') + path)
        response = connection.getresponse()
        body = response.read(1 << 20)
        try:
            return response.status, json.loads(body or b'null')
        except ValueError:
            return response.status, None
    except (OSError, http.client.HTTPException):
        return 0, None
    finally:
        connection.close()


def server_info(url):
    """A llama-server's state: ready, loading or down, its model file and slot count."""
    status, _ = http_json(url, '/health')
    if status == 503:
        return dict(state='loading')
    if status != 200:
        return dict(state='down')
    _, props = http_json(url, '/props')
    props = props if isinstance(props, dict) else {}
    return dict(state='ready', model=props.get('model_path', ''), slots=props.get('total_slots'))


def ps5_mode(address, worker_port=WORKER_PORT):
    """The console's mode (see the module docstring), with the model it serves.
    At most three short connection attempts for a console that is off."""
    worker = port_probe(address, worker_port)
    if worker == 'open':
        return dict(mode='worker')
    ftp = port_open(address, FTP_PORT)
    if worker == 'silent' and ftp:
        # llama.cpp's RPC server serves one client at a time with a listen queue
        # of one: while a split uses the worker, new connections go unanswered.
        # A console without a worker refuses them instead.
        return dict(mode='worker', busy=True)
    if not ftp and not port_open(address, APP_PORT):
        return dict(mode='offline')
    info = server_info(f'http://{address}:{APP_PORT}')
    if info['state'] == 'ready':
        return dict(mode='serving', model=info.get('model', ''), slots=info.get('slots'))
    if info['state'] == 'loading':
        return dict(mode='loading')
    return dict(mode='idle' if ftp else 'offline')


# --- the console over FTP ------------------------------------------------------------------

@contextlib.contextmanager
def console_ftp(address, timeout=30):
    ftp = ftplib.FTP()
    ftp.connect(address, FTP_PORT, timeout=timeout)
    try:
        ftp.login()
        yield ftp
    finally:
        with contextlib.suppress(OSError, EOFError, ftplib.Error):
            ftp.quit()
        ftp.close()


def put_text(ftp, path, text):
    ftp.storbinary(f'STOR {path}', io.BytesIO(text.encode('utf-8')))


def get_text(ftp, path, limit=1 << 20):
    """A small text file over an open connection, or '' when it is not there."""
    buffer = io.BytesIO()
    try:
        ftp.retrbinary(f'RETR {path}', lambda chunk: buffer.write(chunk) if buffer.tell() < limit else None)
    except ftplib.error_perm:
        return ''
    return buffer.getvalue()[:limit].decode('utf-8', 'replace')


def read_text(address, path, limit=1 << 20):
    """A small text file from the console, or '' when it is not there."""
    with console_ftp(address) as ftp:
        return get_text(ftp, path, limit)


def worker_env(screen=True, cache_gib=0):
    """The env.txt lines a console's settings give its worker (PS5LM after v0.11.0):
    PS5LM_WORKER_SCREEN=off starts it without its screen, which leaves about 350 MiB
    more for the model; PS5LM_RPC_CACHE_GB keeps up to that many GiB of the weights
    hosts send on the console, so loading them again comes from its storage."""
    return {'PS5LM_WORKER_SCREEN': 'on' if screen else 'off', 'PS5LM_RPC_CACHE_GB': str(cache_gib)}


def merged_env(text, settings):
    """env.txt with `settings` set, its other lines kept as they were."""
    kept = [line for line in text.splitlines() if line.split('=', 1)[0].strip() not in settings]
    return '\n'.join(kept + [f'{name}={value}' for name, value in settings.items()]) + '\n'


def set_env(ftp, settings):
    """Set lines of env.txt over an open connection. True when the file changed:
    PS5LM reads it when it starts, so a running app or worker needs a restart."""
    current = get_text(ftp, ENV_FILE)
    wanted = merged_env(current, settings)
    if wanted == current:
        return False
    put_text(ftp, ENV_FILE, wanted)
    return True


def cache_files(ftp):
    """The worker's cache files as name -> bytes, or None when it has no cache folder."""
    lines = []
    try:
        ftp.retrlines(f'LIST {CACHE_DIR}', lines.append)
    except ftplib.error_perm:
        return None
    found = {}
    for line in lines:
        parts = line.split(None, 8)
        if len(parts) == 9 and parts[4].isdigit() and _CACHE_NAME.fullmatch(parts[8]):
            found[parts[8]] = int(parts[4])
    return found


def cache_use(address):
    """What a console's weight cache holds: dict(files, mib), or None without one."""
    with console_ftp(address) as ftp:
        found = cache_files(ftp)
    if found is None:
        return None
    return dict(files=len(found), mib=round(sum(found.values()) / MIB))


def clear_cache(address):
    """Delete the worker's cache files (only the names its cache writes). Returns
    dict(files, mib) deleted. A worker loading as they go only sends them again."""
    with console_ftp(address) as ftp:
        found = cache_files(ftp) or {}
        for name in found:
            delete(ftp, f'{CACHE_DIR}/{name}')
    return dict(files=len(found), mib=round(sum(found.values()) / MIB))


def delete(ftp, path):
    """DELE that accepts any 2xx reply (ftpsrv answers 226, which ftplib.delete refuses)."""
    with contextlib.suppress(ftplib.error_perm):
        ftp.sendcmd(f'DELE {path}')


def remote_size(ftp, path):
    """Size of a file on the console, or None. ftpsrv's SIZE is not relied on: LIST the folder."""
    lines = []
    try:
        ftp.retrlines(f'LIST {posixpath.dirname(path)}', lines.append)
    except ftplib.Error:
        return None
    name = posixpath.basename(path)
    for line in lines:
        parts = line.split(None, 8)
        if len(parts) == 9 and parts[8] == name and parts[4].isdigit():
            return int(parts[4])
    return None


def request_consumed(ftp, name, timeout):
    """Wait until the running app takes request file `name` (it checks once a second)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if remote_size(ftp, f'{DATA}/{name}') is None:
            return True
        time.sleep(1)
    return False


def worker_file(address, port):
    """What fleet-worker.txt says: bind the console's own address, port below 40000."""
    return f'{address}:{port}\n{ACK}\n'


def make_worker(address, port=WORKER_PORT, env=None):
    """Write fleet-worker.txt (and `env`'s lines of env.txt) and ask a running PS5LM
    to restart (unload) into worker mode. Returns True when the app took the
    request; False when PS5LM is not open, in which case it starts as a worker
    when it is next opened."""
    with console_ftp(address) as ftp:
        delete(ftp, ARGS_FILE)            # a model replica's arguments would not apply to a worker
        for stale in ('leave-worker', 'quit', 'load'):   # left behind, they would undo this at once
            delete(ftp, f'{DATA}/{stale}')
        if env:
            set_env(ftp, env)
        put_text(ftp, WORKER_FILE, worker_file(address, port))
        put_text(ftp, f'{DATA}/unload', '-\n')
        if request_consumed(ftp, 'unload', TAKE_SECONDS):
            return True
        delete(ftp, f'{DATA}/unload')     # a worker never reads it; the app would later
        return False


def leave_worker(address):
    """Take a console out of worker mode. Returns 'restarted' when the app took the
    leave-worker request and restarts as the app, 'quit' when only an older build's
    quit was possible (then PS5LM must be opened on the console)."""
    with console_ftp(address) as ftp:
        delete(ftp, f'{DATA}/unload')
        put_text(ftp, f'{DATA}/leave-worker', '-\n')
        if request_consumed(ftp, 'leave-worker', TAKE_SECONDS):
            return 'restarted'
        # A build without leave-worker: forget the config and quit instead.
        delete(ftp, f'{DATA}/leave-worker')
        delete(ftp, WORKER_FILE)
        put_text(ftp, f'{DATA}/quit', 'quit\n')
        return 'quit'


def restart(address, env=None):
    """Ask a running PS5LM to restart as it is, after setting `env`'s lines of
    env.txt: a worker as a worker (PS5LM after v0.11.0), the app with the model
    app-args.txt names. False when the app did not take the request (closed, or a
    worker of an older build)."""
    with console_ftp(address) as ftp:
        for stale in ('quit', 'load', 'leave-worker'):
            delete(ftp, f'{DATA}/{stale}')
        if env:
            set_env(ftp, env)
        put_text(ftp, f'{DATA}/unload', '-\n')
        if request_consumed(ftp, 'unload', TAKE_SECONDS):
            return True
        delete(ftp, f'{DATA}/unload')
        return False


def write_env(address, env):
    """Set `env`'s lines of env.txt now; True when it changed (they apply at the next start)."""
    with console_ftp(address) as ftp:
        return set_env(ftp, env)


def quit_app(address):
    with console_ftp(address) as ftp:
        put_text(ftp, f'{DATA}/quit', 'quit\n')


ELF_LOADER_PORT = 9021      # elfldr (ps5-payload-dev), or an OnionHEN plugin of it


def launch_app(address, launcher, timeout=10):
    """Open PS5LM on a console from here: send the launcher payload (PS5LM's
    ps5/app/launch_app.c, built as build/app-ps5/launch-PPSA99575.elf) to the
    console's ELF loader, which runs it; it asks the system to start the title
    for the foreground user. False when no loader listens or the file is missing."""
    launcher = Path(launcher)
    if not launcher.is_file() or port_probe(address, ELF_LOADER_PORT) != 'open':
        return False
    with socket.create_connection((address, ELF_LOADER_PORT), timeout=timeout) as sock:
        sock.sendall(launcher.read_bytes())
    return True


def replica_args(ps5_path, context, slots, cache_type, flash_attn, unified=False):
    """llama-server arguments for a console serving a model by itself, one per line
    in app-args.txt. The planner's settings for the console (ubatch 64, one
    checkpoint, no RAM prompt cache), with the pool's context, slots and cache."""
    args = ['-m', ps5_path, '-ngl', '999', '-fit', 'off', '-lm', 'none',
            '-c', str(context * (1 if unified else slots)), '-np', str(slots),
            '-fa', 'on' if flash_attn else 'off', '-ctk', cache_type, '-ctv', cache_type,
            '-b', '256', '-ub', '64', '-ctxcp', '1', '-cram', '0', '--host', '0.0.0.0', '--port', str(APP_PORT)]
    return args + (['-kvu'] if unified else [])


def serve_model(address, args, worker=False):
    """Have the console serve a model with these llama-server arguments: write
    app-args.txt and ask the app to restart; it then loads them. A worker is
    asked to leave worker mode instead, which restarts it the same way (PS5LM
    builds with leave-worker; returns False when the app did not take it)."""
    with console_ftp(address) as ftp:
        for stale in ('quit', 'load', 'unload', 'leave-worker'):
            delete(ftp, f'{DATA}/{stale}')
        put_text(ftp, ARGS_FILE, '\n'.join(args) + '\n')
        if not worker:
            # Not open: PS5LM loads app-args.txt when it is next opened.
            delete(ftp, WORKER_FILE)
            put_text(ftp, f'{DATA}/unload', '-\n')
            if request_consumed(ftp, 'unload', TAKE_SECONDS):
                return True
            delete(ftp, f'{DATA}/unload')
            return False
        put_text(ftp, f'{DATA}/leave-worker', '-\n')
        if request_consumed(ftp, 'leave-worker', TAKE_SECONDS):
            return True
        delete(ftp, f'{DATA}/leave-worker')
        return False


def stop_serving(address):
    """Stop the console's model: no more app-args.txt, and the app restarts with no model."""
    with console_ftp(address) as ftp:
        delete(ftp, ARGS_FILE)
        put_text(ftp, f'{DATA}/unload', '-\n')
        if not request_consumed(ftp, 'unload', TAKE_SECONDS):
            delete(ftp, f'{DATA}/unload')


def push_file(address, local, remote, progress=None, block=1 << 20):
    """Copy a model to the console: to remote + '.part', then renamed, so a cut
    copy never looks complete. progress(sent, total) is called as it goes."""
    total = Path(local).stat().st_size
    sent = 0

    def count(chunk):
        nonlocal sent
        sent += len(chunk)
        if progress:
            progress(sent, total)

    with console_ftp(address, timeout=120) as ftp:
        with contextlib.suppress(ftplib.error_perm):
            ftp.mkd(posixpath.dirname(remote))
        with open(local, 'rb') as stream:
            ftp.storbinary(f'STOR {remote}.part', stream, blocksize=block, callback=count)
        delete(ftp, remote)
        ftp.sendcmd(f'RNFR {remote}.part')
        ftp.sendcmd(f'RNTO {remote}')
        if remote_size(ftp, remote) != total:
            raise OSError(f'{remote} on the console is not {total} bytes after the copy')


def console_has(address, remote, size):
    with console_ftp(address) as ftp:
        return remote_size(ftp, remote) == size


# --- local processes -------------------------------------------------------------------

class Process:
    """A child process with its last output lines."""

    def __init__(self, argv, name, lines=400):
        self.name = name
        self.argv = argv
        self.lines = collections.deque(maxlen=lines)
        self.started = time.time()
        self.proc = subprocess.Popen(argv, env=controller.child_environment(), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                                     errors='replace', bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip('\n'))

    def running(self):
        return self.proc.poll() is None

    def stop(self, timeout=15):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def local_devices(binary, timeout=60):
    """This computer's ggml devices with free memory: {'CUDA0': (total MiB, free MiB), ...}."""
    try:
        result = subprocess.run([str(binary), '--list-devices'], capture_output=True, text=True,
                                timeout=timeout, env=controller.child_environment())
    except (OSError, subprocess.TimeoutExpired):
        return {}
    found = {}
    for name, total, free in re.findall(r'^\s*([A-Za-z]+[0-9]*): .*\((\d+) MiB, (\d+) MiB free\)\s*$',
                                        result.stdout, re.MULTILINE):
        found[name] = (int(total), int(free))
    return found


def free_port(start=18100, end=18999):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise OSError('No free local port for a server')
