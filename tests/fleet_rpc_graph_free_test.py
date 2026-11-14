#!/usr/bin/env python3
"""Regression for patches/fleet/0006: an RPC worker gives back the graphs it is told to free.

A worker keeps every graph a host sends (to compute it again by uid) until 1,024
have come, and a host sends a new one for each prompt batch whose shape differs
from the last. A PS5 worker serving a 27B model lost about 4 MiB per 512-token
batch that way and died after about 175K prompt tokens.

Starts one loopback CPU worker and speaks the wire protocol to it, never model
data: every graph is one ADD over a 4 KiB buffer, listed with 4,000 tensors so
that its stored copy is about 1.4 MiB.
  1. HELLO's fourth byte (padding before the patch) advertises GRAPH_FREE.
  2. 200 graphs, each followed by a GRAPH_FREE for the one 8 before it: the
     worker's resident memory grows by less than 48 MiB, a kept graph still
     recomputes, and a free of a uid it never had is not an error.
  3. The same 200 graphs without GRAPH_FREE, as a host without the patch sends
     them: the worker grows by more than 150 MiB, which shows that step 2
     measures what it claims to.
An unpatched worker fails step 1.
"""
import argparse
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

HELLO, DEVICE_COUNT, ALLOC_BUFFER, BUFFER_GET_BASE = 14, 15, 0, 3
GRAPH_COMPUTE, GRAPH_RECOMPUTE, GRAPH_FREE = 10, 16, 23
OP_ADD = 2
GRAPHS, KEEP, TENSORS = 200, 8, 4000


def receive(sock, length):
    result = bytearray()
    while len(result) < length:
        chunk = sock.recv(length-len(result))
        if not chunk:
            raise AssertionError('RPC closed before the response completed')
        result.extend(chunk)
    return bytes(result)


def send(sock, command, payload):
    sock.sendall(struct.pack('<BQ', command, len(payload)) + payload)


def request(sock, command, payload, expected_size):
    send(sock, command, payload)
    size, = struct.unpack('<Q', receive(sock, 8))
    assert size == expected_size, (size, expected_size)
    return receive(sock, size)


def tensor(ident, buffer, data, op=0, src=()):
    """rpc_tensor: one F32 vector of 16, packed as ggml-rpc.cpp declares it (296 bytes)."""
    sources = list(src) + [0]*(10-len(src))
    packed = struct.pack('<QIQ4I4II16ii10QQQQ64si', ident, 0, buffer, 16, 1, 1, 1, 4, 64, 64, 64, op,
                         *([0]*16), 0, *sources, 0, 0, data, b't%d' % ident, 0)
    assert len(packed) == 296
    return packed


def graph(uid, buffer, base):
    a, b, c = 1, 2, 3
    tensors = [tensor(a, buffer, base), tensor(b, buffer, base+64), tensor(c, buffer, base+128, OP_ADD, (a, b))]
    # listed but not reached from the node: they only size the worker's copy
    tensors += [tensor(ident, 0, 0) for ident in range(4, TENSORS+1)]
    return struct.pack('<IQIQI', 0, uid, 1, c, len(tensors)) + b''.join(tensors)


def resident_mib(pid):
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('VmRSS'):
            return int(line.split()[1])/1024
    raise AssertionError('no VmRSS')


def connect(port):
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    sock.settimeout(30)
    hello = request(sock, HELLO, bytes(24), 28)
    assert hello[0] == 8, hello[:3]
    remote, _ = struct.unpack('<QQ', request(sock, ALLOC_BUFFER, struct.pack('<IQ', 0, 4096), 16))
    base, = struct.unpack('<Q', request(sock, BUFFER_GET_BASE, struct.pack('<Q', remote), 8))
    return sock, hello, remote, base


def send_graphs(sock, pid, remote, base, first_uid, free):
    # one graph first: the worker's threads and first allocations are not what is measured
    send(sock, GRAPH_COMPUTE, graph(first_uid, remote, base))
    assert request(sock, DEVICE_COUNT, b'', 4) == struct.pack('<I', 1)
    before = resident_mib(pid)
    for uid in range(first_uid+1, first_uid+GRAPHS):
        if free and uid-KEEP >= first_uid:
            send(sock, GRAPH_FREE, struct.pack('<IQ', 0, uid-KEEP))
        send(sock, GRAPH_COMPUTE, graph(uid, remote, base))
    # an answered request: everything before it has been handled
    assert request(sock, DEVICE_COUNT, b'', 4) == struct.pack('<I', 1)
    return resident_mib(pid)-before


def run(binary):
    chooser = socket.socket()
    chooser.bind(('127.0.0.1', 0))
    port = chooser.getsockname()[1]
    chooser.close()
    log_path = ROOT / 'build/fleet-smoke/graph-free.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log:
        worker = subprocess.Popen([str(binary), '--host', '127.0.0.1', '--port', str(port),
                                   '--device', 'CPU', '--threads', '1'],
                                  stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, GGML_RPC_NO_COMM='1', GGML_RPC_NO_RDMA='1'))
        try:
            deadline = time.monotonic()+20
            while True:
                if worker.poll() is not None:
                    raise AssertionError(f'worker exited {worker.returncode}; see {log_path}')
                try:
                    socket.create_connection(('127.0.0.1', port), timeout=0.1).close()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('worker did not bind its loopback port')
                    time.sleep(0.05)

            sock, hello, remote, base = connect(port)
            assert hello[3] & 1, 'the worker does not advertise GRAPH_FREE (patches/fleet/0006 missing?)'
            freed = send_graphs(sock, worker.pid, remote, base, 1000, free=True)
            assert freed < 48, f'the worker grew {freed:.0f} MiB although it was told to free the graphs'
            last = 1000+GRAPHS-1
            send(sock, GRAPH_RECOMPUTE, struct.pack('<IQ', 0, last))        # kept: still computes
            send(sock, GRAPH_FREE, struct.pack('<IQ', 0, 1000))             # freed already
            send(sock, GRAPH_FREE, struct.pack('<IQ', 0, 999999))           # never sent
            assert request(sock, DEVICE_COUNT, b'', 4) == struct.pack('<I', 1)
            sock.close()

            sock, _, remote, base = connect(port)
            kept = send_graphs(sock, worker.pid, remote, base, 5000, free=False)
            assert kept > 150, f'without GRAPH_FREE the worker grew only {kept:.0f} MiB: the test measures nothing'
            sock.close()
            assert worker.poll() is None
            print(f'RPC GRAPH_FREE: {GRAPHS} graphs grew the worker {freed:.1f} MiB when freed, {kept:.1f} MiB when kept')
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/fleet-host/bin/ggml-rpc-server')
    run(parser.parse_args().binary.resolve())
