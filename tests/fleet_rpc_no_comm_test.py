#!/usr/bin/env python3
"""Regression for the pinned v8 RPC server's server-side NO_COMM gate.

Starts one loopback CPU worker. Sends only valid HELLO/COMM_INIT/DEVICE_COUNT
frames, never model data. The requested extra port is already reserved, so even
an unpatched worker cannot open a wildcard listener there; rank 1 only targets
loopback. This is a narrow wire regression, not a general RPC implementation.
"""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def receive(sock, length):
    result = bytearray()
    while len(result) < length:
        chunk = sock.recv(length-len(result))
        if not chunk:
            raise AssertionError('RPC closed before the response completed')
        result.extend(chunk)
    return bytes(result)


def request(sock, command, payload, expected_size):
    sock.sendall(struct.pack('<BQ', command, len(payload)) + payload)
    size, = struct.unpack('<Q', receive(sock, 8))
    assert size == expected_size, (size, expected_size)
    return receive(sock, size)


def run(binary):
    with ExitStack() as stack:
        # Reserve a separate port for the rejected COMM_INIT. A bound loopback
        # socket (no SO_REUSEADDR) excludes wildcard bind on that same port.
        reserved = stack.enter_context(socket.socket())
        reserved.bind(('127.0.0.1', 0))
        comm_port = reserved.getsockname()[1]
        chooser = socket.socket()
        chooser.bind(('127.0.0.1', 0))
        port = chooser.getsockname()[1]
        chooser.close()
        log_path = ROOT / 'build/fleet-smoke/no-comm.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = stack.enter_context(log_path.open('w'))
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
                    sock = stack.enter_context(socket.create_connection(('127.0.0.1', port), timeout=0.1))
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('worker did not bind its loopback port')
                    time.sleep(0.05)
            sock.settimeout(1)
            hello = request(sock, 14, bytes(24), 28)
            assert hello[:3] == bytes([8, 0, 0]), hello[:3]
            for rank in (0, 1):
                body = struct.pack('<IIII64s', 0, rank, 2, comm_port, b'127.0.0.1')
                assert request(sock, 20, body, 1) == b'\0', f'rank {rank} was not refused'
                # The refusal must leave this connection and normal RPC intact.
                assert request(sock, 15, b'', 4) == struct.pack('<I', 1)
            assert worker.poll() is None
            print('RPC NO_COMM: rank 0 and rank 1 refused; ordinary RPC remains usable; no extra listeners')
        finally:
            # Closing the connection before terminating avoids a blocked read.
            if 'sock' in locals():
                sock.close()
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
