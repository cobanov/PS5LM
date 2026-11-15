#!/usr/bin/env python3
"""Device parking (patches/fleet/0009, patches/0015, patches/0016), with real CPU RPC.

One loopback ggml-rpc-server with a park folder and llama-server with every
layer on it and --slot-park-path. A slot's state is parked on the worker under a
key and resumed from there; this machine keeps only the tokens and a manifest.
Checked here:

  swap      save, another conversation takes the slot, restore: the next turn
            re-reads only its new tokens and answers exactly as without the swap
  firings   five firings that each add tokens: each save writes only the new
            cells plus one recurrent state, and a restore from the five files is
            exact; a request that stops short of the parked end goes back to the
            save before, or with "checkpoint": true re-reads a handful of tokens
  many      several keys parked at once, resumed in another order
  restart   resumed after a server restart, and after a worker restart
  torn      the worker dies in the middle of a save: the previous state loads
  stale     the history was changed before the parked end: the key is rewritten
            from the change, never resumed as it was
  prune     a small cap drops the least recently used key, and the server says so
  mixed     a worker without parking, and with --old-build an older host or worker
  busy      a save or a restore sent while the slot generates, or with a request
            queued behind it, waits its turn in the order it came; two saves of
            one key, and a save racing a restore of it, leave the key whole
  crash     the server dies with a save written but its manifest not yet
            replaced, in an append and in a rewrite: the state before loads
  memory    the worker's memory does not grow with the size of what it parks

By default the model is the untrained llama fixture (attention only). With
--model it is a real GGUF: Qwen3.5-0.8B has the hybrid
layout of Qwen3.8-27B (attention cells plus a recurrent state), which is what
the consoles serve.

This is the host's CPU, not a PS5. Run after scripts/build-fleet-host.sh.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import os
import random
import resource
import shutil
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

from fleet_rpc_smoke import free_ports, spawn, stop, wait_ready
from make_fleet_fixture import ROOT, make_fixture

REPLY = 16          # tokens written per turn
KEPT = REPLY - 1    # of which the slot holds all but the last: it is sampled, and read only by a next turn
WORDS = ("stone river lamp garden orbit copper window signal harbor maple thread candle "
         "meadow anchor pepper violin summit ribbon lantern canyon").split()


def resident(pid):
    """A process's anonymous memory now and its peak resident memory, in MiB."""
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("RssAnon:", "VmHWM:")):
            out[line.split(":")[0]] = int(line.split()[1]) / 1024
    return out["RssAnon"], out["VmHWM"]


class Rig:
    """One worker and one server, started and stopped as the cases need."""

    def __init__(self, args, output):
        self.args, self.out = args, output
        self.build = args.build
        self.worker_bin = args.build / "bin/ggml-rpc-server"
        self.server_bin = args.build / "bin/llama-server"
        self.model = args.model
        self.real = args.model is not None
        if not self.real:
            self.model = make_fixture(output / "park-llama.gguf", context=args.ctx)
        self.park_dir = output / "worker-park"
        self.host_dir = output / "host-park"
        self.host_dir.mkdir(parents=True)
        self.worker_port, self.api = free_ports(2)
        self.env = dict(os.environ, GGML_RPC_NO_RDMA="1", GGML_RPC_NO_COMM="1", CUDA_VISIBLE_DEVICES="-1")
        self.stack = None
        self.worker = self.server = None
        self.n_start = 0

    # -- processes --

    def start(self, park=True, cap_gb=None, worker_env=None, worker_bin=None, server_bin=None, server_park=True, server_env=None):
        self.stack = ExitStack()
        self.n_start += 1
        env = dict(self.env)
        if park:
            env.update(GGML_RPC_PARK_DIR=str(self.park_dir), GGML_RPC_PARK_RESERVE_GB="0")
            if cap_gb is not None:
                env["GGML_RPC_PARK_GB"] = repr(cap_gb)
        env.update(worker_env or {})
        threads = str(self.args.threads)
        self.worker = spawn(self.stack, [str(worker_bin or self.worker_bin), "--host", "127.0.0.1", "--port", str(self.worker_port),
                                         "--device", "CPU", "--threads", threads], self.out / f"worker-{self.n_start}.log", env)
        wait_ready(self.worker, self.worker_port)
        command = [str(server_bin or self.server_bin), "--model", str(self.model), "--host", "127.0.0.1", "--port", str(self.api),
                   "--ctx-size", str(self.args.ctx), "--parallel", "1", "--threads", threads, "--threads-batch", threads,
                   "--batch-size", "2048", "--ubatch-size", "512", "--fit", "off", "--no-warmup", "--no-op-offload",
                   "--no-context-shift", "--cache-ram", "0", "--ctx-checkpoints", "3", "--slots",
                   "--rpc", f"127.0.0.1:{self.worker_port}", "--device", "RPC0", "--gpu-layers", "all"]
        if self.real:
            command += ["-no-kvu", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--flash-attn", "on"]
        else:
            command += ["--cache-type-k", "f16", "--cache-type-v", "f16", "--flash-attn", "off", "--load-mode", "mmap", "--lazy-mode", "off"]
        if server_park:
            command += ["--slot-park-path", str(self.host_dir)]
        self.server = spawn(self.stack, command, self.out / f"server-{self.n_start}.log", dict(self.env, **(server_env or {})))
        wait_ready(self.server, self.api, http=True)

    def stop(self):
        if self.stack is not None:
            self.stack.close()
            self.stack = None

    def restart_server(self):
        """The server only: the worker and its files stay."""
        stop(self.server)
        self.n_start += 1
        command = self.server.args
        stream = self.stack.enter_context((self.out / f"server-{self.n_start}.log").open("w"))
        import subprocess
        self.server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=self.env, start_new_session=True)
        self.stack.callback(stop, self.server)
        wait_ready(self.server, self.api, http=True)

    # -- requests --

    def call(self, path, body=None, timeout=1800):
        data = json.dumps(body).encode() if body is not None else None    # no body: a GET
        request = urllib.request.Request(f"http://127.0.0.1:{self.api}{path}", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def slot(self, action, key=None, expect=200, **more):
        status, body = self.call(f"/slots/0?action={action}", dict({} if key is None else {"filename": key}, **more) or {})
        assert status == expect, (action, key, status, body)
        return body if status == 200 else body["error"]

    def turn(self, tokens, n=REPLY):
        """One greedy turn on the slot: the reply's tokens, and how many prompt tokens were read rather than reused."""
        status, body = self.call("/completion", {"prompt": tokens, "n_predict": n, "temperature": 0, "seed": 42,
                                                 "ignore_eos": True, "return_tokens": True, "cache_prompt": True, "id_slot": 0})
        assert status == 200, body
        assert len(body["tokens"]) == n, body
        return body["tokens"], body["timings"]["prompt_n"]

    def background(self, function, *args, **kwargs):
        """Runs a request on a thread of its own: .result (or .error) and .done, the time it answered."""
        def run():
            try:
                thread.result = function(*args, **kwargs)
            except BaseException as error:   # reported by the case that joins it
                thread.error = error
            thread.done = time.monotonic()
        thread = threading.Thread(target=run, daemon=True)
        thread.result = thread.error = thread.done = None
        thread.start()
        return thread

    def generating(self):
        status, body = self.call("/slots", None)
        return status == 200 and body[0]["is_processing"]

    def wait_generating(self):
        deadline = time.monotonic() + 60
        while not self.generating():
            assert time.monotonic() < deadline, "the long generation did not start"
            time.sleep(0.02)

    def history(self, seed, n):
        """n tokens of a conversation of its own."""
        rng = random.Random(seed)
        if not self.real:
            return [1] + [rng.randrange(3, 259) for _ in range(n - 1)]
        tokens = []
        while len(tokens) < n:
            text = " ".join(rng.choice(WORDS) for _ in range(n))
            status, body = self.call("/tokenize", {"content": text, "add_special": not tokens})
            assert status == 200, body
            tokens += body["tokens"]
        return tokens[:n]

    def device_files(self, key):
        folder = self.park_dir / key
        return {f.name: f.stat().st_size for f in folder.iterdir()} if folder.is_dir() else {}

    def host_bytes(self, key):
        return sum(f.stat().st_size for f in (self.host_dir / key).iterdir())


def case_swap(rig, report):
    """Save, another conversation takes the slot, restore; and what a restore costs when the request stops before the parked end."""
    a, b, more = rig.history(11, rig.args.tokens), rig.history(12, rig.args.tokens), rig.history(13, 40)
    # without a swap: the same two turns on a slot nothing else touches
    rig.slot("erase")
    ref1, _ = rig.turn(a)
    ref2, ref_read = rig.turn(a + ref1 + more)
    rig.slot("erase")

    r1, _ = rig.turn(a)
    assert r1 == ref1
    saved = rig.slot("save", "swap-a")
    assert saved["mode"] == "new" and saved["n_saved"] == len(a) + KEPT and saved["n_appended"] == saved["n_saved"], saved
    assert saved["n_written"] == saved["n_written_kv"] + saved["n_written_state"] + saved["n_written_checkpoint"] > 0, saved
    rig.turn(b)                                   # another conversation takes the slot
    rig.slot("save", "swap-b")
    restored = rig.slot("restore", "swap-a")
    assert restored["n_restored"] == saved["n_saved"] and restored["n_read"] == saved["n_parked_bytes"], (saved, restored)
    r2, read = rig.turn(a + r1 + more)
    assert r2 == ref2, "the turn after a restore answers differently than without the swap"
    assert read == ref_read, f"read {read} prompt tokens after the restore, {ref_read} without the swap"

    # the same history again, as a client that does not send the last reply back: the request ends
    # REPLY tokens before the parked end. A recurrent state cannot be rewound, so this needs the checkpoint.
    rig.slot("restore", "swap-a")
    again, read_back = rig.turn(a)
    assert again == r1
    if saved["n_written_state"] > 0:
        # a first save has no earlier save to go back to, so it parks the slot's own checkpoint
        assert saved["checkpoint_kind"] == "slot_checkpoint", "a model with a recurrent state was parked without a checkpoint"
        assert read_back <= 8, f"a return that stops before the parked end re-read {read_back} tokens"
    assert rig.host_bytes("swap-a") < 64 * 1024 + 40 * saved["n_saved"], "bookkeeping on this machine is not small"
    report["swap"] = dict(tokens=saved["n_saved"], device_bytes=saved["n_parked_bytes"], host_bytes=rig.host_bytes("swap-a"),
                          read_after_restore=read, read_without_swap=ref_read, read_when_request_ends_early=read_back,
                          fixed_bytes=saved["n_written_state"] + saved["n_written_checkpoint"],
                          save_ms=saved["timings"]["save_ms"], restore_ms=restored["timings"]["restore_ms"])


def case_firings(rig, report):
    """Five firings that each append: only the new cells are written, and the chain restores exactly."""
    step = rig.args.firing
    other = rig.history(20, rig.args.tokens)
    adds = [rig.history(21 + i, step) for i in range(5)]
    rig.slot("erase")
    reference, history = [], []
    for add in adds:                               # without parking
        history = history + add
        reply, _ = rig.turn(history)
        reference.append(reply)
        history = history + reply
    rig.slot("erase")

    history, saves = [], []
    for i, add in enumerate(adds):
        if i > 0:
            restored = rig.slot("restore", "fire")
            assert restored["n_restored"] == len(history) - 1 and restored["n_segments"] == i, restored
        history = history + add
        reply, read = rig.turn(history)
        assert reply == reference[i], f"firing {i + 1} answers differently than without parking"
        assert read <= step + 1, f"firing {i + 1} re-read {read} tokens for {step} new ones"
        history = history + reply
        saved = rig.slot("save", "fire")
        assert saved["mode"] == ("new" if i == 0 else "append") and saved["n_segments"] == i + 1, saved
        assert saved["n_appended"] == step + (REPLY if i else KEPT) and saved["n_saved"] == len(history) - 1, saved
        saves.append(saved)
        rig.turn(other)                            # another conversation takes the slot
    per_token = saves[0]["n_written_kv"] / saves[0]["n_appended"]
    fixed = saves[1]["n_written"] - saves[1]["n_written_kv"]
    recurrent = saves[0]["n_written_state"] > 0
    for i, saved in enumerate(saves):
        # the cells of the tokens this firing added, give or take the token a reply ends on (written, not yet read)
        assert abs(saved["n_written_kv"] - per_token * saved["n_appended"]) <= 2 * per_token, (i, saved, per_token)
        # and one recurrent state: after the first save nothing else, the save before is the state to go back to
        assert saved["n_written_state"] == saves[0]["n_written_state"], (i, saved)
        if i > 0:
            assert saved["n_written"] - saved["n_written_kv"] == fixed and saved["n_written_checkpoint"] == 0, (i, saved, fixed)
            assert saved["checkpoint_kind"] == ("previous_save" if recurrent else None), (i, saved)
    assert saves[-1]["n_written"] < saves[-1]["n_parked_bytes"] / 3 or per_token * step < fixed, saves[-1]
    assert sorted(f for f in rig.device_files("fire") if f.endswith(".kv")) == [f"{n:06d}.kv" for n in range(1, 6)], rig.device_files("fire")
    assert sum(not f.endswith(".kv") for f in rig.device_files("fire")) <= 2, rig.device_files("fire")   # the last two recurrent states

    # A request that stops short of the parked end (the client does not send the last reply back as it was
    # written): the state to go back to is the save before, so the last firing is read again. Exact, not cheap.
    rig.slot("restore", "fire")
    again, back_previous = rig.turn(history[:-REPLY])
    assert again == reference[4], "a request that stops short of the parked end answers differently"
    back_slot = None
    if recurrent:
        assert 8 < back_previous <= step + REPLY + 8, f"went back {back_previous} tokens, the last firing added {step}"
        # with the slot's own checkpoint parked as well, it is a handful of tokens: one more block per save
        more = rig.history(29, 40)
        rig.slot("restore", "fire")
        reply6, _ = rig.turn(history + more)
        with_ck = rig.slot("save", "fire", checkpoint=True)
        assert with_ck["checkpoint_kind"] == "slot_checkpoint" and with_ck["n_written_checkpoint"] > 0, with_ck
        assert with_ck["n_written_state"] == saves[0]["n_written_state"], with_ck
        rig.turn(other)
        rig.slot("restore", "fire")
        again6, back_slot = rig.turn(history + more)
        assert again6 == reply6 and back_slot <= 8, (back_slot, "with its checkpoint parked a short request re-reads too much")
    rig.turn(other)
    assert rig.slot("save", "fire", expect=200)["mode"] in ("rewrite", "new")   # the slot holds `other` now: not this key's history
    report["firings"] = dict(tokens_per_firing=step + REPLY, bytes_per_token=per_token, fixed_bytes_per_save=fixed,
                             fixed_bytes_first_save=saves[0]["n_written"] - saves[0]["n_written_kv"],
                             reread_when_request_stops_short=dict(previous_save=back_previous, slot_checkpoint=back_slot),
                             written=[s["n_written"] for s in saves], written_cells=[s["n_written_kv"] for s in saves],
                             parked_bytes=saves[-1]["n_parked_bytes"], save_ms=[round(s["timings"]["save_ms"], 1) for s in saves])


def case_many(rig, report):
    """Several keys parked at once, resumed in another order; then after a server restart and a worker restart."""
    n = rig.args.tokens
    keys = {f"many-{i}": rig.history(40 + i, n + 37 * i) for i in range(4)}
    more = rig.history(49, 24)
    first, reference = {}, {}
    for key, tokens in keys.items():
        rig.slot("erase")
        first[key], _ = rig.turn(tokens)
        reference[key], _ = rig.turn(tokens + first[key] + more)
    rig.slot("erase")
    for key, tokens in keys.items():
        reply, _ = rig.turn(tokens)
        assert reply == first[key]
        rig.slot("save", key)
    listed = {k["key"]: k for k in rig.slot("park-list")["keys"]}
    for key, tokens in keys.items():
        assert listed[key]["state"] == "parked" and listed[key]["n_tokens"] == len(tokens) + KEPT, listed[key]

    def resume(key):
        restored = rig.slot("restore", key)
        assert restored["n_restored"] == len(keys[key]) + KEPT, restored
        reply, read = rig.turn(keys[key] + first[key] + more)
        assert reply == reference[key], f"{key} answers differently after its restore"
        assert read <= len(more) + 1, f"{key} re-read {read} tokens"

    order = ["many-2", "many-0", "many-3", "many-1"]
    for key in order:
        resume(key)
    rig.restart_server()
    resume("many-1")
    resume("many-3")
    rig.stop()
    rig.start()
    resume("many-0")
    resume("many-2")
    report["many"] = dict(keys=len(keys), order=order, after_server_restart=["many-1", "many-3"], after_worker_restart=["many-0", "many-2"])


def case_torn(rig, report):
    """The worker dies in the middle of a save, in the cells and in the recurrent state: the previous state loads."""
    base, add, more = rig.history(60, rig.args.tokens), rig.history(61, 300), rig.history(62, 24)
    rig.slot("erase")
    r1, _ = rig.turn(base)
    ref2, _ = rig.turn(base + r1 + more)
    rig.slot("erase")
    rig.turn(base)
    saved = rig.slot("save", "torn")
    files = rig.device_files("torn")
    died = []
    for where in ("kv:4096", "rs:64", "ck:64"):
        if where != "kv:4096" and saved["n_written_state"] == 0:
            continue                                # the fixture has no recurrent state and no checkpoint
        rig.stop()
        rig.start(worker_env={"GGML_RPC_PARK_DIE_AT": where})
        rig.slot("restore", "torn")
        rig.turn(base + r1 + add)
        try:
            status, body = rig.call("/slots/0?action=save", {"filename": "torn", "checkpoint": True}, timeout=120)
            raise AssertionError(f"the save answered {status} {body} although the worker died in it")
        except (urllib.error.URLError, ConnectionError, socket.timeout, json.JSONDecodeError):
            pass                                    # the worker is gone and the server with it, as for any dead worker
        assert rig.worker.wait(timeout=30) != 0
        died.append(where)
        rig.stop()
        rig.start()
        assert not any(f.endswith(".part") for f in rig.device_files("torn")), rig.device_files("torn")
        restored = rig.slot("restore", "torn")
        assert restored["n_restored"] == saved["n_saved"], restored
        reply, read = rig.turn(base + r1 + more)
        assert reply == ref2 and read <= len(more) + 1, (read, "the state before the torn save does not answer as before")
    # and the key still takes a save: what the torn ones left on the worker is cleared away
    rig.slot("restore", "torn")
    rig.turn(base + r1 + add)
    after = rig.slot("save", "torn")
    assert after["mode"] == "append" and after["n_appended"] == len(add) + REPLY, after
    left = rig.device_files("torn")
    assert sum(f.endswith(".kv") for f in left) == 2 and len(left) <= 4, left
    assert set(files) <= set(left) | {"000001.rs", "000001.ck"}, (files, left)
    report["torn"] = dict(worker_died_in=died, restored_tokens=saved["n_saved"])


def case_stale(rig, report):
    """The history is changed before the parked end: the save rewrites the key from there; a wrong restore cannot happen."""
    step = rig.args.firing
    parts = [rig.history(70 + i, step) for i in range(3)]
    changed = rig.history(79, step)
    rig.slot("erase")
    history = []
    for part in parts:                              # three firings, three files
        history = history + part
        reply, _ = rig.turn(history)
        history = history + reply
        rig.slot("save", "stale")
    # the client compacts: the second firing's text is replaced, everything after it changes
    edited = history[:step + REPLY] + changed
    rig.slot("erase")
    ref, _ = rig.turn(edited)
    ref2, _ = rig.turn(edited + ref + parts[2][:50])
    # the parked state is of the old history: a restore gives the old tokens, and the server reuses only
    # what the new request shares with them
    restored = rig.slot("restore", "stale")
    assert restored["n_restored"] == len(history) - 1, restored
    reply, read = rig.turn(edited)
    assert reply == ref, "an edited history answers differently on top of a restore of the old one"
    # (every fixture history starts with the same token, so the first token of the change can be shared)
    assert read >= len(changed) - 1, f"only {read} tokens read: state of the old history was reused past the edit"
    saved = rig.slot("save", "stale")
    assert saved["mode"] == "rewrite" and saved["rewritten_from"] <= step + REPLY and saved["n_saved"] == len(edited) + KEPT, saved
    assert saved["n_appended"] == saved["n_saved"] - saved["rewritten_from"], saved
    assert saved["n_segments"] == 2, saved         # the first firing's file is kept, one new file replaces the other two
    rig.turn(rig.history(78, rig.args.tokens))
    restored = rig.slot("restore", "stale")
    assert restored["n_restored"] == len(edited) + KEPT, restored
    reply2, read = rig.turn(edited + reply + parts[2][:50])
    assert reply2 == ref2 and read <= 51, (read, "the rewritten key does not answer as the edited history does")
    report["stale"] = dict(rewritten_from=saved["rewritten_from"], kept_files=1, written=saved["n_written"])


def case_prune(rig, report):
    """A cap that holds two keys: the third save drops the one used longest ago; the server says so, and a restore of it is refused."""
    n = rig.args.tokens
    rig.slot("erase")
    rig.turn(rig.history(80, n))
    size = rig.slot("save", "prune-size")["n_parked_bytes"]
    rig.slot("park-delete", "prune-size")
    for key in [k["key"] for k in rig.slot("park-list")["keys"]]:
        rig.slot("park-delete", key)
    rig.stop()
    rig.start(cap_gb=2.6 * size / 2**30)
    status = rig.slot("park-status")
    assert status["n_keys"] == 0 and abs(status["cap_bytes"] - 2.6 * size) < 4096, status
    for i, key in enumerate(("prune-1", "prune-2")):
        rig.turn(rig.history(81 + i, n))
        assert rig.slot("save", key)["pruned"] == []
        time.sleep(1.1)                             # the keys' times of last use differ
    rig.slot("restore", "prune-1")                  # used now: prune-2 is the least recently used
    time.sleep(1.1)
    rig.turn(rig.history(83, n))
    saved = rig.slot("save", "prune-3")
    assert saved["pruned"] == ["prune-2"], saved
    listed = {k["key"]: k["state"] for k in rig.slot("park-list")["keys"]}
    assert listed == {"prune-1": "parked", "prune-2": "pruned", "prune-3": "parked"}, listed
    refused = rig.slot("restore", "prune-2", expect=410)
    assert refused["park_error"] == "park_pruned", refused
    rig.slot("restore", "never-parked", expect=404)   # llama-server answers every 404 with its own "File Not Found" body
    # a key that does not fit at all is refused, and nothing in use is dropped for it
    rig.stop()
    rig.start(cap_gb=0.5 * size / 2**30)
    rig.turn(rig.history(84, n))
    full = rig.slot("save", "prune-4", expect=507)
    assert full["park_error"] == "park_no_space", full
    assert rig.device_files("prune-4") == {}, rig.device_files("prune-4")
    reply, _ = rig.turn(rig.history(84, n))         # the server and the slot are as before
    deleted = rig.slot("park-delete", "prune-1")
    assert deleted["deleted"] and rig.device_files("prune-1") == {} and not (rig.host_dir / "prune-1").exists(), deleted
    report["prune"] = dict(key_bytes=size, cap_bytes=int(2.6 * size), pruned=saved["pruned"])


LONG = 240   # tokens of a generation long enough to send other requests into


def joined(*threads):
    for thread in threads:
        thread.join(timeout=600)
        assert not thread.is_alive(), "a request did not answer"
        if thread.error is not None:
            raise thread.error
    return [thread.result for thread in threads]


def case_busy(rig, report):
    """Park actions against a slot that is generating or has a request queued: each waits its turn, in the order it came."""
    n = rig.args.tokens
    a, b, more, more2 = rig.history(110, n), rig.history(111, n), rig.history(112, 30), rig.history(113, 30)
    # what each history answers on a slot nothing else touches
    rig.slot("erase")
    a_long, _ = rig.turn(a, LONG)
    a2, _ = rig.turn(a + a_long + more)
    a3, _ = rig.turn(a + a_long + more + a2 + more2)
    rig.slot("erase")
    b_long, _ = rig.turn(b, LONG)
    b2, _ = rig.turn(b + b_long + more)
    rig.slot("erase")

    # 1. a save sent while the slot generates waits for the generation and parks what it left
    gen = rig.background(rig.turn, a, LONG)
    rig.wait_generating()
    save = rig.background(rig.slot, "save", "busy-a")
    (reply, _), saved = joined(gen, save)
    assert reply == a_long
    assert save.done >= gen.done - 0.05, "the save answered before the generation it was sent into had finished"
    assert saved["n_saved"] == len(a) + LONG - 1 and saved["mode"] == "new", saved

    # 2. a restore sent while the slot generates waits too: the generation is whole, then the slot holds the key
    gen = rig.background(rig.turn, b, LONG)
    rig.wait_generating()
    restore = rig.background(rig.slot, "restore", "busy-a")
    (reply, _), restored = joined(gen, restore)
    assert reply == b_long, "a generation was disturbed by a restore sent into it"
    assert restore.done >= gen.done - 0.05 and restored["n_restored"] == len(a) + LONG - 1, restored
    reply, read = rig.turn(a + a_long + more)
    assert reply == a2 and read <= len(more) + 1, (read, "the key restored after a generation does not answer as its history does")

    # 3. a save, then a request, both sent while the slot generates: they run in that order, and neither is left waiting
    rig.slot("erase")
    gen = rig.background(rig.turn, b, LONG)
    rig.wait_generating()
    save = rig.background(rig.slot, "save", "busy-b")
    time.sleep(0.3)
    queued = rig.background(rig.turn, a + a_long + more + a2 + more2)
    (reply, _), saved, (queued_reply, _) = joined(gen, save, queued)
    assert reply == b_long and queued_reply == a3
    assert saved["n_saved"] == len(b) + LONG - 1, ("the save parked the queued request's state, not the generation's", saved)
    assert gen.done - 0.05 <= save.done <= queued.done + 0.05, (gen.done, save.done, queued.done)
    rig.slot("restore", "busy-b")
    reply, read = rig.turn(b + b_long + more)
    assert reply == b2 and read <= len(more) + 1, read
    # and the other way round: the request first, the save after it, parks what the request left. (The requests
    # here name their slot, "id_slot": 0, as the swarm's do. One that names none is taken up after the park
    # actions that wait for the slot, whenever it came.)
    gen = rig.background(rig.turn, a, LONG)
    rig.wait_generating()
    queued = rig.background(rig.turn, b + b_long + more)
    time.sleep(0.3)
    save = rig.background(rig.slot, "save", "busy-c")
    joined(gen, queued, save)
    assert save.result["n_saved"] == len(b) + LONG + len(more) + KEPT, save.result
    assert queued.done - 0.05 <= save.done

    # the keys can be listed while the slot generates; a save whose client gives up while it waits may or may not run
    gen = rig.background(rig.turn, a, LONG)
    rig.wait_generating()
    listed = rig.slot("park-list")
    assert rig.generating(), "park-list waited for the generation"
    assert "busy-gone" not in [k["key"] for k in listed["keys"]]
    gave_up = False
    try:
        rig.call("/slots/0?action=save", {"filename": "busy-gone"}, timeout=0.2)
    except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError):
        gave_up = True                                    # the usual case: the generation takes seconds
    joined(gen)
    rig.turn(a + a_long + more)                           # the slot has been idle and busy again since
    # The server drops a waiting action when it notices its client has gone, which it looks for about once a
    # second: one whose turn comes sooner is run all the same. Either is fine here; the key is whole if it exists.
    if "busy-gone" in [k["key"] for k in rig.slot("park-list")["keys"]]:
        assert rig.slot("restore", "busy-gone")["n_restored"] == len(a) + LONG - 1
        rig.slot("park-delete", "busy-gone")

    # 4. two saves of one key at once: one parks, the other finds nothing to add
    rig.slot("restore", "busy-a")
    rig.turn(a + a_long + more)
    first, second = joined(rig.background(rig.slot, "save", "busy-a"), rig.background(rig.slot, "save", "busy-a"))
    assert sorted([first["mode"], second["mode"]]) == ["append", "unchanged"], (first, second)
    assert first["n_saved"] == second["n_saved"] == len(a) + LONG + len(more) + KEPT

    # 5. a save racing a restore of the same key: whichever runs first, the key is whole afterwards
    rig.turn(a + a_long + more + a2 + more2)             # the slot is one turn ahead of the key
    joined(rig.background(rig.slot, "save", "busy-a"), rig.background(rig.slot, "restore", "busy-a"))
    rig.turn(b)                                           # another conversation takes the slot
    restored = rig.slot("restore", "busy-a")
    ahead = len(a) + LONG + len(more) + REPLY + len(more2) + KEPT
    assert restored["n_restored"] in (len(a) + LONG + len(more) + KEPT, ahead), restored
    # either way the history one turn ahead answers as it should, reading at most that turn again
    reply, read = rig.turn(a + a_long + more + a2 + more2)
    assert reply == a3 and read <= len(more2) + REPLY + 8, read
    report["busy"] = dict(save_during_generation="waits, parks the generation's end", restore_during_generation="waits, generation whole",
                          order="save, restore and requests for the slot run in the order they were sent",
                          save_racing_restore="save first" if restored["n_restored"] == ahead else "restore first")


def case_crash(rig, report):
    """The server dies with a save written but the manifest not replaced, in an append and in a rewrite: the state before loads."""
    step = rig.args.firing
    parts = [rig.history(120 + i, step) for i in range(2)]
    add, changed, more = rig.history(125, 200), rig.history(126, step), rig.history(127, 24)
    rig.slot("erase")
    history = []
    for part in parts:                                    # two firings, two files
        history = history + part
        reply, _ = rig.turn(history)
        history = history + reply
    ref, _ = rig.turn(history + more)
    rig.slot("erase")
    history = []
    for part in parts:
        history = history + part
        reply, _ = rig.turn(history)
        history = history + reply
        saved = rig.slot("save", "crash")
    before = dict(rig.device_files("crash"))
    edited = history[:step + REPLY] + changed             # the second firing replaced: a save of this is a rewrite
    for name, request in (("append", history + add), ("rewrite", edited)):
        rig.stop()
        rig.start(server_env={"LLAMA_PARK_DIE_AT": "manifest"})
        rig.slot("restore", "crash")
        rig.turn(request)
        try:
            status, body = rig.call("/slots/0?action=save", {"filename": "crash"}, timeout=120)
            raise AssertionError(f"the save answered {status} {body} although the server died in it")
        except (urllib.error.URLError, ConnectionError, socket.timeout, json.JSONDecodeError):
            pass
        assert rig.server.wait(timeout=30) == 3, "the server did not stop where the test asked"
        assert set(before) < set(rig.device_files("crash")), (name, "the dying save left nothing on the worker: it did not get far enough")
        rig.stop()
        rig.start()
        restored = rig.slot("restore", "crash")
        assert restored["n_restored"] == saved["n_saved"] and restored["n_segments"] == 2, (name, restored)
        reply, read = rig.turn(history + more)
        assert reply == ref and read <= len(more) + 1, (name, read, "the state before the dying save does not answer as before")
    # and the key still takes the rewrite; what the dying saves left is cleared away
    rig.slot("restore", "crash")
    rig.turn(edited)
    after = rig.slot("save", "crash")
    assert after["mode"] == "rewrite" and after["n_segments"] == 2, after
    assert sum(f.endswith(".kv") for f in rig.device_files("crash")) == 2, rig.device_files("crash")
    here = sorted(f.name for f in (rig.host_dir / "crash").iterdir())
    assert len(here) == 3 and here[0] == "manifest.json", here
    report["crash"] = dict(server_died_before_manifest_in=["append", "rewrite"], restored_tokens=saved["n_saved"])


def hello_features(rig, park):
    """The fourth byte of a worker's HELLO answer; a worker of its own, as one serves a single client at a time."""
    port, = free_ports(1)
    env = dict(rig.env)
    if park:
        env["GGML_RPC_PARK_DIR"] = str(rig.out / "hello-park")
    with ExitStack() as stack:
        worker = spawn(stack, [str(rig.worker_bin), "--host", "127.0.0.1", "--port", str(port), "--device", "CPU", "--threads", "1"],
                       rig.out / f"worker-hello-{int(park)}.log", env)
        wait_ready(worker, port)
        return hello_byte(port)


def hello_byte(port):
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(struct.pack("<BQ", 14, 24) + bytes(24))
        data = b""
        while len(data) < 36:
            chunk = sock.recv(36 - len(data))
            assert chunk, "the worker closed the connection in HELLO"
            data += chunk
    assert struct.unpack_from("<Q", data)[0] == 28, data
    return data[11]


def case_mixed(rig, report):
    """Either side without parking still serves; only the park actions are refused, cleanly."""
    tokens = rig.history(90, rig.args.tokens)
    rig.slot("erase")
    reference, _ = rig.turn(tokens)
    rig.stop()
    seen = {}

    # this worker with no park folder: it does not offer the command, and the server keeps serving
    rig.start(park=False)
    assert rig.turn(tokens)[0] == reference
    refused = rig.slot("save", "mixed", expect=501)
    assert refused["park_error"] == "park_unsupported", refused
    assert rig.turn(tokens)[0] == reference
    seen["new host, worker without a park folder"] = "serves; save answers 501 park_unsupported"
    rig.stop()

    old = rig.args.old_build
    if old is not None:
        # an older worker (no park command at all)
        rig.start(worker_bin=old / "bin/ggml-rpc-server")
        assert rig.turn(tokens)[0] == reference
        assert rig.slot("save", "mixed", expect=501)["park_error"] == "park_unsupported"
        assert rig.slot("park-list", expect=501)["park_error"] == "park_unsupported"
        assert rig.turn(tokens)[0] == reference
        seen["new host, old worker"] = "serves; save and park-list answer 501 park_unsupported"
        rig.stop()
        # an older host against this worker with its park folder: it never sends the command
        rig.start(server_bin=old / "bin/llama-server", server_park=False)
        assert rig.turn(tokens)[0] == reference
        status, _ = rig.call("/slots/0?action=save", {"filename": "mixed"})
        assert status == 501, status
        assert rig.turn(tokens)[0] == reference
        seen["old host, new worker"] = "serves; has no park actions"
        rig.stop()

    # what a worker says in its handshake: bit 0 is the graph-free of patches/fleet/0006, bit 1 parking
    assert hello_features(rig, park=True) & 3 == 3
    assert hello_features(rig, park=False) & 3 == 1
    seen["handshake"] = "parking is offered only with a park folder"
    rig.start()
    report["mixed"] = seen


def case_memory(rig, report):
    """The worker's memory stays flat over saves, restores and reads, and its peak does not follow the size of the state."""
    n = rig.args.big
    a, b = rig.history(100, n), rig.history(101, n)
    rig.slot("erase")
    rig.turn(a[: n // 8])                           # settles the worker's buffers
    rig.slot("save", "mem-warm")
    rig.slot("restore", "mem-warm")
    pid = rig.worker.pid
    rig.turn(a)
    start_now, start_peak = resident(pid)
    saved = rig.slot("save", "mem-a")
    after_save_now, after_save_peak = resident(pid)
    rig.turn(b)
    rig.slot("save", "mem-b")
    restored = rig.slot("restore", "mem-a")
    after_now, after_peak = resident(pid)
    state = saved["n_parked_bytes"]
    assert restored["n_read"] == state
    rise = after_peak - start_peak
    rounds = []
    for _ in range(4):                              # and it does not creep: four more swaps
        rig.slot("save", "mem-a")
        rig.slot("restore", "mem-b")
        rig.turn(b[: n // 2])
        rig.slot("save", "mem-b")
        rig.slot("restore", "mem-a")
        rounds.append(round(resident(pid)[0], 1))
    report["memory"] = dict(state_MiB=round(state / 2**20, 1), worker_peak_rise_MiB=round(rise, 1),
                            worker_anon_MiB=dict(before=round(start_now, 1), after_save=round(after_save_now, 1),
                                                 after_restore=round(after_now, 1), four_more_swaps=rounds),
                            checked=state >= 24 * 2**20)
    assert max(rounds) - min(rounds) <= 4 and rounds[-1] - after_now <= 12, report["memory"]
    if state >= 24 * 2**20:
        # one 8 MiB buffer, whatever the state's size
        assert rise <= 14, f"the worker's peak memory rose {rise:.1f} MiB while parking and loading {state / 2**20:.1f} MiB"


CASES = dict(swap=case_swap, firings=case_firings, many=case_many, torn=case_torn, stale=case_stale,
             prune=case_prune, mixed=case_mixed, busy=case_busy, crash=case_crash, memory=case_memory)


def no_core_dumps():
    """The torn case kills the worker in a save, and llama-server then aborts, as it does for any dead
    worker. Neither may leave a core dump (several GiB each with a real model). The servers inherit this
    limit. 1, not 0: where the system pipes dumps to a program (WSL does), 0 is ignored and 1 means none."""
    hard = resource.getrlimit(resource.RLIMIT_CORE)[1]
    limit = 1 if hard == resource.RLIM_INFINITY else min(1, hard)
    resource.setrlimit(resource.RLIMIT_CORE, (limit, limit))


def run(args):
    no_core_dumps()
    output = args.output
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    for binary in (args.build / "bin/ggml-rpc-server", args.build / "bin/llama-server"):
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}; first run scripts/build-fleet-host.sh")
    rig = Rig(args, output)
    report = dict(model=str(args.model) if args.model else "untrained llama fixture (attention only)", context=args.ctx)
    try:
        rig.start()
        for name in args.only or list(CASES):
            started = time.monotonic()
            CASES[name](rig, report)
            print(f"  {name}: ok ({time.monotonic() - started:.0f} s)", flush=True)
    finally:
        rig.stop()
    result = dict(status="passed", scope="Linux CPU TCP RPC only; no PS5", **report)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/fleet-park")
    parser.add_argument("--model", type=Path, default=None,
                        help="a real GGUF instead of the untrained fixture (Qwen3.5-0.8B for the hybrid layout)")
    parser.add_argument("--old-build", type=Path, default=os.environ.get("PS5LM_PARK_TEST_OLD_BUILD") or None,
                        help="a fleet host build without device parking, for the mixed case")
    parser.add_argument("--ctx", type=int, default=None, help="context size (default 4096, or 20480 with --model)")
    parser.add_argument("--tokens", type=int, default=None, help="tokens of an ordinary history (default 200, or 1200 with --model)")
    parser.add_argument("--firing", type=int, default=None, help="tokens a firing adds (default 300, or 2500 with --model)")
    parser.add_argument("--big", type=int, default=None, help="tokens of the memory case's histories (default 3000, or 16000 with --model)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--only", nargs="*", choices=list(CASES))
    args = parser.parse_args()
    args.build, args.output = args.build.resolve(), args.output.resolve()
    if args.model is not None:
        args.model = Path(args.model).expanduser().resolve()
    if args.old_build is not None:
        args.old_build = Path(args.old_build).expanduser().resolve()
    real = args.model is not None
    args.ctx    = args.ctx    or (20480 if real else 4096)
    args.tokens = args.tokens or (1200 if real else 200)
    args.firing = args.firing or (2500 if real else 300)
    args.big    = args.big    or (16000 if real else 3000)
    try:
        run(args)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
