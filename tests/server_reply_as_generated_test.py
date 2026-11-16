#!/usr/bin/env python3
"""patches/0017 on a host llama-server: a reply the client sends back is kept as it was generated.

A chat client sends the previous reply back as a message. The template writes it out and it is tokenised again,
and that is not always the tokens the slot holds: the same text can be spelled with other tokens, and the
template trims the reasoning, so whitespace the model wrote right before the end-of-reasoning marker is gone.
A recurrent state cannot go back a few tokens, so without the patch the slot returns to a context checkpoint
and reads the whole reply again.

The slot is given each reply token by token (a raw prompt), then the chat request that sends it back:

  same      the reply in the tokens the template renders: nothing is read again (with and without the patch)
  split     one letter as its byte token instead of its letter token (the same text): nothing is read again
  trim      a space before the end-of-reasoning marker, which the template trims: nothing is read again
  both      split and trim in one reply, then a second and a third turn on top: nothing is read again,
            and the reply two turns back, rendered again by every later prompt, is still kept
  after     with that reply kept two turns back: a second reply that was cut off (the slot holds it less
            its last token) is not read again; one that comes back changed, and one that was cancelled half
            way, are read again from a checkpoint after the kept reply, not from before it
  edits     an earlier message changed; the reply's reasoning changed; the reply's reasoning changed in a
            reply that is also split: each is read again from the checkpoint, as without the patch
  off       LLAMA_REPLY_AS_GENERATED=0: split and trim are read again from the checkpoint

Run after scripts/build-fleet-host.sh. The model is an untrained recurrent fixture with one token per letter,
so only token counts and the server's timings are checked, not the words. With --model the same cases run on a
real GGUF with its own template (Qwen3.5-0.8B: the consoles' hybrid layout and reasoning format).
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import os
import urllib.request

from fleet_rpc_smoke import free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_recurrent_fixture

# Roles in angle brackets (bytes), the reasoning between two marker tokens and trimmed, as the Qwen template
# trims it; an assistant turn ends with the end-of-sequence token.
TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'assistant' -%}"
    "<assistant><think>{{ (m.reasoning_content if m.reasoning_content is string else '') | trim }}</think>{{ m.content }}</s>"
    "{%- else -%}<{{ m.role }}>{{ m.content }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<assistant><think>{%- endif -%}"
)
MARKERS = ("<think>", "</think>")
# The same for a real ChatML model with <think> tokens: every reply keeps its reasoning, trimmed, as the
# Qwen3.8 template writes it with preserve_thinking.
REAL_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'assistant' -%}"
    "{{ '<|im_start|>assistant\\n<think>\\n' + ((m.reasoning_content if m.reasoning_content is string else '') | trim)"
    " + '\\n</think>\\n\\n' + m.content + '<|im_end|>\\n' }}"
    "{%- else -%}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\\n<think>\\n' }}{%- endif -%}"
)


class Server:
    def __init__(self, build, output, model, name, env=None):
        self.output, self.real = output, model is not None
        binary = build / "bin/llama-server"
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}; first run scripts/build-fleet-host.sh")
        (self.port,) = free_ports(1)
        self.log = output / f"server-{name}.log"
        command = [str(binary), "--host", "127.0.0.1", "--port", str(self.port), "--parallel", "1", "--threads", "2",
                   "--device", "none", "--gpu-layers", "0", "--fit", "off", "--no-warmup", "--jinja",
                   "--ctx-checkpoints", "3", "--checkpoint-min-step", "1", "--cache-ram", "0", "--no-context-shift"]
        if self.real:
            template = output / "reasoning-chatml.jinja"
            template.write_text(REAL_TEMPLATE)
            command += ["--model", str(model), "--ctx-size", "4096", "--batch-size", "512", "--ubatch-size", "64",
                        "--chat-template-file", str(template)]
        else:
            fixture = make_recurrent_fixture(output / "tiny-mamba-markers.gguf", markers=MARKERS)
            template = output / "reasoning.jinja"
            template.write_text(TEMPLATE)
            command += ["--model", str(fixture), "--ctx-size", "1024", "--batch-size", "16", "--ubatch-size", "16",
                        "--chat-template-file", str(template)]
        self.kwargs = {}
        self.command, self.env = command, dict(os.environ, CUDA_VISIBLE_DEVICES="-1", **(env or {}))

    def start(self, stack):
        wait_ready(spawn(stack, self.command, self.log, self.env), self.port, http=True)
        return self

    def call(self, path, body):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def tokens(self, messages):
        """The prompt of a chat request, as token ids."""
        prompt = self.call("/apply-template", {"messages": messages, **self.kwargs})["prompt"]
        return self.call("/tokenize", {"content": prompt, "add_special": not self.real, "parse_special": True})["tokens"]

    def ids(self, text):
        return self.call("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

    def give(self, tokens):
        """The slot holds exactly these tokens afterwards, as after generating the last of them."""
        body = self.call("/completion", {"prompt": tokens, "n_predict": 0, "cache_prompt": True, "id_slot": 0, "temperature": 0})
        assert body["tokens_cached"] == len(tokens), (body["tokens_cached"], len(tokens))

    def chat(self, messages):
        body = self.call("/v1/chat/completions", {"messages": messages, "max_tokens": 1, "temperature": 0, "cache_prompt": True, "id_slot": 0, **self.kwargs})
        return body["timings"]["cache_n"], body["timings"]["prompt_n"], body["usage"]["prompt_tokens"]


class Case:
    """A conversation of its own (no two share a first message), its reply in the template's tokens, and ways
    to hold the same reply in other tokens."""

    def __init__(self, server, name):
        self.s = server
        self.messages = [{"role": "system", "content": f"case {name} " + "answer in one word " * 6},
                         {"role": "user", "content": "name a colour that is not red or blue"}]
        self.reply = {"role": "assistant", "reasoning_content": "they want a colour so green\n", "content": "green"}
        self.next = {"role": "user", "content": "and another"}
        self.prompt = server.tokens(self.messages)
        after = server.tokens(self.messages + [self.reply, self.next])
        assert after[:len(self.prompt)] == self.prompt, "the prompt is not a prefix of the next one"
        (self.end_think,) = server.ids("</think>")
        eos = server.call("/tokenize", {"content": "<|im_end|>" if server.real else "</s>", "add_special": False, "parse_special": True})["tokens"]
        assert len(eos) == 1, eos
        self.rendered = after[len(self.prompt):after.index(eos[0], len(self.prompt))]   # the reply, without its end-of-turn token
        assert self.end_think in self.rendered

    def split(self, reply):
        """One token of the reasoning as two or as another one, the text unchanged."""
        if self.s.real:
            (whole,) = self.s.ids(" colour")
            parts = self.s.ids(" col") + self.s.ids("our")
            assert whole in reply and whole not in parts, (whole, parts)
            i = reply.index(whole)
            return reply[:i] + parts + reply[i + 1:]
        letter = self.s.ids("w")[-1]                     # fixture: the letter's byte token, <0x77>
        i = reply.index(letter)
        return reply[:i] + [3 + ord("w")] + reply[i + 1:]

    def trim(self, reply):
        """A space before the end-of-reasoning marker (and its newline, where the template writes one)."""
        i = reply.index(self.end_think)
        if self.s.real:
            (newline,) = self.s.ids("\n")
            assert reply[i - 1] == newline, reply[i - 3:i + 1]
            return reply[:i - 1] + self.s.ids(" ") + reply[i - 1:]
        space = self.s.ids("a b")[-2]                    # the word boundary token
        return reply[:i] + [space] + reply[i:]

    def run(self, held, messages=None):
        """The slot holds the prompt and `held`; then the request that sends the reply back.
        Returns tokens kept, tokens read, and how many the slot held."""
        self.s.give(self.prompt + held)
        kept, read, _ = self.s.chat(messages or self.messages + [self.reply, self.next])
        return kept, read, len(self.prompt) + len(held)


def cases(server, patched, report):
    def kept_whole(name, change):
        case = Case(server, name)
        kept, read, held = case.run(change(case, case.rendered))
        if patched:
            assert kept == held, f"{name}: kept {kept} of the slot's {held} tokens; the reply was read again"
        else:
            assert kept < held, f"{name}: kept {kept} of {held} with the patch switched off"   # back at a checkpoint
        report.append(f"{name} kept {kept}/{held}")
        return case

    same = Case(server, "same")
    kept, read, held = same.run(same.rendered)
    assert kept == held, f"same: kept {kept} of {held}"
    kept_whole("split", lambda c, r: c.split(r))
    kept_whole("trim", lambda c, r: c.trim(r))
    if not patched:
        return

    # both at once, and two more turns on top: every later prompt renders the first reply again
    case = Case(server, "both")
    first = case.trim(case.split(case.rendered))
    kept, read, held = case.run(first)
    assert kept == held, f"both: kept {kept} of {held}"
    second = {"role": "assistant", "reasoning_content": "another colour then\n", "content": "amber"}
    history = case.messages + [case.reply, case.next]
    slot = case.prompt + first + server.tokens(history)[len(case.prompt) + len(case.rendered):]   # as the server merged it
    more = server.tokens(history + [second, {"role": "user", "content": "one more"}])
    canonical = server.tokens(history)
    reply2 = more[len(canonical):more.index(more[len(case.prompt) + len(case.rendered)], len(canonical))]
    server.give(slot + reply2)
    held = len(slot) + len(reply2)
    kept, read, _ = server.chat(history + [second, {"role": "user", "content": "one more"}])
    assert kept == held, f"both, third turn: kept {kept} of {held}"
    report.append(f"both and a third turn kept {kept}/{held}")

    # with a reply kept two turns back, a request that is not a plain continuation of the slot keeps what it
    # shares with the slot; it does not go back to before the kept reply
    third = [second, {"role": "user", "content": "one more"}]
    floor = len(case.prompt) + len(first)                 # the end of the kept reply, in the slot's tokens

    # the second reply was cut off: the slot holds it less its last token, the request has all of it
    server.give(slot + reply2[:-1])
    kept, read, _ = server.chat(history + third)
    assert kept == len(slot) + len(reply2) - 1, f"cut off: kept {kept} of {len(slot) + len(reply2) - 1}"
    report.append(f"cut-off reply after a kept one kept {kept}")

    # the second reply comes back with another answer: read again from a checkpoint after the kept reply
    server.give(slot)                                     # a checkpoint 4 tokens before the second reply
    server.give(slot + reply2)
    changed = dict(second, content="umber")
    kept, read, _ = server.chat(history + [changed, third[1]])
    assert floor < kept < len(slot) + len(reply2), f"changed later reply: kept {kept}, the kept reply ends at {floor}"
    report.append(f"changed later reply read again from {kept} (kept reply ends at {floor})")

    # the second reply was cancelled half way and the client asks something else instead of the second user turn's
    # reply: nothing past the generation prompt is shared, and what is before it is kept up to a checkpoint
    server.give(slot)
    server.give(slot + reply2[:len(reply2) // 2])
    kept, read, _ = server.chat(history)                  # the same request again, as after a cancel
    assert floor < kept <= len(slot), f"cancelled reply: kept {kept}, the kept reply ends at {floor}, the prompt at {len(slot)}"
    report.append(f"cancelled reply read again from {kept} (prompt ends at {len(slot)})")

    # real edits are read again, from a checkpoint before the change
    def edited(name, held_change, edit):
        case = Case(server, name)
        messages = edit(case)
        kept, read, held = case.run(held_change(case, case.rendered), messages)
        new = server.tokens(messages)
        old = case.prompt + case.rendered
        common = next((i for i in range(min(len(old), len(new))) if old[i] != new[i]), min(len(old), len(new)))
        assert kept <= common and kept < held, f"{name}: kept {kept} tokens, the request differs at {common}"
        report.append(f"{name} read again from {kept} (differs at {common}, slot {held})")

    def earlier(case):
        return [case.messages[0], {"role": "user", "content": "name a colour that is not red or green"}, case.reply, case.next]

    def reasoning(case):
        return case.messages + [dict(case.reply, reasoning_content="they want a colour so greens\n"), case.next]

    edited("edit-earlier", lambda c, r: r, earlier)
    edited("edit-reply", lambda c, r: r, reasoning)
    edited("edit-reply-split", lambda c, r: c.split(r), reasoning)


def run(build, output, model):
    output.mkdir(parents=True, exist_ok=True)
    report = []
    with ExitStack() as stack:
        cases(Server(build, output, model, "on").start(stack), True, report)
    with ExitStack() as stack:
        cases(Server(build, output, model, "off", {"LLAMA_REPLY_AS_GENERATED": "0"}).start(stack), False, report)
    print("reply as generated: " + "; ".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/server-reply-as-generated")
    parser.add_argument("--model", type=Path, help="a real GGUF with a reasoning template instead of the fixture")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve(), args.model)
