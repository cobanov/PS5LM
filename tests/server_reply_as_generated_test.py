#!/usr/bin/env python3
"""patches/0017 on a host llama-server: a reply the client sends back is kept as it was generated.

A chat client sends the previous reply back as a message. The template writes it out and it is tokenised again,
and that is not always the tokens the slot holds: the same text can be spelled with other tokens, and the
whitespace around the end of the reasoning is the template's, not the model's. A recurrent state cannot go back
a few tokens, so without the patch the slot returns to a context checkpoint and reads the whole reply again.

The slot is given each reply token by token (a raw prompt), then the chat request that sends it back.

Kept, nothing read again:
  same      the reply in the template's own tokens (with and without the patch)
  split     one token as two, or as another one, the text unchanged
  before    a space before the marker that ends the reasoning (the template trims it)
  after     whitespace after that marker: a space, and a newline and spaces (the parser drops it from the
            content, the template writes its own)
  tool      a reply with a tool call, split
  stream    the request that sends the reply back is a streaming one
  both      split and whitespace in one reply, then a second and a third turn on top: the reply two turns
            back, rendered again by every later prompt, is still kept
  cut off   with a reply kept two turns back, a later reply that was cut off (the slot holds it less its last
            token)
  fresh     what the model writes after a kept reply is what it writes after reading the slot's tokens and the
            rest of the prompt from nothing

Read again, from a checkpoint:
  second    the content has an end-of-reasoning marker of its own (the model wrote one into its answer) and the
            whitespace before it, or after it, differs: only the marker that closes the template's thinking
            block is excepted
  space     two spaces against one in the content, and in a tool call's argument
  marker    the slot has the marker's characters as plain text where the prompt has the marker token
  later     with a reply kept two turns back: a later reply that comes back changed, and one that was cancelled
            half way, are read again from a checkpoint after the kept reply, not from before it
  edits     an earlier message changed; the reply's reasoning changed; the same in a reply that is also split
  bound     the prompt with the slot's tokens would not fit the context (fixture only)
  reject    the patch's last check refuses (forced with LLAMA_REPLY_AS_GENERATED_TEST_REJECT): the request is
            served as without the patch
  off       LLAMA_REPLY_AS_GENERATED=0, false, OFF or no: split and before are read again; any other value is on

Run after scripts/build-fleet-host.sh. The model is an untrained recurrent fixture with one token per letter,
so only token counts and the server's timings are checked, not the words. With --model the same cases run on a
real GGUF with a ChatML template (Qwen3.5-0.8B: the consoles' hybrid layout and reasoning markers).
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import os
import urllib.request

from fleet_rpc_smoke import free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, make_recurrent_fixture

# A marker token starts a user and an assistant turn. The reasoning between two marker tokens and trimmed, the content trimmed, as
# the Qwen template trims them; tool calls after the content; an assistant turn ends with the end-of-sequence token.
TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'assistant' -%}"
    "<|assistant|><think>{{ (m.reasoning_content if m.reasoning_content is string else '') | trim }}</think>"
    "{{ (m.content if m.content is string else '') | trim }}"
    "{%- if m.tool_calls -%}{%- for call in m.tool_calls -%}"
    "<tool_call>{{ call.function.name }}{{ call.function.arguments | tojson }}</tool_call>"
    "{%- endfor -%}{%- endif -%}"
    "</s>"
    "{%- else -%}<|{{ m.role }}|>{{ m.content }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}"
)
MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<|user|>", "<|assistant|>")
# The same for a real ChatML model with <think> tokens: every reply keeps its reasoning, as the Qwen3.8 template
# writes it with preserve_thinking.
REAL_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'assistant' -%}"
    "{{ '<|im_start|>assistant\\n<think>\\n' + ((m.reasoning_content if m.reasoning_content is string else '') | trim)"
    " + '\\n</think>\\n\\n' + ((m.content if m.content is string else '') | trim) }}"
    "{%- if m.tool_calls -%}{%- for call in m.tool_calls -%}"
    "{{ '\\n<tool_call>\\n' + call.function.name + '\\n' + (call.function.arguments | tojson) + '\\n</tool_call>' }}"
    "{%- endfor -%}{%- endif -%}"
    "{{ '<|im_end|>\\n' }}"
    "{%- else -%}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\\n<think>\\n' }}{%- endif -%}"
)
CONTEXT = 1024   # the fixture's


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
            command += ["--model", str(fixture), "--ctx-size", str(CONTEXT), "--batch-size", "16", "--ubatch-size", "16",
                        "--chat-template-file", str(template)]
        self.end_of_turn = "<|im_end|>" if self.real else "</s>"
        self.command, self.env = command, dict(os.environ, CUDA_VISIBLE_DEVICES="-1", **(env or {}))

    def start(self, stack):
        wait_ready(spawn(stack, self.command, self.log, self.env), self.port, http=True)
        return self

    def call(self, path, body):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def text(self, messages):
        return self.call("/apply-template", {"messages": messages})["prompt"]

    def tokens(self, messages):
        """The prompt of a chat request, as token ids."""
        return self.call("/tokenize", {"content": self.text(messages), "add_special": not self.real, "parse_special": True})["tokens"]

    def ids(self, text, special=True):
        """A piece of text as tokens, on its own."""
        return self.call("/tokenize", {"content": text, "add_special": False, "parse_special": special})["tokens"]

    def give(self, tokens):
        """The slot holds exactly these tokens afterwards, as after generating the last of them."""
        body = self.call("/completion", {"prompt": tokens, "n_predict": 0, "cache_prompt": True, "id_slot": 0, "temperature": 0})
        assert body["tokens_cached"] == len(tokens), (body["tokens_cached"], len(tokens))

    def chat(self, messages, stream=False):
        """Tokens kept, tokens read and the prompt's tokens, of the request that sends a conversation."""
        body = {"messages": messages, "max_tokens": 1, "temperature": 0, "cache_prompt": True, "id_slot": 0}
        if not stream:
            answer = self.call("/v1/chat/completions", body)
            return answer["timings"]["cache_n"], answer["timings"]["prompt_n"], answer["usage"]["prompt_tokens"]
        body.update(stream=True, stream_options={"include_usage": True})
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        timings = usage = None
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                if line.startswith(b"data: {"):
                    event = json.loads(line[6:])
                    timings, usage = event.get("timings") or timings, event.get("usage") or usage
        return timings["cache_n"], timings["prompt_n"], usage["prompt_tokens"]

    def continue_from(self, tokens, n=8):
        """n greedy tokens after a raw prompt: the tokens, and how many of the prompt's the slot already had."""
        body = self.call("/completion", {"prompt": tokens, "n_predict": n, "temperature": 0, "seed": 1, "ignore_eos": True,
                                         "return_tokens": True, "cache_prompt": True, "id_slot": 0})
        assert len(body["tokens"]) == n, body
        return body["tokens"], body["timings"]["cache_n"]


CALL = {"type": "function", "id": "call0", "function": {"name": "read", "arguments": "{\"path\": \"wall a b\"}"}}


class Case:
    """A conversation of its own (no two share a first message), its reply as the template writes it, and ways
    to hold the same reply, or nearly the same, in other tokens."""

    def __init__(self, server, name, content="green", call=False):
        self.s = server
        self.messages = [{"role": "system", "content": f"case {name} " + "answer in one word " * 6},
                         {"role": "user", "content": "name a colour that is not red or blue"}]
        self.reply = {"role": "assistant", "reasoning_content": "they want a colour so green\n", "content": content}
        self.next = {"role": "user", "content": "and another"}
        if call:
            self.reply = dict(self.reply, content=None, tool_calls=[CALL])
            self.next = {"role": "tool", "tool_call_id": "call0", "content": "ok"}
        self.prompt = server.tokens(self.messages)
        self.head, after = server.text(self.messages), server.text(self.messages + [self.reply, self.next])
        assert after.startswith(self.head), "the prompt is not a prefix of the next one"
        self.text = after[len(self.head):after.index(server.end_of_turn, len(self.head))]   # the reply, without its end of turn
        self.rendered = self.ids(self.text)
        self.full = server.tokens(self.messages + [self.reply, self.next])
        assert self.full[:len(self.prompt) + len(self.rendered)] == self.prompt + self.rendered, "a reply does not tokenise behind its prompt as in the next one"
        self.end_think = server.ids("</think>")[-1]
        self.close = self.text.index("</think>")           # the marker that closes the template's thinking block
        self.after = self.close + len("</think>")
        self.separator = len(self.text[self.after:]) - len(self.text[self.after:].lstrip())   # the template's own whitespace there

    def ids(self, text):
        """A reply's text as tokens: tokenised behind the prompt, as the template's is."""
        tokens = self.s.call("/tokenize", {"content": self.head + text, "add_special": not self.s.real, "parse_special": True})["tokens"]
        assert tokens[:len(self.prompt)] == self.prompt
        return tokens[len(self.prompt):]

    def edited(self, at, old, new):
        """The reply with `old` at character `at` replaced by `new`, as tokens."""
        assert self.text[at:at + len(old)] == old, (self.text[at:at + len(old)], old)
        return self.ids(self.text[:at] + new + self.text[at + len(old):])

    def split(self, reply=None):
        """One token as two or as another one, the text unchanged."""
        reply = list(self.rendered if reply is None else reply)
        if self.s.real:
            whole = self.s.ids(" colour")[-1]
            parts = self.s.ids(" col") + self.s.ids("our")
            assert whole in reply and whole not in parts, (whole, parts)
            i = reply.index(whole)
            return reply[:i] + parts + reply[i + 1:]
        letter = self.s.ids("w")[-1]                     # fixture: the letter's byte token, <0x77>, instead of its own
        i = len(reply) - 1 - reply[::-1].index(letter)   # the last one: in the tool call when there is one
        return reply[:i] + [3 + ord("w")] + reply[i + 1:]

    def before(self):
        """A space before the closing marker (before the template's own newline, where it writes one)."""
        stripped = len(self.text[:self.close].rstrip())
        return self.edited(stripped, "", " ")

    def after_space(self, whitespace):
        """Other whitespace than the template's right after the closing marker."""
        return self.edited(self.after, self.text[self.after:self.after + self.separator], whitespace)

    def run(self, held, messages=None, stream=False):
        """The slot holds the prompt and `held`; then the request that sends the reply back.
        Returns tokens kept, tokens read, and how many the slot held."""
        self.s.give(self.prompt + held)
        kept, read, _ = self.s.chat(messages or self.messages + [self.reply, self.next], stream)
        return kept, read, len(self.prompt) + len(held)


def cases(server, patched, report):
    def kept_whole(name, held, **how):
        case = held[0]
        kept, read, n = case.run(held[1], **how)
        if patched:
            assert kept == n, f"{name}: kept {kept} of the slot's {n} tokens; the reply was read again"
        else:
            assert kept < n, f"{name}: kept {kept} of {n} with the patch switched off"   # back at a checkpoint
        report.append(f"{name} {kept}/{n}")

    def read_again(name, held, messages=None):
        case = held[0]
        kept, read, n = case.run(held[1], messages)
        assert kept < n, f"{name}: kept all {n} tokens of a slot that does not say what the request says"
        report.append(f"{name} from {kept} of {n}")

    def on(name, change, **kind):
        case = Case(server, name, **kind)
        return case, change(case)

    if patched:
        report.append("KEPT")
    same = Case(server, "same")
    kept, read, held = same.run(same.rendered)
    assert kept == held, f"same: kept {kept} of {held}"
    kept_whole("split", on("split", lambda c: c.split()))
    kept_whole("before", on("before", lambda c: c.before()))
    if not patched:
        return
    kept_whole("after: a space", on("after a", lambda c: c.after_space(" ")))
    kept_whole("after: newline and spaces", on("after b", lambda c: c.after_space("\n    ")))
    kept_whole("after: none", on("after c", lambda c: c.after_space("")))
    kept_whole("before and after", on("around", lambda c: c.ids(c.text[:c.close].rstrip() +"  </think> \n " + c.text[c.after:].lstrip())))
    kept_whole("tool call", on("tool", lambda c: c.split(), call=True))
    kept_whole("tool call, whitespace after the marker", on("tool after", lambda c: c.after_space(" "), call=True))
    kept_whole("streamed", on("stream", lambda c: c.split(c.before())), stream=True)

    # both at once, and two more turns on top: every later prompt renders the first reply again
    case = Case(server, "both")
    first = case.split(case.after_space("\n "))
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
    report.append(f"both and a third turn {kept}/{held}")

    # with a reply kept two turns back, a request that is not a plain continuation of the slot keeps what it
    # shares with the slot; it does not go back to before the kept reply
    third = [second, {"role": "user", "content": "one more"}]
    floor = len(case.prompt) + len(first)                 # the end of the kept reply, in the slot's tokens

    # the second reply was cut off: the slot holds it less its last token, the request has all of it
    server.give(slot + reply2[:-1])
    kept, read, _ = server.chat(history + third)
    assert kept == len(slot) + len(reply2) - 1, f"cut off: kept {kept} of {len(slot) + len(reply2) - 1}"
    report.append(f"cut-off reply after a kept one {kept}/{len(slot) + len(reply2) - 1}")

    # what the model writes after a kept reply is what it writes from the same tokens read from nothing
    # (raw prompts, which carry no template: there only tokens that say the same byte for byte are kept)
    case = Case(server, "fresh")
    held = case.split()
    merged = case.prompt + held + case.full[len(case.prompt) + len(case.rendered):]
    server.give(case.prompt + held)
    after_keep, had = server.continue_from(case.full)
    assert had == len(case.prompt) + len(held), f"fresh: the slot's {len(case.prompt) + len(held)} tokens were not kept ({had})"
    server.give(Case(server, "another").prompt)           # the slot leaves the conversation
    from_nothing, had = server.continue_from(merged)
    assert had < len(case.prompt), had
    assert after_keep == from_nothing, f"after a kept reply the model wrote {after_keep}, from the same tokens read afresh {from_nothing}"
    report.append(f"after a keep the next {len(after_keep)} tokens are those of a fresh read")

    report.append("READ AGAIN")
    # an end-of-reasoning marker of the content's own is compared with its whitespace
    twice = dict(content="green</think>ok")
    second_marker = lambda c: c.text.index("</think>", c.after)
    read_again("second marker, a space before it", on("second a", lambda c: c.edited(second_marker(c), "", " "), **twice))
    read_again("second marker, a space after it", on("second b", lambda c: c.edited(second_marker(c) + len("</think>"), "", " "), **twice))
    read_again("second marker, a newline before it", on("second c", lambda c: c.edited(second_marker(c), "", "\n"), **twice))
    kept_whole("second marker, the closing one's whitespace", on("second d", lambda c: c.after_space(" "), **twice))
    # whitespace anywhere else is text
    spaced = dict(content="green a b")
    read_again("two spaces for one in the content", on("space a", lambda c: c.edited(c.text.index(" b", c.after), " ", "  "), **spaced))
    read_again("a space at the end of the content", on("space b", lambda c: c.edited(len(c.text), "", " "), **spaced))
    read_again("two spaces for one in a tool argument", on("space c", lambda c: c.edited(c.text.index(" b", c.after), " ", "  "), call=True))
    # the marker's characters as plain text are not the marker
    def plain(c):
        at = len(c.rendered) - 1 - c.rendered[::-1].index(c.end_think)       # the content's own marker
        letters = c.s.ids("</thi") + c.s.ids("nk>") if c.s.real else [3 + ord(x) for x in "</think>"]   # in two halves, or it is the marker
        assert c.end_think not in letters and at > c.rendered.index(c.end_think)
        return c.rendered[:at] + letters + c.rendered[at + 1:]
    case, held = on("marker", plain, **twice)
    read_again("the marker as plain text", (case, held))

    # with a kept reply two turns back: a later reply that comes back with another answer
    case = Case(server, "later")
    first = case.split(case.after_space("\n "))
    history = case.messages + [case.reply, case.next]
    slot = case.prompt + first + server.tokens(history)[len(case.prompt) + len(case.rendered):]
    floor = len(case.prompt) + len(first)
    more = server.tokens(history + third)
    reply2 = more[len(server.tokens(history)):more.index(more[len(case.prompt) + len(case.rendered)], len(server.tokens(history)))]
    server.give(case.prompt + first)
    server.give(slot)                                     # a checkpoint 4 tokens before the second reply
    server.give(slot + reply2)
    changed = dict(second, content="umber")
    kept, read, _ = server.chat(history + [changed, third[1]])
    assert floor < kept < len(slot) + len(reply2), f"changed later reply: kept {kept}, the kept reply ends at {floor}"
    report.append(f"changed later reply from {kept} (kept reply ends at {floor})")

    # the second reply was cancelled half way and the client sends the request again
    server.give(slot)
    server.give(slot + reply2[:len(reply2) // 2])
    kept, read, _ = server.chat(history)
    assert floor < kept <= len(slot), f"cancelled reply: kept {kept}, the kept reply ends at {floor}, the prompt at {len(slot)}"
    report.append(f"cancelled reply from {kept} (prompt ends at {len(slot)})")

    # real edits are read again, from a checkpoint before the change
    def edited(name, change, edit):
        case = Case(server, name)
        messages = edit(case)
        kept, read, held = case.run(change(case), messages)
        new = server.tokens(messages)
        old = case.prompt + case.rendered
        common = next((i for i in range(min(len(old), len(new))) if old[i] != new[i]), min(len(old), len(new)))
        assert kept <= common and kept < held, f"{name}: kept {kept} tokens, the request differs at {common}"
        report.append(f"{name} from {kept} (differs at {common}, slot {held})")

    def earlier(case):
        return [case.messages[0], {"role": "user", "content": "name a colour that is not red or green"}, case.reply, case.next]

    def reasoning(case):
        return case.messages + [dict(case.reply, reasoning_content="they want a colour so greens\n"), case.next]

    edited("edit-earlier", lambda c: c.rendered, earlier)
    edited("edit-reply", lambda c: c.rendered, reasoning)
    edited("edit-reply-split", lambda c: c.split(), reasoning)

    if not server.real:
        # the prompt with the slot's tokens must fit the context: one token short of it, it is kept; at it, not
        for room, name in ((2, "fits"), (1, "bound")):
            case = Case(server, name)
            held = case.before()                          # one token more than the template's
            assert len(held) == len(case.rendered) + 1
            pad = CONTEXT - room - len(case.full)
            messages = case.messages + [case.reply, dict(case.next, content=case.next["content"] + "x" * pad)]
            assert len(server.tokens(messages)) == CONTEXT - room, (len(server.tokens(messages)), CONTEXT - room)
            kept, read, n = case.run(held, messages)
            if room == 2:
                assert kept == n, f"a prompt of {CONTEXT - 1} tokens with the slot's: kept {kept} of {n}"
            else:
                assert kept < n, f"a prompt of {CONTEXT} tokens with the slot's was kept"
                report.append(f"context bound from {kept} of {n}")


def run(build, output, model):
    output.mkdir(parents=True, exist_ok=True)
    report = []
    with ExitStack() as stack:
        cases(Server(build, output, model, "on").start(stack), True, report)
    report.append("OFF")
    for value in ("0", "false", "OFF", "no"):
        with ExitStack() as stack:
            cases(Server(build, output, model, "off-" + value, {"LLAMA_REPLY_AS_GENERATED": value}).start(stack), False, report)
    report.append("ON with LLAMA_REPLY_AS_GENERATED=yes")
    with ExitStack() as stack:
        server = Server(build, output, model, "on-yes", {"LLAMA_REPLY_AS_GENERATED": "yes"}).start(stack)
        case = Case(server, "yes")
        kept, read, held = case.run(case.split())
        assert kept == held, f"LLAMA_REPLY_AS_GENERATED=yes: kept {kept} of {held}"
        report.append(f"split {kept}/{held}")
    report.append("LAST CHECK REFUSES")
    with ExitStack() as stack:
        server = Server(build, output, model, "reject", {"LLAMA_REPLY_AS_GENERATED_TEST_REJECT": "1"}).start(stack)
        case = Case(server, "reject")
        kept, read, held = case.run(case.split())
        assert kept < held, f"the check refused and {kept} of {held} were kept all the same"
        assert "previous reply not kept" in server.log.read_text(errors="replace"), "no warning in the server's log"
        report.append(f"split from {kept} of {held}, with a warning")
    print("reply as generated: " + "; ".join(report).replace("; KEPT; ", " KEPT: ").replace("KEPT; ", "KEPT: ")
          .replace("; READ AGAIN; ", " | READ AGAIN: ").replace("; OFF; ", " | OFF: ").replace("; ON with", " | ON with")
          .replace("=yes; ", "=yes: ").replace("; LAST CHECK REFUSES; ", " | LAST CHECK REFUSES: "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/server-reply-as-generated")
    parser.add_argument("--model", type=Path, help="a real GGUF with a reasoning template instead of the fixture")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve(), args.model)
