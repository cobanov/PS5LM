#!/usr/bin/env python3
"""patches/0021 on a host llama-server: under --image-max-tokens a Qwen-VL vision encoder is warmed
with exactly that many tokens, so no image rebuilds its compute buffers later.

Images are resized to a multiple of 32 px a side and stay within the cap, so their token counts
reach it: 1280x1261 gives 36 x 35 = 1,260 tokens and 1280x1024 gives 40 x 32 = 1,280, while the
largest square under 1,280 is 35 x 35 = 1,225. Each image larger than the warm-up rebuilds the
encoder's compute buffers, which the patch logs ("compute buffer grew").

Run after scripts/build-fleet-host.sh. The text model and the projector are untrained fixtures
(a Qwen3-VL text model and projector of 32 wide), so only sizes and log lines are checked.
--projector runs the same with a real Qwen3-VL projector (an mmproj GGUF) and a text fixture as
wide as its output, and reports the encoder's compute buffer.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import base64
import json
import os
import struct
import time
import urllib.request

from fleet_rpc_smoke import free_ports, spawn, wait_ready
from make_fleet_fixture import ROOT, add_tokenizer

CAP = 1280
WARMUP = "warmup with image size = 1280 x 1024"
GREW = "compute buffer grew"
# (width, height) of each image and the tokens it must get under the cap
IMAGES = [((1120, 1120), 1225), ((1297, 1037), 1240), ((1280, 1261), 1260), ((1280, 1024), 1280)]
TEXT_EMBD = 32
MARKER = "<__media__>"
BUFFER = "compute buffer size ="


def make_text_fixture(path: Path, embd: int = TEXT_EMBD) -> Path:
    """A 2-block Qwen3-VL text model (interleaved M-RoPE, q/k norms) that takes the projector's
    output: 4 query and 2 KV heads of 8 at the fixture's width, heads of 128 at a real one's."""
    import numpy as np
    import gguf

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path, "qwen3vl")
    writer.add_name("PS5LM vision warm-up test fixture; untrained")
    writer.add_context_length(4096)
    head_dim = 8 if embd == TEXT_EMBD else 128
    heads = embd // head_dim
    kv, ff = (heads // 2) * head_dim, 64
    writer.add_embedding_length(embd)
    writer.add_block_count(2)
    writer.add_feed_forward_length(ff)
    writer.add_head_count(heads)
    writer.add_head_count_kv(heads // 2)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_dimension_sections([2, 1, 1, 0] if head_dim == 8 else [24, 20, 20, 0])
    writer.add_rope_freq_base(10000.0)
    writer.add_uint32("qwen3vl.n_deepstack_layers", 0)
    tokens = add_tokenizer(writer)
    rng = np.random.default_rng(20261122)

    def weight(name, shape):
        writer.add_tensor(name, (rng.standard_normal(shape) * 0.08).astype(np.float32))

    weight("token_embd.weight", (len(tokens), embd))
    writer.add_tensor("output_norm.weight", np.ones(embd, dtype=np.float32))
    weight("output.weight", (len(tokens), embd))
    for layer in range(2):
        prefix = f"blk.{layer}."
        for name, size in (("attn_norm", embd), ("ffn_norm", embd), ("attn_q_norm", head_dim),
                           ("attn_k_norm", head_dim)):
            writer.add_tensor(prefix + name + ".weight", np.ones(size, dtype=np.float32))
        for name, shape in (
            ("attn_q.weight", (embd, embd)),
            ("attn_k.weight", (kv, embd)),
            ("attn_v.weight", (kv, embd)),
            ("attn_output.weight", (embd, embd)),
            ("ffn_gate.weight", (ff, embd)),
            ("ffn_down.weight", (embd, ff)),
            ("ffn_up.weight", (ff, embd)),
        ):
            weight(prefix + name, shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def make_projector_fixture(path: Path) -> Path:
    """A 2-block qwen3vl_merger projector with the real one's layout (16 px patches, 2x2 merge,
    fused qkv, learned positions interpolated to the image), 32 wide."""
    import numpy as np
    import gguf

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path, "clip")
    writer.add_string("general.type", "mmproj")
    writer.add_name("PS5LM vision warm-up test projector; untrained")
    embd, ff, blocks, positions = 32, 64, 2, 64
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.projector_type", "qwen3vl_merger")
    writer.add_bool("clip.use_gelu", True)
    writer.add_uint32("clip.vision.projection_dim", TEXT_EMBD)
    writer.add_uint32("clip.vision.image_size", 768)
    writer.add_uint32("clip.vision.patch_size", 16)
    writer.add_uint32("clip.vision.embedding_length", embd)
    writer.add_uint32("clip.vision.feed_forward_length", ff)
    writer.add_uint32("clip.vision.block_count", blocks)
    writer.add_uint32("clip.vision.attention.head_count", 2)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    writer.add_uint32("clip.vision.spatial_merge_size", 2)
    writer.add_array("clip.vision.is_deepstack_layers", [False] * blocks)
    rng = np.random.default_rng(20261123)

    def weight(name, shape):
        writer.add_tensor(name, (rng.standard_normal(shape) * 0.05).astype(np.float32))

    def ones(name, size):
        writer.add_tensor(name, np.ones(size, dtype=np.float32))

    def zeros(name, size):
        writer.add_tensor(name, np.zeros(size, dtype=np.float32))

    # numpy shapes are ggml's ne reversed
    weight("v.patch_embd.weight", (embd, 3, 16, 16))
    weight("v.patch_embd.weight.1", (embd, 3, 16, 16))
    zeros("v.patch_embd.bias", embd)
    weight("v.position_embd.weight", (positions, embd))
    for layer in range(blocks):
        prefix = f"v.blk.{layer}."
        weight(prefix + "attn_qkv.weight", (3 * embd, embd))
        zeros(prefix + "attn_qkv.bias", 3 * embd)
        weight(prefix + "attn_out.weight", (embd, embd))
        zeros(prefix + "attn_out.bias", embd)
        weight(prefix + "ffn_up.weight", (ff, embd))
        zeros(prefix + "ffn_up.bias", ff)
        weight(prefix + "ffn_down.weight", (embd, ff))
        zeros(prefix + "ffn_down.bias", embd)
        for norm in ("ln1", "ln2"):
            ones(prefix + norm + ".weight", embd)
            zeros(prefix + norm + ".bias", embd)
    ones("v.post_ln.weight", embd)
    zeros("v.post_ln.bias", embd)
    weight("mm.0.weight", (4 * embd, 4 * embd))
    zeros("mm.0.bias", 4 * embd)
    weight("mm.2.weight", (TEXT_EMBD, 4 * embd))
    zeros("mm.2.bias", TEXT_EMBD)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def bmp(width, height):
    """A 24-bit BMP of a colour gradient."""
    stride = (width * 3 + 3) & ~3
    rows = bytearray()
    pad = bytes(stride - width * 3)
    for y in range(height - 1, -1, -1):
        green = (y * 255) // height
        rows += bytes(v for x in range(width)
                      for v in ((x * 255) // width, green, ((x + y) * 255) // (width + height))) + pad
    header = struct.pack("<2sIHHI", b"BM", 54 + len(rows), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(rows), 2835, 2835, 0, 0)
    return header + info + bytes(rows)


def complete(port, image):
    payload = {"prompt": {"prompt_string": f"a {MARKER} b",
                          "multimodal_data": [base64.b64encode(image).decode()]},
               "n_predict": 1, "temperature": 0, "cache_prompt": False}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def projection_dim(projector: Path) -> int:
    import gguf

    field = gguf.GGUFReader(projector).get_field("clip.vision.projection_dim")
    return int(field.parts[field.data[0]][0])


def run(build, output, projector=None):
    output.mkdir(parents=True, exist_ok=True)
    server = build / "bin/llama-server"
    if not server.is_file():
        raise FileNotFoundError(f"{server}; first run scripts/build-fleet-host.sh")
    if projector:
        model = make_text_fixture(output / "qwen3vl-text.gguf", projection_dim(projector))
    else:
        model = make_text_fixture(output / "tiny-qwen3vl.gguf")
        projector = make_projector_fixture(output / "tiny-qwen3vl-mmproj.gguf")
    (api,) = free_ports(1)
    log = output / "server.log"
    command = [str(server), "--model", str(model), "--mmproj", str(projector), "--no-mmproj-offload",
               "--image-max-tokens", str(CAP), "--host", "127.0.0.1", "--port", str(api),
               "--ctx-size", "2048", "--batch-size", "2048", "--ubatch-size", "512", "--parallel", "1",
               "--threads", "2", "--threads-batch", "2", "--device", "none", "--gpu-layers", "0",
               "--fit", "off", "--log-verbosity", "4"]
    result = {"status": "passed", "images": []}
    with ExitStack() as stack:
        # a fixed marker instead of the server's random one
        process = spawn(stack, command, log, dict(os.environ, LLAMA_MEDIA_MARKER=MARKER))
        wait_ready(process, api, http=True)
        seen = len(log.read_text(errors="replace"))
        for (width, height), want in IMAGES:
            complete(api, bmp(width, height))
            # the log is written by its own thread: wait for this image's lines
            deadline = time.monotonic() + 30
            while True:
                text = log.read_text(errors="replace")
                if "decoded (batch" in text[seen:] or time.monotonic() > deadline:
                    break
                time.sleep(0.05)
            # mtmd-helper: "decoding image batch 1/1, n_tokens_batch = N"
            tokens = sum(int(line.rsplit("=", 1)[1]) for line in text[seen:].splitlines()
                         if "decoding image batch" in line and "n_tokens_batch" in line)
            seen = len(text)
            result["images"].append({"size": f"{width}x{height}", "tokens": tokens})
            assert tokens == want, f"{width}x{height}: {tokens} image tokens, expected {want}"
    text = log.read_text(errors="replace")
    assert WARMUP in text, "the encoder was not warmed at 1280 x 1024 (40 x 32 tokens)"
    result["warmup_buffers"] = [line.split(BUFFER)[1].strip() for line in text.splitlines()
                                if BUFFER in line and "reserve_compute_meta" in line]
    grew = [line for line in text.splitlines() if GREW in line]
    result["grew"] = grew
    print(json.dumps(result))
    assert not grew, f"an image rebuilt the encoder's buffers: {grew}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fleet-host")
    parser.add_argument("--output", type=Path, default=ROOT / "build/server-vision-warmup")
    parser.add_argument("--projector", type=Path, help="a real Qwen3-VL mmproj GGUF instead of the fixture")
    args = parser.parse_args()
    run(args.build.resolve(), args.output.resolve(), args.projector and args.projector.resolve())
