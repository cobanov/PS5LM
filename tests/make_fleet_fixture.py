#!/usr/bin/env python3
"""Create a deterministic, untrained 4-block GGUF for RPC correctness tests.

Uses the pinned llama.cpp GGUF writer and NumPy. This is not a useful language
model and its output is intentionally meaningless. No weights are downloaded.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))


def make_fixture(path: Path) -> Path:
    import numpy as np
    import gguf

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_name("PS5LM deterministic fleet test fixture; untrained")
    writer.add_context_length(256)
    writer.add_embedding_length(32)
    writer.add_block_count(4)
    writer.add_feed_forward_length(64)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(8)
    writer.add_rope_freq_base(10000.0)
    writer.add_tokenizer_model("llama")
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    tokens += [chr(i) for i in range(97, 123)] + ["\u2581"]
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * len(tokens))
    writer.add_token_types([2, 3, 3] + [6] * 256 + [1] * 27)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    rng = np.random.default_rng(20261021)

    def weight(name, shape):
        writer.add_tensor(name, (rng.standard_normal(shape) * 0.08).astype(np.float32))

    weight("token_embd.weight", (len(tokens), 32))
    writer.add_tensor("output_norm.weight", np.ones(32, dtype=np.float32))
    weight("output.weight", (len(tokens), 32))
    for layer in range(4):
        prefix = f"blk.{layer}."
        writer.add_tensor(prefix + "attn_norm.weight", np.ones(32, dtype=np.float32))
        writer.add_tensor(prefix + "ffn_norm.weight", np.ones(32, dtype=np.float32))
        for name, shape in (
            ("attn_q.weight", (32, 32)),
            ("attn_k.weight", (16, 32)),
            ("attn_v.weight", (16, 32)),
            ("attn_output.weight", (32, 32)),
            ("ffn_gate.weight", (64, 32)),
            ("ffn_down.weight", (32, 64)),
            ("ffn_up.weight", (64, 32)),
        ):
            weight(prefix + name, shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path,
                        default=ROOT / "build/fleet-smoke/tiny-llama.gguf")
    args = parser.parse_args()
    result = make_fixture(args.output)
    print(f"{result}: {result.stat().st_size} bytes (untrained)")
