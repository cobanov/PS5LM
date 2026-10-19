# Ternary Bonsai models on the PS5

`third_party/llama.cpp` now points at the [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
(branch `prism`, pinned at `1180918`) instead of the mainline submodule pin. The fork adds the
ternary Bonsai formats and their kernels on top of mainline llama.cpp:

| Format | ggml id | What it is | File to load |
|---|---|---|---|
| `PQ2_0` | 142 | group-128 ternary 2-bit, the fork's preferred format | `*-PQ2_0.gguf` |
| `PTQ1_0` | fork | group-64 post-quantized ternary, smaller and lossless over ternary checkpoints | `*-PTQ1_0.gguf` |

Mainline llama.cpp has neither, so a model like
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
only loads with the fork's tree. The fork carries the kernels PS5LM needs:

- **Vulkan** (the console GPU): dequant, `mul_mat_vec` and coopmat matmul paths for `PQ2_0` and `PTQ1_0`.
- **CPU** (Zen 2 hybrid split): `vec_dot` and the AVX2 tinyBLAS gemv/gemm for `PQ2_0`.

## What fits

On a PS5 Slim the RADV heap holds 11.44 GiB. Measured file sizes from the HF repo:

| File | Size | Fits? |
|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 6.71 GiB | yes, with room for a long context |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5.54 GiB | yes, most room |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.60 GiB | only if you want images too |

The planner gives the model the longest context that fits, as always; put the `.gguf` in
`/data/PS5LM/models` or download it in the app.

## Integration notes

- The submodule URL in `.gitmodules` points at the fork; the pin is its HEAD commit.
- PS5LM's own patches (`patches/0001..0007`) apply on the fork tree unchanged, except that
  `0005` and `0007` were rebased onto the fork's older single-listener `server-http.cpp`
  (the fork predates mainline's multi-listener refactor). They keep the same intent:
  the listener rebinds when the network drops, and the activity hooks feed the TV feed.
- The fork predates llama.cpp's decision-model server (`/v1/systemone`): decision models are
  not served by this build. Chat, completions, Anthropic-style `/v1/messages`, the web UI,
  image mmproj, and the dashboard all work.
