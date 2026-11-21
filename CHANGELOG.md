# Changelog

Versions follow [semver](https://semver.org). The app's version is set in
`ps5/app/version.hpp` and shown on the dashboard; each release is tagged
`v<version>` and ships `ps5lm-app-v<version>.zip` (see `scripts/release-app.sh`).
v0.1.x are the upstream payload releases by cobanov.

## [Unreleased]

### Added
- **Inkling 2 Mini** (Thinking Machines' looped MoE, 26B with 3.4B active) in Suggested, as
  Unsloth's `UD-Q2_K_XL` from `Mainline_Support` (upstream llama.cpp cannot read the files in
  the repository root, which use Unsloth's fork layout). The planner counts its cache slots:
  layers 6 to 17 run three times a token, each pass with its own KV and conv state, so 48
  slots, 8 of them full attention and 40 sliding window. Its relative position bias adds about
  208 bytes of compute buffer per context and ubatch token (832 MiB at 64k, `-ub 64`, measured
  with ggml-vulkan). At 11.38 GiB that is a 64k `q8_0` cache, 11.24 GiB planned. Checked on a PC
  with ggml-vulkan and the patched tree: it answers correctly.
- **A fleet worker leaves worker mode on request**: a `/data/PS5LM/leave-worker` file
  makes it delete `fleet-worker.txt` and restart as the app, the way a model change
  restarts it. Before, a worker could only quit, and someone had to open PS5LM again
  on the console.
- **The fleet splits modern models.** Any single-file model the pinned llama.cpp
  loads as one stack of blocks is admitted. That includes Qwen 3.5/3.6/3.8 (hybrid
  Gated DeltaNet, dense and MoE), Llama 3.1+, Gemma 3/4 and gpt-oss, in all 35
  ggml tensor types. Before, only dense `llama` and `qwen2` were admitted.
  - The estimator now follows llama.cpp's placement and caches: recurrent state,
    NextN blocks, sliding windows (listed in the file or llama.cpp's own pattern),
    shared KV, widened V rows without flash attention, and RoPE and layer-0
    tensors.
  - It matched `llama-fit-params --fit-print` within 1 MiB on every device for
    nine models at 4k and 32k. Qwen 3.5 and Gemma 3 split over two RPC workers
    gave the same greedy tokens as one worker (docs/FLEET.md).
- Before llama-server starts, `fleet serve` checks the plan with llama.cpp itself:
  `llama-fit-params --fit-print` against the live workers, compute buffers
  included. It refuses a worker that would come within 256 MiB of its budget, or
  an eighth of a smaller budget.
  `scripts/build-fleet-host.sh` builds `llama-fit-params`.
- **`python3 -m fleet control`** runs a fleet of any size from one JSON
  configuration: devices (consoles, this PC's GPUs, other RPC servers), and pools.
  A pool is one model with its context, slots and cache. Its replicas are layer
  splits, single consoles that serve the model with PS5LM itself, local GPUs, or
  servers elsewhere.
  - **One API** for every pool on port 8081 (OpenAI and Anthropic routes,
    streaming, an optional key). The request's `model` picks the pool, and the
    least busy replica gets it.
  - **Running pools stay up.** It starts their replicas (making consoles workers
    and starting the GPU's RPC worker as needed) and watches every device and
    replica. It restarts what fails, with back-off.
  - **Consoles are switched over the network**: found with the PlayStation
    discovery protocol (also from WSL), and put in or out of worker mode over FTP.
  - **`python3 -m fleet ctl`**: the same from a shell or an agent, as JSON:
    `status`, `pool add --auto`, `pool start --wait`, `device leave`, and
    `config` / `apply` for the whole configuration.
  - **A console's worker settings**, on the page or with `ctl device set`:
    `screen` (off by default, about 350 MiB more for the model) and `cache_gib`
    (0 by default, no cache, as in the app; set, the GiB of weights its worker
    keeps, so a reload reads them on the console: 32 holds a 27B model's share).
    They go into its env.txt; a change restarts an idle worker, and one
    in use takes it at its next restart. The page shows what each cache holds.
    New actions: **restart** (a worker as a worker, with its settings) and
    **clear-cache**.
  - Consoles keep back 512 MiB for llama.cpp's compute buffers by default, GPUs
    and RPC servers 1024.
  - A split restarts its consoles' workers before it loads: the console's driver
    keeps memory an earlier host allocated, and a second load ran out of it. The
    watcher does not probe a console while a split on it starts.
  - Splits sample on the worker with the output layer (`--backend-sampling`):
    18.4 instead of 16.0 tokens/s for Qwen3.8-27B on a PS5 alone.
  - A closed or crashed PS5LM is opened from the control server (`launch`, and
    automatically by splits, `worker` and `restart`), through an ELF loader on
    the console's port 9021 and PS5LM's launch payload.
  - A version 1 `control.json` (one list of workers) becomes devices and a stopped
    pool.
  - **Who can use it:**
    - The API is shared on the network (`api_lan`) only with an `api_key`; a
      configuration without one is refused, when loaded and when changed.
    - The API sends CORS headers only with a key set. Without one it sends
      none, drops a replica's own, and refuses requests that carry another
      site's `Origin`, so a page open in the browser cannot drive a keyless API.
    - Behind `--allow-host`, the page and `ctl` change things only with a token
      (`--token` or `$PS5LM_FLEET_TOKEN`; without one, a new token is printed
      at each start). The page asks for it once; `ctl` takes `--token`. Both
      also read it from a file that only you can read
      (`~/.config/ps5lm-fleet/token`, or `--token-file`), so scripts that call
      `ctl` need no variable.
    - The launch payload comes from one folder (`build/app-ps5`, or
      `--launcher-dir`): `launcher` picks a `.elf` in it, and only an ELF is
      sent. A model given as a path is copied to a console only if it is a GGUF.
    - A POST with a `Content-Length` that is not a number, negative or over
      4 MiB is answered 413; a negative one used to wait on the connection.
- **Long contexts in a split**:
  - The planner and `fleet serve` take a quantized cache with flash attention
    (`--cache-type q8_0 --flash-attn on`) and parallel slots, sized as llama.cpp
    sizes them.
  - Qwen3.8 27B (Q4_K_XL) ran with a 128k context, q4_0 cache, over one PS5 and an
    RTX 3080. With a q8_0 cache, 80k fits.
- Manual layer splits in the planner (`layer_counts`). A split can have one node.
- **A fleet worker without its screen**: `PS5LM_WORKER_SCREEN=off` in `env.txt`
  starts a worker without its screen and dashboard. That leaves about 350 MiB more
  for the model: on a console the worker reported 12,217 MiB free instead of 11,854.
- **A fleet worker restarts on request**: `/data/PS5LM/unload`, which restarts the
  app, restarts a worker as a worker. It reads `env.txt` again, so its settings
  change without anyone at the console.
- A worker logs its memory once it has started, and when `/data/PS5LM/memory` is there.
- **A fleet worker can keep the weights hosts send** (`PS5LM_RPC_CACHE_GB` in
  `env.txt`, off by default): up to that many GiB in `/data/PS5LM/rpc-cache`, the
  least recently used dropped first. A host loading the same weights again sends
  only their hashes. On a console, Ornith 1.5 9B loaded in 17.6 s from the
  cache against 48.9 s without it (its first load, saving the cache, took
  113 s).
  - `patches/fleet/0005`: a cache file is written whole under a temporary name,
    used only when it hashes to what the host sent, and read in 8 MiB reads.
    Before, a write cut short left a file that later loaded as the weights.
  - The files are written through ftpsrv: a title's own writes slow to about
    1.2 MB/s after 2 GB.
- **Parked state stays on the console** (`PS5LM_PARK_GB` in `env.txt`, 128 GiB
  unless set, 0 for off; `--slot-park-path` on the host): a host that serves
  many conversations on one slot parks the one leaving in files on the worker,
  in `/data/PS5LM/park`, instead of fetching it into its own memory, and loads
  it back from the console's storage. `POST /slots/N?action=save|restore` with
  the conversation's name; `park-list`, `park-delete`, `park-status`
  ([FLEET.md](docs/FLEET.md#parked-state)). On a console serving Qwen3.8-27B
  a 153K-token conversation came back in 4.6 s (65 s through the host prompt
  cache) and a hand-over's save took 2.2 s.
  - A save writes only what was added: the attention cells of the new tokens
    and one recurrent state. For Qwen3.8-27B that is 18 MiB per 1,000 new tokens
    plus 150 MiB, where the host prompt cache moved 2.8 GiB each way for a 153K
    history. The host keeps 28 bytes a token.
  - The worker drops the least recently used key when the folder is full, by
    its own record of saves and restores (a console's file times do not move
    when a key is read), and files are read with `read()` (the console's libc
    reads an unbuffered stream a byte per call).
  - `patches/fleet/0009`: park commands in the RPC worker, offered in its
    handshake, every byte through one 8 MiB buffer, files whole or absent, the
    folder kept to its size by dropping the key used longest ago.
    `patches/0015`: a sequence's state in parts, the tensor data left on the
    device. `patches/0016`: the server's side and its bookkeeping.
  - A history changed before its parked end is rewritten from the change, never
    resumed as it was; a save that dies half way leaves the one before loadable.
  - `scripts/ps5lm-app.sh bench-park` measures the write path: sustained writes
    through ftpsrv with the speed of every GiB.
- **Large tensors go to a fleet worker in 8 MiB pieces** (`patches/fleet/0007`).
  A worker took every tensor write, and built every answer to a read, in one
  buffer of its size. A host that saved or restored a slot moved whole KV layers
  that way, and every prompt batch sent an attention mask that grows with the
  depth: a console serving Qwen3.8-27B with 60 MiB to spare died saving a
  100K-token history. Split by the host, the same saves and restores ran to
  153K tokens with the console's free memory unmoved. `GGML_RPC_CHUNK_MB` sets
  the piece size (0 for off).

- **A host keeps its idle connection to a worker alive** (`patches/fleet/0008`):
  TCP keepalive on the RPC link, first probe after `GGML_RPC_KEEPALIVE_S`
  seconds idle (60; 0 for off). A host behind a NAT (WSL2) lost the connection
  after about half an hour without a request, and llama-server aborted on the
  next one although the worker was fine.

### Changed
- **More memory for models**: `sce_sys/param.json` declares a 256 MiB flexible pool
  (the system's default is 448 MiB), so the direct pool, where models go, is
  12.1875 GiB instead of 12. It applies when the app starts from the home screen,
  not when it restarts for a model.
- **llama.cpp moves to d2bceda3d** (39 commits after b12602, 2026-11-22), 816 commits on from
  b11827, for Inkling 2 Mini (`inkling2_loop`, ggml-org/llama.cpp 32694, and the fixes it needed on
  a console: 32669, 32704, 32735, 32799). Proven on one PS5 with Qwen3.8-27B at 1 x 155,648 against
  the build before it (same loads, parked conversations restored identically, writing 19.1 tokens a
  second where it was 18.6, memory no lower), and Inkling 2 Mini runs on the console
  ([FLEET.md](docs/FLEET.md#inkling-2-mini)).
  - A host and a worker must both be on this pin. llama.cpp's RPC protocol went from 8 to
    10 and a handshake between the two refuses, so a console and its host change together.
  - Replies are not token for token what the old pin wrote: the same prompt diverges after a few
    dozen tokens, greedy included (new attention kernels in ggml-vulkan).
  - `patches/0011` (the KV cache's rotation matrices kept on its device) is gone: llama.cpp
    does it itself (31073). `tests/server_kv_rotation_test.py` stays and holds llama.cpp to
    it.
  - `patches/0012` (the causal mask built on the KV cache's device) is written anew on
    llama.cpp's `llm_attn_mask`. It now also covers the full-attention half of a
    sliding-window model's cache (Gemma 3); a model whose attention has relative
    positions (Inkling 2) makes a dense mask itself and keeps the host mask.
  - The worker's handshake bits moved: llama.cpp took the byte they were in for flags of
    its own, counted up from bit 0, so graph-free (`patches/fleet/0006`) is bit 6 and
    parking (`patches/fleet/0009`) bit 7.
  - `GGML_RPC_GRAPH_KEEP` (host, `patches/fleet/0006`): how many graphs the host leaves on a
    worker for reuse, 8 by default. On this llama.cpp a 27B model's graph costs a console about
    6.5 MiB (it was 4): with 8 a console at 1 x 155,648 settled at 41 MiB of free pool where it
    had 61, with 1 it keeps 72 or more, at the same speed. A host that serves one slot should
    set 1.
  - `patches/fleet/0010`, new: an RPC device no longer claims flash attention over K/V
    rows, which llama.cpp now uses for a unified KV cache when a device says it can. The
    RPC device said yes for every op without asking its worker, and ggml-vulkan has no
    such kernel, so a console would have been sent an op it cannot run.
    `GGML_RPC_KV_ROWS=1` on the host offers it again.
  - The other patches are re-based; `0016`, `fleet/0005`, `fleet/0007` and `fleet/0009`
    with small changes for llama.cpp's reference-counted checkpoints and its new RPC
    send path. `--defrag-thold` is gone from llama-server.
  - Limits for Inkling 2 Mini: its state cannot be parked on the worker yet, and its context is
    bounded by the batch size (108K at `--ubatch-size 128`, 352K at 8).
- **A fleet host keeps sampling on the worker with a reasoning budget and with tools**
  (`patches/0018`, host only): with `--backend-sampling` the worker samples and the
  scores of the whole vocabulary stay there, 970 KiB a token for Qwen3.8-27B. A
  reasoning budget or a tool-call grammar switched that off for the request ("backend
  sampling is not compatible with reasoning budget, disabling"), and every agent
  request has both: a PS5 wrote 16.3 tokens a second that way against 18.8 without
  the budget. For such a request the worker now returns the top-k plus one candidate
  and the host runs its chain, the budget and the grammar on those. The whole
  vocabulary is fetched only for a token the budget forces, a token the grammar
  rejects, and two equal scores among the candidates. The tokens are the ones host
  sampling gives for the same seed. On a CPU worker with Qwen3.5-0.8B a token cost
  184 bytes from the worker instead of 993,288, and 19,179 of 19,354 tokens of
  budget-and-tools replies never needed more. It needs a top-k of at most 1024 as the
  first active sampler (no logit bias or penalties before it), no mirostat, token
  probabilities or speculative decoding; otherwise, and with
  `LLAMA_BACKEND_SAMPLING_CHECKS=0`, a request is sampled on the host as before.
  `LLAMA_BACKEND_SAMPLING_CHECKS=verify` chooses every token both ways and counts
  the ones that differ, for testing a build on a worker.
  `tests/server_backend_sampling_test.py` checks it
  ([FLEET.md](docs/FLEET.md#sampling-on-the-worker)).
- **A reply sent back is not read again** (`patches/0017`): a chat client sends the
  previous reply back as a message, and the template and the tokeniser do not always
  give the tokens the model wrote. The same text can be split into other tokens
  (` book` `marks` written, ` bookmarks` on the way back), and the whitespace around
  the end of the reasoning is the template's: what a reasoning budget left before the
  end-of-reasoning marker, or the model wrote after it, is not what comes back. A hybrid
  model cannot step back a few tokens, so the slot returned to its checkpoint and read
  the whole reply again, or the whole last firing after a parked restore, and the
  parked key was rewritten from there. Qwen3.8-27B serving agents did this on 10 of
  227 follow-up turns, 84 to 10,508 tokens each. The server now compares the slot's
  remaining tokens with the prompt as text, and as far as they say the same the prompt
  continues from the slot's tokens. Whitespace is compared like any text, except right
  before and right after the marker that closes the thinking block the template opened
  in an assistant turn. A history that was really changed is read again as
  before. `prompt_tokens` then counts the kept tokens, a token or two away from a
  fresh tokenisation. It is off unless `LLAMA_REPLY_AS_GENERATED` is `1` (or `true`, `on`,
  `yes`); the fleet controller sets it for its hosts, so the app's own server is unchanged.
  `tests/server_reply_as_generated_test.py` checks it on a host build
  ([FLEET.md](docs/FLEET.md#a-reply-sent-back)).
- The fleet's llama-server keeps one context checkpoint and no RAM prompt cache
  (`--ctx-checkpoints 1 --cache-ram 0`): saving either reads caches back from
  every worker over the network.
- **Qwen vision models get a narrow ubatch**: a model with a Qwen projector (`clip.projector_type`
  `qwen*`) plans `-ub 256`, `-b 1280` and `--image-max-tokens 1280` instead of the 2048 ubatch,
  and keeps 0.5 GiB of headroom for photos. llama.cpp needs a whole image in one ubatch only for
  non-causal encoders (Gemma's), which keep 2048. On the console Qwen 3.8 27B UD-IQ2_S with its
  F16 projector had 0.90 GiB free when ready at `-ub 1024` and 1.47 GiB at `-ub 256`, at the
  same speed (median 11.1 s a photo) and with the same answers.
- **Hybrid models start answering sooner**: with one context checkpoint a slot (`-ctxcp 1`,
  which the app's plan and the fleet use), llama-server made three checkpoints for every prompt
  (at the last user message, one batch before the end and 4 tokens before it) and evicted two of
  them before the prompt was done. Each reads the model's recurrent state back from the GPU,
  150 MiB for Qwen3.8-27B, and for a fleet split that read crosses the network.
  `patches/0010` makes only the checkpoint a prompt keeps, so the next turn rolls back to the
  same place as before. On a PS5 serving Qwen3.8-27B (IQ3_XXS, 128k context) to a PC over RPC,
  LocalMaxxing's 348-token `reasoning-v1` prompt took 7.6 s to its first token and now takes
  4.5 s, and a follow-up turn 6.6 s and now 3.3 s; generation stays at 18.1 tokens a second.
  On the console the read stays in memory, and that gain is not measured.
  `tests/server_checkpoints_test.py` checks it on a host build with an untrained Mamba fixture.
- **Faster generation with a quantized KV cache on another device**: llama.cpp rotates a
  quantized KV cache with Hadamard matrices, and it kept them in PC memory and uploaded them
  to the device as inputs to every graph. For a fleet split those 272 KiB (Qwen3.8-27B's 256- and
  64-wide heads) crossed the network on each token, about 2.3 ms a graph at gigabit.
  `patches/0011` keeps one copy in the cache's device buffer when the whole cache is on one
  device, so they are sent once at load. A PS5 serving Qwen3.8-27B (IQ3_XXS, q4_0 cache) to a
  PC over RPC on LocalMaxxing's `reasoning-v1` test: 18.1 to 18.7 tokens a second at 128k
  context, and with MTP self-speculation (2 drafts, 104k context) 20.1 to 21.9. Greedy output is
  unchanged. A cache split over devices keeps the old inputs. `tests/server_kv_rotation_test.py`
  checks it on a host build against one CPU RPC worker.
- **Causal KQ mask built on the KV cache's device**: llama.cpp filled the attention mask in PC
  memory and uploaded it as an `[n_kv, n_tokens]` input to every graph, about 21 MB per 512-token
  batch at 20k context. `patches/0012` uploads only the cache cells' and the batch tokens'
  positions and builds the mask on the device, when the whole cache is on non-host devices and
  the batch needs a plain causal mask (no ALiBi, SWA or M-RoPE). A PS5 serving Qwen3.8-27B
  (IQ3_XXS, q4_0 cache, `--parallel 2 -no-kvu`) to a PC over RPC: a 20k-token prompt 106.4 to
  108.6 tokens a second, generation unchanged. Greedy output is byte-identical.
  `LLAMA_KQ_MASK_HOST=1` keeps the old mask. `tests/server_kq_mask_test.py` checks it on a host
  build against one CPU RPC worker, with and without flash attention.

### Fixed
- **A new message after a Qwen3.5 turn no longer processes the whole turn again** (#84). Qwen3.5's
  chat template renders an assistant message's thinking only after the last user message, so the
  next user message rendered the turn before it differently and the cache no longer matched: on the
  console, a 37k-token chat took 190 s before its answer started. llama.cpp keeps the thinking of
  earlier turns by default (`--reasoning-preserve`) for templates that support it; Qwen3.5's
  template ignored it, and now honours it (`patches/0013`, in the same place llama.cpp already
  fixes up other templates): a message has its thinking block wherever it is in the history, and
  only when it has thinking. A client that sends the thinking back keeps it in the context; Open
  WebUI without a provider set sends none, and its turns render as Qwen3.5 sees past turns, with no
  block. `--no-reasoning-preserve` (or `LLAMA_ARG_REASONING_PRESERVE=false` in `env.txt`) restores
  the template's own behaviour. Measured on the console with the template as shipped, a 4.7k-token
  agent turn followed by a new message processed all 4,684 tokens again in 10.5 s; with the thinking
  kept, 52 tokens in 0.2 s; in an Open WebUI chat, a follow-up after an 8.4k-token agent turn
  processed 37 tokens in 0.2 s. Keeping an empty block on every earlier turn instead made Qwen3.5
  leave its visible reply empty in 4 of 6 Open WebUI follow-ups, so no empty block is rendered.
  `tests/test-chat.cpp` checks that a new message only appends and that no empty block is rendered.
  From @MCsumz's fork, [PS5LM-memfix](https://github.com/MCsumz/PS5LM-memfix).
- **A fleet worker no longer loses memory with every prompt batch**: llama.cpp's
  RPC server kept a copy of every graph a host sent (about 4 MiB for a 27B
  model) until 1,024 had come, and a host sends a new one for nearly every batch
  of a prompt. A console serving Qwen3.8-27B lost 7.5 to 9 MiB per 1,000 prompt
  tokens and died after about 175K. `patches/fleet/0006`: the host keeps the 8
  graphs it used last on each device and tells the worker to free the others.
  Both the host build and the title need the patch; either alone behaves as
  before.
- **A worker restart waits for the restarted worker**: the control server took the
  worker port accepting as the restart being over, but the old worker goes on
  accepting for some seconds after it takes the request. A host started right after
  `ctl device restart --wait` could then find nothing listening and stop with
  `Failed to connect`. The port cannot tell the two apart (a worker that has just
  lost a host also refuses connections for some seconds while it frees what the
  host held, then accepts again), so the wait is now for PS5LM to have started
  anew, read from the first line of its `app.log`, and then for the port to accept;
  the job's result gives both times. A console set to be a worker that reads as
  idle or busy at that moment is looked at again and restarted as a worker, with
  the same wait, where before the restart returned at once.
- **Generation no longer flatlines on a log write**: the GPU driver prints a stats line from its
  submission path, and a write to `app.log` on `/data` could block for 9-26 s with the driver's
  lock held; each blocked print made the next one due, so generation stayed stopped until the app
  restarted. `patches/mesa/0010` turns that line off unless `RADV_PS5_SUBMIT_STATS=1` is set in
  `env.txt`.
- **The app no longer stops while its log is written** (#70): stdout and stderr went straight to
  `app.log` on `/data`, stderr unbuffered, and a write there can wait for seconds. Every thread that
  printed waited with it: the GPU driver with its submit lock held (#69), the GL library on the
  render thread, llama.cpp's log thread. With `-lv 4` the whole app, screen included, stalled for up
  to a minute. Now printf, fprintf, vprintf, vfprintf, puts, fputs, fputc, putc, putchar and fwrite
  are wrapped at link time (`scripts/build-app.sh`): a print to stdout or stderr copies its text
  into a 2 MiB ring and returns, and one thread writes the ring to `app.log`. A full ring drops
  text and says how much in its place. `abort()` is wrapped too, so a crash writes the ring out
  (its reason included) before the process ends. The memory lines in `app.log` give the sink's figures.
  On the console at `-lv 4`, 5 minutes of ~1,800-token prompts: 16 requests, 7 over 15 s (worst 58 s)
  before; 25 requests, none stalled, after.

## [0.11.0] - 2026-11-02

### Added
- **Settings > Debug report**: one file to attach to a GitHub issue, `/data/PS5LM/debug-report.txt`
  and a copy on a USB drive when one is in. It holds the app's version, the system software, where
  the app keeps its files and the room on each drive, the models and their plans, the downloads,
  the settings, `env.txt` and `app-args.txt`, the last 256 KiB of `app.log` and `llama.log`, and
  which payload ports answer (ELF loader, FTP, kernel log, shell, ps5-exporter, ...). API keys,
  tokens and passwords are taken out of everything in it. The kernel log goes in only when
  **Include the kernel log** is on (off by default), at most its last 1 MiB, read from klogsrv
  or etaHEN's. Before the first report a dialog says what it holds.
- **Ternary Bonsai 2**: a **Bonsai 2** preset (PrismML's sampling, 64k context at most) and
  **Bonsai 2 27B** in Suggested, upstream's `Q2_0` (7.1 GiB). The app's llama.cpp already reads the
  `prism.hadamard.*` rotation those files carry: checked on a PC with the 2B and the 27B, on the CPU
  and on ggml-vulkan with the console's shader paths. On a console, fetched from Suggested (the
  download carried on across two restarts of the app): a 64k context with a q8_0 cache, 1.36 GiB
  of the pool left once it serves, about 17 tokens a second, right answers with and without
  thinking.
- `scripts/bonsai-pq2-to-q2.py`: PrismML's `PQ2_0` files, which upstream llama.cpp (b11827 too)
  refuses, to `Q2_0`, with no loss: each 128-weight block becomes two 64-weight blocks with the same
  scale and codes. Python only, about 80 s for the 27B; the 2B and the 27B come out byte for byte
  as the `Q2_0` files PrismML publishes. The library says what to do with a `PQ2_0` file (and that a
  `PTQ1_0` one does not load) instead of failing at load.
- `scripts/bench-ubatch.py`: prompt and generation speed and memory of one model at several prompt
  batch sizes on the console (2,048-token prompt, each size a restart with the planner's arguments),
  as a table; and `scripts/ps5lm-app.sh memory`, which logs the memory pools on request.
- **The app's own storage** when the HEN keeps `/data` closed (#16, #23, #24). The app used to
  stop there: no log, no settings, every download failing. It now keeps its files in
  `/download0/PS5LM` instead, and on the first such start asks, over the screen: use the app's
  storage, or how to open full storage (OnionHEN's `exact_title_ids`, etaHEN, other HENs) and start
  again, which asks the HEN once more. **Settings > Sound and storage > Where PS5LM keeps its
  files** shows which one is in use and how full it is. Models saved there stay in the library
  once full storage opens. Checked on a console with `/data` made unwritable to the app: the
  question on the screen, a model searched for, fetched to `/download0/PS5LM/models` and loaded
  from there, the choice kept across the restart, and the model still in the library once `/data`
  opened again.

- Two probes for later decisions, out of the app's way (`ps5/app/probes.cpp`, run from the PC):
  `scripts/ps5lm-app.sh bench-read <model file>` measures a drive's sequential and random read
  speed at the sizes MoE experts come in (64 KiB to 4 MiB, one and four threads, cached and
  `O_DIRECT`), and `bench-memory` logs what the system keeps of the title's memory and the
  largest block that really allocates. The download and write probes moved there from `main.cpp`.
- The TV's request feed shows OpenAI Responses requests (`/v1/responses`, what codex sends): the
  question from `input`, the answer from the response or its `output_text` deltas. They showed
  blank before. The API page lists the endpoint, and docs/CLI.md says what it takes.
- **A PC's GPU as a fleet worker.** `PS5LM_FLEET_GPU=metal` (or `cuda`, `vulkan`)
  `scripts/build-fleet-host.sh` adds that backend, so `ggml-rpc-server -d MTL0` serves the PC's
  GPU next to the consoles. The default host build stays CPU only. Run with a console: a PS5 and a
  Mac's GPU (Metal), and a PS5 and an RTX 3090 over CUDA and over Vulkan, each serving
  `qwen2.5-0.5b-instruct-q4_k_m.gguf` with blocks 0–19 on the PS5. See docs/FLEET.md.
- `scripts/ps5lm-app.sh launch` works again, with PS5LM's own launcher payload
  (`ps5/app/launch_app.c`, built by `build-app.sh`) instead of the removed
  ps5-homebrew-dev-protocol.

### Changed
- **The fleet worker is part of the app.** There is no separate fleet build or
  `ps5lm-app-fleet` zip any more: every PS5LM can be a fleet worker, and stays the normal app
  unless `/data/PS5LM/fleet-worker.txt` asks for one, as before. `PS5LM_FLEET` is gone from
  `scripts/build-app.sh`, and CI builds one title again.
- llama.cpp moves to b11827 (970b51ce7, 2026-10-22), 219 commits on, among them ggml-vulkan's
  fixes for GPU-visible host memory (#30468) and concurrent first device init (#29766), its strided
  f16 KV copy on AMD (#27703), and upstream's own `tojson` floats as Python writes them (#30362,
  #30483), so `patches/0006-jinja-tojson-floats-as-python.patch` is gone; `patches/0005` is
  re-based on the new `server-http.cpp`, and `patches/0009` (introspection) on the new
  `server-context.cpp`. It builds and links for the console as before.
- The planner counts the compute buffers' growth with the prompt batch (`-ub`, about 38 bytes a
  token per unit of feed-forward width, measured with ggml-vulkan on the console's shader paths),
  and the batch for language models is one constant, `kPromptUbatch`, 64. `scripts/bench-ubatch.py`
  measured the others on a console (Qwen 3.8 27B UD-IQ2_XXS, 64k, 2,048-token prompt): 124.6
  tokens a second at 64, 107.1 at 128, 122.8 at 256, 102.0 at 512, and the larger sizes leave less
  memory, so it stays 64. See docs/CONSOLE.md.
- `downloadDataSize` in `sce_sys/param.json` is 16 GiB (16384 MiB, was 256 MiB): the size of
  the app's own storage. The app counts what it holds itself, as a title's `statvfs` is a stub.
  The system makes it an 18.8 GB image, `/user/download/PPSA99575/download0.dat`, at the first
  start; whether all of that is taken on the drive at once was not measured. Lowering it later
  frees nothing.
- A download is refused before it starts, with the reason on the TV, when the drive has no room
  for it, the downloads already queued to that drive, and 1 GiB to spare. A model and its image
  projector are queued together or not at all.
- More memory for models, expected about 0.56 GiB: `sce_sys/param.json` no longer declares the
  `amm` block and the 256 MiB CPU and GPU page tables it carried over from ps5-homebrew-ui's
  template. With them the system gave the app 11.44 GiB of direct memory (the 12,280,922,112 bytes
  in #23's log); titles that leave them out, PS5_Vulkan's and PS5_RetroArch's, get 12 GiB, and
  neither the GL stack nor RADV uses what the blocks configure. The planner reads the memory free,
  so every model's plan grows with it: with the dashboard up, v0.9.1 had 11.04 GiB free before a
  model and gave Qwen 3.8 27B UD-Q2_K_XL a 32k context, not the 64k it was suggested with; with
  the 12 GiB pool it plans 64k again. On a console the pool is 12.0000 GiB and the planner has
  11.6 GiB for a model (11.0 before); `app.log` records the pool and the flexible memory at start
  (`memory at start`) and once the model serves (`memory when ready`).
- Every line the app writes to `app.log` starts with the console's local time, as
  `22:56:01 ps5lm-app: start` (#40). Lines from the graphics stack keep their own form.

### Fixed
- **A black screen on OnionHEN** with `PPSA99575` in `exact_title_ids` (#23, #32, #54). OnionHEN
  opens `/data` with a full app jailbreak that also moves the process out of its sandbox, so
  `/app0` and `/download0` were gone and the fonts did not load. Once `/data` is open, the app
  finds its sandbox in `/mnt/sandbox/<title id>_<slot>` and reads its own files from there;
  `app.log` says when it runs outside its sandbox. Unchanged on etaHEN, which keeps the sandbox.
  Not yet run on an OnionHEN console.
- A download that meets a full drive anyway stops at once, "Failed: App storage is full", instead
  of retrying twenty times.
- At the end of a download, Models > Downloads said "starting" for the minutes the SHA-256 check
  takes, so it looked hung (#30). It now says "Checking the SHA-256, 45%". The queue is saved
  before the check, so a restart during it checks the file again but no longer fetches the last
  piece first.
- codex failed on Qwen models with "System message must be at the beginning" (#55): it sends two
  developer messages after its instructions, and Qwen's template refuses a system message that is
  not first. `patches/0004` folded such messages into the first one only for the Anthropic API
  (Claude Code); it now does so for Chat Completions, Responses and Anthropic alike, with
  `tests/server_system_messages_test.py` checking all three on a host build. docs/CLI.md's codex
  setup is the one codex 0.166 reads: the profile in its own `~/.codex/ps5.config.toml`. Checked
  with codex 0.166 against a console.
- Models with sliding-window attention got far less context than fits (#57): the planner counted
  every attention layer at full length. It now reads `<arch>.attention.sliding_window_pattern` and
  counts the window layers at their window, as llama.cpp keeps them. Spark-X2.6 4B (27 of 36 layers
  keep 512 tokens) plans 256k with a q8_0 cache instead of 128k with q4_0; on a console the server
  took the 262,144-token context with 2 GiB of the pool left, and found a needle in a 52,821-token
  prompt. **Longest context** in Settings goes to 192k and 256k, and the planner tries both.
- A console serving a model went into rest mode an hour after the last button press: requests
  from the network are not controller input. Rest mode ends the app, and on a jailbroken console the
  controllers do not connect again until a restart. While a model is loaded, a download is queued
  or running, or the app is a fleet worker, it now tells the system every 30 s that it is in use
  (`sceSystemServicePowerTick`, as a video player does); with nothing to do, the console's own
  rest-mode setting applies. `app.log` says when it starts doing so.
- `scripts/bench-ubatch.py` took the old app's last lines, as it stopped, for the restarted one,
  so a size that loads was reported as not loading, or its prompt went to the stopping server;
  it never removed `app-args.txt` (ftpsrv answers a delete with 226, which ftplib refuses), and an
  error mid-run left the test arguments for every later start. All three are fixed.

## [0.10.0] - 2026-11-01

### Added
- **Fleet, experimental: one model across several consoles.** A PC runs the controller
  (`python3 -m fleet serve`) and one llama-server; each PS5 runs the fleet build of the app as an
  RPC worker that holds a contiguous run of the model's layers on its GPU. Dense `llama` and `qwen2`
  GGUFs only for now; Qwen 3.8's `qwen35` layout is refused by the planner. The fleet build is a
  separate zip, `ps5lm-app-fleet-vX.Y.Z.zip` (`PS5LM_FLEET=1 scripts/build-app.sh`); the normal
  app has no RPC code. A worker starts only when `/data/PS5LM/fleet-worker.txt` names the
  console's own private address, a port below 40000 (the PS5 refuses higher ones) and the
  acknowledgement, because RPC has no authentication:
  keep it on an isolated network. `docs/FLEET.md` is the guide. By @ntindle in #35.
- **An opt-in endpoint for the model's own numbers**: `POST /v1/introspect` answers with the whole
  logit row at the positions you name, the top candidates with their probabilities, the entropy, and
  one row of a named graph tensor (`l_out`, `ffn_swiglu`, `ffn_out`, `attn_out`, `attn_norm`, `ffn_norm`,
  `result_norm`, `result_output`) for a named layer. A request can also change one channel in place,
  with zero, scale or add, at one position or at all of them. `messages` applies the chat template
  first. It is off by default: Settings > Developer > "Logits and activations API" (a change
  restarts the app with the loaded model), or `LLAMA_INTROSPECT=1`. `docs/INTROSPECTION.md` is the
  reference, `scripts/ps5lm-introspect.py` is a client, and `examples/interpret/neuron_ablation.py`
  zeroes the FFN neurons (`ffn_swiglu`) of one layer one at a time.
- The patch reads the logits from the context's own output buffer and copies the tensors out of the
  compute graph with the eval callback that llama.cpp already has, so no work is repeated and no
  request pays for it unless it asks.

### Fixed
- **Downloads keep their speed past 2 GB.** The console slows a title's own writes to about
  1.2 MB/s after the first ~2.2 GB, while a payload's keep going, so the app now writes each piece
  through a loopback ftpsrv: in file order, with `REST`/`STOR` at its place, done once ftpsrv answers
  `226`. It uses the user's ftpsrv on 2121 when a test write through it works, else its own copy,
  `ps5lm-ftpsrv` on `127.0.0.1:2123`, which it sends to elfldr when needed and which leaves other
  ftpsrv copies on 2121 and 2122 running. With an ftpsrv that has no `EPSV` it uses `PASV`. When
  no ftpsrv answers, the app writes the file itself as before. Ported from
  [PS5SD](https://github.com/Darkatek7/PS5SD) (#41).
- Models saved to an M.2 or USB drive show up in the library wherever that drive is mounted. The
  library looked only in `/mnt/ext0`, `/mnt/usb0` and `/mnt/usb1`, while downloads can go to any
  drive the exporter reports; it now also reads the download folder and every mounted drive's
  `PS5LM/models` (#43).
- The SHA-256 check at the end of a download shows how far it has read, once a second, instead of
  a line that did not move for minutes on a large file. The queue is saved when the check starts,
  so a restart during the check repeats only the check, not the download (#30, #51).

## [0.9.1] - 2026-10-20

### Fixed
- A full log queue no longer stops the server. llama.cpp's log holds 512 lines, and the app's
  writer prints each one to `app.log` and `llama.log`; when it fell behind, the server thread
  waited for room, so `/slots` and `/metrics` stopped answering while `/health` still did, the
  hang in #20. On the console a full queue now drops the line and the writer says how many it
  dropped (`patches/0008-log-never-blocks.patch`, by @bilawalriaz in #28).
- Storage on consoles where the HEN does not open `/data` to the app by itself (#16, #23, #24).
  The app asks the HEN at every start, the restart for a model included, by writing
  `{"PID":n}` to `/download0/etahen_jailbreak`, which OnionHEN honours for a title on its
  allowlist: add `PPSA99575` to `app_jailbreak.exact_title_ids` in `/data/OnionHEN/config.ini`.
  If `/data` stays closed, a notification on the TV says so and names the OnionHEN setting.
  From @abhi245y's PS5LM Plus.
- `/data/PS5LM` is created when it is missing.

### Changed
- ps5-homebrew-ui and ps5-opengl are gone from GitHub: the submodule now comes from a mirror at
  the same pinned commit, and [docs/BUILDING.md](docs/BUILDING.md) says where the ps5-opengl SDK
  archive is still found.

## [0.9.0] - 2026-10-16

### Added
- A new screen, designed from scratch: **Home** (the model, its speed, and every request and its
  answer streaming as it happens), **Models** (on this PS5, suggested models measured on the console,
  Hugging Face search and downloads), **API** (the endpoints and an example) and **Settings** (with
  the logs under Diagnostics). Manrope type, interface sounds from PocketVibe, and a full-screen
  splash while the app restarts for a model.
- **Decision models**: llama.cpp's `/v1/systemone` (TypeSafe's API, the one Ollaya serves) with d1,
  d1 Omni, Clef, OpenJev and others. The planner gives them a 2048-token batch and a 4k context.
- MiniCPM-V 4.7's image projector, and the other models of llama.cpp 04f7226.
- `/data/PS5LM/env.txt`: driver and ggml switches set before the GPU opens, for measuring.
- **Console** page: temperature, power, GPU and CPU load charted over the last two minutes or the
  last hour, and a table of every reading.
- Home's request feed reads as a chat, and the console can ask itself: Square sends a demo
  request, Triangle opens a keyboard (with Turkish letters) for a message of your own.
- Settings with several choices open a list under the row; Cross picks, Circle cancels.

### Changed
- The app is **PS5LM** under its own title ID, **PPSA99575**, with a new icon and home screen
  background. PS5LM Plus (PPSA99581) stays @abhi245y's app.
- llama.cpp moves to 04f7226; RADV carries three of PS5CEMU-HAR's driver patches (`patches/mesa`),
  among them freed buffers waiting for the GPU.
- Models with an image projector get a 2048-token ubatch, so images are no longer cut to 64 tokens.

### Fixed
- The app opens on Home, not on the model list, and draws its splash before the GPU opens, so a
  restart for a model no longer shows a black screen.
- Home shows the load as it happens, how much of the model is on the GPU, instead of a still
  splash, and offers Demo and Message only once the server answers: requests sent while the model
  loaded were lost.
- One sound for each kind of action, at one pitch: no two actions share a sound by accident.
- A float in any request (`"temperature": 0.7`) stopped the server: the console's `localeconv()`
  reports an empty decimal point. The app now gives the C locale's.
- An image projector llama.cpp cannot read no longer stops the server: the app restarts once and
  loads the model text only.
- `scripts/build-app.sh` fetches the OpenGL SDK through ps5-homebrew-ui's script, creates `build/`,
  and no longer needs ImageMagick.

### Known issues
- A decision model can stop answering after about 55 requests in a row on the console (the server
  waits on the GPU); restart the app or load the model again. Language models are not affected.

## [0.8.3] - 2026-10-13

### Changed
- The app is now called **PS5LM Plus**, to tell it apart from Mert Cobanov's
  original PS5LM payload: on the home screen, in the dashboard header and in
  the console's notifications. The title ID (PPSA99581), the data folder
  (`/data/PS5LM`) and the repository stay as they were, so updating keeps
  settings, models and downloads.
- The dashboard header shows the chat's address without "Chat at http://",
  so it clears the page tabs with any IP address.

## [0.8.2] - 2026-10-13

### Changed
- `contentVersion` in `sce_sys/param.json` now follows the app's version
  (major+1, minor, patch: 0.8.2 is `01.008.002`). It had stayed at
  `01.000.000` since the first release, so stores such as ProsperoStore
  could not tell an installed copy was out of date. `scripts/release-app.sh`
  refuses to package a release whose `contentVersion` doesn't match.
- README: new screenshots from v0.8.1 (Qwen 3.8 27B generating, the library,
  the loaded-model panel) and a shorter text.

## [0.8.1] - 2026-10-13

### Fixed
- The chat server stopped answering after the network dropped: when the
  console's link went down, accept() failed with `SCE_NET_EINACTIVEDISABLED`
  (errno 163, the same `0x804101a3` a dropped download reported), cpp-httplib
  took that as fatal and closed the listening socket, and the model stayed
  loaded with nothing to reach it. llama-server now binds again once a second
  until it works (`patches/0005-listener-survives-network-drops.patch`).
  Checked on the console: the listener came back within a second after each
  of dozens of drops. A reply streaming at the moment of a drop is still cut.

### Added
- README: the web UI with a sample chat.

### Found
- The repeated "Ethernet disconnected" notifications are the console's link
  going down and up (`eth0: link state changed to DOWN` in the kernel log,
  up to eight times in a few seconds), with or without PS5LM running: a cable,
  port or switch problem, not the app.

## [0.8.0] - 2026-10-13

### Added
- Downloads page (the fifth on L1/R1): a queue kept in
  `/data/PS5LM/downloads.json`, each entry with its speed, time left and
  state. Cross pauses or resumes (or tries a failed one again), Square twice
  cancels and deletes the partial file, or clears a finished one.
- Downloads survive a dropped connection, a pause and the app restarting: the
  file is fetched in 32 MiB pieces on two connections, the part done from its
  start is saved with the queue, and a lost connection is retried on its own
  (4, 8, 16, 32, then 60 s apart, up to 20 times in a row). The SHA-256 is
  checked once the file is whole.
- Get models: Square checks whether the file under the cursor fits, and the
  row says CHECKING while its header is read.
- A busy screen (a spinner and what is happening) while a model is switched or
  unloaded, before the restart for "unload and download", and while the
  models are planned again after a context or cache change.
- README rewritten around the app: features, the payloads it needs, controls.
  The payload's README moved to `docs/PAYLOAD.md`.

### Fixed
- A download that lost its connection failed and threw away what it had
  (2.0 of 10.4 GB, `sceHttp2ReadData 0x804101a3`); now it carries on.
- A connection that went quiet could hang a download: requests now time out
  after 30 s without data and are asked again.
- Settings froze the app for seconds on every change, each one reading every
  model's header again. Only the context and cache settings re-plan now, in
  the background behind the busy screen; the others apply at once.

### Measured (download speed on the console)
- The PC on the same network: 19.6 MB/s from Hugging Face on one connection.
- libSceHttp2 (HTTP/2), one connection: 2.8 to 5.9 MB/s, varying more from run
  to run than with any setting. Reads of 256 KiB gave 0.65 MB/s (each read
  waits about half a second), 4 MiB and larger 3 to 6 MB/s. Four requests on
  one template share one connection (3.5 to 6 MB/s); on four templates, made
  one at a time, 2.8 MB/s; on two, 4.2 MB/s. Made in four threads at once,
  the templates crashed the app.
- libSceHttp2 takes at most 4 requests (512 KiB pool); 8 or 16 are refused
  (`0x817b121c`).
- libSceHttp (HTTP/1.1): 0.44 MB/s per connection whatever the read or receive
  block size, adding up with more connections (4: 1.6, 8: 2.5 MB/s, one of
  the eight refused).
- In the app, two pieces at once on their own templates: 7.0 to 9.8 MB/s for
  a 1.9 GiB file (v0.7: 3.8 MB/s), still well under the PC's speed. Going
  further needs sockets and TLS of our own, for a later release.

### Checked on the console
- A 1.9 GiB download: paused (the partial file stopped growing), resumed from
  the saved 128 MiB, closed and relaunched mid-download (carried on from
  905 MiB), finished with its SHA-256 matching.
- Get models during a download: search, listing and fit checks work.
- The busy screen on a context change; the library re-planned behind it.
- Payloads stayed up throughout.

## [0.7.0] - 2026-10-10

### Added
- Downloads check free space first: the file and 1 GiB to spare must fit on
  the drive. When it does not, a dialog offers the drive with the most room
  for this one download; the download itself refuses too, for the
  after-restart path.
- M.2 and every mounted USB drive: Download to in Settings steps through the
  drives that are mounted (internal, `/mnt/ext0`, `/mnt/usb0..3`), the
  STORAGE tile and the usage details list each one, and the library groups
  models by drive. Models in `PS5LM/models` on the M.2 drive are found too.
- Before a download while a model holds the memory (under 2 GiB free), a
  dialog offers to unload it first: the app restarts without the model and
  the download starts on its own.

### Fixed
- Storage figures were made up: a title's `statvfs` is a stub that reports
  64 GiB with 16 GiB free for every path, and `statfs` is not exported to
  titles. Free and total space now come from ps5-exporter (bundled since
  v0.4.0), which reads the kernel's; internal storage is `/user`. Without the
  exporter the space shows as unknown and does not block a download.
- Get models opened after a restart now lists the popular repositories.
- A download without a checksum survives the unload-and-download restart.

### Checked on the console
- Real space: internal 166.7 of 627.6 GiB free, USB 461 of 477 GiB, as
  ps5-exporter reports them; the drive's 33 MB second partition is left out.
- A 500 GiB job refused on internal storage; a download to the USB drive;
  Download to stepping between internal and USB, saved; unload and download
  (SmolLM2 Q2_K); the scratch size warning and Clear.
- No M.2 drive is fitted, so that path is checked in the preview only.

## [0.6.0] - 2026-10-10

### Added
- Get models page (the fourth on L1/R1): Hugging Face's GGUF repositories,
  most downloaded or by family (L2/R2), each file's size and whether it fits,
  planned from its first 16 MiB, and downloads to internal storage or USB
  (Settings), SHA-256 checked, added to the library when done. HTTPS through
  the system's own libSceHttp2 and libSceSsl. Checked on the console:
  SmolLM2-135M F16 (271 MB) at 3.4 MB/s.
- Chat tools: the browser chat's model can read, write, edit and search files
  (llama-server's built-in tools), confined to `/data/PS5LM/scratch` by
  `patches/0003-tools-root.patch`. Settings has a switch, a size warning
  (default 2 GiB) and a Clear action.
- Images: a vision model's `mmproj` file beside it is loaded with it
  (`--mmproj`), and the planner counts its memory.
- The planner can plan from the first bytes of a GGUF (a range request), the
  base of the model market's fit column.

### Measured
- Writes from the app: stdio 18.5 MB/s, write() 232 MB/s on /data (USB: 293
  and 621 MB/s). Downloads use write().
- Chat tools on the console: a write to /data/homebrew lands in the scratch
  folder; relative paths start there. Gemma 4 E4B with its projector
  described a test image correctly.

## [0.5.0] - 2026-10-10

### Added
- `docs/CLI.md`: opencode, codex and Claude Code against the model on the
  console, each checked on a small edit with Qwen3.8-27B UD-IQ2_XXS.

### Fixed
- Claude Code got HTTP 500 on every request: it sends system messages in the
  middle of a conversation and Qwen's template refuses them.
  `patches/0004` folds them into the first system message.

## [0.4.0] - 2026-10-10

### Added
- POWER card: the SoC's power (CPU, GPU and memory together), its history,
  energy used this session and joules per generated token; expanded, min,
  average and max, fan duty and every temperature sensor. The reading is
  `sceKernelGetSocPowerConsumption`'s low 32 bits in milliwatts (22 W idle,
  86 W while the 27B generates), through ps5-exporter.
- USAGE card: GPU and CPU on one chart; expanded, GPU, CPU, memory, storage
  and the models on disk.

### Changed
- USAGE replaces the GPU and CPU tiles; THERMALS shows the hottest sensor.

### Fixed
- Speed and GPU use during a long reply: llama-server's counters move only
  when a request ends, so the live figures come from the slot's token count.

## [0.3.0] - 2026-10-10

### Added
- Pages on L1/R1: Dashboard, Settings and Logs, with tabs in the header.
- Settings page (rail and rows after the kit's Control Room design): load a
  model at launch and which one, the planner's longest context and KV cache
  type, interface sounds; saved to `/data/PS5LM/settings.json`.
- Logs page: `app.log` and `llama.log`, scrollable, with an errors-only filter.
- Model details on Cross over the Model tile: every llama-server argument, the
  plan, cache and preset; Triangle opens the library.
- THERMALS tile (SoC and CPU temperature, fan duty), STORAGE tile (free space
  on /data and USB), and the CPU clock. A title may not read the sensors
  (`0x80020002`), so the app bundles Marice's ps5-exporter (GPL-3.0, pinned
  v0.2.0) and sends it to elfldr when it is not running.
- `scripts/ps5lm-app.sh press <button>`: drive the screens from a PC.

### Changed
- One MEMORY tile replaces MEMORY and GPU MEMORY (the GPU and CPU share one
  pool); CONTEXT takes in the session counts.

## [0.2.0] - 2026-10-10

First release of the native app (title PPSA99581), from this fork.

### Added
- Native PS5 app running llama-server on the GPU through ggml-vulkan and RADV,
  with an 11.44 GiB unified device heap.
- Dashboard on the TV (ps5-homebrew-ui): generation speed, GPU busy, CPU use,
  memory pool, GPU memory split, session, context and model tiles.
- Model library with internal storage and USB drive sections; models load
  straight from USB.
- Planner that picks context length and KV cache type for the memory free.
- Sampling presets for Qwen 3.5/3.6/3.8, Gemma 4, gpt-oss, Granite, Nemotron,
  Mistral and Llama.
- Model switching by restarting the app (in-place unload leaks GPU memory).
- `scripts/ps5lm-app.sh` to drive the app from a PC, and the version on the
  dashboard and in the log.

### Fixed
- Unloading during a reply no longer hangs the app: it waits at most 10 s for
  llama-server before restarting.

### Changed
- A plain launch loads no model and opens the library.
