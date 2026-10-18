<p align="center">
  <img src="ps5/app/sce_sys/icon0.png" width="128" alt="PS5LM">
</p>

<h1 align="center">PS5LM</h1>

<p align="center">
  <b>Your PS5 runs the language model. Every device on your network uses it.</b><br>
  llama.cpp on the console's GPU, with OpenAI and Anthropic APIs and a screen made for the TV.
</p>

<p align="center">
  <b><a href="https://github.com/cobanov/PS5LM/releases/latest">Get the app</a></b> ·
  <a href="https://ps5lm.cobanov.dev">Website</a> ·
  <a href="#try-it">Get started</a> ·
  <a href="https://github.com/cobanov/PS5LM/issues">Issues</a>
</p>

> **New: decision models, powered by [Ollaya](https://ollaya.dev).** PS5LM now runs Jev-like
> decision models (d1, Clef, OpenJev and more) on the GPU and answers the same `/v1/systemone`
> API as Ollaya: typed decisions with calibrated probabilities, in a fraction of a second.

PS5LM is for anyone with a jailbroken PS5 who wants a private model server at home. It loads a GGUF
model onto the console's GPU and serves it to your laptop, your phone and your coding tools. It needs
no PC while it runs, no cloud and no account.

<p align="center">
  <a href="docs/media/ps5lm-home.mp4"><img src="docs/media/ps5lm-home.gif" width="720" alt="Qwen 3.5 9B generating on a PS5, with the requests streaming across the TV"></a><br>
  <sub>Qwen 3.5 9B answering requests from a laptop on a PS5 Slim. Click for the video.</sub>
</p>

<p align="center">
  <img src="docs/media/app/home.jpg" width="49%" alt="Home: the model, its speed and the live requests">
  <img src="docs/media/app/models.jpg" width="49%" alt="Models: the ones suggested for the PS5">
  <img src="docs/media/app/home-decision.jpg" width="49%" alt="Home with a decision model">
  <img src="docs/media/app/console.jpg" width="49%" alt="Console: temperature, power and load over time">
  <img src="docs/media/app/api.jpg" width="49%" alt="API: the endpoints and an example">
  <img src="docs/media/app/settings.jpg" width="49%" alt="Settings: a list of choices under the row">
  <br><sub>Home, Models, Console, API and Settings on the TV.</sub>
</p>

## A model server under your TV

- **It runs on the GPU.** llama.cpp's Vulkan backend on the console's RDNA 2 GPU, with 11.4 GB of the
  shared memory for the model and its context. Every architecture and quantization llama.cpp knows
  comes along.
- **It answers your network.** OpenAI's `/v1/chat/completions`, Anthropic's `/v1/messages` (Claude Code
  works against it) and a chat page in any browser. Point a client at `http://<console>:8081` and use
  any model name: the loaded one answers.
- **It decides, too.** Load a decision model and the console answers TypeSafe's `/v1/systemone`:
  choice, score and yes-or-no questions about text and images, the API Ollaya serves on a PC.
- **It shows its work.** Every request streams across the TV as a chat: which device asked, what it
  asked, and the reply as it is written. Square sends a demo and Triangle opens a keyboard, so the
  console can be tried from the couch.
- **It watches the console.** A Console page charts temperature, power, GPU and CPU load over the
  last two minutes or the last hour, beside a table of every reading.
- **It picks what fits.** The planner reads each GGUF and gives it the longest context and best cache
  that fit the memory free. A Suggested list holds models measured on the console, and the app
  downloads from Hugging Face by itself, to internal storage, the M.2 drive or a USB drive.
- **It reads images.** Models with an image projector (Gemma 4, MiniCPM-V, d1) take pictures in chat
  and in decisions.

| Model | Size | On a PS5 Slim |
|---|---|---|
| Qwen 3.5 0.8B, Q4_K_M | 0.5 GB | 207 tokens a second |
| Qwen 3.5 9B, Q4_K_M | 5.3 GB | 55 tokens a second |
| Qwen 3.8 27B, UD-Q2_K_XL | 9.2 GB | about 20 tokens a second, 64k context |
| d1 Omni 600M, a decision model | 1.1 GB | 3 questions in 0.3 s; a 720p image and 7 questions in 2.9 s |

## Decision models, powered by Ollaya

[Ollaya](https://github.com/ollaya-dev/ollaya) runs open decision models on a PC the way Ollama runs
language models. PS5LM brings the same idea to the console: a decision model reads a state and
answers typed questions in one forward pass, with no text generated.

```sh
curl http://<console>:8081/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice this month. Refund me or I cancel.",
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"refund": null, "information": null, "cancel": null}},
    "angry":  {"type": "noul", "instructions": "Is the customer angry?"}
  }
}'
```

The Suggested list has d1 Omni 600M, d1 3B and Clef Flash. Any GGUF that carries llama.cpp's
decision metadata loads the same way, and the app labels it a decision model.

## Try it

You need a PS5 on firmware 7.00 to 13.60, jailbroken, with these payloads loaded:

| Payload | For | |
|---|---|---|
| [kstuff-lite](https://github.com/EchoStretch/kstuff-lite) | the kernel patches homebrew runs on | required |
| [etaHEN](https://github.com/etaHEN/etaHEN) | homebrew support | required |
| [ShadowMountPlus](https://github.com/drakmor/ShadowMountPlus) | puts PS5LM on the home screen from `/data/homebrew` | required |
| [elfldr](https://github.com/ps5-payload-dev/elfldr) (port 9021) | temperature, power and real free space, through ps5-exporter | recommended |
| [ftpsrv](https://github.com/ps5-payload-dev/ftpsrv) (port 2121) | installing the app, copying models from a PC | recommended |

1. Download `ps5lm-app-vX.Y.Z.zip` from the [latest release](https://github.com/cobanov/PS5LM/releases/latest) and unzip it.
2. Copy the `PPSA99575` folder to `/data/homebrew` over FTP. ShadowMountPlus adds **PS5LM** to the
   home screen.
3. Open PS5LM, go to **Models > Suggested** and get a model. Or copy a `.gguf` to
   `/data/PS5LM/models`, or to `PS5LM/models` on a USB drive.
4. Load it. Once Home says Ready, open `http://<console>:8081` in a browser, or point your OpenAI or
   Anthropic client at it.

To update, copy the new folder over the old one. Settings, models and downloads stay in
`/data/PS5LM`. Using it from coding tools (opencode, Codex, Claude Code) is in
[docs/CLI.md](docs/CLI.md).

| Where | Button | Does |
|---|---|---|
| Everywhere | L1 / R1 | Home, Models, API, Console, Settings |
| Home | Cross | change the model |
| Home | Square / Triangle | send a demo request / type a message |
| Models | D-pad, Cross | choose a section and a model; load, unload or download it |
| Models, Search | L2 / R2, Square | change the search; check whether a file fits |
| Models, Downloads | Cross, Square | pause or resume; cancel (press twice) or clear |
| Console | L2 / R2 | the last two minutes / the last hour |
| Settings | D-pad, Cross | choose a category and a row; flip it, or pick from its list |
| Settings, Diagnostics | Square, Triangle | app or server log; errors only |

## Looking inside the model

This fork adds an opt-in endpoint that returns the numbers the model computed.
`POST /v1/introspect` answers with the whole logit row at the positions you name,
the probabilities and the entropy, with one row of a named graph tensor for a
named layer, and it can also change one channel in place before the rest of the
model reads it.

The endpoint is off by default. Turn it on in **Settings > Developer > Logits and
activations API**.

```sh
python3 scripts/ps5lm-introspect.py --host <console> --prompt "Once upon a time" \
    --positions last --top-k 10 --tensors l_out --layers 12
```

[Introspection](docs/INTROSPECTION.md) records the request fields, the tensor
names, and the cost of each option.

## Want to tinker?

C++ on [ps5-homebrew-ui](https://github.com/blackbearreloaded/ps5-homebrew-ui), llama.cpp with
ggml-vulkan on [PS5_Vulkan](https://github.com/mihawk-99/PS5_Vulkan)'s RADV, built on Linux.

[Building](docs/BUILDING.md) · [Console notes](docs/CONSOLE.md) · [Porting](docs/PORTING.md) ·
[Research](docs/RESEARCH.md) · [Models that fit](docs/MODELS.md) · [Roadmap](docs/ROADMAP.md) ·
[Changelog](CHANGELOG.md) · [The original payload](docs/PAYLOAD.md)

## Thanks

- [abhi245y](https://github.com/abhi245y), who built the native GPU app this one grew from (PS5LM Plus).
- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp), which do the actual work.
- [Mihawk-99](https://github.com/mihawk-99) for PS5_Vulkan and RADV on the PS5, and
  [PS5CEMU-HAR](https://github.com/premohq/PS5CEMU-HAR) for the driver patches in `patches/mesa`.
- [BlackBearReloaded](https://github.com/blackbearreloaded) for ps5-homebrew-ui, ps5-opengl and
  ps5-homebrew-dev-protocol, and [John Törnblom](https://github.com/ps5-payload-dev) for the SDK.
- [Marice](https://github.com/Marice/ps5-exporter) for ps5-exporter, which the app carries.
- [Manrope](https://github.com/sharanda/manrope) (OFL) for the type, and
  [PocketVibe](https://github.com/cobanov/pocketvibe) for the interface sounds.

---

GPL-3.0-or-later, see [LICENSE](LICENSE). llama.cpp keeps its MIT licence, Manrope its OFL. No Sony
SDK files, keys, firmware or model weights are in this repository.

PS5LM is not affiliated with Sony Interactive Entertainment. "PlayStation" and "PS5" are trademarks
of Sony Interactive Entertainment.
