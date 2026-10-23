// The models the Models page suggests: the ones that run well on this console,
// picked by hand and checked on it. Each is one file (and its image encoder,
// when it reads images) from Hugging Face, pinned by size and sha256.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>

namespace ps5lm {

struct SuggestedFile {
    const char * file;      // in the repository
    uint64_t     bytes;
    const char * sha256;
    const char * save_as;   // the name on the console, or null for the same
};

struct Suggestion {
    const char *  name;     // what the list calls it
    bool          decision; // answers /v1/systemone, not chat
    const char *  about;    // one line: what it is good at here
    float         tps;      // tokens a second it generates on the console, 0 when not measured
    const char *  repo;
    SuggestedFile model;
    SuggestedFile mmproj;   // file null: text only
};

inline constexpr Suggestion kSuggestions[] = {
    // UD-Q2_K_XL, not the smaller IQ2_XXS: the K-quants' kernels run faster here
    // (20 against 15 tokens a second), and it still holds a 64k context.
    { "Qwen 3.8 27B", false, "The strongest model that fits, with a 64k context", 20, "unsloth/Qwen3.8-27B-GGUF",
      { "Qwen3.8-27B-UD-Q2_K_XL.gguf", 9828981664ull, "fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0", nullptr },
      {} },
    // Bonsai 2 27B, PrismML's ternary Qwen 3.8, as upstream's Q2_0: the app's
    // llama.cpp applies its Hadamard rotation (prism.hadamard.*). PrismML keeps
    // this file in its -dev repo under a "prism-fork-required" name from before
    // upstream had the rotation; it is byte for byte what scripts/bonsai-pq2-to-q2.py
    // makes from the main repo's PQ2_0, and it answers correctly with
    // ggml-vulkan on a PC. Not yet run on the console.
    { "Bonsai 2 27B", false, "Qwen 3.8 27B in ternary weights, with a 64k context", 0, "prism-ml/Ternary-Bonsai-2-27B-gguf-dev",
      { "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf", 7626008928ull,
        "4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e", "Ternary-Bonsai-2-27B-Q2_0.gguf" },
      {} },
    { "Qwen 3.5 9B", false, "Fast all-rounder that leaves room for long chats", 55, "unsloth/Qwen3.5-9B-GGUF",
      { "Qwen3.5-9B-Q4_K_M.gguf", 5680522464ull, "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8", nullptr },
      {} },
    { "Gemma 4 E4B", false, "Reads images, quick replies", 0, "ggml-org/gemma-4-E4B-it-GGUF",
      { "gemma-4-E4B-it-Q4_0.gguf", 4590807392ull, "a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2", nullptr },
      { "mmproj-gemma-4-E4B-it-Q8_0.gguf", 559874816ull, "197f49a93027f9843772bd24a6a9e0be2a32a788de5a3def330e9c585d86edd1", nullptr } },
    { "MiniCPM-V 4.7 1B", false, "Small and very fast, reads images", 205, "openbmb/MiniCPM-V-4.7-1B-gguf",
      { "MiniCPM-V-4_7-1B-Q4_K_M.gguf", 672003680ull, "588c6c94c82f756b86dba35c42ed9ba9ec187d65824eb52f039c347df563e63d", nullptr },
      { "mmproj-model-f16.gguf", 1110117984ull, "fb9cc047a3cdad5317fd8433a4aeda2d22f00f02ca4bd93b34ad84165b25a900",
        "mmproj-MiniCPM-V-4_7-1B-f16.gguf" } },
    { "Qwen 3.5 0.8B", false, "Tiny and instant, for simple tasks", 207, "unsloth/Qwen3.5-0.8B-GGUF",
      { "Qwen3.5-0.8B-Q4_K_M.gguf", 532517120ull, "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517", nullptr },
      {} },
    { "d1 Omni 600M", true, "Decisions about text, images and sound in a fraction of a second", 0, "LiquidAI/d1-omni-600M-GGUF",
      { "d1-omni-600M-BF16.gguf", 764049056ull, "9bf4f344835f64264978907a0b0bc43c9f0e59804f15176f8b5630c930c3c9df", nullptr },
      { "mmproj-d1-omni-600M-BF16.gguf", 442441792ull, "1aed7f0ff41f354c66cf3fa41c9eb43f1a76b3ea9453edd8e804567674a9041e", nullptr } },
    { "d1 3B", true, "Sharper decisions about text and images", 0, "LiquidAI/d1-3B-GGUF",
      { "d1-3B-Q8_0.gguf", 2874781280ull, "2f0942d5a5f64cf69c3356d2be439b71644b1f0eb3a580912a5a0179eeaba77d", nullptr },
      { "mmproj-d1-3B-Q8_0.gguf", 583109728ull, "2505920ce464b7c54acd8b7b74a931c0c671f06d0c8833760b2fbceeb253e92a", nullptr } },
    { "Clef Flash", true, "Answers every question of a request together", 0, "ggml-org/Clef-Flash-GGUF",
      { "Clef-Flash-Q4_K_M.gguf", 6486448288ull, "3626116623ca8d89641952471fac9ce12af84fb3517520029a2ac6f5146868a2", nullptr },
      { "mmproj-Clef-Flash-Q8_0.gguf", 624229728ull, "9dd8bc279f57d33b0697d4d0fe900fe751a24c9e3d30ab90969d4714ffc0722d", nullptr } },
};
inline constexpr int kSuggestionCount = (int) (sizeof(kSuggestions) / sizeof(kSuggestions[0]));

}  // namespace ps5lm
