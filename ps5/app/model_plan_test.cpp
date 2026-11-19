// Host check for model_plan.cpp:
//   c++ -std=c++17 -O1 ps5/app/model_plan_test.cpp ps5/app/model_plan.cpp -o build/model_plan_test
//   build/model_plan_test          synthetic headers only
//   build/model_plan_test models   also the real GGUF files in models/
// Uses the budget the console reported (11.38 GiB free) and the two
// configurations measured there.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "model_plan.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// A GGUF header with string and uint32 metadata, no tensors.
struct Gguf {
    std::string kvs;
    uint64_t    n = 0;
    template <class T> static std::string raw(T v) { return std::string((const char *) &v, sizeof(v)); }
    static std::string str(const std::string & v) { return raw<uint64_t>(v.size()) + v; }
    Gguf & kv(const std::string & k, const std::string & v) { kvs += str(k) + raw<uint32_t>(8) + str(v); n++; return *this; }
    Gguf & kv(const std::string & k, uint32_t v) { kvs += str(k) + raw<uint32_t>(4) + raw(v); n++; return *this; }
    Gguf & kv(const std::string & k, const std::vector<bool> & v) {  // a bool array
        kvs += str(k) + raw<uint32_t>(9) + raw<uint32_t>(7) + raw<uint64_t>(v.size());
        for (bool b : v) {
            kvs += (char) b;
        }
        n++;
        return *this;
    }
    // magic, version 3, tensor count, metadata count, metadata
    std::string bytes() const { return "GGUF" + raw<uint32_t>(3) + raw<uint64_t>(0) + raw(n) + kvs; }
};

bool has_args(const ps5lm::Plan & p, const char * flag, const char * value) {
    auto it = std::find(p.args.begin(), p.args.end(), flag);
    return it != p.args.end() && it + 1 != p.args.end() && *(it + 1) == value;
}

}  // namespace

int main(int argc, char ** argv) {
    const double budget = 11.38;

    // A decision model (Clef is Qwen 3.5 9B with a decision head): found from
    // "<arch>.decision.type", a wide ubatch, a 4k cap and no chat sampling.
    Gguf clef;
    clef.kv("general.architecture", "qwen35").kv("general.name", "Clef Flash Qwen3.5 9B")
        .kv("qwen35.block_count", 32u).kv("qwen35.attention.head_count", 16u)
        .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 4096u)
        .kv("qwen35.context_length", 262144u).kv("qwen35.decision.type", "clef");
    const ps5lm::ModelInfo d = ps5lm::read_model_header("clef-flash-Q4_K_M.gguf", clef.bytes(), 5ull << 30);
    assert(d.ok && d.decision == "clef");
    assert(!ps5lm::find_preset(d));
    const ps5lm::Plan dp = ps5lm::plan_model(d, budget);
    assert(dp.fits && dp.ctx == 4096 && has_args(dp, "-ub", "2048") && has_args(dp, "-b", "2048"));
    assert(dp.why.find("decision model (clef)") == 0);

    // The same model without the decision key is a chat model with the narrow batch.
    Gguf chat;
    chat.kv("general.architecture", "qwen35").kv("general.name", "Qwen3.5 9B")
        .kv("qwen35.block_count", 32u).kv("qwen35.attention.head_count", 16u)
        .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 4096u)
        .kv("qwen35.context_length", 262144u);
    const ps5lm::ModelInfo c = ps5lm::read_model_header("Qwen3.5-9B-Q4_K_M.gguf", chat.bytes(), 5ull << 30);
    assert(c.ok && c.decision.empty() && ps5lm::find_preset(c));
    const ps5lm::Plan cp = ps5lm::plan_model(c, budget);
    assert(cp.fits && cp.ctx > 4096 && has_args(cp, "-ub", "64"));

    // A larger prompt batch costs compute buffer, counted by the feed-forward width.
    Gguf wide_ff = chat;
    wide_ff.kv("qwen35.feed_forward_length", 12288u);
    const ps5lm::ModelInfo w = ps5lm::read_model_header("Qwen3.5-9B-Q4_K_M.gguf", wide_ff.bytes(), 5ull << 30);
    const ps5lm::Plan w64 = ps5lm::plan_model(w, budget), w512 = ps5lm::plan_model(w, budget, 65536, nullptr, 512);
    assert(w.n_ff == 12288 && has_args(w64, "-ub", "64") && has_args(w512, "-ub", "512") && has_args(w512, "-b", "512"));
    assert(w512.reserve_gib - w64.reserve_gib > 0.18 && w512.reserve_gib - w64.reserve_gib < 0.21);

    // Bonsai 2: upstream's Q2_0 loads with its preset (64k at most, PrismML's
    // sampling); PrismML's own PQ2_0 and PTQ1_0 say what to do instead.
    const auto bonsai = [](uint32_t ftype) {
        Gguf g;
        g.kv("general.architecture", "qwen35").kv("general.name", "Ternary-Bonsai-2-27B")
            .kv("qwen35.block_count", 64u).kv("qwen35.attention.head_count", 24u)
            .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 5120u)
            .kv("qwen35.attention.key_length", 256u).kv("qwen35.attention.value_length", 256u)
            .kv("qwen35.full_attention_interval", 4u).kv("qwen35.context_length", 262144u)
            .kv("prism.hadamard.version", 1u).kv("general.file_type", ftype);
        return g.bytes();
    };
    const ps5lm::ModelInfo b = ps5lm::read_model_header("Ternary-Bonsai-2-27B-Q2_0.gguf", bonsai(41), 7626008928ull);
    const ps5lm::Preset * bp = ps5lm::find_preset(b);
    assert(b.ok && bp && std::string(bp->label) == "Bonsai 2");
    const ps5lm::Plan bpl = ps5lm::plan_model(b, 12.0, 131072);
    assert(bpl.fits && bpl.ctx == 65536 && has_args(bpl, "--min-p", "0.05"));
    const ps5lm::ModelInfo pq = ps5lm::read_model_header("Ternary-Bonsai-2-27B-PQ2_0.gguf", bonsai(141), 7206168928ull);
    assert(!pq.ok && pq.error.find("bonsai-pq2-to-q2.py") != std::string::npos);
    const ps5lm::ModelInfo pt = ps5lm::read_model_header("Ternary-Bonsai-2-27B-PTQ1_0.gguf", bonsai(143), 5946648928ull);
    assert(!pt.ok && pt.error.find("PTQ1_0") != std::string::npos);

    // Sliding-window attention (#57): Spark-X2.6 4B's header. 27 of its 36
    // layers keep a 512-token window, so 256k fits; counted as full attention
    // it stopped at 128k.
    std::vector<bool> swa;
    for (int i = 0; i < 36; i++) {
        swa.push_back(i % 4 != 3);
    }
    Gguf spark;
    spark.kv("general.architecture", "spark2_5").kv("spark2_5.block_count", 36u)
        .kv("spark2_5.attention.head_count", 16u).kv("spark2_5.attention.head_count_kv", 4u)
        .kv("spark2_5.embedding_length", 2560u).kv("spark2_5.attention.key_length", 256u)
        .kv("spark2_5.attention.value_length", 256u).kv("spark2_5.context_length", 1048576u)
        .kv("spark2_5.attention.sliding_window", 512u).kv("spark2_5.attention.sliding_window_pattern", swa);
    const ps5lm::ModelInfo s = ps5lm::read_model_header("Spark-X2.6-4B-Q8_0.gguf", spark.bytes(), 4375021824ull);
    assert(s.ok && s.n_swa_layers == 27 && s.swa_window == 512);
    assert(kv_bytes_per_token(s, "q8_0") == 9 * 4 * 512 * 34.0 / 32);
    const ps5lm::Plan sp = ps5lm::plan_model(s, budget, 262144);
    assert(sp.fits && sp.ctx == 262144 && sp.kv_type == "q8_0");
    ps5lm::ModelInfo full = s;
    full.n_swa_layers = 0;
    assert(ps5lm::plan_model(full, budget, 262144).ctx == 131072);

    // A Qwen projector beside its model takes images in pieces: a narrow
    // ubatch, a 1280 batch and image cap, and the headroom kept. Another
    // projector type keeps the wide ubatch.
    {
        char dir[] = "/tmp/model_plan_test.XXXXXX";
        assert(mkdtemp(dir));
        const std::string at = dir;
        const auto write = [&](const std::string & name, const Gguf & g) {
            std::ofstream(at + "/" + name, std::ios::binary) << g.bytes();
        };
        Gguf projector;
        projector.kv("general.architecture", "clip").kv("clip.projector_type", "qwen3vl_merger");
        write("Qwen3.8-27B-UD-IQ2_S.gguf", chat);
        write("mmproj-Qwen3.8-27B-F16.gguf", projector);
        const ps5lm::ModelInfo m = ps5lm::read_model_info(at + "/Qwen3.8-27B-UD-IQ2_S.gguf");
        assert(m.mmproj == at + "/mmproj-Qwen3.8-27B-F16.gguf" && m.projector_type == "qwen3vl_merger");
        const ps5lm::Plan vp = ps5lm::plan_model(m, budget, 8192);
        assert(has_args(vp, "-ub", "256") && has_args(vp, "-b", "1280") && has_args(vp, "--image-max-tokens", "1280"));
        assert(vp.reserve_gib >= 0.75 + 0.25 + 0.5);
        assert(!ps5lm::causal_projector("gemma3") && ps5lm::causal_projector("qwen2vl_merger"));
        ps5lm::ModelInfo g = m;
        g.projector_type = "gemma3";
        const ps5lm::Plan gp = ps5lm::plan_model(g, budget, 8192);
        assert(has_args(gp, "-ub", "2048") && has_args(gp, "-b", "2048"));
        assert(std::find(gp.args.begin(), gp.args.end(), "--image-max-tokens") == gp.args.end());
        unlink((at + "/Qwen3.8-27B-UD-IQ2_S.gguf").c_str());
        unlink((at + "/mmproj-Qwen3.8-27B-F16.gguf").c_str());
        rmdir(dir);
    }

    // Tight-VRAM planner, -ctxcp 1 vs 0 accounting, and --cache-disk modes:
    {
        const auto no_arg = [](const ps5lm::Plan & p, const char * flag) {
            return std::find(p.args.begin(), p.args.end(), flag) == p.args.end();
        };
        // Never emit -ctxcp 4 across any mode/budget (preserves patches/0010 single-checkpoint fast path).
        const ps5lm::Plan c_off  = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, "off");
        const ps5lm::Plan c_auto = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, "auto");
        const ps5lm::Plan c_on   = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, "on");
        assert(c_off.fits && has_args(c_off, "-ctxcp", "1") && has_args(c_off, "-cram", "0") && no_arg(c_off, "--cache-disk"));
        assert(c_auto.fits && has_args(c_auto, "-ctxcp", "1") && has_args(c_auto, "-cram", "0") && no_arg(c_auto, "--cache-disk"));
        assert(c_on.fits && has_args(c_on, "-ctxcp", "1") && has_args(c_on, "-cram", "0") &&
               has_args(c_on, "--cache-disk", "/data/PS5LM/cache/prompts"));
        assert(!has_args(c_auto, "-ctxcp", "4") && !has_args(c_on, "-ctxcp", "4"));

        // Exact 2 * state_gib on normal (-ctxcp 1) vs 1 * state_gib on tight (-ctxcp 0) compaction.
        ps5lm::ModelInfo hyb = c;
        hyb.state_bytes = 160ull << 20;
        const ps5lm::Plan hyb_norm = ps5lm::plan_model(hyb, budget, 65536, nullptr, ps5lm::kPromptUbatch, "off");
        assert(std::fabs(hyb_norm.reserve_gib - (0.75 + hyb_norm.state_gib)) < 1e-6);
        assert(std::fabs(hyb_norm.total_gib - (hyb_norm.weights_gib + hyb_norm.kv_gib + 2.0 * hyb_norm.state_gib + 0.75)) < 1e-6);

        // Tight 18B MoE (10.59 GiB weights + 160 MiB state): "auto" unlocks tight compaction (-ub 32, -ctxcp 0)
        // and auto-drops 0.25 GiB introspect buffer when needed to keep >= 8k context with >= 0.15 GiB margin.
        Gguf moe;
        moe.kv("general.architecture", "qwen35moe").kv("general.name", "Qwen3.5 18B A3B")
            .kv("qwen35moe.block_count", 48u).kv("qwen35moe.attention.head_count", 16u)
            .kv("qwen35moe.attention.head_count_kv", 2u).kv("qwen35moe.embedding_length", 2048u)
            .kv("qwen35moe.attention.key_length", 128u).kv("qwen35moe.attention.value_length", 128u)
            .kv("qwen35moe.full_attention_interval", 4u).kv("qwen35moe.context_length", 262144u)
            .kv("qwen35moe.expert_count", 64u);
        ps5lm::ModelInfo moe18 = ps5lm::read_model_header("qwen3.5-18b-a3b-Q4_K_M.gguf", moe.bytes(), 11375418272ull);
        moe18.state_bytes = 160ull << 20;
        const ps5lm::Plan p18_off  = ps5lm::plan_model(moe18, 11.74, 65536, "q8_0", ps5lm::kPromptUbatch, "off");
        const ps5lm::Plan p18_auto = ps5lm::plan_model(moe18, 11.74, 65536, "q8_0", ps5lm::kPromptUbatch, "auto", 8192);
        assert(p18_auto.fits && p18_auto.ctx >= 8192 && has_args(p18_auto, "-ctxcp", "0") &&
               has_args(p18_auto, "-ub", "32") && has_args(p18_auto, "-b", "128") && has_args(p18_auto, "-cram", "0") &&
               has_args(p18_auto, "--cache-disk", "/data/PS5LM/cache/prompts"));
        assert(!p18_off.fits || p18_auto.ctx >= p18_off.ctx);
        assert((11.74 - p18_auto.total_gib) >= 0.15);

        const ps5lm::Plan p18_intro = ps5lm::plan_model(moe18, 11.56, 65536, "q8_0", ps5lm::kPromptUbatch,
                                                        "auto", 8192, "/data/PS5LM/cache/prompts", true);
        assert(p18_intro.fits && p18_intro.disable_introspect && p18_intro.ctx >= 8192);
        assert((11.56 - p18_intro.total_gib) >= 0.15);
        assert(has_args(p18_intro, "-ctxcp", "0") && !has_args(p18_intro, "-ctxcp", "4"));

        // Tight 27B hybrid (IQ2_M, 9.54 GiB + 160 MiB state):
        // - "off" stays conservative (-ub 64 -ctxcp 1) and drops to 16k at 11.60 GiB
        // - "auto" + q8_0 @ 11.74 GiB with introspect keeps 32k q8_0 by auto-dropping introspect & compacting (-ctxcp 0, >= 0.32 GiB margin)
        // - "on" + q8_0 @ 11.30 GiB with introspect expands 16k -> 32k q4_0 ("(expanded)") with >= 0.32 GiB margin
        ps5lm::ModelInfo iq2_27b;
        iq2_27b.ok = true; iq2_27b.file_bytes = 10243816864ull; iq2_27b.n_layer = 64; iq2_27b.full_attn_every = 4;
        iq2_27b.n_head_kv = 4; iq2_27b.key_length = iq2_27b.value_length = 256; iq2_27b.state_bytes = 160ull << 20;

        const ps5lm::Plan p27_off = ps5lm::plan_model(iq2_27b, 11.74, 65536, "q8_0", ps5lm::kPromptUbatch, "off");
        assert(p27_off.fits && p27_off.ctx == 32768 && p27_off.kv_type == "q8_0" &&
               has_args(p27_off, "-ctxcp", "1") && has_args(p27_off, "-ub", "64") && no_arg(p27_off, "--cache-disk"));
        const ps5lm::Plan p27_off_tight = ps5lm::plan_model(iq2_27b, 11.60, 65536, "q8_0", ps5lm::kPromptUbatch, "off");
        assert(p27_off_tight.fits && p27_off_tight.ctx == 16384 && p27_off_tight.kv_type == "q8_0");

        const ps5lm::Plan p27_q8 = ps5lm::plan_model(iq2_27b, 11.74, 65536, "q8_0", ps5lm::kPromptUbatch,
                                                     "auto", 2048, "/data/PS5LM/cache/prompts", true);
        assert(p27_q8.fits && p27_q8.ctx == 32768 && p27_q8.kv_type == "q8_0" && p27_q8.disable_introspect &&
               has_args(p27_q8, "-ctxcp", "0") && has_args(p27_q8, "-ub", "32") &&
               has_args(p27_q8, "--cache-disk", "/data/PS5LM/cache/prompts") &&
               (11.74 - p27_q8.total_gib) >= 0.32);

        const ps5lm::Plan p27_on_1130 = ps5lm::plan_model(iq2_27b, 11.30, 65536, "q8_0", ps5lm::kPromptUbatch,
                                                          "on", 2048, "/data/PS5LM/cache/prompts", true);
        assert(p27_on_1130.fits && p27_on_1130.ctx == 32768 && p27_on_1130.kv_type == "q4_0" &&
               p27_on_1130.disable_introspect && has_args(p27_on_1130, "-ctxcp", "0") &&
               p27_on_1130.why.find("(expanded)") != std::string::npos && (11.30 - p27_on_1130.total_gib) >= 0.32);

        const ps5lm::Plan p27_auto = ps5lm::plan_model(iq2_27b, 11.74, 65536, "auto", ps5lm::kPromptUbatch,
                                                       "auto", 2048, "/data/PS5LM/cache/prompts", true);
        assert(p27_auto.fits && p27_auto.ctx == 32768 && p27_auto.kv_type == "q4_0" &&
               has_args(p27_auto, "-ctxcp", "1") && !has_args(p27_auto, "-ctxcp", "4"));

        // Edge cases: invalid disk_cache_mode -> auto, 0 MiB omits --cache-disk, ctx_cap honored strictly.
        const ps5lm::Plan fb = ps5lm::plan_model(iq2_27b, 11.74, 65536, "auto", ps5lm::kPromptUbatch, "bogus", 2048);
        assert(fb.fits && fb.ctx == 65536 && has_args(fb, "--cache-disk", "/data/PS5LM/cache/prompts") &&
               has_args(fb, "--cache-disk-mib", "2048"));
        const ps5lm::Plan zero_mib = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, "on", 0);
        assert(zero_mib.fits && no_arg(zero_mib, "--cache-disk"));
        const ps5lm::Plan big_disk = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, "on", 32768);
        assert(big_disk.fits && big_disk.ctx == 65536 && has_args(big_disk, "--cache-disk-mib", "32768"));
        const ps5lm::Plan cap_128k = ps5lm::plan_model(c, budget, 131072, "auto", ps5lm::kPromptUbatch, "auto", 4096);
        assert(cap_128k.fits && cap_128k.ctx == 131072);
    }

    if (argc < 2) {
        std::puts("model_plan_test: ok (synthetic only)");
        return 0;
    }
    bool saw_q2k = false, saw_iq2xxs = false;
    for (const auto & path : ps5lm::find_models({ argv[1] })) {
        const ps5lm::ModelInfo m = ps5lm::read_model_info(path);
        assert(m.ok);
        const ps5lm::Plan p = ps5lm::plan_model(m, budget);
        const ps5lm::Preset * pre = ps5lm::find_preset(m);
        std::printf("%-34s %s L%u kvL%u kvh%u k%u v%u state %.0f MiB | %s | %s\n", m.file_name.c_str(),
                    m.arch.c_str(), m.n_layer, m.n_kv_layers(), m.n_head_kv, m.key_length, m.value_length,
                    m.state_bytes / 1048576.0, p.why.c_str(), pre ? pre->label : "-");
        if (m.arch == "qwen35") {
            // Qwen 3.8 27B: 16 of 64 layers keep a KV cache, 4 heads of 256.
            assert(m.n_layer == 64 && m.n_kv_layers() == 16 && m.n_head_kv == 4 && m.key_length == 256);
            assert(kv_bytes_per_token(m, "f16") == 65536.0);
            assert(m.state_bytes > 140.0 * 1048576 && m.state_bytes < 170.0 * 1048576);
            assert(pre && std::string(pre->label) == "Qwen 3.8");
        }
        if (m.file_name.find("UD-Q2_K_XL") != std::string::npos) {
            // Ran on the console: 64k with a q4_0 cache.
            assert(p.fits && p.ctx == 65536 && p.kv_type == "q4_0");
            saw_q2k = true;
        }
        if (m.file_name.find("UD-IQ2_XXS") != std::string::npos) {
            // 4k at f16 ran; 64k needs at most q8_0 at this size.
            assert(p.fits && p.ctx == 65536 && p.kv_type == "q8_0");
            saw_iq2xxs = true;
        }
    }
    assert(saw_q2k && saw_iq2xxs);
    // Nothing fits in too little memory.
    ps5lm::ModelInfo big;
    big.ok = true; big.file_bytes = 20ull << 30; big.n_layer = 32; big.n_head_kv = 8;
    big.key_length = big.value_length = 128;
    assert(!ps5lm::plan_model(big, budget).fits);
    std::puts("model_plan_test: ok");
}
