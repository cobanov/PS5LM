// Host check for report.cpp:
//   c++ -std=c++17 -O1 ps5/app/report_test.cpp ps5/app/report.cpp -o build/report_test && build/report_test
// SPDX-License-Identifier: GPL-3.0-or-later
#include "report.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failed = 0;

void expect(const std::string & in, const std::string & want) {
    const std::string got = ps5lm::redact_secrets(in);
    if (got != want) {
        std::printf("FAIL\n  in:   %s\n  got:  %s\n  want: %s\n", in.c_str(), got.c_str(), want.c_str());
        ++g_failed;
    }
}

void expect_kept(const std::string & in) { expect(in, in); }

}  // namespace

int main() {
    using ps5lm::build_report;
    using ps5lm::last_bytes;

    // Flags with a secret value, in every form the app writes them.
    expect("--api-key sk-local-123", "--api-key [removed]");
    expect("-m model.gguf --api-key=abc --port 8081", "-m model.gguf --api-key=[removed] --port 8081");
    expect("ps5lm-app: arg --api-key\nps5lm-app: arg hunter2\nps5lm-app: arg --port",
           "ps5lm-app: arg --api-key\nps5lm-app: arg [removed]\nps5lm-app: arg --port");
    expect("--hf-token\nhf_abcdefghijklmnopqrstuvwxyz0123456789\n-m\nx.gguf", "--hf-token\n[removed]\n-m\nx.gguf");
    expect("--api-key-file /data/PS5LM/key.txt", "--api-key-file [removed]");

    // Headers and key-value pairs.
    expect("Authorization: Bearer abc.def.ghi", "Authorization: [removed]");
    expect("x-api-key: 12345", "x-api-key: [removed]");
    expect("{\"api_key\": \"s3cret\", \"model\": \"ps5\"}", "{\"api_key\": \"[removed]\", \"model\": \"ps5\"}");
    expect("HF_TOKEN=hf_x password=pw1 user=me", "HF_TOKEN=[removed] password=[removed] user=me");
    expect("\"client_secret\":\"abc\"", "\"client_secret\":\"[removed]\"");

    // Token shapes anywhere.
    expect("got hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456 from env", "got [removed] from env");
    expect("key sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUV end", "key [removed] end");
    expect("ghp_0123456789abcdefghijklmnopqrstuvwxyz", "[removed]");

    // What llama.cpp and the app log every day stays as it is.
    expect_kept("slot update_slots: id  0 | task 3 | n_tokens = 512, memory_seq_rm [512, end)");
    expect_kept("srv  log_server_r: request: POST /v1/chat/completions 192.168.8.12 200");
    expect_kept("{\"max_tokens\": 512, \"stream\": true, \"temperature\": 0.7}");
    expect_kept("print_info: EOS token        = 248046 '<|im_end|>'");
    expect_kept("llama_model_loader: - kv  31: tokenizer.ggml.eos_token_id u32 = 248046");
    expect_kept("ps5lm-app: arg -ctk\nps5lm-app: arg q8_0");
    expect_kept("prompt_tokens_total 9120\nllamacpp:tokens_predicted_total 4410");
    expect_kept("mask_hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456");
    expect_kept("load_tensors: Vulkan0 model buffer size = 9369.58 MiB");

    // The last part of a log starts at a line.
    if (last_bytes("one\ntwo\nthree\n", 8) != "three\n" || last_bytes("short", 100) != "short") {
        std::printf("FAIL last_bytes\n");
        ++g_failed;
    }

    // The report itself: every section, secrets out of the logs and settings,
    // the kernel log only when asked.
    ps5lm::ReportInput in;
    in.created = "2026-10-23 14:32";
    in.version = "0.9.2";
    in.firmware = "13.60 (0x13600000)";
    in.storage = "/data/PS5LM, full storage";
    in.drives = { "Internal storage: 120.4 of 825.0 GiB free, models in /data/PS5LM/models" };
    in.args = { "-m", "/data/PS5LM/models/x.gguf", "--api-key", "topsecret" };
    in.ports = ps5lm::report_ports();
    in.open_ports = { 9021, 8081 };
    in.settings_json = "{\"ctx_cap\": 65536}";
    in.logs = { { "app.log (last 256 KiB)", "ps5lm-app: arg --api-key\nps5lm-app: arg topsecret\n" } };
    const std::string r = build_report(in);
    const char * must[] = { "# PS5LM debug report", "System software: 13.60", "## Drives", "## Payload ports",
                            "- 9021 (ELF loader (elfldr, etaHEN)): open", "- 2121 (ftpsrv): closed", "## Models\n\n- none",
                            "## settings.json", "app.log (last 256 KiB)", "Not included" };
    for (const char * m : must) {
        if (r.find(m) == std::string::npos) {
            std::printf("FAIL report lacks: %s\n", m);
            ++g_failed;
        }
    }
    if (r.find("topsecret") != std::string::npos) {
        std::printf("FAIL report holds the API key\n%s\n", r.c_str());
        ++g_failed;
    }
    in.kernel_log_asked = true;
    in.kernel_log = "[KERNEL] eth0: link state changed to UP\n";
    if (build_report(in).find("## Kernel log\n\n```text\n[KERNEL] eth0") == std::string::npos) {
        std::printf("FAIL kernel log missing when asked\n");
        ++g_failed;
    }

    std::printf(g_failed ? "report_test: %d failed\n" : "report_test: all passed\n", g_failed);
    return g_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
