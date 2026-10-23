// The debug report: one text file to attach to a GitHub issue, made from
// Settings > Debug report. It holds what a bug report needs and nothing
// secret: the app's version, the console's system software, where the app
// keeps its files and the room on each drive, the models and their plans, the
// settings, the last part of both logs, which payload ports answer, and the
// kernel log only when the user switched it on. API keys and tokens are
// removed from everything that goes in (redact_secrets).
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ps5lm {

// What the report says, gathered by the app (main.cpp); building it is plain
// text work, checked on the PC by report_test.cpp.
struct ReportInput {
    std::string created;          // "2026-10-23 14:32", the console's clock
    std::string version;          // the app's
    std::string firmware;         // "13.60" and the raw word, or empty
    std::string storage;          // where settings, logs and models live, and why
    std::vector<std::string> drives;    // one line each
    std::vector<std::string> memory;    // the pools and ggml's devices
    std::vector<std::string> server;    // state, address, model and plan
    std::vector<std::string> args;      // llama-server's arguments
    std::vector<std::pair<int, std::string>> ports;  // port, what usually listens there
    std::vector<int> open_ports;
    std::vector<std::string> models;    // one line each: file, drive, size, plan
    std::vector<std::string> downloads; // one line each
    std::string settings_json;
    std::vector<std::pair<std::string, std::string>> files;  // other small files: name, text (env.txt, app-args.txt)
    std::vector<std::pair<std::string, std::string>> logs;   // name, its last part
    bool        kernel_log_asked = false;
    std::string kernel_log;       // when asked: the last part of it, or why there is none
};

// Payload ports worth knowing about, and what usually listens on each.
const std::vector<std::pair<int, std::string>> & report_ports();

// The text with API keys, tokens and passwords replaced by [removed]: the
// value after --api-key and its kin, Authorization and API-key headers,
// key=value and "key": "value" pairs whose name says key, token, secret or
// password, and strings shaped like Hugging Face (hf_...), OpenAI (sk-...)
// or GitHub (ghp_...) tokens.
std::string redact_secrets(const std::string & text);

// The last `max_bytes` of a text, from the start of a line.
std::string last_bytes(const std::string & text, size_t max_bytes);

std::string build_report(const ReportInput & in);

}  // namespace ps5lm
