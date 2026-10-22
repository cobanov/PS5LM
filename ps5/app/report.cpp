// SPDX-License-Identifier: GPL-3.0-or-later
#include "report.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace ps5lm {

namespace {

constexpr const char * kRemoved = "[removed]";

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

bool ends_with(const std::string & s, const char * tail) {
    const size_t n = std::strlen(tail);
    return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}

bool name_char(char c) { return std::isalnum((unsigned char) c) || c == '_' || c == '-' || c == '.'; }

// llama-server's options whose value is a secret.
bool secret_flag(const std::string & word) {
    static const char * flags[] = { "--api-key", "--api-key-file", "--hf-token", "-hft", "--ssl-key-file" };
    for (const char * f : flags) {
        if (word == f) {
            return true;
        }
    }
    return false;
}

// A key whose value is a secret: "api_key", "hf_token", "client_secret",
// "password". Not "max_tokens", "n_tokens" or "eos_token_id".
bool secret_key(std::string key) {
    key = lower(key);
    const size_t dot = key.find_last_of('.');
    if (dot != std::string::npos) {
        key = key.substr(dot + 1);  // "env.HF_TOKEN", "x.api_key"
    }
    static const char * exact[] = { "token", "apikey", "api_key", "api-key", "x-api-key", "authorization", "secret",
                                    "password", "passwd", "bearer" };
    for (const char * e : exact) {
        if (key == e) {
            return true;
        }
    }
    static const char * tails[] = { "_api_key", "-api-key", "_apikey", "_secret", "-secret", "_password", "-password",
                                    "access_token", "auth_token", "hf_token", "api_token", "refresh_token", "id_token",
                                    "bearer_token", "github_token", "-token" };
    for (const char * t : tails) {
        if (ends_with(key, t)) {
            return true;
        }
    }
    return false;
}

// Where a value that starts at `at` ends: a quote closes a quoted one, else
// space or punctuation does.
size_t value_end(const std::string & s, size_t at) {
    if (at < s.size() && (s[at] == '"' || s[at] == '\'')) {
        const char q = s[at];
        const size_t close = s.find(q, at + 1);
        return close == std::string::npos ? s.size() : close + 1;
    }
    size_t e = at;
    while (e < s.size() && !std::isspace((unsigned char) s[e]) && s[e] != ',' && s[e] != ';' && s[e] != '&' && s[e] != '}' &&
           s[e] != ')') {
        ++e;
    }
    return e;
}

// Replaces s[from, to) with [removed], keeping the quotes of a quoted value.
void blank(std::string & s, size_t from, size_t to) {
    if (to <= from) {
        return;
    }
    if (to - from >= 2 && (s[from] == '"' || s[from] == '\'') && s[to - 1] == s[from]) {
        ++from;
        --to;
    }
    s.replace(from, to - from, kRemoved);
}

// Strings shaped like an access token, whatever key they are under.
void redact_shapes(std::string & s) {
    struct Shape {
        const char * prefix;
        size_t       min_tail;
    };
    static const Shape shapes[] = { { "hf_", 30 }, { "sk-", 20 }, { "ghp_", 30 }, { "gho_", 30 }, { "ghs_", 30 },
                                    { "github_pat_", 30 }, { "xoxb-", 20 }, { "xoxp-", 20 } };
    for (const auto & sh : shapes) {
        const size_t plen = std::strlen(sh.prefix);
        for (size_t at = s.find(sh.prefix); at != std::string::npos; at = s.find(sh.prefix, at + 1)) {
            if (at > 0 && (std::isalnum((unsigned char) s[at - 1]) || s[at - 1] == '_')) {
                continue;  // inside a word: "mask_hf_x"
            }
            size_t e = at + plen;
            while (e < s.size() && (std::isalnum((unsigned char) s[e]) || s[e] == '_' || s[e] == '-')) {
                ++e;
            }
            if (e - at - plen >= sh.min_tail) {
                s.replace(at, e - at, kRemoved);
            }
        }
    }
}

// One line: flags and their values, "Header: value", key=value and
// "key": "value".
std::string redact_line(std::string s, bool & next_is_secret) {
    // The value of a flag on the line before ("ps5lm-app: arg --api-key", then
    // "ps5lm-app: arg <key>"), or app-args.txt's one argument a line.
    if (next_is_secret) {
        next_is_secret = false;
        size_t end = s.size();
        while (end > 0 && std::isspace((unsigned char) s[end - 1])) {
            --end;
        }
        size_t start = end;
        while (start > 0 && !std::isspace((unsigned char) s[start - 1])) {
            --start;
        }
        if (end > start) {
            s.replace(start, end - start, kRemoved);
        }
    }
    // Words: "--api-key X", "--api-key=X", and a flag ending the line.
    for (size_t at = 0; at < s.size();) {
        while (at < s.size() && std::isspace((unsigned char) s[at])) {
            ++at;
        }
        size_t e = at;
        while (e < s.size() && !std::isspace((unsigned char) s[e])) {
            ++e;
        }
        std::string word = s.substr(at, e - at);
        const size_t eq = word.find('=');
        if (eq != std::string::npos && secret_flag(word.substr(0, eq))) {
            blank(s, at + eq + 1, e);
            e = at + eq + 1 + std::strlen(kRemoved);
        } else if (!word.empty() && secret_flag(word)) {
            size_t v = e;
            while (v < s.size() && std::isspace((unsigned char) s[v])) {
                ++v;
            }
            if (v < s.size()) {
                const size_t ve = value_end(s, v);
                blank(s, v, ve);
                e = v + std::strlen(kRemoved);
            } else {
                next_is_secret = true;
            }
        }
        at = e;
    }
    // "name: value", "name=value", "name": "value", for secret names.
    for (size_t at = 0; at < s.size(); ++at) {
        if (!name_char(s[at]) || (at > 0 && name_char(s[at - 1]))) {
            continue;
        }
        size_t e = at;
        while (e < s.size() && name_char(s[e])) {
            ++e;
        }
        const std::string name = s.substr(at, e - at);
        size_t p = e;
        if (p < s.size() && (s[p] == '"' || s[p] == '\'')) {
            ++p;  // the end of a quoted key
        }
        // The separator right after the name: "token=", "token:", "token":,
        // not llama.cpp's "EOS token        = 248046".
        if (p >= s.size() || (s[p] != ':' && s[p] != '=') || !secret_key(name)) {
            at = e - 1;
            continue;
        }
        const bool header = s[p] == ':' && (lower(name) == "authorization" || ends_with(lower(name), "api-key"));
        ++p;
        while (p < s.size() && s[p] == ' ') {
            ++p;
        }
        if (header) {
            blank(s, p, s.size());  // "Authorization: Bearer <token>": all of it
            break;
        }
        const size_t ve = value_end(s, p);
        if (ve > p) {
            blank(s, p, ve);
        }
        at = p;
    }
    redact_shapes(s);
    return s;
}

void add_lines(std::string & out, const std::vector<std::string> & lines) {
    for (const auto & l : lines) {
        out += "- " + l + "\n";
    }
    if (lines.empty()) {
        out += "- none\n";
    }
}

void add_block(std::string & out, const std::string & title, const std::string & text) {
    out += "\n## " + title + "\n\n```text\n" + text;
    if (!text.empty() && text.back() != '\n') {
        out += "\n";
    }
    out += "```\n";
}

}  // namespace

const std::vector<std::pair<int, std::string>> & report_ports() {
    static const std::vector<std::pair<int, std::string>> ports = {
        { 9021, "ELF loader (elfldr, etaHEN)" }, { 9020, "ELF loader (older HENs)" }, { 2121, "ftpsrv" },
        { 1337, "etaHEN FTP" },                  { 3232, "klogsrv" },                 { 9081, "etaHEN kernel log" },
        { 2323, "shsrv" },                       { 8080, "websrv" },                  { 9100, "ps5-exporter" },
        { 12800, "Payload Manager" },            { 8081, "PS5LM's server" },
    };
    return ports;
}

std::string redact_secrets(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    bool next = false;
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        const bool has_nl = nl != std::string::npos;
        if (!has_nl) {
            nl = text.size();
        }
        out += redact_line(text.substr(at, nl - at), next);
        if (has_nl) {
            out += '\n';
        }
        at = nl + 1;
    }
    return out;
}

std::string last_bytes(const std::string & text, size_t max_bytes) {
    if (text.size() <= max_bytes) {
        return text;
    }
    const size_t from = text.size() - max_bytes;
    const size_t nl = text.find('\n', from);
    return nl == std::string::npos ? text.substr(from) : text.substr(nl + 1);
}

std::string build_report(const ReportInput & in) {
    std::string out = "# PS5LM debug report\n\n";
    out += "Attach this file to a GitHub issue. API keys and tokens are removed from it.\n\n";
    out += "- Created: " + in.created + " (the console's clock)\n";
    out += "- PS5LM: " + in.version + "\n";
    out += "- System software: " + (in.firmware.empty() ? std::string("unknown") : in.firmware) + "\n";
    out += "- Storage: " + in.storage + "\n";

    out += "\n## Drives\n\n";
    add_lines(out, in.drives);
    out += "\n## Memory\n\n";
    add_lines(out, in.memory);
    out += "\n## Server\n\n";
    add_lines(out, in.server);
    if (!in.args.empty()) {
        std::string args;
        for (const auto & a : in.args) {
            args += a + "\n";
        }
        add_block(out, "llama-server arguments", redact_secrets(args));
    }

    out += "\n## Payload ports\n\n";
    for (const auto & [port, what] : in.ports) {
        const bool open = std::find(in.open_ports.begin(), in.open_ports.end(), port) != in.open_ports.end();
        out += "- " + std::to_string(port) + " (" + what + "): " + (open ? "open" : "closed") + "\n";
    }

    out += "\n## Models\n\n";
    add_lines(out, in.models);
    out += "\n## Downloads\n\n";
    add_lines(out, in.downloads);

    add_block(out, "settings.json", redact_secrets(in.settings_json));
    for (const auto & [name, text] : in.files) {
        add_block(out, name, redact_secrets(text));
    }
    for (const auto & [name, text] : in.logs) {
        add_block(out, name, redact_secrets(text));
    }
    if (in.kernel_log_asked) {
        add_block(out, "Kernel log", redact_secrets(in.kernel_log));
    } else {
        out += "\n## Kernel log\n\nNot included (Settings > Debug report > Include the kernel log).\n";
    }
    return out;
}

}  // namespace ps5lm
