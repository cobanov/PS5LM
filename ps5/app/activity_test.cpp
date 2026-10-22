// Host check for activity.cpp, with payloads llama-server sent:
//   c++ -std=c++20 -O1 -Ithird_party/llama.cpp/vendor ps5/app/activity_test.cpp ps5/app/activity.cpp
//       ps5/app/stats.cpp ps5/app/storage.cpp -o build/activity_test -pthread && build/activity_test
// SPDX-License-Identifier: GPL-3.0-or-later
#include "stats.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

extern "C" {
void ps5lm_activity_begin(int id, const char * method, const char * path, const char * remote, const char * body,
                          size_t body_len);
void ps5lm_activity_data(int id, const char * data, size_t len);
void ps5lm_activity_end(int id, int status);
}

namespace {

int g_failed = 0;

// One request through the hooks, its answer in pieces of `chunk` bytes, as a
// stream arrives; what the Home feed would show.
ps5lm::Exchange run(ps5lm::StatsCollector & stats, int id, const char * path, const std::string & body,
                    const std::string & answer, size_t chunk) {
    ps5lm_activity_begin(id, "POST", path, "192.168.1.24", body.data(), body.size());
    for (size_t at = 0; at < answer.size(); at += chunk) {
        const std::string piece = answer.substr(at, chunk);
        ps5lm_activity_data(id, piece.data(), piece.size());
    }
    ps5lm_activity_end(id, 200);
    for (const auto & e : stats.snapshot().activity) {
        if (e.id == id) {
            return e;
        }
    }
    return {};
}

void expect(const ps5lm::Exchange & e, const char * asked, const char * answered) {
    if (e.asked != asked || e.answered != answered || e.status != 200) {
        std::printf("FAIL %s\n  asked:    \"%s\" (want \"%s\")\n  answered: \"%s\" (want \"%s\")\n", e.path.c_str(),
                    e.asked.c_str(), asked, e.answered.c_str(), answered);
        ++g_failed;
    }
}

}  // namespace

int main() {
    ps5lm::StatsCollector stats;
    ps5lm::set_activity_sink(&stats);

    // OpenAI Responses, whole.
    expect(run(stats, 1, "/v1/responses", R"({"input":"Say hi in one word.","max_output_tokens":30})",
               R"({"object":"response","status":"completed","output":[{"content":[{"type":"output_text","annotations":[],)"
               R"("logprobs":[],"text":"Hi."}],"id":"msg_1","role":"assistant","status":"completed","type":"message"}]})", 1 << 20),
           "Say hi in one word.", "Hi.");

    // OpenAI Responses, streamed: the deltas once, not again from the closing events.
    const std::string sse =
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n"
        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"content\":[],\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"type\":\"message\"}}\n\n"
        "event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"item_id\":\"msg_1\",\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n"
        "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"One\"}\n\n"
        "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\", two\"}\n\n"
        "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\", three.\"}\n\n"
        "event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"item_id\":\"msg_1\",\"text\":\"One, two, three.\"}\n\n"
        "event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"item_id\":\"msg_1\",\"part\":{\"type\":\"output_text\",\"text\":\"One, two, three.\"}}\n\n"
        "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"One, two, three.\"}],\"role\":\"assistant\"}}\n\n"
        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"One, two, three.\"}],\"role\":\"assistant\"}]}}\n\n";
    const std::string items = R"({"input":[{"role":"user","content":[{"type":"input_text","text":"Count to three."}]}],"stream":true})";
    expect(run(stats, 2, "/v1/responses", items, sse, 37), "Count to three.", "One, two, three.");
    expect(run(stats, 3, "/v1/responses", items, sse, 1 << 20), "Count to three.", "One, two, three.");

    // Chat Completions and Anthropic Messages, as before.
    expect(run(stats, 4, "/v1/chat/completions", R"({"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hello?"}],"stream":true})",
               "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo!\"}}]}\n\ndata: [DONE]\n\n", 11),
           "Hello?", "Hello!");
    expect(run(stats, 5, "/v1/messages", R"({"messages":[{"role":"user","content":[{"type":"text","text":"Hi Claude"}]}],"stream":true})",
               "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi \"}}\n\n"
               "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"there\"}}\n\n", 5),
           "Hi Claude", "Hi there");

    std::printf(g_failed ? "activity_test: %d failed\n" : "activity_test: all passed\n", g_failed);
    return g_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
