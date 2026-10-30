// Host check for ftp_writer.cpp:
//   c++ -std=c++17 -O1 ps5/app/ftp_writer_test.cpp ps5/app/ftp_writer.cpp -o build/ftp_writer_test
// (scripts/test-native-build.sh runs it in CI)
//   build/ftp_writer_test
// A path with a line break is refused before anything connects, so no server
// is needed: a path that got past the check would fail to connect instead.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "ftp_writer.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ps5lm {
// ftp_writer.cpp starts PS5LM's own ftpsrv with it; nothing here gets that far.
bool send_payload(const char *, int) {
    return false;
}
}  // namespace ps5lm

namespace {

bool refused(bool ok, const std::string & error) {
    return !ok && error.find("line break") != std::string::npos;
}

}  // namespace

int main() {
    ps5lm::FtpWriter w;
    uint64_t         size = 0;
    for (const std::string & path : { std::string("x.gguf\r\nDELE /data/a"), std::string("x\n.gguf"),
                                      std::string("x\0.gguf", 7) }) {
        std::string error;
        assert(refused(w.write_at(path, 0, "x", 1, error), error));
        error.clear();
        assert(refused(w.size(path, size, error), error));
    }
    std::string error;
    assert(!refused(w.size("/data/PS5LM/models/x.gguf.part", size, error), error));
    std::puts("ftp_writer_test: ok");
}
