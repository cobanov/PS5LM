// SPDX-License-Identifier: GPL-3.0-or-later
#include "storage.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace ps5lm {

namespace {

std::string g_root = kFullRoot;

constexpr const char *  kParam  = "/app0/sce_sys/param.json";
constexpr const char *  kChoice = "/download0/PS5LM/storage-choice";
constexpr uint64_t      kDefaultSize = 16ull << 30;

}  // namespace

void set_data_root(const std::string & root) { g_root = root; }

const std::string & data_root() { return g_root; }

bool app_storage() { return g_root == kAppRoot; }

std::string data_path(const std::string & name) { return g_root + "/" + name; }

uint64_t folder_bytes(const std::string & path) {
    std::error_code ec;
    uint64_t total = 0;
    for (auto it = std::filesystem::recursive_directory_iterator(path, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec)) {
            total += it->file_size(ec);
        }
    }
    return total;
}

uint64_t app_storage_size() {
    // "downloadDataSize": 16384, in MiB.
    static const uint64_t size = [] {
        std::stringstream text;
        text << std::ifstream(kParam).rdbuf();
        const std::string json = text.str();
        const size_t key = json.find("\"downloadDataSize\"");
        const size_t colon = key == std::string::npos ? key : json.find(':', key);
        const uint64_t mib = colon == std::string::npos ? 0 : std::strtoull(json.c_str() + colon + 1, nullptr, 10);
        return mib ? mib << 20 : kDefaultSize;
    }();
    return size;
}

// Everything in /download0 counts, the HEN's request file and the app's notes included.
uint64_t app_storage_used() { return folder_bytes("/download0"); }

double app_storage_free_gib() {
    const uint64_t size = app_storage_size(), used = app_storage_used();
    return (double) (size > used ? size - used : 0) / 1073741824.0;
}

std::string drive_name(const std::string & path) {
    return path.rfind("/mnt/ext", 0) == 0 ? "M.2 drive" : path.rfind("/mnt/usb", 0) == 0 ? "USB drive"
         : path.rfind("/download0", 0) == 0 ? "App storage" : "Internal storage";
}

std::string storage_choice() {
    std::string choice;
    std::ifstream(kChoice) >> choice;
    return choice;
}

void set_storage_choice(const std::string & choice) {
    std::error_code ec;
    std::filesystem::create_directories(kAppRoot, ec);
    std::ofstream(kChoice) << choice << "\n";
}

}  // namespace ps5lm
