// Unit tests with a deliberately fake backend. No GPU or socket is opened.
// Build with fleet_worker.cpp and include ps5/app plus third_party/llama.cpp/ggml/include.
#include "fleet_worker.hpp"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

struct ggml_backend_reg { int unused; };
struct ggml_backend_device { enum ggml_backend_dev_type type; };
static ggml_backend_reg registry{};
static ggml_backend_device gpu{GGML_BACKEND_DEVICE_TYPE_IGPU};
static bool have_vulkan = true, throw_from_rpc = false;
static size_t device_count = 1;
static int rpc_calls = 0;
static std::string last_endpoint;
static const char * last_cache = nullptr;

extern "C" {
void ggml_backend_load_all() {}
void ggml_log_set(ggml_log_callback, void *) {}
ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    return have_vulkan && std::strcmp(name, "Vulkan") == 0 ? &registry : nullptr;
}
size_t ggml_backend_reg_dev_count(ggml_backend_reg_t) { return device_count; }
ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t, size_t) { return &gpu; }
enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) { return device->type; }
const char * ggml_backend_dev_name(ggml_backend_dev_t) { return "FakeVulkan0"; }
void ggml_backend_rpc_start_server(const char * endpoint, const char * cache, size_t threads,
                                   size_t count, ggml_backend_dev_t * devices) {
    const char * no_comm = std::getenv("GGML_RPC_NO_COMM");
    if (threads != 6 || count != 1 || devices[0] != &gpu || !no_comm || std::strcmp(no_comm, "1") != 0) {
        std::abort();
    }
    ++rpc_calls;
    last_endpoint = endpoint;
    last_cache = cache;
    if (throw_from_rpc) {
        throw std::runtime_error("test failure");
    }
}
}

int main() {
    using namespace ps5lm;
    int checks = 0;
    auto check = [&](bool ok, const char * label) {
        ++checks;
        if (!ok) {
            std::cerr << "FAIL: " << label << '\n';
            std::exit(1);
        }
    };
    const std::string text = "192.168.1.2:30052\nI_UNDERSTAND_RPC_IS_UNAUTHENTICATED\n";
    const auto valid = parse_fleet_worker_config(text);
    check(!run_fleet_worker({}).empty() && rpc_calls == 0, "disabled cannot open listener");
    check(!run_fleet_worker({FleetWorkerConfig::enabled, "0.0.0.0:30052", ""}).empty() && rpc_calls == 0,
          "forged wildcard config revalidated");
    have_vulkan = false;
    check(run_fleet_worker(valid).find("No Vulkan") != std::string::npos && rpc_calls == 0, "no Vulkan");
    have_vulkan = true;
    device_count = 0;
    check(run_fleet_worker(valid).find("No Vulkan GPU") != std::string::npos && rpc_calls == 0, "empty backend");
    device_count = 2;
    check(run_fleet_worker(valid).find("exactly one") != std::string::npos && rpc_calls == 0, "ambiguous devices");
    device_count = 1;
    gpu.type = GGML_BACKEND_DEVICE_TYPE_CPU;
    check(run_fleet_worker(valid).find("No Vulkan GPU") != std::string::npos && rpc_calls == 0, "no CPU fallback");
    gpu.type = GGML_BACKEND_DEVICE_TYPE_IGPU;
    setenv("GGML_RPC_NO_COMM", "0", 1);  // Worker must override any app env setting.
    check(run_fleet_worker(valid).find("stopped") != std::string::npos && rpc_calls == 1 &&
          last_endpoint == valid.endpoint && last_cache == nullptr, "one Vulkan device, return is failure not success");
    check(std::string(std::getenv("GGML_RPC_NO_COMM")) == "1", "worker disables extra listeners");
    throw_from_rpc = true;
    check(run_fleet_worker(valid).find("test failure") != std::string::npos && rpc_calls == 2, "exception reported");
    throw_from_rpc = false;

    // The weight cache: off unless asked for, whole GiB, refused values say so.
    check(rpc_cache_config(nullptr, "/c").dir.empty() && rpc_cache_config(nullptr, "/c").error.empty(), "cache off by default");
    check(rpc_cache_config("0", "/c").dir.empty() && rpc_cache_config("0", "/c").error.empty(), "cache 0 is off");
    check(rpc_cache_config("32", "/c").dir == "/c" && rpc_cache_config("32", "/c").limit == (32ull << 30), "cache 32 GiB");
    check(rpc_cache_config("4096", "/c").limit == (4096ull << 30), "cache upper bound");
    for (const char * bad : { "", "-1", "032", "4097", "12G", "1.5", "99999" }) {
        const RpcCache refused = rpc_cache_config(bad, "/c");
        check(refused.dir.empty() && !refused.error.empty(), "bad cache size refused");
    }
    {
        char dir[] = "/tmp/ps5lm-rpc-cache-XXXXXX";
        check(mkdtemp(dir) != nullptr, "temporary cache folder");
        const std::string root = std::string(dir) + "/rpc-cache";
        RpcCache cache = rpc_cache_config("1", root);
        check(prune_rpc_cache(cache, true).files == 0 && access(root.c_str(), F_OK) == 0, "cache folder made at start");
        run_fleet_worker(valid, cache);
        check(last_cache && root == last_cache, "worker gets the cache folder");
        cache.limit = 5u << 20;  // small files for the test: the limit is bytes
        const auto write = [&](const std::string & name, size_t mib, int age_s) {
            const std::string path = root + "/" + name;
            std::ofstream(path, std::ios::binary) << std::string(mib << 20, 'w');
            const auto t = std::filesystem::file_time_type::clock::now() - std::chrono::seconds(age_s);
            std::filesystem::last_write_time(path, t);
            return path;
        };
        const std::string oldest = write("00000000000000a1", 2, 300);
        const std::string middle = write("00000000000000a2", 2, 200);
        const std::string newest = write("00000000000000a3", 2, 100);
        const std::string part = write("00000000000000a4.00000000000000ff.part", 1, 50);
        const std::string other = write("notes.txt", 1, 400);
        const std::string upper = write("00000000000000A5", 1, 500);  // not a name the cache writes
        RpcCacheUse use = prune_rpc_cache(cache, false);
        check(use.files == 2 && use.bytes == (4ull << 20) && use.dropped == 1 && access(oldest.c_str(), F_OK) != 0 &&
              access(middle.c_str(), F_OK) == 0 && access(newest.c_str(), F_OK) == 0, "least recently used dropped first");
        check(access(part.c_str(), F_OK) == 0, "a write in progress is left alone while the worker runs");
        use = prune_rpc_cache(cache, true);
        check(access(part.c_str(), F_OK) != 0 && use.dropped == 1 && use.files == 2, "a cut-off write is removed at start");
        check(access(other.c_str(), F_OK) == 0 && access(upper.c_str(), F_OK) == 0, "other files are never touched");
        check(prune_rpc_cache(RpcCache{}, true).files == 0, "no cache, nothing done");
        std::filesystem::remove_all(dir);
    }

    char path[] = "/tmp/ps5lm-fleet-config-XXXXXX";
    int fd = mkstemp(path);
    check(fd >= 0, "temporary config file");
    close(fd);
    check(read_fleet_worker_config(path).state == FleetWorkerConfig::invalid, "empty file fails closed");
    std::ofstream(path) << text;
    check(read_fleet_worker_config(path).state == FleetWorkerConfig::enabled, "file enabled");
    std::ofstream(path) << std::string(1025, 'x');
    check(read_fleet_worker_config(path).state == FleetWorkerConfig::invalid, "oversized file fails closed");
    unlink(path);
    check(read_fleet_worker_config(path).state == FleetWorkerConfig::disabled, "missing file disabled");
    check(read_fleet_worker_config("/tmp").state == FleetWorkerConfig::invalid, "directory fails closed");
    check(mkfifo(path, 0600) == 0, "temporary FIFO");
    check(read_fleet_worker_config(path).state == FleetWorkerConfig::invalid, "FIFO fails without blocking");
    unlink(path);
    check(symlink("/not-a-file-ps5lm-fleet", path) == 0, "temporary dangling symlink");
    check(read_fleet_worker_config(path).state == FleetWorkerConfig::invalid, "dangling link does not disable fleet mode");
    unlink(path);
    std::cout << "PASS: " << checks << " fleet worker runtime/file checks (fake backend, no network)\n";
}
