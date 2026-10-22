// Unit tests with a deliberately fake backend. No GPU or socket is opened.
// Build with fleet_worker.cpp and include ps5/app plus third_party/llama.cpp/ggml/include.
#include "fleet_worker.hpp"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

extern "C" {
void ggml_backend_load_all() {}
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
    if (cache || threads != 6 || count != 1 || devices[0] != &gpu || !no_comm || std::strcmp(no_comm, "1") != 0) {
        std::abort();
    }
    ++rpc_calls;
    last_endpoint = endpoint;
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
    const std::string text = "192.168.1.2:50052\nI_UNDERSTAND_RPC_IS_UNAUTHENTICATED\n";
    const auto valid = parse_fleet_worker_config(text);
    check(!run_fleet_worker({}).empty() && rpc_calls == 0, "disabled cannot open listener");
    check(!run_fleet_worker({FleetWorkerConfig::enabled, "0.0.0.0:50052", ""}).empty() && rpc_calls == 0,
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
          last_endpoint == valid.endpoint, "one Vulkan device, return is failure not success");
    check(std::string(std::getenv("GGML_RPC_NO_COMM")) == "1", "worker disables extra listeners");
    throw_from_rpc = true;
    check(run_fleet_worker(valid).find("test failure") != std::string::npos && rpc_calls == 2, "exception reported");

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
