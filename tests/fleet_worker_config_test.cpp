// Portable safety-gate checks; no PS5 SDK, network, or ggml runtime required.
// c++ -std=c++17 -Wall -Wextra -Werror -I ps5/app tests/fleet_worker_config_test.cpp -o build/fleet-worker-config-test
#include "fleet_worker.hpp"

#include <cstdlib>
#include <iostream>

int main() {
    using ps5lm::FleetWorkerConfig;
    using ps5lm::parse_fleet_worker_config;
    const std::string ack = "\nI_UNDERSTAND_RPC_IS_UNAUTHENTICATED\n";
    int checks = 0;
    auto check = [&](bool ok, const std::string & label) {
        ++checks;
        if (!ok) {
            std::cerr << "FAIL: " << label << '\n';
            std::exit(1);
        }
    };
    for (const std::string endpoint : { "10.0.0.1:50052", "172.16.0.1:1024", "172.31.255.254:65535",
                                        "192.168.1.2:50052", "127.0.0.1:50052" }) {
        const auto config = parse_fleet_worker_config(endpoint + ack);
        check(config.state == FleetWorkerConfig::enabled && config.endpoint == endpoint && config.error.empty(), endpoint);
    }
    for (const std::string endpoint : { "", "0.0.0.0:50052", "8.8.8.8:50052", "172.15.0.1:50052",
            "172.32.0.1:50052", "192.169.0.1:50052", "localhost:50052", "ps5:50052",
            "10.0.0.1", "10.0.0.1:", "10.0.0.1:0", "10.0.0.1:1023", "10.0.0.1:65536",
            "10.0.0.1:-1", "10.0.0.1:+50052", "10.0.0.1:050052", "10.0.0.1:50052junk",
            "010.0.0.1:50052", "10.0.0.256:50052", "10.0.1:50052", "10.0.0.1.2:50052",
            "10.0.0.1:50052:50053", "10.0.0.0:50052", "10.0.0.255:50052", "[::1]:50052",
            " 10.0.0.1:50052", "10.0.0.1:50052 ", "127.0.0.2:50052", "169.254.0.1:50052" }) {
        const auto config = parse_fleet_worker_config(endpoint + ack);
        check(config.state == FleetWorkerConfig::invalid && config.endpoint.empty() && !config.error.empty(), endpoint);
    }
    for (const std::string & text : std::vector<std::string>{ "", "10.0.0.1:50052\n", "10.0.0.1:50052\nYES\n",
            "10.0.0.1:50052" + ack + "extra\n", "10.0.0.1:50052" + ack + "\n",
            std::string(1025, 'x'), std::string("10.0.0.1:50052\0", 16) + ack }) {
        check(parse_fleet_worker_config(text).state == FleetWorkerConfig::invalid, "invalid contents");
    }
    check(parse_fleet_worker_config("192.168.1.2:50052\r\nI_UNDERSTAND_RPC_IS_UNAUTHENTICATED\r\n").state == FleetWorkerConfig::enabled, "CRLF");
    check(parse_fleet_worker_config("192.168.1.2:50052\nI_UNDERSTAND_RPC_IS_UNAUTHENTICATED").state == FleetWorkerConfig::enabled, "no final newline");
    std::cout << "PASS: " << checks << " fleet worker config checks\n";
}
