// Experimental, opt-in fleet worker. RPC has no authentication or encryption.
// Never enable it on an untrusted network. SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ps5lm {

struct FleetWorkerConfig {
    enum State { disabled, enabled, invalid } state = disabled;
    std::string endpoint;
    std::string error;
};

// Deliberately tiny, dependency-free parser so the safety gate is host-testable.
// Exactly two nonempty lines: the console's private IPv4:port and the acknowledgement.
// Do not accept hostnames, wildcard addresses, public addresses, or extra options.
inline FleetWorkerConfig parse_fleet_worker_config(const std::string & text) {
    FleetWorkerConfig config;
    config.state = FleetWorkerConfig::invalid;
    config.error = "Expected private IPv4:port and I_UNDERSTAND_RPC_IS_UNAUTHENTICATED on two lines";
    if (text.size() > 1024 || text.find('\0') != std::string::npos) {
        return config;
    }
    std::vector<std::string> lines;
    for (size_t pos = 0; pos < text.size();) {
        const size_t end = text.find('\n', pos);
        std::string line = text.substr(pos, end == std::string::npos ? end : end - pos);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    if (lines.size() != 2 || lines[1] != "I_UNDERSTAND_RPC_IS_UNAUTHENTICATED") {
        return config;
    }
    const std::string & endpoint = lines[0];
    const size_t colon = endpoint.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == endpoint.size()) {
        return config;
    }
    auto number = [](const std::string & s, unsigned limit, unsigned & out) {
        if (s.empty() || s.size() > 5 || (s.size() > 1 && s[0] == '0')) {
            return false;
        }
        out = 0;
        for (char c : s) {
            if (c < '0' || c > '9') {
                return false;
            }
            out = out * 10 + static_cast<unsigned>(c - '0');
            if (out > limit) {
                return false;
            }
        }
        return true;
    };
    const std::string host = endpoint.substr(0, colon);
    unsigned octets[4] = {}, port = 0;
    size_t pos = 0;
    for (size_t i = 0; i < 4; ++i) {
        const size_t dot = host.find('.', pos);
        if ((i < 3) != (dot != std::string::npos) ||
            !number(host.substr(pos, dot == std::string::npos ? dot : dot - pos), 255, octets[i])) {
            return config;
        }
        pos = dot == std::string::npos ? host.size() : dot + 1;
    }
    const bool private_ip = octets[0] == 10 ||
        (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31) ||
        (octets[0] == 192 && octets[1] == 168);
    const bool loopback = host == "127.0.0.1";
    if ((!private_ip && !loopback) || octets[3] == 0 || octets[3] == 255 ||
        !number(endpoint.substr(colon + 1), 39999, port) || port < 1024) {
        // A PS5 title gets EACCES binding port 40000 and up (39999 binds; measured).
        config.error = "Bind only a private console IPv4 (or 127.0.0.1), port 1024..39999; never 0.0.0.0";
        return config;
    }
    config.state = FleetWorkerConfig::enabled;
    config.endpoint = endpoint;
    config.error.clear();
    return config;
}

FleetWorkerConfig read_fleet_worker_config(const char * path);

// The worker's copy of the weights hosts send it (llama.cpp's RPC cache): a host
// that loads the same weights again sends each big tensor's hash, and the worker
// reads the tensor from the console's storage instead of the network.
// patches/fleet/0005 keeps every file whole and checks it against the hash when
// it is read. Off unless env.txt sets PS5LM_RPC_CACHE_GB: then at most that many
// GiB, the least recently used files dropped first (prune_rpc_cache).
struct RpcCache {
    std::string dir;         // empty: no cache
    uint64_t    limit = 0;   // bytes
    std::string error;       // a value that was refused (the cache is then off)
};

constexpr unsigned kRpcCacheMaxGib = 4096;

// PS5LM_RPC_CACHE_GB's value (null when it is not set): 0 or not set is off,
// 1..4096 whole GiB in `dir`.
inline RpcCache rpc_cache_config(const char * gib, const std::string & dir) {
    RpcCache cache;
    if (!gib) {
        return cache;
    }
    unsigned value = 0;
    const std::string text(gib);
    bool ok = !text.empty() && text.size() <= 4 && (text.size() == 1 || text[0] != '0');
    for (char c : text) {
        ok = ok && c >= '0' && c <= '9';
        value = ok ? value * 10 + static_cast<unsigned>(c - '0') : 0;
    }
    if (!ok || value > kRpcCacheMaxGib) {
        cache.error = "PS5LM_RPC_CACHE_GB is whole GiB from 0 to 4096, not \"" + text + "\"; no weight cache";
        return cache;
    }
    if (value > 0) {
        cache.dir = dir;
        cache.limit = static_cast<uint64_t>(value) << 30;
    }
    return cache;
}

// What a cache holds, and what prune_rpc_cache dropped to keep it to its limit.
struct RpcCacheUse {
    uint64_t files = 0, bytes = 0;
    uint64_t dropped = 0, dropped_bytes = 0;
};

// Creates the folder when `at_start` and removes what cut-off writes left
// (*.part; the worker is not running yet then), then drops the least recently
// used cache files until the rest fit the limit. Only names llama.cpp's cache
// writes are touched. Safe beside a running worker: it writes under a temporary
// name, and a file dropped as it is read is a miss, sent again by the host.
RpcCacheUse prune_rpc_cache(const RpcCache & cache, bool at_start);

// Blocks in upstream RPC until it fails; returns the reason, never silently
// falls back to CPU. The title must be closed/restarted to stop this worker.
std::string run_fleet_worker(const FleetWorkerConfig & config, const RpcCache & cache = {});

}  // namespace ps5lm
