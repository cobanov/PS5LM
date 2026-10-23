// SPDX-License-Identifier: GPL-3.0-or-later
#include "stats.hpp"
#include "log.hpp"
#include "storage.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef __PROSPERO__
#include <ps5platform/heap.h>
#include <ps5platform/kernel.h>
extern "C" int scePthreadGetaffinity(void * thread, uint64_t * mask);
extern "C" void * scePthreadSelf(void);
// libkernel; signatures as Marice/ps5-exporter and the SDK's hwinfo sample use them.
extern "C" int  sceKernelGetCpuTemperature(int * celsius);
extern "C" int  sceKernelGetSocSensorTemperature(int sensor, int * celsius);
extern "C" int  sceKernelGetSocPowerConsumption(uint64_t * out, double reserved);
extern "C" long sceKernelGetCpuFrequency(void);
#endif

namespace ps5lm {

namespace {

constexpr size_t kHistory = 120;  // seconds of history in the charts
constexpr double kGiB     = 1073741824.0;

void push(std::vector<float> & v, float x) {
    v.push_back(x);
    if (v.size() > kHistory) {
        v.erase(v.begin());
    }
}

// sceKernelGetSocPowerConsumption's first word: milliwatts in the low half.
float soc_watts(uint64_t raw) {
    const uint64_t mw = raw & 0xffffffffu;
    return mw >= 1000 && mw <= 350000 ? (float) (mw / 1000.0) : 0.0f;
}

// Free and total GiB of the filesystem holding `path`, or zeros.
void space(const char * path, double * free, double * total) {
    struct statvfs v = {};
    if (statvfs(path, &v) != 0 || v.f_blocks == 0) {
        *free = *total = 0;
        return;
    }
    *free  = (double) v.f_bavail * v.f_frsize / kGiB;
    *total = (double) v.f_blocks * v.f_frsize / kGiB;
}

}  // namespace

// A title's statvfs is a stub (the same 64 GiB for every path) and statfs is
// not exported to titles, so the space comes from ps5-exporter, a payload.
// /data is /user/data. Without the exporter the space is unknown (total 0).
std::vector<Drive> drives_from_metrics(const std::string & text) {
    std::vector<Drive> out = { { "Internal storage", "/data/PS5LM/models" } };
    const std::string key = "ps5_filesystem_size_bytes{mountpoint=\"";
    for (size_t at = text.find(key); at != std::string::npos; at = text.find(key, at + 1)) {
        const size_t from = at + key.size();
        const std::string mount = text.substr(from, text.find('"', from) - from);
        const size_t brace = text.find('{', at);
        const std::string labels = text.substr(brace, text.find('}', from) + 1 - brace);  // {mountpoint=...,fstype=...}
        const double total = sample(text, "ps5_filesystem_size_bytes" + labels, 0) / kGiB;
        const double free = sample(text, "ps5_filesystem_avail_bytes" + labels, 0) / kGiB;
        if (mount == "/user") {
            out[0].free = free;
            out[0].total = total;
        } else if ((mount.rfind("/mnt/ext", 0) == 0 || mount.rfind("/mnt/usb", 0) == 0) && total >= 1.0) {
            // ponytail: drives under 1 GiB are left out (a USB stick's small partition)
            out.push_back({ drive_name(mount), mount + "/PS5LM/models", free, total });
        }
    }
    // Without full storage the app's own storage stands in for internal storage.
    if (app_storage()) {
        out[0] = { "App storage", std::string(kAppRoot) + "/models", app_storage_free_gib(), app_storage_size() / kGiB };
    }
    return out;
}

std::vector<Drive> mounted_drives() { return drives_from_metrics(http_get(9100, "/metrics", 2)); }


namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double cpu_seconds() {
    rusage u = {};
    getrusage(RUSAGE_SELF, &u);
    return u.ru_utime.tv_sec + u.ru_stime.tv_sec + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e6;
}

int usable_cpus() {
#ifdef __PROSPERO__
    // The CPUs the title may run on (the app is not given all sixteen).
    uint64_t mask = 0;
    if (scePthreadGetaffinity(scePthreadSelf(), &mask) == 0 && mask != 0) {
        return __builtin_popcountll(mask);
    }
#endif
    const unsigned n = std::thread::hardware_concurrency();
    return n ? (int) n : 1;
}

// A JSON integer after "key": in the first slot of /slots (one slot runs).
long json_int(const std::string & text, const char * key) {
    const std::string k = std::string("\"") + key + "\":";
    const size_t at = text.find(k);
    return at == std::string::npos ? 0 : std::strtol(text.c_str() + at + k.size(), nullptr, 10);
}

}  // namespace

std::string http_get(int port, const char * path, int timeout_s) { return http_post(port, path, {}, timeout_s); }

std::string http_post(int port, const char * path, const std::string & body, int timeout_s) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return {};
    }
    timeval tv = { timeout_s, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    sockaddr_in addr = {};
#ifdef __FreeBSD__
    addr.sin_len = sizeof(addr);
#endif
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((uint16_t) port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    std::string out;
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) == 0) {
        std::string req = std::string(body.empty() ? "GET " : "POST ") + path + " HTTP/1.0\r\nHost: localhost\r\n";
        if (!body.empty()) {
            req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
        }
        req += "\r\n" + body;
        send(fd, req.data(), req.size(), 0);
        char buf[4096];
        for (ssize_t got; (got = recv(fd, buf, sizeof(buf), 0)) > 0;) {
            out.append(buf, (size_t) got);
            if (out.size() > (1u << 20)) {
                break;
            }
        }
    }
    close(fd);
    const size_t at = out.find("\r\n\r\n");
    return at == std::string::npos ? std::string() : out.substr(at + 4);
}

double metric(const std::string & text, const char * name) {
    const std::string key = std::string("\nllamacpp:") + name + " ";
    const size_t at = text.find(key);
    return at == std::string::npos ? 0.0 : std::strtod(text.c_str() + at + key.size(), nullptr);
}

// One sample from Prometheus text by its full name and labels
// ("ps5_temperature_celsius{sensor=\"cpu\"}"), or `missing`.
double sample(const std::string & text, const std::string & key, double missing) {
    const size_t at = text.find("\n" + key + " ");
    return at == std::string::npos ? missing : std::strtod(text.c_str() + at + key.size() + 2, nullptr);
}

// Sends a payload to an ELF loader on this console (elfldr, port 9021).
bool send_payload(const char * path, int port) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in to = {};
#ifdef __FreeBSD__
    to.sin_len = sizeof(to);
#endif
    to.sin_family = AF_INET;
    to.sin_port   = htons((uint16_t) port);
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bool ok = fd >= 0 && connect(fd, (sockaddr *) &to, sizeof(to)) == 0;
    char buf[65536];
    for (size_t n; ok && (n = std::fread(buf, 1, sizeof(buf), f)) > 0;) {
        ok = send(fd, buf, n, 0) == (ssize_t) n;
    }
    std::fclose(f);
    if (fd >= 0) {
        close(fd);
    }
    return ok;
}

std::string local_address() {
    const int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        return {};
    }
    sockaddr_in to = {};
#ifdef __FreeBSD__
    to.sin_len = sizeof(to);
#endif
    to.sin_family = AF_INET;
    to.sin_port   = htons(53);
    inet_pton(AF_INET, "1.1.1.1", &to.sin_addr);
    std::string ip;
    if (connect(fd, (sockaddr *) &to, sizeof(to)) == 0) {
        sockaddr_in me = {};
        socklen_t   len = sizeof(me);
        char        buf[INET_ADDRSTRLEN] = {};
        if (getsockname(fd, (sockaddr *) &me, &len) == 0 && inet_ntop(AF_INET, &me.sin_addr, buf, sizeof(buf))) {
            ip = buf;
        }
    }
    close(fd);
    return ip == "0.0.0.0" ? std::string() : ip;
}

void StatsCollector::start(int port) {
    port_ = port;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        live_.cpus = usable_cpus();
        const std::string ip = local_address();
        live_.address = (ip.empty() ? std::string("<console IP>") : ip) + ":" + std::to_string(port);
    }
    std::thread([this] { run(); }).detach();
}

Live StatsCollector::snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_;
}

void StatsCollector::set_model(const Live & m) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_.model_label = m.model_label;
    live_.model_arch  = m.model_arch;
    live_.preset      = m.preset;
    live_.decision    = m.decision;
    live_.model_gib   = m.model_gib;
    live_.ctx         = m.ctx;
    live_.kv_type     = m.kv_type;
    live_.kv_gib      = m.kv_gib;
    live_.model_file  = m.model_file;
    live_.plan        = m.plan;
    live_.args        = m.args;
}

void StatsCollector::set_phase(std::string phase) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_.phase = std::move(phase);
}

void StatsCollector::set_models(const std::vector<ModelRow> & rows) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_.models = rows;
}

namespace {

double steady_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

void StatsCollector::activity_begin(int id, std::string client, std::string path, std::string asked, int questions) {
    std::lock_guard<std::mutex> lock(mutex_);
    Exchange e;
    e.id = id;
    e.client = std::move(client);
    e.path = std::move(path);
    e.asked = std::move(asked);
    e.questions = questions;
    e.started = steady_s();
    live_.activity.push_back(std::move(e));
    if (live_.activity.size() > 6) {
        live_.activity.erase(live_.activity.begin());
    }
}

void StatsCollector::activity_text(int id, const std::string & text) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & e : live_.activity) {
        if (e.id == id) {
            e.answered += text;
            if (e.answered.size() > 4000) {  // the screen shows the end of it
                size_t cut = e.answered.size() - 3000;
                while (cut < e.answered.size() && ((unsigned char) e.answered[cut] & 0xC0) == 0x80) {
                    ++cut;
                }
                e.answered.erase(0, cut);
            }
        }
    }
}

void StatsCollector::activity_end(int id, int status) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & e : live_.activity) {
        if (e.id == id) {
            live_.served += status < 400;
            e.status = status;
            e.seconds = steady_s() - e.started;
        }
    }
}

void StatsCollector::set_scratch(double gib) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_.scratch_gib = gib;
}

void StatsCollector::set_state(ServerState s) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_.state = s;
    // Loading starts from what the pool holds now; progress is measured against
    // what the model and its cache will add.
    load_base_ = s == ServerState::loading ? live_.pool_gib - live_.free_gib : -1;
    live_.load_progress = s == ServerState::loading ? 0.0f : -1.0f;
}

void StatsCollector::run() {
    const double start = now_s();
    double last_wall = start, last_cpu = cpu_seconds();
    double last_busy = 0, last_tokens = 0, last_gen_s = 0;
    bool   have_last = false, last_working = false;
    double last_ptok = 0, last_psec = 0, last_decoded = 0;
    // The hour charts' sums since their last point: temperature, power, GPU, CPU.
    double hour_sum[4] = {};
    int    hour_n[4] = {}, seconds = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const double wall = now_s(), cpu = cpu_seconds();

        const std::string m = http_get(port_, "/metrics");
        // Only real metrics: while the model loads the server already answers,
        // with 503 "Loading model", and that is not ready.
        const bool        up = m.find("llamacpp:") != std::string::npos;
        const double busy_s = metric(m, "prompt_seconds_total") + metric(m, "tokens_predicted_seconds_total");
        const double tokens = metric(m, "tokens_predicted_total");
        const double gen_s  = metric(m, "tokens_predicted_seconds_total");
        const bool   working = metric(m, "requests_processing") > 0;
        std::string  slots;
        if (up) {
            slots = http_get(port_, "/slots");
        }

        double pool = 0, free = 0, heap = 0;
#ifdef __PROSPERO__
        const int64_t pool_bytes = sceKernelGetDirectMemorySize();
        int64_t       at = 0;
        size_t        avail = 0;
        sceKernelAvailableDirectMemorySize(0, pool_bytes, 2 * 1024 * 1024, &at, &avail);
        struct ps5_heap_stats hs = {};
        ps5_heap_stats(&hs);
        pool = pool_bytes / kGiB;
        free = avail / kGiB;
        heap = hs.mapped_bytes / kGiB;
#endif

        float soc_t = -1000, cpu_t = -1000, watts = 0, ghz = 0;
#ifdef __PROSPERO__
        int t = 0;
        if (sceKernelGetSocSensorTemperature(0, &t) == 0 && t > -100 && t < 200) {
            soc_t = (float) t;
        }
        if (sceKernelGetCpuTemperature(&t) == 0 && t > -100 && t < 200) {
            cpu_t = (float) t;
        }
        // Milliwatts in the low 32 bits; the high half counts slowly (a
        // sample number). Measured on the console: 22 W idle, 86 W while the
        // 27B generates.
        uint64_t raw[16] = {};
        if (sceKernelGetSocPowerConsumption(raw, 0.0) == 0) {
            watts = soc_watts(raw[0]);
        }
        const long hz = sceKernelGetCpuFrequency();
        ghz = hz > 0 ? (float) (hz / 1e9) : 0.0f;
        static bool logged = false;  // ponytail: once, to see what a title is allowed to read
        if (!logged) {
            logged = true;
            int t0 = -1000, t1 = -1000;
            const int r0 = sceKernelGetSocSensorTemperature(0, &t0), r1 = sceKernelGetCpuTemperature(&t1);
            ps5lm::log_line("sensors: soc rc 0x%x t %d, cpu rc 0x%x t %d, power raw %llu, cpu hz %ld\n", r0, t0,
                            r1, t1, (unsigned long long) raw[0], hz);
        }
#endif
        // A title may not read the sensors (0x80020002); a payload may.
        // Marice/ps5-exporter, when loaded, serves them on port 9100.
        float fan = -1;
        bool  from_exporter = false;
        std::vector<std::pair<std::string, float>> temps;
        std::string p;  // ps5-exporter's metrics
        if (soc_t < -100) {
            p = http_get(9100, "/metrics");
            // Not running: hand the bundled copy to elfldr, at start and then
            // once a minute (elfldr may come later).
            if (p.empty() && (long) (wall - start) % 60 == 1) {
                const bool sent = send_payload(sandbox_path("/app0/payloads/ps5-exporter.elf").c_str(), 9021);
                ps5lm::log_line("ps5-exporter not running; %s\n",
                                sent ? "sent the bundled copy to elfldr" : "no elfldr on 9021 to load it");
            }
            if (!p.empty()) {
                from_exporter = true;
                soc_t = (float) sample(p, "ps5_temperature_celsius{sensor=\"soc0\"}", -1000);
                cpu_t = (float) sample(p, "ps5_temperature_celsius{sensor=\"cpu\"}", -1000);
                watts = soc_watts((uint64_t) sample(p, "ps5_soc_power_raw_value", 0));
                for (int k = -1; k < 16; ++k) {
                    const std::string name = k < 0 ? std::string("cpu") : "soc" + std::to_string(k);
                    const double t = sample(p, "ps5_temperature_celsius{sensor=\"" + name + "\"}", -1000);
                    if (t > -100) {
                        temps.emplace_back(name, (float) t);
                    }
                }
                fan   = (float) sample(p, "ps5_fan_duty_ratio", -1);
                if (ghz <= 0) {
                    ghz = (float) (sample(p, "ps5_cpu_frequency_hertz", 0) / 1e9);
                }
            }
        }
        std::vector<Drive> drives = drives_from_metrics(p);

        std::lock_guard<std::mutex> lock(mutex_);
        const double dt = wall - last_wall;
        live_.soc_temp = soc_t;
        live_.cpu_temp = cpu_t;
        live_.soc_power_w = watts;
        live_.cpu_ghz = ghz;
        live_.fan = fan;
        live_.temps = temps;
        if (watts > 0) {
            push(live_.power_history, watts);
            live_.energy_wh += watts * dt / 3600.0;
        }
        live_.sensors_from_exporter = from_exporter;
        if (soc_t > -100) {
            push(live_.temp_history, soc_t);
        }
        live_.drives = std::move(drives);
        live_.uptime_s = wall - start;
        live_.pool_gib = pool;
        live_.free_gib = free;
        live_.heap_gib = heap;
        live_.cpu_use  = (float) std::min(1.0, (cpu - last_cpu) / (dt * live_.cpus));
        if (live_.state == ServerState::loading && load_base_ >= 0 && live_.model_gib + live_.kv_gib > 0) {
            const double added = (pool - free) - load_base_;
            live_.load_progress = (float) std::clamp(added / (live_.model_gib + live_.kv_gib), 0.0, 0.99);
        }
        push(live_.cpu_history, live_.cpu_use);
        if (up) {
            if (live_.state == ServerState::loading || live_.state == ServerState::ready ||
                live_.state == ServerState::generating) {
                live_.state = working ? ServerState::generating : ServerState::ready;
                live_.load_progress = -1;
            }
            live_.tokens        = (uint64_t) tokens;
            live_.prompt_tokens = (uint64_t) metric(m, "prompt_tokens_total");
            if (last_working && !working && tokens > last_tokens) {
                live_.requests++;  // a reply finished
            }
            last_working = working;
            // The last prompt's speed, from the counters (the server's own
            // average reads 0 on this build).
            const double ptok = metric(m, "prompt_tokens_total"), psec = metric(m, "prompt_seconds_total");
            if (have_last && psec - last_psec > 0.01) {
                live_.prompt_tps = (float) ((ptok - last_ptok) / (psec - last_psec));
            }
            last_ptok = ptok;
            last_psec = psec;
            if (have_last) {
                live_.gpu_busy = (float) std::min(1.0, std::max(0.0, (busy_s - last_busy) / dt));
                const double dgen = gen_s - last_gen_s;
                live_.gen_tps = dgen > 0.05 ? (float) ((tokens - last_tokens) / dgen) : 0.0f;
            }
            live_.ctx_used = (uint32_t) (json_int(slots, "n_prompt_tokens") + json_int(slots, "n_decoded"));
            // The server's counters move only when a request ends, so during a
            // long reply the speed comes from the slot's decoded count.
            const double decoded = (double) json_int(slots, "n_decoded");
            if (working && have_last && decoded > last_decoded && dt > 0) {
                live_.gen_tps  = (float) ((decoded - last_decoded) / dt);
                live_.gpu_busy = 1.0f;  // ponytail: a decoding slot keeps the GPU busy; no finer reading mid-request
            }
            last_decoded = working ? decoded : 0;
            have_last   = true;
            last_busy   = busy_s;
            last_tokens = tokens;
            last_gen_s  = gen_s;
        }
        push(live_.gpu_history, live_.gpu_busy);
        push(live_.gen_history, live_.gen_tps);
        const float now[4] = { soc_t, watts, live_.gpu_busy, live_.cpu_use };
        const bool  read[4] = { soc_t > -100, watts > 0, true, true };
        std::vector<float> * hour[4] = { &live_.temp_hour, &live_.power_hour, &live_.gpu_hour, &live_.cpu_hour };
        ++seconds;
        for (int i = 0; i < 4; ++i) {
            if (read[i]) {
                hour_sum[i] += now[i];
                ++hour_n[i];
            }
            if (seconds % 30 == 0) {
                if (hour_n[i] > 0) {
                    push(*hour[i], (float) (hour_sum[i] / hour_n[i]));
                }
                hour_sum[i] = 0;
                hour_n[i] = 0;
            }
        }
        last_wall = wall;
        last_cpu  = cpu;
    }
}

}  // namespace ps5lm
