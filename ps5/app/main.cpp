// PS5LM native app: llama-server on the PS5's GPU through RADV (ggml-vulkan),
// with a dashboard on the TV drawn by ps5-homebrew-ui through ps5-opengl.
//
// A title gets the console's direct memory (12 GiB) where a payload gets none,
// and RADV turns it into Vulkan device memory, so the whole model sits on the
// GPU. llama-server runs on its own thread on port 8081 (chat page and OpenAI
// API); the main thread draws the dashboard, reads the controller and plays
// the interface sounds. Models come from /data/PS5LM/models and USB drives; the
// planner (model_plan.cpp) picks each one's context and cache for the memory
// free. /data/PS5LM/app-args.txt, one argument per line, overrides all of it.
// The app logs to /data/PS5LM/app.log and llama.log. When the HEN keeps /data
// closed, all of it lives in the app's own storage, /download0/PS5LM, instead.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <pthread.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <vulkan/vulkan_core.h>

#include <GL/glcorearb.h>

#include "dashboard.hpp"
#include "log.hpp"
#include "fetch.hpp"
#include "market.hpp"
#include "display.hpp"
#include "ggml-backend.h"
#include "model_plan.hpp"
#include "probes.hpp"
#include "report.hpp"
#include "settings.hpp"
#include "storage.hpp"
#include "stats.hpp"
#include "suggested.hpp"
#include "version.hpp"

#include <ps5platform/heap.h>
#include <ps5platform/kernel.h>

#include "audio/mixer.hpp"
#include "core/input.hpp"
#include "core/save_file.hpp"
#include "gfx/renderer.hpp"
#include "platform/ps5/audio_out.hpp"
#include "platform/ps5/pad.hpp"
#include "platform/ps5/system.hpp"
#include "ui/fonts.hpp"

int  llama_server(int argc, char ** argv);
void llama_server_terminate();

// ---- Vulkan without a loader ----------------------------------------------------
// RADV is linked in whole, and ggml-vulkan's dispatcher starts from
// vkGetInstanceProcAddr, which the ICD entry point answers for every command.
// The few commands ggml calls by name go to Mesa's common implementations,
// which dispatch through the object's own table.
extern "C" PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char * name);
extern "C" PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char * name) {
    return vk_icdGetInstanceProcAddr(instance, name);
}
extern "C" PFN_vkVoidFunction vk_common_GetDeviceProcAddr(VkDevice device, const char * name);
extern "C" void vk_common_GetPhysicalDeviceFeatures2(VkPhysicalDevice dev, VkPhysicalDeviceFeatures2 * features);
extern "C" void vk_common_CmdCopyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, uint32_t n,
                                        const VkBufferCopy * regions);
extern "C" PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice device, const char * name) {
    return vk_common_GetDeviceProcAddr(device, name);
}
extern "C" void vkGetPhysicalDeviceFeatures2(VkPhysicalDevice dev, VkPhysicalDeviceFeatures2 * features) {
    vk_common_GetPhysicalDeviceFeatures2(dev, features);
}
extern "C" void vkCmdCopyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, uint32_t n,
                                const VkBufferCopy * regions) {
    vk_common_CmdCopyBuffer(cmd, src, dst, n, regions);
}

extern "C" {
int sceSystemServiceHideSplashScreen(void);
int sceKernelSendNotificationRequest(uint32_t device, void * request, size_t size, int blocking);
int sceKernelUsleep(uint32_t microseconds);
int sceSystemServiceLoadExec(const char * path, const char * const * args);
}

namespace {

using namespace ps5lm;

// The app's files, in /data/PS5LM with full storage, else in the app's own
// storage (storage.hpp). Set once at start, before any thread starts.
struct Files {
    std::string log, args, choice, env, text_only, quit, settings, scratch, llama_log, models;
    explicit Files(const std::string & root)
        : log(root + "/app.log"), args(root + "/app-args.txt"), choice(root + "/model.txt"),
          env(root + "/env.txt"),              // driver and ggml switches, for measuring
          text_only(root + "/text-only"),      // the last load's projector failed: load without it once
          quit(root + "/quit"), settings(root + "/settings.json"), scratch(root + "/scratch"),
          llama_log(root + "/llama.log"), models(root + "/models") {}
};
Files g_files(kFullRoot);
// The file tools only: a title cannot start processes (no shell command).
constexpr const char * kTools       = "read_file,write_file,edit_file,file_glob_search,grep_search,get_info";
constexpr const char * kAssets      = "/app0/assets";
constexpr int          kPort        = 8081;
// Both homes' models are listed: a model saved to the app's storage still
// loads once full storage opens.
std::vector<std::string> model_dirs() {
    return { "/data/PS5LM/models", sandbox_path("/download0/PS5LM/models"), "/mnt/ext0/PS5LM/models",
             "/mnt/usb0/PS5LM/models", "/mnt/usb1/PS5LM/models", "/mnt/usb0", "/mnt/usb1" };
}

// The last lines of a log, up to about 64 KiB of it, oldest first.
std::vector<std::string> tail_lines(const std::string & path) {
    std::vector<std::string> lines;
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return lines;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    const long from = std::max(0L, size - 65536L);
    std::fseek(f, from, SEEK_SET);
    std::string text((size_t) (size - from), '\0');
    text.resize(std::fread(text.data(), 1, text.size(), f));
    std::fclose(f);
    size_t start = from > 0 ? text.find('\n') + 1 : 0;  // the first line is cut
    for (size_t end; start < text.size() && (end = text.find('\n', start)) != std::string::npos; start = end + 1) {
        lines.emplace_back(text, start, end - start);
    }
    return lines;
}

// A system notification on the TV.
void notify(const char * text) {
    struct {
        uint8_t reserved[45];
        char    message[3075];
    } request = {};
    std::snprintf(request.message, sizeof(request.message), "%s", text);
    sceKernelSendNotificationRequest(0, &request, sizeof(request), 0);
}

// A title sees only its own /app0 and /download0; /data is the HEN's to open.
// etaHEN opens it before the title starts; OnionHEN opens it on request,
// {"PID":n} in /download0/etahen_jailbreak, for a title on its app jailbreak
// allowlist. Every process asks, since the restart for a model (LoadExec)
// starts a new one, sandboxed again (#16, #23; from @abhi245y's PS5LM Plus).
// Open means the models folder lists and a file can be written there.
bool data_open() {
    mkdir("/data/PS5LM", 0777);
    mkdir("/data/PS5LM/models", 0777);
    DIR * d = opendir("/data/PS5LM/models");
    if (!d) {
        return false;
    }
    closedir(d);
    const int fd = open("/data/PS5LM/.probe", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        return false;
    }
    const bool ok = write(fd, "1", 1) == 1;
    close(fd);
    unlink("/data/PS5LM/.probe");
    return ok;
}

// How /data ended up, for the log. A refusal is remembered in /download0:
// the next starts (one per model change) still ask, but wait 2 s, not 10.
std::string open_data(const std::string & title) {
    constexpr const char * kRefused = "/download0/PS5LM/data-closed";
    if (data_open()) {
        find_sandbox(title);
        unlink(sandbox_path(kRefused).c_str());
        return "open";
    }
    const int err = errno;
    const bool again = access(kRefused, F_OK) == 0;
    std::ofstream("/download0/etahen_jailbreak.tmp") << "{\"PID\":" << getpid() << "}\n";
    rename("/download0/etahen_jailbreak.tmp", "/download0/etahen_jailbreak");
    for (int i = 1; i <= (again ? 8 : 40); i++) {  // OnionHEN answered in about 3 s
        sceKernelUsleep(250000);
        if (data_open()) {
            find_sandbox(title);  // OnionHEN's answer also leaves the sandbox
            unlink(sandbox_path(kRefused).c_str());
            return "opened by the HEN after " + std::to_string(i / 4.0).substr(0, 4) + " s";
        }
    }
    mkdir("/download0/PS5LM", 0777);
    std::ofstream(kRefused) << "1\n";
    return std::string("closed (") + strerror(err) + "), the HEN did not open it";
}

// A quit request from the PC (scripts/ps5lm-app.sh close): the app ends
// itself the way the system expects; killing a title that is rendering has
// been followed by lost consoles (ps5-homebrew-ui's AGENTS.md).
void quit_if_asked() {
    if (access(g_files.quit.c_str(), F_OK) != 0) {
        return;
    }
    unlink(g_files.quit.c_str());
    ps5lm::log_line("quit requested\n");
    std::fflush(nullptr);
    sceSystemServiceLoadExec("exit", nullptr);
}

// The screen as a 24-bit BMP (glReadPixels of the TV's framebuffer), for
// checking the dashboard from the PC.
bool save_screen(const char * path, int width, int height) {
    std::vector<unsigned char> pixels((size_t) width * height * 4);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    const uint32_t row = ((uint32_t) width * 3 + 3) & ~3u;
    const uint32_t size = 54 + row * (uint32_t) height;
    unsigned char header[54] = { 'B', 'M' };
    auto put = [&](int at, uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            header[at + i] = (unsigned char) (v >> (8 * i));
        }
    };
    put(2, size);
    put(10, 54);
    put(14, 40);
    put(18, (uint32_t) width);
    put(22, (uint32_t) height);
    header[26] = 1;
    header[28] = 24;
    put(34, size - 54);
    FILE * f = std::fopen(path, "wb");
    if (!f) {
        return false;
    }
    bool ok = std::fwrite(header, 1, sizeof(header), f) == sizeof(header);
    std::vector<unsigned char> line(row);
    for (int y = 0; ok && y < height; ++y) {
        const unsigned char * in = pixels.data() + (size_t) y * width * 4;
        for (int x = 0; x < width; ++x) {
            line[(size_t) x * 3 + 0] = in[x * 4 + 2];
            line[(size_t) x * 3 + 1] = in[x * 4 + 1];
            line[(size_t) x * 3 + 2] = in[x * 4 + 0];
        }
        ok = std::fwrite(line.data(), 1, row, f) == row;
    }
    std::fclose(f);
    return ok;
}

// A request file from the PC: its first line, and the file is consumed.
std::string take_request_file(const std::string & path) {
    if (access(path.c_str(), F_OK) != 0) {
        return {};
    }
    std::string line;
    std::ifstream(path) >> line;
    unlink(path.c_str());
    return line.empty() ? std::string("-") : line;
}

// Where the memory is: the title's direct-memory pool and its free part,
// the flexible pool libc and the system libraries use, the CPU heap, and
// what ggml says about each device (its GPU figure wraps once the unified
// heap is over-committed; the raw numbers still show a change). `devices`
// false before the GPU opens: asking ggml would open it.
void log_memory(const char * when, bool devices = true) {
    int64_t at = 0;
    size_t  avail = 0, flex = 0, flex_free = 0;
    const int64_t pool = sceKernelGetDirectMemorySize();
    sceKernelAvailableDirectMemorySize(0, pool, 2 * 1024 * 1024, &at, &avail);
    sceKernelConfiguredFlexibleMemorySize(&flex);
    sceKernelAvailableFlexibleMemorySize(&flex_free);
    struct ps5_heap_stats heap = {};
    ps5_heap_stats(&heap);
    ps5lm::log_line("memory %s: pool %.4f GiB, pool free %.2f GiB, flexible %zu MiB (%zu free), cpu heap %.2f GiB "
                    "(peak %.2f)", when, pool / 1073741824.0, avail / 1073741824.0, flex >> 20, flex_free >> 20,
                    heap.mapped_bytes / 1073741824.0, heap.peak_bytes / 1073741824.0);
    for (size_t i = 0; devices && i < ggml_backend_dev_count(); i++) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_dev_get(i), &free, &total);
        std::printf(", %s free %zu total %zu", ggml_backend_dev_name(ggml_backend_dev_get(i)), free, total);
    }
    std::printf("\n");
}

std::vector<std::string> read_lines(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
            line.pop_back();
        }
        if (!line.empty() && line[0] != '#') {
            out.push_back(line);
        }
    }
    return out;
}

// Free GiB on the drive whose models folder is `dir`, or -1 when unknown.
double free_gib(const std::string & dir) {
    for (const auto & d : mounted_drives()) {
        if (d.models == dir && d.total > 0) {
            return d.free;
        }
    }
    return -1;
}

std::string base_name(const std::string & path) {
    std::string n = path.substr(path.find_last_of('/') + 1);
    if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0) {
        n.resize(n.size() - 5);
    }
    return n;
}

// ---- the debug report ---------------------------------------------------------------

std::string read_text(const std::string & path, size_t max_bytes = 4u << 20) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    const std::streamoff from = size > (std::streamoff) max_bytes ? size - (std::streamoff) max_bytes : 0;
    in.seekg(from);
    std::string text((size_t) (size - from), '\0');
    in.read(text.data(), (std::streamsize) text.size());
    text.resize((size_t) in.gcount());
    return text;
}

// Whether something listens on a port of this console. The connection is
// reset at once (SO_LINGER 0), so an ELF loader sees an error, never a payload.
bool port_open(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in to = {};
#ifdef __FreeBSD__
    to.sin_len = sizeof(to);
#endif
    to.sin_family = AF_INET;
    to.sin_port = htons((uint16_t) port);
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool open = connect(fd, (sockaddr *) &to, sizeof(to)) == 0;
    const linger reset = { 1, 0 };
    setsockopt(fd, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset));
    close(fd);
    return open;
}

// The kernel log, from klogsrv (3232) or etaHEN's (9081): what it sends in a
// few seconds, the last `max_bytes` of it.
std::string kernel_log(size_t max_bytes) {
    for (int port : { 3232, 9081 }) {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            continue;
        }
        timeval tv = { 1, 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        sockaddr_in to = {};
#ifdef __FreeBSD__
        to.sin_len = sizeof(to);
#endif
        to.sin_family = AF_INET;
        to.sin_port = htons((uint16_t) port);
        to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        std::string text;
        if (connect(fd, (sockaddr *) &to, sizeof(to)) == 0) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(4);
            char buf[65536];
            while (std::chrono::steady_clock::now() < until) {
                const ssize_t got = recv(fd, buf, sizeof(buf), 0);
                if (got <= 0) {
                    break;  // quiet for a second, or closed
                }
                text.append(buf, (size_t) got);
                if (text.size() > 2 * max_bytes) {
                    text = last_bytes(text, max_bytes);
                }
            }
            close(fd);
            return text.empty() ? "klog on port " + std::to_string(port) + " sent nothing in 4 s\n"
                                : last_bytes(text, max_bytes);
        }
        close(fd);
    }
    return "No kernel log server answers (klogsrv on 3232, etaHEN on 9081).\n";
}

const char * state_name(ServerState s) {
    switch (s) {
        case ServerState::loading: return "loading";
        case ServerState::ready: return "ready";
        case ServerState::generating: return "generating";
        case ServerState::failed: return "stopped with an error";
        default: return "no model";
    }
}

// Where the files are, in a line for the log and the debug report.
std::string storage_line(const std::string & data) {
    if (!app_storage()) {
        return data_root() + ", full storage (/data " + data + ")";
    }
    char t[160];
    std::snprintf(t, sizeof(t), "%s, the app's own storage: %.1f of %.0f GiB used (/data ", data_root().c_str(),
                  app_storage_used() / 1073741824.0, app_storage_size() / 1073741824.0);
    return t + data + ")";
}

// What the report is made from, copied on the main thread.
struct ReportJob {
    Live                  live;
    std::vector<Download> downloads;
    Settings              settings;
    std::string           storage;   // the data folder and how /data ended up
    std::string           dir;       // where the report goes
};

std::mutex  g_report_mutex;
std::string g_report_status;         // for the Settings row
std::atomic<bool> g_report_busy{ false };

void write_report(ReportJob job) {
    const auto gib = [](double v) { char t[32]; std::snprintf(t, sizeof(t), "%.2f GiB", v); return std::string(t); };
    ReportInput in;
    const std::time_t now = std::time(nullptr);
    std::tm tm = {};
    char when[32] = "unknown";
    if (localtime_r(&now, &tm)) {
        std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M", &tm);
    }
    in.created = when;
    in.version = PS5LM_VERSION;
    struct ps5_kernel_sw_version sw = {};
    sw.size = sizeof(sw);
    if (sceKernelGetSystemSwVersion(&sw) == 0) {
        sw.text[sizeof(sw.text) - 1] = '\0';
        char t[64];
        std::snprintf(t, sizeof(t), "%s (0x%08x)", sw.text, sw.version);
        in.firmware = t;
        in.firmware.erase(0, in.firmware.find_first_not_of(' '));
    }
    in.storage = job.storage;
    for (const auto & d : job.live.drives) {
        in.drives.push_back(d.name + ": " + (d.total > 0 ? gib(d.free) + " free of " + gib(d.total) : "space unknown (ps5-exporter is not running)") +
                            ", models in " + d.models);
    }
    size_t flex = 0, flex_free = 0;
    sceKernelConfiguredFlexibleMemorySize(&flex);
    sceKernelAvailableFlexibleMemorySize(&flex_free);
    in.memory = { "Direct memory pool " + gib(job.live.pool_gib) + ", " + gib(job.live.free_gib) + " free, CPU heap " + gib(job.live.heap_gib),
                  "Flexible memory " + std::to_string(flex >> 20) + " MiB, " + std::to_string(flex_free >> 20) + " MiB free" };
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_dev_get(i), &free, &total);
        in.memory.push_back(std::string(ggml_backend_dev_name(ggml_backend_dev_get(i))) + " (" +
                            ggml_backend_dev_description(ggml_backend_dev_get(i)) + "): " + gib(free / 1073741824.0) + " free of " +
                            gib(total / 1073741824.0));
    }
    const Live & L = job.live;
    in.server = { std::string("State: ") + state_name(L.state), "Address: " + L.address };
    if (!L.model_label.empty()) {
        in.server.push_back("Model: " + L.model_file + (L.model_arch.empty() ? "" : " (" + L.model_arch + ")"));
        in.server.push_back("Plan: " + L.plan);
        in.server.push_back("Requests answered: " + std::to_string(L.served) + ", tokens generated: " + std::to_string(L.tokens));
    }
    in.args = L.args;
    in.ports = report_ports();
    for (const auto & [port, what] : in.ports) {
        if (port_open(port)) {
            in.open_ports.push_back(port);
        }
    }
    for (const auto & m : L.models) {
        in.models.push_back(m.file + " (" + m.drive + ", " + gib(m.size_gib) + (m.arch.empty() ? "" : ", " + m.arch) + "): " +
                            m.plan + (m.current ? " [loaded]" : ""));
    }
    for (const auto & d : job.downloads) {
        static const char * states[] = { "queued", "running", "retrying", "paused", "failed", "done" };
        in.downloads.push_back(d.repo + "/" + d.file + " to " + d.dir + ": " + states[(int) d.state] + ", " +
                               std::to_string(d.done >> 20) + " of " + std::to_string(d.bytes >> 20) + " MiB" +
                               (d.note.empty() ? "" : ", " + d.note));
    }
    in.settings_json = read_text(g_files.settings);
    for (const std::string & f : { g_files.env, g_files.args }) {
        const std::string text = read_text(f, 64u << 10);
        if (!text.empty()) {
            in.files.emplace_back(f.substr(f.find_last_of('/') + 1), text);
        }
    }
    in.logs = { { "app.log, the last 256 KiB", last_bytes(read_text(g_files.log), 256u << 10) },
                { "llama.log, the last 256 KiB", last_bytes(read_text(g_files.llama_log), 256u << 10) } };
    in.kernel_log_asked = job.settings.report_kernel_log;
    if (in.kernel_log_asked) {
        in.kernel_log = kernel_log(1u << 20);
    }

    const std::string text = build_report(in);
    const std::string path = job.dir + "/debug-report.txt";
    bool ok = false;
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        ok = (bool) (out << text);
    }
    // A copy on a USB drive, for a console without FTP.
    std::string usb;
    for (const char * drive : { "/mnt/usb0", "/mnt/usb1" }) {
        std::ofstream out(std::string(drive) + "/PS5LM-debug-report.txt", std::ios::binary | std::ios::trunc);
        if (out && (out << text)) {
            usb = drive;
            break;
        }
    }
    char status[256];
    if (ok) {
        std::snprintf(status, sizeof(status), "Saved to %s%s, %zu KB", path.c_str(),
                      usb.empty() ? "" : (" and " + usb + "/PS5LM-debug-report.txt").c_str(), (text.size() + 1023) / 1024);
    } else {
        std::snprintf(status, sizeof(status), "Failed: cannot write %s%s", path.c_str(),
                      usb.empty() ? "" : ("; saved to " + usb + "/PS5LM-debug-report.txt").c_str());
    }
    ps5lm::log_line("debug report: %s\n", status);
    notify((std::string("PS5LM: debug report ") + (ok ? "saved to " + path : status)).c_str());
    std::lock_guard<std::mutex> lock(g_report_mutex);
    g_report_status = status;
    g_report_busy = false;
}

// ---- the server thread ------------------------------------------------------------

std::atomic<int> g_server_rc{ -1 };   // llama-server's exit code once it returns; -1 while it runs
std::vector<std::string> g_server_args;

void * run_server(void *) {
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>("llama-server"));
    for (auto & a : g_server_args) {
        argv.push_back(a.data());
    }
    argv.push_back(nullptr);
    int rc = 1;
    try {
        rc = llama_server((int) argv.size() - 1, argv.data());
    } catch (const std::exception & e) {
        ps5lm::log_line("exception: %s\n", e.what());
    } catch (...) {
        ps5lm::log_line("unknown exception\n");
    }
    ps5lm::log_line("llama_server returned %d\n", rc);
    std::fflush(nullptr);
    g_server_rc = rc == 0 ? 0 : 1;
    return nullptr;
}

// ---- models -------------------------------------------------------------------------

struct Models {
    double                 budget_gib = 0;   // GPU memory free before any model
    std::vector<ModelRow>  rows;
    std::string            running;          // path of the model llama-server has
    pthread_t              server{};
    bool                   server_started = false;
    Settings               settings;

    const char * kv_force() const { return settings.kv_type == "auto" ? nullptr : settings.kv_type.c_str(); }

    void scan(StatsCollector & stats) {
        rows = plan_rows(settings, budget_gib, running);
        publish(stats);
    }

    // scan() on its own thread, for a settings change: the dashboard shows it
    // working instead of freezing while every header is read.
    std::mutex             scan_mutex;
    std::vector<ModelRow>  scanned;
    std::atomic<int>       scan_state{ 0 };  // 0 idle, 1 running, 2 done
    bool                   scan_again = false;
    bool                   scan_shown = true;  // the dashboard says it is working

    void rescan(StatsCollector & stats, bool shown = true) {
        scan_shown = shown;
        if (scan_state != 0) {
            scan_again = true;  // the running one read the old settings
            return;
        }
        scan_state = 1;
        std::thread([this, s = settings, b = budget_gib, r = running] {
            std::vector<ModelRow> out = plan_rows(s, b, r);
            std::lock_guard<std::mutex> lock(scan_mutex);
            scanned = std::move(out);
            scan_state = 2;
        }).detach();
        (void) stats;
    }

    // Takes a finished rescan; true while one is still running.
    bool poll_rescan(StatsCollector & stats) {
        if (scan_state == 2) {
            {
                std::lock_guard<std::mutex> lock(scan_mutex);
                rows = std::move(scanned);
            }
            scan_state = 0;
            publish(stats);
            if (scan_again) {
                scan_again = false;
                rescan(stats);
            }
        }
        return scan_state != 0;
    }

    static std::vector<ModelRow> plan_rows(const Settings & settings, double budget_gib, const std::string & running) {
        std::vector<ModelRow> rows;
        const char * kv = settings.kv_type == "auto" ? nullptr : settings.kv_type.c_str();
        for (const auto & path : find_models(model_dirs())) {
            const ModelInfo m = read_model_info(path);
            ModelRow row;
            row.file = path;
            row.label = base_name(path);
            row.drive = drive_name(path);
            if (!m.ok) {
                row.plan = m.error;
                rows.push_back(row);
                continue;
            }
            const Plan p = plan_model(m, budget_gib, settings.ctx_cap, kv);
            const Preset * pre = find_preset(m);
            row.arch = m.arch;
            row.plan = p.why;
            row.fits = p.fits;
            row.size_gib = m.file_bytes / 1073741824.0;
            row.preset = pre ? pre->label : "";
            row.decision = m.decision;
            row.current = path == running;
            rows.push_back(row);
            ps5lm::log_line("model %s (%s): %s\n", m.file_name.c_str(), m.arch.c_str(), p.why.c_str());
        }
        return rows;
    }

    void publish(StatsCollector & stats) {
        std::vector<ModelRow> out = rows;
        for (auto & r : out) {
            r.current = r.file == running;
        }
        stats.set_models(out);
    }

    // The arguments for a model: the planner's, or app-args.txt's verbatim.
    std::vector<std::string> args_for(const std::string & path, Live & model_part) {
        const ModelInfo m = read_model_info(path);
        const Plan p = plan_model(m, budget_gib, settings.ctx_cap, kv_force());
        const Preset * pre = find_preset(m);
        model_part.model_label = base_name(path);
        model_part.model_file  = path;
        model_part.plan        = p.why;
        model_part.args        = p.args;
        model_part.model_arch  = m.arch;
        model_part.preset      = pre ? pre->label : "";
        model_part.decision    = m.decision;
        model_part.model_gib   = m.file_bytes / 1073741824.0;
        model_part.ctx         = p.ctx;
        model_part.kv_type     = p.kv_type;
        model_part.kv_gib      = p.kv_gib;
        return p.fits ? p.args : std::vector<std::string>{};
    }

    void start(const std::string & path, std::vector<std::string> args, StatsCollector & stats, const Live & model_part) {
        // llama.cpp's own log, written as it goes (it survives a crash), and
        // the metrics the dashboard reads.
        args.insert(args.begin(), { "--log-file", g_files.llama_log });
        if (std::find(args.begin(), args.end(), "--metrics") == args.end()) {
            args.push_back("--metrics");
        }
        // The browser chat's file tools, confined to the scratch folder by
        // patches/0003 (LLAMA_TOOLS_ROOT) and working there by default.
        if (settings.tools && std::find(args.begin(), args.end(), "--tools") == args.end()) {
            std::error_code ec;
            std::filesystem::create_directories(g_files.scratch, ec);
            setenv("LLAMA_TOOLS_ROOT", g_files.scratch.c_str(), 1);
            chdir(g_files.scratch.c_str());
            args.insert(args.end(), { "--tools", kTools });
        }
        for (const auto & a : args) {
            ps5lm::log_line("arg %s\n", a.c_str());
        }
        g_server_args = args;
        g_server_rc = -1;
        running = path;
        stats.set_model(model_part);
        stats.set_state(ServerState::loading);
        publish(stats);
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 16u << 20);  // the console's default stack is small
        pthread_create(&server, &attr, run_server, nullptr);
        server_started = true;
        notify("PS5LM: loading the model onto the GPU");
    }

    void stop() {
        if (!server_started) {
            return;
        }
        llama_server_terminate();
        // With a reply in flight llama-server can stay inside its HTTP thread
        // pool forever (unload during generation hung the app). We are about
        // to be replaced by LoadExec anyway, so wait 10 s at most.
        for (int i = 0; i < 100 && g_server_rc < 0; ++i) {
            usleep(100 * 1000);
        }
        if (g_server_rc >= 0) {
            pthread_join(server, nullptr);
        } else {
            ps5lm::log_line("llama_server did not return in 10 s; restarting anyway\n");
        }
        server_started = false;
        running.clear();
        log_memory("after unloading");
    }
};

// A thread with a 16 MiB stack, as llama-server's: the console's default is small.
pthread_t start_thread(std::function<void()> & fn) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16u << 20);
    pthread_t t{};
    pthread_create(&t, &attr, [](void * p) -> void * {
        (*static_cast<std::function<void()> *>(p))();
        return nullptr;
    }, &fn);
    return t;
}

bool load_font(hui::gfx::Renderer & renderer, const char * name, hui::gfx::Font * font, hui::ui::FontRef * ref) {
    std::string data;
    const std::string path = sandbox_path(kAssets) + "/fonts/" + name;
    if (!hui::save::read_file(path, &data) || !font->load(data)) {
        ps5lm::log_line("font %s failed\n", name);
        return false;
    }
    ref->font = font;
    ref->texture = renderer.batch().create_font_texture(*font);
    return true;
}

}  // namespace

int main(int, char **) {
    // /data first: the log, the settings and the models live there when the
    // HEN opens it, else in the app's own storage, /download0/PS5LM.
    const std::string title = title_id();
    const std::string data = open_data(title);
    const bool full = data.rfind("closed", 0) != 0;
    set_data_root(full ? kFullRoot : kAppRoot);
    g_files = Files(data_root());
    mkdir(data_root().c_str(), 0777);
    // stdout and stderr to a file the PC can fetch over FTP.
    if (std::freopen(g_files.log.c_str(), "w", stdout) != nullptr) {
        setvbuf(stdout, nullptr, _IOLBF, 0);
    }
    if (std::freopen(g_files.log.c_str(), "a", stderr) != nullptr) {
        setvbuf(stderr, nullptr, _IONBF, 0);
    }
    ps5lm::log_line("version " PS5LM_VERSION "\n");
    ps5lm::log_line("/data %s\n", data.c_str());
    ps5lm::log_line("files in %s\n", storage_line(data).c_str());
    if (!sandbox_path("").empty()) {
        ps5lm::log_line("the HEN moved the app out of its sandbox; /app0 is %s\n", sandbox_path("/app0").c_str());
    } else if (access("/app0/eboot.bin", F_OK) != 0) {
        ps5lm::log_line("/app0 is not visible, and no /mnt/sandbox/%s_* holds the app\n", title.c_str());
    }
    // The first start without full storage asks on the screen what to do;
    // the notification is for a screen that does not come up.
    if (!full && storage_choice().empty()) {
        notify("PS5LM: full storage (/data) is closed to the app, so it keeps its files in its own storage. "
               "On OnionHEN, add PPSA99575 to app_jailbreak.exact_title_ids in /data/OnionHEN/config.ini.");
    }
    // One device-local heap over the whole pool (RADV splits an APU's 2/3 to
    // 1/3), and a HOME for llama.cpp, which throws without one.
    setenv("radv_enable_unified_heap_on_apu", "true", 1);
    setenv("HOME", data_root().c_str(), 0);
    setenv("LLAMA_CACHE", data_path("cache").c_str(), 0);
    // Driver and backend switches to measure with, one NAME=value a line in
    // env.txt, set before the GPU opens; only the driver's, ggml's and llama's.
    for (const auto & line : read_lines(g_files.env)) {
        const size_t eq = line.find('=');
        const std::string name = line.substr(0, eq == std::string::npos ? 0 : eq);
        bool known = false;
        for (const char * prefix : { "RADV_", "MESA_", "ACO_", "GGML_", "LLAMA_" }) {
            known = known || name.rfind(prefix, 0) == 0;
        }
        if (known) {
            setenv(name.c_str(), line.c_str() + eq + 1, 1);
        }
        ps5lm::log_line("env %s%s\n", line.c_str(), known ? "" : " (ignored)");
    }
    mkdir(g_files.models.c_str(), 0777);
    ps5lm::log_line("start\n");
    // What the system gave the title, before the screen and the GPU take any
    // of it: the pool size follows sce_sys/param.json.
    log_memory("at start", false);

    // The screen first: it takes from the same memory as the model.
    Display display;
    if (!display.open(1920, 1080)) {
        ps5lm::log_line("no display\n");
    }
    hui::gfx::Renderer renderer;
    hui::gfx::Font regular, semibold, headline, mono;
    hui::ui::Fonts fonts;
    const bool ui_ok = renderer.init() && load_font(renderer, "manrope-500.huifont", &regular, &fonts.regular) &&
                       load_font(renderer, "manrope-600.huifont", &semibold, &fonts.semibold) &&
                       load_font(renderer, "manrope-200.huifont", &headline, &fonts.display) &&
                       load_font(renderer, "dejavu-sans-mono.huifont", &mono, &fonts.mono);
    fonts.pixel = fonts.mono;
    fonts.hand = fonts.regular;
    ps5lm::log_line("dashboard %s\n", ui_ok ? "ready" : "failed");

    Dashboard dash(fonts);
    DashboardFrame frame;
    const auto render = [&] {
        frame.backdrop = {};
        frame.scene.clear();
        frame.overlay.clear();
        frame.glass = false;
        frame.glass_texture = renderer.glass_texture();
        dash.draw(frame);
        renderer.begin();
        renderer.backdrop(frame.backdrop);
        renderer.draw(frame.scene);
        if (frame.glass) {
            renderer.glass();
        }
        renderer.draw(frame.overlay);
        display.present(renderer);
    };
    // The splash at once, so a restart for a model is never a black screen.
    if (ui_ok) {
        dash.set_busy("Starting", "Opening the GPU and reading the models");
        render();
        display.swap();
    }
    sceSystemServiceHideSplashScreen();

    StatsCollector stats;
    stats.start(kPort);
    set_activity_sink(&stats);  // llama-server's request hooks (patches/0007) report here
    Models models;
    models.settings = load_settings(g_files.settings.c_str());
    // Without full storage, downloads go where the app can write.
    if (app_storage() && models.settings.download_dir.rfind(kFullRoot, 0) == 0) {
        models.settings.download_dir = g_files.models;
    }
    // The model about to load, by name, for Home while the GPU opens.
    {
        std::vector<std::string> next = read_lines(g_files.args);
        const auto m = std::find(next.begin(), next.end(), "-m");
        next = m != next.end() && m + 1 != next.end() ? std::vector<std::string>{ *(m + 1) } : read_lines(g_files.choice);
        if (next.empty() && models.settings.auto_load) {
            next = { models.settings.default_model };
        }
        if (!next.empty() && !next[0].empty() && next[0] != "none") {
            Live pending;
            pending.model_label = base_name(next[0]);
            stats.set_model(pending);
            stats.set_state(ServerState::loading);
        }
        stats.set_phase("Opening the GPU and reading the models");
    }

    // The slow part on a thread of its own (the GPU opening, every model's
    // header, the model's start) while the screen keeps moving: the splash
    // for a moment, then Home, where the load shows.
    std::function<void()> boot = [&] {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            ps5lm::log_line("device %zu %s (%s): %.2f / %.2f GiB free\n", i, ggml_backend_dev_name(dev),
                            ggml_backend_dev_description(dev), free / 1073741824.0, total / 1073741824.0);
            // The PS5 is an APU: ggml calls its GPU integrated.
            const auto type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                models.budget_gib = free / 1073741824.0;
            }
        }

        models.scan(stats);

        // Which model: app-args.txt verbatim, else model.txt, which the library
        // writes just before restarting the app for a switch and which is used
        // once, else the default model when Settings asks for it at launch. A
        // plain launch otherwise loads nothing and opens the library.
        Live model_part;
        std::vector<std::string> args = read_lines(g_files.args);
        std::string path;
        if (!args.empty()) {
            const auto m = std::find(args.begin(), args.end(), "-m");
            path = m != args.end() && m + 1 != args.end() ? *(m + 1) : "";
            model_part.model_label = base_name(path);
            const ModelInfo info = read_model_info(path);
            model_part.model_arch = info.arch;
            model_part.model_gib = info.file_bytes / 1073741824.0;
            model_part.model_file = path;
            model_part.args = args;
            ps5lm::log_line("app-args.txt overrides the planner\n");
        } else {
            auto chosen = read_lines(g_files.choice);
            unlink(g_files.choice.c_str());
            if (chosen.empty() && models.settings.auto_load && !models.settings.default_model.empty()) {
                chosen = { models.settings.default_model };  // "none" after an unload keeps it from coming back
            }
            for (const auto & r : models.rows) {
                if (r.fits && !chosen.empty() && r.file == chosen[0]) {
                    path = r.file;
                }
            }
            if (!path.empty()) {
                args = models.args_for(path, model_part);
            }
        }
        // A projector this llama.cpp cannot read stops the server (#15); the
        // restart after it loads the same model text only, this once.
        if (access(g_files.text_only.c_str(), F_OK) == 0) {
            unlink(g_files.text_only.c_str());
            const auto mm = std::find(args.begin(), args.end(), "--mmproj");
            if (mm != args.end()) {
                ps5lm::log_line("loading without %s\n", mm + 1 != args.end() ? (mm + 1)->c_str() : "the projector");
                args.erase(mm, std::min(mm + 2, args.end()));
                model_part.args = args;
            }
        }
        log_memory("before loading");
        if (!args.empty()) {
            models.start(path, args, stats, model_part);
        } else {
            stats.set_state(ServerState::no_model);
            if (models.rows.empty()) {
                notify(("PS5LM: no models; copy a .gguf to " + g_files.models + " or PS5LM/models on a USB drive").c_str());
            }
        }
        stats.set_phase("");
    };
    std::atomic<bool> booted{ false };
    std::function<void()> boot_then = [&] {
        boot();
        booted = true;
    };
    const pthread_t booting = start_thread(boot_then);
    {
        hui::ui::Feedback quiet;  // no sounds yet: the mixer starts after
        const int64_t start = hui::sys::monotonic_us();
        int64_t at = start;
        while (ui_ok && !booted) {
            const int64_t now = hui::sys::monotonic_us();
            if (now - start > 1200000) {
                dash.set_busy("");
            }
            dash.set_live(stats.snapshot());
            quiet.clear();
            dash.update(hui::InputFrame{}, std::min(0.05f, (float) (now - at) / 1e6f), quiet);
            at = now;
            render();
            display.swap();
        }
    }
    pthread_join(booting, nullptr);

    // Input and sound, as ps5-homebrew-ui's own app does.
    hui::ps5::Pad pad;
    pad.open();
    hui::InputTracker tracker;
    hui::audio::Mixer mixer;
    hui::ps5::AudioOut audio_out;
    audio_out.start(mixer);
    hui::audio::SoundBank sounds;
    sounds.load(sandbox_path(kAssets) + "/audio/sfx");

    dash.set_busy("");
    Market    market;
    market.start(models.budget_gib);
    int  saved_seen = 0;  // downloads finished, as last seen
    dash.set_settings(models.settings);
    dash.set_storage(!app_storage(), app_storage_size() / 1073741824.0, app_storage() && storage_choice().empty());
    bool      scratch_warned = false;
    // Downloads left queued or running carry on, on their own page.
    unlink(data_path("download.txt").c_str());  // the queue file replaced this one (v0.7.0)
    market.load_queue();
    for (const auto & d : market.snapshot().downloads) {
        // Running already, maybe: the queue's thread starts it at once.
        if (d.state == DownloadState::queued || d.state == DownloadState::running) {
            dash.show_downloads();
            break;
        }
    }
    hui::ui::Feedback feedback;
    hui::PadSample samples[64];
    int64_t last = hui::sys::monotonic_us();
    bool was_serving = false;
    bool rescanning = false;
    uint64_t restart_at = 0;  // the frame to stop the model and restart on, or 0
    for (uint64_t n = 0;; ++n) {
        const int64_t now = hui::sys::monotonic_us();
        const float dt = n == 0 ? 1.0f / 60.0f : std::min(0.05f, (float) (now - last) / 1e6f);
        last = now;

        const size_t count = pad.read(samples);
        const hui::InputFrame input = tracker.update(std::span<const hui::PadSample>(samples, count), (uint64_t) now);
        Live live = stats.snapshot();
        if (g_server_rc >= 0 && live.state != ServerState::failed && live.state != ServerState::no_model) {
            stats.set_state(ServerState::failed);
            const bool projector = std::find(g_server_args.begin(), g_server_args.end(), "--mmproj") != g_server_args.end();
            if (live.state == ServerState::loading && projector && restart_at == 0) {
                std::ofstream(g_files.choice) << models.running << "\n";
                std::ofstream(g_files.text_only) << models.running << "\n";
                notify("PS5LM: the image projector did not load; loading the model text only");
                dash.set_busy("Loading " + base_name(models.running) + " text only",
                              "Its image projector did not load, see " + g_files.llama_log);
                restart_at = n + 3;
            } else {
                notify(("PS5LM: the server stopped, see " + g_files.llama_log).c_str());
            }
        }
        const bool serving = live.state == ServerState::ready || live.state == ServerState::generating;
        if (serving && !was_serving) {
            notify(live.decision.empty() ? "PS5LM: ready, chat in a browser on this network" : "PS5LM: ready for decisions at /v1/systemone");
            log_memory("when ready");  // the weights, the cache and the compute buffers are all allocated by now
        }
        was_serving = serving;
        if (n % 30 == 0) {
            models.publish(stats);
        }
        dash.set_live(live);
        feedback.clear();
        // A button pressed from the PC (scripts/ps5lm-app.sh press r1), for
        // testing the screens without a controller in hand.
        hui::InputFrame in = input;
        if (n % 20 == 10) {
            const std::string key = take_request_file(data_path("press"));
            const struct {
                const char *   name;
                hui::Action    action;
                hui::Direction nav;
            } keys[] = {
                { "up", hui::Action::up, hui::Direction::up },
                { "down", hui::Action::down, hui::Direction::down },
                { "left", hui::Action::left, hui::Direction::left },
                { "right", hui::Action::right, hui::Direction::right },
                { "cross", hui::Action::confirm, hui::Direction::none },
                { "circle", hui::Action::back, hui::Direction::none },
                { "triangle", hui::Action::north, hui::Direction::none },
                { "square", hui::Action::west, hui::Direction::none },
                { "l1", hui::Action::page_prev, hui::Direction::none },
                { "r1", hui::Action::page_next, hui::Direction::none },
                { "l2", hui::Action::jump_prev, hui::Direction::none },
                { "r2", hui::Action::jump_next, hui::Direction::none },
            };
            for (const auto & k : keys) {
                if (key == k.name) {
                    in.connected = true;
                    in.pressed |= hui::action_bit(k.action);
                    in.nav = k.nav;
                }
            }
        }
        dash.update(in, dt, feedback);
        for (const auto & cue : feedback.cues) {
            if (!models.settings.sounds) {
                break;
            }
            sounds.play(mixer, cue.set == hui::audio::SoundSet::count ? hui::audio::SoundSet::glass : cue.set, cue);
        }
        if (feedback.rumble_strength > 0.0f) {
            pad.rumble(feedback.rumble_strength, feedback.rumble_seconds);
        }
        pad.tick(dt);

        // The scratch folder: its size for the dashboard, and one warning each
        // time it grows past the limit.
        if (n % 600 == 300) {
            const double gib = folder_bytes(g_files.scratch) / 1073741824.0;
            stats.set_scratch(gib);
            const bool over = gib > models.settings.scratch_limit_gib;
            if (over && !scratch_warned) {
                char text[128];
                std::snprintf(text, sizeof(text), "PS5LM: the chat's scratch folder holds %.1f GiB; clear it in Settings", gib);
                notify(text);
                ps5lm::log_line("%s\n", text + 7);
            }
            scratch_warned = over;
        }
        if (n % 10 == 0) {
            MarketView mv = market.snapshot();
            // A finished download joins the library.
            if (mv.saved != saved_seen) {
                saved_seen = mv.saved;
                models.rescan(stats, false);
                notify(("PS5LM: " + mv.note).c_str());
            }
            dash.set_market(std::move(mv));
        }
        if (dash.wants_logs() && n % 60 == 0) {
            dash.set_logs(tail_lines(g_files.log), tail_lines(g_files.llama_log));
        }

        // A settings rescan in the background: say so until it lands.
        if (models.poll_rescan(stats) && models.scan_shown) {
            char text[96];
            std::snprintf(text, sizeof(text), "Planning every model again for %uk tokens and the %s cache",
                          models.settings.ctx_cap / 1024, models.settings.kv_type.c_str());
            dash.set_busy("Applying the new limits", text);
            rescanning = true;
        } else if (rescanning) {
            rescanning = false;
            dash.set_busy("");
        }
        // A restart asked for a few frames ago, once the busy screen is up.
        if (restart_at != 0 && n >= restart_at) {
            std::fflush(nullptr);
            models.stop();
            sceSystemServiceLoadExec("/app0/eboot.bin", nullptr);
        }

        Dashboard::Request req = dash.take_request();
        if (req.kind == Dashboard::Request::settings) {
            // The planner's limits change every row's plan; the loaded model
            // keeps its settings until it is loaded again.
            // Only the context and cache limits change the plans; rescanning
            // reads every model's header, seconds with a USB drive.
            const Settings was = models.settings;
            models.settings = dash.settings();
            save_settings(g_files.settings.c_str(), models.settings);
            if (was.ctx_cap != models.settings.ctx_cap || was.kv_type != models.settings.kv_type) {
                models.rescan(stats);
            }
            ps5lm::log_line("settings saved\n");
        }
        switch (req.kind) {
            case Dashboard::Request::ask: ask_self(kPort, req.path, !live.decision.empty()); break;
            case Dashboard::Request::market_search: market.search(req.path); break;
            case Dashboard::Request::market_open: market.open_repo(req.repo); break;
            case Dashboard::Request::market_plan: market.plan_file(req.repo, req.file); break;
            case Dashboard::Request::market_download: {
                MarketJob job;
                if (market.job(req.repo, req.file, req.path.empty() ? models.settings.download_dir : req.path, &job) &&
                    !market.enqueue(job, free_gib(job.dir))) {
                    notify(("PS5LM: " + market.snapshot().note).c_str());
                }
                break;
            }
            case Dashboard::Request::suggested_download: {
                const Suggestion & s = kSuggestions[std::clamp(req.repo, 0, kSuggestionCount - 1)];
                const std::string dir = req.path.empty() ? models.settings.download_dir : req.path;
                std::vector<MarketJob> jobs;
                for (const SuggestedFile * f : { &s.model, &s.mmproj }) {
                    if (f->file) {
                        jobs.push_back({ s.repo, f->file, f->sha256, dir, f->bytes, f->save_as ? f->save_as : "" });
                    }
                }
                if (!market.enqueue(jobs, free_gib(dir))) {
                    notify(("PS5LM: " + market.snapshot().note).c_str());
                }
                dash.show_downloads();
                break;
            }
            case Dashboard::Request::download_pause: market.pause(req.repo); break;
            case Dashboard::Request::download_resume: market.resume(req.repo); break;
            case Dashboard::Request::download_remove: market.remove(req.repo); break;
            case Dashboard::Request::market_download_unload: {
                // Unload by restarting (as the library does); the queue,
                // saved to disk, starts the download in the new process.
                MarketJob job;
                if (market.job(req.repo, req.file, req.path.empty() ? models.settings.download_dir : req.path, &job) &&
                    market.enqueue(job, free_gib(job.dir))) {
                    std::ofstream(g_files.choice) << "none\n";
                    ps5lm::log_line("restarting to download %s/%s without a model\n", job.repo.c_str(), job.file.c_str());
                    dash.set_busy("Unloading " + base_name(models.running), "The app restarts without it, then the download starts");
                    restart_at = n + 3;
                }
                break;
            }
            default: break;
        }
        if (req.kind == Dashboard::Request::clear_scratch) {
            std::error_code ec;
            for (const auto & e : std::filesystem::directory_iterator(g_files.scratch, ec)) {
                std::filesystem::remove_all(e.path(), ec);
            }
            stats.set_scratch(folder_bytes(g_files.scratch) / 1073741824.0);
            notify("PS5LM: scratch folder cleared");
            ps5lm::log_line("scratch folder cleared\n");
        }
        // Full storage closed: the app's own storage is the answer from now on,
        // or a fresh start asks the HEN again.
        if (req.kind == Dashboard::Request::storage_app) {
            set_storage_choice("app");
            ps5lm::log_line("the app's own storage, chosen\n");
        } else if (req.kind == Dashboard::Request::restart) {
            if (!models.running.empty()) {
                std::ofstream(g_files.choice) << models.running << "\n";
            }
            ps5lm::log_line("restarting to ask for /data again\n");
            dash.set_busy("Starting again", "PS5LM asks the HEN for full storage once more");
            restart_at = n + 3;
        }
        // A debug report, on a thread of its own: the ports and the kernel log
        // take seconds. The first one was explained on the screen; remember it.
        if (req.kind == Dashboard::Request::debug_report && !g_report_busy.exchange(true)) {
            if (!models.settings.report_explained) {
                models.settings.report_explained = true;
                dash.set_settings(models.settings);
                save_settings(g_files.settings.c_str(), models.settings);
            }
            {
                std::lock_guard<std::mutex> lock(g_report_mutex);
                g_report_status = "Writing the report";
            }
            ReportJob job{ stats.snapshot(), market.snapshot().downloads, models.settings,
                           storage_line(data), data_root() };
            std::thread(write_report, std::move(job)).detach();
        }
        if (n % 10 == 5) {
            std::lock_guard<std::mutex> lock(g_report_mutex);
            dash.set_report_status(g_report_busy, g_report_status);
        }
        // The same requests from the PC (scripts/ps5lm-app.sh load|unload).
        if (n % 60 == 30 && req.kind == Dashboard::Request::none) {
            if (std::string p = take_request_file(data_path("load")); !p.empty()) {
                req = { Dashboard::Request::load, p };
            } else if (!take_request_file(data_path("unload")).empty()) {
                req = { Dashboard::Request::unload, "" };
            } else if (std::string u = take_request_file(data_path("fetch")); !u.empty()) {
                std::thread(probe_fetch, u).detach();
            } else if (!take_request_file(data_path("bench-write")).empty()) {
                std::thread(probe_write).detach();
            } else if (!take_request_file(data_path("memory")).empty()) {
                log_memory("asked");  // scripts/bench-ubatch.py: what a prompt left allocated
            } else if (std::string f = take_request_file(data_path("bench-read")); !f.empty()) {
                std::thread(probe_read, f).detach();
            } else if (!take_request_file(data_path("bench-memory")).empty()) {
                std::thread(probe_memory, !models.running.empty()).detach();
            } else if (std::string q = take_request_file(data_path("search")); !q.empty()) {
                market.search(q == "-" ? std::string() : q);  // a query typed on the PC
            }
        }
        if (req.kind == Dashboard::Request::unload || req.kind == Dashboard::Request::load) {
            // Unloading in place leaves part of the model's GPU memory held
            // (about 5 GiB of a 6.8 GiB model, measured), so a model change
            // restarts the app: the system replaces the process (LoadExec on
            // our own eboot.bin, "Kill for LoadExec" in the log) and the new
            // one starts from the whole pool, with model.txt saying what to load.
            std::ofstream(g_files.choice) << (req.kind == Dashboard::Request::load ? req.path : std::string("none")) << "\n";
            if (req.kind == Dashboard::Request::load) {
                unlink(g_files.args.c_str());  // the library's choice replaces a manual override
            }
            ps5lm::log_line("restarting for %s\n", req.kind == Dashboard::Request::load ? req.path.c_str() : "no model");
            const bool loaded = !models.running.empty();
            if (req.kind == Dashboard::Request::load) {
                dash.set_busy(loaded ? "Switching models" : "Loading " + base_name(req.path),
                              loaded ? "Unloading " + base_name(models.running) + ", then the app restarts with " + base_name(req.path)
                                     : "The app restarts to give the model the whole memory");
            } else {
                dash.set_busy("Unloading " + base_name(models.running), "The app restarts without a model, about 30 seconds");
            }
            restart_at = n + 3;  // three frames with the busy screen on the TV first
        }

        if (ui_ok) {
            render();
            if (n % 60 == 45 && !take_request_file(data_path("screenshot")).empty()) {
                const bool ok = save_screen(data_path("screen.bmp").c_str(), display.width(), display.height());
                ps5lm::log_line("screenshot %s\n", ok ? "saved" : "failed");
            }
        }
        display.swap();
        if (n % 60 == 0) {
            quit_if_asked();
        }
    }
}
