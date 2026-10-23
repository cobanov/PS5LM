// SPDX-License-Identifier: GPL-3.0-or-later
#include "market.hpp"
#include "log.hpp"

#include "fetch.hpp"
#include "model_plan.hpp"
#include "storage.hpp"

#include <nlohmann/json.hpp>

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <cerrno>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace ps5lm {

namespace {

constexpr const char * kHub = "https://huggingface.co";

std::string url_encode(const std::string & s) {
    std::string out;
    char hex[4];
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += (char) c;
        } else {
            std::snprintf(hex, sizeof(hex), "%%%02X", c);
            out += hex;
        }
    }
    return out;
}

// A whole response as a string, or empty with `error` set.
std::string get_text(const std::string & url, const std::string & range, std::string * error, size_t limit = 64u << 20) {
    std::string body;
    const FetchResult r = https_get(url, range, [&](const char * p, size_t n) {
        body.append(p, n);
        return body.size() < limit;
    });
    if (r.status < 200 || r.status >= 300) {
        char text[160];
        std::snprintf(text, sizeof(text), "HTTP %d%s%s", r.status, r.error.empty() ? "" : ": ", r.error.c_str());
        *error = text;
        return {};
    }
    return body;
}

}  // namespace

void Market::start(double budget_gib) {
    budget_gib_ = budget_gib;
}

MarketView Market::snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    MarketView v = view_;
    v.busy = browsing_;
    return v;
}

void Market::run(std::atomic<bool> & flag, std::function<void()> job) {
    if (flag.exchange(true)) {
        return;  // one at a time; the screen shows it is busy
    }
    std::thread([&flag, job = std::move(job)] {
        job();
        flag = false;
    }).detach();
}

void Market::search(const std::string & query) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        view_.query = query;
        view_.status = "Searching";
    }
    run(browsing_, [this, query] {
        std::string error;
        const std::string body = get_text(std::string(kHub) + "/api/models?filter=gguf&sort=downloads&direction=-1&limit=30" +
                                              (query.empty() ? "" : "&search=" + url_encode(query)),
                                          "", &error);
        const nlohmann::json j = nlohmann::json::parse(body, nullptr, false);
        std::vector<MarketRepo> repos;
        if (j.is_array()) {
            for (const auto & m : j) {
                MarketRepo r;
                r.id = m.value("id", std::string());
                r.downloads = m.value("downloads", (uint64_t) 0);
                if (!r.id.empty()) {
                    repos.push_back(r);
                }
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (view_.query != query) {
            return;  // a newer search replaced this one
        }
        view_.repos = repos;
        view_.open = -1;
        view_.status = !error.empty() ? error : repos.empty() ? "Nothing found" : "";
    });
}

void Market::open_repo(int index) {
    std::string id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < 0 || index >= (int) view_.repos.size()) {
            return;
        }
        id = view_.repos[(size_t) index].id;
        view_.open = index;
        view_.status = "Listing files";
    }
    run(browsing_, [this, index, id] {
        std::string error;
        const std::string body = get_text(std::string(kHub) + "/api/models/" + id + "/tree/main", "", &error);
        const nlohmann::json j = nlohmann::json::parse(body, nullptr, false);
        std::vector<MarketFile> files;
        if (j.is_array()) {
            for (const auto & e : j) {
                const std::string path = e.value("path", std::string());
                const bool gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
                // Projectors and later parts of split models are not models of their own.
                if (!gguf || path.find("mmproj") != std::string::npos ||
                    (path.find("-of-") != std::string::npos && path.find("-00001-of-") == std::string::npos)) {
                    continue;
                }
                MarketFile f;
                f.name = path;
                f.bytes = e.value("size", (uint64_t) 0);
                if (e.contains("lfs") && e["lfs"].is_object()) {
                    f.sha256 = e["lfs"].value("oid", std::string());
                    f.bytes = e["lfs"].value("size", f.bytes);
                }
                // Before the header is read: fits if the weights and the
                // planner's smallest reserve do.
                f.fits = f.bytes / 1073741824.0 + 1.0 <= budget_gib_ ? -1 : 0;
                if (f.fits == 0) {
                    f.plan = "too big: the weights alone need more than is free";
                }
                files.push_back(f);
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (view_.open == index && index < (int) view_.repos.size()) {
            view_.repos[(size_t) index].files = files;
        }
        view_.status = !error.empty() ? error : files.empty() ? "No GGUF files in this repository" : "";
    });
}

// Checks wait in a queue and run one at a time on their own thread, so one
// asked for while another (or a search) runs is not lost. They are kept by
// repository and file name: the lists can change while one waits.
void Market::plan_file(int repo, int file) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (repo < 0 || repo >= (int) view_.repos.size() || file < 0 ||
            file >= (int) view_.repos[(size_t) repo].files.size()) {
            return;
        }
        MarketFile & f = view_.repos[(size_t) repo].files[(size_t) file];
        if ((f.fits != -1 && f.fits != -2) || f.checking) {
            return;  // planned already, too big by its size alone, or waiting
        }
        f.checking = true;
        plans_.push_back({ view_.repos[(size_t) repo].id, f.name, f.bytes });
    }
    if (planning_.exchange(true)) {
        return;  // the running thread takes it
    }
    std::thread([this] {
        for (;;) {
            PlanJob job;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (plans_.empty()) {
                    planning_ = false;
                    return;
                }
                job = plans_.back();  // the newest first: what the cursor is on
                plans_.pop_back();
            }
            std::string error;
            const std::string head = get_text(std::string(kHub) + "/" + job.repo + "/resolve/main/" + job.file,
                                              "bytes=0-16777215", &error, 16u << 20);
            const ModelInfo m = read_model_header(job.file, head, job.bytes);
            const Plan      p = plan_model(m, budget_gib_);
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto & r : view_.repos) {
                for (auto & f : r.files) {
                    if (r.id == job.repo && f.name == job.file) {
                        f.fits = m.ok ? (p.fits ? 1 : 0) : -2;
                        f.checking = false;
                        f.plan = m.ok ? p.why : (!error.empty() ? error : m.error);
                    }
                }
            }
        }
    }).detach();
}

bool Market::job(int repo, int file, const std::string & dir, MarketJob * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (repo < 0 || repo >= (int) view_.repos.size() || file < 0 || file >= (int) view_.repos[(size_t) repo].files.size()) {
        return false;
    }
    const MarketFile & f = view_.repos[(size_t) repo].files[(size_t) file];
    *out = { view_.repos[(size_t) repo].id, f.name, f.sha256, dir, f.bytes };
    return true;
}

namespace {


// Pieces at once, each on its own template (made once, one after the
// other). Measured on the console (CHANGELOG 0.8.0): one connection gets 3
// to 6 MB/s and more do not add much, but two never hurt, and the library's
// four requests leave two for Get models while a download runs.
constexpr int kConnections = 2;
FetchResult kGet(int worker, const std::string & url, const std::string & range,
                 const std::function<bool(const char *, size_t)> & sink) {
    static const int tmpl[kConnections] = { https_template(), https_template() };
    return https_get(url, range, sink, tmpl[worker % kConnections] >= 0 ? tmpl[worker % kConnections] : -1, 4u << 20);
}

const char * state_name(DownloadState s) {
    switch (s) {
        case DownloadState::queued: return "queued";
        case DownloadState::running: return "running";
        case DownloadState::retrying: return "retrying";
        case DownloadState::paused: return "paused";
        case DownloadState::failed: return "failed";
        case DownloadState::done: return "done";
    }
    return "queued";
}

std::string part_path(const Download & d) {
    const size_t slash = d.file.find_last_of('/');
    return d.dir + "/" + (d.save_as.empty() ? d.file.substr(slash == std::string::npos ? 0 : slash + 1) : d.save_as) + ".part";
}

uint64_t file_size(const std::string & path) {
    struct stat st = {};
    return stat(path.c_str(), &st) == 0 ? (uint64_t) st.st_size : 0;
}

}  // namespace

void Market::save_queue() {
    nlohmann::json j = nlohmann::json::array();
    for (const auto & d : view_.downloads) {
        j.push_back({ { "repo", d.repo }, { "file", d.file }, { "sha256", d.sha256 }, { "dir", d.dir }, { "bytes", d.bytes },
                      { "state", state_name(d.state) }, { "note", d.note }, { "resume", d.resume }, { "save_as", d.save_as } });
    }
    std::ofstream(data_path("downloads.json")) << j.dump(1) << "\n";
}

void Market::load_queue() {
    std::ifstream in(data_path("downloads.json"));
    const nlohmann::json j = nlohmann::json::parse(in, nullptr, false);
    if (!j.is_array()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & e : j) {
        Download d;
        d.repo = e.value("repo", std::string());
        d.file = e.value("file", std::string());
        d.sha256 = e.value("sha256", std::string());
        d.dir = e.value("dir", std::string());
        d.bytes = e.value("bytes", (uint64_t) 0);
        d.note = e.value("note", std::string());
        d.save_as = e.value("save_as", std::string());
        const std::string st = e.value("state", std::string());
        // Running when the app stopped: carry on.
        d.state = st == "paused" ? DownloadState::paused : st == "failed" ? DownloadState::failed
                : st == "done" ? DownloadState::done : DownloadState::queued;
        // v0.7 queues had no "resume": their partial files were written in order.
        d.resume = e.value("resume", file_size(part_path(d)));
        d.done = d.state == DownloadState::done ? d.bytes : d.resume;
        if (!d.repo.empty() && !d.file.empty()) {
            view_.downloads.push_back(d);
        }
    }
    if (!working_.exchange(true)) {
        std::thread([this] { worker(); }).detach();
    }
}

bool Market::enqueue(const MarketJob & job, double free) { return enqueue(std::vector<MarketJob>{ job }, free); }

bool Market::enqueue(const std::vector<MarketJob> & jobs, double free) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (jobs.empty()) {
        return false;
    }
    for (const auto & job : jobs) {
        for (const auto & d : view_.downloads) {
            if (d.repo == job.repo && d.file == job.file && d.state != DownloadState::done) {
                view_.note = job.file + " is in Downloads already";
                return false;
            }
        }
    }
    // Room for the files, the downloads queued to the same drive before them,
    // and 1 GiB to spare, so a download never fills a drive (a full /data
    // upsets the system, not only this app) and never stops halfway for want
    // of room. A model and its image projector are asked for together.
    const std::string & dir = jobs[0].dir;
    double need = 0, queued = 0;
    for (const auto & job : jobs) {
        need += job.bytes / 1073741824.0;
    }
    for (const auto & d : view_.downloads) {
        if (d.dir == dir && d.state != DownloadState::done) {
            queued += (double) (d.bytes - std::min(d.done, d.bytes)) / 1073741824.0;
        }
    }
    if (free >= 0 && free - queued < need + 1.0) {
        char text[320];
        std::snprintf(text, sizeof(text), "Not enough space: %s%s needs %.1f GiB and 1 GiB spare, %s has %.1f GiB free%s",
                      jobs[0].file.c_str(), jobs.size() > 1 ? " and its image projector" : "", need, drive_name(dir).c_str(),
                      std::max(0.0, free - queued), queued > 0 ? " after the downloads queued there" : "");
        view_.note = text;
        ps5lm::log_line("download %s/%s: %s\n", jobs[0].repo.c_str(), jobs[0].file.c_str(), text);
        return false;
    }
    for (const auto & job : jobs) {
        Download d;
        static_cast<MarketJob &>(d) = job;
        d.resume = d.done = 0;  // ponytail: a stray partial file is fetched again, its progress is not known
        view_.downloads.push_back(d);
    }
    view_.note = jobs[0].file + " added to Downloads";
    save_queue();
    if (!working_.exchange(true)) {
        std::thread([this] { worker(); }).detach();
    }
    return true;
}

void Market::pause(int i) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (i < 0 || i >= (int) view_.downloads.size()) {
        return;
    }
    Download & d = view_.downloads[(size_t) i];
    if (d.state == DownloadState::running || d.state == DownloadState::retrying) {
        stop_ = 1;  // the worker marks it paused
    } else if (d.state == DownloadState::queued) {
        d.state = DownloadState::paused;
        save_queue();
    }
}

void Market::resume(int i) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (i < 0 || i >= (int) view_.downloads.size()) {
        return;
    }
    Download & d = view_.downloads[(size_t) i];
    if (d.state == DownloadState::paused || d.state == DownloadState::failed) {
        d.state = DownloadState::queued;
        d.attempt = 0;
        d.note.clear();
        save_queue();
        if (!working_.exchange(true)) {
            std::thread([this] { worker(); }).detach();
        }
    }
}

void Market::remove(int i) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (i < 0 || i >= (int) view_.downloads.size()) {
        return;
    }
    Download & d = view_.downloads[(size_t) i];
    if (d.state == DownloadState::running || d.state == DownloadState::retrying) {
        stop_ = 2;  // the worker deletes the partial file and the entry
        return;
    }
    if (d.state != DownloadState::done) {
        unlink(part_path(d).c_str());
    }
    view_.downloads.erase(view_.downloads.begin() + i);
    save_queue();
}

void Market::worker() {
    for (;;) {
        Download d;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = std::find_if(view_.downloads.begin(), view_.downloads.end(),
                                   [](const Download & x) { return x.state == DownloadState::queued; });
            if (it == view_.downloads.end()) {
                working_ = false;
                return;
            }
            it->state = DownloadState::running;
            d = *it;
            stop_ = 0;
        }
        fetch(d);
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = std::find_if(view_.downloads.begin(), view_.downloads.end(),
                               [&](const Download & x) { return x.repo == d.repo && x.file == d.file; });
        if (stop_ == 2) {
            unlink(part_path(d).c_str());
            if (it != view_.downloads.end()) {
                view_.downloads.erase(it);
            }
        } else if (it != view_.downloads.end()) {
            *it = d;
            if (d.state == DownloadState::done) {
                ++view_.saved;
                view_.note = d.note;
            }
        }
        stop_ = 0;
        save_queue();
    }
}

// Several connections at once, each fetching 32 MiB pieces of the file and
// writing them in place: one connection on the console tops out far below
// the line's speed. What is done from the start of the file (`resume`) is
// saved with the queue, so a dropped connection, a pause or a restart costs
// at most the pieces in flight. The SHA-256 is checked once it is whole.
bool Market::fetch(Download & d) {
    constexpr uint64_t kPiece = 32ull << 20;
    const std::string part = part_path(d);
    const std::string to = part.substr(0, part.size() - 5);
    const std::string url = std::string(kHub) + "/" + d.repo + "/resolve/main/" + d.file;
    std::error_code ec;
    std::filesystem::create_directories(d.dir, ec);
    const auto publish = [&](bool save) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto & x : view_.downloads) {
            if (x.repo == d.repo && x.file == d.file) {
                x.state = d.state;
                x.done = d.done;
                x.resume = d.resume;
                x.mbps = d.mbps;
                x.attempt = d.attempt;
                x.note = d.note;
            }
        }
        if (save) {
            save_queue();
        }
    };
    if (d.bytes == 0) {
        d.state = DownloadState::failed;
        d.note = "Failed: the repository does not give this file's size";
        publish(true);
        return false;
    }
    if (d.resume > d.bytes || d.resume > file_size(part)) {
        d.resume = 0;  // not this file, or the partial file is gone
    }
    const int fd = open(part.c_str(), O_WRONLY | O_CREAT, 0666);
    if (fd < 0) {
        d.state = DownloadState::failed;
        d.note = "Failed: cannot write to " + d.dir;
        publish(true);
        return false;
    }
    const uint64_t base = d.resume;
    const size_t   pieces = (size_t) ((d.bytes - base + kPiece - 1) / kPiece);
    std::vector<uint8_t> state(pieces, 0);  // 0 to fetch, 1 in flight, 2 written
    std::mutex   m;
    std::atomic<uint64_t> received{ base };
    std::atomic<int>      failures{ 0 };   // in a row, across connections
    std::atomic<int>      waiting{ 0 };    // connections waiting to try again
    std::atomic<bool>     fatal{ false };
    std::atomic<int>      write_errno{ 0 };  // a write refused for want of room ends the download
    std::string           last_error;
    const auto worker = [&](int id) {
        for (;;) {
            size_t k = pieces;
            {
                std::lock_guard<std::mutex> lock(m);
                for (size_t i = 0; i < pieces; ++i) {
                    if (state[i] == 0) {
                        k = i;
                        state[i] = 1;
                        break;
                    }
                }
            }
            if (k == pieces || stop_ != 0 || fatal) {
                return;
            }
            const uint64_t off = base + k * kPiece, len = std::min(kPiece, d.bytes - off);
            char range[64];
            std::snprintf(range, sizeof(range), "bytes=%llu-%llu", (unsigned long long) off, (unsigned long long) (off + len - 1));
            uint64_t got = 0;
            const FetchResult r = kGet(id, url, range, [&](const char * p, size_t n) {
                if (stop_ != 0 || got + n > len) {
                    return false;
                }
                if (pwrite(fd, p, n, (off_t) (off + got)) != (ssize_t) n) {
                    write_errno = errno;
                    return false;
                }
                got += n;
                received += n;
                return true;
            });
            std::lock_guard<std::mutex> lock(m);
            if (got == len && r.status == 206) {
                state[k] = 2;
                failures = 0;
                continue;
            }
            state[k] = 0;  // again, from its start
            received -= got;
            if (stop_ != 0) {
                return;
            }
            char text[200];
            std::snprintf(text, sizeof(text), "%s%s%s", r.status > 0 ? ("HTTP " + std::to_string(r.status)).c_str() : "",
                          r.status > 0 && !r.error.empty() ? ", " : "", r.error.c_str());
            last_error = text;
            const int f = ++failures;
            ps5lm::log_line("download %s: piece at %llu stopped after %llu bytes: %s\n", d.file.c_str(),
                            (unsigned long long) off, (unsigned long long) got, text);
            const int werr = write_errno;
            if (werr == ENOSPC || werr == EDQUOT || werr == EFBIG) {
                last_error = drive_name(d.dir) + " is full";
                fatal = true;
                return;
            }
            // A 4xx other than 408/429 will not get better by asking again.
            if ((r.status >= 400 && r.status < 500 && r.status != 408 && r.status != 429) || r.status == 200 || f > 20) {
                fatal = true;
                return;
            }
            const int wait = std::min(60, 2 << std::min(f, 5));  // 4, 8, 16, 32, 60 s
            ++waiting;
            m.unlock();
            for (int s = 0; s < wait * 4 && stop_ == 0; ++s) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            m.lock();
            --waiting;
        }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < kConnections && i < (int) pieces; ++i) {
        threads.emplace_back(worker, i);
    }
    // Progress once a second, the queue saved every ten.
    auto window_t = std::chrono::steady_clock::now();
    uint64_t window_bytes = received;
    for (int tick = 0;; ++tick) {
        bool running = false;
        {
            std::lock_guard<std::mutex> lock(m);
            size_t prefix = 0;
            while (prefix < pieces && state[prefix] == 2) {
                ++prefix;
            }
            d.resume = std::min(d.bytes, base + prefix * kPiece);
            for (uint8_t s : state) {
                running = running || s != 2;
            }
            running = running && !fatal;
            d.done = received;
            d.attempt = failures;
            if (waiting > 0 && waiting >= (int) threads.size()) {
                d.state = DownloadState::retrying;
                d.note = "Connection lost (" + last_error + "); trying again, attempt " + std::to_string((int) failures);
            } else {
                d.state = DownloadState::running;
                d.note.clear();
            }
        }
        const auto now = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(now - window_t).count();
        if (s >= 3.0) {
            d.mbps = (d.done - window_bytes) / 1e6 / s;
            window_t = now;
            window_bytes = d.done;
        }
        if (!running || stop_ != 0) {
            break;
        }
        publish(tick % 10 == 0);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    for (auto & t : threads) {
        t.join();
    }
    fsync(fd);
    close(fd);
    d.mbps = 0;
    if (stop_ != 0 || fatal) {
        d.state = stop_ == 1 ? DownloadState::paused : DownloadState::failed;
        d.note = stop_ == 1 ? "Paused" : stop_ == 2 ? "" : "Failed: " + last_error + "  \xC2\xB7  Cross tries again";
        d.done = d.resume;  // what a restart keeps
        publish(true);
        return false;
    }
    // Whole: check it. The queue is saved first, with `resume` at the file's
    // size, so a restart during the check repeats only the check, not the last
    // piece; and the row says how far the check has read (#30).
    d.note = "Checking the SHA-256";
    publish(true);
    sha256_t h;
    sha256_init(&h);
    if (FILE * in = std::fopen(part.c_str(), "rb")) {
        std::vector<unsigned char> buf(8u << 20);
        uint64_t read = 0;
        auto shown = std::chrono::steady_clock::now();
        for (size_t n; (n = std::fread(buf.data(), 1, buf.size(), in)) > 0;) {
            sha256_update(&h, buf.data(), n);
            read += n;
            const auto now = std::chrono::steady_clock::now();
            if (now - shown >= std::chrono::seconds(1)) {
                shown = now;
                d.note = "Checking the SHA-256, " + std::to_string((int) (100.0 * (double) read / (double) d.bytes)) + "%";
                publish(false);
            }
        }
        std::fclose(in);
    }
    unsigned char digest[32];
    sha256_final(&h, digest);
    char hex[65];
    for (int i = 0; i < 32; ++i) {
        std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    }
    if (!d.sha256.empty() && d.sha256 != hex) {
        unlink(part.c_str());
        d.resume = d.done = 0;
        d.state = DownloadState::failed;
        d.note = "Failed: the checksum does not match; Cross downloads it again";
    } else if (std::rename(part.c_str(), to.c_str()) != 0) {
        d.state = DownloadState::failed;
        d.note = "Failed: could not rename the file";
    } else {
        d.state = DownloadState::done;
        d.note = "Saved to " + to;
    }
    ps5lm::log_line("download %s/%s: %s\n", d.repo.c_str(), d.file.c_str(), d.note.c_str());
    publish(true);
    return d.state == DownloadState::done;
}

}  // namespace ps5lm
