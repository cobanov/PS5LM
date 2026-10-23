// The model market: search Hugging Face for GGUF models, show which files
// fit this console, and download one into the models folder. Network work
// runs on its own thread; the screen reads a snapshot.
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ps5lm {

struct MarketFile {
    std::string name;            // "Qwen3.8-27B-UD-Q2_K_XL.gguf"
    std::string sha256;          // from the repository's LFS pointer
    uint64_t    bytes = 0;
    std::string plan;            // the planner's line, once its header was read
    int         fits  = -1;      // 1 fits, 0 too big, -1 not planned yet, -2 could not tell
    bool        checking = false;  // its header is being read
};

struct MarketRepo {
    std::string id;              // "unsloth/Qwen3.8-27B-GGUF"
    uint64_t    downloads = 0;
    std::vector<MarketFile> files;  // filled when the repository is opened
};

// One file to download, by name.
struct MarketJob {
    std::string repo, file, sha256, dir;
    uint64_t    bytes = 0;
    std::string save_as;         // the name on the console when it is not the file's ("mmproj-<model>.gguf")
};

enum class DownloadState : uint8_t { queued, running, retrying, paused, failed, done };

// A queued file. The queue is kept in /data/PS5LM/downloads.json, and the
// partial file (name.gguf.part) stays on disk, so a download carries on
// after a dropped connection, a pause or the app restarting.
struct Download : MarketJob {
    DownloadState state = DownloadState::queued;
    uint64_t    done = 0;        // bytes fetched
    uint64_t    resume = 0;      // bytes done from the start of the file, kept across restarts
    double      mbps = 0;        // over the last seconds, while it runs
    int         attempt = 0;     // retries since the last progress
    std::string note;            // "Saved to …", the error, or why it waits
};

struct MarketView {
    std::string query;
    bool        busy = false;    // a search, listing or plan is running
    std::string status;          // the last error or note, for the screen
    std::vector<MarketRepo> repos;
    int         open = -1;       // the repository whose files are listed
    std::vector<Download> downloads;  // the queue, oldest first
    int         saved = 0;       // downloads finished since the app started
    std::string note;            // the last download refused or finished, for the Get models page
};

class Market {
  public:
    // budget_gib: GPU memory free before any model, for the fit column.
    void start(double budget_gib);
    MarketView snapshot();

    void search(const std::string & query);
    void open_repo(int index);
    void plan_file(int repo, int file);  // reads its header (16 MiB range request)
    bool job(int repo, int file, const std::string & dir, MarketJob * out);

    // The queue. `free_gib`: room on the drive it goes to, or -1 when unknown
    // (the check is skipped). False, with the reason in the view's note, when
    // it does not fit or is queued already.
    bool enqueue(const MarketJob & job, double free_gib);
    bool enqueue(const std::vector<MarketJob> & jobs, double free_gib);  // all on one drive, all or none
    void pause(int index);    // stops it, keeping the partial file
    void resume(int index);   // a paused or failed one goes back in the queue
    void remove(int index);   // cancels it and deletes the partial file; a finished one only leaves the list
    void load_queue();        // at start: what was queued or running carries on

  private:
    void run(std::atomic<bool> & flag, std::function<void()> job);
    void worker();            // the download thread: the queue in order, one at a time
    bool fetch(Download & d); // one download to the end, retrying; false when stopped
    void save_queue();        // under mutex_
    std::mutex mutex_;
    MarketView view_;
    double     budget_gib_ = 0;
    std::atomic<bool> browsing_{ false };
    struct PlanJob { std::string repo, file; uint64_t bytes = 0; };
    std::vector<PlanJob> plans_;           // fit checks asked for, newest last
    std::atomic<bool> planning_{ false };  // their thread runs
    std::atomic<bool> working_{ false };  // the worker thread runs
    std::atomic<int>  stop_{ 0 };         // 1 pause, 2 cancel the running download
};

}  // namespace ps5lm
