// Where PS5LM keeps its files. With full storage the HEN opened /data to the
// app and everything lives in /data/PS5LM, as it always has. Without it the app
// uses its own storage, /download0/PS5LM: the system gives every title a
// /download0 sized by downloadDataSize in sce_sys/param.json (16 GiB for this
// app), and the app keeps its settings, logs and models there instead.
//
// A title's statvfs is a stub, so the room left in the app's storage is
// counted here: its size less what the files in it take. (Whether the system
// sets the whole size aside on internal storage up front is not known; a
// download that meets a full drive anyway stops with that reason.)
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>

namespace ps5lm {

constexpr const char * kFullRoot = "/data/PS5LM";
constexpr const char * kAppRoot  = "/download0/PS5LM";

// The folder in use, chosen once at start (main.cpp) before any thread reads it.
void                set_data_root(const std::string & root);
const std::string & data_root();
bool                app_storage();  // the root is the app's own storage
std::string         data_path(const std::string & name);  // data_root() + "/" + name

// The app's own storage: its size from sce_sys/param.json (16 GiB when the file
// cannot be read), the bytes its files take, and what is left.
uint64_t app_storage_size();
uint64_t app_storage_used();
double   app_storage_free_gib();

// The choice made on the first start without full storage: "app" once the
// user chose the app's storage, empty while they have not.
std::string storage_choice();
void        set_storage_choice(const std::string & choice);

// Bytes in the files under a folder, or 0.
uint64_t folder_bytes(const std::string & path);

// The kind of drive holding `path`: "Internal storage", "App storage",
// "M.2 drive", "USB drive".
std::string drive_name(const std::string & path);

}  // namespace ps5lm
