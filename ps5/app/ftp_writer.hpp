// Writes downloads through an FTP server on this console (ftpsrv, a
// payload) instead of the title's own write(). The console slows a title's
// writes to /data: after about 2.2 GB they drop to about 1.2 MB/s, while
// ftpsrv keeps writing at 80 MB/s and more in the same moment (#41,
// docs/CONSOLE.md).
//
// ftpsrv's REST n + STOR cuts the file at n and writes from there; a REST past
// the end is refused (551). So a file is written from its start, in order: one
// STOR per piece, at the end of what is already there.
//
// The data connection is passive: EPSV, or PASV for a server without it
// (ftpsrv up to 0.15 answers EPSV with 502).
//
// connect_ftp_writer finds the server: the user's ftpsrv on 2121 when a test
// write through it works, else PS5LM's own copy on 2123 (third_party/ftpsrv
// with patches/ftpsrv, loopback only), which the title sends to the ELF loader
// itself when nothing answers there.
//
// Ported from PS5SD (Darkatek7/PS5SD, ps5/app/ftp_writer.*).
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace ps5lm {

class FtpWriter {
public:
    explicit FtpWriter(std::string host = "127.0.0.1", int port = 2121) : host_(std::move(host)), port_(port) {}
    ~FtpWriter() { close(); }
    FtpWriter(const FtpWriter &)             = delete;
    FtpWriter & operator=(const FtpWriter &) = delete;

    // The control connection: connects and logs in, unless it already is.
    bool connect(std::string & error);
    // Writes n bytes to path at offset, which must not lie past the file's end;
    // the file is cut to offset + n. True once the server confirmed (226), so
    // the bytes are in the file.
    bool write_at(const std::string & path, uint64_t offset, const char * data, size_t n, std::string & error);
    // One long transfer, for a file in one stream: begin at offset, send, end.
    // end() is true once the server confirmed the whole transfer.
    bool begin(const std::string & path, uint64_t offset, std::string & error);
    bool send(const char * data, size_t n, std::string & error);
    bool end(std::string & error);
    bool rename(const std::string & from, const std::string & to, std::string & error);
    // The file's size; false if it is not there (550) or the server is gone.
    bool size(const std::string & path, uint64_t & out, std::string & error);
    void close();

    void set_timeout(int seconds) { timeout_ = seconds; }
    // Only while not connected.
    void set_port(int port) { port_ = port; }
    int  port() const { return port_; }
    // The last connect() found nothing listening (the TCP connect itself failed).
    bool refused() const { return refused_; }

private:
    bool command(const std::string & line, int & code, std::string & reply, std::string & error);
    bool reply(int & code, std::string & text, std::string & error);
    bool open_data(std::string & error);
    int  dial(int port, std::string & error);
    bool send_all(int fd, const char * data, size_t n, std::string & error);

    std::string host_;
    int         port_;
    int         timeout_ = 30;
    bool        refused_ = false;
    bool        pasv_    = false;  // the server does not know EPSV
    int         ctrl_    = -1;
    int         data_    = -1;
    std::string inbox_;  // control bytes read past the last reply line
};

constexpr int kFtpsrvPort      = 2121;  // the user's ftpsrv
constexpr int kOwnFtpsrvPort   = 2123;  // PS5LM's copy (patches/ftpsrv)
constexpr int kElfLoaderPort   = 9021;
constexpr const char * kOwnFtpsrvElf = "/app0/payloads/ps5lm-ftpsrv.elf";
constexpr const char * kFtpCheckFile = "/data/PS5LM/.ps5lm-ftp-check";

// Connects ftp (not connected yet) to 2121, else 2123. The first time 2121
// answers, one byte is written to kFtpCheckFile through it; when that fails
// (an ftpsrv without a command PS5LM needs), 2121 is left alone for the rest
// of the run. When nothing listens on 2123, the first call that needs it sends
// kOwnFtpsrvElf to the ELF loader on 9021 and waits up to 5 s for it; that
// happens at most once per app run, also when several threads get here at
// once. False with error when no server answers: the caller writes the file
// itself then.
bool connect_ftp_writer(FtpWriter & ftp, std::string & error);

// ftpsrv's answer when the disk is full (552, or 451/452 with errno's text).
bool ftp_disk_full(const std::string & error);

}  // namespace ps5lm
