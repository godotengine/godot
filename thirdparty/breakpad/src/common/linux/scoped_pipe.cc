// Copyright 2022 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/linux/scoped_pipe.h"

#include <unistd.h>

#include "common/linux/eintr_wrapper.h"

namespace google_breakpad {

ScopedPipe::ScopedPipe() {
  fds_[0] = -1;
  fds_[1] = -1;
}

ScopedPipe::~ScopedPipe() {
  CloseReadFd();
  CloseWriteFd();
}

bool ScopedPipe::Init() {
  return pipe(fds_) == 0;
}

void ScopedPipe::CloseReadFd() {
  if (fds_[0] != -1) {
    close(fds_[0]);
    fds_[0] = -1;
  }
}

void ScopedPipe::CloseWriteFd() {
  if (fds_[1] != -1) {
    close(fds_[1]);
    fds_[1] = -1;
  }
}

bool ScopedPipe::ReadLine(std::string& line) {
  // Simple buffered file read. `read_buffer_` stores previously read bytes, and
  // we either return a line from this buffer, or we append blocks of read bytes
  // to the buffer until we have a complete line.
  size_t eol_index = read_buffer_.find('\n');

  // While we don't have a full line, and read pipe is valid.
  while (eol_index == std::string::npos && GetReadFd() != -1) {
    // Read a block of 128 bytes from the read pipe.
    char read_buf[128];
    ssize_t read_len = HANDLE_EINTR(
      read(GetReadFd(), read_buf, sizeof(read_buf)));
    if (read_len <= 0) {
      // Pipe error, or pipe has been closed.
      CloseReadFd();
      break;
    }

    // Append the block, and check if we have a full line now.
    read_buffer_.append(read_buf, read_len);
    eol_index = read_buffer_.find('\n');
  }

  if (eol_index != std::string::npos) {
    // We have a full line to output.
    line = read_buffer_.substr(0, eol_index);
    if (eol_index < read_buffer_.size()) {
      read_buffer_ = read_buffer_.substr(eol_index + 1);
    } else {
      read_buffer_ = "";
    }

    return true;
  }

  if (read_buffer_.size()) {
    // We don't have a full line to output, but we can only reach here if the
    // pipe has closed and there are some bytes left at the end, so we should
    // return those bytes.
    line = std::move(read_buffer_);
    read_buffer_ = "";

    return true;
  }

  // We don't have any buffered data left, and the pipe has closed.
  return false;
}

int ScopedPipe::Dup2WriteFd(int new_fd) const {
  return dup2(fds_[1], new_fd);
}

bool ScopedPipe::WriteForTesting(const void* bytes, size_t bytes_len) {
  ssize_t r = HANDLE_EINTR(write(GetWriteFd(), bytes, bytes_len));
  if (r != static_cast<ssize_t>(bytes_len)) {
    CloseWriteFd();
    return false;
  }

  return true;
}

}  // namespace google_breakpad
