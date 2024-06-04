// Copyright (c) 2009, Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_DIRECTORY_READER_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_DIRECTORY_READER_H_

#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include <assert.h>
#include <errno.h>
#include <string.h>

#include "common/linux/linux_libc_support.h"
#include "third_party/lss/linux_syscall_support.h"

namespace google_breakpad {

// A class for enumerating a directory without using diropen/readdir or other
// functions which may allocate memory.
class DirectoryReader {
 public:
  DirectoryReader(int fd)
      : fd_(fd),
        buf_used_(0) {
  }

  // Return the next entry from the directory
  //   name: (output) the NUL terminated entry name
  //
  // Returns true iff successful (false on EOF).
  //
  // After calling this, one must call |PopEntry| otherwise you'll get the same
  // entry over and over.
  bool GetNextEntry(const char** name) {
    struct kernel_dirent* const dent =
      reinterpret_cast<kernel_dirent*>(buf_);

    if (buf_used_ == 0) {
      // need to read more entries.
      const int n = sys_getdents(fd_, dent, sizeof(buf_));
      if (n < 0) {
        return false;
      } else if (n == 0) {
        hit_eof_ = true;
      } else {
        buf_used_ += n;
      }
    }

    if (buf_used_ == 0 && hit_eof_)
      return false;

    assert(buf_used_ > 0);

    *name = dent->d_name;
    return true;
  }

  void PopEntry() {
    if (!buf_used_)
      return;

    const struct kernel_dirent* const dent =
      reinterpret_cast<kernel_dirent*>(buf_);

    buf_used_ -= dent->d_reclen;
    my_memmove(buf_, buf_ + dent->d_reclen, buf_used_);
  }

 private:
  const int fd_;
  bool hit_eof_;
  unsigned buf_used_;
  uint8_t buf_[sizeof(struct kernel_dirent) + NAME_MAX + 1];
};

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_MINIDUMP_WRITER_DIRECTORY_READER_H_
