// Copyright 2009 Google LLC
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

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_LINE_READER_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_LINE_READER_H_

#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "common/linux/linux_libc_support.h"
#include "third_party/lss/linux_syscall_support.h"

namespace google_breakpad {

// A class for reading a file, line by line, without using fopen/fgets or other
// functions which may allocate memory.
class LineReader {
 public:
  LineReader(int fd)
      : fd_(fd),
        hit_eof_(false),
        buf_used_(0) {
  }

  // The maximum length of a line.
  static const size_t kMaxLineLen = 512;

  // Return the next line from the file.
  //   line: (output) a pointer to the start of the line. The line is NUL
  //     terminated.
  //   len: (output) the length of the line (not inc the NUL byte)
  //
  // Returns true iff successful (false on EOF).
  //
  // One must call |PopLine| after this function, otherwise you'll continue to
  // get the same line over and over.
  bool GetNextLine(const char** line, unsigned* len) {
    for (;;) {
      if (buf_used_ == 0 && hit_eof_)
        return false;

      for (unsigned i = 0; i < buf_used_; ++i) {
        if (buf_[i] == '\n' || buf_[i] == 0) {
          buf_[i] = 0;
          *len = i;
          *line = buf_;
          return true;
        }
      }

      if (buf_used_ == sizeof(buf_)) {
        // we scanned the whole buffer and didn't find an end-of-line marker.
        // This line is too long to process.
        return false;
      }

      // We didn't find any end-of-line terminators in the buffer. However, if
      // this is the last line in the file it might not have one:
      if (hit_eof_) {
        assert(buf_used_);
        // There's room for the NUL because of the buf_used_ == sizeof(buf_)
        // check above.
        buf_[buf_used_] = 0;
        *len = buf_used_;
        buf_used_ += 1;  // since we appended the NUL.
        *line = buf_;
        return true;
      }

      // Otherwise, we should pull in more data from the file
      const ssize_t n = sys_read(fd_, buf_ + buf_used_,
                                 sizeof(buf_) - buf_used_);
      if (n < 0) {
        return false;
      } else if (n == 0) {
        hit_eof_ = true;
      } else {
        buf_used_ += n;
      }

      // At this point, we have either set the hit_eof_ flag, or we have more
      // data to process...
    }
  }

  void PopLine(unsigned len) {
    // len doesn't include the NUL byte at the end.

    assert(buf_used_ >= len + 1);
    buf_used_ -= len + 1;
    my_memmove(buf_, buf_ + len + 1, buf_used_);
  }

 private:
  const int fd_;

  bool hit_eof_;
  unsigned buf_used_;
  char buf_[kMaxLineLen];
};

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_MINIDUMP_WRITER_LINE_READER_H_
