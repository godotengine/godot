// Copyright (c) 2013, Google Inc.
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

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_PROC_CPUINFO_READER_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_PROC_CPUINFO_READER_H_

#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "client/linux/minidump_writer/line_reader.h"
#include "common/linux/linux_libc_support.h"
#include "third_party/lss/linux_syscall_support.h"

namespace google_breakpad {

// A class for reading /proc/cpuinfo without using fopen/fgets or other
// functions which may allocate memory.
class ProcCpuInfoReader {
public:
  ProcCpuInfoReader(int fd)
    : line_reader_(fd), pop_count_(-1) {
  }

  // Return the next field name, or NULL in case of EOF.
  // field: (output) Pointer to zero-terminated field name.
  // Returns true on success, or false on EOF or error (line too long).
  bool GetNextField(const char** field) {
    for (;;) {
      const char* line;
      unsigned line_len;

      // Try to read next line.
      if (pop_count_ >= 0) {
        line_reader_.PopLine(pop_count_);
        pop_count_ = -1;
      }

      if (!line_reader_.GetNextLine(&line, &line_len))
        return false;

      pop_count_ = static_cast<int>(line_len);

      const char* line_end = line + line_len;

      // Expected format: <field-name> <space>+ ':' <space> <value>
      // Note that:
      //   - empty lines happen.
      //   - <field-name> can contain spaces.
      //   - some fields have an empty <value>
      char* sep = static_cast<char*>(my_memchr(line, ':', line_len));
      if (sep == NULL)
        continue;

      // Record the value. Skip leading space after the column to get
      // its start.
      const char* val = sep+1;
      while (val < line_end && my_isspace(*val))
        val++;

      value_ = val;
      value_len_ = static_cast<size_t>(line_end - val);

      // Remove trailing spaces before the column to properly 0-terminate
      // the field name.
      while (sep > line && my_isspace(sep[-1]))
        sep--;

      if (sep == line)
        continue;

      // zero-terminate field name.
      *sep = '\0';

      *field = line;
      return true;
    }
  }

  // Return the field value. This must be called after a succesful
  // call to GetNextField().
  const char* GetValue() {
    assert(value_);
    return value_;
  }

  // Same as GetValue(), but also returns the length in characters of
  // the value.
  const char* GetValueAndLen(size_t* length) {
    assert(value_);
    *length = value_len_;
    return value_;
  }

private:
  LineReader line_reader_;
  int pop_count_;
  const char* value_;
  size_t value_len_;
};

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_MINIDUMP_WRITER_PROC_CPUINFO_READER_H_
