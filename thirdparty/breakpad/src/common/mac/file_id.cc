// Copyright (c) 2006, Google Inc.
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

// file_id.cc: Return a unique identifier for a file
//
// See file_id.h for documentation
//
// Author: Dan Waylonis

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "common/mac/file_id.h"
#include "common/mac/macho_id.h"

using MacFileUtilities::MachoID;

namespace google_breakpad {

FileID::FileID(const char *path) {
  snprintf(path_, sizeof(path_), "%s", path);
}

bool FileID::FileIdentifier(unsigned char identifier[16]) {
  int fd = open(path_, O_RDONLY);
  if (fd == -1)
    return false;

  MD5Context md5;
  MD5Init(&md5);

  // Read 4k x 2 bytes at a time.  This is faster than just 4k bytes, but
  // doesn't seem to be an unreasonable size for the stack.
  unsigned char buffer[4096 * 2];
  size_t buffer_size = sizeof(buffer);
  while ((buffer_size = read(fd, buffer, buffer_size) > 0)) {
    MD5Update(&md5, buffer, static_cast<unsigned>(buffer_size));
  }

  close(fd);
  MD5Final(identifier, &md5);

  return true;
}

bool FileID::MachoIdentifier(cpu_type_t cpu_type,
                             cpu_subtype_t cpu_subtype,
                             unsigned char identifier[16]) {
  MachoID macho(path_);

  if (macho.UUIDCommand(cpu_type, cpu_subtype, identifier))
    return true;

  return macho.MD5(cpu_type, cpu_subtype, identifier);
}

// static
void FileID::ConvertIdentifierToString(const unsigned char identifier[16],
                                       char *buffer, int buffer_length) {
  int buffer_idx = 0;
  for (int idx = 0; (buffer_idx < buffer_length) && (idx < 16); ++idx) {
    int hi = (identifier[idx] >> 4) & 0x0F;
    int lo = (identifier[idx]) & 0x0F;

    if (idx == 4 || idx == 6 || idx == 8 || idx == 10)
      buffer[buffer_idx++] = '-';

    buffer[buffer_idx++] =
        static_cast<char>((hi >= 10) ? ('A' + hi - 10) : ('0' + hi));
    buffer[buffer_idx++] =
        static_cast<char>((lo >= 10) ? ('A' + lo - 10) : ('0' + lo));
  }

  // NULL terminate
  buffer[(buffer_idx < buffer_length) ? buffer_idx : buffer_idx - 1] = 0;
}

}  // namespace google_breakpad
