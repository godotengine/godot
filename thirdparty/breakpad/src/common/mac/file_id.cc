// Copyright 2006 Google LLC
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

// file_id.cc: Return a unique identifier for a file
//
// See file_id.h for documentation
//
// Author: Dan Waylonis

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/mac/file_id.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>

#include "common/mac/macho_id.h"
#include "common/scoped_ptr.h"

using MacFileUtilities::MachoID;

namespace google_breakpad {
namespace mach_o {
// Constructs a FileID given a path to a file
FileID::FileID(const char* path) : memory_(nullptr), size_(0) {
  snprintf(path_, sizeof(path_), "%s", path);
}

// Constructs a FileID given the contents of a file and its size
FileID::FileID(void* memory, size_t size)
    : path_(), memory_(memory), size_(size) {}

bool FileID::MachoIdentifier(cpu_type_t cpu_type,
                             cpu_subtype_t cpu_subtype,
                             unsigned char identifier[16]) {
  scoped_ptr<MachoID> macho;
  if (memory_) {
    macho.reset(new MachoID(memory_, size_));
  } else {
    macho.reset(new MachoID(path_));
  }
  if (macho->UUIDCommand(cpu_type, cpu_subtype, identifier))
    return true;

  return macho->MD5(cpu_type, cpu_subtype, identifier);
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

}  // namespace mach_o
}  // namespace google_breakpad
