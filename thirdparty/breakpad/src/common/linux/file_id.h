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
//
// file_id.h: Return a unique identifier for a file
//

#ifndef COMMON_LINUX_FILE_ID_H__
#define COMMON_LINUX_FILE_ID_H__

#include <limits.h>
#include <string>

#include "common/linux/guid_creator.h"
#include "common/memory_allocator.h"
#include "common/using_std_string.h"

namespace google_breakpad {

// GNU binutils' ld defaults to 'sha1', which is 160 bits == 20 bytes,
// so this is enough to fit that, which most binaries will use.
// This is just a sensible default for auto_wasteful_vector so most
// callers can get away with stack allocation.
static const size_t kDefaultBuildIdSize = 20;

class FileID {
 public:
  explicit FileID(const char* path);
  ~FileID() {}

  // Load the identifier for the elf file path specified in the constructor into
  // |identifier|.
  //
  // The current implementation will look for a .note.gnu.build-id
  // section and use that as the file id, otherwise it falls back to
  // XORing the first 4096 bytes of the .text section to generate an identifier.
  bool ElfFileIdentifier(wasteful_vector<uint8_t>& identifier);

  // Load the identifier for the elf file mapped into memory at |base| into
  // |identifier|. Return false if the identifier could not be created for this
  // file.
  static bool ElfFileIdentifierFromMappedFile(
      const void* base,
      wasteful_vector<uint8_t>& identifier);

  // Convert the |identifier| data to a string.  The string will
  // be formatted as a UUID in all uppercase without dashes.
  // (e.g., 22F065BBFC9C49F780FE26A7CEBD7BCE).
  static string ConvertIdentifierToUUIDString(
      const wasteful_vector<uint8_t>& identifier);

  // Convert the entire |identifier| data to a hex string.
  static string ConvertIdentifierToString(
      const wasteful_vector<uint8_t>& identifier);

 private:
  // Storage for the path specified
  string path_;
};

}  // namespace google_breakpad

#endif  // COMMON_LINUX_FILE_ID_H__
