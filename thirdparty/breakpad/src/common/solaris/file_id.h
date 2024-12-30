// Copyright 2007 Google LLC
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

// file_id.h: Return a unique identifier for a file
//
// Author: Alfred Peng

#ifndef COMMON_SOLARIS_FILE_ID_H__
#define COMMON_SOLARIS_FILE_ID_H__

#include <limits.h>

namespace google_breakpad {
namespace elf {

class FileID {
 public:
  FileID(const char *path);
  ~FileID() {};

  // Load the identifier for the elf file path specified in the constructor into
  // |identifier|.  Return false if the identifier could not be created for the
  // file.
  // The current implementation will return the MD5 hash of the file's bytes.
  bool ElfFileIdentifier(unsigned char identifier[16]);

  // Convert the |identifier| data to a NULL terminated string.  The string will
  // be formatted as a MDCVInfoPDB70 struct.
  // The |buffer| should be at least 34 bytes long to receive all of the data
  // and termination. Shorter buffers will return false.
  static bool ConvertIdentifierToString(const unsigned char identifier[16],
                                        char *buffer, int buffer_length);

 private:
  // Storage for the path specified
  char path_[PATH_MAX];
};

}  // elf
}  // namespace google_breakpad

#endif  // COMMON_SOLARIS_FILE_ID_H__
