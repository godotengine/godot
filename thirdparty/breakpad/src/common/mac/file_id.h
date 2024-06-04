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

// file_id.h: Return a unique identifier for a file
//
// Author: Dan Waylonis

#ifndef COMMON_MAC_FILE_ID_H__
#define COMMON_MAC_FILE_ID_H__

#include <limits.h>
#include <mach/machine.h>

namespace google_breakpad {

class FileID {
 public:
  FileID(const char *path);
  ~FileID() {}

  // Load the identifier for the file path specified in the constructor into
  // |identifier|.  Return false if the identifier could not be created for the
  // file.
  // The current implementation will return the MD5 hash of the file's bytes.
  bool FileIdentifier(unsigned char identifier[16]);

  // Treat the file as a mach-o file that will contain one or more archicture.
  // Accepted values for |cpu_type| and |cpu_subtype| (e.g., CPU_TYPE_X86 or
  // CPU_TYPE_POWERPC) are listed in /usr/include/mach/machine.h.
  // If |cpu_type| is 0, then the native cpu type is used. If |cpu_subtype| is
  // CPU_SUBTYPE_MULTIPLE, the match is only done on |cpu_type|.
  // Returns false if opening the file failed or if the |cpu_type|/|cpu_subtype|
  // is not present in the file.
  // Return the unique identifier in |identifier|.
  // The current implementation will look for the (in order of priority):
  // LC_UUID, LC_ID_DYLIB, or MD5 hash of the given |cpu_type|.
  bool MachoIdentifier(cpu_type_t cpu_type,
                       cpu_subtype_t cpu_subtype,
                       unsigned char identifier[16]);

  // Convert the |identifier| data to a NULL terminated string.  The string will
  // be formatted as a UUID (e.g., 22F065BB-FC9C-49F7-80FE-26A7CEBD7BCE).
  // The |buffer| should be at least 37 bytes long to receive all of the data
  // and termination.  Shorter buffers will contain truncated data.
  static void ConvertIdentifierToString(const unsigned char identifier[16],
                                        char *buffer, int buffer_length);

 private:
  // Storage for the path specified
  char path_[PATH_MAX];
};

}  // namespace google_breakpad

#endif  // COMMON_MAC_FILE_ID_H__
