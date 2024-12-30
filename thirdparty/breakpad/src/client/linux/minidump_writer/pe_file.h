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

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_PE_FILE_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_PE_FILE_H_

#include "client/linux/minidump_writer/pe_structs.h"

namespace google_breakpad {

typedef enum {
  notPeCoff = 0,
  peWithoutBuildId = 1,
  peWithBuildId = 2
} PEFileFormat;

class PEFile {
 public:
  /**
   * Attempts to parse RSDS_DEBUG_FORMAT record from a PE (Portable
   * Executable) file. To do this we check whether the loaded file is a PE
   * file, and if it is - try to find IMAGE_DEBUG_DIRECTORY structure with
   * its type set to IMAGE_DEBUG_TYPE_CODEVIEW.
   *
   * @param filename Filename for the module to parse.
   * @param debug_info RSDS_DEBUG_FORMAT struct to be populated with PE debug
   * info (GUID and age).
   * @return
   *   notPeCoff: not PE/COFF file;
   *   peWithoutBuildId: a PE/COFF file but build-id is not set;
   *   peWithBuildId: a PE/COFF file and build-id is set.
   */
  static PEFileFormat TryGetDebugInfo(const char* filename,
                                      PRSDS_DEBUG_FORMAT debug_info);

 private:
  template <class TStruct>
  static const TStruct* TryReadStruct(const void* base,
                                      const DWORD position,
                                      const size_t file_size) {
    if (position + sizeof(TStruct) >= file_size){
      return nullptr;
    }

    const void* ptr = static_cast<const char*>(base) + position;
    return reinterpret_cast<const TStruct*>(ptr);
  }
};

}  // namespace google_breakpad
#endif  // CLIENT_LINUX_MINIDUMP_WRITER_PE_FILE_H_