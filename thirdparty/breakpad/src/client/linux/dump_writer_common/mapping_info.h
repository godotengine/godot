// Copyright 2014 Google LLC
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

#ifndef CLIENT_LINUX_DUMP_WRITER_COMMON_MAPPING_INFO_H_
#define CLIENT_LINUX_DUMP_WRITER_COMMON_MAPPING_INFO_H_

#include <limits.h>
#include <list>
#include <stdint.h>

#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

// One of these is produced for each mapping in the process (i.e. line in
// /proc/$x/maps).
struct MappingInfo {
  // On Android, relocation packing can mean that the reported start
  // address of the mapping must be adjusted by a bias in order to
  // compensate for the compression of the relocation section. The
  // following two members hold (after LateInit) the adjusted mapping
  // range. See crbug.com/606972 for more information.
  uintptr_t start_addr;
  size_t size;
  // When Android relocation packing causes |start_addr| and |size| to
  // be modified with a load bias, we need to remember the unbiased
  // address range. The following structure holds the original mapping
  // address range as reported by the operating system.
  struct {
    uintptr_t start_addr;
    uintptr_t end_addr;
  } system_mapping_info;
  size_t offset;  // offset into the backed file.
  bool exec;  // true if the mapping has the execute bit set.
  char name[NAME_MAX];
};

struct MappingEntry {
  MappingInfo first;
  uint8_t second[sizeof(MDGUID)];
};

// A list of <MappingInfo, GUID>
typedef std::list<MappingEntry> MappingList;

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_DUMP_WRITER_COMMON_MAPPING_INFO_H_
