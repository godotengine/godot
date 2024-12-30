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

// macho_id.h: Functions to gather identifying information from a macho file
//
// Author: Dan Waylonis

#ifndef COMMON_MAC_MACHO_ID_H__
#define COMMON_MAC_MACHO_ID_H__

#include <limits.h>
#include <mach/machine.h>
#include <mach-o/loader.h>

#include "common/mac/macho_walker.h"
#include "common/md5.h"

namespace MacFileUtilities {

class MachoID {
 public:
  MachoID(const char* path);
  MachoID(void* memory, size_t size);
  ~MachoID();

  // For the given |cpu_type| and |cpu_subtype|, return a UUID from the LC_UUID
  // command.
  // Return false if there isn't a LC_UUID command.
  bool UUIDCommand(cpu_type_t cpu_type,
                   cpu_subtype_t cpu_subtype,
                   unsigned char identifier[16]);

  // For the given |cpu_type|, and |cpu_subtype| return the MD5 for the mach-o
  // data segment(s).
  // Return true on success, false otherwise
  bool MD5(cpu_type_t cpu_type,
           cpu_subtype_t cpu_subtype,
           unsigned char identifier[16]);

 private:
  // Signature of class member function to be called with data read from file
  typedef void (MachoID::*UpdateFunction)(unsigned char* bytes, size_t size);

  // Update the MD5 value by examining |size| |bytes| and applying the algorithm
  // to each byte.
  void UpdateMD5(unsigned char* bytes, size_t size);

  // Bottleneck for update routines
  void Update(MachoWalker* walker, off_t offset, size_t size);

  // Factory for the MachoWalker
  bool WalkHeader(cpu_type_t cpu_type, cpu_subtype_t cpu_subtype,
                  MachoWalker::LoadCommandCallback callback, void* context);

  // The callback from the MachoWalker for CRC and MD5
  static bool WalkerCB(MachoWalker* walker, load_command* cmd, off_t offset,
                       bool swap, void* context);

  // The callback from the MachoWalker for LC_UUID
  static bool UUIDWalkerCB(MachoWalker* walker, load_command* cmd, off_t offset,
                           bool swap, void* context);

  // File path
  char path_[PATH_MAX];

  // Memory region to read from
  void* memory_;

  // Size of the memory region
  size_t memory_size_;

  // The MD5 context
  google_breakpad::MD5Context md5_context_;

  // The current update to call from the Update callback
  UpdateFunction update_function_;
};

}  // namespace MacFileUtilities

#endif  // COMMON_MAC_MACHO_ID_H__
