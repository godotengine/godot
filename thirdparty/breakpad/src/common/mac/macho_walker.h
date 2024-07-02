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

// macho_walker.h: Iterate over the load commands in a mach-o file
//
// Author: Dan Waylonis

#ifndef COMMON_MAC_MACHO_WALKER_H__
#define COMMON_MAC_MACHO_WALKER_H__

#include <mach/machine.h>
#include <mach-o/loader.h>
#include <sys/types.h>

namespace MacFileUtilities {

class MachoWalker {
 public:
  // A callback function executed when a new load command is read.  If no
  // further processing of load commands is desired, return false.  Otherwise,
  // return true.
  // |cmd| is the current command, and |offset| is the location relative to the
  // beginning of the file (not header) where the command was read.  If |swap|
  // is set, then any command data (other than the returned load_command) should
  // be swapped when read
  typedef bool (*LoadCommandCallback)(MachoWalker* walker, load_command* cmd,
                                      off_t offset, bool swap, void* context);

  MachoWalker(const char* path, LoadCommandCallback callback, void* context);
  MachoWalker(void* memory, size_t size, LoadCommandCallback callback,
              void* context);
  ~MachoWalker();

  // Begin walking the header for |cpu_type| and |cpu_subtype|.  If |cpu_type|
  // is 0, then the native cpu type is used. Otherwise, accepted values are
  // listed in /usr/include/mach/machine.h (e.g., CPU_TYPE_X86 or
  // CPU_TYPE_POWERPC). If |cpu_subtype| is CPU_SUBTYPE_MULTIPLE, the match is
  // only done on |cpu_type|.
  // Returns false if opening the file failed or if the |cpu_type|/|cpu_subtype|
  // is not present in the file.
  bool WalkHeader(cpu_type_t cpu_type, cpu_subtype_t cpu_subtype);

  // Read |size| bytes from the opened file at |offset| into |buffer|
  bool ReadBytes(void* buffer, size_t size, off_t offset);

  // Return the current header and header offset
  bool CurrentHeader(struct mach_header_64* header, off_t* offset);

 private:
  // Locate (if any) the header offset for |cpu_type| and return in |offset|.
  // Return true if found, false otherwise.
  bool FindHeader(cpu_type_t cpu_type,
                  cpu_subtype_t cpu_subtype,
                  off_t& offset);

  // Process an individual header starting at |offset| from the start of the
  // file.  Return true if successful, false otherwise.
  bool WalkHeaderAtOffset(off_t offset);
  bool WalkHeader64AtOffset(off_t offset);

  // Bottleneck for walking the load commands
  bool WalkHeaderCore(off_t offset, uint32_t number_of_commands, bool swap);

  // File descriptor to the opened file
  int file_;

  // Memory location to read from.
  void* memory_;

  // Size of the memory segment we can read from.
  size_t memory_size_;

  // User specified callback & context
  LoadCommandCallback callback_;
  void* callback_context_;

  // Current header, size, and offset.  The mach_header_64 is used for both
  // 32-bit and 64-bit headers because they only differ in their last field
  // (reserved).  By adding the |current_header_size_| and the
  // |current_header_offset_|, you can determine the offset in the file just
  // after the header.
  struct mach_header_64* current_header_;
  unsigned long current_header_size_;
  off_t current_header_offset_;

 private:
  MachoWalker(const MachoWalker&);
  MachoWalker& operator=(const MachoWalker&);
};

}  // namespace MacFileUtilities

#endif  // COMMON_MAC_MACHO_WALKER_H__
