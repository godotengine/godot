// Copyright (c) 2015, Google Inc.
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

// Original author: Erik Chen <erikchen@chromium.org>

// super_fat_arch.h: A class to handle 64-bit object files. Has conversions to
// and from struct fat_arch.

#ifndef BREAKPAD_COMMON_MAC_SUPER_FAT_ARCH_H_
#define BREAKPAD_COMMON_MAC_SUPER_FAT_ARCH_H_

#include <limits>
#include <mach-o/fat.h>
#include <stdint.h>

// Similar to struct fat_arch, except size-related parameters support
// 64-bits.
class SuperFatArch {
 public:
  uint32_t cputype;
  uint32_t cpusubtype;
  uint64_t offset;
  uint64_t size;
  uint64_t align;

  SuperFatArch() :
      cputype(0),
      cpusubtype(0),
      offset(0),
      size(0),
      align(0) {
  }

  explicit SuperFatArch(const struct fat_arch& arch) :
      cputype(arch.cputype),
      cpusubtype(arch.cpusubtype),
      offset(arch.offset),
      size(arch.size),
      align(arch.align) {
  }

  // Returns false if the conversion cannot be made.
  // If the conversion succeeds, the result is placed in |output_arch|.
  bool ConvertToFatArch(struct fat_arch* output_arch) const {
    if (offset > std::numeric_limits<uint32_t>::max())
      return false;
    if (size > std::numeric_limits<uint32_t>::max())
      return false;
    if (align > std::numeric_limits<uint32_t>::max())
      return false;
    struct fat_arch arch;
    arch.cputype = cputype;
    arch.cpusubtype = cpusubtype;
    arch.offset = offset;
    arch.size = size;
    arch.align = align;
    *output_arch = arch;
    return true;
  }
};

#endif  // BREAKPAD_COMMON_MAC_SUPER_FAT_ARCH_H_
