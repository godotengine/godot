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

// macho_walker.cc: Iterate over the load commands in a mach-o file
//
// See macho_walker.h for documentation
//
// Author: Dan Waylonis

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <assert.h>
#include <fcntl.h>
#include <mach-o/fat.h>
#include <mach-o/loader.h>
#include <string.h>
#include <unistd.h>

#include "common/mac/arch_utilities.h"
#include "common/mac/byteswap.h"
#include "common/mac/macho_utilities.h"
#include "common/mac/macho_walker.h"

namespace MacFileUtilities {

MachoWalker::MachoWalker(const char* path, LoadCommandCallback callback,
                         void* context)
    : file_(-1),
      memory_(NULL),
      memory_size_(0),
      callback_(callback),
      callback_context_(context),
      current_header_(NULL),
      current_header_size_(0),
      current_header_offset_(0) {
  file_ = open(path, O_RDONLY);
}

MachoWalker::MachoWalker(void* memory, size_t size,
                         LoadCommandCallback callback, void* context)
    : file_(-1),
      memory_(memory),
      memory_size_(size),
      callback_(callback),
      callback_context_(context),
      current_header_(NULL),
      current_header_size_(0),
      current_header_offset_(0) {
}

MachoWalker::~MachoWalker() {
  if (file_ != -1)
    close(file_);
}

bool MachoWalker::WalkHeader(cpu_type_t cpu_type, cpu_subtype_t cpu_subtype) {
  cpu_type_t valid_cpu_type = cpu_type;
  cpu_subtype_t valid_cpu_subtype = cpu_subtype;
  // if |cpu_type| is 0, use the native cpu type.
  if (cpu_type == 0) {
    ArchInfo arch = GetLocalArchInfo();
    valid_cpu_type = arch.cputype;
    valid_cpu_subtype = CPU_SUBTYPE_MULTIPLE;
  }
  off_t offset;
  if (FindHeader(valid_cpu_type, valid_cpu_subtype, offset)) {
    if (cpu_type & CPU_ARCH_ABI64)
      return WalkHeader64AtOffset(offset);

    return WalkHeaderAtOffset(offset);
  }

  return false;
}

bool MachoWalker::ReadBytes(void* buffer, size_t size, off_t offset) {
  if (memory_) {
    if (offset < 0)
      return false;
    bool result = true;
    if (offset + size > memory_size_) {
      if (static_cast<size_t>(offset) >= memory_size_)
        return false;
      size = memory_size_ - static_cast<size_t>(offset);
      result = false;
    }
    memcpy(buffer, static_cast<char*>(memory_) + offset, size);
    return result;
  } else {
    return pread(file_, buffer, size, offset) == (ssize_t)size;
  }
}

bool MachoWalker::CurrentHeader(struct mach_header_64* header, off_t* offset) {
  if (current_header_) {
    memcpy(header, current_header_, sizeof(mach_header_64));
    *offset = current_header_offset_;
    return true;
  }

  return false;
}

bool MachoWalker::FindHeader(cpu_type_t cpu_type,
                             cpu_subtype_t cpu_subtype,
                             off_t& offset) {
  // Read the magic bytes that's common amongst all mach-o files
  uint32_t magic;
  if (!ReadBytes(&magic, sizeof(magic), 0))
    return false;

  offset = sizeof(magic);

  // Figure out what type of file we've got
  bool is_fat = false;
  if (magic == FAT_MAGIC || magic == FAT_CIGAM) {
    is_fat = true;
  }
  else if (magic != MH_MAGIC && magic != MH_CIGAM && magic != MH_MAGIC_64 &&
           magic != MH_CIGAM_64) {
    return false;
  }

  if (!is_fat) {
    // If we don't have a fat header, check if the cpu type matches the single
    // header
    struct mach_header header;
    if (!ReadBytes(&header, sizeof(header), 0))
      return false;

    if (magic == MH_CIGAM || magic == MH_CIGAM_64)
      breakpad_swap_mach_header(&header);

    if (cpu_type != header.cputype ||
        (cpu_subtype != CPU_SUBTYPE_MULTIPLE &&
         cpu_subtype != header.cpusubtype)) {
      return false;
    }

    offset = 0;
    return true;
  } else {
    // Read the fat header and find an appropriate architecture
    offset = 0;
    struct fat_header fat;
    if (!ReadBytes(&fat, sizeof(fat), offset))
      return false;

    if (NXHostByteOrder() != NX_BigEndian)
      breakpad_swap_fat_header(&fat);

    offset += sizeof(fat);

    // Search each architecture for the desired one
    struct fat_arch arch;
    for (uint32_t i = 0; i < fat.nfat_arch; ++i) {
      if (!ReadBytes(&arch, sizeof(arch), offset))
        return false;

      if (NXHostByteOrder() != NX_BigEndian)
        breakpad_swap_fat_arch(&arch, 1);

      if (arch.cputype == cpu_type &&
          (cpu_subtype == CPU_SUBTYPE_MULTIPLE ||
           arch.cpusubtype == cpu_subtype)) {
        offset = arch.offset;
        return true;
      }

      offset += sizeof(arch);
    }
  }

  return false;
}

bool MachoWalker::WalkHeaderAtOffset(off_t offset) {
  struct mach_header header;
  if (!ReadBytes(&header, sizeof(header), offset))
    return false;

  bool swap = (header.magic == MH_CIGAM);
  if (swap)
    breakpad_swap_mach_header(&header);

  // Copy the data into the mach_header_64 structure.  Since the 32-bit and
  // 64-bit only differ in the last field (reserved), this is safe to do.
  struct mach_header_64 header64;
  memcpy((void*)&header64, (const void*)&header, sizeof(header));
  header64.reserved = 0;

  current_header_ = &header64;
  current_header_size_ = sizeof(header); // 32-bit, not 64-bit
  current_header_offset_ = offset;
  offset += current_header_size_;
  bool result = WalkHeaderCore(offset, header.ncmds, swap);
  current_header_ = NULL;
  current_header_size_ = 0;
  current_header_offset_ = 0;
  return result;
}

bool MachoWalker::WalkHeader64AtOffset(off_t offset) {
  struct mach_header_64 header;
  if (!ReadBytes(&header, sizeof(header), offset))
    return false;

  bool swap = (header.magic == MH_CIGAM_64);
  if (swap)
    breakpad_swap_mach_header_64(&header);

  current_header_ = &header;
  current_header_size_ = sizeof(header);
  current_header_offset_ = offset;
  offset += current_header_size_;
  bool result = WalkHeaderCore(offset, header.ncmds, swap);
  current_header_ = NULL;
  current_header_size_ = 0;
  current_header_offset_ = 0;
  return result;
}

bool MachoWalker::WalkHeaderCore(off_t offset, uint32_t number_of_commands,
                                 bool swap) {
  for (uint32_t i = 0; i < number_of_commands; ++i) {
    struct load_command cmd;
    if (!ReadBytes(&cmd, sizeof(cmd), offset))
      return false;

    if (swap)
      breakpad_swap_load_command(&cmd);

    // Call the user callback
    if (callback_ && !callback_(this, &cmd, offset, swap, callback_context_))
      break;

    offset += cmd.cmdsize;
  }

  return true;
}

}  // namespace MacFileUtilities
