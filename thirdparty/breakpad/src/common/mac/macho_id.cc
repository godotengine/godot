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

// macho_id.cc: Functions to gather identifying information from a macho file
//
// See macho_id.h for documentation
//
// Author: Dan Waylonis


#include <fcntl.h>
#include <mach-o/loader.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/mac/macho_id.h"
#include "common/mac/macho_walker.h"
#include "common/mac/macho_utilities.h"

namespace MacFileUtilities {

using google_breakpad::MD5Init;
using google_breakpad::MD5Update;
using google_breakpad::MD5Final;

MachoID::MachoID(const char* path)
   : memory_(0),
     memory_size_(0),
     crc_(0), 
     md5_context_(), 
     update_function_(NULL) {
  snprintf(path_, sizeof(path_), "%s", path);
}

MachoID::MachoID(const char* path, void* memory, size_t size)
   : memory_(memory),
     memory_size_(size),
     crc_(0), 
     md5_context_(), 
     update_function_(NULL) {
  snprintf(path_, sizeof(path_), "%s", path);
}

MachoID::~MachoID() {
}

// The CRC info is from http://en.wikipedia.org/wiki/Adler-32
// With optimizations from http://www.zlib.net/

// The largest prime smaller than 65536
#define MOD_ADLER 65521
// MAX_BLOCK is the largest n such that 255n(n+1)/2 + (n+1)(MAX_BLOCK-1) <= 2^32-1
#define MAX_BLOCK 5552

void MachoID::UpdateCRC(unsigned char* bytes, size_t size) {
// Unrolled loops for summing
#define DO1(buf,i)  {sum1 += (buf)[i]; sum2 += sum1;}
#define DO2(buf,i)  DO1(buf,i); DO1(buf,i+1);
#define DO4(buf,i)  DO2(buf,i); DO2(buf,i+2);
#define DO8(buf,i)  DO4(buf,i); DO4(buf,i+4);
#define DO16(buf)   DO8(buf,0); DO8(buf,8);
  // Split up the crc
  uint32_t sum1 = crc_ & 0xFFFF;
  uint32_t sum2 = (crc_ >> 16) & 0xFFFF;

  // Do large blocks
  while (size >= MAX_BLOCK) {
    size -= MAX_BLOCK;
    int block_count = MAX_BLOCK / 16;
    do {
      DO16(bytes);
      bytes += 16;
    } while (--block_count);
    sum1 %= MOD_ADLER;
    sum2 %= MOD_ADLER;
  }

  // Do remaining bytes
  if (size) {
    while (size >= 16) {
      size -= 16;
      DO16(bytes);
      bytes += 16;
    }
    while (size--) {
      sum1 += *bytes++;
      sum2 += sum1;
    }
    sum1 %= MOD_ADLER;
    sum2 %= MOD_ADLER;
    crc_ = (sum2 << 16) | sum1;
  }
}

void MachoID::UpdateMD5(unsigned char* bytes, size_t size) {
  MD5Update(&md5_context_, bytes, static_cast<unsigned>(size));
}

void MachoID::Update(MachoWalker* walker, off_t offset, size_t size) {
  if (!update_function_ || !size)
    return;

  // Read up to 4k bytes at a time
  unsigned char buffer[4096];
  size_t buffer_size;
  off_t file_offset = offset;
  while (size > 0) {
    if (size > sizeof(buffer)) {
      buffer_size = sizeof(buffer);
      size -= buffer_size;
    } else {
      buffer_size = size;
      size = 0;
    }

    if (!walker->ReadBytes(buffer, buffer_size, file_offset))
      return;

    (this->*update_function_)(buffer, buffer_size);
    file_offset += buffer_size;
  }
}

bool MachoID::UUIDCommand(cpu_type_t cpu_type,
                          cpu_subtype_t cpu_subtype,
                          unsigned char bytes[16]) {
  struct breakpad_uuid_command uuid_cmd;
  uuid_cmd.cmd = 0;
  if (!WalkHeader(cpu_type, cpu_subtype, UUIDWalkerCB, &uuid_cmd))
    return false;

  // If we found the command, we'll have initialized the uuid_command
  // structure
  if (uuid_cmd.cmd == LC_UUID) {
    memcpy(bytes, uuid_cmd.uuid, sizeof(uuid_cmd.uuid));
    return true;
  }

  return false;
}

bool MachoID::IDCommand(cpu_type_t cpu_type,
                        cpu_subtype_t cpu_subtype,
                        unsigned char identifier[16]) {
  struct dylib_command dylib_cmd;
  dylib_cmd.cmd = 0;
  if (!WalkHeader(cpu_type, cpu_subtype, IDWalkerCB, &dylib_cmd))
    return false;

  // If we found the command, we'll have initialized the dylib_command
  // structure
  if (dylib_cmd.cmd == LC_ID_DYLIB) {
    // Take the hashed filename, version, and compatability version bytes
    // to form the first 12 bytes, pad the rest with zeros

    // create a crude hash of the filename to generate the first 4 bytes
    identifier[0] = 0;
    identifier[1] = 0;
    identifier[2] = 0;
    identifier[3] = 0;

    for (int j = 0, i = (int)strlen(path_)-1; i>=0 && path_[i]!='/'; ++j, --i) {
      identifier[j%4] += path_[i];
    }

    identifier[4] = (dylib_cmd.dylib.current_version >> 24) & 0xFF;
    identifier[5] = (dylib_cmd.dylib.current_version >> 16) & 0xFF;
    identifier[6] = (dylib_cmd.dylib.current_version >> 8) & 0xFF;
    identifier[7] = dylib_cmd.dylib.current_version & 0xFF;
    identifier[8] = (dylib_cmd.dylib.compatibility_version >> 24) & 0xFF;
    identifier[9] = (dylib_cmd.dylib.compatibility_version >> 16) & 0xFF;
    identifier[10] = (dylib_cmd.dylib.compatibility_version >> 8) & 0xFF;
    identifier[11] = dylib_cmd.dylib.compatibility_version & 0xFF;
    identifier[12] = (cpu_type >> 24) & 0xFF;
    identifier[13] = (cpu_type >> 16) & 0xFF;
    identifier[14] = (cpu_type >> 8) & 0xFF;
    identifier[15] = cpu_type & 0xFF;

    return true;
  }

  return false;
}

uint32_t MachoID::Adler32(cpu_type_t cpu_type, cpu_subtype_t cpu_subtype) {
  update_function_ = &MachoID::UpdateCRC;
  crc_ = 0;

  if (!WalkHeader(cpu_type, cpu_subtype, WalkerCB, this))
    return 0;

  return crc_;
}

bool MachoID::MD5(cpu_type_t cpu_type, cpu_subtype_t cpu_subtype, unsigned char identifier[16]) {
  update_function_ = &MachoID::UpdateMD5;

  MD5Init(&md5_context_);

  if (!WalkHeader(cpu_type, cpu_subtype, WalkerCB, this))
    return false;

  MD5Final(identifier, &md5_context_);
  return true;
}

bool MachoID::WalkHeader(cpu_type_t cpu_type,
                         cpu_subtype_t cpu_subtype,
                         MachoWalker::LoadCommandCallback callback,
                         void* context) {
  if (memory_) {
    MachoWalker walker(memory_, memory_size_, callback, context);
    return walker.WalkHeader(cpu_type, cpu_subtype);
  } else {
    MachoWalker walker(path_, callback, context);
    return walker.WalkHeader(cpu_type, cpu_subtype);
  }
}

// static
bool MachoID::WalkerCB(MachoWalker* walker, load_command* cmd, off_t offset,
                       bool swap, void* context) {
  MachoID* macho_id = (MachoID*)context;

  if (cmd->cmd == LC_SEGMENT) {
    struct segment_command seg;

    if (!walker->ReadBytes(&seg, sizeof(seg), offset))
      return false;

    if (swap)
      breakpad_swap_segment_command(&seg);

    struct mach_header_64 header;
    off_t header_offset;
    
    if (!walker->CurrentHeader(&header, &header_offset))
      return false;
        
    // Process segments that have sections:
    // (e.g., __TEXT, __DATA, __IMPORT, __OBJC)
    offset += sizeof(struct segment_command);
    struct section sec;
    for (unsigned long i = 0; i < seg.nsects; ++i) {
      if (!walker->ReadBytes(&sec, sizeof(sec), offset))
        return false;

      if (swap)
        breakpad_swap_section(&sec, 1);

      // sections of type S_ZEROFILL are "virtual" and contain no data
      // in the file itself
      if ((sec.flags & SECTION_TYPE) != S_ZEROFILL && sec.offset != 0)
        macho_id->Update(walker, header_offset + sec.offset, sec.size);

      offset += sizeof(struct section);
    }
  } else if (cmd->cmd == LC_SEGMENT_64) {
    struct segment_command_64 seg64;

    if (!walker->ReadBytes(&seg64, sizeof(seg64), offset))
      return false;

    if (swap)
      breakpad_swap_segment_command_64(&seg64);

    struct mach_header_64 header;
    off_t header_offset;
    
    if (!walker->CurrentHeader(&header, &header_offset))
      return false;
    
    // Process segments that have sections:
    // (e.g., __TEXT, __DATA, __IMPORT, __OBJC)
    offset += sizeof(struct segment_command_64);
    struct section_64 sec64;
    for (unsigned long i = 0; i < seg64.nsects; ++i) {
      if (!walker->ReadBytes(&sec64, sizeof(sec64), offset))
        return false;

      if (swap)
        breakpad_swap_section_64(&sec64, 1);

      // sections of type S_ZEROFILL are "virtual" and contain no data
      // in the file itself
      if ((sec64.flags & SECTION_TYPE) != S_ZEROFILL && sec64.offset != 0)
        macho_id->Update(walker, 
                         header_offset + sec64.offset, 
                         (size_t)sec64.size);

      offset += sizeof(struct section_64);
    }
  }

  // Continue processing
  return true;
}

// static
bool MachoID::UUIDWalkerCB(MachoWalker* walker, load_command* cmd, off_t offset,
                           bool swap, void* context) {
  if (cmd->cmd == LC_UUID) {
    struct breakpad_uuid_command* uuid_cmd =
      (struct breakpad_uuid_command*)context;

    if (!walker->ReadBytes(uuid_cmd, sizeof(struct breakpad_uuid_command),
                           offset))
      return false;

    if (swap)
      breakpad_swap_uuid_command(uuid_cmd);

    return false;
  }

  // Continue processing
  return true;
}

// static
bool MachoID::IDWalkerCB(MachoWalker* walker, load_command* cmd, off_t offset,
                         bool swap, void* context) {
  if (cmd->cmd == LC_ID_DYLIB) {
    struct dylib_command* dylib_cmd = (struct dylib_command*)context;

    if (!walker->ReadBytes(dylib_cmd, sizeof(struct dylib_command), offset))
      return false;

    if (swap)
      breakpad_swap_dylib_command(dylib_cmd);

    return false;
  }

  // Continue processing
  return true;
}

}  // namespace MacFileUtilities
