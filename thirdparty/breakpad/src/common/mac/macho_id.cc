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

// macho_id.cc: Functions to gather identifying information from a macho file
//
// See macho_id.h for documentation
//
// Author: Dan Waylonis


#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <fcntl.h>
#include <mach-o/loader.h>
#include <stdio.h>
#include <string.h>

#include "common/mac/macho_id.h"
#include "common/mac/macho_walker.h"
#include "common/mac/macho_utilities.h"

namespace MacFileUtilities {

using google_breakpad::MD5Init;
using google_breakpad::MD5Update;
using google_breakpad::MD5Final;

MachoID::MachoID(const char* path)
    : memory_(0), memory_size_(0), md5_context_(), update_function_(NULL) {
  snprintf(path_, sizeof(path_), "%s", path);
}

MachoID::MachoID(void* memory, size_t size)
    : path_(),
      memory_(memory),
      memory_size_(size),
      md5_context_(),
      update_function_(NULL) {}

MachoID::~MachoID() {}

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
}  // namespace MacFileUtilities
