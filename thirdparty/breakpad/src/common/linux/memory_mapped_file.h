// Copyright (c) 2011, Google Inc.
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

// memory_mapped_file.h: Define the google_breakpad::MemoryMappedFile
// class, which maps a file into memory for read-only access.

#ifndef COMMON_LINUX_MEMORY_MAPPED_FILE_H_
#define COMMON_LINUX_MEMORY_MAPPED_FILE_H_

#include <stddef.h>
#include "common/basictypes.h"
#include "common/memory_range.h"

namespace google_breakpad {

// A utility class for mapping a file into memory for read-only access of
// the file content. Its implementation avoids calling into libc functions
// by directly making system calls for open, close, mmap, and munmap.
class MemoryMappedFile {
 public:
  MemoryMappedFile();

  // Constructor that calls Map() to map a file at |path| into memory.
  // If Map() fails, the object behaves as if it is default constructed.
  MemoryMappedFile(const char* path, size_t offset);

  ~MemoryMappedFile();

  // Maps a file at |path| into memory, which can then be accessed via
  // content() as a MemoryRange object or via data(), and returns true on
  // success. Mapping an empty file will succeed but with data() and size()
  // returning NULL and 0, respectively. An existing mapping is unmapped
  // before a new mapping is created.
  bool Map(const char* path, size_t offset);

  // Unmaps the memory for the mapped file. It's a no-op if no file is
  // mapped.
  void Unmap();

  // Returns a MemoryRange object that covers the memory for the mapped
  // file. The MemoryRange object is empty if no file is mapped.
  const MemoryRange& content() const { return content_; }

  // Returns a pointer to the beginning of the memory for the mapped file.
  // or NULL if no file is mapped or the mapped file is empty.
  const void* data() const { return content_.data(); }

  // Returns the size in bytes of the mapped file, or zero if no file
  // is mapped.
  size_t size() const { return content_.length(); }

 private:
  // Mapped file content as a MemoryRange object.
  MemoryRange content_;

  DISALLOW_COPY_AND_ASSIGN(MemoryMappedFile);
};

}  // namespace google_breakpad

#endif  // COMMON_LINUX_MEMORY_MAPPED_FILE_H_
