// Copyright 2009 Google LLC
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

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_MINIDUMP_WRITER_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_MINIDUMP_WRITER_H_

#include <stdint.h>
#include <sys/types.h>
#include <sys/ucontext.h>
#include <unistd.h>

#include <list>
#include <type_traits>
#include <utility>

#include "client/linux/minidump_writer/linux_dumper.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

class ExceptionHandler;

#if defined(__aarch64__)
typedef struct fpsimd_context fpstate_t;
#elif !defined(__ARM_EABI__) && !defined(__mips__)
typedef std::remove_pointer<fpregset_t>::type fpstate_t;
#endif

// These entries store a list of memory regions that the client wants included
// in the minidump.
struct AppMemory {
  void* ptr;
  size_t length;

  bool operator==(const struct AppMemory& other) const {
    return ptr == other.ptr;
  }

  bool operator==(const void* other) const {
    return ptr == other;
  }
};
typedef std::list<AppMemory> AppMemoryList;

// Writes a minidump to the filesystem. These functions do not malloc nor use
// libc functions which may. Thus, it can be used in contexts where the state
// of the heap may be corrupt.
//   minidump_path: the path to the file to write to. This is opened O_EXCL and
//     fails open fails.
//   crashing_process: the pid of the crashing process. This must be trusted.
//   blob: a blob of data from the crashing process. See exception_handler.h
//   blob_size: the length of |blob|, in bytes
//
// Returns true iff successful.
bool WriteMinidump(const char* minidump_path, pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   bool skip_stacks_if_mapping_unreferenced = false,
                   uintptr_t principal_mapping_address = 0,
                   bool sanitize_stacks = false);
// Same as above but takes an open file descriptor instead of a path.
bool WriteMinidump(int minidump_fd, pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   bool skip_stacks_if_mapping_unreferenced = false,
                   uintptr_t principal_mapping_address = 0,
                   bool sanitize_stacks = false);

// Alternate form of WriteMinidump() that works with processes that
// are not expected to have crashed.  If |process_blamed_thread| is
// meaningful, it will be the one from which a crash signature is
// extracted.  It is not expected that this function will be called
// from a compromised context, but it is safe to do so.
bool WriteMinidump(const char* minidump_path, pid_t process,
                   pid_t process_blamed_thread);

// These overloads also allow passing a list of known mappings and
// a list of additional memory regions to be included in the minidump.
bool WriteMinidump(const char* minidump_path, pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   const MappingList& mappings,
                   const AppMemoryList& appdata,
                   bool skip_stacks_if_mapping_unreferenced = false,
                   uintptr_t principal_mapping_address = 0,
                   bool sanitize_stacks = false);
bool WriteMinidump(int minidump_fd, pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   const MappingList& mappings,
                   const AppMemoryList& appdata,
                   bool skip_stacks_if_mapping_unreferenced = false,
                   uintptr_t principal_mapping_address = 0,
                   bool sanitize_stacks = false);

// These overloads also allow passing a file size limit for the minidump.
bool WriteMinidump(const char* minidump_path, off_t minidump_size_limit,
                   pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   const MappingList& mappings,
                   const AppMemoryList& appdata,
                   bool skip_stacks_if_mapping_unreferenced = false,
                   uintptr_t principal_mapping_address = 0,
                   bool sanitize_stacks = false);
bool WriteMinidump(int minidump_fd, off_t minidump_size_limit,
                   pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   const MappingList& mappings,
                   const AppMemoryList& appdata,
                   bool skip_stacks_if_mapping_unreferenced = false,
                   uintptr_t principal_mapping_address = 0,
                   bool sanitize_stacks = false);

bool WriteMinidump(const char* filename,
                   const MappingList& mappings,
                   const AppMemoryList& appdata,
                   LinuxDumper* dumper);

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_MINIDUMP_WRITER_MINIDUMP_WRITER_H_
