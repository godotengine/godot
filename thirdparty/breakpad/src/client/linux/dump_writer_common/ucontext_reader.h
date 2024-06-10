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

#ifndef CLIENT_LINUX_DUMP_WRITER_COMMON_UCONTEXT_READER_H
#define CLIENT_LINUX_DUMP_WRITER_COMMON_UCONTEXT_READER_H

#include <sys/ucontext.h>
#include <sys/user.h>

#include "client/linux/dump_writer_common/raw_context_cpu.h"
#include "client/linux/minidump_writer/minidump_writer.h"
#include "common/memory_allocator.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

// Wraps platform-dependent implementations of accessors to ucontext_t structs.
struct UContextReader {
  static uintptr_t GetStackPointer(const ucontext_t* uc);

  static uintptr_t GetInstructionPointer(const ucontext_t* uc);

  // Juggle a arch-specific ucontext_t into a minidump format
  //   out: the minidump structure
  //   info: the collection of register structures.
#if defined(__i386__) || defined(__x86_64)
  static void FillCPUContext(RawContextCPU* out, const ucontext_t* uc,
                             const fpstate_t* fp);
#elif defined(__aarch64__)
  static void FillCPUContext(RawContextCPU* out, const ucontext_t* uc,
                             const struct fpsimd_context* fpregs);
#else
  static void FillCPUContext(RawContextCPU* out, const ucontext_t* uc);
#endif
};

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_DUMP_WRITER_COMMON_UCONTEXT_READER_H
