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

#ifndef CLIENT_LINUX_DUMP_WRITER_COMMON_THREAD_INFO_H_
#define CLIENT_LINUX_DUMP_WRITER_COMMON_THREAD_INFO_H_

#include <sys/ucontext.h>
#include <sys/user.h>

#include "client/linux/dump_writer_common/raw_context_cpu.h"
#include "common/memory_allocator.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

#if defined(__i386) || defined(__x86_64)
typedef __typeof__(((struct user*) 0)->u_debugreg[0]) debugreg_t;
#endif

// We produce one of these structures for each thread in the crashed process.
struct ThreadInfo {
  pid_t tgid;   // thread group id
  pid_t ppid;   // parent process

  uintptr_t stack_pointer;  // thread stack pointer


#if defined(__i386) || defined(__x86_64)
  user_regs_struct regs;
  user_fpregs_struct fpregs;
  static const unsigned kNumDebugRegisters = 8;
  debugreg_t dregs[8];
#if defined(__i386)
  user_fpxregs_struct fpxregs;
#endif  // defined(__i386)

#elif defined(__ARM_EABI__)
  // Mimicking how strace does this(see syscall.c, search for GETREGS)
  struct user_regs regs;
  struct user_fpregs fpregs;
#elif defined(__aarch64__)
  // Use the structures defined in <sys/user.h>
  struct user_regs_struct regs;
  struct user_fpsimd_struct fpregs;
#elif defined(__mips__) || defined(__riscv)
  // Use the structure defined in <sys/ucontext.h>.
  mcontext_t mcontext;
#endif

  // Returns the instruction pointer (platform-dependent impl.).
  uintptr_t GetInstructionPointer() const;

  // Fills a RawContextCPU using the context in the ThreadInfo object.
  void FillCPUContext(RawContextCPU* out) const;

  // Returns the pointer and size of general purpose register area.
  void GetGeneralPurposeRegisters(void** gp_regs, size_t* size);

  // Returns the pointer and size of float point register area.
  void GetFloatingPointRegisters(void** fp_regs, size_t* size);
};

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_DUMP_WRITER_COMMON_THREAD_INFO_H_
