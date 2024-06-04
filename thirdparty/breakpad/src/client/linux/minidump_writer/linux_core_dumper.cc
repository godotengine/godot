// Copyright (c) 2012, Google Inc.
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

// linux_core_dumper.cc: Implement google_breakpad::LinuxCoreDumper.
// See linux_core_dumper.h for details.

#include "client/linux/minidump_writer/linux_core_dumper.h"

#include <asm/ptrace.h>
#include <assert.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <sys/procfs.h>
#if defined(__mips__) && defined(__ANDROID__)
// To get register definitions.
#include <asm/reg.h>
#endif

#include "common/linux/elf_gnu_compat.h"
#include "common/linux/linux_libc_support.h"

namespace google_breakpad {

LinuxCoreDumper::LinuxCoreDumper(pid_t pid,
                                 const char* core_path,
                                 const char* procfs_path,
                                 const char* root_prefix)
    : LinuxDumper(pid, root_prefix),
      core_path_(core_path),
      procfs_path_(procfs_path),
      thread_infos_(&allocator_, 8) {
  assert(core_path_);
}

bool LinuxCoreDumper::BuildProcPath(char* path, pid_t pid,
                                    const char* node) const {
  if (!path || !node)
    return false;

  size_t node_len = my_strlen(node);
  if (node_len == 0)
    return false;

  size_t procfs_path_len = my_strlen(procfs_path_);
  size_t total_length = procfs_path_len + 1 + node_len;
  if (total_length >= NAME_MAX)
    return false;

  memcpy(path, procfs_path_, procfs_path_len);
  path[procfs_path_len] = '/';
  memcpy(path + procfs_path_len + 1, node, node_len);
  path[total_length] = '\0';
  return true;
}

bool LinuxCoreDumper::CopyFromProcess(void* dest, pid_t child,
                                      const void* src, size_t length) {
  ElfCoreDump::Addr virtual_address = reinterpret_cast<ElfCoreDump::Addr>(src);
  // TODO(benchan): Investigate whether the data to be copied could span
  // across multiple segments in the core dump file. ElfCoreDump::CopyData
  // and this method do not handle that case yet.
  if (!core_.CopyData(dest, virtual_address, length)) {
    // If the data segment is not found in the core dump, fill the result
    // with marker characters.
    memset(dest, 0xab, length);
    return false;
  }
  return true;
}

bool LinuxCoreDumper::GetThreadInfoByIndex(size_t index, ThreadInfo* info) {
  if (index >= thread_infos_.size())
    return false;

  *info = thread_infos_[index];
  const uint8_t* stack_pointer;
#if defined(__i386)
  memcpy(&stack_pointer, &info->regs.esp, sizeof(info->regs.esp));
#elif defined(__x86_64)
  memcpy(&stack_pointer, &info->regs.rsp, sizeof(info->regs.rsp));
#elif defined(__ARM_EABI__)
  memcpy(&stack_pointer, &info->regs.ARM_sp, sizeof(info->regs.ARM_sp));
#elif defined(__aarch64__)
  memcpy(&stack_pointer, &info->regs.sp, sizeof(info->regs.sp));
#elif defined(__mips__)
  stack_pointer =
      reinterpret_cast<uint8_t*>(info->mcontext.gregs[MD_CONTEXT_MIPS_REG_SP]);
#else
#error "This code hasn't been ported to your platform yet."
#endif
  info->stack_pointer = reinterpret_cast<uintptr_t>(stack_pointer);
  return true;
}

bool LinuxCoreDumper::IsPostMortem() const {
  return true;
}

bool LinuxCoreDumper::ThreadsSuspend() {
  return true;
}

bool LinuxCoreDumper::ThreadsResume() {
  return true;
}

bool LinuxCoreDumper::EnumerateThreads() {
  if (!mapped_core_file_.Map(core_path_, 0)) {
    fprintf(stderr, "Could not map core dump file into memory\n");
    return false;
  }

  char proc_mem_path[NAME_MAX];
  if (BuildProcPath(proc_mem_path, pid_, "mem")) {
    int fd = open(proc_mem_path, O_RDONLY | O_LARGEFILE | O_CLOEXEC);
    if (fd != -1) {
      core_.SetProcMem(fd);
    } else {
      fprintf(stderr, "Cannot open %s (%s)\n", proc_mem_path, strerror(errno));
    }
  }

  core_.SetContent(mapped_core_file_.content());
  if (!core_.IsValid()) {
    fprintf(stderr, "Invalid core dump file\n");
    return false;
  }

  ElfCoreDump::Note note = core_.GetFirstNote();
  if (!note.IsValid()) {
    fprintf(stderr, "PT_NOTE section not found\n");
    return false;
  }

  bool first_thread = true;
  do {
    ElfCoreDump::Word type = note.GetType();
    MemoryRange name = note.GetName();
    MemoryRange description = note.GetDescription();

    if (type == 0 || name.IsEmpty() || description.IsEmpty()) {
      fprintf(stderr, "Could not found a valid PT_NOTE.\n");
      return false;
    }

    // Based on write_note_info() in linux/kernel/fs/binfmt_elf.c, notes are
    // ordered as follows (NT_PRXFPREG and NT_386_TLS are i386 specific):
    //   Thread           Name          Type
    //   -------------------------------------------------------------------
    //   1st thread       CORE          NT_PRSTATUS
    //   process-wide     CORE          NT_PRPSINFO
    //   process-wide     CORE          NT_SIGINFO
    //   process-wide     CORE          NT_AUXV
    //   1st thread       CORE          NT_FPREGSET
    //   1st thread       LINUX         NT_PRXFPREG
    //   1st thread       LINUX         NT_386_TLS
    //
    //   2nd thread       CORE          NT_PRSTATUS
    //   2nd thread       CORE          NT_FPREGSET
    //   2nd thread       LINUX         NT_PRXFPREG
    //   2nd thread       LINUX         NT_386_TLS
    //
    //   3rd thread       CORE          NT_PRSTATUS
    //   3rd thread       CORE          NT_FPREGSET
    //   3rd thread       LINUX         NT_PRXFPREG
    //   3rd thread       LINUX         NT_386_TLS
    //
    // The following code only works if notes are ordered as expected.
    switch (type) {
      case NT_PRSTATUS: {
        if (description.length() != sizeof(elf_prstatus)) {
          fprintf(stderr, "Found NT_PRSTATUS descriptor of unexpected size\n");
          return false;
        }

        const elf_prstatus* status =
            reinterpret_cast<const elf_prstatus*>(description.data());
        pid_t pid = status->pr_pid;
        ThreadInfo info;
        memset(&info, 0, sizeof(ThreadInfo));
        info.tgid = status->pr_pgrp;
        info.ppid = status->pr_ppid;
#if defined(__mips__)
#if defined(__ANDROID__)
        for (int i = EF_R0; i <= EF_R31; i++)
          info.mcontext.gregs[i - EF_R0] = status->pr_reg[i];
#else  // __ANDROID__
        for (int i = EF_REG0; i <= EF_REG31; i++)
          info.mcontext.gregs[i - EF_REG0] = status->pr_reg[i];
#endif  // __ANDROID__
        info.mcontext.mdlo = status->pr_reg[EF_LO];
        info.mcontext.mdhi = status->pr_reg[EF_HI];
        info.mcontext.pc = status->pr_reg[EF_CP0_EPC];
#else  // __mips__
        memcpy(&info.regs, status->pr_reg, sizeof(info.regs));
#endif  // __mips__
        if (first_thread) {
          crash_thread_ = pid;
          crash_signal_ = status->pr_info.si_signo;
          crash_signal_code_ = status->pr_info.si_code;
        }
        first_thread = false;
        threads_.push_back(pid);
        thread_infos_.push_back(info);
        break;
      }
      case NT_SIGINFO: {
        if (description.length() != sizeof(siginfo_t)) {
          fprintf(stderr, "Found NT_SIGINFO descriptor of unexpected size\n");
          return false;
        }

        const siginfo_t* info =
            reinterpret_cast<const siginfo_t*>(description.data());

        // Set crash_address when si_addr is valid for the signal.
        switch (info->si_signo) {
          case MD_EXCEPTION_CODE_LIN_SIGBUS:
          case MD_EXCEPTION_CODE_LIN_SIGFPE:
          case MD_EXCEPTION_CODE_LIN_SIGILL:
          case MD_EXCEPTION_CODE_LIN_SIGSEGV:
          case MD_EXCEPTION_CODE_LIN_SIGSYS:
          case MD_EXCEPTION_CODE_LIN_SIGTRAP:
            crash_address_ = reinterpret_cast<uintptr_t>(info->si_addr);
            break;
        }

        // Set crash_exception_info for common signals.  Since exception info is
        // unsigned, but some of these fields might be signed, we always cast.
        switch (info->si_signo) {
          case MD_EXCEPTION_CODE_LIN_SIGKILL:
            set_crash_exception_info({
              static_cast<uint64_t>(info->si_pid),
              static_cast<uint64_t>(info->si_uid),
            });
            break;
          case MD_EXCEPTION_CODE_LIN_SIGSYS:
#ifdef si_syscall
            set_crash_exception_info({
              static_cast<uint64_t>(info->si_syscall),
              static_cast<uint64_t>(info->si_arch),
            });
#endif
            break;
        }
        break;
      }
#if defined(__i386) || defined(__x86_64)
      case NT_FPREGSET: {
        if (thread_infos_.empty())
          return false;

        ThreadInfo* info = &thread_infos_.back();
        if (description.length() != sizeof(info->fpregs)) {
          fprintf(stderr, "Found NT_FPREGSET descriptor of unexpected size\n");
          return false;
        }

        memcpy(&info->fpregs, description.data(), sizeof(info->fpregs));
        break;
      }
#endif
#if defined(__i386)
      case NT_PRXFPREG: {
        if (thread_infos_.empty())
          return false;

        ThreadInfo* info = &thread_infos_.back();
        if (description.length() != sizeof(info->fpxregs)) {
          fprintf(stderr, "Found NT_PRXFPREG descriptor of unexpected size\n");
          return false;
        }

        memcpy(&info->fpxregs, description.data(), sizeof(info->fpxregs));
        break;
      }
#endif
    }
    note = note.GetNextNote();
  } while (note.IsValid());

  return true;
}

}  // namespace google_breakpad
