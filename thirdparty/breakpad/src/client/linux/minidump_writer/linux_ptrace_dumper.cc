// Copyright 2012 Google LLC
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

// linux_ptrace_dumper.cc: Implement google_breakpad::LinuxPtraceDumper.
// See linux_ptrace_dumper.h for detals.
// This class was originally splitted from google_breakpad::LinuxDumper.

// This code deals with the mechanics of getting information about a crashed
// process. Since this code may run in a compromised address space, the same
// rules apply as detailed at the top of minidump_writer.h: no libc calls and
// use the alternative allocator.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "client/linux/minidump_writer/linux_ptrace_dumper.h"

#include <asm/ptrace.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/uio.h>
#include <sys/wait.h>

#if defined(__i386)
#include <cpuid.h>
#endif

#include "client/linux/minidump_writer/directory_reader.h"
#include "client/linux/minidump_writer/line_reader.h"
#include "common/linux/eintr_wrapper.h"
#include "common/linux/linux_libc_support.h"
#include "third_party/lss/linux_syscall_support.h"

#if defined(__arm__)
/*
 * https://elixir.bootlin.com/linux/v6.8-rc2/source/arch/arm/include/asm/user.h#L81
 * User specific VFP registers. If only VFPv2 is present, registers 16 to 31
 * are ignored by the ptrace system call and the signal handler.
 */
typedef struct {
  unsigned long long fpregs[32];
  unsigned long fpscr;
// Kernel just appends fpscr to the copy of fpregs, so we need to force
// compiler to build the same layout.
} __attribute__((packed, aligned(4))) user_vfp_t;
#endif  // defined(__arm__)

// Suspends a thread by attaching to it.
static bool SuspendThread(pid_t pid) {
  // This may fail if the thread has just died or debugged.
  errno = 0;
  if (sys_ptrace(PTRACE_ATTACH, pid, NULL, NULL) != 0 &&
      errno != 0) {
    return false;
  }
  while (true) {
    int status;
    int r = HANDLE_EINTR(sys_waitpid(pid, &status, __WALL));
    if (r < 0) {
      sys_ptrace(PTRACE_DETACH, pid, NULL, NULL);
      return false;
    }

    if (!WIFSTOPPED(status))
      return false;

    // Any signal will stop the thread, make sure it is SIGSTOP. Otherwise, this
    // signal will be delivered after PTRACE_DETACH, and the thread will enter
    // the "T (stopped)" state.
    if (WSTOPSIG(status) == SIGSTOP)
      break;

    // Signals other than SIGSTOP that are received need to be reinjected,
    // or they will otherwise get lost.
    r = sys_ptrace(PTRACE_CONT, pid, NULL,
                   reinterpret_cast<void*>(WSTOPSIG(status)));
    if (r < 0)
      return false;
  }
#if defined(__i386) || defined(__x86_64)
  // On x86, the stack pointer is NULL or -1, when executing trusted code in
  // the seccomp sandbox. Not only does this cause difficulties down the line
  // when trying to dump the thread's stack, it also results in the minidumps
  // containing information about the trusted threads. This information is
  // generally completely meaningless and just pollutes the minidumps.
  // We thus test the stack pointer and exclude any threads that are part of
  // the seccomp sandbox's trusted code.
  user_regs_struct regs;
  if (sys_ptrace(PTRACE_GETREGS, pid, NULL, &regs) == -1 ||
#if defined(__i386)
      !regs.esp
#elif defined(__x86_64)
      !regs.rsp
#endif
      ) {
    sys_ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return false;
  }
#endif
  return true;
}

// Resumes a thread by detaching from it.
static bool ResumeThread(pid_t pid) {
  return sys_ptrace(PTRACE_DETACH, pid, NULL, NULL) >= 0;
}

namespace google_breakpad {

LinuxPtraceDumper::LinuxPtraceDumper(pid_t pid)
    : LinuxDumper(pid),
      threads_suspended_(false) {
}

bool LinuxPtraceDumper::BuildProcPath(char* path, pid_t pid,
                                      const char* node) const {
  if (!path || !node || pid <= 0)
    return false;

  size_t node_len = my_strlen(node);
  if (node_len == 0)
    return false;

  const unsigned pid_len = my_uint_len(pid);
  const size_t total_length = 6 + pid_len + 1 + node_len;
  if (total_length >= NAME_MAX)
    return false;

  my_memcpy(path, "/proc/", 6);
  my_uitos(path + 6, pid, pid_len);
  path[6 + pid_len] = '/';
  my_memcpy(path + 6 + pid_len + 1, node, node_len);
  path[total_length] = '\0';
  return true;
}

bool LinuxPtraceDumper::CopyFromProcess(void* dest, pid_t child,
                                        const void* src, size_t length) {
  unsigned long tmp = 55;
  size_t done = 0;
  static const size_t word_size = sizeof(tmp);
  uint8_t* const local = (uint8_t*) dest;
  uint8_t* const remote = (uint8_t*) src;

  while (done < length) {
    const size_t l = (length - done > word_size) ? word_size : (length - done);
    if (sys_ptrace(PTRACE_PEEKDATA, child, remote + done, &tmp) == -1) {
      tmp = 0;
    }
    my_memcpy(local + done, &tmp, l);
    done += l;
  }
  return true;
}

// This read VFP registers via either PTRACE_GETREGSET or PTRACE_GETREGS
#if defined(__arm__)
static bool ReadVFPRegistersArm32(pid_t tid, struct iovec* io) {
#ifdef PTRACE_GETREGSET
  if (sys_ptrace(PTRACE_GETREGSET, tid, reinterpret_cast<void*>(NT_ARM_VFP),
                 io) == 0 && io->iov_len == sizeof(user_vfp_t)) {
    return true;
  }
#endif  // PTRACE_GETREGSET
#ifdef PTRACE_GETVFPREGS
  if (sys_ptrace(PTRACE_GETVFPREGS, tid, nullptr, io->iov_base) == 0) {
    return true;
  }
#endif  // PTRACE_GETVFPREGS
  return false;
}
#endif  // defined(__arm__)

bool LinuxPtraceDumper::ReadRegisterSet(ThreadInfo* info, pid_t tid)
{
#ifdef PTRACE_GETREGSET
  struct iovec io;
  info->GetGeneralPurposeRegisters(&io.iov_base, &io.iov_len);
  if (sys_ptrace(PTRACE_GETREGSET, tid, (void*)NT_PRSTATUS, (void*)&io) == -1) {
    return false;
  }

  info->GetFloatingPointRegisters(&io.iov_base, &io.iov_len);
  if (sys_ptrace(PTRACE_GETREGSET, tid, (void*)NT_FPREGSET, (void*)&io) == -1) {
  // We are going to check if we can read VFP registers on ARM32.
  // Currently breakpad does not support VFP registers to be a part of minidump,
  // so this is only to confirm that we can actually read FP registers.
  // That is needed to prevent a false-positive minidumps failures with ARM32
  // binaries running on top of ARM64 Linux kernels.
#if defined(__arm__)
    switch (errno) {
      case EIO:
      case EINVAL:
        user_vfp_t vfp;
        struct iovec io;
        io.iov_base = &vfp;
        io.iov_len = sizeof(vfp);
        return ReadVFPRegistersArm32(tid, &io);
      default:
        return false;
    }
#endif  // defined(__arm__)
  }
  return true;
#else
  return false;
#endif
}

bool LinuxPtraceDumper::ReadRegisters(ThreadInfo* info, pid_t tid) {
#ifdef PTRACE_GETREGS
  void* gp_addr;
  info->GetGeneralPurposeRegisters(&gp_addr, NULL);
  if (sys_ptrace(PTRACE_GETREGS, tid, NULL, gp_addr) == -1) {
    return false;
  }

  // When running on arm processors the binary may be built with softfp or
  // hardfp. If built with softfp we have no hardware registers to read from,
  // so the following read will always fail. gcc defines __SOFTFP__ macro,
  // clang13 does not do so. see: https://reviews.llvm.org/D135680.
  // If you are using clang and the macro is NOT defined, please include the
  // macro define for applicable targets.
#if !defined(__SOFTFP__)
#if !(defined(__ANDROID__) && defined(__ARM_EABI__))
  // When running an arm build on an arm64 device, attempting to get the
  // floating point registers fails. On Android, the floating point registers
  // aren't written to the cpu context anyway, so just don't get them here.
  // See http://crbug.com/508324
  void* fp_addr;
  info->GetFloatingPointRegisters(&fp_addr, NULL);
  if (sys_ptrace(PTRACE_GETFPREGS, tid, NULL, fp_addr) == -1) {
  // We are going to check if we can read VFP registers on ARM32.
  // Currently breakpad does not support VFP registers to be a part of minidump,
  // so this is only to confirm that we can actually read FP registers.
  // That is needed to prevent a false-positive minidumps failures with ARM32
  // binaries running on top of ARM64 Linux kernels.
#if defined(__arm__)
    switch (errno) {
      case EIO:
      case EINVAL:
        user_vfp_t vfp;
        struct iovec io;
        io.iov_base = &vfp;
        io.iov_len = sizeof(vfp);
        return ReadVFPRegistersArm32(tid, &io);
      default:
        return false;
    }
#endif  // defined(__arm__)
  }
#endif  // !(defined(__ANDROID__) && defined(__ARM_EABI__))
#endif  // !defined(__SOFTFP__)
  return true;
#else  // PTRACE_GETREGS
  return false;
#endif
}

// Read thread info from /proc/$pid/status.
// Fill out the |tgid|, |ppid| and |pid| members of |info|. If unavailable,
// these members are set to -1. Returns true iff all three members are
// available.
bool LinuxPtraceDumper::GetThreadInfoByIndex(size_t index, ThreadInfo* info) {
  if (index >= threads_.size())
    return false;

  pid_t tid = threads_[index];

  assert(info != NULL);
  char status_path[NAME_MAX];
  if (!BuildProcPath(status_path, tid, "status"))
    return false;

  const int fd = sys_open(status_path, O_RDONLY, 0);
  if (fd < 0)
    return false;

  LineReader* const line_reader = new(allocator_) LineReader(fd);
  const char* line;
  unsigned line_len;

  info->ppid = info->tgid = -1;

  while (line_reader->GetNextLine(&line, &line_len)) {
    if (my_strncmp("Tgid:\t", line, 6) == 0) {
      my_strtoui(&info->tgid, line + 6);
    } else if (my_strncmp("PPid:\t", line, 6) == 0) {
      my_strtoui(&info->ppid, line + 6);
    }

    line_reader->PopLine(line_len);
  }
  sys_close(fd);

  if (info->ppid == -1 || info->tgid == -1)
    return false;

  if (!ReadRegisterSet(info, tid)) {
    if (!ReadRegisters(info, tid)) {
      return false;
    }
  }

#if defined(__i386)
#if !defined(bit_FXSAVE)  // e.g. Clang
#define bit_FXSAVE bit_FXSR
#endif
  // Detect if the CPU supports the FXSAVE/FXRSTOR instructions
  int eax, ebx, ecx, edx;
  __cpuid(1, eax, ebx, ecx, edx);
  if (edx & bit_FXSAVE) {
    if (sys_ptrace(PTRACE_GETFPXREGS, tid, NULL, &info->fpxregs) == -1) {
      return false;
    }
  } else {
    memset(&info->fpxregs, 0, sizeof(info->fpxregs));
  }
#endif  // defined(__i386)

#if defined(__i386) || defined(__x86_64)
  for (unsigned i = 0; i < ThreadInfo::kNumDebugRegisters; ++i) {
    if (sys_ptrace(
        PTRACE_PEEKUSER, tid,
        reinterpret_cast<void*> (offsetof(struct user,
                                          u_debugreg[0]) + i *
                                 sizeof(debugreg_t)),
        &info->dregs[i]) == -1) {
      return false;
    }
  }
#endif

#if defined(__mips__)
  sys_ptrace(PTRACE_PEEKUSER, tid,
             reinterpret_cast<void*>(PC), &info->mcontext.pc);
  sys_ptrace(PTRACE_PEEKUSER, tid,
             reinterpret_cast<void*>(DSP_BASE), &info->mcontext.hi1);
  sys_ptrace(PTRACE_PEEKUSER, tid,
             reinterpret_cast<void*>(DSP_BASE + 1), &info->mcontext.lo1);
  sys_ptrace(PTRACE_PEEKUSER, tid,
             reinterpret_cast<void*>(DSP_BASE + 2), &info->mcontext.hi2);
  sys_ptrace(PTRACE_PEEKUSER, tid,
             reinterpret_cast<void*>(DSP_BASE + 3), &info->mcontext.lo2);
  sys_ptrace(PTRACE_PEEKUSER, tid,
             reinterpret_cast<void*>(DSP_BASE + 4), &info->mcontext.hi3);
  sys_ptrace(PTRACE_PEEKUSER, tid,
             reinterpret_cast<void*>(DSP_BASE + 5), &info->mcontext.lo3);
  sys_ptrace(PTRACE_PEEKUSER, tid,
             reinterpret_cast<void*>(DSP_CONTROL), &info->mcontext.dsp);
#endif

  const uint8_t* stack_pointer;
#if defined(__i386)
  my_memcpy(&stack_pointer, &info->regs.esp, sizeof(info->regs.esp));
#elif defined(__x86_64)
  my_memcpy(&stack_pointer, &info->regs.rsp, sizeof(info->regs.rsp));
#elif defined(__ARM_EABI__)
  my_memcpy(&stack_pointer, &info->regs.ARM_sp, sizeof(info->regs.ARM_sp));
#elif defined(__aarch64__)
  my_memcpy(&stack_pointer, &info->regs.sp, sizeof(info->regs.sp));
#elif defined(__mips__)
  stack_pointer =
      reinterpret_cast<uint8_t*>(info->mcontext.gregs[MD_CONTEXT_MIPS_REG_SP]);
#elif defined(__riscv)
  stack_pointer = reinterpret_cast<uint8_t*>(
      info->mcontext.__gregs[MD_CONTEXT_RISCV_REG_SP]);
#else
# error "This code hasn't been ported to your platform yet."
#endif
  info->stack_pointer = reinterpret_cast<uintptr_t>(stack_pointer);

  return true;
}

bool LinuxPtraceDumper::IsPostMortem() const {
  return false;
}

bool LinuxPtraceDumper::ThreadsSuspend() {
  if (threads_suspended_)
    return true;
  for (size_t i = 0; i < threads_.size(); ++i) {
    if (!SuspendThread(threads_[i])) {
      // If the thread either disappeared before we could attach to it, or if
      // it was part of the seccomp sandbox's trusted code, it is OK to
      // silently drop it from the minidump.
      if (i < threads_.size() - 1) {
        my_memmove(&threads_[i], &threads_[i + 1],
                   (threads_.size() - i - 1) * sizeof(threads_[i]));
      }
      threads_.resize(threads_.size() - 1);
      --i;
    }
  }
  threads_suspended_ = true;
  return threads_.size() > 0;
}

bool LinuxPtraceDumper::ThreadsResume() {
  if (!threads_suspended_)
    return false;
  bool good = true;
  for (size_t i = 0; i < threads_.size(); ++i)
    good &= ResumeThread(threads_[i]);
  threads_suspended_ = false;
  return good;
}

// Parse /proc/$pid/task to list all the threads of the process identified by
// pid.
bool LinuxPtraceDumper::EnumerateThreads() {
  char task_path[NAME_MAX];
  if (!BuildProcPath(task_path, pid_, "task"))
    return false;

  const int fd = sys_open(task_path, O_RDONLY | O_DIRECTORY, 0);
  if (fd < 0)
    return false;
  DirectoryReader* dir_reader = new(allocator_) DirectoryReader(fd);

  // The directory may contain duplicate entries which we filter by assuming
  // that they are consecutive.
  int last_tid = -1;
  const char* dent_name;
  while (dir_reader->GetNextEntry(&dent_name)) {
    if (my_strcmp(dent_name, ".") &&
        my_strcmp(dent_name, "..")) {
      int tid = 0;
      if (my_strtoui(&tid, dent_name) &&
          last_tid != tid) {
        last_tid = tid;
        threads_.push_back(tid);
      }
    }
    dir_reader->PopEntry();
  }

  sys_close(fd);
  return true;
}

}  // namespace google_breakpad
