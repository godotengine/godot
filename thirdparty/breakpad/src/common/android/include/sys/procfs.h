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

#ifndef GOOGLE_BREAKPAD_COMMON_ANDROID_SYS_PROCFS_H
#define GOOGLE_BREAKPAD_COMMON_ANDROID_SYS_PROCFS_H

#ifdef __BIONIC_HAVE_SYS_PROCFS_H

#include_next <sys/procfs.h>

#else

#include <asm/ptrace.h>
#include <sys/cdefs.h>
#if defined (__mips__)
#include <sys/types.h>
#endif
#include <sys/user.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if defined(__x86_64__) || defined(__aarch64__)
typedef unsigned long long elf_greg_t;
#else
typedef unsigned long  elf_greg_t;
#endif

#ifdef __arm__
#define ELF_NGREG (sizeof(struct user_regs) / sizeof(elf_greg_t))
#elif defined(__aarch64__)
#define ELF_NGREG (sizeof(struct user_pt_regs) / sizeof(elf_greg_t))
#elif defined(__mips__)
#define ELF_NGREG 45
#else
#define ELF_NGREG (sizeof(struct user_regs_struct) / sizeof(elf_greg_t))
#endif

typedef elf_greg_t elf_gregset_t[ELF_NGREG];

struct elf_siginfo {
  int si_signo;
  int si_code;
  int si_errno;
};

struct elf_prstatus {
  struct elf_siginfo pr_info;
  short              pr_cursig;
  unsigned long      pr_sigpend;
  unsigned long      pr_sighold;
  pid_t              pr_pid;
  pid_t              pr_ppid;
  pid_t              pr_pgrp;
  pid_t              pd_sid;
  struct timeval     pr_utime;
  struct timeval     pr_stime;
  struct timeval     pr_cutime;
  struct timeval     pr_cstime;
  elf_gregset_t      pr_reg;
  int                pr_fpvalid;
};

#define ELF_PRARGSZ 80

struct elf_prpsinfo {
  char           pr_state;
  char           pr_sname;
  char           pr_zomb;
  char           pr_nice;
  unsigned long  pr_flags;
#ifdef __x86_64__
  unsigned int   pr_uid;
  unsigned int   pr_gid;
#elif defined(__mips__)
  __kernel_uid_t pr_uid;
  __kernel_gid_t pr_gid;
#else
  unsigned short pr_uid;
  unsigned short pr_gid;
#endif
  int pr_pid;
  int pr_ppid;
  int pr_pgrp;
  int pr_sid;
  char pr_fname[16];
  char pr_psargs[ELF_PRARGSZ];
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // __BIONIC_HAVE_SYS_PROCFS_H

#endif  // GOOGLE_BREAKPAD_COMMON_ANDROID_SYS_PROCFS_H
