/* Copyright (c) 2006, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

/* minidump_exception_linux.h: A definition of exception codes for
 * Linux
 *
 * (This is C99 source, please don't corrupt it with C++.)
 *
 * Author: Mark Mentovai
 * Split into its own file: Neal Sidhwaney */
 

#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_LINUX_H__
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_LINUX_H__

#include <stddef.h>

#include "google_breakpad/common/breakpad_types.h"


/* For (MDException).exception_code.  These values come from bits/signum.h.
 */
typedef enum {
  MD_EXCEPTION_CODE_LIN_SIGHUP = 1,      /* Hangup (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGINT = 2,      /* Interrupt (ANSI) */
  MD_EXCEPTION_CODE_LIN_SIGQUIT = 3,     /* Quit (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGILL = 4,      /* Illegal instruction (ANSI) */
  MD_EXCEPTION_CODE_LIN_SIGTRAP = 5,     /* Trace trap (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGABRT = 6,     /* Abort (ANSI) */
  MD_EXCEPTION_CODE_LIN_SIGBUS = 7,      /* BUS error (4.2 BSD) */
  MD_EXCEPTION_CODE_LIN_SIGFPE = 8,      /* Floating-point exception (ANSI) */
  MD_EXCEPTION_CODE_LIN_SIGKILL = 9,     /* Kill, unblockable (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGUSR1 = 10,    /* User-defined signal 1 (POSIX).  */
  MD_EXCEPTION_CODE_LIN_SIGSEGV = 11,    /* Segmentation violation (ANSI) */
  MD_EXCEPTION_CODE_LIN_SIGUSR2 = 12,    /* User-defined signal 2 (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGPIPE = 13,    /* Broken pipe (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGALRM = 14,    /* Alarm clock (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGTERM = 15,    /* Termination (ANSI) */
  MD_EXCEPTION_CODE_LIN_SIGSTKFLT = 16,  /* Stack faultd */
  MD_EXCEPTION_CODE_LIN_SIGCHLD = 17,    /* Child status has changed (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGCONT = 18,    /* Continue (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGSTOP = 19,    /* Stop, unblockable (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGTSTP = 20,    /* Keyboard stop (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGTTIN = 21,    /* Background read from tty (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGTTOU = 22,    /* Background write to tty (POSIX) */
  MD_EXCEPTION_CODE_LIN_SIGURG = 23,
    /* Urgent condition on socket (4.2 BSD) */
  MD_EXCEPTION_CODE_LIN_SIGXCPU = 24,    /* CPU limit exceeded (4.2 BSD) */
  MD_EXCEPTION_CODE_LIN_SIGXFSZ = 25,
    /* File size limit exceeded (4.2 BSD) */
  MD_EXCEPTION_CODE_LIN_SIGVTALRM = 26,  /* Virtual alarm clock (4.2 BSD) */
  MD_EXCEPTION_CODE_LIN_SIGPROF = 27,    /* Profiling alarm clock (4.2 BSD) */
  MD_EXCEPTION_CODE_LIN_SIGWINCH = 28,   /* Window size change (4.3 BSD, Sun) */
  MD_EXCEPTION_CODE_LIN_SIGIO = 29,      /* I/O now possible (4.2 BSD) */
  MD_EXCEPTION_CODE_LIN_SIGPWR = 30,     /* Power failure restart (System V) */
  MD_EXCEPTION_CODE_LIN_SIGSYS = 31,     /* Bad system call */
  MD_EXCEPTION_CODE_LIN_DUMP_REQUESTED = 0xFFFFFFFF /* No exception,
                                                       dump requested. */
} MDExceptionCodeLinux;

/* For (MDException).exception_flags.  These values come from
 * asm-generic/siginfo.h.
 */
typedef enum {
  /* SIGILL */
  MD_EXCEPTION_FLAG_LIN_ILL_ILLOPC = 1,
  MD_EXCEPTION_FLAG_LIN_ILL_ILLOPN = 2,
  MD_EXCEPTION_FLAG_LIN_ILL_ILLADR = 3,
  MD_EXCEPTION_FLAG_LIN_ILL_ILLTRP = 4,
  MD_EXCEPTION_FLAG_LIN_ILL_PRVOPC = 5,
  MD_EXCEPTION_FLAG_LIN_ILL_PRVREG = 6,
  MD_EXCEPTION_FLAG_LIN_ILL_COPROC = 7,
  MD_EXCEPTION_FLAG_LIN_ILL_BADSTK = 8,

  /* SIGFPE */
  MD_EXCEPTION_FLAG_LIN_FPE_INTDIV = 1,
  MD_EXCEPTION_FLAG_LIN_FPE_INTOVF = 2,
  MD_EXCEPTION_FLAG_LIN_FPE_FLTDIV = 3,
  MD_EXCEPTION_FLAG_LIN_FPE_FLTOVF = 4,
  MD_EXCEPTION_FLAG_LIN_FPE_FLTUND = 5,
  MD_EXCEPTION_FLAG_LIN_FPE_FLTRES = 6,
  MD_EXCEPTION_FLAG_LIN_FPE_FLTINV = 7,
  MD_EXCEPTION_FLAG_LIN_FPE_FLTSUB = 8,

  /* SIGSEGV */
  MD_EXCEPTION_FLAG_LIN_SEGV_MAPERR = 1,
  MD_EXCEPTION_FLAG_LIN_SEGV_ACCERR = 2,
  MD_EXCEPTION_FLAG_LIN_SEGV_BNDERR = 3,
  MD_EXCEPTION_FLAG_LIN_SEGV_PKUERR = 4,

  /* SIGBUS */
  MD_EXCEPTION_FLAG_LIN_BUS_ADRALN = 1,
  MD_EXCEPTION_FLAG_LIN_BUS_ADRERR = 2,
  MD_EXCEPTION_FLAG_LIN_BUS_OBJERR = 3,
  MD_EXCEPTION_FLAG_LIN_BUS_MCEERR_AR = 4,
  MD_EXCEPTION_FLAG_LIN_BUS_MCEERR_AO = 5,
} MDExceptionFlagLinux;

#endif  /* GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_LINUX_H__ */
