/* Copyright 2006 Google LLC
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
 *     * Neither the name of Google LLC nor the names of its
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

/* minidump_exception_solaris.h: A definition of exception codes for
 * Solaris
 *
 * (This is C99 source, please don't corrupt it with C++.)
 *
 * Author: Mark Mentovai
 * Split into its own file: Neal Sidhwaney */


#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_SOLARIS_H__
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_SOLARIS_H__

#include <stddef.h>

#include "google_breakpad/common/breakpad_types.h"

/* For (MDException).exception_code.  These values come from sys/iso/signal_iso.h
 */
typedef enum {
  MD_EXCEPTION_CODE_SOL_SIGHUP = 1,      /* Hangup */
  MD_EXCEPTION_CODE_SOL_SIGINT = 2,      /* interrupt (rubout) */
  MD_EXCEPTION_CODE_SOL_SIGQUIT = 3,     /* quit (ASCII FS) */
  MD_EXCEPTION_CODE_SOL_SIGILL = 4,      /* illegal instruction (not reset when caught) */
  MD_EXCEPTION_CODE_SOL_SIGTRAP = 5,     /* trace trap (not reset when caught) */
  MD_EXCEPTION_CODE_SOL_SIGIOT = 6,      /* IOT instruction */
  MD_EXCEPTION_CODE_SOL_SIGABRT = 6,     /* used by abort, replace SIGIOT in the future */
  MD_EXCEPTION_CODE_SOL_SIGEMT = 7,      /* EMT instruction */
  MD_EXCEPTION_CODE_SOL_SIGFPE = 8,      /* floating point exception */
  MD_EXCEPTION_CODE_SOL_SIGKILL = 9,     /* kill (cannot be caught or ignored) */
  MD_EXCEPTION_CODE_SOL_SIGBUS = 10,     /* bus error */
  MD_EXCEPTION_CODE_SOL_SIGSEGV = 11,    /* segmentation violation */
  MD_EXCEPTION_CODE_SOL_SIGSYS = 12,     /* bad argument to system call */
  MD_EXCEPTION_CODE_SOL_SIGPIPE = 13,    /* write on a pipe with no one to read it */
  MD_EXCEPTION_CODE_SOL_SIGALRM = 14,    /* alarm clock */
  MD_EXCEPTION_CODE_SOL_SIGTERM = 15,    /* software termination signal from kill */
  MD_EXCEPTION_CODE_SOL_SIGUSR1 = 16,    /* user defined signal 1 */
  MD_EXCEPTION_CODE_SOL_SIGUSR2 = 17,    /* user defined signal 2 */
  MD_EXCEPTION_CODE_SOL_SIGCLD = 18,     /* child status change */
  MD_EXCEPTION_CODE_SOL_SIGCHLD = 18,    /* child status change alias (POSIX) */
  MD_EXCEPTION_CODE_SOL_SIGPWR = 19,     /* power-fail restart */
  MD_EXCEPTION_CODE_SOL_SIGWINCH = 20,   /* window size change */
  MD_EXCEPTION_CODE_SOL_SIGURG = 21,     /* urgent socket condition */
  MD_EXCEPTION_CODE_SOL_SIGPOLL = 22,    /* pollable event occurred */
  MD_EXCEPTION_CODE_SOL_SIGIO = 22,      /* socket I/O possible (SIGPOLL alias) */
  MD_EXCEPTION_CODE_SOL_SIGSTOP = 23,    /* stop (cannot be caught or ignored) */
  MD_EXCEPTION_CODE_SOL_SIGTSTP = 24,    /* user stop requested from tty */
  MD_EXCEPTION_CODE_SOL_SIGCONT = 25,    /* stopped process has been continued */
  MD_EXCEPTION_CODE_SOL_SIGTTIN = 26,    /* background tty read attempted */
  MD_EXCEPTION_CODE_SOL_SIGTTOU = 27,    /* background tty write attempted */
  MD_EXCEPTION_CODE_SOL_SIGVTALRM = 28,  /* virtual timer expired */
  MD_EXCEPTION_CODE_SOL_SIGPROF = 29,    /* profiling timer expired */
  MD_EXCEPTION_CODE_SOL_SIGXCPU = 30,    /* exceeded cpu limit */
  MD_EXCEPTION_CODE_SOL_SIGXFSZ = 31,    /* exceeded file size limit */
  MD_EXCEPTION_CODE_SOL_SIGWAITING = 32, /* reserved signal no longer used by threading code */
  MD_EXCEPTION_CODE_SOL_SIGLWP = 33,     /* reserved signal no longer used by threading code */
  MD_EXCEPTION_CODE_SOL_SIGFREEZE = 34,  /* special signal used by CPR */
  MD_EXCEPTION_CODE_SOL_SIGTHAW = 35,    /* special signal used by CPR */
  MD_EXCEPTION_CODE_SOL_SIGCANCEL = 36,  /* reserved signal for thread cancellation */
  MD_EXCEPTION_CODE_SOL_SIGLOST = 37,    /* resource lost (eg, record-lock lost) */
  MD_EXCEPTION_CODE_SOL_SIGXRES = 38,    /* resource control exceeded */
  MD_EXCEPTION_CODE_SOL_SIGJVM1 = 39,    /* reserved signal for Java Virtual Machine */
  MD_EXCEPTION_CODE_SOL_SIGJVM2 = 40     /* reserved signal for Java Virtual Machine */
} MDExceptionCodeSolaris;

#endif  /* GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_SOLARIS_H__ */
