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

/* minidump_exception_mac.h: A definition of exception codes for Mac
 * OS X
 *
 * (This is C99 source, please don't corrupt it with C++.)
 *
 * Author: Mark Mentovai
 * Split into its own file: Neal Sidhwaney */


#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_MAC_H__
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_MAC_H__

#include <stddef.h>

#include "google_breakpad/common/breakpad_types.h"

/* For (MDException).exception_code.  Breakpad minidump extension for Mac OS X
 * support.  Based on Darwin/Mac OS X' mach/exception_types.h.  This is
 * what Mac OS X calls an "exception", not a "code". */
typedef enum {
  /* Exception code.  The high 16 bits of exception_code contains one of
   * these values. */
  MD_EXCEPTION_MAC_BAD_ACCESS      = 1,  /* code can be a kern_return_t */
      /* EXC_BAD_ACCESS */
  MD_EXCEPTION_MAC_BAD_INSTRUCTION = 2,  /* code is CPU-specific */
      /* EXC_BAD_INSTRUCTION */
  MD_EXCEPTION_MAC_ARITHMETIC      = 3,  /* code is CPU-specific */
      /* EXC_ARITHMETIC */
  MD_EXCEPTION_MAC_EMULATION       = 4,  /* code is CPU-specific */
      /* EXC_EMULATION */
  MD_EXCEPTION_MAC_SOFTWARE        = 5,
      /* EXC_SOFTWARE */
  MD_EXCEPTION_MAC_BREAKPOINT      = 6,  /* code is CPU-specific */
      /* EXC_BREAKPOINT */
  MD_EXCEPTION_MAC_SYSCALL         = 7,
      /* EXC_SYSCALL */
  MD_EXCEPTION_MAC_MACH_SYSCALL    = 8,
      /* EXC_MACH_SYSCALL */
  MD_EXCEPTION_MAC_RPC_ALERT       = 9,
      /* EXC_RESOURCE */
  MD_EXCEPTION_MAC_RESOURCE        = 11,
      /* EXC_GUARD */
  MD_EXCEPTION_MAC_GUARD           = 12,
      /* EXC_RPC_ALERT */
  MD_EXCEPTION_MAC_SIMULATED       = 0x43507378,
      /* Fake exception code used by Crashpad's SimulateCrash ('CPsx'). */
  MD_NS_EXCEPTION_SIMULATED       = 0x43506E78
      /* Fake exception code used by Crashpad's uncaught exceptions ('CPnx'). */
} MDExceptionMac;

/* For (MDException).exception_flags.  Breakpad minidump extension for Mac OS X
 * support.  Based on Darwin/Mac OS X' mach/ppc/exception.h and
 * mach/i386/exception.h.  This is what Mac OS X calls a "code". */
typedef enum {
  /* With MD_EXCEPTION_BAD_ACCESS.  These are relevant kern_return_t values
   * from mach/kern_return.h. */
  MD_EXCEPTION_CODE_MAC_INVALID_ADDRESS    =  1,
      /* KERN_INVALID_ADDRESS */
  MD_EXCEPTION_CODE_MAC_PROTECTION_FAILURE =  2,
      /* KERN_PROTECTION_FAILURE */
  MD_EXCEPTION_CODE_MAC_NO_ACCESS          =  8,
      /* KERN_NO_ACCESS */
  MD_EXCEPTION_CODE_MAC_MEMORY_FAILURE     =  9,
      /* KERN_MEMORY_FAILURE */
  MD_EXCEPTION_CODE_MAC_MEMORY_ERROR       = 10,
      /* KERN_MEMORY_ERROR */
  MD_EXCEPTION_CODE_MAC_CODESIGN_ERROR     = 50,
      /* KERN_CODESIGN_ERROR */

  /* With MD_EXCEPTION_SOFTWARE */
  MD_EXCEPTION_CODE_MAC_BAD_SYSCALL  = 0x00010000,  /* Mach SIGSYS */
  MD_EXCEPTION_CODE_MAC_BAD_PIPE     = 0x00010001,  /* Mach SIGPIPE */
  MD_EXCEPTION_CODE_MAC_ABORT        = 0x00010002,  /* Mach SIGABRT */
  /* Custom values */
  MD_EXCEPTION_CODE_MAC_NS_EXCEPTION = 0xDEADC0DE,  /* uncaught NSException */

  /* With MD_EXCEPTION_MAC_BAD_ACCESS on arm */
  MD_EXCEPTION_CODE_MAC_ARM_DA_ALIGN = 0x0101,  /* EXC_ARM_DA_ALIGN */
  MD_EXCEPTION_CODE_MAC_ARM_DA_DEBUG = 0x0102,  /* EXC_ARM_DA_DEBUG */

  /* With MD_EXCEPTION_MAC_BAD_INSTRUCTION on arm */
  MD_EXCEPTION_CODE_MAC_ARM_UNDEFINED = 1,  /* EXC_ARM_UNDEFINED */

  /* With MD_EXCEPTION_MAC_BREAKPOINT on arm */
  MD_EXCEPTION_CODE_MAC_ARM_BREAKPOINT = 1, /* EXC_ARM_BREAKPOINT */

  /* With MD_EXCEPTION_MAC_BAD_ACCESS on ppc */
  MD_EXCEPTION_CODE_MAC_PPC_VM_PROT_READ = 0x0101,
      /* EXC_PPC_VM_PROT_READ */
  MD_EXCEPTION_CODE_MAC_PPC_BADSPACE     = 0x0102,
      /* EXC_PPC_BADSPACE */
  MD_EXCEPTION_CODE_MAC_PPC_UNALIGNED    = 0x0103,
      /* EXC_PPC_UNALIGNED */

  /* With MD_EXCEPTION_MAC_BAD_INSTRUCTION on ppc */
  MD_EXCEPTION_CODE_MAC_PPC_INVALID_SYSCALL           = 1,
      /* EXC_PPC_INVALID_SYSCALL */
  MD_EXCEPTION_CODE_MAC_PPC_UNIMPLEMENTED_INSTRUCTION = 2,
      /* EXC_PPC_UNIPL_INST */
  MD_EXCEPTION_CODE_MAC_PPC_PRIVILEGED_INSTRUCTION    = 3,
      /* EXC_PPC_PRIVINST */
  MD_EXCEPTION_CODE_MAC_PPC_PRIVILEGED_REGISTER       = 4,
      /* EXC_PPC_PRIVREG */
  MD_EXCEPTION_CODE_MAC_PPC_TRACE                     = 5,
      /* EXC_PPC_TRACE */
  MD_EXCEPTION_CODE_MAC_PPC_PERFORMANCE_MONITOR       = 6,
      /* EXC_PPC_PERFMON */

  /* With MD_EXCEPTION_MAC_ARITHMETIC on ppc */
  MD_EXCEPTION_CODE_MAC_PPC_OVERFLOW           = 1,
      /* EXC_PPC_OVERFLOW */
  MD_EXCEPTION_CODE_MAC_PPC_ZERO_DIVIDE        = 2,
      /* EXC_PPC_ZERO_DIVIDE */
  MD_EXCEPTION_CODE_MAC_PPC_FLOAT_INEXACT      = 3,
      /* EXC_FLT_INEXACT */
  MD_EXCEPTION_CODE_MAC_PPC_FLOAT_ZERO_DIVIDE  = 4,
      /* EXC_PPC_FLT_ZERO_DIVIDE */
  MD_EXCEPTION_CODE_MAC_PPC_FLOAT_UNDERFLOW    = 5,
      /* EXC_PPC_FLT_UNDERFLOW */
  MD_EXCEPTION_CODE_MAC_PPC_FLOAT_OVERFLOW     = 6,
      /* EXC_PPC_FLT_OVERFLOW */
  MD_EXCEPTION_CODE_MAC_PPC_FLOAT_NOT_A_NUMBER = 7,
      /* EXC_PPC_FLT_NOT_A_NUMBER */

  /* With MD_EXCEPTION_MAC_EMULATION on ppc */
  MD_EXCEPTION_CODE_MAC_PPC_NO_EMULATION   = 8,
      /* EXC_PPC_NOEMULATION */
  MD_EXCEPTION_CODE_MAC_PPC_ALTIVEC_ASSIST = 9,
      /* EXC_PPC_ALTIVECASSIST */

  /* With MD_EXCEPTION_MAC_SOFTWARE on ppc */
  MD_EXCEPTION_CODE_MAC_PPC_TRAP    = 0x00000001,  /* EXC_PPC_TRAP */
  MD_EXCEPTION_CODE_MAC_PPC_MIGRATE = 0x00010100,  /* EXC_PPC_MIGRATE */

  /* With MD_EXCEPTION_MAC_BREAKPOINT on ppc */
  MD_EXCEPTION_CODE_MAC_PPC_BREAKPOINT = 1,  /* EXC_PPC_BREAKPOINT */

  /* With MD_EXCEPTION_MAC_BAD_INSTRUCTION on x86, see also x86 interrupt
   * values below. */
  MD_EXCEPTION_CODE_MAC_X86_INVALID_OPERATION = 1,  /* EXC_I386_INVOP */

  /* With MD_EXCEPTION_MAC_ARITHMETIC on x86 */
  MD_EXCEPTION_CODE_MAC_X86_DIV       = 1,  /* EXC_I386_DIV */
  MD_EXCEPTION_CODE_MAC_X86_INTO      = 2,  /* EXC_I386_INTO */
  MD_EXCEPTION_CODE_MAC_X86_NOEXT     = 3,  /* EXC_I386_NOEXT */
  MD_EXCEPTION_CODE_MAC_X86_EXTOVR    = 4,  /* EXC_I386_EXTOVR */
  MD_EXCEPTION_CODE_MAC_X86_EXTERR    = 5,  /* EXC_I386_EXTERR */
  MD_EXCEPTION_CODE_MAC_X86_EMERR     = 6,  /* EXC_I386_EMERR */
  MD_EXCEPTION_CODE_MAC_X86_BOUND     = 7,  /* EXC_I386_BOUND */
  MD_EXCEPTION_CODE_MAC_X86_SSEEXTERR = 8,  /* EXC_I386_SSEEXTERR */

  /* With MD_EXCEPTION_MAC_BREAKPOINT on x86 */
  MD_EXCEPTION_CODE_MAC_X86_SGL = 1,  /* EXC_I386_SGL */
  MD_EXCEPTION_CODE_MAC_X86_BPT = 2,  /* EXC_I386_BPT */

  /* With MD_EXCEPTION_MAC_BAD_INSTRUCTION on x86.  These are the raw
   * x86 interrupt codes.  Most of these are mapped to other Mach
   * exceptions and codes, are handled, or should not occur in user space.
   * A few of these will do occur with MD_EXCEPTION_MAC_BAD_INSTRUCTION. */
  /* EXC_I386_DIVERR    =  0: mapped to EXC_ARITHMETIC/EXC_I386_DIV */
  /* EXC_I386_SGLSTP    =  1: mapped to EXC_BREAKPOINT/EXC_I386_SGL */
  /* EXC_I386_NMIFLT    =  2: should not occur in user space */
  /* EXC_I386_BPTFLT    =  3: mapped to EXC_BREAKPOINT/EXC_I386_BPT */
  /* EXC_I386_INTOFLT   =  4: mapped to EXC_ARITHMETIC/EXC_I386_INTO */
  /* EXC_I386_BOUNDFLT  =  5: mapped to EXC_ARITHMETIC/EXC_I386_BOUND */
  /* EXC_I386_INVOPFLT  =  6: mapped to EXC_BAD_INSTRUCTION/EXC_I386_INVOP */
  /* EXC_I386_NOEXTFLT  =  7: should be handled by the kernel */
  /* EXC_I386_DBLFLT    =  8: should be handled (if possible) by the kernel */
  /* EXC_I386_EXTOVRFLT =  9: mapped to EXC_BAD_ACCESS/(PROT_READ|PROT_EXEC) */
  MD_EXCEPTION_CODE_MAC_X86_INVALID_TASK_STATE_SEGMENT = 10,
      /* EXC_INVTSSFLT */
  MD_EXCEPTION_CODE_MAC_X86_SEGMENT_NOT_PRESENT        = 11,
      /* EXC_SEGNPFLT */
  MD_EXCEPTION_CODE_MAC_X86_STACK_FAULT                = 12,
      /* EXC_STKFLT */
  MD_EXCEPTION_CODE_MAC_X86_GENERAL_PROTECTION_FAULT   = 13,
      /* EXC_GPFLT */
  /* EXC_I386_PGFLT     = 14: should not occur in user space */
  /* EXC_I386_EXTERRFLT = 16: mapped to EXC_ARITHMETIC/EXC_I386_EXTERR */
  MD_EXCEPTION_CODE_MAC_X86_ALIGNMENT_FAULT            = 17
      /* EXC_ALIGNFLT (for vector operations) */
  /* EXC_I386_ENOEXTFLT = 32: should be handled by the kernel */
  /* EXC_I386_ENDPERR   = 33: should not occur */
} MDExceptionCodeMac;

#endif  /* GOOGLE_BREAKPAD_COMMON_MINIDUMP_EXCEPTION_MAC_OSX_H__ */
