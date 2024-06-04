/* Copyright 2013 Google Inc.
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

/* minidump_format.h: A cross-platform reimplementation of minidump-related
 * portions of DbgHelp.h from the Windows Platform SDK.
 *
 * (This is C99 source, please don't corrupt it with C++.)
 *
 * This file contains the necessary definitions to read minidump files
 * produced on ARM.  These files may be read on any platform provided
 * that the alignments of these structures on the processing system are
 * identical to the alignments of these structures on the producing system.
 * For this reason, precise-sized types are used.  The structures defined
 * by this file have been laid out to minimize alignment problems by
 * ensuring that all members are aligned on their natural boundaries.
 * In some cases, tail-padding may be significant when different ABIs specify
 * different tail-padding behaviors.  To avoid problems when reading or
 * writing affected structures, MD_*_SIZE macros are provided where needed,
 * containing the useful size of the structures without padding.
 *
 * Structures that are defined by Microsoft to contain a zero-length array
 * are instead defined here to contain an array with one element, as
 * zero-length arrays are forbidden by standard C and C++.  In these cases,
 * *_minsize constants are provided to be used in place of sizeof.  For a
 * cleaner interface to these sizes when using C++, see minidump_size.h.
 *
 * These structures are also sufficient to populate minidump files.
 *
 * Because precise data type sizes are crucial for this implementation to
 * function properly and portably, a set of primitive types with known sizes
 * are used as the basis of each structure defined by this file.
 *
 * Author: Colin Blundell
 */

/*
 * ARM64 support
 */

#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_ARM64_H__
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_ARM64_H__

#include "google_breakpad/common/breakpad_types.h"

#define MD_FLOATINGSAVEAREA_ARM64_FPR_COUNT 32
#define MD_CONTEXT_ARM64_GPR_COUNT 33

typedef struct {
  /* 32 128-bit floating point registers, d0 .. d31. */
  uint128_struct regs[MD_FLOATINGSAVEAREA_ARM64_FPR_COUNT];

  uint32_t fpcr;       /* FPU control register */
  uint32_t fpsr;       /* FPU status register */
} MDFloatingSaveAreaARM64;

/* For (MDRawContextARM64).context_flags.  These values indicate the type of
 * context stored in the structure. */
#define MD_CONTEXT_ARM64 0x00400000
#define MD_CONTEXT_ARM64_CONTROL (MD_CONTEXT_ARM64 | 0x00000001)
#define MD_CONTEXT_ARM64_INTEGER (MD_CONTEXT_ARM64 | 0x00000002)
#define MD_CONTEXT_ARM64_FLOATING_POINT (MD_CONTEXT_ARM64 | 0x00000004)
#define MD_CONTEXT_ARM64_DEBUG (MD_CONTEXT_ARM64 | 0x00000008)
#define MD_CONTEXT_ARM64_FULL (MD_CONTEXT_ARM64_CONTROL | \
                               MD_CONTEXT_ARM64_INTEGER | \
                               MD_CONTEXT_ARM64_FLOATING_POINT)
#define MD_CONTEXT_ARM64_ALL (MD_CONTEXT_ARM64_FULL | MD_CONTEXT_ARM64_DEBUG)

typedef struct {
  /* Determines which fields of this struct are populated */
  uint32_t context_flags;

  /* CPSR (flags, basically): 32 bits:
        bit 31 - N (negative)
        bit 30 - Z (zero)
        bit 29 - C (carry)
        bit 28 - V (overflow)
        bit 27 - Q (saturation flag, sticky)
     All other fields -- ignore */
  uint32_t cpsr;

  /* 33 64-bit integer registers, x0 .. x31 + the PC
   * Note the following fixed uses:
   *   x29 is the frame pointer
   *   x30 is the link register
   *   x31 is the stack pointer
   *   The PC is effectively x32.
   */
  uint64_t iregs[MD_CONTEXT_ARM64_GPR_COUNT];

  /* The next field is included with MD_CONTEXT64_ARM_FLOATING_POINT */
  MDFloatingSaveAreaARM64 float_save;

  uint32_t bcr[8];
  uint64_t bvr[8];
  uint32_t wcr[2];
  uint64_t wvr[2];
} MDRawContextARM64;

typedef struct {
  uint32_t       fpsr;      /* FPU status register */
  uint32_t       fpcr;      /* FPU control register */

  /* 32 128-bit floating point registers, d0 .. d31. */
  uint128_struct regs[MD_FLOATINGSAVEAREA_ARM64_FPR_COUNT];
} MDFloatingSaveAreaARM64_Old;

/* Use the same 32-bit alignment when accessing this structure from 64-bit code
 * as is used natively in 32-bit code. */
#pragma pack(push, 4)

typedef struct {
  /* The next field determines the layout of the structure, and which parts
   * of it are populated
   */
  uint64_t      context_flags;

  /* 33 64-bit integer registers, x0 .. x31 + the PC
   * Note the following fixed uses:
   *   x29 is the frame pointer
   *   x30 is the link register
   *   x31 is the stack pointer
   *   The PC is effectively x32.
   */
  uint64_t     iregs[MD_CONTEXT_ARM64_GPR_COUNT];

  /* CPSR (flags, basically): 32 bits:
        bit 31 - N (negative)
        bit 30 - Z (zero)
        bit 29 - C (carry)
        bit 28 - V (overflow)
        bit 27 - Q (saturation flag, sticky)
     All other fields -- ignore */
  uint32_t    cpsr;

  /* The next field is included with MD_CONTEXT64_ARM_FLOATING_POINT */
  MDFloatingSaveAreaARM64_Old float_save;

} MDRawContextARM64_Old;

#pragma pack(pop)

/* Indices into iregs for registers with a dedicated or conventional
 * purpose.
 */
enum MDARM64RegisterNumbers {
  MD_CONTEXT_ARM64_REG_FP     = 29,
  MD_CONTEXT_ARM64_REG_LR     = 30,
  MD_CONTEXT_ARM64_REG_SP     = 31,
  MD_CONTEXT_ARM64_REG_PC     = 32
};

/* For (MDRawContextARM64_Old).context_flags.  These values indicate the type of
 * context stored in the structure. MD_CONTEXT_ARM64_OLD is Breakpad-defined.
 * This value was chosen to avoid likely conflicts with MD_CONTEXT_*
 * for other CPUs. */
#define MD_CONTEXT_ARM64_OLD                   0x80000000
#define MD_CONTEXT_ARM64_INTEGER_OLD           (MD_CONTEXT_ARM64_OLD | 0x00000002)
#define MD_CONTEXT_ARM64_FLOATING_POINT_OLD    (MD_CONTEXT_ARM64_OLD | 0x00000004)

#define MD_CONTEXT_ARM64_FULL_OLD              (MD_CONTEXT_ARM64_INTEGER_OLD | \
                                          MD_CONTEXT_ARM64_FLOATING_POINT_OLD)

#define MD_CONTEXT_ARM64_ALL_OLD               (MD_CONTEXT_ARM64_INTEGER_OLD | \
                                          MD_CONTEXT_ARM64_FLOATING_POINT_OLD)

#endif  /* GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_ARM64_H__ */
