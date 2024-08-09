/* Copyright 2009 Google LLC
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
 * Author: Julian Seward
 */

/*
 * ARM support
 */

#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_ARM_H__
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_ARM_H__

#define MD_FLOATINGSAVEAREA_ARM_FPR_COUNT 32
#define MD_FLOATINGSAVEAREA_ARM_FPEXTRA_COUNT 8

/*
 * Note that these structures *do not* map directly to the CONTEXT
 * structure defined in WinNT.h in the Windows Mobile SDK. That structure
 * does not accomodate VFPv3, and I'm unsure if it was ever used in the
 * wild anyway, as Windows CE only seems to produce "cedumps" which
 * are not exactly minidumps.
 */
typedef struct {
  uint64_t      fpscr;      /* FPU status register */

  /* 32 64-bit floating point registers, d0 .. d31. */
  uint64_t      regs[MD_FLOATINGSAVEAREA_ARM_FPR_COUNT];

  /* Miscellaneous control words */
  uint32_t     extra[MD_FLOATINGSAVEAREA_ARM_FPEXTRA_COUNT];
} MDFloatingSaveAreaARM;

#define MD_CONTEXT_ARM_GPR_COUNT 16

typedef struct {
  /* The next field determines the layout of the structure, and which parts
   * of it are populated
   */
  uint32_t      context_flags;

  /* 16 32-bit integer registers, r0 .. r15
   * Note the following fixed uses:
   *   r13 is the stack pointer
   *   r14 is the link register
   *   r15 is the program counter
   */
  uint32_t     iregs[MD_CONTEXT_ARM_GPR_COUNT];

  /* CPSR (flags, basically): 32 bits:
        bit 31 - N (negative)
        bit 30 - Z (zero)
        bit 29 - C (carry)
        bit 28 - V (overflow)
        bit 27 - Q (saturation flag, sticky)
     All other fields -- ignore */
  uint32_t    cpsr;

  /* The next field is included with MD_CONTEXT_ARM_FLOATING_POINT */
  MDFloatingSaveAreaARM float_save;

} MDRawContextARM;

/* Indices into iregs for registers with a dedicated or conventional
 * purpose.
 */
enum MDARMRegisterNumbers {
  MD_CONTEXT_ARM_REG_IOS_FP = 7,
  MD_CONTEXT_ARM_REG_FP     = 11,
  MD_CONTEXT_ARM_REG_SP     = 13,
  MD_CONTEXT_ARM_REG_LR     = 14,
  MD_CONTEXT_ARM_REG_PC     = 15
};

/* For (MDRawContextARM).context_flags.  These values indicate the type of
 * context stored in the structure. */
/* CONTEXT_ARM from the Windows CE 5.0 SDK. This value isn't correct
 * because this bit can be used for flags. Presumably this value was
 * never actually used in minidumps, but only in "CEDumps" which
 * are a whole parallel minidump file format for Windows CE.
 * Therefore, Breakpad defines its own value for ARM CPUs.
 */
#define MD_CONTEXT_ARM_OLD               0x00000040
/* This value was chosen to avoid likely conflicts with MD_CONTEXT_*
 * for other CPUs. */
#define MD_CONTEXT_ARM                   0x40000000
#define MD_CONTEXT_ARM_INTEGER           (MD_CONTEXT_ARM | 0x00000002)
#define MD_CONTEXT_ARM_FLOATING_POINT    (MD_CONTEXT_ARM | 0x00000004)

#define MD_CONTEXT_ARM_FULL              (MD_CONTEXT_ARM_INTEGER | \
                                          MD_CONTEXT_ARM_FLOATING_POINT)

#define MD_CONTEXT_ARM_ALL               (MD_CONTEXT_ARM_INTEGER | \
                                          MD_CONTEXT_ARM_FLOATING_POINT)

#endif  /* GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_ARM_H__ */
