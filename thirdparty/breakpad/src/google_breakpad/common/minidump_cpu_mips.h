/* Copyright 2013 Google LLC
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
 * produced on MIPS.  These files may be read on any platform provided
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
 * Author: Chris Dearman
 */

/*
 * MIPS support
 */

#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_MIPS_H__
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_MIPS_H__

#define MD_CONTEXT_MIPS_GPR_COUNT 32
#define MD_FLOATINGSAVEAREA_MIPS_FPR_COUNT 32
#define MD_CONTEXT_MIPS_DSP_COUNT 3

/*
 * Note that these structures *do not* map directly to the CONTEXT
 * structure defined in WinNT.h in the Windows Mobile SDK. That structure
 * does not accomodate VFPv3, and I'm unsure if it was ever used in the
 * wild anyway, as Windows CE only seems to produce "cedumps" which
 * are not exactly minidumps.
 */
typedef struct {
  /* 32 64-bit floating point registers, f0..f31 */
  uint64_t regs[MD_FLOATINGSAVEAREA_MIPS_FPR_COUNT];

  uint32_t fpcsr; /* FPU status register. */
  uint32_t fir; /* FPU implementation register. */
} MDFloatingSaveAreaMIPS;

typedef struct {
  /* The next field determines the layout of the structure, and which parts
   * of it are populated.
   */
  uint32_t context_flags;
  uint32_t _pad0;

  /* 32 64-bit integer registers, r0..r31.
   * Note the following fixed uses:
   *   r29 is the stack pointer.
   *   r31 is the return address.
   */
  uint64_t iregs[MD_CONTEXT_MIPS_GPR_COUNT];

  /* multiply/divide result. */
  uint64_t mdhi, mdlo;

  /* DSP accumulators. */
  uint32_t hi[MD_CONTEXT_MIPS_DSP_COUNT];
  uint32_t lo[MD_CONTEXT_MIPS_DSP_COUNT];
  uint32_t dsp_control;
  uint32_t _pad1;

  uint64_t epc;
  uint64_t badvaddr;
  uint32_t status;
  uint32_t cause;

  /* The next field is included with MD_CONTEXT_MIPS_FLOATING_POINT. */
  MDFloatingSaveAreaMIPS float_save;

} MDRawContextMIPS;

/* Indices into iregs for registers with a dedicated or conventional
 * purpose.
 */
enum MDMIPSRegisterNumbers {
  MD_CONTEXT_MIPS_REG_S0     = 16,
  MD_CONTEXT_MIPS_REG_S1     = 17,
  MD_CONTEXT_MIPS_REG_S2     = 18,
  MD_CONTEXT_MIPS_REG_S3     = 19,
  MD_CONTEXT_MIPS_REG_S4     = 20,
  MD_CONTEXT_MIPS_REG_S5     = 21,
  MD_CONTEXT_MIPS_REG_S6     = 22,
  MD_CONTEXT_MIPS_REG_S7     = 23,
  MD_CONTEXT_MIPS_REG_GP     = 28,
  MD_CONTEXT_MIPS_REG_SP     = 29,
  MD_CONTEXT_MIPS_REG_FP     = 30,
  MD_CONTEXT_MIPS_REG_RA     = 31,
};

/* For (MDRawContextMIPS).context_flags.  These values indicate the type of
 * context stored in the structure. */
/* CONTEXT_MIPS from the Windows CE 5.0 SDK. This value isn't correct
 * because this bit can be used for flags. Presumably this value was
 * never actually used in minidumps, but only in "CEDumps" which
 * are a whole parallel minidump file format for Windows CE.
 * Therefore, Breakpad defines its own value for MIPS CPUs.
 */
#define MD_CONTEXT_MIPS  0x00040000
#define MD_CONTEXT_MIPS_INTEGER           (MD_CONTEXT_MIPS | 0x00000002)
#define MD_CONTEXT_MIPS_FLOATING_POINT    (MD_CONTEXT_MIPS | 0x00000004)
#define MD_CONTEXT_MIPS_DSP               (MD_CONTEXT_MIPS | 0x00000008)

#define MD_CONTEXT_MIPS_FULL              (MD_CONTEXT_MIPS_INTEGER | \
                                           MD_CONTEXT_MIPS_FLOATING_POINT | \
                                           MD_CONTEXT_MIPS_DSP)

#define MD_CONTEXT_MIPS_ALL               (MD_CONTEXT_MIPS_INTEGER | \
                                           MD_CONTEXT_MIPS_FLOATING_POINT \
                                           MD_CONTEXT_MIPS_DSP)

/**
 * Breakpad defines for MIPS64
 */
#define MD_CONTEXT_MIPS64  0x00080000
#define MD_CONTEXT_MIPS64_INTEGER           (MD_CONTEXT_MIPS64 | 0x00000002)
#define MD_CONTEXT_MIPS64_FLOATING_POINT    (MD_CONTEXT_MIPS64 | 0x00000004)
#define MD_CONTEXT_MIPS64_DSP               (MD_CONTEXT_MIPS64 | 0x00000008)

#define MD_CONTEXT_MIPS64_FULL              (MD_CONTEXT_MIPS64_INTEGER | \
                                             MD_CONTEXT_MIPS64_FLOATING_POINT | \
                                             MD_CONTEXT_MIPS64_DSP)

#define MD_CONTEXT_MIPS64_ALL               (MD_CONTEXT_MIPS64_INTEGER | \
                                             MD_CONTEXT_MIPS64_FLOATING_POINT \
                                             MD_CONTEXT_MIPS64_DSP)

#endif  // GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_MIPS_H__
