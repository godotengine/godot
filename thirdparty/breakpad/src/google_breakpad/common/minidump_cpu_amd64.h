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

/* minidump_format.h: A cross-platform reimplementation of minidump-related
 * portions of DbgHelp.h from the Windows Platform SDK.
 *
 * (This is C99 source, please don't corrupt it with C++.)
 *
 * This file contains the necessary definitions to read minidump files
 * produced on amd64.  These files may be read on any platform provided
 * that the alignments of these structures on the processing system are
 * identical to the alignments of these structures on the producing system.
 * For this reason, precise-sized types are used.  The structures defined
 * by this file have been laid out to minimize alignment problems by ensuring
 * ensuring that all members are aligned on their natural boundaries.  In
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
 * These definitions may be extended to support handling minidump files
 * for other CPUs and other operating systems.
 *
 * Because precise data type sizes are crucial for this implementation to
 * function properly and portably in terms of interoperability with minidumps
 * produced by DbgHelp on Windows, a set of primitive types with known sizes
 * are used as the basis of each structure defined by this file.  DbgHelp
 * on Windows is assumed to be the reference implementation; this file
 * seeks to provide a cross-platform compatible implementation.  To avoid
 * collisions with the types and values defined and used by DbgHelp in the
 * event that this implementation is used on Windows, each type and value
 * defined here is given a new name, beginning with "MD".  Names of the
 * equivalent types and values in the Windows Platform SDK are given in
 * comments.
 *
 * Author: Mark Mentovai
 * Change to split into its own file: Neal Sidhwaney */

#ifndef GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_AMD64_H__
#define GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_AMD64_H__


/*
 * AMD64 support, see WINNT.H
 */

typedef struct {
  uint16_t       control_word;
  uint16_t       status_word;
  uint8_t        tag_word;
  uint8_t        reserved1;
  uint16_t       error_opcode;
  uint32_t       error_offset;
  uint16_t       error_selector;
  uint16_t       reserved2;
  uint32_t       data_offset;
  uint16_t       data_selector;
  uint16_t       reserved3;
  uint32_t       mx_csr;
  uint32_t       mx_csr_mask;
  uint128_struct float_registers[8];
  uint128_struct xmm_registers[16];
  uint8_t        reserved4[96];
} MDXmmSaveArea32AMD64;  /* XMM_SAVE_AREA32 */

#define MD_CONTEXT_AMD64_VR_COUNT 26

typedef struct {
  /*
   * Register parameter home addresses.
   */
  uint64_t  p1_home;
  uint64_t  p2_home;
  uint64_t  p3_home;
  uint64_t  p4_home;
  uint64_t  p5_home;
  uint64_t  p6_home;

  /* The next field determines the layout of the structure, and which parts
   * of it are populated */
  uint32_t  context_flags;
  uint32_t  mx_csr;

  /* The next register is included with MD_CONTEXT_AMD64_CONTROL */
  uint16_t  cs;

  /* The next 4 registers are included with MD_CONTEXT_AMD64_SEGMENTS */
  uint16_t  ds;
  uint16_t  es;
  uint16_t  fs;
  uint16_t  gs;

  /* The next 2 registers are included with MD_CONTEXT_AMD64_CONTROL */
  uint16_t  ss;
  uint32_t  eflags;

  /* The next 6 registers are included with MD_CONTEXT_AMD64_DEBUG_REGISTERS */
  uint64_t  dr0;
  uint64_t  dr1;
  uint64_t  dr2;
  uint64_t  dr3;
  uint64_t  dr6;
  uint64_t  dr7;

  /* The next 4 registers are included with MD_CONTEXT_AMD64_INTEGER */
  uint64_t  rax;
  uint64_t  rcx;
  uint64_t  rdx;
  uint64_t  rbx;

  /* The next register is included with MD_CONTEXT_AMD64_CONTROL */
  uint64_t  rsp;

  /* The next 11 registers are included with MD_CONTEXT_AMD64_INTEGER */
  uint64_t  rbp;
  uint64_t  rsi;
  uint64_t  rdi;
  uint64_t  r8;
  uint64_t  r9;
  uint64_t  r10;
  uint64_t  r11;
  uint64_t  r12;
  uint64_t  r13;
  uint64_t  r14;
  uint64_t  r15;

  /* The next register is included with MD_CONTEXT_AMD64_CONTROL */
  uint64_t  rip;

  /* The next set of registers are included with
   * MD_CONTEXT_AMD64_FLOATING_POINT
   */
  union {
    MDXmmSaveArea32AMD64 flt_save;
    struct {
      uint128_struct header[2];
      uint128_struct legacy[8];
      uint128_struct xmm0;
      uint128_struct xmm1;
      uint128_struct xmm2;
      uint128_struct xmm3;
      uint128_struct xmm4;
      uint128_struct xmm5;
      uint128_struct xmm6;
      uint128_struct xmm7;
      uint128_struct xmm8;
      uint128_struct xmm9;
      uint128_struct xmm10;
      uint128_struct xmm11;
      uint128_struct xmm12;
      uint128_struct xmm13;
      uint128_struct xmm14;
      uint128_struct xmm15;
    } sse_registers;
  };

  uint128_struct vector_register[MD_CONTEXT_AMD64_VR_COUNT];
  uint64_t       vector_control;

  /* The next 5 registers are included with MD_CONTEXT_AMD64_DEBUG_REGISTERS */
  uint64_t debug_control;
  uint64_t last_branch_to_rip;
  uint64_t last_branch_from_rip;
  uint64_t last_exception_to_rip;
  uint64_t last_exception_from_rip;

} MDRawContextAMD64;  /* CONTEXT */

/* For (MDRawContextAMD64).context_flags.  These values indicate the type of
 * context stored in the structure.  The high 24 bits identify the CPU, the
 * low 8 bits identify the type of context saved. */
#define MD_CONTEXT_AMD64 0x00100000  /* CONTEXT_AMD64 */
#define MD_CONTEXT_AMD64_CONTROL         (MD_CONTEXT_AMD64 | 0x00000001)
     /* CONTEXT_CONTROL */
#define MD_CONTEXT_AMD64_INTEGER         (MD_CONTEXT_AMD64 | 0x00000002)
     /* CONTEXT_INTEGER */
#define MD_CONTEXT_AMD64_SEGMENTS        (MD_CONTEXT_AMD64 | 0x00000004)
     /* CONTEXT_SEGMENTS */
#define MD_CONTEXT_AMD64_FLOATING_POINT  (MD_CONTEXT_AMD64 | 0x00000008)
     /* CONTEXT_FLOATING_POINT */
#define MD_CONTEXT_AMD64_DEBUG_REGISTERS (MD_CONTEXT_AMD64 | 0x00000010)
     /* CONTEXT_DEBUG_REGISTERS */
#define MD_CONTEXT_AMD64_XSTATE          (MD_CONTEXT_AMD64 | 0x00000040)
     /* CONTEXT_XSTATE */

/* WinNT.h refers to CONTEXT_MMX_REGISTERS but doesn't appear to define it
 * I think it really means CONTEXT_FLOATING_POINT.
 */

#define MD_CONTEXT_AMD64_FULL            (MD_CONTEXT_AMD64_CONTROL | \
                                          MD_CONTEXT_AMD64_INTEGER | \
                                          MD_CONTEXT_AMD64_FLOATING_POINT)
     /* CONTEXT_FULL */

#define MD_CONTEXT_AMD64_ALL             (MD_CONTEXT_AMD64_FULL | \
                                          MD_CONTEXT_AMD64_SEGMENTS | \
                                          MD_CONTEXT_X86_DEBUG_REGISTERS)
     /* CONTEXT_ALL */


#endif /* GOOGLE_BREAKPAD_COMMON_MINIDUMP_CPU_AMD64_H__ */
