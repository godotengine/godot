/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_ASMDEFS_MMI_H_
#define VPX_VPX_PORTS_ASMDEFS_MMI_H_

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

#if HAVE_MMI

#if HAVE_MIPS64
#define mips_reg int64_t
#define MMI_ADDU(reg1, reg2, reg3) \
  "daddu       " #reg1 ",       " #reg2 ",       " #reg3 "         \n\t"

#define MMI_ADDIU(reg1, reg2, immediate) \
  "daddiu      " #reg1 ",       " #reg2 ",       " #immediate "    \n\t"

#define MMI_ADDI(reg1, reg2, immediate) \
  "daddi       " #reg1 ",       " #reg2 ",       " #immediate "    \n\t"

#define MMI_SUBU(reg1, reg2, reg3) \
  "dsubu       " #reg1 ",       " #reg2 ",       " #reg3 "         \n\t"

#define MMI_L(reg, addr, bias) \
  "ld          " #reg ",        " #bias "(" #addr ")               \n\t"

#define MMI_SRL(reg1, reg2, shift) \
  "ssrld       " #reg1 ",       " #reg2 ",       " #shift "        \n\t"

#define MMI_SLL(reg1, reg2, shift) \
  "dsll        " #reg1 ",       " #reg2 ",       " #shift "        \n\t"

#define MMI_MTC1(reg, fp) \
  "dmtc1       " #reg ",        " #fp "                            \n\t"

#define MMI_LI(reg, immediate) \
  "dli         " #reg ",        " #immediate "                     \n\t"

#else
#define mips_reg int32_t
#define MMI_ADDU(reg1, reg2, reg3) \
  "addu        " #reg1 ",       " #reg2 ",       " #reg3 "         \n\t"

#define MMI_ADDIU(reg1, reg2, immediate) \
  "addiu       " #reg1 ",       " #reg2 ",       " #immediate "    \n\t"

#define MMI_ADDI(reg1, reg2, immediate) \
  "addi        " #reg1 ",       " #reg2 ",       " #immediate "    \n\t"

#define MMI_SUBU(reg1, reg2, reg3) \
  "subu        " #reg1 ",       " #reg2 ",       " #reg3 "         \n\t"

#define MMI_L(reg, addr, bias) \
  "lw          " #reg ",        " #bias "(" #addr ")               \n\t"

#define MMI_SRL(reg1, reg2, shift) \
  "ssrlw       " #reg1 ",       " #reg2 ",       " #shift "        \n\t"

#define MMI_SLL(reg1, reg2, shift) \
  "sll         " #reg1 ",       " #reg2 ",       " #shift "        \n\t"

#define MMI_MTC1(reg, fp) \
  "mtc1        " #reg ",        " #fp "                            \n\t"

#define MMI_LI(reg, immediate) \
  "li          " #reg ",        " #immediate "                     \n\t"

#endif /* HAVE_MIPS64 */

#endif /* HAVE_MMI */

#endif  // VPX_VPX_PORTS_ASMDEFS_MMI_H_
