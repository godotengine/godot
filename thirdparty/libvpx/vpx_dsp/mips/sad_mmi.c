/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/asmdefs_mmi.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

#define SAD_SRC_REF_ABS_SUB_64                                      \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[src])                        \n\t" \
  "gsldlc1    %[ftmp3],   0x07(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x00(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x0f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x08(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x17(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x10(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x1f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x18(%[src])                        \n\t" \
  "gsldlc1    %[ftmp3],   0x17(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x10(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x1f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x18(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x27(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x20(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x2f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x28(%[src])                        \n\t" \
  "gsldlc1    %[ftmp3],   0x27(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x20(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x2f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x28(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x37(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x30(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x3f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x38(%[src])                        \n\t" \
  "gsldlc1    %[ftmp3],   0x37(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x30(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x3f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x38(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

#define SAD_SRC_REF_ABS_SUB_32                                      \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[src])                        \n\t" \
  "gsldlc1    %[ftmp3],   0x07(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x00(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x0f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x08(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x17(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x10(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x1f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x18(%[src])                        \n\t" \
  "gsldlc1    %[ftmp3],   0x17(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x10(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x1f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x18(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

#define SAD_SRC_REF_ABS_SUB_16                                      \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[src])                        \n\t" \
  "gsldlc1    %[ftmp3],   0x07(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x00(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x0f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x08(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

#define SAD_SRC_REF_ABS_SUB_8                                       \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x07(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x00(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "paddw      %[ftmp3],   %[ftmp3],       %[ftmp1]            \n\t"

#if _MIPS_SIM == _ABIO32
#define SAD_SRC_REF_ABS_SUB_4                                       \
  "ulw        %[tmp0],    0x00(%[src])                        \n\t" \
  "mtc1       %[tmp0],    %[ftmp1]                            \n\t" \
  "ulw        %[tmp0],    0x00(%[ref])                        \n\t" \
  "mtc1       %[tmp0],    %[ftmp2]                            \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t" \
  "mthc1      $0,         %[ftmp1]                            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "paddw      %[ftmp3],   %[ftmp3],       %[ftmp1]            \n\t"
#else /* _MIPS_SIM == _ABI64 || _MIPS_SIM == _ABIN32 */
#define SAD_SRC_REF_ABS_SUB_4                                       \
  "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t" \
  "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "gslwlc1    %[ftmp2],   0x03(%[ref])                        \n\t" \
  "gslwrc1    %[ftmp2],   0x00(%[ref])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t" \
  "mthc1      $0,         %[ftmp1]                            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "paddw      %[ftmp3],   %[ftmp3],       %[ftmp1]            \n\t"
#endif /* _MIPS_SIM == _ABIO32 */

#define SAD_SRC_AVGREF_ABS_SUB_64                                   \
  "gsldlc1    %[ftmp1],   0x07(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp3],   0x07(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x00(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x0f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x08(%[ref])                        \n\t" \
  "pavgb      %[ftmp3],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pavgb      %[ftmp4],   %[ftmp2],       %[ftmp4]            \n\t" \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x17(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp1],   0x10(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp2],   0x1f(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp2],   0x18(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp3],   0x17(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x10(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x1f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x18(%[ref])                        \n\t" \
  "pavgb      %[ftmp3],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pavgb      %[ftmp4],   %[ftmp2],       %[ftmp4]            \n\t" \
  "gsldlc1    %[ftmp1],   0x17(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x10(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x1f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x18(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x27(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp1],   0x20(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp2],   0x2f(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp2],   0x28(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp3],   0x27(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x20(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x2f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x28(%[ref])                        \n\t" \
  "pavgb      %[ftmp3],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pavgb      %[ftmp4],   %[ftmp2],       %[ftmp4]            \n\t" \
  "gsldlc1    %[ftmp1],   0x27(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x20(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x2f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x28(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x37(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp1],   0x30(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp2],   0x3f(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp2],   0x38(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp3],   0x37(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x30(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x3f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x38(%[ref])                        \n\t" \
  "pavgb      %[ftmp3],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pavgb      %[ftmp4],   %[ftmp2],       %[ftmp4]            \n\t" \
  "gsldlc1    %[ftmp1],   0x37(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x30(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x3f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x38(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

#define SAD_SRC_AVGREF_ABS_SUB_32                                   \
  "gsldlc1    %[ftmp1],   0x07(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp3],   0x07(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x00(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x0f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x08(%[ref])                        \n\t" \
  "pavgb      %[ftmp3],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pavgb      %[ftmp4],   %[ftmp2],       %[ftmp4]            \n\t" \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x17(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp1],   0x10(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp2],   0x1f(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp2],   0x18(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp3],   0x17(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x10(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x1f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x18(%[ref])                        \n\t" \
  "pavgb      %[ftmp3],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pavgb      %[ftmp4],   %[ftmp2],       %[ftmp4]            \n\t" \
  "gsldlc1    %[ftmp1],   0x17(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x10(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x1f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x18(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

#define SAD_SRC_AVGREF_ABS_SUB_16                                   \
  "gsldlc1    %[ftmp1],   0x07(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp3],   0x07(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp3],   0x00(%[ref])                        \n\t" \
  "gsldlc1    %[ftmp4],   0x0f(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp4],   0x08(%[ref])                        \n\t" \
  "pavgb      %[ftmp3],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pavgb      %[ftmp4],   %[ftmp2],       %[ftmp4]            \n\t" \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[src])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t" \
  "pasubub    %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "biadd      %[ftmp2],   %[ftmp2]                            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "paddw      %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

#define SAD_SRC_AVGREF_ABS_SUB_8                                    \
  "gsldlc1    %[ftmp1],   0x07(%[second_pred])                \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[second_pred])                \n\t" \
  "gsldlc1    %[ftmp2],   0x07(%[ref])                        \n\t" \
  "gsldrc1    %[ftmp2],   0x00(%[ref])                        \n\t" \
  "pavgb      %[ftmp2],   %[ftmp1],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "paddw      %[ftmp3],   %[ftmp3],       %[ftmp1]            \n\t"

#if _MIPS_SIM == _ABIO32
#define SAD_SRC_AVGREF_ABS_SUB_4                                    \
  "ulw        %[tmp0],    0x00(%[second_pred])                \n\t" \
  "mtc1       %[tmp0],    %[ftmp1]                            \n\t" \
  "ulw        %[tmp0],    0x00(%[ref])                        \n\t" \
  "mtc1       %[tmp0],    %[ftmp2]                            \n\t" \
  "pavgb      %[ftmp2],   %[ftmp1],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t" \
  "mthc1      $0,         %[ftmp1]                            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "paddw      %[ftmp3],   %[ftmp3],       %[ftmp1]            \n\t"
#else /* _MIPS_SIM == _ABI64 || _MIPS_SIM == _ABIN32 */
#define SAD_SRC_AVGREF_ABS_SUB_4                                    \
  "gslwlc1    %[ftmp1],   0x03(%[second_pred])                \n\t" \
  "gslwrc1    %[ftmp1],   0x00(%[second_pred])                \n\t" \
  "gslwlc1    %[ftmp2],   0x03(%[ref])                        \n\t" \
  "gslwrc1    %[ftmp2],   0x00(%[ref])                        \n\t" \
  "pavgb      %[ftmp2],   %[ftmp1],       %[ftmp2]            \n\t" \
  "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t" \
  "pasubub    %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t" \
  "mthc1      $0,         %[ftmp1]                            \n\t" \
  "biadd      %[ftmp1],   %[ftmp1]                            \n\t" \
  "paddw      %[ftmp3],   %[ftmp3],       %[ftmp1]            \n\t"
#endif /* _MIPS_SIM == _ABIO32 */

#define sadMxNx4D_mmi(m, n)                                                  \
  void vpx_sad##m##x##n##x4d_mmi(const uint8_t *src, int src_stride,         \
                                 const uint8_t *const ref_array[],           \
                                 int ref_stride, uint32_t *sad_array) {      \
    int i;                                                                   \
    for (i = 0; i < 4; ++i)                                                  \
      sad_array[i] =                                                         \
          vpx_sad##m##x##n##_mmi(src, src_stride, ref_array[i], ref_stride); \
  }

static inline unsigned int vpx_sad64x(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride,
                                      int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3, ftmp4, ftmp5;
  mips_reg l_counter = counter;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp5],   %[ftmp5],       %[ftmp5]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_REF_ABS_SUB_64
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_REF_ABS_SUB_64
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp5]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [ftmp4]"=&f"(ftmp4), [ftmp5]"=&f"(ftmp5), [counter]"+&r"(l_counter),
      [src]"+&r"(src), [ref]"+&r"(ref), [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad64xN(H)                                                   \
  unsigned int vpx_sad64x##H##_mmi(const uint8_t *src, int src_stride,   \
                                   const uint8_t *ref, int ref_stride) { \
    return vpx_sad64x(src, src_stride, ref, ref_stride, H);              \
  }

vpx_sad64xN(64);
vpx_sad64xN(32);
sadMxNx4D_mmi(64, 64);
sadMxNx4D_mmi(64, 32);

static inline unsigned int vpx_sad_avg64x(const uint8_t *src, int src_stride,
                                          const uint8_t *ref, int ref_stride,
                                          const uint8_t *second_pred,
                                          int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3, ftmp4, ftmp5;
  mips_reg l_counter = counter;
  mips_reg l_second_pred = (mips_reg)second_pred;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp5],   %[ftmp5],       %[ftmp5]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_AVGREF_ABS_SUB_64
    MMI_ADDIU(%[second_pred], %[second_pred], 0x40)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_AVGREF_ABS_SUB_64
    MMI_ADDIU(%[second_pred], %[second_pred], 0x40)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp5]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [ftmp4]"=&f"(ftmp4), [ftmp5]"=&f"(ftmp5), [counter]"+&r"(l_counter),
      [src]"+&r"(src), [ref]"+&r"(ref),
      [second_pred]"+&r"(l_second_pred),
      [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad_avg64xN(H)                                                   \
  unsigned int vpx_sad64x##H##_avg_mmi(const uint8_t *src, int src_stride,   \
                                       const uint8_t *ref, int ref_stride,   \
                                       const uint8_t *second_pred) {         \
    return vpx_sad_avg64x(src, src_stride, ref, ref_stride, second_pred, H); \
  }

vpx_sad_avg64xN(64);
vpx_sad_avg64xN(32);

static inline unsigned int vpx_sad32x(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride,
                                      int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3, ftmp4, ftmp5;
  mips_reg l_counter = counter;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp5],   %[ftmp5],       %[ftmp5]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_REF_ABS_SUB_32
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_REF_ABS_SUB_32
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp5]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [ftmp4]"=&f"(ftmp4), [ftmp5]"=&f"(ftmp5), [counter]"+&r"(l_counter),
      [src]"+&r"(src), [ref]"+&r"(ref), [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad32xN(H)                                                   \
  unsigned int vpx_sad32x##H##_mmi(const uint8_t *src, int src_stride,   \
                                   const uint8_t *ref, int ref_stride) { \
    return vpx_sad32x(src, src_stride, ref, ref_stride, H);              \
  }

vpx_sad32xN(64);
vpx_sad32xN(32);
vpx_sad32xN(16);
sadMxNx4D_mmi(32, 64);
sadMxNx4D_mmi(32, 32);
sadMxNx4D_mmi(32, 16);

static inline unsigned int vpx_sad_avg32x(const uint8_t *src, int src_stride,
                                          const uint8_t *ref, int ref_stride,
                                          const uint8_t *second_pred,
                                          int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3, ftmp4, ftmp5;
  mips_reg l_counter = counter;
  mips_reg l_second_pred = (mips_reg)second_pred;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp5],   %[ftmp5],       %[ftmp5]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_AVGREF_ABS_SUB_32
    MMI_ADDIU(%[second_pred], %[second_pred], 0x20)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_AVGREF_ABS_SUB_32
    MMI_ADDIU(%[second_pred], %[second_pred], 0x20)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp5]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [ftmp4]"=&f"(ftmp4), [ftmp5]"=&f"(ftmp5), [counter]"+&r"(l_counter),
      [src]"+&r"(src), [ref]"+&r"(ref),
      [second_pred]"+&r"(l_second_pred),
      [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad_avg32xN(H)                                                   \
  unsigned int vpx_sad32x##H##_avg_mmi(const uint8_t *src, int src_stride,   \
                                       const uint8_t *ref, int ref_stride,   \
                                       const uint8_t *second_pred) {         \
    return vpx_sad_avg32x(src, src_stride, ref, ref_stride, second_pred, H); \
  }

vpx_sad_avg32xN(64);
vpx_sad_avg32xN(32);
vpx_sad_avg32xN(16);

static inline unsigned int vpx_sad16x(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride,
                                      int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3, ftmp4, ftmp5;
  mips_reg l_counter = counter;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp5],   %[ftmp5],       %[ftmp5]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_REF_ABS_SUB_16
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_REF_ABS_SUB_16
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp5]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [ftmp4]"=&f"(ftmp4), [ftmp5]"=&f"(ftmp5), [counter]"+&r"(l_counter),
      [src]"+&r"(src), [ref]"+&r"(ref), [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad16xN(H)                                                   \
  unsigned int vpx_sad16x##H##_mmi(const uint8_t *src, int src_stride,   \
                                   const uint8_t *ref, int ref_stride) { \
    return vpx_sad16x(src, src_stride, ref, ref_stride, H);              \
  }

vpx_sad16xN(32);
vpx_sad16xN(16);
vpx_sad16xN(8);
sadMxNx4D_mmi(16, 32);
sadMxNx4D_mmi(16, 16);
sadMxNx4D_mmi(16, 8);

static inline unsigned int vpx_sad_avg16x(const uint8_t *src, int src_stride,
                                          const uint8_t *ref, int ref_stride,
                                          const uint8_t *second_pred,
                                          int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3, ftmp4, ftmp5;
  mips_reg l_counter = counter;
  mips_reg l_second_pred = (mips_reg)second_pred;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp5],   %[ftmp5],       %[ftmp5]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_AVGREF_ABS_SUB_16
    MMI_ADDIU(%[second_pred], %[second_pred], 0x10)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_AVGREF_ABS_SUB_16
    MMI_ADDIU(%[second_pred], %[second_pred], 0x10)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp5]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [ftmp4]"=&f"(ftmp4), [ftmp5]"=&f"(ftmp5), [counter]"+&r"(l_counter),
      [src]"+&r"(src), [ref]"+&r"(ref),
      [second_pred]"+&r"(l_second_pred),
      [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad_avg16xN(H)                                                   \
  unsigned int vpx_sad16x##H##_avg_mmi(const uint8_t *src, int src_stride,   \
                                       const uint8_t *ref, int ref_stride,   \
                                       const uint8_t *second_pred) {         \
    return vpx_sad_avg16x(src, src_stride, ref, ref_stride, second_pred, H); \
  }

vpx_sad_avg16xN(32);
vpx_sad_avg16xN(16);
vpx_sad_avg16xN(8);

static inline unsigned int vpx_sad8x(const uint8_t *src, int src_stride,
                                     const uint8_t *ref, int ref_stride,
                                     int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3;
  mips_reg l_counter = counter;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp3],   %[ftmp3],       %[ftmp3]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_REF_ABS_SUB_8
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_REF_ABS_SUB_8
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp3]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [counter]"+&r"(l_counter), [src]"+&r"(src), [ref]"+&r"(ref),
      [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad8xN(H)                                                   \
  unsigned int vpx_sad8x##H##_mmi(const uint8_t *src, int src_stride,   \
                                  const uint8_t *ref, int ref_stride) { \
    return vpx_sad8x(src, src_stride, ref, ref_stride, H);              \
  }

vpx_sad8xN(16);
vpx_sad8xN(8);
vpx_sad8xN(4);
sadMxNx4D_mmi(8, 16);
sadMxNx4D_mmi(8, 8);
sadMxNx4D_mmi(8, 4);

static inline unsigned int vpx_sad_avg8x(const uint8_t *src, int src_stride,
                                         const uint8_t *ref, int ref_stride,
                                         const uint8_t *second_pred,
                                         int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3;
  mips_reg l_counter = counter;
  mips_reg l_second_pred = (mips_reg)second_pred;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp3],   %[ftmp3],       %[ftmp3]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_AVGREF_ABS_SUB_8
    MMI_ADDIU(%[second_pred], %[second_pred], 0x08)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_AVGREF_ABS_SUB_8
    MMI_ADDIU(%[second_pred], %[second_pred], 0x08)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp3]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [counter]"+&r"(l_counter), [src]"+&r"(src), [ref]"+&r"(ref),
      [second_pred]"+&r"(l_second_pred),
      [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad_avg8xN(H)                                                   \
  unsigned int vpx_sad8x##H##_avg_mmi(const uint8_t *src, int src_stride,   \
                                      const uint8_t *ref, int ref_stride,   \
                                      const uint8_t *second_pred) {         \
    return vpx_sad_avg8x(src, src_stride, ref, ref_stride, second_pred, H); \
  }

vpx_sad_avg8xN(16);
vpx_sad_avg8xN(8);
vpx_sad_avg8xN(4);

static inline unsigned int vpx_sad4x(const uint8_t *src, int src_stride,
                                     const uint8_t *ref, int ref_stride,
                                     int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3;
  mips_reg l_counter = counter;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp3],   %[ftmp3],       %[ftmp3]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_REF_ABS_SUB_4
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_REF_ABS_SUB_4
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp3]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [counter]"+&r"(l_counter), [src]"+&r"(src), [ref]"+&r"(ref),
      [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad4xN(H)                                                   \
  unsigned int vpx_sad4x##H##_mmi(const uint8_t *src, int src_stride,   \
                                  const uint8_t *ref, int ref_stride) { \
    return vpx_sad4x(src, src_stride, ref, ref_stride, H);              \
  }

vpx_sad4xN(8);
vpx_sad4xN(4);
sadMxNx4D_mmi(4, 8);
sadMxNx4D_mmi(4, 4);

static inline unsigned int vpx_sad_avg4x(const uint8_t *src, int src_stride,
                                         const uint8_t *ref, int ref_stride,
                                         const uint8_t *second_pred,
                                         int counter) {
  unsigned int sad;
  double ftmp1, ftmp2, ftmp3;
  mips_reg l_counter = counter;
  mips_reg l_second_pred = (mips_reg)second_pred;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp3],   %[ftmp3],       %[ftmp3]            \n\t"
    "1:                                                         \n\t"
    // Include two loop body, to reduce loop time.
    SAD_SRC_AVGREF_ABS_SUB_4
    MMI_ADDIU(%[second_pred], %[second_pred], 0x04)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    SAD_SRC_AVGREF_ABS_SUB_4
    MMI_ADDIU(%[second_pred], %[second_pred], 0x04)
    MMI_ADDU(%[src],     %[src],         %[src_stride])
    MMI_ADDU(%[ref],     %[ref],         %[ref_stride])
    MMI_ADDIU(%[counter], %[counter], -0x02)
    "bnez       %[counter], 1b                                  \n\t"
    "mfc1       %[sad],     %[ftmp3]                            \n\t"
    : [ftmp1]"=&f"(ftmp1), [ftmp2]"=&f"(ftmp2), [ftmp3]"=&f"(ftmp3),
      [counter]"+&r"(l_counter), [src]"+&r"(src), [ref]"+&r"(ref),
      [second_pred]"+&r"(l_second_pred),
      [sad]"=&r"(sad)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride)
  );
  /* clang-format on */

  return sad;
}

#define vpx_sad_avg4xN(H)                                                   \
  unsigned int vpx_sad4x##H##_avg_mmi(const uint8_t *src, int src_stride,   \
                                      const uint8_t *ref, int ref_stride,   \
                                      const uint8_t *second_pred) {         \
    return vpx_sad_avg4x(src, src_stride, ref, ref_stride, second_pred, H); \
  }

vpx_sad_avg4xN(8);
vpx_sad_avg4xN(4);
