// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// MIPS version of rescaling functions
//
// Author(s): Djordje Pesut (djordje.pesut@imgtec.com)

#include "./dsp.h"

#if defined(WEBP_USE_MIPS_DSP_R2)

#include <assert.h>
#include "../utils/rescaler.h"

#define ROUNDER (WEBP_RESCALER_ONE >> 1)
#define MULT_FIX(x, y) (((uint64_t)(x) * (y) + ROUNDER) >> WEBP_RESCALER_RFIX)

//------------------------------------------------------------------------------
// Row export

static void ExportRowShrink(WebPRescaler* const wrk) {
  int i;
  const int x_out_max = wrk->dst_width * wrk->num_channels;
  uint8_t* dst = wrk->dst;
  rescaler_t* irow = wrk->irow;
  const rescaler_t* frow = wrk->frow;
  const int yscale = wrk->fy_scale * (-wrk->y_accum);
  int temp0, temp1, temp2, temp3, temp4, temp5, loop_end;
  const int temp7 = (int)wrk->fxy_scale;
  const int temp6 = (x_out_max & ~0x3) << 2;
  assert(!WebPRescalerOutputDone(wrk));
  assert(wrk->y_accum <= 0);
  assert(!wrk->y_expand);
  assert(wrk->fxy_scale != 0);
  if (yscale) {
    if (x_out_max >= 4) {
      int temp8, temp9, temp10, temp11;
      __asm__ volatile (
        "li       %[temp3],    0x10000                    \n\t"
        "li       %[temp4],    0x8000                     \n\t"
        "addu     %[loop_end], %[frow],     %[temp6]      \n\t"
      "1:                                                 \n\t"
        "lw       %[temp0],    0(%[frow])                 \n\t"
        "lw       %[temp1],    4(%[frow])                 \n\t"
        "lw       %[temp2],    8(%[frow])                 \n\t"
        "lw       %[temp5],    12(%[frow])                \n\t"
        "mult     $ac0,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac0,        %[temp0],    %[yscale]     \n\t"
        "mult     $ac1,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac1,        %[temp1],    %[yscale]     \n\t"
        "mult     $ac2,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac2,        %[temp2],    %[yscale]     \n\t"
        "mult     $ac3,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac3,        %[temp5],    %[yscale]     \n\t"
        "addiu    %[frow],     %[frow],     16            \n\t"
        "mfhi     %[temp0],    $ac0                       \n\t"
        "mfhi     %[temp1],    $ac1                       \n\t"
        "mfhi     %[temp2],    $ac2                       \n\t"
        "mfhi     %[temp5],    $ac3                       \n\t"
        "lw       %[temp8],    0(%[irow])                 \n\t"
        "lw       %[temp9],    4(%[irow])                 \n\t"
        "lw       %[temp10],   8(%[irow])                 \n\t"
        "lw       %[temp11],   12(%[irow])                \n\t"
        "addiu    %[dst],      %[dst],      4             \n\t"
        "addiu    %[irow],     %[irow],     16            \n\t"
        "subu     %[temp8],    %[temp8],    %[temp0]      \n\t"
        "subu     %[temp9],    %[temp9],    %[temp1]      \n\t"
        "subu     %[temp10],   %[temp10],   %[temp2]      \n\t"
        "subu     %[temp11],   %[temp11],   %[temp5]      \n\t"
        "mult     $ac0,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac0,        %[temp8],    %[temp7]      \n\t"
        "mult     $ac1,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac1,        %[temp9],    %[temp7]      \n\t"
        "mult     $ac2,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac2,        %[temp10],   %[temp7]      \n\t"
        "mult     $ac3,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac3,        %[temp11],   %[temp7]      \n\t"
        "mfhi     %[temp8],    $ac0                       \n\t"
        "mfhi     %[temp9],    $ac1                       \n\t"
        "mfhi     %[temp10],   $ac2                       \n\t"
        "mfhi     %[temp11],   $ac3                       \n\t"
        "sw       %[temp0],    -16(%[irow])               \n\t"
        "sw       %[temp1],    -12(%[irow])               \n\t"
        "sw       %[temp2],    -8(%[irow])                \n\t"
        "sw       %[temp5],    -4(%[irow])                \n\t"
        "sb       %[temp8],    -4(%[dst])                 \n\t"
        "sb       %[temp9],    -3(%[dst])                 \n\t"
        "sb       %[temp10],   -2(%[dst])                 \n\t"
        "sb       %[temp11],   -1(%[dst])                 \n\t"
        "bne      %[frow],     %[loop_end], 1b            \n\t"
        : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp3]"=&r"(temp3),
          [temp4]"=&r"(temp4), [temp5]"=&r"(temp5), [frow]"+r"(frow),
          [irow]"+r"(irow), [dst]"+r"(dst), [loop_end]"=&r"(loop_end),
          [temp8]"=&r"(temp8), [temp9]"=&r"(temp9), [temp10]"=&r"(temp10),
          [temp11]"=&r"(temp11), [temp2]"=&r"(temp2)
        : [temp7]"r"(temp7), [yscale]"r"(yscale), [temp6]"r"(temp6)
        : "memory", "hi", "lo", "$ac1hi", "$ac1lo",
          "$ac2hi", "$ac2lo", "$ac3hi", "$ac3lo"
      );
    }
    for (i = 0; i < (x_out_max & 0x3); ++i) {
      const uint32_t frac = (uint32_t)MULT_FIX(*frow++, yscale);
      const int v = (int)MULT_FIX(*irow - frac, wrk->fxy_scale);
      assert(v >= 0 && v <= 255);
      *dst++ = v;
      *irow++ = frac;   // new fractional start
    }
  } else {
    if (x_out_max >= 4) {
      __asm__ volatile (
        "li       %[temp3],    0x10000                    \n\t"
        "li       %[temp4],    0x8000                     \n\t"
        "addu     %[loop_end], %[irow],     %[temp6]      \n\t"
      "1:                                                 \n\t"
        "lw       %[temp0],    0(%[irow])                 \n\t"
        "lw       %[temp1],    4(%[irow])                 \n\t"
        "lw       %[temp2],    8(%[irow])                 \n\t"
        "lw       %[temp5],    12(%[irow])                \n\t"
        "addiu    %[dst],      %[dst],      4             \n\t"
        "addiu    %[irow],     %[irow],     16            \n\t"
        "mult     $ac0,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac0,        %[temp0],    %[temp7]      \n\t"
        "mult     $ac1,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac1,        %[temp1],    %[temp7]      \n\t"
        "mult     $ac2,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac2,        %[temp2],    %[temp7]      \n\t"
        "mult     $ac3,        %[temp3],    %[temp4]      \n\t"
        "maddu    $ac3,        %[temp5],    %[temp7]      \n\t"
        "mfhi     %[temp0],    $ac0                       \n\t"
        "mfhi     %[temp1],    $ac1                       \n\t"
        "mfhi     %[temp2],    $ac2                       \n\t"
        "mfhi     %[temp5],    $ac3                       \n\t"
        "sw       $zero,       -16(%[irow])               \n\t"
        "sw       $zero,       -12(%[irow])               \n\t"
        "sw       $zero,       -8(%[irow])                \n\t"
        "sw       $zero,       -4(%[irow])                \n\t"
        "sb       %[temp0],    -4(%[dst])                 \n\t"
        "sb       %[temp1],    -3(%[dst])                 \n\t"
        "sb       %[temp2],    -2(%[dst])                 \n\t"
        "sb       %[temp5],    -1(%[dst])                 \n\t"
        "bne      %[irow],     %[loop_end], 1b            \n\t"
        : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp3]"=&r"(temp3),
          [temp4]"=&r"(temp4), [temp5]"=&r"(temp5), [irow]"+r"(irow),
          [dst]"+r"(dst), [loop_end]"=&r"(loop_end), [temp2]"=&r"(temp2)
        : [temp7]"r"(temp7), [temp6]"r"(temp6)
        : "memory", "hi", "lo", "$ac1hi", "$ac1lo",
          "$ac2hi", "$ac2lo", "$ac3hi", "$ac3lo"
      );
    }
    for (i = 0; i < (x_out_max & 0x3); ++i) {
      const int v = (int)MULT_FIX(*irow, wrk->fxy_scale);
      assert(v >= 0 && v <= 255);
      *dst++ = v;
      *irow++ = 0;
    }
  }
}

static void ExportRowExpand(WebPRescaler* const wrk) {
  int i;
  uint8_t* dst = wrk->dst;
  rescaler_t* irow = wrk->irow;
  const int x_out_max = wrk->dst_width * wrk->num_channels;
  const rescaler_t* frow = wrk->frow;
  int temp0, temp1, temp2, temp3, temp4, temp5, loop_end;
  const int temp6 = (x_out_max & ~0x3) << 2;
  const int temp7 = (int)wrk->fy_scale;
  assert(!WebPRescalerOutputDone(wrk));
  assert(wrk->y_accum <= 0);
  assert(wrk->y_expand);
  assert(wrk->y_sub != 0);
  if (wrk->y_accum == 0) {
    if (x_out_max >= 4) {
      __asm__ volatile (
        "li       %[temp4],    0x10000                    \n\t"
        "li       %[temp5],    0x8000                     \n\t"
        "addu     %[loop_end], %[frow],     %[temp6]      \n\t"
      "1:                                                 \n\t"
        "lw       %[temp0],    0(%[frow])                 \n\t"
        "lw       %[temp1],    4(%[frow])                 \n\t"
        "lw       %[temp2],    8(%[frow])                 \n\t"
        "lw       %[temp3],    12(%[frow])                \n\t"
        "addiu    %[dst],      %[dst],      4             \n\t"
        "addiu    %[frow],     %[frow],     16            \n\t"
        "mult     $ac0,        %[temp4],    %[temp5]      \n\t"
        "maddu    $ac0,        %[temp0],    %[temp7]      \n\t"
        "mult     $ac1,        %[temp4],    %[temp5]      \n\t"
        "maddu    $ac1,        %[temp1],    %[temp7]      \n\t"
        "mult     $ac2,        %[temp4],    %[temp5]      \n\t"
        "maddu    $ac2,        %[temp2],    %[temp7]      \n\t"
        "mult     $ac3,        %[temp4],    %[temp5]      \n\t"
        "maddu    $ac3,        %[temp3],    %[temp7]      \n\t"
        "mfhi     %[temp0],    $ac0                       \n\t"
        "mfhi     %[temp1],    $ac1                       \n\t"
        "mfhi     %[temp2],    $ac2                       \n\t"
        "mfhi     %[temp3],    $ac3                       \n\t"
        "sb       %[temp0],    -4(%[dst])                 \n\t"
        "sb       %[temp1],    -3(%[dst])                 \n\t"
        "sb       %[temp2],    -2(%[dst])                 \n\t"
        "sb       %[temp3],    -1(%[dst])                 \n\t"
        "bne      %[frow],     %[loop_end], 1b            \n\t"
        : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp3]"=&r"(temp3),
          [temp4]"=&r"(temp4), [temp5]"=&r"(temp5), [frow]"+r"(frow),
          [dst]"+r"(dst), [loop_end]"=&r"(loop_end), [temp2]"=&r"(temp2)
        : [temp7]"r"(temp7), [temp6]"r"(temp6)
        : "memory", "hi", "lo", "$ac1hi", "$ac1lo",
          "$ac2hi", "$ac2lo", "$ac3hi", "$ac3lo"
      );
    }
    for (i = 0; i < (x_out_max & 0x3); ++i) {
      const uint32_t J = *frow++;
      const int v = (int)MULT_FIX(J, wrk->fy_scale);
      assert(v >= 0 && v <= 255);
      *dst++ = v;
    }
  } else {
    const uint32_t B = WEBP_RESCALER_FRAC(-wrk->y_accum, wrk->y_sub);
    const uint32_t A = (uint32_t)(WEBP_RESCALER_ONE - B);
    if (x_out_max >= 4) {
      int temp8, temp9, temp10, temp11;
      __asm__ volatile (
        "li       %[temp8],    0x10000                    \n\t"
        "li       %[temp9],    0x8000                     \n\t"
        "addu     %[loop_end], %[frow],     %[temp6]      \n\t"
      "1:                                                 \n\t"
        "lw       %[temp0],    0(%[frow])                 \n\t"
        "lw       %[temp1],    4(%[frow])                 \n\t"
        "lw       %[temp2],    8(%[frow])                 \n\t"
        "lw       %[temp3],    12(%[frow])                \n\t"
        "lw       %[temp4],    0(%[irow])                 \n\t"
        "lw       %[temp5],    4(%[irow])                 \n\t"
        "lw       %[temp10],   8(%[irow])                 \n\t"
        "lw       %[temp11],   12(%[irow])                \n\t"
        "addiu    %[dst],      %[dst],      4             \n\t"
        "mult     $ac0,        %[temp8],    %[temp9]      \n\t"
        "maddu    $ac0,        %[A],        %[temp0]      \n\t"
        "maddu    $ac0,        %[B],        %[temp4]      \n\t"
        "mult     $ac1,        %[temp8],    %[temp9]      \n\t"
        "maddu    $ac1,        %[A],        %[temp1]      \n\t"
        "maddu    $ac1,        %[B],        %[temp5]      \n\t"
        "mult     $ac2,        %[temp8],    %[temp9]      \n\t"
        "maddu    $ac2,        %[A],        %[temp2]      \n\t"
        "maddu    $ac2,        %[B],        %[temp10]     \n\t"
        "mult     $ac3,        %[temp8],    %[temp9]      \n\t"
        "maddu    $ac3,        %[A],        %[temp3]      \n\t"
        "maddu    $ac3,        %[B],        %[temp11]     \n\t"
        "addiu    %[frow],     %[frow],     16            \n\t"
        "addiu    %[irow],     %[irow],     16            \n\t"
        "mfhi     %[temp0],    $ac0                       \n\t"
        "mfhi     %[temp1],    $ac1                       \n\t"
        "mfhi     %[temp2],    $ac2                       \n\t"
        "mfhi     %[temp3],    $ac3                       \n\t"
        "mult     $ac0,        %[temp8],    %[temp9]      \n\t"
        "maddu    $ac0,        %[temp0],    %[temp7]      \n\t"
        "mult     $ac1,        %[temp8],    %[temp9]      \n\t"
        "maddu    $ac1,        %[temp1],    %[temp7]      \n\t"
        "mult     $ac2,        %[temp8],    %[temp9]      \n\t"
        "maddu    $ac2,        %[temp2],    %[temp7]      \n\t"
        "mult     $ac3,        %[temp8],    %[temp9]      \n\t"
        "maddu    $ac3,        %[temp3],    %[temp7]      \n\t"
        "mfhi     %[temp0],    $ac0                       \n\t"
        "mfhi     %[temp1],    $ac1                       \n\t"
        "mfhi     %[temp2],    $ac2                       \n\t"
        "mfhi     %[temp3],    $ac3                       \n\t"
        "sb       %[temp0],    -4(%[dst])                 \n\t"
        "sb       %[temp1],    -3(%[dst])                 \n\t"
        "sb       %[temp2],    -2(%[dst])                 \n\t"
        "sb       %[temp3],    -1(%[dst])                 \n\t"
        "bne      %[frow],     %[loop_end], 1b            \n\t"
        : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp3]"=&r"(temp3),
          [temp4]"=&r"(temp4), [temp5]"=&r"(temp5), [frow]"+r"(frow),
          [irow]"+r"(irow), [dst]"+r"(dst), [loop_end]"=&r"(loop_end),
          [temp8]"=&r"(temp8), [temp9]"=&r"(temp9), [temp10]"=&r"(temp10),
          [temp11]"=&r"(temp11), [temp2]"=&r"(temp2)
        : [temp7]"r"(temp7), [temp6]"r"(temp6), [A]"r"(A), [B]"r"(B)
        : "memory", "hi", "lo", "$ac1hi", "$ac1lo",
          "$ac2hi", "$ac2lo", "$ac3hi", "$ac3lo"
      );
    }
    for (i = 0; i < (x_out_max & 0x3); ++i) {
      const uint64_t I = (uint64_t)A * *frow++
                       + (uint64_t)B * *irow++;
      const uint32_t J = (uint32_t)((I + ROUNDER) >> WEBP_RESCALER_RFIX);
      const int v = (int)MULT_FIX(J, wrk->fy_scale);
      assert(v >= 0 && v <= 255);
      *dst++ = v;
    }
  }
}

#undef MULT_FIX
#undef ROUNDER

//------------------------------------------------------------------------------
// Entry point

extern void WebPRescalerDspInitMIPSdspR2(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPRescalerDspInitMIPSdspR2(void) {
  WebPRescalerExportRowExpand = ExportRowExpand;
  WebPRescalerExportRowShrink = ExportRowShrink;
}

#else  // !WEBP_USE_MIPS_DSP_R2

WEBP_DSP_INIT_STUB(WebPRescalerDspInitMIPSdspR2)

#endif  // WEBP_USE_MIPS_DSP_R2
