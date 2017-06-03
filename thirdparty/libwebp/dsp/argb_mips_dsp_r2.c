// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//   ARGB making functions (mips version).
//
// Author: Djordje Pesut (djordje.pesut@imgtec.com)

#include "./dsp.h"

#if defined(WEBP_USE_MIPS_DSP_R2)

static void PackARGB(const uint8_t* a, const uint8_t* r, const uint8_t* g,
                     const uint8_t* b, int len, uint32_t* out) {
  int temp0, temp1, temp2, temp3, offset;
  const int rest = len & 1;
  const uint32_t* const loop_end = out + len - rest;
  const int step = 4;
  __asm__ volatile (
    "xor          %[offset],   %[offset], %[offset]    \n\t"
    "beq          %[loop_end], %[out],    0f           \n\t"
  "2:                                                  \n\t"
    "lbux         %[temp0],    %[offset](%[a])         \n\t"
    "lbux         %[temp1],    %[offset](%[r])         \n\t"
    "lbux         %[temp2],    %[offset](%[g])         \n\t"
    "lbux         %[temp3],    %[offset](%[b])         \n\t"
    "ins          %[temp1],    %[temp0],  16,     16   \n\t"
    "ins          %[temp3],    %[temp2],  16,     16   \n\t"
    "addiu        %[out],      %[out],    4            \n\t"
    "precr.qb.ph  %[temp0],    %[temp1],  %[temp3]     \n\t"
    "sw           %[temp0],    -4(%[out])              \n\t"
    "addu         %[offset],   %[offset], %[step]      \n\t"
    "bne          %[loop_end], %[out],    2b           \n\t"
  "0:                                                  \n\t"
    "beq          %[rest],     $zero,     1f           \n\t"
    "lbux         %[temp0],    %[offset](%[a])         \n\t"
    "lbux         %[temp1],    %[offset](%[r])         \n\t"
    "lbux         %[temp2],    %[offset](%[g])         \n\t"
    "lbux         %[temp3],    %[offset](%[b])         \n\t"
    "ins          %[temp1],    %[temp0],  16,     16   \n\t"
    "ins          %[temp3],    %[temp2],  16,     16   \n\t"
    "precr.qb.ph  %[temp0],    %[temp1],  %[temp3]     \n\t"
    "sw           %[temp0],    0(%[out])               \n\t"
  "1:                                                  \n\t"
    : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp2]"=&r"(temp2),
      [temp3]"=&r"(temp3), [offset]"=&r"(offset), [out]"+&r"(out)
    : [a]"r"(a), [r]"r"(r), [g]"r"(g), [b]"r"(b), [step]"r"(step),
      [loop_end]"r"(loop_end), [rest]"r"(rest)
    : "memory"
  );
}

static void PackRGB(const uint8_t* r, const uint8_t* g, const uint8_t* b,
                    int len, int step, uint32_t* out) {
  int temp0, temp1, temp2, offset;
  const int rest = len & 1;
  const int a = 0xff;
  const uint32_t* const loop_end = out + len - rest;
  __asm__ volatile (
    "xor          %[offset],   %[offset], %[offset]    \n\t"
    "beq          %[loop_end], %[out],    0f           \n\t"
  "2:                                                  \n\t"
    "lbux         %[temp0],    %[offset](%[r])         \n\t"
    "lbux         %[temp1],    %[offset](%[g])         \n\t"
    "lbux         %[temp2],    %[offset](%[b])         \n\t"
    "ins          %[temp0],    %[a],      16,     16   \n\t"
    "ins          %[temp2],    %[temp1],  16,     16   \n\t"
    "addiu        %[out],      %[out],    4            \n\t"
    "precr.qb.ph  %[temp0],    %[temp0],  %[temp2]     \n\t"
    "sw           %[temp0],    -4(%[out])              \n\t"
    "addu         %[offset],   %[offset], %[step]      \n\t"
    "bne          %[loop_end], %[out],    2b           \n\t"
  "0:                                                  \n\t"
    "beq          %[rest],     $zero,     1f           \n\t"
    "lbux         %[temp0],    %[offset](%[r])         \n\t"
    "lbux         %[temp1],    %[offset](%[g])         \n\t"
    "lbux         %[temp2],    %[offset](%[b])         \n\t"
    "ins          %[temp0],    %[a],      16,     16   \n\t"
    "ins          %[temp2],    %[temp1],  16,     16   \n\t"
    "precr.qb.ph  %[temp0],    %[temp0],  %[temp2]     \n\t"
    "sw           %[temp0],    0(%[out])               \n\t"
  "1:                                                  \n\t"
    : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp2]"=&r"(temp2),
      [offset]"=&r"(offset), [out]"+&r"(out)
    : [a]"r"(a), [r]"r"(r), [g]"r"(g), [b]"r"(b), [step]"r"(step),
      [loop_end]"r"(loop_end), [rest]"r"(rest)
    : "memory"
  );
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8EncDspARGBInitMIPSdspR2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspARGBInitMIPSdspR2(void) {
  VP8PackARGB = PackARGB;
  VP8PackRGB = PackRGB;
}

#else  // !WEBP_USE_MIPS_DSP_R2

WEBP_DSP_INIT_STUB(VP8EncDspARGBInitMIPSdspR2)

#endif  // WEBP_USE_MIPS_DSP_R2
