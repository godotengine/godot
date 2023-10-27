// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Speed-critical functions for Sharp YUV.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "sharpyuv/sharpyuv_dsp.h"

#include <assert.h>
#include <stdlib.h>

#include "sharpyuv/sharpyuv_cpu.h"

//-----------------------------------------------------------------------------

#if !WEBP_NEON_OMIT_C_CODE
static uint16_t clip(int v, int max) {
  return (v < 0) ? 0 : (v > max) ? max : (uint16_t)v;
}

static uint64_t SharpYuvUpdateY_C(const uint16_t* ref, const uint16_t* src,
                                  uint16_t* dst, int len, int bit_depth) {
  uint64_t diff = 0;
  int i;
  const int max_y = (1 << bit_depth) - 1;
  for (i = 0; i < len; ++i) {
    const int diff_y = ref[i] - src[i];
    const int new_y = (int)dst[i] + diff_y;
    dst[i] = clip(new_y, max_y);
    diff += (uint64_t)abs(diff_y);
  }
  return diff;
}

static void SharpYuvUpdateRGB_C(const int16_t* ref, const int16_t* src,
                                int16_t* dst, int len) {
  int i;
  for (i = 0; i < len; ++i) {
    const int diff_uv = ref[i] - src[i];
    dst[i] += diff_uv;
  }
}

static void SharpYuvFilterRow_C(const int16_t* A, const int16_t* B, int len,
                                const uint16_t* best_y, uint16_t* out,
                                int bit_depth) {
  int i;
  const int max_y = (1 << bit_depth) - 1;
  for (i = 0; i < len; ++i, ++A, ++B) {
    const int v0 = (A[0] * 9 + A[1] * 3 + B[0] * 3 + B[1] + 8) >> 4;
    const int v1 = (A[1] * 9 + A[0] * 3 + B[1] * 3 + B[0] + 8) >> 4;
    out[2 * i + 0] = clip(best_y[2 * i + 0] + v0, max_y);
    out[2 * i + 1] = clip(best_y[2 * i + 1] + v1, max_y);
  }
}
#endif  // !WEBP_NEON_OMIT_C_CODE

//-----------------------------------------------------------------------------

uint64_t (*SharpYuvUpdateY)(const uint16_t* src, const uint16_t* ref,
                            uint16_t* dst, int len, int bit_depth);
void (*SharpYuvUpdateRGB)(const int16_t* src, const int16_t* ref, int16_t* dst,
                          int len);
void (*SharpYuvFilterRow)(const int16_t* A, const int16_t* B, int len,
                          const uint16_t* best_y, uint16_t* out,
                          int bit_depth);

extern VP8CPUInfo SharpYuvGetCPUInfo;
extern void InitSharpYuvSSE2(void);
extern void InitSharpYuvNEON(void);

void SharpYuvInitDsp(void) {
#if !WEBP_NEON_OMIT_C_CODE
  SharpYuvUpdateY = SharpYuvUpdateY_C;
  SharpYuvUpdateRGB = SharpYuvUpdateRGB_C;
  SharpYuvFilterRow = SharpYuvFilterRow_C;
#endif

  if (SharpYuvGetCPUInfo != NULL) {
#if defined(WEBP_HAVE_SSE2)
    if (SharpYuvGetCPUInfo(kSSE2)) {
      InitSharpYuvSSE2();
    }
#endif  // WEBP_HAVE_SSE2
  }

#if defined(WEBP_HAVE_NEON)
  if (WEBP_NEON_OMIT_C_CODE ||
      (SharpYuvGetCPUInfo != NULL && SharpYuvGetCPUInfo(kNEON))) {
    InitSharpYuvNEON();
  }
#endif  // WEBP_HAVE_NEON

  assert(SharpYuvUpdateY != NULL);
  assert(SharpYuvUpdateRGB != NULL);
  assert(SharpYuvFilterRow != NULL);
}
