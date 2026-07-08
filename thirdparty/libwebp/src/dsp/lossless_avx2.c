// Copyright 2025 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// AVX2 variant of methods for lossless decoder
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_AVX2)

#include <stddef.h>
#include <immintrin.h>

#include "src/dsp/cpu.h"
#include "src/dsp/lossless.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// Predictor Transform

static WEBP_INLINE void Average2_m256i(const __m256i* const a0,
                                       const __m256i* const a1,
                                       __m256i* const avg) {
  // (a + b) >> 1 = ((a + b + 1) >> 1) - ((a ^ b) & 1)
  const __m256i ones = _mm256_set1_epi8(1);
  const __m256i avg1 = _mm256_avg_epu8(*a0, *a1);
  const __m256i one = _mm256_and_si256(_mm256_xor_si256(*a0, *a1), ones);
  *avg = _mm256_sub_epi8(avg1, one);
}

// Batch versions of those functions.

// Predictor0: ARGB_BLACK.
static void PredictorAdd0_AVX2(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m256i black = _mm256_set1_epi32((int)ARGB_BLACK);
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    const __m256i res = _mm256_add_epi8(src, black);
    _mm256_storeu_si256((__m256i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_SSE[0](in + i, NULL, num_pixels - i, out + i);
  }
  (void)upper;
}

// Predictor1: left.
static void PredictorAdd1_AVX2(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  __m256i prev = _mm256_set1_epi32((int)out[-1]);
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    // h | g | f | e | d | c | b | a
    const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    // g | f | e | 0 | c | b | a | 0
    const __m256i shift0 = _mm256_slli_si256(src, 4);
    // g + h | f + g | e + f | e | c + d | b + c | a + b | a
    const __m256i sum0 = _mm256_add_epi8(src, shift0);
    // e + f | e | 0 | 0 | a + b | a | 0 | 0
    const __m256i shift1 = _mm256_slli_si256(sum0, 8);
    // e + f + g + h | e + f + g | e + f | e | a + b + c + d | a + b + c | a + b
    // | a
    const __m256i sum1 = _mm256_add_epi8(sum0, shift1);
    // Add a + b + c + d to the upper lane.
    const int32_t sum_abcd = _mm256_extract_epi32(sum1, 3);
    const __m256i sum2 = _mm256_add_epi8(
        sum1,
        _mm256_set_epi32(sum_abcd, sum_abcd, sum_abcd, sum_abcd, 0, 0, 0, 0));

    const __m256i res = _mm256_add_epi8(sum2, prev);
    _mm256_storeu_si256((__m256i*)&out[i], res);
    // replicate last res output in prev.
    prev = _mm256_permutevar8x32_epi32(
        res, _mm256_set_epi32(7, 7, 7, 7, 7, 7, 7, 7));
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_SSE[1](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Macro that adds 32-bit integers from IN using mod 256 arithmetic
// per 8 bit channel.
#define GENERATE_PREDICTOR_1(X, IN)                                         \
  static void PredictorAdd##X##_AVX2(const uint32_t* in,                    \
                                     const uint32_t* upper, int num_pixels, \
                                     uint32_t* WEBP_RESTRICT out) {         \
    int i;                                                                  \
    for (i = 0; i + 8 <= num_pixels; i += 8) {                              \
      const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);       \
      const __m256i other = _mm256_loadu_si256((const __m256i*)&(IN));      \
      const __m256i res = _mm256_add_epi8(src, other);                      \
      _mm256_storeu_si256((__m256i*)&out[i], res);                          \
    }                                                                       \
    if (i != num_pixels) {                                                  \
      VP8LPredictorsAdd_SSE[(X)](in + i, upper + i, num_pixels - i, out + i); \
    }                                                                       \
  }

// Predictor2: Top.
GENERATE_PREDICTOR_1(2, upper[i])
// Predictor3: Top-right.
GENERATE_PREDICTOR_1(3, upper[i + 1])
// Predictor4: Top-left.
GENERATE_PREDICTOR_1(4, upper[i - 1])
#undef GENERATE_PREDICTOR_1

// Due to averages with integers, values cannot be accumulated in parallel for
// predictors 5 to 7.

#define GENERATE_PREDICTOR_2(X, IN)                                         \
  static void PredictorAdd##X##_AVX2(const uint32_t* in,                    \
                                     const uint32_t* upper, int num_pixels, \
                                     uint32_t* WEBP_RESTRICT out) {         \
    int i;                                                                  \
    for (i = 0; i + 8 <= num_pixels; i += 8) {                              \
      const __m256i Tother = _mm256_loadu_si256((const __m256i*)&(IN));     \
      const __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);      \
      const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);       \
      __m256i avg, res;                                                     \
      Average2_m256i(&T, &Tother, &avg);                                    \
      res = _mm256_add_epi8(avg, src);                                      \
      _mm256_storeu_si256((__m256i*)&out[i], res);                          \
    }                                                                       \
    if (i != num_pixels) {                                                  \
      VP8LPredictorsAdd_SSE[(X)](in + i, upper + i, num_pixels - i, out + i); \
    }                                                                       \
  }
// Predictor8: average TL T.
GENERATE_PREDICTOR_2(8, upper[i - 1])
// Predictor9: average T TR.
GENERATE_PREDICTOR_2(9, upper[i + 1])
#undef GENERATE_PREDICTOR_2

// Predictor10: average of (average of (L,TL), average of (T, TR)).
#define DO_PRED10(OUT)                                  \
  do {                                                  \
    __m256i avgLTL, avg;                                \
    Average2_m256i(&L, &TL, &avgLTL);                   \
    Average2_m256i(&avgTTR, &avgLTL, &avg);             \
    L = _mm256_add_epi8(avg, src);                      \
    out[i + (OUT)] = (uint32_t)_mm256_cvtsi256_si32(L); \
  } while (0)

#define DO_PRED10_SHIFT                                         \
  do {                                                          \
    /* Rotate the pre-computed values for the next iteration.*/ \
    avgTTR = _mm256_srli_si256(avgTTR, 4);                      \
    TL = _mm256_srli_si256(TL, 4);                              \
    src = _mm256_srli_si256(src, 4);                            \
  } while (0)

static void PredictorAdd10_AVX2(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i, j;
  __m256i L = _mm256_setr_epi32((int)out[-1], 0, 0, 0, 0, 0, 0, 0);
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    __m256i TL = _mm256_loadu_si256((const __m256i*)&upper[i - 1]);
    const __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);
    const __m256i TR = _mm256_loadu_si256((const __m256i*)&upper[i + 1]);
    __m256i avgTTR;
    Average2_m256i(&T, &TR, &avgTTR);
    {
      const __m256i avgTTR_bak = avgTTR;
      const __m256i TL_bak = TL;
      const __m256i src_bak = src;
      for (j = 0; j < 4; ++j) {
        DO_PRED10(j);
        DO_PRED10_SHIFT;
      }
      avgTTR = _mm256_permute2x128_si256(avgTTR_bak, avgTTR_bak, 1);
      TL = _mm256_permute2x128_si256(TL_bak, TL_bak, 1);
      src = _mm256_permute2x128_si256(src_bak, src_bak, 1);
      for (; j < 8; ++j) {
        DO_PRED10(j);
        DO_PRED10_SHIFT;
      }
    }
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_SSE[10](in + i, upper + i, num_pixels - i, out + i);
  }
}
#undef DO_PRED10
#undef DO_PRED10_SHIFT

// Predictor11: select.
#define DO_PRED11(OUT)                                                      \
  do {                                                                      \
    const __m256i L_lo = _mm256_unpacklo_epi32(L, T);                       \
    const __m256i TL_lo = _mm256_unpacklo_epi32(TL, T);                     \
    const __m256i pb = _mm256_sad_epu8(L_lo, TL_lo); /* pb = sum |L-TL|*/   \
    const __m256i mask = _mm256_cmpgt_epi32(pb, pa);                        \
    const __m256i A = _mm256_and_si256(mask, L);                            \
    const __m256i B = _mm256_andnot_si256(mask, T);                         \
    const __m256i pred = _mm256_or_si256(A, B); /* pred = (pa > b)? L : T*/ \
    L = _mm256_add_epi8(src, pred);                                         \
    out[i + (OUT)] = (uint32_t)_mm256_cvtsi256_si32(L);                     \
  } while (0)

#define DO_PRED11_SHIFT                                       \
  do {                                                        \
    /* Shift the pre-computed value for the next iteration.*/ \
    T = _mm256_srli_si256(T, 4);                              \
    TL = _mm256_srli_si256(TL, 4);                            \
    src = _mm256_srli_si256(src, 4);                          \
    pa = _mm256_srli_si256(pa, 4);                            \
  } while (0)

static void PredictorAdd11_AVX2(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i, j;
  __m256i pa;
  __m256i L = _mm256_setr_epi32((int)out[-1], 0, 0, 0, 0, 0, 0, 0);
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);
    __m256i TL = _mm256_loadu_si256((const __m256i*)&upper[i - 1]);
    __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    {
      // We can unpack with any value on the upper 32 bits, provided it's the
      // same on both operands (so that their sum of abs diff is zero). Here we
      // use T.
      const __m256i T_lo = _mm256_unpacklo_epi32(T, T);
      const __m256i TL_lo = _mm256_unpacklo_epi32(TL, T);
      const __m256i T_hi = _mm256_unpackhi_epi32(T, T);
      const __m256i TL_hi = _mm256_unpackhi_epi32(TL, T);
      const __m256i s_lo = _mm256_sad_epu8(T_lo, TL_lo);
      const __m256i s_hi = _mm256_sad_epu8(T_hi, TL_hi);
      pa = _mm256_packs_epi32(s_lo, s_hi);  // pa = sum |T-TL|
    }
    {
      const __m256i T_bak = T;
      const __m256i TL_bak = TL;
      const __m256i src_bak = src;
      const __m256i pa_bak = pa;
      for (j = 0; j < 4; ++j) {
        DO_PRED11(j);
        DO_PRED11_SHIFT;
      }
      T = _mm256_permute2x128_si256(T_bak, T_bak, 1);
      TL = _mm256_permute2x128_si256(TL_bak, TL_bak, 1);
      src = _mm256_permute2x128_si256(src_bak, src_bak, 1);
      pa = _mm256_permute2x128_si256(pa_bak, pa_bak, 1);
      for (; j < 8; ++j) {
        DO_PRED11(j);
        DO_PRED11_SHIFT;
      }
    }
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_SSE[11](in + i, upper + i, num_pixels - i, out + i);
  }
}
#undef DO_PRED11
#undef DO_PRED11_SHIFT

// Predictor12: ClampedAddSubtractFull.
#define DO_PRED12(DIFF, OUT)                              \
  do {                                                    \
    const __m256i all = _mm256_add_epi16(L, (DIFF));      \
    const __m256i alls = _mm256_packus_epi16(all, all);   \
    const __m256i res = _mm256_add_epi8(src, alls);       \
    out[i + (OUT)] = (uint32_t)_mm256_cvtsi256_si32(res); \
    L = _mm256_unpacklo_epi8(res, zero);                  \
  } while (0)

#define DO_PRED12_SHIFT(DIFF, LANE)                           \
  do {                                                        \
    /* Shift the pre-computed value for the next iteration.*/ \
    if ((LANE) == 0) (DIFF) = _mm256_srli_si256(DIFF, 8);     \
    src = _mm256_srli_si256(src, 4);                          \
  } while (0)

static void PredictorAdd12_AVX2(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m256i zero = _mm256_setzero_si256();
  const __m256i L8 = _mm256_setr_epi32((int)out[-1], 0, 0, 0, 0, 0, 0, 0);
  __m256i L = _mm256_unpacklo_epi8(L8, zero);
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    // Load 8 pixels at a time.
    __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    const __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);
    const __m256i T_lo = _mm256_unpacklo_epi8(T, zero);
    const __m256i T_hi = _mm256_unpackhi_epi8(T, zero);
    const __m256i TL = _mm256_loadu_si256((const __m256i*)&upper[i - 1]);
    const __m256i TL_lo = _mm256_unpacklo_epi8(TL, zero);
    const __m256i TL_hi = _mm256_unpackhi_epi8(TL, zero);
    __m256i diff_lo = _mm256_sub_epi16(T_lo, TL_lo);
    __m256i diff_hi = _mm256_sub_epi16(T_hi, TL_hi);
    const __m256i diff_lo_bak = diff_lo;
    const __m256i diff_hi_bak = diff_hi;
    const __m256i src_bak = src;
    DO_PRED12(diff_lo, 0);
    DO_PRED12_SHIFT(diff_lo, 0);
    DO_PRED12(diff_lo, 1);
    DO_PRED12_SHIFT(diff_lo, 0);
    DO_PRED12(diff_hi, 2);
    DO_PRED12_SHIFT(diff_hi, 0);
    DO_PRED12(diff_hi, 3);
    DO_PRED12_SHIFT(diff_hi, 0);

    // Process the upper lane.
    diff_lo = _mm256_permute2x128_si256(diff_lo_bak, diff_lo_bak, 1);
    diff_hi = _mm256_permute2x128_si256(diff_hi_bak, diff_hi_bak, 1);
    src = _mm256_permute2x128_si256(src_bak, src_bak, 1);

    DO_PRED12(diff_lo, 4);
    DO_PRED12_SHIFT(diff_lo, 0);
    DO_PRED12(diff_lo, 5);
    DO_PRED12_SHIFT(diff_lo, 1);
    DO_PRED12(diff_hi, 6);
    DO_PRED12_SHIFT(diff_hi, 0);
    DO_PRED12(diff_hi, 7);
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_SSE[12](in + i, upper + i, num_pixels - i, out + i);
  }
}
#undef DO_PRED12
#undef DO_PRED12_SHIFT

// Due to averages with integers, values cannot be accumulated in parallel for
// predictors 13.

//------------------------------------------------------------------------------
// Subtract-Green Transform

static void AddGreenToBlueAndRed_AVX2(const uint32_t* const src, int num_pixels,
                                      uint32_t* dst) {
  int i;
  const __m256i kCstShuffle = _mm256_set_epi8(
      -1, 29, -1, 29, -1, 25, -1, 25, -1, 21, -1, 21, -1, 17, -1, 17, -1, 13,
      -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1);
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i in = _mm256_loadu_si256((const __m256i*)&src[i]);  // argb
    const __m256i in_0g0g = _mm256_shuffle_epi8(in, kCstShuffle);    // 0g0g
    const __m256i out = _mm256_add_epi8(in, in_0g0g);
    _mm256_storeu_si256((__m256i*)&dst[i], out);
  }
  // fallthrough and finish off with SSE.
  if (i != num_pixels) {
    VP8LAddGreenToBlueAndRed_SSE(src + i, num_pixels - i, dst + i);
  }
}

//------------------------------------------------------------------------------
// Color Transform

static void TransformColorInverse_AVX2(const VP8LMultipliers* const m,
                                       const uint32_t* const src,
                                       int num_pixels, uint32_t* dst) {
// sign-extended multiplying constants, pre-shifted by 5.
#define CST(X)  (((int16_t)(m->X << 8)) >> 5)   // sign-extend
  const __m256i mults_rb =
      _mm256_set1_epi32((int)((uint32_t)CST(green_to_red) << 16 |
                              (CST(green_to_blue) & 0xffff)));
  const __m256i mults_b2 = _mm256_set1_epi32(CST(red_to_blue));
#undef CST
  const __m256i mask_ag = _mm256_set1_epi32((int)0xff00ff00);
  const __m256i perm1 = _mm256_setr_epi8(
      -1, 1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13, -1, 17, -1, 17,
      -1, 21, -1, 21, -1, 25, -1, 25, -1, 29, -1, 29);
  const __m256i perm2 = _mm256_setr_epi8(
      -1, 2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1, 18, -1,
      -1, -1, 22, -1, -1, -1, 26, -1, -1, -1, 30, -1, -1);
  int i;
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i A = _mm256_loadu_si256((const __m256i*)(src + i));
    const __m256i B = _mm256_shuffle_epi8(A, perm1);  // argb -> g0g0
    const __m256i C = _mm256_mulhi_epi16(B, mults_rb);
    const __m256i D = _mm256_add_epi8(A, C);
    const __m256i E = _mm256_shuffle_epi8(D, perm2);
    const __m256i F = _mm256_mulhi_epi16(E, mults_b2);
    const __m256i G = _mm256_add_epi8(D, F);
    const __m256i out = _mm256_blendv_epi8(G, A, mask_ag);
    _mm256_storeu_si256((__m256i*)&dst[i], out);
  }
  // Fall-back to SSE-version for left-overs.
  if (i != num_pixels) {
    VP8LTransformColorInverse_SSE(m, src + i, num_pixels - i, dst + i);
  }
}

//------------------------------------------------------------------------------
// Color-space conversion functions

static void ConvertBGRAToRGBA_AVX2(const uint32_t* WEBP_RESTRICT src,
                                   int num_pixels, uint8_t* WEBP_RESTRICT dst) {
  const __m256i* in = (const __m256i*)src;
  __m256i* out = (__m256i*)dst;
  while (num_pixels >= 8) {
    const __m256i A = _mm256_loadu_si256(in++);
    const __m256i B = _mm256_shuffle_epi8(
        A,
        _mm256_set_epi8(15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2,
                        15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2));
    _mm256_storeu_si256(out++, B);
    num_pixels -= 8;
  }
  // left-overs
  if (num_pixels > 0) {
    VP8LConvertBGRAToRGBA_SSE((const uint32_t*)in, num_pixels, (uint8_t*)out);
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LDspInitAVX2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LDspInitAVX2(void) {
  VP8LPredictorsAdd[0] = PredictorAdd0_AVX2;
  VP8LPredictorsAdd[1] = PredictorAdd1_AVX2;
  VP8LPredictorsAdd[2] = PredictorAdd2_AVX2;
  VP8LPredictorsAdd[3] = PredictorAdd3_AVX2;
  VP8LPredictorsAdd[4] = PredictorAdd4_AVX2;
  VP8LPredictorsAdd[8] = PredictorAdd8_AVX2;
  VP8LPredictorsAdd[9] = PredictorAdd9_AVX2;
  VP8LPredictorsAdd[10] = PredictorAdd10_AVX2;
  VP8LPredictorsAdd[11] = PredictorAdd11_AVX2;
  VP8LPredictorsAdd[12] = PredictorAdd12_AVX2;

  VP8LAddGreenToBlueAndRed = AddGreenToBlueAndRed_AVX2;
  VP8LTransformColorInverse = TransformColorInverse_AVX2;
  VP8LConvertBGRAToRGBA = ConvertBGRAToRGBA_AVX2;
}

#else  // !WEBP_USE_AVX2

WEBP_DSP_INIT_STUB(VP8LDspInitAVX2)

#endif  // WEBP_USE_AVX2
