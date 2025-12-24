// Copyright 2025 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// AVX2 variant of methods for lossless encoder
//
// Author: Vincent Rabaud (vrabaud@google.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_AVX2)
#include <emmintrin.h>
#include <immintrin.h>

#include <assert.h>
#include <stddef.h>

#include "src/dsp/cpu.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// Subtract-Green Transform

static void SubtractGreenFromBlueAndRed_AVX2(uint32_t* argb_data,
                                             int num_pixels) {
  int i;
  const __m256i kCstShuffle = _mm256_set_epi8(
      -1, 29, -1, 29, -1, 25, -1, 25, -1, 21, -1, 21, -1, 17, -1, 17, -1, 13,
      -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1);
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i in = _mm256_loadu_si256((__m256i*)&argb_data[i]);  // argb
    const __m256i in_0g0g = _mm256_shuffle_epi8(in, kCstShuffle);
    const __m256i out = _mm256_sub_epi8(in, in_0g0g);
    _mm256_storeu_si256((__m256i*)&argb_data[i], out);
  }
  // fallthrough and finish off with plain-SSE
  if (i != num_pixels) {
    VP8LSubtractGreenFromBlueAndRed_SSE(argb_data + i, num_pixels - i);
  }
}

//------------------------------------------------------------------------------
// Color Transform

// For sign-extended multiplying constants, pre-shifted by 5:
#define CST_5b(X) (((int16_t)((uint16_t)(X) << 8)) >> 5)

#define MK_CST_16(HI, LO) \
  _mm256_set1_epi32((int)(((uint32_t)(HI) << 16) | ((LO) & 0xffff)))

static void TransformColor_AVX2(const VP8LMultipliers* WEBP_RESTRICT const m,
                                uint32_t* WEBP_RESTRICT argb_data,
                                int num_pixels) {
  const __m256i mults_rb =
      MK_CST_16(CST_5b(m->green_to_red), CST_5b(m->green_to_blue));
  const __m256i mults_b2 = MK_CST_16(CST_5b(m->red_to_blue), 0);
  const __m256i mask_rb = _mm256_set1_epi32(0x00ff00ff);  // red-blue masks
  const __m256i kCstShuffle = _mm256_set_epi8(
      29, -1, 29, -1, 25, -1, 25, -1, 21, -1, 21, -1, 17, -1, 17, -1, 13, -1,
      13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1);
  int i;
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i in = _mm256_loadu_si256((__m256i*)&argb_data[i]);  // argb
    const __m256i A = _mm256_shuffle_epi8(in, kCstShuffle);          // g0g0
    const __m256i B = _mm256_mulhi_epi16(A, mults_rb);  // x dr  x db1
    const __m256i C = _mm256_slli_epi16(in, 8);         // r 0   b   0
    const __m256i D = _mm256_mulhi_epi16(C, mults_b2);  // x db2 0   0
    const __m256i E = _mm256_srli_epi32(D, 16);         // 0 0   x db2
    const __m256i F = _mm256_add_epi8(E, B);            // x dr  x  db
    const __m256i G = _mm256_and_si256(F, mask_rb);     // 0 dr  0  db
    const __m256i out = _mm256_sub_epi8(in, G);
    _mm256_storeu_si256((__m256i*)&argb_data[i], out);
  }
  // fallthrough and finish off with plain-C
  if (i != num_pixels) {
    VP8LTransformColor_SSE(m, argb_data + i, num_pixels - i);
  }
}

//------------------------------------------------------------------------------
#define SPAN 16
static void CollectColorBlueTransforms_AVX2(const uint32_t* WEBP_RESTRICT argb,
                                            int stride, int tile_width,
                                            int tile_height, int green_to_blue,
                                            int red_to_blue, uint32_t histo[]) {
  const __m256i mult =
      MK_CST_16(CST_5b(red_to_blue) + 256, CST_5b(green_to_blue));
  const __m256i perm = _mm256_setr_epi8(
      -1, 1, -1, 2, -1, 5, -1, 6, -1, 9, -1, 10, -1, 13, -1, 14, -1, 17, -1, 18,
      -1, 21, -1, 22, -1, 25, -1, 26, -1, 29, -1, 30);
  if (tile_width >= 8) {
    int y, i;
    for (y = 0; y < tile_height; ++y) {
      uint8_t values[32];
      const uint32_t* const src = argb + y * stride;
      const __m256i A1 = _mm256_loadu_si256((const __m256i*)src);
      const __m256i B1 = _mm256_shuffle_epi8(A1, perm);
      const __m256i C1 = _mm256_mulhi_epi16(B1, mult);
      const __m256i D1 = _mm256_sub_epi16(A1, C1);
      __m256i E = _mm256_add_epi16(_mm256_srli_epi32(D1, 16), D1);
      int x;
      for (x = 8; x + 8 <= tile_width; x += 8) {
        const __m256i A2 = _mm256_loadu_si256((const __m256i*)(src + x));
        __m256i B2, C2, D2;
        _mm256_storeu_si256((__m256i*)values, E);
        for (i = 0; i < 32; i += 4) ++histo[values[i]];
        B2 = _mm256_shuffle_epi8(A2, perm);
        C2 = _mm256_mulhi_epi16(B2, mult);
        D2 = _mm256_sub_epi16(A2, C2);
        E = _mm256_add_epi16(_mm256_srli_epi32(D2, 16), D2);
      }
      _mm256_storeu_si256((__m256i*)values, E);
      for (i = 0; i < 32; i += 4) ++histo[values[i]];
    }
  }
  {
    const int left_over = tile_width & 7;
    if (left_over > 0) {
      VP8LCollectColorBlueTransforms_SSE(argb + tile_width - left_over, stride,
                                         left_over, tile_height, green_to_blue,
                                         red_to_blue, histo);
    }
  }
}

static void CollectColorRedTransforms_AVX2(const uint32_t* WEBP_RESTRICT argb,
                                           int stride, int tile_width,
                                           int tile_height, int green_to_red,
                                           uint32_t histo[]) {
  const __m256i mult = MK_CST_16(0, CST_5b(green_to_red));
  const __m256i mask_g = _mm256_set1_epi32(0x0000ff00);
  if (tile_width >= 8) {
    int y, i;
    for (y = 0; y < tile_height; ++y) {
      uint8_t values[32];
      const uint32_t* const src = argb + y * stride;
      const __m256i A1 = _mm256_loadu_si256((const __m256i*)src);
      const __m256i B1 = _mm256_and_si256(A1, mask_g);
      const __m256i C1 = _mm256_madd_epi16(B1, mult);
      __m256i D = _mm256_sub_epi16(A1, C1);
      int x;
      for (x = 8; x + 8 <= tile_width; x += 8) {
        const __m256i A2 = _mm256_loadu_si256((const __m256i*)(src + x));
        __m256i B2, C2;
        _mm256_storeu_si256((__m256i*)values, D);
        for (i = 2; i < 32; i += 4) ++histo[values[i]];
        B2 = _mm256_and_si256(A2, mask_g);
        C2 = _mm256_madd_epi16(B2, mult);
        D = _mm256_sub_epi16(A2, C2);
      }
      _mm256_storeu_si256((__m256i*)values, D);
      for (i = 2; i < 32; i += 4) ++histo[values[i]];
    }
  }
  {
    const int left_over = tile_width & 7;
    if (left_over > 0) {
      VP8LCollectColorRedTransforms_SSE(argb + tile_width - left_over, stride,
                                        left_over, tile_height, green_to_red,
                                        histo);
    }
  }
}
#undef SPAN
#undef MK_CST_16

//------------------------------------------------------------------------------

// Note we are adding uint32_t's as *signed* int32's (using _mm256_add_epi32).
// But that's ok since the histogram values are less than 1<<28 (max picture
// size).
static void AddVector_AVX2(const uint32_t* WEBP_RESTRICT a,
                           const uint32_t* WEBP_RESTRICT b,
                           uint32_t* WEBP_RESTRICT out, int size) {
  int i = 0;
  int aligned_size = size & ~31;
  // Size is, at minimum, NUM_DISTANCE_CODES (40) and may be as large as
  // NUM_LITERAL_CODES (256) + NUM_LENGTH_CODES (24) + (0 or a non-zero power of
  // 2). See the usage in VP8LHistogramAdd().
  assert(size >= 32);
  assert(size % 2 == 0);

  do {
    const __m256i a0 = _mm256_loadu_si256((const __m256i*)&a[i + 0]);
    const __m256i a1 = _mm256_loadu_si256((const __m256i*)&a[i + 8]);
    const __m256i a2 = _mm256_loadu_si256((const __m256i*)&a[i + 16]);
    const __m256i a3 = _mm256_loadu_si256((const __m256i*)&a[i + 24]);
    const __m256i b0 = _mm256_loadu_si256((const __m256i*)&b[i + 0]);
    const __m256i b1 = _mm256_loadu_si256((const __m256i*)&b[i + 8]);
    const __m256i b2 = _mm256_loadu_si256((const __m256i*)&b[i + 16]);
    const __m256i b3 = _mm256_loadu_si256((const __m256i*)&b[i + 24]);
    _mm256_storeu_si256((__m256i*)&out[i + 0], _mm256_add_epi32(a0, b0));
    _mm256_storeu_si256((__m256i*)&out[i + 8], _mm256_add_epi32(a1, b1));
    _mm256_storeu_si256((__m256i*)&out[i + 16], _mm256_add_epi32(a2, b2));
    _mm256_storeu_si256((__m256i*)&out[i + 24], _mm256_add_epi32(a3, b3));
    i += 32;
  } while (i != aligned_size);

  if ((size & 16) != 0) {
    const __m256i a0 = _mm256_loadu_si256((const __m256i*)&a[i + 0]);
    const __m256i a1 = _mm256_loadu_si256((const __m256i*)&a[i + 8]);
    const __m256i b0 = _mm256_loadu_si256((const __m256i*)&b[i + 0]);
    const __m256i b1 = _mm256_loadu_si256((const __m256i*)&b[i + 8]);
    _mm256_storeu_si256((__m256i*)&out[i + 0], _mm256_add_epi32(a0, b0));
    _mm256_storeu_si256((__m256i*)&out[i + 8], _mm256_add_epi32(a1, b1));
    i += 16;
  }

  size &= 15;
  if (size == 8) {
    const __m256i a0 = _mm256_loadu_si256((const __m256i*)&a[i]);
    const __m256i b0 = _mm256_loadu_si256((const __m256i*)&b[i]);
    _mm256_storeu_si256((__m256i*)&out[i], _mm256_add_epi32(a0, b0));
  } else {
    for (; size--; ++i) {
      out[i] = a[i] + b[i];
    }
  }
}

static void AddVectorEq_AVX2(const uint32_t* WEBP_RESTRICT a,
                             uint32_t* WEBP_RESTRICT out, int size) {
  int i = 0;
  int aligned_size = size & ~31;
  // Size is, at minimum, NUM_DISTANCE_CODES (40) and may be as large as
  // NUM_LITERAL_CODES (256) + NUM_LENGTH_CODES (24) + (0 or a non-zero power of
  // 2). See the usage in VP8LHistogramAdd().
  assert(size >= 32);
  assert(size % 2 == 0);

  do {
    const __m256i a0 = _mm256_loadu_si256((const __m256i*)&a[i + 0]);
    const __m256i a1 = _mm256_loadu_si256((const __m256i*)&a[i + 8]);
    const __m256i a2 = _mm256_loadu_si256((const __m256i*)&a[i + 16]);
    const __m256i a3 = _mm256_loadu_si256((const __m256i*)&a[i + 24]);
    const __m256i b0 = _mm256_loadu_si256((const __m256i*)&out[i + 0]);
    const __m256i b1 = _mm256_loadu_si256((const __m256i*)&out[i + 8]);
    const __m256i b2 = _mm256_loadu_si256((const __m256i*)&out[i + 16]);
    const __m256i b3 = _mm256_loadu_si256((const __m256i*)&out[i + 24]);
    _mm256_storeu_si256((__m256i*)&out[i + 0], _mm256_add_epi32(a0, b0));
    _mm256_storeu_si256((__m256i*)&out[i + 8], _mm256_add_epi32(a1, b1));
    _mm256_storeu_si256((__m256i*)&out[i + 16], _mm256_add_epi32(a2, b2));
    _mm256_storeu_si256((__m256i*)&out[i + 24], _mm256_add_epi32(a3, b3));
    i += 32;
  } while (i != aligned_size);

  if ((size & 16) != 0) {
    const __m256i a0 = _mm256_loadu_si256((const __m256i*)&a[i + 0]);
    const __m256i a1 = _mm256_loadu_si256((const __m256i*)&a[i + 8]);
    const __m256i b0 = _mm256_loadu_si256((const __m256i*)&out[i + 0]);
    const __m256i b1 = _mm256_loadu_si256((const __m256i*)&out[i + 8]);
    _mm256_storeu_si256((__m256i*)&out[i + 0], _mm256_add_epi32(a0, b0));
    _mm256_storeu_si256((__m256i*)&out[i + 8], _mm256_add_epi32(a1, b1));
    i += 16;
  }

  size &= 15;
  if (size == 8) {
    const __m256i a0 = _mm256_loadu_si256((const __m256i*)&a[i]);
    const __m256i b0 = _mm256_loadu_si256((const __m256i*)&out[i]);
    _mm256_storeu_si256((__m256i*)&out[i], _mm256_add_epi32(a0, b0));
  } else {
    for (; size--; ++i) {
      out[i] += a[i];
    }
  }
}

//------------------------------------------------------------------------------
// Entropy

#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ)

static uint64_t CombinedShannonEntropy_AVX2(const uint32_t X[256],
                                            const uint32_t Y[256]) {
  int i;
  uint64_t retval = 0;
  uint32_t sumX = 0, sumXY = 0;
  const __m256i zero = _mm256_setzero_si256();

  for (i = 0; i < 256; i += 32) {
    const __m256i x0 = _mm256_loadu_si256((const __m256i*)(X + i + 0));
    const __m256i y0 = _mm256_loadu_si256((const __m256i*)(Y + i + 0));
    const __m256i x1 = _mm256_loadu_si256((const __m256i*)(X + i + 8));
    const __m256i y1 = _mm256_loadu_si256((const __m256i*)(Y + i + 8));
    const __m256i x2 = _mm256_loadu_si256((const __m256i*)(X + i + 16));
    const __m256i y2 = _mm256_loadu_si256((const __m256i*)(Y + i + 16));
    const __m256i x3 = _mm256_loadu_si256((const __m256i*)(X + i + 24));
    const __m256i y3 = _mm256_loadu_si256((const __m256i*)(Y + i + 24));
    const __m256i x4 = _mm256_packs_epi16(_mm256_packs_epi32(x0, x1),
                                          _mm256_packs_epi32(x2, x3));
    const __m256i y4 = _mm256_packs_epi16(_mm256_packs_epi32(y0, y1),
                                          _mm256_packs_epi32(y2, y3));
    // Packed pixels are actually in order: ... 17 16 12 11 10 9 8 3 2 1 0
    const __m256i x5 = _mm256_permutevar8x32_epi32(
        x4, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    const __m256i y5 = _mm256_permutevar8x32_epi32(
        y4, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    const uint32_t mx =
        (uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(x5, zero));
    uint32_t my =
        (uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(y5, zero)) | mx;
    while (my) {
      const int32_t j = BitsCtz(my);
      uint32_t xy;
      if ((mx >> j) & 1) {
        const int x = X[i + j];
        sumXY += x;
        retval += VP8LFastSLog2(x);
      }
      xy = X[i + j] + Y[i + j];
      sumX += xy;
      retval += VP8LFastSLog2(xy);
      my &= my - 1;
    }
  }
  retval = VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY) - retval;
  return retval;
}

#else

#define DONT_USE_COMBINED_SHANNON_ENTROPY_SSE2_FUNC   // won't be faster

#endif

//------------------------------------------------------------------------------

static int VectorMismatch_AVX2(const uint32_t* const array1,
                               const uint32_t* const array2, int length) {
  int match_len;

  if (length >= 24) {
    __m256i A0 = _mm256_loadu_si256((const __m256i*)&array1[0]);
    __m256i A1 = _mm256_loadu_si256((const __m256i*)&array2[0]);
    match_len = 0;
    do {
      // Loop unrolling and early load both provide a speedup of 10% for the
      // current function. Also, max_limit can be MAX_LENGTH=4096 at most.
      const __m256i cmpA = _mm256_cmpeq_epi32(A0, A1);
      const __m256i B0 =
          _mm256_loadu_si256((const __m256i*)&array1[match_len + 8]);
      const __m256i B1 =
          _mm256_loadu_si256((const __m256i*)&array2[match_len + 8]);
      if ((uint32_t)_mm256_movemask_epi8(cmpA) != 0xffffffff) break;
      match_len += 8;

      {
        const __m256i cmpB = _mm256_cmpeq_epi32(B0, B1);
        A0 = _mm256_loadu_si256((const __m256i*)&array1[match_len + 8]);
        A1 = _mm256_loadu_si256((const __m256i*)&array2[match_len + 8]);
        if ((uint32_t)_mm256_movemask_epi8(cmpB) != 0xffffffff) break;
        match_len += 8;
      }
    } while (match_len + 24 < length);
  } else {
    match_len = 0;
    // Unroll the potential first two loops.
    if (length >= 8 &&
        (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi32(
            _mm256_loadu_si256((const __m256i*)&array1[0]),
            _mm256_loadu_si256((const __m256i*)&array2[0]))) == 0xffffffff) {
      match_len = 8;
      if (length >= 16 &&
          (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi32(
              _mm256_loadu_si256((const __m256i*)&array1[8]),
              _mm256_loadu_si256((const __m256i*)&array2[8]))) == 0xffffffff) {
        match_len = 16;
      }
    }
  }

  while (match_len < length && array1[match_len] == array2[match_len]) {
    ++match_len;
  }
  return match_len;
}

// Bundles multiple (1, 2, 4 or 8) pixels into a single pixel.
static void BundleColorMap_AVX2(const uint8_t* WEBP_RESTRICT const row,
                                int width, int xbits,
                                uint32_t* WEBP_RESTRICT dst) {
  int x = 0;
  assert(xbits >= 0);
  assert(xbits <= 3);
  switch (xbits) {
    case 0: {
      const __m256i ff = _mm256_set1_epi16((short)0xff00);
      const __m256i zero = _mm256_setzero_si256();
      // Store 0xff000000 | (row[x] << 8).
      for (x = 0; x + 32 <= width; x += 32, dst += 32) {
        const __m256i in = _mm256_loadu_si256((const __m256i*)&row[x]);
        const __m256i in_lo = _mm256_unpacklo_epi8(zero, in);
        const __m256i dst0 = _mm256_unpacklo_epi16(in_lo, ff);
        const __m256i dst1 = _mm256_unpackhi_epi16(in_lo, ff);
        const __m256i in_hi = _mm256_unpackhi_epi8(zero, in);
        const __m256i dst2 = _mm256_unpacklo_epi16(in_hi, ff);
        const __m256i dst3 = _mm256_unpackhi_epi16(in_hi, ff);
        _mm256_storeu2_m128i((__m128i*)&dst[16], (__m128i*)&dst[0], dst0);
        _mm256_storeu2_m128i((__m128i*)&dst[20], (__m128i*)&dst[4], dst1);
        _mm256_storeu2_m128i((__m128i*)&dst[24], (__m128i*)&dst[8], dst2);
        _mm256_storeu2_m128i((__m128i*)&dst[28], (__m128i*)&dst[12], dst3);
      }
      break;
    }
    case 1: {
      const __m256i ff = _mm256_set1_epi16((short)0xff00);
      const __m256i mul = _mm256_set1_epi16(0x110);
      for (x = 0; x + 32 <= width; x += 32, dst += 16) {
        // 0a0b | (where a/b are 4 bits).
        const __m256i in = _mm256_loadu_si256((const __m256i*)&row[x]);
        const __m256i tmp = _mm256_mullo_epi16(in, mul);  // aba0
        const __m256i pack = _mm256_and_si256(tmp, ff);   // ab00
        const __m256i dst0 = _mm256_unpacklo_epi16(pack, ff);
        const __m256i dst1 = _mm256_unpackhi_epi16(pack, ff);
        _mm256_storeu2_m128i((__m128i*)&dst[8], (__m128i*)&dst[0], dst0);
        _mm256_storeu2_m128i((__m128i*)&dst[12], (__m128i*)&dst[4], dst1);
      }
      break;
    }
    case 2: {
      const __m256i mask_or = _mm256_set1_epi32((int)0xff000000);
      const __m256i mul_cst = _mm256_set1_epi16(0x0104);
      const __m256i mask_mul = _mm256_set1_epi16(0x0f00);
      for (x = 0; x + 32 <= width; x += 32, dst += 8) {
        // 000a000b000c000d | (where a/b/c/d are 2 bits).
        const __m256i in = _mm256_loadu_si256((const __m256i*)&row[x]);
        const __m256i mul =
            _mm256_mullo_epi16(in, mul_cst);  // 00ab00b000cd00d0
        const __m256i tmp =
            _mm256_and_si256(mul, mask_mul);               //  00ab000000cd0000
        const __m256i shift = _mm256_srli_epi32(tmp, 12);  // 00000000ab000000
        const __m256i pack = _mm256_or_si256(shift, tmp);  // 00000000abcd0000
        // Convert to 0xff00**00.
        const __m256i res = _mm256_or_si256(pack, mask_or);
        _mm256_storeu_si256((__m256i*)dst, res);
      }
      break;
    }
    default: {
      assert(xbits == 3);
      for (x = 0; x + 32 <= width; x += 32, dst += 4) {
        // 0000000a00000000b... | (where a/b are 1 bit).
        const __m256i in = _mm256_loadu_si256((const __m256i*)&row[x]);
        const __m256i shift = _mm256_slli_epi64(in, 7);
        const uint32_t move = _mm256_movemask_epi8(shift);
        dst[0] = 0xff000000 | ((move & 0xff) << 8);
        dst[1] = 0xff000000 | (move & 0xff00);
        dst[2] = 0xff000000 | ((move & 0xff0000) >> 8);
        dst[3] = 0xff000000 | ((move & 0xff000000) >> 16);
      }
      break;
    }
  }
  if (x != width) {
    VP8LBundleColorMap_SSE(row + x, width - x, xbits, dst);
  }
}

//------------------------------------------------------------------------------
// Batch version of Predictor Transform subtraction

static WEBP_INLINE void Average2_m256i(const __m256i* const a0,
                                       const __m256i* const a1,
                                       __m256i* const avg) {
  // (a + b) >> 1 = ((a + b + 1) >> 1) - ((a ^ b) & 1)
  const __m256i ones = _mm256_set1_epi8(1);
  const __m256i avg1 = _mm256_avg_epu8(*a0, *a1);
  const __m256i one = _mm256_and_si256(_mm256_xor_si256(*a0, *a1), ones);
  *avg = _mm256_sub_epi8(avg1, one);
}

// Predictor0: ARGB_BLACK.
static void PredictorSub0_AVX2(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m256i black = _mm256_set1_epi32((int)ARGB_BLACK);
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    const __m256i res = _mm256_sub_epi8(src, black);
    _mm256_storeu_si256((__m256i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[0](in + i, NULL, num_pixels - i, out + i);
  }
  (void)upper;
}

#define GENERATE_PREDICTOR_1(X, IN)                                          \
  static void PredictorSub##X##_AVX2(                                        \
      const uint32_t* const in, const uint32_t* const upper, int num_pixels, \
      uint32_t* WEBP_RESTRICT const out) {                                   \
    int i;                                                                   \
    for (i = 0; i + 8 <= num_pixels; i += 8) {                               \
      const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);        \
      const __m256i pred = _mm256_loadu_si256((const __m256i*)&(IN));        \
      const __m256i res = _mm256_sub_epi8(src, pred);                        \
      _mm256_storeu_si256((__m256i*)&out[i], res);                           \
    }                                                                        \
    if (i != num_pixels) {                                                   \
      VP8LPredictorsSub_SSE[(X)](in + i, WEBP_OFFSET_PTR(upper, i),          \
                                 num_pixels - i, out + i);                   \
    }                                                                        \
  }

GENERATE_PREDICTOR_1(1, in[i - 1])       // Predictor1: L
GENERATE_PREDICTOR_1(2, upper[i])        // Predictor2: T
GENERATE_PREDICTOR_1(3, upper[i + 1])    // Predictor3: TR
GENERATE_PREDICTOR_1(4, upper[i - 1])    // Predictor4: TL
#undef GENERATE_PREDICTOR_1

// Predictor5: avg2(avg2(L, TR), T)
static void PredictorSub5_AVX2(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i L = _mm256_loadu_si256((const __m256i*)&in[i - 1]);
    const __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);
    const __m256i TR = _mm256_loadu_si256((const __m256i*)&upper[i + 1]);
    const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    __m256i avg, pred, res;
    Average2_m256i(&L, &TR, &avg);
    Average2_m256i(&avg, &T, &pred);
    res = _mm256_sub_epi8(src, pred);
    _mm256_storeu_si256((__m256i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[5](in + i, upper + i, num_pixels - i, out + i);
  }
}

#define GENERATE_PREDICTOR_2(X, A, B)                                         \
  static void PredictorSub##X##_AVX2(const uint32_t* in,                      \
                                     const uint32_t* upper, int num_pixels,   \
                                     uint32_t* WEBP_RESTRICT out) {           \
    int i;                                                                    \
    for (i = 0; i + 8 <= num_pixels; i += 8) {                                \
      const __m256i tA = _mm256_loadu_si256((const __m256i*)&(A));            \
      const __m256i tB = _mm256_loadu_si256((const __m256i*)&(B));            \
      const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);         \
      __m256i pred, res;                                                      \
      Average2_m256i(&tA, &tB, &pred);                                        \
      res = _mm256_sub_epi8(src, pred);                                       \
      _mm256_storeu_si256((__m256i*)&out[i], res);                            \
    }                                                                         \
    if (i != num_pixels) {                                                    \
      VP8LPredictorsSub_SSE[(X)](in + i, upper + i, num_pixels - i, out + i); \
    }                                                                         \
  }

GENERATE_PREDICTOR_2(6, in[i - 1], upper[i - 1])   // Predictor6: avg(L, TL)
GENERATE_PREDICTOR_2(7, in[i - 1], upper[i])       // Predictor7: avg(L, T)
GENERATE_PREDICTOR_2(8, upper[i - 1], upper[i])    // Predictor8: avg(TL, T)
GENERATE_PREDICTOR_2(9, upper[i], upper[i + 1])    // Predictor9: average(T, TR)
#undef GENERATE_PREDICTOR_2

// Predictor10: avg(avg(L,TL), avg(T, TR)).
static void PredictorSub10_AVX2(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i L = _mm256_loadu_si256((const __m256i*)&in[i - 1]);
    const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    const __m256i TL = _mm256_loadu_si256((const __m256i*)&upper[i - 1]);
    const __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);
    const __m256i TR = _mm256_loadu_si256((const __m256i*)&upper[i + 1]);
    __m256i avgTTR, avgLTL, avg, res;
    Average2_m256i(&T, &TR, &avgTTR);
    Average2_m256i(&L, &TL, &avgLTL);
    Average2_m256i(&avgTTR, &avgLTL, &avg);
    res = _mm256_sub_epi8(src, avg);
    _mm256_storeu_si256((__m256i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[10](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor11: select.
static void GetSumAbsDiff32_AVX2(const __m256i* const A, const __m256i* const B,
                                 __m256i* const out) {
  // We can unpack with any value on the upper 32 bits, provided it's the same
  // on both operands (to that their sum of abs diff is zero). Here we use *A.
  const __m256i A_lo = _mm256_unpacklo_epi32(*A, *A);
  const __m256i B_lo = _mm256_unpacklo_epi32(*B, *A);
  const __m256i A_hi = _mm256_unpackhi_epi32(*A, *A);
  const __m256i B_hi = _mm256_unpackhi_epi32(*B, *A);
  const __m256i s_lo = _mm256_sad_epu8(A_lo, B_lo);
  const __m256i s_hi = _mm256_sad_epu8(A_hi, B_hi);
  *out = _mm256_packs_epi32(s_lo, s_hi);
}

static void PredictorSub11_AVX2(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i L = _mm256_loadu_si256((const __m256i*)&in[i - 1]);
    const __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);
    const __m256i TL = _mm256_loadu_si256((const __m256i*)&upper[i - 1]);
    const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    __m256i pa, pb;
    GetSumAbsDiff32_AVX2(&T, &TL, &pa);  // pa = sum |T-TL|
    GetSumAbsDiff32_AVX2(&L, &TL, &pb);  // pb = sum |L-TL|
    {
      const __m256i mask = _mm256_cmpgt_epi32(pb, pa);
      const __m256i A = _mm256_and_si256(mask, L);
      const __m256i B = _mm256_andnot_si256(mask, T);
      const __m256i pred = _mm256_or_si256(A, B);  // pred = (L > T)? L : T
      const __m256i res = _mm256_sub_epi8(src, pred);
      _mm256_storeu_si256((__m256i*)&out[i], res);
    }
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[11](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor12: ClampedSubSubtractFull.
static void PredictorSub12_AVX2(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m256i zero = _mm256_setzero_si256();
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    const __m256i L = _mm256_loadu_si256((const __m256i*)&in[i - 1]);
    const __m256i L_lo = _mm256_unpacklo_epi8(L, zero);
    const __m256i L_hi = _mm256_unpackhi_epi8(L, zero);
    const __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);
    const __m256i T_lo = _mm256_unpacklo_epi8(T, zero);
    const __m256i T_hi = _mm256_unpackhi_epi8(T, zero);
    const __m256i TL = _mm256_loadu_si256((const __m256i*)&upper[i - 1]);
    const __m256i TL_lo = _mm256_unpacklo_epi8(TL, zero);
    const __m256i TL_hi = _mm256_unpackhi_epi8(TL, zero);
    const __m256i diff_lo = _mm256_sub_epi16(T_lo, TL_lo);
    const __m256i diff_hi = _mm256_sub_epi16(T_hi, TL_hi);
    const __m256i pred_lo = _mm256_add_epi16(L_lo, diff_lo);
    const __m256i pred_hi = _mm256_add_epi16(L_hi, diff_hi);
    const __m256i pred = _mm256_packus_epi16(pred_lo, pred_hi);
    const __m256i res = _mm256_sub_epi8(src, pred);
    _mm256_storeu_si256((__m256i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[12](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictors13: ClampedAddSubtractHalf
static void PredictorSub13_AVX2(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m256i zero = _mm256_setzero_si256();
  for (i = 0; i + 8 <= num_pixels; i += 8) {
    const __m256i L = _mm256_loadu_si256((const __m256i*)&in[i - 1]);
    const __m256i src = _mm256_loadu_si256((const __m256i*)&in[i]);
    const __m256i T = _mm256_loadu_si256((const __m256i*)&upper[i]);
    const __m256i TL = _mm256_loadu_si256((const __m256i*)&upper[i - 1]);
    // lo.
    const __m256i L_lo = _mm256_unpacklo_epi8(L, zero);
    const __m256i T_lo = _mm256_unpacklo_epi8(T, zero);
    const __m256i TL_lo = _mm256_unpacklo_epi8(TL, zero);
    const __m256i sum_lo = _mm256_add_epi16(T_lo, L_lo);
    const __m256i avg_lo = _mm256_srli_epi16(sum_lo, 1);
    const __m256i A1_lo = _mm256_sub_epi16(avg_lo, TL_lo);
    const __m256i bit_fix_lo = _mm256_cmpgt_epi16(TL_lo, avg_lo);
    const __m256i A2_lo = _mm256_sub_epi16(A1_lo, bit_fix_lo);
    const __m256i A3_lo = _mm256_srai_epi16(A2_lo, 1);
    const __m256i A4_lo = _mm256_add_epi16(avg_lo, A3_lo);
    // hi.
    const __m256i L_hi = _mm256_unpackhi_epi8(L, zero);
    const __m256i T_hi = _mm256_unpackhi_epi8(T, zero);
    const __m256i TL_hi = _mm256_unpackhi_epi8(TL, zero);
    const __m256i sum_hi = _mm256_add_epi16(T_hi, L_hi);
    const __m256i avg_hi = _mm256_srli_epi16(sum_hi, 1);
    const __m256i A1_hi = _mm256_sub_epi16(avg_hi, TL_hi);
    const __m256i bit_fix_hi = _mm256_cmpgt_epi16(TL_hi, avg_hi);
    const __m256i A2_hi = _mm256_sub_epi16(A1_hi, bit_fix_hi);
    const __m256i A3_hi = _mm256_srai_epi16(A2_hi, 1);
    const __m256i A4_hi = _mm256_add_epi16(avg_hi, A3_hi);

    const __m256i pred = _mm256_packus_epi16(A4_lo, A4_hi);
    const __m256i res = _mm256_sub_epi8(src, pred);
    _mm256_storeu_si256((__m256i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[13](in + i, upper + i, num_pixels - i, out + i);
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LEncDspInitAVX2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitAVX2(void) {
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_AVX2;
  VP8LTransformColor = TransformColor_AVX2;
  VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_AVX2;
  VP8LCollectColorRedTransforms = CollectColorRedTransforms_AVX2;
  VP8LAddVector = AddVector_AVX2;
  VP8LAddVectorEq = AddVectorEq_AVX2;
  VP8LCombinedShannonEntropy = CombinedShannonEntropy_AVX2;
  VP8LVectorMismatch = VectorMismatch_AVX2;
  VP8LBundleColorMap = BundleColorMap_AVX2;

  VP8LPredictorsSub[0] = PredictorSub0_AVX2;
  VP8LPredictorsSub[1] = PredictorSub1_AVX2;
  VP8LPredictorsSub[2] = PredictorSub2_AVX2;
  VP8LPredictorsSub[3] = PredictorSub3_AVX2;
  VP8LPredictorsSub[4] = PredictorSub4_AVX2;
  VP8LPredictorsSub[5] = PredictorSub5_AVX2;
  VP8LPredictorsSub[6] = PredictorSub6_AVX2;
  VP8LPredictorsSub[7] = PredictorSub7_AVX2;
  VP8LPredictorsSub[8] = PredictorSub8_AVX2;
  VP8LPredictorsSub[9] = PredictorSub9_AVX2;
  VP8LPredictorsSub[10] = PredictorSub10_AVX2;
  VP8LPredictorsSub[11] = PredictorSub11_AVX2;
  VP8LPredictorsSub[12] = PredictorSub12_AVX2;
  VP8LPredictorsSub[13] = PredictorSub13_AVX2;
  VP8LPredictorsSub[14] = PredictorSub0_AVX2;  // <- padding security sentinels
  VP8LPredictorsSub[15] = PredictorSub0_AVX2;
}

#else  // !WEBP_USE_AVX2

WEBP_DSP_INIT_STUB(VP8LEncDspInitAVX2)

#endif  // WEBP_USE_AVX2
