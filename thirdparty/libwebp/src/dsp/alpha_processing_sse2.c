// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Utilities for processing transparent channel.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_SSE2)
#include <emmintrin.h>

#include "src/webp/types.h"
#include "src/dsp/cpu.h"

//------------------------------------------------------------------------------

static int DispatchAlpha_SSE2(const uint8_t* WEBP_RESTRICT alpha,
                              int alpha_stride, int width, int height,
                              uint8_t* WEBP_RESTRICT dst, int dst_stride) {
  // alpha_and stores an 'and' operation of all the alpha[] values. The final
  // value is not 0xff if any of the alpha[] is not equal to 0xff.
  uint32_t alpha_and = 0xff;
  int i, j;
  const __m128i zero = _mm_setzero_si128();
  const __m128i alpha_mask = _mm_set1_epi32((int)0xff);  // to preserve A
  const __m128i all_0xff = _mm_set1_epi8((char)0xff);
  __m128i all_alphas16 = all_0xff;
  __m128i all_alphas8 = all_0xff;

  // We must be able to access 3 extra bytes after the last written byte
  // 'dst[4 * width - 4]', because we don't know if alpha is the first or the
  // last byte of the quadruplet.
  for (j = 0; j < height; ++j) {
    char* ptr = (char*)dst;
    for (i = 0; i + 16 <= width - 1; i += 16) {
      // load 16 alpha bytes
      const __m128i a0 = _mm_loadu_si128((const __m128i*)&alpha[i]);
      const __m128i a1_lo = _mm_unpacklo_epi8(a0, zero);
      const __m128i a1_hi = _mm_unpackhi_epi8(a0, zero);
      const __m128i a2_lo_lo = _mm_unpacklo_epi16(a1_lo, zero);
      const __m128i a2_lo_hi = _mm_unpackhi_epi16(a1_lo, zero);
      const __m128i a2_hi_lo = _mm_unpacklo_epi16(a1_hi, zero);
      const __m128i a2_hi_hi = _mm_unpackhi_epi16(a1_hi, zero);
      _mm_maskmoveu_si128(a2_lo_lo, alpha_mask, ptr + 0);
      _mm_maskmoveu_si128(a2_lo_hi, alpha_mask, ptr + 16);
      _mm_maskmoveu_si128(a2_hi_lo, alpha_mask, ptr + 32);
      _mm_maskmoveu_si128(a2_hi_hi, alpha_mask, ptr + 48);
      // accumulate 16 alpha 'and' in parallel
      all_alphas16 = _mm_and_si128(all_alphas16, a0);
      ptr += 64;
    }
    if (i + 8 <= width - 1) {
      // load 8 alpha bytes
      const __m128i a0 = _mm_loadl_epi64((const __m128i*)&alpha[i]);
      const __m128i a1 = _mm_unpacklo_epi8(a0, zero);
      const __m128i a2_lo = _mm_unpacklo_epi16(a1, zero);
      const __m128i a2_hi = _mm_unpackhi_epi16(a1, zero);
      _mm_maskmoveu_si128(a2_lo, alpha_mask, ptr);
      _mm_maskmoveu_si128(a2_hi, alpha_mask, ptr + 16);
      // accumulate 8 alpha 'and' in parallel
      all_alphas8 = _mm_and_si128(all_alphas8, a0);
      i += 8;
    }
    for (; i < width; ++i) {
      const uint32_t alpha_value = alpha[i];
      dst[4 * i] = alpha_value;
      alpha_and &= alpha_value;
    }
    alpha += alpha_stride;
    dst += dst_stride;
  }
  // Combine the eight alpha 'and' into a 8-bit mask.
  alpha_and &= _mm_movemask_epi8(_mm_cmpeq_epi8(all_alphas8, all_0xff)) & 0xff;
  return (alpha_and != 0xff ||
          _mm_movemask_epi8(_mm_cmpeq_epi8(all_alphas16, all_0xff)) != 0xffff);
}

static void DispatchAlphaToGreen_SSE2(const uint8_t* WEBP_RESTRICT alpha,
                                      int alpha_stride, int width, int height,
                                      uint32_t* WEBP_RESTRICT dst,
                                      int dst_stride) {
  int i, j;
  const __m128i zero = _mm_setzero_si128();
  const int limit = width & ~15;
  for (j = 0; j < height; ++j) {
    for (i = 0; i < limit; i += 16) {   // process 16 alpha bytes
      const __m128i a0 = _mm_loadu_si128((const __m128i*)&alpha[i]);
      const __m128i a1 = _mm_unpacklo_epi8(zero, a0);  // note the 'zero' first!
      const __m128i b1 = _mm_unpackhi_epi8(zero, a0);
      const __m128i a2_lo = _mm_unpacklo_epi16(a1, zero);
      const __m128i b2_lo = _mm_unpacklo_epi16(b1, zero);
      const __m128i a2_hi = _mm_unpackhi_epi16(a1, zero);
      const __m128i b2_hi = _mm_unpackhi_epi16(b1, zero);
      _mm_storeu_si128((__m128i*)&dst[i +  0], a2_lo);
      _mm_storeu_si128((__m128i*)&dst[i +  4], a2_hi);
      _mm_storeu_si128((__m128i*)&dst[i +  8], b2_lo);
      _mm_storeu_si128((__m128i*)&dst[i + 12], b2_hi);
    }
    for (; i < width; ++i) dst[i] = alpha[i] << 8;
    alpha += alpha_stride;
    dst += dst_stride;
  }
}

static int ExtractAlpha_SSE2(const uint8_t* WEBP_RESTRICT argb, int argb_stride,
                             int width, int height,
                             uint8_t* WEBP_RESTRICT alpha, int alpha_stride) {
  // alpha_and stores an 'and' operation of all the alpha[] values. The final
  // value is not 0xff if any of the alpha[] is not equal to 0xff.
  uint32_t alpha_and = 0xff;
  int i, j;
  const __m128i a_mask = _mm_set1_epi32(0xff);  // to preserve alpha
  const __m128i all_0xff = _mm_set_epi32(0, 0, ~0, ~0);
  __m128i all_alphas = all_0xff;

  // We must be able to access 3 extra bytes after the last written byte
  // 'src[4 * width - 4]', because we don't know if alpha is the first or the
  // last byte of the quadruplet.
  const int limit = (width - 1) & ~7;

  for (j = 0; j < height; ++j) {
    const __m128i* src = (const __m128i*)argb;
    for (i = 0; i < limit; i += 8) {
      // load 32 argb bytes
      const __m128i a0 = _mm_loadu_si128(src + 0);
      const __m128i a1 = _mm_loadu_si128(src + 1);
      const __m128i b0 = _mm_and_si128(a0, a_mask);
      const __m128i b1 = _mm_and_si128(a1, a_mask);
      const __m128i c0 = _mm_packs_epi32(b0, b1);
      const __m128i d0 = _mm_packus_epi16(c0, c0);
      // store
      _mm_storel_epi64((__m128i*)&alpha[i], d0);
      // accumulate eight alpha 'and' in parallel
      all_alphas = _mm_and_si128(all_alphas, d0);
      src += 2;
    }
    for (; i < width; ++i) {
      const uint32_t alpha_value = argb[4 * i];
      alpha[i] = alpha_value;
      alpha_and &= alpha_value;
    }
    argb += argb_stride;
    alpha += alpha_stride;
  }
  // Combine the eight alpha 'and' into a 8-bit mask.
  alpha_and &= _mm_movemask_epi8(_mm_cmpeq_epi8(all_alphas, all_0xff));
  return (alpha_and == 0xff);
}

static void ExtractGreen_SSE2(const uint32_t* WEBP_RESTRICT argb,
                              uint8_t* WEBP_RESTRICT alpha, int size) {
  int i;
  const __m128i mask = _mm_set1_epi32(0xff);
  const __m128i* src = (const __m128i*)argb;

  for (i = 0; i + 16 <= size; i += 16, src += 4) {
    const __m128i a0 = _mm_loadu_si128(src + 0);
    const __m128i a1 = _mm_loadu_si128(src + 1);
    const __m128i a2 = _mm_loadu_si128(src + 2);
    const __m128i a3 = _mm_loadu_si128(src + 3);
    const __m128i b0 = _mm_srli_epi32(a0, 8);
    const __m128i b1 = _mm_srli_epi32(a1, 8);
    const __m128i b2 = _mm_srli_epi32(a2, 8);
    const __m128i b3 = _mm_srli_epi32(a3, 8);
    const __m128i c0 = _mm_and_si128(b0, mask);
    const __m128i c1 = _mm_and_si128(b1, mask);
    const __m128i c2 = _mm_and_si128(b2, mask);
    const __m128i c3 = _mm_and_si128(b3, mask);
    const __m128i d0 = _mm_packs_epi32(c0, c1);
    const __m128i d1 = _mm_packs_epi32(c2, c3);
    const __m128i e = _mm_packus_epi16(d0, d1);
    // store
    _mm_storeu_si128((__m128i*)&alpha[i], e);
  }
  if (i + 8 <= size) {
    const __m128i a0 = _mm_loadu_si128(src + 0);
    const __m128i a1 = _mm_loadu_si128(src + 1);
    const __m128i b0 = _mm_srli_epi32(a0, 8);
    const __m128i b1 = _mm_srli_epi32(a1, 8);
    const __m128i c0 = _mm_and_si128(b0, mask);
    const __m128i c1 = _mm_and_si128(b1, mask);
    const __m128i d = _mm_packs_epi32(c0, c1);
    const __m128i e = _mm_packus_epi16(d, d);
    _mm_storel_epi64((__m128i*)&alpha[i], e);
    i += 8;
  }
  for (; i < size; ++i) alpha[i] = argb[i] >> 8;
}

//------------------------------------------------------------------------------
// Non-dither premultiplied modes

#define MULTIPLIER(a)   ((a) * 0x8081)
#define PREMULTIPLY(x, m) (((x) * (m)) >> 23)

// We can't use a 'const int' for the SHUFFLE value, because it has to be an
// immediate in the _mm_shufflexx_epi16() instruction. We really need a macro.
// We use: v / 255 = (v * 0x8081) >> 23, where v = alpha * {r,g,b} is a 16bit
// value.
#define APPLY_ALPHA(RGBX, SHUFFLE) do {                              \
  const __m128i argb0 = _mm_loadu_si128((const __m128i*)&(RGBX));    \
  const __m128i argb1_lo = _mm_unpacklo_epi8(argb0, zero);           \
  const __m128i argb1_hi = _mm_unpackhi_epi8(argb0, zero);           \
  const __m128i alpha0_lo = _mm_or_si128(argb1_lo, kMask);           \
  const __m128i alpha0_hi = _mm_or_si128(argb1_hi, kMask);           \
  const __m128i alpha1_lo = _mm_shufflelo_epi16(alpha0_lo, SHUFFLE); \
  const __m128i alpha1_hi = _mm_shufflelo_epi16(alpha0_hi, SHUFFLE); \
  const __m128i alpha2_lo = _mm_shufflehi_epi16(alpha1_lo, SHUFFLE); \
  const __m128i alpha2_hi = _mm_shufflehi_epi16(alpha1_hi, SHUFFLE); \
  /* alpha2 = [ff a0 a0 a0][ff a1 a1 a1] */                          \
  const __m128i A0_lo = _mm_mullo_epi16(alpha2_lo, argb1_lo);        \
  const __m128i A0_hi = _mm_mullo_epi16(alpha2_hi, argb1_hi);        \
  const __m128i A1_lo = _mm_mulhi_epu16(A0_lo, kMult);               \
  const __m128i A1_hi = _mm_mulhi_epu16(A0_hi, kMult);               \
  const __m128i A2_lo = _mm_srli_epi16(A1_lo, 7);                    \
  const __m128i A2_hi = _mm_srli_epi16(A1_hi, 7);                    \
  const __m128i A3 = _mm_packus_epi16(A2_lo, A2_hi);                 \
  _mm_storeu_si128((__m128i*)&(RGBX), A3);                           \
} while (0)

static void ApplyAlphaMultiply_SSE2(uint8_t* rgba, int alpha_first,
                                    int w, int h, int stride) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i kMult = _mm_set1_epi16((short)0x8081);
  const __m128i kMask = _mm_set_epi16(0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0);
  const int kSpan = 4;
  while (h-- > 0) {
    uint32_t* const rgbx = (uint32_t*)rgba;
    int i;
    if (!alpha_first) {
      for (i = 0; i + kSpan <= w; i += kSpan) {
        APPLY_ALPHA(rgbx[i], _MM_SHUFFLE(2, 3, 3, 3));
      }
    } else {
      for (i = 0; i + kSpan <= w; i += kSpan) {
        APPLY_ALPHA(rgbx[i], _MM_SHUFFLE(0, 0, 0, 1));
      }
    }
    // Finish with left-overs.
    for (; i < w; ++i) {
      uint8_t* const rgb = rgba + (alpha_first ? 1 : 0);
      const uint8_t* const alpha = rgba + (alpha_first ? 0 : 3);
      const uint32_t a = alpha[4 * i];
      if (a != 0xff) {
        const uint32_t mult = MULTIPLIER(a);
        rgb[4 * i + 0] = PREMULTIPLY(rgb[4 * i + 0], mult);
        rgb[4 * i + 1] = PREMULTIPLY(rgb[4 * i + 1], mult);
        rgb[4 * i + 2] = PREMULTIPLY(rgb[4 * i + 2], mult);
      }
    }
    rgba += stride;
  }
}
#undef MULTIPLIER
#undef PREMULTIPLY

//------------------------------------------------------------------------------
// Alpha detection

static int HasAlpha8b_SSE2(const uint8_t* src, int length) {
  const __m128i all_0xff = _mm_set1_epi8((char)0xff);
  int i = 0;
  for (; i + 16 <= length; i += 16) {
    const __m128i v = _mm_loadu_si128((const __m128i*)(src + i));
    const __m128i bits = _mm_cmpeq_epi8(v, all_0xff);
    const int mask = _mm_movemask_epi8(bits);
    if (mask != 0xffff) return 1;
  }
  for (; i < length; ++i) if (src[i] != 0xff) return 1;
  return 0;
}

static int HasAlpha32b_SSE2(const uint8_t* src, int length) {
  const __m128i alpha_mask = _mm_set1_epi32(0xff);
  const __m128i all_0xff = _mm_set1_epi8((char)0xff);
  int i = 0;
  // We don't know if we can access the last 3 bytes after the last alpha
  // value 'src[4 * length - 4]' (because we don't know if alpha is the first
  // or the last byte of the quadruplet). Hence the '-3' protection below.
  length = length * 4 - 3;   // size in bytes
  for (; i + 64 <= length; i += 64) {
    const __m128i a0 = _mm_loadu_si128((const __m128i*)(src + i +  0));
    const __m128i a1 = _mm_loadu_si128((const __m128i*)(src + i + 16));
    const __m128i a2 = _mm_loadu_si128((const __m128i*)(src + i + 32));
    const __m128i a3 = _mm_loadu_si128((const __m128i*)(src + i + 48));
    const __m128i b0 = _mm_and_si128(a0, alpha_mask);
    const __m128i b1 = _mm_and_si128(a1, alpha_mask);
    const __m128i b2 = _mm_and_si128(a2, alpha_mask);
    const __m128i b3 = _mm_and_si128(a3, alpha_mask);
    const __m128i c0 = _mm_packs_epi32(b0, b1);
    const __m128i c1 = _mm_packs_epi32(b2, b3);
    const __m128i d  = _mm_packus_epi16(c0, c1);
    const __m128i bits = _mm_cmpeq_epi8(d, all_0xff);
    const int mask = _mm_movemask_epi8(bits);
    if (mask != 0xffff) return 1;
  }
  for (; i + 32 <= length; i += 32) {
    const __m128i a0 = _mm_loadu_si128((const __m128i*)(src + i +  0));
    const __m128i a1 = _mm_loadu_si128((const __m128i*)(src + i + 16));
    const __m128i b0 = _mm_and_si128(a0, alpha_mask);
    const __m128i b1 = _mm_and_si128(a1, alpha_mask);
    const __m128i c  = _mm_packs_epi32(b0, b1);
    const __m128i d  = _mm_packus_epi16(c, c);
    const __m128i bits = _mm_cmpeq_epi8(d, all_0xff);
    const int mask = _mm_movemask_epi8(bits);
    if (mask != 0xffff) return 1;
  }
  for (; i <= length; i += 4) if (src[i] != 0xff) return 1;
  return 0;
}

static void AlphaReplace_SSE2(uint32_t* src, int length, uint32_t color) {
  const __m128i m_color = _mm_set1_epi32((int)color);
  const __m128i zero = _mm_setzero_si128();
  int i = 0;
  for (; i + 8 <= length; i += 8) {
    const __m128i a0 = _mm_loadu_si128((const __m128i*)(src + i + 0));
    const __m128i a1 = _mm_loadu_si128((const __m128i*)(src + i + 4));
    const __m128i b0 = _mm_srai_epi32(a0, 24);
    const __m128i b1 = _mm_srai_epi32(a1, 24);
    const __m128i c0 = _mm_cmpeq_epi32(b0, zero);
    const __m128i c1 = _mm_cmpeq_epi32(b1, zero);
    const __m128i d0 = _mm_and_si128(c0, m_color);
    const __m128i d1 = _mm_and_si128(c1, m_color);
    const __m128i e0 = _mm_andnot_si128(c0, a0);
    const __m128i e1 = _mm_andnot_si128(c1, a1);
    _mm_storeu_si128((__m128i*)(src + i + 0), _mm_or_si128(d0, e0));
    _mm_storeu_si128((__m128i*)(src + i + 4), _mm_or_si128(d1, e1));
  }
  for (; i < length; ++i) if ((src[i] >> 24) == 0) src[i] = color;
}

// -----------------------------------------------------------------------------
// Apply alpha value to rows

static void MultARGBRow_SSE2(uint32_t* const ptr, int width, int inverse) {
  int x = 0;
  if (!inverse) {
    const int kSpan = 2;
    const __m128i zero = _mm_setzero_si128();
    const __m128i k128 = _mm_set1_epi16(128);
    const __m128i kMult = _mm_set1_epi16(0x0101);
    const __m128i kMask = _mm_set_epi16(0, 0xff, 0, 0, 0, 0xff, 0, 0);
    for (x = 0; x + kSpan <= width; x += kSpan) {
      // To compute 'result = (int)(a * x / 255. + .5)', we use:
      //   tmp = a * v + 128, result = (tmp * 0x0101u) >> 16
      const __m128i A0 = _mm_loadl_epi64((const __m128i*)&ptr[x]);
      const __m128i A1 = _mm_unpacklo_epi8(A0, zero);
      const __m128i A2 = _mm_or_si128(A1, kMask);
      const __m128i A3 = _mm_shufflelo_epi16(A2, _MM_SHUFFLE(2, 3, 3, 3));
      const __m128i A4 = _mm_shufflehi_epi16(A3, _MM_SHUFFLE(2, 3, 3, 3));
      // here, A4 = [ff a0 a0 a0][ff a1 a1 a1]
      const __m128i A5 = _mm_mullo_epi16(A4, A1);
      const __m128i A6 = _mm_add_epi16(A5, k128);
      const __m128i A7 = _mm_mulhi_epu16(A6, kMult);
      const __m128i A10 = _mm_packus_epi16(A7, zero);
      _mm_storel_epi64((__m128i*)&ptr[x], A10);
    }
  }
  width -= x;
  if (width > 0) WebPMultARGBRow_C(ptr + x, width, inverse);
}

static void MultRow_SSE2(uint8_t* WEBP_RESTRICT const ptr,
                         const uint8_t* WEBP_RESTRICT const alpha,
                         int width, int inverse) {
  int x = 0;
  if (!inverse) {
    const __m128i zero = _mm_setzero_si128();
    const __m128i k128 = _mm_set1_epi16(128);
    const __m128i kMult = _mm_set1_epi16(0x0101);
    for (x = 0; x + 8 <= width; x += 8) {
      const __m128i v0 = _mm_loadl_epi64((__m128i*)&ptr[x]);
      const __m128i a0 = _mm_loadl_epi64((const __m128i*)&alpha[x]);
      const __m128i v1 = _mm_unpacklo_epi8(v0, zero);
      const __m128i a1 = _mm_unpacklo_epi8(a0, zero);
      const __m128i v2 = _mm_mullo_epi16(v1, a1);
      const __m128i v3 = _mm_add_epi16(v2, k128);
      const __m128i v4 = _mm_mulhi_epu16(v3, kMult);
      const __m128i v5 = _mm_packus_epi16(v4, zero);
      _mm_storel_epi64((__m128i*)&ptr[x], v5);
    }
  }
  width -= x;
  if (width > 0) WebPMultRow_C(ptr + x, alpha + x, width, inverse);
}

//------------------------------------------------------------------------------
// Entry point

extern void WebPInitAlphaProcessingSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPInitAlphaProcessingSSE2(void) {
  WebPMultARGBRow = MultARGBRow_SSE2;
  WebPMultRow = MultRow_SSE2;
  WebPApplyAlphaMultiply = ApplyAlphaMultiply_SSE2;
  WebPDispatchAlpha = DispatchAlpha_SSE2;
  WebPDispatchAlphaToGreen = DispatchAlphaToGreen_SSE2;
  WebPExtractAlpha = ExtractAlpha_SSE2;
  WebPExtractGreen = ExtractGreen_SSE2;

  WebPHasAlpha8b = HasAlpha8b_SSE2;
  WebPHasAlpha32b = HasAlpha32b_SSE2;
  WebPAlphaReplace = AlphaReplace_SSE2;
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(WebPInitAlphaProcessingSSE2)

#endif  // WEBP_USE_SSE2
