// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// SSE2 version of YUV to RGB upsampling functions.
//
// Author: somnath@google.com (Somnath Banerjee)

#include "./dsp.h"

#if defined(WEBP_USE_SSE2)

#include <assert.h>
#include <emmintrin.h>
#include <string.h>
#include "./yuv.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#ifdef FANCY_UPSAMPLING

// We compute (9*a + 3*b + 3*c + d + 8) / 16 as follows
// u = (9*a + 3*b + 3*c + d + 8) / 16
//   = (a + (a + 3*b + 3*c + d) / 8 + 1) / 2
//   = (a + m + 1) / 2
// where m = (a + 3*b + 3*c + d) / 8
//         = ((a + b + c + d) / 2 + b + c) / 4
//
// Let's say  k = (a + b + c + d) / 4.
// We can compute k as
// k = (s + t + 1) / 2 - ((a^d) | (b^c) | (s^t)) & 1
// where s = (a + d + 1) / 2 and t = (b + c + 1) / 2
//
// Then m can be written as
// m = (k + t + 1) / 2 - (((b^c) & (s^t)) | (k^t)) & 1

// Computes out = (k + in + 1) / 2 - ((ij & (s^t)) | (k^in)) & 1
#define GET_M(ij, in, out) do {                                                \
  const __m128i tmp0 = _mm_avg_epu8(k, (in));     /* (k + in + 1) / 2 */       \
  const __m128i tmp1 = _mm_and_si128((ij), st);   /* (ij) & (s^t) */           \
  const __m128i tmp2 = _mm_xor_si128(k, (in));    /* (k^in) */                 \
  const __m128i tmp3 = _mm_or_si128(tmp1, tmp2);  /* ((ij) & (s^t)) | (k^in) */\
  const __m128i tmp4 = _mm_and_si128(tmp3, one);  /* & 1 -> lsb_correction */  \
  (out) = _mm_sub_epi8(tmp0, tmp4);    /* (k + in + 1) / 2 - lsb_correction */ \
} while (0)

// pack and store two alterning pixel rows
#define PACK_AND_STORE(a, b, da, db, out) do {                                 \
  const __m128i ta = _mm_avg_epu8(a, da);  /* (9a + 3b + 3c +  d + 8) / 16 */  \
  const __m128i tb = _mm_avg_epu8(b, db);  /* (3a + 9b +  c + 3d + 8) / 16 */  \
  const __m128i t1 = _mm_unpacklo_epi8(ta, tb);                                \
  const __m128i t2 = _mm_unpackhi_epi8(ta, tb);                                \
  _mm_store_si128(((__m128i*)(out)) + 0, t1);                                  \
  _mm_store_si128(((__m128i*)(out)) + 1, t2);                                  \
} while (0)

// Loads 17 pixels each from rows r1 and r2 and generates 32 pixels.
#define UPSAMPLE_32PIXELS(r1, r2, out) {                                       \
  const __m128i one = _mm_set1_epi8(1);                                        \
  const __m128i a = _mm_loadu_si128((__m128i*)&(r1)[0]);                       \
  const __m128i b = _mm_loadu_si128((__m128i*)&(r1)[1]);                       \
  const __m128i c = _mm_loadu_si128((__m128i*)&(r2)[0]);                       \
  const __m128i d = _mm_loadu_si128((__m128i*)&(r2)[1]);                       \
                                                                               \
  const __m128i s = _mm_avg_epu8(a, d);        /* s = (a + d + 1) / 2 */       \
  const __m128i t = _mm_avg_epu8(b, c);        /* t = (b + c + 1) / 2 */       \
  const __m128i st = _mm_xor_si128(s, t);      /* st = s^t */                  \
                                                                               \
  const __m128i ad = _mm_xor_si128(a, d);      /* ad = a^d */                  \
  const __m128i bc = _mm_xor_si128(b, c);      /* bc = b^c */                  \
                                                                               \
  const __m128i t1 = _mm_or_si128(ad, bc);     /* (a^d) | (b^c) */             \
  const __m128i t2 = _mm_or_si128(t1, st);     /* (a^d) | (b^c) | (s^t) */     \
  const __m128i t3 = _mm_and_si128(t2, one);   /* (a^d) | (b^c) | (s^t) & 1 */ \
  const __m128i t4 = _mm_avg_epu8(s, t);                                       \
  const __m128i k = _mm_sub_epi8(t4, t3);      /* k = (a + b + c + d) / 4 */   \
  __m128i diag1, diag2;                                                        \
                                                                               \
  GET_M(bc, t, diag1);                  /* diag1 = (a + 3b + 3c + d) / 8 */    \
  GET_M(ad, s, diag2);                  /* diag2 = (3a + b + c + 3d) / 8 */    \
                                                                               \
  /* pack the alternate pixels */                                              \
  PACK_AND_STORE(a, b, diag1, diag2, &(out)[0 * 32]);                          \
  PACK_AND_STORE(c, d, diag2, diag1, &(out)[2 * 32]);                          \
}

// Turn the macro into a function for reducing code-size when non-critical
static void Upsample32Pixels(const uint8_t r1[], const uint8_t r2[],
                             uint8_t* const out) {
  UPSAMPLE_32PIXELS(r1, r2, out);
}

#define UPSAMPLE_LAST_BLOCK(tb, bb, num_pixels, out) {                         \
  uint8_t r1[17], r2[17];                                                      \
  memcpy(r1, (tb), (num_pixels));                                              \
  memcpy(r2, (bb), (num_pixels));                                              \
  /* replicate last byte */                                                    \
  memset(r1 + (num_pixels), r1[(num_pixels) - 1], 17 - (num_pixels));          \
  memset(r2 + (num_pixels), r2[(num_pixels) - 1], 17 - (num_pixels));          \
  /* using the shared function instead of the macro saves ~3k code size */     \
  Upsample32Pixels(r1, r2, out);                                               \
}

#define CONVERT2RGB(FUNC, XSTEP, top_y, bottom_y, uv,                          \
                    top_dst, bottom_dst, cur_x, num_pixels) {                  \
  int n;                                                                       \
  if (top_y) {                                                                 \
    for (n = 0; n < (num_pixels); ++n) {                                       \
      FUNC(top_y[(cur_x) + n], (uv)[n], (uv)[32 + n],                          \
           top_dst + ((cur_x) + n) * XSTEP);                                   \
    }                                                                          \
  }                                                                            \
  if (bottom_y) {                                                              \
    for (n = 0; n < (num_pixels); ++n) {                                       \
      FUNC(bottom_y[(cur_x) + n], (uv)[64 + n], (uv)[64 + 32 + n],             \
           bottom_dst + ((cur_x) + n) * XSTEP);                                \
    }                                                                          \
  }                                                                            \
}

#define SSE2_UPSAMPLE_FUNC(FUNC_NAME, FUNC, XSTEP)                             \
static void FUNC_NAME(const uint8_t* top_y, const uint8_t* bottom_y,           \
                      const uint8_t* top_u, const uint8_t* top_v,              \
                      const uint8_t* cur_u, const uint8_t* cur_v,              \
                      uint8_t* top_dst, uint8_t* bottom_dst, int len) {        \
  int b;                                                                       \
  /* 16 byte aligned array to cache reconstructed u and v */                   \
  uint8_t uv_buf[4 * 32 + 15];                                                 \
  uint8_t* const r_uv = (uint8_t*)((uintptr_t)(uv_buf + 15) & ~15);            \
  const int uv_len = (len + 1) >> 1;                                           \
  /* 17 pixels must be read-able for each block */                             \
  const int num_blocks = (uv_len - 1) >> 4;                                    \
  const int leftover = uv_len - num_blocks * 16;                               \
  const int last_pos = 1 + 32 * num_blocks;                                    \
                                                                               \
  const int u_diag = ((top_u[0] + cur_u[0]) >> 1) + 1;                         \
  const int v_diag = ((top_v[0] + cur_v[0]) >> 1) + 1;                         \
                                                                               \
  assert(len > 0);                                                             \
  /* Treat the first pixel in regular way */                                   \
  if (top_y) {                                                                 \
    const int u0 = (top_u[0] + u_diag) >> 1;                                   \
    const int v0 = (top_v[0] + v_diag) >> 1;                                   \
    FUNC(top_y[0], u0, v0, top_dst);                                           \
  }                                                                            \
  if (bottom_y) {                                                              \
    const int u0 = (cur_u[0] + u_diag) >> 1;                                   \
    const int v0 = (cur_v[0] + v_diag) >> 1;                                   \
    FUNC(bottom_y[0], u0, v0, bottom_dst);                                     \
  }                                                                            \
                                                                               \
  for (b = 0; b < num_blocks; ++b) {                                           \
    UPSAMPLE_32PIXELS(top_u, cur_u, r_uv + 0 * 32);                            \
    UPSAMPLE_32PIXELS(top_v, cur_v, r_uv + 1 * 32);                            \
    CONVERT2RGB(FUNC, XSTEP, top_y, bottom_y, r_uv, top_dst, bottom_dst,       \
                32 * b + 1, 32)                                                \
    top_u += 16;                                                               \
    cur_u += 16;                                                               \
    top_v += 16;                                                               \
    cur_v += 16;                                                               \
  }                                                                            \
                                                                               \
  UPSAMPLE_LAST_BLOCK(top_u, cur_u, leftover, r_uv + 0 * 32);                  \
  UPSAMPLE_LAST_BLOCK(top_v, cur_v, leftover, r_uv + 1 * 32);                  \
  CONVERT2RGB(FUNC, XSTEP, top_y, bottom_y, r_uv, top_dst, bottom_dst,         \
              last_pos, len - last_pos);                                       \
}

// SSE2 variants of the fancy upsampler.
SSE2_UPSAMPLE_FUNC(UpsampleRgbLinePairSSE2,  VP8YuvToRgb,  3)
SSE2_UPSAMPLE_FUNC(UpsampleBgrLinePairSSE2,  VP8YuvToBgr,  3)
SSE2_UPSAMPLE_FUNC(UpsampleRgbaLinePairSSE2, VP8YuvToRgba, 4)
SSE2_UPSAMPLE_FUNC(UpsampleBgraLinePairSSE2, VP8YuvToBgra, 4)

#undef GET_M
#undef PACK_AND_STORE
#undef UPSAMPLE_32PIXELS
#undef UPSAMPLE_LAST_BLOCK
#undef CONVERT2RGB
#undef SSE2_UPSAMPLE_FUNC

//------------------------------------------------------------------------------

extern WebPUpsampleLinePairFunc WebPUpsamplers[/* MODE_LAST */];

void WebPInitUpsamplersSSE2(void) {
  WebPUpsamplers[MODE_RGB]  = UpsampleRgbLinePairSSE2;
  WebPUpsamplers[MODE_RGBA] = UpsampleRgbaLinePairSSE2;
  WebPUpsamplers[MODE_BGR]  = UpsampleBgrLinePairSSE2;
  WebPUpsamplers[MODE_BGRA] = UpsampleBgraLinePairSSE2;
}

void WebPInitPremultiplySSE2(void) {
  WebPUpsamplers[MODE_rgbA] = UpsampleRgbaLinePairSSE2;
  WebPUpsamplers[MODE_bgrA] = UpsampleBgraLinePairSSE2;
}

#endif  // FANCY_UPSAMPLING

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif   // WEBP_USE_SSE2
