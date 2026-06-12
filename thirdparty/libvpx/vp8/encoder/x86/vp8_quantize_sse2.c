/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx_ports/x86.h"
#include "vpx_mem/vpx_mem.h"
#include "vp8/encoder/block.h"
#include "vp8/common/entropy.h" /* vp8_default_inv_zig_zag */

#include <mmintrin.h>  /* MMX */
#include <xmmintrin.h> /* SSE */
#include <emmintrin.h> /* SSE2 */

#define SELECT_EOB(i, z)                    \
  do {                                      \
    short boost = *zbin_boost_ptr;          \
    int cmp = (x[z] < boost) | (y[z] == 0); \
    zbin_boost_ptr++;                       \
    if (cmp) break;                         \
    qcoeff_ptr[z] = y[z];                   \
    eob = i;                                \
    zbin_boost_ptr = b->zrun_zbin_boost;    \
  } while (0)

void vp8_regular_quantize_b_sse2(BLOCK *b, BLOCKD *d) {
  char eob = 0;
  short *zbin_boost_ptr;
  short *qcoeff_ptr = d->qcoeff;
  DECLARE_ALIGNED(16, short, x[16]);
  DECLARE_ALIGNED(16, short, y[16]);

  __m128i sz0, x0, sz1, x1, y0, y1, x_minus_zbin0, x_minus_zbin1;
  __m128i quant_shift0 = _mm_load_si128((__m128i *)(b->quant_shift));
  __m128i quant_shift1 = _mm_load_si128((__m128i *)(b->quant_shift + 8));
  __m128i z0 = _mm_load_si128((__m128i *)(b->coeff));
  __m128i z1 = _mm_load_si128((__m128i *)(b->coeff + 8));
  __m128i zbin_extra = _mm_cvtsi32_si128(b->zbin_extra);
  __m128i zbin0 = _mm_load_si128((__m128i *)(b->zbin));
  __m128i zbin1 = _mm_load_si128((__m128i *)(b->zbin + 8));
  __m128i round0 = _mm_load_si128((__m128i *)(b->round));
  __m128i round1 = _mm_load_si128((__m128i *)(b->round + 8));
  __m128i quant0 = _mm_load_si128((__m128i *)(b->quant));
  __m128i quant1 = _mm_load_si128((__m128i *)(b->quant + 8));
  __m128i dequant0 = _mm_load_si128((__m128i *)(d->dequant));
  __m128i dequant1 = _mm_load_si128((__m128i *)(d->dequant + 8));

  memset(qcoeff_ptr, 0, 32);

  /* Duplicate to all lanes. */
  zbin_extra = _mm_shufflelo_epi16(zbin_extra, 0);
  zbin_extra = _mm_unpacklo_epi16(zbin_extra, zbin_extra);

  /* Sign of z: z >> 15 */
  sz0 = _mm_srai_epi16(z0, 15);
  sz1 = _mm_srai_epi16(z1, 15);

  /* x = abs(z): (z ^ sz) - sz */
  x0 = _mm_xor_si128(z0, sz0);
  x1 = _mm_xor_si128(z1, sz1);
  x0 = _mm_sub_epi16(x0, sz0);
  x1 = _mm_sub_epi16(x1, sz1);

  /* zbin[] + zbin_extra */
  zbin0 = _mm_add_epi16(zbin0, zbin_extra);
  zbin1 = _mm_add_epi16(zbin1, zbin_extra);

  /* In C x is compared to zbin where zbin = zbin[] + boost + extra. Rebalance
   * the equation because boost is the only value which can change:
   * x - (zbin[] + extra) >= boost */
  x_minus_zbin0 = _mm_sub_epi16(x0, zbin0);
  x_minus_zbin1 = _mm_sub_epi16(x1, zbin1);

  _mm_store_si128((__m128i *)(x), x_minus_zbin0);
  _mm_store_si128((__m128i *)(x + 8), x_minus_zbin1);

  /* All the remaining calculations are valid whether they are done now with
   * simd or later inside the loop one at a time. */
  x0 = _mm_add_epi16(x0, round0);
  x1 = _mm_add_epi16(x1, round1);

  y0 = _mm_mulhi_epi16(x0, quant0);
  y1 = _mm_mulhi_epi16(x1, quant1);

  y0 = _mm_add_epi16(y0, x0);
  y1 = _mm_add_epi16(y1, x1);

  /* Instead of shifting each value independently we convert the scaling
   * factor with 1 << (16 - shift) so we can use multiply/return high half. */
  y0 = _mm_mulhi_epi16(y0, quant_shift0);
  y1 = _mm_mulhi_epi16(y1, quant_shift1);

  /* Return the sign: (y ^ sz) - sz */
  y0 = _mm_xor_si128(y0, sz0);
  y1 = _mm_xor_si128(y1, sz1);
  y0 = _mm_sub_epi16(y0, sz0);
  y1 = _mm_sub_epi16(y1, sz1);

  _mm_store_si128((__m128i *)(y), y0);
  _mm_store_si128((__m128i *)(y + 8), y1);

  zbin_boost_ptr = b->zrun_zbin_boost;

  /* The loop gets unrolled anyway. Avoid the vp8_default_zig_zag1d lookup. */
  SELECT_EOB(1, 0);
  SELECT_EOB(2, 1);
  SELECT_EOB(3, 4);
  SELECT_EOB(4, 8);
  SELECT_EOB(5, 5);
  SELECT_EOB(6, 2);
  SELECT_EOB(7, 3);
  SELECT_EOB(8, 6);
  SELECT_EOB(9, 9);
  SELECT_EOB(10, 12);
  SELECT_EOB(11, 13);
  SELECT_EOB(12, 10);
  SELECT_EOB(13, 7);
  SELECT_EOB(14, 11);
  SELECT_EOB(15, 14);
  SELECT_EOB(16, 15);

  y0 = _mm_load_si128((__m128i *)(d->qcoeff));
  y1 = _mm_load_si128((__m128i *)(d->qcoeff + 8));

  /* dqcoeff = qcoeff * dequant */
  y0 = _mm_mullo_epi16(y0, dequant0);
  y1 = _mm_mullo_epi16(y1, dequant1);

  _mm_store_si128((__m128i *)(d->dqcoeff), y0);
  _mm_store_si128((__m128i *)(d->dqcoeff + 8), y1);

  *d->eob = eob;
}

void vp8_fast_quantize_b_sse2(BLOCK *b, BLOCKD *d) {
  __m128i z0 = _mm_load_si128((__m128i *)(b->coeff));
  __m128i z1 = _mm_load_si128((__m128i *)(b->coeff + 8));
  __m128i round0 = _mm_load_si128((__m128i *)(b->round));
  __m128i round1 = _mm_load_si128((__m128i *)(b->round + 8));
  __m128i quant_fast0 = _mm_load_si128((__m128i *)(b->quant_fast));
  __m128i quant_fast1 = _mm_load_si128((__m128i *)(b->quant_fast + 8));
  __m128i dequant0 = _mm_load_si128((__m128i *)(d->dequant));
  __m128i dequant1 = _mm_load_si128((__m128i *)(d->dequant + 8));
  __m128i inv_zig_zag0 =
      _mm_load_si128((const __m128i *)(vp8_default_inv_zig_zag));
  __m128i inv_zig_zag1 =
      _mm_load_si128((const __m128i *)(vp8_default_inv_zig_zag + 8));

  __m128i sz0, sz1, x0, x1, y0, y1, xdq0, xdq1, zeros, ones;

  /* sign of z: z >> 15 */
  sz0 = _mm_srai_epi16(z0, 15);
  sz1 = _mm_srai_epi16(z1, 15);

  /* x = abs(z): (z ^ sz) - sz */
  x0 = _mm_xor_si128(z0, sz0);
  x1 = _mm_xor_si128(z1, sz1);
  x0 = _mm_sub_epi16(x0, sz0);
  x1 = _mm_sub_epi16(x1, sz1);

  /* x += round */
  x0 = _mm_add_epi16(x0, round0);
  x1 = _mm_add_epi16(x1, round1);

  /* y = (x * quant) >> 16 */
  y0 = _mm_mulhi_epi16(x0, quant_fast0);
  y1 = _mm_mulhi_epi16(x1, quant_fast1);

  /* x = abs(y) = (y ^ sz) - sz */
  y0 = _mm_xor_si128(y0, sz0);
  y1 = _mm_xor_si128(y1, sz1);
  x0 = _mm_sub_epi16(y0, sz0);
  x1 = _mm_sub_epi16(y1, sz1);

  /* qcoeff = x */
  _mm_store_si128((__m128i *)(d->qcoeff), x0);
  _mm_store_si128((__m128i *)(d->qcoeff + 8), x1);

  /* x * dequant */
  xdq0 = _mm_mullo_epi16(x0, dequant0);
  xdq1 = _mm_mullo_epi16(x1, dequant1);

  /* dqcoeff = x * dequant */
  _mm_store_si128((__m128i *)(d->dqcoeff), xdq0);
  _mm_store_si128((__m128i *)(d->dqcoeff + 8), xdq1);

  /* build a mask for the zig zag */
  zeros = _mm_setzero_si128();

  x0 = _mm_cmpeq_epi16(x0, zeros);
  x1 = _mm_cmpeq_epi16(x1, zeros);

  ones = _mm_cmpeq_epi16(zeros, zeros);

  x0 = _mm_xor_si128(x0, ones);
  x1 = _mm_xor_si128(x1, ones);

  x0 = _mm_and_si128(x0, inv_zig_zag0);
  x1 = _mm_and_si128(x1, inv_zig_zag1);

  x0 = _mm_max_epi16(x0, x1);

  /* now down to 8 */
  x1 = _mm_shuffle_epi32(x0, 0xE);  // 0b00001110

  x0 = _mm_max_epi16(x0, x1);

  /* only 4 left */
  x1 = _mm_shufflelo_epi16(x0, 0xE);  // 0b00001110

  x0 = _mm_max_epi16(x0, x1);

  /* okay, just 2! */
  x1 = _mm_shufflelo_epi16(x0, 0x1);  // 0b00000001

  x0 = _mm_max_epi16(x0, x1);

  *d->eob = 0xFF & _mm_cvtsi128_si32(x0);
}
