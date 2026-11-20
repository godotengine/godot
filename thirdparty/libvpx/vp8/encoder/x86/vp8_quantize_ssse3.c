/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <tmmintrin.h> /* SSSE3 */

#include "./vp8_rtcd.h"
#include "vp8/encoder/block.h"
#include "vpx_ports/bitops.h" /* get_msb */

void vp8_fast_quantize_b_ssse3(BLOCK *b, BLOCKD *d) {
  int eob, mask;

  __m128i z0 = _mm_load_si128((__m128i *)(b->coeff));
  __m128i z1 = _mm_load_si128((__m128i *)(b->coeff + 8));
  __m128i round0 = _mm_load_si128((__m128i *)(b->round));
  __m128i round1 = _mm_load_si128((__m128i *)(b->round + 8));
  __m128i quant_fast0 = _mm_load_si128((__m128i *)(b->quant_fast));
  __m128i quant_fast1 = _mm_load_si128((__m128i *)(b->quant_fast + 8));
  __m128i dequant0 = _mm_load_si128((__m128i *)(d->dequant));
  __m128i dequant1 = _mm_load_si128((__m128i *)(d->dequant + 8));

  __m128i sz0, sz1, x, x0, x1, y0, y1, zeros, abs0, abs1;

  DECLARE_ALIGNED(16, const uint8_t,
                  pshufb_zig_zag_mask[16]) = { 0, 1,  4,  8,  5, 2,  3,  6,
                                               9, 12, 13, 10, 7, 11, 14, 15 };
  __m128i zig_zag = _mm_load_si128((const __m128i *)pshufb_zig_zag_mask);

  /* sign of z: z >> 15 */
  sz0 = _mm_srai_epi16(z0, 15);
  sz1 = _mm_srai_epi16(z1, 15);

  /* x = abs(z) */
  x0 = _mm_abs_epi16(z0);
  x1 = _mm_abs_epi16(z1);

  /* x += round */
  x0 = _mm_add_epi16(x0, round0);
  x1 = _mm_add_epi16(x1, round1);

  /* y = (x * quant) >> 16 */
  y0 = _mm_mulhi_epi16(x0, quant_fast0);
  y1 = _mm_mulhi_epi16(x1, quant_fast1);

  /* ASM saves Y for EOB */
  /* I think we can ignore that because adding the sign doesn't change anything
   * and multiplying 0 by dequant is OK as well */
  abs0 = y0;
  abs1 = y1;

  /* Restore the sign bit. */
  y0 = _mm_xor_si128(y0, sz0);
  y1 = _mm_xor_si128(y1, sz1);
  x0 = _mm_sub_epi16(y0, sz0);
  x1 = _mm_sub_epi16(y1, sz1);

  /* qcoeff = x */
  _mm_store_si128((__m128i *)(d->qcoeff), x0);
  _mm_store_si128((__m128i *)(d->qcoeff + 8), x1);

  /* x * dequant */
  x0 = _mm_mullo_epi16(x0, dequant0);
  x1 = _mm_mullo_epi16(x1, dequant1);

  /* dqcoeff = x * dequant */
  _mm_store_si128((__m128i *)(d->dqcoeff), x0);
  _mm_store_si128((__m128i *)(d->dqcoeff + 8), x1);

  zeros = _mm_setzero_si128();

  x0 = _mm_cmpgt_epi16(abs0, zeros);
  x1 = _mm_cmpgt_epi16(abs1, zeros);

  x = _mm_packs_epi16(x0, x1);

  x = _mm_shuffle_epi8(x, zig_zag);

  mask = _mm_movemask_epi8(x);

  /* x2 is needed to increase the result from non-zero masks by 1,
   * +1 is needed to mask undefined behavior for a null argument,
   * the result of get_msb(1) is 0 */
  eob = get_msb(mask * 2 + 1);

  *d->eob = eob;
}
