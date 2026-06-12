/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>  // SSE2

#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/mem.h"

static INLINE __m128i signed_char_clamp_bd_sse2(__m128i value, int bd) {
  __m128i ubounded;
  __m128i lbounded;
  __m128i retval;

  const __m128i zero = _mm_setzero_si128();
  const __m128i one = _mm_set1_epi16(1);
  __m128i t80, max, min;

  if (bd == 8) {
    t80 = _mm_set1_epi16(0x80);
    max = _mm_subs_epi16(_mm_subs_epi16(_mm_slli_epi16(one, 8), one), t80);
  } else if (bd == 10) {
    t80 = _mm_set1_epi16(0x200);
    max = _mm_subs_epi16(_mm_subs_epi16(_mm_slli_epi16(one, 10), one), t80);
  } else {  // bd == 12
    t80 = _mm_set1_epi16(0x800);
    max = _mm_subs_epi16(_mm_subs_epi16(_mm_slli_epi16(one, 12), one), t80);
  }

  min = _mm_subs_epi16(zero, t80);

  ubounded = _mm_cmpgt_epi16(value, max);
  lbounded = _mm_cmplt_epi16(value, min);
  retval = _mm_andnot_si128(_mm_or_si128(ubounded, lbounded), value);
  ubounded = _mm_and_si128(ubounded, max);
  lbounded = _mm_and_si128(lbounded, min);
  retval = _mm_or_si128(retval, ubounded);
  retval = _mm_or_si128(retval, lbounded);
  return retval;
}

// TODO(debargha, peter): Break up large functions into smaller ones
// in this file.
void vpx_highbd_lpf_horizontal_16_sse2(uint16_t *s, int pitch,
                                       const uint8_t *blimit,
                                       const uint8_t *limit,
                                       const uint8_t *thresh, int bd) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i one = _mm_set1_epi16(1);
  __m128i blimit_v, limit_v, thresh_v;
  __m128i q7, p7, q6, p6, q5, p5, q4, p4, q3, p3, q2, p2, q1, p1, q0, p0;
  __m128i mask, hev, flat, flat2, abs_p1p0, abs_q1q0;
  __m128i ps1, qs1, ps0, qs0;
  __m128i abs_p0q0, abs_p1q1, ffff, work;
  __m128i filt, work_a, filter1, filter2;
  __m128i flat2_q6, flat2_p6, flat2_q5, flat2_p5, flat2_q4, flat2_p4;
  __m128i flat2_q3, flat2_p3, flat2_q2, flat2_p2, flat2_q1, flat2_p1;
  __m128i flat2_q0, flat2_p0;
  __m128i flat_q2, flat_p2, flat_q1, flat_p1, flat_q0, flat_p0;
  __m128i pixelFilter_p, pixelFilter_q;
  __m128i pixetFilter_p2p1p0, pixetFilter_q2q1q0;
  __m128i sum_p7, sum_q7, sum_p3, sum_q3;
  __m128i t4, t3, t80, t1;
  __m128i eight, four;

  if (bd == 8) {
    blimit_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero);
    limit_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero);
    thresh_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero);
  } else if (bd == 10) {
    blimit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero), 2);
    limit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero), 2);
    thresh_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero), 2);
  } else {  // bd == 12
    blimit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero), 4);
    limit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero), 4);
    thresh_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero), 4);
  }

  q4 = _mm_load_si128((__m128i *)(s + 4 * pitch));
  p4 = _mm_load_si128((__m128i *)(s - 5 * pitch));
  q3 = _mm_load_si128((__m128i *)(s + 3 * pitch));
  p3 = _mm_load_si128((__m128i *)(s - 4 * pitch));
  q2 = _mm_load_si128((__m128i *)(s + 2 * pitch));
  p2 = _mm_load_si128((__m128i *)(s - 3 * pitch));
  q1 = _mm_load_si128((__m128i *)(s + 1 * pitch));
  p1 = _mm_load_si128((__m128i *)(s - 2 * pitch));
  q0 = _mm_load_si128((__m128i *)(s + 0 * pitch));
  p0 = _mm_load_si128((__m128i *)(s - 1 * pitch));

  //  highbd_filter_mask
  abs_p1p0 = _mm_or_si128(_mm_subs_epu16(p1, p0), _mm_subs_epu16(p0, p1));
  abs_q1q0 = _mm_or_si128(_mm_subs_epu16(q1, q0), _mm_subs_epu16(q0, q1));

  ffff = _mm_cmpeq_epi16(abs_p1p0, abs_p1p0);

  abs_p0q0 = _mm_or_si128(_mm_subs_epu16(p0, q0), _mm_subs_epu16(q0, p0));
  abs_p1q1 = _mm_or_si128(_mm_subs_epu16(p1, q1), _mm_subs_epu16(q1, p1));

  //  highbd_hev_mask (in C code this is actually called from highbd_filter4)
  flat = _mm_max_epi16(abs_p1p0, abs_q1q0);
  hev = _mm_subs_epu16(flat, thresh_v);
  hev = _mm_xor_si128(_mm_cmpeq_epi16(hev, zero), ffff);

  abs_p0q0 = _mm_adds_epu16(abs_p0q0, abs_p0q0);  // abs(p0 - q0) * 2
  abs_p1q1 = _mm_srli_epi16(abs_p1q1, 1);         // abs(p1 - q1) / 2
  mask = _mm_subs_epu16(_mm_adds_epu16(abs_p0q0, abs_p1q1), blimit_v);
  mask = _mm_xor_si128(_mm_cmpeq_epi16(mask, zero), ffff);
  mask = _mm_and_si128(mask, _mm_adds_epu16(limit_v, one));
  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p1, p0), _mm_subs_epu16(p0, p1)),
      _mm_or_si128(_mm_subs_epu16(q1, q0), _mm_subs_epu16(q0, q1)));
  mask = _mm_max_epi16(work, mask);
  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p2, p1), _mm_subs_epu16(p1, p2)),
      _mm_or_si128(_mm_subs_epu16(q2, q1), _mm_subs_epu16(q1, q2)));
  mask = _mm_max_epi16(work, mask);
  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p3, p2), _mm_subs_epu16(p2, p3)),
      _mm_or_si128(_mm_subs_epu16(q3, q2), _mm_subs_epu16(q2, q3)));
  mask = _mm_max_epi16(work, mask);

  mask = _mm_subs_epu16(mask, limit_v);
  mask = _mm_cmpeq_epi16(mask, zero);  // return ~mask

  // lp filter
  // highbd_filter4
  t4 = _mm_set1_epi16(4);
  t3 = _mm_set1_epi16(3);
  if (bd == 8)
    t80 = _mm_set1_epi16(0x80);
  else if (bd == 10)
    t80 = _mm_set1_epi16(0x200);
  else  // bd == 12
    t80 = _mm_set1_epi16(0x800);

  t1 = _mm_set1_epi16(0x1);

  ps1 = _mm_subs_epi16(p1, t80);
  qs1 = _mm_subs_epi16(q1, t80);
  ps0 = _mm_subs_epi16(p0, t80);
  qs0 = _mm_subs_epi16(q0, t80);

  filt = _mm_and_si128(signed_char_clamp_bd_sse2(_mm_subs_epi16(ps1, qs1), bd),
                       hev);
  work_a = _mm_subs_epi16(qs0, ps0);
  filt = _mm_adds_epi16(filt, work_a);
  filt = _mm_adds_epi16(filt, work_a);
  filt = signed_char_clamp_bd_sse2(_mm_adds_epi16(filt, work_a), bd);
  filt = _mm_and_si128(filt, mask);
  filter1 = signed_char_clamp_bd_sse2(_mm_adds_epi16(filt, t4), bd);
  filter2 = signed_char_clamp_bd_sse2(_mm_adds_epi16(filt, t3), bd);

  // Filter1 >> 3
  filter1 = _mm_srai_epi16(filter1, 0x3);
  filter2 = _mm_srai_epi16(filter2, 0x3);

  qs0 = _mm_adds_epi16(
      signed_char_clamp_bd_sse2(_mm_subs_epi16(qs0, filter1), bd), t80);
  ps0 = _mm_adds_epi16(
      signed_char_clamp_bd_sse2(_mm_adds_epi16(ps0, filter2), bd), t80);
  filt = _mm_adds_epi16(filter1, t1);
  filt = _mm_srai_epi16(filt, 1);
  filt = _mm_andnot_si128(hev, filt);
  qs1 = _mm_adds_epi16(signed_char_clamp_bd_sse2(_mm_subs_epi16(qs1, filt), bd),
                       t80);
  ps1 = _mm_adds_epi16(signed_char_clamp_bd_sse2(_mm_adds_epi16(ps1, filt), bd),
                       t80);

  // end highbd_filter4
  // loopfilter done

  // highbd_flat_mask4
  flat = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p2, p0), _mm_subs_epu16(p0, p2)),
      _mm_or_si128(_mm_subs_epu16(p3, p0), _mm_subs_epu16(p0, p3)));
  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(q2, q0), _mm_subs_epu16(q0, q2)),
      _mm_or_si128(_mm_subs_epu16(q3, q0), _mm_subs_epu16(q0, q3)));
  flat = _mm_max_epi16(work, flat);
  work = _mm_max_epi16(abs_p1p0, abs_q1q0);
  flat = _mm_max_epi16(work, flat);

  if (bd == 8)
    flat = _mm_subs_epu16(flat, one);
  else if (bd == 10)
    flat = _mm_subs_epu16(flat, _mm_slli_epi16(one, 2));
  else  // bd == 12
    flat = _mm_subs_epu16(flat, _mm_slli_epi16(one, 4));

  flat = _mm_cmpeq_epi16(flat, zero);
  // end flat_mask4

  // flat & mask = flat && mask (as used in filter8)
  // (because, in both vars, each block of 16 either all 1s or all 0s)
  flat = _mm_and_si128(flat, mask);

  p5 = _mm_load_si128((__m128i *)(s - 6 * pitch));
  q5 = _mm_load_si128((__m128i *)(s + 5 * pitch));
  p6 = _mm_load_si128((__m128i *)(s - 7 * pitch));
  q6 = _mm_load_si128((__m128i *)(s + 6 * pitch));
  p7 = _mm_load_si128((__m128i *)(s - 8 * pitch));
  q7 = _mm_load_si128((__m128i *)(s + 7 * pitch));

  // highbd_flat_mask5 (arguments passed in are p0, q0, p4-p7, q4-q7
  // but referred to as p0-p4 & q0-q4 in fn)
  flat2 = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p4, p0), _mm_subs_epu16(p0, p4)),
      _mm_or_si128(_mm_subs_epu16(q4, q0), _mm_subs_epu16(q0, q4)));

  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p5, p0), _mm_subs_epu16(p0, p5)),
      _mm_or_si128(_mm_subs_epu16(q5, q0), _mm_subs_epu16(q0, q5)));
  flat2 = _mm_max_epi16(work, flat2);

  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p6, p0), _mm_subs_epu16(p0, p6)),
      _mm_or_si128(_mm_subs_epu16(q6, q0), _mm_subs_epu16(q0, q6)));
  flat2 = _mm_max_epi16(work, flat2);

  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p7, p0), _mm_subs_epu16(p0, p7)),
      _mm_or_si128(_mm_subs_epu16(q7, q0), _mm_subs_epu16(q0, q7)));
  flat2 = _mm_max_epi16(work, flat2);

  if (bd == 8)
    flat2 = _mm_subs_epu16(flat2, one);
  else if (bd == 10)
    flat2 = _mm_subs_epu16(flat2, _mm_slli_epi16(one, 2));
  else  // bd == 12
    flat2 = _mm_subs_epu16(flat2, _mm_slli_epi16(one, 4));

  flat2 = _mm_cmpeq_epi16(flat2, zero);
  flat2 = _mm_and_si128(flat2, flat);  // flat2 & flat & mask
  // end highbd_flat_mask5

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // flat and wide flat calculations
  eight = _mm_set1_epi16(8);
  four = _mm_set1_epi16(4);

  pixelFilter_p = _mm_add_epi16(_mm_add_epi16(p6, p5), _mm_add_epi16(p4, p3));
  pixelFilter_q = _mm_add_epi16(_mm_add_epi16(q6, q5), _mm_add_epi16(q4, q3));

  pixetFilter_p2p1p0 = _mm_add_epi16(p0, _mm_add_epi16(p2, p1));
  pixelFilter_p = _mm_add_epi16(pixelFilter_p, pixetFilter_p2p1p0);

  pixetFilter_q2q1q0 = _mm_add_epi16(q0, _mm_add_epi16(q2, q1));
  pixelFilter_q = _mm_add_epi16(pixelFilter_q, pixetFilter_q2q1q0);
  pixelFilter_p =
      _mm_add_epi16(eight, _mm_add_epi16(pixelFilter_p, pixelFilter_q));
  pixetFilter_p2p1p0 = _mm_add_epi16(
      four, _mm_add_epi16(pixetFilter_p2p1p0, pixetFilter_q2q1q0));
  flat2_p0 =
      _mm_srli_epi16(_mm_add_epi16(pixelFilter_p, _mm_add_epi16(p7, p0)), 4);
  flat2_q0 =
      _mm_srli_epi16(_mm_add_epi16(pixelFilter_p, _mm_add_epi16(q7, q0)), 4);
  flat_p0 = _mm_srli_epi16(
      _mm_add_epi16(pixetFilter_p2p1p0, _mm_add_epi16(p3, p0)), 3);
  flat_q0 = _mm_srli_epi16(
      _mm_add_epi16(pixetFilter_p2p1p0, _mm_add_epi16(q3, q0)), 3);

  sum_p7 = _mm_add_epi16(p7, p7);
  sum_q7 = _mm_add_epi16(q7, q7);
  sum_p3 = _mm_add_epi16(p3, p3);
  sum_q3 = _mm_add_epi16(q3, q3);

  pixelFilter_q = _mm_sub_epi16(pixelFilter_p, p6);
  pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q6);
  flat2_p1 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_p, _mm_add_epi16(sum_p7, p1)), 4);
  flat2_q1 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_q, _mm_add_epi16(sum_q7, q1)), 4);

  pixetFilter_q2q1q0 = _mm_sub_epi16(pixetFilter_p2p1p0, p2);
  pixetFilter_p2p1p0 = _mm_sub_epi16(pixetFilter_p2p1p0, q2);
  flat_p1 = _mm_srli_epi16(
      _mm_add_epi16(pixetFilter_p2p1p0, _mm_add_epi16(sum_p3, p1)), 3);
  flat_q1 = _mm_srli_epi16(
      _mm_add_epi16(pixetFilter_q2q1q0, _mm_add_epi16(sum_q3, q1)), 3);

  sum_p7 = _mm_add_epi16(sum_p7, p7);
  sum_q7 = _mm_add_epi16(sum_q7, q7);
  sum_p3 = _mm_add_epi16(sum_p3, p3);
  sum_q3 = _mm_add_epi16(sum_q3, q3);

  pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q5);
  pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p5);
  flat2_p2 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_p, _mm_add_epi16(sum_p7, p2)), 4);
  flat2_q2 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_q, _mm_add_epi16(sum_q7, q2)), 4);

  pixetFilter_p2p1p0 = _mm_sub_epi16(pixetFilter_p2p1p0, q1);
  pixetFilter_q2q1q0 = _mm_sub_epi16(pixetFilter_q2q1q0, p1);
  flat_p2 = _mm_srli_epi16(
      _mm_add_epi16(pixetFilter_p2p1p0, _mm_add_epi16(sum_p3, p2)), 3);
  flat_q2 = _mm_srli_epi16(
      _mm_add_epi16(pixetFilter_q2q1q0, _mm_add_epi16(sum_q3, q2)), 3);

  sum_p7 = _mm_add_epi16(sum_p7, p7);
  sum_q7 = _mm_add_epi16(sum_q7, q7);
  pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q4);
  pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p4);
  flat2_p3 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_p, _mm_add_epi16(sum_p7, p3)), 4);
  flat2_q3 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_q, _mm_add_epi16(sum_q7, q3)), 4);

  sum_p7 = _mm_add_epi16(sum_p7, p7);
  sum_q7 = _mm_add_epi16(sum_q7, q7);
  pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q3);
  pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p3);
  flat2_p4 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_p, _mm_add_epi16(sum_p7, p4)), 4);
  flat2_q4 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_q, _mm_add_epi16(sum_q7, q4)), 4);

  sum_p7 = _mm_add_epi16(sum_p7, p7);
  sum_q7 = _mm_add_epi16(sum_q7, q7);
  pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q2);
  pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p2);
  flat2_p5 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_p, _mm_add_epi16(sum_p7, p5)), 4);
  flat2_q5 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_q, _mm_add_epi16(sum_q7, q5)), 4);

  sum_p7 = _mm_add_epi16(sum_p7, p7);
  sum_q7 = _mm_add_epi16(sum_q7, q7);
  pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q1);
  pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p1);
  flat2_p6 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_p, _mm_add_epi16(sum_p7, p6)), 4);
  flat2_q6 = _mm_srli_epi16(
      _mm_add_epi16(pixelFilter_q, _mm_add_epi16(sum_q7, q6)), 4);

  //  wide flat
  //  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  //  highbd_filter8
  p2 = _mm_andnot_si128(flat, p2);
  //  p2 remains unchanged if !(flat && mask)
  flat_p2 = _mm_and_si128(flat, flat_p2);
  //  when (flat && mask)
  p2 = _mm_or_si128(p2, flat_p2);  // full list of p2 values
  q2 = _mm_andnot_si128(flat, q2);
  flat_q2 = _mm_and_si128(flat, flat_q2);
  q2 = _mm_or_si128(q2, flat_q2);  // full list of q2 values

  ps1 = _mm_andnot_si128(flat, ps1);
  //  p1 takes the value assigned to in in filter4 if !(flat && mask)
  flat_p1 = _mm_and_si128(flat, flat_p1);
  //  when (flat && mask)
  p1 = _mm_or_si128(ps1, flat_p1);  // full list of p1 values
  qs1 = _mm_andnot_si128(flat, qs1);
  flat_q1 = _mm_and_si128(flat, flat_q1);
  q1 = _mm_or_si128(qs1, flat_q1);  // full list of q1 values

  ps0 = _mm_andnot_si128(flat, ps0);
  //  p0 takes the value assigned to in in filter4 if !(flat && mask)
  flat_p0 = _mm_and_si128(flat, flat_p0);
  //  when (flat && mask)
  p0 = _mm_or_si128(ps0, flat_p0);  // full list of p0 values
  qs0 = _mm_andnot_si128(flat, qs0);
  flat_q0 = _mm_and_si128(flat, flat_q0);
  q0 = _mm_or_si128(qs0, flat_q0);  // full list of q0 values
  // end highbd_filter8

  // highbd_filter16
  p6 = _mm_andnot_si128(flat2, p6);
  //  p6 remains unchanged if !(flat2 && flat && mask)
  flat2_p6 = _mm_and_si128(flat2, flat2_p6);
  //  get values for when (flat2 && flat && mask)
  p6 = _mm_or_si128(p6, flat2_p6);  // full list of p6 values
  q6 = _mm_andnot_si128(flat2, q6);
  //  q6 remains unchanged if !(flat2 && flat && mask)
  flat2_q6 = _mm_and_si128(flat2, flat2_q6);
  //  get values for when (flat2 && flat && mask)
  q6 = _mm_or_si128(q6, flat2_q6);  // full list of q6 values
  _mm_store_si128((__m128i *)(s - 7 * pitch), p6);
  _mm_store_si128((__m128i *)(s + 6 * pitch), q6);

  p5 = _mm_andnot_si128(flat2, p5);
  //  p5 remains unchanged if !(flat2 && flat && mask)
  flat2_p5 = _mm_and_si128(flat2, flat2_p5);
  //  get values for when (flat2 && flat && mask)
  p5 = _mm_or_si128(p5, flat2_p5);
  //  full list of p5 values
  q5 = _mm_andnot_si128(flat2, q5);
  //  q5 remains unchanged if !(flat2 && flat && mask)
  flat2_q5 = _mm_and_si128(flat2, flat2_q5);
  //  get values for when (flat2 && flat && mask)
  q5 = _mm_or_si128(q5, flat2_q5);
  //  full list of q5 values
  _mm_store_si128((__m128i *)(s - 6 * pitch), p5);
  _mm_store_si128((__m128i *)(s + 5 * pitch), q5);

  p4 = _mm_andnot_si128(flat2, p4);
  //  p4 remains unchanged if !(flat2 && flat && mask)
  flat2_p4 = _mm_and_si128(flat2, flat2_p4);
  //  get values for when (flat2 && flat && mask)
  p4 = _mm_or_si128(p4, flat2_p4);  // full list of p4 values
  q4 = _mm_andnot_si128(flat2, q4);
  //  q4 remains unchanged if !(flat2 && flat && mask)
  flat2_q4 = _mm_and_si128(flat2, flat2_q4);
  //  get values for when (flat2 && flat && mask)
  q4 = _mm_or_si128(q4, flat2_q4);  // full list of q4 values
  _mm_store_si128((__m128i *)(s - 5 * pitch), p4);
  _mm_store_si128((__m128i *)(s + 4 * pitch), q4);

  p3 = _mm_andnot_si128(flat2, p3);
  //  p3 takes value from highbd_filter8 if !(flat2 && flat && mask)
  flat2_p3 = _mm_and_si128(flat2, flat2_p3);
  //  get values for when (flat2 && flat && mask)
  p3 = _mm_or_si128(p3, flat2_p3);  // full list of p3 values
  q3 = _mm_andnot_si128(flat2, q3);
  //  q3 takes value from highbd_filter8 if !(flat2 && flat && mask)
  flat2_q3 = _mm_and_si128(flat2, flat2_q3);
  //  get values for when (flat2 && flat && mask)
  q3 = _mm_or_si128(q3, flat2_q3);  // full list of q3 values
  _mm_store_si128((__m128i *)(s - 4 * pitch), p3);
  _mm_store_si128((__m128i *)(s + 3 * pitch), q3);

  p2 = _mm_andnot_si128(flat2, p2);
  //  p2 takes value from highbd_filter8 if !(flat2 && flat && mask)
  flat2_p2 = _mm_and_si128(flat2, flat2_p2);
  //  get values for when (flat2 && flat && mask)
  p2 = _mm_or_si128(p2, flat2_p2);
  //  full list of p2 values
  q2 = _mm_andnot_si128(flat2, q2);
  //  q2 takes value from highbd_filter8 if !(flat2 && flat && mask)
  flat2_q2 = _mm_and_si128(flat2, flat2_q2);
  //  get values for when (flat2 && flat && mask)
  q2 = _mm_or_si128(q2, flat2_q2);  // full list of q2 values
  _mm_store_si128((__m128i *)(s - 3 * pitch), p2);
  _mm_store_si128((__m128i *)(s + 2 * pitch), q2);

  p1 = _mm_andnot_si128(flat2, p1);
  //  p1 takes value from highbd_filter8 if !(flat2 && flat && mask)
  flat2_p1 = _mm_and_si128(flat2, flat2_p1);
  //  get values for when (flat2 && flat && mask)
  p1 = _mm_or_si128(p1, flat2_p1);  // full list of p1 values
  q1 = _mm_andnot_si128(flat2, q1);
  //  q1 takes value from highbd_filter8 if !(flat2 && flat && mask)
  flat2_q1 = _mm_and_si128(flat2, flat2_q1);
  //  get values for when (flat2 && flat && mask)
  q1 = _mm_or_si128(q1, flat2_q1);  // full list of q1 values
  _mm_store_si128((__m128i *)(s - 2 * pitch), p1);
  _mm_store_si128((__m128i *)(s + 1 * pitch), q1);

  p0 = _mm_andnot_si128(flat2, p0);
  //  p0 takes value from highbd_filter8 if !(flat2 && flat && mask)
  flat2_p0 = _mm_and_si128(flat2, flat2_p0);
  //  get values for when (flat2 && flat && mask)
  p0 = _mm_or_si128(p0, flat2_p0);  // full list of p0 values
  q0 = _mm_andnot_si128(flat2, q0);
  //  q0 takes value from highbd_filter8 if !(flat2 && flat && mask)
  flat2_q0 = _mm_and_si128(flat2, flat2_q0);
  //  get values for when (flat2 && flat && mask)
  q0 = _mm_or_si128(q0, flat2_q0);  // full list of q0 values
  _mm_store_si128((__m128i *)(s - 1 * pitch), p0);
  _mm_store_si128((__m128i *)(s - 0 * pitch), q0);
}

void vpx_highbd_lpf_horizontal_16_dual_sse2(uint16_t *s, int pitch,
                                            const uint8_t *blimit,
                                            const uint8_t *limit,
                                            const uint8_t *thresh, int bd) {
  vpx_highbd_lpf_horizontal_16_sse2(s, pitch, blimit, limit, thresh, bd);
  vpx_highbd_lpf_horizontal_16_sse2(s + 8, pitch, blimit, limit, thresh, bd);
}

void vpx_highbd_lpf_horizontal_8_sse2(uint16_t *s, int pitch,
                                      const uint8_t *blimit,
                                      const uint8_t *limit,
                                      const uint8_t *thresh, int bd) {
  DECLARE_ALIGNED(16, uint16_t, flat_op2[16]);
  DECLARE_ALIGNED(16, uint16_t, flat_op1[16]);
  DECLARE_ALIGNED(16, uint16_t, flat_op0[16]);
  DECLARE_ALIGNED(16, uint16_t, flat_oq2[16]);
  DECLARE_ALIGNED(16, uint16_t, flat_oq1[16]);
  DECLARE_ALIGNED(16, uint16_t, flat_oq0[16]);
  const __m128i zero = _mm_setzero_si128();
  __m128i blimit_v, limit_v, thresh_v;
  __m128i mask, hev, flat;
  __m128i p3 = _mm_load_si128((__m128i *)(s - 4 * pitch));
  __m128i q3 = _mm_load_si128((__m128i *)(s + 3 * pitch));
  __m128i p2 = _mm_load_si128((__m128i *)(s - 3 * pitch));
  __m128i q2 = _mm_load_si128((__m128i *)(s + 2 * pitch));
  __m128i p1 = _mm_load_si128((__m128i *)(s - 2 * pitch));
  __m128i q1 = _mm_load_si128((__m128i *)(s + 1 * pitch));
  __m128i p0 = _mm_load_si128((__m128i *)(s - 1 * pitch));
  __m128i q0 = _mm_load_si128((__m128i *)(s + 0 * pitch));
  const __m128i one = _mm_set1_epi16(1);
  const __m128i ffff = _mm_cmpeq_epi16(one, one);
  __m128i abs_p1q1, abs_p0q0, abs_q1q0, abs_p1p0, work;
  const __m128i four = _mm_set1_epi16(4);
  __m128i workp_a, workp_b, workp_shft;

  const __m128i t4 = _mm_set1_epi16(4);
  const __m128i t3 = _mm_set1_epi16(3);
  __m128i t80;
  const __m128i t1 = _mm_set1_epi16(0x1);
  __m128i ps1, ps0, qs0, qs1;
  __m128i filt;
  __m128i work_a;
  __m128i filter1, filter2;

  if (bd == 8) {
    blimit_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero);
    limit_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero);
    thresh_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero);
    t80 = _mm_set1_epi16(0x80);
  } else if (bd == 10) {
    blimit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero), 2);
    limit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero), 2);
    thresh_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero), 2);
    t80 = _mm_set1_epi16(0x200);
  } else {  // bd == 12
    blimit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero), 4);
    limit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero), 4);
    thresh_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero), 4);
    t80 = _mm_set1_epi16(0x800);
  }

  ps1 = _mm_subs_epi16(p1, t80);
  ps0 = _mm_subs_epi16(p0, t80);
  qs0 = _mm_subs_epi16(q0, t80);
  qs1 = _mm_subs_epi16(q1, t80);

  // filter_mask and hev_mask
  abs_p1p0 = _mm_or_si128(_mm_subs_epu16(p1, p0), _mm_subs_epu16(p0, p1));
  abs_q1q0 = _mm_or_si128(_mm_subs_epu16(q1, q0), _mm_subs_epu16(q0, q1));

  abs_p0q0 = _mm_or_si128(_mm_subs_epu16(p0, q0), _mm_subs_epu16(q0, p0));
  abs_p1q1 = _mm_or_si128(_mm_subs_epu16(p1, q1), _mm_subs_epu16(q1, p1));
  flat = _mm_max_epi16(abs_p1p0, abs_q1q0);
  hev = _mm_subs_epu16(flat, thresh_v);
  hev = _mm_xor_si128(_mm_cmpeq_epi16(hev, zero), ffff);

  abs_p0q0 = _mm_adds_epu16(abs_p0q0, abs_p0q0);
  abs_p1q1 = _mm_srli_epi16(abs_p1q1, 1);
  mask = _mm_subs_epu16(_mm_adds_epu16(abs_p0q0, abs_p1q1), blimit_v);
  mask = _mm_xor_si128(_mm_cmpeq_epi16(mask, zero), ffff);
  // mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > blimit) * -1;
  // So taking maximums continues to work:
  mask = _mm_and_si128(mask, _mm_adds_epu16(limit_v, one));
  mask = _mm_max_epi16(abs_p1p0, mask);
  // mask |= (abs(p1 - p0) > limit) * -1;
  mask = _mm_max_epi16(abs_q1q0, mask);
  // mask |= (abs(q1 - q0) > limit) * -1;

  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p2, p1), _mm_subs_epu16(p1, p2)),
      _mm_or_si128(_mm_subs_epu16(q2, q1), _mm_subs_epu16(q1, q2)));
  mask = _mm_max_epi16(work, mask);
  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p3, p2), _mm_subs_epu16(p2, p3)),
      _mm_or_si128(_mm_subs_epu16(q3, q2), _mm_subs_epu16(q2, q3)));
  mask = _mm_max_epi16(work, mask);
  mask = _mm_subs_epu16(mask, limit_v);
  mask = _mm_cmpeq_epi16(mask, zero);

  // flat_mask4
  flat = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p2, p0), _mm_subs_epu16(p0, p2)),
      _mm_or_si128(_mm_subs_epu16(q2, q0), _mm_subs_epu16(q0, q2)));
  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p3, p0), _mm_subs_epu16(p0, p3)),
      _mm_or_si128(_mm_subs_epu16(q3, q0), _mm_subs_epu16(q0, q3)));
  flat = _mm_max_epi16(work, flat);
  flat = _mm_max_epi16(abs_p1p0, flat);
  flat = _mm_max_epi16(abs_q1q0, flat);

  if (bd == 8)
    flat = _mm_subs_epu16(flat, one);
  else if (bd == 10)
    flat = _mm_subs_epu16(flat, _mm_slli_epi16(one, 2));
  else  // bd == 12
    flat = _mm_subs_epu16(flat, _mm_slli_epi16(one, 4));

  flat = _mm_cmpeq_epi16(flat, zero);
  flat = _mm_and_si128(flat, mask);  // flat & mask

  // Added before shift for rounding part of ROUND_POWER_OF_TWO

  workp_a = _mm_add_epi16(_mm_add_epi16(p3, p3), _mm_add_epi16(p2, p1));
  workp_a = _mm_add_epi16(_mm_add_epi16(workp_a, four), p0);
  workp_b = _mm_add_epi16(_mm_add_epi16(q0, p2), p3);
  workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
  _mm_store_si128((__m128i *)&flat_op2[0], workp_shft);

  workp_b = _mm_add_epi16(_mm_add_epi16(q0, q1), p1);
  workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
  _mm_store_si128((__m128i *)&flat_op1[0], workp_shft);

  workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3), q2);
  workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p1), p0);
  workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
  _mm_store_si128((__m128i *)&flat_op0[0], workp_shft);

  workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3), q3);
  workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p0), q0);
  workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
  _mm_store_si128((__m128i *)&flat_oq0[0], workp_shft);

  workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p2), q3);
  workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q0), q1);
  workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
  _mm_store_si128((__m128i *)&flat_oq1[0], workp_shft);

  workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p1), q3);
  workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q1), q2);
  workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
  _mm_store_si128((__m128i *)&flat_oq2[0], workp_shft);

  // lp filter
  filt = signed_char_clamp_bd_sse2(_mm_subs_epi16(ps1, qs1), bd);
  filt = _mm_and_si128(filt, hev);
  work_a = _mm_subs_epi16(qs0, ps0);
  filt = _mm_adds_epi16(filt, work_a);
  filt = _mm_adds_epi16(filt, work_a);
  filt = _mm_adds_epi16(filt, work_a);
  // (vpx_filter + 3 * (qs0 - ps0)) & mask
  filt = signed_char_clamp_bd_sse2(filt, bd);
  filt = _mm_and_si128(filt, mask);

  filter1 = _mm_adds_epi16(filt, t4);
  filter2 = _mm_adds_epi16(filt, t3);

  // Filter1 >> 3
  filter1 = signed_char_clamp_bd_sse2(filter1, bd);
  filter1 = _mm_srai_epi16(filter1, 3);

  // Filter2 >> 3
  filter2 = signed_char_clamp_bd_sse2(filter2, bd);
  filter2 = _mm_srai_epi16(filter2, 3);

  // filt >> 1
  filt = _mm_adds_epi16(filter1, t1);
  filt = _mm_srai_epi16(filt, 1);
  // filter = ROUND_POWER_OF_TWO(filter1, 1) & ~hev;
  filt = _mm_andnot_si128(hev, filt);

  work_a = signed_char_clamp_bd_sse2(_mm_subs_epi16(qs0, filter1), bd);
  work_a = _mm_adds_epi16(work_a, t80);
  q0 = _mm_load_si128((__m128i *)flat_oq0);
  work_a = _mm_andnot_si128(flat, work_a);
  q0 = _mm_and_si128(flat, q0);
  q0 = _mm_or_si128(work_a, q0);

  work_a = signed_char_clamp_bd_sse2(_mm_subs_epi16(qs1, filt), bd);
  work_a = _mm_adds_epi16(work_a, t80);
  q1 = _mm_load_si128((__m128i *)flat_oq1);
  work_a = _mm_andnot_si128(flat, work_a);
  q1 = _mm_and_si128(flat, q1);
  q1 = _mm_or_si128(work_a, q1);

  work_a = _mm_loadu_si128((__m128i *)(s + 2 * pitch));
  q2 = _mm_load_si128((__m128i *)flat_oq2);
  work_a = _mm_andnot_si128(flat, work_a);
  q2 = _mm_and_si128(flat, q2);
  q2 = _mm_or_si128(work_a, q2);

  work_a = signed_char_clamp_bd_sse2(_mm_adds_epi16(ps0, filter2), bd);
  work_a = _mm_adds_epi16(work_a, t80);
  p0 = _mm_load_si128((__m128i *)flat_op0);
  work_a = _mm_andnot_si128(flat, work_a);
  p0 = _mm_and_si128(flat, p0);
  p0 = _mm_or_si128(work_a, p0);

  work_a = signed_char_clamp_bd_sse2(_mm_adds_epi16(ps1, filt), bd);
  work_a = _mm_adds_epi16(work_a, t80);
  p1 = _mm_load_si128((__m128i *)flat_op1);
  work_a = _mm_andnot_si128(flat, work_a);
  p1 = _mm_and_si128(flat, p1);
  p1 = _mm_or_si128(work_a, p1);

  work_a = _mm_loadu_si128((__m128i *)(s - 3 * pitch));
  p2 = _mm_load_si128((__m128i *)flat_op2);
  work_a = _mm_andnot_si128(flat, work_a);
  p2 = _mm_and_si128(flat, p2);
  p2 = _mm_or_si128(work_a, p2);

  _mm_store_si128((__m128i *)(s - 3 * pitch), p2);
  _mm_store_si128((__m128i *)(s - 2 * pitch), p1);
  _mm_store_si128((__m128i *)(s - 1 * pitch), p0);
  _mm_store_si128((__m128i *)(s + 0 * pitch), q0);
  _mm_store_si128((__m128i *)(s + 1 * pitch), q1);
  _mm_store_si128((__m128i *)(s + 2 * pitch), q2);
}

void vpx_highbd_lpf_horizontal_8_dual_sse2(
    uint16_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  vpx_highbd_lpf_horizontal_8_sse2(s, pitch, blimit0, limit0, thresh0, bd);
  vpx_highbd_lpf_horizontal_8_sse2(s + 8, pitch, blimit1, limit1, thresh1, bd);
}

void vpx_highbd_lpf_horizontal_4_sse2(uint16_t *s, int pitch,
                                      const uint8_t *blimit,
                                      const uint8_t *limit,
                                      const uint8_t *thresh, int bd) {
  const __m128i zero = _mm_setzero_si128();
  __m128i blimit_v, limit_v, thresh_v;
  __m128i mask, hev, flat;
  __m128i p3 = _mm_loadu_si128((__m128i *)(s - 4 * pitch));
  __m128i p2 = _mm_loadu_si128((__m128i *)(s - 3 * pitch));
  __m128i p1 = _mm_loadu_si128((__m128i *)(s - 2 * pitch));
  __m128i p0 = _mm_loadu_si128((__m128i *)(s - 1 * pitch));
  __m128i q0 = _mm_loadu_si128((__m128i *)(s - 0 * pitch));
  __m128i q1 = _mm_loadu_si128((__m128i *)(s + 1 * pitch));
  __m128i q2 = _mm_loadu_si128((__m128i *)(s + 2 * pitch));
  __m128i q3 = _mm_loadu_si128((__m128i *)(s + 3 * pitch));
  const __m128i abs_p1p0 =
      _mm_or_si128(_mm_subs_epu16(p1, p0), _mm_subs_epu16(p0, p1));
  const __m128i abs_q1q0 =
      _mm_or_si128(_mm_subs_epu16(q1, q0), _mm_subs_epu16(q0, q1));
  const __m128i ffff = _mm_cmpeq_epi16(abs_p1p0, abs_p1p0);
  const __m128i one = _mm_set1_epi16(1);
  __m128i abs_p0q0 =
      _mm_or_si128(_mm_subs_epu16(p0, q0), _mm_subs_epu16(q0, p0));
  __m128i abs_p1q1 =
      _mm_or_si128(_mm_subs_epu16(p1, q1), _mm_subs_epu16(q1, p1));
  __m128i work;
  const __m128i t4 = _mm_set1_epi16(4);
  const __m128i t3 = _mm_set1_epi16(3);
  __m128i t80;
  __m128i tff80;
  __m128i tffe0;
  __m128i t1f;
  // equivalent to shifting 0x1f left by bitdepth - 8
  // and setting new bits to 1
  const __m128i t1 = _mm_set1_epi16(0x1);
  __m128i t7f;
  // equivalent to shifting 0x7f left by bitdepth - 8
  // and setting new bits to 1
  __m128i ps1, ps0, qs0, qs1;
  __m128i filt;
  __m128i work_a;
  __m128i filter1, filter2;

  if (bd == 8) {
    blimit_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero);
    limit_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero);
    thresh_v = _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero);
    t80 = _mm_set1_epi16(0x80);
    tff80 = _mm_set1_epi16((int16_t)0xff80);
    tffe0 = _mm_set1_epi16((int16_t)0xffe0);
    t1f = _mm_srli_epi16(_mm_set1_epi16(0x1fff), 8);
    t7f = _mm_srli_epi16(_mm_set1_epi16(0x7fff), 8);
  } else if (bd == 10) {
    blimit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero), 2);
    limit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero), 2);
    thresh_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero), 2);
    t80 = _mm_slli_epi16(_mm_set1_epi16(0x80), 2);
    tff80 = _mm_slli_epi16(_mm_set1_epi16((int16_t)0xff80), 2);
    tffe0 = _mm_slli_epi16(_mm_set1_epi16((int16_t)0xffe0), 2);
    t1f = _mm_srli_epi16(_mm_set1_epi16(0x1fff), 6);
    t7f = _mm_srli_epi16(_mm_set1_epi16(0x7fff), 6);
  } else {  // bd == 12
    blimit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)blimit), zero), 4);
    limit_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)limit), zero), 4);
    thresh_v = _mm_slli_epi16(
        _mm_unpacklo_epi8(_mm_load_si128((const __m128i *)thresh), zero), 4);
    t80 = _mm_slli_epi16(_mm_set1_epi16(0x80), 4);
    tff80 = _mm_slli_epi16(_mm_set1_epi16((int16_t)0xff80), 4);
    tffe0 = _mm_slli_epi16(_mm_set1_epi16((int16_t)0xffe0), 4);
    t1f = _mm_srli_epi16(_mm_set1_epi16(0x1fff), 4);
    t7f = _mm_srli_epi16(_mm_set1_epi16(0x7fff), 4);
  }

  ps1 = _mm_subs_epi16(_mm_loadu_si128((__m128i *)(s - 2 * pitch)), t80);
  ps0 = _mm_subs_epi16(_mm_loadu_si128((__m128i *)(s - 1 * pitch)), t80);
  qs0 = _mm_subs_epi16(_mm_loadu_si128((__m128i *)(s + 0 * pitch)), t80);
  qs1 = _mm_subs_epi16(_mm_loadu_si128((__m128i *)(s + 1 * pitch)), t80);

  // filter_mask and hev_mask
  flat = _mm_max_epi16(abs_p1p0, abs_q1q0);
  hev = _mm_subs_epu16(flat, thresh_v);
  hev = _mm_xor_si128(_mm_cmpeq_epi16(hev, zero), ffff);

  abs_p0q0 = _mm_adds_epu16(abs_p0q0, abs_p0q0);
  abs_p1q1 = _mm_srli_epi16(abs_p1q1, 1);
  mask = _mm_subs_epu16(_mm_adds_epu16(abs_p0q0, abs_p1q1), blimit_v);
  mask = _mm_xor_si128(_mm_cmpeq_epi16(mask, zero), ffff);
  // mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > blimit) * -1;
  // So taking maximums continues to work:
  mask = _mm_and_si128(mask, _mm_adds_epu16(limit_v, one));
  mask = _mm_max_epi16(flat, mask);
  // mask |= (abs(p1 - p0) > limit) * -1;
  // mask |= (abs(q1 - q0) > limit) * -1;
  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(p2, p1), _mm_subs_epu16(p1, p2)),
      _mm_or_si128(_mm_subs_epu16(p3, p2), _mm_subs_epu16(p2, p3)));
  mask = _mm_max_epi16(work, mask);
  work = _mm_max_epi16(
      _mm_or_si128(_mm_subs_epu16(q2, q1), _mm_subs_epu16(q1, q2)),
      _mm_or_si128(_mm_subs_epu16(q3, q2), _mm_subs_epu16(q2, q3)));
  mask = _mm_max_epi16(work, mask);
  mask = _mm_subs_epu16(mask, limit_v);
  mask = _mm_cmpeq_epi16(mask, zero);

  // filter4
  filt = signed_char_clamp_bd_sse2(_mm_subs_epi16(ps1, qs1), bd);
  filt = _mm_and_si128(filt, hev);
  work_a = _mm_subs_epi16(qs0, ps0);
  filt = _mm_adds_epi16(filt, work_a);
  filt = _mm_adds_epi16(filt, work_a);
  filt = signed_char_clamp_bd_sse2(_mm_adds_epi16(filt, work_a), bd);

  // (vpx_filter + 3 * (qs0 - ps0)) & mask
  filt = _mm_and_si128(filt, mask);

  filter1 = signed_char_clamp_bd_sse2(_mm_adds_epi16(filt, t4), bd);
  filter2 = signed_char_clamp_bd_sse2(_mm_adds_epi16(filt, t3), bd);

  // Filter1 >> 3
  work_a = _mm_cmpgt_epi16(zero, filter1);  // get the values that are <0
  filter1 = _mm_srli_epi16(filter1, 3);
  work_a = _mm_and_si128(work_a, tffe0);    // sign bits for the values < 0
  filter1 = _mm_and_si128(filter1, t1f);    // clamp the range
  filter1 = _mm_or_si128(filter1, work_a);  // reinsert the sign bits

  // Filter2 >> 3
  work_a = _mm_cmpgt_epi16(zero, filter2);
  filter2 = _mm_srli_epi16(filter2, 3);
  work_a = _mm_and_si128(work_a, tffe0);
  filter2 = _mm_and_si128(filter2, t1f);
  filter2 = _mm_or_si128(filter2, work_a);

  // filt >> 1
  filt = _mm_adds_epi16(filter1, t1);
  work_a = _mm_cmpgt_epi16(zero, filt);
  filt = _mm_srli_epi16(filt, 1);
  work_a = _mm_and_si128(work_a, tff80);
  filt = _mm_and_si128(filt, t7f);
  filt = _mm_or_si128(filt, work_a);

  filt = _mm_andnot_si128(hev, filt);

  q0 = _mm_adds_epi16(
      signed_char_clamp_bd_sse2(_mm_subs_epi16(qs0, filter1), bd), t80);
  q1 = _mm_adds_epi16(signed_char_clamp_bd_sse2(_mm_subs_epi16(qs1, filt), bd),
                      t80);
  p0 = _mm_adds_epi16(
      signed_char_clamp_bd_sse2(_mm_adds_epi16(ps0, filter2), bd), t80);
  p1 = _mm_adds_epi16(signed_char_clamp_bd_sse2(_mm_adds_epi16(ps1, filt), bd),
                      t80);

  _mm_storeu_si128((__m128i *)(s - 2 * pitch), p1);
  _mm_storeu_si128((__m128i *)(s - 1 * pitch), p0);
  _mm_storeu_si128((__m128i *)(s + 0 * pitch), q0);
  _mm_storeu_si128((__m128i *)(s + 1 * pitch), q1);
}

void vpx_highbd_lpf_horizontal_4_dual_sse2(
    uint16_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  vpx_highbd_lpf_horizontal_4_sse2(s, pitch, blimit0, limit0, thresh0, bd);
  vpx_highbd_lpf_horizontal_4_sse2(s + 8, pitch, blimit1, limit1, thresh1, bd);
}

static INLINE void highbd_transpose(uint16_t *src[], int in_p, uint16_t *dst[],
                                    int out_p, int num_8x8_to_transpose) {
  int idx8x8 = 0;
  __m128i p0, p1, p2, p3, p4, p5, p6, p7, x0, x1, x2, x3, x4, x5, x6, x7;
  do {
    uint16_t *in = src[idx8x8];
    uint16_t *out = dst[idx8x8];

    p0 =
        _mm_loadu_si128((__m128i *)(in + 0 * in_p));  // 00 01 02 03 04 05 06 07
    p1 =
        _mm_loadu_si128((__m128i *)(in + 1 * in_p));  // 10 11 12 13 14 15 16 17
    p2 =
        _mm_loadu_si128((__m128i *)(in + 2 * in_p));  // 20 21 22 23 24 25 26 27
    p3 =
        _mm_loadu_si128((__m128i *)(in + 3 * in_p));  // 30 31 32 33 34 35 36 37
    p4 =
        _mm_loadu_si128((__m128i *)(in + 4 * in_p));  // 40 41 42 43 44 45 46 47
    p5 =
        _mm_loadu_si128((__m128i *)(in + 5 * in_p));  // 50 51 52 53 54 55 56 57
    p6 =
        _mm_loadu_si128((__m128i *)(in + 6 * in_p));  // 60 61 62 63 64 65 66 67
    p7 =
        _mm_loadu_si128((__m128i *)(in + 7 * in_p));  // 70 71 72 73 74 75 76 77
    // 00 10 01 11 02 12 03 13
    x0 = _mm_unpacklo_epi16(p0, p1);
    // 20 30 21 31 22 32 23 33
    x1 = _mm_unpacklo_epi16(p2, p3);
    // 40 50 41 51 42 52 43 53
    x2 = _mm_unpacklo_epi16(p4, p5);
    // 60 70 61 71 62 72 63 73
    x3 = _mm_unpacklo_epi16(p6, p7);
    // 00 10 20 30 01 11 21 31
    x4 = _mm_unpacklo_epi32(x0, x1);
    // 40 50 60 70 41 51 61 71
    x5 = _mm_unpacklo_epi32(x2, x3);
    // 00 10 20 30 40 50 60 70
    x6 = _mm_unpacklo_epi64(x4, x5);
    // 01 11 21 31 41 51 61 71
    x7 = _mm_unpackhi_epi64(x4, x5);

    _mm_storeu_si128((__m128i *)(out + 0 * out_p), x6);
    // 00 10 20 30 40 50 60 70
    _mm_storeu_si128((__m128i *)(out + 1 * out_p), x7);
    // 01 11 21 31 41 51 61 71

    // 02 12 22 32 03 13 23 33
    x4 = _mm_unpackhi_epi32(x0, x1);
    // 42 52 62 72 43 53 63 73
    x5 = _mm_unpackhi_epi32(x2, x3);
    // 02 12 22 32 42 52 62 72
    x6 = _mm_unpacklo_epi64(x4, x5);
    // 03 13 23 33 43 53 63 73
    x7 = _mm_unpackhi_epi64(x4, x5);

    _mm_storeu_si128((__m128i *)(out + 2 * out_p), x6);
    // 02 12 22 32 42 52 62 72
    _mm_storeu_si128((__m128i *)(out + 3 * out_p), x7);
    // 03 13 23 33 43 53 63 73

    // 04 14 05 15 06 16 07 17
    x0 = _mm_unpackhi_epi16(p0, p1);
    // 24 34 25 35 26 36 27 37
    x1 = _mm_unpackhi_epi16(p2, p3);
    // 44 54 45 55 46 56 47 57
    x2 = _mm_unpackhi_epi16(p4, p5);
    // 64 74 65 75 66 76 67 77
    x3 = _mm_unpackhi_epi16(p6, p7);
    // 04 14 24 34 05 15 25 35
    x4 = _mm_unpacklo_epi32(x0, x1);
    // 44 54 64 74 45 55 65 75
    x5 = _mm_unpacklo_epi32(x2, x3);
    // 04 14 24 34 44 54 64 74
    x6 = _mm_unpacklo_epi64(x4, x5);
    // 05 15 25 35 45 55 65 75
    x7 = _mm_unpackhi_epi64(x4, x5);

    _mm_storeu_si128((__m128i *)(out + 4 * out_p), x6);
    // 04 14 24 34 44 54 64 74
    _mm_storeu_si128((__m128i *)(out + 5 * out_p), x7);
    // 05 15 25 35 45 55 65 75

    // 06 16 26 36 07 17 27 37
    x4 = _mm_unpackhi_epi32(x0, x1);
    // 46 56 66 76 47 57 67 77
    x5 = _mm_unpackhi_epi32(x2, x3);
    // 06 16 26 36 46 56 66 76
    x6 = _mm_unpacklo_epi64(x4, x5);
    // 07 17 27 37 47 57 67 77
    x7 = _mm_unpackhi_epi64(x4, x5);

    _mm_storeu_si128((__m128i *)(out + 6 * out_p), x6);
    // 06 16 26 36 46 56 66 76
    _mm_storeu_si128((__m128i *)(out + 7 * out_p), x7);
    // 07 17 27 37 47 57 67 77
  } while (++idx8x8 < num_8x8_to_transpose);
}

static INLINE void highbd_transpose8x16(uint16_t *in0, uint16_t *in1, int in_p,
                                        uint16_t *out, int out_p) {
  uint16_t *src0[1];
  uint16_t *src1[1];
  uint16_t *dest0[1];
  uint16_t *dest1[1];
  src0[0] = in0;
  src1[0] = in1;
  dest0[0] = out;
  dest1[0] = out + 8;
  highbd_transpose(src0, in_p, dest0, out_p, 1);
  highbd_transpose(src1, in_p, dest1, out_p, 1);
}

void vpx_highbd_lpf_vertical_4_sse2(uint16_t *s, int pitch,
                                    const uint8_t *blimit, const uint8_t *limit,
                                    const uint8_t *thresh, int bd) {
  DECLARE_ALIGNED(16, uint16_t, t_dst[8 * 8]);
  uint16_t *src[1];
  uint16_t *dst[1];

  // Transpose 8x8
  src[0] = s - 4;
  dst[0] = t_dst;

  highbd_transpose(src, pitch, dst, 8, 1);

  // Loop filtering
  vpx_highbd_lpf_horizontal_4_sse2(t_dst + 4 * 8, 8, blimit, limit, thresh, bd);

  src[0] = t_dst;
  dst[0] = s - 4;

  // Transpose back
  highbd_transpose(src, 8, dst, pitch, 1);
}

void vpx_highbd_lpf_vertical_4_dual_sse2(
    uint16_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  DECLARE_ALIGNED(16, uint16_t, t_dst[16 * 8]);
  uint16_t *src[2];
  uint16_t *dst[2];

  // Transpose 8x16
  highbd_transpose8x16(s - 4, s - 4 + pitch * 8, pitch, t_dst, 16);

  // Loop filtering
  vpx_highbd_lpf_horizontal_4_dual_sse2(t_dst + 4 * 16, 16, blimit0, limit0,
                                        thresh0, blimit1, limit1, thresh1, bd);
  src[0] = t_dst;
  src[1] = t_dst + 8;
  dst[0] = s - 4;
  dst[1] = s - 4 + pitch * 8;

  // Transpose back
  highbd_transpose(src, 16, dst, pitch, 2);
}

void vpx_highbd_lpf_vertical_8_sse2(uint16_t *s, int pitch,
                                    const uint8_t *blimit, const uint8_t *limit,
                                    const uint8_t *thresh, int bd) {
  DECLARE_ALIGNED(16, uint16_t, t_dst[8 * 8]);
  uint16_t *src[1];
  uint16_t *dst[1];

  // Transpose 8x8
  src[0] = s - 4;
  dst[0] = t_dst;

  highbd_transpose(src, pitch, dst, 8, 1);

  // Loop filtering
  vpx_highbd_lpf_horizontal_8_sse2(t_dst + 4 * 8, 8, blimit, limit, thresh, bd);

  src[0] = t_dst;
  dst[0] = s - 4;

  // Transpose back
  highbd_transpose(src, 8, dst, pitch, 1);
}

void vpx_highbd_lpf_vertical_8_dual_sse2(
    uint16_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1,
    const uint8_t *thresh1, int bd) {
  DECLARE_ALIGNED(16, uint16_t, t_dst[16 * 8]);
  uint16_t *src[2];
  uint16_t *dst[2];

  // Transpose 8x16
  highbd_transpose8x16(s - 4, s - 4 + pitch * 8, pitch, t_dst, 16);

  // Loop filtering
  vpx_highbd_lpf_horizontal_8_dual_sse2(t_dst + 4 * 16, 16, blimit0, limit0,
                                        thresh0, blimit1, limit1, thresh1, bd);
  src[0] = t_dst;
  src[1] = t_dst + 8;

  dst[0] = s - 4;
  dst[1] = s - 4 + pitch * 8;

  // Transpose back
  highbd_transpose(src, 16, dst, pitch, 2);
}

void vpx_highbd_lpf_vertical_16_sse2(uint16_t *s, int pitch,
                                     const uint8_t *blimit,
                                     const uint8_t *limit,
                                     const uint8_t *thresh, int bd) {
  DECLARE_ALIGNED(16, uint16_t, t_dst[8 * 16]);
  uint16_t *src[2];
  uint16_t *dst[2];

  src[0] = s - 8;
  src[1] = s;
  dst[0] = t_dst;
  dst[1] = t_dst + 8 * 8;

  // Transpose 16x8
  highbd_transpose(src, pitch, dst, 8, 2);

  // Loop filtering
  vpx_highbd_lpf_horizontal_16_sse2(t_dst + 8 * 8, 8, blimit, limit, thresh,
                                    bd);
  src[0] = t_dst;
  src[1] = t_dst + 8 * 8;
  dst[0] = s - 8;
  dst[1] = s;

  // Transpose back
  highbd_transpose(src, 8, dst, pitch, 2);
}

void vpx_highbd_lpf_vertical_16_dual_sse2(uint16_t *s, int pitch,
                                          const uint8_t *blimit,
                                          const uint8_t *limit,
                                          const uint8_t *thresh, int bd) {
  DECLARE_ALIGNED(16, uint16_t, t_dst[256]);

  //  Transpose 16x16
  highbd_transpose8x16(s - 8, s - 8 + 8 * pitch, pitch, t_dst, 16);
  highbd_transpose8x16(s, s + 8 * pitch, pitch, t_dst + 8 * 16, 16);

  //  Loop filtering
  vpx_highbd_lpf_horizontal_16_dual_sse2(t_dst + 8 * 16, 16, blimit, limit,
                                         thresh, bd);

  //  Transpose back
  highbd_transpose8x16(t_dst, t_dst + 8 * 16, 16, s - 8, pitch);
  highbd_transpose8x16(t_dst + 8, t_dst + 8 + 8 * 16, 16, s - 8 + 8 * pitch,
                       pitch);
}
