/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
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
#include "vpx_ports/emmintrin_compat.h"

static INLINE __m128i abs_diff(__m128i a, __m128i b) {
  return _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
}

// filter_mask and hev_mask
#define FILTER_HEV_MASK do {                                                   \
  /* (abs(q1 - q0), abs(p1 - p0) */                                            \
  __m128i flat = abs_diff(q1p1, q0p0);                                         \
  /* abs(p1 - q1), abs(p0 - q0) */                                             \
  const __m128i abs_p1q1p0q0 = abs_diff(p1p0, q1q0);                           \
  __m128i abs_p0q0, abs_p1q1, work;                                            \
                                                                               \
  /* const uint8_t hev = hev_mask(thresh, *op1, *op0, *oq0, *oq1); */          \
  hev = _mm_unpacklo_epi8(_mm_max_epu8(flat, _mm_srli_si128(flat, 8)), zero);  \
  hev = _mm_cmpgt_epi16(hev, thresh);                                          \
  hev = _mm_packs_epi16(hev, hev);                                             \
                                                                               \
  /* const int8_t mask = filter_mask(*limit, *blimit, */                       \
  /*                                 p3, p2, p1, p0, q0, q1, q2, q3); */       \
  abs_p0q0 = _mm_adds_epu8(abs_p1q1p0q0, abs_p1q1p0q0);  /* abs(p0 - q0) * 2 */\
  abs_p1q1 = _mm_unpackhi_epi8(abs_p1q1p0q0, abs_p1q1p0q0);  /* abs(p1 - q1) */\
  abs_p1q1 = _mm_srli_epi16(abs_p1q1, 9);                                      \
  abs_p1q1 = _mm_packs_epi16(abs_p1q1, abs_p1q1);  /* abs(p1 - q1) / 2 */      \
  /* abs(p0 - q0) * 2 + abs(p1 - q1) / 2 */                                    \
  mask = _mm_adds_epu8(abs_p0q0, abs_p1q1);                                    \
  /* abs(p3 - p2), abs(p2 - p1) */                                             \
  work = abs_diff(p3p2, p2p1);                                                 \
  flat = _mm_max_epu8(work, flat);                                             \
  /* abs(q3 - q2), abs(q2 - q1) */                                             \
  work = abs_diff(q3q2, q2q1);                                                 \
  flat = _mm_max_epu8(work, flat);                                             \
  flat = _mm_max_epu8(flat, _mm_srli_si128(flat, 8));                          \
  mask = _mm_unpacklo_epi64(mask, flat);                                       \
  mask = _mm_subs_epu8(mask, limit);                                           \
  mask = _mm_cmpeq_epi8(mask, zero);                                           \
  mask = _mm_and_si128(mask, _mm_srli_si128(mask, 8));                         \
} while (0)

#define FILTER4 do {                                                           \
  const __m128i t3t4 = _mm_set_epi8(3, 3, 3, 3, 3, 3, 3, 3,                    \
                                    4, 4, 4, 4, 4, 4, 4, 4);                   \
  const __m128i t80 = _mm_set1_epi8(0x80);                                     \
  __m128i filter, filter2filter1, work;                                        \
                                                                               \
  ps1ps0 = _mm_xor_si128(p1p0, t80);  /* ^ 0x80 */                             \
  qs1qs0 = _mm_xor_si128(q1q0, t80);                                           \
                                                                               \
  /* int8_t filter = signed_char_clamp(ps1 - qs1) & hev; */                    \
  work = _mm_subs_epi8(ps1ps0, qs1qs0);                                        \
  filter = _mm_and_si128(_mm_srli_si128(work, 8), hev);                        \
  /* filter = signed_char_clamp(filter + 3 * (qs0 - ps0)) & mask; */           \
  filter = _mm_subs_epi8(filter, work);                                        \
  filter = _mm_subs_epi8(filter, work);                                        \
  filter = _mm_subs_epi8(filter, work);  /* + 3 * (qs0 - ps0) */               \
  filter = _mm_and_si128(filter, mask);  /* & mask */                          \
  filter = _mm_unpacklo_epi64(filter, filter);                                 \
                                                                               \
  /* filter1 = signed_char_clamp(filter + 4) >> 3; */                          \
  /* filter2 = signed_char_clamp(filter + 3) >> 3; */                          \
  filter2filter1 = _mm_adds_epi8(filter, t3t4);  /* signed_char_clamp */       \
  filter = _mm_unpackhi_epi8(filter2filter1, filter2filter1);                  \
  filter2filter1 = _mm_unpacklo_epi8(filter2filter1, filter2filter1);          \
  filter2filter1 = _mm_srai_epi16(filter2filter1, 11);  /* >> 3 */             \
  filter = _mm_srai_epi16(filter, 11);  /* >> 3 */                             \
  filter2filter1 = _mm_packs_epi16(filter2filter1, filter);                    \
                                                                               \
  /* filter = ROUND_POWER_OF_TWO(filter1, 1) & ~hev; */                        \
  filter = _mm_subs_epi8(filter2filter1, ff);  /* + 1 */                       \
  filter = _mm_unpacklo_epi8(filter, filter);                                  \
  filter = _mm_srai_epi16(filter, 9);  /* round */                             \
  filter = _mm_packs_epi16(filter, filter);                                    \
  filter = _mm_andnot_si128(hev, filter);                                      \
                                                                               \
  hev = _mm_unpackhi_epi64(filter2filter1, filter);                            \
  filter2filter1 = _mm_unpacklo_epi64(filter2filter1, filter);                 \
                                                                               \
  /* signed_char_clamp(qs1 - filter), signed_char_clamp(qs0 - filter1) */      \
  qs1qs0 = _mm_subs_epi8(qs1qs0, filter2filter1);                              \
  /* signed_char_clamp(ps1 + filter), signed_char_clamp(ps0 + filter2) */      \
  ps1ps0 = _mm_adds_epi8(ps1ps0, hev);                                         \
  qs1qs0 = _mm_xor_si128(qs1qs0, t80);  /* ^ 0x80 */                           \
  ps1ps0 = _mm_xor_si128(ps1ps0, t80);  /* ^ 0x80 */                           \
} while (0)

void vpx_lpf_horizontal_4_sse2(uint8_t *s, int p /* pitch */,
                               const uint8_t *_blimit, const uint8_t *_limit,
                               const uint8_t *_thresh) {
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i limit =
      _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)_blimit),
                         _mm_loadl_epi64((const __m128i *)_limit));
  const __m128i thresh =
      _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)_thresh), zero);
  const __m128i ff = _mm_cmpeq_epi8(zero, zero);
  __m128i q1p1, q0p0, p3p2, p2p1, p1p0, q3q2, q2q1, q1q0, ps1ps0, qs1qs0;
  __m128i mask, hev;

  p3p2 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i *)(s - 3 * p)),
                            _mm_loadl_epi64((__m128i *)(s - 4 * p)));
  q1p1 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i *)(s - 2 * p)),
                            _mm_loadl_epi64((__m128i *)(s + 1 * p)));
  q0p0 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i *)(s - 1 * p)),
                            _mm_loadl_epi64((__m128i *)(s + 0 * p)));
  q3q2 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i *)(s + 2 * p)),
                            _mm_loadl_epi64((__m128i *)(s + 3 * p)));
  p1p0 = _mm_unpacklo_epi64(q0p0, q1p1);
  p2p1 = _mm_unpacklo_epi64(q1p1, p3p2);
  q1q0 = _mm_unpackhi_epi64(q0p0, q1p1);
  q2q1 = _mm_unpacklo_epi64(_mm_srli_si128(q1p1, 8), q3q2);

  FILTER_HEV_MASK;
  FILTER4;

  _mm_storeh_pi((__m64 *)(s - 2 * p), _mm_castsi128_ps(ps1ps0));  // *op1
  _mm_storel_epi64((__m128i *)(s - 1 * p), ps1ps0);  // *op0
  _mm_storel_epi64((__m128i *)(s + 0 * p), qs1qs0);  // *oq0
  _mm_storeh_pi((__m64 *)(s + 1 * p), _mm_castsi128_ps(qs1qs0));  // *oq1
}

void vpx_lpf_vertical_4_sse2(uint8_t *s, int p /* pitch */,
                             const uint8_t *_blimit, const uint8_t *_limit,
                             const uint8_t *_thresh) {
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i limit =
      _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)_blimit),
                         _mm_loadl_epi64((const __m128i *)_limit));
  const __m128i thresh =
      _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)_thresh), zero);
  const __m128i ff = _mm_cmpeq_epi8(zero, zero);
  __m128i x0, x1, x2, x3;
  __m128i q1p1, q0p0, p3p2, p2p1, p1p0, q3q2, q2q1, q1q0, ps1ps0, qs1qs0;
  __m128i mask, hev;

  // 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
  q1q0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(s + 0 * p - 4)),
                           _mm_loadl_epi64((__m128i *)(s + 1 * p - 4)));

  // 20 30 21 31 22 32 23 33 24 34 25 35 26 36 27 37
  x1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(s + 2 * p - 4)),
                         _mm_loadl_epi64((__m128i *)(s + 3 * p - 4)));

  // 40 50 41 51 42 52 43 53 44 54 45 55 46 56 47 57
  x2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(s + 4 * p - 4)),
                         _mm_loadl_epi64((__m128i *)(s + 5 * p - 4)));

  // 60 70 61 71 62 72 63 73 64 74 65 75 66 76 67 77
  x3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(s + 6 * p - 4)),
                         _mm_loadl_epi64((__m128i *)(s + 7 * p - 4)));

  // Transpose 8x8
  // 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
  p1p0 = _mm_unpacklo_epi16(q1q0, x1);
  // 40 50 60 70 41 51 61 71  42 52 62 72 43 53 63 73
  x0 = _mm_unpacklo_epi16(x2, x3);
  // 00 10 20 30 40 50 60 70  01 11 21 31 41 51 61 71
  p3p2 = _mm_unpacklo_epi32(p1p0, x0);
  // 02 12 22 32 42 52 62 72  03 13 23 33 43 53 63 73
  p1p0 = _mm_unpackhi_epi32(p1p0, x0);
  p3p2 = _mm_unpackhi_epi64(p3p2, _mm_slli_si128(p3p2, 8));  // swap lo and high
  p1p0 = _mm_unpackhi_epi64(p1p0, _mm_slli_si128(p1p0, 8));  // swap lo and high

  // 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
  q1q0 = _mm_unpackhi_epi16(q1q0, x1);
  // 44 54 64 74 45 55 65 75  46 56 66 76 47 57 67 77
  x2 = _mm_unpackhi_epi16(x2, x3);
  // 06 16 26 36 46 56 66 76  07 17 27 37 47 57 67 77
  q3q2 = _mm_unpackhi_epi32(q1q0, x2);
  // 04 14 24 34 44 54 64 74  05 15 25 35 45 55 65 75
  q1q0 = _mm_unpacklo_epi32(q1q0, x2);

  q0p0 = _mm_unpacklo_epi64(p1p0, q1q0);
  q1p1 = _mm_unpackhi_epi64(p1p0, q1q0);
  p1p0 = _mm_unpacklo_epi64(q0p0, q1p1);
  p2p1 = _mm_unpacklo_epi64(q1p1, p3p2);
  q2q1 = _mm_unpacklo_epi64(_mm_srli_si128(q1p1, 8), q3q2);

  FILTER_HEV_MASK;
  FILTER4;

  // Transpose 8x4 to 4x8
  // qs1qs0: 20 21 22 23 24 25 26 27  30 31 32 33 34 34 36 37
  // ps1ps0: 10 11 12 13 14 15 16 17  00 01 02 03 04 05 06 07
  // 00 01 02 03 04 05 06 07  10 11 12 13 14 15 16 17
  ps1ps0 = _mm_unpackhi_epi64(ps1ps0, _mm_slli_si128(ps1ps0, 8));
  // 10 30 11 31 12 32 13 33  14 34 15 35 16 36 17 37
  x0 = _mm_unpackhi_epi8(ps1ps0, qs1qs0);
  // 00 20 01 21 02 22 03 23  04 24 05 25 06 26 07 27
  ps1ps0 = _mm_unpacklo_epi8(ps1ps0, qs1qs0);
  // 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
  qs1qs0 = _mm_unpackhi_epi8(ps1ps0, x0);
  // 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
  ps1ps0 = _mm_unpacklo_epi8(ps1ps0, x0);

  *(int *)(s + 0 * p - 2) = _mm_cvtsi128_si32(ps1ps0);
  ps1ps0 = _mm_srli_si128(ps1ps0, 4);
  *(int *)(s + 1 * p - 2) = _mm_cvtsi128_si32(ps1ps0);
  ps1ps0 = _mm_srli_si128(ps1ps0, 4);
  *(int *)(s + 2 * p - 2) = _mm_cvtsi128_si32(ps1ps0);
  ps1ps0 = _mm_srli_si128(ps1ps0, 4);
  *(int *)(s + 3 * p - 2) = _mm_cvtsi128_si32(ps1ps0);

  *(int *)(s + 4 * p - 2) = _mm_cvtsi128_si32(qs1qs0);
  qs1qs0 = _mm_srli_si128(qs1qs0, 4);
  *(int *)(s + 5 * p - 2) = _mm_cvtsi128_si32(qs1qs0);
  qs1qs0 = _mm_srli_si128(qs1qs0, 4);
  *(int *)(s + 6 * p - 2) = _mm_cvtsi128_si32(qs1qs0);
  qs1qs0 = _mm_srli_si128(qs1qs0, 4);
  *(int *)(s + 7 * p - 2) = _mm_cvtsi128_si32(qs1qs0);
}

void vpx_lpf_horizontal_edge_8_sse2(unsigned char *s, int p,
                                    const unsigned char *_blimit,
                                    const unsigned char *_limit,
                                    const unsigned char *_thresh) {
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i one = _mm_set1_epi8(1);
  const __m128i blimit = _mm_load_si128((const __m128i *)_blimit);
  const __m128i limit = _mm_load_si128((const __m128i *)_limit);
  const __m128i thresh = _mm_load_si128((const __m128i *)_thresh);
  __m128i mask, hev, flat, flat2;
  __m128i q7p7, q6p6, q5p5, q4p4, q3p3, q2p2, q1p1, q0p0, p0q0, p1q1;
  __m128i abs_p1p0;

  q4p4 = _mm_loadl_epi64((__m128i *)(s - 5 * p));
  q4p4 = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(q4p4),
                                       (__m64 *)(s + 4 * p)));
  q3p3 = _mm_loadl_epi64((__m128i *)(s - 4 * p));
  q3p3 = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(q3p3),
                                       (__m64 *)(s + 3 * p)));
  q2p2 = _mm_loadl_epi64((__m128i *)(s - 3 * p));
  q2p2 = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(q2p2),
                                       (__m64 *)(s + 2 * p)));
  q1p1 = _mm_loadl_epi64((__m128i *)(s - 2 * p));
  q1p1 = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(q1p1),
                                       (__m64 *)(s + 1 * p)));
  p1q1 = _mm_shuffle_epi32(q1p1, 78);
  q0p0 = _mm_loadl_epi64((__m128i *)(s - 1 * p));
  q0p0 = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(q0p0),
                                       (__m64 *)(s - 0 * p)));
  p0q0 = _mm_shuffle_epi32(q0p0, 78);

  {
    __m128i abs_p1q1, abs_p0q0, abs_q1q0, fe, ff, work;
    abs_p1p0 = abs_diff(q1p1, q0p0);
    abs_q1q0 =  _mm_srli_si128(abs_p1p0, 8);
    fe = _mm_set1_epi8(0xfe);
    ff = _mm_cmpeq_epi8(abs_p1p0, abs_p1p0);
    abs_p0q0 = abs_diff(q0p0, p0q0);
    abs_p1q1 = abs_diff(q1p1, p1q1);
    flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
    hev = _mm_subs_epu8(flat, thresh);
    hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);

    abs_p0q0 =_mm_adds_epu8(abs_p0q0, abs_p0q0);
    abs_p1q1 = _mm_srli_epi16(_mm_and_si128(abs_p1q1, fe), 1);
    mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
    mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
    // mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > blimit) * -1;
    mask = _mm_max_epu8(abs_p1p0, mask);
    // mask |= (abs(p1 - p0) > limit) * -1;
    // mask |= (abs(q1 - q0) > limit) * -1;

    work = _mm_max_epu8(abs_diff(q2p2, q1p1),
                        abs_diff(q3p3, q2p2));
    mask = _mm_max_epu8(work, mask);
    mask = _mm_max_epu8(mask, _mm_srli_si128(mask, 8));
    mask = _mm_subs_epu8(mask, limit);
    mask = _mm_cmpeq_epi8(mask, zero);
  }

  // lp filter
  {
    const __m128i t4 = _mm_set1_epi8(4);
    const __m128i t3 = _mm_set1_epi8(3);
    const __m128i t80 = _mm_set1_epi8(0x80);
    const __m128i t1 = _mm_set1_epi16(0x1);
    __m128i qs1ps1 = _mm_xor_si128(q1p1, t80);
    __m128i qs0ps0 = _mm_xor_si128(q0p0, t80);
    __m128i qs0 = _mm_xor_si128(p0q0, t80);
    __m128i qs1 = _mm_xor_si128(p1q1, t80);
    __m128i filt;
    __m128i work_a;
    __m128i filter1, filter2;
    __m128i flat2_q6p6, flat2_q5p5, flat2_q4p4, flat2_q3p3, flat2_q2p2;
    __m128i flat2_q1p1, flat2_q0p0, flat_q2p2, flat_q1p1, flat_q0p0;

    filt = _mm_and_si128(_mm_subs_epi8(qs1ps1, qs1), hev);
    work_a = _mm_subs_epi8(qs0, qs0ps0);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    // (vpx_filter + 3 * (qs0 - ps0)) & mask
    filt = _mm_and_si128(filt, mask);

    filter1 = _mm_adds_epi8(filt, t4);
    filter2 = _mm_adds_epi8(filt, t3);

    filter1 = _mm_unpacklo_epi8(zero, filter1);
    filter1 = _mm_srai_epi16(filter1, 0xB);
    filter2 = _mm_unpacklo_epi8(zero, filter2);
    filter2 = _mm_srai_epi16(filter2, 0xB);

    // Filter1 >> 3
    filt = _mm_packs_epi16(filter2, _mm_subs_epi16(zero, filter1));
    qs0ps0 = _mm_xor_si128(_mm_adds_epi8(qs0ps0, filt), t80);

    // filt >> 1
    filt = _mm_adds_epi16(filter1, t1);
    filt = _mm_srai_epi16(filt, 1);
    filt = _mm_andnot_si128(_mm_srai_epi16(_mm_unpacklo_epi8(zero, hev), 0x8),
                            filt);
    filt = _mm_packs_epi16(filt, _mm_subs_epi16(zero, filt));
    qs1ps1 = _mm_xor_si128(_mm_adds_epi8(qs1ps1, filt), t80);
    // loopfilter done

    {
      __m128i work;
      flat = _mm_max_epu8(abs_diff(q2p2, q0p0), abs_diff(q3p3, q0p0));
      flat = _mm_max_epu8(abs_p1p0, flat);
      flat = _mm_max_epu8(flat, _mm_srli_si128(flat, 8));
      flat = _mm_subs_epu8(flat, one);
      flat = _mm_cmpeq_epi8(flat, zero);
      flat = _mm_and_si128(flat, mask);

      q5p5 = _mm_loadl_epi64((__m128i *)(s - 6 * p));
      q5p5 = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(q5p5),
                                           (__m64 *)(s + 5 * p)));

      q6p6 = _mm_loadl_epi64((__m128i *)(s - 7 * p));
      q6p6 = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(q6p6),
                                           (__m64 *)(s + 6 * p)));
      flat2 = _mm_max_epu8(abs_diff(q4p4, q0p0), abs_diff(q5p5, q0p0));

      q7p7 = _mm_loadl_epi64((__m128i *)(s - 8 * p));
      q7p7 = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(q7p7),
                                           (__m64 *)(s + 7 * p)));
      work = _mm_max_epu8(abs_diff(q6p6, q0p0), abs_diff(q7p7, q0p0));
      flat2 = _mm_max_epu8(work, flat2);
      flat2 = _mm_max_epu8(flat2, _mm_srli_si128(flat2, 8));
      flat2 = _mm_subs_epu8(flat2, one);
      flat2 = _mm_cmpeq_epi8(flat2, zero);
      flat2 = _mm_and_si128(flat2, flat);  // flat2 & flat & mask
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // flat and wide flat calculations
    {
      const __m128i eight = _mm_set1_epi16(8);
      const __m128i four = _mm_set1_epi16(4);
      __m128i p7_16, p6_16, p5_16, p4_16, p3_16, p2_16, p1_16, p0_16;
      __m128i q7_16, q6_16, q5_16, q4_16, q3_16, q2_16, q1_16, q0_16;
      __m128i pixelFilter_p, pixelFilter_q;
      __m128i pixetFilter_p2p1p0, pixetFilter_q2q1q0;
      __m128i sum_p7, sum_q7, sum_p3, sum_q3, res_p, res_q;

      p7_16 = _mm_unpacklo_epi8(q7p7, zero);;
      p6_16 = _mm_unpacklo_epi8(q6p6, zero);
      p5_16 = _mm_unpacklo_epi8(q5p5, zero);
      p4_16 = _mm_unpacklo_epi8(q4p4, zero);
      p3_16 = _mm_unpacklo_epi8(q3p3, zero);
      p2_16 = _mm_unpacklo_epi8(q2p2, zero);
      p1_16 = _mm_unpacklo_epi8(q1p1, zero);
      p0_16 = _mm_unpacklo_epi8(q0p0, zero);
      q0_16 = _mm_unpackhi_epi8(q0p0, zero);
      q1_16 = _mm_unpackhi_epi8(q1p1, zero);
      q2_16 = _mm_unpackhi_epi8(q2p2, zero);
      q3_16 = _mm_unpackhi_epi8(q3p3, zero);
      q4_16 = _mm_unpackhi_epi8(q4p4, zero);
      q5_16 = _mm_unpackhi_epi8(q5p5, zero);
      q6_16 = _mm_unpackhi_epi8(q6p6, zero);
      q7_16 = _mm_unpackhi_epi8(q7p7, zero);

      pixelFilter_p = _mm_add_epi16(_mm_add_epi16(p6_16, p5_16),
                                    _mm_add_epi16(p4_16, p3_16));
      pixelFilter_q = _mm_add_epi16(_mm_add_epi16(q6_16, q5_16),
                                    _mm_add_epi16(q4_16, q3_16));

      pixetFilter_p2p1p0 = _mm_add_epi16(p0_16, _mm_add_epi16(p2_16, p1_16));
      pixelFilter_p =  _mm_add_epi16(pixelFilter_p, pixetFilter_p2p1p0);

      pixetFilter_q2q1q0 = _mm_add_epi16(q0_16, _mm_add_epi16(q2_16, q1_16));
      pixelFilter_q =  _mm_add_epi16(pixelFilter_q, pixetFilter_q2q1q0);
      pixelFilter_p =  _mm_add_epi16(eight, _mm_add_epi16(pixelFilter_p,
                                                         pixelFilter_q));
      pixetFilter_p2p1p0 =   _mm_add_epi16(four,
                                           _mm_add_epi16(pixetFilter_p2p1p0,
                                                         pixetFilter_q2q1q0));
      res_p = _mm_srli_epi16(_mm_add_epi16(pixelFilter_p,
                                           _mm_add_epi16(p7_16, p0_16)), 4);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixelFilter_p,
                                           _mm_add_epi16(q7_16, q0_16)), 4);
      flat2_q0p0 = _mm_packus_epi16(res_p, res_q);
      res_p = _mm_srli_epi16(_mm_add_epi16(pixetFilter_p2p1p0,
                                           _mm_add_epi16(p3_16, p0_16)), 3);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixetFilter_p2p1p0,
                                           _mm_add_epi16(q3_16, q0_16)), 3);

      flat_q0p0 = _mm_packus_epi16(res_p, res_q);

      sum_p7 = _mm_add_epi16(p7_16, p7_16);
      sum_q7 = _mm_add_epi16(q7_16, q7_16);
      sum_p3 = _mm_add_epi16(p3_16, p3_16);
      sum_q3 = _mm_add_epi16(q3_16, q3_16);

      pixelFilter_q = _mm_sub_epi16(pixelFilter_p, p6_16);
      pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q6_16);
      res_p = _mm_srli_epi16(_mm_add_epi16(pixelFilter_p,
                             _mm_add_epi16(sum_p7, p1_16)), 4);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixelFilter_q,
                             _mm_add_epi16(sum_q7, q1_16)), 4);
      flat2_q1p1 = _mm_packus_epi16(res_p, res_q);

      pixetFilter_q2q1q0 = _mm_sub_epi16(pixetFilter_p2p1p0, p2_16);
      pixetFilter_p2p1p0 = _mm_sub_epi16(pixetFilter_p2p1p0, q2_16);
      res_p = _mm_srli_epi16(_mm_add_epi16(pixetFilter_p2p1p0,
                             _mm_add_epi16(sum_p3, p1_16)), 3);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixetFilter_q2q1q0,
                             _mm_add_epi16(sum_q3, q1_16)), 3);
      flat_q1p1 = _mm_packus_epi16(res_p, res_q);

      sum_p7 = _mm_add_epi16(sum_p7, p7_16);
      sum_q7 = _mm_add_epi16(sum_q7, q7_16);
      sum_p3 = _mm_add_epi16(sum_p3, p3_16);
      sum_q3 = _mm_add_epi16(sum_q3, q3_16);

      pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q5_16);
      pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p5_16);
      res_p = _mm_srli_epi16(_mm_add_epi16(pixelFilter_p,
                             _mm_add_epi16(sum_p7, p2_16)), 4);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixelFilter_q,
                             _mm_add_epi16(sum_q7, q2_16)), 4);
      flat2_q2p2 = _mm_packus_epi16(res_p, res_q);

      pixetFilter_p2p1p0 = _mm_sub_epi16(pixetFilter_p2p1p0, q1_16);
      pixetFilter_q2q1q0 = _mm_sub_epi16(pixetFilter_q2q1q0, p1_16);

      res_p = _mm_srli_epi16(_mm_add_epi16(pixetFilter_p2p1p0,
                                           _mm_add_epi16(sum_p3, p2_16)), 3);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixetFilter_q2q1q0,
                                           _mm_add_epi16(sum_q3, q2_16)), 3);
      flat_q2p2 = _mm_packus_epi16(res_p, res_q);

      sum_p7 = _mm_add_epi16(sum_p7, p7_16);
      sum_q7 = _mm_add_epi16(sum_q7, q7_16);
      pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q4_16);
      pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p4_16);
      res_p = _mm_srli_epi16(_mm_add_epi16(pixelFilter_p,
                             _mm_add_epi16(sum_p7, p3_16)), 4);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixelFilter_q,
                             _mm_add_epi16(sum_q7, q3_16)), 4);
      flat2_q3p3 = _mm_packus_epi16(res_p, res_q);

      sum_p7 = _mm_add_epi16(sum_p7, p7_16);
      sum_q7 = _mm_add_epi16(sum_q7, q7_16);
      pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q3_16);
      pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p3_16);
      res_p = _mm_srli_epi16(_mm_add_epi16(pixelFilter_p,
                             _mm_add_epi16(sum_p7, p4_16)), 4);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixelFilter_q,
                             _mm_add_epi16(sum_q7, q4_16)), 4);
      flat2_q4p4 = _mm_packus_epi16(res_p, res_q);

      sum_p7 = _mm_add_epi16(sum_p7, p7_16);
      sum_q7 = _mm_add_epi16(sum_q7, q7_16);
      pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q2_16);
      pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p2_16);
      res_p = _mm_srli_epi16(_mm_add_epi16(pixelFilter_p,
                             _mm_add_epi16(sum_p7, p5_16)), 4);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixelFilter_q,
                             _mm_add_epi16(sum_q7, q5_16)), 4);
      flat2_q5p5 = _mm_packus_epi16(res_p, res_q);

      sum_p7 = _mm_add_epi16(sum_p7, p7_16);
      sum_q7 = _mm_add_epi16(sum_q7, q7_16);
      pixelFilter_p = _mm_sub_epi16(pixelFilter_p, q1_16);
      pixelFilter_q = _mm_sub_epi16(pixelFilter_q, p1_16);
      res_p = _mm_srli_epi16(_mm_add_epi16(pixelFilter_p,
                             _mm_add_epi16(sum_p7, p6_16)), 4);
      res_q = _mm_srli_epi16(_mm_add_epi16(pixelFilter_q,
                             _mm_add_epi16(sum_q7, q6_16)), 4);
      flat2_q6p6 = _mm_packus_epi16(res_p, res_q);
    }
    // wide flat
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    flat = _mm_shuffle_epi32(flat, 68);
    flat2 = _mm_shuffle_epi32(flat2, 68);

    q2p2 = _mm_andnot_si128(flat, q2p2);
    flat_q2p2 = _mm_and_si128(flat, flat_q2p2);
    q2p2 = _mm_or_si128(q2p2, flat_q2p2);

    qs1ps1 = _mm_andnot_si128(flat, qs1ps1);
    flat_q1p1 = _mm_and_si128(flat, flat_q1p1);
    q1p1 = _mm_or_si128(qs1ps1, flat_q1p1);

    qs0ps0 = _mm_andnot_si128(flat, qs0ps0);
    flat_q0p0 = _mm_and_si128(flat, flat_q0p0);
    q0p0 = _mm_or_si128(qs0ps0, flat_q0p0);

    q6p6 = _mm_andnot_si128(flat2, q6p6);
    flat2_q6p6 = _mm_and_si128(flat2, flat2_q6p6);
    q6p6 = _mm_or_si128(q6p6, flat2_q6p6);
    _mm_storel_epi64((__m128i *)(s - 7 * p), q6p6);
    _mm_storeh_pi((__m64 *)(s + 6 * p), _mm_castsi128_ps(q6p6));

    q5p5 = _mm_andnot_si128(flat2, q5p5);
    flat2_q5p5 = _mm_and_si128(flat2, flat2_q5p5);
    q5p5 = _mm_or_si128(q5p5, flat2_q5p5);
    _mm_storel_epi64((__m128i *)(s - 6 * p), q5p5);
    _mm_storeh_pi((__m64 *)(s + 5 * p), _mm_castsi128_ps(q5p5));

    q4p4 = _mm_andnot_si128(flat2, q4p4);
    flat2_q4p4 = _mm_and_si128(flat2, flat2_q4p4);
    q4p4 = _mm_or_si128(q4p4, flat2_q4p4);
    _mm_storel_epi64((__m128i *)(s - 5 * p), q4p4);
    _mm_storeh_pi((__m64 *)(s + 4 * p), _mm_castsi128_ps(q4p4));

    q3p3 = _mm_andnot_si128(flat2, q3p3);
    flat2_q3p3 = _mm_and_si128(flat2, flat2_q3p3);
    q3p3 = _mm_or_si128(q3p3, flat2_q3p3);
    _mm_storel_epi64((__m128i *)(s - 4 * p), q3p3);
    _mm_storeh_pi((__m64 *)(s + 3 * p), _mm_castsi128_ps(q3p3));

    q2p2 = _mm_andnot_si128(flat2, q2p2);
    flat2_q2p2 = _mm_and_si128(flat2, flat2_q2p2);
    q2p2 = _mm_or_si128(q2p2, flat2_q2p2);
    _mm_storel_epi64((__m128i *)(s - 3 * p), q2p2);
    _mm_storeh_pi((__m64 *)(s + 2 * p), _mm_castsi128_ps(q2p2));

    q1p1 = _mm_andnot_si128(flat2, q1p1);
    flat2_q1p1 = _mm_and_si128(flat2, flat2_q1p1);
    q1p1 = _mm_or_si128(q1p1, flat2_q1p1);
    _mm_storel_epi64((__m128i *)(s - 2 * p), q1p1);
    _mm_storeh_pi((__m64 *)(s + 1 * p), _mm_castsi128_ps(q1p1));

    q0p0 = _mm_andnot_si128(flat2, q0p0);
    flat2_q0p0 = _mm_and_si128(flat2, flat2_q0p0);
    q0p0 = _mm_or_si128(q0p0, flat2_q0p0);
    _mm_storel_epi64((__m128i *)(s - 1 * p), q0p0);
    _mm_storeh_pi((__m64 *)(s - 0 * p),  _mm_castsi128_ps(q0p0));
  }
}

static INLINE __m128i filter_add2_sub2(const __m128i *const total,
                                       const __m128i *const a1,
                                       const __m128i *const a2,
                                       const __m128i *const s1,
                                       const __m128i *const s2) {
  __m128i x = _mm_add_epi16(*a1, *total);
  x = _mm_add_epi16(_mm_sub_epi16(x, _mm_add_epi16(*s1, *s2)), *a2);
  return x;
}

static INLINE __m128i filter8_mask(const __m128i *const flat,
                                   const __m128i *const other_filt,
                                   const __m128i *const f8_lo,
                                   const __m128i *const f8_hi) {
  const __m128i f8 = _mm_packus_epi16(_mm_srli_epi16(*f8_lo, 3),
                                      _mm_srli_epi16(*f8_hi, 3));
  const __m128i result = _mm_and_si128(*flat, f8);
  return _mm_or_si128(_mm_andnot_si128(*flat, *other_filt), result);
}

static INLINE __m128i filter16_mask(const __m128i *const flat,
                                    const __m128i *const other_filt,
                                    const __m128i *const f_lo,
                                    const __m128i *const f_hi) {
  const __m128i f = _mm_packus_epi16(_mm_srli_epi16(*f_lo, 4),
                                     _mm_srli_epi16(*f_hi, 4));
  const __m128i result = _mm_and_si128(*flat, f);
  return _mm_or_si128(_mm_andnot_si128(*flat, *other_filt), result);
}

void vpx_lpf_horizontal_edge_16_sse2(unsigned char *s, int p,
                                     const unsigned char *_blimit,
                                     const unsigned char *_limit,
                                     const unsigned char *_thresh) {
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i one = _mm_set1_epi8(1);
  const __m128i blimit = _mm_load_si128((const __m128i *)_blimit);
  const __m128i limit = _mm_load_si128((const __m128i *)_limit);
  const __m128i thresh = _mm_load_si128((const __m128i *)_thresh);
  __m128i mask, hev, flat, flat2;
  __m128i p7, p6, p5;
  __m128i p4, p3, p2, p1, p0, q0, q1, q2, q3, q4;
  __m128i q5, q6, q7;

  __m128i op2, op1, op0, oq0, oq1, oq2;

  __m128i max_abs_p1p0q1q0;

  p7 = _mm_loadu_si128((__m128i *)(s - 8 * p));
  p6 = _mm_loadu_si128((__m128i *)(s - 7 * p));
  p5 = _mm_loadu_si128((__m128i *)(s - 6 * p));
  p4 = _mm_loadu_si128((__m128i *)(s - 5 * p));
  p3 = _mm_loadu_si128((__m128i *)(s - 4 * p));
  p2 = _mm_loadu_si128((__m128i *)(s - 3 * p));
  p1 = _mm_loadu_si128((__m128i *)(s - 2 * p));
  p0 = _mm_loadu_si128((__m128i *)(s - 1 * p));
  q0 = _mm_loadu_si128((__m128i *)(s - 0 * p));
  q1 = _mm_loadu_si128((__m128i *)(s + 1 * p));
  q2 = _mm_loadu_si128((__m128i *)(s + 2 * p));
  q3 = _mm_loadu_si128((__m128i *)(s + 3 * p));
  q4 = _mm_loadu_si128((__m128i *)(s + 4 * p));
  q5 = _mm_loadu_si128((__m128i *)(s + 5 * p));
  q6 = _mm_loadu_si128((__m128i *)(s + 6 * p));
  q7 = _mm_loadu_si128((__m128i *)(s + 7 * p));

  {
    const __m128i abs_p1p0 = abs_diff(p1, p0);
    const __m128i abs_q1q0 = abs_diff(q1, q0);
    const __m128i fe = _mm_set1_epi8(0xfe);
    const __m128i ff = _mm_cmpeq_epi8(zero, zero);
    __m128i abs_p0q0 = abs_diff(p0, q0);
    __m128i abs_p1q1 = abs_diff(p1, q1);
    __m128i work;
    max_abs_p1p0q1q0 = _mm_max_epu8(abs_p1p0, abs_q1q0);

    abs_p0q0 =_mm_adds_epu8(abs_p0q0, abs_p0q0);
    abs_p1q1 = _mm_srli_epi16(_mm_and_si128(abs_p1q1, fe), 1);
    mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
    mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
    // mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > blimit) * -1;
    mask = _mm_max_epu8(max_abs_p1p0q1q0, mask);
    // mask |= (abs(p1 - p0) > limit) * -1;
    // mask |= (abs(q1 - q0) > limit) * -1;
    work = _mm_max_epu8(abs_diff(p2, p1), abs_diff(p3, p2));
    mask = _mm_max_epu8(work, mask);
    work = _mm_max_epu8(abs_diff(q2, q1), abs_diff(q3, q2));
    mask = _mm_max_epu8(work, mask);
    mask = _mm_subs_epu8(mask, limit);
    mask = _mm_cmpeq_epi8(mask, zero);
  }

  {
    __m128i work;
    work = _mm_max_epu8(abs_diff(p2, p0), abs_diff(q2, q0));
    flat = _mm_max_epu8(work, max_abs_p1p0q1q0);
    work = _mm_max_epu8(abs_diff(p3, p0), abs_diff(q3, q0));
    flat = _mm_max_epu8(work, flat);
    work = _mm_max_epu8(abs_diff(p4, p0), abs_diff(q4, q0));
    flat = _mm_subs_epu8(flat, one);
    flat = _mm_cmpeq_epi8(flat, zero);
    flat = _mm_and_si128(flat, mask);
    flat2 = _mm_max_epu8(abs_diff(p5, p0), abs_diff(q5, q0));
    flat2 = _mm_max_epu8(work, flat2);
    work = _mm_max_epu8(abs_diff(p6, p0), abs_diff(q6, q0));
    flat2 = _mm_max_epu8(work, flat2);
    work = _mm_max_epu8(abs_diff(p7, p0), abs_diff(q7, q0));
    flat2 = _mm_max_epu8(work, flat2);
    flat2 = _mm_subs_epu8(flat2, one);
    flat2 = _mm_cmpeq_epi8(flat2, zero);
    flat2 = _mm_and_si128(flat2, flat);  // flat2 & flat & mask
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // filter4
  {
    const __m128i t4 = _mm_set1_epi8(4);
    const __m128i t3 = _mm_set1_epi8(3);
    const __m128i t80 = _mm_set1_epi8(0x80);
    const __m128i te0 = _mm_set1_epi8(0xe0);
    const __m128i t1f = _mm_set1_epi8(0x1f);
    const __m128i t1 = _mm_set1_epi8(0x1);
    const __m128i t7f = _mm_set1_epi8(0x7f);
    const __m128i ff = _mm_cmpeq_epi8(t4, t4);

    __m128i filt;
    __m128i work_a;
    __m128i filter1, filter2;

    op1 = _mm_xor_si128(p1, t80);
    op0 = _mm_xor_si128(p0, t80);
    oq0 = _mm_xor_si128(q0, t80);
    oq1 = _mm_xor_si128(q1, t80);

    hev = _mm_subs_epu8(max_abs_p1p0q1q0, thresh);
    hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);
    filt = _mm_and_si128(_mm_subs_epi8(op1, oq1), hev);

    work_a = _mm_subs_epi8(oq0, op0);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    // (vpx_filter + 3 * (qs0 - ps0)) & mask
    filt = _mm_and_si128(filt, mask);
    filter1 = _mm_adds_epi8(filt, t4);
    filter2 = _mm_adds_epi8(filt, t3);

    // Filter1 >> 3
    work_a = _mm_cmpgt_epi8(zero, filter1);
    filter1 = _mm_srli_epi16(filter1, 3);
    work_a = _mm_and_si128(work_a, te0);
    filter1 = _mm_and_si128(filter1, t1f);
    filter1 = _mm_or_si128(filter1, work_a);
    oq0 = _mm_xor_si128(_mm_subs_epi8(oq0, filter1), t80);

    // Filter2 >> 3
    work_a = _mm_cmpgt_epi8(zero, filter2);
    filter2 = _mm_srli_epi16(filter2, 3);
    work_a = _mm_and_si128(work_a, te0);
    filter2 = _mm_and_si128(filter2, t1f);
    filter2 = _mm_or_si128(filter2, work_a);
    op0 = _mm_xor_si128(_mm_adds_epi8(op0, filter2), t80);

    // filt >> 1
    filt = _mm_adds_epi8(filter1, t1);
    work_a = _mm_cmpgt_epi8(zero, filt);
    filt = _mm_srli_epi16(filt, 1);
    work_a = _mm_and_si128(work_a, t80);
    filt = _mm_and_si128(filt, t7f);
    filt = _mm_or_si128(filt, work_a);
    filt = _mm_andnot_si128(hev, filt);
    op1 = _mm_xor_si128(_mm_adds_epi8(op1, filt), t80);
    oq1 = _mm_xor_si128(_mm_subs_epi8(oq1, filt), t80);
    // loopfilter done

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // filter8
    {
      const __m128i four = _mm_set1_epi16(4);
      const __m128i p3_lo = _mm_unpacklo_epi8(p3, zero);
      const __m128i p2_lo = _mm_unpacklo_epi8(p2, zero);
      const __m128i p1_lo = _mm_unpacklo_epi8(p1, zero);
      const __m128i p0_lo = _mm_unpacklo_epi8(p0, zero);
      const __m128i q0_lo = _mm_unpacklo_epi8(q0, zero);
      const __m128i q1_lo = _mm_unpacklo_epi8(q1, zero);
      const __m128i q2_lo = _mm_unpacklo_epi8(q2, zero);
      const __m128i q3_lo = _mm_unpacklo_epi8(q3, zero);

      const __m128i p3_hi = _mm_unpackhi_epi8(p3, zero);
      const __m128i p2_hi = _mm_unpackhi_epi8(p2, zero);
      const __m128i p1_hi = _mm_unpackhi_epi8(p1, zero);
      const __m128i p0_hi = _mm_unpackhi_epi8(p0, zero);
      const __m128i q0_hi = _mm_unpackhi_epi8(q0, zero);
      const __m128i q1_hi = _mm_unpackhi_epi8(q1, zero);
      const __m128i q2_hi = _mm_unpackhi_epi8(q2, zero);
      const __m128i q3_hi = _mm_unpackhi_epi8(q3, zero);
      __m128i f8_lo, f8_hi;

      f8_lo = _mm_add_epi16(_mm_add_epi16(p3_lo, four),
                            _mm_add_epi16(p3_lo, p2_lo));
      f8_lo = _mm_add_epi16(_mm_add_epi16(p3_lo, f8_lo),
                            _mm_add_epi16(p2_lo, p1_lo));
      f8_lo = _mm_add_epi16(_mm_add_epi16(p0_lo, q0_lo), f8_lo);

      f8_hi = _mm_add_epi16(_mm_add_epi16(p3_hi, four),
                            _mm_add_epi16(p3_hi, p2_hi));
      f8_hi = _mm_add_epi16(_mm_add_epi16(p3_hi, f8_hi),
                            _mm_add_epi16(p2_hi, p1_hi));
      f8_hi = _mm_add_epi16(_mm_add_epi16(p0_hi, q0_hi), f8_hi);

      op2 = filter8_mask(&flat, &p2, &f8_lo, &f8_hi);

      f8_lo = filter_add2_sub2(&f8_lo, &q1_lo, &p1_lo, &p2_lo, &p3_lo);
      f8_hi = filter_add2_sub2(&f8_hi, &q1_hi, &p1_hi, &p2_hi, &p3_hi);
      op1 = filter8_mask(&flat, &op1, &f8_lo, &f8_hi);

      f8_lo = filter_add2_sub2(&f8_lo, &q2_lo, &p0_lo, &p1_lo, &p3_lo);
      f8_hi = filter_add2_sub2(&f8_hi, &q2_hi, &p0_hi, &p1_hi, &p3_hi);
      op0 = filter8_mask(&flat, &op0, &f8_lo, &f8_hi);

      f8_lo = filter_add2_sub2(&f8_lo, &q3_lo, &q0_lo, &p0_lo, &p3_lo);
      f8_hi = filter_add2_sub2(&f8_hi, &q3_hi, &q0_hi, &p0_hi, &p3_hi);
      oq0 = filter8_mask(&flat, &oq0, &f8_lo, &f8_hi);

      f8_lo = filter_add2_sub2(&f8_lo, &q3_lo, &q1_lo, &q0_lo, &p2_lo);
      f8_hi = filter_add2_sub2(&f8_hi, &q3_hi, &q1_hi, &q0_hi, &p2_hi);
      oq1 = filter8_mask(&flat, &oq1, &f8_lo, &f8_hi);

      f8_lo = filter_add2_sub2(&f8_lo, &q3_lo, &q2_lo, &q1_lo, &p1_lo);
      f8_hi = filter_add2_sub2(&f8_hi, &q3_hi, &q2_hi, &q1_hi, &p1_hi);
      oq2 = filter8_mask(&flat, &q2, &f8_lo, &f8_hi);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // wide flat calculations
    {
      const __m128i eight = _mm_set1_epi16(8);
      const __m128i p7_lo = _mm_unpacklo_epi8(p7, zero);
      const __m128i p6_lo = _mm_unpacklo_epi8(p6, zero);
      const __m128i p5_lo = _mm_unpacklo_epi8(p5, zero);
      const __m128i p4_lo = _mm_unpacklo_epi8(p4, zero);
      const __m128i p3_lo = _mm_unpacklo_epi8(p3, zero);
      const __m128i p2_lo = _mm_unpacklo_epi8(p2, zero);
      const __m128i p1_lo = _mm_unpacklo_epi8(p1, zero);
      const __m128i p0_lo = _mm_unpacklo_epi8(p0, zero);
      const __m128i q0_lo = _mm_unpacklo_epi8(q0, zero);
      const __m128i q1_lo = _mm_unpacklo_epi8(q1, zero);
      const __m128i q2_lo = _mm_unpacklo_epi8(q2, zero);
      const __m128i q3_lo = _mm_unpacklo_epi8(q3, zero);
      const __m128i q4_lo = _mm_unpacklo_epi8(q4, zero);
      const __m128i q5_lo = _mm_unpacklo_epi8(q5, zero);
      const __m128i q6_lo = _mm_unpacklo_epi8(q6, zero);
      const __m128i q7_lo = _mm_unpacklo_epi8(q7, zero);

      const __m128i p7_hi = _mm_unpackhi_epi8(p7, zero);
      const __m128i p6_hi = _mm_unpackhi_epi8(p6, zero);
      const __m128i p5_hi = _mm_unpackhi_epi8(p5, zero);
      const __m128i p4_hi = _mm_unpackhi_epi8(p4, zero);
      const __m128i p3_hi = _mm_unpackhi_epi8(p3, zero);
      const __m128i p2_hi = _mm_unpackhi_epi8(p2, zero);
      const __m128i p1_hi = _mm_unpackhi_epi8(p1, zero);
      const __m128i p0_hi = _mm_unpackhi_epi8(p0, zero);
      const __m128i q0_hi = _mm_unpackhi_epi8(q0, zero);
      const __m128i q1_hi = _mm_unpackhi_epi8(q1, zero);
      const __m128i q2_hi = _mm_unpackhi_epi8(q2, zero);
      const __m128i q3_hi = _mm_unpackhi_epi8(q3, zero);
      const __m128i q4_hi = _mm_unpackhi_epi8(q4, zero);
      const __m128i q5_hi = _mm_unpackhi_epi8(q5, zero);
      const __m128i q6_hi = _mm_unpackhi_epi8(q6, zero);
      const __m128i q7_hi = _mm_unpackhi_epi8(q7, zero);

      __m128i f_lo;
      __m128i f_hi;

      f_lo = _mm_sub_epi16(_mm_slli_epi16(p7_lo, 3), p7_lo);  // p7 * 7
      f_lo = _mm_add_epi16(_mm_slli_epi16(p6_lo, 1),
                           _mm_add_epi16(p4_lo, f_lo));
      f_lo = _mm_add_epi16(_mm_add_epi16(p3_lo, f_lo),
                           _mm_add_epi16(p2_lo, p1_lo));
      f_lo = _mm_add_epi16(_mm_add_epi16(p0_lo, q0_lo), f_lo);
      f_lo = _mm_add_epi16(_mm_add_epi16(p5_lo, eight), f_lo);

      f_hi = _mm_sub_epi16(_mm_slli_epi16(p7_hi, 3), p7_hi);  // p7 * 7
      f_hi = _mm_add_epi16(_mm_slli_epi16(p6_hi, 1),
                           _mm_add_epi16(p4_hi, f_hi));
      f_hi = _mm_add_epi16(_mm_add_epi16(p3_hi, f_hi),
                           _mm_add_epi16(p2_hi, p1_hi));
      f_hi = _mm_add_epi16(_mm_add_epi16(p0_hi, q0_hi), f_hi);
      f_hi = _mm_add_epi16(_mm_add_epi16(p5_hi, eight), f_hi);

      p6 = filter16_mask(&flat2, &p6, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s - 7 * p), p6);

      f_lo = filter_add2_sub2(&f_lo, &q1_lo, &p5_lo, &p6_lo, &p7_lo);
      f_hi = filter_add2_sub2(&f_hi, &q1_hi, &p5_hi, &p6_hi, &p7_hi);
      p5 = filter16_mask(&flat2, &p5, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s - 6 * p), p5);

      f_lo = filter_add2_sub2(&f_lo, &q2_lo, &p4_lo, &p5_lo, &p7_lo);
      f_hi = filter_add2_sub2(&f_hi, &q2_hi, &p4_hi, &p5_hi, &p7_hi);
      p4 = filter16_mask(&flat2, &p4, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s - 5 * p), p4);

      f_lo = filter_add2_sub2(&f_lo, &q3_lo, &p3_lo, &p4_lo, &p7_lo);
      f_hi = filter_add2_sub2(&f_hi, &q3_hi, &p3_hi, &p4_hi, &p7_hi);
      p3 = filter16_mask(&flat2, &p3, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s - 4 * p), p3);

      f_lo = filter_add2_sub2(&f_lo, &q4_lo, &p2_lo, &p3_lo, &p7_lo);
      f_hi = filter_add2_sub2(&f_hi, &q4_hi, &p2_hi, &p3_hi, &p7_hi);
      op2 = filter16_mask(&flat2, &op2, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s - 3 * p), op2);

      f_lo = filter_add2_sub2(&f_lo, &q5_lo, &p1_lo, &p2_lo, &p7_lo);
      f_hi = filter_add2_sub2(&f_hi, &q5_hi, &p1_hi, &p2_hi, &p7_hi);
      op1 = filter16_mask(&flat2, &op1, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s - 2 * p), op1);

      f_lo = filter_add2_sub2(&f_lo, &q6_lo, &p0_lo, &p1_lo, &p7_lo);
      f_hi = filter_add2_sub2(&f_hi, &q6_hi, &p0_hi, &p1_hi, &p7_hi);
      op0 = filter16_mask(&flat2, &op0, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s - 1 * p), op0);

      f_lo = filter_add2_sub2(&f_lo, &q7_lo, &q0_lo, &p0_lo, &p7_lo);
      f_hi = filter_add2_sub2(&f_hi, &q7_hi, &q0_hi, &p0_hi, &p7_hi);
      oq0 = filter16_mask(&flat2, &oq0, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s - 0 * p), oq0);

      f_lo = filter_add2_sub2(&f_lo, &q7_lo, &q1_lo, &p6_lo, &q0_lo);
      f_hi = filter_add2_sub2(&f_hi, &q7_hi, &q1_hi, &p6_hi, &q0_hi);
      oq1 = filter16_mask(&flat2, &oq1, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s + 1 * p), oq1);

      f_lo = filter_add2_sub2(&f_lo, &q7_lo, &q2_lo, &p5_lo, &q1_lo);
      f_hi = filter_add2_sub2(&f_hi, &q7_hi, &q2_hi, &p5_hi, &q1_hi);
      oq2 = filter16_mask(&flat2, &oq2, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s + 2 * p), oq2);

      f_lo = filter_add2_sub2(&f_lo, &q7_lo, &q3_lo, &p4_lo, &q2_lo);
      f_hi = filter_add2_sub2(&f_hi, &q7_hi, &q3_hi, &p4_hi, &q2_hi);
      q3 = filter16_mask(&flat2, &q3, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s + 3 * p), q3);

      f_lo = filter_add2_sub2(&f_lo, &q7_lo, &q4_lo, &p3_lo, &q3_lo);
      f_hi = filter_add2_sub2(&f_hi, &q7_hi, &q4_hi, &p3_hi, &q3_hi);
      q4 = filter16_mask(&flat2, &q4, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s + 4 * p), q4);

      f_lo = filter_add2_sub2(&f_lo, &q7_lo, &q5_lo, &p2_lo, &q4_lo);
      f_hi = filter_add2_sub2(&f_hi, &q7_hi, &q5_hi, &p2_hi, &q4_hi);
      q5 = filter16_mask(&flat2, &q5, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s + 5 * p), q5);

      f_lo = filter_add2_sub2(&f_lo, &q7_lo, &q6_lo, &p1_lo, &q5_lo);
      f_hi = filter_add2_sub2(&f_hi, &q7_hi, &q6_hi, &p1_hi, &q5_hi);
      q6 = filter16_mask(&flat2, &q6, &f_lo, &f_hi);
      _mm_storeu_si128((__m128i *)(s + 6 * p), q6);
    }
    // wide flat
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  }
}

void vpx_lpf_horizontal_8_sse2(unsigned char *s, int p,
                               const unsigned char *_blimit,
                               const unsigned char *_limit,
                               const unsigned char *_thresh) {
  DECLARE_ALIGNED(16, unsigned char, flat_op2[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_op1[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_op0[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_oq2[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_oq1[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_oq0[16]);
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i blimit = _mm_load_si128((const __m128i *)_blimit);
  const __m128i limit = _mm_load_si128((const __m128i *)_limit);
  const __m128i thresh = _mm_load_si128((const __m128i *)_thresh);
  __m128i mask, hev, flat;
  __m128i p3, p2, p1, p0, q0, q1, q2, q3;
  __m128i q3p3, q2p2, q1p1, q0p0, p1q1, p0q0;

  q3p3 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i *)(s - 4 * p)),
                            _mm_loadl_epi64((__m128i *)(s + 3 * p)));
  q2p2 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i *)(s - 3 * p)),
                            _mm_loadl_epi64((__m128i *)(s + 2 * p)));
  q1p1 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i *)(s - 2 * p)),
                            _mm_loadl_epi64((__m128i *)(s + 1 * p)));
  q0p0 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i *)(s - 1 * p)),
                            _mm_loadl_epi64((__m128i *)(s - 0 * p)));
  p1q1 = _mm_shuffle_epi32(q1p1, 78);
  p0q0 = _mm_shuffle_epi32(q0p0, 78);

  {
    // filter_mask and hev_mask
    const __m128i one = _mm_set1_epi8(1);
    const __m128i fe = _mm_set1_epi8(0xfe);
    const __m128i ff = _mm_cmpeq_epi8(fe, fe);
    __m128i abs_p1q1, abs_p0q0, abs_q1q0, abs_p1p0, work;
    abs_p1p0 = abs_diff(q1p1, q0p0);
    abs_q1q0 =  _mm_srli_si128(abs_p1p0, 8);

    abs_p0q0 = abs_diff(q0p0, p0q0);
    abs_p1q1 = abs_diff(q1p1, p1q1);
    flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
    hev = _mm_subs_epu8(flat, thresh);
    hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);

    abs_p0q0 =_mm_adds_epu8(abs_p0q0, abs_p0q0);
    abs_p1q1 = _mm_srli_epi16(_mm_and_si128(abs_p1q1, fe), 1);
    mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
    mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
    // mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > blimit) * -1;
    mask = _mm_max_epu8(abs_p1p0, mask);
    // mask |= (abs(p1 - p0) > limit) * -1;
    // mask |= (abs(q1 - q0) > limit) * -1;

    work = _mm_max_epu8(abs_diff(q2p2, q1p1),
                        abs_diff(q3p3, q2p2));
    mask = _mm_max_epu8(work, mask);
    mask = _mm_max_epu8(mask, _mm_srli_si128(mask, 8));
    mask = _mm_subs_epu8(mask, limit);
    mask = _mm_cmpeq_epi8(mask, zero);

    // flat_mask4

    flat = _mm_max_epu8(abs_diff(q2p2, q0p0),
                        abs_diff(q3p3, q0p0));
    flat = _mm_max_epu8(abs_p1p0, flat);
    flat = _mm_max_epu8(flat, _mm_srli_si128(flat, 8));
    flat = _mm_subs_epu8(flat, one);
    flat = _mm_cmpeq_epi8(flat, zero);
    flat = _mm_and_si128(flat, mask);
  }

  {
    const __m128i four = _mm_set1_epi16(4);
    unsigned char *src = s;
    {
      __m128i workp_a, workp_b, workp_shft;
      p3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 4 * p)), zero);
      p2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 3 * p)), zero);
      p1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 2 * p)), zero);
      p0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 1 * p)), zero);
      q0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 0 * p)), zero);
      q1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src + 1 * p)), zero);
      q2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src + 2 * p)), zero);
      q3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src + 3 * p)), zero);

      workp_a = _mm_add_epi16(_mm_add_epi16(p3, p3), _mm_add_epi16(p2, p1));
      workp_a = _mm_add_epi16(_mm_add_epi16(workp_a, four), p0);
      workp_b = _mm_add_epi16(_mm_add_epi16(q0, p2), p3);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_op2[0],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_b = _mm_add_epi16(_mm_add_epi16(q0, q1), p1);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_op1[0],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3), q2);
      workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p1), p0);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_op0[0],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3), q3);
      workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p0), q0);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_oq0[0],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p2), q3);
      workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q0), q1);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_oq1[0],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p1), q3);
      workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q1), q2);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_oq2[0],
                       _mm_packus_epi16(workp_shft, workp_shft));
    }
  }
  // lp filter
  {
    const __m128i t4 = _mm_set1_epi8(4);
    const __m128i t3 = _mm_set1_epi8(3);
    const __m128i t80 = _mm_set1_epi8(0x80);
    const __m128i t1 = _mm_set1_epi8(0x1);
    const __m128i ps1 = _mm_xor_si128(_mm_loadl_epi64((__m128i *)(s - 2 * p)),
                                      t80);
    const __m128i ps0 = _mm_xor_si128(_mm_loadl_epi64((__m128i *)(s - 1 * p)),
                                      t80);
    const __m128i qs0 = _mm_xor_si128(_mm_loadl_epi64((__m128i *)(s + 0 * p)),
                                      t80);
    const __m128i qs1 = _mm_xor_si128(_mm_loadl_epi64((__m128i *)(s + 1 * p)),
                                      t80);
    __m128i filt;
    __m128i work_a;
    __m128i filter1, filter2;

    filt = _mm_and_si128(_mm_subs_epi8(ps1, qs1), hev);
    work_a = _mm_subs_epi8(qs0, ps0);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    // (vpx_filter + 3 * (qs0 - ps0)) & mask
    filt = _mm_and_si128(filt, mask);

    filter1 = _mm_adds_epi8(filt, t4);
    filter2 = _mm_adds_epi8(filt, t3);

    // Filter1 >> 3
    filter1 = _mm_unpacklo_epi8(zero, filter1);
    filter1 = _mm_srai_epi16(filter1, 11);
    filter1 = _mm_packs_epi16(filter1, filter1);

    // Filter2 >> 3
    filter2 = _mm_unpacklo_epi8(zero, filter2);
    filter2 = _mm_srai_epi16(filter2, 11);
    filter2 = _mm_packs_epi16(filter2, zero);

    // filt >> 1
    filt = _mm_adds_epi8(filter1, t1);
    filt = _mm_unpacklo_epi8(zero, filt);
    filt = _mm_srai_epi16(filt, 9);
    filt = _mm_packs_epi16(filt, zero);

    filt = _mm_andnot_si128(hev, filt);

    work_a = _mm_xor_si128(_mm_subs_epi8(qs0, filter1), t80);
    q0 = _mm_loadl_epi64((__m128i *)flat_oq0);
    work_a = _mm_andnot_si128(flat, work_a);
    q0 = _mm_and_si128(flat, q0);
    q0 = _mm_or_si128(work_a, q0);

    work_a = _mm_xor_si128(_mm_subs_epi8(qs1, filt), t80);
    q1 = _mm_loadl_epi64((__m128i *)flat_oq1);
    work_a = _mm_andnot_si128(flat, work_a);
    q1 = _mm_and_si128(flat, q1);
    q1 = _mm_or_si128(work_a, q1);

    work_a = _mm_loadu_si128((__m128i *)(s + 2 * p));
    q2 = _mm_loadl_epi64((__m128i *)flat_oq2);
    work_a = _mm_andnot_si128(flat, work_a);
    q2 = _mm_and_si128(flat, q2);
    q2 = _mm_or_si128(work_a, q2);

    work_a = _mm_xor_si128(_mm_adds_epi8(ps0, filter2), t80);
    p0 = _mm_loadl_epi64((__m128i *)flat_op0);
    work_a = _mm_andnot_si128(flat, work_a);
    p0 = _mm_and_si128(flat, p0);
    p0 = _mm_or_si128(work_a, p0);

    work_a = _mm_xor_si128(_mm_adds_epi8(ps1, filt), t80);
    p1 = _mm_loadl_epi64((__m128i *)flat_op1);
    work_a = _mm_andnot_si128(flat, work_a);
    p1 = _mm_and_si128(flat, p1);
    p1 = _mm_or_si128(work_a, p1);

    work_a = _mm_loadu_si128((__m128i *)(s - 3 * p));
    p2 = _mm_loadl_epi64((__m128i *)flat_op2);
    work_a = _mm_andnot_si128(flat, work_a);
    p2 = _mm_and_si128(flat, p2);
    p2 = _mm_or_si128(work_a, p2);

    _mm_storel_epi64((__m128i *)(s - 3 * p), p2);
    _mm_storel_epi64((__m128i *)(s - 2 * p), p1);
    _mm_storel_epi64((__m128i *)(s - 1 * p), p0);
    _mm_storel_epi64((__m128i *)(s + 0 * p), q0);
    _mm_storel_epi64((__m128i *)(s + 1 * p), q1);
    _mm_storel_epi64((__m128i *)(s + 2 * p), q2);
  }
}

void vpx_lpf_horizontal_8_dual_sse2(uint8_t *s, int p,
                                    const uint8_t *_blimit0,
                                    const uint8_t *_limit0,
                                    const uint8_t *_thresh0,
                                    const uint8_t *_blimit1,
                                    const uint8_t *_limit1,
                                    const uint8_t *_thresh1) {
  DECLARE_ALIGNED(16, unsigned char, flat_op2[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_op1[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_op0[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_oq2[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_oq1[16]);
  DECLARE_ALIGNED(16, unsigned char, flat_oq0[16]);
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i blimit =
      _mm_unpacklo_epi64(_mm_load_si128((const __m128i *)_blimit0),
                         _mm_load_si128((const __m128i *)_blimit1));
  const __m128i limit =
      _mm_unpacklo_epi64(_mm_load_si128((const __m128i *)_limit0),
                         _mm_load_si128((const __m128i *)_limit1));
  const __m128i thresh =
      _mm_unpacklo_epi64(_mm_load_si128((const __m128i *)_thresh0),
                         _mm_load_si128((const __m128i *)_thresh1));

  __m128i mask, hev, flat;
  __m128i p3, p2, p1, p0, q0, q1, q2, q3;

  p3 = _mm_loadu_si128((__m128i *)(s - 4 * p));
  p2 = _mm_loadu_si128((__m128i *)(s - 3 * p));
  p1 = _mm_loadu_si128((__m128i *)(s - 2 * p));
  p0 = _mm_loadu_si128((__m128i *)(s - 1 * p));
  q0 = _mm_loadu_si128((__m128i *)(s - 0 * p));
  q1 = _mm_loadu_si128((__m128i *)(s + 1 * p));
  q2 = _mm_loadu_si128((__m128i *)(s + 2 * p));
  q3 = _mm_loadu_si128((__m128i *)(s + 3 * p));
  {
    const __m128i abs_p1p0 = _mm_or_si128(_mm_subs_epu8(p1, p0),
                                          _mm_subs_epu8(p0, p1));
    const __m128i abs_q1q0 = _mm_or_si128(_mm_subs_epu8(q1, q0),
                                          _mm_subs_epu8(q0, q1));
    const __m128i one = _mm_set1_epi8(1);
    const __m128i fe = _mm_set1_epi8(0xfe);
    const __m128i ff = _mm_cmpeq_epi8(abs_p1p0, abs_p1p0);
    __m128i abs_p0q0 = _mm_or_si128(_mm_subs_epu8(p0, q0),
                                    _mm_subs_epu8(q0, p0));
    __m128i abs_p1q1 = _mm_or_si128(_mm_subs_epu8(p1, q1),
                                    _mm_subs_epu8(q1, p1));
    __m128i work;

    // filter_mask and hev_mask
    flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
    hev = _mm_subs_epu8(flat, thresh);
    hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);

    abs_p0q0 =_mm_adds_epu8(abs_p0q0, abs_p0q0);
    abs_p1q1 = _mm_srli_epi16(_mm_and_si128(abs_p1q1, fe), 1);
    mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
    mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
    // mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > blimit) * -1;
    mask = _mm_max_epu8(flat, mask);
    // mask |= (abs(p1 - p0) > limit) * -1;
    // mask |= (abs(q1 - q0) > limit) * -1;
    work = _mm_max_epu8(_mm_or_si128(_mm_subs_epu8(p2, p1),
                                     _mm_subs_epu8(p1, p2)),
                         _mm_or_si128(_mm_subs_epu8(p3, p2),
                                      _mm_subs_epu8(p2, p3)));
    mask = _mm_max_epu8(work, mask);
    work = _mm_max_epu8(_mm_or_si128(_mm_subs_epu8(q2, q1),
                                     _mm_subs_epu8(q1, q2)),
                         _mm_or_si128(_mm_subs_epu8(q3, q2),
                                      _mm_subs_epu8(q2, q3)));
    mask = _mm_max_epu8(work, mask);
    mask = _mm_subs_epu8(mask, limit);
    mask = _mm_cmpeq_epi8(mask, zero);

    // flat_mask4
    work = _mm_max_epu8(_mm_or_si128(_mm_subs_epu8(p2, p0),
                                     _mm_subs_epu8(p0, p2)),
                         _mm_or_si128(_mm_subs_epu8(q2, q0),
                                      _mm_subs_epu8(q0, q2)));
    flat = _mm_max_epu8(work, flat);
    work = _mm_max_epu8(_mm_or_si128(_mm_subs_epu8(p3, p0),
                                     _mm_subs_epu8(p0, p3)),
                         _mm_or_si128(_mm_subs_epu8(q3, q0),
                                      _mm_subs_epu8(q0, q3)));
    flat = _mm_max_epu8(work, flat);
    flat = _mm_subs_epu8(flat, one);
    flat = _mm_cmpeq_epi8(flat, zero);
    flat = _mm_and_si128(flat, mask);
  }
  {
    const __m128i four = _mm_set1_epi16(4);
    unsigned char *src = s;
    int i = 0;

    do {
      __m128i workp_a, workp_b, workp_shft;
      p3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 4 * p)), zero);
      p2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 3 * p)), zero);
      p1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 2 * p)), zero);
      p0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 1 * p)), zero);
      q0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src - 0 * p)), zero);
      q1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src + 1 * p)), zero);
      q2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src + 2 * p)), zero);
      q3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(src + 3 * p)), zero);

      workp_a = _mm_add_epi16(_mm_add_epi16(p3, p3), _mm_add_epi16(p2, p1));
      workp_a = _mm_add_epi16(_mm_add_epi16(workp_a, four), p0);
      workp_b = _mm_add_epi16(_mm_add_epi16(q0, p2), p3);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_op2[i * 8],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_b = _mm_add_epi16(_mm_add_epi16(q0, q1), p1);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_op1[i * 8],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3), q2);
      workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p1), p0);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_op0[i * 8],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3), q3);
      workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p0), q0);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_oq0[i * 8],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p2), q3);
      workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q0), q1);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_oq1[i * 8],
                       _mm_packus_epi16(workp_shft, workp_shft));

      workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p1), q3);
      workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q1), q2);
      workp_shft = _mm_srli_epi16(_mm_add_epi16(workp_a, workp_b), 3);
      _mm_storel_epi64((__m128i *)&flat_oq2[i * 8],
                       _mm_packus_epi16(workp_shft, workp_shft));

      src += 8;
    } while (++i < 2);
  }
  // lp filter
  {
    const __m128i t4 = _mm_set1_epi8(4);
    const __m128i t3 = _mm_set1_epi8(3);
    const __m128i t80 = _mm_set1_epi8(0x80);
    const __m128i te0 = _mm_set1_epi8(0xe0);
    const __m128i t1f = _mm_set1_epi8(0x1f);
    const __m128i t1 = _mm_set1_epi8(0x1);
    const __m128i t7f = _mm_set1_epi8(0x7f);

    const __m128i ps1 = _mm_xor_si128(_mm_loadu_si128((__m128i *)(s - 2 * p)),
                                      t80);
    const __m128i ps0 = _mm_xor_si128(_mm_loadu_si128((__m128i *)(s - 1 * p)),
                                      t80);
    const __m128i qs0 = _mm_xor_si128(_mm_loadu_si128((__m128i *)(s + 0 * p)),
                                      t80);
    const __m128i qs1 = _mm_xor_si128(_mm_loadu_si128((__m128i *)(s + 1 * p)),
                                      t80);
    __m128i filt;
    __m128i work_a;
    __m128i filter1, filter2;

    filt = _mm_and_si128(_mm_subs_epi8(ps1, qs1), hev);
    work_a = _mm_subs_epi8(qs0, ps0);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    // (vpx_filter + 3 * (qs0 - ps0)) & mask
    filt = _mm_and_si128(filt, mask);

    filter1 = _mm_adds_epi8(filt, t4);
    filter2 = _mm_adds_epi8(filt, t3);

    // Filter1 >> 3
    work_a = _mm_cmpgt_epi8(zero, filter1);
    filter1 = _mm_srli_epi16(filter1, 3);
    work_a = _mm_and_si128(work_a, te0);
    filter1 = _mm_and_si128(filter1, t1f);
    filter1 = _mm_or_si128(filter1, work_a);

    // Filter2 >> 3
    work_a = _mm_cmpgt_epi8(zero, filter2);
    filter2 = _mm_srli_epi16(filter2, 3);
    work_a = _mm_and_si128(work_a, te0);
    filter2 = _mm_and_si128(filter2, t1f);
    filter2 = _mm_or_si128(filter2, work_a);

    // filt >> 1
    filt = _mm_adds_epi8(filter1, t1);
    work_a = _mm_cmpgt_epi8(zero, filt);
    filt = _mm_srli_epi16(filt, 1);
    work_a = _mm_and_si128(work_a, t80);
    filt = _mm_and_si128(filt, t7f);
    filt = _mm_or_si128(filt, work_a);

    filt = _mm_andnot_si128(hev, filt);

    work_a = _mm_xor_si128(_mm_subs_epi8(qs0, filter1), t80);
    q0 = _mm_load_si128((__m128i *)flat_oq0);
    work_a = _mm_andnot_si128(flat, work_a);
    q0 = _mm_and_si128(flat, q0);
    q0 = _mm_or_si128(work_a, q0);

    work_a = _mm_xor_si128(_mm_subs_epi8(qs1, filt), t80);
    q1 = _mm_load_si128((__m128i *)flat_oq1);
    work_a = _mm_andnot_si128(flat, work_a);
    q1 = _mm_and_si128(flat, q1);
    q1 = _mm_or_si128(work_a, q1);

    work_a = _mm_loadu_si128((__m128i *)(s + 2 * p));
    q2 = _mm_load_si128((__m128i *)flat_oq2);
    work_a = _mm_andnot_si128(flat, work_a);
    q2 = _mm_and_si128(flat, q2);
    q2 = _mm_or_si128(work_a, q2);

    work_a = _mm_xor_si128(_mm_adds_epi8(ps0, filter2), t80);
    p0 = _mm_load_si128((__m128i *)flat_op0);
    work_a = _mm_andnot_si128(flat, work_a);
    p0 = _mm_and_si128(flat, p0);
    p0 = _mm_or_si128(work_a, p0);

    work_a = _mm_xor_si128(_mm_adds_epi8(ps1, filt), t80);
    p1 = _mm_load_si128((__m128i *)flat_op1);
    work_a = _mm_andnot_si128(flat, work_a);
    p1 = _mm_and_si128(flat, p1);
    p1 = _mm_or_si128(work_a, p1);

    work_a = _mm_loadu_si128((__m128i *)(s - 3 * p));
    p2 = _mm_load_si128((__m128i *)flat_op2);
    work_a = _mm_andnot_si128(flat, work_a);
    p2 = _mm_and_si128(flat, p2);
    p2 = _mm_or_si128(work_a, p2);

    _mm_storeu_si128((__m128i *)(s - 3 * p), p2);
    _mm_storeu_si128((__m128i *)(s - 2 * p), p1);
    _mm_storeu_si128((__m128i *)(s - 1 * p), p0);
    _mm_storeu_si128((__m128i *)(s + 0 * p), q0);
    _mm_storeu_si128((__m128i *)(s + 1 * p), q1);
    _mm_storeu_si128((__m128i *)(s + 2 * p), q2);
  }
}

void vpx_lpf_horizontal_4_dual_sse2(unsigned char *s, int p,
                                    const unsigned char *_blimit0,
                                    const unsigned char *_limit0,
                                    const unsigned char *_thresh0,
                                    const unsigned char *_blimit1,
                                    const unsigned char *_limit1,
                                    const unsigned char *_thresh1) {
  const __m128i blimit =
      _mm_unpacklo_epi64(_mm_load_si128((const __m128i *)_blimit0),
                         _mm_load_si128((const __m128i *)_blimit1));
  const __m128i limit =
      _mm_unpacklo_epi64(_mm_load_si128((const __m128i *)_limit0),
                         _mm_load_si128((const __m128i *)_limit1));
  const __m128i thresh =
      _mm_unpacklo_epi64(_mm_load_si128((const __m128i *)_thresh0),
                         _mm_load_si128((const __m128i *)_thresh1));
  const __m128i zero = _mm_set1_epi16(0);
  __m128i p3, p2, p1, p0, q0, q1, q2, q3;
  __m128i mask, hev, flat;

  p3 = _mm_loadu_si128((__m128i *)(s - 4 * p));
  p2 = _mm_loadu_si128((__m128i *)(s - 3 * p));
  p1 = _mm_loadu_si128((__m128i *)(s - 2 * p));
  p0 = _mm_loadu_si128((__m128i *)(s - 1 * p));
  q0 = _mm_loadu_si128((__m128i *)(s - 0 * p));
  q1 = _mm_loadu_si128((__m128i *)(s + 1 * p));
  q2 = _mm_loadu_si128((__m128i *)(s + 2 * p));
  q3 = _mm_loadu_si128((__m128i *)(s + 3 * p));

  // filter_mask and hev_mask
  {
    const __m128i abs_p1p0 = _mm_or_si128(_mm_subs_epu8(p1, p0),
                                          _mm_subs_epu8(p0, p1));
    const __m128i abs_q1q0 = _mm_or_si128(_mm_subs_epu8(q1, q0),
                                          _mm_subs_epu8(q0, q1));
    const __m128i fe = _mm_set1_epi8(0xfe);
    const __m128i ff = _mm_cmpeq_epi8(abs_p1p0, abs_p1p0);
    __m128i abs_p0q0 = _mm_or_si128(_mm_subs_epu8(p0, q0),
                                    _mm_subs_epu8(q0, p0));
    __m128i abs_p1q1 = _mm_or_si128(_mm_subs_epu8(p1, q1),
                                    _mm_subs_epu8(q1, p1));
    __m128i work;

    flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
    hev = _mm_subs_epu8(flat, thresh);
    hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);

    abs_p0q0 =_mm_adds_epu8(abs_p0q0, abs_p0q0);
    abs_p1q1 = _mm_srli_epi16(_mm_and_si128(abs_p1q1, fe), 1);
    mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
    mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
    // mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > blimit) * -1;
    mask = _mm_max_epu8(flat, mask);
    // mask |= (abs(p1 - p0) > limit) * -1;
    // mask |= (abs(q1 - q0) > limit) * -1;
    work = _mm_max_epu8(_mm_or_si128(_mm_subs_epu8(p2, p1),
                                     _mm_subs_epu8(p1, p2)),
                         _mm_or_si128(_mm_subs_epu8(p3, p2),
                                      _mm_subs_epu8(p2, p3)));
    mask = _mm_max_epu8(work, mask);
    work = _mm_max_epu8(_mm_or_si128(_mm_subs_epu8(q2, q1),
                                     _mm_subs_epu8(q1, q2)),
                         _mm_or_si128(_mm_subs_epu8(q3, q2),
                                      _mm_subs_epu8(q2, q3)));
    mask = _mm_max_epu8(work, mask);
    mask = _mm_subs_epu8(mask, limit);
    mask = _mm_cmpeq_epi8(mask, zero);
  }

  // filter4
  {
    const __m128i t4 = _mm_set1_epi8(4);
    const __m128i t3 = _mm_set1_epi8(3);
    const __m128i t80 = _mm_set1_epi8(0x80);
    const __m128i te0 = _mm_set1_epi8(0xe0);
    const __m128i t1f = _mm_set1_epi8(0x1f);
    const __m128i t1 = _mm_set1_epi8(0x1);
    const __m128i t7f = _mm_set1_epi8(0x7f);

    const __m128i ps1 = _mm_xor_si128(_mm_loadu_si128((__m128i *)(s - 2 * p)),
                                      t80);
    const __m128i ps0 = _mm_xor_si128(_mm_loadu_si128((__m128i *)(s - 1 * p)),
                                      t80);
    const __m128i qs0 = _mm_xor_si128(_mm_loadu_si128((__m128i *)(s + 0 * p)),
                                      t80);
    const __m128i qs1 = _mm_xor_si128(_mm_loadu_si128((__m128i *)(s + 1 * p)),
                                      t80);
    __m128i filt;
    __m128i work_a;
    __m128i filter1, filter2;

    filt = _mm_and_si128(_mm_subs_epi8(ps1, qs1), hev);
    work_a = _mm_subs_epi8(qs0, ps0);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    filt = _mm_adds_epi8(filt, work_a);
    // (vpx_filter + 3 * (qs0 - ps0)) & mask
    filt = _mm_and_si128(filt, mask);

    filter1 = _mm_adds_epi8(filt, t4);
    filter2 = _mm_adds_epi8(filt, t3);

    // Filter1 >> 3
    work_a = _mm_cmpgt_epi8(zero, filter1);
    filter1 = _mm_srli_epi16(filter1, 3);
    work_a = _mm_and_si128(work_a, te0);
    filter1 = _mm_and_si128(filter1, t1f);
    filter1 = _mm_or_si128(filter1, work_a);

    // Filter2 >> 3
    work_a = _mm_cmpgt_epi8(zero, filter2);
    filter2 = _mm_srli_epi16(filter2, 3);
    work_a = _mm_and_si128(work_a, te0);
    filter2 = _mm_and_si128(filter2, t1f);
    filter2 = _mm_or_si128(filter2, work_a);

    // filt >> 1
    filt = _mm_adds_epi8(filter1, t1);
    work_a = _mm_cmpgt_epi8(zero, filt);
    filt = _mm_srli_epi16(filt, 1);
    work_a = _mm_and_si128(work_a, t80);
    filt = _mm_and_si128(filt, t7f);
    filt = _mm_or_si128(filt, work_a);

    filt = _mm_andnot_si128(hev, filt);

    q0 = _mm_xor_si128(_mm_subs_epi8(qs0, filter1), t80);
    q1 = _mm_xor_si128(_mm_subs_epi8(qs1, filt), t80);
    p0 = _mm_xor_si128(_mm_adds_epi8(ps0, filter2), t80);
    p1 = _mm_xor_si128(_mm_adds_epi8(ps1, filt), t80);

    _mm_storeu_si128((__m128i *)(s - 2 * p), p1);
    _mm_storeu_si128((__m128i *)(s - 1 * p), p0);
    _mm_storeu_si128((__m128i *)(s + 0 * p), q0);
    _mm_storeu_si128((__m128i *)(s + 1 * p), q1);
  }
}

static INLINE void transpose8x16(unsigned char *in0, unsigned char *in1,
                                 int in_p, unsigned char *out, int out_p) {
  __m128i x0, x1, x2, x3, x4, x5, x6, x7;
  __m128i x8, x9, x10, x11, x12, x13, x14, x15;

  // 2-way interleave w/hoisting of unpacks
  x0 = _mm_loadl_epi64((__m128i *)in0);  // 1
  x1 = _mm_loadl_epi64((__m128i *)(in0 + in_p));  // 3
  x0 = _mm_unpacklo_epi8(x0, x1);  // 1

  x2 = _mm_loadl_epi64((__m128i *)(in0 + 2 * in_p));  // 5
  x3 = _mm_loadl_epi64((__m128i *)(in0 + 3*in_p));  // 7
  x1 = _mm_unpacklo_epi8(x2, x3);  // 2

  x4 = _mm_loadl_epi64((__m128i *)(in0 + 4*in_p));  // 9
  x5 = _mm_loadl_epi64((__m128i *)(in0 + 5*in_p));  // 11
  x2 = _mm_unpacklo_epi8(x4, x5);  // 3

  x6 = _mm_loadl_epi64((__m128i *)(in0 + 6*in_p));  // 13
  x7 = _mm_loadl_epi64((__m128i *)(in0 + 7*in_p));  // 15
  x3 = _mm_unpacklo_epi8(x6, x7);  // 4
  x4 = _mm_unpacklo_epi16(x0, x1);  // 9

  x8 = _mm_loadl_epi64((__m128i *)in1);  // 2
  x9 = _mm_loadl_epi64((__m128i *)(in1 + in_p));  // 4
  x8 = _mm_unpacklo_epi8(x8, x9);  // 5
  x5 = _mm_unpacklo_epi16(x2, x3);  // 10

  x10 = _mm_loadl_epi64((__m128i *)(in1 + 2 * in_p));  // 6
  x11 = _mm_loadl_epi64((__m128i *)(in1 + 3*in_p));  // 8
  x9 = _mm_unpacklo_epi8(x10, x11);  // 6

  x12 = _mm_loadl_epi64((__m128i *)(in1 + 4*in_p));  // 10
  x13 = _mm_loadl_epi64((__m128i *)(in1 + 5*in_p));  // 12
  x10 = _mm_unpacklo_epi8(x12, x13);  // 7
  x12 = _mm_unpacklo_epi16(x8, x9);  // 11

  x14 = _mm_loadl_epi64((__m128i *)(in1 + 6*in_p));  // 14
  x15 = _mm_loadl_epi64((__m128i *)(in1 + 7*in_p));  // 16
  x11 = _mm_unpacklo_epi8(x14, x15);  // 8
  x13 = _mm_unpacklo_epi16(x10, x11);  // 12

  x6 = _mm_unpacklo_epi32(x4, x5);  // 13
  x7 = _mm_unpackhi_epi32(x4, x5);  // 14
  x14 = _mm_unpacklo_epi32(x12, x13);  // 15
  x15 = _mm_unpackhi_epi32(x12, x13);  // 16

  // Store first 4-line result
  _mm_storeu_si128((__m128i *)out, _mm_unpacklo_epi64(x6, x14));
  _mm_storeu_si128((__m128i *)(out + out_p), _mm_unpackhi_epi64(x6, x14));
  _mm_storeu_si128((__m128i *)(out + 2 * out_p), _mm_unpacklo_epi64(x7, x15));
  _mm_storeu_si128((__m128i *)(out + 3 * out_p), _mm_unpackhi_epi64(x7, x15));

  x4 = _mm_unpackhi_epi16(x0, x1);
  x5 = _mm_unpackhi_epi16(x2, x3);
  x12 = _mm_unpackhi_epi16(x8, x9);
  x13 = _mm_unpackhi_epi16(x10, x11);

  x6 = _mm_unpacklo_epi32(x4, x5);
  x7 = _mm_unpackhi_epi32(x4, x5);
  x14 = _mm_unpacklo_epi32(x12, x13);
  x15 = _mm_unpackhi_epi32(x12, x13);

  // Store second 4-line result
  _mm_storeu_si128((__m128i *)(out + 4 * out_p), _mm_unpacklo_epi64(x6, x14));
  _mm_storeu_si128((__m128i *)(out + 5 * out_p), _mm_unpackhi_epi64(x6, x14));
  _mm_storeu_si128((__m128i *)(out + 6 * out_p), _mm_unpacklo_epi64(x7, x15));
  _mm_storeu_si128((__m128i *)(out + 7 * out_p), _mm_unpackhi_epi64(x7, x15));
}

static INLINE void transpose(unsigned char *src[], int in_p,
                             unsigned char *dst[], int out_p,
                             int num_8x8_to_transpose) {
  int idx8x8 = 0;
  __m128i x0, x1, x2, x3, x4, x5, x6, x7;
  do {
    unsigned char *in = src[idx8x8];
    unsigned char *out = dst[idx8x8];

    x0 = _mm_loadl_epi64((__m128i *)(in + 0*in_p));  // 00 01 02 03 04 05 06 07
    x1 = _mm_loadl_epi64((__m128i *)(in + 1*in_p));  // 10 11 12 13 14 15 16 17
    // 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
    x0 = _mm_unpacklo_epi8(x0, x1);

    x2 = _mm_loadl_epi64((__m128i *)(in + 2*in_p));  // 20 21 22 23 24 25 26 27
    x3 = _mm_loadl_epi64((__m128i *)(in + 3*in_p));  // 30 31 32 33 34 35 36 37
    // 20 30 21 31 22 32 23 33 24 34 25 35 26 36 27 37
    x1 = _mm_unpacklo_epi8(x2, x3);

    x4 = _mm_loadl_epi64((__m128i *)(in + 4*in_p));  // 40 41 42 43 44 45 46 47
    x5 = _mm_loadl_epi64((__m128i *)(in + 5*in_p));  // 50 51 52 53 54 55 56 57
    // 40 50 41 51 42 52 43 53 44 54 45 55 46 56 47 57
    x2 = _mm_unpacklo_epi8(x4, x5);

    x6 = _mm_loadl_epi64((__m128i *)(in + 6*in_p));  // 60 61 62 63 64 65 66 67
    x7 = _mm_loadl_epi64((__m128i *)(in + 7*in_p));  // 70 71 72 73 74 75 76 77
    // 60 70 61 71 62 72 63 73 64 74 65 75 66 76 67 77
    x3 = _mm_unpacklo_epi8(x6, x7);

    // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
    x4 = _mm_unpacklo_epi16(x0, x1);
    // 40 50 60 70 41 51 61 71 42 52 62 72 43 53 63 73
    x5 = _mm_unpacklo_epi16(x2, x3);
    // 00 10 20 30 40 50 60 70 01 11 21 31 41 51 61 71
    x6 = _mm_unpacklo_epi32(x4, x5);
    _mm_storel_pd((double *)(out + 0*out_p),
                  _mm_castsi128_pd(x6));  // 00 10 20 30 40 50 60 70
    _mm_storeh_pd((double *)(out + 1*out_p),
                  _mm_castsi128_pd(x6));  // 01 11 21 31 41 51 61 71
    // 02 12 22 32 42 52 62 72 03 13 23 33 43 53 63 73
    x7 = _mm_unpackhi_epi32(x4, x5);
    _mm_storel_pd((double *)(out + 2*out_p),
                  _mm_castsi128_pd(x7));  // 02 12 22 32 42 52 62 72
    _mm_storeh_pd((double *)(out + 3*out_p),
                  _mm_castsi128_pd(x7));  // 03 13 23 33 43 53 63 73

    // 04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37
    x4 = _mm_unpackhi_epi16(x0, x1);
    // 44 54 64 74 45 55 65 75 46 56 66 76 47 57 67 77
    x5 = _mm_unpackhi_epi16(x2, x3);
    // 04 14 24 34 44 54 64 74 05 15 25 35 45 55 65 75
    x6 = _mm_unpacklo_epi32(x4, x5);
    _mm_storel_pd((double *)(out + 4*out_p),
                  _mm_castsi128_pd(x6));  // 04 14 24 34 44 54 64 74
    _mm_storeh_pd((double *)(out + 5*out_p),
                  _mm_castsi128_pd(x6));  // 05 15 25 35 45 55 65 75
    // 06 16 26 36 46 56 66 76 07 17 27 37 47 57 67 77
    x7 = _mm_unpackhi_epi32(x4, x5);

    _mm_storel_pd((double *)(out + 6*out_p),
                  _mm_castsi128_pd(x7));  // 06 16 26 36 46 56 66 76
    _mm_storeh_pd((double *)(out + 7*out_p),
                  _mm_castsi128_pd(x7));  // 07 17 27 37 47 57 67 77
  } while (++idx8x8 < num_8x8_to_transpose);
}

void vpx_lpf_vertical_4_dual_sse2(uint8_t *s, int p, const uint8_t *blimit0,
                                  const uint8_t *limit0,
                                  const uint8_t *thresh0,
                                  const uint8_t *blimit1,
                                  const uint8_t *limit1,
                                  const uint8_t *thresh1) {
  DECLARE_ALIGNED(16, unsigned char, t_dst[16 * 8]);
  unsigned char *src[2];
  unsigned char *dst[2];

  // Transpose 8x16
  transpose8x16(s - 4, s - 4 + p * 8, p, t_dst, 16);

  // Loop filtering
  vpx_lpf_horizontal_4_dual_sse2(t_dst + 4 * 16, 16, blimit0, limit0, thresh0,
                                 blimit1, limit1, thresh1);
  src[0] = t_dst;
  src[1] = t_dst + 8;
  dst[0] = s - 4;
  dst[1] = s - 4 + p * 8;

  // Transpose back
  transpose(src, 16, dst, p, 2);
}

void vpx_lpf_vertical_8_sse2(unsigned char *s, int p,
                             const unsigned char *blimit,
                             const unsigned char *limit,
                             const unsigned char *thresh) {
  DECLARE_ALIGNED(8, unsigned char, t_dst[8 * 8]);
  unsigned char *src[1];
  unsigned char *dst[1];

  // Transpose 8x8
  src[0] = s - 4;
  dst[0] = t_dst;

  transpose(src, p, dst, 8, 1);

  // Loop filtering
  vpx_lpf_horizontal_8_sse2(t_dst + 4 * 8, 8, blimit, limit, thresh);

  src[0] = t_dst;
  dst[0] = s - 4;

  // Transpose back
  transpose(src, 8, dst, p, 1);
}

void vpx_lpf_vertical_8_dual_sse2(uint8_t *s, int p, const uint8_t *blimit0,
                                  const uint8_t *limit0,
                                  const uint8_t *thresh0,
                                  const uint8_t *blimit1,
                                  const uint8_t *limit1,
                                  const uint8_t *thresh1) {
  DECLARE_ALIGNED(16, unsigned char, t_dst[16 * 8]);
  unsigned char *src[2];
  unsigned char *dst[2];

  // Transpose 8x16
  transpose8x16(s - 4, s - 4 + p * 8, p, t_dst, 16);

  // Loop filtering
  vpx_lpf_horizontal_8_dual_sse2(t_dst + 4 * 16, 16, blimit0, limit0, thresh0,
                                 blimit1, limit1, thresh1);
  src[0] = t_dst;
  src[1] = t_dst + 8;

  dst[0] = s - 4;
  dst[1] = s - 4 + p * 8;

  // Transpose back
  transpose(src, 16, dst, p, 2);
}

void vpx_lpf_vertical_16_sse2(unsigned char *s, int p,
                              const unsigned char *blimit,
                              const unsigned char *limit,
                              const unsigned char *thresh) {
  DECLARE_ALIGNED(8, unsigned char, t_dst[8 * 16]);
  unsigned char *src[2];
  unsigned char *dst[2];

  src[0] = s - 8;
  src[1] = s;
  dst[0] = t_dst;
  dst[1] = t_dst + 8 * 8;

  // Transpose 16x8
  transpose(src, p, dst, 8, 2);

  // Loop filtering
  vpx_lpf_horizontal_edge_8_sse2(t_dst + 8 * 8, 8, blimit, limit, thresh);

  src[0] = t_dst;
  src[1] = t_dst + 8 * 8;
  dst[0] = s - 8;
  dst[1] = s;

  // Transpose back
  transpose(src, 8, dst, p, 2);
}

void vpx_lpf_vertical_16_dual_sse2(unsigned char *s, int p,
                                   const uint8_t *blimit, const uint8_t *limit,
                                   const uint8_t *thresh) {
  DECLARE_ALIGNED(16, unsigned char, t_dst[256]);

  // Transpose 16x16
  transpose8x16(s - 8, s - 8 + 8 * p, p, t_dst, 16);
  transpose8x16(s, s + 8 * p, p, t_dst + 8 * 16, 16);

  // Loop filtering
  vpx_lpf_horizontal_edge_16_sse2(t_dst + 8 * 16, 16, blimit, limit, thresh);

  // Transpose back
  transpose8x16(t_dst, t_dst + 8 * 16, 16, s - 8, p);
  transpose8x16(t_dst + 8, t_dst + 8 + 8 * 16, 16, s - 8 + 8 * p, p);
}
