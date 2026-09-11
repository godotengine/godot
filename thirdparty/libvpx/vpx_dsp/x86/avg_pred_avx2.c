/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <immintrin.h>

#include "./vpx_dsp_rtcd.h"

void vpx_comp_avg_pred_avx2(uint8_t *comp_pred, const uint8_t *pred, int width,
                            int height, const uint8_t *ref, int ref_stride) {
  int row = 0;
  // comp_pred and pred must be 32 byte aligned.
  assert(((intptr_t)comp_pred % 32) == 0);
  assert(((intptr_t)pred % 32) == 0);

  if (width == 8) {
    assert(height % 4 == 0);
    do {
      const __m256i p = _mm256_load_si256((const __m256i *)pred);
      const __m128i r_0 = _mm_loadl_epi64((const __m128i *)ref);
      const __m128i r_1 =
          _mm_loadl_epi64((const __m128i *)(ref + 2 * ref_stride));

      const __m128i r1 = _mm_castps_si128(_mm_loadh_pi(
          _mm_castsi128_ps(r_0), (const __m64 *)(ref + ref_stride)));
      const __m128i r2 = _mm_castps_si128(_mm_loadh_pi(
          _mm_castsi128_ps(r_1), (const __m64 *)(ref + 3 * ref_stride)));

      const __m256i ref_0123 =
          _mm256_inserti128_si256(_mm256_castsi128_si256(r1), r2, 1);
      const __m256i avg = _mm256_avg_epu8(p, ref_0123);

      _mm256_store_si256((__m256i *)comp_pred, avg);

      row += 4;
      pred += 32;
      comp_pred += 32;
      ref += 4 * ref_stride;
    } while (row < height);
  } else if (width == 16) {
    assert(height % 4 == 0);
    do {
      const __m256i pred_0 = _mm256_load_si256((const __m256i *)pred);
      const __m256i pred_1 = _mm256_load_si256((const __m256i *)(pred + 32));
      const __m256i tmp0 =
          _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)ref));
      const __m256i ref_0 = _mm256_inserti128_si256(
          tmp0, _mm_loadu_si128((const __m128i *)(ref + ref_stride)), 1);
      const __m256i tmp1 = _mm256_castsi128_si256(
          _mm_loadu_si128((const __m128i *)(ref + 2 * ref_stride)));
      const __m256i ref_1 = _mm256_inserti128_si256(
          tmp1, _mm_loadu_si128((const __m128i *)(ref + 3 * ref_stride)), 1);
      const __m256i average_0 = _mm256_avg_epu8(pred_0, ref_0);
      const __m256i average_1 = _mm256_avg_epu8(pred_1, ref_1);
      _mm256_store_si256((__m256i *)comp_pred, average_0);
      _mm256_store_si256((__m256i *)(comp_pred + 32), average_1);

      row += 4;
      pred += 64;
      comp_pred += 64;
      ref += 4 * ref_stride;
    } while (row < height);
  } else if (width == 32) {
    assert(height % 2 == 0);
    do {
      const __m256i pred_0 = _mm256_load_si256((const __m256i *)pred);
      const __m256i pred_1 = _mm256_load_si256((const __m256i *)(pred + 32));
      const __m256i ref_0 = _mm256_loadu_si256((const __m256i *)ref);
      const __m256i ref_1 =
          _mm256_loadu_si256((const __m256i *)(ref + ref_stride));
      const __m256i average_0 = _mm256_avg_epu8(pred_0, ref_0);
      const __m256i average_1 = _mm256_avg_epu8(pred_1, ref_1);
      _mm256_store_si256((__m256i *)comp_pred, average_0);
      _mm256_store_si256((__m256i *)(comp_pred + 32), average_1);

      row += 2;
      pred += 64;
      comp_pred += 64;
      ref += 2 * ref_stride;
    } while (row < height);
  } else if (width % 64 == 0) {
    do {
      int x;
      for (x = 0; x < width; x += 64) {
        const __m256i pred_0 = _mm256_load_si256((const __m256i *)(pred + x));
        const __m256i pred_1 =
            _mm256_load_si256((const __m256i *)(pred + x + 32));
        const __m256i ref_0 = _mm256_loadu_si256((const __m256i *)(ref + x));
        const __m256i ref_1 =
            _mm256_loadu_si256((const __m256i *)(ref + x + 32));
        const __m256i average_0 = _mm256_avg_epu8(pred_0, ref_0);
        const __m256i average_1 = _mm256_avg_epu8(pred_1, ref_1);
        _mm256_store_si256((__m256i *)(comp_pred + x), average_0);
        _mm256_store_si256((__m256i *)(comp_pred + x + 32), average_1);
      }
      row++;
      pred += width;
      comp_pred += width;
      ref += ref_stride;
    } while (row < height);
  } else {
    vpx_comp_avg_pred_sse2(comp_pred, pred, width, height, ref, ref_stride);
  }
}
