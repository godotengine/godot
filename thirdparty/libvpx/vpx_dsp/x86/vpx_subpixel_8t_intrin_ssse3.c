/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <tmmintrin.h>  // SSSE3

#include <string.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_dsp/x86/convolve.h"
#include "vpx_dsp/x86/convolve_sse2.h"
#include "vpx_dsp/x86/convolve_ssse3.h"
#include "vpx_dsp/x86/mem_sse2.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

static INLINE __m128i shuffle_filter_convolve8_8_ssse3(
    const __m128i *const s, const int16_t *const filter) {
  __m128i f[4];
  shuffle_filter_ssse3(filter, f);
  return convolve8_8_ssse3(s, f);
}

// Used by the avx2 implementation.
#if VPX_ARCH_X86_64
// Use the intrinsics below
filter8_1dfunction vpx_filter_block1d4_h8_intrin_ssse3;
filter8_1dfunction vpx_filter_block1d8_h8_intrin_ssse3;
filter8_1dfunction vpx_filter_block1d8_v8_intrin_ssse3;
#define vpx_filter_block1d4_h8_ssse3 vpx_filter_block1d4_h8_intrin_ssse3
#define vpx_filter_block1d8_h8_ssse3 vpx_filter_block1d8_h8_intrin_ssse3
#define vpx_filter_block1d8_v8_ssse3 vpx_filter_block1d8_v8_intrin_ssse3
#else  // VPX_ARCH_X86
// Use the assembly in vpx_dsp/x86/vpx_subpixel_8t_ssse3.asm.
filter8_1dfunction vpx_filter_block1d4_h8_ssse3;
filter8_1dfunction vpx_filter_block1d8_h8_ssse3;
filter8_1dfunction vpx_filter_block1d8_v8_ssse3;
#endif

#if VPX_ARCH_X86_64
void vpx_filter_block1d4_h8_intrin_ssse3(
    const uint8_t *src_ptr, ptrdiff_t src_pitch, uint8_t *output_ptr,
    ptrdiff_t output_pitch, uint32_t output_height, const int16_t *filter) {
  __m128i firstFilters, secondFilters, shuffle1, shuffle2;
  __m128i srcRegFilt1, srcRegFilt2;
  __m128i addFilterReg64, filtersReg, srcReg;
  unsigned int i;

  // create a register with 0,64,0,64,0,64,0,64,0,64,0,64,0,64,0,64
  addFilterReg64 = _mm_set1_epi32((int)0x0400040u);
  filtersReg = _mm_loadu_si128((const __m128i *)filter);
  // converting the 16 bit (short) to  8 bit (byte) and have the same data
  // in both lanes of 128 bit register.
  filtersReg = _mm_packs_epi16(filtersReg, filtersReg);

  // duplicate only the first 16 bits in the filter into the first lane
  firstFilters = _mm_shufflelo_epi16(filtersReg, 0);
  // duplicate only the third 16 bit in the filter into the first lane
  secondFilters = _mm_shufflelo_epi16(filtersReg, 0xAAu);
  // duplicate only the seconds 16 bits in the filter into the second lane
  // firstFilters: k0 k1 k0 k1 k0 k1 k0 k1 k2 k3 k2 k3 k2 k3 k2 k3
  firstFilters = _mm_shufflehi_epi16(firstFilters, 0x55u);
  // duplicate only the forth 16 bits in the filter into the second lane
  // secondFilters: k4 k5 k4 k5 k4 k5 k4 k5 k6 k7 k6 k7 k6 k7 k6 k7
  secondFilters = _mm_shufflehi_epi16(secondFilters, 0xFFu);

  // loading the local filters
  shuffle1 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 2, 3, 3, 4, 4, 5, 5, 6);
  shuffle2 = _mm_setr_epi8(4, 5, 5, 6, 6, 7, 7, 8, 6, 7, 7, 8, 8, 9, 9, 10);

  for (i = 0; i < output_height; i++) {
    srcReg = _mm_loadu_si128((const __m128i *)(src_ptr - 3));

    // filter the source buffer
    srcRegFilt1 = _mm_shuffle_epi8(srcReg, shuffle1);
    srcRegFilt2 = _mm_shuffle_epi8(srcReg, shuffle2);

    // multiply 2 adjacent elements with the filter and add the result
    srcRegFilt1 = _mm_maddubs_epi16(srcRegFilt1, firstFilters);
    srcRegFilt2 = _mm_maddubs_epi16(srcRegFilt2, secondFilters);

    // sum the results together, saturating only on the final step
    // the specific order of the additions prevents outranges
    srcRegFilt1 = _mm_add_epi16(srcRegFilt1, srcRegFilt2);

    // extract the higher half of the register
    srcRegFilt2 = _mm_srli_si128(srcRegFilt1, 8);

    // add the rounding offset early to avoid another saturated add
    srcRegFilt1 = _mm_add_epi16(srcRegFilt1, addFilterReg64);
    srcRegFilt1 = _mm_adds_epi16(srcRegFilt1, srcRegFilt2);

    // shift by 7 bit each 16 bits
    srcRegFilt1 = _mm_srai_epi16(srcRegFilt1, 7);

    // shrink to 8 bit each 16 bits
    srcRegFilt1 = _mm_packus_epi16(srcRegFilt1, srcRegFilt1);
    src_ptr += src_pitch;

    // save only 4 bytes
    *((int *)&output_ptr[0]) = _mm_cvtsi128_si32(srcRegFilt1);

    output_ptr += output_pitch;
  }
}

void vpx_filter_block1d8_h8_intrin_ssse3(
    const uint8_t *src_ptr, ptrdiff_t src_pitch, uint8_t *output_ptr,
    ptrdiff_t output_pitch, uint32_t output_height, const int16_t *filter) {
  unsigned int i;
  __m128i f[4], filt[4], s[4];

  shuffle_filter_ssse3(filter, f);
  filt[0] = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
  filt[1] = _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10);
  filt[2] = _mm_setr_epi8(4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12);
  filt[3] =
      _mm_setr_epi8(6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14);

  for (i = 0; i < output_height; i++) {
    const __m128i srcReg = _mm_loadu_si128((const __m128i *)(src_ptr - 3));

    // filter the source buffer
    s[0] = _mm_shuffle_epi8(srcReg, filt[0]);
    s[1] = _mm_shuffle_epi8(srcReg, filt[1]);
    s[2] = _mm_shuffle_epi8(srcReg, filt[2]);
    s[3] = _mm_shuffle_epi8(srcReg, filt[3]);
    s[0] = convolve8_8_ssse3(s, f);

    // shrink to 8 bit each 16 bits
    s[0] = _mm_packus_epi16(s[0], s[0]);

    src_ptr += src_pitch;

    // save only 8 bytes
    _mm_storel_epi64((__m128i *)&output_ptr[0], s[0]);

    output_ptr += output_pitch;
  }
}

void vpx_filter_block1d8_v8_intrin_ssse3(
    const uint8_t *src_ptr, ptrdiff_t src_pitch, uint8_t *output_ptr,
    ptrdiff_t out_pitch, uint32_t output_height, const int16_t *filter) {
  unsigned int i;
  __m128i f[4], s[8], ss[4];

  shuffle_filter_ssse3(filter, f);

  // load the first 7 rows of 8 bytes
  s[0] = _mm_loadl_epi64((const __m128i *)(src_ptr + 0 * src_pitch));
  s[1] = _mm_loadl_epi64((const __m128i *)(src_ptr + 1 * src_pitch));
  s[2] = _mm_loadl_epi64((const __m128i *)(src_ptr + 2 * src_pitch));
  s[3] = _mm_loadl_epi64((const __m128i *)(src_ptr + 3 * src_pitch));
  s[4] = _mm_loadl_epi64((const __m128i *)(src_ptr + 4 * src_pitch));
  s[5] = _mm_loadl_epi64((const __m128i *)(src_ptr + 5 * src_pitch));
  s[6] = _mm_loadl_epi64((const __m128i *)(src_ptr + 6 * src_pitch));

  for (i = 0; i < output_height; i++) {
    // load the last 8 bytes
    s[7] = _mm_loadl_epi64((const __m128i *)(src_ptr + 7 * src_pitch));

    // merge the result together
    ss[0] = _mm_unpacklo_epi8(s[0], s[1]);
    ss[1] = _mm_unpacklo_epi8(s[2], s[3]);

    // merge the result together
    ss[2] = _mm_unpacklo_epi8(s[4], s[5]);
    ss[3] = _mm_unpacklo_epi8(s[6], s[7]);

    ss[0] = convolve8_8_ssse3(ss, f);
    // shrink to 8 bit each 16 bits
    ss[0] = _mm_packus_epi16(ss[0], ss[0]);

    src_ptr += src_pitch;

    // shift down a row
    s[0] = s[1];
    s[1] = s[2];
    s[2] = s[3];
    s[3] = s[4];
    s[4] = s[5];
    s[5] = s[6];
    s[6] = s[7];

    // save only 8 bytes convolve result
    _mm_storel_epi64((__m128i *)&output_ptr[0], ss[0]);

    output_ptr += out_pitch;
  }
}
#endif  // VPX_ARCH_X86_64

static void vpx_filter_block1d16_h4_ssse3(const uint8_t *src_ptr,
                                          ptrdiff_t src_stride,
                                          uint8_t *dst_ptr,
                                          ptrdiff_t dst_stride, uint32_t height,
                                          const int16_t *kernel) {
  // We will cast the kernel from 16-bit words to 8-bit words, and then extract
  // the middle four elements of the kernel into two registers in the form
  // ... k[3] k[2] k[3] k[2]
  // ... k[5] k[4] k[5] k[4]
  // Then we shuffle the source into
  // ... s[1] s[0] s[0] s[-1]
  // ... s[3] s[2] s[2] s[1]
  // Calling multiply and add gives us half of the sum. Calling add gives us
  // first half of the output. Repeat again to get the second half of the
  // output. Finally we shuffle again to combine the two outputs.

  __m128i kernel_reg;                         // Kernel
  __m128i kernel_reg_23, kernel_reg_45;       // Segments of the kernel used
  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding
  int h;

  __m128i src_reg, src_reg_shift_0, src_reg_shift_2;
  __m128i dst_first, dst_second;
  __m128i tmp_0, tmp_1;
  __m128i idx_shift_0 =
      _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
  __m128i idx_shift_2 =
      _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10);

  // Start one pixel before as we need tap/2 - 1 = 1 sample from the past
  src_ptr -= 1;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg = _mm_packs_epi16(kernel_reg, kernel_reg);
  kernel_reg_23 = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi16(0x0302u));
  kernel_reg_45 = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi16(0x0504u));

  for (h = height; h > 0; --h) {
    // Load the source
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    src_reg_shift_0 = _mm_shuffle_epi8(src_reg, idx_shift_0);
    src_reg_shift_2 = _mm_shuffle_epi8(src_reg, idx_shift_2);

    // Partial result for first half
    tmp_0 = _mm_maddubs_epi16(src_reg_shift_0, kernel_reg_23);
    tmp_1 = _mm_maddubs_epi16(src_reg_shift_2, kernel_reg_45);
    dst_first = _mm_adds_epi16(tmp_0, tmp_1);

    // Do again to get the second half of dst
    // Load the source
    src_reg = _mm_loadu_si128((const __m128i *)(src_ptr + 8));
    src_reg_shift_0 = _mm_shuffle_epi8(src_reg, idx_shift_0);
    src_reg_shift_2 = _mm_shuffle_epi8(src_reg, idx_shift_2);

    // Partial result for first half
    tmp_0 = _mm_maddubs_epi16(src_reg_shift_0, kernel_reg_23);
    tmp_1 = _mm_maddubs_epi16(src_reg_shift_2, kernel_reg_45);
    dst_second = _mm_adds_epi16(tmp_0, tmp_1);

    // Round each result
    dst_first = mm_round_epi16_sse2(&dst_first, &reg_32, 6);
    dst_second = mm_round_epi16_sse2(&dst_second, &reg_32, 6);

    // Finally combine to get the final dst
    dst_first = _mm_packus_epi16(dst_first, dst_second);
    _mm_store_si128((__m128i *)dst_ptr, dst_first);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void vpx_filter_block1d16_v4_ssse3(const uint8_t *src_ptr,
                                          ptrdiff_t src_stride,
                                          uint8_t *dst_ptr,
                                          ptrdiff_t dst_stride, uint32_t height,
                                          const int16_t *kernel) {
  // We will load two rows of pixels as 8-bit words, rearrange them into the
  // form
  // ... s[0,1] s[-1,1] s[0,0] s[-1,0]
  // ... s[0,9] s[-1,9] s[0,8] s[-1,8]
  // so that we can call multiply and add with the kernel to get 16-bit words of
  // the form
  // ... s[0,1]k[3]+s[-1,1]k[2] s[0,0]k[3]+s[-1,0]k[2]
  // Finally, we can add multiple rows together to get the desired output.

  // Register for source s[-1:3, :]
  __m128i src_reg_m1, src_reg_0, src_reg_1, src_reg_2, src_reg_3;
  // Interleaved rows of the source. lo is first half, hi second
  __m128i src_reg_m10_lo, src_reg_m10_hi, src_reg_01_lo, src_reg_01_hi;
  __m128i src_reg_12_lo, src_reg_12_hi, src_reg_23_lo, src_reg_23_hi;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used

  // Result after multiply and add
  __m128i res_reg_m10_lo, res_reg_01_lo, res_reg_12_lo, res_reg_23_lo;
  __m128i res_reg_m10_hi, res_reg_01_hi, res_reg_12_hi, res_reg_23_hi;
  __m128i res_reg_m1012, res_reg_0123;
  __m128i res_reg_m1012_lo, res_reg_0123_lo, res_reg_m1012_hi, res_reg_0123_hi;

  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding

  // We will compute the result two rows at a time
  const ptrdiff_t src_stride_unrolled = src_stride << 1;
  const ptrdiff_t dst_stride_unrolled = dst_stride << 1;
  int h;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg = _mm_packs_epi16(kernel_reg, kernel_reg);
  kernel_reg_23 = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi16(0x0302u));
  kernel_reg_45 = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi16(0x0504u));

  // First shuffle the data
  src_reg_m1 = _mm_loadu_si128((const __m128i *)src_ptr);
  src_reg_0 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride));
  src_reg_m10_lo = _mm_unpacklo_epi8(src_reg_m1, src_reg_0);
  src_reg_m10_hi = _mm_unpackhi_epi8(src_reg_m1, src_reg_0);

  // More shuffling
  src_reg_1 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 2));
  src_reg_01_lo = _mm_unpacklo_epi8(src_reg_0, src_reg_1);
  src_reg_01_hi = _mm_unpackhi_epi8(src_reg_0, src_reg_1);

  for (h = height; h > 1; h -= 2) {
    src_reg_2 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 3));

    src_reg_12_lo = _mm_unpacklo_epi8(src_reg_1, src_reg_2);
    src_reg_12_hi = _mm_unpackhi_epi8(src_reg_1, src_reg_2);

    src_reg_3 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 4));

    src_reg_23_lo = _mm_unpacklo_epi8(src_reg_2, src_reg_3);
    src_reg_23_hi = _mm_unpackhi_epi8(src_reg_2, src_reg_3);

    // Partial output from first half
    res_reg_m10_lo = _mm_maddubs_epi16(src_reg_m10_lo, kernel_reg_23);
    res_reg_01_lo = _mm_maddubs_epi16(src_reg_01_lo, kernel_reg_23);

    res_reg_12_lo = _mm_maddubs_epi16(src_reg_12_lo, kernel_reg_45);
    res_reg_23_lo = _mm_maddubs_epi16(src_reg_23_lo, kernel_reg_45);

    // Add to get first half of the results
    res_reg_m1012_lo = _mm_adds_epi16(res_reg_m10_lo, res_reg_12_lo);
    res_reg_0123_lo = _mm_adds_epi16(res_reg_01_lo, res_reg_23_lo);

    // Partial output for second half
    res_reg_m10_hi = _mm_maddubs_epi16(src_reg_m10_hi, kernel_reg_23);
    res_reg_01_hi = _mm_maddubs_epi16(src_reg_01_hi, kernel_reg_23);

    res_reg_12_hi = _mm_maddubs_epi16(src_reg_12_hi, kernel_reg_45);
    res_reg_23_hi = _mm_maddubs_epi16(src_reg_23_hi, kernel_reg_45);

    // Second half of the results
    res_reg_m1012_hi = _mm_adds_epi16(res_reg_m10_hi, res_reg_12_hi);
    res_reg_0123_hi = _mm_adds_epi16(res_reg_01_hi, res_reg_23_hi);

    // Round the words
    res_reg_m1012_lo = mm_round_epi16_sse2(&res_reg_m1012_lo, &reg_32, 6);
    res_reg_0123_lo = mm_round_epi16_sse2(&res_reg_0123_lo, &reg_32, 6);
    res_reg_m1012_hi = mm_round_epi16_sse2(&res_reg_m1012_hi, &reg_32, 6);
    res_reg_0123_hi = mm_round_epi16_sse2(&res_reg_0123_hi, &reg_32, 6);

    // Combine to get the result
    res_reg_m1012 = _mm_packus_epi16(res_reg_m1012_lo, res_reg_m1012_hi);
    res_reg_0123 = _mm_packus_epi16(res_reg_0123_lo, res_reg_0123_hi);

    _mm_store_si128((__m128i *)dst_ptr, res_reg_m1012);
    _mm_store_si128((__m128i *)(dst_ptr + dst_stride), res_reg_0123);

    // Update the source by two rows
    src_ptr += src_stride_unrolled;
    dst_ptr += dst_stride_unrolled;

    src_reg_m10_lo = src_reg_12_lo;
    src_reg_m10_hi = src_reg_12_hi;
    src_reg_01_lo = src_reg_23_lo;
    src_reg_01_hi = src_reg_23_hi;
    src_reg_1 = src_reg_3;
  }
}

static void vpx_filter_block1d8_h4_ssse3(const uint8_t *src_ptr,
                                         ptrdiff_t src_stride, uint8_t *dst_ptr,
                                         ptrdiff_t dst_stride, uint32_t height,
                                         const int16_t *kernel) {
  // We will cast the kernel from 16-bit words to 8-bit words, and then extract
  // the middle four elements of the kernel into two registers in the form
  // ... k[3] k[2] k[3] k[2]
  // ... k[5] k[4] k[5] k[4]
  // Then we shuffle the source into
  // ... s[1] s[0] s[0] s[-1]
  // ... s[3] s[2] s[2] s[1]
  // Calling multiply and add gives us half of the sum. Calling add gives us
  // first half of the output. Repeat again to get the second half of the
  // output. Finally we shuffle again to combine the two outputs.

  __m128i kernel_reg;                         // Kernel
  __m128i kernel_reg_23, kernel_reg_45;       // Segments of the kernel used
  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding
  int h;

  __m128i src_reg, src_reg_shift_0, src_reg_shift_2;
  __m128i dst_first;
  __m128i tmp_0, tmp_1;
  __m128i idx_shift_0 =
      _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
  __m128i idx_shift_2 =
      _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10);

  // Start one pixel before as we need tap/2 - 1 = 1 sample from the past
  src_ptr -= 1;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg = _mm_packs_epi16(kernel_reg, kernel_reg);
  kernel_reg_23 = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi16(0x0302u));
  kernel_reg_45 = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi16(0x0504u));

  for (h = height; h > 0; --h) {
    // Load the source
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    src_reg_shift_0 = _mm_shuffle_epi8(src_reg, idx_shift_0);
    src_reg_shift_2 = _mm_shuffle_epi8(src_reg, idx_shift_2);

    // Get the result
    tmp_0 = _mm_maddubs_epi16(src_reg_shift_0, kernel_reg_23);
    tmp_1 = _mm_maddubs_epi16(src_reg_shift_2, kernel_reg_45);
    dst_first = _mm_adds_epi16(tmp_0, tmp_1);

    // Round round result
    dst_first = mm_round_epi16_sse2(&dst_first, &reg_32, 6);

    // Pack to 8-bits
    dst_first = _mm_packus_epi16(dst_first, _mm_setzero_si128());
    _mm_storel_epi64((__m128i *)dst_ptr, dst_first);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void vpx_filter_block1d8_v4_ssse3(const uint8_t *src_ptr,
                                         ptrdiff_t src_stride, uint8_t *dst_ptr,
                                         ptrdiff_t dst_stride, uint32_t height,
                                         const int16_t *kernel) {
  // We will load two rows of pixels as 8-bit words, rearrange them into the
  // form
  // ... s[0,1] s[-1,1] s[0,0] s[-1,0]
  // so that we can call multiply and add with the kernel to get 16-bit words of
  // the form
  // ... s[0,1]k[3]+s[-1,1]k[2] s[0,0]k[3]+s[-1,0]k[2]
  // Finally, we can add multiple rows together to get the desired output.

  // Register for source s[-1:3, :]
  __m128i src_reg_m1, src_reg_0, src_reg_1, src_reg_2, src_reg_3;
  // Interleaved rows of the source. lo is first half, hi second
  __m128i src_reg_m10, src_reg_01;
  __m128i src_reg_12, src_reg_23;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used

  // Result after multiply and add
  __m128i res_reg_m10, res_reg_01, res_reg_12, res_reg_23;
  __m128i res_reg_m1012, res_reg_0123;

  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding

  // We will compute the result two rows at a time
  const ptrdiff_t src_stride_unrolled = src_stride << 1;
  const ptrdiff_t dst_stride_unrolled = dst_stride << 1;
  int h;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg = _mm_packs_epi16(kernel_reg, kernel_reg);
  kernel_reg_23 = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi16(0x0302u));
  kernel_reg_45 = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi16(0x0504u));

  // First shuffle the data
  src_reg_m1 = _mm_loadl_epi64((const __m128i *)src_ptr);
  src_reg_0 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride));
  src_reg_m10 = _mm_unpacklo_epi8(src_reg_m1, src_reg_0);

  // More shuffling
  src_reg_1 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 2));
  src_reg_01 = _mm_unpacklo_epi8(src_reg_0, src_reg_1);

  for (h = height; h > 1; h -= 2) {
    src_reg_2 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 3));

    src_reg_12 = _mm_unpacklo_epi8(src_reg_1, src_reg_2);

    src_reg_3 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 4));

    src_reg_23 = _mm_unpacklo_epi8(src_reg_2, src_reg_3);

    // Partial output
    res_reg_m10 = _mm_maddubs_epi16(src_reg_m10, kernel_reg_23);
    res_reg_01 = _mm_maddubs_epi16(src_reg_01, kernel_reg_23);

    res_reg_12 = _mm_maddubs_epi16(src_reg_12, kernel_reg_45);
    res_reg_23 = _mm_maddubs_epi16(src_reg_23, kernel_reg_45);

    // Add to get entire output
    res_reg_m1012 = _mm_adds_epi16(res_reg_m10, res_reg_12);
    res_reg_0123 = _mm_adds_epi16(res_reg_01, res_reg_23);

    // Round the words
    res_reg_m1012 = mm_round_epi16_sse2(&res_reg_m1012, &reg_32, 6);
    res_reg_0123 = mm_round_epi16_sse2(&res_reg_0123, &reg_32, 6);

    // Pack from 16-bit to 8-bit
    res_reg_m1012 = _mm_packus_epi16(res_reg_m1012, _mm_setzero_si128());
    res_reg_0123 = _mm_packus_epi16(res_reg_0123, _mm_setzero_si128());

    _mm_storel_epi64((__m128i *)dst_ptr, res_reg_m1012);
    _mm_storel_epi64((__m128i *)(dst_ptr + dst_stride), res_reg_0123);

    // Update the source by two rows
    src_ptr += src_stride_unrolled;
    dst_ptr += dst_stride_unrolled;

    src_reg_m10 = src_reg_12;
    src_reg_01 = src_reg_23;
    src_reg_1 = src_reg_3;
  }
}

static void vpx_filter_block1d4_h4_ssse3(const uint8_t *src_ptr,
                                         ptrdiff_t src_stride, uint8_t *dst_ptr,
                                         ptrdiff_t dst_stride, uint32_t height,
                                         const int16_t *kernel) {
  // We will cast the kernel from 16-bit words to 8-bit words, and then extract
  // the middle four elements of the kernel into a single register in the form
  // k[5:2] k[5:2] k[5:2] k[5:2]
  // Then we shuffle the source into
  // s[5:2] s[4:1] s[3:0] s[2:-1]
  // Calling multiply and add gives us half of the sum next to each other.
  // Calling horizontal add then gives us the output.

  __m128i kernel_reg;                         // Kernel
  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding
  int h;

  __m128i src_reg, src_reg_shuf;
  __m128i dst_first;
  __m128i shuf_idx =
      _mm_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6);

  // Start one pixel before as we need tap/2 - 1 = 1 sample from the past
  src_ptr -= 1;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg = _mm_packs_epi16(kernel_reg, kernel_reg);
  kernel_reg = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi32(0x05040302u));

  for (h = height; h > 0; --h) {
    // Load the source
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    src_reg_shuf = _mm_shuffle_epi8(src_reg, shuf_idx);

    // Get the result
    dst_first = _mm_maddubs_epi16(src_reg_shuf, kernel_reg);
    dst_first = _mm_hadds_epi16(dst_first, _mm_setzero_si128());

    // Round result
    dst_first = mm_round_epi16_sse2(&dst_first, &reg_32, 6);

    // Pack to 8-bits
    dst_first = _mm_packus_epi16(dst_first, _mm_setzero_si128());
    *((int *)(dst_ptr)) = _mm_cvtsi128_si32(dst_first);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void vpx_filter_block1d4_v4_ssse3(const uint8_t *src_ptr,
                                         ptrdiff_t src_stride, uint8_t *dst_ptr,
                                         ptrdiff_t dst_stride, uint32_t height,
                                         const int16_t *kernel) {
  // We will load two rows of pixels as 8-bit words, rearrange them into the
  // form
  // ... s[2,0] s[1,0] s[0,0] s[-1,0]
  // so that we can call multiply and add with the kernel partial output. Then
  // we can call horizontal add to get the output.
  // Finally, we can add multiple rows together to get the desired output.
  // This is done two rows at a time

  // Register for source s[-1:3, :]
  __m128i src_reg_m1, src_reg_0, src_reg_1, src_reg_2, src_reg_3;
  // Interleaved rows of the source.
  __m128i src_reg_m10, src_reg_01;
  __m128i src_reg_12, src_reg_23;
  __m128i src_reg_m1001, src_reg_1223;
  __m128i src_reg_m1012_1023_lo, src_reg_m1012_1023_hi;

  __m128i kernel_reg;  // Kernel

  // Result after multiply and add
  __m128i reg_0, reg_1;

  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding

  // We will compute the result two rows at a time
  const ptrdiff_t src_stride_unrolled = src_stride << 1;
  const ptrdiff_t dst_stride_unrolled = dst_stride << 1;
  int h;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg = _mm_packs_epi16(kernel_reg, kernel_reg);
  kernel_reg = _mm_shuffle_epi8(kernel_reg, _mm_set1_epi32(0x05040302u));

  // First shuffle the data
  src_reg_m1 = _mm_loadl_epi64((const __m128i *)src_ptr);
  src_reg_0 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride));
  src_reg_m10 = _mm_unpacklo_epi32(src_reg_m1, src_reg_0);

  // More shuffling
  src_reg_1 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 2));
  src_reg_01 = _mm_unpacklo_epi32(src_reg_0, src_reg_1);

  // Put three rows next to each other
  src_reg_m1001 = _mm_unpacklo_epi8(src_reg_m10, src_reg_01);

  for (h = height; h > 1; h -= 2) {
    src_reg_2 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 3));
    src_reg_12 = _mm_unpacklo_epi32(src_reg_1, src_reg_2);

    src_reg_3 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 4));
    src_reg_23 = _mm_unpacklo_epi32(src_reg_2, src_reg_3);

    // Put three rows next to each other
    src_reg_1223 = _mm_unpacklo_epi8(src_reg_12, src_reg_23);

    // Put all four rows next to each other
    src_reg_m1012_1023_lo = _mm_unpacklo_epi16(src_reg_m1001, src_reg_1223);
    src_reg_m1012_1023_hi = _mm_unpackhi_epi16(src_reg_m1001, src_reg_1223);

    // Get the results
    reg_0 = _mm_maddubs_epi16(src_reg_m1012_1023_lo, kernel_reg);
    reg_1 = _mm_maddubs_epi16(src_reg_m1012_1023_hi, kernel_reg);
    reg_0 = _mm_hadds_epi16(reg_0, _mm_setzero_si128());
    reg_1 = _mm_hadds_epi16(reg_1, _mm_setzero_si128());

    // Round the words
    reg_0 = mm_round_epi16_sse2(&reg_0, &reg_32, 6);
    reg_1 = mm_round_epi16_sse2(&reg_1, &reg_32, 6);

    // Pack from 16-bit to 8-bit and put them in the right order
    reg_0 = _mm_packus_epi16(reg_0, reg_0);
    reg_1 = _mm_packus_epi16(reg_1, reg_1);

    // Save the result
    *((int *)(dst_ptr)) = _mm_cvtsi128_si32(reg_0);
    *((int *)(dst_ptr + dst_stride)) = _mm_cvtsi128_si32(reg_1);

    // Update the source by two rows
    src_ptr += src_stride_unrolled;
    dst_ptr += dst_stride_unrolled;

    src_reg_m1001 = src_reg_1223;
    src_reg_1 = src_reg_3;
  }
}

// From vpx_dsp/x86/vpx_subpixel_8t_ssse3.asm
filter8_1dfunction vpx_filter_block1d16_v8_ssse3;
filter8_1dfunction vpx_filter_block1d16_h8_ssse3;
filter8_1dfunction vpx_filter_block1d4_v8_ssse3;
filter8_1dfunction vpx_filter_block1d16_v8_avg_ssse3;
filter8_1dfunction vpx_filter_block1d16_h8_avg_ssse3;
filter8_1dfunction vpx_filter_block1d8_v8_avg_ssse3;
filter8_1dfunction vpx_filter_block1d8_h8_avg_ssse3;
filter8_1dfunction vpx_filter_block1d4_v8_avg_ssse3;
filter8_1dfunction vpx_filter_block1d4_h8_avg_ssse3;

// Use the [vh]8 version because there is no [vh]4 implementation.
#define vpx_filter_block1d16_v4_avg_ssse3 vpx_filter_block1d16_v8_avg_ssse3
#define vpx_filter_block1d16_h4_avg_ssse3 vpx_filter_block1d16_h8_avg_ssse3
#define vpx_filter_block1d8_v4_avg_ssse3 vpx_filter_block1d8_v8_avg_ssse3
#define vpx_filter_block1d8_h4_avg_ssse3 vpx_filter_block1d8_h8_avg_ssse3
#define vpx_filter_block1d4_v4_avg_ssse3 vpx_filter_block1d4_v8_avg_ssse3
#define vpx_filter_block1d4_h4_avg_ssse3 vpx_filter_block1d4_h8_avg_ssse3

// From vpx_dsp/x86/vpx_subpixel_bilinear_ssse3.asm
filter8_1dfunction vpx_filter_block1d16_v2_ssse3;
filter8_1dfunction vpx_filter_block1d16_h2_ssse3;
filter8_1dfunction vpx_filter_block1d8_v2_ssse3;
filter8_1dfunction vpx_filter_block1d8_h2_ssse3;
filter8_1dfunction vpx_filter_block1d4_v2_ssse3;
filter8_1dfunction vpx_filter_block1d4_h2_ssse3;
filter8_1dfunction vpx_filter_block1d16_v2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d16_h2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d8_v2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d8_h2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d4_v2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d4_h2_avg_ssse3;

// void vpx_convolve8_horiz_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                                uint8_t *dst, ptrdiff_t dst_stride,
//                                const InterpKernel *filter, int x0_q4,
//                                int32_t x_step_q4, int y0_q4, int y_step_q4,
//                                int w, int h);
// void vpx_convolve8_vert_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                               uint8_t *dst, ptrdiff_t dst_stride,
//                               const InterpKernel *filter, int x0_q4,
//                               int32_t x_step_q4, int y0_q4, int y_step_q4,
//                               int w, int h);
// void vpx_convolve8_avg_horiz_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                                    uint8_t *dst, ptrdiff_t dst_stride,
//                                    const InterpKernel *filter, int x0_q4,
//                                    int32_t x_step_q4, int y0_q4,
//                                    int y_step_q4, int w, int h);
// void vpx_convolve8_avg_vert_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                                   uint8_t *dst, ptrdiff_t dst_stride,
//                                   const InterpKernel *filter, int x0_q4,
//                                   int32_t x_step_q4, int y0_q4,
//                                   int y_step_q4, int w, int h);
FUN_CONV_1D(horiz, x0_q4, x_step_q4, h, src, , ssse3, 0)
FUN_CONV_1D(vert, y0_q4, y_step_q4, v, src - src_stride * (num_taps / 2 - 1), ,
            ssse3, 0)
FUN_CONV_1D(avg_horiz, x0_q4, x_step_q4, h, src, avg_, ssse3, 1)
FUN_CONV_1D(avg_vert, y0_q4, y_step_q4, v,
            src - src_stride * (num_taps / 2 - 1), avg_, ssse3, 1)

static void filter_horiz_w8_ssse3(const uint8_t *const src,
                                  const ptrdiff_t src_stride,
                                  uint8_t *const dst,
                                  const int16_t *const x_filter) {
  __m128i s[8], ss[4], temp;

  load_8bit_8x8(src, src_stride, s);
  // 00 01 10 11 20 21 30 31  40 41 50 51 60 61 70 71
  // 02 03 12 13 22 23 32 33  42 43 52 53 62 63 72 73
  // 04 05 14 15 24 25 34 35  44 45 54 55 64 65 74 75
  // 06 07 16 17 26 27 36 37  46 47 56 57 66 67 76 77
  transpose_16bit_4x8(s, ss);
  temp = shuffle_filter_convolve8_8_ssse3(ss, x_filter);
  // shrink to 8 bit each 16 bits
  temp = _mm_packus_epi16(temp, temp);
  // save only 8 bytes convolve result
  _mm_storel_epi64((__m128i *)dst, temp);
}

static void transpose8x8_to_dst(const uint8_t *const src,
                                const ptrdiff_t src_stride, uint8_t *const dst,
                                const ptrdiff_t dst_stride) {
  __m128i s[8];

  load_8bit_8x8(src, src_stride, s);
  transpose_8bit_8x8(s, s);
  store_8bit_8x8(s, dst, dst_stride);
}

static void scaledconvolve_horiz_w8(const uint8_t *src,
                                    const ptrdiff_t src_stride, uint8_t *dst,
                                    const ptrdiff_t dst_stride,
                                    const InterpKernel *const x_filters,
                                    const int x0_q4, const int x_step_q4,
                                    const int w, const int h) {
  DECLARE_ALIGNED(16, uint8_t, temp[8 * 8]);
  int x, y, z;
  src -= SUBPEL_TAPS / 2 - 1;

  // This function processes 8x8 areas. The intermediate height is not always
  // a multiple of 8, so force it to be a multiple of 8 here.
  y = h + (8 - (h & 0x7));

  do {
    int x_q4 = x0_q4;
    for (x = 0; x < w; x += 8) {
      // process 8 src_x steps
      for (z = 0; z < 8; ++z) {
        const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
        const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
        if (x_q4 & SUBPEL_MASK) {
          filter_horiz_w8_ssse3(src_x, src_stride, temp + (z * 8), x_filter);
        } else {
          int i;
          for (i = 0; i < 8; ++i) {
            temp[z * 8 + i] = src_x[i * src_stride + 3];
          }
        }
        x_q4 += x_step_q4;
      }

      // transpose the 8x8 filters values back to dst
      transpose8x8_to_dst(temp, 8, dst + x, dst_stride);
    }

    src += src_stride * 8;
    dst += dst_stride * 8;
  } while (y -= 8);
}

static void filter_horiz_w4_ssse3(const uint8_t *const src,
                                  const ptrdiff_t src_stride,
                                  uint8_t *const dst,
                                  const int16_t *const filter) {
  __m128i s[4], ss[2];
  __m128i temp;

  load_8bit_8x4(src, src_stride, s);
  transpose_16bit_4x4(s, ss);
  // 00 01 10 11 20 21 30 31
  s[0] = ss[0];
  // 02 03 12 13 22 23 32 33
  s[1] = _mm_srli_si128(ss[0], 8);
  // 04 05 14 15 24 25 34 35
  s[2] = ss[1];
  // 06 07 16 17 26 27 36 37
  s[3] = _mm_srli_si128(ss[1], 8);

  temp = shuffle_filter_convolve8_8_ssse3(s, filter);
  // shrink to 8 bit each 16 bits
  temp = _mm_packus_epi16(temp, temp);
  // save only 4 bytes
  *(int *)dst = _mm_cvtsi128_si32(temp);
}

static void transpose4x4_to_dst(const uint8_t *const src,
                                const ptrdiff_t src_stride, uint8_t *const dst,
                                const ptrdiff_t dst_stride) {
  __m128i s[4];

  load_8bit_4x4(src, src_stride, s);
  s[0] = transpose_8bit_4x4(s);
  s[1] = _mm_srli_si128(s[0], 4);
  s[2] = _mm_srli_si128(s[0], 8);
  s[3] = _mm_srli_si128(s[0], 12);
  store_8bit_4x4(s, dst, dst_stride);
}

static void scaledconvolve_horiz_w4(const uint8_t *src,
                                    const ptrdiff_t src_stride, uint8_t *dst,
                                    const ptrdiff_t dst_stride,
                                    const InterpKernel *const x_filters,
                                    const int x0_q4, const int x_step_q4,
                                    const int w, const int h) {
  DECLARE_ALIGNED(16, uint8_t, temp[4 * 4]);
  int x, y, z;
  src -= SUBPEL_TAPS / 2 - 1;

  for (y = 0; y < h; y += 4) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; x += 4) {
      // process 4 src_x steps
      for (z = 0; z < 4; ++z) {
        const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
        const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
        if (x_q4 & SUBPEL_MASK) {
          filter_horiz_w4_ssse3(src_x, src_stride, temp + (z * 4), x_filter);
        } else {
          int i;
          for (i = 0; i < 4; ++i) {
            temp[z * 4 + i] = src_x[i * src_stride + 3];
          }
        }
        x_q4 += x_step_q4;
      }

      // transpose the 4x4 filters values back to dst
      transpose4x4_to_dst(temp, 4, dst + x, dst_stride);
    }

    src += src_stride * 4;
    dst += dst_stride * 4;
  }
}

static __m128i filter_vert_kernel(const __m128i *const s,
                                  const int16_t *const filter) {
  __m128i ss[4];
  __m128i temp;

  // 00 10 01 11 02 12 03 13
  ss[0] = _mm_unpacklo_epi8(s[0], s[1]);
  // 20 30 21 31 22 32 23 33
  ss[1] = _mm_unpacklo_epi8(s[2], s[3]);
  // 40 50 41 51 42 52 43 53
  ss[2] = _mm_unpacklo_epi8(s[4], s[5]);
  // 60 70 61 71 62 72 63 73
  ss[3] = _mm_unpacklo_epi8(s[6], s[7]);

  temp = shuffle_filter_convolve8_8_ssse3(ss, filter);
  // shrink to 8 bit each 16 bits
  return _mm_packus_epi16(temp, temp);
}

static void filter_vert_w4_ssse3(const uint8_t *const src,
                                 const ptrdiff_t src_stride, uint8_t *const dst,
                                 const int16_t *const filter) {
  __m128i s[8];
  __m128i temp;

  load_8bit_4x8(src, src_stride, s);
  temp = filter_vert_kernel(s, filter);
  // save only 4 bytes
  *(int *)dst = _mm_cvtsi128_si32(temp);
}

static void scaledconvolve_vert_w4(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *const dst,
    const ptrdiff_t dst_stride, const InterpKernel *const y_filters,
    const int y0_q4, const int y_step_q4, const int w, const int h) {
  int y;
  int y_q4 = y0_q4;

  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  for (y = 0; y < h; ++y) {
    const unsigned char *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];

    if (y_q4 & SUBPEL_MASK) {
      filter_vert_w4_ssse3(src_y, src_stride, &dst[y * dst_stride], y_filter);
    } else {
      memcpy(&dst[y * dst_stride], &src_y[3 * src_stride], w);
    }

    y_q4 += y_step_q4;
  }
}

static void filter_vert_w8_ssse3(const uint8_t *const src,
                                 const ptrdiff_t src_stride, uint8_t *const dst,
                                 const int16_t *const filter) {
  __m128i s[8], temp;

  load_8bit_8x8(src, src_stride, s);
  temp = filter_vert_kernel(s, filter);
  // save only 8 bytes convolve result
  _mm_storel_epi64((__m128i *)dst, temp);
}

static void scaledconvolve_vert_w8(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *const dst,
    const ptrdiff_t dst_stride, const InterpKernel *const y_filters,
    const int y0_q4, const int y_step_q4, const int w, const int h) {
  int y;
  int y_q4 = y0_q4;

  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  for (y = 0; y < h; ++y) {
    const unsigned char *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
    if (y_q4 & SUBPEL_MASK) {
      filter_vert_w8_ssse3(src_y, src_stride, &dst[y * dst_stride], y_filter);
    } else {
      memcpy(&dst[y * dst_stride], &src_y[3 * src_stride], w);
    }
    y_q4 += y_step_q4;
  }
}

static void filter_vert_w16_ssse3(const uint8_t *src,
                                  const ptrdiff_t src_stride,
                                  uint8_t *const dst,
                                  const int16_t *const filter, const int w) {
  int i;
  __m128i f[4];
  shuffle_filter_ssse3(filter, f);

  for (i = 0; i < w; i += 16) {
    __m128i s[8], s_lo[4], s_hi[4], temp_lo, temp_hi;

    loadu_8bit_16x8(src, src_stride, s);

    // merge the result together
    s_lo[0] = _mm_unpacklo_epi8(s[0], s[1]);
    s_hi[0] = _mm_unpackhi_epi8(s[0], s[1]);
    s_lo[1] = _mm_unpacklo_epi8(s[2], s[3]);
    s_hi[1] = _mm_unpackhi_epi8(s[2], s[3]);
    s_lo[2] = _mm_unpacklo_epi8(s[4], s[5]);
    s_hi[2] = _mm_unpackhi_epi8(s[4], s[5]);
    s_lo[3] = _mm_unpacklo_epi8(s[6], s[7]);
    s_hi[3] = _mm_unpackhi_epi8(s[6], s[7]);
    temp_lo = convolve8_8_ssse3(s_lo, f);
    temp_hi = convolve8_8_ssse3(s_hi, f);

    // shrink to 8 bit each 16 bits, the first lane contain the first convolve
    // result and the second lane contain the second convolve result
    temp_hi = _mm_packus_epi16(temp_lo, temp_hi);
    src += 16;
    // save 16 bytes convolve result
    _mm_store_si128((__m128i *)&dst[i], temp_hi);
  }
}

static void scaledconvolve_vert_w16(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *const dst,
    const ptrdiff_t dst_stride, const InterpKernel *const y_filters,
    const int y0_q4, const int y_step_q4, const int w, const int h) {
  int y;
  int y_q4 = y0_q4;

  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  for (y = 0; y < h; ++y) {
    const unsigned char *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
    if (y_q4 & SUBPEL_MASK) {
      filter_vert_w16_ssse3(src_y, src_stride, &dst[y * dst_stride], y_filter,
                            w);
    } else {
      memcpy(&dst[y * dst_stride], &src_y[3 * src_stride], w);
    }
    y_q4 += y_step_q4;
  }
}

void vpx_scaled_2d_ssse3(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                         ptrdiff_t dst_stride, const InterpKernel *filter,
                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                         int w, int h) {
  // Note: Fixed size intermediate buffer, temp, places limits on parameters.
  // 2d filtering proceeds in 2 steps:
  //   (1) Interpolate horizontally into an intermediate buffer, temp.
  //   (2) Interpolate temp vertically to derive the sub-pixel result.
  // Deriving the maximum number of rows in the temp buffer (135):
  // --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
  // --Largest block size is 64x64 pixels.
  // --64 rows in the downscaled frame span a distance of (64 - 1) * 32 in the
  //   original frame (in 1/16th pixel units).
  // --Must round-up because block may be located at sub-pixel position.
  // --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
  // --((64 - 1) * 32 + 15) >> 4 + 8 = 135.
  // --Require an additional 8 rows for the horiz_w8 transpose tail.
  // When calling in frame scaling function, the smallest scaling factor is x1/4
  // ==> y_step_q4 = 64. Since w and h are at most 16, the temp buffer is still
  // big enough.
  DECLARE_ALIGNED(16, uint8_t, temp[(135 + 8) * 64]);
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

  assert(w <= 64);
  assert(h <= 64);
  assert(y_step_q4 <= 32 || (y_step_q4 <= 64 && h <= 32));
  assert(x_step_q4 <= 64);

  if (w >= 8) {
    scaledconvolve_horiz_w8(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                            src_stride, temp, 64, filter, x0_q4, x_step_q4, w,
                            intermediate_height);
  } else {
    scaledconvolve_horiz_w4(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                            src_stride, temp, 64, filter, x0_q4, x_step_q4, w,
                            intermediate_height);
  }

  if (w >= 16) {
    scaledconvolve_vert_w16(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                            dst_stride, filter, y0_q4, y_step_q4, w, h);
  } else if (w == 8) {
    scaledconvolve_vert_w8(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                           dst_stride, filter, y0_q4, y_step_q4, w, h);
  } else {
    scaledconvolve_vert_w4(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                           dst_stride, filter, y0_q4, y_step_q4, w, h);
  }
}

// void vpx_convolve8_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                          uint8_t *dst, ptrdiff_t dst_stride,
//                          const InterpKernel *filter, int x0_q4,
//                          int32_t x_step_q4, int y0_q4, int y_step_q4,
//                          int w, int h);
// void vpx_convolve8_avg_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                              uint8_t *dst, ptrdiff_t dst_stride,
//                              const InterpKernel *filter, int x0_q4,
//                              int32_t x_step_q4, int y0_q4, int y_step_q4,
//                              int w, int h);
FUN_CONV_2D(, ssse3, 0)
FUN_CONV_2D(avg_, ssse3, 1)
