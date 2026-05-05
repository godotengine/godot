/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_pred_common.h"
#include "vpx_mem/vpx_mem.h"

namespace {

using libvpx_test::ACMRandom;

const int count_test_block = 100000;

using IntraPredFunc = void (*)(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left);

struct IntraPredParam {
  IntraPredParam(IntraPredFunc pred = nullptr, IntraPredFunc ref = nullptr,
                 int block_size_value = 0, int bit_depth_value = 0)
      : pred_fn(pred), ref_fn(ref), block_size(block_size_value),
        bit_depth(bit_depth_value) {}

  IntraPredFunc pred_fn;
  IntraPredFunc ref_fn;
  int block_size;
  int bit_depth;
};

template <typename Pixel, typename PredParam>
class IntraPredTest : public ::testing::TestWithParam<PredParam> {
 public:
  void RunTest(Pixel *left_col, Pixel *above_data, Pixel *dst, Pixel *ref_dst) {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int block_size = params_.block_size;
    above_row_ = above_data + 16;
    left_col_ = left_col;
    dst_ = dst;
    ref_dst_ = ref_dst;
    int error_count = 0;
    for (int i = 0; i < count_test_block; ++i) {
      // TODO(webm:1797): Some of the optimised predictor implementations rely
      // on the trailing half of the above_row_ being a copy of the final
      // element, however relying on this in some cases can cause the MD5 tests
      // to fail. We have fixed all of these cases for Neon, so fill the whole
      // of above_row_ randomly.
#if HAVE_NEON
      // Fill edges with random data, try first with saturated values.
      for (int x = -1; x < 2 * block_size; x++) {
        if (i == 0) {
          above_row_[x] = mask_;
        } else {
          above_row_[x] = rnd.Rand16() & mask_;
        }
      }
#else
      // Fill edges with random data, try first with saturated values.
      for (int x = -1; x < block_size; x++) {
        if (i == 0) {
          above_row_[x] = mask_;
        } else {
          above_row_[x] = rnd.Rand16() & mask_;
        }
      }
      for (int x = block_size; x < 2 * block_size; x++) {
        above_row_[x] = above_row_[block_size - 1];
      }
#endif
      for (int y = 0; y < block_size; y++) {
        if (i == 0) {
          left_col_[y] = mask_;
        } else {
          left_col_[y] = rnd.Rand16() & mask_;
        }
      }
      Predict();
      CheckPrediction(i, &error_count);
    }
    ASSERT_EQ(0, error_count);
  }

 protected:
  void SetUp() override {
    params_ = this->GetParam();
    stride_ = params_.block_size * 3;
    mask_ = (1 << params_.bit_depth) - 1;
  }

  void Predict();

  void CheckPrediction(int test_case_number, int *error_count) const {
    // For each pixel ensure that the calculated value is the same as reference.
    const int block_size = params_.block_size;
    for (int y = 0; y < block_size; y++) {
      for (int x = 0; x < block_size; x++) {
        *error_count += ref_dst_[x + y * stride_] != dst_[x + y * stride_];
        if (*error_count == 1) {
          ASSERT_EQ(ref_dst_[x + y * stride_], dst_[x + y * stride_])
              << " Failed on Test Case Number " << test_case_number;
        }
      }
    }
  }

  Pixel *above_row_;
  Pixel *left_col_;
  Pixel *dst_;
  Pixel *ref_dst_;
  ptrdiff_t stride_;
  int mask_;

  PredParam params_;
};

template <>
void IntraPredTest<uint8_t, IntraPredParam>::Predict() {
  params_.ref_fn(ref_dst_, stride_, above_row_, left_col_);
  ASM_REGISTER_STATE_CHECK(
      params_.pred_fn(dst_, stride_, above_row_, left_col_));
}

using VP9IntraPredTest = IntraPredTest<uint8_t, IntraPredParam>;

TEST_P(VP9IntraPredTest, IntraPredTests) {
  // max block size is 32
  DECLARE_ALIGNED(16, uint8_t, left_col[2 * 32]);
  DECLARE_ALIGNED(16, uint8_t, above_data[2 * 32 + 32]);
  DECLARE_ALIGNED(16, uint8_t, dst[3 * 32 * 32]);
  DECLARE_ALIGNED(16, uint8_t, ref_dst[3 * 32 * 32]);
  RunTest(left_col, above_data, dst, ref_dst);
}

// Instantiate a token test to avoid -Wuninitialized warnings when none of the
// other tests are enabled.
INSTANTIATE_TEST_SUITE_P(
    C, VP9IntraPredTest,
    ::testing::Values(IntraPredParam(&vpx_d45_predictor_4x4_c,
                                     &vpx_d45_predictor_4x4_c, 4, 8)));
#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, VP9IntraPredTest,
    ::testing::Values(
        IntraPredParam(&vpx_d45_predictor_4x4_sse2, &vpx_d45_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_d45_predictor_8x8_sse2, &vpx_d45_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_d207_predictor_4x4_sse2, &vpx_d207_predictor_4x4_c,
                       4, 8),
        IntraPredParam(&vpx_dc_128_predictor_4x4_sse2,
                       &vpx_dc_128_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_128_predictor_8x8_sse2,
                       &vpx_dc_128_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_128_predictor_16x16_sse2,
                       &vpx_dc_128_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_128_predictor_32x32_sse2,
                       &vpx_dc_128_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_dc_left_predictor_4x4_sse2,
                       &vpx_dc_left_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_left_predictor_8x8_sse2,
                       &vpx_dc_left_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_left_predictor_16x16_sse2,
                       &vpx_dc_left_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_left_predictor_32x32_sse2,
                       &vpx_dc_left_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_dc_predictor_4x4_sse2, &vpx_dc_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_dc_predictor_8x8_sse2, &vpx_dc_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_dc_predictor_16x16_sse2, &vpx_dc_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_dc_predictor_32x32_sse2, &vpx_dc_predictor_32x32_c,
                       32, 8),
        IntraPredParam(&vpx_dc_top_predictor_4x4_sse2,
                       &vpx_dc_top_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_top_predictor_8x8_sse2,
                       &vpx_dc_top_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_top_predictor_16x16_sse2,
                       &vpx_dc_top_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_top_predictor_32x32_sse2,
                       &vpx_dc_top_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_h_predictor_4x4_sse2, &vpx_h_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_h_predictor_8x8_sse2, &vpx_h_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_h_predictor_16x16_sse2, &vpx_h_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_h_predictor_32x32_sse2, &vpx_h_predictor_32x32_c,
                       32, 8),
        IntraPredParam(&vpx_tm_predictor_4x4_sse2, &vpx_tm_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_tm_predictor_8x8_sse2, &vpx_tm_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_tm_predictor_16x16_sse2, &vpx_tm_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_tm_predictor_32x32_sse2, &vpx_tm_predictor_32x32_c,
                       32, 8),
        IntraPredParam(&vpx_v_predictor_4x4_sse2, &vpx_v_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_v_predictor_8x8_sse2, &vpx_v_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_v_predictor_16x16_sse2, &vpx_v_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_v_predictor_32x32_sse2, &vpx_v_predictor_32x32_c,
                       32, 8)));
#endif  // HAVE_SSE2

#if HAVE_SSSE3
INSTANTIATE_TEST_SUITE_P(
    SSSE3, VP9IntraPredTest,
    ::testing::Values(IntraPredParam(&vpx_d45_predictor_16x16_ssse3,
                                     &vpx_d45_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_d45_predictor_32x32_ssse3,
                                     &vpx_d45_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_d63_predictor_4x4_ssse3,
                                     &vpx_d63_predictor_4x4_c, 4, 8),
                      IntraPredParam(&vpx_d63_predictor_8x8_ssse3,
                                     &vpx_d63_predictor_8x8_c, 8, 8),
                      IntraPredParam(&vpx_d63_predictor_16x16_ssse3,
                                     &vpx_d63_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_d63_predictor_32x32_ssse3,
                                     &vpx_d63_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_d153_predictor_4x4_ssse3,
                                     &vpx_d153_predictor_4x4_c, 4, 8),
                      IntraPredParam(&vpx_d153_predictor_8x8_ssse3,
                                     &vpx_d153_predictor_8x8_c, 8, 8),
                      IntraPredParam(&vpx_d153_predictor_16x16_ssse3,
                                     &vpx_d153_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_d153_predictor_32x32_ssse3,
                                     &vpx_d153_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_d207_predictor_8x8_ssse3,
                                     &vpx_d207_predictor_8x8_c, 8, 8),
                      IntraPredParam(&vpx_d207_predictor_16x16_ssse3,
                                     &vpx_d207_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_d207_predictor_32x32_ssse3,
                                     &vpx_d207_predictor_32x32_c, 32, 8)));
#endif  // HAVE_SSSE3

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, VP9IntraPredTest,
    ::testing::Values(
        IntraPredParam(&vpx_d45_predictor_4x4_neon, &vpx_d45_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_d45_predictor_8x8_neon, &vpx_d45_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_d45_predictor_16x16_neon,
                       &vpx_d45_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_d45_predictor_32x32_neon,
                       &vpx_d45_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_d63_predictor_4x4_neon, &vpx_d63_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_d63_predictor_8x8_neon, &vpx_d63_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_d63_predictor_16x16_neon,
                       &vpx_d63_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_d63_predictor_32x32_neon,
                       &vpx_d63_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_d117_predictor_4x4_neon, &vpx_d117_predictor_4x4_c,
                       4, 8),
        IntraPredParam(&vpx_d117_predictor_8x8_neon, &vpx_d117_predictor_8x8_c,
                       8, 8),
        IntraPredParam(&vpx_d117_predictor_16x16_neon,
                       &vpx_d117_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_d117_predictor_32x32_neon,
                       &vpx_d117_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_d135_predictor_4x4_neon, &vpx_d135_predictor_4x4_c,
                       4, 8),
        IntraPredParam(&vpx_d135_predictor_8x8_neon, &vpx_d135_predictor_8x8_c,
                       8, 8),
        IntraPredParam(&vpx_d135_predictor_16x16_neon,
                       &vpx_d135_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_d135_predictor_32x32_neon,
                       &vpx_d135_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_d153_predictor_4x4_neon, &vpx_d153_predictor_4x4_c,
                       4, 8),
        IntraPredParam(&vpx_d153_predictor_8x8_neon, &vpx_d153_predictor_8x8_c,
                       8, 8),
        IntraPredParam(&vpx_d153_predictor_16x16_neon,
                       &vpx_d153_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_d153_predictor_32x32_neon,
                       &vpx_d153_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_d207_predictor_4x4_neon, &vpx_d207_predictor_4x4_c,
                       4, 8),
        IntraPredParam(&vpx_d207_predictor_8x8_neon, &vpx_d207_predictor_8x8_c,
                       8, 8),
        IntraPredParam(&vpx_d207_predictor_16x16_neon,
                       &vpx_d207_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_d207_predictor_32x32_neon,
                       &vpx_d207_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_dc_128_predictor_4x4_neon,
                       &vpx_dc_128_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_128_predictor_8x8_neon,
                       &vpx_dc_128_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_128_predictor_16x16_neon,
                       &vpx_dc_128_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_128_predictor_32x32_neon,
                       &vpx_dc_128_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_dc_left_predictor_4x4_neon,
                       &vpx_dc_left_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_left_predictor_8x8_neon,
                       &vpx_dc_left_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_left_predictor_16x16_neon,
                       &vpx_dc_left_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_left_predictor_32x32_neon,
                       &vpx_dc_left_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_dc_predictor_4x4_neon, &vpx_dc_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_dc_predictor_8x8_neon, &vpx_dc_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_dc_predictor_16x16_neon, &vpx_dc_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_dc_predictor_32x32_neon, &vpx_dc_predictor_32x32_c,
                       32, 8),
        IntraPredParam(&vpx_dc_top_predictor_4x4_neon,
                       &vpx_dc_top_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_top_predictor_8x8_neon,
                       &vpx_dc_top_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_top_predictor_16x16_neon,
                       &vpx_dc_top_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_top_predictor_32x32_neon,
                       &vpx_dc_top_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_h_predictor_4x4_neon, &vpx_h_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_h_predictor_8x8_neon, &vpx_h_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_h_predictor_16x16_neon, &vpx_h_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_h_predictor_32x32_neon, &vpx_h_predictor_32x32_c,
                       32, 8),
        IntraPredParam(&vpx_tm_predictor_4x4_neon, &vpx_tm_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_tm_predictor_8x8_neon, &vpx_tm_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_tm_predictor_16x16_neon, &vpx_tm_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_tm_predictor_32x32_neon, &vpx_tm_predictor_32x32_c,
                       32, 8),
        IntraPredParam(&vpx_v_predictor_4x4_neon, &vpx_v_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_v_predictor_8x8_neon, &vpx_v_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_v_predictor_16x16_neon, &vpx_v_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_v_predictor_32x32_neon, &vpx_v_predictor_32x32_c,
                       32, 8)));
#endif  // HAVE_NEON

#if HAVE_DSPR2
INSTANTIATE_TEST_SUITE_P(
    DSPR2, VP9IntraPredTest,
    ::testing::Values(IntraPredParam(&vpx_dc_predictor_4x4_dspr2,
                                     &vpx_dc_predictor_4x4_c, 4, 8),
                      IntraPredParam(&vpx_dc_predictor_8x8_dspr2,
                                     &vpx_dc_predictor_8x8_c, 8, 8),
                      IntraPredParam(&vpx_dc_predictor_16x16_dspr2,
                                     &vpx_dc_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_h_predictor_4x4_dspr2,
                                     &vpx_h_predictor_4x4_c, 4, 8),
                      IntraPredParam(&vpx_h_predictor_8x8_dspr2,
                                     &vpx_h_predictor_8x8_c, 8, 8),
                      IntraPredParam(&vpx_h_predictor_16x16_dspr2,
                                     &vpx_h_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_tm_predictor_4x4_dspr2,
                                     &vpx_tm_predictor_4x4_c, 4, 8),
                      IntraPredParam(&vpx_tm_predictor_8x8_dspr2,
                                     &vpx_tm_predictor_8x8_c, 8, 8)));
#endif  // HAVE_DSPR2

#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(
    MSA, VP9IntraPredTest,
    ::testing::Values(
        IntraPredParam(&vpx_dc_128_predictor_4x4_msa,
                       &vpx_dc_128_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_128_predictor_8x8_msa,
                       &vpx_dc_128_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_128_predictor_16x16_msa,
                       &vpx_dc_128_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_128_predictor_32x32_msa,
                       &vpx_dc_128_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_dc_left_predictor_4x4_msa,
                       &vpx_dc_left_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_left_predictor_8x8_msa,
                       &vpx_dc_left_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_left_predictor_16x16_msa,
                       &vpx_dc_left_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_left_predictor_32x32_msa,
                       &vpx_dc_left_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_dc_predictor_4x4_msa, &vpx_dc_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_dc_predictor_8x8_msa, &vpx_dc_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_dc_predictor_16x16_msa, &vpx_dc_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_dc_predictor_32x32_msa, &vpx_dc_predictor_32x32_c,
                       32, 8),
        IntraPredParam(&vpx_dc_top_predictor_4x4_msa,
                       &vpx_dc_top_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_dc_top_predictor_8x8_msa,
                       &vpx_dc_top_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_dc_top_predictor_16x16_msa,
                       &vpx_dc_top_predictor_16x16_c, 16, 8),
        IntraPredParam(&vpx_dc_top_predictor_32x32_msa,
                       &vpx_dc_top_predictor_32x32_c, 32, 8),
        IntraPredParam(&vpx_h_predictor_4x4_msa, &vpx_h_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_h_predictor_8x8_msa, &vpx_h_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_h_predictor_16x16_msa, &vpx_h_predictor_16x16_c, 16,
                       8),
        IntraPredParam(&vpx_h_predictor_32x32_msa, &vpx_h_predictor_32x32_c, 32,
                       8),
        IntraPredParam(&vpx_tm_predictor_4x4_msa, &vpx_tm_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_tm_predictor_8x8_msa, &vpx_tm_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_tm_predictor_16x16_msa, &vpx_tm_predictor_16x16_c,
                       16, 8),
        IntraPredParam(&vpx_tm_predictor_32x32_msa, &vpx_tm_predictor_32x32_c,
                       32, 8),
        IntraPredParam(&vpx_v_predictor_4x4_msa, &vpx_v_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_v_predictor_8x8_msa, &vpx_v_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_v_predictor_16x16_msa, &vpx_v_predictor_16x16_c, 16,
                       8),
        IntraPredParam(&vpx_v_predictor_32x32_msa, &vpx_v_predictor_32x32_c, 32,
                       8)));
#endif  // HAVE_MSA

// TODO(crbug.com/webm/1522): Fix test failures.
#if 0
        IntraPredParam(&vpx_d45_predictor_8x8_vsx, &vpx_d45_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_d63_predictor_8x8_vsx, &vpx_d63_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_dc_predictor_8x8_vsx, &vpx_dc_predictor_8x8_c, 8,
                       8),
        IntraPredParam(&vpx_h_predictor_4x4_vsx, &vpx_h_predictor_4x4_c, 4, 8),
        IntraPredParam(&vpx_h_predictor_8x8_vsx, &vpx_h_predictor_8x8_c, 8, 8),
        IntraPredParam(&vpx_tm_predictor_4x4_vsx, &vpx_tm_predictor_4x4_c, 4,
                       8),
        IntraPredParam(&vpx_tm_predictor_8x8_vsx, &vpx_tm_predictor_8x8_c, 8,
                       8),
#endif

#if HAVE_VSX
INSTANTIATE_TEST_SUITE_P(
    VSX, VP9IntraPredTest,
    ::testing::Values(IntraPredParam(&vpx_d45_predictor_16x16_vsx,
                                     &vpx_d45_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_d45_predictor_32x32_vsx,
                                     &vpx_d45_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_d63_predictor_16x16_vsx,
                                     &vpx_d63_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_d63_predictor_32x32_vsx,
                                     &vpx_d63_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_dc_128_predictor_16x16_vsx,
                                     &vpx_dc_128_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_dc_128_predictor_32x32_vsx,
                                     &vpx_dc_128_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_dc_left_predictor_16x16_vsx,
                                     &vpx_dc_left_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_dc_left_predictor_32x32_vsx,
                                     &vpx_dc_left_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_dc_predictor_16x16_vsx,
                                     &vpx_dc_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_dc_predictor_32x32_vsx,
                                     &vpx_dc_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_dc_top_predictor_16x16_vsx,
                                     &vpx_dc_top_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_dc_top_predictor_32x32_vsx,
                                     &vpx_dc_top_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_h_predictor_16x16_vsx,
                                     &vpx_h_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_h_predictor_32x32_vsx,
                                     &vpx_h_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_tm_predictor_16x16_vsx,
                                     &vpx_tm_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_tm_predictor_32x32_vsx,
                                     &vpx_tm_predictor_32x32_c, 32, 8),
                      IntraPredParam(&vpx_v_predictor_16x16_vsx,
                                     &vpx_v_predictor_16x16_c, 16, 8),
                      IntraPredParam(&vpx_v_predictor_32x32_vsx,
                                     &vpx_v_predictor_32x32_c, 32, 8)));
#endif  // HAVE_VSX

#if HAVE_LSX
INSTANTIATE_TEST_SUITE_P(
    LSX, VP9IntraPredTest,
    ::testing::Values(IntraPredParam(&vpx_dc_predictor_8x8_lsx,
                                     &vpx_dc_predictor_8x8_c, 8, 8),
                      IntraPredParam(&vpx_dc_predictor_16x16_lsx,
                                     &vpx_dc_predictor_16x16_c, 16, 8)));
#endif  // HAVE_LSX

#if CONFIG_VP9_HIGHBITDEPTH
using HighbdIntraPred = void (*)(uint16_t *dst, ptrdiff_t stride,
                                 const uint16_t *above, const uint16_t *left,
                                 int bps);

struct HighbdIntraPredParam {
  HighbdIntraPredParam(HighbdIntraPred pred = nullptr,
                       HighbdIntraPred ref = nullptr, int block_size_value = 0,
                       int bit_depth_value = 0)
      : pred_fn(pred), ref_fn(ref), block_size(block_size_value),
        bit_depth(bit_depth_value) {}

  HighbdIntraPred pred_fn;
  HighbdIntraPred ref_fn;
  int block_size;
  int bit_depth;
};

#if HAVE_SSSE3 || HAVE_NEON || HAVE_SSE2
template <>
void IntraPredTest<uint16_t, HighbdIntraPredParam>::Predict() {
  const int bit_depth = params_.bit_depth;
  params_.ref_fn(ref_dst_, stride_, above_row_, left_col_, bit_depth);
  ASM_REGISTER_STATE_CHECK(
      params_.pred_fn(dst_, stride_, above_row_, left_col_, bit_depth));
}

using VP9HighbdIntraPredTest = IntraPredTest<uint16_t, HighbdIntraPredParam>;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VP9HighbdIntraPredTest);

TEST_P(VP9HighbdIntraPredTest, HighbdIntraPredTests) {
  // max block size is 32
  DECLARE_ALIGNED(16, uint16_t, left_col[2 * 32]);
  DECLARE_ALIGNED(16, uint16_t, above_data[2 * 32 + 32]);
  DECLARE_ALIGNED(16, uint16_t, dst[3 * 32 * 32]);
  DECLARE_ALIGNED(16, uint16_t, ref_dst[3 * 32 * 32]);
  RunTest(left_col, above_data, dst, ref_dst);
}
#endif

#if HAVE_SSSE3
INSTANTIATE_TEST_SUITE_P(
    SSSE3_TO_C_8, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_4x4_ssse3,
                             &vpx_highbd_d45_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_8x8_ssse3,
                             &vpx_highbd_d45_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_16x16_ssse3,
                             &vpx_highbd_d45_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_32x32_ssse3,
                             &vpx_highbd_d45_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_8x8_ssse3,
                             &vpx_highbd_d63_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_16x16_ssse3,
                             &vpx_highbd_d63_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_32x32_c,
                             &vpx_highbd_d63_predictor_32x32_ssse3, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_8x8_ssse3,
                             &vpx_highbd_d117_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_16x16_ssse3,
                             &vpx_highbd_d117_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_32x32_c,
                             &vpx_highbd_d117_predictor_32x32_ssse3, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_8x8_ssse3,
                             &vpx_highbd_d135_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_16x16_ssse3,
                             &vpx_highbd_d135_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_32x32_ssse3,
                             &vpx_highbd_d135_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_8x8_ssse3,
                             &vpx_highbd_d153_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_16x16_ssse3,
                             &vpx_highbd_d153_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_32x32_ssse3,
                             &vpx_highbd_d153_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_8x8_ssse3,
                             &vpx_highbd_d207_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_16x16_ssse3,
                             &vpx_highbd_d207_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_32x32_ssse3,
                             &vpx_highbd_d207_predictor_32x32_c, 32, 8)));

INSTANTIATE_TEST_SUITE_P(
    SSSE3_TO_C_10, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_4x4_ssse3,
                             &vpx_highbd_d45_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_8x8_ssse3,
                             &vpx_highbd_d45_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_16x16_ssse3,
                             &vpx_highbd_d45_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_32x32_ssse3,
                             &vpx_highbd_d45_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_8x8_ssse3,
                             &vpx_highbd_d63_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_16x16_ssse3,
                             &vpx_highbd_d63_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_32x32_c,
                             &vpx_highbd_d63_predictor_32x32_ssse3, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_8x8_ssse3,
                             &vpx_highbd_d117_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_16x16_ssse3,
                             &vpx_highbd_d117_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_32x32_c,
                             &vpx_highbd_d117_predictor_32x32_ssse3, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_8x8_ssse3,
                             &vpx_highbd_d135_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_16x16_ssse3,
                             &vpx_highbd_d135_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_32x32_ssse3,
                             &vpx_highbd_d135_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_8x8_ssse3,
                             &vpx_highbd_d153_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_16x16_ssse3,
                             &vpx_highbd_d153_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_32x32_ssse3,
                             &vpx_highbd_d153_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_8x8_ssse3,
                             &vpx_highbd_d207_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_16x16_ssse3,
                             &vpx_highbd_d207_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_32x32_ssse3,
                             &vpx_highbd_d207_predictor_32x32_c, 32, 10)));

INSTANTIATE_TEST_SUITE_P(
    SSSE3_TO_C_12, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_4x4_ssse3,
                             &vpx_highbd_d45_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_8x8_ssse3,
                             &vpx_highbd_d45_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_16x16_ssse3,
                             &vpx_highbd_d45_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_32x32_ssse3,
                             &vpx_highbd_d45_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_8x8_ssse3,
                             &vpx_highbd_d63_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_16x16_ssse3,
                             &vpx_highbd_d63_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_32x32_c,
                             &vpx_highbd_d63_predictor_32x32_ssse3, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_8x8_ssse3,
                             &vpx_highbd_d117_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_16x16_ssse3,
                             &vpx_highbd_d117_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_32x32_c,
                             &vpx_highbd_d117_predictor_32x32_ssse3, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_8x8_ssse3,
                             &vpx_highbd_d135_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_16x16_ssse3,
                             &vpx_highbd_d135_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_32x32_ssse3,
                             &vpx_highbd_d135_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_8x8_ssse3,
                             &vpx_highbd_d153_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_16x16_ssse3,
                             &vpx_highbd_d153_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_32x32_ssse3,
                             &vpx_highbd_d153_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_8x8_ssse3,
                             &vpx_highbd_d207_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_16x16_ssse3,
                             &vpx_highbd_d207_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_32x32_ssse3,
                             &vpx_highbd_d207_predictor_32x32_c, 32, 12)));
#endif  // HAVE_SSSE3

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2_TO_C_8, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_4x4_sse2,
                             &vpx_highbd_dc_128_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_8x8_sse2,
                             &vpx_highbd_dc_128_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_16x16_sse2,
                             &vpx_highbd_dc_128_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_32x32_sse2,
                             &vpx_highbd_dc_128_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_4x4_sse2,
                             &vpx_highbd_d63_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_4x4_sse2,
                             &vpx_highbd_d117_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_4x4_sse2,
                             &vpx_highbd_d135_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_4x4_sse2,
                             &vpx_highbd_d153_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_4x4_sse2,
                             &vpx_highbd_d207_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_4x4_sse2,
                             &vpx_highbd_dc_left_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_8x8_sse2,
                             &vpx_highbd_dc_left_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_16x16_sse2,
                             &vpx_highbd_dc_left_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_32x32_sse2,
                             &vpx_highbd_dc_left_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_4x4_sse2,
                             &vpx_highbd_dc_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_8x8_sse2,
                             &vpx_highbd_dc_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_16x16_sse2,
                             &vpx_highbd_dc_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_32x32_sse2,
                             &vpx_highbd_dc_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_4x4_sse2,
                             &vpx_highbd_dc_top_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_8x8_sse2,
                             &vpx_highbd_dc_top_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_16x16_sse2,
                             &vpx_highbd_dc_top_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_32x32_sse2,
                             &vpx_highbd_dc_top_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_4x4_sse2,
                             &vpx_highbd_tm_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_8x8_sse2,
                             &vpx_highbd_tm_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_16x16_sse2,
                             &vpx_highbd_tm_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_32x32_sse2,
                             &vpx_highbd_tm_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_4x4_sse2,
                             &vpx_highbd_h_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_8x8_sse2,
                             &vpx_highbd_h_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_16x16_sse2,
                             &vpx_highbd_h_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_32x32_sse2,
                             &vpx_highbd_h_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_4x4_sse2,
                             &vpx_highbd_v_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_8x8_sse2,
                             &vpx_highbd_v_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_16x16_sse2,
                             &vpx_highbd_v_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_32x32_sse2,
                             &vpx_highbd_v_predictor_32x32_c, 32, 8)));

INSTANTIATE_TEST_SUITE_P(
    SSE2_TO_C_10, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_4x4_sse2,
                             &vpx_highbd_dc_128_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_8x8_sse2,
                             &vpx_highbd_dc_128_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_16x16_sse2,
                             &vpx_highbd_dc_128_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_32x32_sse2,
                             &vpx_highbd_dc_128_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_4x4_sse2,
                             &vpx_highbd_d63_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_4x4_sse2,
                             &vpx_highbd_d117_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_4x4_sse2,
                             &vpx_highbd_d135_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_4x4_sse2,
                             &vpx_highbd_d153_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_4x4_sse2,
                             &vpx_highbd_d207_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_4x4_sse2,
                             &vpx_highbd_dc_left_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_8x8_sse2,
                             &vpx_highbd_dc_left_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_16x16_sse2,
                             &vpx_highbd_dc_left_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_32x32_sse2,
                             &vpx_highbd_dc_left_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_4x4_sse2,
                             &vpx_highbd_dc_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_8x8_sse2,
                             &vpx_highbd_dc_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_16x16_sse2,
                             &vpx_highbd_dc_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_32x32_sse2,
                             &vpx_highbd_dc_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_4x4_sse2,
                             &vpx_highbd_dc_top_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_8x8_sse2,
                             &vpx_highbd_dc_top_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_16x16_sse2,
                             &vpx_highbd_dc_top_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_32x32_sse2,
                             &vpx_highbd_dc_top_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_4x4_sse2,
                             &vpx_highbd_tm_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_8x8_sse2,
                             &vpx_highbd_tm_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_16x16_sse2,
                             &vpx_highbd_tm_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_32x32_sse2,
                             &vpx_highbd_tm_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_4x4_sse2,
                             &vpx_highbd_h_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_8x8_sse2,
                             &vpx_highbd_h_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_16x16_sse2,
                             &vpx_highbd_h_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_32x32_sse2,
                             &vpx_highbd_h_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_4x4_sse2,
                             &vpx_highbd_v_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_8x8_sse2,
                             &vpx_highbd_v_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_16x16_sse2,
                             &vpx_highbd_v_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_32x32_sse2,
                             &vpx_highbd_v_predictor_32x32_c, 32, 10)));

INSTANTIATE_TEST_SUITE_P(
    SSE2_TO_C_12, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_4x4_sse2,
                             &vpx_highbd_dc_128_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_8x8_sse2,
                             &vpx_highbd_dc_128_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_16x16_sse2,
                             &vpx_highbd_dc_128_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_32x32_sse2,
                             &vpx_highbd_dc_128_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_4x4_sse2,
                             &vpx_highbd_d63_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_4x4_sse2,
                             &vpx_highbd_d117_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_4x4_sse2,
                             &vpx_highbd_d135_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_4x4_sse2,
                             &vpx_highbd_d153_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_4x4_sse2,
                             &vpx_highbd_d207_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_4x4_sse2,
                             &vpx_highbd_dc_left_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_8x8_sse2,
                             &vpx_highbd_dc_left_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_16x16_sse2,
                             &vpx_highbd_dc_left_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_32x32_sse2,
                             &vpx_highbd_dc_left_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_4x4_sse2,
                             &vpx_highbd_dc_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_8x8_sse2,
                             &vpx_highbd_dc_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_16x16_sse2,
                             &vpx_highbd_dc_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_32x32_sse2,
                             &vpx_highbd_dc_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_4x4_sse2,
                             &vpx_highbd_dc_top_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_8x8_sse2,
                             &vpx_highbd_dc_top_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_16x16_sse2,
                             &vpx_highbd_dc_top_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_32x32_sse2,
                             &vpx_highbd_dc_top_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_4x4_sse2,
                             &vpx_highbd_tm_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_8x8_sse2,
                             &vpx_highbd_tm_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_16x16_sse2,
                             &vpx_highbd_tm_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_32x32_sse2,
                             &vpx_highbd_tm_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_4x4_sse2,
                             &vpx_highbd_h_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_8x8_sse2,
                             &vpx_highbd_h_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_16x16_sse2,
                             &vpx_highbd_h_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_32x32_sse2,
                             &vpx_highbd_h_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_4x4_sse2,
                             &vpx_highbd_v_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_8x8_sse2,
                             &vpx_highbd_v_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_16x16_sse2,
                             &vpx_highbd_v_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_32x32_sse2,
                             &vpx_highbd_v_predictor_32x32_c, 32, 12)));
#endif  // HAVE_SSE2

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON_TO_C_8, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_4x4_neon,
                             &vpx_highbd_d45_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_8x8_neon,
                             &vpx_highbd_d45_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_16x16_neon,
                             &vpx_highbd_d45_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_32x32_neon,
                             &vpx_highbd_d45_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_4x4_neon,
                             &vpx_highbd_d63_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_8x8_neon,
                             &vpx_highbd_d63_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_16x16_neon,
                             &vpx_highbd_d63_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_32x32_neon,
                             &vpx_highbd_d63_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_4x4_neon,
                             &vpx_highbd_d117_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_8x8_neon,
                             &vpx_highbd_d117_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_16x16_neon,
                             &vpx_highbd_d117_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_32x32_neon,
                             &vpx_highbd_d117_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_4x4_neon,
                             &vpx_highbd_d135_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_8x8_neon,
                             &vpx_highbd_d135_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_16x16_neon,
                             &vpx_highbd_d135_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_32x32_neon,
                             &vpx_highbd_d135_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_4x4_neon,
                             &vpx_highbd_d153_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_8x8_neon,
                             &vpx_highbd_d153_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_16x16_neon,
                             &vpx_highbd_d153_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_32x32_neon,
                             &vpx_highbd_d153_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_4x4_neon,
                             &vpx_highbd_d207_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_8x8_neon,
                             &vpx_highbd_d207_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_16x16_neon,
                             &vpx_highbd_d207_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_32x32_neon,
                             &vpx_highbd_d207_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_4x4_neon,
                             &vpx_highbd_dc_128_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_8x8_neon,
                             &vpx_highbd_dc_128_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_16x16_neon,
                             &vpx_highbd_dc_128_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_32x32_neon,
                             &vpx_highbd_dc_128_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_4x4_neon,
                             &vpx_highbd_dc_left_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_8x8_neon,
                             &vpx_highbd_dc_left_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_16x16_neon,
                             &vpx_highbd_dc_left_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_32x32_neon,
                             &vpx_highbd_dc_left_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_4x4_neon,
                             &vpx_highbd_dc_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_8x8_neon,
                             &vpx_highbd_dc_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_16x16_neon,
                             &vpx_highbd_dc_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_32x32_neon,
                             &vpx_highbd_dc_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_4x4_neon,
                             &vpx_highbd_dc_top_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_8x8_neon,
                             &vpx_highbd_dc_top_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_16x16_neon,
                             &vpx_highbd_dc_top_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_32x32_neon,
                             &vpx_highbd_dc_top_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_4x4_neon,
                             &vpx_highbd_h_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_8x8_neon,
                             &vpx_highbd_h_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_16x16_neon,
                             &vpx_highbd_h_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_32x32_neon,
                             &vpx_highbd_h_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_4x4_neon,
                             &vpx_highbd_tm_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_8x8_neon,
                             &vpx_highbd_tm_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_16x16_neon,
                             &vpx_highbd_tm_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_32x32_neon,
                             &vpx_highbd_tm_predictor_32x32_c, 32, 8),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_4x4_neon,
                             &vpx_highbd_v_predictor_4x4_c, 4, 8),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_8x8_neon,
                             &vpx_highbd_v_predictor_8x8_c, 8, 8),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_16x16_neon,
                             &vpx_highbd_v_predictor_16x16_c, 16, 8),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_32x32_neon,
                             &vpx_highbd_v_predictor_32x32_c, 32, 8)));

INSTANTIATE_TEST_SUITE_P(
    NEON_TO_C_10, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_4x4_neon,
                             &vpx_highbd_d45_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_8x8_neon,
                             &vpx_highbd_d45_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_16x16_neon,
                             &vpx_highbd_d45_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_32x32_neon,
                             &vpx_highbd_d45_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_4x4_neon,
                             &vpx_highbd_d63_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_8x8_neon,
                             &vpx_highbd_d63_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_16x16_neon,
                             &vpx_highbd_d63_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_32x32_neon,
                             &vpx_highbd_d63_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_4x4_neon,
                             &vpx_highbd_d117_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_8x8_neon,
                             &vpx_highbd_d117_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_16x16_neon,
                             &vpx_highbd_d117_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_32x32_neon,
                             &vpx_highbd_d117_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_4x4_neon,
                             &vpx_highbd_d135_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_8x8_neon,
                             &vpx_highbd_d135_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_16x16_neon,
                             &vpx_highbd_d135_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_32x32_neon,
                             &vpx_highbd_d135_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_4x4_neon,
                             &vpx_highbd_d153_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_8x8_neon,
                             &vpx_highbd_d153_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_16x16_neon,
                             &vpx_highbd_d153_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_32x32_neon,
                             &vpx_highbd_d153_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_4x4_neon,
                             &vpx_highbd_d207_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_8x8_neon,
                             &vpx_highbd_d207_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_16x16_neon,
                             &vpx_highbd_d207_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_32x32_neon,
                             &vpx_highbd_d207_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_4x4_neon,
                             &vpx_highbd_dc_128_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_8x8_neon,
                             &vpx_highbd_dc_128_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_16x16_neon,
                             &vpx_highbd_dc_128_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_32x32_neon,
                             &vpx_highbd_dc_128_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_4x4_neon,
                             &vpx_highbd_dc_left_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_8x8_neon,
                             &vpx_highbd_dc_left_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_16x16_neon,
                             &vpx_highbd_dc_left_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_32x32_neon,
                             &vpx_highbd_dc_left_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_4x4_neon,
                             &vpx_highbd_dc_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_8x8_neon,
                             &vpx_highbd_dc_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_16x16_neon,
                             &vpx_highbd_dc_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_32x32_neon,
                             &vpx_highbd_dc_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_4x4_neon,
                             &vpx_highbd_dc_top_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_8x8_neon,
                             &vpx_highbd_dc_top_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_16x16_neon,
                             &vpx_highbd_dc_top_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_32x32_neon,
                             &vpx_highbd_dc_top_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_4x4_neon,
                             &vpx_highbd_h_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_8x8_neon,
                             &vpx_highbd_h_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_16x16_neon,
                             &vpx_highbd_h_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_32x32_neon,
                             &vpx_highbd_h_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_4x4_neon,
                             &vpx_highbd_tm_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_8x8_neon,
                             &vpx_highbd_tm_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_16x16_neon,
                             &vpx_highbd_tm_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_32x32_neon,
                             &vpx_highbd_tm_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_4x4_neon,
                             &vpx_highbd_v_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_8x8_neon,
                             &vpx_highbd_v_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_16x16_neon,
                             &vpx_highbd_v_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_32x32_neon,
                             &vpx_highbd_v_predictor_32x32_c, 32, 10)));

INSTANTIATE_TEST_SUITE_P(
    NEON_TO_C_12, VP9HighbdIntraPredTest,
    ::testing::Values(
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_4x4_neon,
                             &vpx_highbd_d45_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_8x8_neon,
                             &vpx_highbd_d45_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_16x16_neon,
                             &vpx_highbd_d45_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d45_predictor_32x32_neon,
                             &vpx_highbd_d45_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_4x4_neon,
                             &vpx_highbd_d63_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_8x8_neon,
                             &vpx_highbd_d63_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_16x16_neon,
                             &vpx_highbd_d63_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d63_predictor_32x32_neon,
                             &vpx_highbd_d63_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_4x4_neon,
                             &vpx_highbd_d117_predictor_4x4_c, 4, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_8x8_neon,
                             &vpx_highbd_d117_predictor_8x8_c, 8, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_16x16_neon,
                             &vpx_highbd_d117_predictor_16x16_c, 16, 10),
        HighbdIntraPredParam(&vpx_highbd_d117_predictor_32x32_neon,
                             &vpx_highbd_d117_predictor_32x32_c, 32, 10),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_4x4_neon,
                             &vpx_highbd_d135_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_8x8_neon,
                             &vpx_highbd_d135_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_16x16_neon,
                             &vpx_highbd_d135_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d135_predictor_32x32_neon,
                             &vpx_highbd_d135_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_4x4_neon,
                             &vpx_highbd_d153_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_8x8_neon,
                             &vpx_highbd_d153_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_16x16_neon,
                             &vpx_highbd_d153_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d153_predictor_32x32_neon,
                             &vpx_highbd_d153_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_4x4_neon,
                             &vpx_highbd_d207_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_8x8_neon,
                             &vpx_highbd_d207_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_16x16_neon,
                             &vpx_highbd_d207_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_d207_predictor_32x32_neon,
                             &vpx_highbd_d207_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_4x4_neon,
                             &vpx_highbd_dc_128_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_8x8_neon,
                             &vpx_highbd_dc_128_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_16x16_neon,
                             &vpx_highbd_dc_128_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_128_predictor_32x32_neon,
                             &vpx_highbd_dc_128_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_4x4_neon,
                             &vpx_highbd_dc_left_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_8x8_neon,
                             &vpx_highbd_dc_left_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_16x16_neon,
                             &vpx_highbd_dc_left_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_left_predictor_32x32_neon,
                             &vpx_highbd_dc_left_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_4x4_neon,
                             &vpx_highbd_dc_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_8x8_neon,
                             &vpx_highbd_dc_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_16x16_neon,
                             &vpx_highbd_dc_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_predictor_32x32_neon,
                             &vpx_highbd_dc_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_4x4_neon,
                             &vpx_highbd_dc_top_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_8x8_neon,
                             &vpx_highbd_dc_top_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_16x16_neon,
                             &vpx_highbd_dc_top_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_dc_top_predictor_32x32_neon,
                             &vpx_highbd_dc_top_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_4x4_neon,
                             &vpx_highbd_h_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_8x8_neon,
                             &vpx_highbd_h_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_16x16_neon,
                             &vpx_highbd_h_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_h_predictor_32x32_neon,
                             &vpx_highbd_h_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_4x4_neon,
                             &vpx_highbd_tm_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_8x8_neon,
                             &vpx_highbd_tm_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_16x16_neon,
                             &vpx_highbd_tm_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_tm_predictor_32x32_neon,
                             &vpx_highbd_tm_predictor_32x32_c, 32, 12),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_4x4_neon,
                             &vpx_highbd_v_predictor_4x4_c, 4, 12),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_8x8_neon,
                             &vpx_highbd_v_predictor_8x8_c, 8, 12),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_16x16_neon,
                             &vpx_highbd_v_predictor_16x16_c, 16, 12),
        HighbdIntraPredParam(&vpx_highbd_v_predictor_32x32_neon,
                             &vpx_highbd_v_predictor_32x32_c, 32, 12)));
#endif  // HAVE_NEON

#endif  // CONFIG_VP9_HIGHBITDEPTH
}  // namespace
