/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <cmath>
#include <cstdlib>
#include <string>
#include <tuple>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vp9/common/vp9_entropy.h"
#include "vp9/common/vp9_loopfilter.h"
#include "vpx/vpx_integer.h"

using libvpx_test::ACMRandom;

namespace {
// Horizontally and Vertically need 32x32: 8  Coeffs preceeding filtered section
//                                         16 Coefs within filtered section
//                                         8  Coeffs following filtered section
const int kNumCoeffs = 1024;

const int number_of_iterations = 10000;

#if CONFIG_VP9_HIGHBITDEPTH
using Pixel = uint16_t;
#define PIXEL_WIDTH 16

using loop_op_t = void (*)(Pixel *s, int p, const uint8_t *blimit,
                           const uint8_t *limit, const uint8_t *thresh, int bd);
using dual_loop_op_t = void (*)(Pixel *s, int p, const uint8_t *blimit0,
                                const uint8_t *limit0, const uint8_t *thresh0,
                                const uint8_t *blimit1, const uint8_t *limit1,
                                const uint8_t *thresh1, int bd);
#else
using Pixel = uint8_t;
#define PIXEL_WIDTH 8

using loop_op_t = void (*)(Pixel *s, int p, const uint8_t *blimit,
                           const uint8_t *limit, const uint8_t *thresh);
using dual_loop_op_t = void (*)(Pixel *s, int p, const uint8_t *blimit0,
                                const uint8_t *limit0, const uint8_t *thresh0,
                                const uint8_t *blimit1, const uint8_t *limit1,
                                const uint8_t *thresh1);
#endif  // CONFIG_VP9_HIGHBITDEPTH

using loop8_param_t = std::tuple<loop_op_t, loop_op_t, int>;
using dualloop8_param_t = std::tuple<dual_loop_op_t, dual_loop_op_t, int>;

void InitInput(Pixel *s, Pixel *ref_s, ACMRandom *rnd, const uint8_t limit,
               const int mask, const int32_t p, const int i) {
  uint16_t tmp_s[kNumCoeffs];

  for (int j = 0; j < kNumCoeffs;) {
    const uint8_t val = rnd->Rand8();
    if (val & 0x80) {  // 50% chance to choose a new value.
      tmp_s[j] = rnd->Rand16();
      j++;
    } else {  // 50% chance to repeat previous value in row X times.
      int k = 0;
      while (k++ < ((val & 0x1f) + 1) && j < kNumCoeffs) {
        if (j < 1) {
          tmp_s[j] = rnd->Rand16();
        } else if (val & 0x20) {  // Increment by a value within the limit.
          tmp_s[j] = static_cast<uint16_t>(tmp_s[j - 1] + (limit - 1));
        } else {  // Decrement by a value within the limit.
          tmp_s[j] = static_cast<uint16_t>(tmp_s[j - 1] - (limit - 1));
        }
        j++;
      }
    }
  }

  for (int j = 0; j < kNumCoeffs;) {
    const uint8_t val = rnd->Rand8();
    if (val & 0x80) {
      j++;
    } else {  // 50% chance to repeat previous value in column X times.
      int k = 0;
      while (k++ < ((val & 0x1f) + 1) && j < kNumCoeffs) {
        if (j < 1) {
          tmp_s[j] = rnd->Rand16();
        } else if (val & 0x20) {  // Increment by a value within the limit.
          tmp_s[(j % 32) * 32 + j / 32] = static_cast<uint16_t>(
              tmp_s[((j - 1) % 32) * 32 + (j - 1) / 32] + (limit - 1));
        } else {  // Decrement by a value within the limit.
          tmp_s[(j % 32) * 32 + j / 32] = static_cast<uint16_t>(
              tmp_s[((j - 1) % 32) * 32 + (j - 1) / 32] - (limit - 1));
        }
        j++;
      }
    }
  }

  for (int j = 0; j < kNumCoeffs; j++) {
    if (i % 2) {
      s[j] = tmp_s[j] & mask;
    } else {
      s[j] = tmp_s[p * (j % p) + j / p] & mask;
    }
    ref_s[j] = s[j];
  }
}

uint8_t GetOuterThresh(ACMRandom *rnd) {
  return static_cast<uint8_t>(rnd->RandRange(3 * MAX_LOOP_FILTER + 5));
}

uint8_t GetInnerThresh(ACMRandom *rnd) {
  return static_cast<uint8_t>(rnd->RandRange(MAX_LOOP_FILTER + 1));
}

uint8_t GetHevThresh(ACMRandom *rnd) {
  return static_cast<uint8_t>(rnd->RandRange(MAX_LOOP_FILTER + 1) >> 4);
}

class Loop8Test6Param : public ::testing::TestWithParam<loop8_param_t> {
 public:
  ~Loop8Test6Param() override = default;
  void SetUp() override {
    loopfilter_op_ = GET_PARAM(0);
    ref_loopfilter_op_ = GET_PARAM(1);
    bit_depth_ = GET_PARAM(2);
    mask_ = (1 << bit_depth_) - 1;
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  int bit_depth_;
  int mask_;
  loop_op_t loopfilter_op_;
  loop_op_t ref_loopfilter_op_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Loop8Test6Param);

#if HAVE_NEON || HAVE_SSE2 || (HAVE_LSX && !CONFIG_VP9_HIGHBITDEPTH) || \
    (HAVE_DSPR2 || HAVE_MSA && !CONFIG_VP9_HIGHBITDEPTH)
class Loop8Test9Param : public ::testing::TestWithParam<dualloop8_param_t> {
 public:
  ~Loop8Test9Param() override = default;
  void SetUp() override {
    loopfilter_op_ = GET_PARAM(0);
    ref_loopfilter_op_ = GET_PARAM(1);
    bit_depth_ = GET_PARAM(2);
    mask_ = (1 << bit_depth_) - 1;
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  int bit_depth_;
  int mask_;
  dual_loop_op_t loopfilter_op_;
  dual_loop_op_t ref_loopfilter_op_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Loop8Test9Param);
#endif  // HAVE_NEON || HAVE_SSE2 || (HAVE_DSPR2 || HAVE_MSA &&
        // (!CONFIG_VP9_HIGHBITDEPTH) || (HAVE_LSX && !CONFIG_VP9_HIGHBITDEPTH))

TEST_P(Loop8Test6Param, OperationCheck) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  const int count_test_block = number_of_iterations;
  const int32_t p = kNumCoeffs / 32;
  DECLARE_ALIGNED(PIXEL_WIDTH, Pixel, s[kNumCoeffs]);
  DECLARE_ALIGNED(PIXEL_WIDTH, Pixel, ref_s[kNumCoeffs]);
  int err_count_total = 0;
  int first_failure = -1;
  for (int i = 0; i < count_test_block; ++i) {
    int err_count = 0;
    uint8_t tmp = GetOuterThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    blimit[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                    tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetInnerThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    limit[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                   tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetHevThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    thresh[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                    tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    InitInput(s, ref_s, &rnd, *limit, mask_, p, i);
#if CONFIG_VP9_HIGHBITDEPTH
    ref_loopfilter_op_(ref_s + 8 + p * 8, p, blimit, limit, thresh, bit_depth_);
    ASM_REGISTER_STATE_CHECK(
        loopfilter_op_(s + 8 + p * 8, p, blimit, limit, thresh, bit_depth_));
#else
    ref_loopfilter_op_(ref_s + 8 + p * 8, p, blimit, limit, thresh);
    ASM_REGISTER_STATE_CHECK(
        loopfilter_op_(s + 8 + p * 8, p, blimit, limit, thresh));
#endif  // CONFIG_VP9_HIGHBITDEPTH

    for (int j = 0; j < kNumCoeffs; ++j) {
      err_count += ref_s[j] != s[j];
    }
    if (err_count && !err_count_total) {
      first_failure = i;
    }
    err_count_total += err_count;
  }
  EXPECT_EQ(0, err_count_total)
      << "Error: Loop8Test6Param, C output doesn't match SSE2 "
         "loopfilter output. "
      << "First failed at test case " << first_failure;
}

TEST_P(Loop8Test6Param, ValueCheck) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  const int count_test_block = number_of_iterations;
  DECLARE_ALIGNED(PIXEL_WIDTH, Pixel, s[kNumCoeffs]);
  DECLARE_ALIGNED(PIXEL_WIDTH, Pixel, ref_s[kNumCoeffs]);
  int err_count_total = 0;
  int first_failure = -1;

  // NOTE: The code in vp9_loopfilter.c:update_sharpness computes mblim as a
  // function of sharpness_lvl and the loopfilter lvl as:
  // block_inside_limit = lvl >> ((sharpness_lvl > 0) + (sharpness_lvl > 4));
  // ...
  // memset(lfi->lfthr[lvl].mblim, (2 * (lvl + 2) + block_inside_limit),
  //        SIMD_WIDTH);
  // This means that the largest value for mblim will occur when sharpness_lvl
  // is equal to 0, and lvl is equal to its greatest value (MAX_LOOP_FILTER).
  // In this case block_inside_limit will be equal to MAX_LOOP_FILTER and
  // therefore mblim will be equal to (2 * (lvl + 2) + block_inside_limit) =
  // 2 * (MAX_LOOP_FILTER + 2) + MAX_LOOP_FILTER = 3 * MAX_LOOP_FILTER + 4

  for (int i = 0; i < count_test_block; ++i) {
    int err_count = 0;
    uint8_t tmp = GetOuterThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    blimit[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                    tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetInnerThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    limit[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                   tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetHevThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    thresh[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                    tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    int32_t p = kNumCoeffs / 32;
    for (int j = 0; j < kNumCoeffs; ++j) {
      s[j] = rnd.Rand16() & mask_;
      ref_s[j] = s[j];
    }
#if CONFIG_VP9_HIGHBITDEPTH
    ref_loopfilter_op_(ref_s + 8 + p * 8, p, blimit, limit, thresh, bit_depth_);
    ASM_REGISTER_STATE_CHECK(
        loopfilter_op_(s + 8 + p * 8, p, blimit, limit, thresh, bit_depth_));
#else
    ref_loopfilter_op_(ref_s + 8 + p * 8, p, blimit, limit, thresh);
    ASM_REGISTER_STATE_CHECK(
        loopfilter_op_(s + 8 + p * 8, p, blimit, limit, thresh));
#endif  // CONFIG_VP9_HIGHBITDEPTH

    for (int j = 0; j < kNumCoeffs; ++j) {
      err_count += ref_s[j] != s[j];
    }
    if (err_count && !err_count_total) {
      first_failure = i;
    }
    err_count_total += err_count;
  }
  EXPECT_EQ(0, err_count_total)
      << "Error: Loop8Test6Param, C output doesn't match SSE2 "
         "loopfilter output. "
      << "First failed at test case " << first_failure;
}

#if HAVE_NEON || HAVE_SSE2 || (HAVE_LSX && (!CONFIG_VP9_HIGHBITDEPTH)) || \
    (HAVE_DSPR2 || HAVE_MSA && (!CONFIG_VP9_HIGHBITDEPTH))
TEST_P(Loop8Test9Param, OperationCheck) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  const int count_test_block = number_of_iterations;
  DECLARE_ALIGNED(PIXEL_WIDTH, Pixel, s[kNumCoeffs]);
  DECLARE_ALIGNED(PIXEL_WIDTH, Pixel, ref_s[kNumCoeffs]);
  int err_count_total = 0;
  int first_failure = -1;
  for (int i = 0; i < count_test_block; ++i) {
    int err_count = 0;
    uint8_t tmp = GetOuterThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    blimit0[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                     tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetInnerThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    limit0[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                    tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetHevThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    thresh0[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                     tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetOuterThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    blimit1[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                     tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetInnerThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    limit1[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                    tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetHevThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    thresh1[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                     tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    int32_t p = kNumCoeffs / 32;
    const uint8_t limit = *limit0 < *limit1 ? *limit0 : *limit1;
    InitInput(s, ref_s, &rnd, limit, mask_, p, i);
#if CONFIG_VP9_HIGHBITDEPTH
    ref_loopfilter_op_(ref_s + 8 + p * 8, p, blimit0, limit0, thresh0, blimit1,
                       limit1, thresh1, bit_depth_);
    ASM_REGISTER_STATE_CHECK(loopfilter_op_(s + 8 + p * 8, p, blimit0, limit0,
                                            thresh0, blimit1, limit1, thresh1,
                                            bit_depth_));
#else
    ref_loopfilter_op_(ref_s + 8 + p * 8, p, blimit0, limit0, thresh0, blimit1,
                       limit1, thresh1);
    ASM_REGISTER_STATE_CHECK(loopfilter_op_(s + 8 + p * 8, p, blimit0, limit0,
                                            thresh0, blimit1, limit1, thresh1));
#endif  // CONFIG_VP9_HIGHBITDEPTH

    for (int j = 0; j < kNumCoeffs; ++j) {
      err_count += ref_s[j] != s[j];
    }
    if (err_count && !err_count_total) {
      first_failure = i;
    }
    err_count_total += err_count;
  }
  EXPECT_EQ(0, err_count_total)
      << "Error: Loop8Test9Param, C output doesn't match SSE2 "
         "loopfilter output. "
      << "First failed at test case " << first_failure;
}

TEST_P(Loop8Test9Param, ValueCheck) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  const int count_test_block = number_of_iterations;
  DECLARE_ALIGNED(PIXEL_WIDTH, Pixel, s[kNumCoeffs]);
  DECLARE_ALIGNED(PIXEL_WIDTH, Pixel, ref_s[kNumCoeffs]);
  int err_count_total = 0;
  int first_failure = -1;
  for (int i = 0; i < count_test_block; ++i) {
    int err_count = 0;
    uint8_t tmp = GetOuterThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    blimit0[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                     tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetInnerThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    limit0[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                    tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetHevThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    thresh0[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                     tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetOuterThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    blimit1[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                     tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetInnerThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    limit1[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                    tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    tmp = GetHevThresh(&rnd);
    DECLARE_ALIGNED(16, const uint8_t,
                    thresh1[16]) = { tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp,
                                     tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp };
    int32_t p = kNumCoeffs / 32;  // TODO(pdlf) can we have non-square here?
    for (int j = 0; j < kNumCoeffs; ++j) {
      s[j] = rnd.Rand16() & mask_;
      ref_s[j] = s[j];
    }
#if CONFIG_VP9_HIGHBITDEPTH
    ref_loopfilter_op_(ref_s + 8 + p * 8, p, blimit0, limit0, thresh0, blimit1,
                       limit1, thresh1, bit_depth_);
    ASM_REGISTER_STATE_CHECK(loopfilter_op_(s + 8 + p * 8, p, blimit0, limit0,
                                            thresh0, blimit1, limit1, thresh1,
                                            bit_depth_));
#else
    ref_loopfilter_op_(ref_s + 8 + p * 8, p, blimit0, limit0, thresh0, blimit1,
                       limit1, thresh1);
    ASM_REGISTER_STATE_CHECK(loopfilter_op_(s + 8 + p * 8, p, blimit0, limit0,
                                            thresh0, blimit1, limit1, thresh1));
#endif  // CONFIG_VP9_HIGHBITDEPTH

    for (int j = 0; j < kNumCoeffs; ++j) {
      err_count += ref_s[j] != s[j];
    }
    if (err_count && !err_count_total) {
      first_failure = i;
    }
    err_count_total += err_count;
  }
  EXPECT_EQ(0, err_count_total)
      << "Error: Loop8Test9Param, C output doesn't match SSE2"
         "loopfilter output. "
      << "First failed at test case " << first_failure;
}
#endif  // HAVE_NEON || HAVE_SSE2 || (HAVE_DSPR2 || HAVE_MSA &&
        // (!CONFIG_VP9_HIGHBITDEPTH)) || (HAVE_LSX &&
        // (!CONFIG_VP9_HIGHBITDEPTH))

using std::make_tuple;

#if HAVE_SSE2
#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    SSE2, Loop8Test6Param,
    ::testing::Values(make_tuple(&vpx_highbd_lpf_horizontal_4_sse2,
                                 &vpx_highbd_lpf_horizontal_4_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_4_sse2,
                                 &vpx_highbd_lpf_vertical_4_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_sse2,
                                 &vpx_highbd_lpf_horizontal_8_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_sse2,
                                 &vpx_highbd_lpf_horizontal_16_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_16_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_8_sse2,
                                 &vpx_highbd_lpf_vertical_8_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_16_sse2,
                                 &vpx_highbd_lpf_vertical_16_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_4_sse2,
                                 &vpx_highbd_lpf_horizontal_4_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_4_sse2,
                                 &vpx_highbd_lpf_vertical_4_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_sse2,
                                 &vpx_highbd_lpf_horizontal_8_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_sse2,
                                 &vpx_highbd_lpf_horizontal_16_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_16_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_8_sse2,
                                 &vpx_highbd_lpf_vertical_8_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_16_sse2,
                                 &vpx_highbd_lpf_vertical_16_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_4_sse2,
                                 &vpx_highbd_lpf_horizontal_4_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_4_sse2,
                                 &vpx_highbd_lpf_vertical_4_c, 12),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_sse2,
                                 &vpx_highbd_lpf_horizontal_8_c, 12),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_sse2,
                                 &vpx_highbd_lpf_horizontal_16_c, 12),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_16_dual_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_8_sse2,
                                 &vpx_highbd_lpf_vertical_8_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_16_sse2,
                                 &vpx_highbd_lpf_vertical_16_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_16_dual_sse2,
                                 &vpx_highbd_lpf_vertical_16_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_16_dual_sse2,
                                 &vpx_highbd_lpf_vertical_16_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_16_dual_sse2,
                                 &vpx_highbd_lpf_vertical_16_dual_c, 12)));
#else
INSTANTIATE_TEST_SUITE_P(
    SSE2, Loop8Test6Param,
    ::testing::Values(
        make_tuple(&vpx_lpf_horizontal_4_sse2, &vpx_lpf_horizontal_4_c, 8),
        make_tuple(&vpx_lpf_horizontal_8_sse2, &vpx_lpf_horizontal_8_c, 8),
        make_tuple(&vpx_lpf_horizontal_16_sse2, &vpx_lpf_horizontal_16_c, 8),
        make_tuple(&vpx_lpf_horizontal_16_dual_sse2,
                   &vpx_lpf_horizontal_16_dual_c, 8),
        make_tuple(&vpx_lpf_vertical_4_sse2, &vpx_lpf_vertical_4_c, 8),
        make_tuple(&vpx_lpf_vertical_8_sse2, &vpx_lpf_vertical_8_c, 8),
        make_tuple(&vpx_lpf_vertical_16_sse2, &vpx_lpf_vertical_16_c, 8),
        make_tuple(&vpx_lpf_vertical_16_dual_sse2, &vpx_lpf_vertical_16_dual_c,
                   8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif

#if HAVE_AVX2 && (!CONFIG_VP9_HIGHBITDEPTH)
INSTANTIATE_TEST_SUITE_P(
    AVX2, Loop8Test6Param,
    ::testing::Values(make_tuple(&vpx_lpf_horizontal_16_avx2,
                                 &vpx_lpf_horizontal_16_c, 8),
                      make_tuple(&vpx_lpf_horizontal_16_dual_avx2,
                                 &vpx_lpf_horizontal_16_dual_c, 8)));
#endif

#if HAVE_SSE2
#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    SSE2, Loop8Test9Param,
    ::testing::Values(make_tuple(&vpx_highbd_lpf_horizontal_4_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_4_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_8_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_4_dual_sse2,
                                 &vpx_highbd_lpf_vertical_4_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_8_dual_sse2,
                                 &vpx_highbd_lpf_vertical_8_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_4_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_4_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_8_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_4_dual_sse2,
                                 &vpx_highbd_lpf_vertical_4_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_8_dual_sse2,
                                 &vpx_highbd_lpf_vertical_8_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_4_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_4_dual_c, 12),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_dual_sse2,
                                 &vpx_highbd_lpf_horizontal_8_dual_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_4_dual_sse2,
                                 &vpx_highbd_lpf_vertical_4_dual_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_8_dual_sse2,
                                 &vpx_highbd_lpf_vertical_8_dual_c, 12)));
#else
INSTANTIATE_TEST_SUITE_P(
    SSE2, Loop8Test9Param,
    ::testing::Values(make_tuple(&vpx_lpf_horizontal_4_dual_sse2,
                                 &vpx_lpf_horizontal_4_dual_c, 8),
                      make_tuple(&vpx_lpf_horizontal_8_dual_sse2,
                                 &vpx_lpf_horizontal_8_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_4_dual_sse2,
                                 &vpx_lpf_vertical_4_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_8_dual_sse2,
                                 &vpx_lpf_vertical_8_dual_c, 8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif

#if HAVE_NEON
#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    NEON, Loop8Test6Param,
    ::testing::Values(make_tuple(&vpx_highbd_lpf_horizontal_4_neon,
                                 &vpx_highbd_lpf_horizontal_4_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_4_neon,
                                 &vpx_highbd_lpf_horizontal_4_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_4_neon,
                                 &vpx_highbd_lpf_horizontal_4_c, 12),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_neon,
                                 &vpx_highbd_lpf_horizontal_8_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_neon,
                                 &vpx_highbd_lpf_horizontal_8_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_neon,
                                 &vpx_highbd_lpf_horizontal_8_c, 12),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_neon,
                                 &vpx_highbd_lpf_horizontal_16_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_neon,
                                 &vpx_highbd_lpf_horizontal_16_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_neon,
                                 &vpx_highbd_lpf_horizontal_16_c, 12),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_dual_neon,
                                 &vpx_highbd_lpf_horizontal_16_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_dual_neon,
                                 &vpx_highbd_lpf_horizontal_16_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_16_dual_neon,
                                 &vpx_highbd_lpf_horizontal_16_dual_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_4_neon,
                                 &vpx_highbd_lpf_vertical_4_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_4_neon,
                                 &vpx_highbd_lpf_vertical_4_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_4_neon,
                                 &vpx_highbd_lpf_vertical_4_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_8_neon,
                                 &vpx_highbd_lpf_vertical_8_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_8_neon,
                                 &vpx_highbd_lpf_vertical_8_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_8_neon,
                                 &vpx_highbd_lpf_vertical_8_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_16_neon,
                                 &vpx_highbd_lpf_vertical_16_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_16_neon,
                                 &vpx_highbd_lpf_vertical_16_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_16_neon,
                                 &vpx_highbd_lpf_vertical_16_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_16_dual_neon,
                                 &vpx_highbd_lpf_vertical_16_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_16_dual_neon,
                                 &vpx_highbd_lpf_vertical_16_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_16_dual_neon,
                                 &vpx_highbd_lpf_vertical_16_dual_c, 12)));
INSTANTIATE_TEST_SUITE_P(
    NEON, Loop8Test9Param,
    ::testing::Values(make_tuple(&vpx_highbd_lpf_horizontal_4_dual_neon,
                                 &vpx_highbd_lpf_horizontal_4_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_4_dual_neon,
                                 &vpx_highbd_lpf_horizontal_4_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_4_dual_neon,
                                 &vpx_highbd_lpf_horizontal_4_dual_c, 12),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_dual_neon,
                                 &vpx_highbd_lpf_horizontal_8_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_dual_neon,
                                 &vpx_highbd_lpf_horizontal_8_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_horizontal_8_dual_neon,
                                 &vpx_highbd_lpf_horizontal_8_dual_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_4_dual_neon,
                                 &vpx_highbd_lpf_vertical_4_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_4_dual_neon,
                                 &vpx_highbd_lpf_vertical_4_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_4_dual_neon,
                                 &vpx_highbd_lpf_vertical_4_dual_c, 12),
                      make_tuple(&vpx_highbd_lpf_vertical_8_dual_neon,
                                 &vpx_highbd_lpf_vertical_8_dual_c, 8),
                      make_tuple(&vpx_highbd_lpf_vertical_8_dual_neon,
                                 &vpx_highbd_lpf_vertical_8_dual_c, 10),
                      make_tuple(&vpx_highbd_lpf_vertical_8_dual_neon,
                                 &vpx_highbd_lpf_vertical_8_dual_c, 12)));
#else
INSTANTIATE_TEST_SUITE_P(
    NEON, Loop8Test6Param,
    ::testing::Values(
        make_tuple(&vpx_lpf_horizontal_16_neon, &vpx_lpf_horizontal_16_c, 8),
        make_tuple(&vpx_lpf_horizontal_16_dual_neon,
                   &vpx_lpf_horizontal_16_dual_c, 8),
        make_tuple(&vpx_lpf_vertical_16_neon, &vpx_lpf_vertical_16_c, 8),
        make_tuple(&vpx_lpf_vertical_16_dual_neon, &vpx_lpf_vertical_16_dual_c,
                   8),
        make_tuple(&vpx_lpf_horizontal_8_neon, &vpx_lpf_horizontal_8_c, 8),
        make_tuple(&vpx_lpf_vertical_8_neon, &vpx_lpf_vertical_8_c, 8),
        make_tuple(&vpx_lpf_horizontal_4_neon, &vpx_lpf_horizontal_4_c, 8),
        make_tuple(&vpx_lpf_vertical_4_neon, &vpx_lpf_vertical_4_c, 8)));
INSTANTIATE_TEST_SUITE_P(
    NEON, Loop8Test9Param,
    ::testing::Values(make_tuple(&vpx_lpf_horizontal_8_dual_neon,
                                 &vpx_lpf_horizontal_8_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_8_dual_neon,
                                 &vpx_lpf_vertical_8_dual_c, 8),
                      make_tuple(&vpx_lpf_horizontal_4_dual_neon,
                                 &vpx_lpf_horizontal_4_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_4_dual_neon,
                                 &vpx_lpf_vertical_4_dual_c, 8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // HAVE_NEON

#if HAVE_DSPR2 && !CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    DSPR2, Loop8Test6Param,
    ::testing::Values(
        make_tuple(&vpx_lpf_horizontal_4_dspr2, &vpx_lpf_horizontal_4_c, 8),
        make_tuple(&vpx_lpf_horizontal_8_dspr2, &vpx_lpf_horizontal_8_c, 8),
        make_tuple(&vpx_lpf_horizontal_16_dspr2, &vpx_lpf_horizontal_16_c, 8),
        make_tuple(&vpx_lpf_horizontal_16_dual_dspr2,
                   &vpx_lpf_horizontal_16_dual_c, 8),
        make_tuple(&vpx_lpf_vertical_4_dspr2, &vpx_lpf_vertical_4_c, 8),
        make_tuple(&vpx_lpf_vertical_8_dspr2, &vpx_lpf_vertical_8_c, 8),
        make_tuple(&vpx_lpf_vertical_16_dspr2, &vpx_lpf_vertical_16_c, 8),
        make_tuple(&vpx_lpf_vertical_16_dual_dspr2, &vpx_lpf_vertical_16_dual_c,
                   8)));

INSTANTIATE_TEST_SUITE_P(
    DSPR2, Loop8Test9Param,
    ::testing::Values(make_tuple(&vpx_lpf_horizontal_4_dual_dspr2,
                                 &vpx_lpf_horizontal_4_dual_c, 8),
                      make_tuple(&vpx_lpf_horizontal_8_dual_dspr2,
                                 &vpx_lpf_horizontal_8_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_4_dual_dspr2,
                                 &vpx_lpf_vertical_4_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_8_dual_dspr2,
                                 &vpx_lpf_vertical_8_dual_c, 8)));
#endif  // HAVE_DSPR2 && !CONFIG_VP9_HIGHBITDEPTH

#if HAVE_MSA && (!CONFIG_VP9_HIGHBITDEPTH)
INSTANTIATE_TEST_SUITE_P(
    MSA, Loop8Test6Param,
    ::testing::Values(
        make_tuple(&vpx_lpf_horizontal_4_msa, &vpx_lpf_horizontal_4_c, 8),
        make_tuple(&vpx_lpf_horizontal_8_msa, &vpx_lpf_horizontal_8_c, 8),
        make_tuple(&vpx_lpf_horizontal_16_msa, &vpx_lpf_horizontal_16_c, 8),
        make_tuple(&vpx_lpf_horizontal_16_dual_msa,
                   &vpx_lpf_horizontal_16_dual_c, 8),
        make_tuple(&vpx_lpf_vertical_4_msa, &vpx_lpf_vertical_4_c, 8),
        make_tuple(&vpx_lpf_vertical_8_msa, &vpx_lpf_vertical_8_c, 8),
        make_tuple(&vpx_lpf_vertical_16_msa, &vpx_lpf_vertical_16_c, 8)));

INSTANTIATE_TEST_SUITE_P(
    MSA, Loop8Test9Param,
    ::testing::Values(make_tuple(&vpx_lpf_horizontal_4_dual_msa,
                                 &vpx_lpf_horizontal_4_dual_c, 8),
                      make_tuple(&vpx_lpf_horizontal_8_dual_msa,
                                 &vpx_lpf_horizontal_8_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_4_dual_msa,
                                 &vpx_lpf_vertical_4_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_8_dual_msa,
                                 &vpx_lpf_vertical_8_dual_c, 8)));
#endif  // HAVE_MSA && (!CONFIG_VP9_HIGHBITDEPTH)

#if HAVE_LSX && (!CONFIG_VP9_HIGHBITDEPTH)
INSTANTIATE_TEST_SUITE_P(
    LSX, Loop8Test6Param,
    ::testing::Values(
        make_tuple(&vpx_lpf_horizontal_4_lsx, &vpx_lpf_horizontal_4_c, 8),
        make_tuple(&vpx_lpf_horizontal_8_lsx, &vpx_lpf_horizontal_8_c, 8),
        make_tuple(&vpx_lpf_horizontal_16_dual_lsx,
                   &vpx_lpf_horizontal_16_dual_c, 8),
        make_tuple(&vpx_lpf_vertical_4_lsx, &vpx_lpf_vertical_4_c, 8),
        make_tuple(&vpx_lpf_vertical_8_lsx, &vpx_lpf_vertical_8_c, 8),
        make_tuple(&vpx_lpf_vertical_16_dual_lsx, &vpx_lpf_vertical_16_dual_c,
                   8)));

INSTANTIATE_TEST_SUITE_P(
    LSX, Loop8Test9Param,
    ::testing::Values(make_tuple(&vpx_lpf_horizontal_4_dual_lsx,
                                 &vpx_lpf_horizontal_4_dual_c, 8),
                      make_tuple(&vpx_lpf_horizontal_8_dual_lsx,
                                 &vpx_lpf_horizontal_8_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_4_dual_lsx,
                                 &vpx_lpf_vertical_4_dual_c, 8),
                      make_tuple(&vpx_lpf_vertical_8_dual_lsx,
                                 &vpx_lpf_vertical_8_dual_c, 8)));
#endif  // HAVE_LSX && (!CONFIG_VP9_HIGHBITDEPTH)

}  // namespace
