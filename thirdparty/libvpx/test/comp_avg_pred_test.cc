/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "gtest/gtest.h"

#include "./vpx_dsp_rtcd.h"

#include "test/acm_random.h"
#include "test/buffer.h"
#include "test/register_state_check.h"
#include "vpx_config.h"
#include "vpx_ports/vpx_timer.h"

namespace {

using ::libvpx_test::ACMRandom;
using ::libvpx_test::Buffer;

template <typename Pixel>
Pixel avg_with_rounding(Pixel a, Pixel b) {
  return (a + b + 1) >> 1;
}

template <typename Pixel>
void reference_pred(const Buffer<Pixel> &pred, const Buffer<Pixel> &ref,
                    int width, int height, Buffer<Pixel> *avg) {
  ASSERT_NE(avg->TopLeftPixel(), nullptr);
  ASSERT_NE(pred.TopLeftPixel(), nullptr);
  ASSERT_NE(ref.TopLeftPixel(), nullptr);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      avg->TopLeftPixel()[y * avg->stride() + x] =
          avg_with_rounding<Pixel>(pred.TopLeftPixel()[y * pred.stride() + x],
                                   ref.TopLeftPixel()[y * ref.stride() + x]);
    }
  }
}

using AvgPredFunc = void (*)(uint8_t *a, const uint8_t *b, int w, int h,
                             const uint8_t *c, int c_stride);

template <int bitdepth, typename Pixel>
class AvgPredTest : public ::testing::TestWithParam<AvgPredFunc> {
 public:
  void SetUp() override {
    avg_pred_func_ = GetParam();
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  void TestSizeCombinations();
  void TestCompareReferenceRandom();
  void TestSpeed();

 protected:
  AvgPredFunc avg_pred_func_;
  ACMRandom rnd_;
};

template <int bitdepth, typename Pixel>
void AvgPredTest<bitdepth, Pixel>::TestSizeCombinations() {
  // This is called as part of the sub pixel variance. As such it must be one of
  // the variance block sizes.
  for (int width_pow = 2; width_pow <= 6; ++width_pow) {
    for (int height_pow = width_pow - 1; height_pow <= width_pow + 1;
         ++height_pow) {
      // Don't test 4x2 or 64x128
      if (height_pow == 1 || height_pow == 7) continue;

      // The sse2 special-cases when ref width == stride, so make sure to test
      // it.
      for (int ref_padding = 0; ref_padding < 2; ref_padding++) {
        const int width = 1 << width_pow;
        const int height = 1 << height_pow;
        // Only the reference buffer may have a stride not equal to width.
        Buffer<Pixel> ref = Buffer<Pixel>(width, height, ref_padding ? 8 : 0);
        ASSERT_TRUE(ref.Init());
        Buffer<Pixel> pred = Buffer<Pixel>(width, height, 0, 32);
        ASSERT_TRUE(pred.Init());
        Buffer<Pixel> avg_ref = Buffer<Pixel>(width, height, 0, 32);
        ASSERT_TRUE(avg_ref.Init());
        Buffer<Pixel> avg_chk = Buffer<Pixel>(width, height, 0, 32);
        ASSERT_TRUE(avg_chk.Init());
        const int bitdepth_mask = (1 << bitdepth) - 1;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            ref.TopLeftPixel()[w + h * width] = rnd_.Rand16() & bitdepth_mask;
          }
        }
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            pred.TopLeftPixel()[w + h * width] = rnd_.Rand16() & bitdepth_mask;
          }
        }

        reference_pred<Pixel>(pred, ref, width, height, &avg_ref);
        ASM_REGISTER_STATE_CHECK(avg_pred_func_(
            (uint8_t *)avg_chk.TopLeftPixel(), (uint8_t *)pred.TopLeftPixel(),
            width, height, (uint8_t *)ref.TopLeftPixel(), ref.stride()));

        EXPECT_TRUE(avg_chk.CheckValues(avg_ref));
        if (HasFailure()) {
          printf("Width: %d Height: %d\n", width, height);
          avg_chk.PrintDifference(avg_ref);
          return;
        }
      }
    }
  }
}

template <int bitdepth, typename Pixel>
void AvgPredTest<bitdepth, Pixel>::TestCompareReferenceRandom() {
  const int width = 64;
  const int height = 32;
  Buffer<Pixel> ref = Buffer<Pixel>(width, height, 8);
  ASSERT_TRUE(ref.Init());
  Buffer<Pixel> pred = Buffer<Pixel>(width, height, 0, 32);
  ASSERT_TRUE(pred.Init());
  Buffer<Pixel> avg_ref = Buffer<Pixel>(width, height, 0, 32);
  ASSERT_TRUE(avg_ref.Init());
  Buffer<Pixel> avg_chk = Buffer<Pixel>(width, height, 0, 32);
  ASSERT_TRUE(avg_chk.Init());

  for (int i = 0; i < 500; ++i) {
    const int bitdepth_mask = (1 << bitdepth) - 1;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        ref.TopLeftPixel()[w + h * width] = rnd_.Rand16() & bitdepth_mask;
      }
    }
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        pred.TopLeftPixel()[w + h * width] = rnd_.Rand16() & bitdepth_mask;
      }
    }

    reference_pred<Pixel>(pred, ref, width, height, &avg_ref);
    ASM_REGISTER_STATE_CHECK(avg_pred_func_(
        (uint8_t *)avg_chk.TopLeftPixel(), (uint8_t *)pred.TopLeftPixel(),
        width, height, (uint8_t *)ref.TopLeftPixel(), ref.stride()));
    EXPECT_TRUE(avg_chk.CheckValues(avg_ref));
    if (HasFailure()) {
      printf("Width: %d Height: %d\n", width, height);
      avg_chk.PrintDifference(avg_ref);
      return;
    }
  }
}

template <int bitdepth, typename Pixel>
void AvgPredTest<bitdepth, Pixel>::TestSpeed() {
  for (int width_pow = 2; width_pow <= 6; ++width_pow) {
    for (int height_pow = width_pow - 1; height_pow <= width_pow + 1;
         ++height_pow) {
      // Don't test 4x2 or 64x128
      if (height_pow == 1 || height_pow == 7) continue;

      for (int ref_padding = 0; ref_padding < 2; ref_padding++) {
        const int width = 1 << width_pow;
        const int height = 1 << height_pow;
        Buffer<Pixel> ref = Buffer<Pixel>(width, height, ref_padding ? 8 : 0);
        ASSERT_TRUE(ref.Init());
        Buffer<Pixel> pred = Buffer<Pixel>(width, height, 0, 32);
        ASSERT_TRUE(pred.Init());
        Buffer<Pixel> avg = Buffer<Pixel>(width, height, 0, 32);
        ASSERT_TRUE(avg.Init());
        const int bitdepth_mask = (1 << bitdepth) - 1;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            ref.TopLeftPixel()[w + h * width] = rnd_.Rand16() & bitdepth_mask;
          }
        }
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            pred.TopLeftPixel()[w + h * width] = rnd_.Rand16() & bitdepth_mask;
          }
        }

        vpx_usec_timer timer;
        vpx_usec_timer_start(&timer);
        for (int i = 0; i < 100000000 / (width * height); ++i) {
          avg_pred_func_((uint8_t *)avg.TopLeftPixel(),
                         (uint8_t *)pred.TopLeftPixel(), width, height,
                         (uint8_t *)ref.TopLeftPixel(), ref.stride());
        }
        vpx_usec_timer_mark(&timer);

        const int elapsed_time =
            static_cast<int>(vpx_usec_timer_elapsed(&timer));
        printf("Average Test (ref_padding: %d) %dx%d time: %5d us\n",
               ref_padding, width, height, elapsed_time);
      }
    }
  }
}

using AvgPredTestLBD = AvgPredTest<8, uint8_t>;

TEST_P(AvgPredTestLBD, SizeCombinations) { TestSizeCombinations(); }

TEST_P(AvgPredTestLBD, CompareReferenceRandom) { TestCompareReferenceRandom(); }

TEST_P(AvgPredTestLBD, DISABLED_Speed) { TestSpeed(); }

INSTANTIATE_TEST_SUITE_P(C, AvgPredTestLBD,
                         ::testing::Values(&vpx_comp_avg_pred_c));

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, AvgPredTestLBD,
                         ::testing::Values(&vpx_comp_avg_pred_sse2));
#endif  // HAVE_SSE2

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(AVX2, AvgPredTestLBD,
                         ::testing::Values(&vpx_comp_avg_pred_avx2));
#endif  // HAVE_AVX2

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, AvgPredTestLBD,
                         ::testing::Values(&vpx_comp_avg_pred_neon));
#endif  // HAVE_NEON

#if HAVE_VSX
INSTANTIATE_TEST_SUITE_P(VSX, AvgPredTestLBD,
                         ::testing::Values(&vpx_comp_avg_pred_vsx));
#endif  // HAVE_VSX

#if HAVE_LSX
INSTANTIATE_TEST_SUITE_P(LSX, AvgPredTestLBD,
                         ::testing::Values(&vpx_comp_avg_pred_lsx));
#endif  // HAVE_LSX

#if CONFIG_VP9_HIGHBITDEPTH
using HighbdAvgPredFunc = void (*)(uint16_t *a, const uint16_t *b, int w, int h,
                                   const uint16_t *c, int c_stride);

template <HighbdAvgPredFunc fn>
void highbd_wrapper(uint8_t *a, const uint8_t *b, int w, int h,
                    const uint8_t *c, int c_stride) {
  fn((uint16_t *)a, (const uint16_t *)b, w, h, (const uint16_t *)c, c_stride);
}

using AvgPredTestHBD = AvgPredTest<12, uint16_t>;

TEST_P(AvgPredTestHBD, SizeCombinations) { TestSizeCombinations(); }

TEST_P(AvgPredTestHBD, CompareReferenceRandom) { TestCompareReferenceRandom(); }

TEST_P(AvgPredTestHBD, DISABLED_Speed) { TestSpeed(); }

INSTANTIATE_TEST_SUITE_P(
    C, AvgPredTestHBD,
    ::testing::Values(&highbd_wrapper<vpx_highbd_comp_avg_pred_c>));

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, AvgPredTestHBD,
    ::testing::Values(&highbd_wrapper<vpx_highbd_comp_avg_pred_sse2>));
#endif  // HAVE_SSE2

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, AvgPredTestHBD,
    ::testing::Values(&highbd_wrapper<vpx_highbd_comp_avg_pred_neon>));
#endif  // HAVE_NEON

#endif  // CONFIG_VP9_HIGHBITDEPTH
}  // namespace
