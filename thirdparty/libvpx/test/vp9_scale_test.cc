/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "gtest/gtest.h"

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "./vpx_scale_rtcd.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/vpx_scale_test.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/vpx_timer.h"
#include "vpx_scale/yv12config.h"

namespace libvpx_test {

using ScaleFrameFunc = void (*)(const YV12_BUFFER_CONFIG *src,
                                YV12_BUFFER_CONFIG *dst,
                                INTERP_FILTER filter_type, int phase_scaler);

class ScaleTest : public VpxScaleBase,
                  public ::testing::TestWithParam<ScaleFrameFunc> {
 public:
  ~ScaleTest() override = default;

 protected:
  void SetUp() override { scale_fn_ = GetParam(); }

  void ReferenceScaleFrame(INTERP_FILTER filter_type, int phase_scaler) {
    vp9_scale_and_extend_frame_c(&img_, &ref_img_, filter_type, phase_scaler);
  }

  void ScaleFrame(INTERP_FILTER filter_type, int phase_scaler) {
    ASM_REGISTER_STATE_CHECK(
        scale_fn_(&img_, &dst_img_, filter_type, phase_scaler));
  }

  void RunTest(INTERP_FILTER filter_type) {
    static const int kNumSizesToTest = 22;
    static const int kNumScaleFactorsToTest = 4;
    static const int kSizesToTest[] = { 1,  2,  3,  4,  6,   8,  10, 12,
                                        14, 16, 18, 20, 22,  24, 26, 28,
                                        30, 32, 34, 68, 128, 134 };
    static const int kScaleFactors[] = { 1, 2, 3, 4 };
    for (int phase_scaler = 0; phase_scaler < 16; ++phase_scaler) {
      for (int h = 0; h < kNumSizesToTest; ++h) {
        const int src_height = kSizesToTest[h];
        for (int w = 0; w < kNumSizesToTest; ++w) {
          const int src_width = kSizesToTest[w];
          for (int sf_up_idx = 0; sf_up_idx < kNumScaleFactorsToTest;
               ++sf_up_idx) {
            const int sf_up = kScaleFactors[sf_up_idx];
            for (int sf_down_idx = 0; sf_down_idx < kNumScaleFactorsToTest;
                 ++sf_down_idx) {
              const int sf_down = kScaleFactors[sf_down_idx];
              const int dst_width = src_width * sf_up / sf_down;
              const int dst_height = src_height * sf_up / sf_down;
              if (sf_up == sf_down && sf_up != 1) {
                continue;
              }
              // I420 frame width and height must be even.
              if (!dst_width || !dst_height || dst_width & 1 ||
                  dst_height & 1) {
                continue;
              }
              // vpx_convolve8_c() has restriction on the step which cannot
              // exceed 64 (ratio 1 to 4).
              if (src_width > 4 * dst_width || src_height > 4 * dst_height) {
                continue;
              }
              ASSERT_NO_FATAL_FAILURE(ResetScaleImages(src_width, src_height,
                                                       dst_width, dst_height));
              ReferenceScaleFrame(filter_type, phase_scaler);
              ScaleFrame(filter_type, phase_scaler);
              if (memcmp(dst_img_.buffer_alloc, ref_img_.buffer_alloc,
                         ref_img_.frame_size)) {
                printf(
                    "filter_type = %d, phase_scaler = %d, src_width = %4d, "
                    "src_height = %4d, dst_width = %4d, dst_height = %4d, "
                    "scale factor = %d:%d\n",
                    filter_type, phase_scaler, src_width, src_height, dst_width,
                    dst_height, sf_down, sf_up);
                PrintDiff();
              }
              CompareImages(dst_img_);
              DeallocScaleImages();
            }
          }
        }
      }
    }
  }

  void PrintDiffComponent(const uint8_t *const ref, const uint8_t *const opt,
                          const int stride, const int width, const int height,
                          const int plane_idx) const {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (ref[y * stride + x] != opt[y * stride + x]) {
          printf("Plane %d pixel[%d][%d] diff:%6d (ref),%6d (opt)\n", plane_idx,
                 y, x, ref[y * stride + x], opt[y * stride + x]);
          break;
        }
      }
    }
  }

  void PrintDiff() const {
    assert(ref_img_.y_stride == dst_img_.y_stride);
    assert(ref_img_.y_width == dst_img_.y_width);
    assert(ref_img_.y_height == dst_img_.y_height);
    assert(ref_img_.uv_stride == dst_img_.uv_stride);
    assert(ref_img_.uv_width == dst_img_.uv_width);
    assert(ref_img_.uv_height == dst_img_.uv_height);

    if (memcmp(dst_img_.buffer_alloc, ref_img_.buffer_alloc,
               ref_img_.frame_size)) {
      PrintDiffComponent(ref_img_.y_buffer, dst_img_.y_buffer,
                         ref_img_.y_stride, ref_img_.y_width, ref_img_.y_height,
                         0);
      PrintDiffComponent(ref_img_.u_buffer, dst_img_.u_buffer,
                         ref_img_.uv_stride, ref_img_.uv_width,
                         ref_img_.uv_height, 1);
      PrintDiffComponent(ref_img_.v_buffer, dst_img_.v_buffer,
                         ref_img_.uv_stride, ref_img_.uv_width,
                         ref_img_.uv_height, 2);
    }
  }

  ScaleFrameFunc scale_fn_;
};

TEST_P(ScaleTest, ScaleFrame_EightTap) { RunTest(EIGHTTAP); }
TEST_P(ScaleTest, ScaleFrame_EightTapSmooth) { RunTest(EIGHTTAP_SMOOTH); }
TEST_P(ScaleTest, ScaleFrame_EightTapSharp) { RunTest(EIGHTTAP_SHARP); }
TEST_P(ScaleTest, ScaleFrame_Bilinear) { RunTest(BILINEAR); }

TEST_P(ScaleTest, DISABLED_Speed) {
  static const int kCountSpeedTestBlock = 100;
  static const int kNumScaleFactorsToTest = 4;
  static const int kScaleFactors[] = { 1, 2, 3, 4 };
  const int src_width = 1280;
  const int src_height = 720;
  for (INTERP_FILTER filter_type = 2; filter_type < 4; ++filter_type) {
    for (int phase_scaler = 0; phase_scaler < 2; ++phase_scaler) {
      for (int sf_up_idx = 0; sf_up_idx < kNumScaleFactorsToTest; ++sf_up_idx) {
        const int sf_up = kScaleFactors[sf_up_idx];
        for (int sf_down_idx = 0; sf_down_idx < kNumScaleFactorsToTest;
             ++sf_down_idx) {
          const int sf_down = kScaleFactors[sf_down_idx];
          const int dst_width = src_width * sf_up / sf_down;
          const int dst_height = src_height * sf_up / sf_down;
          if (sf_up == sf_down && sf_up != 1) {
            continue;
          }
          // I420 frame width and height must be even.
          if (dst_width & 1 || dst_height & 1) {
            continue;
          }
          ASSERT_NO_FATAL_FAILURE(
              ResetScaleImages(src_width, src_height, dst_width, dst_height));
          ASM_REGISTER_STATE_CHECK(
              ReferenceScaleFrame(filter_type, phase_scaler));

          vpx_usec_timer timer;
          vpx_usec_timer_start(&timer);
          for (int i = 0; i < kCountSpeedTestBlock; ++i) {
            ScaleFrame(filter_type, phase_scaler);
          }
          libvpx_test::ClearSystemState();
          vpx_usec_timer_mark(&timer);
          const int elapsed_time =
              static_cast<int>(vpx_usec_timer_elapsed(&timer) / 1000);
          CompareImages(dst_img_);
          DeallocScaleImages();

          printf(
              "filter_type = %d, phase_scaler = %d, src_width = %4d, "
              "src_height = %4d, dst_width = %4d, dst_height = %4d, "
              "scale factor = %d:%d, scale time: %5d ms\n",
              filter_type, phase_scaler, src_width, src_height, dst_width,
              dst_height, sf_down, sf_up, elapsed_time);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(C, ScaleTest,
                         ::testing::Values(vp9_scale_and_extend_frame_c));

#if HAVE_SSSE3
INSTANTIATE_TEST_SUITE_P(SSSE3, ScaleTest,
                         ::testing::Values(vp9_scale_and_extend_frame_ssse3));
#endif  // HAVE_SSSE3

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, ScaleTest,
                         ::testing::Values(vp9_scale_and_extend_frame_neon));
#endif  // HAVE_NEON

}  // namespace libvpx_test
