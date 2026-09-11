/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <tuple>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#if CONFIG_VP9_ENCODER
#include "./vp9_rtcd.h"
#endif

#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vpx_dsp/ssim.h"
#include "vpx_mem/vpx_mem.h"

extern "C" double vpx_get_ssim_metrics(uint8_t *img1, int img1_pitch,
                                       uint8_t *img2, int img2_pitch, int width,
                                       int height, Ssimv *sv2, Metrics *m,
                                       int do_inconsistency);

using libvpx_test::ACMRandom;

namespace {
class ConsistencyTestBase : public ::testing::Test {
 public:
  ConsistencyTestBase(int width, int height) : width_(width), height_(height) {}

  static void SetUpTestSuite() {
    source_data_[0] = reinterpret_cast<uint8_t *>(
        vpx_memalign(kDataAlignment, kDataBufferSize));
    reference_data_[0] = reinterpret_cast<uint8_t *>(
        vpx_memalign(kDataAlignment, kDataBufferSize));
    source_data_[1] = reinterpret_cast<uint8_t *>(
        vpx_memalign(kDataAlignment, kDataBufferSize));
    reference_data_[1] = reinterpret_cast<uint8_t *>(
        vpx_memalign(kDataAlignment, kDataBufferSize));
    ssim_array_ = new Ssimv[kDataBufferSize / 16];
  }

  static void ClearSsim() { memset(ssim_array_, 0, kDataBufferSize / 16); }
  static void TearDownTestSuite() {
    vpx_free(source_data_[0]);
    source_data_[0] = nullptr;
    vpx_free(reference_data_[0]);
    reference_data_[0] = nullptr;
    vpx_free(source_data_[1]);
    source_data_[1] = nullptr;
    vpx_free(reference_data_[1]);
    reference_data_[1] = nullptr;

    delete[] ssim_array_;
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  // Handle frames up to 640x480
  static const int kDataAlignment = 16;
  static const int kDataBufferSize = 640 * 480;

  void SetUp() override {
    source_stride_ = (width_ + 31) & ~31;
    reference_stride_ = width_ * 2;
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  void FillRandom(uint8_t *data, int stride, int width, int height) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data[h * stride + w] = rnd_.Rand8();
      }
    }
  }

  void FillRandom(uint8_t *data, int stride) {
    FillRandom(data, stride, width_, height_);
  }

  void Copy(uint8_t *reference, uint8_t *source) {
    memcpy(reference, source, kDataBufferSize);
  }

  void Blur(uint8_t *data, int stride, int taps) {
    int sum = 0;
    int half_taps = taps / 2;
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < taps; ++w) {
        sum += data[w + h * stride];
      }
      for (int w = taps; w < width_; ++w) {
        sum += data[w + h * stride] - data[w - taps + h * stride];
        data[w - half_taps + h * stride] = (sum + half_taps) / taps;
      }
    }
    for (int w = 0; w < width_; ++w) {
      for (int h = 0; h < taps; ++h) {
        sum += data[h + w * stride];
      }
      for (int h = taps; h < height_; ++h) {
        sum += data[w + h * stride] - data[(h - taps) * stride + w];
        data[(h - half_taps) * stride + w] = (sum + half_taps) / taps;
      }
    }
  }
  int width_, height_;
  static uint8_t *source_data_[2];
  int source_stride_;
  static uint8_t *reference_data_[2];
  int reference_stride_;
  static Ssimv *ssim_array_;
  Metrics metrics_;

  ACMRandom rnd_;
};

#if CONFIG_VP9_ENCODER
using ConsistencyParam = std::tuple<int, int>;
class ConsistencyVP9Test
    : public ConsistencyTestBase,
      public ::testing::WithParamInterface<ConsistencyParam> {
 public:
  ConsistencyVP9Test() : ConsistencyTestBase(GET_PARAM(0), GET_PARAM(1)) {}

 protected:
  double CheckConsistency(int frame) {
    EXPECT_LT(frame, 2) << "Frame to check has to be less than 2.";
    return vpx_get_ssim_metrics(source_data_[frame], source_stride_,
                                reference_data_[frame], reference_stride_,
                                width_, height_, ssim_array_, &metrics_, 1);
  }
};
#endif  // CONFIG_VP9_ENCODER

uint8_t *ConsistencyTestBase::source_data_[2] = { nullptr, nullptr };
uint8_t *ConsistencyTestBase::reference_data_[2] = { nullptr, nullptr };
Ssimv *ConsistencyTestBase::ssim_array_ = nullptr;

#if CONFIG_VP9_ENCODER
TEST_P(ConsistencyVP9Test, ConsistencyIsZero) {
  FillRandom(source_data_[0], source_stride_);
  Copy(source_data_[1], source_data_[0]);
  Copy(reference_data_[0], source_data_[0]);
  Blur(reference_data_[0], reference_stride_, 3);
  Copy(reference_data_[1], source_data_[0]);
  Blur(reference_data_[1], reference_stride_, 3);

  double inconsistency = CheckConsistency(1);
  inconsistency = CheckConsistency(0);
  EXPECT_EQ(inconsistency, 0.0)
      << "Should have 0 inconsistency if they are exactly the same.";

  // If sources are not consistent reference frames inconsistency should
  // be less than if the source is consistent.
  FillRandom(source_data_[0], source_stride_);
  FillRandom(source_data_[1], source_stride_);
  FillRandom(reference_data_[0], reference_stride_);
  FillRandom(reference_data_[1], reference_stride_);
  CheckConsistency(0);
  inconsistency = CheckConsistency(1);

  Copy(source_data_[1], source_data_[0]);
  CheckConsistency(0);
  double inconsistency2 = CheckConsistency(1);
  EXPECT_LT(inconsistency, inconsistency2)
      << "Should have less inconsistency if source itself is inconsistent.";

  // Less of a blur should be less inconsistent than more blur coming off a
  // a frame with no blur.
  ClearSsim();
  FillRandom(source_data_[0], source_stride_);
  Copy(source_data_[1], source_data_[0]);
  Copy(reference_data_[0], source_data_[0]);
  Copy(reference_data_[1], source_data_[0]);
  Blur(reference_data_[1], reference_stride_, 4);
  CheckConsistency(0);
  inconsistency = CheckConsistency(1);
  ClearSsim();
  Copy(reference_data_[1], source_data_[0]);
  Blur(reference_data_[1], reference_stride_, 8);
  CheckConsistency(0);
  inconsistency2 = CheckConsistency(1);

  EXPECT_LT(inconsistency, inconsistency2)
      << "Stronger Blur should produce more inconsistency.";
}
#endif  // CONFIG_VP9_ENCODER

using std::make_tuple;

//------------------------------------------------------------------------------
// C functions

#if CONFIG_VP9_ENCODER
const ConsistencyParam c_vp9_tests[] = { make_tuple(320, 240),
                                         make_tuple(318, 242),
                                         make_tuple(318, 238) };
INSTANTIATE_TEST_SUITE_P(C, ConsistencyVP9Test,
                         ::testing::ValuesIn(c_vp9_tests));
#endif

}  // namespace
