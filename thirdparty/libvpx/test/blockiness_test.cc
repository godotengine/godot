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

#include "vpx_mem/vpx_mem.h"
#include "vp9/encoder/vp9_blockiness.h"

using libvpx_test::ACMRandom;

namespace {
class BlockinessTestBase : public ::testing::Test {
 public:
  BlockinessTestBase(int width, int height) : width_(width), height_(height) {}

  static void SetUpTestSuite() {
    source_data_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(kDataAlignment, kDataBufferSize));
    reference_data_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(kDataAlignment, kDataBufferSize));
  }

  static void TearDownTestSuite() {
    vpx_free(source_data_);
    source_data_ = nullptr;
    vpx_free(reference_data_);
    reference_data_ = nullptr;
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

  void FillConstant(uint8_t *data, int stride, uint8_t fill_constant, int width,
                    int height) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data[h * stride + w] = fill_constant;
      }
    }
  }

  void FillConstant(uint8_t *data, int stride, uint8_t fill_constant) {
    FillConstant(data, stride, fill_constant, width_, height_);
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

  void FillRandomBlocky(uint8_t *data, int stride) {
    for (int h = 0; h < height_; h += 4) {
      for (int w = 0; w < width_; w += 4) {
        FillRandom(data + h * stride + w, stride, 4, 4);
      }
    }
  }

  void FillCheckerboard(uint8_t *data, int stride) {
    for (int h = 0; h < height_; h += 4) {
      for (int w = 0; w < width_; w += 4) {
        if (((h / 4) ^ (w / 4)) & 1) {
          FillConstant(data + h * stride + w, stride, 255, 4, 4);
        } else {
          FillConstant(data + h * stride + w, stride, 0, 4, 4);
        }
      }
    }
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
  static uint8_t *source_data_;
  int source_stride_;
  static uint8_t *reference_data_;
  int reference_stride_;

  ACMRandom rnd_;
};

#if CONFIG_VP9_ENCODER
using BlockinessParam = std::tuple<int, int>;
class BlockinessVP9Test
    : public BlockinessTestBase,
      public ::testing::WithParamInterface<BlockinessParam> {
 public:
  BlockinessVP9Test() : BlockinessTestBase(GET_PARAM(0), GET_PARAM(1)) {}

 protected:
  double GetBlockiness() const {
    return vp9_get_blockiness(source_data_, source_stride_, reference_data_,
                              reference_stride_, width_, height_);
  }
};
#endif  // CONFIG_VP9_ENCODER

uint8_t *BlockinessTestBase::source_data_ = nullptr;
uint8_t *BlockinessTestBase::reference_data_ = nullptr;

#if CONFIG_VP9_ENCODER
TEST_P(BlockinessVP9Test, SourceBlockierThanReference) {
  // Source is blockier than reference.
  FillRandomBlocky(source_data_, source_stride_);
  FillConstant(reference_data_, reference_stride_, 128);
  const double super_blocky = GetBlockiness();

  EXPECT_DOUBLE_EQ(0.0, super_blocky)
      << "Blocky source should produce 0 blockiness.";
}

TEST_P(BlockinessVP9Test, ReferenceBlockierThanSource) {
  // Source is blockier than reference.
  FillConstant(source_data_, source_stride_, 128);
  FillRandomBlocky(reference_data_, reference_stride_);
  const double super_blocky = GetBlockiness();

  EXPECT_GT(super_blocky, 0.0)
      << "Blocky reference should score high for blockiness.";
}

TEST_P(BlockinessVP9Test, BlurringDecreasesBlockiness) {
  // Source is blockier than reference.
  FillConstant(source_data_, source_stride_, 128);
  FillRandomBlocky(reference_data_, reference_stride_);
  const double super_blocky = GetBlockiness();

  Blur(reference_data_, reference_stride_, 4);
  const double less_blocky = GetBlockiness();

  EXPECT_GT(super_blocky, less_blocky)
      << "A straight blur should decrease blockiness.";
}

TEST_P(BlockinessVP9Test, WorstCaseBlockiness) {
  // Source is blockier than reference.
  FillConstant(source_data_, source_stride_, 128);
  FillCheckerboard(reference_data_, reference_stride_);

  const double super_blocky = GetBlockiness();

  Blur(reference_data_, reference_stride_, 4);
  const double less_blocky = GetBlockiness();

  EXPECT_GT(super_blocky, less_blocky)
      << "A straight blur should decrease blockiness.";
}
#endif  // CONFIG_VP9_ENCODER

using std::make_tuple;

//------------------------------------------------------------------------------
// C functions

#if CONFIG_VP9_ENCODER
const BlockinessParam c_vp9_tests[] = { make_tuple(320, 240),
                                        make_tuple(318, 242),
                                        make_tuple(318, 238) };
INSTANTIATE_TEST_SUITE_P(C, BlockinessVP9Test,
                         ::testing::ValuesIn(c_vp9_tests));
#endif

}  // namespace
