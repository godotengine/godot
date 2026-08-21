/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <cstdio>
#include <tuple>

#include "gtest/gtest.h"

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/bench.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vp9/common/vp9_blockd.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/vpx_timer.h"

using SubtractFunc = void (*)(int rows, int cols, int16_t *diff_ptr,
                              ptrdiff_t diff_stride, const uint8_t *src_ptr,
                              ptrdiff_t src_stride, const uint8_t *pred_ptr,
                              ptrdiff_t pred_stride);

namespace vp9 {

class VP9SubtractBlockTest : public AbstractBench,
                             public ::testing::TestWithParam<SubtractFunc> {
 public:
  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  void Run() override {
    GetParam()(block_height_, block_width_, diff_, block_width_, src_,
               block_width_, pred_, block_width_);
  }

  void SetupBlocks(BLOCK_SIZE bsize) {
    block_width_ = 4 * num_4x4_blocks_wide_lookup[bsize];
    block_height_ = 4 * num_4x4_blocks_high_lookup[bsize];
    diff_ = reinterpret_cast<int16_t *>(
        vpx_memalign(16, sizeof(*diff_) * block_width_ * block_height_ * 2));
    pred_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(16, block_width_ * block_height_ * 2));
    src_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(16, block_width_ * block_height_ * 2));
  }

  int block_width_;
  int block_height_;
  int16_t *diff_;
  uint8_t *pred_;
  uint8_t *src_;
};

using libvpx_test::ACMRandom;

TEST_P(VP9SubtractBlockTest, DISABLED_Speed) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());

  for (BLOCK_SIZE bsize = BLOCK_4X4; bsize < BLOCK_SIZES;
       bsize = static_cast<BLOCK_SIZE>(static_cast<int>(bsize) + 1)) {
    SetupBlocks(bsize);

    RunNTimes(100000000 / (block_height_ * block_width_));
    char block_size[16];
    snprintf(block_size, sizeof(block_size), "%dx%d", block_height_,
             block_width_);
    char title[100];
    snprintf(title, sizeof(title), "%8s ", block_size);
    PrintMedian(title);

    vpx_free(diff_);
    vpx_free(pred_);
    vpx_free(src_);
  }
}

TEST_P(VP9SubtractBlockTest, SimpleSubtract) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());

  for (BLOCK_SIZE bsize = BLOCK_4X4; bsize < BLOCK_SIZES;
       bsize = static_cast<BLOCK_SIZE>(static_cast<int>(bsize) + 1)) {
    SetupBlocks(bsize);

    for (int n = 0; n < 100; n++) {
      for (int r = 0; r < block_height_; ++r) {
        for (int c = 0; c < block_width_ * 2; ++c) {
          src_[r * block_width_ * 2 + c] = rnd.Rand8();
          pred_[r * block_width_ * 2 + c] = rnd.Rand8();
        }
      }

      GetParam()(block_height_, block_width_, diff_, block_width_, src_,
                 block_width_, pred_, block_width_);

      for (int r = 0; r < block_height_; ++r) {
        for (int c = 0; c < block_width_; ++c) {
          EXPECT_EQ(diff_[r * block_width_ + c],
                    (src_[r * block_width_ + c] - pred_[r * block_width_ + c]))
              << "r = " << r << ", c = " << c
              << ", bs = " << static_cast<int>(bsize);
        }
      }

      GetParam()(block_height_, block_width_, diff_, block_width_ * 2, src_,
                 block_width_ * 2, pred_, block_width_ * 2);

      for (int r = 0; r < block_height_; ++r) {
        for (int c = 0; c < block_width_; ++c) {
          EXPECT_EQ(diff_[r * block_width_ * 2 + c],
                    (src_[r * block_width_ * 2 + c] -
                     pred_[r * block_width_ * 2 + c]))
              << "r = " << r << ", c = " << c
              << ", bs = " << static_cast<int>(bsize);
        }
      }
    }
    vpx_free(diff_);
    vpx_free(pred_);
    vpx_free(src_);
  }
}

INSTANTIATE_TEST_SUITE_P(C, VP9SubtractBlockTest,
                         ::testing::Values(vpx_subtract_block_c));

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, VP9SubtractBlockTest,
                         ::testing::Values(vpx_subtract_block_sse2));
#endif
#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(AVX2, VP9SubtractBlockTest,
                         ::testing::Values(vpx_subtract_block_avx2));
#endif
#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, VP9SubtractBlockTest,
                         ::testing::Values(vpx_subtract_block_neon));
#endif
#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(MSA, VP9SubtractBlockTest,
                         ::testing::Values(vpx_subtract_block_msa));
#endif

#if HAVE_MMI
INSTANTIATE_TEST_SUITE_P(MMI, VP9SubtractBlockTest,
                         ::testing::Values(vpx_subtract_block_mmi));
#endif

#if HAVE_VSX
INSTANTIATE_TEST_SUITE_P(VSX, VP9SubtractBlockTest,
                         ::testing::Values(vpx_subtract_block_vsx));
#endif

#if HAVE_LSX
INSTANTIATE_TEST_SUITE_P(LSX, VP9SubtractBlockTest,
                         ::testing::Values(vpx_subtract_block_lsx));
#endif

#if CONFIG_VP9_HIGHBITDEPTH

using HBDSubtractFunc = void (*)(int rows, int cols, int16_t *diff_ptr,
                                 ptrdiff_t diff_stride, const uint8_t *src_ptr,
                                 ptrdiff_t src_stride, const uint8_t *pred_ptr,
                                 ptrdiff_t pred_stride, int bd);

// <BLOCK_SIZE, bit_depth, optimized subtract func, reference subtract func>
using Params = std::tuple<BLOCK_SIZE, int, HBDSubtractFunc, HBDSubtractFunc>;

class VPXHBDSubtractBlockTest : public ::testing::TestWithParam<Params> {
 public:
  void SetUp() override {
    block_width_ = 4 * num_4x4_blocks_wide_lookup[GET_PARAM(0)];
    block_height_ = 4 * num_4x4_blocks_high_lookup[GET_PARAM(0)];
    bit_depth_ = static_cast<vpx_bit_depth_t>(GET_PARAM(1));
    func_ = GET_PARAM(2);
    ref_func_ = GET_PARAM(3);

    rnd_.Reset(ACMRandom::DeterministicSeed());

    constexpr size_t kMaxWidth = 128;
    constexpr size_t kMaxBlockSize = kMaxWidth * kMaxWidth;
    src_ = CONVERT_TO_BYTEPTR(reinterpret_cast<uint16_t *>(
        vpx_memalign(16, kMaxBlockSize * sizeof(uint16_t))));
    ASSERT_NE(src_, nullptr);
    pred_ = CONVERT_TO_BYTEPTR(reinterpret_cast<uint16_t *>(
        vpx_memalign(16, kMaxBlockSize * sizeof(uint16_t))));
    ASSERT_NE(pred_, nullptr);
    diff_ = reinterpret_cast<int16_t *>(
        vpx_memalign(16, kMaxBlockSize * sizeof(int16_t)));
    ASSERT_NE(diff_, nullptr);
  }

  void TearDown() override {
    vpx_free(CONVERT_TO_SHORTPTR(src_));
    vpx_free(CONVERT_TO_SHORTPTR(pred_));
    vpx_free(diff_);
  }

 protected:
  void CheckResult();
  void RunForSpeed();

 private:
  ACMRandom rnd_;
  int block_height_;
  int block_width_;
  vpx_bit_depth_t bit_depth_;
  HBDSubtractFunc func_;
  HBDSubtractFunc ref_func_;
  uint8_t *src_;
  uint8_t *pred_;
  int16_t *diff_;
};

void VPXHBDSubtractBlockTest::CheckResult() {
  constexpr int kTestNum = 100;
  constexpr int kMaxWidth = 128;
  constexpr int kMaxBlockSize = kMaxWidth * kMaxWidth;
  const int mask = (1 << bit_depth_) - 1;
  for (int i = 0; i < kTestNum; ++i) {
    for (int j = 0; j < kMaxBlockSize; ++j) {
      CONVERT_TO_SHORTPTR(src_)[j] = rnd_.Rand16() & mask;
      CONVERT_TO_SHORTPTR(pred_)[j] = rnd_.Rand16() & mask;
    }

    func_(block_height_, block_width_, diff_, block_width_, src_, block_width_,
          pred_, block_width_, bit_depth_);

    for (int r = 0; r < block_height_; ++r) {
      for (int c = 0; c < block_width_; ++c) {
        EXPECT_EQ(diff_[r * block_width_ + c],
                  (CONVERT_TO_SHORTPTR(src_)[r * block_width_ + c] -
                   CONVERT_TO_SHORTPTR(pred_)[r * block_width_ + c]))
            << "r = " << r << ", c = " << c << ", test: " << i;
      }
    }
  }
}

TEST_P(VPXHBDSubtractBlockTest, CheckResult) { CheckResult(); }

void VPXHBDSubtractBlockTest::RunForSpeed() {
  constexpr int kTestNum = 200000;
  constexpr int kMaxWidth = 128;
  constexpr int kMaxBlockSize = kMaxWidth * kMaxWidth;
  const int mask = (1 << bit_depth_) - 1;

  if (ref_func_ == func_) GTEST_SKIP();

  for (int j = 0; j < kMaxBlockSize; ++j) {
    CONVERT_TO_SHORTPTR(src_)[j] = rnd_.Rand16() & mask;
    CONVERT_TO_SHORTPTR(pred_)[j] = rnd_.Rand16() & mask;
  }

  vpx_usec_timer ref_timer;
  vpx_usec_timer_start(&ref_timer);
  for (int i = 0; i < kTestNum; ++i) {
    ref_func_(block_height_, block_width_, diff_, block_width_, src_,
              block_width_, pred_, block_width_, bit_depth_);
  }
  vpx_usec_timer_mark(&ref_timer);
  const int64_t ref_elapsed_time = vpx_usec_timer_elapsed(&ref_timer);

  for (int j = 0; j < kMaxBlockSize; ++j) {
    CONVERT_TO_SHORTPTR(src_)[j] = rnd_.Rand16() & mask;
    CONVERT_TO_SHORTPTR(pred_)[j] = rnd_.Rand16() & mask;
  }

  vpx_usec_timer timer;
  vpx_usec_timer_start(&timer);
  for (int i = 0; i < kTestNum; ++i) {
    func_(block_height_, block_width_, diff_, block_width_, src_, block_width_,
          pred_, block_width_, bit_depth_);
  }
  vpx_usec_timer_mark(&timer);
  const int64_t elapsed_time = vpx_usec_timer_elapsed(&timer);

  printf(
      "[%dx%d]: "
      "ref_time=%6" PRId64 " \t simd_time=%6" PRId64
      " \t "
      "gain=%f \n",
      block_width_, block_height_, ref_elapsed_time, elapsed_time,
      static_cast<double>(ref_elapsed_time) /
          static_cast<double>(elapsed_time));
}

TEST_P(VPXHBDSubtractBlockTest, DISABLED_Speed) { RunForSpeed(); }

const BLOCK_SIZE kValidBlockSize[] = { BLOCK_4X4,   BLOCK_4X8,   BLOCK_8X4,
                                       BLOCK_8X8,   BLOCK_8X16,  BLOCK_16X8,
                                       BLOCK_16X16, BLOCK_16X32, BLOCK_32X16,
                                       BLOCK_32X32, BLOCK_32X64, BLOCK_64X32,
                                       BLOCK_64X64 };

INSTANTIATE_TEST_SUITE_P(
    C, VPXHBDSubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(12),
                       ::testing::Values(&vpx_highbd_subtract_block_c),
                       ::testing::Values(&vpx_highbd_subtract_block_c)));

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, VPXHBDSubtractBlockTest,
    ::testing::Combine(::testing::ValuesIn(kValidBlockSize),
                       ::testing::Values(12),
                       ::testing::Values(&vpx_highbd_subtract_block_avx2),
                       ::testing::Values(&vpx_highbd_subtract_block_c)));
#endif  // HAVE_AVX2

#endif  // CONFIG_VP9_HIGHBITDEPTH
}  // namespace vp9
