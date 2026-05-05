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

#include <memory>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "gtest/gtest.h"
#include "test/acm_random.h"
#include "test/bench.h"
#include "test/buffer.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"

using libvpx_test::ACMRandom;
using libvpx_test::Buffer;

using VpxPostProcDownAndAcrossMbRowFunc = void (*)(
    unsigned char *src_ptr, unsigned char *dst_ptr, int src_pixels_per_line,
    int dst_pixels_per_line, int cols, unsigned char *flimit, int size);

using VpxMbPostProcAcrossIpFunc = void (*)(unsigned char *src, int pitch,
                                           int rows, int cols, int flimit);

using VpxMbPostProcDownFunc = void (*)(unsigned char *dst, int pitch, int rows,
                                       int cols, int flimit);

namespace {
// Compute the filter level used in post proc from the loop filter strength
int q2mbl(int x) {
  if (x < 20) x = 20;

  x = 50 + (x - 50) * 10 / 8;
  return x * x / 3;
}

class VpxPostProcDownAndAcrossMbRowTest
    : public AbstractBench,
      public ::testing::TestWithParam<VpxPostProcDownAndAcrossMbRowFunc> {
 public:
  VpxPostProcDownAndAcrossMbRowTest()
      : mb_post_proc_down_and_across_(GetParam()) {}
  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  void Run() override;

  const VpxPostProcDownAndAcrossMbRowFunc mb_post_proc_down_and_across_;
  // Size of the underlying data block that will be filtered.
  int block_width_;
  int block_height_;
  Buffer<uint8_t> *src_image_;
  Buffer<uint8_t> *dst_image_;
  uint8_t *flimits_;
};

void VpxPostProcDownAndAcrossMbRowTest::Run() {
  mb_post_proc_down_and_across_(
      src_image_->TopLeftPixel(), dst_image_->TopLeftPixel(),
      src_image_->stride(), dst_image_->stride(), block_width_, flimits_, 16);
}

// Test routine for the VPx post-processing function
// vpx_post_proc_down_and_across_mb_row_c.

TEST_P(VpxPostProcDownAndAcrossMbRowTest, CheckFilterOutput) {
  // Size of the underlying data block that will be filtered.
  block_width_ = 16;
  block_height_ = 16;

  // 5-tap filter needs 2 padding rows above and below the block in the input.
  Buffer<uint8_t> src_image = Buffer<uint8_t>(block_width_, block_height_, 2);
  ASSERT_TRUE(src_image.Init());

  // Filter extends output block by 8 samples at left and right edges.
  // Though the left padding is only 8 bytes, the assembly code tries to
  // read 16 bytes before the pointer.
  Buffer<uint8_t> dst_image =
      Buffer<uint8_t>(block_width_, block_height_, 8, 16, 8, 8);
  ASSERT_TRUE(dst_image.Init());

  flimits_ = reinterpret_cast<uint8_t *>(vpx_memalign(16, block_width_));
  (void)memset(flimits_, 255, block_width_);

  // Initialize pixels in the input:
  //   block pixels to value 1,
  //   border pixels to value 10.
  src_image.SetPadding(10);
  src_image.Set(1);

  // Initialize pixels in the output to 99.
  dst_image.Set(99);

  ASM_REGISTER_STATE_CHECK(mb_post_proc_down_and_across_(
      src_image.TopLeftPixel(), dst_image.TopLeftPixel(), src_image.stride(),
      dst_image.stride(), block_width_, flimits_, 16));

  static const uint8_t kExpectedOutput[] = { 4, 3, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 3, 4 };

  uint8_t *pixel_ptr = dst_image.TopLeftPixel();
  for (int i = 0; i < block_height_; ++i) {
    for (int j = 0; j < block_width_; ++j) {
      ASSERT_EQ(kExpectedOutput[i], pixel_ptr[j])
          << "at (" << i << ", " << j << ")";
    }
    pixel_ptr += dst_image.stride();
  }

  vpx_free(flimits_);
}

TEST_P(VpxPostProcDownAndAcrossMbRowTest, CheckCvsAssembly) {
  // Size of the underlying data block that will be filtered.
  // Y blocks are always a multiple of 16 wide and exactly 16 high. U and V
  // blocks are always a multiple of 8 wide and exactly 8 high.
  block_width_ = 136;
  block_height_ = 16;

  // 5-tap filter needs 2 padding rows above and below the block in the input.
  // SSE2 reads in blocks of 16. Pad an extra 8 in case the width is not %16.
  Buffer<uint8_t> src_image =
      Buffer<uint8_t>(block_width_, block_height_, 2, 2, 10, 2);
  ASSERT_TRUE(src_image.Init());

  // Filter extends output block by 8 samples at left and right edges.
  // Though the left padding is only 8 bytes, there is 'above' padding as well
  // so when the assembly code tries to read 16 bytes before the pointer it is
  // not a problem.
  // SSE2 reads in blocks of 16. Pad an extra 8 in case the width is not %16.
  Buffer<uint8_t> dst_image =
      Buffer<uint8_t>(block_width_, block_height_, 8, 8, 16, 8);
  ASSERT_TRUE(dst_image.Init());
  Buffer<uint8_t> dst_image_ref =
      Buffer<uint8_t>(block_width_, block_height_, 8);
  ASSERT_TRUE(dst_image_ref.Init());

  // Filter values are set in blocks of 16 for Y and 8 for U/V. Each macroblock
  // can have a different filter. SSE2 assembly reads flimits in blocks of 16 so
  // it must be padded out.
  const int flimits_width = block_width_ % 16 ? block_width_ + 8 : block_width_;
  flimits_ = reinterpret_cast<uint8_t *>(vpx_memalign(16, flimits_width));

  ACMRandom rnd;
  rnd.Reset(ACMRandom::DeterministicSeed());
  // Initialize pixels in the input:
  //   block pixels to random values.
  //   border pixels to value 10.
  src_image.SetPadding(10);
  src_image.Set(&rnd, &ACMRandom::Rand8);

  for (int blocks = 0; blocks < block_width_; blocks += 8) {
    (void)memset(flimits_, 0, sizeof(*flimits_) * flimits_width);

    for (int f = 0; f < 255; f++) {
      (void)memset(flimits_ + blocks, f, sizeof(*flimits_) * 8);
      dst_image.Set(0);
      dst_image_ref.Set(0);

      vpx_post_proc_down_and_across_mb_row_c(
          src_image.TopLeftPixel(), dst_image_ref.TopLeftPixel(),
          src_image.stride(), dst_image_ref.stride(), block_width_, flimits_,
          block_height_);
      ASM_REGISTER_STATE_CHECK(mb_post_proc_down_and_across_(
          src_image.TopLeftPixel(), dst_image.TopLeftPixel(),
          src_image.stride(), dst_image.stride(), block_width_, flimits_,
          block_height_));

      ASSERT_TRUE(dst_image.CheckValues(dst_image_ref));
    }
  }

  vpx_free(flimits_);
}

TEST_P(VpxPostProcDownAndAcrossMbRowTest, DISABLED_Speed) {
  // Size of the underlying data block that will be filtered.
  block_width_ = 16;
  block_height_ = 16;

  // 5-tap filter needs 2 padding rows above and below the block in the input.
  Buffer<uint8_t> src_image = Buffer<uint8_t>(block_width_, block_height_, 2);
  ASSERT_TRUE(src_image.Init());
  this->src_image_ = &src_image;

  // Filter extends output block by 8 samples at left and right edges.
  // Though the left padding is only 8 bytes, the assembly code tries to
  // read 16 bytes before the pointer.
  Buffer<uint8_t> dst_image =
      Buffer<uint8_t>(block_width_, block_height_, 8, 16, 8, 8);
  ASSERT_TRUE(dst_image.Init());
  this->dst_image_ = &dst_image;

  flimits_ = reinterpret_cast<uint8_t *>(vpx_memalign(16, block_width_));
  (void)memset(flimits_, 255, block_width_);

  // Initialize pixels in the input:
  //   block pixels to value 1,
  //   border pixels to value 10.
  src_image.SetPadding(10);
  src_image.Set(1);

  // Initialize pixels in the output to 99.
  dst_image.Set(99);

  RunNTimes(INT16_MAX);
  PrintMedian("16x16");

  vpx_free(flimits_);
}

class VpxMbPostProcAcrossIpTest
    : public AbstractBench,
      public ::testing::TestWithParam<VpxMbPostProcAcrossIpFunc> {
 public:
  VpxMbPostProcAcrossIpTest()
      : rows_(16), cols_(16), mb_post_proc_across_ip_(GetParam()),
        src_(Buffer<uint8_t>(rows_, cols_, 8, 8, 17, 8)) {}
  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  void Run() override;

  void SetCols(unsigned char *s, int rows, int cols, int src_width) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        s[c] = c;
      }
      s += src_width;
    }
  }

  void RunComparison(const unsigned char *expected_output, unsigned char *src_c,
                     int rows, int cols, int src_pitch) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        ASSERT_EQ(expected_output[c], src_c[c])
            << "at (" << r << ", " << c << ")";
      }
      src_c += src_pitch;
    }
  }

  void RunFilterLevel(unsigned char *s, int rows, int cols, int src_width,
                      int filter_level, const unsigned char *expected_output) {
    ASM_REGISTER_STATE_CHECK(
        GetParam()(s, src_width, rows, cols, filter_level));
    RunComparison(expected_output, s, rows, cols, src_width);
  }

  const int rows_;
  const int cols_;
  const VpxMbPostProcAcrossIpFunc mb_post_proc_across_ip_;
  Buffer<uint8_t> src_;
};

void VpxMbPostProcAcrossIpTest::Run() {
  mb_post_proc_across_ip_(src_.TopLeftPixel(), src_.stride(), rows_, cols_,
                          q2mbl(0));
}

TEST_P(VpxMbPostProcAcrossIpTest, CheckLowFilterOutput) {
  ASSERT_TRUE(src_.Init());
  src_.SetPadding(10);
  SetCols(src_.TopLeftPixel(), rows_, cols_, src_.stride());

  Buffer<uint8_t> expected_output = Buffer<uint8_t>(cols_, rows_, 0);
  ASSERT_TRUE(expected_output.Init());
  SetCols(expected_output.TopLeftPixel(), rows_, cols_,
          expected_output.stride());

  RunFilterLevel(src_.TopLeftPixel(), rows_, cols_, src_.stride(), q2mbl(0),
                 expected_output.TopLeftPixel());
}

TEST_P(VpxMbPostProcAcrossIpTest, CheckMediumFilterOutput) {
  ASSERT_TRUE(src_.Init());
  src_.SetPadding(10);
  SetCols(src_.TopLeftPixel(), rows_, cols_, src_.stride());

  static const unsigned char kExpectedOutput[] = {
    2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 13
  };

  RunFilterLevel(src_.TopLeftPixel(), rows_, cols_, src_.stride(), q2mbl(70),
                 kExpectedOutput);
}

TEST_P(VpxMbPostProcAcrossIpTest, CheckHighFilterOutput) {
  ASSERT_TRUE(src_.Init());
  src_.SetPadding(10);
  SetCols(src_.TopLeftPixel(), rows_, cols_, src_.stride());

  static const unsigned char kExpectedOutput[] = {
    2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 13
  };

  RunFilterLevel(src_.TopLeftPixel(), rows_, cols_, src_.stride(), INT_MAX,
                 kExpectedOutput);

  SetCols(src_.TopLeftPixel(), rows_, cols_, src_.stride());

  RunFilterLevel(src_.TopLeftPixel(), rows_, cols_, src_.stride(), q2mbl(100),
                 kExpectedOutput);
}

TEST_P(VpxMbPostProcAcrossIpTest, CheckCvsAssembly) {
  Buffer<uint8_t> c_mem = Buffer<uint8_t>(cols_, rows_, 8, 8, 17, 8);
  ASSERT_TRUE(c_mem.Init());
  Buffer<uint8_t> asm_mem = Buffer<uint8_t>(cols_, rows_, 8, 8, 17, 8);
  ASSERT_TRUE(asm_mem.Init());

  // When level >= 100, the filter behaves the same as the level = INT_MAX
  // When level < 20, it behaves the same as the level = 0
  for (int level = 0; level < 100; level++) {
    c_mem.SetPadding(10);
    asm_mem.SetPadding(10);
    SetCols(c_mem.TopLeftPixel(), rows_, cols_, c_mem.stride());
    SetCols(asm_mem.TopLeftPixel(), rows_, cols_, asm_mem.stride());

    vpx_mbpost_proc_across_ip_c(c_mem.TopLeftPixel(), c_mem.stride(), rows_,
                                cols_, q2mbl(level));
    ASM_REGISTER_STATE_CHECK(GetParam()(
        asm_mem.TopLeftPixel(), asm_mem.stride(), rows_, cols_, q2mbl(level)));

    ASSERT_TRUE(asm_mem.CheckValues(c_mem));
  }
}

TEST_P(VpxMbPostProcAcrossIpTest, DISABLED_Speed) {
  ASSERT_TRUE(src_.Init());
  src_.SetPadding(10);

  SetCols(src_.TopLeftPixel(), rows_, cols_, src_.stride());

  RunNTimes(100000);
  PrintMedian("16x16");
}

class VpxMbPostProcDownTest
    : public AbstractBench,
      public ::testing::TestWithParam<VpxMbPostProcDownFunc> {
 public:
  VpxMbPostProcDownTest()
      : rows_(16), cols_(16), mb_post_proc_down_(GetParam()),
        src_c_(Buffer<uint8_t>(rows_, cols_, 8, 8, 8, 17)) {}

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  void Run() override;

  void SetRows(unsigned char *src_c, int rows, int cols, int src_width) {
    for (int r = 0; r < rows; r++) {
      memset(src_c, r, cols);
      src_c += src_width;
    }
  }

  void RunComparison(const unsigned char *expected_output, unsigned char *src_c,
                     int rows, int cols, int src_pitch) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        ASSERT_EQ(expected_output[r * rows + c], src_c[c])
            << "at (" << r << ", " << c << ")";
      }
      src_c += src_pitch;
    }
  }

  void RunFilterLevel(unsigned char *s, int rows, int cols, int src_width,
                      int filter_level, const unsigned char *expected_output) {
    ASM_REGISTER_STATE_CHECK(
        mb_post_proc_down_(s, src_width, rows, cols, filter_level));
    RunComparison(expected_output, s, rows, cols, src_width);
  }

  const int rows_;
  const int cols_;
  const VpxMbPostProcDownFunc mb_post_proc_down_;
  Buffer<uint8_t> src_c_;
};

void VpxMbPostProcDownTest::Run() {
  mb_post_proc_down_(src_c_.TopLeftPixel(), src_c_.stride(), rows_, cols_,
                     q2mbl(0));
}

TEST_P(VpxMbPostProcDownTest, CheckHighFilterOutput) {
  ASSERT_TRUE(src_c_.Init());
  src_c_.SetPadding(10);

  SetRows(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride());

  static const unsigned char kExpectedOutput[] = {
    2,  2,  1,  1,  2,  2,  2,  2,  2,  2,  1,  1,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  2,  2,  2,  2,  2,  2,  2,  3,  2,  2,  2,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  3,  4,  4,  3,  3,  3,
    4,  4,  3,  4,  4,  3,  3,  4,  5,  4,  4,  4,  4,  4,  4,  4,  5,  4,  4,
    4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
    5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,
    7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  8,  9,  9,  8,  8,  8,  9,
    9,  8,  9,  9,  8,  8,  8,  9,  9,  10, 10, 9,  9,  9,  10, 10, 9,  10, 10,
    9,  9,  9,  10, 10, 10, 11, 10, 10, 10, 11, 10, 11, 10, 11, 10, 10, 10, 11,
    10, 11, 11, 11, 11, 11, 11, 11, 12, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 12,
    13, 12, 13, 12, 12, 12, 13, 12, 13, 12, 13, 12, 13, 13, 13, 14, 13, 13, 13,
    13, 13, 13, 13, 14, 13, 13, 13, 13
  };

  RunFilterLevel(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride(), INT_MAX,
                 kExpectedOutput);

  src_c_.SetPadding(10);
  SetRows(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride());
  RunFilterLevel(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride(),
                 q2mbl(100), kExpectedOutput);
}

TEST_P(VpxMbPostProcDownTest, CheckMediumFilterOutput) {
  ASSERT_TRUE(src_c_.Init());
  src_c_.SetPadding(10);

  SetRows(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride());

  static const unsigned char kExpectedOutput[] = {
    2,  2,  1,  1,  2,  2,  2,  2,  2,  2,  1,  1,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  2,  2,  2,  2,  2,  2,  2,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
    5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,
    7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,
    9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 13, 12,
    13, 12, 13, 12, 12, 12, 13, 12, 13, 12, 13, 12, 13, 13, 13, 14, 13, 13, 13,
    13, 13, 13, 13, 14, 13, 13, 13, 13
  };

  RunFilterLevel(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride(),
                 q2mbl(70), kExpectedOutput);
}

TEST_P(VpxMbPostProcDownTest, CheckLowFilterOutput) {
  ASSERT_TRUE(src_c_.Init());
  src_c_.SetPadding(10);

  SetRows(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride());

  std::unique_ptr<unsigned char[]> expected_output(
      new unsigned char[rows_ * cols_]);
  ASSERT_NE(expected_output, nullptr);
  SetRows(expected_output.get(), rows_, cols_, cols_);

  RunFilterLevel(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride(), q2mbl(0),
                 expected_output.get());
}

TEST_P(VpxMbPostProcDownTest, CheckCvsAssembly) {
  ACMRandom rnd;
  rnd.Reset(ACMRandom::DeterministicSeed());

  ASSERT_TRUE(src_c_.Init());
  Buffer<uint8_t> src_asm = Buffer<uint8_t>(cols_, rows_, 8, 8, 8, 17);
  ASSERT_TRUE(src_asm.Init());

  for (int level = 0; level < 100; level++) {
    src_c_.SetPadding(10);
    src_asm.SetPadding(10);
    src_c_.Set(&rnd, &ACMRandom::Rand8);
    src_asm.CopyFrom(src_c_);

    vpx_mbpost_proc_down_c(src_c_.TopLeftPixel(), src_c_.stride(), rows_, cols_,
                           q2mbl(level));
    ASM_REGISTER_STATE_CHECK(mb_post_proc_down_(
        src_asm.TopLeftPixel(), src_asm.stride(), rows_, cols_, q2mbl(level)));
    ASSERT_TRUE(src_asm.CheckValues(src_c_));

    src_c_.SetPadding(10);
    src_asm.SetPadding(10);
    src_c_.Set(&rnd, &ACMRandom::Rand8Extremes);
    src_asm.CopyFrom(src_c_);

    vpx_mbpost_proc_down_c(src_c_.TopLeftPixel(), src_c_.stride(), rows_, cols_,
                           q2mbl(level));
    ASM_REGISTER_STATE_CHECK(mb_post_proc_down_(
        src_asm.TopLeftPixel(), src_asm.stride(), rows_, cols_, q2mbl(level)));
    ASSERT_TRUE(src_asm.CheckValues(src_c_));
  }
}

TEST_P(VpxMbPostProcDownTest, DISABLED_Speed) {
  ASSERT_TRUE(src_c_.Init());
  src_c_.SetPadding(10);

  SetRows(src_c_.TopLeftPixel(), rows_, cols_, src_c_.stride());

  RunNTimes(100000);
  PrintMedian("16x16");
}

INSTANTIATE_TEST_SUITE_P(
    C, VpxPostProcDownAndAcrossMbRowTest,
    ::testing::Values(vpx_post_proc_down_and_across_mb_row_c));

INSTANTIATE_TEST_SUITE_P(C, VpxMbPostProcAcrossIpTest,
                         ::testing::Values(vpx_mbpost_proc_across_ip_c));

INSTANTIATE_TEST_SUITE_P(C, VpxMbPostProcDownTest,
                         ::testing::Values(vpx_mbpost_proc_down_c));

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxPostProcDownAndAcrossMbRowTest,
    ::testing::Values(vpx_post_proc_down_and_across_mb_row_sse2));

INSTANTIATE_TEST_SUITE_P(SSE2, VpxMbPostProcAcrossIpTest,
                         ::testing::Values(vpx_mbpost_proc_across_ip_sse2));

INSTANTIATE_TEST_SUITE_P(SSE2, VpxMbPostProcDownTest,
                         ::testing::Values(vpx_mbpost_proc_down_sse2));
#endif  // HAVE_SSE2

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, VpxPostProcDownAndAcrossMbRowTest,
    ::testing::Values(vpx_post_proc_down_and_across_mb_row_neon));

INSTANTIATE_TEST_SUITE_P(NEON, VpxMbPostProcAcrossIpTest,
                         ::testing::Values(vpx_mbpost_proc_across_ip_neon));

INSTANTIATE_TEST_SUITE_P(NEON, VpxMbPostProcDownTest,
                         ::testing::Values(vpx_mbpost_proc_down_neon));
#endif  // HAVE_NEON

#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(
    MSA, VpxPostProcDownAndAcrossMbRowTest,
    ::testing::Values(vpx_post_proc_down_and_across_mb_row_msa));

INSTANTIATE_TEST_SUITE_P(MSA, VpxMbPostProcAcrossIpTest,
                         ::testing::Values(vpx_mbpost_proc_across_ip_msa));

INSTANTIATE_TEST_SUITE_P(MSA, VpxMbPostProcDownTest,
                         ::testing::Values(vpx_mbpost_proc_down_msa));
#endif  // HAVE_MSA

#if HAVE_VSX
INSTANTIATE_TEST_SUITE_P(
    VSX, VpxPostProcDownAndAcrossMbRowTest,
    ::testing::Values(vpx_post_proc_down_and_across_mb_row_vsx));

INSTANTIATE_TEST_SUITE_P(VSX, VpxMbPostProcAcrossIpTest,
                         ::testing::Values(vpx_mbpost_proc_across_ip_vsx));

INSTANTIATE_TEST_SUITE_P(VSX, VpxMbPostProcDownTest,
                         ::testing::Values(vpx_mbpost_proc_down_vsx));
#endif  // HAVE_VSX

}  // namespace
