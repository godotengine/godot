/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <tuple>

#include "gtest/gtest.h"

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_scan.h"
#include "vpx/vpx_integer.h"
#include "vpx_config.h"
#include "vpx_ports/vpx_timer.h"

using libvpx_test::ACMRandom;

namespace {

using FwdTxfmFunc = void (*)(const int16_t *in, tran_low_t *out, int stride);
using InvTxfmFunc = void (*)(const tran_low_t *in, uint8_t *out, int stride);
using InvTxfmWithBdFunc = void (*)(const tran_low_t *in, uint8_t *out,
                                   int stride, int bd);

template <InvTxfmFunc fn>
void wrapper(const tran_low_t *in, uint8_t *out, int stride, int bd) {
  (void)bd;
  fn(in, out, stride);
}

#if CONFIG_VP9_HIGHBITDEPTH
using InvTxfmHighbdFunc = void (*)(const tran_low_t *in, uint16_t *out,
                                   int stride, int bd);

template <InvTxfmHighbdFunc fn>
void highbd_wrapper(const tran_low_t *in, uint8_t *out, int stride, int bd) {
  fn(in, CAST_TO_SHORTPTR(out), stride, bd);
}
#endif

using PartialInvTxfmParam =
    std::tuple<FwdTxfmFunc, InvTxfmWithBdFunc, InvTxfmWithBdFunc, TX_SIZE, int,
               int, int>;
const int kMaxNumCoeffs = 1024;
const int kCountTestBlock = 1000;

class PartialIDctTest : public ::testing::TestWithParam<PartialInvTxfmParam> {
 public:
  ~PartialIDctTest() override = default;
  void SetUp() override {
    rnd_.Reset(ACMRandom::DeterministicSeed());
    fwd_txfm_ = GET_PARAM(0);
    full_inv_txfm_ = GET_PARAM(1);
    partial_inv_txfm_ = GET_PARAM(2);
    tx_size_ = GET_PARAM(3);
    last_nonzero_ = GET_PARAM(4);
    bit_depth_ = GET_PARAM(5);
    pixel_size_ = GET_PARAM(6);
    mask_ = (1 << bit_depth_) - 1;

    switch (tx_size_) {
      case TX_4X4: size_ = 4; break;
      case TX_8X8: size_ = 8; break;
      case TX_16X16: size_ = 16; break;
      case TX_32X32: size_ = 32; break;
      default: FAIL() << "Wrong Size!";
    }

    // Randomize stride_ to a value less than or equal to 1024
    stride_ = rnd_(1024) + 1;
    if (stride_ < size_) {
      stride_ = size_;
    }
    // Align stride_ to 16 if it's bigger than 16.
    if (stride_ > 16) {
      stride_ &= ~15;
    }

    input_block_size_ = size_ * size_;
    output_block_size_ = size_ * stride_;

    input_block_ = reinterpret_cast<tran_low_t *>(
        vpx_memalign(16, sizeof(*input_block_) * input_block_size_));
    output_block_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(16, pixel_size_ * output_block_size_));
    output_block_ref_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(16, pixel_size_ * output_block_size_));
  }

  void TearDown() override {
    vpx_free(input_block_);
    input_block_ = nullptr;
    vpx_free(output_block_);
    output_block_ = nullptr;
    vpx_free(output_block_ref_);
    output_block_ref_ = nullptr;
    libvpx_test::ClearSystemState();
  }

  void InitMem() {
    memset(input_block_, 0, sizeof(*input_block_) * input_block_size_);
    if (pixel_size_ == 1) {
      for (int j = 0; j < output_block_size_; ++j) {
        output_block_[j] = output_block_ref_[j] = rnd_.Rand16() & mask_;
      }
    } else {
      ASSERT_EQ(2, pixel_size_);
      uint16_t *const output = reinterpret_cast<uint16_t *>(output_block_);
      uint16_t *const output_ref =
          reinterpret_cast<uint16_t *>(output_block_ref_);
      for (int j = 0; j < output_block_size_; ++j) {
        output[j] = output_ref[j] = rnd_.Rand16() & mask_;
      }
    }
  }

  void InitInput() {
    const int64_t max_coeff = (32766 << (bit_depth_ - 8)) / 4;
    int64_t max_energy_leftover = max_coeff * max_coeff;
    for (int j = 0; j < last_nonzero_; ++j) {
      tran_low_t coeff = static_cast<tran_low_t>(
          sqrt(1.0 * max_energy_leftover) * (rnd_.Rand16() - 32768) / 65536);
      max_energy_leftover -= static_cast<int64_t>(coeff) * coeff;
      if (max_energy_leftover < 0) {
        max_energy_leftover = 0;
        coeff = 0;
      }
      input_block_[vp9_default_scan_orders[tx_size_].scan[j]] = coeff;
    }
  }

  void PrintDiff() {
    if (memcmp(output_block_ref_, output_block_,
               pixel_size_ * output_block_size_)) {
      uint16_t ref, opt;
      for (int y = 0; y < size_; y++) {
        for (int x = 0; x < size_; x++) {
          if (pixel_size_ == 1) {
            ref = output_block_ref_[y * stride_ + x];
            opt = output_block_[y * stride_ + x];
          } else {
            ref = reinterpret_cast<uint16_t *>(
                output_block_ref_)[y * stride_ + x];
            opt = reinterpret_cast<uint16_t *>(output_block_)[y * stride_ + x];
          }
          if (ref != opt) {
            printf("dest[%d][%d] diff:%6d (ref),%6d (opt)\n", y, x, ref, opt);
          }
        }
      }

      printf("\ninput_block_:\n");
      for (int y = 0; y < size_; y++) {
        for (int x = 0; x < size_; x++) {
          printf("%6d,", input_block_[y * size_ + x]);
        }
        printf("\n");
      }
    }
  }

 protected:
  int last_nonzero_;
  TX_SIZE tx_size_;
  tran_low_t *input_block_;
  uint8_t *output_block_;
  uint8_t *output_block_ref_;
  int size_;
  int stride_;
  int pixel_size_;
  int input_block_size_;
  int output_block_size_;
  int bit_depth_;
  int mask_;
  FwdTxfmFunc fwd_txfm_;
  InvTxfmWithBdFunc full_inv_txfm_;
  InvTxfmWithBdFunc partial_inv_txfm_;
  ACMRandom rnd_;
};

TEST_P(PartialIDctTest, RunQuantCheck) {
  const int count_test_block = (size_ != 4) ? kCountTestBlock : 65536;
  DECLARE_ALIGNED(16, int16_t, input_extreme_block[kMaxNumCoeffs]);
  DECLARE_ALIGNED(16, tran_low_t, output_ref_block[kMaxNumCoeffs]);

  InitMem();

  for (int i = 0; i < count_test_block; ++i) {
    // Initialize a test block with input range [-mask_, mask_].
    if (size_ != 4) {
      if (i == 0) {
        for (int k = 0; k < input_block_size_; ++k) {
          input_extreme_block[k] = mask_;
        }
      } else if (i == 1) {
        for (int k = 0; k < input_block_size_; ++k) {
          input_extreme_block[k] = -mask_;
        }
      } else {
        for (int k = 0; k < input_block_size_; ++k) {
          input_extreme_block[k] = rnd_.Rand8() % 2 ? mask_ : -mask_;
        }
      }
    } else {
      // Try all possible combinations.
      for (int k = 0; k < input_block_size_; ++k) {
        input_extreme_block[k] = (i & (1 << k)) ? mask_ : -mask_;
      }
    }

    fwd_txfm_(input_extreme_block, output_ref_block, size_);

    // quantization with minimum allowed step sizes
    input_block_[0] = (output_ref_block[0] / 4) * 4;
    for (int k = 1; k < last_nonzero_; ++k) {
      const int pos = vp9_default_scan_orders[tx_size_].scan[k];
      input_block_[pos] = (output_ref_block[pos] / 4) * 4;
    }

    ASM_REGISTER_STATE_CHECK(
        full_inv_txfm_(input_block_, output_block_ref_, stride_, bit_depth_));
    ASM_REGISTER_STATE_CHECK(
        partial_inv_txfm_(input_block_, output_block_, stride_, bit_depth_));
    ASSERT_EQ(0, memcmp(output_block_ref_, output_block_,
                        pixel_size_ * output_block_size_))
        << "Error: partial inverse transform produces different results";
  }
}

TEST_P(PartialIDctTest, ResultsMatch) {
  for (int i = 0; i < kCountTestBlock; ++i) {
    InitMem();
    InitInput();

    ASM_REGISTER_STATE_CHECK(
        full_inv_txfm_(input_block_, output_block_ref_, stride_, bit_depth_));
    ASM_REGISTER_STATE_CHECK(
        partial_inv_txfm_(input_block_, output_block_, stride_, bit_depth_));
    ASSERT_EQ(0, memcmp(output_block_ref_, output_block_,
                        pixel_size_ * output_block_size_))
        << "Error: partial inverse transform produces different results";
  }
}

TEST_P(PartialIDctTest, AddOutputBlock) {
  for (int i = 0; i < kCountTestBlock; ++i) {
    InitMem();
    for (int j = 0; j < last_nonzero_; ++j) {
      input_block_[vp9_default_scan_orders[tx_size_].scan[j]] = 10;
    }

    ASM_REGISTER_STATE_CHECK(
        full_inv_txfm_(input_block_, output_block_ref_, stride_, bit_depth_));
    ASM_REGISTER_STATE_CHECK(
        partial_inv_txfm_(input_block_, output_block_, stride_, bit_depth_));
    ASSERT_EQ(0, memcmp(output_block_ref_, output_block_,
                        pixel_size_ * output_block_size_))
        << "Error: Transform results are not correctly added to output.";
  }
}

TEST_P(PartialIDctTest, SingleExtremeCoeff) {
  const int16_t max_coeff = std::numeric_limits<int16_t>::max();
  const int16_t min_coeff = std::numeric_limits<int16_t>::min();
  for (int i = 0; i < last_nonzero_; ++i) {
    memset(input_block_, 0, sizeof(*input_block_) * input_block_size_);
    // Run once for min and once for max.
    for (int j = 0; j < 2; ++j) {
      const int coeff = j ? min_coeff : max_coeff;

      memset(output_block_, 0, pixel_size_ * output_block_size_);
      memset(output_block_ref_, 0, pixel_size_ * output_block_size_);
      input_block_[vp9_default_scan_orders[tx_size_].scan[i]] = coeff;

      ASM_REGISTER_STATE_CHECK(
          full_inv_txfm_(input_block_, output_block_ref_, stride_, bit_depth_));
      ASM_REGISTER_STATE_CHECK(
          partial_inv_txfm_(input_block_, output_block_, stride_, bit_depth_));
      ASSERT_EQ(0, memcmp(output_block_ref_, output_block_,
                          pixel_size_ * output_block_size_))
          << "Error: Fails with single coeff of " << coeff << " at " << i
          << ".";
    }
  }
}

TEST_P(PartialIDctTest, DISABLED_Speed) {
  // Keep runtime stable with transform size.
  const int kCountSpeedTestBlock = 500000000 / input_block_size_;
  InitMem();
  InitInput();

  for (int i = 0; i < kCountSpeedTestBlock; ++i) {
    ASM_REGISTER_STATE_CHECK(
        full_inv_txfm_(input_block_, output_block_ref_, stride_, bit_depth_));
  }
  vpx_usec_timer timer;
  vpx_usec_timer_start(&timer);
  for (int i = 0; i < kCountSpeedTestBlock; ++i) {
    partial_inv_txfm_(input_block_, output_block_, stride_, bit_depth_);
  }
  libvpx_test::ClearSystemState();
  vpx_usec_timer_mark(&timer);
  const int elapsed_time =
      static_cast<int>(vpx_usec_timer_elapsed(&timer) / 1000);
  printf("idct%dx%d_%d (%s %d) time: %5d ms\n", size_, size_, last_nonzero_,
         (pixel_size_ == 1) ? "bitdepth" : "high bitdepth", bit_depth_,
         elapsed_time);
  ASSERT_EQ(0, memcmp(output_block_ref_, output_block_,
                      pixel_size_ * output_block_size_))
      << "Error: partial inverse transform produces different results";
}

using std::make_tuple;

const PartialInvTxfmParam c_partial_idct_tests[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>, TX_32X32, 1024, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>, TX_32X32, 1024, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>, TX_32X32, 1024, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>, TX_32X32, 135, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>, TX_32X32, 135, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>, TX_32X32, 135, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>, TX_32X32, 34, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>, TX_32X32, 34, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>, TX_32X32, 34, 12, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>, TX_32X32, 1, 8, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>, TX_32X32, 1, 10, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>, TX_32X32, 1, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>, TX_16X16, 256, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>, TX_16X16, 256, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>, TX_16X16, 256, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>, TX_16X16, 38, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>, TX_16X16, 38, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>, TX_16X16, 38, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>, TX_16X16, 10, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>, TX_16X16, 10, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>, TX_16X16, 10, 12, 2),
  make_tuple(&vpx_highbd_fdct16x16_c,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
             &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>, TX_16X16, 1, 8, 2),
  make_tuple(&vpx_highbd_fdct16x16_c,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
             &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>, TX_16X16, 1, 10, 2),
  make_tuple(&vpx_highbd_fdct16x16_c,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
             &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>, TX_16X16, 1, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>, TX_8X8, 64, 8, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>, TX_8X8, 64, 10, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>, TX_8X8, 64, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>, TX_8X8, 12, 8, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>, TX_8X8, 12, 10, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>, TX_8X8, 12, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>, TX_8X8, 1, 8, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>, TX_8X8, 1, 10, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>, TX_8X8, 1, 12, 2),
  make_tuple(&vpx_highbd_fdct4x4_c,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>, TX_4X4, 16, 8, 2),
  make_tuple(&vpx_highbd_fdct4x4_c,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>, TX_4X4, 16, 10, 2),
  make_tuple(&vpx_highbd_fdct4x4_c,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>, TX_4X4, 16, 12, 2),
  make_tuple(&vpx_highbd_fdct4x4_c,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>, TX_4X4, 1, 8, 2),
  make_tuple(&vpx_highbd_fdct4x4_c,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>, TX_4X4, 1, 10, 2),
  make_tuple(&vpx_highbd_fdct4x4_c,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>, TX_4X4, 1, 12, 2),
#endif  // CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_1024_add_c>, TX_32X32, 1024, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_135_add_c>, TX_32X32, 135, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_34_add_c>, TX_32X32, 34, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_1_add_c>, TX_32X32, 1, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_256_add_c>,
             &wrapper<vpx_idct16x16_256_add_c>, TX_16X16, 256, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_256_add_c>,
             &wrapper<vpx_idct16x16_38_add_c>, TX_16X16, 38, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_256_add_c>,
             &wrapper<vpx_idct16x16_10_add_c>, TX_16X16, 10, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_256_add_c>,
             &wrapper<vpx_idct16x16_1_add_c>, TX_16X16, 1, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_64_add_c>,
             &wrapper<vpx_idct8x8_64_add_c>, TX_8X8, 64, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_64_add_c>,
             &wrapper<vpx_idct8x8_12_add_c>, TX_8X8, 12, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_64_add_c>,
             &wrapper<vpx_idct8x8_1_add_c>, TX_8X8, 1, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_16_add_c>,
             &wrapper<vpx_idct4x4_16_add_c>, TX_4X4, 16, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_16_add_c>,
             &wrapper<vpx_idct4x4_1_add_c>, TX_4X4, 1, 8, 1)
};

INSTANTIATE_TEST_SUITE_P(C, PartialIDctTest,
                         ::testing::ValuesIn(c_partial_idct_tests));

#if !CONFIG_EMULATE_HARDWARE

#if HAVE_NEON
const PartialInvTxfmParam neon_partial_idct_tests[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_neon>, TX_32X32,
             1024, 8, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_neon>, TX_32X32,
             1024, 10, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_neon>, TX_32X32,
             1024, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_neon>, TX_32X32, 135, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_neon>, TX_32X32, 135, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_neon>, TX_32X32, 135, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_neon>, TX_32X32, 34, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_neon>, TX_32X32, 34, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_neon>, TX_32X32, 34, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1_add_neon>, TX_32X32, 1, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1_add_neon>, TX_32X32, 1, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1_add_neon>, TX_32X32, 1, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_neon>, TX_16X16, 256, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_neon>, TX_16X16, 256, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_neon>, TX_16X16, 256, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_neon>, TX_16X16, 38, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_neon>, TX_16X16, 38, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_neon>, TX_16X16, 38, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_neon>, TX_16X16, 10, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_neon>, TX_16X16, 10, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_neon>, TX_16X16, 10, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_1_add_neon>, TX_16X16, 1, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_1_add_neon>, TX_16X16, 1, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_1_add_neon>, TX_16X16, 1, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_neon>, TX_8X8, 64, 8, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_64_add_neon>, TX_8X8, 64, 10, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_64_add_neon>, TX_8X8, 64, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_12_add_neon>, TX_8X8, 12, 8, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_12_add_neon>, TX_8X8, 12, 10, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_12_add_neon>, TX_8X8, 12, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_neon>, TX_8X8, 1, 8, 2),
  make_tuple(&vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_neon>, TX_8X8, 1, 10, 2),
  make_tuple(&vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_neon>, TX_8X8, 1, 12, 2),
  make_tuple(&vpx_highbd_fdct4x4_c,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_neon>, TX_4X4, 16, 8, 2),
  make_tuple(
      &vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
      &highbd_wrapper<vpx_highbd_idct4x4_16_add_neon>, TX_4X4, 16, 10, 2),
  make_tuple(
      &vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
      &highbd_wrapper<vpx_highbd_idct4x4_16_add_neon>, TX_4X4, 16, 12, 2),
  make_tuple(&vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_neon>, TX_4X4, 1, 8, 2),
  make_tuple(&vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_neon>, TX_4X4, 1, 10, 2),
  make_tuple(&vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_neon>, TX_4X4, 1, 12, 2),
#endif  // CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_1024_add_neon>, TX_32X32, 1024, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_135_add_c>,
             &wrapper<vpx_idct32x32_135_add_neon>, TX_32X32, 135, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_34_add_c>,
             &wrapper<vpx_idct32x32_34_add_neon>, TX_32X32, 34, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1_add_c>,
             &wrapper<vpx_idct32x32_1_add_neon>, TX_32X32, 1, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_256_add_c>,
             &wrapper<vpx_idct16x16_256_add_neon>, TX_16X16, 256, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_38_add_c>,
             &wrapper<vpx_idct16x16_38_add_neon>, TX_16X16, 38, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_10_add_c>,
             &wrapper<vpx_idct16x16_10_add_neon>, TX_16X16, 10, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_1_add_c>,
             &wrapper<vpx_idct16x16_1_add_neon>, TX_16X16, 1, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_64_add_c>,
             &wrapper<vpx_idct8x8_64_add_neon>, TX_8X8, 64, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_12_add_c>,
             &wrapper<vpx_idct8x8_12_add_neon>, TX_8X8, 12, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_1_add_c>,
             &wrapper<vpx_idct8x8_1_add_neon>, TX_8X8, 1, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_16_add_c>,
             &wrapper<vpx_idct4x4_16_add_neon>, TX_4X4, 16, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_1_add_c>,
             &wrapper<vpx_idct4x4_1_add_neon>, TX_4X4, 1, 8, 1)
};

INSTANTIATE_TEST_SUITE_P(NEON, PartialIDctTest,
                         ::testing::ValuesIn(neon_partial_idct_tests));
#endif  // HAVE_NEON

#if HAVE_SSE2
// 32x32_135_ is implemented using the 1024 version.
const PartialInvTxfmParam sse2_partial_idct_tests[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_sse2>, TX_32X32,
             1024, 8, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_sse2>, TX_32X32,
             1024, 10, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_sse2>, TX_32X32,
             1024, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_sse2>, TX_32X32, 135, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_sse2>, TX_32X32, 135, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_135_add_sse2>, TX_32X32, 135, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_sse2>, TX_32X32, 34, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_sse2>, TX_32X32, 34, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_sse2>, TX_32X32, 34, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1_add_sse2>, TX_32X32, 1, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1_add_sse2>, TX_32X32, 1, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_1_add_sse2>, TX_32X32, 1, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_sse2>, TX_16X16, 256, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_sse2>, TX_16X16, 256, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_256_add_sse2>, TX_16X16, 256, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_sse2>, TX_16X16, 38, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_sse2>, TX_16X16, 38, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_sse2>, TX_16X16, 38, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_sse2>, TX_16X16, 10, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_sse2>, TX_16X16, 10, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_sse2>, TX_16X16, 10, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_1_add_sse2>, TX_16X16, 1, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_1_add_sse2>, TX_16X16, 1, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_1_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_1_add_sse2>, TX_16X16, 1, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_64_add_sse2>, TX_8X8, 64, 8, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_64_add_sse2>, TX_8X8, 64, 10, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_64_add_sse2>, TX_8X8, 64, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c,
             &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_12_add_sse2>, TX_8X8, 12, 8, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_12_add_sse2>, TX_8X8, 12, 10, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_12_add_sse2>, TX_8X8, 12, 12, 2),
  make_tuple(&vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_sse2>, TX_8X8, 1, 8, 2),
  make_tuple(&vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_sse2>, TX_8X8, 1, 10, 2),
  make_tuple(&vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct8x8_1_add_sse2>, TX_8X8, 1, 12, 2),
  make_tuple(&vpx_highbd_fdct4x4_c,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_16_add_sse2>, TX_4X4, 16, 8, 2),
  make_tuple(
      &vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
      &highbd_wrapper<vpx_highbd_idct4x4_16_add_sse2>, TX_4X4, 16, 10, 2),
  make_tuple(
      &vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
      &highbd_wrapper<vpx_highbd_idct4x4_16_add_sse2>, TX_4X4, 16, 12, 2),
  make_tuple(&vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_sse2>, TX_4X4, 1, 8, 2),
  make_tuple(&vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_sse2>, TX_4X4, 1, 10, 2),
  make_tuple(&vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_1_add_c>,
             &highbd_wrapper<vpx_highbd_idct4x4_1_add_sse2>, TX_4X4, 1, 12, 2),
#endif  // CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_1024_add_sse2>, TX_32X32, 1024, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_135_add_c>,
             &wrapper<vpx_idct32x32_135_add_sse2>, TX_32X32, 135, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_34_add_c>,
             &wrapper<vpx_idct32x32_34_add_sse2>, TX_32X32, 34, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1_add_c>,
             &wrapper<vpx_idct32x32_1_add_sse2>, TX_32X32, 1, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_256_add_c>,
             &wrapper<vpx_idct16x16_256_add_sse2>, TX_16X16, 256, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_38_add_c>,
             &wrapper<vpx_idct16x16_38_add_sse2>, TX_16X16, 38, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_10_add_c>,
             &wrapper<vpx_idct16x16_10_add_sse2>, TX_16X16, 10, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_1_add_c>,
             &wrapper<vpx_idct16x16_1_add_sse2>, TX_16X16, 1, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_64_add_c>,
             &wrapper<vpx_idct8x8_64_add_sse2>, TX_8X8, 64, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_12_add_c>,
             &wrapper<vpx_idct8x8_12_add_sse2>, TX_8X8, 12, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_1_add_c>,
             &wrapper<vpx_idct8x8_1_add_sse2>, TX_8X8, 1, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_16_add_c>,
             &wrapper<vpx_idct4x4_16_add_sse2>, TX_4X4, 16, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_1_add_c>,
             &wrapper<vpx_idct4x4_1_add_sse2>, TX_4X4, 1, 8, 1)
};

INSTANTIATE_TEST_SUITE_P(SSE2, PartialIDctTest,
                         ::testing::ValuesIn(sse2_partial_idct_tests));

#endif  // HAVE_SSE2

#if HAVE_SSSE3
const PartialInvTxfmParam ssse3_partial_idct_tests[] = {
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_135_add_c>,
             &wrapper<vpx_idct32x32_135_add_ssse3>, TX_32X32, 135, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_34_add_c>,
             &wrapper<vpx_idct32x32_34_add_ssse3>, TX_32X32, 34, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_12_add_c>,
             &wrapper<vpx_idct8x8_12_add_ssse3>, TX_8X8, 12, 8, 1)
};

INSTANTIATE_TEST_SUITE_P(SSSE3, PartialIDctTest,
                         ::testing::ValuesIn(ssse3_partial_idct_tests));
#endif  // HAVE_SSSE3

#if HAVE_SSE4_1 && CONFIG_VP9_HIGHBITDEPTH
const PartialInvTxfmParam sse4_1_partial_idct_tests[] = {
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_sse4_1>, TX_32X32,
             1024, 8, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_sse4_1>, TX_32X32,
             1024, 10, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_1024_add_sse4_1>, TX_32X32,
             1024, 12, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_135_add_sse4_1>, TX_32X32,
             135, 8, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_135_add_sse4_1>, TX_32X32,
             135, 10, 2),
  make_tuple(&vpx_highbd_fdct32x32_c,
             &highbd_wrapper<vpx_highbd_idct32x32_135_add_c>,
             &highbd_wrapper<vpx_highbd_idct32x32_135_add_sse4_1>, TX_32X32,
             135, 12, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_sse4_1>, TX_32X32, 34, 8, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_sse4_1>, TX_32X32, 34, 10, 2),
  make_tuple(
      &vpx_highbd_fdct32x32_c, &highbd_wrapper<vpx_highbd_idct32x32_34_add_c>,
      &highbd_wrapper<vpx_highbd_idct32x32_34_add_sse4_1>, TX_32X32, 34, 12, 2),
  make_tuple(&vpx_highbd_fdct16x16_c,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_sse4_1>, TX_16X16,
             256, 8, 2),
  make_tuple(&vpx_highbd_fdct16x16_c,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_sse4_1>, TX_16X16,
             256, 10, 2),
  make_tuple(&vpx_highbd_fdct16x16_c,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_c>,
             &highbd_wrapper<vpx_highbd_idct16x16_256_add_sse4_1>, TX_16X16,
             256, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_sse4_1>, TX_16X16, 38, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_sse4_1>, TX_16X16, 38, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_38_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_38_add_sse4_1>, TX_16X16, 38, 12, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_sse4_1>, TX_16X16, 10, 8, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_sse4_1>, TX_16X16, 10, 10, 2),
  make_tuple(
      &vpx_highbd_fdct16x16_c, &highbd_wrapper<vpx_highbd_idct16x16_10_add_c>,
      &highbd_wrapper<vpx_highbd_idct16x16_10_add_sse4_1>, TX_16X16, 10, 12, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_64_add_sse4_1>, TX_8X8, 64, 8, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_64_add_sse4_1>, TX_8X8, 64, 10, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_64_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_64_add_sse4_1>, TX_8X8, 64, 12, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_12_add_sse4_1>, TX_8X8, 12, 8, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_12_add_sse4_1>, TX_8X8, 12, 10, 2),
  make_tuple(
      &vpx_highbd_fdct8x8_c, &highbd_wrapper<vpx_highbd_idct8x8_12_add_c>,
      &highbd_wrapper<vpx_highbd_idct8x8_12_add_sse4_1>, TX_8X8, 12, 12, 2),
  make_tuple(
      &vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
      &highbd_wrapper<vpx_highbd_idct4x4_16_add_sse4_1>, TX_4X4, 16, 8, 2),
  make_tuple(
      &vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
      &highbd_wrapper<vpx_highbd_idct4x4_16_add_sse4_1>, TX_4X4, 16, 10, 2),
  make_tuple(
      &vpx_highbd_fdct4x4_c, &highbd_wrapper<vpx_highbd_idct4x4_16_add_c>,
      &highbd_wrapper<vpx_highbd_idct4x4_16_add_sse4_1>, TX_4X4, 16, 12, 2)
};

INSTANTIATE_TEST_SUITE_P(SSE4_1, PartialIDctTest,
                         ::testing::ValuesIn(sse4_1_partial_idct_tests));
#endif  // HAVE_SSE4_1 && CONFIG_VP9_HIGHBITDEPTH

#if HAVE_DSPR2 && !CONFIG_VP9_HIGHBITDEPTH
const PartialInvTxfmParam dspr2_partial_idct_tests[] = {
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_1024_add_dspr2>, TX_32X32, 1024, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_34_add_c>,
             &wrapper<vpx_idct32x32_34_add_dspr2>, TX_32X32, 34, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1_add_c>,
             &wrapper<vpx_idct32x32_1_add_dspr2>, TX_32X32, 1, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_256_add_c>,
             &wrapper<vpx_idct16x16_256_add_dspr2>, TX_16X16, 256, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_10_add_c>,
             &wrapper<vpx_idct16x16_10_add_dspr2>, TX_16X16, 10, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_1_add_c>,
             &wrapper<vpx_idct16x16_1_add_dspr2>, TX_16X16, 1, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_64_add_c>,
             &wrapper<vpx_idct8x8_64_add_dspr2>, TX_8X8, 64, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_12_add_c>,
             &wrapper<vpx_idct8x8_12_add_dspr2>, TX_8X8, 12, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_1_add_c>,
             &wrapper<vpx_idct8x8_1_add_dspr2>, TX_8X8, 1, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_16_add_c>,
             &wrapper<vpx_idct4x4_16_add_dspr2>, TX_4X4, 16, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_1_add_c>,
             &wrapper<vpx_idct4x4_1_add_dspr2>, TX_4X4, 1, 8, 1)
};

INSTANTIATE_TEST_SUITE_P(DSPR2, PartialIDctTest,
                         ::testing::ValuesIn(dspr2_partial_idct_tests));
#endif  // HAVE_DSPR2 && !CONFIG_VP9_HIGHBITDEPTH

#if HAVE_MSA && !CONFIG_VP9_HIGHBITDEPTH
// 32x32_135_ is implemented using the 1024 version.
const PartialInvTxfmParam msa_partial_idct_tests[] = {
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_1024_add_msa>, TX_32X32, 1024, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_34_add_c>,
             &wrapper<vpx_idct32x32_34_add_msa>, TX_32X32, 34, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1_add_c>,
             &wrapper<vpx_idct32x32_1_add_msa>, TX_32X32, 1, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_256_add_c>,
             &wrapper<vpx_idct16x16_256_add_msa>, TX_16X16, 256, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_10_add_c>,
             &wrapper<vpx_idct16x16_10_add_msa>, TX_16X16, 10, 8, 1),
  make_tuple(&vpx_fdct16x16_c, &wrapper<vpx_idct16x16_1_add_c>,
             &wrapper<vpx_idct16x16_1_add_msa>, TX_16X16, 1, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_64_add_c>,
             &wrapper<vpx_idct8x8_64_add_msa>, TX_8X8, 64, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_12_add_c>,
             &wrapper<vpx_idct8x8_12_add_msa>, TX_8X8, 12, 8, 1),
  make_tuple(&vpx_fdct8x8_c, &wrapper<vpx_idct8x8_1_add_c>,
             &wrapper<vpx_idct8x8_1_add_msa>, TX_8X8, 1, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_16_add_c>,
             &wrapper<vpx_idct4x4_16_add_msa>, TX_4X4, 16, 8, 1),
  make_tuple(&vpx_fdct4x4_c, &wrapper<vpx_idct4x4_1_add_c>,
             &wrapper<vpx_idct4x4_1_add_msa>, TX_4X4, 1, 8, 1)
};

INSTANTIATE_TEST_SUITE_P(MSA, PartialIDctTest,
                         ::testing::ValuesIn(msa_partial_idct_tests));
#endif  // HAVE_MSA && !CONFIG_VP9_HIGHBITDEPTH

#if HAVE_LSX && !CONFIG_VP9_HIGHBITDEPTH
const PartialInvTxfmParam lsx_partial_idct_tests[] = {
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1024_add_c>,
             &wrapper<vpx_idct32x32_1024_add_lsx>, TX_32X32, 1024, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_34_add_c>,
             &wrapper<vpx_idct32x32_34_add_lsx>, TX_32X32, 34, 8, 1),
  make_tuple(&vpx_fdct32x32_c, &wrapper<vpx_idct32x32_1_add_c>,
             &wrapper<vpx_idct32x32_1_add_lsx>, TX_32X32, 1, 8, 1),
};

INSTANTIATE_TEST_SUITE_P(LSX, PartialIDctTest,
                         ::testing::ValuesIn(lsx_partial_idct_tests));
#endif  // HAVE_LSX && !CONFIG_VP9_HIGHBITDEPTH

#endif  // !CONFIG_EMULATE_HARDWARE

}  // namespace
