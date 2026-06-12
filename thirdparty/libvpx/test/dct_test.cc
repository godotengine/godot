/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
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
#include <tuple>

#include "gtest/gtest.h"

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/buffer.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vp9/common/vp9_entropy.h"
#include "vpx_config.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

using libvpx_test::ACMRandom;
using libvpx_test::Buffer;
using std::make_tuple;
using std::tuple;

namespace {
using FdctFunc = void (*)(const int16_t *in, tran_low_t *out, int stride);
using IdctFunc = void (*)(const tran_low_t *in, uint8_t *out, int stride);
using FhtFunc = void (*)(const int16_t *in, tran_low_t *out, int stride,
                         int tx_type);
using FhtFuncRef = void (*)(const Buffer<int16_t> &in, Buffer<tran_low_t> *out,
                            int size, int tx_type);
using IhtFunc = void (*)(const tran_low_t *in, uint8_t *out, int stride,
                         int tx_type);
using IhtWithBdFunc = void (*)(const tran_low_t *in, uint8_t *out, int stride,
                               int tx_type, int bd);

template <FdctFunc fn>
void fdct_wrapper(const int16_t *in, tran_low_t *out, int stride, int tx_type) {
  (void)tx_type;
  fn(in, out, stride);
}

template <IdctFunc fn>
void idct_wrapper(const tran_low_t *in, uint8_t *out, int stride, int tx_type,
                  int bd) {
  (void)tx_type;
  (void)bd;
  fn(in, out, stride);
}

template <IhtFunc fn>
void iht_wrapper(const tran_low_t *in, uint8_t *out, int stride, int tx_type,
                 int bd) {
  (void)bd;
  fn(in, out, stride, tx_type);
}

#if CONFIG_VP9_HIGHBITDEPTH
using HighbdIdctFunc = void (*)(const tran_low_t *in, uint16_t *out, int stride,
                                int bd);

using HighbdIhtFunc = void (*)(const tran_low_t *in, uint16_t *out, int stride,
                               int tx_type, int bd);

template <HighbdIdctFunc fn>
void highbd_idct_wrapper(const tran_low_t *in, uint8_t *out, int stride,
                         int tx_type, int bd) {
  (void)tx_type;
  fn(in, CAST_TO_SHORTPTR(out), stride, bd);
}

template <HighbdIhtFunc fn>
void highbd_iht_wrapper(const tran_low_t *in, uint8_t *out, int stride,
                        int tx_type, int bd) {
  fn(in, CAST_TO_SHORTPTR(out), stride, tx_type, bd);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

struct FuncInfo {
  FhtFunc ft_func;
  IhtWithBdFunc it_func;
  int size;
  int pixel_size;
};

/* forward transform, inverse transform, size, transform type, bit depth */
using DctParam = tuple<int, const FuncInfo *, int, vpx_bit_depth_t>;

void fdct_ref(const Buffer<int16_t> &in, Buffer<tran_low_t> *out, int size,
              int /*tx_type*/) {
  const int16_t *i = in.TopLeftPixel();
  const int i_stride = in.stride();
  tran_low_t *o = out->TopLeftPixel();
  if (size == 4) {
    vpx_fdct4x4_c(i, o, i_stride);
  } else if (size == 8) {
    vpx_fdct8x8_c(i, o, i_stride);
  } else if (size == 16) {
    vpx_fdct16x16_c(i, o, i_stride);
  } else if (size == 32) {
    vpx_fdct32x32_c(i, o, i_stride);
  }
}

void fht_ref(const Buffer<int16_t> &in, Buffer<tran_low_t> *out, int size,
             int tx_type) {
  const int16_t *i = in.TopLeftPixel();
  const int i_stride = in.stride();
  tran_low_t *o = out->TopLeftPixel();
  if (size == 4) {
    vp9_fht4x4_c(i, o, i_stride, tx_type);
  } else if (size == 8) {
    vp9_fht8x8_c(i, o, i_stride, tx_type);
  } else if (size == 16) {
    vp9_fht16x16_c(i, o, i_stride, tx_type);
  }
}

void fwht_ref(const Buffer<int16_t> &in, Buffer<tran_low_t> *out, int size,
              int /*tx_type*/) {
  ASSERT_EQ(size, 4);
  vp9_fwht4x4_c(in.TopLeftPixel(), out->TopLeftPixel(), in.stride());
}

class TransTestBase : public ::testing::TestWithParam<DctParam> {
 public:
  void SetUp() override {
    rnd_.Reset(ACMRandom::DeterministicSeed());
    const int idx = GET_PARAM(0);
    const FuncInfo *func_info = &(GET_PARAM(1)[idx]);
    tx_type_ = GET_PARAM(2);
    bit_depth_ = GET_PARAM(3);
    fwd_txfm_ = func_info->ft_func;
    inv_txfm_ = func_info->it_func;
    size_ = func_info->size;
    pixel_size_ = func_info->pixel_size;
    max_pixel_value_ = (1 << bit_depth_) - 1;

    // Randomize stride_ to a value less than or equal to 1024
    stride_ = rnd_(1024) + 1;
    if (stride_ < size_) {
      stride_ = size_;
    }
    // Align stride_ to 16 if it's bigger than 16.
    if (stride_ > 16) {
      stride_ &= ~15;
    }

    block_size_ = size_ * stride_;

    src_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(16, pixel_size_ * block_size_));
    ASSERT_NE(src_, nullptr);
    dst_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(16, pixel_size_ * block_size_));
    ASSERT_NE(dst_, nullptr);
  }

  void TearDown() override {
    vpx_free(src_);
    src_ = nullptr;
    vpx_free(dst_);
    dst_ = nullptr;
    libvpx_test::ClearSystemState();
  }

  void InitMem() {
    if (pixel_size_ == 1 && bit_depth_ > VPX_BITS_8) return;
    if (pixel_size_ == 1) {
      for (int j = 0; j < block_size_; ++j) {
        src_[j] = rnd_.Rand16() & max_pixel_value_;
      }
      for (int j = 0; j < block_size_; ++j) {
        dst_[j] = rnd_.Rand16() & max_pixel_value_;
      }
    } else {
      ASSERT_EQ(pixel_size_, 2);
      uint16_t *const src = reinterpret_cast<uint16_t *>(src_);
      uint16_t *const dst = reinterpret_cast<uint16_t *>(dst_);
      for (int j = 0; j < block_size_; ++j) {
        src[j] = rnd_.Rand16() & max_pixel_value_;
      }
      for (int j = 0; j < block_size_; ++j) {
        dst[j] = rnd_.Rand16() & max_pixel_value_;
      }
    }
  }

  void RunFwdTxfm(const Buffer<int16_t> &in, Buffer<tran_low_t> *out) {
    fwd_txfm_(in.TopLeftPixel(), out->TopLeftPixel(), in.stride(), tx_type_);
  }

  void RunInvTxfm(const Buffer<tran_low_t> &in, uint8_t *out) {
    inv_txfm_(in.TopLeftPixel(), out, stride_, tx_type_, bit_depth_);
  }

 protected:
  void RunAccuracyCheck(int limit) {
    if (pixel_size_ == 1 && bit_depth_ > VPX_BITS_8) return;
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    Buffer<int16_t> test_input_block =
        Buffer<int16_t>(size_, size_, 8, size_ == 4 ? 0 : 16);
    ASSERT_TRUE(test_input_block.Init());
    ASSERT_NE(test_input_block.TopLeftPixel(), nullptr);
    Buffer<tran_low_t> test_temp_block =
        Buffer<tran_low_t>(size_, size_, 0, 16);
    ASSERT_TRUE(test_temp_block.Init());
    uint32_t max_error = 0;
    int64_t total_error = 0;
    const int count_test_block = 10000;
    for (int i = 0; i < count_test_block; ++i) {
      InitMem();
      for (int h = 0; h < size_; ++h) {
        for (int w = 0; w < size_; ++w) {
          if (pixel_size_ == 1) {
            test_input_block.TopLeftPixel()[h * test_input_block.stride() + w] =
                src_[h * stride_ + w] - dst_[h * stride_ + w];
          } else {
            ASSERT_EQ(pixel_size_, 2);
            const uint16_t *const src = reinterpret_cast<uint16_t *>(src_);
            const uint16_t *const dst = reinterpret_cast<uint16_t *>(dst_);
            test_input_block.TopLeftPixel()[h * test_input_block.stride() + w] =
                src[h * stride_ + w] - dst[h * stride_ + w];
          }
        }
      }

      ASM_REGISTER_STATE_CHECK(RunFwdTxfm(test_input_block, &test_temp_block));
      ASM_REGISTER_STATE_CHECK(RunInvTxfm(test_temp_block, dst_));

      for (int h = 0; h < size_; ++h) {
        for (int w = 0; w < size_; ++w) {
          int diff;
          if (pixel_size_ == 1) {
            diff = dst_[h * stride_ + w] - src_[h * stride_ + w];
          } else {
            ASSERT_EQ(pixel_size_, 2);
            const uint16_t *const src = reinterpret_cast<uint16_t *>(src_);
            const uint16_t *const dst = reinterpret_cast<uint16_t *>(dst_);
            diff = dst[h * stride_ + w] - src[h * stride_ + w];
          }
          const uint32_t error = diff * diff;
          if (max_error < error) max_error = error;
          total_error += error;
        }
      }
    }

    EXPECT_GE(static_cast<uint32_t>(limit), max_error)
        << "Error: " << size_ << "x" << size_
        << " transform/inverse transform has an individual round trip error > "
        << limit;

    EXPECT_GE(count_test_block * limit, total_error)
        << "Error: " << size_ << "x" << size_
        << " transform/inverse transform has average round trip error > "
        << limit << " per block";
  }

  void RunCoeffCheck() {
    if (pixel_size_ == 1 && bit_depth_ > VPX_BITS_8) return;
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int count_test_block = 5000;
    Buffer<int16_t> input_block =
        Buffer<int16_t>(size_, size_, 8, size_ == 4 ? 0 : 16);
    ASSERT_TRUE(input_block.Init());
    Buffer<tran_low_t> output_ref_block = Buffer<tran_low_t>(size_, size_, 0);
    ASSERT_TRUE(output_ref_block.Init());
    Buffer<tran_low_t> output_block = Buffer<tran_low_t>(size_, size_, 0, 16);
    ASSERT_TRUE(output_block.Init());

    for (int i = 0; i < count_test_block; ++i) {
      // Initialize a test block with input range [-max_pixel_value_,
      // max_pixel_value_].
      input_block.Set(&rnd, -max_pixel_value_, max_pixel_value_);

      fwd_txfm_ref(input_block, &output_ref_block, size_, tx_type_);
      ASM_REGISTER_STATE_CHECK(RunFwdTxfm(input_block, &output_block));

      // The minimum quant value is 4.
      EXPECT_TRUE(output_block.CheckValues(output_ref_block));
      if (::testing::Test::HasFailure()) {
        printf("Size: %d Transform type: %d\n", size_, tx_type_);
        output_block.PrintDifference(output_ref_block);
        return;
      }
    }
  }

  void RunMemCheck() {
    if (pixel_size_ == 1 && bit_depth_ > VPX_BITS_8) return;
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int count_test_block = 5000;
    Buffer<int16_t> input_extreme_block =
        Buffer<int16_t>(size_, size_, 8, size_ == 4 ? 0 : 16);
    ASSERT_TRUE(input_extreme_block.Init());
    Buffer<tran_low_t> output_ref_block = Buffer<tran_low_t>(size_, size_, 0);
    ASSERT_TRUE(output_ref_block.Init());
    Buffer<tran_low_t> output_block = Buffer<tran_low_t>(size_, size_, 0, 16);
    ASSERT_TRUE(output_block.Init());

    for (int i = 0; i < count_test_block; ++i) {
      // Initialize a test block with -max_pixel_value_ or max_pixel_value_.
      if (i == 0) {
        input_extreme_block.Set(max_pixel_value_);
      } else if (i == 1) {
        input_extreme_block.Set(-max_pixel_value_);
      } else {
        ASSERT_NE(input_extreme_block.TopLeftPixel(), nullptr);
        for (int h = 0; h < size_; ++h) {
          for (int w = 0; w < size_; ++w) {
            input_extreme_block
                .TopLeftPixel()[h * input_extreme_block.stride() + w] =
                rnd.Rand8() % 2 ? max_pixel_value_ : -max_pixel_value_;
          }
        }
      }

      fwd_txfm_ref(input_extreme_block, &output_ref_block, size_, tx_type_);
      ASM_REGISTER_STATE_CHECK(RunFwdTxfm(input_extreme_block, &output_block));

      // The minimum quant value is 4.
      EXPECT_TRUE(output_block.CheckValues(output_ref_block));
      ASSERT_NE(output_block.TopLeftPixel(), nullptr);
      for (int h = 0; h < size_; ++h) {
        for (int w = 0; w < size_; ++w) {
          EXPECT_GE(
              4 * DCT_MAX_VALUE << (bit_depth_ - 8),
              abs(output_block.TopLeftPixel()[h * output_block.stride() + w]))
              << "Error: " << size_ << "x" << size_
              << " transform has coefficient larger than 4*DCT_MAX_VALUE"
              << " at " << w << "," << h;
          if (::testing::Test::HasFailure()) {
            printf("Size: %d Transform type: %d\n", size_, tx_type_);
            output_block.DumpBuffer();
            return;
          }
        }
      }
    }
  }

  void RunInvAccuracyCheck(int limit) {
    if (pixel_size_ == 1 && bit_depth_ > VPX_BITS_8) return;
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int count_test_block = 1000;
    Buffer<int16_t> in = Buffer<int16_t>(size_, size_, 4);
    ASSERT_TRUE(in.Init());
    Buffer<tran_low_t> coeff = Buffer<tran_low_t>(size_, size_, 0, 16);
    ASSERT_TRUE(coeff.Init());

    for (int i = 0; i < count_test_block; ++i) {
      InitMem();
      ASSERT_NE(in.TopLeftPixel(), nullptr);
      // Initialize a test block with input range [-max_pixel_value_,
      // max_pixel_value_].
      for (int h = 0; h < size_; ++h) {
        for (int w = 0; w < size_; ++w) {
          if (pixel_size_ == 1) {
            in.TopLeftPixel()[h * in.stride() + w] =
                src_[h * stride_ + w] - dst_[h * stride_ + w];
          } else {
            ASSERT_EQ(pixel_size_, 2);
            const uint16_t *const src = reinterpret_cast<uint16_t *>(src_);
            const uint16_t *const dst = reinterpret_cast<uint16_t *>(dst_);
            in.TopLeftPixel()[h * in.stride() + w] =
                src[h * stride_ + w] - dst[h * stride_ + w];
          }
        }
      }

      fwd_txfm_ref(in, &coeff, size_, tx_type_);

      ASM_REGISTER_STATE_CHECK(RunInvTxfm(coeff, dst_));

      for (int h = 0; h < size_; ++h) {
        for (int w = 0; w < size_; ++w) {
          int diff;
          if (pixel_size_ == 1) {
            diff = dst_[h * stride_ + w] - src_[h * stride_ + w];
          } else {
            ASSERT_EQ(pixel_size_, 2);
            const uint16_t *const src = reinterpret_cast<uint16_t *>(src_);
            const uint16_t *const dst = reinterpret_cast<uint16_t *>(dst_);
            diff = dst[h * stride_ + w] - src[h * stride_ + w];
          }
          const uint32_t error = diff * diff;
          EXPECT_GE(static_cast<uint32_t>(limit), error)
              << "Error: " << size_ << "x" << size_
              << " inverse transform has error " << error << " at " << w << ","
              << h;
          if (::testing::Test::HasFailure()) {
            printf("Size: %d Transform type: %d\n", size_, tx_type_);
            return;
          }
        }
      }
    }
  }

  FhtFunc fwd_txfm_;
  FhtFuncRef fwd_txfm_ref;
  IhtWithBdFunc inv_txfm_;
  ACMRandom rnd_;
  uint8_t *src_;
  uint8_t *dst_;
  vpx_bit_depth_t bit_depth_;
  int tx_type_;
  int max_pixel_value_;
  int size_;
  int stride_;
  int pixel_size_;
  int block_size_;
};

/* -------------------------------------------------------------------------- */

class TransDCT : public TransTestBase {
 public:
  TransDCT() { fwd_txfm_ref = fdct_ref; }
};

TEST_P(TransDCT, AccuracyCheck) {
  int t = 1;
  if (size_ == 16 && bit_depth_ > 10 && pixel_size_ == 2) {
    t = 2;
  } else if (size_ == 32 && bit_depth_ > 10 && pixel_size_ == 2) {
    t = 7;
  }
  RunAccuracyCheck(t);
}

TEST_P(TransDCT, CoeffCheck) { RunCoeffCheck(); }

TEST_P(TransDCT, MemCheck) { RunMemCheck(); }

TEST_P(TransDCT, InvAccuracyCheck) { RunInvAccuracyCheck(1); }

static const FuncInfo dct_c_func_info[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  { &fdct_wrapper<vpx_highbd_fdct4x4_c>,
    &highbd_idct_wrapper<vpx_highbd_idct4x4_16_add_c>, 4, 2 },
  { &fdct_wrapper<vpx_highbd_fdct8x8_c>,
    &highbd_idct_wrapper<vpx_highbd_idct8x8_64_add_c>, 8, 2 },
  { &fdct_wrapper<vpx_highbd_fdct16x16_c>,
    &highbd_idct_wrapper<vpx_highbd_idct16x16_256_add_c>, 16, 2 },
  { &fdct_wrapper<vpx_highbd_fdct32x32_c>,
    &highbd_idct_wrapper<vpx_highbd_idct32x32_1024_add_c>, 32, 2 },
#endif
  { &fdct_wrapper<vpx_fdct4x4_c>, &idct_wrapper<vpx_idct4x4_16_add_c>, 4, 1 },
  { &fdct_wrapper<vpx_fdct8x8_c>, &idct_wrapper<vpx_idct8x8_64_add_c>, 8, 1 },
  { &fdct_wrapper<vpx_fdct16x16_c>, &idct_wrapper<vpx_idct16x16_256_add_c>, 16,
    1 },
  { &fdct_wrapper<vpx_fdct32x32_c>, &idct_wrapper<vpx_idct32x32_1024_add_c>, 32,
    1 }
};

INSTANTIATE_TEST_SUITE_P(
    C, TransDCT,
    ::testing::Combine(
        ::testing::Range(0, static_cast<int>(sizeof(dct_c_func_info) /
                                             sizeof(dct_c_func_info[0]))),
        ::testing::Values(dct_c_func_info), ::testing::Values(0),
        ::testing::Values(VPX_BITS_8, VPX_BITS_10, VPX_BITS_12)));

#if !CONFIG_EMULATE_HARDWARE

#if HAVE_SSE2
static const FuncInfo dct_sse2_func_info[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  { &fdct_wrapper<vpx_highbd_fdct4x4_sse2>,
    &highbd_idct_wrapper<vpx_highbd_idct4x4_16_add_sse2>, 4, 2 },
  { &fdct_wrapper<vpx_highbd_fdct8x8_sse2>,
    &highbd_idct_wrapper<vpx_highbd_idct8x8_64_add_sse2>, 8, 2 },
  { &fdct_wrapper<vpx_highbd_fdct16x16_sse2>,
    &highbd_idct_wrapper<vpx_highbd_idct16x16_256_add_sse2>, 16, 2 },
  { &fdct_wrapper<vpx_highbd_fdct32x32_sse2>,
    &highbd_idct_wrapper<vpx_highbd_idct32x32_1024_add_sse2>, 32, 2 },
#endif
  { &fdct_wrapper<vpx_fdct4x4_sse2>, &idct_wrapper<vpx_idct4x4_16_add_sse2>, 4,
    1 },
  { &fdct_wrapper<vpx_fdct8x8_sse2>, &idct_wrapper<vpx_idct8x8_64_add_sse2>, 8,
    1 },
  { &fdct_wrapper<vpx_fdct16x16_sse2>,
    &idct_wrapper<vpx_idct16x16_256_add_sse2>, 16, 1 },
  { &fdct_wrapper<vpx_fdct32x32_sse2>,
    &idct_wrapper<vpx_idct32x32_1024_add_sse2>, 32, 1 }
};

INSTANTIATE_TEST_SUITE_P(
    SSE2, TransDCT,
    ::testing::Combine(
        ::testing::Range(0, static_cast<int>(sizeof(dct_sse2_func_info) /
                                             sizeof(dct_sse2_func_info[0]))),
        ::testing::Values(dct_sse2_func_info), ::testing::Values(0),
        ::testing::Values(VPX_BITS_8, VPX_BITS_10, VPX_BITS_12)));
#endif  // HAVE_SSE2

#if HAVE_SSSE3 && !CONFIG_VP9_HIGHBITDEPTH && VPX_ARCH_X86_64
// vpx_fdct8x8_ssse3 is only available in 64 bit builds.
static const FuncInfo dct_ssse3_func_info = {
  &fdct_wrapper<vpx_fdct8x8_ssse3>, &idct_wrapper<vpx_idct8x8_64_add_sse2>, 8, 1
};

// TODO(johannkoenig): high bit depth fdct8x8.
INSTANTIATE_TEST_SUITE_P(SSSE3, TransDCT,
                         ::testing::Values(make_tuple(0, &dct_ssse3_func_info,
                                                      0, VPX_BITS_8)));
#endif  // HAVE_SSSE3 && !CONFIG_VP9_HIGHBITDEPTH && VPX_ARCH_X86_64

#if HAVE_AVX2 && !CONFIG_VP9_HIGHBITDEPTH
static const FuncInfo dct_avx2_func_info = {
  &fdct_wrapper<vpx_fdct32x32_avx2>, &idct_wrapper<vpx_idct32x32_1024_add_sse2>,
  32, 1
};

// TODO(johannkoenig): high bit depth fdct32x32.
INSTANTIATE_TEST_SUITE_P(AVX2, TransDCT,
                         ::testing::Values(make_tuple(0, &dct_avx2_func_info, 0,
                                                      VPX_BITS_8)));
#endif  // HAVE_AVX2 && !CONFIG_VP9_HIGHBITDEPTH

#if HAVE_NEON
#if CONFIG_VP9_HIGHBITDEPTH
static const FuncInfo dct_neon_func_info[] = {
  { &fdct_wrapper<vpx_highbd_fdct4x4_neon>,
    &highbd_idct_wrapper<vpx_highbd_idct4x4_16_add_neon>, 4, 2 },
  { &fdct_wrapper<vpx_highbd_fdct8x8_neon>,
    &highbd_idct_wrapper<vpx_highbd_idct8x8_64_add_neon>, 8, 2 },
  { &fdct_wrapper<vpx_highbd_fdct16x16_neon>,
    &highbd_idct_wrapper<vpx_highbd_idct16x16_256_add_neon>, 16, 2 },
  /* { &fdct_wrapper<vpx_highbd_fdct32x32_neon>,
       &highbd_idct_wrapper<vpx_highbd_idct32x32_1024_add_neon>, 32, 2 },*/
};
#else
static const FuncInfo dct_neon_func_info[4] = {
  { &fdct_wrapper<vpx_fdct4x4_neon>, &idct_wrapper<vpx_idct4x4_16_add_neon>, 4,
    1 },
  { &fdct_wrapper<vpx_fdct8x8_neon>, &idct_wrapper<vpx_idct8x8_64_add_neon>, 8,
    1 },
  { &fdct_wrapper<vpx_fdct16x16_neon>,
    &idct_wrapper<vpx_idct16x16_256_add_neon>, 16, 1 },
  { &fdct_wrapper<vpx_fdct32x32_neon>,
    &idct_wrapper<vpx_idct32x32_1024_add_neon>, 32, 1 }
};
#endif  // CONFIG_VP9_HIGHBITDEPTH

INSTANTIATE_TEST_SUITE_P(
    NEON, TransDCT,
    ::testing::Combine(
        ::testing::Range(0, static_cast<int>(sizeof(dct_neon_func_info) /
                                             sizeof(dct_neon_func_info[0]))),
        ::testing::Values(dct_neon_func_info), ::testing::Values(0),
        ::testing::Values(VPX_BITS_8, VPX_BITS_10, VPX_BITS_12)));
#endif  // HAVE_NEON

#if HAVE_MSA && !CONFIG_VP9_HIGHBITDEPTH
static const FuncInfo dct_msa_func_info[4] = {
  { &fdct_wrapper<vpx_fdct4x4_msa>, &idct_wrapper<vpx_idct4x4_16_add_msa>, 4,
    1 },
  { &fdct_wrapper<vpx_fdct8x8_msa>, &idct_wrapper<vpx_idct8x8_64_add_msa>, 8,
    1 },
  { &fdct_wrapper<vpx_fdct16x16_msa>, &idct_wrapper<vpx_idct16x16_256_add_msa>,
    16, 1 },
  { &fdct_wrapper<vpx_fdct32x32_msa>, &idct_wrapper<vpx_idct32x32_1024_add_msa>,
    32, 1 }
};

INSTANTIATE_TEST_SUITE_P(
    MSA, TransDCT,
    ::testing::Combine(::testing::Range(0, 4),
                       ::testing::Values(dct_msa_func_info),
                       ::testing::Values(0), ::testing::Values(VPX_BITS_8)));
#endif  // HAVE_MSA && !CONFIG_VP9_HIGHBITDEPTH

#if HAVE_VSX && !CONFIG_VP9_HIGHBITDEPTH
static const FuncInfo dct_vsx_func_info = {
  &fdct_wrapper<vpx_fdct4x4_c>, &idct_wrapper<vpx_idct4x4_16_add_vsx>, 4, 1
};

INSTANTIATE_TEST_SUITE_P(VSX, TransDCT,
                         ::testing::Values(make_tuple(0, &dct_vsx_func_info, 0,
                                                      VPX_BITS_8)));
#endif  // HAVE_VSX && !CONFIG_VP9_HIGHBITDEPTH &&

#if HAVE_LSX && !CONFIG_VP9_HIGHBITDEPTH
static const FuncInfo dct_lsx_func_info[4] = {
  { &fdct_wrapper<vpx_fdct4x4_lsx>, &idct_wrapper<vpx_idct4x4_16_add_c>, 4, 1 },
  { &fdct_wrapper<vpx_fdct8x8_lsx>, &idct_wrapper<vpx_idct8x8_64_add_c>, 8, 1 },
  { &fdct_wrapper<vpx_fdct16x16_lsx>, &idct_wrapper<vpx_idct16x16_256_add_c>,
    16, 1 },
  { &fdct_wrapper<vpx_fdct32x32_lsx>, &idct_wrapper<vpx_idct32x32_1024_add_lsx>,
    32, 1 }
};

INSTANTIATE_TEST_SUITE_P(
    LSX, TransDCT,
    ::testing::Combine(::testing::Range(0, 4),
                       ::testing::Values(dct_lsx_func_info),
                       ::testing::Values(0), ::testing::Values(VPX_BITS_8)));
#endif  // HAVE_LSX && !CONFIG_VP9_HIGHBITDEPTH

#endif  // !CONFIG_EMULATE_HARDWARE

/* -------------------------------------------------------------------------- */

class TransHT : public TransTestBase {
 public:
  TransHT() { fwd_txfm_ref = fht_ref; }
};

TEST_P(TransHT, AccuracyCheck) {
  RunAccuracyCheck(size_ == 16 && bit_depth_ > 10 && pixel_size_ == 2 ? 2 : 1);
}

TEST_P(TransHT, CoeffCheck) { RunCoeffCheck(); }

TEST_P(TransHT, MemCheck) { RunMemCheck(); }

TEST_P(TransHT, InvAccuracyCheck) { RunInvAccuracyCheck(1); }

static const FuncInfo ht_c_func_info[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  { &vp9_highbd_fht4x4_c, &highbd_iht_wrapper<vp9_highbd_iht4x4_16_add_c>, 4,
    2 },
  { &vp9_highbd_fht8x8_c, &highbd_iht_wrapper<vp9_highbd_iht8x8_64_add_c>, 8,
    2 },
  { &vp9_highbd_fht16x16_c, &highbd_iht_wrapper<vp9_highbd_iht16x16_256_add_c>,
    16, 2 },
#endif
  { &vp9_fht4x4_c, &iht_wrapper<vp9_iht4x4_16_add_c>, 4, 1 },
  { &vp9_fht8x8_c, &iht_wrapper<vp9_iht8x8_64_add_c>, 8, 1 },
  { &vp9_fht16x16_c, &iht_wrapper<vp9_iht16x16_256_add_c>, 16, 1 }
};

INSTANTIATE_TEST_SUITE_P(
    C, TransHT,
    ::testing::Combine(
        ::testing::Range(0, static_cast<int>(sizeof(ht_c_func_info) /
                                             sizeof(ht_c_func_info[0]))),
        ::testing::Values(ht_c_func_info), ::testing::Range(0, 4),
        ::testing::Values(VPX_BITS_8, VPX_BITS_10, VPX_BITS_12)));

#if !CONFIG_EMULATE_HARDWARE

#if HAVE_NEON

static const FuncInfo ht_neon_func_info[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  { &vp9_highbd_fht4x4_c, &highbd_iht_wrapper<vp9_highbd_iht4x4_16_add_neon>, 4,
    2 },
  { &vp9_highbd_fht4x4_neon, &highbd_iht_wrapper<vp9_highbd_iht4x4_16_add_neon>,
    4, 2 },
  { &vp9_highbd_fht8x8_c, &highbd_iht_wrapper<vp9_highbd_iht8x8_64_add_neon>, 8,
    2 },
  { &vp9_highbd_fht8x8_neon, &highbd_iht_wrapper<vp9_highbd_iht8x8_64_add_neon>,
    8, 2 },
  { &vp9_highbd_fht16x16_c,
    &highbd_iht_wrapper<vp9_highbd_iht16x16_256_add_neon>, 16, 2 },
  { &vp9_highbd_fht16x16_neon,
    &highbd_iht_wrapper<vp9_highbd_iht16x16_256_add_neon>, 16, 2 },
#endif
  { &vp9_fht4x4_c, &iht_wrapper<vp9_iht4x4_16_add_neon>, 4, 1 },
  { &vp9_fht4x4_neon, &iht_wrapper<vp9_iht4x4_16_add_neon>, 4, 1 },
  { &vp9_fht8x8_c, &iht_wrapper<vp9_iht8x8_64_add_neon>, 8, 1 },
  { &vp9_fht8x8_neon, &iht_wrapper<vp9_iht8x8_64_add_neon>, 8, 1 },
  { &vp9_fht16x16_c, &iht_wrapper<vp9_iht16x16_256_add_neon>, 16, 1 },
  { &vp9_fht16x16_neon, &iht_wrapper<vp9_iht16x16_256_add_neon>, 16, 1 }
};

INSTANTIATE_TEST_SUITE_P(
    NEON, TransHT,
    ::testing::Combine(
        ::testing::Range(0, static_cast<int>(sizeof(ht_neon_func_info) /
                                             sizeof(ht_neon_func_info[0]))),
        ::testing::Values(ht_neon_func_info), ::testing::Range(0, 4),
        ::testing::Values(VPX_BITS_8, VPX_BITS_10, VPX_BITS_12)));
#endif  // HAVE_NEON

#if HAVE_SSE2

static const FuncInfo ht_sse2_func_info[3] = {
  { &vp9_fht4x4_sse2, &iht_wrapper<vp9_iht4x4_16_add_sse2>, 4, 1 },
  { &vp9_fht8x8_sse2, &iht_wrapper<vp9_iht8x8_64_add_sse2>, 8, 1 },
  { &vp9_fht16x16_sse2, &iht_wrapper<vp9_iht16x16_256_add_sse2>, 16, 1 }
};

INSTANTIATE_TEST_SUITE_P(
    SSE2, TransHT,
    ::testing::Combine(::testing::Range(0, 3),
                       ::testing::Values(ht_sse2_func_info),
                       ::testing::Range(0, 4), ::testing::Values(VPX_BITS_8)));
#endif  // HAVE_SSE2

#if HAVE_SSE4_1 && CONFIG_VP9_HIGHBITDEPTH
static const FuncInfo ht_sse4_1_func_info[3] = {
  { &vp9_highbd_fht4x4_c, &highbd_iht_wrapper<vp9_highbd_iht4x4_16_add_sse4_1>,
    4, 2 },
  { vp9_highbd_fht8x8_c, &highbd_iht_wrapper<vp9_highbd_iht8x8_64_add_sse4_1>,
    8, 2 },
  { &vp9_highbd_fht16x16_c,
    &highbd_iht_wrapper<vp9_highbd_iht16x16_256_add_sse4_1>, 16, 2 }
};

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, TransHT,
    ::testing::Combine(::testing::Range(0, 3),
                       ::testing::Values(ht_sse4_1_func_info),
                       ::testing::Range(0, 4),
                       ::testing::Values(VPX_BITS_8, VPX_BITS_10,
                                         VPX_BITS_12)));
#endif  // HAVE_SSE4_1 && CONFIG_VP9_HIGHBITDEPTH

#if HAVE_VSX && !CONFIG_EMULATE_HARDWARE && !CONFIG_VP9_HIGHBITDEPTH
static const FuncInfo ht_vsx_func_info[3] = {
  { &vp9_fht4x4_c, &iht_wrapper<vp9_iht4x4_16_add_vsx>, 4, 1 },
  { &vp9_fht8x8_c, &iht_wrapper<vp9_iht8x8_64_add_vsx>, 8, 1 },
  { &vp9_fht16x16_c, &iht_wrapper<vp9_iht16x16_256_add_vsx>, 16, 1 }
};

INSTANTIATE_TEST_SUITE_P(VSX, TransHT,
                         ::testing::Combine(::testing::Range(0, 3),
                                            ::testing::Values(ht_vsx_func_info),
                                            ::testing::Range(0, 4),
                                            ::testing::Values(VPX_BITS_8)));
#endif  // HAVE_VSX
#endif  // !CONFIG_EMULATE_HARDWARE

/* -------------------------------------------------------------------------- */

class TransWHT : public TransTestBase {
 public:
  TransWHT() { fwd_txfm_ref = fwht_ref; }
};

TEST_P(TransWHT, AccuracyCheck) { RunAccuracyCheck(0); }

TEST_P(TransWHT, CoeffCheck) { RunCoeffCheck(); }

TEST_P(TransWHT, MemCheck) { RunMemCheck(); }

TEST_P(TransWHT, InvAccuracyCheck) { RunInvAccuracyCheck(0); }

static const FuncInfo wht_c_func_info[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  { &fdct_wrapper<vp9_highbd_fwht4x4_c>,
    &highbd_idct_wrapper<vpx_highbd_iwht4x4_16_add_c>, 4, 2 },
#endif
  { &fdct_wrapper<vp9_fwht4x4_c>, &idct_wrapper<vpx_iwht4x4_16_add_c>, 4, 1 }
};

INSTANTIATE_TEST_SUITE_P(
    C, TransWHT,
    ::testing::Combine(
        ::testing::Range(0, static_cast<int>(sizeof(wht_c_func_info) /
                                             sizeof(wht_c_func_info[0]))),
        ::testing::Values(wht_c_func_info), ::testing::Values(0),
        ::testing::Values(VPX_BITS_8, VPX_BITS_10, VPX_BITS_12)));

#if HAVE_SSE2 && !CONFIG_EMULATE_HARDWARE
static const FuncInfo wht_sse2_func_info = {
  &fdct_wrapper<vp9_fwht4x4_sse2>, &idct_wrapper<vpx_iwht4x4_16_add_sse2>, 4, 1
};

INSTANTIATE_TEST_SUITE_P(SSE2, TransWHT,
                         ::testing::Values(make_tuple(0, &wht_sse2_func_info, 0,
                                                      VPX_BITS_8)));
#endif  // HAVE_SSE2 && !CONFIG_EMULATE_HARDWARE

#if HAVE_VSX && !CONFIG_EMULATE_HARDWARE && !CONFIG_VP9_HIGHBITDEPTH
static const FuncInfo wht_vsx_func_info = {
  &fdct_wrapper<vp9_fwht4x4_c>, &idct_wrapper<vpx_iwht4x4_16_add_vsx>, 4, 1
};

INSTANTIATE_TEST_SUITE_P(VSX, TransWHT,
                         ::testing::Values(make_tuple(0, &wht_vsx_func_info, 0,
                                                      VPX_BITS_8)));
#endif  // HAVE_VSX && !CONFIG_EMULATE_HARDWARE

}  // namespace
