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
#include <limits>
#include <tuple>

#include "gtest/gtest.h"

#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/buffer.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vpx_config.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

using libvpx_test::ACMRandom;
using libvpx_test::Buffer;
using std::make_tuple;
using std::tuple;

namespace {
using PartialFdctFunc = void (*)(const int16_t *in, tran_low_t *out,
                                 int stride);

using PartialFdctParam = tuple<PartialFdctFunc, int /*size*/, vpx_bit_depth_t>;

tran_low_t partial_fdct_ref(const Buffer<int16_t> &in, int size) {
  int64_t sum = 0;
  if (in.TopLeftPixel() != nullptr) {
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        sum += in.TopLeftPixel()[y * in.stride() + x];
      }
    }
  } else {
    assert(0);
  }

  switch (size) {
    case 4: sum *= 2; break;
    case 8: /*sum = sum;*/ break;
    case 16: sum >>= 1; break;
    case 32: sum >>= 3; break;
  }

  return static_cast<tran_low_t>(sum);
}

class PartialFdctTest : public ::testing::TestWithParam<PartialFdctParam> {
 public:
  PartialFdctTest() {
    fwd_txfm_ = GET_PARAM(0);
    size_ = GET_PARAM(1);
    bit_depth_ = GET_PARAM(2);
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  void RunTest() {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int16_t maxvalue =
        clip_pixel_highbd(std::numeric_limits<int16_t>::max(), bit_depth_);
    const int16_t minvalue = -maxvalue;
    Buffer<int16_t> input_block =
        Buffer<int16_t>(size_, size_, 8, size_ == 4 ? 0 : 16);
    ASSERT_TRUE(input_block.Init());
    Buffer<tran_low_t> output_block = Buffer<tran_low_t>(size_, size_, 0, 16);
    ASSERT_TRUE(output_block.Init());

    if (output_block.TopLeftPixel() != nullptr) {
      for (int i = 0; i < 100; ++i) {
        if (i == 0) {
          input_block.Set(maxvalue);
        } else if (i == 1) {
          input_block.Set(minvalue);
        } else {
          input_block.Set(&rnd, minvalue, maxvalue);
        }

        ASM_REGISTER_STATE_CHECK(fwd_txfm_(input_block.TopLeftPixel(),
                                           output_block.TopLeftPixel(),
                                           input_block.stride()));

        EXPECT_EQ(partial_fdct_ref(input_block, size_),
                  output_block.TopLeftPixel()[0]);
      }
    } else {
      assert(0);
    }
  }

  PartialFdctFunc fwd_txfm_;
  vpx_bit_depth_t bit_depth_;
  int size_;
};

TEST_P(PartialFdctTest, PartialFdctTest) { RunTest(); }

#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    C, PartialFdctTest,
    ::testing::Values(make_tuple(&vpx_highbd_fdct32x32_1_c, 32, VPX_BITS_12),
                      make_tuple(&vpx_highbd_fdct32x32_1_c, 32, VPX_BITS_10),
                      make_tuple(&vpx_fdct32x32_1_c, 32, VPX_BITS_8),
                      make_tuple(&vpx_highbd_fdct16x16_1_c, 16, VPX_BITS_12),
                      make_tuple(&vpx_highbd_fdct16x16_1_c, 16, VPX_BITS_10),
                      make_tuple(&vpx_fdct16x16_1_c, 16, VPX_BITS_8),
                      make_tuple(&vpx_highbd_fdct8x8_1_c, 8, VPX_BITS_12),
                      make_tuple(&vpx_highbd_fdct8x8_1_c, 8, VPX_BITS_10),
                      make_tuple(&vpx_fdct8x8_1_c, 8, VPX_BITS_8),
                      make_tuple(&vpx_fdct4x4_1_c, 4, VPX_BITS_8)));
#else
INSTANTIATE_TEST_SUITE_P(
    C, PartialFdctTest,
    ::testing::Values(make_tuple(&vpx_fdct32x32_1_c, 32, VPX_BITS_8),
                      make_tuple(&vpx_fdct16x16_1_c, 16, VPX_BITS_8),
                      make_tuple(&vpx_fdct8x8_1_c, 8, VPX_BITS_8),
                      make_tuple(&vpx_fdct4x4_1_c, 4, VPX_BITS_8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, PartialFdctTest,
    ::testing::Values(make_tuple(&vpx_fdct32x32_1_sse2, 32, VPX_BITS_8),
                      make_tuple(&vpx_fdct16x16_1_sse2, 16, VPX_BITS_8),
                      make_tuple(&vpx_fdct8x8_1_sse2, 8, VPX_BITS_8),
                      make_tuple(&vpx_fdct4x4_1_sse2, 4, VPX_BITS_8)));
#endif  // HAVE_SSE2

#if HAVE_NEON
#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    NEON, PartialFdctTest,
    ::testing::Values(make_tuple(&vpx_highbd_fdct32x32_1_neon, 32, VPX_BITS_12),
                      make_tuple(&vpx_highbd_fdct32x32_1_neon, 32, VPX_BITS_10),
                      make_tuple(&vpx_highbd_fdct32x32_1_neon, 32, VPX_BITS_8),
                      make_tuple(&vpx_highbd_fdct16x16_1_neon, 16, VPX_BITS_12),
                      make_tuple(&vpx_highbd_fdct16x16_1_neon, 16, VPX_BITS_10),
                      make_tuple(&vpx_highbd_fdct16x16_1_neon, 16, VPX_BITS_8),
                      make_tuple(&vpx_fdct8x8_1_neon, 8, VPX_BITS_12),
                      make_tuple(&vpx_fdct8x8_1_neon, 8, VPX_BITS_10),
                      make_tuple(&vpx_fdct8x8_1_neon, 8, VPX_BITS_8),
                      make_tuple(&vpx_fdct4x4_1_neon, 4, VPX_BITS_12),
                      make_tuple(&vpx_fdct4x4_1_neon, 4, VPX_BITS_10),
                      make_tuple(&vpx_fdct4x4_1_neon, 4, VPX_BITS_8)));
#else
INSTANTIATE_TEST_SUITE_P(
    NEON, PartialFdctTest,
    ::testing::Values(make_tuple(&vpx_fdct32x32_1_neon, 32, VPX_BITS_8),
                      make_tuple(&vpx_fdct16x16_1_neon, 16, VPX_BITS_8),
                      make_tuple(&vpx_fdct8x8_1_neon, 8, VPX_BITS_8),
                      make_tuple(&vpx_fdct4x4_1_neon, 4, VPX_BITS_8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // HAVE_NEON

#if HAVE_MSA
#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(MSA, PartialFdctTest,
                         ::testing::Values(make_tuple(&vpx_fdct8x8_1_msa, 8,
                                                      VPX_BITS_8)));
#else   // !CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    MSA, PartialFdctTest,
    ::testing::Values(make_tuple(&vpx_fdct32x32_1_msa, 32, VPX_BITS_8),
                      make_tuple(&vpx_fdct16x16_1_msa, 16, VPX_BITS_8),
                      make_tuple(&vpx_fdct8x8_1_msa, 8, VPX_BITS_8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // HAVE_MSA
}  // namespace
