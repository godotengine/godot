/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "./vp8_rtcd.h"

#include "gtest/gtest.h"

#include "test/buffer.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "vpx/vpx_integer.h"

using IdctFunc = void (*)(int16_t *input, unsigned char *pred_ptr,
                          int pred_stride, unsigned char *dst_ptr,
                          int dst_stride);
namespace {

using libvpx_test::Buffer;

class IDCTTest : public ::testing::TestWithParam<IdctFunc> {
 protected:
  void SetUp() override {
    UUT = GetParam();

    input = new Buffer<int16_t>(4, 4, 0);
    ASSERT_NE(input, nullptr);
    ASSERT_TRUE(input->Init());
    predict = new Buffer<uint8_t>(4, 4, 3);
    ASSERT_NE(predict, nullptr);
    ASSERT_TRUE(predict->Init());
    output = new Buffer<uint8_t>(4, 4, 3);
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(output->Init());
  }

  void TearDown() override {
    delete input;
    delete predict;
    delete output;
    libvpx_test::ClearSystemState();
  }

  IdctFunc UUT;
  Buffer<int16_t> *input;
  Buffer<uint8_t> *predict;
  Buffer<uint8_t> *output;
};

TEST_P(IDCTTest, TestAllZeros) {
  // When the input is '0' the output will be '0'.
  input->Set(0);
  predict->Set(0);
  output->Set(0);

  ASM_REGISTER_STATE_CHECK(UUT(input->TopLeftPixel(), predict->TopLeftPixel(),
                               predict->stride(), output->TopLeftPixel(),
                               output->stride()));

  ASSERT_TRUE(input->CheckValues(0));
  ASSERT_TRUE(input->CheckPadding());
  ASSERT_TRUE(output->CheckValues(0));
  ASSERT_TRUE(output->CheckPadding());
}

TEST_P(IDCTTest, TestAllOnes) {
  input->Set(0);
  ASSERT_NE(input->TopLeftPixel(), nullptr);
  // When the first element is '4' it will fill the output buffer with '1'.
  input->TopLeftPixel()[0] = 4;
  predict->Set(0);
  output->Set(0);

  ASM_REGISTER_STATE_CHECK(UUT(input->TopLeftPixel(), predict->TopLeftPixel(),
                               predict->stride(), output->TopLeftPixel(),
                               output->stride()));

  ASSERT_TRUE(output->CheckValues(1));
  ASSERT_TRUE(output->CheckPadding());
}

TEST_P(IDCTTest, TestAddOne) {
  // Set the transform output to '1' and make sure it gets added to the
  // prediction buffer.
  input->Set(0);
  ASSERT_NE(input->TopLeftPixel(), nullptr);
  input->TopLeftPixel()[0] = 4;
  output->Set(0);

  uint8_t *pred = predict->TopLeftPixel();
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      pred[y * predict->stride() + x] = y * 4 + x;
    }
  }

  ASM_REGISTER_STATE_CHECK(UUT(input->TopLeftPixel(), predict->TopLeftPixel(),
                               predict->stride(), output->TopLeftPixel(),
                               output->stride()));

  uint8_t const *out = output->TopLeftPixel();
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      EXPECT_EQ(1 + y * 4 + x, out[y * output->stride() + x]);
    }
  }

  if (HasFailure()) {
    output->DumpBuffer();
  }

  ASSERT_TRUE(output->CheckPadding());
}

TEST_P(IDCTTest, TestWithData) {
  // Test a single known input.
  predict->Set(0);

  int16_t *in = input->TopLeftPixel();
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      in[y * input->stride() + x] = y * 4 + x;
    }
  }

  ASM_REGISTER_STATE_CHECK(UUT(input->TopLeftPixel(), predict->TopLeftPixel(),
                               predict->stride(), output->TopLeftPixel(),
                               output->stride()));

  uint8_t *out = output->TopLeftPixel();
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      switch (y * 4 + x) {
        case 0: EXPECT_EQ(11, out[y * output->stride() + x]); break;
        case 2:
        case 5:
        case 8: EXPECT_EQ(3, out[y * output->stride() + x]); break;
        case 10: EXPECT_EQ(1, out[y * output->stride() + x]); break;
        default: EXPECT_EQ(0, out[y * output->stride() + x]);
      }
    }
  }

  if (HasFailure()) {
    output->DumpBuffer();
  }

  ASSERT_TRUE(output->CheckPadding());
}

INSTANTIATE_TEST_SUITE_P(C, IDCTTest,
                         ::testing::Values(vp8_short_idct4x4llm_c));

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, IDCTTest,
                         ::testing::Values(vp8_short_idct4x4llm_neon));
#endif  // HAVE_NEON

#if HAVE_MMX
INSTANTIATE_TEST_SUITE_P(MMX, IDCTTest,
                         ::testing::Values(vp8_short_idct4x4llm_mmx));
#endif  // HAVE_MMX

#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(MSA, IDCTTest,
                         ::testing::Values(vp8_short_idct4x4llm_msa));
#endif  // HAVE_MSA

#if HAVE_MMI
INSTANTIATE_TEST_SUITE_P(MMI, IDCTTest,
                         ::testing::Values(vp8_short_idct4x4llm_mmi));
#endif  // HAVE_MMI
}  // namespace
