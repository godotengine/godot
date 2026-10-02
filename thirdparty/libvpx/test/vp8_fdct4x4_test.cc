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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vp8_rtcd.h"
#include "test/acm_random.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

namespace {

using FdctFunc = void (*)(int16_t *a, int16_t *b, int a_stride);

const int cospi8sqrt2minus1 = 20091;
const int sinpi8sqrt2 = 35468;

void reference_idct4x4(const int16_t *input, int16_t *output) {
  const int16_t *ip = input;
  int16_t *op = output;

  for (int i = 0; i < 4; ++i) {
    const int a1 = ip[0] + ip[8];
    const int b1 = ip[0] - ip[8];
    const int temp1 = (ip[4] * sinpi8sqrt2) >> 16;
    const int temp2 = ip[12] + ((ip[12] * cospi8sqrt2minus1) >> 16);
    const int c1 = temp1 - temp2;
    const int temp3 = ip[4] + ((ip[4] * cospi8sqrt2minus1) >> 16);
    const int temp4 = (ip[12] * sinpi8sqrt2) >> 16;
    const int d1 = temp3 + temp4;
    op[0] = a1 + d1;
    op[12] = a1 - d1;
    op[4] = b1 + c1;
    op[8] = b1 - c1;
    ++ip;
    ++op;
  }
  ip = output;
  op = output;
  for (int i = 0; i < 4; ++i) {
    const int a1 = ip[0] + ip[2];
    const int b1 = ip[0] - ip[2];
    const int temp1 = (ip[1] * sinpi8sqrt2) >> 16;
    const int temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1) >> 16);
    const int c1 = temp1 - temp2;
    const int temp3 = ip[1] + ((ip[1] * cospi8sqrt2minus1) >> 16);
    const int temp4 = (ip[3] * sinpi8sqrt2) >> 16;
    const int d1 = temp3 + temp4;
    op[0] = (a1 + d1 + 4) >> 3;
    op[3] = (a1 - d1 + 4) >> 3;
    op[1] = (b1 + c1 + 4) >> 3;
    op[2] = (b1 - c1 + 4) >> 3;
    ip += 4;
    op += 4;
  }
}

using libvpx_test::ACMRandom;

class FdctTest : public ::testing::TestWithParam<FdctFunc> {
 public:
  void SetUp() override {
    fdct_func_ = GetParam();
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

 protected:
  FdctFunc fdct_func_;
  ACMRandom rnd_;
};

TEST_P(FdctTest, SignBiasCheck) {
  int16_t test_input_block[16];
  DECLARE_ALIGNED(16, int16_t, test_output_block[16]);
  const int pitch = 8;
  int count_sign_block[16][2];
  const int count_test_block = 1000000;

  memset(count_sign_block, 0, sizeof(count_sign_block));

  for (int i = 0; i < count_test_block; ++i) {
    // Initialize a test block with input range [-255, 255].
    for (int j = 0; j < 16; ++j) {
      test_input_block[j] = rnd_.Rand8() - rnd_.Rand8();
    }

    fdct_func_(test_input_block, test_output_block, pitch);

    for (int j = 0; j < 16; ++j) {
      if (test_output_block[j] < 0) {
        ++count_sign_block[j][0];
      } else if (test_output_block[j] > 0) {
        ++count_sign_block[j][1];
      }
    }
  }

  bool bias_acceptable = true;
  for (int j = 0; j < 16; ++j) {
    bias_acceptable =
        bias_acceptable &&
        (abs(count_sign_block[j][0] - count_sign_block[j][1]) < 10000);
  }

  EXPECT_EQ(true, bias_acceptable)
      << "Error: 4x4 FDCT has a sign bias > 1% for input range [-255, 255]";

  memset(count_sign_block, 0, sizeof(count_sign_block));

  for (int i = 0; i < count_test_block; ++i) {
    // Initialize a test block with input range [-15, 15].
    for (int j = 0; j < 16; ++j) {
      test_input_block[j] = (rnd_.Rand8() >> 4) - (rnd_.Rand8() >> 4);
    }

    fdct_func_(test_input_block, test_output_block, pitch);

    for (int j = 0; j < 16; ++j) {
      if (test_output_block[j] < 0) {
        ++count_sign_block[j][0];
      } else if (test_output_block[j] > 0) {
        ++count_sign_block[j][1];
      }
    }
  }

  bias_acceptable = true;
  for (int j = 0; j < 16; ++j) {
    bias_acceptable =
        bias_acceptable &&
        (abs(count_sign_block[j][0] - count_sign_block[j][1]) < 100000);
  }

  EXPECT_EQ(true, bias_acceptable)
      << "Error: 4x4 FDCT has a sign bias > 10% for input range [-15, 15]";
}

TEST_P(FdctTest, RoundTripErrorCheck) {
  int max_error = 0;
  double total_error = 0;
  const int count_test_block = 1000000;
  for (int i = 0; i < count_test_block; ++i) {
    int16_t test_input_block[16];
    int16_t test_output_block[16];
    DECLARE_ALIGNED(16, int16_t, test_temp_block[16]);

    // Initialize a test block with input range [-255, 255].
    for (int j = 0; j < 16; ++j) {
      test_input_block[j] = rnd_.Rand8() - rnd_.Rand8();
    }

    const int pitch = 8;
    fdct_func_(test_input_block, test_temp_block, pitch);
    reference_idct4x4(test_temp_block, test_output_block);

    for (int j = 0; j < 16; ++j) {
      const int diff = test_input_block[j] - test_output_block[j];
      const int error = diff * diff;
      if (max_error < error) max_error = error;
      total_error += error;
    }
  }

  EXPECT_GE(1, max_error)
      << "Error: FDCT/IDCT has an individual roundtrip error > 1";

  EXPECT_GE(count_test_block, total_error)
      << "Error: FDCT/IDCT has average roundtrip error > 1 per block";
}

INSTANTIATE_TEST_SUITE_P(C, FdctTest, ::testing::Values(vp8_short_fdct4x4_c));

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, FdctTest,
                         ::testing::Values(vp8_short_fdct4x4_neon));
#endif  // HAVE_NEON

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, FdctTest,
                         ::testing::Values(vp8_short_fdct4x4_sse2));
#endif  // HAVE_SSE2

#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(MSA, FdctTest,
                         ::testing::Values(vp8_short_fdct4x4_msa));
#endif  // HAVE_MSA
#if HAVE_MMI
INSTANTIATE_TEST_SUITE_P(MMI, FdctTest,
                         ::testing::Values(vp8_short_fdct4x4_mmi));
#endif  // HAVE_MMI

#if HAVE_LSX
INSTANTIATE_TEST_SUITE_P(LSX, FdctTest,
                         ::testing::Values(vp8_short_fdct4x4_lsx));
#endif  // HAVE_LSX
}  // namespace
