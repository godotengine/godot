/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <algorithm>

#include "gtest/gtest.h"

#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/vpx_timer.h"

#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "vpx_config.h"

namespace {

using ::libvpx_test::ACMRandom;

using HadamardFunc = void (*)(const int16_t *a, ptrdiff_t a_stride,
                              tran_low_t *b);

void hadamard_loop(const tran_low_t *a, tran_low_t *out) {
  tran_low_t b[8];
  for (int i = 0; i < 8; i += 2) {
    b[i + 0] = a[i * 8] + a[(i + 1) * 8];
    b[i + 1] = a[i * 8] - a[(i + 1) * 8];
  }
  tran_low_t c[8];
  for (int i = 0; i < 8; i += 4) {
    c[i + 0] = b[i + 0] + b[i + 2];
    c[i + 1] = b[i + 1] + b[i + 3];
    c[i + 2] = b[i + 0] - b[i + 2];
    c[i + 3] = b[i + 1] - b[i + 3];
  }
  out[0] = c[0] + c[4];
  out[7] = c[1] + c[5];
  out[3] = c[2] + c[6];
  out[4] = c[3] + c[7];
  out[2] = c[0] - c[4];
  out[6] = c[1] - c[5];
  out[1] = c[2] - c[6];
  out[5] = c[3] - c[7];
}

void reference_hadamard8x8(const int16_t *a, int a_stride, tran_low_t *b) {
  tran_low_t input[64];
  tran_low_t buf[64];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      input[i * 8 + j] = static_cast<tran_low_t>(a[i * a_stride + j]);
    }
  }
  for (int i = 0; i < 8; ++i) hadamard_loop(input + i, buf + i * 8);
  for (int i = 0; i < 8; ++i) hadamard_loop(buf + i, b + i * 8);
}

void reference_hadamard16x16(const int16_t *a, int a_stride, tran_low_t *b) {
  /* The source is a 16x16 block. The destination is rearranged to 8x32.
   * Input is 9 bit. */
  reference_hadamard8x8(a + 0 + 0 * a_stride, a_stride, b + 0);
  reference_hadamard8x8(a + 8 + 0 * a_stride, a_stride, b + 64);
  reference_hadamard8x8(a + 0 + 8 * a_stride, a_stride, b + 128);
  reference_hadamard8x8(a + 8 + 8 * a_stride, a_stride, b + 192);

  /* Overlay the 8x8 blocks and combine. */
  for (int i = 0; i < 64; ++i) {
    /* 8x8 steps the range up to 15 bits. */
    const tran_low_t a0 = b[0];
    const tran_low_t a1 = b[64];
    const tran_low_t a2 = b[128];
    const tran_low_t a3 = b[192];

    /* Prevent the result from escaping int16_t. */
    const tran_low_t b0 = (a0 + a1) >> 1;
    const tran_low_t b1 = (a0 - a1) >> 1;
    const tran_low_t b2 = (a2 + a3) >> 1;
    const tran_low_t b3 = (a2 - a3) >> 1;

    /* Store a 16 bit value. */
    b[0] = b0 + b2;
    b[64] = b1 + b3;
    b[128] = b0 - b2;
    b[192] = b1 - b3;

    ++b;
  }
}

void reference_hadamard32x32(const int16_t *a, int a_stride, tran_low_t *b) {
  reference_hadamard16x16(a + 0 + 0 * a_stride, a_stride, b + 0);
  reference_hadamard16x16(a + 16 + 0 * a_stride, a_stride, b + 256);
  reference_hadamard16x16(a + 0 + 16 * a_stride, a_stride, b + 512);
  reference_hadamard16x16(a + 16 + 16 * a_stride, a_stride, b + 768);

  for (int i = 0; i < 256; ++i) {
    const tran_low_t a0 = b[0];
    const tran_low_t a1 = b[256];
    const tran_low_t a2 = b[512];
    const tran_low_t a3 = b[768];

    const tran_low_t b0 = (a0 + a1) >> 2;
    const tran_low_t b1 = (a0 - a1) >> 2;
    const tran_low_t b2 = (a2 + a3) >> 2;
    const tran_low_t b3 = (a2 - a3) >> 2;

    b[0] = b0 + b2;
    b[256] = b1 + b3;
    b[512] = b0 - b2;
    b[768] = b1 - b3;

    ++b;
  }
}

struct HadamardFuncWithSize {
  HadamardFuncWithSize(HadamardFunc f, int s) : func(f), block_size(s) {}
  HadamardFunc func;
  int block_size;
};

std::ostream &operator<<(std::ostream &os, const HadamardFuncWithSize &hfs) {
  return os << "block size: " << hfs.block_size;
}

class HadamardTestBase : public ::testing::TestWithParam<HadamardFuncWithSize> {
 public:
  void SetUp() override {
    h_func_ = GetParam().func;
    bwh_ = GetParam().block_size;
    block_size_ = bwh_ * bwh_;
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  // The Rand() function generates values in the range [-((1 << BitDepth) - 1),
  // (1 << BitDepth) - 1]. This is because the input to the Hadamard transform
  // is the residual pixel, which is defined as 'source pixel - predicted
  // pixel'. Source pixel and predicted pixel take values in the range
  // [0, (1 << BitDepth) - 1] and thus the residual pixel ranges from
  // -((1 << BitDepth) - 1) to ((1 << BitDepth) - 1).
  virtual int16_t Rand() = 0;

  void ReferenceHadamard(const int16_t *a, int a_stride, tran_low_t *b,
                         int bwh) {
    if (bwh == 32)
      reference_hadamard32x32(a, a_stride, b);
    else if (bwh == 16)
      reference_hadamard16x16(a, a_stride, b);
    else
      reference_hadamard8x8(a, a_stride, b);
  }

  void CompareReferenceRandom() {
    const int kMaxBlockSize = 32 * 32;
    DECLARE_ALIGNED(16, int16_t, a[kMaxBlockSize]);
    DECLARE_ALIGNED(16, tran_low_t, b[kMaxBlockSize]);
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));

    tran_low_t b_ref[kMaxBlockSize];
    memset(b_ref, 0, sizeof(b_ref));

    for (int i = 0; i < block_size_; ++i) a[i] = Rand();

    ReferenceHadamard(a, bwh_, b_ref, bwh_);
    ASM_REGISTER_STATE_CHECK(h_func_(a, bwh_, b));

    // The order of the output is not important. Sort before checking.
    std::sort(b, b + block_size_);
    std::sort(b_ref, b_ref + block_size_);
    EXPECT_EQ(0, memcmp(b, b_ref, sizeof(b)));
  }

  void ExtremeValuesTest() {
    const int kMaxBlockSize = 32 * 32;
    DECLARE_ALIGNED(16, int16_t, input_extreme_block[kMaxBlockSize]);
    DECLARE_ALIGNED(16, tran_low_t, b[kMaxBlockSize]);
    memset(b, 0, sizeof(b));

    tran_low_t b_ref[kMaxBlockSize];
    memset(b_ref, 0, sizeof(b_ref));

    for (int i = 0; i < 2; ++i) {
      // Initialize a test block with input range [-mask_, mask_].
      const int sign = (i == 0) ? 1 : -1;
      for (int j = 0; j < kMaxBlockSize; ++j)
        input_extreme_block[j] = sign * 255;

      ReferenceHadamard(input_extreme_block, bwh_, b_ref, bwh_);
      ASM_REGISTER_STATE_CHECK(h_func_(input_extreme_block, bwh_, b));

      // The order of the output is not important. Sort before checking.
      std::sort(b, b + block_size_);
      std::sort(b_ref, b_ref + block_size_);
      EXPECT_EQ(0, memcmp(b, b_ref, sizeof(b)));
    }
  }

  void VaryStride() {
    const int kMaxBlockSize = 32 * 32;
    DECLARE_ALIGNED(16, int16_t, a[kMaxBlockSize * 8]);
    DECLARE_ALIGNED(16, tran_low_t, b[kMaxBlockSize]);
    memset(a, 0, sizeof(a));
    for (int i = 0; i < block_size_ * 8; ++i) a[i] = Rand();

    tran_low_t b_ref[kMaxBlockSize];
    for (int i = 8; i < 64; i += 8) {
      memset(b, 0, sizeof(b));
      memset(b_ref, 0, sizeof(b_ref));

      ReferenceHadamard(a, i, b_ref, bwh_);
      ASM_REGISTER_STATE_CHECK(h_func_(a, i, b));

      // The order of the output is not important. Sort before checking.
      std::sort(b, b + block_size_);
      std::sort(b_ref, b_ref + block_size_);
      EXPECT_EQ(0, memcmp(b, b_ref, sizeof(b)));
    }
  }

  void SpeedTest(int times) {
    const int kMaxBlockSize = 32 * 32;
    DECLARE_ALIGNED(16, int16_t, input[kMaxBlockSize]);
    DECLARE_ALIGNED(16, tran_low_t, output[kMaxBlockSize]);
    memset(input, 1, sizeof(input));
    memset(output, 0, sizeof(output));

    vpx_usec_timer timer;
    vpx_usec_timer_start(&timer);
    for (int i = 0; i < times; ++i) {
      h_func_(input, bwh_, output);
    }
    vpx_usec_timer_mark(&timer);

    const int elapsed_time = static_cast<int>(vpx_usec_timer_elapsed(&timer));
    printf("Hadamard%dx%d[%12d runs]: %d us\n", bwh_, bwh_, times,
           elapsed_time);
  }

 protected:
  int bwh_;
  int block_size_;
  HadamardFunc h_func_;
  ACMRandom rnd_;
};

class HadamardLowbdTest : public HadamardTestBase {
 protected:
  // Use values between -255 (0xFF01) and 255 (0x00FF)
  int16_t Rand() override {
    int16_t src = rnd_.Rand8();
    int16_t pred = rnd_.Rand8();
    return src - pred;
  }
};

TEST_P(HadamardLowbdTest, CompareReferenceRandom) { CompareReferenceRandom(); }

TEST_P(HadamardLowbdTest, ExtremeValuesTest) { ExtremeValuesTest(); }

TEST_P(HadamardLowbdTest, VaryStride) { VaryStride(); }

TEST_P(HadamardLowbdTest, DISABLED_Speed) {
  SpeedTest(10);
  SpeedTest(10000);
  SpeedTest(10000000);
}

INSTANTIATE_TEST_SUITE_P(
    C, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_hadamard_8x8_c, 8),
                      HadamardFuncWithSize(&vpx_hadamard_16x16_c, 16),
                      HadamardFuncWithSize(&vpx_hadamard_32x32_c, 32)));

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_hadamard_8x8_sse2, 8),
                      HadamardFuncWithSize(&vpx_hadamard_16x16_sse2, 16),
                      HadamardFuncWithSize(&vpx_hadamard_32x32_sse2, 32)));
#endif  // HAVE_SSE2

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_hadamard_16x16_avx2, 16),
                      HadamardFuncWithSize(&vpx_hadamard_32x32_avx2, 32)));
#endif  // HAVE_AVX2

#if HAVE_SSSE3 && VPX_ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSSE3, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_hadamard_8x8_ssse3, 8)));
#endif  // HAVE_SSSE3 && VPX_ARCH_X86_64

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_hadamard_8x8_neon, 8),
                      HadamardFuncWithSize(&vpx_hadamard_16x16_neon, 16),
                      HadamardFuncWithSize(&vpx_hadamard_32x32_neon, 32)));
#endif  // HAVE_NEON

// TODO(jingning): Remove highbitdepth flag when the SIMD functions are
// in place and turn on the unit test.
#if !CONFIG_VP9_HIGHBITDEPTH
#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(
    MSA, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_hadamard_8x8_msa, 8),
                      HadamardFuncWithSize(&vpx_hadamard_16x16_msa, 16)));
#endif  // HAVE_MSA
#endif  // !CONFIG_VP9_HIGHBITDEPTH

#if HAVE_VSX
INSTANTIATE_TEST_SUITE_P(
    VSX, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_hadamard_8x8_vsx, 8),
                      HadamardFuncWithSize(&vpx_hadamard_16x16_vsx, 16)));
#endif  // HAVE_VSX

#if HAVE_LSX
INSTANTIATE_TEST_SUITE_P(
    LSX, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_hadamard_8x8_lsx, 8),
                      HadamardFuncWithSize(&vpx_hadamard_16x16_lsx, 16)));
#endif  // HAVE_LSX

#if CONFIG_VP9_HIGHBITDEPTH
class HadamardHighbdTest : public HadamardTestBase {
 protected:
  // Use values between -4095 (0xF001) and 4095 (0x0FFF)
  int16_t Rand() override {
    int16_t src = rnd_.Rand12();
    int16_t pred = rnd_.Rand12();
    return src - pred;
  }
};

TEST_P(HadamardHighbdTest, CompareReferenceRandom) { CompareReferenceRandom(); }

TEST_P(HadamardHighbdTest, VaryStride) { VaryStride(); }

TEST_P(HadamardHighbdTest, DISABLED_Speed) {
  SpeedTest(10);
  SpeedTest(10000);
  SpeedTest(10000000);
}

INSTANTIATE_TEST_SUITE_P(
    C, HadamardHighbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_highbd_hadamard_8x8_c, 8),
                      HadamardFuncWithSize(&vpx_highbd_hadamard_16x16_c, 16),
                      HadamardFuncWithSize(&vpx_highbd_hadamard_32x32_c, 32)));

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, HadamardHighbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_highbd_hadamard_8x8_avx2, 8),
                      HadamardFuncWithSize(&vpx_highbd_hadamard_16x16_avx2, 16),
                      HadamardFuncWithSize(&vpx_highbd_hadamard_32x32_avx2,
                                           32)));
#endif  // HAVE_AVX2

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, HadamardHighbdTest,
    ::testing::Values(HadamardFuncWithSize(&vpx_highbd_hadamard_8x8_neon, 8),
                      HadamardFuncWithSize(&vpx_highbd_hadamard_16x16_neon, 16),
                      HadamardFuncWithSize(&vpx_highbd_hadamard_32x32_neon,
                                           32)));
#endif

#endif  // CONFIG_VP9_HIGHBITDEPTH
}  // namespace
