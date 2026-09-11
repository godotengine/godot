/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <tuple>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/vpx_timer.h"

using libvpx_test::ACMRandom;
using ::testing::Combine;
using ::testing::Range;
using ::testing::ValuesIn;

namespace {
const int kNumIterations = 10000;

using SSI16Func = uint64_t (*)(const int16_t *src, int stride, int size);
using SumSquaresParam = std::tuple<SSI16Func, SSI16Func>;

class SumSquaresTest : public ::testing::TestWithParam<SumSquaresParam> {
 public:
  ~SumSquaresTest() override = default;
  void SetUp() override {
    ref_func_ = GET_PARAM(0);
    tst_func_ = GET_PARAM(1);
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  SSI16Func ref_func_;
  SSI16Func tst_func_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SumSquaresTest);

TEST_P(SumSquaresTest, OperationCheck) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  DECLARE_ALIGNED(16, int16_t, src[256 * 256]);
  const int msb = 11;  // Up to 12 bit input
  const int limit = 1 << (msb + 1);

  for (int k = 0; k < kNumIterations; k++) {
    const int size = 4 << rnd(6);  // Up to 128x128
    int stride = 4 << rnd(7);      // Up to 256 stride
    while (stride < size) {        // Make sure it's valid
      stride = 4 << rnd(7);
    }

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        src[i * stride + j] = rnd(2) ? rnd(limit) : -rnd(limit);
      }
    }

    const uint64_t res_ref = ref_func_(src, stride, size);
    uint64_t res_tst;
    ASM_REGISTER_STATE_CHECK(res_tst = tst_func_(src, stride, size));

    ASSERT_EQ(res_ref, res_tst) << "Error: Sum Squares Test"
                                << " C output does not match optimized output.";
  }
}

TEST_P(SumSquaresTest, ExtremeValues) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  DECLARE_ALIGNED(16, int16_t, src[256 * 256]);
  const int msb = 11;  // Up to 12 bit input
  const int limit = 1 << (msb + 1);

  for (int k = 0; k < kNumIterations; k++) {
    const int size = 4 << rnd(6);  // Up to 128x128
    int stride = 4 << rnd(7);      // Up to 256 stride
    while (stride < size) {        // Make sure it's valid
      stride = 4 << rnd(7);
    }

    const int val = rnd(2) ? limit - 1 : -(limit - 1);
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        src[i * stride + j] = val;
      }
    }

    const uint64_t res_ref = ref_func_(src, stride, size);
    uint64_t res_tst;
    ASM_REGISTER_STATE_CHECK(res_tst = tst_func_(src, stride, size));

    ASSERT_EQ(res_ref, res_tst) << "Error: Sum Squares Test"
                                << " C output does not match optimized output.";
  }
}

using std::make_tuple;

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, SumSquaresTest,
    ::testing::Values(make_tuple(&vpx_sum_squares_2d_i16_c,
                                 &vpx_sum_squares_2d_i16_neon)));
#endif  // HAVE_NEON

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, SumSquaresTest,
    ::testing::Values(make_tuple(&vpx_sum_squares_2d_i16_c,
                                 &vpx_sum_squares_2d_i16_sve)));
#endif  // HAVE_SVE

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, SumSquaresTest,
    ::testing::Values(make_tuple(&vpx_sum_squares_2d_i16_c,
                                 &vpx_sum_squares_2d_i16_sse2)));
#endif  // HAVE_SSE2

#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(
    MSA, SumSquaresTest,
    ::testing::Values(make_tuple(&vpx_sum_squares_2d_i16_c,
                                 &vpx_sum_squares_2d_i16_msa)));
#endif  // HAVE_MSA

using SSEFunc = int64_t (*)(const uint8_t *a, int a_stride, const uint8_t *b,
                            int b_stride, int width, int height);

struct TestSSEFuncs {
  TestSSEFuncs(SSEFunc ref = nullptr, SSEFunc tst = nullptr, int depth = 0)
      : ref_func(ref), tst_func(tst), bit_depth(depth) {}
  SSEFunc ref_func;  // Pointer to reference function
  SSEFunc tst_func;  // Pointer to tested function
  int bit_depth;
};

using SSETestParam = std::tuple<TestSSEFuncs, int>;

class SSETest : public ::testing::TestWithParam<SSETestParam> {
 public:
  ~SSETest() override = default;
  void SetUp() override {
    params_ = GET_PARAM(0);
    width_ = GET_PARAM(1);
    is_hbd_ =
#if CONFIG_VP9_HIGHBITDEPTH
        params_.ref_func == vpx_highbd_sse_c;
#else
        false;
#endif
    rnd_.Reset(ACMRandom::DeterministicSeed());
    src_ = reinterpret_cast<uint8_t *>(vpx_memalign(32, 256 * 256 * 2));
    ref_ = reinterpret_cast<uint8_t *>(vpx_memalign(32, 256 * 256 * 2));
    ASSERT_NE(src_, nullptr);
    ASSERT_NE(ref_, nullptr);
  }

  void TearDown() override {
    vpx_free(src_);
    vpx_free(ref_);
  }
  void RunTest(bool is_random, int width, int height, int run_times);

  void GenRandomData(int width, int height, int stride) {
    uint16_t *src16 = reinterpret_cast<uint16_t *>(src_);
    uint16_t *ref16 = reinterpret_cast<uint16_t *>(ref_);
    const int msb = 11;  // Up to 12 bit input
    const int limit = 1 << (msb + 1);
    for (int ii = 0; ii < height; ii++) {
      for (int jj = 0; jj < width; jj++) {
        if (!is_hbd_) {
          src_[ii * stride + jj] = rnd_.Rand8();
          ref_[ii * stride + jj] = rnd_.Rand8();
        } else {
          src16[ii * stride + jj] = rnd_(limit);
          ref16[ii * stride + jj] = rnd_(limit);
        }
      }
    }
  }

  void GenExtremeData(int width, int height, int stride, uint8_t *data,
                      int16_t val) {
    uint16_t *data16 = reinterpret_cast<uint16_t *>(data);
    for (int ii = 0; ii < height; ii++) {
      for (int jj = 0; jj < width; jj++) {
        if (!is_hbd_) {
          data[ii * stride + jj] = static_cast<uint8_t>(val);
        } else {
          data16[ii * stride + jj] = val;
        }
      }
    }
  }

 protected:
  bool is_hbd_;
  int width_;
  TestSSEFuncs params_;
  uint8_t *src_;
  uint8_t *ref_;
  ACMRandom rnd_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SSETest);

void SSETest::RunTest(bool is_random, int width, int height, int run_times) {
  int failed = 0;
  vpx_usec_timer ref_timer, test_timer;
  for (int k = 0; k < 3; k++) {
    int stride = 4 << rnd_(7);  // Up to 256 stride
    while (stride < width) {    // Make sure it's valid
      stride = 4 << rnd_(7);
    }
    if (is_random) {
      GenRandomData(width, height, stride);
    } else {
      const int msb = is_hbd_ ? 12 : 8;  // Up to 12 bit input
      const int limit = (1 << msb) - 1;
      if (k == 0) {
        GenExtremeData(width, height, stride, src_, 0);
        GenExtremeData(width, height, stride, ref_, limit);
      } else {
        GenExtremeData(width, height, stride, src_, limit);
        GenExtremeData(width, height, stride, ref_, 0);
      }
    }
    int64_t res_ref, res_tst;
    uint8_t *src = src_;
    uint8_t *ref = ref_;
#if CONFIG_VP9_HIGHBITDEPTH
    if (is_hbd_) {
      src = CONVERT_TO_BYTEPTR(src_);
      ref = CONVERT_TO_BYTEPTR(ref_);
    }
#endif
    res_ref = params_.ref_func(src, stride, ref, stride, width, height);
    res_tst = params_.tst_func(src, stride, ref, stride, width, height);
    if (run_times > 1) {
      vpx_usec_timer_start(&ref_timer);
      for (int j = 0; j < run_times; j++) {
        params_.ref_func(src, stride, ref, stride, width, height);
      }
      vpx_usec_timer_mark(&ref_timer);
      const int elapsed_time_c =
          static_cast<int>(vpx_usec_timer_elapsed(&ref_timer));

      vpx_usec_timer_start(&test_timer);
      for (int j = 0; j < run_times; j++) {
        params_.tst_func(src, stride, ref, stride, width, height);
      }
      vpx_usec_timer_mark(&test_timer);
      const int elapsed_time_simd =
          static_cast<int>(vpx_usec_timer_elapsed(&test_timer));

      printf(
          "c_time=%d \t simd_time=%d \t "
          "gain=%d\n",
          elapsed_time_c, elapsed_time_simd,
          (elapsed_time_c / elapsed_time_simd));
    } else {
      if (!failed) {
        failed = res_ref != res_tst;
        EXPECT_EQ(res_ref, res_tst)
            << "Error:" << (is_hbd_ ? "hbd " : " ") << k << " SSE Test ["
            << width << "x" << height
            << "] C output does not match optimized output.";
      }
    }
  }
}

TEST_P(SSETest, OperationCheck) {
  for (int height = 4; height <= 128; height += 4) {
    RunTest(true, width_, height, 1);  // GenRandomData
  }
}

TEST_P(SSETest, ExtremeValues) {
  for (int height = 4; height <= 128; height += 4) {
    RunTest(false, width_, height, 1);
  }
}

TEST_P(SSETest, DISABLED_Speed) {
  for (int height = 4; height <= 128; height += 4) {
    RunTest(true, width_, height, 100);
  }
}

#if HAVE_NEON
TestSSEFuncs sse_neon[] = {
  TestSSEFuncs(&vpx_sse_c, &vpx_sse_neon),
#if CONFIG_VP9_HIGHBITDEPTH
  TestSSEFuncs(&vpx_highbd_sse_c, &vpx_highbd_sse_neon)
#endif
};
INSTANTIATE_TEST_SUITE_P(NEON, SSETest,
                         Combine(ValuesIn(sse_neon), Range(4, 129, 4)));
#endif  // HAVE_NEON

#if HAVE_NEON_DOTPROD
TestSSEFuncs sse_neon_dotprod[] = {
  TestSSEFuncs(&vpx_sse_c, &vpx_sse_neon_dotprod),
};
INSTANTIATE_TEST_SUITE_P(NEON_DOTPROD, SSETest,
                         Combine(ValuesIn(sse_neon_dotprod), Range(4, 129, 4)));
#endif  // HAVE_NEON_DOTPROD

#if HAVE_SSE4_1
TestSSEFuncs sse_sse4[] = {
  TestSSEFuncs(&vpx_sse_c, &vpx_sse_sse4_1),
#if CONFIG_VP9_HIGHBITDEPTH
  TestSSEFuncs(&vpx_highbd_sse_c, &vpx_highbd_sse_sse4_1)
#endif
};
INSTANTIATE_TEST_SUITE_P(SSE4_1, SSETest,
                         Combine(ValuesIn(sse_sse4), Range(4, 129, 4)));
#endif  // HAVE_SSE4_1

#if HAVE_AVX2

TestSSEFuncs sse_avx2[] = {
  TestSSEFuncs(&vpx_sse_c, &vpx_sse_avx2),
#if CONFIG_VP9_HIGHBITDEPTH
  TestSSEFuncs(&vpx_highbd_sse_c, &vpx_highbd_sse_avx2)
#endif
};
INSTANTIATE_TEST_SUITE_P(AVX2, SSETest,
                         Combine(ValuesIn(sse_avx2), Range(4, 129, 4)));
#endif  // HAVE_AVX2
}  // namespace
