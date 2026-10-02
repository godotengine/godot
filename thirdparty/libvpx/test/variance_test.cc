/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <cstdlib>
#include <new>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/variance.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/vpx_timer.h"

namespace {

using Get4x4SseFunc = unsigned int (*)(const uint8_t *a, int a_stride,
                                       const uint8_t *b, int b_stride);
using GetVarianceFunc = void (*)(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *ref_ptr, int ref_stride,
                                 uint32_t *sse, int *sum);
using SumOfSquaresFunction = unsigned int (*)(const int16_t *src);

using libvpx_test::ACMRandom;

// Truncate high bit depth results by downshifting (with rounding) by:
// 2 * (bit_depth - 8) for sse
// (bit_depth - 8) for se
static void RoundHighBitDepth(int bit_depth, int64_t *se, uint64_t *sse) {
  switch (bit_depth) {
    case VPX_BITS_12:
      *sse = (*sse + 128) >> 8;
      *se = (*se + 8) >> 4;
      break;
    case VPX_BITS_10:
      *sse = (*sse + 8) >> 4;
      *se = (*se + 2) >> 2;
      break;
    case VPX_BITS_8:
    default: break;
  }
}

static unsigned int mb_ss_ref(const int16_t *src) {
  unsigned int res = 0;
  for (int i = 0; i < 256; ++i) {
    res += src[i] * src[i];
  }
  return res;
}

/* Note:
 *  Our codebase calculates the "diff" value in the variance algorithm by
 *  (src - ref).
 */
static void variance(const uint8_t *src, int src_stride, const uint8_t *ref,
                     int ref_stride, int w, int h, bool use_high_bit_depth_,
                     uint64_t *sse, int64_t *se, vpx_bit_depth_t bit_depth) {
  int64_t se_long = 0;
  uint64_t sse_long = 0;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int diff = 0;
      if (!use_high_bit_depth_) {
        diff = src[y * src_stride + x] - ref[y * ref_stride + x];
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        diff = CONVERT_TO_SHORTPTR(src)[y * src_stride + x] -
               CONVERT_TO_SHORTPTR(ref)[y * ref_stride + x];
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
      se_long += diff;
      sse_long += diff * diff;
    }
  }

  RoundHighBitDepth(bit_depth, &se_long, &sse_long);

  *sse = sse_long;
  *se = se_long;
}

static void get_variance_ref(const uint8_t *src, int src_stride,
                             const uint8_t *ref, int ref_stride, int l2w,
                             int l2h, bool use_high_bit_depth_, uint32_t *sse,
                             int *se, vpx_bit_depth_t bit_depth) {
  const int w = 1 << l2w;
  const int h = 1 << l2h;
  int64_t se_long = 0;
  uint64_t sse_long = 0;

  variance(src, src_stride, ref, ref_stride, w, h, use_high_bit_depth_,
           &sse_long, &se_long, bit_depth);

  *sse = static_cast<uint32_t>(sse_long);
  *se = static_cast<int>(se_long);
}

static uint32_t variance_ref(const uint8_t *src, const uint8_t *ref, int l2w,
                             int l2h, int src_stride, int ref_stride,
                             uint32_t *sse_ptr, bool use_high_bit_depth_,
                             vpx_bit_depth_t bit_depth) {
  const int w = 1 << l2w;
  const int h = 1 << l2h;
  int64_t se_long = 0;
  uint64_t sse_long = 0;

  variance(src, src_stride, ref, ref_stride, w, h, use_high_bit_depth_,
           &sse_long, &se_long, bit_depth);

  *sse_ptr = static_cast<uint32_t>(sse_long);
  return static_cast<uint32_t>(
      sse_long - ((static_cast<int64_t>(se_long) * se_long) >> (l2w + l2h)));
}

/* The subpel reference functions differ from the codec version in one aspect:
 * they calculate the bilinear factors directly instead of using a lookup table
 * and therefore upshift xoff and yoff by 1. Only every other calculated value
 * is used so the codec version shrinks the table to save space and maintain
 * compatibility with vp8.
 */
static uint32_t subpel_variance_ref(const uint8_t *ref, const uint8_t *src,
                                    int l2w, int l2h, int xoff, int yoff,
                                    uint32_t *sse_ptr, bool use_high_bit_depth_,
                                    vpx_bit_depth_t bit_depth) {
  int64_t se = 0;
  uint64_t sse = 0;
  const int w = 1 << l2w;
  const int h = 1 << l2h;

  xoff <<= 1;
  yoff <<= 1;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // Bilinear interpolation at a 16th pel step.
      if (!use_high_bit_depth_) {
        const int a1 = ref[(w + 1) * (y + 0) + x + 0];
        const int a2 = ref[(w + 1) * (y + 0) + x + 1];
        const int b1 = ref[(w + 1) * (y + 1) + x + 0];
        const int b2 = ref[(w + 1) * (y + 1) + x + 1];
        const int a = a1 + (((a2 - a1) * xoff + 8) >> 4);
        const int b = b1 + (((b2 - b1) * xoff + 8) >> 4);
        const int r = a + (((b - a) * yoff + 8) >> 4);
        const int diff = r - src[w * y + x];
        se += diff;
        sse += diff * diff;
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        uint16_t *ref16 = CONVERT_TO_SHORTPTR(ref);
        uint16_t *src16 = CONVERT_TO_SHORTPTR(src);
        const int a1 = ref16[(w + 1) * (y + 0) + x + 0];
        const int a2 = ref16[(w + 1) * (y + 0) + x + 1];
        const int b1 = ref16[(w + 1) * (y + 1) + x + 0];
        const int b2 = ref16[(w + 1) * (y + 1) + x + 1];
        const int a = a1 + (((a2 - a1) * xoff + 8) >> 4);
        const int b = b1 + (((b2 - b1) * xoff + 8) >> 4);
        const int r = a + (((b - a) * yoff + 8) >> 4);
        const int diff = r - src16[w * y + x];
        se += diff;
        sse += diff * diff;
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
    }
  }
  RoundHighBitDepth(bit_depth, &se, &sse);
  *sse_ptr = static_cast<uint32_t>(sse);
  return static_cast<uint32_t>(
      sse - ((static_cast<int64_t>(se) * se) >> (l2w + l2h)));
}

static uint32_t subpel_avg_variance_ref(const uint8_t *ref, const uint8_t *src,
                                        const uint8_t *second_pred, int l2w,
                                        int l2h, int xoff, int yoff,
                                        uint32_t *sse_ptr,
                                        bool use_high_bit_depth,
                                        vpx_bit_depth_t bit_depth) {
  int64_t se = 0;
  uint64_t sse = 0;
  const int w = 1 << l2w;
  const int h = 1 << l2h;

  xoff <<= 1;
  yoff <<= 1;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // bilinear interpolation at a 16th pel step
      if (!use_high_bit_depth) {
        const int a1 = ref[(w + 1) * (y + 0) + x + 0];
        const int a2 = ref[(w + 1) * (y + 0) + x + 1];
        const int b1 = ref[(w + 1) * (y + 1) + x + 0];
        const int b2 = ref[(w + 1) * (y + 1) + x + 1];
        const int a = a1 + (((a2 - a1) * xoff + 8) >> 4);
        const int b = b1 + (((b2 - b1) * xoff + 8) >> 4);
        const int r = a + (((b - a) * yoff + 8) >> 4);
        const int diff =
            ((r + second_pred[w * y + x] + 1) >> 1) - src[w * y + x];
        se += diff;
        sse += diff * diff;
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        const uint16_t *ref16 = CONVERT_TO_SHORTPTR(ref);
        const uint16_t *src16 = CONVERT_TO_SHORTPTR(src);
        const uint16_t *sec16 = CONVERT_TO_SHORTPTR(second_pred);
        const int a1 = ref16[(w + 1) * (y + 0) + x + 0];
        const int a2 = ref16[(w + 1) * (y + 0) + x + 1];
        const int b1 = ref16[(w + 1) * (y + 1) + x + 0];
        const int b2 = ref16[(w + 1) * (y + 1) + x + 1];
        const int a = a1 + (((a2 - a1) * xoff + 8) >> 4);
        const int b = b1 + (((b2 - b1) * xoff + 8) >> 4);
        const int r = a + (((b - a) * yoff + 8) >> 4);
        const int diff = ((r + sec16[w * y + x] + 1) >> 1) - src16[w * y + x];
        se += diff;
        sse += diff * diff;
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
    }
  }
  RoundHighBitDepth(bit_depth, &se, &sse);
  *sse_ptr = static_cast<uint32_t>(sse);
  return static_cast<uint32_t>(
      sse - ((static_cast<int64_t>(se) * se) >> (l2w + l2h)));
}

////////////////////////////////////////////////////////////////////////////////

class SumOfSquaresTest : public ::testing::TestWithParam<SumOfSquaresFunction> {
 public:
  SumOfSquaresTest() : func_(GetParam()) {}

  ~SumOfSquaresTest() override { libvpx_test::ClearSystemState(); }

 protected:
  void ConstTest();
  void RefTest();

  SumOfSquaresFunction func_;
  ACMRandom rnd_;
};

void SumOfSquaresTest::ConstTest() {
  int16_t mem[256];
  unsigned int res;
  for (int v = 0; v < 256; ++v) {
    for (int i = 0; i < 256; ++i) {
      mem[i] = v;
    }
    ASM_REGISTER_STATE_CHECK(res = func_(mem));
    EXPECT_EQ(256u * (v * v), res);
  }
}

void SumOfSquaresTest::RefTest() {
  int16_t mem[256];
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 256; ++j) {
      mem[j] = rnd_.Rand8() - rnd_.Rand8();
    }

    const unsigned int expected = mb_ss_ref(mem);
    unsigned int res;
    ASM_REGISTER_STATE_CHECK(res = func_(mem));
    EXPECT_EQ(expected, res);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Encapsulating struct to store the function to test along with
// some testing context.
// Can be used for MSE, SSE, Variance, etc.

template <typename Func>
struct TestParams {
  TestParams(int log2w = 0, int log2h = 0, Func function = nullptr,
             int bit_depth_value = 0)
      : log2width(log2w), log2height(log2h), func(function) {
    use_high_bit_depth = (bit_depth_value > 0);
    if (use_high_bit_depth) {
      bit_depth = static_cast<vpx_bit_depth_t>(bit_depth_value);
    } else {
      bit_depth = VPX_BITS_8;
    }
    width = 1 << log2width;
    height = 1 << log2height;
    block_size = width * height;
    mask = (1u << bit_depth) - 1;
  }

  int log2width, log2height;
  int width, height;
  int block_size;
  Func func;
  vpx_bit_depth_t bit_depth;
  bool use_high_bit_depth;
  uint32_t mask;
};

template <typename Func>
std::ostream &operator<<(std::ostream &os, const TestParams<Func> &p) {
  return os << "log2width/height:" << p.log2width << "/" << p.log2height
            << " function:" << reinterpret_cast<const void *>(p.func)
            << " bit-depth:" << p.bit_depth;
}

// Main class for testing a function type
template <typename FunctionType>
class MainTestClass
    : public ::testing::TestWithParam<TestParams<FunctionType> > {
 public:
  void SetUp() override {
    params_ = this->GetParam();

    rnd_.Reset(ACMRandom::DeterministicSeed());
    const size_t unit =
        use_high_bit_depth() ? sizeof(uint16_t) : sizeof(uint8_t);
    src_ = reinterpret_cast<uint8_t *>(vpx_memalign(16, block_size() * unit));
    ref_ = new uint8_t[block_size() * unit];
    ASSERT_NE(src_, nullptr);
    ASSERT_NE(ref_, nullptr);
#if CONFIG_VP9_HIGHBITDEPTH
    if (use_high_bit_depth()) {
      // TODO(skal): remove!
      src_ = CONVERT_TO_BYTEPTR(src_);
      ref_ = CONVERT_TO_BYTEPTR(ref_);
    }
#endif
  }

  void TearDown() override {
#if CONFIG_VP9_HIGHBITDEPTH
    if (use_high_bit_depth()) {
      // TODO(skal): remove!
      src_ = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(src_));
      ref_ = reinterpret_cast<uint8_t *>(CONVERT_TO_SHORTPTR(ref_));
    }
#endif

    vpx_free(src_);
    delete[] ref_;
    src_ = nullptr;
    ref_ = nullptr;
    libvpx_test::ClearSystemState();
  }

 protected:
  // We could sub-class MainTestClass into dedicated class for Variance
  // and MSE/SSE, but it involves a lot of 'this->xxx' dereferencing
  // to access top class fields xxx. That's cumbersome, so for now we'll just
  // implement the testing methods here:

  // Variance tests
  void ZeroTest();
  void RefTest();
  void RefStrideTest();
  void OneQuarterTest();
  void SpeedTest();

  // GetVariance tests
  void RefTestGetVar();

  // MSE/SSE tests
  void RefTestMse();
  void RefTestSse();
  void MaxTestMse();
  void MaxTestSse();

 protected:
  ACMRandom rnd_;
  uint8_t *src_;
  uint8_t *ref_;
  TestParams<FunctionType> params_;

  // some relay helpers
  bool use_high_bit_depth() const { return params_.use_high_bit_depth; }
  int byte_shift() const { return params_.bit_depth - 8; }
  int block_size() const { return params_.block_size; }
  int width() const { return params_.width; }
  int height() const { return params_.height; }
  uint32_t mask() const { return params_.mask; }
};

////////////////////////////////////////////////////////////////////////////////
// Tests related to variance.

template <typename VarianceFunctionType>
void MainTestClass<VarianceFunctionType>::ZeroTest() {
  for (int i = 0; i <= 255; ++i) {
    if (!use_high_bit_depth()) {
      memset(src_, i, block_size());
    } else {
      uint16_t *const src16 = CONVERT_TO_SHORTPTR(src_);
      for (int k = 0; k < block_size(); ++k) src16[k] = i << byte_shift();
    }
    for (int j = 0; j <= 255; ++j) {
      if (!use_high_bit_depth()) {
        memset(ref_, j, block_size());
      } else {
        uint16_t *const ref16 = CONVERT_TO_SHORTPTR(ref_);
        for (int k = 0; k < block_size(); ++k) ref16[k] = j << byte_shift();
      }
      unsigned int sse, var;
      ASM_REGISTER_STATE_CHECK(
          var = params_.func(src_, width(), ref_, width(), &sse));
      EXPECT_EQ(0u, var) << "src values: " << i << " ref values: " << j;
    }
  }
}

template <typename VarianceFunctionType>
void MainTestClass<VarianceFunctionType>::RefTest() {
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < block_size(); j++) {
      if (!use_high_bit_depth()) {
        src_[j] = rnd_.Rand8();
        ref_[j] = rnd_.Rand8();
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        CONVERT_TO_SHORTPTR(src_)[j] = rnd_.Rand16() & mask();
        CONVERT_TO_SHORTPTR(ref_)[j] = rnd_.Rand16() & mask();
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
    }
    unsigned int sse1, sse2, var1, var2;
    const int stride = width();
    ASM_REGISTER_STATE_CHECK(
        var1 = params_.func(src_, stride, ref_, stride, &sse1));
    var2 =
        variance_ref(src_, ref_, params_.log2width, params_.log2height, stride,
                     stride, &sse2, use_high_bit_depth(), params_.bit_depth);
    EXPECT_EQ(sse1, sse2) << "Error at test index: " << i;
    EXPECT_EQ(var1, var2) << "Error at test index: " << i;
  }
}

template <typename VarianceFunctionType>
void MainTestClass<VarianceFunctionType>::RefStrideTest() {
  for (int i = 0; i < 10; ++i) {
    const int ref_stride = (i & 1) * width();
    const int src_stride = ((i >> 1) & 1) * width();
    for (int j = 0; j < block_size(); j++) {
      const int ref_ind = (j / width()) * ref_stride + j % width();
      const int src_ind = (j / width()) * src_stride + j % width();
      if (!use_high_bit_depth()) {
        src_[src_ind] = rnd_.Rand8();
        ref_[ref_ind] = rnd_.Rand8();
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        CONVERT_TO_SHORTPTR(src_)[src_ind] = rnd_.Rand16() & mask();
        CONVERT_TO_SHORTPTR(ref_)[ref_ind] = rnd_.Rand16() & mask();
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
    }
    unsigned int sse1, sse2;
    unsigned int var1, var2;

    ASM_REGISTER_STATE_CHECK(
        var1 = params_.func(src_, src_stride, ref_, ref_stride, &sse1));
    var2 = variance_ref(src_, ref_, params_.log2width, params_.log2height,
                        src_stride, ref_stride, &sse2, use_high_bit_depth(),
                        params_.bit_depth);
    EXPECT_EQ(sse1, sse2) << "Error at test index: " << i;
    EXPECT_EQ(var1, var2) << "Error at test index: " << i;
  }
}

template <typename VarianceFunctionType>
void MainTestClass<VarianceFunctionType>::OneQuarterTest() {
  const int half = block_size() / 2;
  if (!use_high_bit_depth()) {
    memset(src_, 255, block_size());
    memset(ref_, 255, half);
    memset(ref_ + half, 0, half);
#if CONFIG_VP9_HIGHBITDEPTH
  } else {
    vpx_memset16(CONVERT_TO_SHORTPTR(src_), 255 << byte_shift(), block_size());
    vpx_memset16(CONVERT_TO_SHORTPTR(ref_), 255 << byte_shift(), half);
    vpx_memset16(CONVERT_TO_SHORTPTR(ref_) + half, 0, half);
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }
  unsigned int sse, var, expected;
  ASM_REGISTER_STATE_CHECK(
      var = params_.func(src_, width(), ref_, width(), &sse));
  expected = block_size() * 255 * 255 / 4;
  EXPECT_EQ(expected, var);
}

template <typename VarianceFunctionType>
void MainTestClass<VarianceFunctionType>::SpeedTest() {
  const int half = block_size() / 2;
  if (!use_high_bit_depth()) {
    memset(src_, 255, block_size());
    memset(ref_, 255, half);
    memset(ref_ + half, 0, half);
#if CONFIG_VP9_HIGHBITDEPTH
  } else {
    vpx_memset16(CONVERT_TO_SHORTPTR(src_), 255 << byte_shift(), block_size());
    vpx_memset16(CONVERT_TO_SHORTPTR(ref_), 255 << byte_shift(), half);
    vpx_memset16(CONVERT_TO_SHORTPTR(ref_) + half, 0, half);
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }
  unsigned int sse;

  vpx_usec_timer timer;
  vpx_usec_timer_start(&timer);
  for (int i = 0; i < (1 << 30) / block_size(); ++i) {
    const uint32_t variance = params_.func(src_, width(), ref_, width(), &sse);
    // Ignore return value.
    (void)variance;
  }
  vpx_usec_timer_mark(&timer);
  const int elapsed_time = static_cast<int>(vpx_usec_timer_elapsed(&timer));
  printf("Variance %dx%d %dbpp time: %5d ms\n", width(), height(),
         params_.bit_depth, elapsed_time / 1000);
}

////////////////////////////////////////////////////////////////////////////////
// Tests related to GetVariance.
template <typename GetVarianceFunctionType>
void MainTestClass<GetVarianceFunctionType>::RefTestGetVar() {
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < block_size(); j++) {
      if (!use_high_bit_depth()) {
        src_[j] = rnd_.Rand8();
        ref_[j] = rnd_.Rand8();
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        CONVERT_TO_SHORTPTR(src_)[j] = rnd_.Rand16() & mask();
        CONVERT_TO_SHORTPTR(ref_)[j] = rnd_.Rand16() & mask();
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
    }
    unsigned int sse1, sse2;
    int sum1, sum2;
    const int stride = width();
    ASM_REGISTER_STATE_CHECK(
        params_.func(src_, stride, ref_, stride, &sse1, &sum1));
    get_variance_ref(src_, stride, ref_, stride, params_.log2width,
                     params_.log2height, use_high_bit_depth(), &sse2, &sum2,
                     params_.bit_depth);
    EXPECT_EQ(sse1, sse2) << "Error at test index: " << i;
    EXPECT_EQ(sum1, sum2) << "Error at test index: " << i;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Tests related to MSE / SSE.

template <typename FunctionType>
void MainTestClass<FunctionType>::RefTestMse() {
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < block_size(); ++j) {
      if (!use_high_bit_depth()) {
        src_[j] = rnd_.Rand8();
        ref_[j] = rnd_.Rand8();
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        CONVERT_TO_SHORTPTR(src_)[j] = rnd_.Rand16() & mask();
        CONVERT_TO_SHORTPTR(ref_)[j] = rnd_.Rand16() & mask();
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
    }
    unsigned int sse1, sse2;
    const int stride = width();
    ASM_REGISTER_STATE_CHECK(params_.func(src_, stride, ref_, stride, &sse1));
    variance_ref(src_, ref_, params_.log2width, params_.log2height, stride,
                 stride, &sse2, use_high_bit_depth(), params_.bit_depth);
    EXPECT_EQ(sse1, sse2);
  }
}

template <typename FunctionType>
void MainTestClass<FunctionType>::RefTestSse() {
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < block_size(); ++j) {
      src_[j] = rnd_.Rand8();
      ref_[j] = rnd_.Rand8();
    }
    unsigned int sse2;
    unsigned int var1;
    const int stride = width();
    ASM_REGISTER_STATE_CHECK(var1 = params_.func(src_, stride, ref_, stride));
    variance_ref(src_, ref_, params_.log2width, params_.log2height, stride,
                 stride, &sse2, false, VPX_BITS_8);
    EXPECT_EQ(var1, sse2);
  }
}

template <typename FunctionType>
void MainTestClass<FunctionType>::MaxTestMse() {
  if (!use_high_bit_depth()) {
    memset(src_, 255, block_size());
    memset(ref_, 0, block_size());
#if CONFIG_VP9_HIGHBITDEPTH
  } else {
    vpx_memset16(CONVERT_TO_SHORTPTR(src_), 255 << byte_shift(), block_size());
    vpx_memset16(CONVERT_TO_SHORTPTR(ref_), 0, block_size());
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }
  unsigned int sse;
  ASM_REGISTER_STATE_CHECK(params_.func(src_, width(), ref_, width(), &sse));
  const unsigned int expected = block_size() * 255 * 255;
  EXPECT_EQ(expected, sse);
}

template <typename FunctionType>
void MainTestClass<FunctionType>::MaxTestSse() {
  memset(src_, 255, block_size());
  memset(ref_, 0, block_size());
  unsigned int var;
  ASM_REGISTER_STATE_CHECK(var = params_.func(src_, width(), ref_, width()));
  const unsigned int expected = block_size() * 255 * 255;
  EXPECT_EQ(expected, var);
}

////////////////////////////////////////////////////////////////////////////////

template <typename FunctionType>
class SubpelVarianceTest
    : public ::testing::TestWithParam<TestParams<FunctionType> > {
 public:
  void SetUp() override {
    params_ = this->GetParam();

    rnd_.Reset(ACMRandom::DeterministicSeed());
    if (!use_high_bit_depth()) {
      src_ = reinterpret_cast<uint8_t *>(vpx_memalign(16, block_size()));
      sec_ = reinterpret_cast<uint8_t *>(vpx_memalign(16, block_size()));
      ref_ = reinterpret_cast<uint8_t *>(
          vpx_malloc(block_size() + width() + height() + 1));
#if CONFIG_VP9_HIGHBITDEPTH
    } else {
      src_ = CONVERT_TO_BYTEPTR(reinterpret_cast<uint16_t *>(
          vpx_memalign(16, block_size() * sizeof(uint16_t))));
      sec_ = CONVERT_TO_BYTEPTR(reinterpret_cast<uint16_t *>(
          vpx_memalign(16, block_size() * sizeof(uint16_t))));
      ref_ = CONVERT_TO_BYTEPTR(reinterpret_cast<uint16_t *>(vpx_malloc(
          (block_size() + width() + height() + 1) * sizeof(uint16_t))));
#endif  // CONFIG_VP9_HIGHBITDEPTH
    }
    ASSERT_NE(src_, nullptr);
    ASSERT_NE(sec_, nullptr);
    ASSERT_NE(ref_, nullptr);
  }

  void TearDown() override {
    if (!use_high_bit_depth()) {
      vpx_free(src_);
      vpx_free(sec_);
      vpx_free(ref_);
#if CONFIG_VP9_HIGHBITDEPTH
    } else {
      vpx_free(CONVERT_TO_SHORTPTR(src_));
      vpx_free(CONVERT_TO_SHORTPTR(ref_));
      vpx_free(CONVERT_TO_SHORTPTR(sec_));
#endif  // CONFIG_VP9_HIGHBITDEPTH
    }
    libvpx_test::ClearSystemState();
  }

 protected:
  void RefTest();
  void ExtremeRefTest();
  void SpeedTest();

  ACMRandom rnd_;
  uint8_t *src_;
  uint8_t *ref_;
  uint8_t *sec_;
  TestParams<FunctionType> params_;

  // some relay helpers
  bool use_high_bit_depth() const { return params_.use_high_bit_depth; }
  int byte_shift() const { return params_.bit_depth - 8; }
  int block_size() const { return params_.block_size; }
  int width() const { return params_.width; }
  int height() const { return params_.height; }
  uint32_t mask() const { return params_.mask; }
};

template <typename SubpelVarianceFunctionType>
void SubpelVarianceTest<SubpelVarianceFunctionType>::RefTest() {
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      if (!use_high_bit_depth()) {
        for (int j = 0; j < block_size(); j++) {
          src_[j] = rnd_.Rand8();
        }
        for (int j = 0; j < block_size() + width() + height() + 1; j++) {
          ref_[j] = rnd_.Rand8();
        }
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        for (int j = 0; j < block_size(); j++) {
          CONVERT_TO_SHORTPTR(src_)[j] = rnd_.Rand16() & mask();
        }
        for (int j = 0; j < block_size() + width() + height() + 1; j++) {
          CONVERT_TO_SHORTPTR(ref_)[j] = rnd_.Rand16() & mask();
        }
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
      unsigned int sse1, sse2;
      unsigned int var1;
      ASM_REGISTER_STATE_CHECK(
          var1 = params_.func(ref_, width() + 1, x, y, src_, width(), &sse1));
      const unsigned int var2 = subpel_variance_ref(
          ref_, src_, params_.log2width, params_.log2height, x, y, &sse2,
          use_high_bit_depth(), params_.bit_depth);
      EXPECT_EQ(sse1, sse2) << "at position " << x << ", " << y;
      EXPECT_EQ(var1, var2) << "at position " << x << ", " << y;
    }
  }
}

template <typename SubpelVarianceFunctionType>
void SubpelVarianceTest<SubpelVarianceFunctionType>::ExtremeRefTest() {
  // Compare against reference.
  // Src: Set the first half of values to 0, the second half to the maximum.
  // Ref: Set the first half of values to the maximum, the second half to 0.
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      const int half = block_size() / 2;
      if (!use_high_bit_depth()) {
        memset(src_, 0, half);
        memset(src_ + half, 255, half);
        memset(ref_, 255, half);
        memset(ref_ + half, 0, half + width() + height() + 1);
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        vpx_memset16(CONVERT_TO_SHORTPTR(src_), mask(), half);
        vpx_memset16(CONVERT_TO_SHORTPTR(src_) + half, 0, half);
        vpx_memset16(CONVERT_TO_SHORTPTR(ref_), 0, half);
        vpx_memset16(CONVERT_TO_SHORTPTR(ref_) + half, mask(),
                     half + width() + height() + 1);
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
      unsigned int sse1, sse2;
      unsigned int var1;
      ASM_REGISTER_STATE_CHECK(
          var1 = params_.func(ref_, width() + 1, x, y, src_, width(), &sse1));
      const unsigned int var2 = subpel_variance_ref(
          ref_, src_, params_.log2width, params_.log2height, x, y, &sse2,
          use_high_bit_depth(), params_.bit_depth);
      EXPECT_EQ(sse1, sse2) << "for xoffset " << x << " and yoffset " << y;
      EXPECT_EQ(var1, var2) << "for xoffset " << x << " and yoffset " << y;
    }
  }
}

template <typename SubpelVarianceFunctionType>
void SubpelVarianceTest<SubpelVarianceFunctionType>::SpeedTest() {
  // The only interesting points are 0, 4, and anything else. To make the loops
  // simple we will use 0, 2 and 4.
  for (int x = 0; x <= 4; x += 2) {
    for (int y = 0; y <= 4; y += 2) {
      if (!use_high_bit_depth()) {
        memset(src_, 25, block_size());
        memset(ref_, 50, block_size());
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        vpx_memset16(CONVERT_TO_SHORTPTR(src_), 25, block_size());
        vpx_memset16(CONVERT_TO_SHORTPTR(ref_), 50, block_size());
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
      unsigned int sse;
      vpx_usec_timer timer;
      vpx_usec_timer_start(&timer);
      for (int i = 0; i < 1000000000 / block_size(); ++i) {
        const uint32_t variance =
            params_.func(ref_, width() + 1, x, y, src_, width(), &sse);
        (void)variance;
      }
      vpx_usec_timer_mark(&timer);
      const int elapsed_time = static_cast<int>(vpx_usec_timer_elapsed(&timer));
      printf("SubpelVariance %dx%d xoffset: %d yoffset: %d time: %5d ms\n",
             width(), height(), x, y, elapsed_time / 1000);
    }
  }
}

template <>
void SubpelVarianceTest<vpx_subp_avg_variance_fn_t>::RefTest() {
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      if (!use_high_bit_depth()) {
        for (int j = 0; j < block_size(); j++) {
          src_[j] = rnd_.Rand8();
          sec_[j] = rnd_.Rand8();
        }
        for (int j = 0; j < block_size() + width() + height() + 1; j++) {
          ref_[j] = rnd_.Rand8();
        }
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        for (int j = 0; j < block_size(); j++) {
          CONVERT_TO_SHORTPTR(src_)[j] = rnd_.Rand16() & mask();
          CONVERT_TO_SHORTPTR(sec_)[j] = rnd_.Rand16() & mask();
        }
        for (int j = 0; j < block_size() + width() + height() + 1; j++) {
          CONVERT_TO_SHORTPTR(ref_)[j] = rnd_.Rand16() & mask();
        }
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
      uint32_t sse1, sse2;
      uint32_t var1, var2;
      ASM_REGISTER_STATE_CHECK(var1 = params_.func(ref_, width() + 1, x, y,
                                                   src_, width(), &sse1, sec_));
      var2 = subpel_avg_variance_ref(ref_, src_, sec_, params_.log2width,
                                     params_.log2height, x, y, &sse2,
                                     use_high_bit_depth(), params_.bit_depth);
      EXPECT_EQ(sse1, sse2) << "at position " << x << ", " << y;
      EXPECT_EQ(var1, var2) << "at position " << x << ", " << y;
    }
  }
}

using VpxSseTest = MainTestClass<Get4x4SseFunc>;
using VpxMseTest = MainTestClass<vpx_variance_fn_t>;
using VpxVarianceTest = MainTestClass<vpx_variance_fn_t>;
using VpxGetVarianceTest = MainTestClass<GetVarianceFunc>;
using VpxSubpelVarianceTest = SubpelVarianceTest<vpx_subpixvariance_fn_t>;
using VpxSubpelAvgVarianceTest = SubpelVarianceTest<vpx_subp_avg_variance_fn_t>;

TEST_P(VpxSseTest, RefSse) { RefTestSse(); }
TEST_P(VpxSseTest, MaxSse) { MaxTestSse(); }
TEST_P(VpxMseTest, RefMse) { RefTestMse(); }
TEST_P(VpxMseTest, MaxMse) { MaxTestMse(); }
TEST_P(VpxMseTest, DISABLED_Speed) { SpeedTest(); }
TEST_P(VpxVarianceTest, Zero) { ZeroTest(); }
TEST_P(VpxVarianceTest, Ref) { RefTest(); }
TEST_P(VpxVarianceTest, RefStride) { RefStrideTest(); }
TEST_P(VpxVarianceTest, OneQuarter) { OneQuarterTest(); }
TEST_P(VpxVarianceTest, DISABLED_Speed) { SpeedTest(); }
TEST_P(VpxGetVarianceTest, RefGetVar) { RefTestGetVar(); }
TEST_P(SumOfSquaresTest, Const) { ConstTest(); }
TEST_P(SumOfSquaresTest, Ref) { RefTest(); }
TEST_P(VpxSubpelVarianceTest, Ref) { RefTest(); }
TEST_P(VpxSubpelVarianceTest, ExtremeRef) { ExtremeRefTest(); }
TEST_P(VpxSubpelVarianceTest, DISABLED_Speed) { SpeedTest(); }
TEST_P(VpxSubpelAvgVarianceTest, Ref) { RefTest(); }

INSTANTIATE_TEST_SUITE_P(C, SumOfSquaresTest,
                         ::testing::Values(vpx_get_mb_ss_c));

using SseParams = TestParams<Get4x4SseFunc>;
INSTANTIATE_TEST_SUITE_P(C, VpxSseTest,
                         ::testing::Values(SseParams(2, 2,
                                                     &vpx_get4x4sse_cs_c)));

using MseParams = TestParams<vpx_variance_fn_t>;
INSTANTIATE_TEST_SUITE_P(C, VpxMseTest,
                         ::testing::Values(MseParams(4, 4, &vpx_mse16x16_c),
                                           MseParams(4, 3, &vpx_mse16x8_c),
                                           MseParams(3, 4, &vpx_mse8x16_c),
                                           MseParams(3, 3, &vpx_mse8x8_c)));

using VarianceParams = TestParams<vpx_variance_fn_t>;
INSTANTIATE_TEST_SUITE_P(
    C, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_c),
                      VarianceParams(6, 5, &vpx_variance64x32_c),
                      VarianceParams(5, 6, &vpx_variance32x64_c),
                      VarianceParams(5, 5, &vpx_variance32x32_c),
                      VarianceParams(5, 4, &vpx_variance32x16_c),
                      VarianceParams(4, 5, &vpx_variance16x32_c),
                      VarianceParams(4, 4, &vpx_variance16x16_c),
                      VarianceParams(4, 3, &vpx_variance16x8_c),
                      VarianceParams(3, 4, &vpx_variance8x16_c),
                      VarianceParams(3, 3, &vpx_variance8x8_c),
                      VarianceParams(3, 2, &vpx_variance8x4_c),
                      VarianceParams(2, 3, &vpx_variance4x8_c),
                      VarianceParams(2, 2, &vpx_variance4x4_c)));

using GetVarianceParams = TestParams<GetVarianceFunc>;
INSTANTIATE_TEST_SUITE_P(
    C, VpxGetVarianceTest,
    ::testing::Values(GetVarianceParams(4, 4, &vpx_get16x16var_c),
                      GetVarianceParams(3, 3, &vpx_get8x8var_c),
                      GetVarianceParams(4, 4, &vpx_get16x16var_c),
                      GetVarianceParams(3, 3, &vpx_get8x8var_c),
                      GetVarianceParams(4, 4, &vpx_get16x16var_c),
                      GetVarianceParams(3, 3, &vpx_get8x8var_c)));

using SubpelVarianceParams = TestParams<vpx_subpixvariance_fn_t>;
INSTANTIATE_TEST_SUITE_P(
    C, VpxSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_sub_pixel_variance64x64_c, 0),
        SubpelVarianceParams(6, 5, &vpx_sub_pixel_variance64x32_c, 0),
        SubpelVarianceParams(5, 6, &vpx_sub_pixel_variance32x64_c, 0),
        SubpelVarianceParams(5, 5, &vpx_sub_pixel_variance32x32_c, 0),
        SubpelVarianceParams(5, 4, &vpx_sub_pixel_variance32x16_c, 0),
        SubpelVarianceParams(4, 5, &vpx_sub_pixel_variance16x32_c, 0),
        SubpelVarianceParams(4, 4, &vpx_sub_pixel_variance16x16_c, 0),
        SubpelVarianceParams(4, 3, &vpx_sub_pixel_variance16x8_c, 0),
        SubpelVarianceParams(3, 4, &vpx_sub_pixel_variance8x16_c, 0),
        SubpelVarianceParams(3, 3, &vpx_sub_pixel_variance8x8_c, 0),
        SubpelVarianceParams(3, 2, &vpx_sub_pixel_variance8x4_c, 0),
        SubpelVarianceParams(2, 3, &vpx_sub_pixel_variance4x8_c, 0),
        SubpelVarianceParams(2, 2, &vpx_sub_pixel_variance4x4_c, 0)));

using SubpelAvgVarianceParams = TestParams<vpx_subp_avg_variance_fn_t>;
INSTANTIATE_TEST_SUITE_P(
    C, VpxSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6, &vpx_sub_pixel_avg_variance64x64_c, 0),
        SubpelAvgVarianceParams(6, 5, &vpx_sub_pixel_avg_variance64x32_c, 0),
        SubpelAvgVarianceParams(5, 6, &vpx_sub_pixel_avg_variance32x64_c, 0),
        SubpelAvgVarianceParams(5, 5, &vpx_sub_pixel_avg_variance32x32_c, 0),
        SubpelAvgVarianceParams(5, 4, &vpx_sub_pixel_avg_variance32x16_c, 0),
        SubpelAvgVarianceParams(4, 5, &vpx_sub_pixel_avg_variance16x32_c, 0),
        SubpelAvgVarianceParams(4, 4, &vpx_sub_pixel_avg_variance16x16_c, 0),
        SubpelAvgVarianceParams(4, 3, &vpx_sub_pixel_avg_variance16x8_c, 0),
        SubpelAvgVarianceParams(3, 4, &vpx_sub_pixel_avg_variance8x16_c, 0),
        SubpelAvgVarianceParams(3, 3, &vpx_sub_pixel_avg_variance8x8_c, 0),
        SubpelAvgVarianceParams(3, 2, &vpx_sub_pixel_avg_variance8x4_c, 0),
        SubpelAvgVarianceParams(2, 3, &vpx_sub_pixel_avg_variance4x8_c, 0),
        SubpelAvgVarianceParams(2, 2, &vpx_sub_pixel_avg_variance4x4_c, 0)));

#if CONFIG_VP9_HIGHBITDEPTH
using VpxHBDVarianceTest = MainTestClass<vpx_variance_fn_t>;
using VpxHBDGetVarianceTest = MainTestClass<GetVarianceFunc>;
using VpxHBDSubpelVarianceTest = SubpelVarianceTest<vpx_subpixvariance_fn_t>;
using VpxHBDSubpelAvgVarianceTest =
    SubpelVarianceTest<vpx_subp_avg_variance_fn_t>;

TEST_P(VpxHBDVarianceTest, Zero) { ZeroTest(); }
TEST_P(VpxHBDVarianceTest, Ref) { RefTest(); }
TEST_P(VpxHBDVarianceTest, RefStride) { RefStrideTest(); }
TEST_P(VpxHBDVarianceTest, OneQuarter) { OneQuarterTest(); }
TEST_P(VpxHBDVarianceTest, DISABLED_Speed) { SpeedTest(); }
TEST_P(VpxHBDGetVarianceTest, RefGetVar) { RefTestGetVar(); }
TEST_P(VpxHBDSubpelVarianceTest, Ref) { RefTest(); }
TEST_P(VpxHBDSubpelVarianceTest, ExtremeRef) { ExtremeRefTest(); }
TEST_P(VpxHBDSubpelAvgVarianceTest, Ref) { RefTest(); }

using VpxHBDMseTest = MainTestClass<vpx_variance_fn_t>;
TEST_P(VpxHBDMseTest, RefMse) { RefTestMse(); }
TEST_P(VpxHBDMseTest, MaxMse) { MaxTestMse(); }
TEST_P(VpxHBDMseTest, DISABLED_Speed) { SpeedTest(); }
INSTANTIATE_TEST_SUITE_P(
    C, VpxHBDMseTest,
    ::testing::Values(MseParams(4, 4, &vpx_highbd_12_mse16x16_c, VPX_BITS_12),
                      MseParams(4, 3, &vpx_highbd_12_mse16x8_c, VPX_BITS_12),
                      MseParams(3, 4, &vpx_highbd_12_mse8x16_c, VPX_BITS_12),
                      MseParams(3, 3, &vpx_highbd_12_mse8x8_c, VPX_BITS_12),
                      MseParams(4, 4, &vpx_highbd_10_mse16x16_c, VPX_BITS_10),
                      MseParams(4, 3, &vpx_highbd_10_mse16x8_c, VPX_BITS_10),
                      MseParams(3, 4, &vpx_highbd_10_mse8x16_c, VPX_BITS_10),
                      MseParams(3, 3, &vpx_highbd_10_mse8x8_c, VPX_BITS_10),
                      MseParams(4, 4, &vpx_highbd_8_mse16x16_c, VPX_BITS_8),
                      MseParams(4, 3, &vpx_highbd_8_mse16x8_c, VPX_BITS_8),
                      MseParams(3, 4, &vpx_highbd_8_mse8x16_c, VPX_BITS_8),
                      MseParams(3, 3, &vpx_highbd_8_mse8x8_c, VPX_BITS_8)));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VpxHBDMseTest);

INSTANTIATE_TEST_SUITE_P(
    C, VpxHBDVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_highbd_12_variance64x64_c, 12),
                      VarianceParams(6, 5, &vpx_highbd_12_variance64x32_c, 12),
                      VarianceParams(5, 6, &vpx_highbd_12_variance32x64_c, 12),
                      VarianceParams(5, 5, &vpx_highbd_12_variance32x32_c, 12),
                      VarianceParams(5, 4, &vpx_highbd_12_variance32x16_c, 12),
                      VarianceParams(4, 5, &vpx_highbd_12_variance16x32_c, 12),
                      VarianceParams(4, 4, &vpx_highbd_12_variance16x16_c, 12),
                      VarianceParams(4, 3, &vpx_highbd_12_variance16x8_c, 12),
                      VarianceParams(3, 4, &vpx_highbd_12_variance8x16_c, 12),
                      VarianceParams(3, 3, &vpx_highbd_12_variance8x8_c, 12),
                      VarianceParams(3, 2, &vpx_highbd_12_variance8x4_c, 12),
                      VarianceParams(2, 3, &vpx_highbd_12_variance4x8_c, 12),
                      VarianceParams(2, 2, &vpx_highbd_12_variance4x4_c, 12),
                      VarianceParams(6, 6, &vpx_highbd_10_variance64x64_c, 10),
                      VarianceParams(6, 5, &vpx_highbd_10_variance64x32_c, 10),
                      VarianceParams(5, 6, &vpx_highbd_10_variance32x64_c, 10),
                      VarianceParams(5, 5, &vpx_highbd_10_variance32x32_c, 10),
                      VarianceParams(5, 4, &vpx_highbd_10_variance32x16_c, 10),
                      VarianceParams(4, 5, &vpx_highbd_10_variance16x32_c, 10),
                      VarianceParams(4, 4, &vpx_highbd_10_variance16x16_c, 10),
                      VarianceParams(4, 3, &vpx_highbd_10_variance16x8_c, 10),
                      VarianceParams(3, 4, &vpx_highbd_10_variance8x16_c, 10),
                      VarianceParams(3, 3, &vpx_highbd_10_variance8x8_c, 10),
                      VarianceParams(3, 2, &vpx_highbd_10_variance8x4_c, 10),
                      VarianceParams(2, 3, &vpx_highbd_10_variance4x8_c, 10),
                      VarianceParams(2, 2, &vpx_highbd_10_variance4x4_c, 10),
                      VarianceParams(6, 6, &vpx_highbd_8_variance64x64_c, 8),
                      VarianceParams(6, 5, &vpx_highbd_8_variance64x32_c, 8),
                      VarianceParams(5, 6, &vpx_highbd_8_variance32x64_c, 8),
                      VarianceParams(5, 5, &vpx_highbd_8_variance32x32_c, 8),
                      VarianceParams(5, 4, &vpx_highbd_8_variance32x16_c, 8),
                      VarianceParams(4, 5, &vpx_highbd_8_variance16x32_c, 8),
                      VarianceParams(4, 4, &vpx_highbd_8_variance16x16_c, 8),
                      VarianceParams(4, 3, &vpx_highbd_8_variance16x8_c, 8),
                      VarianceParams(3, 4, &vpx_highbd_8_variance8x16_c, 8),
                      VarianceParams(3, 3, &vpx_highbd_8_variance8x8_c, 8),
                      VarianceParams(3, 2, &vpx_highbd_8_variance8x4_c, 8),
                      VarianceParams(2, 3, &vpx_highbd_8_variance4x8_c, 8),
                      VarianceParams(2, 2, &vpx_highbd_8_variance4x4_c, 8)));

INSTANTIATE_TEST_SUITE_P(
    C, VpxHBDGetVarianceTest,
    ::testing::Values(GetVarianceParams(4, 4, &vpx_highbd_12_get16x16var_c, 12),
                      GetVarianceParams(3, 3, &vpx_highbd_12_get8x8var_c, 12),
                      GetVarianceParams(4, 4, &vpx_highbd_10_get16x16var_c, 10),
                      GetVarianceParams(3, 3, &vpx_highbd_10_get8x8var_c, 10),
                      GetVarianceParams(4, 4, &vpx_highbd_8_get16x16var_c, 8),
                      GetVarianceParams(3, 3, &vpx_highbd_8_get8x8var_c, 8)));

INSTANTIATE_TEST_SUITE_P(
    C, VpxHBDSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_highbd_8_sub_pixel_variance64x64_c, 8),
        SubpelVarianceParams(6, 5, &vpx_highbd_8_sub_pixel_variance64x32_c, 8),
        SubpelVarianceParams(5, 6, &vpx_highbd_8_sub_pixel_variance32x64_c, 8),
        SubpelVarianceParams(5, 5, &vpx_highbd_8_sub_pixel_variance32x32_c, 8),
        SubpelVarianceParams(5, 4, &vpx_highbd_8_sub_pixel_variance32x16_c, 8),
        SubpelVarianceParams(4, 5, &vpx_highbd_8_sub_pixel_variance16x32_c, 8),
        SubpelVarianceParams(4, 4, &vpx_highbd_8_sub_pixel_variance16x16_c, 8),
        SubpelVarianceParams(4, 3, &vpx_highbd_8_sub_pixel_variance16x8_c, 8),
        SubpelVarianceParams(3, 4, &vpx_highbd_8_sub_pixel_variance8x16_c, 8),
        SubpelVarianceParams(3, 3, &vpx_highbd_8_sub_pixel_variance8x8_c, 8),
        SubpelVarianceParams(3, 2, &vpx_highbd_8_sub_pixel_variance8x4_c, 8),
        SubpelVarianceParams(2, 3, &vpx_highbd_8_sub_pixel_variance4x8_c, 8),
        SubpelVarianceParams(2, 2, &vpx_highbd_8_sub_pixel_variance4x4_c, 8),
        SubpelVarianceParams(6, 6, &vpx_highbd_10_sub_pixel_variance64x64_c,
                             10),
        SubpelVarianceParams(6, 5, &vpx_highbd_10_sub_pixel_variance64x32_c,
                             10),
        SubpelVarianceParams(5, 6, &vpx_highbd_10_sub_pixel_variance32x64_c,
                             10),
        SubpelVarianceParams(5, 5, &vpx_highbd_10_sub_pixel_variance32x32_c,
                             10),
        SubpelVarianceParams(5, 4, &vpx_highbd_10_sub_pixel_variance32x16_c,
                             10),
        SubpelVarianceParams(4, 5, &vpx_highbd_10_sub_pixel_variance16x32_c,
                             10),
        SubpelVarianceParams(4, 4, &vpx_highbd_10_sub_pixel_variance16x16_c,
                             10),
        SubpelVarianceParams(4, 3, &vpx_highbd_10_sub_pixel_variance16x8_c, 10),
        SubpelVarianceParams(3, 4, &vpx_highbd_10_sub_pixel_variance8x16_c, 10),
        SubpelVarianceParams(3, 3, &vpx_highbd_10_sub_pixel_variance8x8_c, 10),
        SubpelVarianceParams(3, 2, &vpx_highbd_10_sub_pixel_variance8x4_c, 10),
        SubpelVarianceParams(2, 3, &vpx_highbd_10_sub_pixel_variance4x8_c, 10),
        SubpelVarianceParams(2, 2, &vpx_highbd_10_sub_pixel_variance4x4_c, 10),
        SubpelVarianceParams(6, 6, &vpx_highbd_12_sub_pixel_variance64x64_c,
                             12),
        SubpelVarianceParams(6, 5, &vpx_highbd_12_sub_pixel_variance64x32_c,
                             12),
        SubpelVarianceParams(5, 6, &vpx_highbd_12_sub_pixel_variance32x64_c,
                             12),
        SubpelVarianceParams(5, 5, &vpx_highbd_12_sub_pixel_variance32x32_c,
                             12),
        SubpelVarianceParams(5, 4, &vpx_highbd_12_sub_pixel_variance32x16_c,
                             12),
        SubpelVarianceParams(4, 5, &vpx_highbd_12_sub_pixel_variance16x32_c,
                             12),
        SubpelVarianceParams(4, 4, &vpx_highbd_12_sub_pixel_variance16x16_c,
                             12),
        SubpelVarianceParams(4, 3, &vpx_highbd_12_sub_pixel_variance16x8_c, 12),
        SubpelVarianceParams(3, 4, &vpx_highbd_12_sub_pixel_variance8x16_c, 12),
        SubpelVarianceParams(3, 3, &vpx_highbd_12_sub_pixel_variance8x8_c, 12),
        SubpelVarianceParams(3, 2, &vpx_highbd_12_sub_pixel_variance8x4_c, 12),
        SubpelVarianceParams(2, 3, &vpx_highbd_12_sub_pixel_variance4x8_c, 12),
        SubpelVarianceParams(2, 2, &vpx_highbd_12_sub_pixel_variance4x4_c,
                             12)));

INSTANTIATE_TEST_SUITE_P(
    C, VpxHBDSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_8_sub_pixel_avg_variance64x64_c, 8),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance64x32_c, 8),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_8_sub_pixel_avg_variance32x64_c, 8),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance32x32_c, 8),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance32x16_c, 8),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance16x32_c, 8),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance16x16_c, 8),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_8_sub_pixel_avg_variance16x8_c, 8),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance8x16_c, 8),
        SubpelAvgVarianceParams(3, 3, &vpx_highbd_8_sub_pixel_avg_variance8x8_c,
                                8),
        SubpelAvgVarianceParams(3, 2, &vpx_highbd_8_sub_pixel_avg_variance8x4_c,
                                8),
        SubpelAvgVarianceParams(2, 3, &vpx_highbd_8_sub_pixel_avg_variance4x8_c,
                                8),
        SubpelAvgVarianceParams(2, 2, &vpx_highbd_8_sub_pixel_avg_variance4x4_c,
                                8),
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_10_sub_pixel_avg_variance64x64_c,
                                10),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance64x32_c,
                                10),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_10_sub_pixel_avg_variance32x64_c,
                                10),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance32x32_c,
                                10),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance32x16_c,
                                10),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance16x32_c,
                                10),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance16x16_c,
                                10),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_10_sub_pixel_avg_variance16x8_c,
                                10),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance8x16_c,
                                10),
        SubpelAvgVarianceParams(3, 3,
                                &vpx_highbd_10_sub_pixel_avg_variance8x8_c, 10),
        SubpelAvgVarianceParams(3, 2,
                                &vpx_highbd_10_sub_pixel_avg_variance8x4_c, 10),
        SubpelAvgVarianceParams(2, 3,
                                &vpx_highbd_10_sub_pixel_avg_variance4x8_c, 10),
        SubpelAvgVarianceParams(2, 2,
                                &vpx_highbd_10_sub_pixel_avg_variance4x4_c, 10),
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_12_sub_pixel_avg_variance64x64_c,
                                12),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance64x32_c,
                                12),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_12_sub_pixel_avg_variance32x64_c,
                                12),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance32x32_c,
                                12),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance32x16_c,
                                12),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance16x32_c,
                                12),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance16x16_c,
                                12),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_12_sub_pixel_avg_variance16x8_c,
                                12),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance8x16_c,
                                12),
        SubpelAvgVarianceParams(3, 3,
                                &vpx_highbd_12_sub_pixel_avg_variance8x8_c, 12),
        SubpelAvgVarianceParams(3, 2,
                                &vpx_highbd_12_sub_pixel_avg_variance8x4_c, 12),
        SubpelAvgVarianceParams(2, 3,
                                &vpx_highbd_12_sub_pixel_avg_variance4x8_c, 12),
        SubpelAvgVarianceParams(2, 2,
                                &vpx_highbd_12_sub_pixel_avg_variance4x4_c,
                                12)));
#endif  // CONFIG_VP9_HIGHBITDEPTH

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, SumOfSquaresTest,
                         ::testing::Values(vpx_get_mb_ss_sse2));

INSTANTIATE_TEST_SUITE_P(SSE2, VpxMseTest,
                         ::testing::Values(MseParams(4, 4, &vpx_mse16x16_sse2),
                                           MseParams(4, 3, &vpx_mse16x8_sse2),
                                           MseParams(3, 4, &vpx_mse8x16_sse2),
                                           MseParams(3, 3, &vpx_mse8x8_sse2)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_sse2),
                      VarianceParams(6, 5, &vpx_variance64x32_sse2),
                      VarianceParams(5, 6, &vpx_variance32x64_sse2),
                      VarianceParams(5, 5, &vpx_variance32x32_sse2),
                      VarianceParams(5, 4, &vpx_variance32x16_sse2),
                      VarianceParams(4, 5, &vpx_variance16x32_sse2),
                      VarianceParams(4, 4, &vpx_variance16x16_sse2),
                      VarianceParams(4, 3, &vpx_variance16x8_sse2),
                      VarianceParams(3, 4, &vpx_variance8x16_sse2),
                      VarianceParams(3, 3, &vpx_variance8x8_sse2),
                      VarianceParams(3, 2, &vpx_variance8x4_sse2),
                      VarianceParams(2, 3, &vpx_variance4x8_sse2),
                      VarianceParams(2, 2, &vpx_variance4x4_sse2)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxGetVarianceTest,
    ::testing::Values(GetVarianceParams(4, 4, &vpx_get16x16var_sse2),
                      GetVarianceParams(3, 3, &vpx_get8x8var_sse2),
                      GetVarianceParams(4, 4, &vpx_get16x16var_sse2),
                      GetVarianceParams(3, 3, &vpx_get8x8var_sse2),
                      GetVarianceParams(4, 4, &vpx_get16x16var_sse2),
                      GetVarianceParams(3, 3, &vpx_get8x8var_sse2)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_sub_pixel_variance64x64_sse2, 0),
        SubpelVarianceParams(6, 5, &vpx_sub_pixel_variance64x32_sse2, 0),
        SubpelVarianceParams(5, 6, &vpx_sub_pixel_variance32x64_sse2, 0),
        SubpelVarianceParams(5, 5, &vpx_sub_pixel_variance32x32_sse2, 0),
        SubpelVarianceParams(5, 4, &vpx_sub_pixel_variance32x16_sse2, 0),
        SubpelVarianceParams(4, 5, &vpx_sub_pixel_variance16x32_sse2, 0),
        SubpelVarianceParams(4, 4, &vpx_sub_pixel_variance16x16_sse2, 0),
        SubpelVarianceParams(4, 3, &vpx_sub_pixel_variance16x8_sse2, 0),
        SubpelVarianceParams(3, 4, &vpx_sub_pixel_variance8x16_sse2, 0),
        SubpelVarianceParams(3, 3, &vpx_sub_pixel_variance8x8_sse2, 0),
        SubpelVarianceParams(3, 2, &vpx_sub_pixel_variance8x4_sse2, 0),
        SubpelVarianceParams(2, 3, &vpx_sub_pixel_variance4x8_sse2, 0),
        SubpelVarianceParams(2, 2, &vpx_sub_pixel_variance4x4_sse2, 0)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6, &vpx_sub_pixel_avg_variance64x64_sse2, 0),
        SubpelAvgVarianceParams(6, 5, &vpx_sub_pixel_avg_variance64x32_sse2, 0),
        SubpelAvgVarianceParams(5, 6, &vpx_sub_pixel_avg_variance32x64_sse2, 0),
        SubpelAvgVarianceParams(5, 5, &vpx_sub_pixel_avg_variance32x32_sse2, 0),
        SubpelAvgVarianceParams(5, 4, &vpx_sub_pixel_avg_variance32x16_sse2, 0),
        SubpelAvgVarianceParams(4, 5, &vpx_sub_pixel_avg_variance16x32_sse2, 0),
        SubpelAvgVarianceParams(4, 4, &vpx_sub_pixel_avg_variance16x16_sse2, 0),
        SubpelAvgVarianceParams(4, 3, &vpx_sub_pixel_avg_variance16x8_sse2, 0),
        SubpelAvgVarianceParams(3, 4, &vpx_sub_pixel_avg_variance8x16_sse2, 0),
        SubpelAvgVarianceParams(3, 3, &vpx_sub_pixel_avg_variance8x8_sse2, 0),
        SubpelAvgVarianceParams(3, 2, &vpx_sub_pixel_avg_variance8x4_sse2, 0),
        SubpelAvgVarianceParams(2, 3, &vpx_sub_pixel_avg_variance4x8_sse2, 0),
        SubpelAvgVarianceParams(2, 2, &vpx_sub_pixel_avg_variance4x4_sse2, 0)));

#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxHBDMseTest,
    ::testing::Values(
        MseParams(4, 4, &vpx_highbd_12_mse16x16_sse2, VPX_BITS_12),
        MseParams(3, 3, &vpx_highbd_12_mse8x8_sse2, VPX_BITS_12),
        MseParams(4, 4, &vpx_highbd_10_mse16x16_sse2, VPX_BITS_10),
        MseParams(3, 3, &vpx_highbd_10_mse8x8_sse2, VPX_BITS_10),
        MseParams(4, 4, &vpx_highbd_8_mse16x16_sse2, VPX_BITS_8),
        MseParams(3, 3, &vpx_highbd_8_mse8x8_sse2, VPX_BITS_8)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxHBDVarianceTest,
    ::testing::Values(
        VarianceParams(6, 6, &vpx_highbd_12_variance64x64_sse2, 12),
        VarianceParams(6, 5, &vpx_highbd_12_variance64x32_sse2, 12),
        VarianceParams(5, 6, &vpx_highbd_12_variance32x64_sse2, 12),
        VarianceParams(5, 5, &vpx_highbd_12_variance32x32_sse2, 12),
        VarianceParams(5, 4, &vpx_highbd_12_variance32x16_sse2, 12),
        VarianceParams(4, 5, &vpx_highbd_12_variance16x32_sse2, 12),
        VarianceParams(4, 4, &vpx_highbd_12_variance16x16_sse2, 12),
        VarianceParams(4, 3, &vpx_highbd_12_variance16x8_sse2, 12),
        VarianceParams(3, 4, &vpx_highbd_12_variance8x16_sse2, 12),
        VarianceParams(3, 3, &vpx_highbd_12_variance8x8_sse2, 12),
        VarianceParams(6, 6, &vpx_highbd_10_variance64x64_sse2, 10),
        VarianceParams(6, 5, &vpx_highbd_10_variance64x32_sse2, 10),
        VarianceParams(5, 6, &vpx_highbd_10_variance32x64_sse2, 10),
        VarianceParams(5, 5, &vpx_highbd_10_variance32x32_sse2, 10),
        VarianceParams(5, 4, &vpx_highbd_10_variance32x16_sse2, 10),
        VarianceParams(4, 5, &vpx_highbd_10_variance16x32_sse2, 10),
        VarianceParams(4, 4, &vpx_highbd_10_variance16x16_sse2, 10),
        VarianceParams(4, 3, &vpx_highbd_10_variance16x8_sse2, 10),
        VarianceParams(3, 4, &vpx_highbd_10_variance8x16_sse2, 10),
        VarianceParams(3, 3, &vpx_highbd_10_variance8x8_sse2, 10),
        VarianceParams(6, 6, &vpx_highbd_8_variance64x64_sse2, 8),
        VarianceParams(6, 5, &vpx_highbd_8_variance64x32_sse2, 8),
        VarianceParams(5, 6, &vpx_highbd_8_variance32x64_sse2, 8),
        VarianceParams(5, 5, &vpx_highbd_8_variance32x32_sse2, 8),
        VarianceParams(5, 4, &vpx_highbd_8_variance32x16_sse2, 8),
        VarianceParams(4, 5, &vpx_highbd_8_variance16x32_sse2, 8),
        VarianceParams(4, 4, &vpx_highbd_8_variance16x16_sse2, 8),
        VarianceParams(4, 3, &vpx_highbd_8_variance16x8_sse2, 8),
        VarianceParams(3, 4, &vpx_highbd_8_variance8x16_sse2, 8),
        VarianceParams(3, 3, &vpx_highbd_8_variance8x8_sse2, 8)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxHBDGetVarianceTest,
    ::testing::Values(
        GetVarianceParams(4, 4, &vpx_highbd_12_get16x16var_sse2, 12),
        GetVarianceParams(3, 3, &vpx_highbd_12_get8x8var_sse2, 12),
        GetVarianceParams(4, 4, &vpx_highbd_10_get16x16var_sse2, 10),
        GetVarianceParams(3, 3, &vpx_highbd_10_get8x8var_sse2, 10),
        GetVarianceParams(4, 4, &vpx_highbd_8_get16x16var_sse2, 8),
        GetVarianceParams(3, 3, &vpx_highbd_8_get8x8var_sse2, 8)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxHBDSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_highbd_12_sub_pixel_variance64x64_sse2,
                             12),
        SubpelVarianceParams(6, 5, &vpx_highbd_12_sub_pixel_variance64x32_sse2,
                             12),
        SubpelVarianceParams(5, 6, &vpx_highbd_12_sub_pixel_variance32x64_sse2,
                             12),
        SubpelVarianceParams(5, 5, &vpx_highbd_12_sub_pixel_variance32x32_sse2,
                             12),
        SubpelVarianceParams(5, 4, &vpx_highbd_12_sub_pixel_variance32x16_sse2,
                             12),
        SubpelVarianceParams(4, 5, &vpx_highbd_12_sub_pixel_variance16x32_sse2,
                             12),
        SubpelVarianceParams(4, 4, &vpx_highbd_12_sub_pixel_variance16x16_sse2,
                             12),
        SubpelVarianceParams(4, 3, &vpx_highbd_12_sub_pixel_variance16x8_sse2,
                             12),
        SubpelVarianceParams(3, 4, &vpx_highbd_12_sub_pixel_variance8x16_sse2,
                             12),
        SubpelVarianceParams(3, 3, &vpx_highbd_12_sub_pixel_variance8x8_sse2,
                             12),
        SubpelVarianceParams(3, 2, &vpx_highbd_12_sub_pixel_variance8x4_sse2,
                             12),
        SubpelVarianceParams(6, 6, &vpx_highbd_10_sub_pixel_variance64x64_sse2,
                             10),
        SubpelVarianceParams(6, 5, &vpx_highbd_10_sub_pixel_variance64x32_sse2,
                             10),
        SubpelVarianceParams(5, 6, &vpx_highbd_10_sub_pixel_variance32x64_sse2,
                             10),
        SubpelVarianceParams(5, 5, &vpx_highbd_10_sub_pixel_variance32x32_sse2,
                             10),
        SubpelVarianceParams(5, 4, &vpx_highbd_10_sub_pixel_variance32x16_sse2,
                             10),
        SubpelVarianceParams(4, 5, &vpx_highbd_10_sub_pixel_variance16x32_sse2,
                             10),
        SubpelVarianceParams(4, 4, &vpx_highbd_10_sub_pixel_variance16x16_sse2,
                             10),
        SubpelVarianceParams(4, 3, &vpx_highbd_10_sub_pixel_variance16x8_sse2,
                             10),
        SubpelVarianceParams(3, 4, &vpx_highbd_10_sub_pixel_variance8x16_sse2,
                             10),
        SubpelVarianceParams(3, 3, &vpx_highbd_10_sub_pixel_variance8x8_sse2,
                             10),
        SubpelVarianceParams(3, 2, &vpx_highbd_10_sub_pixel_variance8x4_sse2,
                             10),
        SubpelVarianceParams(6, 6, &vpx_highbd_8_sub_pixel_variance64x64_sse2,
                             8),
        SubpelVarianceParams(6, 5, &vpx_highbd_8_sub_pixel_variance64x32_sse2,
                             8),
        SubpelVarianceParams(5, 6, &vpx_highbd_8_sub_pixel_variance32x64_sse2,
                             8),
        SubpelVarianceParams(5, 5, &vpx_highbd_8_sub_pixel_variance32x32_sse2,
                             8),
        SubpelVarianceParams(5, 4, &vpx_highbd_8_sub_pixel_variance32x16_sse2,
                             8),
        SubpelVarianceParams(4, 5, &vpx_highbd_8_sub_pixel_variance16x32_sse2,
                             8),
        SubpelVarianceParams(4, 4, &vpx_highbd_8_sub_pixel_variance16x16_sse2,
                             8),
        SubpelVarianceParams(4, 3, &vpx_highbd_8_sub_pixel_variance16x8_sse2,
                             8),
        SubpelVarianceParams(3, 4, &vpx_highbd_8_sub_pixel_variance8x16_sse2,
                             8),
        SubpelVarianceParams(3, 3, &vpx_highbd_8_sub_pixel_variance8x8_sse2, 8),
        SubpelVarianceParams(3, 2, &vpx_highbd_8_sub_pixel_variance8x4_sse2,
                             8)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, VpxHBDSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_12_sub_pixel_avg_variance64x64_sse2,
                                12),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance64x32_sse2,
                                12),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_12_sub_pixel_avg_variance32x64_sse2,
                                12),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance32x32_sse2,
                                12),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance32x16_sse2,
                                12),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance16x32_sse2,
                                12),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance16x16_sse2,
                                12),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_12_sub_pixel_avg_variance16x8_sse2,
                                12),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance8x16_sse2,
                                12),
        SubpelAvgVarianceParams(3, 3,
                                &vpx_highbd_12_sub_pixel_avg_variance8x8_sse2,
                                12),
        SubpelAvgVarianceParams(3, 2,
                                &vpx_highbd_12_sub_pixel_avg_variance8x4_sse2,
                                12),
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_10_sub_pixel_avg_variance64x64_sse2,
                                10),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance64x32_sse2,
                                10),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_10_sub_pixel_avg_variance32x64_sse2,
                                10),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance32x32_sse2,
                                10),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance32x16_sse2,
                                10),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance16x32_sse2,
                                10),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance16x16_sse2,
                                10),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_10_sub_pixel_avg_variance16x8_sse2,
                                10),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance8x16_sse2,
                                10),
        SubpelAvgVarianceParams(3, 3,
                                &vpx_highbd_10_sub_pixel_avg_variance8x8_sse2,
                                10),
        SubpelAvgVarianceParams(3, 2,
                                &vpx_highbd_10_sub_pixel_avg_variance8x4_sse2,
                                10),
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_8_sub_pixel_avg_variance64x64_sse2,
                                8),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance64x32_sse2,
                                8),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_8_sub_pixel_avg_variance32x64_sse2,
                                8),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance32x32_sse2,
                                8),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance32x16_sse2,
                                8),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance16x32_sse2,
                                8),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance16x16_sse2,
                                8),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_8_sub_pixel_avg_variance16x8_sse2,
                                8),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance8x16_sse2,
                                8),
        SubpelAvgVarianceParams(3, 3,
                                &vpx_highbd_8_sub_pixel_avg_variance8x8_sse2,
                                8),
        SubpelAvgVarianceParams(3, 2,
                                &vpx_highbd_8_sub_pixel_avg_variance8x4_sse2,
                                8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // HAVE_SSE2

#if HAVE_SSSE3
INSTANTIATE_TEST_SUITE_P(
    SSSE3, VpxSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_sub_pixel_variance64x64_ssse3, 0),
        SubpelVarianceParams(6, 5, &vpx_sub_pixel_variance64x32_ssse3, 0),
        SubpelVarianceParams(5, 6, &vpx_sub_pixel_variance32x64_ssse3, 0),
        SubpelVarianceParams(5, 5, &vpx_sub_pixel_variance32x32_ssse3, 0),
        SubpelVarianceParams(5, 4, &vpx_sub_pixel_variance32x16_ssse3, 0),
        SubpelVarianceParams(4, 5, &vpx_sub_pixel_variance16x32_ssse3, 0),
        SubpelVarianceParams(4, 4, &vpx_sub_pixel_variance16x16_ssse3, 0),
        SubpelVarianceParams(4, 3, &vpx_sub_pixel_variance16x8_ssse3, 0),
        SubpelVarianceParams(3, 4, &vpx_sub_pixel_variance8x16_ssse3, 0),
        SubpelVarianceParams(3, 3, &vpx_sub_pixel_variance8x8_ssse3, 0),
        SubpelVarianceParams(3, 2, &vpx_sub_pixel_variance8x4_ssse3, 0),
        SubpelVarianceParams(2, 3, &vpx_sub_pixel_variance4x8_ssse3, 0),
        SubpelVarianceParams(2, 2, &vpx_sub_pixel_variance4x4_ssse3, 0)));

INSTANTIATE_TEST_SUITE_P(
    SSSE3, VpxSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6, &vpx_sub_pixel_avg_variance64x64_ssse3,
                                0),
        SubpelAvgVarianceParams(6, 5, &vpx_sub_pixel_avg_variance64x32_ssse3,
                                0),
        SubpelAvgVarianceParams(5, 6, &vpx_sub_pixel_avg_variance32x64_ssse3,
                                0),
        SubpelAvgVarianceParams(5, 5, &vpx_sub_pixel_avg_variance32x32_ssse3,
                                0),
        SubpelAvgVarianceParams(5, 4, &vpx_sub_pixel_avg_variance32x16_ssse3,
                                0),
        SubpelAvgVarianceParams(4, 5, &vpx_sub_pixel_avg_variance16x32_ssse3,
                                0),
        SubpelAvgVarianceParams(4, 4, &vpx_sub_pixel_avg_variance16x16_ssse3,
                                0),
        SubpelAvgVarianceParams(4, 3, &vpx_sub_pixel_avg_variance16x8_ssse3, 0),
        SubpelAvgVarianceParams(3, 4, &vpx_sub_pixel_avg_variance8x16_ssse3, 0),
        SubpelAvgVarianceParams(3, 3, &vpx_sub_pixel_avg_variance8x8_ssse3, 0),
        SubpelAvgVarianceParams(3, 2, &vpx_sub_pixel_avg_variance8x4_ssse3, 0),
        SubpelAvgVarianceParams(2, 3, &vpx_sub_pixel_avg_variance4x8_ssse3, 0),
        SubpelAvgVarianceParams(2, 2, &vpx_sub_pixel_avg_variance4x4_ssse3,
                                0)));
#endif  // HAVE_SSSE3

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(AVX2, VpxMseTest,
                         ::testing::Values(MseParams(4, 4, &vpx_mse16x16_avx2),
                                           MseParams(4, 3, &vpx_mse16x8_avx2)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_avx2),
                      VarianceParams(6, 5, &vpx_variance64x32_avx2),
                      VarianceParams(5, 6, &vpx_variance32x64_avx2),
                      VarianceParams(5, 5, &vpx_variance32x32_avx2),
                      VarianceParams(5, 4, &vpx_variance32x16_avx2),
                      VarianceParams(4, 5, &vpx_variance16x32_avx2),
                      VarianceParams(4, 4, &vpx_variance16x16_avx2),
                      VarianceParams(4, 3, &vpx_variance16x8_avx2),
                      VarianceParams(3, 4, &vpx_variance8x16_avx2),
                      VarianceParams(3, 3, &vpx_variance8x8_avx2),
                      VarianceParams(3, 2, &vpx_variance8x4_avx2)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, VpxSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_sub_pixel_variance64x64_avx2, 0),
        SubpelVarianceParams(5, 5, &vpx_sub_pixel_variance32x32_avx2, 0)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, VpxSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6, &vpx_sub_pixel_avg_variance64x64_avx2, 0),
        SubpelAvgVarianceParams(5, 5, &vpx_sub_pixel_avg_variance32x32_avx2,
                                0)));
#endif  // HAVE_AVX2

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, VpxSseTest,
                         ::testing::Values(SseParams(2, 2,
                                                     &vpx_get4x4sse_cs_neon)));

INSTANTIATE_TEST_SUITE_P(NEON, VpxMseTest,
                         ::testing::Values(MseParams(4, 4, &vpx_mse16x16_neon),
                                           MseParams(4, 3, &vpx_mse16x8_neon),
                                           MseParams(3, 4, &vpx_mse8x16_neon),
                                           MseParams(3, 3, &vpx_mse8x8_neon)));

INSTANTIATE_TEST_SUITE_P(
    NEON, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_neon),
                      VarianceParams(6, 5, &vpx_variance64x32_neon),
                      VarianceParams(5, 6, &vpx_variance32x64_neon),
                      VarianceParams(5, 5, &vpx_variance32x32_neon),
                      VarianceParams(5, 4, &vpx_variance32x16_neon),
                      VarianceParams(4, 5, &vpx_variance16x32_neon),
                      VarianceParams(4, 4, &vpx_variance16x16_neon),
                      VarianceParams(4, 3, &vpx_variance16x8_neon),
                      VarianceParams(3, 4, &vpx_variance8x16_neon),
                      VarianceParams(3, 3, &vpx_variance8x8_neon),
                      VarianceParams(3, 2, &vpx_variance8x4_neon),
                      VarianceParams(2, 3, &vpx_variance4x8_neon),
                      VarianceParams(2, 2, &vpx_variance4x4_neon)));

INSTANTIATE_TEST_SUITE_P(
    NEON, VpxGetVarianceTest,
    ::testing::Values(GetVarianceParams(4, 4, &vpx_get16x16var_neon),
                      GetVarianceParams(3, 3, &vpx_get8x8var_neon),
                      GetVarianceParams(4, 4, &vpx_get16x16var_neon),
                      GetVarianceParams(3, 3, &vpx_get8x8var_neon),
                      GetVarianceParams(4, 4, &vpx_get16x16var_neon),
                      GetVarianceParams(3, 3, &vpx_get8x8var_neon)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, VpxSseTest,
    ::testing::Values(SseParams(2, 2, &vpx_get4x4sse_cs_neon_dotprod)));

INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, VpxMseTest,
    ::testing::Values(MseParams(4, 4, &vpx_mse16x16_neon_dotprod),
                      MseParams(4, 3, &vpx_mse16x8_neon_dotprod),
                      MseParams(3, 4, &vpx_mse8x16_neon_dotprod),
                      MseParams(3, 3, &vpx_mse8x8_neon_dotprod)));

INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_neon_dotprod),
                      VarianceParams(6, 5, &vpx_variance64x32_neon_dotprod),
                      VarianceParams(5, 6, &vpx_variance32x64_neon_dotprod),
                      VarianceParams(5, 5, &vpx_variance32x32_neon_dotprod),
                      VarianceParams(5, 4, &vpx_variance32x16_neon_dotprod),
                      VarianceParams(4, 5, &vpx_variance16x32_neon_dotprod),
                      VarianceParams(4, 4, &vpx_variance16x16_neon_dotprod),
                      VarianceParams(4, 3, &vpx_variance16x8_neon_dotprod),
                      VarianceParams(3, 4, &vpx_variance8x16_neon_dotprod),
                      VarianceParams(3, 3, &vpx_variance8x8_neon_dotprod),
                      VarianceParams(3, 2, &vpx_variance8x4_neon_dotprod),
                      VarianceParams(2, 3, &vpx_variance4x8_neon_dotprod),
                      VarianceParams(2, 2, &vpx_variance4x4_neon_dotprod)));

INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, VpxGetVarianceTest,
    ::testing::Values(GetVarianceParams(4, 4, &vpx_get16x16var_neon_dotprod),
                      GetVarianceParams(3, 3, &vpx_get8x8var_neon_dotprod),
                      GetVarianceParams(4, 4, &vpx_get16x16var_neon_dotprod),
                      GetVarianceParams(3, 3, &vpx_get8x8var_neon_dotprod),
                      GetVarianceParams(4, 4, &vpx_get16x16var_neon_dotprod),
                      GetVarianceParams(3, 3, &vpx_get8x8var_neon_dotprod)));
#endif  // HAVE_NEON_DOTPROD

INSTANTIATE_TEST_SUITE_P(
    NEON, VpxSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_sub_pixel_variance64x64_neon, 0),
        SubpelVarianceParams(6, 5, &vpx_sub_pixel_variance64x32_neon, 0),
        SubpelVarianceParams(5, 6, &vpx_sub_pixel_variance32x64_neon, 0),
        SubpelVarianceParams(5, 5, &vpx_sub_pixel_variance32x32_neon, 0),
        SubpelVarianceParams(5, 4, &vpx_sub_pixel_variance32x16_neon, 0),
        SubpelVarianceParams(4, 5, &vpx_sub_pixel_variance16x32_neon, 0),
        SubpelVarianceParams(4, 4, &vpx_sub_pixel_variance16x16_neon, 0),
        SubpelVarianceParams(4, 3, &vpx_sub_pixel_variance16x8_neon, 0),
        SubpelVarianceParams(3, 4, &vpx_sub_pixel_variance8x16_neon, 0),
        SubpelVarianceParams(3, 3, &vpx_sub_pixel_variance8x8_neon, 0),
        SubpelVarianceParams(3, 2, &vpx_sub_pixel_variance8x4_neon, 0),
        SubpelVarianceParams(2, 3, &vpx_sub_pixel_variance4x8_neon, 0),
        SubpelVarianceParams(2, 2, &vpx_sub_pixel_variance4x4_neon, 0)));

INSTANTIATE_TEST_SUITE_P(
    NEON, VpxSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6, &vpx_sub_pixel_avg_variance64x64_neon, 0),
        SubpelAvgVarianceParams(6, 5, &vpx_sub_pixel_avg_variance64x32_neon, 0),
        SubpelAvgVarianceParams(5, 6, &vpx_sub_pixel_avg_variance32x64_neon, 0),
        SubpelAvgVarianceParams(5, 5, &vpx_sub_pixel_avg_variance32x32_neon, 0),
        SubpelAvgVarianceParams(5, 4, &vpx_sub_pixel_avg_variance32x16_neon, 0),
        SubpelAvgVarianceParams(4, 5, &vpx_sub_pixel_avg_variance16x32_neon, 0),
        SubpelAvgVarianceParams(4, 4, &vpx_sub_pixel_avg_variance16x16_neon, 0),
        SubpelAvgVarianceParams(4, 3, &vpx_sub_pixel_avg_variance16x8_neon, 0),
        SubpelAvgVarianceParams(3, 4, &vpx_sub_pixel_avg_variance8x16_neon, 0),
        SubpelAvgVarianceParams(3, 3, &vpx_sub_pixel_avg_variance8x8_neon, 0),
        SubpelAvgVarianceParams(3, 2, &vpx_sub_pixel_avg_variance8x4_neon, 0),
        SubpelAvgVarianceParams(2, 3, &vpx_sub_pixel_avg_variance4x8_neon, 0),
        SubpelAvgVarianceParams(2, 2, &vpx_sub_pixel_avg_variance4x4_neon, 0)));

#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    NEON, VpxHBDMseTest,
    ::testing::Values(
        MseParams(4, 4, &vpx_highbd_12_mse16x16_neon, VPX_BITS_12),
        MseParams(4, 3, &vpx_highbd_12_mse16x8_neon, VPX_BITS_12),
        MseParams(3, 4, &vpx_highbd_12_mse8x16_neon, VPX_BITS_12),
        MseParams(3, 3, &vpx_highbd_12_mse8x8_neon, VPX_BITS_12),
        MseParams(4, 4, &vpx_highbd_10_mse16x16_neon, VPX_BITS_10),
        MseParams(4, 3, &vpx_highbd_10_mse16x8_neon, VPX_BITS_10),
        MseParams(3, 4, &vpx_highbd_10_mse8x16_neon, VPX_BITS_10),
        MseParams(3, 3, &vpx_highbd_10_mse8x8_neon, VPX_BITS_10),
        MseParams(4, 4, &vpx_highbd_8_mse16x16_neon, VPX_BITS_8),
        MseParams(4, 3, &vpx_highbd_8_mse16x8_neon, VPX_BITS_8),
        MseParams(3, 4, &vpx_highbd_8_mse8x16_neon, VPX_BITS_8),
        MseParams(3, 3, &vpx_highbd_8_mse8x8_neon, VPX_BITS_8)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, VpxHBDMseTest,
    ::testing::Values(
        MseParams(4, 4, &vpx_highbd_8_mse16x16_neon_dotprod, VPX_BITS_8),
        MseParams(4, 3, &vpx_highbd_8_mse16x8_neon_dotprod, VPX_BITS_8),
        MseParams(3, 4, &vpx_highbd_8_mse8x16_neon_dotprod, VPX_BITS_8),
        MseParams(3, 3, &vpx_highbd_8_mse8x8_neon_dotprod, VPX_BITS_8)));
#endif  // HAVE_NEON_DOTPROD

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, VpxHBDMseTest,
    ::testing::Values(MseParams(4, 4, &vpx_highbd_12_mse16x16_sve, VPX_BITS_12),
                      MseParams(4, 3, &vpx_highbd_12_mse16x8_sve, VPX_BITS_12),
                      MseParams(3, 4, &vpx_highbd_12_mse8x16_sve, VPX_BITS_12),
                      MseParams(3, 3, &vpx_highbd_12_mse8x8_sve, VPX_BITS_12),
                      MseParams(4, 4, &vpx_highbd_10_mse16x16_sve, VPX_BITS_10),
                      MseParams(4, 3, &vpx_highbd_10_mse16x8_sve, VPX_BITS_10),
                      MseParams(3, 4, &vpx_highbd_10_mse8x16_sve, VPX_BITS_10),
                      MseParams(3, 3, &vpx_highbd_10_mse8x8_sve, VPX_BITS_10)));
#endif  // HAVE_SVE

INSTANTIATE_TEST_SUITE_P(
    NEON, VpxHBDVarianceTest,
    ::testing::Values(
        VarianceParams(6, 6, &vpx_highbd_12_variance64x64_neon, 12),
        VarianceParams(6, 5, &vpx_highbd_12_variance64x32_neon, 12),
        VarianceParams(5, 6, &vpx_highbd_12_variance32x64_neon, 12),
        VarianceParams(5, 5, &vpx_highbd_12_variance32x32_neon, 12),
        VarianceParams(5, 4, &vpx_highbd_12_variance32x16_neon, 12),
        VarianceParams(4, 5, &vpx_highbd_12_variance16x32_neon, 12),
        VarianceParams(4, 4, &vpx_highbd_12_variance16x16_neon, 12),
        VarianceParams(4, 3, &vpx_highbd_12_variance16x8_neon, 12),
        VarianceParams(3, 4, &vpx_highbd_12_variance8x16_neon, 12),
        VarianceParams(3, 3, &vpx_highbd_12_variance8x8_neon, 12),
        VarianceParams(3, 2, &vpx_highbd_12_variance8x4_neon, 12),
        VarianceParams(2, 3, &vpx_highbd_12_variance4x8_neon, 12),
        VarianceParams(2, 2, &vpx_highbd_12_variance4x4_neon, 12),
        VarianceParams(6, 6, &vpx_highbd_10_variance64x64_neon, 10),
        VarianceParams(6, 5, &vpx_highbd_10_variance64x32_neon, 10),
        VarianceParams(5, 6, &vpx_highbd_10_variance32x64_neon, 10),
        VarianceParams(5, 5, &vpx_highbd_10_variance32x32_neon, 10),
        VarianceParams(5, 4, &vpx_highbd_10_variance32x16_neon, 10),
        VarianceParams(4, 5, &vpx_highbd_10_variance16x32_neon, 10),
        VarianceParams(4, 4, &vpx_highbd_10_variance16x16_neon, 10),
        VarianceParams(4, 3, &vpx_highbd_10_variance16x8_neon, 10),
        VarianceParams(3, 4, &vpx_highbd_10_variance8x16_neon, 10),
        VarianceParams(3, 3, &vpx_highbd_10_variance8x8_neon, 10),
        VarianceParams(3, 2, &vpx_highbd_10_variance8x4_neon, 10),
        VarianceParams(2, 3, &vpx_highbd_10_variance4x8_neon, 10),
        VarianceParams(2, 2, &vpx_highbd_10_variance4x4_neon, 10),
        VarianceParams(6, 6, &vpx_highbd_8_variance64x64_neon, 8),
        VarianceParams(6, 5, &vpx_highbd_8_variance64x32_neon, 8),
        VarianceParams(5, 6, &vpx_highbd_8_variance32x64_neon, 8),
        VarianceParams(5, 5, &vpx_highbd_8_variance32x32_neon, 8),
        VarianceParams(5, 4, &vpx_highbd_8_variance32x16_neon, 8),
        VarianceParams(4, 5, &vpx_highbd_8_variance16x32_neon, 8),
        VarianceParams(4, 4, &vpx_highbd_8_variance16x16_neon, 8),
        VarianceParams(4, 3, &vpx_highbd_8_variance16x8_neon, 8),
        VarianceParams(3, 4, &vpx_highbd_8_variance8x16_neon, 8),
        VarianceParams(3, 3, &vpx_highbd_8_variance8x8_neon, 8),
        VarianceParams(3, 2, &vpx_highbd_8_variance8x4_neon, 8),
        VarianceParams(2, 3, &vpx_highbd_8_variance4x8_neon, 8),
        VarianceParams(2, 2, &vpx_highbd_8_variance4x4_neon, 8)));

INSTANTIATE_TEST_SUITE_P(
    NEON, VpxHBDGetVarianceTest,
    ::testing::Values(
        GetVarianceParams(4, 4, &vpx_highbd_12_get16x16var_neon, 12),
        GetVarianceParams(3, 3, &vpx_highbd_12_get8x8var_neon, 12),
        GetVarianceParams(4, 4, &vpx_highbd_10_get16x16var_neon, 10),
        GetVarianceParams(3, 3, &vpx_highbd_10_get8x8var_neon, 10),
        GetVarianceParams(4, 4, &vpx_highbd_8_get16x16var_neon, 8),
        GetVarianceParams(3, 3, &vpx_highbd_8_get8x8var_neon, 8)));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, VpxHBDGetVarianceTest,
    ::testing::Values(
        GetVarianceParams(4, 4, &vpx_highbd_12_get16x16var_sve, 12),
        GetVarianceParams(3, 3, &vpx_highbd_12_get8x8var_sve, 12),
        GetVarianceParams(4, 4, &vpx_highbd_10_get16x16var_sve, 10),
        GetVarianceParams(3, 3, &vpx_highbd_10_get8x8var_sve, 10),
        GetVarianceParams(4, 4, &vpx_highbd_8_get16x16var_sve, 8),
        GetVarianceParams(3, 3, &vpx_highbd_8_get8x8var_sve, 8)));
#endif  // HAVE_SVE

INSTANTIATE_TEST_SUITE_P(
    NEON, VpxHBDSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_highbd_12_sub_pixel_variance64x64_neon,
                             12),
        SubpelVarianceParams(6, 5, &vpx_highbd_12_sub_pixel_variance64x32_neon,
                             12),
        SubpelVarianceParams(5, 6, &vpx_highbd_12_sub_pixel_variance32x64_neon,
                             12),
        SubpelVarianceParams(5, 5, &vpx_highbd_12_sub_pixel_variance32x32_neon,
                             12),
        SubpelVarianceParams(5, 4, &vpx_highbd_12_sub_pixel_variance32x16_neon,
                             12),
        SubpelVarianceParams(4, 5, &vpx_highbd_12_sub_pixel_variance16x32_neon,
                             12),
        SubpelVarianceParams(4, 4, &vpx_highbd_12_sub_pixel_variance16x16_neon,
                             12),
        SubpelVarianceParams(4, 3, &vpx_highbd_12_sub_pixel_variance16x8_neon,
                             12),
        SubpelVarianceParams(3, 4, &vpx_highbd_12_sub_pixel_variance8x16_neon,
                             12),
        SubpelVarianceParams(3, 3, &vpx_highbd_12_sub_pixel_variance8x8_neon,
                             12),
        SubpelVarianceParams(3, 2, &vpx_highbd_12_sub_pixel_variance8x4_neon,
                             12),
        SubpelVarianceParams(2, 3, &vpx_highbd_12_sub_pixel_variance4x8_neon,
                             12),
        SubpelVarianceParams(2, 2, &vpx_highbd_12_sub_pixel_variance4x4_neon,
                             12),
        SubpelVarianceParams(6, 6, &vpx_highbd_10_sub_pixel_variance64x64_neon,
                             10),
        SubpelVarianceParams(6, 5, &vpx_highbd_10_sub_pixel_variance64x32_neon,
                             10),
        SubpelVarianceParams(5, 6, &vpx_highbd_10_sub_pixel_variance32x64_neon,
                             10),
        SubpelVarianceParams(5, 5, &vpx_highbd_10_sub_pixel_variance32x32_neon,
                             10),
        SubpelVarianceParams(5, 4, &vpx_highbd_10_sub_pixel_variance32x16_neon,
                             10),
        SubpelVarianceParams(4, 5, &vpx_highbd_10_sub_pixel_variance16x32_neon,
                             10),
        SubpelVarianceParams(4, 4, &vpx_highbd_10_sub_pixel_variance16x16_neon,
                             10),
        SubpelVarianceParams(4, 3, &vpx_highbd_10_sub_pixel_variance16x8_neon,
                             10),
        SubpelVarianceParams(3, 4, &vpx_highbd_10_sub_pixel_variance8x16_neon,
                             10),
        SubpelVarianceParams(3, 3, &vpx_highbd_10_sub_pixel_variance8x8_neon,
                             10),
        SubpelVarianceParams(3, 2, &vpx_highbd_10_sub_pixel_variance8x4_neon,
                             10),
        SubpelVarianceParams(2, 3, &vpx_highbd_10_sub_pixel_variance4x8_neon,
                             10),
        SubpelVarianceParams(2, 2, &vpx_highbd_10_sub_pixel_variance4x4_neon,
                             10),
        SubpelVarianceParams(6, 6, &vpx_highbd_8_sub_pixel_variance64x64_neon,
                             8),
        SubpelVarianceParams(6, 5, &vpx_highbd_8_sub_pixel_variance64x32_neon,
                             8),
        SubpelVarianceParams(5, 6, &vpx_highbd_8_sub_pixel_variance32x64_neon,
                             8),
        SubpelVarianceParams(5, 5, &vpx_highbd_8_sub_pixel_variance32x32_neon,
                             8),
        SubpelVarianceParams(5, 4, &vpx_highbd_8_sub_pixel_variance32x16_neon,
                             8),
        SubpelVarianceParams(4, 5, &vpx_highbd_8_sub_pixel_variance16x32_neon,
                             8),
        SubpelVarianceParams(4, 4, &vpx_highbd_8_sub_pixel_variance16x16_neon,
                             8),
        SubpelVarianceParams(4, 3, &vpx_highbd_8_sub_pixel_variance16x8_neon,
                             8),
        SubpelVarianceParams(3, 4, &vpx_highbd_8_sub_pixel_variance8x16_neon,
                             8),
        SubpelVarianceParams(3, 3, &vpx_highbd_8_sub_pixel_variance8x8_neon, 8),
        SubpelVarianceParams(3, 2, &vpx_highbd_8_sub_pixel_variance8x4_neon, 8),
        SubpelVarianceParams(2, 3, &vpx_highbd_8_sub_pixel_variance4x8_neon, 8),
        SubpelVarianceParams(2, 2, &vpx_highbd_8_sub_pixel_variance4x4_neon,
                             8)));

INSTANTIATE_TEST_SUITE_P(
    NEON, VpxHBDSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_12_sub_pixel_avg_variance64x64_neon,
                                12),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance64x32_neon,
                                12),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_12_sub_pixel_avg_variance32x64_neon,
                                12),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance32x32_neon,
                                12),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance32x16_neon,
                                12),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_12_sub_pixel_avg_variance16x32_neon,
                                12),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance16x16_neon,
                                12),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_12_sub_pixel_avg_variance16x8_neon,
                                12),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_12_sub_pixel_avg_variance8x16_neon,
                                12),
        SubpelAvgVarianceParams(3, 3,
                                &vpx_highbd_12_sub_pixel_avg_variance8x8_neon,
                                12),
        SubpelAvgVarianceParams(3, 2,
                                &vpx_highbd_12_sub_pixel_avg_variance8x4_neon,
                                12),
        SubpelAvgVarianceParams(2, 3,
                                &vpx_highbd_12_sub_pixel_avg_variance4x8_neon,
                                12),
        SubpelAvgVarianceParams(2, 2,
                                &vpx_highbd_12_sub_pixel_avg_variance4x4_neon,
                                12),
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_10_sub_pixel_avg_variance64x64_neon,
                                10),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance64x32_neon,
                                10),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_10_sub_pixel_avg_variance32x64_neon,
                                10),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance32x32_neon,
                                10),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance32x16_neon,
                                10),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_10_sub_pixel_avg_variance16x32_neon,
                                10),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance16x16_neon,
                                10),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_10_sub_pixel_avg_variance16x8_neon,
                                10),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_10_sub_pixel_avg_variance8x16_neon,
                                10),
        SubpelAvgVarianceParams(3, 3,
                                &vpx_highbd_10_sub_pixel_avg_variance8x8_neon,
                                10),
        SubpelAvgVarianceParams(3, 2,
                                &vpx_highbd_10_sub_pixel_avg_variance8x4_neon,
                                10),
        SubpelAvgVarianceParams(2, 3,
                                &vpx_highbd_10_sub_pixel_avg_variance4x8_neon,
                                10),
        SubpelAvgVarianceParams(2, 2,
                                &vpx_highbd_10_sub_pixel_avg_variance4x4_neon,
                                10),
        SubpelAvgVarianceParams(6, 6,
                                &vpx_highbd_8_sub_pixel_avg_variance64x64_neon,
                                8),
        SubpelAvgVarianceParams(6, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance64x32_neon,
                                8),
        SubpelAvgVarianceParams(5, 6,
                                &vpx_highbd_8_sub_pixel_avg_variance32x64_neon,
                                8),
        SubpelAvgVarianceParams(5, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance32x32_neon,
                                8),
        SubpelAvgVarianceParams(5, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance32x16_neon,
                                8),
        SubpelAvgVarianceParams(4, 5,
                                &vpx_highbd_8_sub_pixel_avg_variance16x32_neon,
                                8),
        SubpelAvgVarianceParams(4, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance16x16_neon,
                                8),
        SubpelAvgVarianceParams(4, 3,
                                &vpx_highbd_8_sub_pixel_avg_variance16x8_neon,
                                8),
        SubpelAvgVarianceParams(3, 4,
                                &vpx_highbd_8_sub_pixel_avg_variance8x16_neon,
                                8),
        SubpelAvgVarianceParams(3, 3,
                                &vpx_highbd_8_sub_pixel_avg_variance8x8_neon,
                                8),
        SubpelAvgVarianceParams(3, 2,
                                &vpx_highbd_8_sub_pixel_avg_variance8x4_neon,
                                8),
        SubpelAvgVarianceParams(2, 3,
                                &vpx_highbd_8_sub_pixel_avg_variance4x8_neon,
                                8),
        SubpelAvgVarianceParams(2, 2,
                                &vpx_highbd_8_sub_pixel_avg_variance4x4_neon,
                                8)));

#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // HAVE_NEON

#if HAVE_SVE
#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    SVE, VpxHBDVarianceTest,
    ::testing::Values(
        VarianceParams(6, 6, &vpx_highbd_12_variance64x64_sve, 12),
        VarianceParams(6, 5, &vpx_highbd_12_variance64x32_sve, 12),
        VarianceParams(5, 6, &vpx_highbd_12_variance32x64_sve, 12),
        VarianceParams(5, 5, &vpx_highbd_12_variance32x32_sve, 12),
        VarianceParams(5, 4, &vpx_highbd_12_variance32x16_sve, 12),
        VarianceParams(4, 5, &vpx_highbd_12_variance16x32_sve, 12),
        VarianceParams(4, 4, &vpx_highbd_12_variance16x16_sve, 12),
        VarianceParams(4, 3, &vpx_highbd_12_variance16x8_sve, 12),
        VarianceParams(3, 4, &vpx_highbd_12_variance8x16_sve, 12),
        VarianceParams(3, 3, &vpx_highbd_12_variance8x8_sve, 12),
        VarianceParams(3, 2, &vpx_highbd_12_variance8x4_sve, 12),
        VarianceParams(2, 3, &vpx_highbd_12_variance4x8_sve, 12),
        VarianceParams(2, 2, &vpx_highbd_12_variance4x4_sve, 12),
        VarianceParams(6, 6, &vpx_highbd_10_variance64x64_sve, 10),
        VarianceParams(6, 5, &vpx_highbd_10_variance64x32_sve, 10),
        VarianceParams(5, 6, &vpx_highbd_10_variance32x64_sve, 10),
        VarianceParams(5, 5, &vpx_highbd_10_variance32x32_sve, 10),
        VarianceParams(5, 4, &vpx_highbd_10_variance32x16_sve, 10),
        VarianceParams(4, 5, &vpx_highbd_10_variance16x32_sve, 10),
        VarianceParams(4, 4, &vpx_highbd_10_variance16x16_sve, 10),
        VarianceParams(4, 3, &vpx_highbd_10_variance16x8_sve, 10),
        VarianceParams(3, 4, &vpx_highbd_10_variance8x16_sve, 10),
        VarianceParams(3, 3, &vpx_highbd_10_variance8x8_sve, 10),
        VarianceParams(3, 2, &vpx_highbd_10_variance8x4_sve, 10),
        VarianceParams(2, 3, &vpx_highbd_10_variance4x8_sve, 10),
        VarianceParams(2, 2, &vpx_highbd_10_variance4x4_sve, 10),
        VarianceParams(6, 6, &vpx_highbd_8_variance64x64_sve, 8),
        VarianceParams(6, 5, &vpx_highbd_8_variance64x32_sve, 8),
        VarianceParams(5, 6, &vpx_highbd_8_variance32x64_sve, 8),
        VarianceParams(5, 5, &vpx_highbd_8_variance32x32_sve, 8),
        VarianceParams(5, 4, &vpx_highbd_8_variance32x16_sve, 8),
        VarianceParams(4, 5, &vpx_highbd_8_variance16x32_sve, 8),
        VarianceParams(4, 4, &vpx_highbd_8_variance16x16_sve, 8),
        VarianceParams(4, 3, &vpx_highbd_8_variance16x8_sve, 8),
        VarianceParams(3, 4, &vpx_highbd_8_variance8x16_sve, 8),
        VarianceParams(3, 3, &vpx_highbd_8_variance8x8_sve, 8),
        VarianceParams(3, 2, &vpx_highbd_8_variance8x4_sve, 8),
        VarianceParams(2, 3, &vpx_highbd_8_variance4x8_sve, 8),
        VarianceParams(2, 2, &vpx_highbd_8_variance4x4_sve, 8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // HAVE_SVE

#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(MSA, SumOfSquaresTest,
                         ::testing::Values(vpx_get_mb_ss_msa));

INSTANTIATE_TEST_SUITE_P(MSA, VpxSseTest,
                         ::testing::Values(SseParams(2, 2,
                                                     &vpx_get4x4sse_cs_msa)));

INSTANTIATE_TEST_SUITE_P(MSA, VpxMseTest,
                         ::testing::Values(MseParams(4, 4, &vpx_mse16x16_msa),
                                           MseParams(4, 3, &vpx_mse16x8_msa),
                                           MseParams(3, 4, &vpx_mse8x16_msa),
                                           MseParams(3, 3, &vpx_mse8x8_msa)));

INSTANTIATE_TEST_SUITE_P(
    MSA, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_msa),
                      VarianceParams(6, 5, &vpx_variance64x32_msa),
                      VarianceParams(5, 6, &vpx_variance32x64_msa),
                      VarianceParams(5, 5, &vpx_variance32x32_msa),
                      VarianceParams(5, 4, &vpx_variance32x16_msa),
                      VarianceParams(4, 5, &vpx_variance16x32_msa),
                      VarianceParams(4, 4, &vpx_variance16x16_msa),
                      VarianceParams(4, 3, &vpx_variance16x8_msa),
                      VarianceParams(3, 4, &vpx_variance8x16_msa),
                      VarianceParams(3, 3, &vpx_variance8x8_msa),
                      VarianceParams(3, 2, &vpx_variance8x4_msa),
                      VarianceParams(2, 3, &vpx_variance4x8_msa),
                      VarianceParams(2, 2, &vpx_variance4x4_msa)));

INSTANTIATE_TEST_SUITE_P(
    MSA, VpxGetVarianceTest,
    ::testing::Values(GetVarianceParams(4, 4, &vpx_get16x16var_msa),
                      GetVarianceParams(3, 3, &vpx_get8x8var_msa),
                      GetVarianceParams(4, 4, &vpx_get16x16var_msa),
                      GetVarianceParams(3, 3, &vpx_get8x8var_msa),
                      GetVarianceParams(4, 4, &vpx_get16x16var_msa),
                      GetVarianceParams(3, 3, &vpx_get8x8var_msa)));

INSTANTIATE_TEST_SUITE_P(
    MSA, VpxSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(2, 2, &vpx_sub_pixel_variance4x4_msa, 0),
        SubpelVarianceParams(2, 3, &vpx_sub_pixel_variance4x8_msa, 0),
        SubpelVarianceParams(3, 2, &vpx_sub_pixel_variance8x4_msa, 0),
        SubpelVarianceParams(3, 3, &vpx_sub_pixel_variance8x8_msa, 0),
        SubpelVarianceParams(3, 4, &vpx_sub_pixel_variance8x16_msa, 0),
        SubpelVarianceParams(4, 3, &vpx_sub_pixel_variance16x8_msa, 0),
        SubpelVarianceParams(4, 4, &vpx_sub_pixel_variance16x16_msa, 0),
        SubpelVarianceParams(4, 5, &vpx_sub_pixel_variance16x32_msa, 0),
        SubpelVarianceParams(5, 4, &vpx_sub_pixel_variance32x16_msa, 0),
        SubpelVarianceParams(5, 5, &vpx_sub_pixel_variance32x32_msa, 0),
        SubpelVarianceParams(5, 6, &vpx_sub_pixel_variance32x64_msa, 0),
        SubpelVarianceParams(6, 5, &vpx_sub_pixel_variance64x32_msa, 0),
        SubpelVarianceParams(6, 6, &vpx_sub_pixel_variance64x64_msa, 0)));

INSTANTIATE_TEST_SUITE_P(
    MSA, VpxSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6, &vpx_sub_pixel_avg_variance64x64_msa, 0),
        SubpelAvgVarianceParams(6, 5, &vpx_sub_pixel_avg_variance64x32_msa, 0),
        SubpelAvgVarianceParams(5, 6, &vpx_sub_pixel_avg_variance32x64_msa, 0),
        SubpelAvgVarianceParams(5, 5, &vpx_sub_pixel_avg_variance32x32_msa, 0),
        SubpelAvgVarianceParams(5, 4, &vpx_sub_pixel_avg_variance32x16_msa, 0),
        SubpelAvgVarianceParams(4, 5, &vpx_sub_pixel_avg_variance16x32_msa, 0),
        SubpelAvgVarianceParams(4, 4, &vpx_sub_pixel_avg_variance16x16_msa, 0),
        SubpelAvgVarianceParams(4, 3, &vpx_sub_pixel_avg_variance16x8_msa, 0),
        SubpelAvgVarianceParams(3, 4, &vpx_sub_pixel_avg_variance8x16_msa, 0),
        SubpelAvgVarianceParams(3, 3, &vpx_sub_pixel_avg_variance8x8_msa, 0),
        SubpelAvgVarianceParams(3, 2, &vpx_sub_pixel_avg_variance8x4_msa, 0),
        SubpelAvgVarianceParams(2, 3, &vpx_sub_pixel_avg_variance4x8_msa, 0),
        SubpelAvgVarianceParams(2, 2, &vpx_sub_pixel_avg_variance4x4_msa, 0)));
#endif  // HAVE_MSA

#if HAVE_VSX
INSTANTIATE_TEST_SUITE_P(VSX, SumOfSquaresTest,
                         ::testing::Values(vpx_get_mb_ss_vsx));

INSTANTIATE_TEST_SUITE_P(VSX, VpxSseTest,
                         ::testing::Values(SseParams(2, 2,
                                                     &vpx_get4x4sse_cs_vsx)));
INSTANTIATE_TEST_SUITE_P(VSX, VpxMseTest,
                         ::testing::Values(MseParams(4, 4, &vpx_mse16x16_vsx),
                                           MseParams(4, 3, &vpx_mse16x8_vsx),
                                           MseParams(3, 4, &vpx_mse8x16_vsx),
                                           MseParams(3, 3, &vpx_mse8x8_vsx)));

INSTANTIATE_TEST_SUITE_P(
    VSX, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_vsx),
                      VarianceParams(6, 5, &vpx_variance64x32_vsx),
                      VarianceParams(5, 6, &vpx_variance32x64_vsx),
                      VarianceParams(5, 5, &vpx_variance32x32_vsx),
                      VarianceParams(5, 4, &vpx_variance32x16_vsx),
                      VarianceParams(4, 5, &vpx_variance16x32_vsx),
                      VarianceParams(4, 4, &vpx_variance16x16_vsx),
                      VarianceParams(4, 3, &vpx_variance16x8_vsx),
                      VarianceParams(3, 4, &vpx_variance8x16_vsx),
                      VarianceParams(3, 3, &vpx_variance8x8_vsx),
                      VarianceParams(3, 2, &vpx_variance8x4_vsx),
                      VarianceParams(2, 3, &vpx_variance4x8_vsx),
                      VarianceParams(2, 2, &vpx_variance4x4_vsx)));

INSTANTIATE_TEST_SUITE_P(
    VSX, VpxGetVarianceTest,
    ::testing::Values(GetVarianceParams(4, 4, &vpx_get16x16var_vsx),
                      GetVarianceParams(3, 3, &vpx_get8x8var_vsx),
                      GetVarianceParams(4, 4, &vpx_get16x16var_vsx),
                      GetVarianceParams(3, 3, &vpx_get8x8var_vsx),
                      GetVarianceParams(4, 4, &vpx_get16x16var_vsx),
                      GetVarianceParams(3, 3, &vpx_get8x8var_vsx)));
#endif  // HAVE_VSX

#if HAVE_MMI
INSTANTIATE_TEST_SUITE_P(MMI, VpxMseTest,
                         ::testing::Values(MseParams(4, 4, &vpx_mse16x16_mmi),
                                           MseParams(4, 3, &vpx_mse16x8_mmi),
                                           MseParams(3, 4, &vpx_mse8x16_mmi),
                                           MseParams(3, 3, &vpx_mse8x8_mmi)));

INSTANTIATE_TEST_SUITE_P(
    MMI, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_mmi),
                      VarianceParams(6, 5, &vpx_variance64x32_mmi),
                      VarianceParams(5, 6, &vpx_variance32x64_mmi),
                      VarianceParams(5, 5, &vpx_variance32x32_mmi),
                      VarianceParams(5, 4, &vpx_variance32x16_mmi),
                      VarianceParams(4, 5, &vpx_variance16x32_mmi),
                      VarianceParams(4, 4, &vpx_variance16x16_mmi),
                      VarianceParams(4, 3, &vpx_variance16x8_mmi),
                      VarianceParams(3, 4, &vpx_variance8x16_mmi),
                      VarianceParams(3, 3, &vpx_variance8x8_mmi),
                      VarianceParams(3, 2, &vpx_variance8x4_mmi),
                      VarianceParams(2, 3, &vpx_variance4x8_mmi),
                      VarianceParams(2, 2, &vpx_variance4x4_mmi)));

INSTANTIATE_TEST_SUITE_P(
    MMI, VpxSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(6, 6, &vpx_sub_pixel_variance64x64_mmi, 0),
        SubpelVarianceParams(6, 5, &vpx_sub_pixel_variance64x32_mmi, 0),
        SubpelVarianceParams(5, 6, &vpx_sub_pixel_variance32x64_mmi, 0),
        SubpelVarianceParams(5, 5, &vpx_sub_pixel_variance32x32_mmi, 0),
        SubpelVarianceParams(5, 4, &vpx_sub_pixel_variance32x16_mmi, 0),
        SubpelVarianceParams(4, 5, &vpx_sub_pixel_variance16x32_mmi, 0),
        SubpelVarianceParams(4, 4, &vpx_sub_pixel_variance16x16_mmi, 0),
        SubpelVarianceParams(4, 3, &vpx_sub_pixel_variance16x8_mmi, 0),
        SubpelVarianceParams(3, 4, &vpx_sub_pixel_variance8x16_mmi, 0),
        SubpelVarianceParams(3, 3, &vpx_sub_pixel_variance8x8_mmi, 0),
        SubpelVarianceParams(3, 2, &vpx_sub_pixel_variance8x4_mmi, 0),
        SubpelVarianceParams(2, 3, &vpx_sub_pixel_variance4x8_mmi, 0),
        SubpelVarianceParams(2, 2, &vpx_sub_pixel_variance4x4_mmi, 0)));

INSTANTIATE_TEST_SUITE_P(
    MMI, VpxSubpelAvgVarianceTest,
    ::testing::Values(
        SubpelAvgVarianceParams(6, 6, &vpx_sub_pixel_avg_variance64x64_mmi, 0),
        SubpelAvgVarianceParams(6, 5, &vpx_sub_pixel_avg_variance64x32_mmi, 0),
        SubpelAvgVarianceParams(5, 6, &vpx_sub_pixel_avg_variance32x64_mmi, 0),
        SubpelAvgVarianceParams(5, 5, &vpx_sub_pixel_avg_variance32x32_mmi, 0),
        SubpelAvgVarianceParams(5, 4, &vpx_sub_pixel_avg_variance32x16_mmi, 0),
        SubpelAvgVarianceParams(4, 5, &vpx_sub_pixel_avg_variance16x32_mmi, 0),
        SubpelAvgVarianceParams(4, 4, &vpx_sub_pixel_avg_variance16x16_mmi, 0),
        SubpelAvgVarianceParams(4, 3, &vpx_sub_pixel_avg_variance16x8_mmi, 0),
        SubpelAvgVarianceParams(3, 4, &vpx_sub_pixel_avg_variance8x16_mmi, 0),
        SubpelAvgVarianceParams(3, 3, &vpx_sub_pixel_avg_variance8x8_mmi, 0),
        SubpelAvgVarianceParams(3, 2, &vpx_sub_pixel_avg_variance8x4_mmi, 0),
        SubpelAvgVarianceParams(2, 3, &vpx_sub_pixel_avg_variance4x8_mmi, 0),
        SubpelAvgVarianceParams(2, 2, &vpx_sub_pixel_avg_variance4x4_mmi, 0)));
#endif  // HAVE_MMI

#if HAVE_LSX
INSTANTIATE_TEST_SUITE_P(LSX, VpxMseTest,
                         ::testing::Values(MseParams(4, 4, &vpx_mse16x16_lsx)));

INSTANTIATE_TEST_SUITE_P(
    LSX, VpxVarianceTest,
    ::testing::Values(VarianceParams(6, 6, &vpx_variance64x64_lsx),
                      VarianceParams(5, 5, &vpx_variance32x32_lsx),
                      VarianceParams(4, 4, &vpx_variance16x16_lsx),
                      VarianceParams(3, 3, &vpx_variance8x8_lsx)));

INSTANTIATE_TEST_SUITE_P(
    LSX, VpxSubpelVarianceTest,
    ::testing::Values(
        SubpelVarianceParams(3, 3, &vpx_sub_pixel_variance8x8_lsx, 0),
        SubpelVarianceParams(4, 4, &vpx_sub_pixel_variance16x16_lsx, 0),
        SubpelVarianceParams(5, 5, &vpx_sub_pixel_variance32x32_lsx, 0)));

INSTANTIATE_TEST_SUITE_P(LSX, VpxSubpelAvgVarianceTest,
                         ::testing::Values(SubpelAvgVarianceParams(
                             6, 6, &vpx_sub_pixel_avg_variance64x64_lsx, 0)));
#endif
}  // namespace
