/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "gtest/gtest.h"

#include "./vp9_rtcd.h"
#include "test/acm_random.h"
#include "test/buffer.h"
#include "test/register_state_check.h"
#include "vpx_config.h"
#include "vpx_ports/vpx_timer.h"

namespace {

using ::libvpx_test::ACMRandom;
using ::libvpx_test::Buffer;

using YUVTemporalFilterFunc = void (*)(
    const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
    int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src,
    int uv_src_stride, const uint8_t *u_pre, const uint8_t *v_pre,
    int uv_pre_stride, unsigned int block_width, unsigned int block_height,
    int ss_x, int ss_y, int strength, const int *const blk_fw, int use_32x32,
    uint32_t *y_accumulator, uint16_t *y_count, uint32_t *u_accumulator,
    uint16_t *u_count, uint32_t *v_accumulator, uint16_t *v_count);

struct TemporalFilterWithBd {
  TemporalFilterWithBd(YUVTemporalFilterFunc func, int bitdepth)
      : temporal_filter(func), bd(bitdepth) {}

  YUVTemporalFilterFunc temporal_filter;
  int bd;
};

std::ostream &operator<<(std::ostream &os, const TemporalFilterWithBd &tf) {
  return os << "Bitdepth: " << tf.bd;
}

int GetFilterWeight(unsigned int row, unsigned int col,
                    unsigned int block_height, unsigned int block_width,
                    const int *const blk_fw, int use_32x32) {
  if (use_32x32) {
    return blk_fw[0];
  }

  return blk_fw[2 * (row >= block_height / 2) + (col >= block_width / 2)];
}

template <typename PixelType>
int GetModIndex(int sum_dist, int index, int rounding, int strength,
                int filter_weight) {
  int mod = sum_dist * 3 / index;
  mod += rounding;
  mod >>= strength;

  mod = VPXMIN(16, mod);

  mod = 16 - mod;
  mod *= filter_weight;

  return mod;
}

template <>
int GetModIndex<uint8_t>(int sum_dist, int index, int rounding, int strength,
                         int filter_weight) {
  unsigned int index_mult[14] = { 0,     0,     0,     0,     49152,
                                  39322, 32768, 28087, 24576, 21846,
                                  19661, 17874, 0,     15124 };

  assert(index >= 0 && index <= 13);
  assert(index_mult[index] != 0);

  int mod = (clamp(sum_dist, 0, UINT16_MAX) * index_mult[index]) >> 16;
  mod += rounding;
  mod >>= strength;

  mod = VPXMIN(16, mod);

  mod = 16 - mod;
  mod *= filter_weight;

  return mod;
}

template <>
int GetModIndex<uint16_t>(int sum_dist, int index, int rounding, int strength,
                          int filter_weight) {
  int64_t index_mult[14] = { 0U,          0U,          0U,          0U,
                             3221225472U, 2576980378U, 2147483648U, 1840700270U,
                             1610612736U, 1431655766U, 1288490189U, 1171354718U,
                             0U,          991146300U };

  assert(index >= 0 && index <= 13);
  assert(index_mult[index] != 0);

  int mod = static_cast<int>((sum_dist * index_mult[index]) >> 32);
  mod += rounding;
  mod >>= strength;

  mod = VPXMIN(16, mod);

  mod = 16 - mod;
  mod *= filter_weight;

  return mod;
}

template <typename PixelType>
void ApplyReferenceFilter(
    const Buffer<PixelType> &y_src, const Buffer<PixelType> &y_pre,
    const Buffer<PixelType> &u_src, const Buffer<PixelType> &v_src,
    const Buffer<PixelType> &u_pre, const Buffer<PixelType> &v_pre,
    unsigned int block_width, unsigned int block_height, int ss_x, int ss_y,
    int strength, const int *const blk_fw, int use_32x32,
    Buffer<uint32_t> *y_accumulator, Buffer<uint16_t> *y_counter,
    Buffer<uint32_t> *u_accumulator, Buffer<uint16_t> *u_counter,
    Buffer<uint32_t> *v_accumulator, Buffer<uint16_t> *v_counter) {
  const PixelType *y_src_ptr = y_src.TopLeftPixel();
  const PixelType *y_pre_ptr = y_pre.TopLeftPixel();
  const PixelType *u_src_ptr = u_src.TopLeftPixel();
  const PixelType *u_pre_ptr = u_pre.TopLeftPixel();
  const PixelType *v_src_ptr = v_src.TopLeftPixel();
  const PixelType *v_pre_ptr = v_pre.TopLeftPixel();

  const int uv_block_width = block_width >> ss_x,
            uv_block_height = block_height >> ss_y;
  const int y_src_stride = y_src.stride(), y_pre_stride = y_pre.stride();
  const int uv_src_stride = u_src.stride(), uv_pre_stride = u_pre.stride();
  const int y_diff_stride = block_width, uv_diff_stride = uv_block_width;

  Buffer<int> y_dif = Buffer<int>(block_width, block_height, 0);
  Buffer<int> u_dif = Buffer<int>(uv_block_width, uv_block_height, 0);
  Buffer<int> v_dif = Buffer<int>(uv_block_width, uv_block_height, 0);

  ASSERT_TRUE(y_dif.Init());
  ASSERT_TRUE(u_dif.Init());
  ASSERT_TRUE(v_dif.Init());
  y_dif.Set(0);
  u_dif.Set(0);
  v_dif.Set(0);

  int *y_diff_ptr = y_dif.TopLeftPixel();
  int *u_diff_ptr = u_dif.TopLeftPixel();
  int *v_diff_ptr = v_dif.TopLeftPixel();

  uint32_t *y_accum = y_accumulator->TopLeftPixel();
  uint32_t *u_accum = u_accumulator->TopLeftPixel();
  uint32_t *v_accum = v_accumulator->TopLeftPixel();
  uint16_t *y_count = y_counter->TopLeftPixel();
  uint16_t *u_count = u_counter->TopLeftPixel();
  uint16_t *v_count = v_counter->TopLeftPixel();

  const int y_accum_stride = y_accumulator->stride();
  const int u_accum_stride = u_accumulator->stride();
  const int v_accum_stride = v_accumulator->stride();
  const int y_count_stride = y_counter->stride();
  const int u_count_stride = u_counter->stride();
  const int v_count_stride = v_counter->stride();

  const int rounding = (1 << strength) >> 1;

  // Get the square diffs
  for (int row = 0; row < static_cast<int>(block_height); row++) {
    for (int col = 0; col < static_cast<int>(block_width); col++) {
      const int diff = y_src_ptr[row * y_src_stride + col] -
                       y_pre_ptr[row * y_pre_stride + col];
      y_diff_ptr[row * y_diff_stride + col] = diff * diff;
    }
  }

  for (int row = 0; row < uv_block_height; row++) {
    for (int col = 0; col < uv_block_width; col++) {
      const int u_diff = u_src_ptr[row * uv_src_stride + col] -
                         u_pre_ptr[row * uv_pre_stride + col];
      const int v_diff = v_src_ptr[row * uv_src_stride + col] -
                         v_pre_ptr[row * uv_pre_stride + col];
      u_diff_ptr[row * uv_diff_stride + col] = u_diff * u_diff;
      v_diff_ptr[row * uv_diff_stride + col] = v_diff * v_diff;
    }
  }

  // Apply the filter to luma
  for (int row = 0; row < static_cast<int>(block_height); row++) {
    for (int col = 0; col < static_cast<int>(block_width); col++) {
      const int uv_row = row >> ss_y;
      const int uv_col = col >> ss_x;
      const int filter_weight = GetFilterWeight(row, col, block_height,
                                                block_width, blk_fw, use_32x32);

      // First we get the modifier for the current y pixel
      const int y_pixel = y_pre_ptr[row * y_pre_stride + col];
      int y_num_used = 0;
      int y_mod = 0;

      // Sum the neighboring 3x3 y pixels
      for (int row_step = -1; row_step <= 1; row_step++) {
        for (int col_step = -1; col_step <= 1; col_step++) {
          const int sub_row = row + row_step;
          const int sub_col = col + col_step;

          if (sub_row >= 0 && sub_row < static_cast<int>(block_height) &&
              sub_col >= 0 && sub_col < static_cast<int>(block_width)) {
            y_mod += y_diff_ptr[sub_row * y_diff_stride + sub_col];
            y_num_used++;
          }
        }
      }

      // Sum the corresponding uv pixels to the current y modifier
      // Note we are rounding down instead of rounding to the nearest pixel.
      y_mod += u_diff_ptr[uv_row * uv_diff_stride + uv_col];
      y_mod += v_diff_ptr[uv_row * uv_diff_stride + uv_col];

      y_num_used += 2;

      // Set the modifier
      y_mod = GetModIndex<PixelType>(y_mod, y_num_used, rounding, strength,
                                     filter_weight);

      // Accumulate the result
      y_count[row * y_count_stride + col] += y_mod;
      y_accum[row * y_accum_stride + col] += y_mod * y_pixel;
    }
  }

  // Apply the filter to chroma
  for (int uv_row = 0; uv_row < uv_block_height; uv_row++) {
    for (int uv_col = 0; uv_col < uv_block_width; uv_col++) {
      const int y_row = uv_row << ss_y;
      const int y_col = uv_col << ss_x;
      const int filter_weight = GetFilterWeight(
          uv_row, uv_col, uv_block_height, uv_block_width, blk_fw, use_32x32);

      const int u_pixel = u_pre_ptr[uv_row * uv_pre_stride + uv_col];
      const int v_pixel = v_pre_ptr[uv_row * uv_pre_stride + uv_col];

      int uv_num_used = 0;
      int u_mod = 0, v_mod = 0;

      // Sum the neighboring 3x3 chromal pixels to the chroma modifier
      for (int row_step = -1; row_step <= 1; row_step++) {
        for (int col_step = -1; col_step <= 1; col_step++) {
          const int sub_row = uv_row + row_step;
          const int sub_col = uv_col + col_step;

          if (sub_row >= 0 && sub_row < uv_block_height && sub_col >= 0 &&
              sub_col < uv_block_width) {
            u_mod += u_diff_ptr[sub_row * uv_diff_stride + sub_col];
            v_mod += v_diff_ptr[sub_row * uv_diff_stride + sub_col];
            uv_num_used++;
          }
        }
      }

      // Sum all the luma pixels associated with the current luma pixel
      for (int row_step = 0; row_step < 1 + ss_y; row_step++) {
        for (int col_step = 0; col_step < 1 + ss_x; col_step++) {
          const int sub_row = y_row + row_step;
          const int sub_col = y_col + col_step;
          const int y_diff = y_diff_ptr[sub_row * y_diff_stride + sub_col];

          u_mod += y_diff;
          v_mod += y_diff;
          uv_num_used++;
        }
      }

      // Set the modifier
      u_mod = GetModIndex<PixelType>(u_mod, uv_num_used, rounding, strength,
                                     filter_weight);
      v_mod = GetModIndex<PixelType>(v_mod, uv_num_used, rounding, strength,
                                     filter_weight);

      // Accumulate the result
      u_count[uv_row * u_count_stride + uv_col] += u_mod;
      u_accum[uv_row * u_accum_stride + uv_col] += u_mod * u_pixel;
      v_count[uv_row * v_count_stride + uv_col] += v_mod;
      v_accum[uv_row * v_accum_stride + uv_col] += v_mod * v_pixel;
    }
  }
}

class YUVTemporalFilterTest
    : public ::testing::TestWithParam<TemporalFilterWithBd> {
 public:
  void SetUp() override {
    filter_func_ = GetParam().temporal_filter;
    bd_ = GetParam().bd;
    use_highbd_ = (bd_ != 8);

    rnd_.Reset(ACMRandom::DeterministicSeed());
    saturate_test_ = 0;
    num_repeats_ = 10;

    ASSERT_TRUE(bd_ == 8 || bd_ == 10 || bd_ == 12);
  }

 protected:
  template <typename PixelType>
  void CompareTestWithParam(int width, int height, int ss_x, int ss_y,
                            int filter_strength, int use_32x32,
                            const int *filter_weight);
  template <typename PixelType>
  void RunTestFilterWithParam(int width, int height, int ss_x, int ss_y,
                              int filter_strength, int use_32x32,
                              const int *filter_weight);
  YUVTemporalFilterFunc filter_func_;
  ACMRandom rnd_;
  int saturate_test_;
  int num_repeats_;
  int use_highbd_;
  int bd_;
};

template <typename PixelType>
void YUVTemporalFilterTest::CompareTestWithParam(int width, int height,
                                                 int ss_x, int ss_y,
                                                 int filter_strength,
                                                 int use_32x32,
                                                 const int *filter_weight) {
  const int uv_width = width >> ss_x, uv_height = height >> ss_y;

  Buffer<PixelType> y_src = Buffer<PixelType>(width, height, 0);
  Buffer<PixelType> y_pre = Buffer<PixelType>(width, height, 0);
  Buffer<uint16_t> y_count_ref = Buffer<uint16_t>(width, height, 0);
  Buffer<uint32_t> y_accum_ref = Buffer<uint32_t>(width, height, 0);
  Buffer<uint16_t> y_count_tst = Buffer<uint16_t>(width, height, 0);
  Buffer<uint32_t> y_accum_tst = Buffer<uint32_t>(width, height, 0);

  Buffer<PixelType> u_src = Buffer<PixelType>(uv_width, uv_height, 0);
  Buffer<PixelType> u_pre = Buffer<PixelType>(uv_width, uv_height, 0);
  Buffer<uint16_t> u_count_ref = Buffer<uint16_t>(uv_width, uv_height, 0);
  Buffer<uint32_t> u_accum_ref = Buffer<uint32_t>(uv_width, uv_height, 0);
  Buffer<uint16_t> u_count_tst = Buffer<uint16_t>(uv_width, uv_height, 0);
  Buffer<uint32_t> u_accum_tst = Buffer<uint32_t>(uv_width, uv_height, 0);

  Buffer<PixelType> v_src = Buffer<PixelType>(uv_width, uv_height, 0);
  Buffer<PixelType> v_pre = Buffer<PixelType>(uv_width, uv_height, 0);
  Buffer<uint16_t> v_count_ref = Buffer<uint16_t>(uv_width, uv_height, 0);
  Buffer<uint32_t> v_accum_ref = Buffer<uint32_t>(uv_width, uv_height, 0);
  Buffer<uint16_t> v_count_tst = Buffer<uint16_t>(uv_width, uv_height, 0);
  Buffer<uint32_t> v_accum_tst = Buffer<uint32_t>(uv_width, uv_height, 0);

  ASSERT_TRUE(y_src.Init());
  ASSERT_TRUE(y_pre.Init());
  ASSERT_TRUE(y_count_ref.Init());
  ASSERT_TRUE(y_accum_ref.Init());
  ASSERT_TRUE(y_count_tst.Init());
  ASSERT_TRUE(y_accum_tst.Init());
  ASSERT_TRUE(u_src.Init());
  ASSERT_TRUE(u_pre.Init());
  ASSERT_TRUE(u_count_ref.Init());
  ASSERT_TRUE(u_accum_ref.Init());
  ASSERT_TRUE(u_count_tst.Init());
  ASSERT_TRUE(u_accum_tst.Init());

  ASSERT_TRUE(v_src.Init());
  ASSERT_TRUE(v_pre.Init());
  ASSERT_TRUE(v_count_ref.Init());
  ASSERT_TRUE(v_accum_ref.Init());
  ASSERT_TRUE(v_count_tst.Init());
  ASSERT_TRUE(v_accum_tst.Init());

  y_accum_ref.Set(0);
  y_accum_tst.Set(0);
  y_count_ref.Set(0);
  y_count_tst.Set(0);
  u_accum_ref.Set(0);
  u_accum_tst.Set(0);
  u_count_ref.Set(0);
  u_count_tst.Set(0);
  v_accum_ref.Set(0);
  v_accum_tst.Set(0);
  v_count_ref.Set(0);
  v_count_tst.Set(0);

  for (int repeats = 0; repeats < num_repeats_; repeats++) {
    if (saturate_test_) {
      const int max_val = (1 << bd_) - 1;
      y_src.Set(max_val);
      y_pre.Set(0);
      u_src.Set(max_val);
      u_pre.Set(0);
      v_src.Set(max_val);
      v_pre.Set(0);
    } else {
      y_src.Set(&rnd_, 0, 7 << (bd_ - 8));
      y_pre.Set(&rnd_, 0, 7 << (bd_ - 8));
      u_src.Set(&rnd_, 0, 7 << (bd_ - 8));
      u_pre.Set(&rnd_, 0, 7 << (bd_ - 8));
      v_src.Set(&rnd_, 0, 7 << (bd_ - 8));
      v_pre.Set(&rnd_, 0, 7 << (bd_ - 8));
    }

    ApplyReferenceFilter<PixelType>(
        y_src, y_pre, u_src, v_src, u_pre, v_pre, width, height, ss_x, ss_y,
        filter_strength, filter_weight, use_32x32, &y_accum_ref, &y_count_ref,
        &u_accum_ref, &u_count_ref, &v_accum_ref, &v_count_ref);

    ASM_REGISTER_STATE_CHECK(filter_func_(
        reinterpret_cast<const uint8_t *>(y_src.TopLeftPixel()), y_src.stride(),
        reinterpret_cast<const uint8_t *>(y_pre.TopLeftPixel()), y_pre.stride(),
        reinterpret_cast<const uint8_t *>(u_src.TopLeftPixel()),
        reinterpret_cast<const uint8_t *>(v_src.TopLeftPixel()), u_src.stride(),
        reinterpret_cast<const uint8_t *>(u_pre.TopLeftPixel()),
        reinterpret_cast<const uint8_t *>(v_pre.TopLeftPixel()), u_pre.stride(),
        width, height, ss_x, ss_y, filter_strength, filter_weight, use_32x32,
        y_accum_tst.TopLeftPixel(), y_count_tst.TopLeftPixel(),
        u_accum_tst.TopLeftPixel(), u_count_tst.TopLeftPixel(),
        v_accum_tst.TopLeftPixel(), v_count_tst.TopLeftPixel()));

    EXPECT_TRUE(y_accum_tst.CheckValues(y_accum_ref));
    EXPECT_TRUE(y_count_tst.CheckValues(y_count_ref));
    EXPECT_TRUE(u_accum_tst.CheckValues(u_accum_ref));
    EXPECT_TRUE(u_count_tst.CheckValues(u_count_ref));
    EXPECT_TRUE(v_accum_tst.CheckValues(v_accum_ref));
    EXPECT_TRUE(v_count_tst.CheckValues(v_count_ref));

    if (HasFailure()) {
      if (use_32x32) {
        printf("SS_X: %d, SS_Y: %d, Strength: %d, Weight: %d\n", ss_x, ss_y,
               filter_strength, *filter_weight);
      } else {
        printf("SS_X: %d, SS_Y: %d, Strength: %d, Weights: %d,%d,%d,%d\n", ss_x,
               ss_y, filter_strength, filter_weight[0], filter_weight[1],
               filter_weight[2], filter_weight[3]);
      }
      y_accum_tst.PrintDifference(y_accum_ref);
      y_count_tst.PrintDifference(y_count_ref);
      u_accum_tst.PrintDifference(u_accum_ref);
      u_count_tst.PrintDifference(u_count_ref);
      v_accum_tst.PrintDifference(v_accum_ref);
      v_count_tst.PrintDifference(v_count_ref);

      return;
    }
  }
}

template <typename PixelType>
void YUVTemporalFilterTest::RunTestFilterWithParam(int width, int height,
                                                   int ss_x, int ss_y,
                                                   int filter_strength,
                                                   int use_32x32,
                                                   const int *filter_weight) {
  const int uv_width = width >> ss_x, uv_height = height >> ss_y;

  Buffer<PixelType> y_src = Buffer<PixelType>(width, height, 0);
  Buffer<PixelType> y_pre = Buffer<PixelType>(width, height, 0);
  Buffer<uint16_t> y_count = Buffer<uint16_t>(width, height, 0);
  Buffer<uint32_t> y_accum = Buffer<uint32_t>(width, height, 0);

  Buffer<PixelType> u_src = Buffer<PixelType>(uv_width, uv_height, 0);
  Buffer<PixelType> u_pre = Buffer<PixelType>(uv_width, uv_height, 0);
  Buffer<uint16_t> u_count = Buffer<uint16_t>(uv_width, uv_height, 0);
  Buffer<uint32_t> u_accum = Buffer<uint32_t>(uv_width, uv_height, 0);

  Buffer<PixelType> v_src = Buffer<PixelType>(uv_width, uv_height, 0);
  Buffer<PixelType> v_pre = Buffer<PixelType>(uv_width, uv_height, 0);
  Buffer<uint16_t> v_count = Buffer<uint16_t>(uv_width, uv_height, 0);
  Buffer<uint32_t> v_accum = Buffer<uint32_t>(uv_width, uv_height, 0);

  ASSERT_TRUE(y_src.Init());
  ASSERT_TRUE(y_pre.Init());
  ASSERT_TRUE(y_count.Init());
  ASSERT_TRUE(y_accum.Init());

  ASSERT_TRUE(u_src.Init());
  ASSERT_TRUE(u_pre.Init());
  ASSERT_TRUE(u_count.Init());
  ASSERT_TRUE(u_accum.Init());

  ASSERT_TRUE(v_src.Init());
  ASSERT_TRUE(v_pre.Init());
  ASSERT_TRUE(v_count.Init());
  ASSERT_TRUE(v_accum.Init());

  y_accum.Set(0);
  y_count.Set(0);

  u_accum.Set(0);
  u_count.Set(0);

  v_accum.Set(0);
  v_count.Set(0);

  y_src.Set(&rnd_, 0, 7 << (bd_ - 8));
  y_pre.Set(&rnd_, 0, 7 << (bd_ - 8));
  u_src.Set(&rnd_, 0, 7 << (bd_ - 8));
  u_pre.Set(&rnd_, 0, 7 << (bd_ - 8));
  v_src.Set(&rnd_, 0, 7 << (bd_ - 8));
  v_pre.Set(&rnd_, 0, 7 << (bd_ - 8));

  for (int repeats = 0; repeats < num_repeats_; repeats++) {
    ASM_REGISTER_STATE_CHECK(filter_func_(
        reinterpret_cast<const uint8_t *>(y_src.TopLeftPixel()), y_src.stride(),
        reinterpret_cast<const uint8_t *>(y_pre.TopLeftPixel()), y_pre.stride(),
        reinterpret_cast<const uint8_t *>(u_src.TopLeftPixel()),
        reinterpret_cast<const uint8_t *>(v_src.TopLeftPixel()), u_src.stride(),
        reinterpret_cast<const uint8_t *>(u_pre.TopLeftPixel()),
        reinterpret_cast<const uint8_t *>(v_pre.TopLeftPixel()), u_pre.stride(),
        width, height, ss_x, ss_y, filter_strength, filter_weight, use_32x32,
        y_accum.TopLeftPixel(), y_count.TopLeftPixel(), u_accum.TopLeftPixel(),
        u_count.TopLeftPixel(), v_accum.TopLeftPixel(),
        v_count.TopLeftPixel()));
  }
}

TEST_P(YUVTemporalFilterTest, Use32x32) {
  const int width = 32, height = 32;
  const int use_32x32 = 1;

  for (int ss_x = 0; ss_x <= 1; ss_x++) {
    for (int ss_y = 0; ss_y <= 1; ss_y++) {
      for (int filter_strength = 0; filter_strength <= 6;
           filter_strength += 2) {
        for (int filter_weight = 0; filter_weight <= 2; filter_weight++) {
          if (use_highbd_) {
            const int adjusted_strength = filter_strength + 2 * (bd_ - 8);
            CompareTestWithParam<uint16_t>(width, height, ss_x, ss_y,
                                           adjusted_strength, use_32x32,
                                           &filter_weight);
          } else {
            CompareTestWithParam<uint8_t>(width, height, ss_x, ss_y,
                                          filter_strength, use_32x32,
                                          &filter_weight);
          }
          ASSERT_FALSE(HasFailure());
        }
      }
    }
  }
}

TEST_P(YUVTemporalFilterTest, Use16x16) {
  const int width = 32, height = 32;
  const int use_32x32 = 0;

  for (int ss_x = 0; ss_x <= 1; ss_x++) {
    for (int ss_y = 0; ss_y <= 1; ss_y++) {
      for (int filter_idx = 0; filter_idx < 3 * 3 * 3 * 3; filter_idx++) {
        // Set up the filter
        int filter_weight[4];
        int filter_idx_cp = filter_idx;
        for (int idx = 0; idx < 4; idx++) {
          filter_weight[idx] = filter_idx_cp % 3;
          filter_idx_cp /= 3;
        }

        // Test each parameter
        for (int filter_strength = 0; filter_strength <= 6;
             filter_strength += 2) {
          if (use_highbd_) {
            const int adjusted_strength = filter_strength + 2 * (bd_ - 8);
            CompareTestWithParam<uint16_t>(width, height, ss_x, ss_y,
                                           adjusted_strength, use_32x32,
                                           filter_weight);
          } else {
            CompareTestWithParam<uint8_t>(width, height, ss_x, ss_y,
                                          filter_strength, use_32x32,
                                          filter_weight);
          }

          ASSERT_FALSE(HasFailure());
        }
      }
    }
  }
}

TEST_P(YUVTemporalFilterTest, SaturationTest) {
  const int width = 32, height = 32;
  const int use_32x32 = 1;
  const int filter_weight = 1;
  saturate_test_ = 1;

  for (int ss_x = 0; ss_x <= 1; ss_x++) {
    for (int ss_y = 0; ss_y <= 1; ss_y++) {
      for (int filter_strength = 0; filter_strength <= 6;
           filter_strength += 2) {
        if (use_highbd_) {
          const int adjusted_strength = filter_strength + 2 * (bd_ - 8);
          CompareTestWithParam<uint16_t>(width, height, ss_x, ss_y,
                                         adjusted_strength, use_32x32,
                                         &filter_weight);
        } else {
          CompareTestWithParam<uint8_t>(width, height, ss_x, ss_y,
                                        filter_strength, use_32x32,
                                        &filter_weight);
        }

        ASSERT_FALSE(HasFailure());
      }
    }
  }
}

TEST_P(YUVTemporalFilterTest, DISABLED_Speed) {
  const int width = 32, height = 32;
  num_repeats_ = 1000;

  for (int use_32x32 = 0; use_32x32 <= 1; use_32x32++) {
    const int num_filter_weights = use_32x32 ? 3 : 3 * 3 * 3 * 3;
    for (int ss_x = 0; ss_x <= 1; ss_x++) {
      for (int ss_y = 0; ss_y <= 1; ss_y++) {
        for (int filter_idx = 0; filter_idx < num_filter_weights;
             filter_idx++) {
          // Set up the filter
          int filter_weight[4];
          int filter_idx_cp = filter_idx;
          for (int idx = 0; idx < 4; idx++) {
            filter_weight[idx] = filter_idx_cp % 3;
            filter_idx_cp /= 3;
          }

          // Test each parameter
          for (int filter_strength = 0; filter_strength <= 6;
               filter_strength += 2) {
            vpx_usec_timer timer;
            vpx_usec_timer_start(&timer);

            if (use_highbd_) {
              RunTestFilterWithParam<uint16_t>(width, height, ss_x, ss_y,
                                               filter_strength, use_32x32,
                                               filter_weight);
            } else {
              RunTestFilterWithParam<uint8_t>(width, height, ss_x, ss_y,
                                              filter_strength, use_32x32,
                                              filter_weight);
            }

            vpx_usec_timer_mark(&timer);
            const int elapsed_time =
                static_cast<int>(vpx_usec_timer_elapsed(&timer));

            printf(
                "Bitdepth: %d, Use 32X32: %d, SS_X: %d, SS_Y: %d, Weight Idx: "
                "%d, Strength: %d, Time: %5d\n",
                bd_, use_32x32, ss_x, ss_y, filter_idx, filter_strength,
                elapsed_time);
          }
        }
      }
    }
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
#define WRAP_HIGHBD_FUNC(func, bd)                                            \
  void wrap_##func##_##bd(                                                    \
      const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,           \
      int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src,           \
      int uv_src_stride, const uint8_t *u_pre, const uint8_t *v_pre,          \
      int uv_pre_stride, unsigned int block_width, unsigned int block_height, \
      int ss_x, int ss_y, int strength, const int *const blk_fw,              \
      int use_32x32, uint32_t *y_accumulator, uint16_t *y_count,              \
      uint32_t *u_accumulator, uint16_t *u_count, uint32_t *v_accumulator,    \
      uint16_t *v_count) {                                                    \
    func(reinterpret_cast<const uint16_t *>(y_src), y_src_stride,             \
         reinterpret_cast<const uint16_t *>(y_pre), y_pre_stride,             \
         reinterpret_cast<const uint16_t *>(u_src),                           \
         reinterpret_cast<const uint16_t *>(v_src), uv_src_stride,            \
         reinterpret_cast<const uint16_t *>(u_pre),                           \
         reinterpret_cast<const uint16_t *>(v_pre), uv_pre_stride,            \
         block_width, block_height, ss_x, ss_y, strength, blk_fw, use_32x32,  \
         y_accumulator, y_count, u_accumulator, u_count, v_accumulator,       \
         v_count);                                                            \
  }

WRAP_HIGHBD_FUNC(vp9_highbd_apply_temporal_filter_c, 10)
WRAP_HIGHBD_FUNC(vp9_highbd_apply_temporal_filter_c, 12)

INSTANTIATE_TEST_SUITE_P(
    C, YUVTemporalFilterTest,
    ::testing::Values(
        TemporalFilterWithBd(&wrap_vp9_highbd_apply_temporal_filter_c_10, 10),
        TemporalFilterWithBd(&wrap_vp9_highbd_apply_temporal_filter_c_12, 12)));
#if HAVE_SSE4_1
WRAP_HIGHBD_FUNC(vp9_highbd_apply_temporal_filter_sse4_1, 10)
WRAP_HIGHBD_FUNC(vp9_highbd_apply_temporal_filter_sse4_1, 12)

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, YUVTemporalFilterTest,
    ::testing::Values(
        TemporalFilterWithBd(&wrap_vp9_highbd_apply_temporal_filter_sse4_1_10,
                             10),
        TemporalFilterWithBd(&wrap_vp9_highbd_apply_temporal_filter_sse4_1_12,
                             12)));
#endif  // HAVE_SSE4_1
#if HAVE_NEON
WRAP_HIGHBD_FUNC(vp9_highbd_apply_temporal_filter_neon, 10)
WRAP_HIGHBD_FUNC(vp9_highbd_apply_temporal_filter_neon, 12)

INSTANTIATE_TEST_SUITE_P(
    NEON, YUVTemporalFilterTest,
    ::testing::Values(
        TemporalFilterWithBd(&wrap_vp9_highbd_apply_temporal_filter_neon_10,
                             10),
        TemporalFilterWithBd(&wrap_vp9_highbd_apply_temporal_filter_neon_12,
                             12)));
#endif  // HAVE_NEON
#else
INSTANTIATE_TEST_SUITE_P(
    C, YUVTemporalFilterTest,
    ::testing::Values(TemporalFilterWithBd(&vp9_apply_temporal_filter_c, 8)));

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE4_1, YUVTemporalFilterTest,
                         ::testing::Values(TemporalFilterWithBd(
                             &vp9_apply_temporal_filter_sse4_1, 8)));
#endif  // HAVE_SSE4_1
#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, YUVTemporalFilterTest,
                         ::testing::Values(TemporalFilterWithBd(
                             &vp9_apply_temporal_filter_neon, 8)));
#endif  // HAVE_NEON
#endif  // CONFIG_VP9_HIGHBITDEPTH

}  // namespace
