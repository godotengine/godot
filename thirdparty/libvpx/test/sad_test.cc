/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/bench.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vpx/vpx_codec.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/vpx_timer.h"

// const[expr] should be sufficient for DECLARE_ALIGNED but early
// implementations of c++11 appear to have some issues with it.
#define kDataAlignment 32

template <typename Function>
struct TestParams {
  TestParams(int w, int h, Function f, int bd = -1)
      : width(w), height(h), bit_depth(bd), func(f) {}
  int width, height, bit_depth;
  Function func;
};

using SadMxNFunc = unsigned int (*)(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride);
using SadMxNParam = TestParams<SadMxNFunc>;

using SadSkipMxNFunc = unsigned int (*)(const uint8_t *src_ptr, int src_stride,
                                        const uint8_t *ref_ptr, int ref_stride);
using SadSkipMxNParam = TestParams<SadSkipMxNFunc>;

using SadMxNAvgFunc = unsigned int (*)(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride,
                                       const uint8_t *second_pred);
using SadMxNAvgParam = TestParams<SadMxNAvgFunc>;

using SadMxNx4Func = void (*)(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *const ref_ptr[], int ref_stride,
                              unsigned int *sad_array);
using SadMxNx4Param = TestParams<SadMxNx4Func>;

using SadSkipMxNx4Func = void (*)(const uint8_t *src_ptr, int src_stride,
                                  const uint8_t *const ref_ptr[],
                                  int ref_stride, unsigned int *sad_array);
using SadSkipMxNx4Param = TestParams<SadSkipMxNx4Func>;

using SadMxNx8Func = void (*)(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *ref_ptr, int ref_stride,
                              unsigned int *sad_array);

using libvpx_test::ACMRandom;

namespace {
template <typename ParamType>
class SADTestBase : public ::testing::TestWithParam<ParamType> {
 public:
  explicit SADTestBase(const ParamType &params) : params_(params) {}

  void SetUp() override {
    source_data8_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(kDataAlignment, kDataBlockSize));
    reference_data8_ = reinterpret_cast<uint8_t *>(
        vpx_memalign(kDataAlignment, kDataBufferSize));
    second_pred8_ =
        reinterpret_cast<uint8_t *>(vpx_memalign(kDataAlignment, 64 * 64));
    source_data16_ = reinterpret_cast<uint16_t *>(
        vpx_memalign(kDataAlignment, kDataBlockSize * sizeof(uint16_t)));
    reference_data16_ = reinterpret_cast<uint16_t *>(
        vpx_memalign(kDataAlignment, kDataBufferSize * sizeof(uint16_t)));
    second_pred16_ = reinterpret_cast<uint16_t *>(
        vpx_memalign(kDataAlignment, 64 * 64 * sizeof(uint16_t)));

    if (params_.bit_depth == -1) {
      use_high_bit_depth_ = false;
      bit_depth_ = VPX_BITS_8;
      source_data_ = source_data8_;
      reference_data_ = reference_data8_;
      second_pred_ = second_pred8_;
#if CONFIG_VP9_HIGHBITDEPTH
    } else {
      use_high_bit_depth_ = true;
      bit_depth_ = static_cast<vpx_bit_depth_t>(params_.bit_depth);
      source_data_ = CONVERT_TO_BYTEPTR(source_data16_);
      reference_data_ = CONVERT_TO_BYTEPTR(reference_data16_);
      second_pred_ = CONVERT_TO_BYTEPTR(second_pred16_);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    }
    mask_ = (1 << bit_depth_) - 1;
    source_stride_ = (params_.width + 63) & ~63;
    reference_stride_ = params_.width * 2;
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  void TearDown() override {
    vpx_free(source_data8_);
    source_data8_ = nullptr;
    vpx_free(reference_data8_);
    reference_data8_ = nullptr;
    vpx_free(second_pred8_);
    second_pred8_ = nullptr;
    vpx_free(source_data16_);
    source_data16_ = nullptr;
    vpx_free(reference_data16_);
    reference_data16_ = nullptr;
    vpx_free(second_pred16_);
    second_pred16_ = nullptr;

    libvpx_test::ClearSystemState();
  }

 protected:
  // Handle blocks up to 4 blocks 64x64 with stride up to 128
  // crbug.com/webm/1660
  static const int kDataBlockSize = 64 * 128;
  static const int kDataBufferSize = 4 * kDataBlockSize;

  int GetBlockRefOffset(int block_idx) const {
    return block_idx * kDataBlockSize;
  }

  uint8_t *GetReferenceFromOffset(int ref_offset) const {
    assert((params_.height - 1) * reference_stride_ + params_.width - 1 +
               ref_offset <
           kDataBufferSize);
#if CONFIG_VP9_HIGHBITDEPTH
    if (use_high_bit_depth_) {
      return CONVERT_TO_BYTEPTR(CONVERT_TO_SHORTPTR(reference_data_) +
                                ref_offset);
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    return reference_data_ + ref_offset;
  }

  uint8_t *GetReference(int block_idx) const {
    return GetReferenceFromOffset(GetBlockRefOffset(block_idx));
  }

  // Sum of Absolute Differences. Given two blocks, calculate the absolute
  // difference between two pixels in the same relative location; accumulate.
  uint32_t ReferenceSAD(int ref_offset) const {
    uint32_t sad = 0;
    const uint8_t *const reference8 = GetReferenceFromOffset(ref_offset);
    const uint8_t *const source8 = source_data_;
#if CONFIG_VP9_HIGHBITDEPTH
    const uint16_t *const reference16 =
        CONVERT_TO_SHORTPTR(GetReferenceFromOffset(ref_offset));
    const uint16_t *const source16 = CONVERT_TO_SHORTPTR(source_data_);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    for (int h = 0; h < params_.height; ++h) {
      for (int w = 0; w < params_.width; ++w) {
        if (!use_high_bit_depth_) {
          sad += abs(source8[h * source_stride_ + w] -
                     reference8[h * reference_stride_ + w]);
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          sad += abs(source16[h * source_stride_ + w] -
                     reference16[h * reference_stride_ + w]);
#endif  // CONFIG_VP9_HIGHBITDEPTH
        }
      }
    }
    return sad;
  }

  // Sum of Absolute Differences Skip rows. Given two blocks, calculate the
  // absolute difference between two pixels in the same relative location every
  // other row; accumulate and double the result at the end.
  uint32_t ReferenceSADSkip(int ref_offset) const {
    uint32_t sad = 0;
    const uint8_t *const reference8 = GetReferenceFromOffset(ref_offset);
    const uint8_t *const source8 = source_data_;
#if CONFIG_VP9_HIGHBITDEPTH
    const uint16_t *const reference16 =
        CONVERT_TO_SHORTPTR(GetReferenceFromOffset(ref_offset));
    const uint16_t *const source16 = CONVERT_TO_SHORTPTR(source_data_);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    for (int h = 0; h < params_.height; h += 2) {
      for (int w = 0; w < params_.width; ++w) {
        if (!use_high_bit_depth_) {
          sad += abs(source8[h * source_stride_ + w] -
                     reference8[h * reference_stride_ + w]);
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          sad += abs(source16[h * source_stride_ + w] -
                     reference16[h * reference_stride_ + w]);
#endif  // CONFIG_VP9_HIGHBITDEPTH
        }
      }
    }
    return sad * 2;
  }

  // Sum of Absolute Differences Average. Given two blocks, and a prediction
  // calculate the absolute difference between one pixel and average of the
  // corresponding and predicted pixels; accumulate.
  unsigned int ReferenceSADavg(int block_idx) const {
    unsigned int sad = 0;
    const uint8_t *const reference8 = GetReference(block_idx);
    const uint8_t *const source8 = source_data_;
    const uint8_t *const second_pred8 = second_pred_;
#if CONFIG_VP9_HIGHBITDEPTH
    const uint16_t *const reference16 =
        CONVERT_TO_SHORTPTR(GetReference(block_idx));
    const uint16_t *const source16 = CONVERT_TO_SHORTPTR(source_data_);
    const uint16_t *const second_pred16 = CONVERT_TO_SHORTPTR(second_pred_);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    for (int h = 0; h < params_.height; ++h) {
      for (int w = 0; w < params_.width; ++w) {
        if (!use_high_bit_depth_) {
          const int tmp = second_pred8[h * params_.width + w] +
                          reference8[h * reference_stride_ + w];
          const uint8_t comp_pred = ROUND_POWER_OF_TWO(tmp, 1);
          sad += abs(source8[h * source_stride_ + w] - comp_pred);
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          const int tmp = second_pred16[h * params_.width + w] +
                          reference16[h * reference_stride_ + w];
          const uint16_t comp_pred = ROUND_POWER_OF_TWO(tmp, 1);
          sad += abs(source16[h * source_stride_ + w] - comp_pred);
#endif  // CONFIG_VP9_HIGHBITDEPTH
        }
      }
    }
    return sad;
  }

  void FillConstant(uint8_t *data, int stride, uint16_t fill_constant) const {
    uint8_t *data8 = data;
#if CONFIG_VP9_HIGHBITDEPTH
    uint16_t *data16 = CONVERT_TO_SHORTPTR(data);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    for (int h = 0; h < params_.height; ++h) {
      for (int w = 0; w < params_.width; ++w) {
        if (!use_high_bit_depth_) {
          data8[h * stride + w] = static_cast<uint8_t>(fill_constant);
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          data16[h * stride + w] = fill_constant;
#endif  // CONFIG_VP9_HIGHBITDEPTH
        }
      }
    }
  }

  void FillRandomWH(uint8_t *data, int stride, int w, int h) {
    uint8_t *data8 = data;
#if CONFIG_VP9_HIGHBITDEPTH
    uint16_t *data16 = CONVERT_TO_SHORTPTR(data);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        if (!use_high_bit_depth_) {
          data8[r * stride + c] = rnd_.Rand8();
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          data16[r * stride + c] = rnd_.Rand16() & mask_;
#endif  // CONFIG_VP9_HIGHBITDEPTH
        }
      }
    }
  }

  void FillRandom(uint8_t *data, int stride) {
    FillRandomWH(data, stride, params_.width, params_.height);
  }

  uint32_t mask_;
  vpx_bit_depth_t bit_depth_;
  int source_stride_;
  int reference_stride_;
  bool use_high_bit_depth_;

  uint8_t *source_data_;
  uint8_t *reference_data_;
  uint8_t *second_pred_;
  uint8_t *source_data8_;
  uint8_t *reference_data8_;
  uint8_t *second_pred8_;
  uint16_t *source_data16_;
  uint16_t *reference_data16_;
  uint16_t *second_pred16_;

  ACMRandom rnd_;
  ParamType params_;
};

class SADx4Test : public SADTestBase<SadMxNx4Param> {
 public:
  SADx4Test() : SADTestBase(GetParam()) {}

 protected:
  void SADs(unsigned int *results) const {
    const uint8_t *references[] = { GetReference(0), GetReference(1),
                                    GetReference(2), GetReference(3) };

    ASM_REGISTER_STATE_CHECK(params_.func(
        source_data_, source_stride_, references, reference_stride_, results));
  }

  void CheckSADs() const {
    uint32_t reference_sad;
    DECLARE_ALIGNED(kDataAlignment, uint32_t, exp_sad[4]);

    SADs(exp_sad);
    for (int block = 0; block < 4; ++block) {
      reference_sad = ReferenceSAD(GetBlockRefOffset(block));

      EXPECT_EQ(reference_sad, exp_sad[block]) << "block " << block;
    }
  }
};

class SADSkipx4Test : public SADTestBase<SadMxNx4Param> {
 public:
  SADSkipx4Test() : SADTestBase(GetParam()) {}

 protected:
  void SADs(unsigned int *results) const {
    const uint8_t *references[] = { GetReference(0), GetReference(1),
                                    GetReference(2), GetReference(3) };

    ASM_REGISTER_STATE_CHECK(params_.func(
        source_data_, source_stride_, references, reference_stride_, results));
  }

  void CheckSADs() const {
    uint32_t reference_sad;
    DECLARE_ALIGNED(kDataAlignment, uint32_t, exp_sad[4]);

    SADs(exp_sad);
    for (int block = 0; block < 4; ++block) {
      reference_sad = ReferenceSADSkip(GetBlockRefOffset(block));

      EXPECT_EQ(reference_sad, exp_sad[block]) << "block " << block;
    }
  }
};

class SADTest : public AbstractBench, public SADTestBase<SadMxNParam> {
 public:
  SADTest() : SADTestBase(GetParam()) {}

 protected:
  unsigned int SAD(int block_idx) const {
    unsigned int ret;
    const uint8_t *const reference = GetReference(block_idx);

    ASM_REGISTER_STATE_CHECK(ret = params_.func(source_data_, source_stride_,
                                                reference, reference_stride_));
    return ret;
  }

  void CheckSAD() const {
    const unsigned int reference_sad = ReferenceSAD(GetBlockRefOffset(0));
    const unsigned int exp_sad = SAD(0);

    ASSERT_EQ(reference_sad, exp_sad);
  }

  void Run() override {
    params_.func(source_data_, source_stride_, reference_data_,
                 reference_stride_);
  }
};

class SADSkipTest : public AbstractBench, public SADTestBase<SadMxNParam> {
 public:
  SADSkipTest() : SADTestBase(GetParam()) {}

 protected:
  unsigned int SAD(int block_idx) const {
    unsigned int ret;
    const uint8_t *const reference = GetReference(block_idx);

    ASM_REGISTER_STATE_CHECK(ret = params_.func(source_data_, source_stride_,
                                                reference, reference_stride_));
    return ret;
  }

  void CheckSAD() const {
    const unsigned int reference_sad = ReferenceSADSkip(GetBlockRefOffset(0));
    const unsigned int exp_sad = SAD(0);

    ASSERT_EQ(reference_sad, exp_sad);
  }

  void Run() override {
    params_.func(source_data_, source_stride_, reference_data_,
                 reference_stride_);
  }
};

class SADavgTest : public AbstractBench, public SADTestBase<SadMxNAvgParam> {
 public:
  SADavgTest() : SADTestBase(GetParam()) {}

 protected:
  unsigned int SAD_avg(int block_idx) const {
    unsigned int ret;
    const uint8_t *const reference = GetReference(block_idx);

    ASM_REGISTER_STATE_CHECK(ret = params_.func(source_data_, source_stride_,
                                                reference, reference_stride_,
                                                second_pred_));
    return ret;
  }

  void CheckSAD() const {
    const unsigned int reference_sad = ReferenceSADavg(0);
    const unsigned int exp_sad = SAD_avg(0);

    ASSERT_EQ(reference_sad, exp_sad);
  }

  void Run() override {
    params_.func(source_data_, source_stride_, reference_data_,
                 reference_stride_, second_pred_);
  }
};

TEST_P(SADTest, MaxRef) {
  FillConstant(source_data_, source_stride_, 0);
  FillConstant(reference_data_, reference_stride_, mask_);
  CheckSAD();
}

TEST_P(SADTest, MaxSrc) {
  FillConstant(source_data_, source_stride_, mask_);
  FillConstant(reference_data_, reference_stride_, 0);
  CheckSAD();
}

TEST_P(SADTest, ShortRef) {
  const int tmp_stride = reference_stride_;
  reference_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  CheckSAD();
  reference_stride_ = tmp_stride;
}

TEST_P(SADTest, UnalignedRef) {
  // The reference frame, but not the source frame, may be unaligned for
  // certain types of searches.
  const int tmp_stride = reference_stride_;
  reference_stride_ -= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  CheckSAD();
  reference_stride_ = tmp_stride;
}

TEST_P(SADTest, ShortSrc) {
  const int tmp_stride = source_stride_;
  source_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  CheckSAD();
  source_stride_ = tmp_stride;
}

TEST_P(SADTest, DISABLED_Speed) {
  const int kCountSpeedTestBlock = 50000000 / (params_.width * params_.height);
  FillRandom(source_data_, source_stride_);

  RunNTimes(kCountSpeedTestBlock);

  char title[16];
  snprintf(title, sizeof(title), "%dx%d", params_.width, params_.height);
  PrintMedian(title);
}

TEST_P(SADSkipTest, MaxRef) {
  FillConstant(source_data_, source_stride_, 0);
  FillConstant(reference_data_, reference_stride_, mask_);
  CheckSAD();
}

TEST_P(SADSkipTest, MaxSrc) {
  FillConstant(source_data_, source_stride_, mask_);
  FillConstant(reference_data_, reference_stride_, 0);
  CheckSAD();
}

TEST_P(SADSkipTest, ShortRef) {
  const int tmp_stride = reference_stride_;
  reference_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  CheckSAD();
  reference_stride_ = tmp_stride;
}

TEST_P(SADSkipTest, UnalignedRef) {
  // The reference frame, but not the source frame, may be unaligned for
  // certain types of searches.
  const int tmp_stride = reference_stride_;
  reference_stride_ -= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  CheckSAD();
  reference_stride_ = tmp_stride;
}

TEST_P(SADSkipTest, ShortSrc) {
  const int tmp_stride = source_stride_;
  source_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  CheckSAD();
  source_stride_ = tmp_stride;
}

TEST_P(SADSkipTest, DISABLED_Speed) {
  const int kCountSpeedTestBlock = 50000000 / (params_.width * params_.height);
  FillRandom(source_data_, source_stride_);

  RunNTimes(kCountSpeedTestBlock);

  char title[16];
  snprintf(title, sizeof(title), "%dx%d", params_.width, params_.height);
  PrintMedian(title);
}

TEST_P(SADavgTest, MaxRef) {
  FillConstant(source_data_, source_stride_, 0);
  FillConstant(reference_data_, reference_stride_, mask_);
  FillConstant(second_pred_, params_.width, 0);
  CheckSAD();
}
TEST_P(SADavgTest, MaxSrc) {
  FillConstant(source_data_, source_stride_, mask_);
  FillConstant(reference_data_, reference_stride_, 0);
  FillConstant(second_pred_, params_.width, 0);
  CheckSAD();
}

TEST_P(SADavgTest, ShortRef) {
  const int tmp_stride = reference_stride_;
  reference_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  FillRandom(second_pred_, params_.width);
  CheckSAD();
  reference_stride_ = tmp_stride;
}

TEST_P(SADavgTest, UnalignedRef) {
  // The reference frame, but not the source frame, may be unaligned for
  // certain types of searches.
  const int tmp_stride = reference_stride_;
  reference_stride_ -= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  FillRandom(second_pred_, params_.width);
  CheckSAD();
  reference_stride_ = tmp_stride;
}

TEST_P(SADavgTest, ShortSrc) {
  const int tmp_stride = source_stride_;
  source_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  FillRandom(second_pred_, params_.width);
  CheckSAD();
  source_stride_ = tmp_stride;
}

TEST_P(SADavgTest, DISABLED_Speed) {
  const int kCountSpeedTestBlock = 50000000 / (params_.width * params_.height);
  FillRandom(source_data_, source_stride_);
  FillRandom(reference_data_, reference_stride_);
  FillRandom(second_pred_, params_.width);

  RunNTimes(kCountSpeedTestBlock);

  char title[16];
  snprintf(title, sizeof(title), "%dx%d", params_.width, params_.height);
  PrintMedian(title);
}

TEST_P(SADx4Test, MaxRef) {
  FillConstant(source_data_, source_stride_, 0);
  FillConstant(GetReference(0), reference_stride_, mask_);
  FillConstant(GetReference(1), reference_stride_, mask_);
  FillConstant(GetReference(2), reference_stride_, mask_);
  FillConstant(GetReference(3), reference_stride_, mask_);
  CheckSADs();
}

TEST_P(SADx4Test, MaxSrc) {
  FillConstant(source_data_, source_stride_, mask_);
  FillConstant(GetReference(0), reference_stride_, 0);
  FillConstant(GetReference(1), reference_stride_, 0);
  FillConstant(GetReference(2), reference_stride_, 0);
  FillConstant(GetReference(3), reference_stride_, 0);
  CheckSADs();
}

TEST_P(SADx4Test, ShortRef) {
  int tmp_stride = reference_stride_;
  reference_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  CheckSADs();
  reference_stride_ = tmp_stride;
}

TEST_P(SADx4Test, UnalignedRef) {
  // The reference frame, but not the source frame, may be unaligned for
  // certain types of searches.
  int tmp_stride = reference_stride_;
  reference_stride_ -= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  CheckSADs();
  reference_stride_ = tmp_stride;
}

TEST_P(SADx4Test, ShortSrc) {
  int tmp_stride = source_stride_;
  source_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  CheckSADs();
  source_stride_ = tmp_stride;
}

TEST_P(SADx4Test, SrcAlignedByWidth) {
  uint8_t *tmp_source_data = source_data_;
  source_data_ += params_.width;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  CheckSADs();
  source_data_ = tmp_source_data;
}

TEST_P(SADx4Test, DISABLED_Speed) {
  int tmp_stride = reference_stride_;
  reference_stride_ -= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  const int kCountSpeedTestBlock = 500000000 / (params_.width * params_.height);
  uint32_t reference_sad[4];
  DECLARE_ALIGNED(kDataAlignment, uint32_t, exp_sad[4]);
  vpx_usec_timer timer;
  for (int block = 0; block < 4; ++block) {
    reference_sad[block] = ReferenceSAD(GetBlockRefOffset(block));
  }
  vpx_usec_timer_start(&timer);
  for (int i = 0; i < kCountSpeedTestBlock; ++i) {
    SADs(exp_sad);
  }
  vpx_usec_timer_mark(&timer);
  for (int block = 0; block < 4; ++block) {
    EXPECT_EQ(reference_sad[block], exp_sad[block]) << "block " << block;
  }
  const int elapsed_time =
      static_cast<int>(vpx_usec_timer_elapsed(&timer) / 1000);
  printf("sad%dx%dx4 (%2dbit) time: %5d ms\n", params_.width, params_.height,
         bit_depth_, elapsed_time);

  reference_stride_ = tmp_stride;
}

TEST_P(SADSkipx4Test, MaxRef) {
  FillConstant(source_data_, source_stride_, 0);
  FillConstant(GetReference(0), reference_stride_, mask_);
  FillConstant(GetReference(1), reference_stride_, mask_);
  FillConstant(GetReference(2), reference_stride_, mask_);
  FillConstant(GetReference(3), reference_stride_, mask_);
  CheckSADs();
}

TEST_P(SADSkipx4Test, MaxSrc) {
  FillConstant(source_data_, source_stride_, mask_);
  FillConstant(GetReference(0), reference_stride_, 0);
  FillConstant(GetReference(1), reference_stride_, 0);
  FillConstant(GetReference(2), reference_stride_, 0);
  FillConstant(GetReference(3), reference_stride_, 0);
  CheckSADs();
}

TEST_P(SADSkipx4Test, ShortRef) {
  int tmp_stride = reference_stride_;
  reference_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  CheckSADs();
  reference_stride_ = tmp_stride;
}

TEST_P(SADSkipx4Test, UnalignedRef) {
  // The reference frame, but not the source frame, may be unaligned for
  // certain types of searches.
  int tmp_stride = reference_stride_;
  reference_stride_ -= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  CheckSADs();
  reference_stride_ = tmp_stride;
}

TEST_P(SADSkipx4Test, ShortSrc) {
  int tmp_stride = source_stride_;
  source_stride_ >>= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  CheckSADs();
  source_stride_ = tmp_stride;
}

TEST_P(SADSkipx4Test, SrcAlignedByWidth) {
  uint8_t *tmp_source_data = source_data_;
  source_data_ += params_.width;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  CheckSADs();
  source_data_ = tmp_source_data;
}

TEST_P(SADSkipx4Test, DISABLED_Speed) {
  int tmp_stride = reference_stride_;
  reference_stride_ -= 1;
  FillRandom(source_data_, source_stride_);
  FillRandom(GetReference(0), reference_stride_);
  FillRandom(GetReference(1), reference_stride_);
  FillRandom(GetReference(2), reference_stride_);
  FillRandom(GetReference(3), reference_stride_);
  const int kCountSpeedTestBlock = 500000000 / (params_.width * params_.height);
  uint32_t reference_sad[4];
  DECLARE_ALIGNED(kDataAlignment, uint32_t, exp_sad[4]);
  vpx_usec_timer timer;
  for (int block = 0; block < 4; ++block) {
    reference_sad[block] = ReferenceSADSkip(GetBlockRefOffset(block));
  }
  vpx_usec_timer_start(&timer);
  for (int i = 0; i < kCountSpeedTestBlock; ++i) {
    SADs(exp_sad);
  }
  vpx_usec_timer_mark(&timer);
  for (int block = 0; block < 4; ++block) {
    EXPECT_EQ(reference_sad[block], exp_sad[block]) << "block " << block;
  }
  const int elapsed_time =
      static_cast<int>(vpx_usec_timer_elapsed(&timer) / 1000);
  printf("sad%dx%dx4 (%2dbit) time: %5d ms\n", params_.width, params_.height,
         bit_depth_, elapsed_time);

  reference_stride_ = tmp_stride;
}

//------------------------------------------------------------------------------
// C functions
const SadMxNParam c_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_c),
  SadMxNParam(64, 32, &vpx_sad64x32_c),
  SadMxNParam(32, 64, &vpx_sad32x64_c),
  SadMxNParam(32, 32, &vpx_sad32x32_c),
  SadMxNParam(32, 16, &vpx_sad32x16_c),
  SadMxNParam(16, 32, &vpx_sad16x32_c),
  SadMxNParam(16, 16, &vpx_sad16x16_c),
  SadMxNParam(16, 8, &vpx_sad16x8_c),
  SadMxNParam(8, 16, &vpx_sad8x16_c),
  SadMxNParam(8, 8, &vpx_sad8x8_c),
  SadMxNParam(8, 4, &vpx_sad8x4_c),
  SadMxNParam(4, 8, &vpx_sad4x8_c),
  SadMxNParam(4, 4, &vpx_sad4x4_c),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_c, 8),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_c, 8),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_c, 8),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_c, 8),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_c, 8),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_c, 8),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_c, 8),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_c, 8),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_c, 8),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_c, 8),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_c, 8),
  SadMxNParam(4, 8, &vpx_highbd_sad4x8_c, 8),
  SadMxNParam(4, 4, &vpx_highbd_sad4x4_c, 8),
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_c, 10),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_c, 10),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_c, 10),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_c, 10),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_c, 10),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_c, 10),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_c, 10),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_c, 10),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_c, 10),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_c, 10),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_c, 10),
  SadMxNParam(4, 8, &vpx_highbd_sad4x8_c, 10),
  SadMxNParam(4, 4, &vpx_highbd_sad4x4_c, 10),
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_c, 12),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_c, 12),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_c, 12),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_c, 12),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_c, 12),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_c, 12),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_c, 12),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_c, 12),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_c, 12),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_c, 12),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_c, 12),
  SadMxNParam(4, 8, &vpx_highbd_sad4x8_c, 12),
  SadMxNParam(4, 4, &vpx_highbd_sad4x4_c, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(C, SADTest, ::testing::ValuesIn(c_tests));

const SadSkipMxNParam skip_c_tests[] = {
  SadSkipMxNParam(64, 64, &vpx_sad_skip_64x64_c),
  SadSkipMxNParam(64, 32, &vpx_sad_skip_64x32_c),
  SadSkipMxNParam(32, 64, &vpx_sad_skip_32x64_c),
  SadSkipMxNParam(32, 32, &vpx_sad_skip_32x32_c),
  SadSkipMxNParam(32, 16, &vpx_sad_skip_32x16_c),
  SadSkipMxNParam(16, 32, &vpx_sad_skip_16x32_c),
  SadSkipMxNParam(16, 16, &vpx_sad_skip_16x16_c),
  SadSkipMxNParam(16, 8, &vpx_sad_skip_16x8_c),
  SadSkipMxNParam(8, 16, &vpx_sad_skip_8x16_c),
  SadSkipMxNParam(8, 8, &vpx_sad_skip_8x8_c),
  SadSkipMxNParam(4, 8, &vpx_sad_skip_4x8_c),
#if CONFIG_VP9_HIGHBITDEPTH
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_c, 8),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_c, 8),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_c, 8),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_c, 8),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_c, 8),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_c, 8),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_c, 8),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_c, 8),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_c, 8),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_c, 8),
  SadSkipMxNParam(4, 8, &vpx_highbd_sad_skip_4x8_c, 8),
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_c, 10),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_c, 10),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_c, 10),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_c, 10),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_c, 10),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_c, 10),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_c, 10),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_c, 10),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_c, 10),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_c, 10),
  SadSkipMxNParam(4, 8, &vpx_highbd_sad_skip_4x8_c, 10),
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_c, 12),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_c, 12),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_c, 12),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_c, 12),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_c, 12),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_c, 12),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_c, 12),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_c, 12),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_c, 12),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_c, 12),
  SadSkipMxNParam(4, 8, &vpx_highbd_sad_skip_4x8_c, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(C, SADSkipTest, ::testing::ValuesIn(skip_c_tests));

const SadMxNAvgParam avg_c_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_c),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_c),
  SadMxNAvgParam(32, 64, &vpx_sad32x64_avg_c),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_c),
  SadMxNAvgParam(32, 16, &vpx_sad32x16_avg_c),
  SadMxNAvgParam(16, 32, &vpx_sad16x32_avg_c),
  SadMxNAvgParam(16, 16, &vpx_sad16x16_avg_c),
  SadMxNAvgParam(16, 8, &vpx_sad16x8_avg_c),
  SadMxNAvgParam(8, 16, &vpx_sad8x16_avg_c),
  SadMxNAvgParam(8, 8, &vpx_sad8x8_avg_c),
  SadMxNAvgParam(8, 4, &vpx_sad8x4_avg_c),
  SadMxNAvgParam(4, 8, &vpx_sad4x8_avg_c),
  SadMxNAvgParam(4, 4, &vpx_sad4x4_avg_c),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_c, 8),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_c, 8),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_c, 8),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_c, 8),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_c, 8),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_c, 8),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_c, 8),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_c, 8),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_c, 8),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_c, 8),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_c, 8),
  SadMxNAvgParam(4, 8, &vpx_highbd_sad4x8_avg_c, 8),
  SadMxNAvgParam(4, 4, &vpx_highbd_sad4x4_avg_c, 8),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_c, 10),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_c, 10),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_c, 10),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_c, 10),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_c, 10),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_c, 10),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_c, 10),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_c, 10),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_c, 10),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_c, 10),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_c, 10),
  SadMxNAvgParam(4, 8, &vpx_highbd_sad4x8_avg_c, 10),
  SadMxNAvgParam(4, 4, &vpx_highbd_sad4x4_avg_c, 10),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_c, 12),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_c, 12),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_c, 12),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_c, 12),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_c, 12),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_c, 12),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_c, 12),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_c, 12),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_c, 12),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_c, 12),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_c, 12),
  SadMxNAvgParam(4, 8, &vpx_highbd_sad4x8_avg_c, 12),
  SadMxNAvgParam(4, 4, &vpx_highbd_sad4x4_avg_c, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(C, SADavgTest, ::testing::ValuesIn(avg_c_tests));

const SadMxNx4Param x4d_c_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_c),
  SadMxNx4Param(64, 32, &vpx_sad64x32x4d_c),
  SadMxNx4Param(32, 64, &vpx_sad32x64x4d_c),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_c),
  SadMxNx4Param(32, 16, &vpx_sad32x16x4d_c),
  SadMxNx4Param(16, 32, &vpx_sad16x32x4d_c),
  SadMxNx4Param(16, 16, &vpx_sad16x16x4d_c),
  SadMxNx4Param(16, 8, &vpx_sad16x8x4d_c),
  SadMxNx4Param(8, 16, &vpx_sad8x16x4d_c),
  SadMxNx4Param(8, 8, &vpx_sad8x8x4d_c),
  SadMxNx4Param(8, 4, &vpx_sad8x4x4d_c),
  SadMxNx4Param(4, 8, &vpx_sad4x8x4d_c),
  SadMxNx4Param(4, 4, &vpx_sad4x4x4d_c),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_c, 8),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_c, 8),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_c, 8),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_c, 8),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_c, 8),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_c, 8),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_c, 8),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_c, 8),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_c, 8),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_c, 8),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_c, 8),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_c, 8),
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_c, 8),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_c, 10),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_c, 10),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_c, 10),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_c, 10),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_c, 10),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_c, 10),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_c, 10),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_c, 10),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_c, 10),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_c, 10),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_c, 10),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_c, 10),
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_c, 10),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_c, 12),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_c, 12),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_c, 12),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_c, 12),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_c, 12),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_c, 12),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_c, 12),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_c, 12),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_c, 12),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_c, 12),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_c, 12),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_c, 12),
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_c, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(C, SADx4Test, ::testing::ValuesIn(x4d_c_tests));

const SadSkipMxNx4Param skip_x4d_c_tests[] = {
  SadSkipMxNx4Param(64, 64, &vpx_sad_skip_64x64x4d_c),
  SadSkipMxNx4Param(64, 32, &vpx_sad_skip_64x32x4d_c),
  SadSkipMxNx4Param(32, 64, &vpx_sad_skip_32x64x4d_c),
  SadSkipMxNx4Param(32, 32, &vpx_sad_skip_32x32x4d_c),
  SadSkipMxNx4Param(32, 16, &vpx_sad_skip_32x16x4d_c),
  SadSkipMxNx4Param(16, 32, &vpx_sad_skip_16x32x4d_c),
  SadSkipMxNx4Param(16, 16, &vpx_sad_skip_16x16x4d_c),
  SadSkipMxNx4Param(16, 8, &vpx_sad_skip_16x8x4d_c),
  SadSkipMxNx4Param(8, 16, &vpx_sad_skip_8x16x4d_c),
  SadSkipMxNx4Param(8, 8, &vpx_sad_skip_8x8x4d_c),
  SadSkipMxNx4Param(4, 8, &vpx_sad_skip_4x8x4d_c),
#if CONFIG_VP9_HIGHBITDEPTH
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_c, 8),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_c, 8),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_c, 8),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_c, 8),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_c, 8),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_c, 8),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_c, 8),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_c, 8),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_c, 8),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_c, 8),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_c, 8),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_c, 10),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_c, 10),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_c, 10),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_c, 10),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_c, 10),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_c, 10),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_c, 10),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_c, 10),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_c, 10),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_c, 10),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_c, 10),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_c, 12),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_c, 12),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_c, 12),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_c, 12),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_c, 12),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_c, 12),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_c, 12),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_c, 12),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_c, 12),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_c, 12),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_c, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(C, SADSkipx4Test,
                         ::testing::ValuesIn(skip_x4d_c_tests));

//------------------------------------------------------------------------------
// ARM functions
#if HAVE_NEON
const SadMxNParam neon_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_neon),
  SadMxNParam(64, 32, &vpx_sad64x32_neon),
  SadMxNParam(32, 32, &vpx_sad32x32_neon),
  SadMxNParam(16, 32, &vpx_sad16x32_neon),
  SadMxNParam(16, 16, &vpx_sad16x16_neon),
  SadMxNParam(16, 8, &vpx_sad16x8_neon),
  SadMxNParam(8, 16, &vpx_sad8x16_neon),
  SadMxNParam(8, 8, &vpx_sad8x8_neon),
  SadMxNParam(8, 4, &vpx_sad8x4_neon),
  SadMxNParam(4, 8, &vpx_sad4x8_neon),
  SadMxNParam(4, 4, &vpx_sad4x4_neon),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNParam(4, 4, &vpx_highbd_sad4x4_neon, 8),
  SadMxNParam(4, 8, &vpx_highbd_sad4x8_neon, 8),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_neon, 8),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_neon, 8),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_neon, 8),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_neon, 8),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_neon, 8),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_neon, 8),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_neon, 8),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_neon, 8),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_neon, 8),
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_neon, 8),
  SadMxNParam(4, 4, &vpx_highbd_sad4x4_neon, 10),
  SadMxNParam(4, 8, &vpx_highbd_sad4x8_neon, 10),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_neon, 10),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_neon, 10),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_neon, 10),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_neon, 10),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_neon, 10),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_neon, 10),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_neon, 10),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_neon, 10),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_neon, 10),
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_neon, 10),
  SadMxNParam(4, 4, &vpx_highbd_sad4x4_neon, 12),
  SadMxNParam(4, 8, &vpx_highbd_sad4x8_neon, 12),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_neon, 12),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_neon, 12),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_neon, 12),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_neon, 12),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_neon, 12),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_neon, 12),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_neon, 12),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_neon, 12),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_neon, 12),
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_neon, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH

};
INSTANTIATE_TEST_SUITE_P(NEON, SADTest, ::testing::ValuesIn(neon_tests));

#if HAVE_NEON_DOTPROD
const SadMxNParam neon_dotprod_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_neon_dotprod),
  SadMxNParam(64, 32, &vpx_sad64x32_neon_dotprod),
  SadMxNParam(32, 64, &vpx_sad32x64_neon_dotprod),
  SadMxNParam(32, 32, &vpx_sad32x32_neon_dotprod),
  SadMxNParam(32, 16, &vpx_sad32x16_neon_dotprod),
  SadMxNParam(16, 32, &vpx_sad16x32_neon_dotprod),
  SadMxNParam(16, 16, &vpx_sad16x16_neon_dotprod),
  SadMxNParam(16, 8, &vpx_sad16x8_neon_dotprod),
};
INSTANTIATE_TEST_SUITE_P(NEON_DOTPROD, SADTest,
                         ::testing::ValuesIn(neon_dotprod_tests));
#endif  // HAVE_NEON_DOTPROD

const SadSkipMxNParam skip_neon_tests[] = {
  SadSkipMxNParam(64, 64, &vpx_sad_skip_64x64_neon),
  SadSkipMxNParam(64, 32, &vpx_sad_skip_64x32_neon),
  SadSkipMxNParam(32, 64, &vpx_sad_skip_32x64_neon),
  SadSkipMxNParam(32, 32, &vpx_sad_skip_32x32_neon),
  SadSkipMxNParam(32, 16, &vpx_sad_skip_32x16_neon),
  SadSkipMxNParam(16, 32, &vpx_sad_skip_16x32_neon),
  SadSkipMxNParam(16, 16, &vpx_sad_skip_16x16_neon),
  SadSkipMxNParam(16, 8, &vpx_sad_skip_16x8_neon),
  SadSkipMxNParam(8, 16, &vpx_sad_skip_8x16_neon),
  SadSkipMxNParam(8, 8, &vpx_sad_skip_8x8_neon),
  SadSkipMxNParam(8, 4, &vpx_sad_skip_8x4_neon),
  SadSkipMxNParam(4, 8, &vpx_sad_skip_4x8_neon),
  SadSkipMxNParam(4, 4, &vpx_sad_skip_4x4_neon),
#if CONFIG_VP9_HIGHBITDEPTH
  SadSkipMxNParam(4, 4, &vpx_highbd_sad_skip_4x4_neon, 8),
  SadSkipMxNParam(4, 8, &vpx_highbd_sad_skip_4x8_neon, 8),
  SadSkipMxNParam(8, 4, &vpx_highbd_sad_skip_8x4_neon, 8),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_neon, 8),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_neon, 8),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_neon, 8),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_neon, 8),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_neon, 8),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_neon, 8),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_neon, 8),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_neon, 8),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_neon, 8),
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_neon, 8),
  SadSkipMxNParam(4, 4, &vpx_highbd_sad_skip_4x4_neon, 10),
  SadSkipMxNParam(4, 8, &vpx_highbd_sad_skip_4x8_neon, 10),
  SadSkipMxNParam(8, 4, &vpx_highbd_sad_skip_8x4_neon, 10),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_neon, 10),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_neon, 10),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_neon, 10),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_neon, 10),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_neon, 10),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_neon, 10),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_neon, 10),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_neon, 10),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_neon, 10),
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_neon, 10),
  SadSkipMxNParam(4, 4, &vpx_highbd_sad_skip_4x4_neon, 12),
  SadSkipMxNParam(4, 8, &vpx_highbd_sad_skip_4x8_neon, 12),
  SadSkipMxNParam(8, 4, &vpx_highbd_sad_skip_8x4_neon, 12),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_neon, 12),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_neon, 12),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_neon, 12),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_neon, 12),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_neon, 12),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_neon, 12),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_neon, 12),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_neon, 12),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_neon, 12),
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_neon, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(NEON, SADSkipTest,
                         ::testing::ValuesIn(skip_neon_tests));

#if HAVE_NEON_DOTPROD
const SadSkipMxNParam skip_neon_dotprod_tests[] = {
  SadSkipMxNParam(64, 64, &vpx_sad_skip_64x64_neon_dotprod),
  SadSkipMxNParam(64, 32, &vpx_sad_skip_64x32_neon_dotprod),
  SadSkipMxNParam(32, 64, &vpx_sad_skip_32x64_neon_dotprod),
  SadSkipMxNParam(32, 32, &vpx_sad_skip_32x32_neon_dotprod),
  SadSkipMxNParam(32, 16, &vpx_sad_skip_32x16_neon_dotprod),
  SadSkipMxNParam(16, 32, &vpx_sad_skip_16x32_neon_dotprod),
  SadSkipMxNParam(16, 16, &vpx_sad_skip_16x16_neon_dotprod),
  SadSkipMxNParam(16, 8, &vpx_sad_skip_16x8_neon_dotprod),
};
INSTANTIATE_TEST_SUITE_P(NEON_DOTPROD, SADSkipTest,
                         ::testing::ValuesIn(skip_neon_dotprod_tests));
#endif  // HAVE_NEON_DOTPROD

const SadMxNAvgParam avg_neon_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_neon),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_neon),
  SadMxNAvgParam(32, 64, &vpx_sad32x64_avg_neon),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_neon),
  SadMxNAvgParam(32, 16, &vpx_sad32x16_avg_neon),
  SadMxNAvgParam(16, 32, &vpx_sad16x32_avg_neon),
  SadMxNAvgParam(16, 16, &vpx_sad16x16_avg_neon),
  SadMxNAvgParam(16, 8, &vpx_sad16x8_avg_neon),
  SadMxNAvgParam(8, 16, &vpx_sad8x16_avg_neon),
  SadMxNAvgParam(8, 8, &vpx_sad8x8_avg_neon),
  SadMxNAvgParam(8, 4, &vpx_sad8x4_avg_neon),
  SadMxNAvgParam(4, 8, &vpx_sad4x8_avg_neon),
  SadMxNAvgParam(4, 4, &vpx_sad4x4_avg_neon),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNAvgParam(4, 4, &vpx_highbd_sad4x4_avg_neon, 8),
  SadMxNAvgParam(4, 8, &vpx_highbd_sad4x8_avg_neon, 8),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_neon, 8),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_neon, 8),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_neon, 8),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_neon, 8),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_neon, 8),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_neon, 8),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_neon, 8),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_neon, 8),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_neon, 8),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_neon, 8),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_neon, 8),
  SadMxNAvgParam(4, 4, &vpx_highbd_sad4x4_avg_neon, 10),
  SadMxNAvgParam(4, 8, &vpx_highbd_sad4x8_avg_neon, 10),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_neon, 10),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_neon, 10),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_neon, 10),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_neon, 10),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_neon, 10),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_neon, 10),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_neon, 10),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_neon, 10),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_neon, 10),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_neon, 10),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_neon, 10),
  SadMxNAvgParam(4, 4, &vpx_highbd_sad4x4_avg_neon, 12),
  SadMxNAvgParam(4, 8, &vpx_highbd_sad4x8_avg_neon, 12),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_neon, 12),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_neon, 12),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_neon, 12),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_neon, 12),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_neon, 12),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_neon, 12),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_neon, 12),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_neon, 12),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_neon, 12),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_neon, 12),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_neon, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(NEON, SADavgTest, ::testing::ValuesIn(avg_neon_tests));

#if HAVE_NEON_DOTPROD
const SadMxNAvgParam avg_neon_dotprod_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_neon_dotprod),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_neon_dotprod),
  SadMxNAvgParam(32, 64, &vpx_sad32x64_avg_neon_dotprod),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_neon_dotprod),
  SadMxNAvgParam(32, 16, &vpx_sad32x16_avg_neon_dotprod),
  SadMxNAvgParam(16, 32, &vpx_sad16x32_avg_neon_dotprod),
  SadMxNAvgParam(16, 16, &vpx_sad16x16_avg_neon_dotprod),
  SadMxNAvgParam(16, 8, &vpx_sad16x8_avg_neon_dotprod),
};
INSTANTIATE_TEST_SUITE_P(NEON_DOTPROD, SADavgTest,
                         ::testing::ValuesIn(avg_neon_dotprod_tests));
#endif  // HAVE_NEON_DOTPROD

const SadMxNx4Param x4d_neon_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_neon),
  SadMxNx4Param(64, 32, &vpx_sad64x32x4d_neon),
  SadMxNx4Param(32, 64, &vpx_sad32x64x4d_neon),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_neon),
  SadMxNx4Param(32, 16, &vpx_sad32x16x4d_neon),
  SadMxNx4Param(16, 32, &vpx_sad16x32x4d_neon),
  SadMxNx4Param(16, 16, &vpx_sad16x16x4d_neon),
  SadMxNx4Param(16, 8, &vpx_sad16x8x4d_neon),
  SadMxNx4Param(8, 16, &vpx_sad8x16x4d_neon),
  SadMxNx4Param(8, 8, &vpx_sad8x8x4d_neon),
  SadMxNx4Param(8, 4, &vpx_sad8x4x4d_neon),
  SadMxNx4Param(4, 8, &vpx_sad4x8x4d_neon),
  SadMxNx4Param(4, 4, &vpx_sad4x4x4d_neon),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_neon, 8),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_neon, 8),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_neon, 8),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_neon, 8),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_neon, 8),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_neon, 8),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_neon, 8),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_neon, 8),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_neon, 8),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_neon, 8),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_neon, 8),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_neon, 8),
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_neon, 10),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_neon, 10),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_neon, 10),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_neon, 10),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_neon, 10),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_neon, 10),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_neon, 10),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_neon, 10),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_neon, 10),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_neon, 10),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_neon, 10),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_neon, 10),
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_neon, 12),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_neon, 12),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_neon, 12),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_neon, 12),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_neon, 12),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_neon, 12),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_neon, 12),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_neon, 12),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_neon, 12),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_neon, 12),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_neon, 12),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_neon, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(NEON, SADx4Test, ::testing::ValuesIn(x4d_neon_tests));

#if HAVE_NEON_DOTPROD
const SadMxNx4Param x4d_neon_dotprod_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_neon_dotprod),
  SadMxNx4Param(64, 32, &vpx_sad64x32x4d_neon_dotprod),
  SadMxNx4Param(32, 64, &vpx_sad32x64x4d_neon_dotprod),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_neon_dotprod),
  SadMxNx4Param(32, 16, &vpx_sad32x16x4d_neon_dotprod),
  SadMxNx4Param(16, 32, &vpx_sad16x32x4d_neon_dotprod),
  SadMxNx4Param(16, 16, &vpx_sad16x16x4d_neon_dotprod),
  SadMxNx4Param(16, 8, &vpx_sad16x8x4d_neon_dotprod),
};
INSTANTIATE_TEST_SUITE_P(NEON_DOTPROD, SADx4Test,
                         ::testing::ValuesIn(x4d_neon_dotprod_tests));
#endif  // HAVE_NEON_DOTPROD

const SadSkipMxNx4Param skip_x4d_neon_tests[] = {
  SadSkipMxNx4Param(64, 64, &vpx_sad_skip_64x64x4d_neon),
  SadSkipMxNx4Param(64, 32, &vpx_sad_skip_64x32x4d_neon),
  SadSkipMxNx4Param(32, 64, &vpx_sad_skip_32x64x4d_neon),
  SadSkipMxNx4Param(32, 32, &vpx_sad_skip_32x32x4d_neon),
  SadSkipMxNx4Param(32, 16, &vpx_sad_skip_32x16x4d_neon),
  SadSkipMxNx4Param(16, 32, &vpx_sad_skip_16x32x4d_neon),
  SadSkipMxNx4Param(16, 16, &vpx_sad_skip_16x16x4d_neon),
  SadSkipMxNx4Param(16, 8, &vpx_sad_skip_16x8x4d_neon),
  SadSkipMxNx4Param(8, 16, &vpx_sad_skip_8x16x4d_neon),
  SadSkipMxNx4Param(8, 8, &vpx_sad_skip_8x8x4d_neon),
  SadSkipMxNx4Param(8, 4, &vpx_sad_skip_8x4x4d_neon),
  SadSkipMxNx4Param(4, 8, &vpx_sad_skip_4x8x4d_neon),
  SadSkipMxNx4Param(4, 4, &vpx_sad_skip_4x4x4d_neon),
#if CONFIG_VP9_HIGHBITDEPTH
  SadSkipMxNx4Param(4, 4, &vpx_highbd_sad_skip_4x4x4d_neon, 8),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_neon, 8),
  SadSkipMxNx4Param(8, 4, &vpx_highbd_sad_skip_8x4x4d_neon, 8),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_neon, 8),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_neon, 8),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_neon, 8),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_neon, 8),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_neon, 8),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_neon, 8),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_neon, 8),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_neon, 8),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_neon, 8),
  SadSkipMxNx4Param(4, 4, &vpx_highbd_sad_skip_4x4x4d_neon, 10),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_neon, 10),
  SadSkipMxNx4Param(8, 4, &vpx_highbd_sad_skip_8x4x4d_neon, 10),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_neon, 10),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_neon, 10),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_neon, 10),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_neon, 10),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_neon, 10),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_neon, 10),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_neon, 10),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_neon, 10),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_neon, 10),
  SadSkipMxNx4Param(4, 4, &vpx_highbd_sad_skip_4x4x4d_neon, 12),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_neon, 12),
  SadSkipMxNx4Param(8, 4, &vpx_highbd_sad_skip_8x4x4d_neon, 12),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_neon, 12),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_neon, 12),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_neon, 12),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_neon, 12),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_neon, 12),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_neon, 12),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_neon, 12),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_neon, 12),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_neon, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(NEON, SADSkipx4Test,
                         ::testing::ValuesIn(skip_x4d_neon_tests));

#if HAVE_NEONE_DOTPROD
const SadSkipMxNx4Param skip_x4d_neon_dotprod_tests[] = {
  SadSkipMxNx4Param(64, 64, &vpx_sad_skip_64x64x4d_neon_dotprod),
  SadSkipMxNx4Param(64, 32, &vpx_sad_skip_64x32x4d_neon_dotprod),
  SadSkipMxNx4Param(32, 64, &vpx_sad_skip_32x64x4d_neon_dotprod),
  SadSkipMxNx4Param(32, 32, &vpx_sad_skip_32x32x4d_neon_dotprod),
  SadSkipMxNx4Param(32, 16, &vpx_sad_skip_32x16x4d_neon_dotprod),
  SadSkipMxNx4Param(16, 32, &vpx_sad_skip_16x32x4d_neon_dotprod),
  SadSkipMxNx4Param(16, 16, &vpx_sad_skip_16x16x4d_neon_dotprod),
  SadSkipMxNx4Param(16, 8, &vpx_sad_skip_16x8x4d_neon_dotprod),
};
INSTANTIATE_TEST_SUITE_P(NEON_DOTPROD, SADSkipx4Test,
                         ::testing::ValuesIn(skip_x4d_neon_dotprod_tests));
#endif  // HAVE_NEON_DOTPROD
#endif  // HAVE_NEON

//------------------------------------------------------------------------------
// x86 functions
#if HAVE_SSE2
const SadMxNParam sse2_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_sse2),
  SadMxNParam(64, 32, &vpx_sad64x32_sse2),
  SadMxNParam(32, 64, &vpx_sad32x64_sse2),
  SadMxNParam(32, 32, &vpx_sad32x32_sse2),
  SadMxNParam(32, 16, &vpx_sad32x16_sse2),
  SadMxNParam(16, 32, &vpx_sad16x32_sse2),
  SadMxNParam(16, 16, &vpx_sad16x16_sse2),
  SadMxNParam(16, 8, &vpx_sad16x8_sse2),
  SadMxNParam(8, 16, &vpx_sad8x16_sse2),
  SadMxNParam(8, 8, &vpx_sad8x8_sse2),
  SadMxNParam(8, 4, &vpx_sad8x4_sse2),
  SadMxNParam(4, 8, &vpx_sad4x8_sse2),
  SadMxNParam(4, 4, &vpx_sad4x4_sse2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_sse2, 8),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_sse2, 8),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_sse2, 8),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_sse2, 8),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_sse2, 8),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_sse2, 8),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_sse2, 8),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_sse2, 8),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_sse2, 8),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_sse2, 8),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_sse2, 8),
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_sse2, 10),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_sse2, 10),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_sse2, 10),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_sse2, 10),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_sse2, 10),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_sse2, 10),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_sse2, 10),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_sse2, 10),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_sse2, 10),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_sse2, 10),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_sse2, 10),
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_sse2, 12),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_sse2, 12),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_sse2, 12),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_sse2, 12),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_sse2, 12),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_sse2, 12),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_sse2, 12),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_sse2, 12),
  SadMxNParam(8, 16, &vpx_highbd_sad8x16_sse2, 12),
  SadMxNParam(8, 8, &vpx_highbd_sad8x8_sse2, 12),
  SadMxNParam(8, 4, &vpx_highbd_sad8x4_sse2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(SSE2, SADTest, ::testing::ValuesIn(sse2_tests));

const SadSkipMxNParam skip_sse2_tests[] = {
  SadSkipMxNParam(64, 64, &vpx_sad_skip_64x64_sse2),
  SadSkipMxNParam(64, 32, &vpx_sad_skip_64x32_sse2),
  SadSkipMxNParam(32, 64, &vpx_sad_skip_32x64_sse2),
  SadSkipMxNParam(32, 32, &vpx_sad_skip_32x32_sse2),
  SadSkipMxNParam(32, 16, &vpx_sad_skip_32x16_sse2),
  SadSkipMxNParam(16, 32, &vpx_sad_skip_16x32_sse2),
  SadSkipMxNParam(16, 16, &vpx_sad_skip_16x16_sse2),
  SadSkipMxNParam(16, 8, &vpx_sad_skip_16x8_sse2),
  SadSkipMxNParam(8, 16, &vpx_sad_skip_8x16_sse2),
  SadSkipMxNParam(8, 8, &vpx_sad_skip_8x8_sse2),
  SadSkipMxNParam(4, 8, &vpx_sad_skip_4x8_sse2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_sse2, 8),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_sse2, 8),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_sse2, 8),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_sse2, 8),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_sse2, 8),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_sse2, 8),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_sse2, 8),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_sse2, 8),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_sse2, 8),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_sse2, 8),
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_sse2, 10),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_sse2, 10),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_sse2, 10),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_sse2, 10),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_sse2, 10),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_sse2, 10),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_sse2, 10),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_sse2, 10),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_sse2, 10),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_sse2, 10),
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_sse2, 12),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_sse2, 12),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_sse2, 12),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_sse2, 12),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_sse2, 12),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_sse2, 12),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_sse2, 12),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_sse2, 12),
  SadSkipMxNParam(8, 16, &vpx_highbd_sad_skip_8x16_sse2, 12),
  SadSkipMxNParam(8, 8, &vpx_highbd_sad_skip_8x8_sse2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(SSE2, SADSkipTest,
                         ::testing::ValuesIn(skip_sse2_tests));

const SadMxNAvgParam avg_sse2_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_sse2),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_sse2),
  SadMxNAvgParam(32, 64, &vpx_sad32x64_avg_sse2),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_sse2),
  SadMxNAvgParam(32, 16, &vpx_sad32x16_avg_sse2),
  SadMxNAvgParam(16, 32, &vpx_sad16x32_avg_sse2),
  SadMxNAvgParam(16, 16, &vpx_sad16x16_avg_sse2),
  SadMxNAvgParam(16, 8, &vpx_sad16x8_avg_sse2),
  SadMxNAvgParam(8, 16, &vpx_sad8x16_avg_sse2),
  SadMxNAvgParam(8, 8, &vpx_sad8x8_avg_sse2),
  SadMxNAvgParam(8, 4, &vpx_sad8x4_avg_sse2),
  SadMxNAvgParam(4, 8, &vpx_sad4x8_avg_sse2),
  SadMxNAvgParam(4, 4, &vpx_sad4x4_avg_sse2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_sse2, 8),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_sse2, 8),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_sse2, 8),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_sse2, 8),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_sse2, 8),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_sse2, 8),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_sse2, 8),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_sse2, 8),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_sse2, 8),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_sse2, 8),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_sse2, 8),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_sse2, 10),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_sse2, 10),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_sse2, 10),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_sse2, 10),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_sse2, 10),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_sse2, 10),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_sse2, 10),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_sse2, 10),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_sse2, 10),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_sse2, 10),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_sse2, 10),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_sse2, 12),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_sse2, 12),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_sse2, 12),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_sse2, 12),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_sse2, 12),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_sse2, 12),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_sse2, 12),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_sse2, 12),
  SadMxNAvgParam(8, 16, &vpx_highbd_sad8x16_avg_sse2, 12),
  SadMxNAvgParam(8, 8, &vpx_highbd_sad8x8_avg_sse2, 12),
  SadMxNAvgParam(8, 4, &vpx_highbd_sad8x4_avg_sse2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(SSE2, SADavgTest, ::testing::ValuesIn(avg_sse2_tests));

const SadMxNx4Param x4d_sse2_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_sse2),
  SadMxNx4Param(64, 32, &vpx_sad64x32x4d_sse2),
  SadMxNx4Param(32, 64, &vpx_sad32x64x4d_sse2),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_sse2),
  SadMxNx4Param(32, 16, &vpx_sad32x16x4d_sse2),
  SadMxNx4Param(16, 32, &vpx_sad16x32x4d_sse2),
  SadMxNx4Param(16, 16, &vpx_sad16x16x4d_sse2),
  SadMxNx4Param(16, 8, &vpx_sad16x8x4d_sse2),
  SadMxNx4Param(8, 16, &vpx_sad8x16x4d_sse2),
  SadMxNx4Param(8, 8, &vpx_sad8x8x4d_sse2),
  SadMxNx4Param(8, 4, &vpx_sad8x4x4d_sse2),
  SadMxNx4Param(4, 8, &vpx_sad4x8x4d_sse2),
  SadMxNx4Param(4, 4, &vpx_sad4x4x4d_sse2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_sse2, 8),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_sse2, 8),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_sse2, 8),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_sse2, 8),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_sse2, 8),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_sse2, 8),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_sse2, 8),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_sse2, 8),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_sse2, 8),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_sse2, 8),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_sse2, 8),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_sse2, 8),
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_sse2, 8),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_sse2, 10),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_sse2, 10),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_sse2, 10),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_sse2, 10),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_sse2, 10),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_sse2, 10),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_sse2, 10),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_sse2, 10),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_sse2, 10),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_sse2, 10),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_sse2, 10),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_sse2, 10),
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_sse2, 10),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_sse2, 12),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_sse2, 12),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_sse2, 12),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_sse2, 12),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_sse2, 12),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_sse2, 12),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_sse2, 12),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_sse2, 12),
  SadMxNx4Param(8, 16, &vpx_highbd_sad8x16x4d_sse2, 12),
  SadMxNx4Param(8, 8, &vpx_highbd_sad8x8x4d_sse2, 12),
  SadMxNx4Param(8, 4, &vpx_highbd_sad8x4x4d_sse2, 12),
  SadMxNx4Param(4, 8, &vpx_highbd_sad4x8x4d_sse2, 12),
  SadMxNx4Param(4, 4, &vpx_highbd_sad4x4x4d_sse2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(SSE2, SADx4Test, ::testing::ValuesIn(x4d_sse2_tests));

const SadSkipMxNx4Param skip_x4d_sse2_tests[] = {
  SadSkipMxNx4Param(64, 64, &vpx_sad_skip_64x64x4d_sse2),
  SadSkipMxNx4Param(64, 32, &vpx_sad_skip_64x32x4d_sse2),
  SadSkipMxNx4Param(32, 64, &vpx_sad_skip_32x64x4d_sse2),
  SadSkipMxNx4Param(32, 32, &vpx_sad_skip_32x32x4d_sse2),
  SadSkipMxNx4Param(32, 16, &vpx_sad_skip_32x16x4d_sse2),
  SadSkipMxNx4Param(16, 32, &vpx_sad_skip_16x32x4d_sse2),
  SadSkipMxNx4Param(16, 16, &vpx_sad_skip_16x16x4d_sse2),
  SadSkipMxNx4Param(16, 8, &vpx_sad_skip_16x8x4d_sse2),
  SadSkipMxNx4Param(8, 16, &vpx_sad_skip_8x16x4d_sse2),
  SadSkipMxNx4Param(8, 8, &vpx_sad_skip_8x8x4d_sse2),
  SadSkipMxNx4Param(4, 8, &vpx_sad_skip_4x8x4d_sse2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_sse2, 8),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_sse2, 8),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_sse2, 8),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_sse2, 8),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_sse2, 8),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_sse2, 8),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_sse2, 8),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_sse2, 8),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_sse2, 8),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_sse2, 8),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_sse2, 8),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_sse2, 10),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_sse2, 10),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_sse2, 10),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_sse2, 10),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_sse2, 10),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_sse2, 10),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_sse2, 10),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_sse2, 10),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_sse2, 10),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_sse2, 10),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_sse2, 10),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_sse2, 12),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_sse2, 12),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_sse2, 12),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_sse2, 12),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_sse2, 12),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_sse2, 12),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_sse2, 12),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_sse2, 12),
  SadSkipMxNx4Param(8, 16, &vpx_highbd_sad_skip_8x16x4d_sse2, 12),
  SadSkipMxNx4Param(8, 8, &vpx_highbd_sad_skip_8x8x4d_sse2, 12),
  SadSkipMxNx4Param(4, 8, &vpx_highbd_sad_skip_4x8x4d_sse2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(SSE2, SADSkipx4Test,
                         ::testing::ValuesIn(skip_x4d_sse2_tests));
#endif  // HAVE_SSE2

#if HAVE_SSE3
// Only functions are x3, which do not have tests.
#endif  // HAVE_SSE3

#if HAVE_SSSE3
// Only functions are x3, which do not have tests.
#endif  // HAVE_SSSE3

#if HAVE_AVX2
const SadMxNParam avx2_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_avx2),
  SadMxNParam(64, 32, &vpx_sad64x32_avx2),
  SadMxNParam(32, 64, &vpx_sad32x64_avx2),
  SadMxNParam(32, 32, &vpx_sad32x32_avx2),
  SadMxNParam(32, 16, &vpx_sad32x16_avx2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNParam(64, 64, &vpx_highbd_sad64x64_avx2, 8),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_avx2, 8),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_avx2, 8),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_avx2, 8),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_avx2, 8),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_avx2, 8),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_avx2, 8),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_avx2, 8),

  SadMxNParam(64, 64, &vpx_highbd_sad64x64_avx2, 10),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_avx2, 10),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_avx2, 10),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_avx2, 10),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_avx2, 10),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_avx2, 10),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_avx2, 10),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_avx2, 10),

  SadMxNParam(64, 64, &vpx_highbd_sad64x64_avx2, 12),
  SadMxNParam(64, 32, &vpx_highbd_sad64x32_avx2, 12),
  SadMxNParam(32, 64, &vpx_highbd_sad32x64_avx2, 12),
  SadMxNParam(32, 32, &vpx_highbd_sad32x32_avx2, 12),
  SadMxNParam(32, 16, &vpx_highbd_sad32x16_avx2, 12),
  SadMxNParam(16, 32, &vpx_highbd_sad16x32_avx2, 12),
  SadMxNParam(16, 16, &vpx_highbd_sad16x16_avx2, 12),
  SadMxNParam(16, 8, &vpx_highbd_sad16x8_avx2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(AVX2, SADTest, ::testing::ValuesIn(avx2_tests));

const SadSkipMxNParam skip_avx2_tests[] = {
  SadSkipMxNParam(64, 64, &vpx_sad_skip_64x64_avx2),
  SadSkipMxNParam(64, 32, &vpx_sad_skip_64x32_avx2),
  SadSkipMxNParam(32, 64, &vpx_sad_skip_32x64_avx2),
  SadSkipMxNParam(32, 32, &vpx_sad_skip_32x32_avx2),
  SadSkipMxNParam(32, 16, &vpx_sad_skip_32x16_avx2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_avx2, 8),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_avx2, 8),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_avx2, 8),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_avx2, 8),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_avx2, 8),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_avx2, 8),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_avx2, 8),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_avx2, 8),

  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_avx2, 10),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_avx2, 10),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_avx2, 10),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_avx2, 10),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_avx2, 10),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_avx2, 10),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_avx2, 10),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_avx2, 10),

  SadSkipMxNParam(64, 64, &vpx_highbd_sad_skip_64x64_avx2, 12),
  SadSkipMxNParam(64, 32, &vpx_highbd_sad_skip_64x32_avx2, 12),
  SadSkipMxNParam(32, 64, &vpx_highbd_sad_skip_32x64_avx2, 12),
  SadSkipMxNParam(32, 32, &vpx_highbd_sad_skip_32x32_avx2, 12),
  SadSkipMxNParam(32, 16, &vpx_highbd_sad_skip_32x16_avx2, 12),
  SadSkipMxNParam(16, 32, &vpx_highbd_sad_skip_16x32_avx2, 12),
  SadSkipMxNParam(16, 16, &vpx_highbd_sad_skip_16x16_avx2, 12),
  SadSkipMxNParam(16, 8, &vpx_highbd_sad_skip_16x8_avx2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(AVX2, SADSkipTest,
                         ::testing::ValuesIn(skip_avx2_tests));

const SadMxNAvgParam avg_avx2_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_avx2),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_avx2),
  SadMxNAvgParam(32, 64, &vpx_sad32x64_avg_avx2),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_avx2),
  SadMxNAvgParam(32, 16, &vpx_sad32x16_avg_avx2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_avx2, 8),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_avx2, 8),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_avx2, 8),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_avx2, 8),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_avx2, 8),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_avx2, 8),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_avx2, 8),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_avx2, 8),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_avx2, 10),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_avx2, 10),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_avx2, 10),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_avx2, 10),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_avx2, 10),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_avx2, 10),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_avx2, 10),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_avx2, 10),
  SadMxNAvgParam(64, 64, &vpx_highbd_sad64x64_avg_avx2, 12),
  SadMxNAvgParam(64, 32, &vpx_highbd_sad64x32_avg_avx2, 12),
  SadMxNAvgParam(32, 64, &vpx_highbd_sad32x64_avg_avx2, 12),
  SadMxNAvgParam(32, 32, &vpx_highbd_sad32x32_avg_avx2, 12),
  SadMxNAvgParam(32, 16, &vpx_highbd_sad32x16_avg_avx2, 12),
  SadMxNAvgParam(16, 32, &vpx_highbd_sad16x32_avg_avx2, 12),
  SadMxNAvgParam(16, 16, &vpx_highbd_sad16x16_avg_avx2, 12),
  SadMxNAvgParam(16, 8, &vpx_highbd_sad16x8_avg_avx2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(AVX2, SADavgTest, ::testing::ValuesIn(avg_avx2_tests));

const SadMxNx4Param x4d_avx2_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_avx2),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_avx2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_avx2, 8),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_avx2, 8),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_avx2, 8),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_avx2, 8),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_avx2, 8),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_avx2, 8),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_avx2, 8),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_avx2, 8),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_avx2, 10),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_avx2, 10),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_avx2, 10),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_avx2, 10),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_avx2, 10),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_avx2, 10),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_avx2, 10),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_avx2, 10),
  SadMxNx4Param(64, 64, &vpx_highbd_sad64x64x4d_avx2, 12),
  SadMxNx4Param(64, 32, &vpx_highbd_sad64x32x4d_avx2, 12),
  SadMxNx4Param(32, 64, &vpx_highbd_sad32x64x4d_avx2, 12),
  SadMxNx4Param(32, 32, &vpx_highbd_sad32x32x4d_avx2, 12),
  SadMxNx4Param(32, 16, &vpx_highbd_sad32x16x4d_avx2, 12),
  SadMxNx4Param(16, 32, &vpx_highbd_sad16x32x4d_avx2, 12),
  SadMxNx4Param(16, 16, &vpx_highbd_sad16x16x4d_avx2, 12),
  SadMxNx4Param(16, 8, &vpx_highbd_sad16x8x4d_avx2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(AVX2, SADx4Test, ::testing::ValuesIn(x4d_avx2_tests));

const SadSkipMxNx4Param skip_x4d_avx2_tests[] = {
  SadSkipMxNx4Param(64, 64, &vpx_sad_skip_64x64x4d_avx2),
  SadSkipMxNx4Param(64, 32, &vpx_sad_skip_64x32x4d_avx2),
  SadSkipMxNx4Param(32, 64, &vpx_sad_skip_32x64x4d_avx2),
  SadSkipMxNx4Param(32, 32, &vpx_sad_skip_32x32x4d_avx2),
  SadSkipMxNx4Param(32, 16, &vpx_sad_skip_32x16x4d_avx2),
#if CONFIG_VP9_HIGHBITDEPTH
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_avx2, 8),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_avx2, 8),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_avx2, 8),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_avx2, 8),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_avx2, 8),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_avx2, 8),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_avx2, 8),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_avx2, 8),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_avx2, 10),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_avx2, 10),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_avx2, 10),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_avx2, 10),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_avx2, 10),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_avx2, 10),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_avx2, 10),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_avx2, 10),
  SadSkipMxNx4Param(64, 64, &vpx_highbd_sad_skip_64x64x4d_avx2, 12),
  SadSkipMxNx4Param(64, 32, &vpx_highbd_sad_skip_64x32x4d_avx2, 12),
  SadSkipMxNx4Param(32, 64, &vpx_highbd_sad_skip_32x64x4d_avx2, 12),
  SadSkipMxNx4Param(32, 32, &vpx_highbd_sad_skip_32x32x4d_avx2, 12),
  SadSkipMxNx4Param(32, 16, &vpx_highbd_sad_skip_32x16x4d_avx2, 12),
  SadSkipMxNx4Param(16, 32, &vpx_highbd_sad_skip_16x32x4d_avx2, 12),
  SadSkipMxNx4Param(16, 16, &vpx_highbd_sad_skip_16x16x4d_avx2, 12),
  SadSkipMxNx4Param(16, 8, &vpx_highbd_sad_skip_16x8x4d_avx2, 12),
#endif  // CONFIG_VP9_HIGHBITDEPTH
};
INSTANTIATE_TEST_SUITE_P(AVX2, SADSkipx4Test,
                         ::testing::ValuesIn(skip_x4d_avx2_tests));

#endif  // HAVE_AVX2

#if HAVE_AVX512
const SadMxNParam avx512_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_avx512),
  SadMxNParam(64, 32, &vpx_sad64x32_avx512),
};
INSTANTIATE_TEST_SUITE_P(AVX512, SADTest, ::testing::ValuesIn(avx512_tests));

const SadSkipMxNParam skip_avx512_tests[] = {
  SadSkipMxNParam(64, 64, &vpx_sad_skip_64x64_avx512),
  SadSkipMxNParam(64, 32, &vpx_sad_skip_64x32_avx512),
};
INSTANTIATE_TEST_SUITE_P(AVX512, SADSkipTest,
                         ::testing::ValuesIn(skip_avx512_tests));

const SadMxNAvgParam avg_avx512_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_avx512),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_avx512),
};
INSTANTIATE_TEST_SUITE_P(AVX512, SADavgTest,
                         ::testing::ValuesIn(avg_avx512_tests));

const SadMxNx4Param x4d_avx512_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_avx512),
};
INSTANTIATE_TEST_SUITE_P(AVX512, SADx4Test,
                         ::testing::ValuesIn(x4d_avx512_tests));

const SadSkipMxNx4Param skip_x4d_avx512_tests[] = {
  SadSkipMxNx4Param(64, 64, &vpx_sad_skip_64x64x4d_avx512),
  SadSkipMxNx4Param(64, 32, &vpx_sad_skip_64x32x4d_avx512),
};
INSTANTIATE_TEST_SUITE_P(AVX512, SADSkipx4Test,
                         ::testing::ValuesIn(skip_x4d_avx512_tests));
#endif  // HAVE_AVX512

//------------------------------------------------------------------------------
// MIPS functions
#if HAVE_MSA
const SadMxNParam msa_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_msa),
  SadMxNParam(64, 32, &vpx_sad64x32_msa),
  SadMxNParam(32, 64, &vpx_sad32x64_msa),
  SadMxNParam(32, 32, &vpx_sad32x32_msa),
  SadMxNParam(32, 16, &vpx_sad32x16_msa),
  SadMxNParam(16, 32, &vpx_sad16x32_msa),
  SadMxNParam(16, 16, &vpx_sad16x16_msa),
  SadMxNParam(16, 8, &vpx_sad16x8_msa),
  SadMxNParam(8, 16, &vpx_sad8x16_msa),
  SadMxNParam(8, 8, &vpx_sad8x8_msa),
  SadMxNParam(8, 4, &vpx_sad8x4_msa),
  SadMxNParam(4, 8, &vpx_sad4x8_msa),
  SadMxNParam(4, 4, &vpx_sad4x4_msa),
};
INSTANTIATE_TEST_SUITE_P(MSA, SADTest, ::testing::ValuesIn(msa_tests));

const SadMxNAvgParam avg_msa_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_msa),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_msa),
  SadMxNAvgParam(32, 64, &vpx_sad32x64_avg_msa),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_msa),
  SadMxNAvgParam(32, 16, &vpx_sad32x16_avg_msa),
  SadMxNAvgParam(16, 32, &vpx_sad16x32_avg_msa),
  SadMxNAvgParam(16, 16, &vpx_sad16x16_avg_msa),
  SadMxNAvgParam(16, 8, &vpx_sad16x8_avg_msa),
  SadMxNAvgParam(8, 16, &vpx_sad8x16_avg_msa),
  SadMxNAvgParam(8, 8, &vpx_sad8x8_avg_msa),
  SadMxNAvgParam(8, 4, &vpx_sad8x4_avg_msa),
  SadMxNAvgParam(4, 8, &vpx_sad4x8_avg_msa),
  SadMxNAvgParam(4, 4, &vpx_sad4x4_avg_msa),
};
INSTANTIATE_TEST_SUITE_P(MSA, SADavgTest, ::testing::ValuesIn(avg_msa_tests));

const SadMxNx4Param x4d_msa_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_msa),
  SadMxNx4Param(64, 32, &vpx_sad64x32x4d_msa),
  SadMxNx4Param(32, 64, &vpx_sad32x64x4d_msa),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_msa),
  SadMxNx4Param(32, 16, &vpx_sad32x16x4d_msa),
  SadMxNx4Param(16, 32, &vpx_sad16x32x4d_msa),
  SadMxNx4Param(16, 16, &vpx_sad16x16x4d_msa),
  SadMxNx4Param(16, 8, &vpx_sad16x8x4d_msa),
  SadMxNx4Param(8, 16, &vpx_sad8x16x4d_msa),
  SadMxNx4Param(8, 8, &vpx_sad8x8x4d_msa),
  SadMxNx4Param(8, 4, &vpx_sad8x4x4d_msa),
  SadMxNx4Param(4, 8, &vpx_sad4x8x4d_msa),
  SadMxNx4Param(4, 4, &vpx_sad4x4x4d_msa),
};
INSTANTIATE_TEST_SUITE_P(MSA, SADx4Test, ::testing::ValuesIn(x4d_msa_tests));
#endif  // HAVE_MSA

//------------------------------------------------------------------------------
// VSX functions
#if HAVE_VSX
const SadMxNParam vsx_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_vsx),
  SadMxNParam(64, 32, &vpx_sad64x32_vsx),
  SadMxNParam(32, 64, &vpx_sad32x64_vsx),
  SadMxNParam(32, 32, &vpx_sad32x32_vsx),
  SadMxNParam(32, 16, &vpx_sad32x16_vsx),
  SadMxNParam(16, 32, &vpx_sad16x32_vsx),
  SadMxNParam(16, 16, &vpx_sad16x16_vsx),
  SadMxNParam(16, 8, &vpx_sad16x8_vsx),
  SadMxNParam(8, 16, &vpx_sad8x16_vsx),
  SadMxNParam(8, 8, &vpx_sad8x8_vsx),
  SadMxNParam(8, 4, &vpx_sad8x4_vsx),
};
INSTANTIATE_TEST_SUITE_P(VSX, SADTest, ::testing::ValuesIn(vsx_tests));

const SadMxNAvgParam avg_vsx_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_vsx),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_vsx),
  SadMxNAvgParam(32, 64, &vpx_sad32x64_avg_vsx),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_vsx),
  SadMxNAvgParam(32, 16, &vpx_sad32x16_avg_vsx),
  SadMxNAvgParam(16, 32, &vpx_sad16x32_avg_vsx),
  SadMxNAvgParam(16, 16, &vpx_sad16x16_avg_vsx),
  SadMxNAvgParam(16, 8, &vpx_sad16x8_avg_vsx),
};
INSTANTIATE_TEST_SUITE_P(VSX, SADavgTest, ::testing::ValuesIn(avg_vsx_tests));

const SadMxNx4Param x4d_vsx_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_vsx),
  SadMxNx4Param(64, 32, &vpx_sad64x32x4d_vsx),
  SadMxNx4Param(32, 64, &vpx_sad32x64x4d_vsx),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_vsx),
  SadMxNx4Param(32, 16, &vpx_sad32x16x4d_vsx),
  SadMxNx4Param(16, 32, &vpx_sad16x32x4d_vsx),
  SadMxNx4Param(16, 16, &vpx_sad16x16x4d_vsx),
  SadMxNx4Param(16, 8, &vpx_sad16x8x4d_vsx),
};
INSTANTIATE_TEST_SUITE_P(VSX, SADx4Test, ::testing::ValuesIn(x4d_vsx_tests));
#endif  // HAVE_VSX

//------------------------------------------------------------------------------
// Loongson functions
#if HAVE_MMI
const SadMxNParam mmi_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_mmi),
  SadMxNParam(64, 32, &vpx_sad64x32_mmi),
  SadMxNParam(32, 64, &vpx_sad32x64_mmi),
  SadMxNParam(32, 32, &vpx_sad32x32_mmi),
  SadMxNParam(32, 16, &vpx_sad32x16_mmi),
  SadMxNParam(16, 32, &vpx_sad16x32_mmi),
  SadMxNParam(16, 16, &vpx_sad16x16_mmi),
  SadMxNParam(16, 8, &vpx_sad16x8_mmi),
  SadMxNParam(8, 16, &vpx_sad8x16_mmi),
  SadMxNParam(8, 8, &vpx_sad8x8_mmi),
  SadMxNParam(8, 4, &vpx_sad8x4_mmi),
  SadMxNParam(4, 8, &vpx_sad4x8_mmi),
  SadMxNParam(4, 4, &vpx_sad4x4_mmi),
};
INSTANTIATE_TEST_SUITE_P(MMI, SADTest, ::testing::ValuesIn(mmi_tests));

const SadMxNAvgParam avg_mmi_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_mmi),
  SadMxNAvgParam(64, 32, &vpx_sad64x32_avg_mmi),
  SadMxNAvgParam(32, 64, &vpx_sad32x64_avg_mmi),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_mmi),
  SadMxNAvgParam(32, 16, &vpx_sad32x16_avg_mmi),
  SadMxNAvgParam(16, 32, &vpx_sad16x32_avg_mmi),
  SadMxNAvgParam(16, 16, &vpx_sad16x16_avg_mmi),
  SadMxNAvgParam(16, 8, &vpx_sad16x8_avg_mmi),
  SadMxNAvgParam(8, 16, &vpx_sad8x16_avg_mmi),
  SadMxNAvgParam(8, 8, &vpx_sad8x8_avg_mmi),
  SadMxNAvgParam(8, 4, &vpx_sad8x4_avg_mmi),
  SadMxNAvgParam(4, 8, &vpx_sad4x8_avg_mmi),
  SadMxNAvgParam(4, 4, &vpx_sad4x4_avg_mmi),
};
INSTANTIATE_TEST_SUITE_P(MMI, SADavgTest, ::testing::ValuesIn(avg_mmi_tests));

const SadMxNx4Param x4d_mmi_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_mmi),
  SadMxNx4Param(64, 32, &vpx_sad64x32x4d_mmi),
  SadMxNx4Param(32, 64, &vpx_sad32x64x4d_mmi),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_mmi),
  SadMxNx4Param(32, 16, &vpx_sad32x16x4d_mmi),
  SadMxNx4Param(16, 32, &vpx_sad16x32x4d_mmi),
  SadMxNx4Param(16, 16, &vpx_sad16x16x4d_mmi),
  SadMxNx4Param(16, 8, &vpx_sad16x8x4d_mmi),
  SadMxNx4Param(8, 16, &vpx_sad8x16x4d_mmi),
  SadMxNx4Param(8, 8, &vpx_sad8x8x4d_mmi),
  SadMxNx4Param(8, 4, &vpx_sad8x4x4d_mmi),
  SadMxNx4Param(4, 8, &vpx_sad4x8x4d_mmi),
  SadMxNx4Param(4, 4, &vpx_sad4x4x4d_mmi),
};
INSTANTIATE_TEST_SUITE_P(MMI, SADx4Test, ::testing::ValuesIn(x4d_mmi_tests));
#endif  // HAVE_MMI

//------------------------------------------------------------------------------
// loongarch functions
#if HAVE_LSX
const SadMxNParam lsx_tests[] = {
  SadMxNParam(64, 64, &vpx_sad64x64_lsx),
  SadMxNParam(32, 32, &vpx_sad32x32_lsx),
  SadMxNParam(16, 16, &vpx_sad16x16_lsx),
  SadMxNParam(8, 8, &vpx_sad8x8_lsx),
};
INSTANTIATE_TEST_SUITE_P(LSX, SADTest, ::testing::ValuesIn(lsx_tests));

const SadMxNAvgParam avg_lsx_tests[] = {
  SadMxNAvgParam(64, 64, &vpx_sad64x64_avg_lsx),
  SadMxNAvgParam(32, 32, &vpx_sad32x32_avg_lsx),
};
INSTANTIATE_TEST_SUITE_P(LSX, SADavgTest, ::testing::ValuesIn(avg_lsx_tests));

const SadMxNx4Param x4d_lsx_tests[] = {
  SadMxNx4Param(64, 64, &vpx_sad64x64x4d_lsx),
  SadMxNx4Param(64, 32, &vpx_sad64x32x4d_lsx),
  SadMxNx4Param(32, 64, &vpx_sad32x64x4d_lsx),
  SadMxNx4Param(32, 32, &vpx_sad32x32x4d_lsx),
  SadMxNx4Param(16, 16, &vpx_sad16x16x4d_lsx),
  SadMxNx4Param(8, 8, &vpx_sad8x8x4d_lsx),
};
INSTANTIATE_TEST_SUITE_P(LSX, SADx4Test, ::testing::ValuesIn(x4d_lsx_tests));
#endif  // HAVE_LSX

}  // namespace
