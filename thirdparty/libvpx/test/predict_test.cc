/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>

#include "gtest/gtest.h"

#include "./vp8_rtcd.h"
#include "./vpx_config.h"
#include "test/acm_random.h"
#include "test/bench.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"

namespace {

using libvpx_test::ACMRandom;
using std::make_tuple;

using PredictFunc = void (*)(uint8_t *src_ptr, int src_pixels_per_line,
                             int xoffset, int yoffset, uint8_t *dst_ptr,
                             int dst_pitch);

using PredictParam = std::tuple<int, int, PredictFunc>;

class PredictTestBase : public AbstractBench,
                        public ::testing::TestWithParam<PredictParam> {
 public:
  PredictTestBase()
      : width_(GET_PARAM(0)), height_(GET_PARAM(1)), predict_(GET_PARAM(2)),
        src_(nullptr), padded_dst_(nullptr), dst_(nullptr), dst_c_(nullptr) {}

  void SetUp() override {
    src_ = new uint8_t[kSrcSize];
    ASSERT_NE(src_, nullptr);

    // padded_dst_ provides a buffer of kBorderSize around the destination
    // memory to facilitate detecting out of bounds writes.
    dst_stride_ = kBorderSize + width_ + kBorderSize;
    padded_dst_size_ = dst_stride_ * (kBorderSize + height_ + kBorderSize);
    padded_dst_ =
        reinterpret_cast<uint8_t *>(vpx_memalign(16, padded_dst_size_));
    ASSERT_NE(padded_dst_, nullptr);
    dst_ = padded_dst_ + (kBorderSize * dst_stride_) + kBorderSize;

    dst_c_ = new uint8_t[16 * 16];
    ASSERT_NE(dst_c_, nullptr);

    memset(src_, 0, kSrcSize);
    memset(padded_dst_, 128, padded_dst_size_);
    memset(dst_c_, 0, 16 * 16);
  }

  void TearDown() override {
    delete[] src_;
    src_ = nullptr;
    vpx_free(padded_dst_);
    padded_dst_ = nullptr;
    dst_ = nullptr;
    delete[] dst_c_;
    dst_c_ = nullptr;
    libvpx_test::ClearSystemState();
  }

 protected:
  // Make reference arrays big enough for 16x16 functions. Six-tap filters need
  // 5 extra pixels outside of the macroblock.
  static const int kSrcStride = 21;
  static const int kSrcSize = kSrcStride * kSrcStride;
  static const int kBorderSize = 16;

  int width_;
  int height_;
  PredictFunc predict_;
  uint8_t *src_;
  uint8_t *padded_dst_;
  uint8_t *dst_;
  int padded_dst_size_;
  uint8_t *dst_c_;
  int dst_stride_;

  bool CompareBuffers(const uint8_t *a, int a_stride, const uint8_t *b,
                      int b_stride) const {
    for (int height = 0; height < height_; ++height) {
      EXPECT_EQ(0, memcmp(a + height * a_stride, b + height * b_stride,
                          sizeof(*a) * width_))
          << "Row " << height << " does not match.";
    }

    return !HasFailure();
  }

  // Given a block of memory 'a' with size 'a_size', determine if all regions
  // excepting block 'b' described by 'b_stride', 'b_height', and 'b_width'
  // match pixel value 'c'.
  bool CheckBorder(const uint8_t *a, int a_size, const uint8_t *b, int b_width,
                   int b_height, int b_stride, uint8_t c) const {
    const uint8_t *a_end = a + a_size;
    const int b_size = (b_stride * b_height) + b_width;
    const uint8_t *b_end = b + b_size;
    const int left_border = (b_stride - b_width) / 2;
    const int right_border = left_border + ((b_stride - b_width) % 2);

    EXPECT_GE(b - left_border, a) << "'b' does not start within 'a'";
    EXPECT_LE(b_end + right_border, a_end) << "'b' does not end within 'a'";

    // Top border.
    for (int pixel = 0; pixel < b - a - left_border; ++pixel) {
      EXPECT_EQ(c, a[pixel]) << "Mismatch at " << pixel << " in top border.";
    }

    // Left border.
    for (int height = 0; height < b_height; ++height) {
      for (int width = left_border; width > 0; --width) {
        EXPECT_EQ(c, b[height * b_stride - width])
            << "Mismatch at row " << height << " column " << left_border - width
            << " in left border.";
      }
    }

    // Right border.
    for (int height = 0; height < b_height; ++height) {
      for (int width = b_width; width < b_width + right_border; ++width) {
        EXPECT_EQ(c, b[height * b_stride + width])
            << "Mismatch at row " << height << " column " << width - b_width
            << " in right border.";
      }
    }

    // Bottom border.
    for (int pixel = static_cast<int>(b - a + b_size); pixel < a_size;
         ++pixel) {
      EXPECT_EQ(c, a[pixel]) << "Mismatch at " << pixel << " in bottom border.";
    }

    return !HasFailure();
  }

  void TestWithRandomData(PredictFunc reference) {
    ACMRandom rnd(ACMRandom::DeterministicSeed());

    // Run tests for almost all possible offsets.
    for (int xoffset = 0; xoffset < 8; ++xoffset) {
      for (int yoffset = 0; yoffset < 8; ++yoffset) {
        if (xoffset == 0 && yoffset == 0) {
          // This represents a copy which is not required to be handled by this
          // module.
          continue;
        }

        for (int i = 0; i < kSrcSize; ++i) {
          src_[i] = rnd.Rand8();
        }
        reference(&src_[kSrcStride * 2 + 2], kSrcStride, xoffset, yoffset,
                  dst_c_, 16);

        ASM_REGISTER_STATE_CHECK(predict_(&src_[kSrcStride * 2 + 2], kSrcStride,
                                          xoffset, yoffset, dst_, dst_stride_));

        ASSERT_TRUE(CompareBuffers(dst_c_, 16, dst_, dst_stride_));
        ASSERT_TRUE(CheckBorder(padded_dst_, padded_dst_size_, dst_, width_,
                                height_, dst_stride_, 128));
      }
    }
  }

  void TestWithUnalignedDst(PredictFunc reference) {
    ACMRandom rnd(ACMRandom::DeterministicSeed());

    // Only the 4x4 need to be able to handle unaligned writes.
    if (width_ == 4 && height_ == 4) {
      for (int xoffset = 0; xoffset < 8; ++xoffset) {
        for (int yoffset = 0; yoffset < 8; ++yoffset) {
          if (xoffset == 0 && yoffset == 0) {
            continue;
          }
          for (int i = 0; i < kSrcSize; ++i) {
            src_[i] = rnd.Rand8();
          }
          reference(&src_[kSrcStride * 2 + 2], kSrcStride, xoffset, yoffset,
                    dst_c_, 16);

          for (int i = 1; i < 4; ++i) {
            memset(padded_dst_, 128, padded_dst_size_);

            ASM_REGISTER_STATE_CHECK(predict_(&src_[kSrcStride * 2 + 2],
                                              kSrcStride, xoffset, yoffset,
                                              dst_ + i, dst_stride_ + i));

            ASSERT_TRUE(CompareBuffers(dst_c_, 16, dst_ + i, dst_stride_ + i));
            ASSERT_TRUE(CheckBorder(padded_dst_, padded_dst_size_, dst_ + i,
                                    width_, height_, dst_stride_ + i, 128));
          }
        }
      }
    }
  }

  void Run() override {
    for (int xoffset = 0; xoffset < 8; ++xoffset) {
      for (int yoffset = 0; yoffset < 8; ++yoffset) {
        if (xoffset == 0 && yoffset == 0) {
          continue;
        }

        predict_(&src_[kSrcStride * 2 + 2], kSrcStride, xoffset, yoffset, dst_,
                 dst_stride_);
      }
    }
  }
};  // namespace

class SixtapPredictTest : public PredictTestBase {};

TEST_P(SixtapPredictTest, TestWithRandomData) {
  TestWithRandomData(vp8_sixtap_predict16x16_c);
}
TEST_P(SixtapPredictTest, TestWithUnalignedDst) {
  TestWithUnalignedDst(vp8_sixtap_predict16x16_c);
}

TEST_P(SixtapPredictTest, TestWithPresetData) {
  // Test input
  static const uint8_t kTestData[kSrcSize] = {
    184, 4,   191, 82,  92,  41,  0,   1,   226, 236, 172, 20,  182, 42,  226,
    177, 79,  94,  77,  179, 203, 206, 198, 22,  192, 19,  75,  17,  192, 44,
    233, 120, 48,  168, 203, 141, 210, 203, 143, 180, 184, 59,  201, 110, 102,
    171, 32,  182, 10,  109, 105, 213, 60,  47,  236, 253, 67,  55,  14,  3,
    99,  247, 124, 148, 159, 71,  34,  114, 19,  177, 38,  203, 237, 239, 58,
    83,  155, 91,  10,  166, 201, 115, 124, 5,   163, 104, 2,   231, 160, 16,
    234, 4,   8,   103, 153, 167, 174, 187, 26,  193, 109, 64,  141, 90,  48,
    200, 174, 204, 36,  184, 114, 237, 43,  238, 242, 207, 86,  245, 182, 247,
    6,   161, 251, 14,  8,   148, 182, 182, 79,  208, 120, 188, 17,  6,   23,
    65,  206, 197, 13,  242, 126, 128, 224, 170, 110, 211, 121, 197, 200, 47,
    188, 207, 208, 184, 221, 216, 76,  148, 143, 156, 100, 8,   89,  117, 14,
    112, 183, 221, 54,  197, 208, 180, 69,  176, 94,  180, 131, 215, 121, 76,
    7,   54,  28,  216, 238, 249, 176, 58,  142, 64,  215, 242, 72,  49,  104,
    87,  161, 32,  52,  216, 230, 4,   141, 44,  181, 235, 224, 57,  195, 89,
    134, 203, 144, 162, 163, 126, 156, 84,  185, 42,  148, 145, 29,  221, 194,
    134, 52,  100, 166, 105, 60,  140, 110, 201, 184, 35,  181, 153, 93,  121,
    243, 227, 68,  131, 134, 232, 2,   35,  60,  187, 77,  209, 76,  106, 174,
    15,  241, 227, 115, 151, 77,  175, 36,  187, 121, 221, 223, 47,  118, 61,
    168, 105, 32,  237, 236, 167, 213, 238, 202, 17,  170, 24,  226, 247, 131,
    145, 6,   116, 117, 121, 11,  194, 41,  48,  126, 162, 13,  93,  209, 131,
    154, 122, 237, 187, 103, 217, 99,  60,  200, 45,  78,  115, 69,  49,  106,
    200, 194, 112, 60,  56,  234, 72,  251, 19,  120, 121, 182, 134, 215, 135,
    10,  114, 2,   247, 46,  105, 209, 145, 165, 153, 191, 243, 12,  5,   36,
    119, 206, 231, 231, 11,  32,  209, 83,  27,  229, 204, 149, 155, 83,  109,
    35,  93,  223, 37,  84,  14,  142, 37,  160, 52,  191, 96,  40,  204, 101,
    77,  67,  52,  53,  43,  63,  85,  253, 147, 113, 226, 96,  6,   125, 179,
    115, 161, 17,  83,  198, 101, 98,  85,  139, 3,   137, 75,  99,  178, 23,
    201, 255, 91,  253, 52,  134, 60,  138, 131, 208, 251, 101, 48,  2,   227,
    228, 118, 132, 245, 202, 75,  91,  44,  160, 231, 47,  41,  50,  147, 220,
    74,  92,  219, 165, 89,  16
  };

  // Expected results for xoffset = 2 and yoffset = 2.
  static const int kExpectedDstStride = 16;
  static const uint8_t kExpectedDst[256] = {
    117, 102, 74,  135, 42,  98,  175, 206, 70,  73,  222, 197, 50,  24,  39,
    49,  38,  105, 90,  47,  169, 40,  171, 215, 200, 73,  109, 141, 53,  85,
    177, 164, 79,  208, 124, 89,  212, 18,  81,  145, 151, 164, 217, 153, 91,
    154, 102, 102, 159, 75,  164, 152, 136, 51,  213, 219, 186, 116, 193, 224,
    186, 36,  231, 208, 84,  211, 155, 167, 35,  59,  42,  76,  216, 149, 73,
    201, 78,  149, 184, 100, 96,  196, 189, 198, 188, 235, 195, 117, 129, 120,
    129, 49,  25,  133, 113, 69,  221, 114, 70,  143, 99,  157, 108, 189, 140,
    78,  6,   55,  65,  240, 255, 245, 184, 72,  90,  100, 116, 131, 39,  60,
    234, 167, 33,  160, 88,  185, 200, 157, 159, 176, 127, 151, 138, 102, 168,
    106, 170, 86,  82,  219, 189, 76,  33,  115, 197, 106, 96,  198, 136, 97,
    141, 237, 151, 98,  137, 191, 185, 2,   57,  95,  142, 91,  255, 185, 97,
    137, 76,  162, 94,  173, 131, 193, 161, 81,  106, 72,  135, 222, 234, 137,
    66,  137, 106, 243, 210, 147, 95,  15,  137, 110, 85,  66,  16,  96,  167,
    147, 150, 173, 203, 140, 118, 196, 84,  147, 160, 19,  95,  101, 123, 74,
    132, 202, 82,  166, 12,  131, 166, 189, 170, 159, 85,  79,  66,  57,  152,
    132, 203, 194, 0,   1,   56,  146, 180, 224, 156, 28,  83,  181, 79,  76,
    80,  46,  160, 175, 59,  106, 43,  87,  75,  136, 85,  189, 46,  71,  200,
    90
  };

  ASM_REGISTER_STATE_CHECK(
      predict_(const_cast<uint8_t *>(kTestData) + kSrcStride * 2 + 2,
               kSrcStride, 2, 2, dst_, dst_stride_));

  ASSERT_TRUE(
      CompareBuffers(kExpectedDst, kExpectedDstStride, dst_, dst_stride_));
}

INSTANTIATE_TEST_SUITE_P(
    C, SixtapPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_sixtap_predict16x16_c),
                      make_tuple(8, 8, &vp8_sixtap_predict8x8_c),
                      make_tuple(8, 4, &vp8_sixtap_predict8x4_c),
                      make_tuple(4, 4, &vp8_sixtap_predict4x4_c)));
#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, SixtapPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_sixtap_predict16x16_neon),
                      make_tuple(8, 8, &vp8_sixtap_predict8x8_neon),
                      make_tuple(8, 4, &vp8_sixtap_predict8x4_neon),
                      make_tuple(4, 4, &vp8_sixtap_predict4x4_neon)));
#endif
#if HAVE_MMX
INSTANTIATE_TEST_SUITE_P(
    MMX, SixtapPredictTest,
    ::testing::Values(make_tuple(4, 4, &vp8_sixtap_predict4x4_mmx)));
#endif
#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, SixtapPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_sixtap_predict16x16_sse2),
                      make_tuple(8, 8, &vp8_sixtap_predict8x8_sse2),
                      make_tuple(8, 4, &vp8_sixtap_predict8x4_sse2)));
#endif
#if HAVE_SSSE3
INSTANTIATE_TEST_SUITE_P(
    SSSE3, SixtapPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_sixtap_predict16x16_ssse3),
                      make_tuple(8, 8, &vp8_sixtap_predict8x8_ssse3),
                      make_tuple(8, 4, &vp8_sixtap_predict8x4_ssse3),
                      make_tuple(4, 4, &vp8_sixtap_predict4x4_ssse3)));
#endif
#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(
    MSA, SixtapPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_sixtap_predict16x16_msa),
                      make_tuple(8, 8, &vp8_sixtap_predict8x8_msa),
                      make_tuple(8, 4, &vp8_sixtap_predict8x4_msa),
                      make_tuple(4, 4, &vp8_sixtap_predict4x4_msa)));
#endif

#if HAVE_MMI
INSTANTIATE_TEST_SUITE_P(
    MMI, SixtapPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_sixtap_predict16x16_mmi),
                      make_tuple(8, 8, &vp8_sixtap_predict8x8_mmi),
                      make_tuple(8, 4, &vp8_sixtap_predict8x4_mmi),
                      make_tuple(4, 4, &vp8_sixtap_predict4x4_mmi)));
#endif

#if HAVE_LSX
INSTANTIATE_TEST_SUITE_P(
    LSX, SixtapPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_sixtap_predict16x16_lsx),
                      make_tuple(8, 8, &vp8_sixtap_predict8x8_lsx),
                      make_tuple(4, 4, &vp8_sixtap_predict4x4_lsx)));
#endif

class BilinearPredictTest : public PredictTestBase {};

TEST_P(BilinearPredictTest, TestWithRandomData) {
  TestWithRandomData(vp8_bilinear_predict16x16_c);
}
TEST_P(BilinearPredictTest, TestWithUnalignedDst) {
  TestWithUnalignedDst(vp8_bilinear_predict16x16_c);
}
TEST_P(BilinearPredictTest, DISABLED_Speed) {
  const int kCountSpeedTestBlock = 5000000 / (width_ * height_);
  RunNTimes(kCountSpeedTestBlock);

  char title[16];
  snprintf(title, sizeof(title), "%dx%d", width_, height_);
  PrintMedian(title);
}

INSTANTIATE_TEST_SUITE_P(
    C, BilinearPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_bilinear_predict16x16_c),
                      make_tuple(8, 8, &vp8_bilinear_predict8x8_c),
                      make_tuple(8, 4, &vp8_bilinear_predict8x4_c),
                      make_tuple(4, 4, &vp8_bilinear_predict4x4_c)));
#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, BilinearPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_bilinear_predict16x16_neon),
                      make_tuple(8, 8, &vp8_bilinear_predict8x8_neon),
                      make_tuple(8, 4, &vp8_bilinear_predict8x4_neon),
                      make_tuple(4, 4, &vp8_bilinear_predict4x4_neon)));
#endif
#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, BilinearPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_bilinear_predict16x16_sse2),
                      make_tuple(8, 8, &vp8_bilinear_predict8x8_sse2),
                      make_tuple(8, 4, &vp8_bilinear_predict8x4_sse2),
                      make_tuple(4, 4, &vp8_bilinear_predict4x4_sse2)));
#endif
#if HAVE_SSSE3
INSTANTIATE_TEST_SUITE_P(
    SSSE3, BilinearPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_bilinear_predict16x16_ssse3),
                      make_tuple(8, 8, &vp8_bilinear_predict8x8_ssse3)));
#endif
#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(
    MSA, BilinearPredictTest,
    ::testing::Values(make_tuple(16, 16, &vp8_bilinear_predict16x16_msa),
                      make_tuple(8, 8, &vp8_bilinear_predict8x8_msa),
                      make_tuple(8, 4, &vp8_bilinear_predict8x4_msa),
                      make_tuple(4, 4, &vp8_bilinear_predict4x4_msa)));
#endif
}  // namespace
