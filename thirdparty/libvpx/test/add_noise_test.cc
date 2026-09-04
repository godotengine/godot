/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <math.h>
#include <tuple>

#include "gtest/gtest.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_config.h"
#include "vpx_dsp/postproc.h"
#include "vpx_mem/vpx_mem.h"

namespace {

static const int kNoiseSize = 3072;

using AddNoiseFunc = void (*)(uint8_t *start, const int8_t *noise,
                              int blackclamp, int whiteclamp, int width,
                              int height, int pitch);

using AddNoiseTestFPParam = std::tuple<double, AddNoiseFunc>;

class AddNoiseTest : public ::testing::Test,
                     public ::testing::WithParamInterface<AddNoiseTestFPParam> {
 public:
  void TearDown() override { libvpx_test::ClearSystemState(); }
  ~AddNoiseTest() override = default;
};

double stddev6(char a, char b, char c, char d, char e, char f) {
  const double n = (a + b + c + d + e + f) / 6.0;
  const double v = ((a - n) * (a - n) + (b - n) * (b - n) + (c - n) * (c - n) +
                    (d - n) * (d - n) + (e - n) * (e - n) + (f - n) * (f - n)) /
                   6.0;
  return sqrt(v);
}

TEST_P(AddNoiseTest, CheckNoiseAdded) {
  const int width = 64;
  const int height = 64;
  const int image_size = width * height;
  int8_t noise[kNoiseSize];
  const int clamp = vpx_setup_noise(GET_PARAM(0), noise, kNoiseSize);
  uint8_t *const s =
      reinterpret_cast<uint8_t *>(vpx_calloc(image_size, sizeof(*s)));
  ASSERT_NE(s, nullptr);
  memset(s, 99, image_size * sizeof(*s));

  ASM_REGISTER_STATE_CHECK(
      GET_PARAM(1)(s, noise, clamp, clamp, width, height, width));

  // Check to make sure we don't end up having either the same or no added
  // noise either vertically or horizontally.
  for (int i = 0; i < image_size - 6 * width - 6; ++i) {
    const double hd = stddev6(s[i] - 99, s[i + 1] - 99, s[i + 2] - 99,
                              s[i + 3] - 99, s[i + 4] - 99, s[i + 5] - 99);
    const double vd = stddev6(s[i] - 99, s[i + width] - 99,
                              s[i + 2 * width] - 99, s[i + 3 * width] - 99,
                              s[i + 4 * width] - 99, s[i + 5 * width] - 99);

    EXPECT_NE(hd, 0);
    EXPECT_NE(vd, 0);
  }

  // Initialize pixels in the image to 255 and check for roll over.
  memset(s, 255, image_size);

  ASM_REGISTER_STATE_CHECK(
      GET_PARAM(1)(s, noise, clamp, clamp, width, height, width));

  // Check to make sure don't roll over.
  for (int i = 0; i < image_size; ++i) {
    EXPECT_GT(static_cast<int>(s[i]), clamp) << "i = " << i;
  }

  // Initialize pixels in the image to 0 and check for roll under.
  memset(s, 0, image_size);

  ASM_REGISTER_STATE_CHECK(
      GET_PARAM(1)(s, noise, clamp, clamp, width, height, width));

  // Check to make sure don't roll under.
  for (int i = 0; i < image_size; ++i) {
    EXPECT_LT(static_cast<int>(s[i]), 255 - clamp) << "i = " << i;
  }

  vpx_free(s);
}

TEST_P(AddNoiseTest, CheckCvsAssembly) {
  const int width = 64;
  const int height = 64;
  const int image_size = width * height;
  int8_t noise[kNoiseSize];
  const int clamp = vpx_setup_noise(4.4, noise, kNoiseSize);

  uint8_t *const s = reinterpret_cast<uint8_t *>(vpx_calloc(image_size, 1));
  uint8_t *const d = reinterpret_cast<uint8_t *>(vpx_calloc(image_size, 1));
  ASSERT_NE(s, nullptr);
  ASSERT_NE(d, nullptr);

  memset(s, 99, image_size);
  memset(d, 99, image_size);

  srand(0);
  ASM_REGISTER_STATE_CHECK(
      GET_PARAM(1)(s, noise, clamp, clamp, width, height, width));
  srand(0);
  ASM_REGISTER_STATE_CHECK(
      vpx_plane_add_noise_c(d, noise, clamp, clamp, width, height, width));

  for (int i = 0; i < image_size; ++i) {
    EXPECT_EQ(static_cast<int>(s[i]), static_cast<int>(d[i])) << "i = " << i;
  }

  vpx_free(d);
  vpx_free(s);
}

using std::make_tuple;

INSTANTIATE_TEST_SUITE_P(
    C, AddNoiseTest,
    ::testing::Values(make_tuple(3.25, vpx_plane_add_noise_c),
                      make_tuple(4.4, vpx_plane_add_noise_c)));

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, AddNoiseTest,
    ::testing::Values(make_tuple(3.25, vpx_plane_add_noise_sse2),
                      make_tuple(4.4, vpx_plane_add_noise_sse2)));
#endif

#if HAVE_MSA
INSTANTIATE_TEST_SUITE_P(
    MSA, AddNoiseTest,
    ::testing::Values(make_tuple(3.25, vpx_plane_add_noise_msa),
                      make_tuple(4.4, vpx_plane_add_noise_msa)));
#endif
}  // namespace
