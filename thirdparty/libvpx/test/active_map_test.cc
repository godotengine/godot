/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <climits>
#include <vector>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"

namespace {

class ActiveMapTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith3Params<libvpx_test::TestMode, int,
                                                 int> {
 protected:
  static const int kWidth = 208;
  static const int kHeight = 144;

  ActiveMapTest() : EncoderTest(GET_PARAM(0)) {}
  ~ActiveMapTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
    cpu_used_ = GET_PARAM(2);
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, cpu_used_);
      encoder->Control(VP9E_SET_AQ_MODE, GET_PARAM(3));
    } else if (video->frame() == 3) {
      vpx_active_map_t map = vpx_active_map_t();
      /* clang-format off */
      uint8_t active_map[9 * 13] = {
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
      };
      /* clang-format on */
      map.cols = (kWidth + 15) / 16;
      map.rows = (kHeight + 15) / 16;
      ASSERT_EQ(map.cols, 13u);
      ASSERT_EQ(map.rows, 9u);
      map.active_map = active_map;
      encoder->Control(VP8E_SET_ACTIVEMAP, &map);
    } else if (video->frame() == 15) {
      vpx_active_map_t map = vpx_active_map_t();
      map.cols = (kWidth + 15) / 16;
      map.rows = (kHeight + 15) / 16;
      map.active_map = nullptr;
      encoder->Control(VP8E_SET_ACTIVEMAP, &map);
    }
  }

  int cpu_used_;
};

TEST_P(ActiveMapTest, Test) {
  // Validate that this non multiple of 64 wide clip encodes
  cfg_.g_lag_in_frames = 0;
  cfg_.rc_target_bitrate = 400;
  cfg_.rc_resize_allowed = 0;
  cfg_.g_pass = VPX_RC_ONE_PASS;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.kf_max_dist = 90000;

  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", kWidth, kHeight, 30, 1,
                                       0, 20);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

VP9_INSTANTIATE_TEST_SUITE(ActiveMapTest,
                           ::testing::Values(::libvpx_test::kRealTime),
                           ::testing::Range(5, 10), ::testing::Values(0, 3));
}  // namespace
