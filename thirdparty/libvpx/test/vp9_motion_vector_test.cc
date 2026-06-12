/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <memory>

#include "gtest/gtest.h"

#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/yuv_video_source.h"
#include "vpx_config.h"

namespace {
#define MAX_EXTREME_MV 1
#define MIN_EXTREME_MV 2

// Encoding modes
const libvpx_test::TestMode kEncodingModeVectors[] = {
#if !CONFIG_REALTIME_ONLY
  ::libvpx_test::kTwoPassGood, ::libvpx_test::kOnePassGood,
#endif
  ::libvpx_test::kRealTime
};

// Encoding speeds
const int kCpuUsedVectors[] = { 0, 1, 2, 3, 4, 5, 6 };

// MV test modes: 1 - always use maximum MV; 2 - always use minimum MV.
const int kMVTestModes[] = { MAX_EXTREME_MV, MIN_EXTREME_MV };

class MotionVectorTestLarge
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith3Params<libvpx_test::TestMode, int,
                                                 int> {
 protected:
  MotionVectorTestLarge()
      : EncoderTest(GET_PARAM(0)), encoding_mode_(GET_PARAM(1)),
        cpu_used_(GET_PARAM(2)), mv_test_mode_(GET_PARAM(3)) {}

  ~MotionVectorTestLarge() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);
    if (encoding_mode_ != ::libvpx_test::kRealTime) {
      cfg_.g_lag_in_frames = 3;
      cfg_.rc_end_usage = VPX_VBR;
    } else {
      cfg_.g_lag_in_frames = 0;
      cfg_.rc_end_usage = VPX_CBR;
      cfg_.rc_buf_sz = 1000;
      cfg_.rc_buf_initial_sz = 500;
      cfg_.rc_buf_optimal_sz = 600;
    }
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, cpu_used_);
      encoder->Control(VP9E_ENABLE_MOTION_VECTOR_UNIT_TEST, mv_test_mode_);
      if (encoding_mode_ != ::libvpx_test::kRealTime) {
        encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
        encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
        encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
        encoder->Control(VP8E_SET_ARNR_TYPE, 3);
      }
    }
  }

  libvpx_test::TestMode encoding_mode_;
  int cpu_used_;
  int mv_test_mode_;
};

TEST_P(MotionVectorTestLarge, OverallTest) {
  cfg_.rc_target_bitrate = 24000;
  cfg_.g_profile = 0;
  init_flags_ = VPX_CODEC_USE_PSNR;

  std::unique_ptr<libvpx_test::VideoSource> video;
  video.reset(new libvpx_test::YUVVideoSource(
      "niklas_640_480_30.yuv", VPX_IMG_FMT_I420, 3840, 2160,  // 2048, 1080,
      30, 1, 0, 5));

  ASSERT_NE(video.get(), nullptr);
  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
}

VP9_INSTANTIATE_TEST_SUITE(MotionVectorTestLarge,
                           ::testing::ValuesIn(kEncodingModeVectors),
                           ::testing::ValuesIn(kCpuUsedVectors),
                           ::testing::ValuesIn(kMVTestModes));
}  // namespace
