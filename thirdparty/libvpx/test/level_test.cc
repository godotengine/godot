/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "vpx_config.h"

namespace {
class LevelTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<libvpx_test::TestMode, int> {
 protected:
  LevelTest()
      : EncoderTest(GET_PARAM(0)), encoding_mode_(GET_PARAM(1)),
        cpu_used_(GET_PARAM(2)), min_gf_internal_(24), target_level_(0),
        level_(0) {}
  ~LevelTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);
    if (encoding_mode_ != ::libvpx_test::kRealTime) {
      cfg_.g_lag_in_frames = 25;
      cfg_.rc_end_usage = VPX_VBR;
    } else {
      cfg_.g_lag_in_frames = 0;
      cfg_.rc_end_usage = VPX_CBR;
    }
    cfg_.rc_2pass_vbr_minsection_pct = 5;
    cfg_.rc_2pass_vbr_maxsection_pct = 2000;
    cfg_.rc_target_bitrate = 400;
    cfg_.rc_max_quantizer = 63;
    cfg_.rc_min_quantizer = 0;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, cpu_used_);
      encoder->Control(VP9E_SET_TARGET_LEVEL, target_level_);
      encoder->Control(VP9E_SET_MIN_GF_INTERVAL, min_gf_internal_);
      if (encoding_mode_ != ::libvpx_test::kRealTime) {
        encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
        encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
        encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
        encoder->Control(VP8E_SET_ARNR_TYPE, 3);
      }
    }
    encoder->Control(VP9E_GET_LEVEL, &level_);
    ASSERT_LE(level_, 51);
    ASSERT_GE(level_, 0);
  }

  ::libvpx_test::TestMode encoding_mode_;
  int cpu_used_;
  int min_gf_internal_;
  int target_level_;
  int level_;
};

TEST_P(LevelTest, TestTargetLevel11Large) {
#if CONFIG_REALTIME_ONLY
  GTEST_SKIP();
#else
  ASSERT_NE(encoding_mode_, ::libvpx_test::kRealTime);
  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0,
                                       60);
  target_level_ = 11;
  cfg_.rc_target_bitrate = 150;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(target_level_, level_);
#endif
}

TEST_P(LevelTest, TestTargetLevel20Large) {
#if CONFIG_REALTIME_ONLY
  GTEST_SKIP();
#else
  ASSERT_NE(encoding_mode_, ::libvpx_test::kRealTime);
  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 60);
  target_level_ = 20;
  cfg_.rc_target_bitrate = 1200;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(target_level_, level_);
#endif
}

TEST_P(LevelTest, TestTargetLevel31Large) {
#if CONFIG_REALTIME_ONLY
  GTEST_SKIP();
#else
  ASSERT_NE(encoding_mode_, ::libvpx_test::kRealTime);
  ::libvpx_test::I420VideoSource video("niklas_1280_720_30.y4m", 1280, 720, 30,
                                       1, 0, 60);
  target_level_ = 31;
  cfg_.rc_target_bitrate = 8000;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(target_level_, level_);
#endif
}

// Test for keeping level stats only
TEST_P(LevelTest, TestTargetLevel0) {
  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0,
                                       40);
  target_level_ = 0;
  min_gf_internal_ = 4;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(11, level_);

  cfg_.rc_target_bitrate = 1600;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(20, level_);
}

// Test for level control being turned off
TEST_P(LevelTest, TestTargetLevel255) {
  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0,
                                       30);
  target_level_ = 255;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(LevelTest, TestTargetLevelApi) {
  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0, 1);
  static vpx_codec_iface_t *codec = &vpx_codec_vp9_cx_algo;
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;
  EXPECT_EQ(VPX_CODEC_OK, vpx_codec_enc_config_default(codec, &cfg, 0));
  cfg.rc_target_bitrate = 100;
  EXPECT_EQ(VPX_CODEC_OK, vpx_codec_enc_init(&enc, codec, &cfg, 0));
  for (int level = 0; level <= 256; ++level) {
    if (level == 10 || level == 11 || level == 20 || level == 21 ||
        level == 30 || level == 31 || level == 40 || level == 41 ||
        level == 50 || level == 51 || level == 52 || level == 60 ||
        level == 61 || level == 62 || level == 0 || level == 1 || level == 255)
      EXPECT_EQ(VPX_CODEC_OK,
                vpx_codec_control(&enc, VP9E_SET_TARGET_LEVEL, level));
    else
      EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
                vpx_codec_control(&enc, VP9E_SET_TARGET_LEVEL, level));
  }
  EXPECT_EQ(VPX_CODEC_OK, vpx_codec_destroy(&enc));
}

VP9_INSTANTIATE_TEST_SUITE(LevelTest, ONE_OR_TWO_PASS_TEST_MODES,
                           ::testing::Range(0, 9));
}  // namespace
