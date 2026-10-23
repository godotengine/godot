/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
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
#include "test/y4m_video_source.h"

namespace {

const int kMaxPSNR = 100;

class CpuSpeedTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<libvpx_test::TestMode, int> {
 protected:
  CpuSpeedTest()
      : EncoderTest(GET_PARAM(0)), encoding_mode_(GET_PARAM(1)),
        set_cpu_used_(GET_PARAM(2)), min_psnr_(kMaxPSNR),
        tune_content_(VP9E_CONTENT_DEFAULT) {}
  ~CpuSpeedTest() override = default;

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
  }

  void BeginPassHook(unsigned int /*pass*/) override { min_psnr_ = kMaxPSNR; }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, set_cpu_used_);
      encoder->Control(VP9E_SET_TUNE_CONTENT, tune_content_);
      if (encoding_mode_ != ::libvpx_test::kRealTime) {
        encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
        encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
        encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
        encoder->Control(VP8E_SET_ARNR_TYPE, 3);
      }
    }
  }

  void PSNRPktHook(const vpx_codec_cx_pkt_t *pkt) override {
    if (pkt->data.psnr.psnr[0] < min_psnr_) min_psnr_ = pkt->data.psnr.psnr[0];
  }

  ::libvpx_test::TestMode encoding_mode_;
  int set_cpu_used_;
  double min_psnr_;
  int tune_content_;
};

TEST_P(CpuSpeedTest, TestQ0) {
  // Validate that this non multiple of 64 wide clip encodes and decodes
  // without a mismatch when passing in a very low max q.  This pushes
  // the encoder to producing lots of big partitions which will likely
  // extend into the border and test the border condition.
  cfg_.rc_2pass_vbr_minsection_pct = 5;
  cfg_.rc_2pass_vbr_maxsection_pct = 2000;
  cfg_.rc_target_bitrate = 400;
  cfg_.rc_max_quantizer = 0;
  cfg_.rc_min_quantizer = 0;

  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0,
                                       20);

  init_flags_ = VPX_CODEC_USE_PSNR;

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  EXPECT_GE(min_psnr_, kMaxPSNR);
}

TEST_P(CpuSpeedTest, TestScreencastQ0) {
  ::libvpx_test::Y4mVideoSource video("screendata.y4m", 0, 25);
  cfg_.g_timebase = video.timebase();
  cfg_.rc_2pass_vbr_minsection_pct = 5;
  cfg_.rc_2pass_vbr_maxsection_pct = 2000;
  cfg_.rc_target_bitrate = 400;
  cfg_.rc_max_quantizer = 0;
  cfg_.rc_min_quantizer = 0;

  init_flags_ = VPX_CODEC_USE_PSNR;

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  EXPECT_GE(min_psnr_, kMaxPSNR);
}

TEST_P(CpuSpeedTest, TestTuneScreen) {
  ::libvpx_test::Y4mVideoSource video("screendata.y4m", 0, 25);
  cfg_.g_timebase = video.timebase();
  cfg_.rc_2pass_vbr_minsection_pct = 5;
  cfg_.rc_2pass_vbr_maxsection_pct = 2000;
  cfg_.rc_target_bitrate = 2000;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_min_quantizer = 0;
  tune_content_ = VP9E_CONTENT_SCREEN;

  init_flags_ = VPX_CODEC_USE_PSNR;

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(CpuSpeedTest, TestEncodeHighBitrate) {
  // Validate that this non multiple of 64 wide clip encodes and decodes
  // without a mismatch when passing in a very low max q.  This pushes
  // the encoder to producing lots of big partitions which will likely
  // extend into the border and test the border condition.
  cfg_.rc_2pass_vbr_minsection_pct = 5;
  cfg_.rc_2pass_vbr_maxsection_pct = 2000;
  cfg_.rc_target_bitrate = 12000;
  cfg_.rc_max_quantizer = 10;
  cfg_.rc_min_quantizer = 0;

  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0,
                                       20);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(CpuSpeedTest, TestLowBitrate) {
  // Validate that this clip encodes and decodes without a mismatch
  // when passing in a very high min q.  This pushes the encoder to producing
  // lots of small partitions which might will test the other condition.
  cfg_.rc_2pass_vbr_minsection_pct = 5;
  cfg_.rc_2pass_vbr_maxsection_pct = 2000;
  cfg_.rc_target_bitrate = 200;
  cfg_.rc_min_quantizer = 40;

  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0,
                                       20);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

VP9_INSTANTIATE_TEST_SUITE(CpuSpeedTest, ONE_PASS_TEST_MODES,
                           ::testing::Range(0, 10));
}  // namespace
