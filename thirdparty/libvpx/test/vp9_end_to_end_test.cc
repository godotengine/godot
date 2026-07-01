/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "memory"

#include "gtest/gtest.h"

#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "test/yuv_video_source.h"
#include "vpx_config.h"

namespace {

const unsigned int kWidth = 160;
const unsigned int kHeight = 90;
const unsigned int kFramerate = 50;
const unsigned int kFrames = 20;
const int kBitrate = 500;
// List of psnr thresholds for speed settings 0-7 and 5 encoding modes
const double kPsnrThreshold[][5] = {
  { 36.0, 37.0, 37.0, 37.0, 37.0 }, { 35.0, 36.0, 36.0, 36.0, 36.0 },
  { 34.0, 35.0, 35.0, 35.0, 35.0 }, { 33.0, 34.0, 34.0, 34.0, 34.0 },
  { 32.0, 33.0, 33.0, 33.0, 33.0 }, { 28.0, 32.0, 32.0, 32.0, 32.0 },
  { 28.4, 31.0, 31.0, 31.0, 31.0 }, { 27.5, 30.0, 30.0, 30.0, 30.0 },
};

struct TestVideoParam {
  const char *filename;
  unsigned int input_bit_depth;
  vpx_img_fmt fmt;
  vpx_bit_depth_t bit_depth;
  unsigned int profile;
};

const TestVideoParam kTestVectors[] = {
  { "park_joy_90p_8_420.y4m", 8, VPX_IMG_FMT_I420, VPX_BITS_8, 0 },
  { "park_joy_90p_8_422.y4m", 8, VPX_IMG_FMT_I422, VPX_BITS_8, 1 },
  { "park_joy_90p_8_444.y4m", 8, VPX_IMG_FMT_I444, VPX_BITS_8, 1 },
  { "park_joy_90p_8_440.yuv", 8, VPX_IMG_FMT_I440, VPX_BITS_8, 1 },
#if CONFIG_VP9_HIGHBITDEPTH
  { "park_joy_90p_10_420_20f.y4m", 10, VPX_IMG_FMT_I42016, VPX_BITS_10, 2 },
  { "park_joy_90p_10_422_20f.y4m", 10, VPX_IMG_FMT_I42216, VPX_BITS_10, 3 },
  { "park_joy_90p_10_444_20f.y4m", 10, VPX_IMG_FMT_I44416, VPX_BITS_10, 3 },
  { "park_joy_90p_10_440.yuv", 10, VPX_IMG_FMT_I44016, VPX_BITS_10, 3 },
  { "park_joy_90p_12_420_20f.y4m", 12, VPX_IMG_FMT_I42016, VPX_BITS_12, 2 },
  { "park_joy_90p_12_422_20f.y4m", 12, VPX_IMG_FMT_I42216, VPX_BITS_12, 3 },
  { "park_joy_90p_12_444_20f.y4m", 12, VPX_IMG_FMT_I44416, VPX_BITS_12, 3 },
  { "park_joy_90p_12_440.yuv", 12, VPX_IMG_FMT_I44016, VPX_BITS_12, 3 },
#endif  // CONFIG_VP9_HIGHBITDEPTH
};

const TestVideoParam kTestVectorsNv12[] = {
  { "hantro_collage_w352h288_nv12.yuv", 8, VPX_IMG_FMT_NV12, VPX_BITS_8, 0 },
};

const TestVideoParam k4x2VideoTestVectors[] = {
  { "4x2.y4m", 8, VPX_IMG_FMT_I420, VPX_BITS_8, 0 },
};

// Encoding modes tested
const libvpx_test::TestMode kEncodingModeVectors[] = {
#if !CONFIG_REALTIME_ONLY
  ::libvpx_test::kTwoPassGood, ::libvpx_test::kOnePassGood,
#endif
  ::libvpx_test::kRealTime
};

// Speed settings tested
const int kCpuUsedVectors[] = { 1, 2, 3, 5, 6, 7 };

int is_extension_y4m(const char *filename) {
  const char *dot = strrchr(filename, '.');
  if (!dot || dot == filename) {
    return 0;
  } else {
    return !strcmp(dot, ".y4m");
  }
}

class EndToEndTestAdaptiveRDThresh
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<int, int> {
 protected:
  EndToEndTestAdaptiveRDThresh()
      : EncoderTest(GET_PARAM(0)), cpu_used_start_(GET_PARAM(1)),
        cpu_used_end_(GET_PARAM(2)) {}

  ~EndToEndTestAdaptiveRDThresh() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    cfg_.g_lag_in_frames = 0;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 600;
    dec_cfg_.threads = 4;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, cpu_used_start_);
      encoder->Control(VP9E_SET_ROW_MT, 1);
      encoder->Control(VP9E_SET_TILE_COLUMNS, 2);
    }
    if (video->frame() == 100)
      encoder->Control(VP8E_SET_CPUUSED, cpu_used_end_);
  }

 private:
  int cpu_used_start_;
  int cpu_used_end_;
};

class EndToEndTestLarge
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith3Params<libvpx_test::TestMode,
                                                 TestVideoParam, int> {
 protected:
  EndToEndTestLarge()
      : EncoderTest(GET_PARAM(0)), test_video_param_(GET_PARAM(2)),
        cpu_used_(GET_PARAM(3)), psnr_(0.0), nframes_(0),
        encoding_mode_(GET_PARAM(1)) {
    cyclic_refresh_ = 0;
    denoiser_on_ = 0;
  }

  ~EndToEndTestLarge() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);
    if (encoding_mode_ != ::libvpx_test::kRealTime) {
      cfg_.g_lag_in_frames = 5;
      cfg_.rc_end_usage = VPX_VBR;
    } else {
      cfg_.g_lag_in_frames = 0;
      cfg_.rc_end_usage = VPX_CBR;
      cfg_.rc_buf_sz = 1000;
      cfg_.rc_buf_initial_sz = 500;
      cfg_.rc_buf_optimal_sz = 600;
    }
    dec_cfg_.threads = 4;
  }

  void BeginPassHook(unsigned int) override {
    psnr_ = 0.0;
    nframes_ = 0;
  }

  void PSNRPktHook(const vpx_codec_cx_pkt_t *pkt) override {
    psnr_ += pkt->data.psnr.psnr[0];
    nframes_++;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP9E_SET_FRAME_PARALLEL_DECODING, 1);
      encoder->Control(VP9E_SET_TILE_COLUMNS, 4);
      encoder->Control(VP8E_SET_CPUUSED, cpu_used_);
      if (encoding_mode_ != ::libvpx_test::kRealTime) {
        encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
        encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
        encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
        encoder->Control(VP8E_SET_ARNR_TYPE, 3);
      } else {
        encoder->Control(VP9E_SET_NOISE_SENSITIVITY, denoiser_on_);
        encoder->Control(VP9E_SET_AQ_MODE, cyclic_refresh_);
      }
    }
  }

  double GetAveragePsnr() const {
    if (nframes_) return psnr_ / nframes_;
    return 0.0;
  }

  double GetPsnrThreshold() {
    return kPsnrThreshold[cpu_used_][encoding_mode_];
  }

  TestVideoParam test_video_param_;
  int cpu_used_;
  int cyclic_refresh_;
  int denoiser_on_;

 private:
  double psnr_;
  unsigned int nframes_;
  libvpx_test::TestMode encoding_mode_;
};

#if CONFIG_VP9_DECODER
// The test parameters control VP9D_SET_LOOP_FILTER_OPT and the number of
// decoder threads.
class EndToEndTestLoopFilterThreading
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<bool, int> {
 protected:
  EndToEndTestLoopFilterThreading()
      : EncoderTest(GET_PARAM(0)), use_loop_filter_opt_(GET_PARAM(1)) {}

  ~EndToEndTestLoopFilterThreading() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    cfg_.g_threads = 2;
    cfg_.g_lag_in_frames = 0;
    cfg_.rc_target_bitrate = 500;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.kf_min_dist = 1;
    cfg_.kf_max_dist = 1;
    dec_cfg_.threads = GET_PARAM(2);
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, 8);
    }
    encoder->Control(VP9E_SET_TILE_COLUMNS, 4 - video->frame() % 5);
  }

  void PreDecodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Decoder *decoder) override {
    if (video->frame() == 0) {
      decoder->Control(VP9D_SET_LOOP_FILTER_OPT, use_loop_filter_opt_ ? 1 : 0);
    }
  }

 private:
  const bool use_loop_filter_opt_;
};
#endif  // CONFIG_VP9_DECODER

class EndToEndNV12 : public EndToEndTestLarge {};

TEST_P(EndToEndNV12, EndtoEndNV12Test) {
  cfg_.rc_target_bitrate = kBitrate;
  cfg_.g_error_resilient = 0;
  cfg_.g_profile = test_video_param_.profile;
  cfg_.g_input_bit_depth = test_video_param_.input_bit_depth;
  cfg_.g_bit_depth = test_video_param_.bit_depth;
  init_flags_ = VPX_CODEC_USE_PSNR;
  if (cfg_.g_bit_depth > 8) init_flags_ |= VPX_CODEC_USE_HIGHBITDEPTH;

  std::unique_ptr<libvpx_test::VideoSource> video;

  video.reset(new libvpx_test::YUVVideoSource(test_video_param_.filename,
                                              test_video_param_.fmt, 352, 288,
                                              30, 1, 0, 100));
  ASSERT_NE(video.get(), nullptr);

  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
}

class EndToEnd4x2Video : public EndToEndTestLarge {};

TEST_P(EndToEnd4x2Video, EndtoEnd4x2VideoTest) {
  cfg_.rc_target_bitrate = kBitrate;
  cfg_.g_error_resilient = 0;
  cfg_.g_profile = test_video_param_.profile;
  cfg_.g_input_bit_depth = test_video_param_.input_bit_depth;
  cfg_.g_bit_depth = test_video_param_.bit_depth;

  std::unique_ptr<libvpx_test::VideoSource> video;

  video.reset(
      new libvpx_test::Y4mVideoSource(test_video_param_.filename, 0, 200));
  ASSERT_NE(video.get(), nullptr);

  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
}

TEST_P(EndToEndTestLarge, EndtoEndPSNRTest) {
  cfg_.rc_target_bitrate = kBitrate;
  cfg_.g_error_resilient = 0;
  cfg_.g_profile = test_video_param_.profile;
  cfg_.g_input_bit_depth = test_video_param_.input_bit_depth;
  cfg_.g_bit_depth = test_video_param_.bit_depth;
  init_flags_ = VPX_CODEC_USE_PSNR;
  if (cfg_.g_bit_depth > 8) init_flags_ |= VPX_CODEC_USE_HIGHBITDEPTH;

  std::unique_ptr<libvpx_test::VideoSource> video;
  if (is_extension_y4m(test_video_param_.filename)) {
    video.reset(new libvpx_test::Y4mVideoSource(test_video_param_.filename, 0,
                                                kFrames));
  } else {
    video.reset(new libvpx_test::YUVVideoSource(
        test_video_param_.filename, test_video_param_.fmt, kWidth, kHeight,
        kFramerate, 1, 0, kFrames));
  }
  ASSERT_NE(video.get(), nullptr);

  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
  const double psnr = GetAveragePsnr();
  EXPECT_GT(psnr, GetPsnrThreshold());
}

TEST_P(EndToEndTestLarge, EndtoEndPSNRDenoiserAQTest) {
  cfg_.rc_target_bitrate = kBitrate;
  cfg_.g_error_resilient = 0;
  cfg_.g_profile = test_video_param_.profile;
  cfg_.g_input_bit_depth = test_video_param_.input_bit_depth;
  cfg_.g_bit_depth = test_video_param_.bit_depth;
  init_flags_ = VPX_CODEC_USE_PSNR;
  cyclic_refresh_ = 3;
  denoiser_on_ = 1;
  if (cfg_.g_bit_depth > 8) init_flags_ |= VPX_CODEC_USE_HIGHBITDEPTH;

  std::unique_ptr<libvpx_test::VideoSource> video;
  if (is_extension_y4m(test_video_param_.filename)) {
    video.reset(new libvpx_test::Y4mVideoSource(test_video_param_.filename, 0,
                                                kFrames));
  } else {
    video.reset(new libvpx_test::YUVVideoSource(
        test_video_param_.filename, test_video_param_.fmt, kWidth, kHeight,
        kFramerate, 1, 0, kFrames));
  }
  ASSERT_NE(video.get(), nullptr);

  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
  const double psnr = GetAveragePsnr();
  EXPECT_GT(psnr, GetPsnrThreshold());
}

TEST_P(EndToEndTestAdaptiveRDThresh, EndtoEndAdaptiveRDThreshRowMT) {
  cfg_.rc_target_bitrate = kBitrate;
  cfg_.g_error_resilient = 0;
  cfg_.g_threads = 2;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

#if CONFIG_VP9_DECODER
TEST_P(EndToEndTestLoopFilterThreading, TileCountChange) {
  ::libvpx_test::RandomVideoSource video;
  video.SetSize(4096, 2160);
  video.set_limit(10);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}
#endif  // CONFIG_VP9_DECODER

VP9_INSTANTIATE_TEST_SUITE(EndToEndTestLarge,
                           ::testing::ValuesIn(kEncodingModeVectors),
                           ::testing::ValuesIn(kTestVectors),
                           ::testing::ValuesIn(kCpuUsedVectors));

VP9_INSTANTIATE_TEST_SUITE(EndToEndNV12,
                           ::testing::Values(::libvpx_test::kRealTime),
                           ::testing::ValuesIn(kTestVectorsNv12),
                           ::testing::Values(6, 7, 8));

VP9_INSTANTIATE_TEST_SUITE(EndToEnd4x2Video,
                           ::testing::Values(::libvpx_test::kTwoPassGood),
                           ::testing::ValuesIn(k4x2VideoTestVectors),
                           ::testing::Values(0, 1));

VP9_INSTANTIATE_TEST_SUITE(EndToEndTestAdaptiveRDThresh,
                           ::testing::Values(5, 6, 7), ::testing::Values(8, 9));

#if CONFIG_VP9_DECODER
VP9_INSTANTIATE_TEST_SUITE(EndToEndTestLoopFilterThreading, ::testing::Bool(),
                           ::testing::Range(2, 6));
#endif  // CONFIG_VP9_DECODER
}  // namespace
