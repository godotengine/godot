/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
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
#include "test/y4m_video_source.h"
#include "test/yuv_video_source.h"
#include "vp9/encoder/vp9_ratectrl.h"
#include "vpx_config.h"

namespace {

const unsigned int kFrames = 100;
const int kBitrate = 500;

#define ARF_NOT_SEEN 1000001
#define ARF_SEEN_ONCE 1000000

struct TestVideoParam {
  const char *filename;
  unsigned int width;
  unsigned int height;
  unsigned int framerate_num;
  unsigned int framerate_den;
  unsigned int input_bit_depth;
  vpx_img_fmt fmt;
  vpx_bit_depth_t bit_depth;
  unsigned int profile;
};

struct TestEncodeParam {
  libvpx_test::TestMode mode;
  int cpu_used;
};

const TestVideoParam kTestVectors[] = {
  // artificially increase framerate to trigger default check
  { "hantro_collage_w352h288.yuv", 352, 288, 5000, 1, 8, VPX_IMG_FMT_I420,
    VPX_BITS_8, 0 },
  { "hantro_collage_w352h288.yuv", 352, 288, 30, 1, 8, VPX_IMG_FMT_I420,
    VPX_BITS_8, 0 },
  { "rush_hour_444.y4m", 352, 288, 30, 1, 8, VPX_IMG_FMT_I444, VPX_BITS_8, 1 },
#if CONFIG_VP9_HIGHBITDEPTH
// Add list of profile 2/3 test videos here ...
#endif  // CONFIG_VP9_HIGHBITDEPTH
};

const TestEncodeParam kEncodeVectors[] = {
  { ::libvpx_test::kOnePassGood, 2 }, { ::libvpx_test::kOnePassGood, 5 },
  { ::libvpx_test::kTwoPassGood, 1 }, { ::libvpx_test::kTwoPassGood, 2 },
  { ::libvpx_test::kTwoPassGood, 5 }, { ::libvpx_test::kRealTime, 5 },
};

const int kMinArfVectors[] = {
  // NOTE: 0 refers to the default built-in logic in:
  //       vp9_rc_get_default_min_gf_interval(...)
  0, 4, 8, 12, 15
};

int is_extension_y4m(const char *filename) {
  const char *dot = strrchr(filename, '.');
  if (!dot || dot == filename) {
    return 0;
  } else {
    return !strcmp(dot, ".y4m");
  }
}

class ArfFreqTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith3Params<TestVideoParam,
                                                 TestEncodeParam, int> {
 protected:
  ArfFreqTest()
      : EncoderTest(GET_PARAM(0)), test_video_param_(GET_PARAM(1)),
        test_encode_param_(GET_PARAM(2)), min_arf_requested_(GET_PARAM(3)) {}

  ~ArfFreqTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(test_encode_param_.mode);
    if (test_encode_param_.mode != ::libvpx_test::kRealTime) {
      cfg_.g_lag_in_frames = 25;
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
    min_run_ = ARF_NOT_SEEN;
    run_of_visible_frames_ = 0;
  }

  int GetNumFramesInPkt(const vpx_codec_cx_pkt_t *pkt) {
    const uint8_t *buffer = reinterpret_cast<uint8_t *>(pkt->data.frame.buf);
    const uint8_t marker = buffer[pkt->data.frame.sz - 1];
    const int mag = ((marker >> 3) & 3) + 1;
    int frames = (marker & 0x7) + 1;
    const unsigned int index_sz = 2 + mag * frames;
    // Check for superframe or not.
    // Assume superframe has only one visible frame, the rest being
    // invisible. If superframe index is not found, then there is only
    // one frame.
    if (!((marker & 0xe0) == 0xc0 && pkt->data.frame.sz >= index_sz &&
          buffer[pkt->data.frame.sz - index_sz] == marker)) {
      frames = 1;
    }
    return frames;
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    if (pkt->kind != VPX_CODEC_CX_FRAME_PKT) return;
    const int frames = GetNumFramesInPkt(pkt);
    if (frames == 1) {
      run_of_visible_frames_++;
    } else if (frames == 2) {
      if (min_run_ == ARF_NOT_SEEN) {
        min_run_ = ARF_SEEN_ONCE;
      } else if (min_run_ == ARF_SEEN_ONCE ||
                 run_of_visible_frames_ < min_run_) {
        min_run_ = run_of_visible_frames_;
      }
      run_of_visible_frames_ = 1;
    } else {
      min_run_ = 0;
      run_of_visible_frames_ = 1;
    }
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP9E_SET_FRAME_PARALLEL_DECODING, 1);
      encoder->Control(VP9E_SET_TILE_COLUMNS, 4);
      encoder->Control(VP8E_SET_CPUUSED, test_encode_param_.cpu_used);
      encoder->Control(VP9E_SET_MIN_GF_INTERVAL, min_arf_requested_);
      if (test_encode_param_.mode != ::libvpx_test::kRealTime) {
        encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
        encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
        encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
        encoder->Control(VP8E_SET_ARNR_TYPE, 3);
      }
    }
  }

  int GetMinVisibleRun() const { return min_run_; }

  int GetMinArfDistanceRequested() const {
    if (min_arf_requested_) {
      return min_arf_requested_;
    } else {
      return vp9_rc_get_default_min_gf_interval(
          test_video_param_.width, test_video_param_.height,
          (double)test_video_param_.framerate_num /
              test_video_param_.framerate_den);
    }
  }

  TestVideoParam test_video_param_;
  TestEncodeParam test_encode_param_;

 private:
  int min_arf_requested_;
  int min_run_;
  int run_of_visible_frames_;
};

TEST_P(ArfFreqTest, MinArfFreqTest) {
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
        test_video_param_.filename, test_video_param_.fmt,
        test_video_param_.width, test_video_param_.height,
        test_video_param_.framerate_num, test_video_param_.framerate_den, 0,
        kFrames));
  }

  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
  const int min_run = GetMinVisibleRun();
  const int min_arf_dist_requested = GetMinArfDistanceRequested();
  if (min_run != ARF_NOT_SEEN && min_run != ARF_SEEN_ONCE) {
    const int min_arf_dist = min_run + 1;
    EXPECT_GE(min_arf_dist, min_arf_dist_requested);
  }
}

VP9_INSTANTIATE_TEST_SUITE(ArfFreqTest, ::testing::ValuesIn(kTestVectors),
                           ::testing::ValuesIn(kEncodeVectors),
                           ::testing::ValuesIn(kMinArfVectors));
}  // namespace
