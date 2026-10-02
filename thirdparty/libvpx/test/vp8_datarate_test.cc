/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "./vpx_config.h"
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "vpx/vpx_encoder.h"

namespace {

class DatarateTestLarge
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<libvpx_test::TestMode, int> {
 public:
  DatarateTestLarge() : EncoderTest(GET_PARAM(0)) {}

  ~DatarateTestLarge() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
    set_cpu_used_ = GET_PARAM(2);
    ResetModel();
  }

  void ResetModel() {
    last_pts_ = 0;
    bits_in_buffer_model_ = cfg_.rc_target_bitrate * cfg_.rc_buf_initial_sz;
    frame_number_ = 0;
    first_drop_ = 0;
    bits_total_ = 0;
    duration_ = 0.0;
    // Denoiser is off by default.
    denoiser_on_ = 0;
    denoiser_offon_test_ = 0;
    denoiser_offon_period_ = -1;
    gf_boost_ = 0;
    use_roi_ = false;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_NOISE_SENSITIVITY, denoiser_on_);
      encoder->Control(VP8E_SET_CPUUSED, set_cpu_used_);
      encoder->Control(VP8E_SET_GF_CBR_BOOST_PCT, gf_boost_);
    }

    if (use_roi_) {
      encoder->Control(VP8E_SET_ROI_MAP, &roi_);
    }

    if (denoiser_offon_test_) {
      ASSERT_GT(denoiser_offon_period_, 0)
          << "denoiser_offon_period_ is not positive.";
      if ((video->frame() + 1) % denoiser_offon_period_ == 0) {
        // Flip denoiser_on_ periodically
        denoiser_on_ ^= 1;
      }
      encoder->Control(VP8E_SET_NOISE_SENSITIVITY, denoiser_on_);
    }

    const vpx_rational_t tb = video->timebase();
    timebase_ = static_cast<double>(tb.num) / tb.den;
    duration_ = 0;
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    // Time since last timestamp = duration.
    vpx_codec_pts_t duration = pkt->data.frame.pts - last_pts_;

    // TODO(jimbankoski): Remove these lines when the issue:
    // http://code.google.com/p/webm/issues/detail?id=496 is fixed.
    // For now the codec assumes buffer starts at starting buffer rate
    // plus one frame's time.
    if (last_pts_ == 0) duration = 1;

    // Add to the buffer the bits we'd expect from a constant bitrate server.
    bits_in_buffer_model_ += static_cast<int64_t>(
        duration * timebase_ * cfg_.rc_target_bitrate * 1000);

    /* Test the buffer model here before subtracting the frame. Do so because
     * the way the leaky bucket model works in libvpx is to allow the buffer to
     * empty - and then stop showing frames until we've got enough bits to
     * show one. As noted in comment below (issue 495), this does not currently
     * apply to key frames. For now exclude key frames in condition below. */
    const bool key_frame =
        (pkt->data.frame.flags & VPX_FRAME_IS_KEY) ? true : false;
    if (!key_frame) {
      ASSERT_GE(bits_in_buffer_model_, 0)
          << "Buffer Underrun at frame " << pkt->data.frame.pts;
    }

    const int64_t frame_size_in_bits = pkt->data.frame.sz * 8;

    // Subtract from the buffer the bits associated with a played back frame.
    bits_in_buffer_model_ -= frame_size_in_bits;

    // Update the running total of bits for end of test datarate checks.
    bits_total_ += frame_size_in_bits;

    // If first drop not set and we have a drop set it to this time.
    if (!first_drop_ && duration > 1) first_drop_ = last_pts_ + 1;

    // Update the most recent pts.
    last_pts_ = pkt->data.frame.pts;

    // We update this so that we can calculate the datarate minus the last
    // frame encoded in the file.
    bits_in_last_frame_ = frame_size_in_bits;

    ++frame_number_;
  }

  void EndPassHook() override {
    if (bits_total_) {
      const double file_size_in_kb = bits_total_ / 1000.;  // bits per kilobit

      duration_ = (last_pts_ + 1) * timebase_;

      // Effective file datarate includes the time spent prebuffering.
      effective_datarate_ = (bits_total_ - bits_in_last_frame_) / 1000.0 /
                            (cfg_.rc_buf_initial_sz / 1000.0 + duration_);

      file_datarate_ = file_size_in_kb / duration_;
    }
  }

  virtual void DenoiserLevelsTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_dropframe_thresh = 1;
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_end_usage = VPX_CBR;
    ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 140);
    for (int j = 1; j < 5; ++j) {
      // Run over the denoiser levels.
      // For the temporal denoiser (#if CONFIG_TEMPORAL_DENOISING) the level j
      // refers to the 4 denoiser modes: denoiserYonly, denoiserOnYUV,
      // denoiserOnAggressive, and denoiserOnAdaptive.
      cfg_.rc_target_bitrate = 300;
      ResetModel();
      denoiser_on_ = j;
      ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_ * 0.95)
          << " The datarate for the file exceeds the target!";

      ASSERT_LE(cfg_.rc_target_bitrate, file_datarate_ * 1.4)
          << " The datarate for the file missed the target!";
    }
  }

  virtual void DenoiserOffOnTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_dropframe_thresh = 1;
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_end_usage = VPX_CBR;
    ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 299);
    cfg_.rc_target_bitrate = 300;
    ResetModel();
    // Set the offon test flag.
    denoiser_offon_test_ = 1;
    denoiser_offon_period_ = 100;
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_ * 0.95)
        << " The datarate for the file exceeds the target!";
    ASSERT_LE(cfg_.rc_target_bitrate, file_datarate_ * 1.4)
        << " The datarate for the file missed the target!";
  }

  virtual void BasicBufferModelTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_dropframe_thresh = 1;
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_end_usage = VPX_CBR;
    // 2 pass cbr datarate control has a bug hidden by the small # of
    // frames selected in this encode. The problem is that even if the buffer is
    // negative we produce a keyframe on a cutscene. Ignoring datarate
    // constraints
    // TODO(jimbankoski): ( Fix when issue
    // http://code.google.com/p/webm/issues/detail?id=495 is addressed. )
    ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 140);

    // There is an issue for low bitrates in real-time mode, where the
    // effective_datarate slightly overshoots the target bitrate.
    // This is same the issue as noted about (#495).
    // TODO(jimbankoski/marpan): Update test to run for lower bitrates (< 100),
    // when the issue is resolved.
    for (int i = 100; i < 800; i += 200) {
      cfg_.rc_target_bitrate = i;
      ResetModel();
      ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_ * 0.95)
          << " The datarate for the file exceeds the target!";
      ASSERT_LE(cfg_.rc_target_bitrate, file_datarate_ * 1.4)
          << " The datarate for the file missed the target!";
    }
  }

  virtual void ChangingDropFrameThreshTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_max_quantizer = 36;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.rc_target_bitrate = 200;
    cfg_.kf_mode = VPX_KF_DISABLED;

    const int frame_count = 40;
    ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, frame_count);

    // Here we check that the first dropped frame gets earlier and earlier
    // as the drop frame threshold is increased.

    const int kDropFrameThreshTestStep = 30;
    vpx_codec_pts_t last_drop = frame_count;
    for (int i = 1; i < 91; i += kDropFrameThreshTestStep) {
      cfg_.rc_dropframe_thresh = i;
      ResetModel();
      ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
      ASSERT_LE(first_drop_, last_drop)
          << " The first dropped frame for drop_thresh " << i
          << " > first dropped frame for drop_thresh "
          << i - kDropFrameThreshTestStep;
      last_drop = first_drop_;
    }
  }

  virtual void DropFramesMultiThreadsTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_dropframe_thresh = 30;
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.g_threads = 2;

    ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352,
                                         288, 30, 1, 0, 140);
    cfg_.rc_target_bitrate = 200;
    ResetModel();
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_ * 0.95)
        << " The datarate for the file exceeds the target!";

    ASSERT_LE(cfg_.rc_target_bitrate, file_datarate_ * 1.4)
        << " The datarate for the file missed the target!";
  }

  virtual void MultiThreadsPSNRTest() {
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_dropframe_thresh = 0;
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.g_threads = 4;
    init_flags_ = VPX_CODEC_USE_PSNR;

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, 30);
    cfg_.rc_target_bitrate = 1000;
    ResetModel();
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_ * 0.5)
        << " The datarate for the file exceeds the target!";

    ASSERT_LE(cfg_.rc_target_bitrate, file_datarate_ * 2.0)
        << " The datarate for the file missed the target!";
  }

  vpx_codec_pts_t last_pts_;
  int64_t bits_in_buffer_model_;
  double timebase_;
  int frame_number_;
  vpx_codec_pts_t first_drop_;
  int64_t bits_total_;
  double duration_;
  double file_datarate_;
  double effective_datarate_;
  int64_t bits_in_last_frame_;
  int denoiser_on_;
  int denoiser_offon_test_;
  int denoiser_offon_period_;
  int set_cpu_used_;
  int gf_boost_;
  bool use_roi_;
  vpx_roi_map_t roi_;
};

#if CONFIG_TEMPORAL_DENOISING
// Check basic datarate targeting, for a single bitrate, but loop over the
// various denoiser settings.
TEST_P(DatarateTestLarge, DenoiserLevels) { DenoiserLevelsTest(); }

// Check basic datarate targeting, for a single bitrate, when denoiser is off
// and on.
TEST_P(DatarateTestLarge, DenoiserOffOn) { DenoiserOffOnTest(); }
#endif  // CONFIG_TEMPORAL_DENOISING

TEST_P(DatarateTestLarge, BasicBufferModel) { BasicBufferModelTest(); }

TEST_P(DatarateTestLarge, ChangingDropFrameThresh) {
  ChangingDropFrameThreshTest();
}

TEST_P(DatarateTestLarge, DropFramesMultiThreads) {
  DropFramesMultiThreadsTest();
}

class DatarateTestRealTime : public DatarateTestLarge {
 public:
  ~DatarateTestRealTime() override = default;
};

#if CONFIG_TEMPORAL_DENOISING
// Check basic datarate targeting, for a single bitrate, but loop over the
// various denoiser settings.
TEST_P(DatarateTestRealTime, DenoiserLevels) { DenoiserLevelsTest(); }

// Check basic datarate targeting, for a single bitrate, when denoiser is off
// and on.
TEST_P(DatarateTestRealTime, DenoiserOffOn) {}
#endif  // CONFIG_TEMPORAL_DENOISING

TEST_P(DatarateTestRealTime, BasicBufferModel) { BasicBufferModelTest(); }

TEST_P(DatarateTestRealTime, ChangingDropFrameThresh) {
  ChangingDropFrameThreshTest();
}

TEST_P(DatarateTestRealTime, DropFramesMultiThreads) {
  DropFramesMultiThreadsTest();
}

TEST_P(DatarateTestRealTime, MultiThreadsPSNR) { MultiThreadsPSNRTest(); }

TEST_P(DatarateTestRealTime, RegionOfInterest) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  // Encode using multiple threads.
  cfg_.g_threads = 2;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 300);
  cfg_.rc_target_bitrate = 450;
  cfg_.g_w = 352;
  cfg_.g_h = 288;

  ResetModel();

  // Set ROI parameters
  use_roi_ = true;
  memset(&roi_, 0, sizeof(roi_));

  roi_.rows = (cfg_.g_h + 15) / 16;
  roi_.cols = (cfg_.g_w + 15) / 16;

  roi_.delta_q[0] = 0;
  roi_.delta_q[1] = -20;
  roi_.delta_q[2] = 0;
  roi_.delta_q[3] = 0;

  roi_.delta_lf[0] = 0;
  roi_.delta_lf[1] = -20;
  roi_.delta_lf[2] = 0;
  roi_.delta_lf[3] = 0;

  roi_.static_threshold[0] = 0;
  roi_.static_threshold[1] = 1000;
  roi_.static_threshold[2] = 0;
  roi_.static_threshold[3] = 0;

  // Use 2 states: 1 is center square, 0 is the rest.
  roi_.roi_map =
      (uint8_t *)calloc(roi_.rows * roi_.cols, sizeof(*roi_.roi_map));
  for (unsigned int i = 0; i < roi_.rows; ++i) {
    for (unsigned int j = 0; j < roi_.cols; ++j) {
      if (i > (roi_.rows >> 2) && i < ((roi_.rows * 3) >> 2) &&
          j > (roi_.cols >> 2) && j < ((roi_.cols * 3) >> 2)) {
        roi_.roi_map[i * roi_.cols + j] = 1;
      }
    }
  }

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_ * 0.95)
      << " The datarate for the file exceeds the target!";

  ASSERT_LE(cfg_.rc_target_bitrate, file_datarate_ * 1.4)
      << " The datarate for the file missed the target!";

  free(roi_.roi_map);
}

TEST_P(DatarateTestRealTime, GFBoost) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_error_resilient = 0;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 300);
  cfg_.rc_target_bitrate = 300;
  ResetModel();
  // Apply a gf boost.
  gf_boost_ = 50;

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_ * 0.95)
      << " The datarate for the file exceeds the target!";

  ASSERT_LE(cfg_.rc_target_bitrate, file_datarate_ * 1.4)
      << " The datarate for the file missed the target!";
}

TEST_P(DatarateTestRealTime, NV12) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_error_resilient = 0;
  ::libvpx_test::YUVVideoSource video("hantro_collage_w352h288_nv12.yuv",
                                      VPX_IMG_FMT_NV12, 352, 288, 30, 1, 0,
                                      100);

  cfg_.rc_target_bitrate = 200;
  ResetModel();

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_ * 0.95)
      << " The datarate for the file exceeds the target!";

  ASSERT_LE(cfg_.rc_target_bitrate, file_datarate_ * 1.4)
      << " The datarate for the file missed the target!";
}

class DatarateTestPsnr : public DatarateTestLarge {
 public:
  DatarateTestPsnr() : DatarateTestLarge() {}
  ~DatarateTestPsnr() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(libvpx_test::kRealTime);
    set_cpu_used_ = 10;
    ResetModel();
    frame_flags_ = VPX_EFLAG_CALCULATE_PSNR;
  }
  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    DatarateTestLarge::PreEncodeFrameHook(video, encoder);
    frame_flags_ ^= VPX_EFLAG_CALCULATE_PSNR;
#if CONFIG_INTERNAL_STATS
    // CONFIG_INTERNAL_STATS unconditionally generates PSNR.
    expect_psnr_ = true;
#else
    expect_psnr_ = (frame_flags_ & VPX_EFLAG_CALCULATE_PSNR) != 0;
#endif  // CONFIG_INTERNAL_STATS
    if (video->img() == nullptr) {
      expect_psnr_ = false;
    }
  }
  void PostEncodeFrameHook(::libvpx_test::Encoder *encoder) override {
    libvpx_test::CxDataIterator iter = encoder->GetCxData();

    bool had_psnr = false;
    while (const vpx_codec_cx_pkt_t *pkt = iter.Next()) {
      if (pkt->kind == VPX_CODEC_PSNR_PKT) had_psnr = true;
    }

    EXPECT_EQ(had_psnr, expect_psnr_);
  }

 private:
  bool expect_psnr_;
};

TEST_P(DatarateTestPsnr, PerFramePsnr) {
  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 100);

  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

VP8_INSTANTIATE_TEST_SUITE(DatarateTestLarge, ALL_TEST_MODES,
                           ::testing::Values(0));
VP8_INSTANTIATE_TEST_SUITE(DatarateTestRealTime,
                           ::testing::Values(::libvpx_test::kRealTime),
                           ::testing::Values(-6, -12));
VP8_INSTANTIATE_TEST_SUITE(DatarateTestPsnr,
                           ::testing::Values(::libvpx_test::kRealTime),
                           ::testing::Values(0));

}  // namespace
