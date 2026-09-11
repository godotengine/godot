/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
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

const int kMaxErrorFrames = 12;
const int kMaxDroppableFrames = 12;

class ErrorResilienceTestLarge
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<libvpx_test::TestMode, bool> {
 protected:
  ErrorResilienceTestLarge()
      : EncoderTest(GET_PARAM(0)), svc_support_(GET_PARAM(2)), psnr_(0.0),
        nframes_(0), mismatch_psnr_(0.0), mismatch_nframes_(0),
        encoding_mode_(GET_PARAM(1)) {
    Reset();
  }

  ~ErrorResilienceTestLarge() override = default;

  void Reset() {
    error_nframes_ = 0;
    droppable_nframes_ = 0;
    pattern_switch_ = 0;
  }

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);
  }

  void BeginPassHook(unsigned int /*pass*/) override {
    psnr_ = 0.0;
    nframes_ = 0;
    mismatch_psnr_ = 0.0;
    mismatch_nframes_ = 0;
  }

  void PSNRPktHook(const vpx_codec_cx_pkt_t *pkt) override {
    psnr_ += pkt->data.psnr.psnr[0];
    nframes_++;
  }

  //
  // Frame flags and layer id for temporal layers.
  // For two layers, test pattern is:
  //   1     3
  // 0    2     .....
  // LAST is updated on base/layer 0, GOLDEN  updated on layer 1.
  // Non-zero pattern_switch parameter means pattern will switch to
  // not using LAST for frame_num >= pattern_switch.
  int SetFrameFlags(int frame_num, int num_temp_layers, int pattern_switch) {
    int frame_flags = 0;
    if (num_temp_layers == 2) {
      if (frame_num % 2 == 0) {
        if (frame_num < pattern_switch || pattern_switch == 0) {
          // Layer 0: predict from LAST and ARF, update LAST.
          frame_flags =
              VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF;
        } else {
          // Layer 0: predict from GF and ARF, update GF.
          frame_flags = VP8_EFLAG_NO_REF_LAST | VP8_EFLAG_NO_UPD_LAST |
                        VP8_EFLAG_NO_UPD_ARF;
        }
      } else {
        if (frame_num < pattern_switch || pattern_switch == 0) {
          // Layer 1: predict from L, GF, and ARF, update GF.
          frame_flags = VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST;
        } else {
          // Layer 1: predict from GF and ARF, update GF.
          frame_flags = VP8_EFLAG_NO_REF_LAST | VP8_EFLAG_NO_UPD_LAST |
                        VP8_EFLAG_NO_UPD_ARF;
        }
      }
    }
    return frame_flags;
  }

  void PreEncodeFrameHook(libvpx_test::VideoSource *video) override {
    frame_flags_ &=
        ~(VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF);
    // For temporal layer case.
    if (cfg_.ts_number_layers > 1) {
      frame_flags_ =
          SetFrameFlags(video->frame(), cfg_.ts_number_layers, pattern_switch_);
      for (unsigned int i = 0; i < droppable_nframes_; ++i) {
        if (droppable_frames_[i] == video->frame()) {
          std::cout << "Encoding droppable frame: " << droppable_frames_[i]
                    << "\n";
        }
      }
    } else {
      if (droppable_nframes_ > 0 &&
          (cfg_.g_pass == VPX_RC_LAST_PASS || cfg_.g_pass == VPX_RC_ONE_PASS)) {
        for (unsigned int i = 0; i < droppable_nframes_; ++i) {
          if (droppable_frames_[i] == video->frame()) {
            std::cout << "Encoding droppable frame: " << droppable_frames_[i]
                      << "\n";
            frame_flags_ |= (VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_UPD_GF |
                             VP8_EFLAG_NO_UPD_ARF);
            return;
          }
        }
      }
    }
  }

  double GetAveragePsnr() const {
    if (nframes_) return psnr_ / nframes_;
    return 0.0;
  }

  double GetAverageMismatchPsnr() const {
    if (mismatch_nframes_) return mismatch_psnr_ / mismatch_nframes_;
    return 0.0;
  }

  bool DoDecode() const override {
    if (error_nframes_ > 0 &&
        (cfg_.g_pass == VPX_RC_LAST_PASS || cfg_.g_pass == VPX_RC_ONE_PASS)) {
      for (unsigned int i = 0; i < error_nframes_; ++i) {
        if (error_frames_[i] == nframes_ - 1) {
          std::cout << "             Skipping decoding frame: "
                    << error_frames_[i] << "\n";
          return false;
        }
      }
    }
    return true;
  }

  void MismatchHook(const vpx_image_t *img1, const vpx_image_t *img2) override {
    double mismatch_psnr = compute_psnr(img1, img2);
    mismatch_psnr_ += mismatch_psnr;
    ++mismatch_nframes_;
    // std::cout << "Mismatch frame psnr: " << mismatch_psnr << "\n";
  }

  void SetErrorFrames(int num, unsigned int *list) {
    if (num > kMaxErrorFrames) {
      num = kMaxErrorFrames;
    } else if (num < 0) {
      num = 0;
    }
    error_nframes_ = num;
    for (unsigned int i = 0; i < error_nframes_; ++i) {
      error_frames_[i] = list[i];
    }
  }

  void SetDroppableFrames(int num, unsigned int *list) {
    if (num > kMaxDroppableFrames) {
      num = kMaxDroppableFrames;
    } else if (num < 0) {
      num = 0;
    }
    droppable_nframes_ = num;
    for (unsigned int i = 0; i < droppable_nframes_; ++i) {
      droppable_frames_[i] = list[i];
    }
  }

  unsigned int GetMismatchFrames() { return mismatch_nframes_; }

  void SetPatternSwitch(int frame_switch) { pattern_switch_ = frame_switch; }

  bool svc_support_;

 private:
  double psnr_;
  unsigned int nframes_;
  unsigned int error_nframes_;
  unsigned int droppable_nframes_;
  unsigned int pattern_switch_;
  double mismatch_psnr_;
  unsigned int mismatch_nframes_;
  unsigned int error_frames_[kMaxErrorFrames];
  unsigned int droppable_frames_[kMaxDroppableFrames];
  libvpx_test::TestMode encoding_mode_;
};

TEST_P(ErrorResilienceTestLarge, OnVersusOff) {
#if CONFIG_REALTIME_ONLY
  GTEST_SKIP()
      << "Non-zero g_lag_in_frames is unsupported with CONFIG_REALTIME_ONLY";
#else
  const vpx_rational timebase = { 33333333, 1000000000 };
  cfg_.g_timebase = timebase;
  cfg_.rc_target_bitrate = 2000;
  cfg_.g_lag_in_frames = 10;

  init_flags_ = VPX_CODEC_USE_PSNR;

  libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                     timebase.den, timebase.num, 0, 30);

  // Error resilient mode OFF.
  cfg_.g_error_resilient = 0;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const double psnr_resilience_off = GetAveragePsnr();
  EXPECT_GT(psnr_resilience_off, 25.0);

  // Error resilient mode ON.
  cfg_.g_error_resilient = 1;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const double psnr_resilience_on = GetAveragePsnr();
  EXPECT_GT(psnr_resilience_on, 25.0);

  // Test that turning on error resilient mode hurts by 10% at most.
  if (psnr_resilience_off > 0.0) {
    const double psnr_ratio = psnr_resilience_on / psnr_resilience_off;
    EXPECT_GE(psnr_ratio, 0.9);
    EXPECT_LE(psnr_ratio, 1.1);
  }
#endif  // CONFIG_REALTIME_ONLY
}

// Check for successful decoding and no encoder/decoder mismatch
// if we lose (i.e., drop before decoding) a set of droppable
// frames (i.e., frames that don't update any reference buffers).
// Check both isolated and consecutive loss.
TEST_P(ErrorResilienceTestLarge, DropFramesWithoutRecovery) {
  const vpx_rational timebase = { 33333333, 1000000000 };
  cfg_.g_timebase = timebase;
  cfg_.rc_target_bitrate = 500;
  // FIXME(debargha): Fix this to work for any lag.
  // Currently this test only works for lag = 0
  cfg_.g_lag_in_frames = 0;

  init_flags_ = VPX_CODEC_USE_PSNR;

  libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                     timebase.den, timebase.num, 0, 40);

  // Error resilient mode ON.
  cfg_.g_error_resilient = 1;
  cfg_.kf_mode = VPX_KF_DISABLED;

  // Set an arbitrary set of error frames same as droppable frames.
  // In addition to isolated loss/drop, add a long consecutive series
  // (of size 9) of dropped frames.
  unsigned int num_droppable_frames = 11;
  unsigned int droppable_frame_list[] = { 5,  16, 22, 23, 24, 25,
                                          26, 27, 28, 29, 30 };
  SetDroppableFrames(num_droppable_frames, droppable_frame_list);
  SetErrorFrames(num_droppable_frames, droppable_frame_list);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Test that no mismatches have been found
  std::cout << "             Mismatch frames: " << GetMismatchFrames() << "\n";
  EXPECT_EQ(GetMismatchFrames(), (unsigned int)0);

  // Reset previously set of error/droppable frames.
  Reset();

#if 0
  // TODO(jkoleszar): This test is disabled for the time being as too
  // sensitive. It's not clear how to set a reasonable threshold for
  // this behavior.

  // Now set an arbitrary set of error frames that are non-droppable
  unsigned int num_error_frames = 3;
  unsigned int error_frame_list[] = {3, 10, 20};
  SetErrorFrames(num_error_frames, error_frame_list);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // Test that dropping an arbitrary set of inter frames does not hurt too much
  // Note the Average Mismatch PSNR is the average of the PSNR between
  // decoded frame and encoder's version of the same frame for all frames
  // with mismatch.
  const double psnr_resilience_mismatch = GetAverageMismatchPsnr();
  std::cout << "             Mismatch PSNR: "
            << psnr_resilience_mismatch << "\n";
  EXPECT_GT(psnr_resilience_mismatch, 20.0);
#endif
}

// Check for successful decoding and no encoder/decoder mismatch
// if we lose (i.e., drop before decoding) the enhancement layer frames for a
// two layer temporal pattern. The base layer does not predict from the top
// layer, so successful decoding is expected.
TEST_P(ErrorResilienceTestLarge, 2LayersDropEnhancement) {
  // This test doesn't run if SVC is not supported.
  if (!svc_support_) return;

  const vpx_rational timebase = { 33333333, 1000000000 };
  cfg_.g_timebase = timebase;
  cfg_.rc_target_bitrate = 500;
  cfg_.g_lag_in_frames = 0;

  cfg_.rc_end_usage = VPX_CBR;
  // 2 Temporal layers, no spatial layers, CBR mode.
  cfg_.ss_number_layers = 1;
  cfg_.ts_number_layers = 2;
  cfg_.ts_rate_decimator[0] = 2;
  cfg_.ts_rate_decimator[1] = 1;
  cfg_.ts_periodicity = 2;
  cfg_.ts_target_bitrate[0] = 60 * cfg_.rc_target_bitrate / 100;
  cfg_.ts_target_bitrate[1] = cfg_.rc_target_bitrate;

  init_flags_ = VPX_CODEC_USE_PSNR;

  libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                     timebase.den, timebase.num, 0, 40);

  // Error resilient mode ON.
  cfg_.g_error_resilient = 1;
  cfg_.kf_mode = VPX_KF_DISABLED;
  SetPatternSwitch(0);

  // The odd frames are the enhancement layer for 2 layer pattern, so set
  // those frames as droppable. Drop the last 7 frames.
  unsigned int num_droppable_frames = 7;
  unsigned int droppable_frame_list[] = { 27, 29, 31, 33, 35, 37, 39 };
  SetDroppableFrames(num_droppable_frames, droppable_frame_list);
  SetErrorFrames(num_droppable_frames, droppable_frame_list);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Test that no mismatches have been found
  std::cout << "             Mismatch frames: " << GetMismatchFrames() << "\n";
  EXPECT_EQ(GetMismatchFrames(), (unsigned int)0);

  // Reset previously set of error/droppable frames.
  Reset();
}

// Check for successful decoding and no encoder/decoder mismatch
// for a two layer temporal pattern, where at some point in the
// sequence, the LAST ref is not used anymore.
TEST_P(ErrorResilienceTestLarge, 2LayersNoRefLast) {
  // This test doesn't run if SVC is not supported.
  if (!svc_support_) return;

  const vpx_rational timebase = { 33333333, 1000000000 };
  cfg_.g_timebase = timebase;
  cfg_.rc_target_bitrate = 500;
  cfg_.g_lag_in_frames = 0;

  cfg_.rc_end_usage = VPX_CBR;
  // 2 Temporal layers, no spatial layers, CBR mode.
  cfg_.ss_number_layers = 1;
  cfg_.ts_number_layers = 2;
  cfg_.ts_rate_decimator[0] = 2;
  cfg_.ts_rate_decimator[1] = 1;
  cfg_.ts_periodicity = 2;
  cfg_.ts_target_bitrate[0] = 60 * cfg_.rc_target_bitrate / 100;
  cfg_.ts_target_bitrate[1] = cfg_.rc_target_bitrate;

  init_flags_ = VPX_CODEC_USE_PSNR;

  libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                     timebase.den, timebase.num, 0, 100);

  // Error resilient mode ON.
  cfg_.g_error_resilient = 1;
  cfg_.kf_mode = VPX_KF_DISABLED;
  SetPatternSwitch(60);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Test that no mismatches have been found
  std::cout << "             Mismatch frames: " << GetMismatchFrames() << "\n";
  EXPECT_EQ(GetMismatchFrames(), (unsigned int)0);

  // Reset previously set of error/droppable frames.
  Reset();
}

class ErrorResilienceTestLargeCodecControls
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  ErrorResilienceTestLargeCodecControls()
      : EncoderTest(GET_PARAM(0)), encoding_mode_(GET_PARAM(1)) {
    Reset();
  }

  ~ErrorResilienceTestLargeCodecControls() override = default;

  void Reset() {
    last_pts_ = 0;
    tot_frame_number_ = 0;
    // For testing up to 3 layers.
    for (int i = 0; i < 3; ++i) {
      bits_total_[i] = 0;
    }
    duration_ = 0.0;
  }

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);
  }

  //
  // Frame flags and layer id for temporal layers.
  //

  // For two layers, test pattern is:
  //   1     3
  // 0    2     .....
  // For three layers, test pattern is:
  //   1      3    5      7
  //      2           6
  // 0          4            ....
  // LAST is always update on base/layer 0, GOLDEN is updated on layer 1,
  // and ALTREF is updated on top layer for 3 layer pattern.
  int SetFrameFlags(int frame_num, int num_temp_layers) {
    int frame_flags = 0;
    if (num_temp_layers == 2) {
      if (frame_num % 2 == 0) {
        // Layer 0: predict from L and ARF, update L.
        frame_flags =
            VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF;
      } else {
        // Layer 1: predict from L, G and ARF, and update G.
        frame_flags = VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST |
                      VP8_EFLAG_NO_UPD_ENTROPY;
      }
    } else if (num_temp_layers == 3) {
      if (frame_num % 4 == 0) {
        // Layer 0: predict from L, update L.
        frame_flags = VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF |
                      VP8_EFLAG_NO_REF_GF | VP8_EFLAG_NO_REF_ARF;
      } else if ((frame_num - 2) % 4 == 0) {
        // Layer 1: predict from L, G,  update G.
        frame_flags =
            VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST | VP8_EFLAG_NO_REF_ARF;
      } else if ((frame_num - 1) % 2 == 0) {
        // Layer 2: predict from L, G, ARF; update ARG.
        frame_flags = VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_LAST;
      }
    }
    return frame_flags;
  }

  int SetLayerId(int frame_num, int num_temp_layers) {
    int layer_id = 0;
    if (num_temp_layers == 2) {
      if (frame_num % 2 == 0) {
        layer_id = 0;
      } else {
        layer_id = 1;
      }
    } else if (num_temp_layers == 3) {
      if (frame_num % 4 == 0) {
        layer_id = 0;
      } else if ((frame_num - 2) % 4 == 0) {
        layer_id = 1;
      } else if ((frame_num - 1) % 2 == 0) {
        layer_id = 2;
      }
    }
    return layer_id;
  }

  void PreEncodeFrameHook(libvpx_test::VideoSource *video,
                          libvpx_test::Encoder *encoder) override {
    if (cfg_.ts_number_layers > 1) {
      int layer_id = SetLayerId(video->frame(), cfg_.ts_number_layers);
      int frame_flags = SetFrameFlags(video->frame(), cfg_.ts_number_layers);
      if (video->frame() > 0) {
        encoder->Control(VP8E_SET_TEMPORAL_LAYER_ID, layer_id);
        encoder->Control(VP8E_SET_FRAME_FLAGS, frame_flags);
      }
      const vpx_rational_t tb = video->timebase();
      timebase_ = static_cast<double>(tb.num) / tb.den;
      duration_ = 0;
      return;
    }
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    // Time since last timestamp = duration.
    vpx_codec_pts_t duration = pkt->data.frame.pts - last_pts_;
    if (duration > 1) {
      // Update counter for total number of frames (#frames input to encoder).
      // Needed for setting the proper layer_id below.
      tot_frame_number_ += static_cast<int>(duration - 1);
    }
    int layer = SetLayerId(tot_frame_number_, cfg_.ts_number_layers);
    const size_t frame_size_in_bits = pkt->data.frame.sz * 8;
    // Update the total encoded bits. For temporal layers, update the cumulative
    // encoded bits per layer.
    for (int i = layer; i < static_cast<int>(cfg_.ts_number_layers); ++i) {
      bits_total_[i] += frame_size_in_bits;
    }
    // Update the most recent pts.
    last_pts_ = pkt->data.frame.pts;
    ++tot_frame_number_;
  }

  void EndPassHook() override {
    duration_ = (last_pts_ + 1) * timebase_;
    if (cfg_.ts_number_layers > 1) {
      for (int layer = 0; layer < static_cast<int>(cfg_.ts_number_layers);
           ++layer) {
        if (bits_total_[layer]) {
          // Effective file datarate:
          effective_datarate_[layer] =
              (bits_total_[layer] / 1000.0) / duration_;
        }
      }
    }
  }

  double effective_datarate_[3];

 private:
  libvpx_test::TestMode encoding_mode_;
  vpx_codec_pts_t last_pts_;
  double timebase_;
  int64_t bits_total_[3];
  double duration_;
  int tot_frame_number_;
};

// Check two codec controls used for:
// (1) for setting temporal layer id, and (2) for settings encoder flags.
// This test invokes those controls for each frame, and verifies encoder/decoder
// mismatch and basic rate control response.
// TODO(marpan): Maybe move this test to datarate_test.cc.
TEST_P(ErrorResilienceTestLargeCodecControls, CodecControl3TemporalLayers) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 2;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.g_lag_in_frames = 0;
  cfg_.kf_mode = VPX_KF_DISABLED;
  cfg_.g_error_resilient = 1;

  // 3 Temporal layers. Framerate decimation (4, 2, 1).
  cfg_.ts_number_layers = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;
  cfg_.ts_periodicity = 4;
  cfg_.ts_layer_id[0] = 0;
  cfg_.ts_layer_id[1] = 2;
  cfg_.ts_layer_id[2] = 1;
  cfg_.ts_layer_id[3] = 2;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 200);
  for (int i = 200; i <= 800; i += 200) {
    cfg_.rc_target_bitrate = i;
    Reset();
    // 40-20-40 bitrate allocation for 3 temporal layers.
    cfg_.ts_target_bitrate[0] = 40 * cfg_.rc_target_bitrate / 100;
    cfg_.ts_target_bitrate[1] = 60 * cfg_.rc_target_bitrate / 100;
    cfg_.ts_target_bitrate[2] = cfg_.rc_target_bitrate;
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    for (int j = 0; j < static_cast<int>(cfg_.ts_number_layers); ++j) {
      ASSERT_GE(effective_datarate_[j], cfg_.ts_target_bitrate[j] * 0.75)
          << " The datarate for the file is lower than target by too much, "
             "for layer: "
          << j;
      ASSERT_LE(effective_datarate_[j], cfg_.ts_target_bitrate[j] * 1.25)
          << " The datarate for the file is greater than target by too much, "
             "for layer: "
          << j;
    }
  }
}

VP8_INSTANTIATE_TEST_SUITE(ErrorResilienceTestLarge, ONE_PASS_TEST_MODES,
                           ::testing::Values(true));
VP8_INSTANTIATE_TEST_SUITE(ErrorResilienceTestLargeCodecControls,
                           ONE_PASS_TEST_MODES);
VP9_INSTANTIATE_TEST_SUITE(ErrorResilienceTestLarge, ONE_PASS_TEST_MODES,
                           ::testing::Values(true));
}  // namespace
