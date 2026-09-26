/*
 *  Copyright (c) 2021 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <fstream>  // NOLINT
#include <string>

#include "./vpx_config.h"
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/video_source.h"
#include "vp8/vp8_ratectrl_rtc.h"
#include "vpx/vpx_codec.h"
#include "vpx_ports/bitops.h"

namespace {

struct Vp8RCTestVideo {
  Vp8RCTestVideo() = default;
  Vp8RCTestVideo(const char *name_, int width_, int height_,
                 unsigned int frames_)
      : name(name_), width(width_), height(height_), frames(frames_) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const Vp8RCTestVideo &video) {
    os << video.name << " " << video.width << " " << video.height << " "
       << video.frames;
    return os;
  }
  const char *name;
  int width;
  int height;
  unsigned int frames;
};

const Vp8RCTestVideo kVp8RCTestVectors[] = {
  Vp8RCTestVideo("niklas_640_480_30.yuv", 640, 480, 470),
  Vp8RCTestVideo("desktop_office1.1280_720-020.yuv", 1280, 720, 300),
  Vp8RCTestVideo("hantro_collage_w352h288.yuv", 352, 288, 100),
};

class Vp8RcInterfaceTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<int, Vp8RCTestVideo> {
 public:
  Vp8RcInterfaceTest()
      : EncoderTest(GET_PARAM(0)), key_interval_(3000), encoder_exit_(false),
        frame_drop_thresh_(0) {}
  ~Vp8RcInterfaceTest() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
  }

  // From error_resilience_test.cc
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

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (rc_cfg_.ts_number_layers > 1) {
      const int layer_id = SetLayerId(video->frame(), cfg_.ts_number_layers);
      const int frame_flags =
          SetFrameFlags(video->frame(), cfg_.ts_number_layers);
      frame_params_.temporal_layer_id = layer_id;
      if (video->frame() > 0) {
        encoder->Control(VP8E_SET_TEMPORAL_LAYER_ID, layer_id);
        encoder->Control(VP8E_SET_FRAME_FLAGS, frame_flags);
      }
    } else {
      if (video->frame() == 0) {
        encoder->Control(VP8E_SET_CPUUSED, -6);
        encoder->Control(VP8E_SET_RTC_EXTERNAL_RATECTRL, 1);
        encoder->Control(VP8E_SET_MAX_INTRA_BITRATE_PCT, 1000);
        if (rc_cfg_.is_screen) {
          encoder->Control(VP8E_SET_SCREEN_CONTENT_MODE, 1);
        }
      } else if (frame_params_.frame_type == libvpx::RcFrameType::kInterFrame) {
        // Disable golden frame update.
        frame_flags_ |= VP8_EFLAG_NO_UPD_GF;
        frame_flags_ |= VP8_EFLAG_NO_UPD_ARF;
      }
    }
    frame_params_.frame_type = video->frame() % key_interval_ == 0
                                   ? libvpx::RcFrameType::kKeyFrame
                                   : libvpx::RcFrameType::kInterFrame;
    encoder_exit_ = video->frame() == test_video_.frames;
  }

  void PostEncodeFrameHook(::libvpx_test::Encoder *encoder) override {
    if (encoder_exit_) {
      return;
    }
    int qp;
    libvpx::UVDeltaQP uv_delta_qp;
    encoder->Control(VP8E_GET_LAST_QUANTIZER, &qp);
    if (rc_api_->ComputeQP(frame_params_) == libvpx::FrameDropDecision::kOk) {
      ASSERT_EQ(rc_api_->GetQP(), qp);
      uv_delta_qp = rc_api_->GetUVDeltaQP();
      // delta_qp for UV channel is only set for screen.
      if (!rc_cfg_.is_screen) {
        ASSERT_EQ(uv_delta_qp.uvdc_delta_q, 0);
        ASSERT_EQ(uv_delta_qp.uvac_delta_q, 0);
      }
    } else {
      num_drops_++;
    }
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    rc_api_->PostEncodeUpdate(pkt->data.frame.sz);
  }

  void RunOneLayer() {
    test_video_ = GET_PARAM(2);
    target_bitrate_ = GET_PARAM(1);
    SetConfig();
    rc_api_ = libvpx::VP8RateControlRTC::Create(rc_cfg_);
    ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));

    ::libvpx_test::I420VideoSource video(test_video_.name, test_video_.width,
                                         test_video_.height, 30, 1, 0,
                                         test_video_.frames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunOneLayerScreen() {
    test_video_ = GET_PARAM(2);
    target_bitrate_ = GET_PARAM(1);
    SetConfig();
    rc_cfg_.is_screen = true;
    rc_api_ = libvpx::VP8RateControlRTC::Create(rc_cfg_);
    ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));

    ::libvpx_test::I420VideoSource video(test_video_.name, test_video_.width,
                                         test_video_.height, 30, 1, 0,
                                         test_video_.frames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunOneLayerDropFrames() {
    test_video_ = GET_PARAM(2);
    target_bitrate_ = GET_PARAM(1);
    frame_drop_thresh_ = 30;
    num_drops_ = 0;
    // Use lower target_bitrate and max_quantizer to trigger drops.
    target_bitrate_ = target_bitrate_ >> 2;
    SetConfig();
    rc_cfg_.max_quantizer = 56;
    cfg_.rc_max_quantizer = 56;
    rc_api_ = libvpx::VP8RateControlRTC::Create(rc_cfg_);
    ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));

    ::libvpx_test::I420VideoSource video(test_video_.name, test_video_.width,
                                         test_video_.height, 30, 1, 0,
                                         test_video_.frames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    // Check that some frames were dropped, otherwise test has no value.
    ASSERT_GE(num_drops_, 1);
  }

  void RunPeriodicKey() {
    test_video_ = GET_PARAM(2);
    target_bitrate_ = GET_PARAM(1);
    key_interval_ = 100;
    frame_drop_thresh_ = 30;
    SetConfig();
    rc_api_ = libvpx::VP8RateControlRTC::Create(rc_cfg_);
    ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));

    ::libvpx_test::I420VideoSource video(test_video_.name, test_video_.width,
                                         test_video_.height, 30, 1, 0,
                                         test_video_.frames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunTemporalLayers2TL() {
    test_video_ = GET_PARAM(2);
    target_bitrate_ = GET_PARAM(1);
    SetConfigTemporalLayers(2);
    rc_api_ = libvpx::VP8RateControlRTC::Create(rc_cfg_);
    ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));

    ::libvpx_test::I420VideoSource video(test_video_.name, test_video_.width,
                                         test_video_.height, 30, 1, 0,
                                         test_video_.frames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunTemporalLayers3TL() {
    test_video_ = GET_PARAM(2);
    target_bitrate_ = GET_PARAM(1);
    SetConfigTemporalLayers(3);
    rc_api_ = libvpx::VP8RateControlRTC::Create(rc_cfg_);
    ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));

    ::libvpx_test::I420VideoSource video(test_video_.name, test_video_.width,
                                         test_video_.height, 30, 1, 0,
                                         test_video_.frames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunTemporalLayers3TLDropFrames() {
    test_video_ = GET_PARAM(2);
    target_bitrate_ = GET_PARAM(1);
    frame_drop_thresh_ = 30;
    num_drops_ = 0;
    // Use lower target_bitrate and max_quantizer to trigger drops.
    target_bitrate_ = target_bitrate_ >> 2;
    SetConfigTemporalLayers(3);
    rc_cfg_.max_quantizer = 56;
    cfg_.rc_max_quantizer = 56;
    rc_api_ = libvpx::VP8RateControlRTC::Create(rc_cfg_);
    ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));

    ::libvpx_test::I420VideoSource video(test_video_.name, test_video_.width,
                                         test_video_.height, 30, 1, 0,
                                         test_video_.frames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    // Check that some frames were dropped, otherwise test has no value.
    ASSERT_GE(num_drops_, 1);
  }

 private:
  void SetConfig() {
    rc_cfg_.width = test_video_.width;
    rc_cfg_.height = test_video_.height;
    rc_cfg_.max_quantizer = 60;
    rc_cfg_.min_quantizer = 2;
    rc_cfg_.target_bandwidth = target_bitrate_;
    rc_cfg_.buf_initial_sz = 600;
    rc_cfg_.buf_optimal_sz = 600;
    rc_cfg_.buf_sz = target_bitrate_;
    rc_cfg_.undershoot_pct = 50;
    rc_cfg_.overshoot_pct = 50;
    rc_cfg_.max_intra_bitrate_pct = 1000;
    rc_cfg_.framerate = 30.0;
    rc_cfg_.layer_target_bitrate[0] = target_bitrate_;
    rc_cfg_.frame_drop_thresh = frame_drop_thresh_;

    // Encoder settings for ground truth.
    cfg_.g_w = test_video_.width;
    cfg_.g_h = test_video_.height;
    cfg_.rc_undershoot_pct = 50;
    cfg_.rc_overshoot_pct = 50;
    cfg_.rc_buf_initial_sz = 600;
    cfg_.rc_buf_optimal_sz = 600;
    cfg_.rc_buf_sz = target_bitrate_;
    cfg_.rc_dropframe_thresh = 0;
    cfg_.rc_min_quantizer = 2;
    cfg_.rc_max_quantizer = 60;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_error_resilient = 1;
    cfg_.rc_target_bitrate = target_bitrate_;
    cfg_.kf_min_dist = key_interval_;
    cfg_.kf_max_dist = key_interval_;
    cfg_.rc_dropframe_thresh = frame_drop_thresh_;
  }

  void SetConfigTemporalLayers(int temporal_layers) {
    rc_cfg_.width = test_video_.width;
    rc_cfg_.height = test_video_.height;
    rc_cfg_.max_quantizer = 60;
    rc_cfg_.min_quantizer = 2;
    rc_cfg_.target_bandwidth = target_bitrate_;
    rc_cfg_.buf_initial_sz = 600;
    rc_cfg_.buf_optimal_sz = 600;
    rc_cfg_.buf_sz = target_bitrate_;
    rc_cfg_.undershoot_pct = 50;
    rc_cfg_.overshoot_pct = 50;
    rc_cfg_.max_intra_bitrate_pct = 1000;
    rc_cfg_.framerate = 30.0;
    rc_cfg_.frame_drop_thresh = frame_drop_thresh_;
    if (temporal_layers == 2) {
      rc_cfg_.layer_target_bitrate[0] = 60 * target_bitrate_ / 100;
      rc_cfg_.layer_target_bitrate[1] = target_bitrate_;
      rc_cfg_.ts_rate_decimator[0] = 2;
      rc_cfg_.ts_rate_decimator[1] = 1;
    } else if (temporal_layers == 3) {
      rc_cfg_.layer_target_bitrate[0] = 40 * target_bitrate_ / 100;
      rc_cfg_.layer_target_bitrate[1] = 60 * target_bitrate_ / 100;
      rc_cfg_.layer_target_bitrate[2] = target_bitrate_;
      rc_cfg_.ts_rate_decimator[0] = 4;
      rc_cfg_.ts_rate_decimator[1] = 2;
      rc_cfg_.ts_rate_decimator[2] = 1;
    }

    rc_cfg_.ts_number_layers = temporal_layers;

    // Encoder settings for ground truth.
    cfg_.g_w = test_video_.width;
    cfg_.g_h = test_video_.height;
    cfg_.rc_undershoot_pct = 50;
    cfg_.rc_overshoot_pct = 50;
    cfg_.rc_buf_initial_sz = 600;
    cfg_.rc_buf_optimal_sz = 600;
    cfg_.rc_buf_sz = target_bitrate_;
    cfg_.rc_dropframe_thresh = 0;
    cfg_.rc_min_quantizer = 2;
    cfg_.rc_max_quantizer = 60;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_error_resilient = 1;
    cfg_.rc_target_bitrate = target_bitrate_;
    cfg_.kf_min_dist = key_interval_;
    cfg_.kf_max_dist = key_interval_;
    cfg_.rc_dropframe_thresh = frame_drop_thresh_;
    // 2 Temporal layers, no spatial layers, CBR mode.
    cfg_.ss_number_layers = 1;
    cfg_.ts_number_layers = temporal_layers;
    if (temporal_layers == 2) {
      cfg_.ts_rate_decimator[0] = 2;
      cfg_.ts_rate_decimator[1] = 1;
      cfg_.ts_periodicity = 2;
      cfg_.ts_target_bitrate[0] = 60 * cfg_.rc_target_bitrate / 100;
      cfg_.ts_target_bitrate[1] = cfg_.rc_target_bitrate;
    } else if (temporal_layers == 3) {
      cfg_.ts_rate_decimator[0] = 4;
      cfg_.ts_rate_decimator[1] = 2;
      cfg_.ts_rate_decimator[2] = 1;
      cfg_.ts_periodicity = 4;
      cfg_.ts_target_bitrate[0] = 40 * cfg_.rc_target_bitrate / 100;
      cfg_.ts_target_bitrate[1] = 60 * cfg_.rc_target_bitrate / 100;
      cfg_.ts_target_bitrate[2] = cfg_.rc_target_bitrate;
    }
  }

  std::unique_ptr<libvpx::VP8RateControlRTC> rc_api_;
  libvpx::VP8RateControlRtcConfig rc_cfg_;
  int key_interval_;
  int target_bitrate_;
  Vp8RCTestVideo test_video_;
  libvpx::VP8FrameParamsQpRTC frame_params_;
  bool encoder_exit_;
  int frame_drop_thresh_;
  int num_drops_;
};

TEST_P(Vp8RcInterfaceTest, OneLayer) { RunOneLayer(); }

TEST_P(Vp8RcInterfaceTest, OneLayerScreen) { RunOneLayerScreen(); }

TEST_P(Vp8RcInterfaceTest, OneLayerDropFrames) { RunOneLayerDropFrames(); }

TEST_P(Vp8RcInterfaceTest, OneLayerPeriodicKey) { RunPeriodicKey(); }

TEST_P(Vp8RcInterfaceTest, TemporalLayers2TL) { RunTemporalLayers2TL(); }

TEST_P(Vp8RcInterfaceTest, TemporalLayers3TL) { RunTemporalLayers3TL(); }

TEST_P(Vp8RcInterfaceTest, TemporalLayers3TLDropFrames) {
  RunTemporalLayers3TLDropFrames();
}

VP8_INSTANTIATE_TEST_SUITE(Vp8RcInterfaceTest,
                           ::testing::Values(200, 400, 1000),
                           ::testing::ValuesIn(kVp8RCTestVectors));

}  // namespace
