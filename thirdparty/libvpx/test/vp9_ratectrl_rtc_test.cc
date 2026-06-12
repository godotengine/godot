/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "vp9/ratectrl_rtc.h"

#include <climits>
#include <fstream>  // NOLINT
#include <string>

#include "./vpx_config.h"
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/video_source.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_svc_layercontext.h"
#include "vpx/vpx_codec.h"
#include "vpx_ports/bitops.h"

namespace {

const size_t kNumFrames = 300;

const int kTemporalId3Layer[4] = { 0, 2, 1, 2 };
const int kTemporalId2Layer[2] = { 0, 1 };
const int kTemporalRateAllocation3Layer[3] = { 50, 70, 100 };
const int kTemporalRateAllocation2Layer[2] = { 60, 100 };
const int kSpatialLayerBitrate[3] = { 200, 400, 1000 };
const int kSpatialLayerBitrateLow[3] = { 50, 100, 400 };

class RcInterfaceTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<int, vpx_rc_mode> {
 public:
  RcInterfaceTest()
      : EncoderTest(GET_PARAM(0)), aq_mode_(GET_PARAM(1)), key_interval_(3000),
        encoder_exit_(false), frame_drop_thresh_(0), num_drops_(0) {}

  ~RcInterfaceTest() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
  }

  void PreEncodeFrameHook(libvpx_test::VideoSource *video,
                          libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, 7);
      encoder->Control(VP9E_SET_AQ_MODE, aq_mode_);
      if (rc_cfg_.is_screen) {
        encoder->Control(VP9E_SET_TUNE_CONTENT, VP9E_CONTENT_SCREEN);
      } else {
        encoder->Control(VP9E_SET_TUNE_CONTENT, VP9E_CONTENT_DEFAULT);
      }
      encoder->Control(VP8E_SET_MAX_INTRA_BITRATE_PCT, 1000);
      encoder->Control(VP9E_SET_RTC_EXTERNAL_RATECTRL, 1);
    }
    frame_params_.frame_type = video->frame() % key_interval_ == 0
                                   ? libvpx::RcFrameType::kKeyFrame
                                   : libvpx::RcFrameType::kInterFrame;
    if (rc_cfg_.rc_mode == VPX_CBR &&
        frame_params_.frame_type == libvpx::RcFrameType::kInterFrame) {
      // Disable golden frame update.
      frame_flags_ |= VP8_EFLAG_NO_UPD_GF;
      frame_flags_ |= VP8_EFLAG_NO_UPD_ARF;
    }
    encoder_exit_ = video->frame() == kNumFrames;
  }

  void PostEncodeFrameHook(::libvpx_test::Encoder *encoder) override {
    if (encoder_exit_) {
      return;
    }
    int loopfilter_level, qp;
    encoder->Control(VP9E_GET_LOOPFILTER_LEVEL, &loopfilter_level);
    encoder->Control(VP8E_GET_LAST_QUANTIZER, &qp);
    if (rc_api_->ComputeQP(frame_params_) == libvpx::FrameDropDecision::kOk) {
      ASSERT_EQ(rc_api_->GetQP(), qp);
      ASSERT_EQ(rc_api_->GetLoopfilterLevel(), loopfilter_level);
    } else {
      num_drops_++;
    }
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    rc_api_->PostEncodeUpdate(pkt->data.frame.sz, frame_params_);
  }

  void RunOneLayer() {
    SetConfig(GET_PARAM(2));
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    frame_params_.spatial_layer_id = 0;
    frame_params_.temporal_layer_id = 0;

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunOneLayerScreen() {
    SetConfig(GET_PARAM(2));
    rc_cfg_.is_screen = true;
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    frame_params_.spatial_layer_id = 0;
    frame_params_.temporal_layer_id = 0;

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunOneLayerDropFramesCBR() {
    if (GET_PARAM(2) != VPX_CBR) {
      GTEST_SKIP() << "Frame dropping is only for CBR mode.";
    }
    frame_drop_thresh_ = 30;
    SetConfig(GET_PARAM(2));
    // Use lower bitrate, lower max-q, and enable frame dropper.
    rc_cfg_.target_bandwidth = 200;
    cfg_.rc_target_bitrate = 200;
    rc_cfg_.max_quantizer = 50;
    cfg_.rc_max_quantizer = 50;
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    frame_params_.spatial_layer_id = 0;
    frame_params_.temporal_layer_id = 0;

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    // Check that some frames were dropped, otherwise test has no value.
    ASSERT_GE(num_drops_, 1);
  }

  void RunOneLayerVBRPeriodicKey() {
    if (GET_PARAM(2) != VPX_VBR) return;
    key_interval_ = 100;
    SetConfig(VPX_VBR);
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    frame_params_.spatial_layer_id = 0;
    frame_params_.temporal_layer_id = 0;

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

 private:
  void SetConfig(vpx_rc_mode rc_mode) {
    rc_cfg_.width = 1280;
    rc_cfg_.height = 720;
    rc_cfg_.max_quantizer = 52;
    rc_cfg_.min_quantizer = 2;
    rc_cfg_.target_bandwidth = 1000;
    rc_cfg_.buf_initial_sz = 600;
    rc_cfg_.buf_optimal_sz = 600;
    rc_cfg_.buf_sz = 1000;
    rc_cfg_.undershoot_pct = 50;
    rc_cfg_.overshoot_pct = 50;
    rc_cfg_.max_intra_bitrate_pct = 1000;
    rc_cfg_.framerate = 30.0;
    rc_cfg_.ss_number_layers = 1;
    rc_cfg_.ts_number_layers = 1;
    rc_cfg_.scaling_factor_num[0] = 1;
    rc_cfg_.scaling_factor_den[0] = 1;
    rc_cfg_.layer_target_bitrate[0] = 1000;
    rc_cfg_.max_quantizers[0] = 52;
    rc_cfg_.min_quantizers[0] = 2;
    rc_cfg_.rc_mode = rc_mode;
    rc_cfg_.aq_mode = aq_mode_;
    rc_cfg_.frame_drop_thresh = frame_drop_thresh_;

    // Encoder settings for ground truth.
    cfg_.g_w = 1280;
    cfg_.g_h = 720;
    cfg_.rc_undershoot_pct = 50;
    cfg_.rc_overshoot_pct = 50;
    cfg_.rc_buf_initial_sz = 600;
    cfg_.rc_buf_optimal_sz = 600;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_dropframe_thresh = 0;
    cfg_.rc_min_quantizer = 2;
    cfg_.rc_max_quantizer = 52;
    cfg_.rc_end_usage = rc_mode;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_error_resilient = 0;
    cfg_.rc_target_bitrate = 1000;
    cfg_.kf_min_dist = key_interval_;
    cfg_.kf_max_dist = key_interval_;
    cfg_.rc_dropframe_thresh = frame_drop_thresh_;
  }

  std::unique_ptr<libvpx::VP9RateControlRTC> rc_api_;
  libvpx::VP9RateControlRtcConfig rc_cfg_;
  int aq_mode_;
  int key_interval_;
  libvpx::VP9FrameParamsQpRTC frame_params_;
  bool encoder_exit_;
  int frame_drop_thresh_;
  int num_drops_;
};

class RcInterfaceSvcTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<int, bool> {
 public:
  RcInterfaceSvcTest()
      : EncoderTest(GET_PARAM(0)), aq_mode_(GET_PARAM(1)), key_interval_(3000),
        dynamic_spatial_layers_(0), inter_layer_pred_off_(GET_PARAM(2)),
        parallel_spatial_layers_(false), frame_drop_thresh_(0),
        max_consec_drop_(INT_MAX), num_drops_(0) {}
  ~RcInterfaceSvcTest() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
  }

  void PreEncodeFrameHook(libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      current_superframe_ = 0;
      encoder->Control(VP8E_SET_CPUUSED, 7);
      encoder->Control(VP9E_SET_AQ_MODE, aq_mode_);
      encoder->Control(VP9E_SET_TUNE_CONTENT, 0);
      encoder->Control(VP8E_SET_MAX_INTRA_BITRATE_PCT, 900);
      encoder->Control(VP9E_SET_RTC_EXTERNAL_RATECTRL, 1);
      encoder->Control(VP9E_SET_SVC, 1);
      encoder->Control(VP9E_SET_SVC_PARAMETERS, &svc_params_);
      if (inter_layer_pred_off_) {
        encoder->Control(VP9E_SET_SVC_INTER_LAYER_PRED,
                         INTER_LAYER_PRED_OFF_NONKEY);
      }
      if (frame_drop_thresh_ > 0) {
        vpx_svc_frame_drop_t svc_drop_frame;
        svc_drop_frame.framedrop_mode = FULL_SUPERFRAME_DROP;
        for (int sl = 0; sl < rc_cfg_.ss_number_layers; ++sl)
          svc_drop_frame.framedrop_thresh[sl] = frame_drop_thresh_;
        svc_drop_frame.max_consec_drop = max_consec_drop_;
        encoder->Control(VP9E_SET_SVC_FRAME_DROP_LAYER, &svc_drop_frame);
      }
    }
    frame_params_.frame_type = video->frame() % key_interval_ == 0
                                   ? libvpx::RcFrameType::kKeyFrame
                                   : libvpx::RcFrameType::kInterFrame;
    encoder_exit_ = video->frame() == kNumFrames;
    if (dynamic_spatial_layers_ == 1) {
      if (video->frame() == 100) {
        // Go down to 2 spatial layers: set top SL to 0 bitrate.
        // Update the encoder config.
        cfg_.rc_target_bitrate -= cfg_.layer_target_bitrate[8];
        cfg_.layer_target_bitrate[6] = 0;
        cfg_.layer_target_bitrate[7] = 0;
        cfg_.layer_target_bitrate[8] = 0;
        encoder->Config(&cfg_);
        // Update the RC config.
        rc_cfg_.target_bandwidth -= rc_cfg_.layer_target_bitrate[8];
        rc_cfg_.layer_target_bitrate[6] = 0;
        rc_cfg_.layer_target_bitrate[7] = 0;
        rc_cfg_.layer_target_bitrate[8] = 0;
        ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));
      } else if (video->frame() == 200) {
        // Go down to 1 spatial layer.
        // Update the encoder config.
        cfg_.rc_target_bitrate -= cfg_.layer_target_bitrate[5];
        cfg_.layer_target_bitrate[3] = 0;
        cfg_.layer_target_bitrate[4] = 0;
        cfg_.layer_target_bitrate[5] = 0;
        encoder->Config(&cfg_);
        // Update the RC config.
        rc_cfg_.target_bandwidth -= rc_cfg_.layer_target_bitrate[5];
        rc_cfg_.layer_target_bitrate[3] = 0;
        rc_cfg_.layer_target_bitrate[4] = 0;
        rc_cfg_.layer_target_bitrate[5] = 0;
        ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));
      } else if (/*DISABLES CODE*/ (false) && video->frame() == 280) {
        // TODO(marpan): Re-enable this going back up when issue is fixed.
        // Go back up to 3 spatial layers.
        // Update the encoder config: use the original bitrates.
        SetEncoderConfigSvc(3, 3);
        encoder->Config(&cfg_);
        // Update the RC config.
        SetRCConfigSvc(3, 3);
        ASSERT_TRUE(rc_api_->UpdateRateControl(rc_cfg_));
      }
    }
  }

  virtual void SetFrameParamsSvc(int sl) {
    frame_params_.spatial_layer_id = sl;
    if (rc_cfg_.ts_number_layers == 3)
      frame_params_.temporal_layer_id =
          kTemporalId3Layer[current_superframe_ % 4];
    else if (rc_cfg_.ts_number_layers == 2)
      frame_params_.temporal_layer_id =
          kTemporalId2Layer[current_superframe_ % 2];
    else
      frame_params_.temporal_layer_id = 0;
    frame_params_.frame_type =
        current_superframe_ % key_interval_ == 0 && sl == 0
            ? libvpx::RcFrameType::kKeyFrame
            : libvpx::RcFrameType::kInterFrame;
  }

  void PostEncodeFrameHook(::libvpx_test::Encoder *encoder) override {
    if (encoder_exit_) {
      return;
    }
    int superframe_is_dropped = false;
    ::libvpx_test::CxDataIterator iter = encoder->GetCxData();
    for (int sl = 0; sl < rc_cfg_.ss_number_layers; sl++) sizes_[sl] = 0;
    std::vector<int> rc_qp;
    // For FULL_SUPERFRAME_DROP: the full superframe drop decision is
    // determined on the base spatial layer.
    SetFrameParamsSvc(0);
    if (rc_api_->ComputeQP(frame_params_) == libvpx::FrameDropDecision::kDrop) {
      superframe_is_dropped = true;
      num_drops_++;
    }
    while (const vpx_codec_cx_pkt_t *pkt = iter.Next()) {
      ASSERT_EQ(superframe_is_dropped, false);
      ParseSuperframeSizes(static_cast<const uint8_t *>(pkt->data.frame.buf),
                           pkt->data.frame.sz);
      if (!parallel_spatial_layers_ || current_superframe_ == 0) {
        for (int sl = 0; sl < rc_cfg_.ss_number_layers; sl++) {
          if (sizes_[sl] > 0) {
            SetFrameParamsSvc(sl);
            // For sl=0 ComputeQP() is already called above (line 310).
            if (sl > 0) rc_api_->ComputeQP(frame_params_);
            rc_api_->PostEncodeUpdate(sizes_[sl], frame_params_);
            rc_qp.push_back(rc_api_->GetQP());
          }
        }
      } else {
        for (int sl = 0; sl < rc_cfg_.ss_number_layers; sl++) {
          // For sl=0 ComputeQP() is already called above (line 310).
          if (sizes_[sl] > 0 && sl > 0) {
            SetFrameParamsSvc(sl);
            rc_api_->ComputeQP(frame_params_);
          }
        }
        for (int sl = 0; sl < rc_cfg_.ss_number_layers; sl++) {
          if (sizes_[sl] > 0) {
            SetFrameParamsSvc(sl);
            rc_api_->PostEncodeUpdate(sizes_[sl], frame_params_);
            rc_qp.push_back(rc_api_->GetQP());
          }
        }
      }
    }
    if (!superframe_is_dropped) {
      int loopfilter_level;
      std::vector<int> encoder_qp(VPX_SS_MAX_LAYERS, 0);
      encoder->Control(VP9E_GET_LOOPFILTER_LEVEL, &loopfilter_level);
      encoder->Control(VP9E_GET_LAST_QUANTIZER_SVC_LAYERS, encoder_qp.data());
      encoder_qp.resize(rc_qp.size());
      ASSERT_EQ(rc_qp, encoder_qp);
      ASSERT_EQ(rc_api_->GetLoopfilterLevel(), loopfilter_level);
      current_superframe_++;
    }
  }
  // This method needs to be overridden because non-reference frames are
  // expected to be mismatched frames as the encoder will avoid loopfilter on
  // these frames.
  void MismatchHook(const vpx_image_t * /*img1*/,
                    const vpx_image_t * /*img2*/) override {}

  void RunSvc() {
    SetRCConfigSvc(3, 3);
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    SetEncoderConfigSvc(3, 3);

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunSvcDropFramesCBR() {
    max_consec_drop_ = 10;
    frame_drop_thresh_ = 30;
    SetRCConfigSvc(3, 3);
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    SetEncoderConfigSvc(3, 3);

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    // Check that some frames were dropped, otherwise test has no value.
    ASSERT_GE(num_drops_, 1);
  }

  void RunSvcPeriodicKey() {
    SetRCConfigSvc(3, 3);
    key_interval_ = 100;
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    SetEncoderConfigSvc(3, 3);

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunSvcDynamicSpatial() {
    dynamic_spatial_layers_ = 1;
    SetRCConfigSvc(3, 3);
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    SetEncoderConfigSvc(3, 3);

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void RunSvcParallelSpatialLayers() {
    if (!inter_layer_pred_off_) return;
    parallel_spatial_layers_ = true;
    SetRCConfigSvc(3, 3);
    rc_api_ = libvpx::VP9RateControlRTC::Create(rc_cfg_);
    SetEncoderConfigSvc(3, 3);

    ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv",
                                         1280, 720, 30, 1, 0, kNumFrames);

    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

 private:
  vpx_codec_err_t ParseSuperframeSizes(const uint8_t *data, size_t data_sz) {
    uint8_t marker = *(data + data_sz - 1);
    if ((marker & 0xe0) == 0xc0) {
      const uint32_t frames = (marker & 0x7) + 1;
      const uint32_t mag = ((marker >> 3) & 0x3) + 1;
      const size_t index_sz = 2 + mag * frames;
      // This chunk is marked as having a superframe index but doesn't have
      // enough data for it, thus it's an invalid superframe index.
      if (data_sz < index_sz) return VPX_CODEC_CORRUPT_FRAME;
      {
        const uint8_t marker2 = *(data + data_sz - index_sz);
        // This chunk is marked as having a superframe index but doesn't have
        // the matching marker byte at the front of the index therefore it's an
        // invalid chunk.
        if (marker != marker2) return VPX_CODEC_CORRUPT_FRAME;
      }
      const uint8_t *x = &data[data_sz - index_sz + 1];
      for (uint32_t i = 0; i < frames; ++i) {
        uint32_t this_sz = 0;

        for (uint32_t j = 0; j < mag; ++j) this_sz |= (*x++) << (j * 8);
        sizes_[i] = this_sz;
      }
    }
    return VPX_CODEC_OK;
  }

  void SetEncoderConfigSvc(int number_spatial_layers,
                           int number_temporal_layers) {
    cfg_.g_w = 1280;
    cfg_.g_h = 720;
    cfg_.ss_number_layers = number_spatial_layers;
    cfg_.ts_number_layers = number_temporal_layers;
    cfg_.g_timebase.num = 1;
    cfg_.g_timebase.den = 30;
    if (number_spatial_layers == 3) {
      svc_params_.scaling_factor_num[0] = 1;
      svc_params_.scaling_factor_den[0] = 4;
      svc_params_.scaling_factor_num[1] = 2;
      svc_params_.scaling_factor_den[1] = 4;
      svc_params_.scaling_factor_num[2] = 4;
      svc_params_.scaling_factor_den[2] = 4;
    } else if (number_spatial_layers == 2) {
      svc_params_.scaling_factor_num[0] = 1;
      svc_params_.scaling_factor_den[0] = 2;
      svc_params_.scaling_factor_num[1] = 2;
      svc_params_.scaling_factor_den[1] = 2;
    } else if (number_spatial_layers == 1) {
      svc_params_.scaling_factor_num[0] = 1;
      svc_params_.scaling_factor_den[0] = 1;
    }

    for (int i = 0; i < VPX_MAX_LAYERS; ++i) {
      svc_params_.max_quantizers[i] = 56;
      svc_params_.min_quantizers[i] = 2;
      svc_params_.speed_per_layer[i] = 7;
      svc_params_.loopfilter_ctrl[i] = LOOPFILTER_ALL;
    }
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_error_resilient = 0;

    if (number_temporal_layers == 3) {
      cfg_.ts_rate_decimator[0] = 4;
      cfg_.ts_rate_decimator[1] = 2;
      cfg_.ts_rate_decimator[2] = 1;
      cfg_.temporal_layering_mode = 3;
    } else if (number_temporal_layers == 2) {
      cfg_.ts_rate_decimator[0] = 2;
      cfg_.ts_rate_decimator[1] = 1;
      cfg_.temporal_layering_mode = 2;
    } else if (number_temporal_layers == 1) {
      cfg_.ts_rate_decimator[0] = 1;
      cfg_.temporal_layering_mode = 0;
    }

    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 600;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_min_quantizer = 2;
    cfg_.rc_max_quantizer = 56;
    cfg_.g_threads = 1;
    cfg_.kf_max_dist = 9999;
    cfg_.rc_overshoot_pct = 50;
    cfg_.rc_undershoot_pct = 50;
    cfg_.rc_dropframe_thresh = frame_drop_thresh_;

    cfg_.rc_target_bitrate = 0;
    for (int sl = 0; sl < number_spatial_layers; sl++) {
      int spatial_bitrate = 0;
      if (number_spatial_layers <= 3)
        spatial_bitrate = frame_drop_thresh_ > 0 ? kSpatialLayerBitrateLow[sl]
                                                 : kSpatialLayerBitrate[sl];
      for (int tl = 0; tl < number_temporal_layers; tl++) {
        int layer = sl * number_temporal_layers + tl;
        if (number_temporal_layers == 3)
          cfg_.layer_target_bitrate[layer] =
              kTemporalRateAllocation3Layer[tl] * spatial_bitrate / 100;
        else if (number_temporal_layers == 2)
          cfg_.layer_target_bitrate[layer] =
              kTemporalRateAllocation2Layer[tl] * spatial_bitrate / 100;
        else if (number_temporal_layers == 1)
          cfg_.layer_target_bitrate[layer] = spatial_bitrate;
      }
      cfg_.rc_target_bitrate += spatial_bitrate;
    }

    cfg_.kf_min_dist = key_interval_;
    cfg_.kf_max_dist = key_interval_;
  }

  void SetRCConfigSvc(int number_spatial_layers, int number_temporal_layers) {
    rc_cfg_.width = 1280;
    rc_cfg_.height = 720;
    rc_cfg_.ss_number_layers = number_spatial_layers;
    rc_cfg_.ts_number_layers = number_temporal_layers;
    rc_cfg_.max_quantizer = 56;
    rc_cfg_.min_quantizer = 2;
    rc_cfg_.buf_initial_sz = 500;
    rc_cfg_.buf_optimal_sz = 600;
    rc_cfg_.buf_sz = 1000;
    rc_cfg_.undershoot_pct = 50;
    rc_cfg_.overshoot_pct = 50;
    rc_cfg_.max_intra_bitrate_pct = 900;
    rc_cfg_.framerate = 30.0;
    rc_cfg_.rc_mode = VPX_CBR;
    rc_cfg_.aq_mode = aq_mode_;
    rc_cfg_.frame_drop_thresh = frame_drop_thresh_;
    rc_cfg_.max_consec_drop = max_consec_drop_;

    if (number_spatial_layers == 3) {
      rc_cfg_.scaling_factor_num[0] = 1;
      rc_cfg_.scaling_factor_den[0] = 4;
      rc_cfg_.scaling_factor_num[1] = 2;
      rc_cfg_.scaling_factor_den[1] = 4;
      rc_cfg_.scaling_factor_num[2] = 4;
      rc_cfg_.scaling_factor_den[2] = 4;
    } else if (number_spatial_layers == 2) {
      rc_cfg_.scaling_factor_num[0] = 1;
      rc_cfg_.scaling_factor_den[0] = 2;
      rc_cfg_.scaling_factor_num[1] = 2;
      rc_cfg_.scaling_factor_den[1] = 2;
    } else if (number_spatial_layers == 1) {
      rc_cfg_.scaling_factor_num[0] = 1;
      rc_cfg_.scaling_factor_den[0] = 1;
    }

    if (number_temporal_layers == 3) {
      rc_cfg_.ts_rate_decimator[0] = 4;
      rc_cfg_.ts_rate_decimator[1] = 2;
      rc_cfg_.ts_rate_decimator[2] = 1;
    } else if (number_temporal_layers == 2) {
      rc_cfg_.ts_rate_decimator[0] = 2;
      rc_cfg_.ts_rate_decimator[1] = 1;
    } else if (number_temporal_layers == 1) {
      rc_cfg_.ts_rate_decimator[0] = 1;
    }

    rc_cfg_.target_bandwidth = 0;
    for (int sl = 0; sl < number_spatial_layers; sl++) {
      int spatial_bitrate = 0;
      if (number_spatial_layers <= 3)
        spatial_bitrate = frame_drop_thresh_ > 0 ? kSpatialLayerBitrateLow[sl]
                                                 : kSpatialLayerBitrate[sl];
      for (int tl = 0; tl < number_temporal_layers; tl++) {
        int layer = sl * number_temporal_layers + tl;
        if (number_temporal_layers == 3)
          rc_cfg_.layer_target_bitrate[layer] =
              kTemporalRateAllocation3Layer[tl] * spatial_bitrate / 100;
        else if (number_temporal_layers == 2)
          rc_cfg_.layer_target_bitrate[layer] =
              kTemporalRateAllocation2Layer[tl] * spatial_bitrate / 100;
        else if (number_temporal_layers == 1)
          rc_cfg_.layer_target_bitrate[layer] = spatial_bitrate;
      }
      rc_cfg_.target_bandwidth += spatial_bitrate;
    }

    for (int sl = 0; sl < rc_cfg_.ss_number_layers; ++sl) {
      for (int tl = 0; tl < rc_cfg_.ts_number_layers; ++tl) {
        const int i = sl * rc_cfg_.ts_number_layers + tl;
        rc_cfg_.max_quantizers[i] = 56;
        rc_cfg_.min_quantizers[i] = 2;
      }
    }
  }

  int aq_mode_;
  std::unique_ptr<libvpx::VP9RateControlRTC> rc_api_;
  libvpx::VP9RateControlRtcConfig rc_cfg_;
  vpx_svc_extra_cfg_t svc_params_;
  libvpx::VP9FrameParamsQpRTC frame_params_;
  bool encoder_exit_;
  int current_superframe_;
  uint32_t sizes_[8];
  int key_interval_;
  int dynamic_spatial_layers_;
  bool inter_layer_pred_off_;
  // ComputeQP() and PostEncodeUpdate() don't need to be sequential for KSVC.
  bool parallel_spatial_layers_;
  int frame_drop_thresh_;
  int max_consec_drop_;
  int num_drops_;
};

TEST_P(RcInterfaceTest, OneLayer) { RunOneLayer(); }

TEST_P(RcInterfaceTest, OneLayerDropFramesCBR) { RunOneLayerDropFramesCBR(); }

TEST_P(RcInterfaceTest, OneLayerScreen) { RunOneLayerScreen(); }

TEST_P(RcInterfaceTest, OneLayerVBRPeriodicKey) { RunOneLayerVBRPeriodicKey(); }

TEST_P(RcInterfaceSvcTest, Svc) { RunSvc(); }

TEST_P(RcInterfaceSvcTest, SvcDropFramesCBR) { RunSvcDropFramesCBR(); }

TEST_P(RcInterfaceSvcTest, SvcParallelSpatialLayers) {
  RunSvcParallelSpatialLayers();
}

TEST_P(RcInterfaceSvcTest, SvcPeriodicKey) { RunSvcPeriodicKey(); }

TEST_P(RcInterfaceSvcTest, SvcDynamicSpatial) { RunSvcDynamicSpatial(); }

VP9_INSTANTIATE_TEST_SUITE(RcInterfaceTest, ::testing::Values(0, 3),
                           ::testing::Values(VPX_CBR, VPX_VBR));
VP9_INSTANTIATE_TEST_SUITE(RcInterfaceSvcTest, ::testing::Values(0, 3),
                           ::testing::Values(true, false));
}  // namespace
