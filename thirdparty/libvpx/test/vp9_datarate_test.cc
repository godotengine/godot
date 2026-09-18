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
#include "test/acm_random.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "vpx/vpx_codec.h"
#include "vpx_ports/bitops.h"

namespace {

class DatarateTestVP9 : public ::libvpx_test::EncoderTest {
 public:
  explicit DatarateTestVP9(const ::libvpx_test::CodecFactory *codec)
      : EncoderTest(codec) {
    tune_content_ = 0;
  }

 protected:
  ~DatarateTestVP9() override = default;

  virtual void ResetModel() {
    last_pts_ = 0;
    bits_in_buffer_model_ = cfg_.rc_target_bitrate * cfg_.rc_buf_initial_sz;
    frame_number_ = 0;
    tot_frame_number_ = 0;
    first_drop_ = 0;
    num_drops_ = 0;
    aq_mode_ = 3;
    // Denoiser is off by default.
    denoiser_on_ = 0;
    // For testing up to 3 layers.
    for (int i = 0; i < 3; ++i) {
      bits_total_[i] = 0;
    }
    denoiser_offon_test_ = 0;
    denoiser_offon_period_ = -1;
    frame_parallel_decoding_mode_ = 1;
    delta_q_uv_ = 0;
    use_roi_ = false;
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
  // LAST is always update on base/layer 0, GOLDEN is updated on layer 1.
  // For this 3 layer example, the 2nd enhancement layer (layer 2) updates
  // the altref frame.
  static int GetFrameFlags(int frame_num, int num_temp_layers) {
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
        // Layer 0: predict from L and ARF; update L.
        frame_flags =
            VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_REF_GF;
      } else if ((frame_num - 2) % 4 == 0) {
        // Layer 1: predict from L, G, ARF; update G.
        frame_flags = VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_LAST;
      } else if ((frame_num - 1) % 2 == 0) {
        // Layer 2: predict from L, G, ARF; update ARF.
        frame_flags = VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_LAST;
      }
    }
    return frame_flags;
  }

  static int SetLayerId(int frame_num, int num_temp_layers) {
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
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, set_cpu_used_);
      encoder->Control(VP9E_SET_AQ_MODE, aq_mode_);
      encoder->Control(VP9E_SET_TUNE_CONTENT, tune_content_);
    }

    if (denoiser_offon_test_) {
      ASSERT_GT(denoiser_offon_period_, 0)
          << "denoiser_offon_period_ is not positive.";
      if ((video->frame() + 1) % denoiser_offon_period_ == 0) {
        // Flip denoiser_on_ periodically
        denoiser_on_ ^= 1;
      }
    }

    encoder->Control(VP9E_SET_NOISE_SENSITIVITY, denoiser_on_);
    encoder->Control(VP9E_SET_TILE_COLUMNS, get_msb(cfg_.g_threads));
    encoder->Control(VP9E_SET_FRAME_PARALLEL_DECODING,
                     frame_parallel_decoding_mode_);

    if (use_roi_) {
      encoder->Control(VP9E_SET_ROI_MAP, &roi_);
      encoder->Control(VP9E_SET_AQ_MODE, 0);
    }

    if (delta_q_uv_ != 0) {
      encoder->Control(VP9E_SET_DELTA_Q_UV, delta_q_uv_);
    }

    if (cfg_.ts_number_layers > 1) {
      if (video->frame() == 0) {
        encoder->Control(VP9E_SET_SVC, 1);
      }
      if (cfg_.temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_BYPASS) {
        vpx_svc_layer_id_t layer_id;
        frame_flags_ = GetFrameFlags(video->frame(), cfg_.ts_number_layers);
        layer_id.spatial_layer_id = 0;
        layer_id.temporal_layer_id =
            SetLayerId(video->frame(), cfg_.ts_number_layers);
        layer_id.temporal_layer_id_per_spatial[0] =
            SetLayerId(video->frame(), cfg_.ts_number_layers);
        encoder->Control(VP9E_SET_SVC_LAYER_ID, &layer_id);
      }
    }
    const vpx_rational_t tb = video->timebase();
    timebase_ = static_cast<double>(tb.num) / tb.den;
    duration_ = 0;
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    // Time since last timestamp = duration.
    vpx_codec_pts_t duration = pkt->data.frame.pts - last_pts_;

    if (duration > 1) {
      // If first drop not set and we have a drop set it to this time.
      if (!first_drop_) first_drop_ = last_pts_ + 1;
      // Update the number of frame drops.
      num_drops_ += static_cast<int>(duration - 1);
      // Update counter for total number of frames (#frames input to encoder).
      // Needed for setting the proper layer_id below.
      tot_frame_number_ += static_cast<int>(duration - 1);
    }

    int layer = SetLayerId(tot_frame_number_, cfg_.ts_number_layers);

    // Add to the buffer the bits we'd expect from a constant bitrate server.
    bits_in_buffer_model_ += static_cast<int64_t>(
        duration * timebase_ * cfg_.rc_target_bitrate * 1000);

    // Buffer should not go negative.
    ASSERT_GE(bits_in_buffer_model_, 0)
        << "Buffer Underrun at frame " << pkt->data.frame.pts;

    const size_t frame_size_in_bits = pkt->data.frame.sz * 8;

    // Update the total encoded bits. For temporal layers, update the cumulative
    // encoded bits per layer.
    for (int i = layer; i < static_cast<int>(cfg_.ts_number_layers); ++i) {
      bits_total_[i] += frame_size_in_bits;
    }

    // Update the most recent pts.
    last_pts_ = pkt->data.frame.pts;
    ++frame_number_;
    ++tot_frame_number_;
  }

  void EndPassHook() override {
    for (int layer = 0; layer < static_cast<int>(cfg_.ts_number_layers);
         ++layer) {
      duration_ = (last_pts_ + 1) * timebase_;
      if (bits_total_[layer]) {
        // Effective file datarate:
        effective_datarate_[layer] = (bits_total_[layer] / 1000.0) / duration_;
      }
    }
  }

  vpx_codec_pts_t last_pts_;
  double timebase_;
  int tune_content_;
  int frame_number_;      // Counter for number of non-dropped/encoded frames.
  int tot_frame_number_;  // Counter for total number of input frames.
  int64_t bits_total_[3];
  double duration_;
  double effective_datarate_[3];
  int set_cpu_used_;
  int64_t bits_in_buffer_model_;
  vpx_codec_pts_t first_drop_;
  int num_drops_;
  int aq_mode_;
  int denoiser_on_;
  int denoiser_offon_test_;
  int denoiser_offon_period_;
  int frame_parallel_decoding_mode_;
  int delta_q_uv_;
  bool use_roi_;
  vpx_roi_map_t roi_;
};

// Params: test mode, speed setting and index for bitrate array.
class DatarateTestVP9RealTimeMultiBR
    : public DatarateTestVP9,
      public ::libvpx_test::CodecTestWith2Params<int, int> {
 public:
  DatarateTestVP9RealTimeMultiBR() : DatarateTestVP9(GET_PARAM(0)) {}

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    set_cpu_used_ = GET_PARAM(1);
    ResetModel();
  }
};

// Params: speed setting and index for bitrate array.
class DatarateTestVP9LargeVBR
    : public DatarateTestVP9,
      public ::libvpx_test::CodecTestWith2Params<int, int> {
 public:
  DatarateTestVP9LargeVBR() : DatarateTestVP9(GET_PARAM(0)) {}

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    set_cpu_used_ = GET_PARAM(1);
    ResetModel();
  }
};

// Check basic rate targeting for VBR mode with 0 lag.
TEST_P(DatarateTestVP9LargeVBR, BasicRateTargetingVBRLagZero) {
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_error_resilient = 0;
  cfg_.rc_end_usage = VPX_VBR;
  cfg_.g_lag_in_frames = 0;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 300);

  const int bitrates[2] = { 400, 800 };
  const int bitrate_index = GET_PARAM(2);
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.75)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.36)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic rate targeting for VBR mode with non-zero lag.
TEST_P(DatarateTestVP9LargeVBR, BasicRateTargetingVBRLagNonZero) {
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_error_resilient = 0;
  cfg_.rc_end_usage = VPX_VBR;
  // For non-zero lag, rate control will work (be within bounds) for
  // real-time mode.
  if (deadline_ == VPX_DL_REALTIME) {
    cfg_.g_lag_in_frames = 15;
  } else {
    cfg_.g_lag_in_frames = 0;
  }

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 300);
  const int bitrates[2] = { 400, 800 };
  const int bitrate_index = GET_PARAM(2);
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.75)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.35)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic rate targeting for VBR mode with non-zero lag, with
// frame_parallel_decoding_mode off. This enables the adapt_coeff/mode/mv probs
// since error_resilience is off.
TEST_P(DatarateTestVP9LargeVBR, BasicRateTargetingVBRLagNonZeroFrameParDecOff) {
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_error_resilient = 0;
  cfg_.rc_end_usage = VPX_VBR;
  // For non-zero lag, rate control will work (be within bounds) for
  // real-time mode.
  if (deadline_ == VPX_DL_REALTIME) {
    cfg_.g_lag_in_frames = 15;
  } else {
    cfg_.g_lag_in_frames = 0;
  }

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 300);
  const int bitrates[2] = { 400, 800 };
  const int bitrate_index = GET_PARAM(2);
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  ResetModel();
  frame_parallel_decoding_mode_ = 0;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.75)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.35)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic rate targeting for CBR mode.
TEST_P(DatarateTestVP9RealTimeMultiBR, BasicRateTargeting) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  const int bitrates[4] = { 150, 350, 550, 750 };
  const int bitrate_index = GET_PARAM(2);
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.15)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic rate targeting for CBR mode, with frame_parallel_decoding_mode
// off( and error_resilience off).
TEST_P(DatarateTestVP9RealTimeMultiBR, BasicRateTargetingFrameParDecOff) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;
  cfg_.g_error_resilient = 0;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  const int bitrates[4] = { 150, 350, 550, 750 };
  const int bitrate_index = GET_PARAM(2);
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  ResetModel();
  frame_parallel_decoding_mode_ = 0;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.15)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic rate targeting for CBR.
TEST_P(DatarateTestVP9RealTimeMultiBR, BasicRateTargeting444) {
  ::libvpx_test::Y4mVideoSource video("rush_hour_444.y4m", 0, 140);

  cfg_.g_profile = 1;
  cfg_.g_timebase = video.timebase();

  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  const int bitrates[4] = { 250, 450, 650, 850 };
  const int bitrate_index = GET_PARAM(2);
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(static_cast<double>(cfg_.rc_target_bitrate),
            effective_datarate_[0] * 0.80)
      << " The datarate for the file exceeds the target by too much!";
  ASSERT_LE(static_cast<double>(cfg_.rc_target_bitrate),
            effective_datarate_[0] * 1.15)
      << " The datarate for the file missed the target!"
      << cfg_.rc_target_bitrate << " " << effective_datarate_;
}

// Check that (1) the first dropped frame gets earlier and earlier
// as the drop frame threshold is increased, and (2) that the total number of
// frame drops does not decrease as we increase frame drop threshold.
// Use a lower qp-max to force some frame drops.
TEST_P(DatarateTestVP9RealTimeMultiBR, ChangingDropFrameThresh) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_undershoot_pct = 20;
  cfg_.rc_undershoot_pct = 20;
  cfg_.rc_dropframe_thresh = 10;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 50;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.rc_target_bitrate = 200;
  cfg_.g_lag_in_frames = 0;
  // TODO(marpan): Investigate datarate target failures with a smaller keyframe
  // interval (128).
  cfg_.kf_max_dist = 9999;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);

  const int kDropFrameThreshTestStep = 30;
  const int bitrates[2] = { 50, 150 };
  const int bitrate_index = GET_PARAM(2);
  if (bitrate_index > 1) return;
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  vpx_codec_pts_t last_drop = 140;
  int last_num_drops = 0;
  for (int i = 10; i < 100; i += kDropFrameThreshTestStep) {
    cfg_.rc_dropframe_thresh = i;
    ResetModel();
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
        << " The datarate for the file is lower than target by too much!";
    ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.25)
        << " The datarate for the file is greater than target by too much!";
    ASSERT_LE(first_drop_, last_drop)
        << " The first dropped frame for drop_thresh " << i
        << " > first dropped frame for drop_thresh "
        << i - kDropFrameThreshTestStep;
    ASSERT_GE(num_drops_, last_num_drops * 0.85)
        << " The number of dropped frames for drop_thresh " << i
        << " < number of dropped frames for drop_thresh "
        << i - kDropFrameThreshTestStep;
    last_drop = first_drop_;
    last_num_drops = num_drops_;
  }
}  // namespace

// Check basic rate targeting for 2 temporal layers.
TEST_P(DatarateTestVP9RealTimeMultiBR, BasicRateTargeting2TemporalLayers) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  // 2 Temporal layers, no spatial layers: Framerate decimation (2, 1).
  cfg_.ss_number_layers = 1;
  cfg_.ts_number_layers = 2;
  cfg_.ts_rate_decimator[0] = 2;
  cfg_.ts_rate_decimator[1] = 1;

  cfg_.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  const int bitrates[4] = { 200, 400, 600, 800 };
  const int bitrate_index = GET_PARAM(2);
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  ResetModel();
  // 60-40 bitrate allocation for 2 temporal layers.
  cfg_.layer_target_bitrate[0] = 60 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[1] = cfg_.rc_target_bitrate;
  aq_mode_ = 0;
  if (deadline_ == VPX_DL_REALTIME) {
    aq_mode_ = 3;
    cfg_.g_error_resilient = 1;
  }
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  for (int j = 0; j < static_cast<int>(cfg_.ts_number_layers); ++j) {
    ASSERT_GE(effective_datarate_[j], cfg_.layer_target_bitrate[j] * 0.85)
        << " The datarate for the file is lower than target by too much, "
           "for layer: "
        << j;
    ASSERT_LE(effective_datarate_[j], cfg_.layer_target_bitrate[j] * 1.15)
        << " The datarate for the file is greater than target by too much, "
           "for layer: "
        << j;
  }
}

// Check basic rate targeting for 3 temporal layers.
TEST_P(DatarateTestVP9RealTimeMultiBR, BasicRateTargeting3TemporalLayers) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  // 3 Temporal layers, no spatial layers: Framerate decimation (4, 2, 1).
  cfg_.ss_number_layers = 1;
  cfg_.ts_number_layers = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;

  cfg_.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  const int bitrates[4] = { 200, 400, 600, 800 };
  const int bitrate_index = GET_PARAM(2);
  cfg_.rc_target_bitrate = bitrates[bitrate_index];
  ResetModel();
  // 40-20-40 bitrate allocation for 3 temporal layers.
  cfg_.layer_target_bitrate[0] = 40 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[1] = 60 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[2] = cfg_.rc_target_bitrate;
  aq_mode_ = 0;
  if (deadline_ == VPX_DL_REALTIME) {
    aq_mode_ = 3;
    cfg_.g_error_resilient = 1;
  }
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  for (int j = 0; j < static_cast<int>(cfg_.ts_number_layers); ++j) {
    // TODO(yaowu): Work out more stable rc control strategy and
    //              Adjust the thresholds to be tighter than .75.
    ASSERT_GE(effective_datarate_[j], cfg_.layer_target_bitrate[j] * 0.75)
        << " The datarate for the file is lower than target by too much, "
           "for layer: "
        << j;
    // TODO(yaowu): Work out more stable rc control strategy and
    //              Adjust the thresholds to be tighter than 1.25.
    ASSERT_LE(effective_datarate_[j], cfg_.layer_target_bitrate[j] * 1.25)
        << " The datarate for the file is greater than target by too much, "
           "for layer: "
        << j;
  }
}

// Params: speed setting.
class DatarateTestVP9RealTime : public DatarateTestVP9,
                                public ::libvpx_test::CodecTestWithParam<int> {
 public:
  DatarateTestVP9RealTime() : DatarateTestVP9(GET_PARAM(0)) {}
  ~DatarateTestVP9RealTime() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    set_cpu_used_ = GET_PARAM(1);
    ResetModel();
  }
};

// Check basic rate targeting for CBR mode, with 2 threads and dropped frames.
TEST_P(DatarateTestVP9RealTime, BasicRateTargetingDropFramesMultiThreads) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;
  // Encode using multiple threads.
  cfg_.g_threads = 2;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  cfg_.rc_target_bitrate = 200;
  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.15)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic rate targeting for 3 temporal layers, with frame dropping.
// Only for one (low) bitrate with lower max_quantizer, and somewhat higher
// frame drop threshold, to force frame dropping.
TEST_P(DatarateTestVP9RealTime,
       BasicRateTargeting3TemporalLayersFrameDropping) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  // Set frame drop threshold and rc_max_quantizer to force some frame drops.
  cfg_.rc_dropframe_thresh = 20;
  cfg_.rc_max_quantizer = 45;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  // 3 Temporal layers, no spatial layers: Framerate decimation (4, 2, 1).
  cfg_.ss_number_layers = 1;
  cfg_.ts_number_layers = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;

  cfg_.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  cfg_.rc_target_bitrate = 200;
  ResetModel();
  // 40-20-40 bitrate allocation for 3 temporal layers.
  cfg_.layer_target_bitrate[0] = 40 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[1] = 60 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[2] = cfg_.rc_target_bitrate;
  aq_mode_ = 0;
  if (deadline_ == VPX_DL_REALTIME) {
    aq_mode_ = 3;
    cfg_.g_error_resilient = 1;
  }
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  for (int j = 0; j < static_cast<int>(cfg_.ts_number_layers); ++j) {
    ASSERT_GE(effective_datarate_[j], cfg_.layer_target_bitrate[j] * 0.85)
        << " The datarate for the file is lower than target by too much, "
           "for layer: "
        << j;
    ASSERT_LE(effective_datarate_[j], cfg_.layer_target_bitrate[j] * 1.20)
        << " The datarate for the file is greater than target by too much, "
           "for layer: "
        << j;
    // Expect some frame drops in this test: for this 200 frames test,
    // expect at least 10% and not more than 60% drops.
    ASSERT_GE(num_drops_, 20);
    ASSERT_LE(num_drops_, 280);
  }
}

// Check VP9 region of interest feature.
TEST_P(DatarateTestVP9RealTime, RegionOfInterest) {
  if (deadline_ != VPX_DL_REALTIME || set_cpu_used_ < 5) return;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);

  cfg_.rc_target_bitrate = 450;
  cfg_.g_w = 640;
  cfg_.g_h = 480;

  ResetModel();

  // Set ROI parameters
  use_roi_ = true;
  memset(&roi_, 0, sizeof(roi_));

  roi_.rows = (cfg_.g_h + 7) / 8;
  roi_.cols = (cfg_.g_w + 7) / 8;

  roi_.delta_q[1] = -20;
  roi_.delta_lf[1] = -20;
  memset(roi_.ref_frame, -1, sizeof(roi_.ref_frame));
  roi_.ref_frame[1] = 1;

  // Use 2 states: 1 is center square, 0 is the rest.
  roi_.roi_map = reinterpret_cast<uint8_t *>(
      calloc(roi_.rows * roi_.cols, sizeof(*roi_.roi_map)));
  ASSERT_NE(roi_.roi_map, nullptr);

  for (unsigned int i = 0; i < roi_.rows; ++i) {
    for (unsigned int j = 0; j < roi_.cols; ++j) {
      if (i > (roi_.rows >> 2) && i < ((roi_.rows * 3) >> 2) &&
          j > (roi_.cols >> 2) && j < ((roi_.cols * 3) >> 2)) {
        roi_.roi_map[i * roi_.cols + j] = 1;
      }
    }
  }

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_[0] * 0.90)
      << " The datarate for the file exceeds the target!";

  ASSERT_LE(cfg_.rc_target_bitrate, effective_datarate_[0] * 1.4)
      << " The datarate for the file missed the target!";

  free(roi_.roi_map);
}

// Params: speed setting, delta q UV.
class DatarateTestVP9RealTimeDeltaQUV
    : public DatarateTestVP9,
      public ::libvpx_test::CodecTestWith2Params<int, int> {
 public:
  DatarateTestVP9RealTimeDeltaQUV() : DatarateTestVP9(GET_PARAM(0)) {}
  ~DatarateTestVP9RealTimeDeltaQUV() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    set_cpu_used_ = GET_PARAM(1);
    ResetModel();
  }
};

TEST_P(DatarateTestVP9RealTimeDeltaQUV, DeltaQUV) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);

  cfg_.rc_target_bitrate = 450;
  cfg_.g_w = 640;
  cfg_.g_h = 480;

  ResetModel();

  delta_q_uv_ = GET_PARAM(2);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(cfg_.rc_target_bitrate, effective_datarate_[0] * 0.90)
      << " The datarate for the file exceeds the target!";

  ASSERT_LE(cfg_.rc_target_bitrate, effective_datarate_[0] * 1.4)
      << " The datarate for the file missed the target!";
}

// Params: test mode, speed setting and index for bitrate array.
class DatarateTestVP9PostEncodeDrop
    : public DatarateTestVP9,
      public ::libvpx_test::CodecTestWithParam<int> {
 public:
  DatarateTestVP9PostEncodeDrop() : DatarateTestVP9(GET_PARAM(0)) {}

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    set_cpu_used_ = GET_PARAM(1);
    ResetModel();
  }
};

// Check basic rate targeting for CBR mode, with 2 threads and dropped frames.
TEST_P(DatarateTestVP9PostEncodeDrop, PostEncodeDropScreenContent) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;
  // Encode using multiple threads.
  cfg_.g_threads = 2;
  cfg_.g_error_resilient = 0;
  tune_content_ = 1;
  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 300);
  cfg_.rc_target_bitrate = 300;
  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.15)
      << " The datarate for the file is greater than target by too much!";
}

using libvpx_test::ACMRandom;

class DatarateTestVP9FrameQp
    : public DatarateTestVP9,
      public ::testing::TestWithParam<const libvpx_test::CodecFactory *> {
 public:
  DatarateTestVP9FrameQp() : DatarateTestVP9(GetParam()), frame_(0) {}
  ~DatarateTestVP9FrameQp() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    ResetModel();
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    set_cpu_used_ = 7;
    DatarateTestVP9::PreEncodeFrameHook(video, encoder);
    frame_qp_ = static_cast<int>(rnd_.RandRange(64));
    encoder->Control(VP9E_SET_QUANTIZER_ONE_PASS, frame_qp_);
    frame_++;
  }

  void PostEncodeFrameHook(::libvpx_test::Encoder *encoder) override {
    int qp = 0;
    vpx_svc_layer_id_t layer_id;
    if (frame_ >= total_frame_) return;
    encoder->Control(VP8E_GET_LAST_QUANTIZER_64, &qp);
    ASSERT_EQ(frame_qp_, qp);
    encoder->Control(VP9E_GET_SVC_LAYER_ID, &layer_id);
    temporal_layer_id_ = layer_id.temporal_layer_id;
  }

  void MismatchHook(const vpx_image_t * /*img1*/,
                    const vpx_image_t * /*img2*/) override {
    if (frame_ >= total_frame_) return;
    ASSERT_TRUE(cfg_.temporal_layering_mode ==
                    VP9E_TEMPORAL_LAYERING_MODE_0212 &&
                temporal_layer_id_ == 2);
  }

 protected:
  int total_frame_;

 private:
  ACMRandom rnd_;
  int frame_qp_;
  int frame_;
  int temporal_layer_id_;
};

TEST_P(DatarateTestVP9FrameQp, VP9SetFrameQp) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  total_frame_ = 400;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, total_frame_);
  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(DatarateTestVP9FrameQp, VP9SetFrameQp3TemporalLayersBypass) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  // 3 Temporal layers, no spatial layers: Framerate decimation (4, 2, 1).
  cfg_.ss_number_layers = 1;
  cfg_.ts_number_layers = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;

  cfg_.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS;
  cfg_.rc_target_bitrate = 200;
  total_frame_ = 400;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, total_frame_);
  ResetModel();
  cfg_.layer_target_bitrate[0] = 40 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[1] = 60 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[2] = cfg_.rc_target_bitrate;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(DatarateTestVP9FrameQp, VP9SetFrameQp3TemporalLayersFixedMode) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  // 3 Temporal layers, no spatial layers: Framerate decimation (4, 2, 1).
  cfg_.ss_number_layers = 1;
  cfg_.ts_number_layers = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;

  cfg_.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_0212;
  cfg_.rc_target_bitrate = 200;
  cfg_.g_error_resilient = 1;
  total_frame_ = 400;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, total_frame_);
  ResetModel();
  cfg_.layer_target_bitrate[0] = 40 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[1] = 60 * cfg_.rc_target_bitrate / 100;
  cfg_.layer_target_bitrate[2] = cfg_.rc_target_bitrate;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

#if CONFIG_VP9_TEMPORAL_DENOISING
// Params: speed setting.
class DatarateTestVP9RealTimeDenoiser : public DatarateTestVP9RealTime {
 public:
  ~DatarateTestVP9RealTimeDenoiser() override = default;
};

// Check basic datarate targeting, for a single bitrate, when denoiser is on.
TEST_P(DatarateTestVP9RealTimeDenoiser, LowNoise) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 2;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);

  // For the temporal denoiser (#if CONFIG_VP9_TEMPORAL_DENOISING),
  // there is only one denoiser mode: denoiserYonly(which is 1),
  // but may add more modes in the future.
  cfg_.rc_target_bitrate = 400;
  ResetModel();
  // Turn on the denoiser.
  denoiser_on_ = 1;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.15)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic datarate targeting, for a single bitrate, when denoiser is on,
// for clip with high noise level. Use 2 threads.
TEST_P(DatarateTestVP9RealTimeDenoiser, HighNoise) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 2;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;
  cfg_.g_threads = 2;

  ::libvpx_test::Y4mVideoSource video("noisy_clip_640_360.y4m", 0, 200);

  // For the temporal denoiser (#if CONFIG_VP9_TEMPORAL_DENOISING),
  // there is only one denoiser mode: kDenoiserOnYOnly(which is 1),
  // but may add more modes in the future.
  cfg_.rc_target_bitrate = 1000;
  ResetModel();
  // Turn on the denoiser.
  denoiser_on_ = 1;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.15)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic datarate targeting, for a single bitrate, when denoiser is on,
// for 1280x720 clip with 4 threads.
TEST_P(DatarateTestVP9RealTimeDenoiser, 4threads) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 2;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;
  cfg_.g_threads = 4;

  ::libvpx_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 0, 300);

  // For the temporal denoiser (#if CONFIG_VP9_TEMPORAL_DENOISING),
  // there is only one denoiser mode: denoiserYonly(which is 1),
  // but may add more modes in the future.
  cfg_.rc_target_bitrate = 1000;
  ResetModel();
  // Turn on the denoiser.
  denoiser_on_ = 1;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.29)
      << " The datarate for the file is greater than target by too much!";
}

// Check basic datarate targeting, for a single bitrate, when denoiser is off
// and on.
TEST_P(DatarateTestVP9RealTimeDenoiser, DenoiserOffOn) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.rc_min_quantizer = 2;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;

  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);

  // For the temporal denoiser (#if CONFIG_VP9_TEMPORAL_DENOISING),
  // there is only one denoiser mode: denoiserYonly(which is 1),
  // but may add more modes in the future.
  cfg_.rc_target_bitrate = 400;
  ResetModel();
  // The denoiser is off by default.
  denoiser_on_ = 0;
  // Set the offon test flag.
  denoiser_offon_test_ = 1;
  denoiser_offon_period_ = 100;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  ASSERT_GE(effective_datarate_[0], cfg_.rc_target_bitrate * 0.85)
      << " The datarate for the file is lower than target by too much!";
  ASSERT_LE(effective_datarate_[0], cfg_.rc_target_bitrate * 1.15)
      << " The datarate for the file is greater than target by too much!";
}
#endif  // CONFIG_VP9_TEMPORAL_DENOISING

class DatarateTestVP9Psnr : public DatarateTestVP9,
                            public ::libvpx_test::CodecTestWithParam<int> {
 protected:
  DatarateTestVP9Psnr() : DatarateTestVP9(GET_PARAM(0)) {}
  ~DatarateTestVP9Psnr() override = default;

  void SetUp() override {
    InitializeConfig();
    cfg_.g_lag_in_frames = 0;
    SetMode(libvpx_test::kRealTime);
    set_cpu_used_ = 10;
    ResetModel();
    frame_flags_ = VPX_EFLAG_CALCULATE_PSNR;
    expect_psnr_ = true;
  }
  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    DatarateTestVP9::PreEncodeFrameHook(video, encoder);
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

TEST_P(DatarateTestVP9Psnr, PerFramePsnr) {
  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 100);

  ResetModel();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

VP9_INSTANTIATE_TEST_SUITE(DatarateTestVP9RealTimeMultiBR,
                           ::testing::Range(5, 10), ::testing::Range(0, 4));

VP9_INSTANTIATE_TEST_SUITE(DatarateTestVP9LargeVBR, ::testing::Range(5, 9),
                           ::testing::Range(0, 2));

VP9_INSTANTIATE_TEST_SUITE(DatarateTestVP9RealTime, ::testing::Range(5, 10));

#if CONFIG_VP9
INSTANTIATE_TEST_SUITE_P(
    VP9, DatarateTestVP9FrameQp,
    ::testing::Values(
        static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP9)));
#endif

VP9_INSTANTIATE_TEST_SUITE(DatarateTestVP9RealTimeDeltaQUV,
                           ::testing::Range(5, 10),
                           ::testing::Values(-5, -10, -15));

VP9_INSTANTIATE_TEST_SUITE(DatarateTestVP9PostEncodeDrop,
                           ::testing::Range(5, 6));

#if CONFIG_VP9_TEMPORAL_DENOISING
VP9_INSTANTIATE_TEST_SUITE(DatarateTestVP9RealTimeDenoiser,
                           ::testing::Range(5, 10));
#endif

VP9_INSTANTIATE_TEST_SUITE(DatarateTestVP9Psnr,
                           ::testing::Values(::libvpx_test::kRealTime));
}  // namespace
