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
#include "test/svc_test.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vpx/vpx_codec.h"
#include "vpx_ports/bitops.h"

namespace svc_test {
namespace {

enum INTER_LAYER_PRED {
  // Inter-layer prediction is on on all frames.
  INTER_LAYER_PRED_ON,
  // Inter-layer prediction is off on all frames.
  INTER_LAYER_PRED_OFF,
  // Inter-layer prediction is off on non-key frames and non-sync frames.
  INTER_LAYER_PRED_OFF_NONKEY,
  // Inter-layer prediction is on on all frames, but constrained such
  // that any layer S (> 0) can only predict from previous spatial
  // layer S-1, from the same superframe.
  INTER_LAYER_PRED_ON_CONSTRAINED
};

class DatarateOnePassCbrSvc : public OnePassCbrSvc {
 public:
  explicit DatarateOnePassCbrSvc(const ::libvpx_test::CodecFactory *codec)
      : OnePassCbrSvc(codec) {
    inter_layer_pred_mode_ = 0;
  }

 protected:
  ~DatarateOnePassCbrSvc() override = default;

  virtual void ResetModel() {
    last_pts_ = 0;
    duration_ = 0.0;
    mismatch_psnr_ = 0.0;
    mismatch_nframes_ = 0;
    denoiser_on_ = 0;
    tune_content_ = 0;
    base_speed_setting_ = 5;
    spatial_layer_id_ = 0;
    temporal_layer_id_ = 0;
    update_pattern_ = 0;
    memset(bits_in_buffer_model_, 0, sizeof(bits_in_buffer_model_));
    memset(bits_total_, 0, sizeof(bits_total_));
    memset(layer_target_avg_bandwidth_, 0, sizeof(layer_target_avg_bandwidth_));
    dynamic_drop_layer_ = false;
    single_layer_resize_ = false;
    change_bitrate_ = false;
    last_pts_ref_ = 0;
    middle_bitrate_ = 0;
    top_bitrate_ = 0;
    superframe_count_ = -1;
    key_frame_spacing_ = 9999;
    num_nonref_frames_ = 0;
    layer_framedrop_ = 0;
    force_key_ = 0;
    force_key_test_ = 0;
    insert_layer_sync_ = 0;
    layer_sync_on_base_ = 0;
    force_intra_only_frame_ = 0;
    superframe_has_intra_only_ = 0;
    use_post_encode_drop_ = 0;
    denoiser_off_on_ = false;
    denoiser_enable_layers_ = false;
    num_resize_down_ = 0;
    num_resize_up_ = 0;
    for (int i = 0; i < VPX_MAX_LAYERS; i++) {
      prev_frame_width_[i] = 320;
      prev_frame_height_[i] = 240;
    }
    ksvc_flex_noupd_tlenh_ = false;
    external_resize_dynamic_drop_layer_ = false;
    external_resize_pattern_ = 0;
    superframe_cnt_ = 0;
  }
  void BeginPassHook(unsigned int /*pass*/) override {}

  // Example pattern for spatial layers and 2 temporal layers used in the
  // bypass/flexible mode. The pattern corresponds to the pattern
  // VP9E_TEMPORAL_LAYERING_MODE_0101 (temporal_layering_mode == 2) used in
  // non-flexible mode, except that we disable inter-layer prediction.
  void set_frame_flags_bypass_mode(int tl, int num_spatial_layers,
                                   int is_key_frame,
                                   vpx_svc_ref_frame_config_t *ref_frame_config,
                                   int noupdate_tlenh) {
    for (int sl = 0; sl < num_spatial_layers; ++sl)
      ref_frame_config->update_buffer_slot[sl] = 0;

    for (int sl = 0; sl < num_spatial_layers; ++sl) {
      if (tl == 0) {
        ref_frame_config->lst_fb_idx[sl] = sl;
        if (sl) {
          if (is_key_frame) {
            ref_frame_config->lst_fb_idx[sl] = sl - 1;
            ref_frame_config->gld_fb_idx[sl] = sl;
          } else {
            ref_frame_config->gld_fb_idx[sl] = sl - 1;
          }
        } else {
          ref_frame_config->gld_fb_idx[sl] = 0;
        }
        ref_frame_config->alt_fb_idx[sl] = 0;
      } else if (tl == 1) {
        ref_frame_config->lst_fb_idx[sl] = sl;
        ref_frame_config->gld_fb_idx[sl] =
            VPXMIN(REF_FRAMES - 1, num_spatial_layers + sl - 1);
        ref_frame_config->alt_fb_idx[sl] =
            VPXMIN(REF_FRAMES - 1, num_spatial_layers + sl);
      }
      if (!tl) {
        if (!sl) {
          ref_frame_config->reference_last[sl] = 1;
          ref_frame_config->reference_golden[sl] = 0;
          ref_frame_config->reference_alt_ref[sl] = 0;
          ref_frame_config->update_buffer_slot[sl] |=
              1 << ref_frame_config->lst_fb_idx[sl];
        } else {
          if (is_key_frame) {
            ref_frame_config->reference_last[sl] = 1;
            ref_frame_config->reference_golden[sl] = 0;
            ref_frame_config->reference_alt_ref[sl] = 0;
            ref_frame_config->update_buffer_slot[sl] |=
                1 << ref_frame_config->gld_fb_idx[sl];
          } else {
            ref_frame_config->reference_last[sl] = 1;
            ref_frame_config->reference_golden[sl] = 0;
            ref_frame_config->reference_alt_ref[sl] = 0;
            ref_frame_config->update_buffer_slot[sl] |=
                1 << ref_frame_config->lst_fb_idx[sl];
          }
        }
      } else if (tl == 1) {
        if (!sl) {
          ref_frame_config->reference_last[sl] = 1;
          ref_frame_config->reference_golden[sl] = 0;
          ref_frame_config->reference_alt_ref[sl] = 0;
          ref_frame_config->update_buffer_slot[sl] |=
              1 << ref_frame_config->alt_fb_idx[sl];
        } else {
          ref_frame_config->reference_last[sl] = 1;
          ref_frame_config->reference_golden[sl] = 0;
          ref_frame_config->reference_alt_ref[sl] = 0;
          // Non reference frame on top temporal top spatial.
          ref_frame_config->update_buffer_slot[sl] = 0;
        }
        // Force no update on all spatial layers for temporal enhancement layer
        // frames.
        if (noupdate_tlenh) ref_frame_config->update_buffer_slot[sl] = 0;
      }
    }
  }

  void CheckLayerRateTargeting(int num_spatial_layers, int num_temporal_layers,
                               double thresh_overshoot,
                               double thresh_undershoot) const {
    for (int sl = 0; sl < num_spatial_layers; ++sl)
      for (int tl = 0; tl < num_temporal_layers; ++tl) {
        const int layer = sl * num_temporal_layers + tl;
        ASSERT_GE(cfg_.layer_target_bitrate[layer],
                  file_datarate_[layer] * thresh_overshoot)
            << " The datarate for the file exceeds the target by too much!";
        ASSERT_LE(cfg_.layer_target_bitrate[layer],
                  file_datarate_[layer] * thresh_undershoot)
            << " The datarate for the file is lower than the target by too "
               "much!";
      }
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    PreEncodeFrameHookSetup(video, encoder);

    if (video->frame() == 0) {
      if (force_intra_only_frame_) {
        // Decoder sets the color_space for Intra-only frames
        // to BT_601 (see line 1810 in vp9_decodeframe.c).
        // So set it here in these tess to avoid encoder-decoder
        // mismatch check on color space setting.
        encoder->Control(VP9E_SET_COLOR_SPACE, VPX_CS_BT_601);
      }
      encoder->Control(VP9E_SET_NOISE_SENSITIVITY, denoiser_on_);
      encoder->Control(VP9E_SET_TUNE_CONTENT, tune_content_);
      encoder->Control(VP9E_SET_SVC_INTER_LAYER_PRED, inter_layer_pred_mode_);

      if (layer_framedrop_) {
        vpx_svc_frame_drop_t svc_drop_frame;
        svc_drop_frame.framedrop_mode = LAYER_DROP;
        for (int i = 0; i < number_spatial_layers_; i++)
          svc_drop_frame.framedrop_thresh[i] = 30;
        svc_drop_frame.max_consec_drop = 30;
        encoder->Control(VP9E_SET_SVC_FRAME_DROP_LAYER, &svc_drop_frame);
      }

      if (use_post_encode_drop_) {
        encoder->Control(VP9E_SET_POSTENCODE_DROP, use_post_encode_drop_);
      }
      // We want to force external resize on the very first frame.
      if (external_resize_dynamic_drop_layer_) video->Next();
    }

    if (denoiser_off_on_) {
      encoder->Control(VP9E_SET_AQ_MODE, 3);
      // Set inter_layer_pred to INTER_LAYER_PRED_OFF_NONKEY (K-SVC).
      encoder->Control(VP9E_SET_SVC_INTER_LAYER_PRED, 2);
      if (!denoiser_enable_layers_) {
        if (video->frame() == 0)
          encoder->Control(VP9E_SET_NOISE_SENSITIVITY, 0);
        else if (video->frame() == 100)
          encoder->Control(VP9E_SET_NOISE_SENSITIVITY, 1);
      } else {
        // Cumulative bitrates for top spatial layers, for
        // 3 temporal layers.
        if (video->frame() == 0) {
          encoder->Control(VP9E_SET_NOISE_SENSITIVITY, 0);
          // Change layer bitrates to set top spatial layer to 0.
          // This is for 3 spatial 3 temporal layers.
          // This will trigger skip encoding/dropping of top spatial layer.
          cfg_.rc_target_bitrate -= cfg_.layer_target_bitrate[8];
          for (int i = 0; i < 3; i++)
            bitrate_sl3_[i] = cfg_.layer_target_bitrate[i + 6];
          cfg_.layer_target_bitrate[6] = 0;
          cfg_.layer_target_bitrate[7] = 0;
          cfg_.layer_target_bitrate[8] = 0;
          encoder->Config(&cfg_);
        } else if (video->frame() == 100) {
          // Change layer bitrates to non-zero on top spatial layer.
          // This will trigger skip encoding of top spatial layer
          // on key frame (period = 100).
          for (int i = 0; i < 3; i++)
            cfg_.layer_target_bitrate[i + 6] = bitrate_sl3_[i];
          cfg_.rc_target_bitrate += cfg_.layer_target_bitrate[8];
          encoder->Config(&cfg_);
        } else if (video->frame() == 120) {
          // Enable denoiser and top spatial layer after key frame (period is
          // 100).
          encoder->Control(VP9E_SET_NOISE_SENSITIVITY, 1);
        }
      }
    }

    if (ksvc_flex_noupd_tlenh_) {
      vpx_svc_layer_id_t layer_id;
      layer_id.spatial_layer_id = 0;
      layer_id.temporal_layer_id = (video->frame() % 2 != 0);
      temporal_layer_id_ = layer_id.temporal_layer_id;
      for (int i = 0; i < number_spatial_layers_; i++) {
        layer_id.temporal_layer_id_per_spatial[i] = temporal_layer_id_;
        ref_frame_config_.duration[i] = 1;
      }
      encoder->Control(VP9E_SET_SVC_LAYER_ID, &layer_id);
      set_frame_flags_bypass_mode(layer_id.temporal_layer_id,
                                  number_spatial_layers_, 0, &ref_frame_config_,
                                  1);
      encoder->Control(VP9E_SET_SVC_REF_FRAME_CONFIG, &ref_frame_config_);
    }

    if (update_pattern_ && video->frame() >= 100) {
      vpx_svc_layer_id_t layer_id;
      if (video->frame() == 100) {
        cfg_.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS;
        encoder->Config(&cfg_);
      }
      // Set layer id since the pattern changed.
      layer_id.spatial_layer_id = 0;
      layer_id.temporal_layer_id = (video->frame() % 2 != 0);
      temporal_layer_id_ = layer_id.temporal_layer_id;
      for (int i = 0; i < number_spatial_layers_; i++) {
        layer_id.temporal_layer_id_per_spatial[i] = temporal_layer_id_;
        ref_frame_config_.duration[i] = 1;
      }
      encoder->Control(VP9E_SET_SVC_LAYER_ID, &layer_id);
      set_frame_flags_bypass_mode(layer_id.temporal_layer_id,
                                  number_spatial_layers_, 0, &ref_frame_config_,
                                  0);
      encoder->Control(VP9E_SET_SVC_REF_FRAME_CONFIG, &ref_frame_config_);
    }

    if (change_bitrate_ && video->frame() == 200) {
      duration_ = (last_pts_ + 1) * timebase_;
      for (int sl = 0; sl < number_spatial_layers_; ++sl) {
        for (int tl = 0; tl < number_temporal_layers_; ++tl) {
          const int layer = sl * number_temporal_layers_ + tl;
          const double file_size_in_kb = bits_total_[layer] / 1000.;
          file_datarate_[layer] = file_size_in_kb / duration_;
        }
      }

      CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_,
                              0.78, 1.15);

      memset(file_datarate_, 0, sizeof(file_datarate_));
      memset(bits_total_, 0, sizeof(bits_total_));
      int64_t bits_in_buffer_model_tmp[VPX_MAX_LAYERS];
      last_pts_ref_ = last_pts_;
      // Set new target bitarate.
      cfg_.rc_target_bitrate = cfg_.rc_target_bitrate >> 1;
      // Buffer level should not reset on dynamic bitrate change.
      memcpy(bits_in_buffer_model_tmp, bits_in_buffer_model_,
             sizeof(bits_in_buffer_model_));
      AssignLayerBitrates();
      memcpy(bits_in_buffer_model_, bits_in_buffer_model_tmp,
             sizeof(bits_in_buffer_model_));

      // Change config to update encoder with new bitrate configuration.
      encoder->Config(&cfg_);
    }

    if (external_resize_dynamic_drop_layer_) {
      frame_flags_ = 0;
      for (int i = 0; i < 9; ++i) {
        svc_params_.min_quantizers[i] = 20;
        svc_params_.max_quantizers[i] = 56;
      }
      if (video->frame() == 1 || video->frame() == 150) {
        // Set the new top width/height for external resize.
        top_sl_width_ = video->img()->d_w;
        top_sl_height_ = video->img()->d_h;
        for (int i = 0; i < 9; ++i) {
          bitrate_layer_[i] = cfg_.layer_target_bitrate[i];
        }
        if (external_resize_pattern_ == 1) {
          // Input size is 1/4. 2 top spatial layers are dropped.
          // This will trigger skip encoding/dropping of two top spatial layers.
          cfg_.rc_target_bitrate -=
              cfg_.layer_target_bitrate[5] + cfg_.layer_target_bitrate[8];
          for (int i = 3; i < 9; ++i) {
            cfg_.layer_target_bitrate[i] = 0;
          }
          for (int sl = 0; sl < 3; sl++) {
            svc_params_.scaling_factor_num[sl] = 1;
            svc_params_.scaling_factor_den[sl] = 1;
          }
        } else if (external_resize_pattern_ == 2) {
          // Input size is 1/2. Top spatial layer is dropped.
          // This will trigger skip encoding/dropping of top spatial layer.
          cfg_.rc_target_bitrate -= cfg_.layer_target_bitrate[8];
          for (int i = 6; i < 9; ++i) {
            cfg_.layer_target_bitrate[i] = 0;
          }
          svc_params_.scaling_factor_num[0] = 1;
          svc_params_.scaling_factor_den[0] = 2;
          svc_params_.scaling_factor_num[1] = 1;
          svc_params_.scaling_factor_den[1] = 1;
          svc_params_.scaling_factor_num[2] = 1;
          svc_params_.scaling_factor_den[2] = 1;
        }
        encoder->Config(&cfg_);
        encoder->Control(VP9E_SET_SVC_PARAMETERS, &svc_params_);
      } else if (video->frame() == 50 || video->frame() == 200) {
        top_sl_width_ = video->img()->d_w;
        top_sl_height_ = video->img()->d_h;
        if (external_resize_pattern_ == 1) {
          // Input size is 1/2. Change layer bitrates to set top layer to 0.
          // This will trigger skip encoding/dropping of top spatial layer.
          cfg_.rc_target_bitrate += bitrate_layer_[5];
          for (int i = 3; i < 6; ++i) {
            cfg_.layer_target_bitrate[i] = bitrate_layer_[i];
          }
          svc_params_.scaling_factor_num[0] = 1;
          svc_params_.scaling_factor_den[0] = 2;
          svc_params_.scaling_factor_num[1] = 1;
          svc_params_.scaling_factor_den[1] = 1;
          svc_params_.scaling_factor_num[2] = 1;
          svc_params_.scaling_factor_den[2] = 1;
        } else if (external_resize_pattern_ == 2) {
          // Input size is 1/4. Change layer bitrates to set two top layers to
          // 0. This will trigger skip encoding/dropping of two top spatial
          // layers.
          cfg_.rc_target_bitrate -= bitrate_layer_[5];
          for (int i = 3; i < 6; ++i) {
            cfg_.layer_target_bitrate[i] = 0;
          }
          for (int sl = 0; sl < 3; sl++) {
            svc_params_.scaling_factor_num[sl] = 1;
            svc_params_.scaling_factor_den[sl] = 1;
          }
        }
        encoder->Config(&cfg_);
        encoder->Control(VP9E_SET_SVC_PARAMETERS, &svc_params_);
      } else if (video->frame() == 100 || video->frame() == 250) {
        top_sl_width_ = video->img()->d_w;
        top_sl_height_ = video->img()->d_h;
        // Input is original size. Change layer bitrates to nonzero for all
        // layers.
        cfg_.rc_target_bitrate =
            bitrate_layer_[2] + bitrate_layer_[5] + bitrate_layer_[8];
        for (int i = 0; i < 9; ++i) {
          cfg_.layer_target_bitrate[i] = bitrate_layer_[i];
        }
        svc_params_.scaling_factor_num[0] = 1;
        svc_params_.scaling_factor_den[0] = 4;
        svc_params_.scaling_factor_num[1] = 1;
        svc_params_.scaling_factor_den[1] = 2;
        svc_params_.scaling_factor_num[2] = 1;
        svc_params_.scaling_factor_den[2] = 1;
        encoder->Config(&cfg_);
        encoder->Control(VP9E_SET_SVC_PARAMETERS, &svc_params_);
      }
    } else if (dynamic_drop_layer_ && !single_layer_resize_) {
      if (video->frame() == 0) {
        // Change layer bitrates to set top layers to 0. This will trigger skip
        // encoding/dropping of top two spatial layers.
        cfg_.rc_target_bitrate -=
            (cfg_.layer_target_bitrate[1] + cfg_.layer_target_bitrate[2]);
        middle_bitrate_ = cfg_.layer_target_bitrate[1];
        top_bitrate_ = cfg_.layer_target_bitrate[2];
        cfg_.layer_target_bitrate[1] = 0;
        cfg_.layer_target_bitrate[2] = 0;
        encoder->Config(&cfg_);
      } else if (video->frame() == 50) {
        // Change layer bitrates to non-zero on two top spatial layers.
        // This will trigger skip encoding of top two spatial layers.
        cfg_.layer_target_bitrate[1] = middle_bitrate_;
        cfg_.layer_target_bitrate[2] = top_bitrate_;
        cfg_.rc_target_bitrate +=
            cfg_.layer_target_bitrate[2] + cfg_.layer_target_bitrate[1];
        encoder->Config(&cfg_);
      } else if (video->frame() == 100) {
        // Change layer bitrates to set top layers to 0. This will trigger skip
        // encoding/dropping of top two spatial layers.
        cfg_.rc_target_bitrate -=
            (cfg_.layer_target_bitrate[1] + cfg_.layer_target_bitrate[2]);
        middle_bitrate_ = cfg_.layer_target_bitrate[1];
        top_bitrate_ = cfg_.layer_target_bitrate[2];
        cfg_.layer_target_bitrate[1] = 0;
        cfg_.layer_target_bitrate[2] = 0;
        encoder->Config(&cfg_);
      } else if (video->frame() == 150) {
        // Change layer bitrate on second layer to non-zero to start
        // encoding it again.
        cfg_.layer_target_bitrate[1] = middle_bitrate_;
        cfg_.rc_target_bitrate += cfg_.layer_target_bitrate[1];
        encoder->Config(&cfg_);
      } else if (video->frame() == 200) {
        // Change layer bitrate on top layer to non-zero to start
        // encoding it again.
        cfg_.layer_target_bitrate[2] = top_bitrate_;
        cfg_.rc_target_bitrate += cfg_.layer_target_bitrate[2];
        encoder->Config(&cfg_);
      }
    } else if (dynamic_drop_layer_ && single_layer_resize_) {
      // Change layer bitrates to set top layers to 0. This will trigger skip
      // encoding/dropping of top spatial layers.
      if (video->frame() == 2) {
        cfg_.rc_target_bitrate -=
            (cfg_.layer_target_bitrate[1] + cfg_.layer_target_bitrate[2]);
        middle_bitrate_ = cfg_.layer_target_bitrate[1];
        top_bitrate_ = cfg_.layer_target_bitrate[2];
        cfg_.layer_target_bitrate[1] = 0;
        cfg_.layer_target_bitrate[2] = 0;
        // Set spatial layer 0 to a very low bitrate to trigger resize.
        cfg_.layer_target_bitrate[0] = 30;
        cfg_.rc_target_bitrate = cfg_.layer_target_bitrate[0];
        encoder->Config(&cfg_);
      } else if (video->frame() == 100) {
        // Set base spatial layer to very high to go back up to original size.
        cfg_.layer_target_bitrate[0] = 400;
        cfg_.rc_target_bitrate = cfg_.layer_target_bitrate[0];
        encoder->Config(&cfg_);
      }
    } else if (!dynamic_drop_layer_ && single_layer_resize_) {
      if (video->frame() == 2) {
        cfg_.layer_target_bitrate[0] = 30;
        cfg_.layer_target_bitrate[1] = 50;
        cfg_.rc_target_bitrate =
            (cfg_.layer_target_bitrate[0] + cfg_.layer_target_bitrate[1]);
        encoder->Config(&cfg_);
      } else if (video->frame() == 160) {
        cfg_.layer_target_bitrate[0] = 1500;
        cfg_.layer_target_bitrate[1] = 2000;
        cfg_.rc_target_bitrate =
            (cfg_.layer_target_bitrate[0] + cfg_.layer_target_bitrate[1]);
        encoder->Config(&cfg_);
      }
    }
    if (force_key_test_ && force_key_) frame_flags_ = VPX_EFLAG_FORCE_KF;

    if (insert_layer_sync_) {
      vpx_svc_spatial_layer_sync_t svc_layer_sync;
      svc_layer_sync.base_layer_intra_only = 0;
      for (int i = 0; i < number_spatial_layers_; i++)
        svc_layer_sync.spatial_layer_sync[i] = 0;
      if (force_intra_only_frame_) {
        superframe_has_intra_only_ = 0;
        if (video->frame() == 0) {
          svc_layer_sync.base_layer_intra_only = 1;
          svc_layer_sync.spatial_layer_sync[0] = 1;
          encoder->Control(VP9E_SET_SVC_SPATIAL_LAYER_SYNC, &svc_layer_sync);
          superframe_has_intra_only_ = 1;
        } else if (video->frame() == 100) {
          svc_layer_sync.base_layer_intra_only = 1;
          svc_layer_sync.spatial_layer_sync[0] = 1;
          encoder->Control(VP9E_SET_SVC_SPATIAL_LAYER_SYNC, &svc_layer_sync);
          superframe_has_intra_only_ = 1;
        }
      } else {
        layer_sync_on_base_ = 0;
        if (video->frame() == 150) {
          svc_layer_sync.spatial_layer_sync[1] = 1;
          encoder->Control(VP9E_SET_SVC_SPATIAL_LAYER_SYNC, &svc_layer_sync);
        } else if (video->frame() == 240) {
          svc_layer_sync.spatial_layer_sync[2] = 1;
          encoder->Control(VP9E_SET_SVC_SPATIAL_LAYER_SYNC, &svc_layer_sync);
        } else if (video->frame() == 320) {
          svc_layer_sync.spatial_layer_sync[0] = 1;
          layer_sync_on_base_ = 1;
          encoder->Control(VP9E_SET_SVC_SPATIAL_LAYER_SYNC, &svc_layer_sync);
        }
      }
    }

    const vpx_rational_t tb = video->timebase();
    timebase_ = static_cast<double>(tb.num) / tb.den;
    duration_ = 0;
    superframe_cnt_++;
  }

  vpx_codec_err_t parse_superframe_index(const uint8_t *data, size_t data_sz,
                                         uint32_t sizes[8], int *count) {
    uint8_t marker;
    marker = *(data + data_sz - 1);
    *count = 0;
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
      {
        uint32_t i, j;
        const uint8_t *x = &data[data_sz - index_sz + 1];
        for (i = 0; i < frames; ++i) {
          uint32_t this_sz = 0;

          for (j = 0; j < mag; ++j) this_sz |= (*x++) << (j * 8);
          sizes[i] = this_sz;
        }
        *count = frames;
      }
    }
    return VPX_CODEC_OK;
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    uint32_t sizes[8] = { 0 };
    uint32_t sizes_parsed[8] = { 0 };
    int count = 0;
    int num_layers_encoded = 0;
    last_pts_ = pkt->data.frame.pts;
    const bool key_frame =
        (pkt->data.frame.flags & VPX_FRAME_IS_KEY) ? true : false;
    if (external_resize_dynamic_drop_layer_) {
      // No key frames expected in stream, except for first.
      if (cfg_.kf_max_dist > 1000) {
        ASSERT_FALSE(key_frame && superframe_cnt_ > 1);
      }
    }
    if (key_frame) {
      // For test that inserts layer sync frames: requesting a layer_sync on
      // the base layer must force key frame. So if any key frame occurs after
      // first superframe it must due to layer sync on base spatial layer.
      if (superframe_count_ > 0 && insert_layer_sync_ &&
          !force_intra_only_frame_) {
        ASSERT_EQ(layer_sync_on_base_, 1);
      }
      temporal_layer_id_ = 0;
      superframe_count_ = 0;
    }
    parse_superframe_index(static_cast<const uint8_t *>(pkt->data.frame.buf),
                           pkt->data.frame.sz, sizes_parsed, &count);
    // Count may be less than number of spatial layers because of frame drops.
    if (number_spatial_layers_ > 1) {
      for (int sl = 0; sl < number_spatial_layers_; ++sl) {
        if (pkt->data.frame.spatial_layer_encoded[sl]) {
          sizes[sl] = sizes_parsed[num_layers_encoded];
          num_layers_encoded++;
        }
      }
    }
    // For superframe with Intra-only count will be +1 larger
    // because of no-show frame.
    if (force_intra_only_frame_ && superframe_has_intra_only_)
      ASSERT_EQ(count, num_layers_encoded + 1);
    else
      ASSERT_EQ(count, num_layers_encoded);

    // In the constrained frame drop mode, if a given spatial is dropped all
    // upper layers must be dropped too.
    if (!layer_framedrop_) {
      int num_layers_dropped = 0;
      for (int sl = 0; sl < number_spatial_layers_; ++sl) {
        if (!pkt->data.frame.spatial_layer_encoded[sl]) {
          // Check that all upper layers are dropped.
          num_layers_dropped++;
          for (int sl2 = sl + 1; sl2 < number_spatial_layers_; ++sl2)
            ASSERT_EQ(pkt->data.frame.spatial_layer_encoded[sl2], 0);
        }
      }
      if (num_layers_dropped == number_spatial_layers_ - 1)
        force_key_ = 1;
      else
        force_key_ = 0;
    }
    // Keep track of number of non-reference frames, needed for mismatch check.
    // Non-reference frames are top spatial and temporal layer frames,
    // for TL > 0.
    if (temporal_layer_id_ == number_temporal_layers_ - 1 &&
        temporal_layer_id_ > 0 &&
        pkt->data.frame.spatial_layer_encoded[number_spatial_layers_ - 1])
      num_nonref_frames_++;
    for (int sl = 0; sl < number_spatial_layers_; ++sl) {
      sizes[sl] = sizes[sl] << 3;
      // Update the total encoded bits per layer.
      // For temporal layers, update the cumulative encoded bits per layer.
      for (int tl = temporal_layer_id_; tl < number_temporal_layers_; ++tl) {
        const int layer = sl * number_temporal_layers_ + tl;
        bits_total_[layer] += static_cast<int64_t>(sizes[sl]);
        // Update the per-layer buffer level with the encoded frame size.
        bits_in_buffer_model_[layer] -= static_cast<int64_t>(sizes[sl]);
        // There should be no buffer underrun, except on the base
        // temporal layer, since there may be key frames there.
        // Fo short key frame spacing, buffer can underrun on individual frames.
        if (!key_frame && tl > 0 && key_frame_spacing_ < 100) {
          ASSERT_GE(bits_in_buffer_model_[layer], 0)
              << "Buffer Underrun at frame " << pkt->data.frame.pts;
        }
      }

      if (!single_layer_resize_ && sl < number_spatial_layers_ - 1) {
        unsigned int scaled_width = top_sl_width_ *
                                    svc_params_.scaling_factor_num[sl] /
                                    svc_params_.scaling_factor_den[sl];
        if (scaled_width % 2 != 0) scaled_width += 1;
        ASSERT_EQ(pkt->data.frame.width[sl], scaled_width);
        unsigned int scaled_height = top_sl_height_ *
                                     svc_params_.scaling_factor_num[sl] /
                                     svc_params_.scaling_factor_den[sl];
        if (scaled_height % 2 != 0) scaled_height += 1;
        ASSERT_EQ(pkt->data.frame.height[sl], scaled_height);
      } else if (superframe_count_ > 0) {
        if (pkt->data.frame.width[sl] < prev_frame_width_[sl] &&
            pkt->data.frame.height[sl] < prev_frame_height_[sl])
          num_resize_down_ += 1;
        if (pkt->data.frame.width[sl] > prev_frame_width_[sl] &&
            pkt->data.frame.height[sl] > prev_frame_height_[sl])
          num_resize_up_ += 1;
      }
      prev_frame_width_[sl] = pkt->data.frame.width[sl];
      prev_frame_height_[sl] = pkt->data.frame.height[sl];
    }
  }

  void EndPassHook() override {
    if (change_bitrate_) last_pts_ = last_pts_ - last_pts_ref_;
    duration_ = (last_pts_ + 1) * timebase_;
    for (int sl = 0; sl < number_spatial_layers_; ++sl) {
      for (int tl = 0; tl < number_temporal_layers_; ++tl) {
        const int layer = sl * number_temporal_layers_ + tl;
        const double file_size_in_kb = bits_total_[layer] / 1000.;
        file_datarate_[layer] = file_size_in_kb / duration_;
      }
    }
  }

  void MismatchHook(const vpx_image_t *img1, const vpx_image_t *img2) override {
    // TODO(marpan): Look into why an assert is triggered in compute_psnr
    // for mismatch frames for the special test case: ksvc_flex_noupd_tlenh.
    // Has to do with dropped frames in bypass/flexible svc mode.
    if (!ksvc_flex_noupd_tlenh_) {
      double mismatch_psnr = compute_psnr(img1, img2);
      mismatch_psnr_ += mismatch_psnr;
      ++mismatch_nframes_;
    }
  }

  unsigned int GetMismatchFrames() { return mismatch_nframes_; }
  unsigned int GetNonRefFrames() { return num_nonref_frames_; }

  vpx_codec_pts_t last_pts_;
  double timebase_;
  int64_t bits_total_[VPX_MAX_LAYERS];
  double duration_;
  double file_datarate_[VPX_MAX_LAYERS];
  size_t bits_in_last_frame_;
  double mismatch_psnr_;
  int denoiser_on_;
  int tune_content_;
  int spatial_layer_id_;
  bool dynamic_drop_layer_;
  bool single_layer_resize_;
  unsigned int top_sl_width_;
  unsigned int top_sl_height_;
  vpx_svc_ref_frame_config_t ref_frame_config_;
  int update_pattern_;
  bool change_bitrate_;
  vpx_codec_pts_t last_pts_ref_;
  int middle_bitrate_;
  int top_bitrate_;
  int key_frame_spacing_;
  int layer_framedrop_;
  int force_key_;
  int force_key_test_;
  int inter_layer_pred_mode_;
  int insert_layer_sync_;
  int layer_sync_on_base_;
  int force_intra_only_frame_;
  int superframe_has_intra_only_;
  int use_post_encode_drop_;
  int bitrate_sl3_[3];
  // Denoiser switched on the fly.
  bool denoiser_off_on_;
  // Top layer enabled on the fly.
  bool denoiser_enable_layers_;
  int num_resize_up_;
  int num_resize_down_;
  unsigned int prev_frame_width_[VPX_MAX_LAYERS];
  unsigned int prev_frame_height_[VPX_MAX_LAYERS];
  bool ksvc_flex_noupd_tlenh_;
  bool external_resize_dynamic_drop_layer_;
  int bitrate_layer_[9];
  int external_resize_pattern_;
  int superframe_cnt_;

 private:
  void SetConfig(const int num_temporal_layer) override {
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_error_resilient = 1;
    if (num_temporal_layer == 3) {
      cfg_.ts_rate_decimator[0] = 4;
      cfg_.ts_rate_decimator[1] = 2;
      cfg_.ts_rate_decimator[2] = 1;
      cfg_.temporal_layering_mode = 3;
    } else if (num_temporal_layer == 2) {
      cfg_.ts_rate_decimator[0] = 2;
      cfg_.ts_rate_decimator[1] = 1;
      cfg_.temporal_layering_mode = 2;
    } else if (num_temporal_layer == 1) {
      cfg_.ts_rate_decimator[0] = 1;
      cfg_.temporal_layering_mode = 0;
    }
  }

  unsigned int num_nonref_frames_;
  unsigned int mismatch_nframes_;
};

void ScaleForFrameNumber(unsigned int frame, unsigned int initial_w,
                         unsigned int initial_h, unsigned int *w,
                         unsigned int *h, int resize_pattern) {
  *w = initial_w;
  *h = initial_h;
  if (resize_pattern == 1) {
    if (frame < 50) {
      *w = initial_w / 4;
      *h = initial_h / 4;
    } else if (frame < 100) {
      *w = initial_w / 2;
      *h = initial_h / 2;
    } else if (frame < 150) {
      *w = initial_w;
      *h = initial_h;
    } else if (frame < 200) {
      *w = initial_w / 4;
      *h = initial_h / 4;
    } else if (frame < 250) {
      *w = initial_w / 2;
      *h = initial_h / 2;
    }
  } else if (resize_pattern == 2) {
    if (frame < 50) {
      *w = initial_w / 2;
      *h = initial_h / 2;
    } else if (frame < 100) {
      *w = initial_w / 4;
      *h = initial_h / 4;
    } else if (frame < 150) {
      *w = initial_w;
      *h = initial_h;
    } else if (frame < 200) {
      *w = initial_w / 2;
      *h = initial_h / 2;
    } else if (frame < 250) {
      *w = initial_w / 4;
      *h = initial_h / 4;
    }
  }
}

class ResizingVideoSource : public ::libvpx_test::DummyVideoSource {
 public:
  ResizingVideoSource(int width, int height) {
    top_width_ = width;
    top_height_ = height;
    SetSize(top_width_, top_height_);
    limit_ = 300;
  }
  int external_resize_pattern_ = 1;
  int force_zero_source_ = 0;
  int top_width_;
  int top_height_;
  ~ResizingVideoSource() override = default;

 protected:
  void Next() override {
    ++frame_;
    unsigned int width = 0;
    unsigned int height = 0;
    libvpx_test::ACMRandom rnd(libvpx_test::ACMRandom::DeterministicSeed());
    ScaleForFrameNumber(frame_, top_width_, top_height_, &width, &height,
                        external_resize_pattern_);
    SetSize(width, height);
    FillFrame();
    unsigned char *image = img_->planes[0];
    for (size_t i = 0; i < raw_sz_; ++i) {
      image[i] = rnd.Rand8();
      if (force_zero_source_ && frame_ % 20 == 0) image[i] = 0;
    }
  }
};

// Params: speed setting.
class DatarateOnePassCbrSvcSingleBR
    : public DatarateOnePassCbrSvc,
      public ::libvpx_test::CodecTestWithParam<int> {
 public:
  DatarateOnePassCbrSvcSingleBR() : DatarateOnePassCbrSvc(GET_PARAM(0)) {
    memset(&svc_params_, 0, sizeof(svc_params_));
  }
  ~DatarateOnePassCbrSvcSingleBR() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    speed_setting_ = GET_PARAM(1);
    ResetModel();
  }
};

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and 3
// temporal layers, for 4:4:4 Profile 1.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL3TL444Profile1) {
  SetSvcConfig(3, 3);
  ::libvpx_test::Y4mVideoSource video("rush_hour_444.y4m", 0, 140);
  cfg_.g_profile = 1;
  cfg_.g_bit_depth = VPX_BITS_8;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.kf_max_dist = 9999;

  top_sl_width_ = 352;
  top_sl_height_ = 288;
  cfg_.rc_target_bitrate = 500;
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 2 spatial layers and 3
// temporal layers, for 4:2:2 Profile 1.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc2SL3TL422Profile1) {
  SetSvcConfig(2, 3);
  ::libvpx_test::Y4mVideoSource video("park_joy_90p_8_422.y4m", 0, 20);
  cfg_.g_profile = 1;
  cfg_.g_bit_depth = VPX_BITS_8;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.kf_max_dist = 9999;

  top_sl_width_ = 160;
  top_sl_height_ = 90;
  cfg_.rc_target_bitrate = 500;
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Use large under/over shoot thresholds as this is a very short clip,
  // so not good for testing rate-targeting.
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.5,
                          1.7);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

#if CONFIG_VP9_HIGHBITDEPTH
// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and 3
// temporal layers, for Profle 2 10bit.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL3TL10bitProfile2) {
  SetSvcConfig(3, 3);
  ::libvpx_test::Y4mVideoSource video("park_joy_90p_10_420_20f.y4m", 0, 20);
  cfg_.g_profile = 2;
  cfg_.g_bit_depth = VPX_BITS_10;
  cfg_.g_input_bit_depth = VPX_BITS_10;
  if (cfg_.g_bit_depth > 8) init_flags_ |= VPX_CODEC_USE_HIGHBITDEPTH;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.kf_max_dist = 9999;

  top_sl_width_ = 160;
  top_sl_height_ = 90;
  cfg_.rc_target_bitrate = 500;
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // TODO(marpan/jianj): Comment out the rate-target checking for now
  // as superframe parsing to get frame size needs to be fixed for
  // high bitdepth.
  /*
  // Use large under/over shoot thresholds as this is a very short clip,
  // so not good for testing rate-targeting.
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.5,
                          1.7);
  */
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and 3
// temporal layers, for Profle 2 12bit.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL3TL12bitProfile2) {
  SetSvcConfig(3, 3);
  ::libvpx_test::Y4mVideoSource video("park_joy_90p_12_420_20f.y4m", 0, 20);
  cfg_.g_profile = 2;
  cfg_.g_bit_depth = VPX_BITS_12;
  cfg_.g_input_bit_depth = VPX_BITS_12;
  if (cfg_.g_bit_depth > 8) init_flags_ |= VPX_CODEC_USE_HIGHBITDEPTH;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.kf_max_dist = 9999;

  top_sl_width_ = 160;
  top_sl_height_ = 90;
  cfg_.rc_target_bitrate = 500;
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // TODO(marpan/jianj): Comment out the rate-target checking for now
  // as superframe parsing to get frame size needs to be fixed for
  // high bitdepth.
  /*
  // Use large under/over shoot thresholds as this is a very short clip,
  // so not good for testing rate-targeting.
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.5,
                          1.7);
  */
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}
#endif

// Check basic rate targeting for 1 pass CBR SVC: 2 spatial layers and 1
// temporal layer, with screen content mode on and same speed setting for all
// layers.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc2SL1TLScreenContent1) {
  SetSvcConfig(2, 1);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 10;
  cfg_.kf_max_dist = 9999;

  ::libvpx_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 0, 60);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  cfg_.rc_target_bitrate = 500;
  ResetModel();
  tune_content_ = 1;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and
// 3 temporal layers, with force key frame after frame drop
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL3TLForceKey) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  cfg_.rc_target_bitrate = 100;
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.25);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and
// 2 temporal layers, with a change on the fly from the fixed SVC pattern to one
// generate via SVC_SET_REF_FRAME_CONFIG. The new pattern also disables
// inter-layer prediction.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL2TLDynamicPatternChange) {
  SetSvcConfig(3, 2);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  cfg_.rc_target_bitrate = 800;
  ResetModel();
  // Change SVC pattern on the fly.
  update_pattern_ = 1;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC with 3 spatial and 3 temporal
// layers, for inter_layer_pred=OffKey (K-SVC) and on the fly switching
// of denoiser from off to on (on at frame = 100). Key frame period is set to
// 1000 so denoise is enabled on non-key.
TEST_P(DatarateOnePassCbrSvcSingleBR,
       OnePassCbrSvc3SL3TL_DenoiserOffOnFixedLayers) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 1000;
  ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv", 1280,
                                       720, 30, 1, 0, 300);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  cfg_.rc_target_bitrate = 1000;
  ResetModel();
  denoiser_off_on_ = true;
  denoiser_enable_layers_ = false;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Don't check rate targeting on two top spatial layer since they will be
  // skipped for part of the sequence.
  CheckLayerRateTargeting(number_spatial_layers_ - 2, number_temporal_layers_,
                          0.78, 1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC with 3 spatial and 3 temporal
// layers, for inter_layer_pred=OffKey (K-SVC) and on the fly switching
// of denoiser from off to on, for dynamic layers. Start at 2 spatial layers
// and enable 3rd spatial layer at frame = 100. Use periodic key frame with
// period 100 so enabling of spatial layer occurs at key frame. Enable denoiser
// at frame > 100, after the key frame sync.
TEST_P(DatarateOnePassCbrSvcSingleBR,
       OnePassCbrSvc3SL3TL_DenoiserOffOnEnableLayers) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 0;
  cfg_.kf_max_dist = 100;
  ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv", 1280,
                                       720, 30, 1, 0, 300);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  cfg_.rc_target_bitrate = 1000;
  ResetModel();
  denoiser_off_on_ = true;
  denoiser_enable_layers_ = true;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Don't check rate targeting on two top spatial layer since they will be
  // skipped for part of the sequence.
  CheckLayerRateTargeting(number_spatial_layers_ - 2, number_temporal_layers_,
                          0.78, 1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC with 3 spatial layers and on
// the fly switching to 1 and then 2 and back to 3 spatial layers. This switch
// is done by setting spatial layer bitrates to 0, and then back to non-zero,
// during the sequence.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL_DisableEnableLayers) {
  SetSvcConfig(3, 1);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.temporal_layering_mode = 0;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  cfg_.rc_target_bitrate = 800;
  ResetModel();
  dynamic_drop_layer_ = true;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Don't check rate targeting on two top spatial layer since they will be
  // skipped for part of the sequence.
  CheckLayerRateTargeting(number_spatial_layers_ - 2, number_temporal_layers_,
                          0.78, 1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC with 2 spatial layers and on
// the fly switching to 1 spatial layer with dynamic resize enabled.
// The resizer will resize the single layer down and back up again, as the
// bitrate goes back up.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc2SL_SingleLayerResize) {
  SetSvcConfig(2, 1);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.temporal_layering_mode = 0;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  cfg_.rc_resize_allowed = 1;
  ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv", 1280,
                                       720, 15, 1, 0, 300);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  cfg_.rc_target_bitrate = 800;
  ResetModel();
  dynamic_drop_layer_ = true;
  single_layer_resize_ = true;
  base_speed_setting_ = speed_setting_;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Expect at least one resize down and at least one resize back up.
  EXPECT_GE(num_resize_down_, 1);
  EXPECT_GE(num_resize_up_, 1);
  // Don't check rate targeting on two top spatial layer since they will be
  // skipped for part of the sequence.
  CheckLayerRateTargeting(number_spatial_layers_ - 2, number_temporal_layers_,
                          0.78, 1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// For 1 pass CBR SVC with 1 spatial and 2 temporal layers with dynamic resize
// and denoiser enabled. The resizer will resize the single layer down and back
// up again, as the bitrate goes back up.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc1SL2TL_DenoiseResize) {
  SetSvcConfig(1, 2);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.temporal_layering_mode = 2;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  cfg_.rc_resize_allowed = 1;
  ::libvpx_test::I420VideoSource video("desktop_office1.1280_720-020.yuv", 1280,
                                       720, 12, 1, 0, 300);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  cfg_.rc_target_bitrate = 800;
  ResetModel();
  dynamic_drop_layer_ = false;
  single_layer_resize_ = true;
  denoiser_on_ = 1;
  base_speed_setting_ = speed_setting_;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  // Expect at least one resize down and at least one resize back up.
  EXPECT_GE(num_resize_down_, 1);
  EXPECT_GE(num_resize_up_, 1);
}

// Run SVC encoder for 1 temporal layer, 2 spatial layers, with spatial
// downscale 5x5.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc2SL1TL5x5MultipleRuns) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;
  cfg_.ss_number_layers = 2;
  cfg_.ts_number_layers = 1;
  cfg_.ts_rate_decimator[0] = 1;
  cfg_.g_error_resilient = 1;
  cfg_.g_threads = 3;
  cfg_.temporal_layering_mode = 0;
  svc_params_.scaling_factor_num[0] = 256;
  svc_params_.scaling_factor_den[0] = 1280;
  svc_params_.scaling_factor_num[1] = 1280;
  svc_params_.scaling_factor_den[1] = 1280;
  cfg_.rc_dropframe_thresh = 10;
  cfg_.kf_max_dist = 999999;
  cfg_.kf_min_dist = 0;
  cfg_.ss_target_bitrate[0] = 300;
  cfg_.ss_target_bitrate[1] = 1400;
  cfg_.layer_target_bitrate[0] = 300;
  cfg_.layer_target_bitrate[1] = 1400;
  cfg_.rc_target_bitrate = 1700;
  number_spatial_layers_ = cfg_.ss_number_layers;
  number_temporal_layers_ = cfg_.ts_number_layers;
  ResetModel();
  layer_target_avg_bandwidth_[0] = cfg_.layer_target_bitrate[0] * 1000 / 30;
  bits_in_buffer_model_[0] =
      cfg_.layer_target_bitrate[0] * cfg_.rc_buf_initial_sz;
  layer_target_avg_bandwidth_[1] = cfg_.layer_target_bitrate[1] * 1000 / 30;
  bits_in_buffer_model_[1] =
      cfg_.layer_target_bitrate[1] * cfg_.rc_buf_initial_sz;
  ::libvpx_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 0, 60);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// For 1 pass CBR SVC with 3 spatial and 3 temporal layers with external resize
// and denoiser enabled. The external resizer will resize down and back up,
// setting 0/nonzero bitrate on spatial enhancement layers to disable/enable
// layers. Resizing starts on first frame and the pattern is:
//  1/4 -> 1/2 -> 1 -> 1/4 -> 1/2.
TEST_P(DatarateOnePassCbrSvcSingleBR,
       OnePassCbrSvc3SL3TL_DenoiseExternalResizePattern1) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 40;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.temporal_layering_mode = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.kf_max_dist = 10000;
  cfg_.kf_min_dist = 10000;
  cfg_.rc_resize_allowed = 0;
  cfg_.g_w = 1280;
  cfg_.g_h = 720;
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  ResizingVideoSource video(1280, 720);
  video.external_resize_pattern_ = 1;
  video.force_zero_source_ = 0;
  cfg_.rc_target_bitrate = 1000;
  ResetModel();
  dynamic_drop_layer_ = false;
  single_layer_resize_ = false;
  denoiser_on_ = 1;
  base_speed_setting_ = speed_setting_;
  external_resize_dynamic_drop_layer_ = true;
  external_resize_pattern_ = video.external_resize_pattern_;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

// For 1 pass CBR SVC with 3 spatial and 3 temporal layers with external resize
// and denoiser enabled. The external resizer will resize down and back up,
// setting 0/nonzero bitrate on spatial enhancement layers to disable/enable
// layers. Resizing starts on first frame and the pattern is:
//  1/2 -> 1/4 -> 1 -> 1/2 -> 1/4.
TEST_P(DatarateOnePassCbrSvcSingleBR,
       OnePassCbrSvc3SL3TL_DenoiseExternalResizePattern2) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 40;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.temporal_layering_mode = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.kf_max_dist = 10000;
  cfg_.kf_min_dist = 10000;
  cfg_.rc_resize_allowed = 0;
  cfg_.g_w = 1280;
  cfg_.g_h = 720;
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  ResizingVideoSource video(1280, 720);
  video.external_resize_pattern_ = 2;
  video.force_zero_source_ = 0;
  cfg_.rc_target_bitrate = 1000;
  ResetModel();
  dynamic_drop_layer_ = false;
  single_layer_resize_ = false;
  denoiser_on_ = 1;
  base_speed_setting_ = speed_setting_;
  external_resize_dynamic_drop_layer_ = true;
  external_resize_pattern_ = video.external_resize_pattern_;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

// For 1 pass CBR SVC with 3 spatial and 3 temporal layers with external resize
// and denoiser enabled. The external resizer will resize down and back up,
// setting 0/nonzero bitrate on spatial enhancement layers to disable/enable
// layers. Resizing starts on first frame and the pattern is:
//  1/2 -> 1/4 -> 1 -> 1/2 -> 1/4. This test uses 4 threads with small keyframe
// spacing, and top resolution is 1280x960.
TEST_P(DatarateOnePassCbrSvcSingleBR,
       OnePassCbrSvc3SL3TL_DenoiseExternalResizePattern2Key4Threads) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 40;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 4;
  cfg_.temporal_layering_mode = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.kf_max_dist = 40;
  cfg_.kf_min_dist = 40;
  cfg_.rc_resize_allowed = 0;
  cfg_.g_w = 1280;
  cfg_.g_h = 960;
  top_sl_width_ = cfg_.g_w;
  top_sl_height_ = cfg_.g_h;
  ResizingVideoSource video(1280, 960);
  video.external_resize_pattern_ = 2;
  video.force_zero_source_ = 0;
  cfg_.rc_target_bitrate = 1000;
  ResetModel();
  dynamic_drop_layer_ = false;
  single_layer_resize_ = false;
  denoiser_on_ = 1;
  base_speed_setting_ = speed_setting_;
  external_resize_dynamic_drop_layer_ = true;
  external_resize_pattern_ = video.external_resize_pattern_;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

// For 1 pass CBR SVC with 3 spatial and 3 temporal layers with external resize
// and denoiser enabled. The external resizer will resize down and back up,
// setting 0/nonzero bitrate on spatial enhancement layers to disable/enable
// layers. Resizing starts on first frame and the pattern is:
//  1/4 -> 1/2 -> 1 -> 1/4 -> 1/2. The source will be set to 0 every x frames,
// otherwise random values, to trigger scene detection in the encoder.
TEST_P(DatarateOnePassCbrSvcSingleBR,
       OnePassCbrSvc3SL3TL_DenoiseExternalResizePattern1SceneChange) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 40;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.temporal_layering_mode = 3;
  cfg_.ts_rate_decimator[0] = 4;
  cfg_.ts_rate_decimator[1] = 2;
  cfg_.ts_rate_decimator[2] = 1;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.kf_max_dist = 10000;
  cfg_.kf_min_dist = 10000;
  cfg_.rc_resize_allowed = 0;
  cfg_.g_w = 1280;
  cfg_.g_h = 720;
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  ResizingVideoSource video(1280, 720);
  video.external_resize_pattern_ = 1;
  video.force_zero_source_ = 1;
  cfg_.rc_target_bitrate = 1000;
  ResetModel();
  dynamic_drop_layer_ = false;
  single_layer_resize_ = false;
  denoiser_on_ = 1;
  base_speed_setting_ = speed_setting_;
  external_resize_dynamic_drop_layer_ = true;
  external_resize_pattern_ = video.external_resize_pattern_;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

// Params: speed setting and index for bitrate array.
class DatarateOnePassCbrSvcMultiBR
    : public DatarateOnePassCbrSvc,
      public ::libvpx_test::CodecTestWith2Params<int, int> {
 public:
  DatarateOnePassCbrSvcMultiBR() : DatarateOnePassCbrSvc(GET_PARAM(0)) {
    memset(&svc_params_, 0, sizeof(svc_params_));
  }
  ~DatarateOnePassCbrSvcMultiBR() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    speed_setting_ = GET_PARAM(1);
    ResetModel();
  }
};

// Check basic rate targeting for 1 pass CBR SVC: 2 spatial layers and
// 3 temporal layers. Run CIF clip with 1 thread.
TEST_P(DatarateOnePassCbrSvcMultiBR, OnePassCbrSvc2SL3TL) {
  SetSvcConfig(2, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  const int bitrates[3] = { 200, 400, 600 };
  // TODO(marpan): Check that effective_datarate for each layer hits the
  // layer target_bitrate.
  cfg_.rc_target_bitrate = bitrates[GET_PARAM(2)];
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.75,
                          1.2);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass VBR SVC: 2 spatial layers and
// 3 temporal layers. Run VGA clip with 1 thread.
TEST_P(DatarateOnePassCbrSvcMultiBR, OnePassVbrSvc2SL3TL) {
  SetSvcConfig(2, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 2;
  cfg_.rc_max_quantizer = 56;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  cfg_.rc_end_usage = VPX_VBR;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  const int bitrates[3] = { 200, 400, 600 };
  cfg_.rc_target_bitrate = bitrates[GET_PARAM(2)];
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.70,
                          1.3);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Params: speed setting, layer framedrop control and index for bitrate array.
class DatarateOnePassCbrSvcFrameDropMultiBR
    : public DatarateOnePassCbrSvc,
      public ::libvpx_test::CodecTestWith3Params<int, int, int> {
 public:
  DatarateOnePassCbrSvcFrameDropMultiBR()
      : DatarateOnePassCbrSvc(GET_PARAM(0)) {
    memset(&svc_params_, 0, sizeof(svc_params_));
  }
  ~DatarateOnePassCbrSvcFrameDropMultiBR() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    speed_setting_ = GET_PARAM(1);
    ResetModel();
  }
};

// Check basic rate targeting for 1 pass CBR SVC: 2 spatial layers and
// 3 temporal layers. Run HD clip with 4 threads.
TEST_P(DatarateOnePassCbrSvcFrameDropMultiBR, OnePassCbrSvc2SL3TL4Threads) {
  SetSvcConfig(2, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 4;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 0, 60);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  layer_framedrop_ = 0;
  const int bitrates[3] = { 200, 400, 600 };
  cfg_.rc_target_bitrate = bitrates[GET_PARAM(3)];
  ResetModel();
  layer_framedrop_ = GET_PARAM(2);
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.64,
                          1.45);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and
// 3 temporal layers. Run HD clip with 4 threads.
TEST_P(DatarateOnePassCbrSvcFrameDropMultiBR, OnePassCbrSvc3SL3TL4Threads) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 4;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 0, 60);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  layer_framedrop_ = 0;
  const int bitrates[3] = { 200, 400, 600 };
  cfg_.rc_target_bitrate = bitrates[GET_PARAM(3)];
  ResetModel();
  layer_framedrop_ = GET_PARAM(2);
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.58,
                          1.2);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and
// 3 temporal layers. Run HD clip with 4 threads, for 1284x770, which
// likely is the issue for Bug: 366146260.
TEST_P(DatarateOnePassCbrSvcFrameDropMultiBR,
       OnePassCbrSvc3SL3TL4Threads1284x770) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 4;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::Y4mVideoSource video("niklas_1284_770_30.y4m", 0, 60);
  top_sl_width_ = 1284;
  top_sl_height_ = 770;
  layer_framedrop_ = 0;
  const int bitrates[3] = { 200, 400, 600 };
  cfg_.rc_target_bitrate = bitrates[GET_PARAM(3)];
  ResetModel();
  layer_framedrop_ = GET_PARAM(2);
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.58,
                          1.2);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and
// 3 temporal layers. Run HD clip with 4 threads, for 1857x167.
TEST_P(DatarateOnePassCbrSvcFrameDropMultiBR,
       OnePassCbrSvc3SL3TL4Threads1857x167) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::Y4mVideoSource video("niklas_1857_167_30.y4m", 0, 60);
  top_sl_width_ = 1857;
  top_sl_height_ = 167;
  layer_framedrop_ = 0;
  const int bitrates[3] = { 200, 400, 600 };
  cfg_.rc_target_bitrate = bitrates[GET_PARAM(3)];
  ResetModel();
  layer_framedrop_ = GET_PARAM(2);
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.58,
                          1.2);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and
// 2 temporal layers, for KSVC in flexible mode with no update of reference
// frames for all spatial layers on TL > 0 superframes.
// Run HD clip with 4 threads.
TEST_P(DatarateOnePassCbrSvcFrameDropMultiBR, OnePassCbrSvc3SL2TL4ThKSVCFlex) {
  SetSvcConfig(3, 2);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 4;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 0, 60);
  top_sl_width_ = 1280;
  top_sl_height_ = 720;
  layer_framedrop_ = 0;
  const int bitrates[3] = { 200, 400, 600 };
  cfg_.rc_target_bitrate = bitrates[GET_PARAM(3)];
  ResetModel();
  layer_framedrop_ = GET_PARAM(2);
  AssignLayerBitrates();
  ksvc_flex_noupd_tlenh_ = true;
  cfg_.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.58,
                          1.2);
}

// Params: speed setting, inter-layer prediction mode.
class DatarateOnePassCbrSvcInterLayerPredSingleBR
    : public DatarateOnePassCbrSvc,
      public ::libvpx_test::CodecTestWith2Params<int, int> {
 public:
  DatarateOnePassCbrSvcInterLayerPredSingleBR()
      : DatarateOnePassCbrSvc(GET_PARAM(0)) {
    memset(&svc_params_, 0, sizeof(svc_params_));
  }
  ~DatarateOnePassCbrSvcInterLayerPredSingleBR() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    speed_setting_ = GET_PARAM(1);
    inter_layer_pred_mode_ = GET_PARAM(2);
    ResetModel();
  }
};

// Check basic rate targeting with different inter-layer prediction modes for 1
// pass CBR SVC: 3 spatial layers and 3 temporal layers. Run CIF clip with 1
// thread.
TEST_P(DatarateOnePassCbrSvcInterLayerPredSingleBR, OnePassCbrSvc3SL3TL) {
  // Disable test for inter-layer pred off for now since simulcast_mode fails.
  if (inter_layer_pred_mode_ == INTER_LAYER_PRED_OFF) return;
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.temporal_layering_mode = 3;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  cfg_.rc_target_bitrate = 800;
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check rate targeting with different inter-layer prediction modes for 1 pass
// CBR SVC: 3 spatial layers and 3 temporal layers, changing the target bitrate
// at the middle of encoding.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL3TLDynamicBitrateChange) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  cfg_.rc_target_bitrate = 800;
  ResetModel();
  change_bitrate_ = true;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

#if CONFIG_VP9_TEMPORAL_DENOISING
// Params: speed setting, noise sensitivity, index for bitrate array and inter
// layer pred mode.
class DatarateOnePassCbrSvcDenoiser
    : public DatarateOnePassCbrSvc,
      public ::libvpx_test::CodecTestWith4Params<int, int, int, int> {
 public:
  DatarateOnePassCbrSvcDenoiser() : DatarateOnePassCbrSvc(GET_PARAM(0)) {
    memset(&svc_params_, 0, sizeof(svc_params_));
  }
  ~DatarateOnePassCbrSvcDenoiser() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    speed_setting_ = GET_PARAM(1);
    inter_layer_pred_mode_ = GET_PARAM(3);
    ResetModel();
  }
};

// Check basic rate targeting for 1 pass CBR SVC with denoising.
// 2 spatial layers and 3 temporal layer. Run HD clip with 2 threads.
TEST_P(DatarateOnePassCbrSvcDenoiser, OnePassCbrSvc2SL3TLDenoiserOn) {
  SetSvcConfig(2, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 2;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  number_spatial_layers_ = cfg_.ss_number_layers;
  number_temporal_layers_ = cfg_.ts_number_layers;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  const int bitrates[3] = { 600, 800, 1000 };
  // TODO(marpan): Check that effective_datarate for each layer hits the
  // layer target_bitrate.
  // For SVC, noise_sen = 1 means denoising only the top spatial layer
  // noise_sen = 2 means denoising the two top spatial layers.
  cfg_.rc_target_bitrate = bitrates[GET_PARAM(3)];
  ResetModel();
  denoiser_on_ = GET_PARAM(2);
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}
#endif

// Params: speed setting, key frame dist.
class DatarateOnePassCbrSvcSmallKF
    : public DatarateOnePassCbrSvc,
      public ::libvpx_test::CodecTestWith2Params<int, int> {
 public:
  DatarateOnePassCbrSvcSmallKF() : DatarateOnePassCbrSvc(GET_PARAM(0)) {
    memset(&svc_params_, 0, sizeof(svc_params_));
  }
  ~DatarateOnePassCbrSvcSmallKF() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    speed_setting_ = GET_PARAM(1);
    ResetModel();
  }
};

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and 3
// temporal layers. Run CIF clip with 1 thread, and few short key frame periods.
TEST_P(DatarateOnePassCbrSvcSmallKF, OnePassCbrSvc3SL3TLSmallKf) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 10;
  cfg_.rc_target_bitrate = 800;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  // For this 3 temporal layer case, pattern repeats every 4 frames, so choose
  // 4 key neighboring key frame periods (so key frame will land on 0-2-1-2).
  const int kf_dist = GET_PARAM(2);
  cfg_.kf_max_dist = kf_dist;
  key_frame_spacing_ = kf_dist;
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.70,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 2 spatial layers and 3
// temporal layers. Run CIF clip with 1 thread, and few short key frame periods.
TEST_P(DatarateOnePassCbrSvcSmallKF, OnePassCbrSvc2SL3TLSmallKf) {
  SetSvcConfig(2, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.rc_dropframe_thresh = 10;
  cfg_.rc_target_bitrate = 400;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  // For this 3 temporal layer case, pattern repeats every 4 frames, so choose
  // 4 key neighboring key frame periods (so key frame will land on 0-2-1-2).
  const int kf_dist = GET_PARAM(2) + 32;
  cfg_.kf_max_dist = kf_dist;
  key_frame_spacing_ = kf_dist;
  ResetModel();
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Check basic rate targeting for 1 pass CBR SVC: 3 spatial layers and 3
// temporal layers. Run VGA clip with 1 thread, and place layer sync frames:
// one at middle layer first, then another one for top layer, and another
// insert for base spatial layer (which forces key frame).
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL3TLSyncFrames) {
  SetSvcConfig(3, 3);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 1;
  cfg_.kf_max_dist = 9999;
  cfg_.rc_dropframe_thresh = 10;
  cfg_.rc_target_bitrate = 400;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  ResetModel();
  insert_layer_sync_ = 1;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.78,
                          1.15);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Run SVC encoder for 3 spatial layers, 1 temporal layer, with
// intra-only frame as sync frame on base spatial layer.
// Intra_only is inserted at start and in middle of sequence.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc3SL1TLSyncWithIntraOnly) {
  SetSvcConfig(3, 1);
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 63;
  cfg_.g_threads = 4;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  cfg_.rc_target_bitrate = 400;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  ResetModel();
  insert_layer_sync_ = 1;
  // Use intra_only frame for sync on base layer.
  force_intra_only_frame_ = 1;
  AssignLayerBitrates();
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.73,
                          1.2);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Run SVC encoder for 2 quality layers (same resolution different,
// bitrates), 1 temporal layer, with screen content mode.
TEST_P(DatarateOnePassCbrSvcSingleBR, OnePassCbrSvc2QL1TLScreen) {
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;
  cfg_.ss_number_layers = 2;
  cfg_.ts_number_layers = 1;
  cfg_.ts_rate_decimator[0] = 1;
  cfg_.temporal_layering_mode = 0;
  cfg_.g_error_resilient = 1;
  cfg_.g_threads = 2;
  svc_params_.scaling_factor_num[0] = 1;
  svc_params_.scaling_factor_den[0] = 1;
  svc_params_.scaling_factor_num[1] = 1;
  svc_params_.scaling_factor_den[1] = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  number_spatial_layers_ = cfg_.ss_number_layers;
  number_temporal_layers_ = cfg_.ts_number_layers;
  ::libvpx_test::I420VideoSource video("niklas_640_480_30.yuv", 640, 480, 30, 1,
                                       0, 400);
  top_sl_width_ = 640;
  top_sl_height_ = 480;
  ResetModel();
  tune_content_ = 1;
  // Set the layer bitrates, for 2 spatial layers, 1 temporal.
  cfg_.rc_target_bitrate = 400;
  cfg_.ss_target_bitrate[0] = 100;
  cfg_.ss_target_bitrate[1] = 300;
  cfg_.layer_target_bitrate[0] = 100;
  cfg_.layer_target_bitrate[1] = 300;
  for (int sl = 0; sl < 2; ++sl) {
    float layer_framerate = 30.0;
    layer_target_avg_bandwidth_[sl] = static_cast<int>(
        cfg_.layer_target_bitrate[sl] * 1000.0 / layer_framerate);
    bits_in_buffer_model_[sl] =
        cfg_.layer_target_bitrate[sl] * cfg_.rc_buf_initial_sz;
  }
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.73,
                          1.25);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

// Params: speed setting.
class DatarateOnePassCbrSvcPostencodeDrop
    : public DatarateOnePassCbrSvc,
      public ::libvpx_test::CodecTestWithParam<int> {
 public:
  DatarateOnePassCbrSvcPostencodeDrop() : DatarateOnePassCbrSvc(GET_PARAM(0)) {
    memset(&svc_params_, 0, sizeof(svc_params_));
  }
  ~DatarateOnePassCbrSvcPostencodeDrop() override = default;

 protected:
  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    speed_setting_ = GET_PARAM(1);
    ResetModel();
  }
};

// Run SVC encoder for 2 quality layers (same resolution different,
// bitrates), 1 temporal layer, with screen content mode.
TEST_P(DatarateOnePassCbrSvcPostencodeDrop, OnePassCbrSvc2QL1TLScreen) {
  cfg_.rc_buf_initial_sz = 200;
  cfg_.rc_buf_optimal_sz = 200;
  cfg_.rc_buf_sz = 400;
  cfg_.rc_min_quantizer = 0;
  cfg_.rc_max_quantizer = 52;
  cfg_.rc_end_usage = VPX_CBR;
  cfg_.g_lag_in_frames = 0;
  cfg_.ss_number_layers = 2;
  cfg_.ts_number_layers = 1;
  cfg_.ts_rate_decimator[0] = 1;
  cfg_.temporal_layering_mode = 0;
  cfg_.g_error_resilient = 1;
  cfg_.g_threads = 2;
  svc_params_.scaling_factor_num[0] = 1;
  svc_params_.scaling_factor_den[0] = 1;
  svc_params_.scaling_factor_num[1] = 1;
  svc_params_.scaling_factor_den[1] = 1;
  cfg_.rc_dropframe_thresh = 30;
  cfg_.kf_max_dist = 9999;
  number_spatial_layers_ = cfg_.ss_number_layers;
  number_temporal_layers_ = cfg_.ts_number_layers;
  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 300);
  top_sl_width_ = 352;
  top_sl_height_ = 288;
  ResetModel();
  base_speed_setting_ = speed_setting_;
  tune_content_ = 1;
  use_post_encode_drop_ = 1;
  // Set the layer bitrates, for 2 spatial layers, 1 temporal.
  cfg_.rc_target_bitrate = 400;
  cfg_.ss_target_bitrate[0] = 100;
  cfg_.ss_target_bitrate[1] = 300;
  cfg_.layer_target_bitrate[0] = 100;
  cfg_.layer_target_bitrate[1] = 300;
  for (int sl = 0; sl < 2; ++sl) {
    float layer_framerate = 30.0;
    layer_target_avg_bandwidth_[sl] = static_cast<int>(
        cfg_.layer_target_bitrate[sl] * 1000.0 / layer_framerate);
    bits_in_buffer_model_[sl] =
        cfg_.layer_target_bitrate[sl] * cfg_.rc_buf_initial_sz;
  }
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  CheckLayerRateTargeting(number_spatial_layers_, number_temporal_layers_, 0.73,
                          1.25);
#if CONFIG_VP9_DECODER
  // The non-reference frames are expected to be mismatched frames as the
  // encoder will avoid loopfilter on these frames.
  EXPECT_EQ(GetNonRefFrames(), GetMismatchFrames());
#endif
}

VP9_INSTANTIATE_TEST_SUITE(DatarateOnePassCbrSvcSingleBR,
                           ::testing::Range(5, 10));

VP9_INSTANTIATE_TEST_SUITE(DatarateOnePassCbrSvcPostencodeDrop,
                           ::testing::Range(5, 6));

VP9_INSTANTIATE_TEST_SUITE(DatarateOnePassCbrSvcInterLayerPredSingleBR,
                           ::testing::Range(5, 10), ::testing::Range(0, 3));

VP9_INSTANTIATE_TEST_SUITE(DatarateOnePassCbrSvcMultiBR,
                           ::testing::Range(5, 10), ::testing::Range(0, 3));

VP9_INSTANTIATE_TEST_SUITE(DatarateOnePassCbrSvcFrameDropMultiBR,
                           ::testing::Range(5, 10), ::testing::Range(0, 2),
                           ::testing::Range(0, 3));

#if CONFIG_VP9_TEMPORAL_DENOISING
VP9_INSTANTIATE_TEST_SUITE(DatarateOnePassCbrSvcDenoiser,
                           ::testing::Range(5, 10), ::testing::Range(1, 3),
                           ::testing::Range(0, 3), ::testing::Range(0, 4));
#endif

VP9_INSTANTIATE_TEST_SUITE(DatarateOnePassCbrSvcSmallKF,
                           ::testing::Range(5, 10), ::testing::Range(32, 36));
}  // namespace
}  // namespace svc_test
