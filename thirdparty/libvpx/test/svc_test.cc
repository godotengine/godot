/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "test/svc_test.h"

namespace svc_test {
void OnePassCbrSvc::SetSvcConfig(const int num_spatial_layer,
                                 const int num_temporal_layer) {
  SetConfig(num_temporal_layer);
  cfg_.ss_number_layers = num_spatial_layer;
  cfg_.ts_number_layers = num_temporal_layer;
  if (num_spatial_layer == 1) {
    svc_params_.scaling_factor_num[0] = 288;
    svc_params_.scaling_factor_den[0] = 288;
  } else if (num_spatial_layer == 2) {
    svc_params_.scaling_factor_num[0] = 144;
    svc_params_.scaling_factor_den[0] = 288;
    svc_params_.scaling_factor_num[1] = 288;
    svc_params_.scaling_factor_den[1] = 288;
  } else if (num_spatial_layer == 3) {
    svc_params_.scaling_factor_num[0] = 72;
    svc_params_.scaling_factor_den[0] = 288;
    svc_params_.scaling_factor_num[1] = 144;
    svc_params_.scaling_factor_den[1] = 288;
    svc_params_.scaling_factor_num[2] = 288;
    svc_params_.scaling_factor_den[2] = 288;
  }
  number_spatial_layers_ = cfg_.ss_number_layers;
  number_temporal_layers_ = cfg_.ts_number_layers;
}

void OnePassCbrSvc::PreEncodeFrameHookSetup(::libvpx_test::VideoSource *video,
                                            ::libvpx_test::Encoder *encoder) {
  if (video->frame() == 0) {
    for (int i = 0; i < VPX_MAX_LAYERS; ++i) {
      svc_params_.max_quantizers[i] = 63;
      svc_params_.min_quantizers[i] = 0;
    }
    if (number_temporal_layers_ > 1 || number_spatial_layers_ > 1) {
      svc_params_.speed_per_layer[0] = base_speed_setting_;
      for (int i = 1; i < VPX_SS_MAX_LAYERS; ++i) {
        svc_params_.speed_per_layer[i] = speed_setting_;
      }
      encoder->Control(VP9E_SET_SVC, 1);
      encoder->Control(VP9E_SET_SVC_PARAMETERS, &svc_params_);
    }
    encoder->Control(VP8E_SET_CPUUSED, speed_setting_);
    encoder->Control(VP9E_SET_AQ_MODE, 3);
    encoder->Control(VP8E_SET_MAX_INTRA_BITRATE_PCT, 300);
    encoder->Control(VP9E_SET_TILE_COLUMNS, get_msb(cfg_.g_threads));
    encoder->Control(VP9E_SET_ROW_MT, 1);
    encoder->Control(VP8E_SET_STATIC_THRESHOLD, 1);
  }

  superframe_count_++;
  temporal_layer_id_ = 0;
  if (number_temporal_layers_ == 2) {
    temporal_layer_id_ = (superframe_count_ % 2 != 0);
  } else if (number_temporal_layers_ == 3) {
    if (superframe_count_ % 2 != 0) temporal_layer_id_ = 2;
    if (superframe_count_ > 1) {
      if ((superframe_count_ - 2) % 4 == 0) temporal_layer_id_ = 1;
    }
  }

  frame_flags_ = 0;
}

void OnePassCbrSvc::PostEncodeFrameHook(::libvpx_test::Encoder *encoder) {
  vpx_svc_layer_id_t layer_id;
  encoder->Control(VP9E_GET_SVC_LAYER_ID, &layer_id);
  temporal_layer_id_ = layer_id.temporal_layer_id;
  for (int sl = 0; sl < number_spatial_layers_; ++sl) {
    for (int tl = temporal_layer_id_; tl < number_temporal_layers_; ++tl) {
      const int layer = sl * number_temporal_layers_ + tl;
      bits_in_buffer_model_[layer] +=
          static_cast<int64_t>(layer_target_avg_bandwidth_[layer]);
    }
  }
}

void OnePassCbrSvc::AssignLayerBitrates() {
  int sl, spatial_layer_target;
  int spatial_layers = cfg_.ss_number_layers;
  int temporal_layers = cfg_.ts_number_layers;
  float total = 0;
  float alloc_ratio[VPX_MAX_LAYERS] = { 0 };
  float framerate = 30.0;
  for (sl = 0; sl < spatial_layers; ++sl) {
    if (svc_params_.scaling_factor_den[sl] > 0) {
      alloc_ratio[sl] =
          static_cast<float>((svc_params_.scaling_factor_num[sl] * 1.0 /
                              svc_params_.scaling_factor_den[sl]));
      total += alloc_ratio[sl];
    }
  }
  for (sl = 0; sl < spatial_layers; ++sl) {
    cfg_.ss_target_bitrate[sl] = spatial_layer_target =
        static_cast<unsigned int>(cfg_.rc_target_bitrate * alloc_ratio[sl] /
                                  total);
    const int index = sl * temporal_layers;
    if (cfg_.temporal_layering_mode == 3) {
      cfg_.layer_target_bitrate[index] = spatial_layer_target >> 1;
      cfg_.layer_target_bitrate[index + 1] =
          (spatial_layer_target >> 1) + (spatial_layer_target >> 2);
      cfg_.layer_target_bitrate[index + 2] = spatial_layer_target;
    } else if (cfg_.temporal_layering_mode == 2) {
      cfg_.layer_target_bitrate[index] = spatial_layer_target * 2 / 3;
      cfg_.layer_target_bitrate[index + 1] = spatial_layer_target;
    } else if (cfg_.temporal_layering_mode <= 1) {
      cfg_.layer_target_bitrate[index] = spatial_layer_target;
    }
  }
  for (sl = 0; sl < spatial_layers; ++sl) {
    for (int tl = 0; tl < temporal_layers; ++tl) {
      const int layer = sl * temporal_layers + tl;
      float layer_framerate = framerate;
      if (temporal_layers == 2 && tl == 0) layer_framerate = framerate / 2;
      if (temporal_layers == 3 && tl == 0) layer_framerate = framerate / 4;
      if (temporal_layers == 3 && tl == 1) layer_framerate = framerate / 2;
      layer_target_avg_bandwidth_[layer] = static_cast<int>(
          cfg_.layer_target_bitrate[layer] * 1000.0 / layer_framerate);
      bits_in_buffer_model_[layer] =
          cfg_.layer_target_bitrate[layer] * cfg_.rc_buf_initial_sz;
    }
  }
}
}  // namespace svc_test
