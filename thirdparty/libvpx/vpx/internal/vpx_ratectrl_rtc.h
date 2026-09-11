/*
 *  Copyright (c) 2021 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_INTERNAL_VPX_RATECTRL_RTC_H_
#define VPX_VPX_INTERNAL_VPX_RATECTRL_RTC_H_

#include "vpx/vpx_encoder.h"

namespace libvpx {

enum class RcFrameType { kKeyFrame = 0, kInterFrame = 1 };

enum class FrameDropDecision {
  kOk,    // Frame is encoded.
  kDrop,  // Frame is dropped.
};

struct UVDeltaQP {
  // For the UV channel: the QP for the dc/ac value is given as
  // GetQP() + uvdc/ac_delta_q, where the uvdc/ac_delta_q are negative numbers.
  int uvdc_delta_q;
  int uvac_delta_q;
};

struct VpxRateControlRtcConfig {
  VpxRateControlRtcConfig() {
    width = 1280;
    height = 720;
    max_quantizer = 63;
    min_quantizer = 2;
    target_bandwidth = 1000;
    buf_initial_sz = 600;
    buf_optimal_sz = 600;
    buf_sz = 1000;
    undershoot_pct = overshoot_pct = 50;
    max_intra_bitrate_pct = 50;
    max_inter_bitrate_pct = 0;
    framerate = 30.0;
    ts_number_layers = 1;
    rc_mode = VPX_CBR;
    aq_mode = 0;
    layer_target_bitrate[0] = static_cast<int>(target_bandwidth);
    ts_rate_decimator[0] = 1;
    frame_drop_thresh = 0;
    is_screen = false;
  }

  int width;
  int height;
  // 0-63
  int max_quantizer;
  int min_quantizer;
  int64_t target_bandwidth;
  int64_t buf_initial_sz;
  int64_t buf_optimal_sz;
  int64_t buf_sz;
  int undershoot_pct;
  int overshoot_pct;
  int max_intra_bitrate_pct;
  int max_inter_bitrate_pct;
  double framerate;
  // Number of temporal layers
  int ts_number_layers;
  int layer_target_bitrate[VPX_MAX_LAYERS];
  int ts_rate_decimator[VPX_TS_MAX_LAYERS];
  // vbr, cbr
  enum vpx_rc_mode rc_mode;
  int aq_mode;
  int frame_drop_thresh;
  bool is_screen;
};
}  // namespace libvpx
#endif  // VPX_VPX_INTERNAL_VPX_RATECTRL_RTC_H_
