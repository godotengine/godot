/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_RATECTRL_RTC_H_
#define VPX_VP9_RATECTRL_RTC_H_

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include "vpx/vpx_encoder.h"
#include "vpx/internal/vpx_ratectrl_rtc.h"

struct VP9_COMP;

namespace libvpx {
struct VP9RateControlRtcConfig : public VpxRateControlRtcConfig {
  VP9RateControlRtcConfig() {
    memset(layer_target_bitrate, 0, sizeof(layer_target_bitrate));
    memset(ts_rate_decimator, 0, sizeof(ts_rate_decimator));
    scaling_factor_num[0] = 1;
    scaling_factor_den[0] = 1;
    max_quantizers[0] = max_quantizer;
    min_quantizers[0] = min_quantizer;
  }

  // Number of spatial layers
  int ss_number_layers = 1;
  int max_quantizers[VPX_MAX_LAYERS] = {};
  int min_quantizers[VPX_MAX_LAYERS] = {};
  int scaling_factor_num[VPX_SS_MAX_LAYERS] = {};
  int scaling_factor_den[VPX_SS_MAX_LAYERS] = {};
  // This is only for SVC for now.
  int max_consec_drop = std::numeric_limits<int>::max();
};

struct VP9FrameParamsQpRTC {
  RcFrameType frame_type;
  int spatial_layer_id;
  int temporal_layer_id;
};

struct VP9SegmentationData {
  const uint8_t *segmentation_map;
  size_t segmentation_map_size;
  const int *delta_q;
  size_t delta_q_size;
};

// This interface allows using VP9 real-time rate control without initializing
// the encoder. To use this interface, you need to link with libvpxrc.a.
//
// #include "vp9/ratectrl_rtc.h"
// VP9RateControlRtcConfig cfg;
// VP9FrameParamsQpRTC frame_params;
//
// YourFunctionToInitializeConfig(cfg);
// std::unique_ptr<VP9RateControlRTC> rc_api = VP9RateControlRTC::Create(cfg);
// // start encoding
// while (frame_to_encode) {
//   if (config_changed)
//     rc_api->UpdateRateControl(cfg);
//   YourFunctionToFillFrameParams(frame_params);
//   rc_api->ComputeQP(frame_params);
//   YourFunctionToUseQP(rc_api->GetQP());
//   YourFunctionToUseLoopfilter(rc_api->GetLoopfilterLevel());
//   // After encoding
//   rc_api->PostEncode(encoded_frame_size, frame_params);
// }
class VP9RateControlRTC {
 public:
  static std::unique_ptr<VP9RateControlRTC> Create(
      const VP9RateControlRtcConfig &cfg);
  ~VP9RateControlRTC();

  bool UpdateRateControl(const VP9RateControlRtcConfig &rc_cfg);
  // GetQP() needs to be called after ComputeQP() to get the latest QP
  int GetQP() const;
  int GetLoopfilterLevel() const;
  bool GetSegmentationData(VP9SegmentationData *segmentation_data) const;
  // ComputeQP computes the QP if the frame is not dropped (kOk return),
  // otherwise it returns kDrop and subsequent GetQP and PostEncodeUpdate
  // are not to be called (vp9_rc_postencode_update_drop_frame is already
  // called via ComputeQP if drop is decided).
  FrameDropDecision ComputeQP(const VP9FrameParamsQpRTC &frame_params);
  // Feedback to rate control with the size of current encoded frame
  void PostEncodeUpdate(uint64_t encoded_frame_size,
                        const VP9FrameParamsQpRTC &frame_params);

 private:
  VP9RateControlRTC() = default;
  bool InitRateControl(const VP9RateControlRtcConfig &cfg);
  struct VP9_COMP *cpi_ = nullptr;
};

}  // namespace libvpx

#endif  // VPX_VP9_RATECTRL_RTC_H_
