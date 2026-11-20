/*
 *  Copyright (c) 2021 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_RATECTRL_RTC_H_
#define VPX_VP8_RATECTRL_RTC_H_

#include <cstdint>
#include <cstring>
#include <memory>

#include "vpx/internal/vpx_ratectrl_rtc.h"

struct VP8_COMP;

namespace libvpx {
struct VP8RateControlRtcConfig : public VpxRateControlRtcConfig {
  VP8RateControlRtcConfig() {
    memset(&layer_target_bitrate, 0, sizeof(layer_target_bitrate));
    memset(&ts_rate_decimator, 0, sizeof(ts_rate_decimator));
  }
};

struct VP8FrameParamsQpRTC {
  RcFrameType frame_type;
  int temporal_layer_id;
};

class VP8RateControlRTC {
 public:
  static std::unique_ptr<VP8RateControlRTC> Create(
      const VP8RateControlRtcConfig &cfg);
  ~VP8RateControlRTC();

  bool UpdateRateControl(const VP8RateControlRtcConfig &rc_cfg);
  // GetQP() needs to be called after ComputeQP() to get the latest QP
  int GetQP() const;
  // GetUVDeltaQP() needs to be called after ComputeQP() to get the latest
  // delta QP for UV.
  UVDeltaQP GetUVDeltaQP() const;
  // GetLoopfilterLevel() needs to be called after ComputeQP() since loopfilter
  // level is calculated from frame qp.
  int GetLoopfilterLevel() const;
  // ComputeQP computes the QP if the frame is not dropped (kOk return),
  // otherwise it returns kDrop and subsequent GetQP and PostEncodeUpdate
  // are not to be called.
  FrameDropDecision ComputeQP(const VP8FrameParamsQpRTC &frame_params);
  // Feedback to rate control with the size of current encoded frame
  void PostEncodeUpdate(uint64_t encoded_frame_size);

 private:
  VP8RateControlRTC() = default;
  bool InitRateControl(const VP8RateControlRtcConfig &cfg);
  struct VP8_COMP *cpi_ = nullptr;
  int q_ = -1;
};

}  // namespace libvpx

#endif  // VPX_VP8_RATECTRL_RTC_H_
