/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_VP9_CX_IFACE_H_
#define VPX_VP9_VP9_CX_IFACE_H_
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/common/vp9_onyxc_int.h"

#ifdef __cplusplus
extern "C" {
#endif

VP9EncoderConfig vp9_get_encoder_config(int frame_width, int frame_height,
                                        vpx_rational_t frame_rate,
                                        int target_bitrate, int encode_speed,
                                        int target_level,
                                        vpx_enc_pass enc_pass);

void vp9_dump_encoder_config(const VP9EncoderConfig *oxcf, FILE *fp);

FRAME_INFO vp9_get_frame_info(const VP9EncoderConfig *oxcf);

static INLINE int64_t
timebase_units_to_ticks(const vpx_rational64_t *timestamp_ratio, int64_t n) {
  return n * timestamp_ratio->num / timestamp_ratio->den;
}

static INLINE int64_t
ticks_to_timebase_units(const vpx_rational64_t *timestamp_ratio, int64_t n) {
  int64_t round = timestamp_ratio->num / 2;
  if (round > 0) --round;
  return (n * timestamp_ratio->den + round) / timestamp_ratio->num;
}

void vp9_set_first_pass_stats(VP9EncoderConfig *oxcf,
                              const vpx_fixed_buf_t *stats);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_VP9_CX_IFACE_H_
