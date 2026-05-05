/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_PICKLPF_H_
#define VPX_VP8_ENCODER_PICKLPF_H_

#ifdef __cplusplus
extern "C" {
#endif

struct VP8_COMP;
struct yv12_buffer_config;

void vp8cx_pick_filter_level_fast(struct yv12_buffer_config *sd,
                                  struct VP8_COMP *cpi);
void vp8cx_set_alt_lf_level(struct VP8_COMP *cpi, int filt_val);
void vp8cx_pick_filter_level(struct yv12_buffer_config *sd, VP8_COMP *cpi);

#ifdef __cplusplus
}
#endif

#endif  // VPX_VP8_ENCODER_PICKLPF_H_
