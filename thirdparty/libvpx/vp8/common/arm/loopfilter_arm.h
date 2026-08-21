/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_ARM_LOOPFILTER_ARM_H_
#define VPX_VP8_COMMON_ARM_LOOPFILTER_ARM_H_

typedef void loopfilter_y_neon(unsigned char *src, int pitch,
                               unsigned char blimit, unsigned char limit,
                               unsigned char thresh);
typedef void loopfilter_uv_neon(unsigned char *u, int pitch,
                                unsigned char blimit, unsigned char limit,
                                unsigned char thresh, unsigned char *v);

loopfilter_y_neon vp8_loop_filter_horizontal_edge_y_neon;
loopfilter_y_neon vp8_loop_filter_vertical_edge_y_neon;
loopfilter_uv_neon vp8_loop_filter_horizontal_edge_uv_neon;
loopfilter_uv_neon vp8_loop_filter_vertical_edge_uv_neon;

loopfilter_y_neon vp8_mbloop_filter_horizontal_edge_y_neon;
loopfilter_y_neon vp8_mbloop_filter_vertical_edge_y_neon;
loopfilter_uv_neon vp8_mbloop_filter_horizontal_edge_uv_neon;
loopfilter_uv_neon vp8_mbloop_filter_vertical_edge_uv_neon;

#endif  // VPX_VP8_COMMON_ARM_LOOPFILTER_ARM_H_
