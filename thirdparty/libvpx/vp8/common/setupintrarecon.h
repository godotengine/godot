/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_SETUPINTRARECON_H_
#define VPX_VP8_COMMON_SETUPINTRARECON_H_

#include "./vpx_config.h"
#include "vpx_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif
extern void vp8_setup_intra_recon(YV12_BUFFER_CONFIG *ybf);
extern void vp8_setup_intra_recon_top_line(YV12_BUFFER_CONFIG *ybf);

static INLINE void setup_intra_recon_left(unsigned char *y_buffer,
                                          unsigned char *u_buffer,
                                          unsigned char *v_buffer, int y_stride,
                                          int uv_stride) {
  int i;

  for (i = 0; i < 16; ++i) y_buffer[y_stride * i] = (unsigned char)129;

  for (i = 0; i < 8; ++i) u_buffer[uv_stride * i] = (unsigned char)129;

  for (i = 0; i < 8; ++i) v_buffer[uv_stride * i] = (unsigned char)129;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_SETUPINTRARECON_H_
