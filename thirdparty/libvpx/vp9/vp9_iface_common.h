/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VP9_VP9_IFACE_COMMON_H_
#define VPX_VP9_VP9_IFACE_COMMON_H_

#include <assert.h>
#include "vpx_ports/mem.h"
#include "vpx/vp8.h"
#include "vpx_scale/yv12config.h"
#include "common/vp9_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

void yuvconfig2image(vpx_image_t *img, const YV12_BUFFER_CONFIG *yv12,
                     void *user_priv);

vpx_codec_err_t image2yuvconfig(const vpx_image_t *img,
                                YV12_BUFFER_CONFIG *yv12);

static INLINE VP9_REFFRAME
ref_frame_to_vp9_reframe(vpx_ref_frame_type_t frame) {
  switch (frame) {
    case VP8_LAST_FRAME: return VP9_LAST_FLAG;
    case VP8_GOLD_FRAME: return VP9_GOLD_FLAG;
    case VP8_ALTR_FRAME: return VP9_ALT_FLAG;
  }
  assert(0 && "Invalid Reference Frame");
  return VP9_LAST_FLAG;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_VP9_IFACE_COMMON_H_
