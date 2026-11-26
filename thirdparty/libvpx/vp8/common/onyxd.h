/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_ONYXD_H_
#define VPX_VP8_COMMON_ONYXD_H_

/* Create/destroy static data structures. */
#ifdef __cplusplus
extern "C" {
#endif
#include "vpx_scale/yv12config.h"
#include "ppflags.h"
#include "vpx_ports/mem.h"
#include "vpx/vpx_codec.h"
#include "vpx/vp8.h"

struct VP8D_COMP;
struct VP8Common;

typedef struct {
  int Width;
  int Height;
  int Version;
  int postprocess;
  int max_threads;
  int error_concealment;
} VP8D_CONFIG;

typedef enum { VP8D_OK = 0 } VP8D_SETTING;

void vp8dx_initialize(void);

void vp8dx_set_setting(struct VP8D_COMP *comp, VP8D_SETTING oxst, int x);

int vp8dx_get_setting(struct VP8D_COMP *comp, VP8D_SETTING oxst);

int vp8dx_receive_compressed_data(struct VP8D_COMP *pbi);
int vp8dx_get_raw_frame(struct VP8D_COMP *pbi, YV12_BUFFER_CONFIG *sd,
                        vp8_ppflags_t *flags);
int vp8dx_references_buffer(struct VP8Common *oci, int ref_frame);

vpx_codec_err_t vp8dx_get_reference(struct VP8D_COMP *pbi,
                                    enum vpx_ref_frame_type ref_frame_flag,
                                    YV12_BUFFER_CONFIG *sd);
vpx_codec_err_t vp8dx_set_reference(struct VP8D_COMP *pbi,
                                    enum vpx_ref_frame_type ref_frame_flag,
                                    YV12_BUFFER_CONFIG *sd);
int vp8dx_get_quantizer(const struct VP8D_COMP *pbi);

#ifdef __cplusplus
}
#endif

#endif  // VPX_VP8_COMMON_ONYXD_H_
