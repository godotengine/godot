/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_WEBMENC_H_
#define VPX_WEBMENC_H_

#include <stdio.h>
#include <stdlib.h>

#include "tools_common.h"
#include "vpx/vpx_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

struct WebmOutputContext {
  int debug;
  FILE *stream;
  int64_t last_pts_ns;
  void *writer;
  void *segment;
};

/* Stereo 3D packed frame format */
typedef enum stereo_format {
  STEREO_FORMAT_MONO = 0,
  STEREO_FORMAT_LEFT_RIGHT = 1,
  STEREO_FORMAT_BOTTOM_TOP = 2,
  STEREO_FORMAT_TOP_BOTTOM = 3,
  STEREO_FORMAT_RIGHT_LEFT = 11
} stereo_format_t;

void write_webm_file_header(struct WebmOutputContext *webm_ctx,
                            const vpx_codec_enc_cfg_t *cfg,
                            stereo_format_t stereo_fmt, unsigned int fourcc,
                            const struct VpxRational *par);

void write_webm_block(struct WebmOutputContext *webm_ctx,
                      const vpx_codec_enc_cfg_t *cfg,
                      const vpx_codec_cx_pkt_t *pkt);

void write_webm_file_footer(struct WebmOutputContext *webm_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_WEBMENC_H_
