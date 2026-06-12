/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_Y4MENC_H_
#define VPX_Y4MENC_H_

#include "./tools_common.h"

#include "vpx/vpx_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Y4M_BUFFER_SIZE 128

int y4m_write_file_header(char *buf, size_t len, int width, int height,
                          const struct VpxRational *framerate,
                          vpx_img_fmt_t fmt, unsigned int bit_depth);
int y4m_write_frame_header(char *buf, size_t len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_Y4MENC_H_
