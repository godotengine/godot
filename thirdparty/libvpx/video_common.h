/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VIDEO_COMMON_H_
#define VPX_VIDEO_COMMON_H_

#include "./tools_common.h"

typedef struct {
  uint32_t codec_fourcc;
  int frame_width;
  int frame_height;
  struct VpxRational time_base;
} VpxVideoInfo;

#endif  // VPX_VIDEO_COMMON_H_
