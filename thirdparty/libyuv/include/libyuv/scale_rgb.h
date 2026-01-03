/*
 *  Copyright 2022 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_SCALE_RGB_H_
#define INCLUDE_LIBYUV_SCALE_RGB_H_

#include "libyuv/basic_types.h"
#include "libyuv/scale.h"  // For FilterMode

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// RGB can be RAW, RGB24 or YUV24
// RGB scales 24 bit images by converting a row at a time to ARGB
// and using ARGB row functions to scale, then convert to RGB.
// TODO(fbarchard): Allow input/output formats to be specified.
LIBYUV_API
int RGBScale(const uint8_t* src_rgb,
             int src_stride_rgb,
             int src_width,
             int src_height,
             uint8_t* dst_rgb,
             int dst_stride_rgb,
             int dst_width,
             int dst_height,
             enum FilterMode filtering);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_SCALE_UV_H_
