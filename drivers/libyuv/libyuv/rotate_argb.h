/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_ROTATE_ARGB_H_  // NOLINT
#define INCLUDE_LIBYUV_ROTATE_ARGB_H_

#include "libyuv/basic_types.h"
#include "libyuv/rotate.h"  // For RotationMode.

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Rotate ARGB frame
LIBYUV_API
int ARGBRotate(const uint8* src_argb, int src_stride_argb,
               uint8* dst_argb, int dst_stride_argb,
               int src_width, int src_height, enum RotationMode mode);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_ROTATE_ARGB_H_  NOLINT
