/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_CONVERT_FROM_H_
#define INCLUDE_LIBYUV_CONVERT_FROM_H_

#include "libyuv/basic_types.h"
#include "libyuv/rotate.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// See Also convert.h for conversions from formats to I420.

// Convert 8 bit YUV to 10 bit.
#define H420ToH010 I420ToI010
LIBYUV_API
int I420ToI010(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert 8 bit YUV to 12 bit.
#define H420ToH012 I420ToI012
LIBYUV_API
int I420ToI012(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

LIBYUV_API
int I420ToI422(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

LIBYUV_API
int I420ToI444(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Copy to I400. Source can be I420, I422, I444, I400, NV12 or NV21.
LIBYUV_API
int I400Copy(const uint8_t* src_y,
             int src_stride_y,
             uint8_t* dst_y,
             int dst_stride_y,
             int width,
             int height);

LIBYUV_API
int I420ToNV12(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

LIBYUV_API
int I420ToNV21(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height);

LIBYUV_API
int I420ToYUY2(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_yuy2,
               int dst_stride_yuy2,
               int width,
               int height);

LIBYUV_API
int I420ToUYVY(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_uyvy,
               int dst_stride_uyvy,
               int width,
               int height);

// The following are from convert_argb.h
// DEPRECATED: The prototypes will be removed in future.  Use convert_argb.h

// Convert I420 to ARGB.
LIBYUV_API
int I420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert I420 to ABGR.
LIBYUV_API
int I420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert I420 to specified format.
// "dst_sample_stride" is bytes in a row for the destination. Pass 0 if the
//    buffer has contiguous rows. Can be negative. A multiple of 16 is optimal.
LIBYUV_API
int ConvertFromI420(const uint8_t* y,
                    int y_stride,
                    const uint8_t* u,
                    int u_stride,
                    const uint8_t* v,
                    int v_stride,
                    uint8_t* dst_sample,
                    int dst_sample_stride,
                    int width,
                    int height,
                    uint32_t fourcc);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_CONVERT_FROM_H_
