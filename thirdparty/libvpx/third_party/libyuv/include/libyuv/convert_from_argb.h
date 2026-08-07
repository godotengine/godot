/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_CONVERT_FROM_ARGB_H_
#define INCLUDE_LIBYUV_CONVERT_FROM_ARGB_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Copy ARGB to ARGB.
#define ARGBToARGB ARGBCopy
LIBYUV_API
int ARGBCopy(const uint8_t* src_argb,
             int src_stride_argb,
             uint8_t* dst_argb,
             int dst_stride_argb,
             int width,
             int height);

// Convert ARGB To BGRA.
LIBYUV_API
int ARGBToBGRA(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_bgra,
               int dst_stride_bgra,
               int width,
               int height);

// Convert ARGB To ABGR.
LIBYUV_API
int ARGBToABGR(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert ARGB To RGBA.
LIBYUV_API
int ARGBToRGBA(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_rgba,
               int dst_stride_rgba,
               int width,
               int height);

// Aliases
#define ARGBToAB30 ABGRToAR30
#define ABGRToAB30 ARGBToAR30

// Convert ABGR To AR30.
LIBYUV_API
int ABGRToAR30(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert ARGB To AR30.
LIBYUV_API
int ARGBToAR30(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert ARGB To RGB24.
LIBYUV_API
int ARGBToRGB24(const uint8_t* src_argb,
                int src_stride_argb,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

// Convert ARGB To RAW.
LIBYUV_API
int ARGBToRAW(const uint8_t* src_argb,
              int src_stride_argb,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height);

// Convert ARGB To RGB565.
LIBYUV_API
int ARGBToRGB565(const uint8_t* src_argb,
                 int src_stride_argb,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height);

// Convert ARGB To RGB565 with 4x4 dither matrix (16 bytes).
// Values in dither matrix from 0 to 7 recommended.
// The order of the dither matrix is first byte is upper left.
// TODO(fbarchard): Consider pointer to 2d array for dither4x4.
// const uint8_t(*dither)[4][4];
LIBYUV_API
int ARGBToRGB565Dither(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const uint8_t* dither4x4,
                       int width,
                       int height);

// Convert ARGB To ARGB1555.
LIBYUV_API
int ARGBToARGB1555(const uint8_t* src_argb,
                   int src_stride_argb,
                   uint8_t* dst_argb1555,
                   int dst_stride_argb1555,
                   int width,
                   int height);

// Convert ARGB To ARGB4444.
LIBYUV_API
int ARGBToARGB4444(const uint8_t* src_argb,
                   int src_stride_argb,
                   uint8_t* dst_argb4444,
                   int dst_stride_argb4444,
                   int width,
                   int height);

// Convert ARGB To I444.
LIBYUV_API
int ARGBToI444(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert ARGB To I422.
LIBYUV_API
int ARGBToI422(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert ARGB To I420. (also in convert.h)
LIBYUV_API
int ARGBToI420(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert ARGB to J420. (JPeg full range I420).
LIBYUV_API
int ARGBToJ420(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_yj,
               int dst_stride_yj,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert ARGB to J422.
LIBYUV_API
int ARGBToJ422(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_yj,
               int dst_stride_yj,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert ARGB to J400. (JPeg full range).
LIBYUV_API
int ARGBToJ400(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_yj,
               int dst_stride_yj,
               int width,
               int height);

// Convert ARGB to I400.
LIBYUV_API
int ARGBToI400(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height);

// Convert ARGB to G. (Reverse of J400toARGB, which replicates G back to ARGB)
LIBYUV_API
int ARGBToG(const uint8_t* src_argb,
            int src_stride_argb,
            uint8_t* dst_g,
            int dst_stride_g,
            int width,
            int height);

// Convert ARGB To NV12.
LIBYUV_API
int ARGBToNV12(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert ARGB To NV21.
LIBYUV_API
int ARGBToNV21(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height);

// Convert ARGB To NV21.
LIBYUV_API
int ARGBToNV21(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height);

// Convert ARGB To YUY2.
LIBYUV_API
int ARGBToYUY2(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_yuy2,
               int dst_stride_yuy2,
               int width,
               int height);

// Convert ARGB To UYVY.
LIBYUV_API
int ARGBToUYVY(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_uyvy,
               int dst_stride_uyvy,
               int width,
               int height);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_CONVERT_FROM_ARGB_H_
