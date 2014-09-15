/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_CONVERT_ARGB_H_  // NOLINT
#define INCLUDE_LIBYUV_CONVERT_ARGB_H_

#include "libyuv/basic_types.h"
// TODO(fbarchard): Remove the following headers includes
#include "libyuv/convert_from.h"
#include "libyuv/planar_functions.h"
#include "libyuv/rotate.h"

// TODO(fbarchard): This set of functions should exactly match convert.h
// Add missing Q420.
// TODO(fbarchard): Add tests. Create random content of right size and convert
// with C vs Opt and or to I420 and compare.
// TODO(fbarchard): Some of these functions lack parameter setting.

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Alias.
#define ARGBToARGB ARGBCopy

// Copy ARGB to ARGB.
LIBYUV_API
int ARGBCopy(const uint8* src_argb, int src_stride_argb,
             uint8* dst_argb, int dst_stride_argb,
             int width, int height);

// Convert I420 to ARGB.
LIBYUV_API
int I420ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Convert I422 to ARGB.
LIBYUV_API
int I422ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Convert I444 to ARGB.
LIBYUV_API
int I444ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Convert I411 to ARGB.
LIBYUV_API
int I411ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Convert I400 (grey) to ARGB.
LIBYUV_API
int I400ToARGB(const uint8* src_y, int src_stride_y,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Alias.
#define YToARGB I400ToARGB_Reference

// Convert I400 to ARGB. Reverse of ARGBToI400.
LIBYUV_API
int I400ToARGB_Reference(const uint8* src_y, int src_stride_y,
                         uint8* dst_argb, int dst_stride_argb,
                         int width, int height);

// Convert NV12 to ARGB.
LIBYUV_API
int NV12ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_uv, int src_stride_uv,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Convert NV21 to ARGB.
LIBYUV_API
int NV21ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_vu, int src_stride_vu,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Convert M420 to ARGB.
LIBYUV_API
int M420ToARGB(const uint8* src_m420, int src_stride_m420,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// TODO(fbarchard): Convert Q420 to ARGB.
// LIBYUV_API
// int Q420ToARGB(const uint8* src_y, int src_stride_y,
//                const uint8* src_yuy2, int src_stride_yuy2,
//                uint8* dst_argb, int dst_stride_argb,
//                int width, int height);

// Convert YUY2 to ARGB.
LIBYUV_API
int YUY2ToARGB(const uint8* src_yuy2, int src_stride_yuy2,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Convert UYVY to ARGB.
LIBYUV_API
int UYVYToARGB(const uint8* src_uyvy, int src_stride_uyvy,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// BGRA little endian (argb in memory) to ARGB.
LIBYUV_API
int BGRAToARGB(const uint8* src_frame, int src_stride_frame,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// ABGR little endian (rgba in memory) to ARGB.
LIBYUV_API
int ABGRToARGB(const uint8* src_frame, int src_stride_frame,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// RGBA little endian (abgr in memory) to ARGB.
LIBYUV_API
int RGBAToARGB(const uint8* src_frame, int src_stride_frame,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Deprecated function name.
#define BG24ToARGB RGB24ToARGB

// RGB little endian (bgr in memory) to ARGB.
LIBYUV_API
int RGB24ToARGB(const uint8* src_frame, int src_stride_frame,
                uint8* dst_argb, int dst_stride_argb,
                int width, int height);

// RGB big endian (rgb in memory) to ARGB.
LIBYUV_API
int RAWToARGB(const uint8* src_frame, int src_stride_frame,
              uint8* dst_argb, int dst_stride_argb,
              int width, int height);

// RGB16 (RGBP fourcc) little endian to ARGB.
LIBYUV_API
int RGB565ToARGB(const uint8* src_frame, int src_stride_frame,
                 uint8* dst_argb, int dst_stride_argb,
                 int width, int height);

// RGB15 (RGBO fourcc) little endian to ARGB.
LIBYUV_API
int ARGB1555ToARGB(const uint8* src_frame, int src_stride_frame,
                   uint8* dst_argb, int dst_stride_argb,
                   int width, int height);

// RGB12 (R444 fourcc) little endian to ARGB.
LIBYUV_API
int ARGB4444ToARGB(const uint8* src_frame, int src_stride_frame,
                   uint8* dst_argb, int dst_stride_argb,
                   int width, int height);

#ifdef HAVE_JPEG
// src_width/height provided by capture
// dst_width/height for clipping determine final size.
LIBYUV_API
int MJPGToARGB(const uint8* sample, size_t sample_size,
               uint8* dst_argb, int dst_stride_argb,
               int src_width, int src_height,
               int dst_width, int dst_height);
#endif

// Note Bayer formats (BGGR) to ARGB are in format_conversion.h.

// Convert camera sample to ARGB with cropping, rotation and vertical flip.
// "src_size" is needed to parse MJPG.
// "dst_stride_argb" number of bytes in a row of the dst_argb plane.
//   Normally this would be the same as dst_width, with recommended alignment
//   to 16 bytes for better efficiency.
//   If rotation of 90 or 270 is used, stride is affected. The caller should
//   allocate the I420 buffer according to rotation.
// "dst_stride_u" number of bytes in a row of the dst_u plane.
//   Normally this would be the same as (dst_width + 1) / 2, with
//   recommended alignment to 16 bytes for better efficiency.
//   If rotation of 90 or 270 is used, stride is affected.
// "crop_x" and "crop_y" are starting position for cropping.
//   To center, crop_x = (src_width - dst_width) / 2
//              crop_y = (src_height - dst_height) / 2
// "src_width" / "src_height" is size of src_frame in pixels.
//   "src_height" can be negative indicating a vertically flipped image source.
// "crop_width" / "crop_height" is the size to crop the src to.
//    Must be less than or equal to src_width/src_height
//    Cropping parameters are pre-rotation.
// "rotation" can be 0, 90, 180 or 270.
// "format" is a fourcc. ie 'I420', 'YUY2'
// Returns 0 for successful; -1 for invalid parameter. Non-zero for failure.
LIBYUV_API
int ConvertToARGB(const uint8* src_frame, size_t src_size,
                  uint8* dst_argb, int dst_stride_argb,
                  int crop_x, int crop_y,
                  int src_width, int src_height,
                  int crop_width, int crop_height,
                  enum RotationMode rotation,
                  uint32 format);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_CONVERT_ARGB_H_  NOLINT
