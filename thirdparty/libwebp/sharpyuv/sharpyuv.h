// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Sharp RGB to YUV conversion.

#ifndef WEBP_SHARPYUV_SHARPYUV_H_
#define WEBP_SHARPYUV_SHARPYUV_H_

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

// SharpYUV API version following the convention from semver.org
#define SHARPYUV_VERSION_MAJOR 0
#define SHARPYUV_VERSION_MINOR 1
#define SHARPYUV_VERSION_PATCH 0
// Version as a uint32_t. The major number is the high 8 bits.
// The minor number is the middle 8 bits. The patch number is the low 16 bits.
#define SHARPYUV_MAKE_VERSION(MAJOR, MINOR, PATCH) \
  (((MAJOR) << 24) | ((MINOR) << 16) | (PATCH))
#define SHARPYUV_VERSION                                                \
  SHARPYUV_MAKE_VERSION(SHARPYUV_VERSION_MAJOR, SHARPYUV_VERSION_MINOR, \
                        SHARPYUV_VERSION_PATCH)

// RGB to YUV conversion matrix, in 16 bit fixed point.
// y = rgb_to_y[0] * r + rgb_to_y[1] * g + rgb_to_y[2] * b + rgb_to_y[3]
// u = rgb_to_u[0] * r + rgb_to_u[1] * g + rgb_to_u[2] * b + rgb_to_u[3]
// v = rgb_to_v[0] * r + rgb_to_v[1] * g + rgb_to_v[2] * b + rgb_to_v[3]
// Then y, u and v values are divided by 1<<16 and rounded.
typedef struct {
  int rgb_to_y[4];
  int rgb_to_u[4];
  int rgb_to_v[4];
} SharpYuvConversionMatrix;

// Converts RGB to YUV420 using a downsampling algorithm that minimizes
// artefacts caused by chroma subsampling.
// This is slower than standard downsampling (averaging of 4 UV values).
// Assumes that the image will be upsampled using a bilinear filter. If nearest
// neighbor is used instead, the upsampled image might look worse than with
// standard downsampling.
// r_ptr, g_ptr, b_ptr: pointers to the source r, g and b channels. Should point
//     to uint8_t buffers if rgb_bit_depth is 8, or uint16_t buffers otherwise.
// rgb_step: distance in bytes between two horizontally adjacent pixels on the
//     r, g and b channels. If rgb_bit_depth is > 8, it should be a
//     multiple of 2.
// rgb_stride: distance in bytes between two vertically adjacent pixels on the
//     r, g, and b channels. If rgb_bit_depth is > 8, it should be a
//     multiple of 2.
// rgb_bit_depth: number of bits for each r/g/b value. One of: 8, 10, 12, 16.
//     Note: 16 bit input is truncated to 14 bits before conversion to yuv.
// yuv_bit_depth: number of bits for each y/u/v value. One of: 8, 10, 12.
// y_ptr, u_ptr, v_ptr: pointers to the destination y, u and v channels.  Should
//     point to uint8_t buffers if yuv_bit_depth is 8, or uint16_t buffers
//     otherwise.
// y_stride, u_stride, v_stride: distance in bytes between two vertically
//     adjacent pixels on the y, u and v channels. If yuv_bit_depth > 8, they
//     should be multiples of 2.
// width, height: width and height of the image in pixels
int SharpYuvConvert(const void* r_ptr, const void* g_ptr, const void* b_ptr,
                    int rgb_step, int rgb_stride, int rgb_bit_depth,
                    void* y_ptr, int y_stride, void* u_ptr, int u_stride,
                    void* v_ptr, int v_stride, int yuv_bit_depth, int width,
                    int height, const SharpYuvConversionMatrix* yuv_matrix);

// TODO(b/194336375): Add YUV444 to YUV420 conversion. Maybe also add 422
// support (it's rarely used in practice, especially for images).

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WEBP_SHARPYUV_SHARPYUV_H_
