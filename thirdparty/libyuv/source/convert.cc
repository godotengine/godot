/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/convert.h"

#include "libyuv/basic_types.h"
#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"
#include "libyuv/rotate.h"
#include "libyuv/row.h"
#include "libyuv/scale.h"      // For ScalePlane()
#include "libyuv/scale_row.h"  // For FixedDiv
#include "libyuv/scale_uv.h"   // For UVScale()

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Subsample amount uses a shift.
//   v is value
//   a is amount to add to round up
//   s is shift to subsample down
#define SUBSAMPLE(v, a, s) (v < 0) ? (-((-v + a) >> s)) : ((v + a) >> s)
static __inline int Abs(int v) {
  return v >= 0 ? v : -v;
}

// Any I4xx To I420 format
static int I4xxToI420(const uint8_t* src_y,
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
                      int src_y_width,
                      int src_y_height,
                      int src_uv_width,
                      int src_uv_height) {
  const int dst_y_width = src_y_width;
  const int dst_y_height = Abs(src_y_height);
  const int dst_uv_width = SUBSAMPLE(dst_y_width, 1, 1);
  const int dst_uv_height = SUBSAMPLE(dst_y_height, 1, 1);
  int r;
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v ||
      src_y_width <= 0 || src_y_height == 0 || src_uv_width <= 0 ||
      src_uv_height == 0) {
    return -1;
  }
  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, src_y_width,
              src_y_height);
  }
  r = ScalePlane(src_u, src_stride_u, src_uv_width, src_uv_height, dst_u,
                 dst_stride_u, dst_uv_width, dst_uv_height, kFilterBilinear);
  if (r != 0) {
    return r;
  }
  r = ScalePlane(src_v, src_stride_v, src_uv_width, src_uv_height, dst_v,
                 dst_stride_v, dst_uv_width, dst_uv_height, kFilterBilinear);
  return r;
}

// Copy I420 with optional vertical flipping using negative height.
LIBYUV_API
int I420Copy(const uint8_t* src_y,
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
             int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (halfheight - 1) * src_stride_u;
    src_v = src_v + (halfheight - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  // Copy UV planes.
  CopyPlane(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, halfheight);
  CopyPlane(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, halfheight);
  return 0;
}

// Copy I010 with optional flipping.
LIBYUV_API
int I010Copy(const uint16_t* src_y,
             int src_stride_y,
             const uint16_t* src_u,
             int src_stride_u,
             const uint16_t* src_v,
             int src_stride_v,
             uint16_t* dst_y,
             int dst_stride_y,
             uint16_t* dst_u,
             int dst_stride_u,
             uint16_t* dst_v,
             int dst_stride_v,
             int width,
             int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (halfheight - 1) * src_stride_u;
    src_v = src_v + (halfheight - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  if (dst_y) {
    CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  // Copy UV planes.
  CopyPlane_16(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, halfheight);
  CopyPlane_16(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, halfheight);
  return 0;
}

static int Planar16bitTo8bit(const uint16_t* src_y,
                             int src_stride_y,
                             const uint16_t* src_u,
                             int src_stride_u,
                             const uint16_t* src_v,
                             int src_stride_v,
                             uint8_t* dst_y,
                             int dst_stride_y,
                             uint8_t* dst_u,
                             int dst_stride_u,
                             uint8_t* dst_v,
                             int dst_stride_v,
                             int width,
                             int height,
                             int subsample_x,
                             int subsample_y,
                             int depth) {
  int uv_width = SUBSAMPLE(width, subsample_x, subsample_x);
  int uv_height = SUBSAMPLE(height, subsample_y, subsample_y);
  int scale = 1 << (24 - depth);
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    uv_height = -uv_height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (uv_height - 1) * src_stride_u;
    src_v = src_v + (uv_height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  // Convert Y plane.
  if (dst_y) {
    Convert16To8Plane(src_y, src_stride_y, dst_y, dst_stride_y, scale, width,
                      height);
  }
  // Convert UV planes.
  Convert16To8Plane(src_u, src_stride_u, dst_u, dst_stride_u, scale, uv_width,
                    uv_height);
  Convert16To8Plane(src_v, src_stride_v, dst_v, dst_stride_v, scale, uv_width,
                    uv_height);
  return 0;
}

static int I41xToI420(const uint16_t* src_y,
                      int src_stride_y,
                      const uint16_t* src_u,
                      int src_stride_u,
                      const uint16_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_y,
                      int dst_stride_y,
                      uint8_t* dst_u,
                      int dst_stride_u,
                      uint8_t* dst_v,
                      int dst_stride_v,
                      int width,
                      int height,
                      int depth) {
  const int scale = 1 << (24 - depth);

  if (width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (height - 1) * src_stride_u;
    src_v = src_v + (height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  {
    const int uv_width = SUBSAMPLE(width, 1, 1);
    const int uv_height = SUBSAMPLE(height, 1, 1);

    Convert16To8Plane(src_y, src_stride_y, dst_y, dst_stride_y, scale, width,
                      height);
    ScalePlaneDown2_16To8(width, height, uv_width, uv_height, src_stride_u,
                          dst_stride_u, src_u, dst_u, scale, kFilterBilinear);
    ScalePlaneDown2_16To8(width, height, uv_width, uv_height, src_stride_v,
                          dst_stride_v, src_v, dst_v, scale, kFilterBilinear);
  }
  return 0;
}

static int I21xToI420(const uint16_t* src_y,
                      int src_stride_y,
                      const uint16_t* src_u,
                      int src_stride_u,
                      const uint16_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_y,
                      int dst_stride_y,
                      uint8_t* dst_u,
                      int dst_stride_u,
                      uint8_t* dst_v,
                      int dst_stride_v,
                      int width,
                      int height,
                      int depth) {
  const int scale = 1 << (24 - depth);

  if (width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (height - 1) * src_stride_u;
    src_v = src_v + (height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  {
    const int uv_width = SUBSAMPLE(width, 1, 1);
    const int uv_height = SUBSAMPLE(height, 1, 1);
    const int dy = FixedDiv(height, uv_height);

    Convert16To8Plane(src_y, src_stride_y, dst_y, dst_stride_y, scale, width,
                      height);
    ScalePlaneVertical_16To8(height, uv_width, uv_height, src_stride_u,
                             dst_stride_u, src_u, dst_u, 0, 32768, dy,
                             /*bpp=*/1, scale, kFilterBilinear);
    ScalePlaneVertical_16To8(height, uv_width, uv_height, src_stride_v,
                             dst_stride_v, src_v, dst_v, 0, 32768, dy,
                             /*bpp=*/1, scale, kFilterBilinear);
  }
  return 0;
}

// Convert 10 bit YUV to 8 bit.
LIBYUV_API
int I010ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return Planar16bitTo8bit(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_y, dst_stride_y, dst_u,
                           dst_stride_u, dst_v, dst_stride_v, width, height, 1,
                           1, 10);
}

LIBYUV_API
int I210ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return I21xToI420(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, 10);
}

LIBYUV_API
int I210ToI422(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return Planar16bitTo8bit(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_y, dst_stride_y, dst_u,
                           dst_stride_u, dst_v, dst_stride_v, width, height, 1,
                           0, 10);
}

LIBYUV_API
int I410ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return I41xToI420(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, 10);
}

LIBYUV_API
int I410ToI444(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return Planar16bitTo8bit(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_y, dst_stride_y, dst_u,
                           dst_stride_u, dst_v, dst_stride_v, width, height, 0,
                           0, 10);
}

LIBYUV_API
int I012ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return Planar16bitTo8bit(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_y, dst_stride_y, dst_u,
                           dst_stride_u, dst_v, dst_stride_v, width, height, 1,
                           1, 12);
}

LIBYUV_API
int I212ToI422(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return Planar16bitTo8bit(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_y, dst_stride_y, dst_u,
                           dst_stride_u, dst_v, dst_stride_v, width, height, 1,
                           0, 12);
}

LIBYUV_API
int I212ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return I21xToI420(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, 12);
}

LIBYUV_API
int I412ToI444(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return Planar16bitTo8bit(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_y, dst_stride_y, dst_u,
                           dst_stride_u, dst_v, dst_stride_v, width, height, 0,
                           0, 12);
}

LIBYUV_API
int I412ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return I41xToI420(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, 12);
}

// Any Ix10 To I010 format
static int Ix10ToI010(const uint16_t* src_y,
                      int src_stride_y,
                      const uint16_t* src_u,
                      int src_stride_u,
                      const uint16_t* src_v,
                      int src_stride_v,
                      uint16_t* dst_y,
                      int dst_stride_y,
                      uint16_t* dst_u,
                      int dst_stride_u,
                      uint16_t* dst_v,
                      int dst_stride_v,
                      int width,
                      int height,
                      int subsample_x,
                      int subsample_y) {
  const int dst_y_width = width;
  const int dst_y_height = Abs(height);
  const int src_uv_width = SUBSAMPLE(width, subsample_x, subsample_x);
  const int src_uv_height = SUBSAMPLE(height, subsample_y, subsample_y);
  const int dst_uv_width = SUBSAMPLE(dst_y_width, 1, 1);
  const int dst_uv_height = SUBSAMPLE(dst_y_height, 1, 1);
  int r;
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  if (dst_y) {
    CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  r = ScalePlane_12(src_u, src_stride_u, src_uv_width, src_uv_height, dst_u,
                    dst_stride_u, dst_uv_width, dst_uv_height, kFilterBilinear);
  if (r != 0) {
    return r;
  }
  r = ScalePlane_12(src_v, src_stride_v, src_uv_width, src_uv_height, dst_v,
                    dst_stride_v, dst_uv_width, dst_uv_height, kFilterBilinear);
  return r;
}

LIBYUV_API
int I410ToI010(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return Ix10ToI010(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, 0, 0);
}

LIBYUV_API
int I210ToI010(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return Ix10ToI010(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, 1, 0);
}

// Any I[420]1[02] to P[420]1[02] format
static int IxxxToPxxx(const uint16_t* src_y,
                      int src_stride_y,
                      const uint16_t* src_u,
                      int src_stride_u,
                      const uint16_t* src_v,
                      int src_stride_v,
                      uint16_t* dst_y,
                      int dst_stride_y,
                      uint16_t* dst_uv,
                      int dst_stride_uv,
                      int width,
                      int height,
                      int subsample_x,
                      int subsample_y,
                      int depth) {
  const int uv_width = SUBSAMPLE(width, subsample_x, subsample_x);
  const int uv_height = SUBSAMPLE(height, subsample_y, subsample_y);
  if (width <= 0 || height == 0) {
    return -1;
  }

  ConvertToMSBPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height,
                       depth);
  MergeUVPlane_16(src_u, src_stride_u, src_v, src_stride_v, dst_uv,
                  dst_stride_uv, uv_width, uv_height, depth);
  return 0;
}

LIBYUV_API
int I010ToP010(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  return IxxxToPxxx(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_uv, dst_stride_uv,
                    width, height, 1, 1, 10);
}

LIBYUV_API
int I010ToNV12(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int y;
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  const int scale = 16385;  // 16384 for 10 bits
  void (*Convert16To8Row)(const uint16_t* src_y, uint8_t* dst_y, int scale,
                          int width) = Convert16To8Row_C;
  void (*MergeUVRow)(const uint8_t* src_u, const uint8_t* src_v,
                     uint8_t* dst_uv, int width) = MergeUVRow_C;
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_uv || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (halfheight - 1) * src_stride_u;
    src_v = src_v + (halfheight - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }
#if defined(HAS_CONVERT16TO8ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Convert16To8Row = Convert16To8Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      Convert16To8Row = Convert16To8Row_NEON;
    }
  }
#endif
#if defined(HAS_CONVERT16TO8ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    Convert16To8Row = Convert16To8Row_SME;
  }
#endif
#if defined(HAS_CONVERT16TO8ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    Convert16To8Row = Convert16To8Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      Convert16To8Row = Convert16To8Row_SSSE3;
    }
  }
#endif
#if defined(HAS_CONVERT16TO8ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Convert16To8Row = Convert16To8Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      Convert16To8Row = Convert16To8Row_AVX2;
    }
  }
#endif
#if defined(HAS_CONVERT16TO8ROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW)) {
    Convert16To8Row = Convert16To8Row_Any_AVX512BW;
    if (IS_ALIGNED(width, 64)) {
      Convert16To8Row = Convert16To8Row_AVX512BW;
    }
  }
#endif

#if defined(HAS_MERGEUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    MergeUVRow = MergeUVRow_Any_SSE2;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeUVRow = MergeUVRow_Any_AVX2;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW)) {
    MergeUVRow = MergeUVRow_Any_AVX512BW;
    if (IS_ALIGNED(halfwidth, 32)) {
      MergeUVRow = MergeUVRow_AVX512BW;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeUVRow = MergeUVRow_Any_NEON;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_NEON;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    MergeUVRow = MergeUVRow_SME;
  }
#endif
#if defined(HAS_MERGEUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    MergeUVRow = MergeUVRow_Any_MSA;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_MSA;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    MergeUVRow = MergeUVRow_Any_LSX;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_LSX;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    MergeUVRow = MergeUVRow_RVV;
  }
#endif

  // Convert Y plane.
  if (dst_y) {
    Convert16To8Plane(src_y, src_stride_y, dst_y, dst_stride_y, scale, width,
                      height);
  }

  {
    // Allocate a row of uv.
    align_buffer_64(row_u, ((halfwidth + 31) & ~31) * 2);
    uint8_t* row_v = row_u + ((halfwidth + 31) & ~31);
    if (!row_u)
      return 1;

    for (y = 0; y < halfheight; ++y) {
      Convert16To8Row(src_u, row_u, scale, halfwidth);
      Convert16To8Row(src_v, row_v, scale, halfwidth);
      MergeUVRow(row_u, row_v, dst_uv, halfwidth);
      src_u += src_stride_u;
      src_v += src_stride_v;
      dst_uv += dst_stride_uv;
    }
    free_aligned_buffer_64(row_u);
  }
  return 0;
}

LIBYUV_API
int I210ToP210(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  return IxxxToPxxx(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_uv, dst_stride_uv,
                    width, height, 1, 0, 10);
}

LIBYUV_API
int I012ToP012(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  return IxxxToPxxx(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_uv, dst_stride_uv,
                    width, height, 1, 1, 12);
}

LIBYUV_API
int I212ToP212(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  return IxxxToPxxx(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_uv, dst_stride_uv,
                    width, height, 1, 0, 12);
}

// 422 chroma is 1/2 width, 1x height
// 420 chroma is 1/2 width, 1/2 height
LIBYUV_API
int I422ToI420(const uint8_t* src_y,
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
               int height) {
  const int src_uv_width = SUBSAMPLE(width, 1, 1);
  return I4xxToI420(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, src_uv_width, height);
}

LIBYUV_API
int I422ToI210(const uint8_t* src_y,
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
               int height) {
  int halfwidth = (width + 1) >> 1;
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (height - 1) * src_stride_u;
    src_v = src_v + (height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  // Convert Y plane.
  Convert8To16Plane(src_y, src_stride_y, dst_y, dst_stride_y, 1024, width,
                    height);
  // Convert UV planes.
  Convert8To16Plane(src_u, src_stride_u, dst_u, dst_stride_u, 1024, halfwidth,
                    height);
  Convert8To16Plane(src_v, src_stride_v, dst_v, dst_stride_v, 1024, halfwidth,
                    height);
  return 0;
}

// TODO(fbarchard): Implement row conversion.
LIBYUV_API
int I422ToNV21(const uint8_t* src_y,
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
               int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (height - 1) * src_stride_u;
    src_v = src_v + (height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  // Allocate u and v buffers
  align_buffer_64(plane_u, halfwidth * halfheight * 2);
  uint8_t* plane_v = plane_u + halfwidth * halfheight;
  if (!plane_u)
    return 1;

  I422ToI420(src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
             dst_y, dst_stride_y, plane_u, halfwidth, plane_v, halfwidth, width,
             height);
  MergeUVPlane(plane_v, halfwidth, plane_u, halfwidth, dst_vu, dst_stride_vu,
               halfwidth, halfheight);
  free_aligned_buffer_64(plane_u);
  return 0;
}

LIBYUV_API
int MM21ToNV12(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  if (!src_uv || !dst_uv || width <= 0) {
    return -1;
  }

  int sign = height < 0 ? -1 : 1;

  if (dst_y) {
    DetilePlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height, 32);
  }
  DetilePlane(src_uv, src_stride_uv, dst_uv, dst_stride_uv, (width + 1) & ~1,
              (height + sign) / 2, 16);

  return 0;
}

LIBYUV_API
int MM21ToI420(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int sign = height < 0 ? -1 : 1;

  if (!src_uv || !dst_u || !dst_v || width <= 0) {
    return -1;
  }

  if (dst_y) {
    DetilePlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height, 32);
  }
  DetileSplitUVPlane(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                     dst_stride_v, (width + 1) & ~1, (height + sign) / 2, 16);

  return 0;
}

LIBYUV_API
int MM21ToYUY2(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_yuy2,
               int dst_stride_yuy2,
               int width,
               int height) {
  if (!src_y || !src_uv || !dst_yuy2 || width <= 0) {
    return -1;
  }

  DetileToYUY2(src_y, src_stride_y, src_uv, src_stride_uv, dst_yuy2,
               dst_stride_yuy2, width, height, 32);

  return 0;
}

// Convert MT2T into P010. See tinyurl.com/mtk-10bit-video-format for format
// documentation.
// TODO(greenjustin): Add an MT2T to I420 conversion.
LIBYUV_API
int MT2TToP010(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  if (width <= 0 || !height || !src_uv || !dst_uv) {
    return -1;
  }

  {
    int uv_width = (width + 1) & ~1;
    int uv_height = (height + 1) / 2;
    int y = 0;
    const int tile_width = 16;
    const int y_tile_height = 32;
    const int uv_tile_height = 16;
    int padded_width = (width + tile_width - 1) & ~(tile_width - 1);
    int y_tile_row_size = padded_width * y_tile_height * 10 / 8;
    int uv_tile_row_size = padded_width * uv_tile_height * 10 / 8;
    size_t row_buf_size = padded_width * y_tile_height * sizeof(uint16_t);
    void (*UnpackMT2T)(const uint8_t* src, uint16_t* dst, size_t size) =
        UnpackMT2T_C;
    align_buffer_64(row_buf, row_buf_size);
    if (!row_buf)
      return 1;

#if defined(HAS_UNPACKMT2T_NEON)
    if (TestCpuFlag(kCpuHasNEON)) {
      UnpackMT2T = UnpackMT2T_NEON;
    }
#endif
    // Negative height means invert the image.
    if (height < 0) {
      height = -height;
      uv_height = (height + 1) / 2;
      if (dst_y) {
        dst_y = dst_y + (height - 1) * dst_stride_y;
        dst_stride_y = -dst_stride_y;
      }
      dst_uv = dst_uv + (uv_height - 1) * dst_stride_uv;
      dst_stride_uv = -dst_stride_uv;
    }

    // Unpack and detile Y in rows of tiles
    if (src_y && dst_y) {
      for (y = 0; y < (height & ~(y_tile_height - 1)); y += y_tile_height) {
        UnpackMT2T(src_y, (uint16_t*)row_buf, y_tile_row_size);
        DetilePlane_16((uint16_t*)row_buf, padded_width, dst_y, dst_stride_y,
                       width, y_tile_height, y_tile_height);
        src_y += src_stride_y * y_tile_height;
        dst_y += dst_stride_y * y_tile_height;
      }
      if (height & (y_tile_height - 1)) {
        UnpackMT2T(src_y, (uint16_t*)row_buf, y_tile_row_size);
        DetilePlane_16((uint16_t*)row_buf, padded_width, dst_y, dst_stride_y,
                       width, height & (y_tile_height - 1), y_tile_height);
      }
    }

    // Unpack and detile UV plane
    for (y = 0; y < (uv_height & ~(uv_tile_height - 1)); y += uv_tile_height) {
      UnpackMT2T(src_uv, (uint16_t*)row_buf, uv_tile_row_size);
      DetilePlane_16((uint16_t*)row_buf, padded_width, dst_uv, dst_stride_uv,
                     uv_width, uv_tile_height, uv_tile_height);
      src_uv += src_stride_uv * uv_tile_height;
      dst_uv += dst_stride_uv * uv_tile_height;
    }
    if (uv_height & (uv_tile_height - 1)) {
      UnpackMT2T(src_uv, (uint16_t*)row_buf, uv_tile_row_size);
      DetilePlane_16((uint16_t*)row_buf, padded_width, dst_uv, dst_stride_uv,
                     uv_width, uv_height & (uv_tile_height - 1),
                     uv_tile_height);
    }
    free_aligned_buffer_64(row_buf);
  }
  return 0;
}

#ifdef I422TONV21_ROW_VERSION
// Unittest fails for this version.
// 422 chroma is 1/2 width, 1x height
// 420 chroma is 1/2 width, 1/2 height
// Swap src_u and src_v to implement I422ToNV12
LIBYUV_API
int I422ToNV21(const uint8_t* src_y,
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
               int height) {
  int y;
  void (*MergeUVRow)(const uint8_t* src_u, const uint8_t* src_v,
                     uint8_t* dst_uv, int width) = MergeUVRow_C;
  void (*InterpolateRow)(uint8_t* dst_ptr, const uint8_t* src_ptr,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_C;
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_vu || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (halfheight - 1) * src_stride_u;
    src_v = src_v + (halfheight - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }
#if defined(HAS_MERGEUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    MergeUVRow = MergeUVRow_Any_SSE2;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeUVRow = MergeUVRow_Any_AVX2;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW)) {
    MergeUVRow = MergeUVRow_Any_AVX512BW;
    if (IS_ALIGNED(halfwidth, 32)) {
      MergeUVRow = MergeUVRow_AVX512BW;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeUVRow = MergeUVRow_Any_NEON;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_NEON;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    MergeUVRow = MergeUVRow_SME;
  }
#endif
#if defined(HAS_MERGEUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    MergeUVRow = MergeUVRow_Any_MSA;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_MSA;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    MergeUVRow = MergeUVRow_Any_LSX;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow = MergeUVRow_LSX;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    MergeUVRow = MergeUVRow_RVV;
  }
#endif
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow = InterpolateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    InterpolateRow = InterpolateRow_SME;
  }
#endif
#if defined(HAS_INTERPOLATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow = InterpolateRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_MSA;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    InterpolateRow = InterpolateRow_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_LSX;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    InterpolateRow = InterpolateRow_RVV;
  }
#endif

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, halfwidth, height);
  }
  {
    // Allocate 2 rows of vu.
    int awidth = halfwidth * 2;
    align_buffer_64(row_vu_0, awidth * 2);
    uint8_t* row_vu_1 = row_vu_0 + awidth;
    if (!row_vu_0)
      return 1;

    for (y = 0; y < height - 1; y += 2) {
      MergeUVRow(src_v, src_u, row_vu_0, halfwidth);
      MergeUVRow(src_v + src_stride_v, src_u + src_stride_u, row_vu_1,
                 halfwidth);
      InterpolateRow(dst_vu, row_vu_0, awidth, awidth, 128);
      src_u += src_stride_u * 2;
      src_v += src_stride_v * 2;
      dst_vu += dst_stride_vu;
    }
    if (height & 1) {
      MergeUVRow(src_v, src_u, dst_vu, halfwidth);
    }
    free_aligned_buffer_64(row_vu_0);
  }
  return 0;
}
#endif  // I422TONV21_ROW_VERSION

// 444 chroma is 1x width, 1x height
// 420 chroma is 1/2 width, 1/2 height
LIBYUV_API
int I444ToI420(const uint8_t* src_y,
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
               int height) {
  return I4xxToI420(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, width, height);
}

LIBYUV_API
int I444ToNV12(const uint8_t* src_y,
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
               int height) {
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_uv || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (height - 1) * src_stride_u;
    src_v = src_v + (height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }
  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  HalfMergeUVPlane(src_u, src_stride_u, src_v, src_stride_v, dst_uv,
                   dst_stride_uv, width, height);
  return 0;
}

LIBYUV_API
int I444ToNV21(const uint8_t* src_y,
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
               int height) {
  return I444ToNV12(src_y, src_stride_y, src_v, src_stride_v, src_u,
                    src_stride_u, dst_y, dst_stride_y, dst_vu, dst_stride_vu,
                    width, height);
}

// I400 is greyscale typically used in MJPG
LIBYUV_API
int I400ToI420(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if ((!src_y && dst_y) || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_stride_y = -src_stride_y;
  }
  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  SetPlane(dst_u, dst_stride_u, halfwidth, halfheight, 128);
  SetPlane(dst_v, dst_stride_v, halfwidth, halfheight, 128);
  return 0;
}

// I400 is greyscale typically used in MJPG
LIBYUV_API
int I400ToNV21(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if ((!src_y && dst_y) || !dst_vu || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_stride_y = -src_stride_y;
  }
  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  SetPlane(dst_vu, dst_stride_vu, halfwidth * 2, halfheight, 128);
  return 0;
}

// Convert NV12 to I420.
// TODO(fbarchard): Consider inverting destination. Faster on ARM with prfm.
LIBYUV_API
int NV12ToI420(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if ((!src_y && dst_y) || !src_uv || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_uv = src_uv + (halfheight - 1) * src_stride_uv;
    src_stride_y = -src_stride_y;
    src_stride_uv = -src_stride_uv;
  }
  // Coalesce rows.
  if (src_stride_y == width && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_y = 0;
  }
  // Coalesce rows.
  if (src_stride_uv == halfwidth * 2 && dst_stride_u == halfwidth &&
      dst_stride_v == halfwidth) {
    halfwidth *= halfheight;
    halfheight = 1;
    src_stride_uv = dst_stride_u = dst_stride_v = 0;
  }

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }

  // Split UV plane - NV12 / NV21
  SplitUVPlane(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v, dst_stride_v,
               halfwidth, halfheight);

  return 0;
}

// Convert NV21 to I420.  Same as NV12 but u and v pointers swapped.
LIBYUV_API
int NV21ToI420(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return NV12ToI420(src_y, src_stride_y, src_vu, src_stride_vu, dst_y,
                    dst_stride_y, dst_v, dst_stride_v, dst_u, dst_stride_u,
                    width, height);
}

LIBYUV_API
int NV12ToNV24(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int r;
  if ((!src_y && dst_y) || !src_uv || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  r = UVScale(src_uv, src_stride_uv, SUBSAMPLE(width, 1, 1),
              SUBSAMPLE(height, 1, 1), dst_uv, dst_stride_uv, Abs(width),
              Abs(height), kFilterBilinear);
  return r;
}

LIBYUV_API
int NV16ToNV24(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int r;
  if ((!src_y && dst_y) || !src_uv || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  r = UVScale(src_uv, src_stride_uv, SUBSAMPLE(width, 1, 1), height, dst_uv,
              dst_stride_uv, Abs(width), Abs(height), kFilterBilinear);
  return r;
}

// Any P[420]1[02] to I[420]1[02] format
static int PxxxToIxxx(const uint16_t* src_y,
                      int src_stride_y,
                      const uint16_t* src_uv,
                      int src_stride_uv,
                      uint16_t* dst_y,
                      int dst_stride_y,
                      uint16_t* dst_u,
                      int dst_stride_u,
                      uint16_t* dst_v,
                      int dst_stride_v,
                      int width,
                      int height,
                      int subsample_x,
                      int subsample_y,
                      int depth) {
  const int uv_width = SUBSAMPLE(width, subsample_x, subsample_x);
  const int uv_height = SUBSAMPLE(height, subsample_y, subsample_y);
  if (!src_y || !dst_y || !src_uv || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  ConvertToLSBPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height,
                       depth);
  SplitUVPlane_16(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                  dst_stride_v, uv_width, uv_height, depth);
  return 0;
}

LIBYUV_API
int P010ToI010(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return PxxxToIxxx(src_y, src_stride_y, src_uv, src_stride_uv, dst_y,
                    dst_stride_y, dst_u, dst_stride_u, dst_v, dst_stride_v,
                    width, height, 1, 1, 10);
}

LIBYUV_API
int P012ToI012(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return PxxxToIxxx(src_y, src_stride_y, src_uv, src_stride_uv, dst_y,
                    dst_stride_y, dst_u, dst_stride_u, dst_v, dst_stride_v,
                    width, height, 1, 1, 12);
}

LIBYUV_API
int P010ToP410(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int r;
  if ((!src_y && dst_y) || !src_uv || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }

  if (dst_y) {
    CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  r = UVScale_16(src_uv, src_stride_uv, SUBSAMPLE(width, 1, 1),
                 SUBSAMPLE(height, 1, 1), dst_uv, dst_stride_uv, Abs(width),
                 Abs(height), kFilterBilinear);
  return r;
}

LIBYUV_API
int P210ToP410(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int r;
  if ((!src_y && dst_y) || !src_uv || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }

  if (dst_y) {
    CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  r = UVScale_16(src_uv, src_stride_uv, SUBSAMPLE(width, 1, 1), height, dst_uv,
                 dst_stride_uv, Abs(width), Abs(height), kFilterBilinear);
  return r;
}

// Convert YUY2 to I420.
LIBYUV_API
int YUY2ToI420(const uint8_t* src_yuy2,
               int src_stride_yuy2,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*YUY2ToUVRow)(const uint8_t* src_yuy2, int src_stride_yuy2,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      YUY2ToUVRow_C;
  void (*YUY2ToYRow)(const uint8_t* src_yuy2, uint8_t* dst_y, int width) =
      YUY2ToYRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_yuy2 = src_yuy2 + (height - 1) * src_stride_yuy2;
    src_stride_yuy2 = -src_stride_yuy2;
  }
#if defined(HAS_YUY2TOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    YUY2ToUVRow = YUY2ToUVRow_Any_SSE2;
    YUY2ToYRow = YUY2ToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToUVRow = YUY2ToUVRow_SSE2;
      YUY2ToYRow = YUY2ToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    YUY2ToUVRow = YUY2ToUVRow_Any_AVX2;
    YUY2ToYRow = YUY2ToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToUVRow = YUY2ToUVRow_AVX2;
      YUY2ToYRow = YUY2ToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    YUY2ToYRow = YUY2ToYRow_Any_NEON;
    YUY2ToUVRow = YUY2ToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToYRow = YUY2ToYRow_NEON;
      YUY2ToUVRow = YUY2ToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_MSA) && defined(HAS_YUY2TOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    YUY2ToYRow = YUY2ToYRow_Any_MSA;
    YUY2ToUVRow = YUY2ToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_MSA;
      YUY2ToUVRow = YUY2ToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_LSX) && defined(HAS_YUY2TOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    YUY2ToYRow = YUY2ToYRow_Any_LSX;
    YUY2ToUVRow = YUY2ToUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToYRow = YUY2ToYRow_LSX;
      YUY2ToUVRow = YUY2ToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_LASX) && defined(HAS_YUY2TOUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    YUY2ToYRow = YUY2ToYRow_Any_LASX;
    YUY2ToUVRow = YUY2ToUVRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_LASX;
      YUY2ToUVRow = YUY2ToUVRow_LASX;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    YUY2ToUVRow(src_yuy2, src_stride_yuy2, dst_u, dst_v, width);
    YUY2ToYRow(src_yuy2, dst_y, width);
    YUY2ToYRow(src_yuy2 + src_stride_yuy2, dst_y + dst_stride_y, width);
    src_yuy2 += src_stride_yuy2 * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    YUY2ToUVRow(src_yuy2, 0, dst_u, dst_v, width);
    YUY2ToYRow(src_yuy2, dst_y, width);
  }
  return 0;
}

// Convert UYVY to I420.
LIBYUV_API
int UYVYToI420(const uint8_t* src_uyvy,
               int src_stride_uyvy,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*UYVYToUVRow)(const uint8_t* src_uyvy, int src_stride_uyvy,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      UYVYToUVRow_C;
  void (*UYVYToYRow)(const uint8_t* src_uyvy, uint8_t* dst_y, int width) =
      UYVYToYRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uyvy = src_uyvy + (height - 1) * src_stride_uyvy;
    src_stride_uyvy = -src_stride_uyvy;
  }
#if defined(HAS_UYVYTOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    UYVYToUVRow = UYVYToUVRow_Any_SSE2;
    UYVYToYRow = UYVYToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      UYVYToUVRow = UYVYToUVRow_SSE2;
      UYVYToYRow = UYVYToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    UYVYToUVRow = UYVYToUVRow_Any_AVX2;
    UYVYToYRow = UYVYToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      UYVYToUVRow = UYVYToUVRow_AVX2;
      UYVYToYRow = UYVYToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    UYVYToYRow = UYVYToYRow_Any_NEON;
    UYVYToUVRow = UYVYToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_NEON;
      UYVYToUVRow = UYVYToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    UYVYToYRow = UYVYToYRow_Any_MSA;
    UYVYToUVRow = UYVYToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      UYVYToYRow = UYVYToYRow_MSA;
      UYVYToUVRow = UYVYToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    UYVYToYRow = UYVYToYRow_Any_LSX;
    UYVYToUVRow = UYVYToUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_LSX;
      UYVYToUVRow = UYVYToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    UYVYToYRow = UYVYToYRow_Any_LSX;
    UYVYToUVRow = UYVYToUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_LSX;
      UYVYToUVRow = UYVYToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    UYVYToYRow = UYVYToYRow_Any_LASX;
    UYVYToUVRow = UYVYToUVRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      UYVYToYRow = UYVYToYRow_LASX;
      UYVYToUVRow = UYVYToUVRow_LASX;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    UYVYToUVRow(src_uyvy, src_stride_uyvy, dst_u, dst_v, width);
    UYVYToYRow(src_uyvy, dst_y, width);
    UYVYToYRow(src_uyvy + src_stride_uyvy, dst_y + dst_stride_y, width);
    src_uyvy += src_stride_uyvy * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    UYVYToUVRow(src_uyvy, 0, dst_u, dst_v, width);
    UYVYToYRow(src_uyvy, dst_y, width);
  }
  return 0;
}

// Convert AYUV to NV12.
LIBYUV_API
int AYUVToNV12(const uint8_t* src_ayuv,
               int src_stride_ayuv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int y;
  void (*AYUVToUVRow)(const uint8_t* src_ayuv, int src_stride_ayuv,
                      uint8_t* dst_uv, int width) = AYUVToUVRow_C;
  void (*AYUVToYRow)(const uint8_t* src_ayuv, uint8_t* dst_y, int width) =
      AYUVToYRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ayuv = src_ayuv + (height - 1) * src_stride_ayuv;
    src_stride_ayuv = -src_stride_ayuv;
  }
// place holders for future intel code
#if defined(HAS_AYUVTOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    AYUVToUVRow = AYUVToUVRow_Any_SSE2;
    AYUVToYRow = AYUVToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      AYUVToUVRow = AYUVToUVRow_SSE2;
      AYUVToYRow = AYUVToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_AYUVTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    AYUVToUVRow = AYUVToUVRow_Any_AVX2;
    AYUVToYRow = AYUVToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      AYUVToUVRow = AYUVToUVRow_AVX2;
      AYUVToYRow = AYUVToYRow_AVX2;
    }
  }
#endif

#if defined(HAS_AYUVTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    AYUVToYRow = AYUVToYRow_Any_NEON;
    AYUVToUVRow = AYUVToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      AYUVToYRow = AYUVToYRow_NEON;
      AYUVToUVRow = AYUVToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_AYUVTOUVROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    AYUVToUVRow = AYUVToUVRow_Any_SVE2;
    if (IS_ALIGNED(width, 2)) {
      AYUVToUVRow = AYUVToUVRow_SVE2;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    AYUVToUVRow(src_ayuv, src_stride_ayuv, dst_uv, width);
    AYUVToYRow(src_ayuv, dst_y, width);
    AYUVToYRow(src_ayuv + src_stride_ayuv, dst_y + dst_stride_y, width);
    src_ayuv += src_stride_ayuv * 2;
    dst_y += dst_stride_y * 2;
    dst_uv += dst_stride_uv;
  }
  if (height & 1) {
    AYUVToUVRow(src_ayuv, 0, dst_uv, width);
    AYUVToYRow(src_ayuv, dst_y, width);
  }
  return 0;
}

// Convert AYUV to NV21.
LIBYUV_API
int AYUVToNV21(const uint8_t* src_ayuv,
               int src_stride_ayuv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height) {
  int y;
  void (*AYUVToVURow)(const uint8_t* src_ayuv, int src_stride_ayuv,
                      uint8_t* dst_vu, int width) = AYUVToVURow_C;
  void (*AYUVToYRow)(const uint8_t* src_ayuv, uint8_t* dst_y, int width) =
      AYUVToYRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ayuv = src_ayuv + (height - 1) * src_stride_ayuv;
    src_stride_ayuv = -src_stride_ayuv;
  }
// place holders for future intel code
#if defined(HAS_AYUVTOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    AYUVToVURow = AYUVToVURow_Any_SSE2;
    AYUVToYRow = AYUVToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      AYUVToVURow = AYUVToVURow_SSE2;
      AYUVToYRow = AYUVToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_AYUVTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    AYUVToVURow = AYUVToVURow_Any_AVX2;
    AYUVToYRow = AYUVToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      AYUVToVURow = AYUVToVURow_AVX2;
      AYUVToYRow = AYUVToYRow_AVX2;
    }
  }
#endif

#if defined(HAS_AYUVTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    AYUVToYRow = AYUVToYRow_Any_NEON;
    AYUVToVURow = AYUVToVURow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      AYUVToYRow = AYUVToYRow_NEON;
      AYUVToVURow = AYUVToVURow_NEON;
    }
  }
#endif
#if defined(HAS_AYUVTOVUROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    AYUVToVURow = AYUVToVURow_Any_SVE2;
    if (IS_ALIGNED(width, 2)) {
      AYUVToVURow = AYUVToVURow_SVE2;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    AYUVToVURow(src_ayuv, src_stride_ayuv, dst_vu, width);
    AYUVToYRow(src_ayuv, dst_y, width);
    AYUVToYRow(src_ayuv + src_stride_ayuv, dst_y + dst_stride_y, width);
    src_ayuv += src_stride_ayuv * 2;
    dst_y += dst_stride_y * 2;
    dst_vu += dst_stride_vu;
  }
  if (height & 1) {
    AYUVToVURow(src_ayuv, 0, dst_vu, width);
    AYUVToYRow(src_ayuv, dst_y, width);
  }
  return 0;
}

// Convert ARGB to I420.
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
               int height) {
  int y;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  if (!src_argb || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
#if defined(HAS_ARGBTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToYRow = ARGBToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd)) {
    ARGBToYRow = ARGBToYRow_Any_NEON_DotProd;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_NEON_DotProd;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToUVRow = ARGBToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_NEON_I8MM)
  if (TestCpuFlag(kCpuHasNeonI8MM)) {
    ARGBToUVRow = ARGBToUVRow_Any_NEON_I8MM;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_NEON_I8MM;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    ARGBToUVRow = ARGBToUVRow_Any_SVE2;
    if (IS_ALIGNED(width, 2)) {
      ARGBToUVRow = ARGBToUVRow_SVE2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    ARGBToUVRow = ARGBToUVRow_Any_SME;
    if (IS_ALIGNED(width, 2)) {
      ARGBToUVRow = ARGBToUVRow_SME;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_MSA) && defined(HAS_ARGBTOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToYRow = ARGBToYRow_Any_MSA;
    ARGBToUVRow = ARGBToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_MSA;
    }
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToYRow = ARGBToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LSX) && defined(HAS_ARGBTOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToYRow = ARGBToYRow_Any_LSX;
    ARGBToUVRow = ARGBToUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_LSX;
      ARGBToUVRow = ARGBToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LASX) && defined(HAS_ARGBTOUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToYRow = ARGBToYRow_Any_LASX;
    ARGBToUVRow = ARGBToUVRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_LASX;
      ARGBToUVRow = ARGBToUVRow_LASX;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    ARGBToUVRow(src_argb, src_stride_argb, dst_u, dst_v, width);
    ARGBToYRow(src_argb, dst_y, width);
    ARGBToYRow(src_argb + src_stride_argb, dst_y + dst_stride_y, width);
    src_argb += src_stride_argb * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    ARGBToUVRow(src_argb, 0, dst_u, dst_v, width);
    ARGBToYRow(src_argb, dst_y, width);
  }
  return 0;
}

#ifdef USE_EXTRACTALPHA
// Convert ARGB to I420 with Alpha
// The following version calls ARGBExtractAlpha on the full image.
LIBYUV_API
int ARGBToI420Alpha(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_y,
                    int dst_stride_y,
                    uint8_t* dst_u,
                    int dst_stride_u,
                    uint8_t* dst_v,
                    int dst_stride_v,
                    uint8_t* dst_a,
                    int dst_stride_a,
                    int width,
                    int height) {
  int r = ARGBToI420(src_argb, src_stride_argb, dst_y, dst_stride_y, dst_u,
                     dst_stride_u, dst_v, dst_stride_v, width, height);
  if (r == 0) {
    r = ARGBExtractAlpha(src_argb, src_stride_argb, dst_a, dst_stride_a, width,
                         height);
  }
  return r;
}
#else  // USE_EXTRACTALPHA
// Convert ARGB to I420 with Alpha
LIBYUV_API
int ARGBToI420Alpha(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_y,
                    int dst_stride_y,
                    uint8_t* dst_u,
                    int dst_stride_u,
                    uint8_t* dst_v,
                    int dst_stride_v,
                    uint8_t* dst_a,
                    int dst_stride_a,
                    int width,
                    int height) {
  int y;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  void (*ARGBExtractAlphaRow)(const uint8_t* src_argb, uint8_t* dst_a,
                              int width) = ARGBExtractAlphaRow_C;
  if (!src_argb || !dst_y || !dst_u || !dst_v || !dst_a || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
#if defined(HAS_ARGBTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToYRow = ARGBToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd)) {
    ARGBToYRow = ARGBToYRow_Any_NEON_DotProd;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_NEON_DotProd;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToUVRow = ARGBToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_NEON_I8MM)
  if (TestCpuFlag(kCpuHasNeonI8MM)) {
    ARGBToUVRow = ARGBToUVRow_Any_NEON_I8MM;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_NEON_I8MM;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    ARGBToUVRow = ARGBToUVRow_Any_SVE2;
    if (IS_ALIGNED(width, 2)) {
      ARGBToUVRow = ARGBToUVRow_SVE2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    ARGBToUVRow = ARGBToUVRow_Any_SME;
    if (IS_ALIGNED(width, 2)) {
      ARGBToUVRow = ARGBToUVRow_SME;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_MSA) && defined(HAS_ARGBTOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToYRow = ARGBToYRow_Any_MSA;
    ARGBToUVRow = ARGBToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_MSA;
    }
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToYRow = ARGBToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LASX) && defined(HAS_ARGBTOUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToYRow = ARGBToYRow_Any_LASX;
    ARGBToUVRow = ARGBToUVRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_LASX;
      ARGBToUVRow = ARGBToUVRow_LASX;
    }
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 8) ? ARGBExtractAlphaRow_SSE2
                                               : ARGBExtractAlphaRow_Any_SSE2;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 32) ? ARGBExtractAlphaRow_AVX2
                                                : ARGBExtractAlphaRow_Any_AVX2;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 16) ? ARGBExtractAlphaRow_NEON
                                                : ARGBExtractAlphaRow_Any_NEON;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 16) ? ARGBExtractAlphaRow_MSA
                                                : ARGBExtractAlphaRow_Any_MSA;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 16) ? ARGBExtractAlphaRow_LSX
                                                : ARGBExtractAlphaRow_Any_LSX;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBExtractAlphaRow = ARGBExtractAlphaRow_RVV;
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    ARGBToUVRow(src_argb, src_stride_argb, dst_u, dst_v, width);
    ARGBToYRow(src_argb, dst_y, width);
    ARGBToYRow(src_argb + src_stride_argb, dst_y + dst_stride_y, width);
    ARGBExtractAlphaRow(src_argb, dst_a, width);
    ARGBExtractAlphaRow(src_argb + src_stride_argb, dst_a + dst_stride_a,
                        width);
    src_argb += src_stride_argb * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
    dst_a += dst_stride_a * 2;
  }
  if (height & 1) {
    ARGBToUVRow(src_argb, 0, dst_u, dst_v, width);
    ARGBToYRow(src_argb, dst_y, width);
    ARGBExtractAlphaRow(src_argb, dst_a, width);
  }
  return 0;
}
#endif  // USE_EXTRACTALPHA

// Convert BGRA to I420.
LIBYUV_API
int BGRAToI420(const uint8_t* src_bgra,
               int src_stride_bgra,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*BGRAToUVRow)(const uint8_t* src_bgra0, int src_stride_bgra,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      BGRAToUVRow_C;
  void (*BGRAToYRow)(const uint8_t* src_bgra, uint8_t* dst_y, int width) =
      BGRAToYRow_C;
  if (!src_bgra || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_bgra = src_bgra + (height - 1) * src_stride_bgra;
    src_stride_bgra = -src_stride_bgra;
  }
#if defined(HAS_BGRATOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    BGRAToYRow = BGRAToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      BGRAToYRow = BGRAToYRow_NEON;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd)) {
    BGRAToYRow = BGRAToYRow_Any_NEON_DotProd;
    if (IS_ALIGNED(width, 16)) {
      BGRAToYRow = BGRAToYRow_NEON_DotProd;
    }
  }
#endif
#if defined(HAS_BGRATOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    BGRAToUVRow = BGRAToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      BGRAToUVRow = BGRAToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_BGRATOUVROW_NEON_I8MM)
  if (TestCpuFlag(kCpuHasNeonI8MM)) {
    BGRAToUVRow = BGRAToUVRow_Any_NEON_I8MM;
    if (IS_ALIGNED(width, 16)) {
      BGRAToUVRow = BGRAToUVRow_NEON_I8MM;
    }
  }
#endif
#if defined(HAS_BGRATOUVROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    BGRAToUVRow = BGRAToUVRow_Any_SVE2;
    if (IS_ALIGNED(width, 2)) {
      BGRAToUVRow = BGRAToUVRow_SVE2;
    }
  }
#endif
#if defined(HAS_BGRATOUVROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    BGRAToUVRow = BGRAToUVRow_Any_SME;
    if (IS_ALIGNED(width, 2)) {
      BGRAToUVRow = BGRAToUVRow_SME;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    BGRAToYRow = BGRAToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      BGRAToYRow = BGRAToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_BGRATOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    BGRAToUVRow = BGRAToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      BGRAToUVRow = BGRAToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    BGRAToYRow = BGRAToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      BGRAToYRow = BGRAToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_BGRATOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    BGRAToUVRow = BGRAToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      BGRAToUVRow = BGRAToUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_MSA) && defined(HAS_BGRATOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    BGRAToYRow = BGRAToYRow_Any_MSA;
    BGRAToUVRow = BGRAToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      BGRAToYRow = BGRAToYRow_MSA;
    }
    if (IS_ALIGNED(width, 32)) {
      BGRAToUVRow = BGRAToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_LSX) && defined(HAS_BGRATOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    BGRAToYRow = BGRAToYRow_Any_LSX;
    BGRAToUVRow = BGRAToUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      BGRAToYRow = BGRAToYRow_LSX;
      BGRAToUVRow = BGRAToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    BGRAToYRow = BGRAToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      BGRAToYRow = BGRAToYRow_LASX;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    BGRAToYRow = BGRAToYRow_RVV;
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    BGRAToUVRow(src_bgra, src_stride_bgra, dst_u, dst_v, width);
    BGRAToYRow(src_bgra, dst_y, width);
    BGRAToYRow(src_bgra + src_stride_bgra, dst_y + dst_stride_y, width);
    src_bgra += src_stride_bgra * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    BGRAToUVRow(src_bgra, 0, dst_u, dst_v, width);
    BGRAToYRow(src_bgra, dst_y, width);
  }
  return 0;
}

// Convert ABGR to I420.
LIBYUV_API
int ABGRToI420(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*ABGRToUVRow)(const uint8_t* src_abgr0, int src_stride_abgr,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ABGRToUVRow_C;
  void (*ABGRToYRow)(const uint8_t* src_abgr, uint8_t* dst_y, int width) =
      ABGRToYRow_C;
  if (!src_abgr || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_abgr = src_abgr + (height - 1) * src_stride_abgr;
    src_stride_abgr = -src_stride_abgr;
  }
#if defined(HAS_ABGRTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ABGRToYRow = ABGRToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ABGRToYRow = ABGRToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ABGRTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ABGRToUVRow = ABGRToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ABGRToUVRow = ABGRToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ABGRToYRow = ABGRToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ABGRToYRow = ABGRToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ABGRTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ABGRToUVRow = ABGRToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ABGRToUVRow = ABGRToUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ABGRToYRow = ABGRToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ABGRToYRow = ABGRToYRow_NEON;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd)) {
    ABGRToYRow = ABGRToYRow_Any_NEON_DotProd;
    if (IS_ALIGNED(width, 16)) {
      ABGRToYRow = ABGRToYRow_NEON_DotProd;
    }
  }
#endif
#if defined(HAS_ABGRTOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ABGRToUVRow = ABGRToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ABGRToUVRow = ABGRToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_ABGRTOUVROW_NEON_I8MM)
  if (TestCpuFlag(kCpuHasNeonI8MM)) {
    ABGRToUVRow = ABGRToUVRow_Any_NEON_I8MM;
    if (IS_ALIGNED(width, 16)) {
      ABGRToUVRow = ABGRToUVRow_NEON_I8MM;
    }
  }
#endif
#if defined(HAS_ABGRTOUVROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    ABGRToUVRow = ABGRToUVRow_Any_SVE2;
    if (IS_ALIGNED(width, 2)) {
      ABGRToUVRow = ABGRToUVRow_SVE2;
    }
  }
#endif
#if defined(HAS_ABGRTOUVROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    ABGRToUVRow = ABGRToUVRow_Any_SME;
    if (IS_ALIGNED(width, 2)) {
      ABGRToUVRow = ABGRToUVRow_SME;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_MSA) && defined(HAS_ABGRTOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ABGRToYRow = ABGRToYRow_Any_MSA;
    ABGRToUVRow = ABGRToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ABGRToYRow = ABGRToYRow_MSA;
      ABGRToUVRow = ABGRToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_LSX) && defined(HAS_ABGRTOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ABGRToYRow = ABGRToYRow_Any_LSX;
    ABGRToUVRow = ABGRToUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ABGRToYRow = ABGRToYRow_LSX;
      ABGRToUVRow = ABGRToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ABGRToYRow = ABGRToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ABGRToYRow = ABGRToYRow_LASX;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ABGRToYRow = ABGRToYRow_RVV;
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    ABGRToUVRow(src_abgr, src_stride_abgr, dst_u, dst_v, width);
    ABGRToYRow(src_abgr, dst_y, width);
    ABGRToYRow(src_abgr + src_stride_abgr, dst_y + dst_stride_y, width);
    src_abgr += src_stride_abgr * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    ABGRToUVRow(src_abgr, 0, dst_u, dst_v, width);
    ABGRToYRow(src_abgr, dst_y, width);
  }
  return 0;
}

// Convert RGBA to I420.
LIBYUV_API
int RGBAToI420(const uint8_t* src_rgba,
               int src_stride_rgba,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*RGBAToUVRow)(const uint8_t* src_rgba0, int src_stride_rgba,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      RGBAToUVRow_C;
  void (*RGBAToYRow)(const uint8_t* src_rgba, uint8_t* dst_y, int width) =
      RGBAToYRow_C;
  if (!src_rgba || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgba = src_rgba + (height - 1) * src_stride_rgba;
    src_stride_rgba = -src_stride_rgba;
  }
#if defined(HAS_RGBATOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGBAToYRow = RGBAToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGBAToYRow = RGBAToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RGBATOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGBAToUVRow = RGBAToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGBAToUVRow = RGBAToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RGBATOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGBAToYRow = RGBAToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGBAToYRow = RGBAToYRow_NEON;
    }
  }
#endif
#if defined(HAS_RGBATOYROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd)) {
    RGBAToYRow = RGBAToYRow_Any_NEON_DotProd;
    if (IS_ALIGNED(width, 16)) {
      RGBAToYRow = RGBAToYRow_NEON_DotProd;
    }
  }
#endif
#if defined(HAS_RGBATOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGBAToUVRow = RGBAToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGBAToUVRow = RGBAToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_RGBATOUVROW_NEON_I8MM)
  if (TestCpuFlag(kCpuHasNeonI8MM)) {
    RGBAToUVRow = RGBAToUVRow_Any_NEON_I8MM;
    if (IS_ALIGNED(width, 16)) {
      RGBAToUVRow = RGBAToUVRow_NEON_I8MM;
    }
  }
#endif
#if defined(HAS_RGBATOUVROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    RGBAToUVRow = RGBAToUVRow_Any_SVE2;
    if (IS_ALIGNED(width, 2)) {
      RGBAToUVRow = RGBAToUVRow_SVE2;
    }
  }
#endif
#if defined(HAS_RGBATOUVROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    RGBAToUVRow = RGBAToUVRow_Any_SME;
    if (IS_ALIGNED(width, 2)) {
      RGBAToUVRow = RGBAToUVRow_SME;
    }
  }
#endif
#if defined(HAS_RGBATOYROW_MSA) && defined(HAS_RGBATOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGBAToYRow = RGBAToYRow_Any_MSA;
    RGBAToUVRow = RGBAToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGBAToYRow = RGBAToYRow_MSA;
      RGBAToUVRow = RGBAToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_RGBATOYROW_LSX) && defined(HAS_RGBATOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RGBAToYRow = RGBAToYRow_Any_LSX;
    RGBAToUVRow = RGBAToUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RGBAToYRow = RGBAToYRow_LSX;
      RGBAToUVRow = RGBAToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_RGBATOYROW_LASX)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGBAToYRow = RGBAToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RGBAToYRow = RGBAToYRow_LASX;
    }
  }
#endif
#if defined(HAS_RGBATOYROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RGBAToYRow = RGBAToYRow_RVV;
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    RGBAToUVRow(src_rgba, src_stride_rgba, dst_u, dst_v, width);
    RGBAToYRow(src_rgba, dst_y, width);
    RGBAToYRow(src_rgba + src_stride_rgba, dst_y + dst_stride_y, width);
    src_rgba += src_stride_rgba * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    RGBAToUVRow(src_rgba, 0, dst_u, dst_v, width);
    RGBAToYRow(src_rgba, dst_y, width);
  }
  return 0;
}

// Enabled if 1 pass is available
#if (defined(HAS_RGB24TOYROW_NEON) || defined(HAS_RGB24TOYROW_MSA) || \
     defined(HAS_RGB24TOYROW_LSX) || defined(HAS_RGB24TOYROW_RVV))
#define HAS_RGB24TOYROW
#endif

// Convert RGB24 to I420.
LIBYUV_API
int RGB24ToI420(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_y,
                int dst_stride_y,
                uint8_t* dst_u,
                int dst_stride_u,
                uint8_t* dst_v,
                int dst_stride_v,
                int width,
                int height) {
  int y;
#if defined(HAS_RGB24TOYROW)
  void (*RGB24ToUVRow)(const uint8_t* src_rgb24, int src_stride_rgb24,
                       uint8_t* dst_u, uint8_t* dst_v, int width) =
      RGB24ToUVRow_C;
  void (*RGB24ToYRow)(const uint8_t* src_rgb24, uint8_t* dst_y, int width) =
      RGB24ToYRow_C;
#else
  void (*RGB24ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RGB24ToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_rgb24 || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb24 = src_rgb24 + (height - 1) * src_stride_rgb24;
    src_stride_rgb24 = -src_stride_rgb24;
  }

#if defined(HAS_RGB24TOYROW)

// Neon version does direct RGB24 to YUV.
#if defined(HAS_RGB24TOYROW_NEON) && defined(HAS_RGB24TOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB24ToUVRow = RGB24ToUVRow_Any_NEON;
    RGB24ToYRow = RGB24ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYRow = RGB24ToYRow_NEON;
      RGB24ToUVRow = RGB24ToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_RGB24TOYROW_MSA) && defined(HAS_RGB24TOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGB24ToUVRow = RGB24ToUVRow_Any_MSA;
    RGB24ToYRow = RGB24ToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYRow = RGB24ToYRow_MSA;
      RGB24ToUVRow = RGB24ToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_RGB24TOYROW_LSX) && defined(HAS_RGB24TOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RGB24ToUVRow = RGB24ToUVRow_Any_LSX;
    RGB24ToYRow = RGB24ToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYRow = RGB24ToYRow_LSX;
      RGB24ToUVRow = RGB24ToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_RGB24TOYROW_LASX) && defined(HAS_RGB24TOUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RGB24ToUVRow = RGB24ToUVRow_Any_LASX;
    RGB24ToYRow = RGB24ToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RGB24ToYRow = RGB24ToYRow_LASX;
      RGB24ToUVRow = RGB24ToUVRow_LASX;
    }
  }
#endif
#if defined(HAS_RGB24TOYROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RGB24ToYRow = RGB24ToYRow_RVV;
  }
#endif

// Other platforms do intermediate conversion from RGB24 to ARGB.
#else  // HAS_RGB24TOYROW

#if defined(HAS_RGB24TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToARGBRow = RGB24ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
    }
  }
#endif
#endif  // HAS_RGB24TOYROW

  {
#if !defined(HAS_RGB24TOYROW)
    // Allocate 2 rows of ARGB.
    const int row_size = (width * 4 + 31) & ~31;
    align_buffer_64(row, row_size * 2);
    if (!row)
      return 1;
#endif

    for (y = 0; y < height - 1; y += 2) {
#if defined(HAS_RGB24TOYROW)
      RGB24ToUVRow(src_rgb24, src_stride_rgb24, dst_u, dst_v, width);
      RGB24ToYRow(src_rgb24, dst_y, width);
      RGB24ToYRow(src_rgb24 + src_stride_rgb24, dst_y + dst_stride_y, width);
#else
      RGB24ToARGBRow(src_rgb24, row, width);
      RGB24ToARGBRow(src_rgb24 + src_stride_rgb24, row + row_size, width);
      ARGBToUVRow(row, row_size, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + row_size, dst_y + dst_stride_y, width);
#endif
      src_rgb24 += src_stride_rgb24 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if defined(HAS_RGB24TOYROW)
      RGB24ToUVRow(src_rgb24, 0, dst_u, dst_v, width);
      RGB24ToYRow(src_rgb24, dst_y, width);
#else
      RGB24ToARGBRow(src_rgb24, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !defined(HAS_RGB24TOYROW)
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}
#undef HAS_RGB24TOYROW

// Enabled if 1 pass is available
#if defined(HAS_RGB24TOYJROW_NEON) || defined(HAS_RGB24TOYJROW_MSA) || \
    defined(HAS_RGB24TOYJROW_RVV)
#define HAS_RGB24TOYJROW
#endif

// Convert RGB24 to J420.
LIBYUV_API
int RGB24ToJ420(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_y,
                int dst_stride_y,
                uint8_t* dst_u,
                int dst_stride_u,
                uint8_t* dst_v,
                int dst_stride_v,
                int width,
                int height) {
  int y;
#if defined(HAS_RGB24TOYJROW)
  void (*RGB24ToUVJRow)(const uint8_t* src_rgb24, int src_stride_rgb24,
                        uint8_t* dst_u, uint8_t* dst_v, int width) =
      RGB24ToUVJRow_C;
  void (*RGB24ToYJRow)(const uint8_t* src_rgb24, uint8_t* dst_y, int width) =
      RGB24ToYJRow_C;
#else
  void (*RGB24ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RGB24ToARGBRow_C;
  void (*ARGBToUVJRow)(const uint8_t* src_argb0, int src_stride_argb,
                       uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVJRow_C;
  void (*ARGBToYJRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYJRow_C;
#endif
  if (!src_rgb24 || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb24 = src_rgb24 + (height - 1) * src_stride_rgb24;
    src_stride_rgb24 = -src_stride_rgb24;
  }

#if defined(HAS_RGB24TOYJROW)

// Neon version does direct RGB24 to YUV.
#if defined(HAS_RGB24TOYJROW_NEON) && defined(HAS_RGB24TOUVJROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB24ToUVJRow = RGB24ToUVJRow_Any_NEON;
    RGB24ToYJRow = RGB24ToYJRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYJRow = RGB24ToYJRow_NEON;
      RGB24ToUVJRow = RGB24ToUVJRow_NEON;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_MSA) && defined(HAS_RGB24TOUVJROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGB24ToUVJRow = RGB24ToUVJRow_Any_MSA;
    RGB24ToYJRow = RGB24ToYJRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYJRow = RGB24ToYJRow_MSA;
      RGB24ToUVJRow = RGB24ToUVJRow_MSA;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RGB24ToYJRow = RGB24ToYJRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYJRow = RGB24ToYJRow_LSX;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RGB24ToYJRow = RGB24ToYJRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RGB24ToYJRow = RGB24ToYJRow_LASX;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RGB24ToYJRow = RGB24ToYJRow_RVV;
  }
#endif

// Other platforms do intermediate conversion from RGB24 to ARGB.
#else  // HAS_RGB24TOYJROW

#if defined(HAS_RGB24TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToARGBRow = RGB24ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYJRow = ARGBToYJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYJRow = ARGBToYJRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYJRow = ARGBToYJRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJRow = ARGBToUVJRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVJRow = ARGBToUVJRow_AVX2;
    }
  }
#endif
#endif  // HAS_RGB24TOYJROW

  {
#if !defined(HAS_RGB24TOYJROW)
    // Allocate 2 rows of ARGB.
    const int row_size = (width * 4 + 31) & ~31;
    align_buffer_64(row, row_size * 2);
    if (!row)
      return 1;
#endif

    for (y = 0; y < height - 1; y += 2) {
#if defined(HAS_RGB24TOYJROW)
      RGB24ToUVJRow(src_rgb24, src_stride_rgb24, dst_u, dst_v, width);
      RGB24ToYJRow(src_rgb24, dst_y, width);
      RGB24ToYJRow(src_rgb24 + src_stride_rgb24, dst_y + dst_stride_y, width);
#else
      RGB24ToARGBRow(src_rgb24, row, width);
      RGB24ToARGBRow(src_rgb24 + src_stride_rgb24, row + row_size, width);
      ARGBToUVJRow(row, row_size, dst_u, dst_v, width);
      ARGBToYJRow(row, dst_y, width);
      ARGBToYJRow(row + row_size, dst_y + dst_stride_y, width);
#endif
      src_rgb24 += src_stride_rgb24 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if defined(HAS_RGB24TOYJROW)
      RGB24ToUVJRow(src_rgb24, 0, dst_u, dst_v, width);
      RGB24ToYJRow(src_rgb24, dst_y, width);
#else
      RGB24ToARGBRow(src_rgb24, row, width);
      ARGBToUVJRow(row, 0, dst_u, dst_v, width);
      ARGBToYJRow(row, dst_y, width);
#endif
    }
#if !defined(HAS_RGB24TOYJROW)
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}
#undef HAS_RGB24TOYJROW

// Enabled if 1 pass is available
#if (defined(HAS_RAWTOYROW_NEON) || defined(HAS_RAWTOYROW_MSA) || \
     defined(HAS_RAWTOYROW_LSX) || defined(HAS_RAWTOYROW_RVV))
#define HAS_RAWTOYROW
#endif

// Convert RAW to I420.
LIBYUV_API
int RAWToI420(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height) {
  int y;
#if defined(HAS_RAWTOYROW)
  void (*RAWToUVRow)(const uint8_t* src_raw, int src_stride_raw, uint8_t* dst_u,
                     uint8_t* dst_v, int width) = RAWToUVRow_C;
  void (*RAWToYRow)(const uint8_t* src_raw, uint8_t* dst_y, int width) =
      RAWToYRow_C;
#else
  void (*RAWToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RAWToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_raw || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }

#if defined(HAS_RAWTOYROW)

// Neon version does direct RAW to YUV.
#if defined(HAS_RAWTOYROW_NEON) && defined(HAS_RAWTOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToUVRow = RAWToUVRow_Any_NEON;
    RAWToYRow = RAWToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RAWToYRow = RAWToYRow_NEON;
      RAWToUVRow = RAWToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_RAWTOYROW_MSA) && defined(HAS_RAWTOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RAWToUVRow = RAWToUVRow_Any_MSA;
    RAWToYRow = RAWToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RAWToYRow = RAWToYRow_MSA;
      RAWToUVRow = RAWToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_RAWTOYROW_LSX) && defined(HAS_RAWTOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RAWToUVRow = RAWToUVRow_Any_LSX;
    RAWToYRow = RAWToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RAWToYRow = RAWToYRow_LSX;
      RAWToUVRow = RAWToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_RAWTOYROW_LASX) && defined(HAS_RAWTOUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RAWToUVRow = RAWToUVRow_Any_LASX;
    RAWToYRow = RAWToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RAWToYRow = RAWToYRow_LASX;
      RAWToUVRow = RAWToUVRow_LASX;
    }
  }
#endif
#if defined(HAS_RAWTOYROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RAWToYRow = RAWToYRow_RVV;
  }
#endif

// Other platforms do intermediate conversion from RAW to ARGB.
#else  // HAS_RAWTOYROW

#if defined(HAS_RAWTOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToARGBRow = RAWToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RAWToARGBRow = RAWToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      RAWToARGBRow = RAWToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
    }
  }
#endif
#endif  // HAS_RAWTOYROW

  {
#if !defined(HAS_RAWTOYROW)
    // Allocate 2 rows of ARGB.
    const int row_size = (width * 4 + 31) & ~31;
    align_buffer_64(row, row_size * 2);
    if (!row)
      return 1;
#endif

    for (y = 0; y < height - 1; y += 2) {
#if defined(HAS_RAWTOYROW)
      RAWToUVRow(src_raw, src_stride_raw, dst_u, dst_v, width);
      RAWToYRow(src_raw, dst_y, width);
      RAWToYRow(src_raw + src_stride_raw, dst_y + dst_stride_y, width);
#else
      RAWToARGBRow(src_raw, row, width);
      RAWToARGBRow(src_raw + src_stride_raw, row + row_size, width);
      ARGBToUVRow(row, row_size, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + row_size, dst_y + dst_stride_y, width);
#endif
      src_raw += src_stride_raw * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if defined(HAS_RAWTOYROW)
      RAWToUVRow(src_raw, 0, dst_u, dst_v, width);
      RAWToYRow(src_raw, dst_y, width);
#else
      RAWToARGBRow(src_raw, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !defined(HAS_RAWTOYROW)
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}
#undef HAS_RAWTOYROW

// Enabled if 1 pass is available
#if defined(HAS_RAWTOYJROW_NEON) || defined(HAS_RAWTOYJROW_MSA) || \
    defined(HAS_RAWTOYJROW_RVV)
#define HAS_RAWTOYJROW
#endif

// Convert RAW to J420.
LIBYUV_API
int RAWToJ420(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height) {
  int y;
#if defined(HAS_RAWTOYJROW)
  void (*RAWToUVJRow)(const uint8_t* src_raw, int src_stride_raw,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      RAWToUVJRow_C;
  void (*RAWToYJRow)(const uint8_t* src_raw, uint8_t* dst_y, int width) =
      RAWToYJRow_C;
#else
  void (*RAWToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RAWToARGBRow_C;
  void (*ARGBToUVJRow)(const uint8_t* src_argb0, int src_stride_argb,
                       uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVJRow_C;
  void (*ARGBToYJRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYJRow_C;
#endif
  if (!src_raw || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }

#if defined(HAS_RAWTOYJROW)

// Neon version does direct RAW to YUV.
#if defined(HAS_RAWTOYJROW_NEON) && defined(HAS_RAWTOUVJROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToUVJRow = RAWToUVJRow_Any_NEON;
    RAWToYJRow = RAWToYJRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RAWToYJRow = RAWToYJRow_NEON;
      RAWToUVJRow = RAWToUVJRow_NEON;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_MSA) && defined(HAS_RAWTOUVJROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RAWToUVJRow = RAWToUVJRow_Any_MSA;
    RAWToYJRow = RAWToYJRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RAWToYJRow = RAWToYJRow_MSA;
      RAWToUVJRow = RAWToUVJRow_MSA;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_LSX) && defined(HAS_RAWTOUVJROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RAWToUVJRow = RAWToUVJRow_Any_LSX;
    RAWToYJRow = RAWToYJRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RAWToYJRow = RAWToYJRow_LSX;
      RAWToUVJRow = RAWToUVJRow_LSX;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_LASX) && defined(HAS_RAWTOUVJROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RAWToUVJRow = RAWToUVJRow_Any_LASX;
    RAWToYJRow = RAWToYJRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RAWToYJRow = RAWToYJRow_LASX;
      RAWToUVJRow = RAWToUVJRow_LASX;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RAWToYJRow = RAWToYJRow_RVV;
  }
#endif

// Other platforms do intermediate conversion from RAW to ARGB.
#else  // HAS_RAWTOYJROW

#if defined(HAS_RAWTOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToARGBRow = RAWToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RAWToARGBRow = RAWToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      RAWToARGBRow = RAWToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYJRow = ARGBToYJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYJRow = ARGBToYJRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYJRow = ARGBToYJRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJRow = ARGBToUVJRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVJRow = ARGBToUVJRow_AVX2;
    }
  }
#endif
#endif  // HAS_RAWTOYJROW

  {
#if !defined(HAS_RAWTOYJROW)
    // Allocate 2 rows of ARGB.
    const int row_size = (width * 4 + 31) & ~31;
    align_buffer_64(row, row_size * 2);
    if (!row)
      return 1;
#endif

    for (y = 0; y < height - 1; y += 2) {
#if defined(HAS_RAWTOYJROW)
      RAWToUVJRow(src_raw, src_stride_raw, dst_u, dst_v, width);
      RAWToYJRow(src_raw, dst_y, width);
      RAWToYJRow(src_raw + src_stride_raw, dst_y + dst_stride_y, width);
#else
      RAWToARGBRow(src_raw, row, width);
      RAWToARGBRow(src_raw + src_stride_raw, row + row_size, width);
      ARGBToUVJRow(row, row_size, dst_u, dst_v, width);
      ARGBToYJRow(row, dst_y, width);
      ARGBToYJRow(row + row_size, dst_y + dst_stride_y, width);
#endif
      src_raw += src_stride_raw * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if defined(HAS_RAWTOYJROW)
      RAWToUVJRow(src_raw, 0, dst_u, dst_v, width);
      RAWToYJRow(src_raw, dst_y, width);
#else
      RAWToARGBRow(src_raw, row, width);
      ARGBToUVJRow(row, 0, dst_u, dst_v, width);
      ARGBToYJRow(row, dst_y, width);
#endif
    }
#if !defined(HAS_RAWTOYJROW)
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}
#undef HAS_RAWTOYJROW

// RAW big endian (rgb in memory) to I444
// 2 step conversion of RAWToARGB then ARGBToY and ARGBToUV444
LIBYUV_API
int RAWToI444(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height) {
  int y;
  void (*RAWToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RAWToARGBRow_C;
  void (*ARGBToYRow)(const uint8_t* src_raw, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  void (*ARGBToUV444Row)(const uint8_t* src_raw, uint8_t* dst_u, uint8_t* dst_v,
                         int width) = ARGBToUV444Row_C;
  if (!src_raw || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }
  // TODO: add row coalesce when main loop handles large width in blocks
  // TODO: implement UV444 or trim the ifdef below
#if defined(HAS_ARGBTOUV444ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUV444Row = ARGBToUV444Row_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUV444ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUV444Row = ARGBToUV444Row_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUV444ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToUV444Row = ARGBToUV444Row_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOUV444ROW_NEON_I8MM)
  if (TestCpuFlag(kCpuHasNeonI8MM)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_NEON_I8MM;
    if (IS_ALIGNED(width, 8)) {
      ARGBToUV444Row = ARGBToUV444Row_NEON_I8MM;
    }
  }
#endif
#if defined(HAS_ARGBTOUV444ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUV444Row = ARGBToUV444Row_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOUV444ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUV444Row = ARGBToUV444Row_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOUV444ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUV444Row = ARGBToUV444Row_LASX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToYRow = ARGBToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd)) {
    ARGBToYRow = ARGBToYRow_Any_NEON_DotProd;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_NEON_DotProd;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToYRow = ARGBToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToYRow = ARGBToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToYRow = ARGBToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_LASX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBToYRow = ARGBToYRow_RVV;
  }
#endif

#if defined(HAS_RAWTOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToARGBRow = RAWToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RAWToARGBRow = RAWToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      RAWToARGBRow = RAWToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToARGBRow = RAWToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RAWToARGBRow = RAWToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    RAWToARGBRow = RAWToARGBRow_SVE2;
  }
#endif
#if defined(HAS_RAWTOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RAWToARGBRow = RAWToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_MSA;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RAWToARGBRow = RAWToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RAWToARGBRow = RAWToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RAWToARGBRow = RAWToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RAWToARGBRow = RAWToARGBRow_RVV;
  }
#endif

  {
    // Allocate a row of ARGB.
    const int row_size = width * 4;
    align_buffer_64(row, row_size);
    if (!row)
      return 1;

    for (y = 0; y < height; ++y) {
      RAWToARGBRow(src_raw, row, width);
      ARGBToUV444Row(row, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      src_raw += src_stride_raw;
      dst_y += dst_stride_y;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    free_aligned_buffer_64(row);
  }
  return 0;
}

// RAW big endian (rgb in memory) to J444
// 2 step conversion of RAWToARGB then ARGBToYJ and ARGBToUVJ444
LIBYUV_API
int RAWToJ444(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height) {
  int y;
  void (*RAWToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RAWToARGBRow_C;
  void (*ARGBToYJRow)(const uint8_t* src_raw, uint8_t* dst_y, int width) =
      ARGBToYJRow_C;
  void (*ARGBToUVJ444Row)(const uint8_t* src_raw, uint8_t* dst_u,
                          uint8_t* dst_v, int width) = ARGBToUVJ444Row_C;
  if (!src_raw || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }
  // TODO: add row coalesce when main loop handles large width in blocks
#if defined(HAS_ARGBTOUVJ444ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVJ444Row = ARGBToUVJ444Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJ444Row = ARGBToUVJ444Row_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJ444ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVJ444Row = ARGBToUVJ444Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVJ444Row = ARGBToUVJ444Row_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJ444ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToUVJ444Row = ARGBToUVJ444Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToUVJ444Row = ARGBToUVJ444Row_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJ444ROW_NEON_I8MM)
  if (TestCpuFlag(kCpuHasNeonI8MM)) {
    ARGBToUVJ444Row = ARGBToUVJ444Row_Any_NEON_I8MM;
    if (IS_ALIGNED(width, 8)) {
      ARGBToUVJ444Row = ARGBToUVJ444Row_NEON_I8MM;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJ444ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToUVJ444Row = ARGBToUVJ444Row_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJ444Row = ARGBToUVJ444Row_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJ444ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToUVJ444Row = ARGBToUVJ444Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJ444Row = ARGBToUVJ444Row_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOUVJ444ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToUVJ444Row = ARGBToUVJ444Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVJ444Row = ARGBToUVJ444Row_LASX;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYJRow = ARGBToYJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYJRow = ARGBToYJRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYJRow = ARGBToYJRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToYJRow = ARGBToYJRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd)) {
    ARGBToYJRow = ARGBToYJRow_Any_NEON_DotProd;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_NEON_DotProd;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToYJRow = ARGBToYJRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToYJRow = ARGBToYJRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToYJRow = ARGBToYJRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYJRow = ARGBToYJRow_LASX;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBToYJRow = ARGBToYJRow_RVV;
  }
#endif

#if defined(HAS_RAWTOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToARGBRow = RAWToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RAWToARGBRow = RAWToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      RAWToARGBRow = RAWToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToARGBRow = RAWToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RAWToARGBRow = RAWToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    RAWToARGBRow = RAWToARGBRow_SVE2;
  }
#endif
#if defined(HAS_RAWTOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RAWToARGBRow = RAWToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_MSA;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RAWToARGBRow = RAWToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RAWToARGBRow = RAWToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RAWToARGBRow = RAWToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RAWToARGBRow = RAWToARGBRow_RVV;
  }
#endif

  {
    // Allocate a row of ARGB.
    const int row_size = width * 4;
    align_buffer_64(row, row_size);
    if (!row)
      return 1;

    for (y = 0; y < height; ++y) {
      RAWToARGBRow(src_raw, row, width);
      ARGBToUVJ444Row(row, dst_u, dst_v, width);
      ARGBToYJRow(row, dst_y, width);
      src_raw += src_stride_raw;
      dst_y += dst_stride_y;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    free_aligned_buffer_64(row);
  }
  return 0;
}

// Convert RGB565 to I420.
LIBYUV_API
int RGB565ToI420(const uint8_t* src_rgb565,
                 int src_stride_rgb565,
                 uint8_t* dst_y,
                 int dst_stride_y,
                 uint8_t* dst_u,
                 int dst_stride_u,
                 uint8_t* dst_v,
                 int dst_stride_v,
                 int width,
                 int height) {
  int y;
#if (defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA) || \
     defined(HAS_RGB565TOYROW_LSX) || defined(HAS_RGB565TOYROW_LASX))
  void (*RGB565ToUVRow)(const uint8_t* src_rgb565, int src_stride_rgb565,
                        uint8_t* dst_u, uint8_t* dst_v, int width) =
      RGB565ToUVRow_C;
  void (*RGB565ToYRow)(const uint8_t* src_rgb565, uint8_t* dst_y, int width) =
      RGB565ToYRow_C;
#else
  void (*RGB565ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb,
                          int width) = RGB565ToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_rgb565 || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb565 = src_rgb565 + (height - 1) * src_stride_rgb565;
    src_stride_rgb565 = -src_stride_rgb565;
  }

// Neon version does direct RGB565 to YUV.
#if defined(HAS_RGB565TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB565ToYRow = RGB565ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToYRow = RGB565ToYRow_NEON;
    }
  }
#endif
// Neon version does direct RGB565 to YUV.
#if defined(HAS_RGB565TOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB565ToUVRow = RGB565ToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToUVRow = RGB565ToUVRow_NEON;
    }
  }
#endif
// MSA version does direct RGB565 to YUV.
#if defined(HAS_RGB565TOYROW_MSA) && defined(HAS_RGB565TOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGB565ToUVRow = RGB565ToUVRow_Any_MSA;
    RGB565ToYRow = RGB565ToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToYRow = RGB565ToYRow_MSA;
      RGB565ToUVRow = RGB565ToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_RGB565TOYROW_LSX) && defined(HAS_RGB565TOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RGB565ToUVRow = RGB565ToUVRow_Any_LSX;
    RGB565ToYRow = RGB565ToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToYRow = RGB565ToYRow_LSX;
      RGB565ToUVRow = RGB565ToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_RGB565TOYROW_LASX) && defined(HAS_RGB565TOUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RGB565ToUVRow = RGB565ToUVRow_Any_LASX;
    RGB565ToYRow = RGB565ToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RGB565ToYRow = RGB565ToYRow_LASX;
      RGB565ToUVRow = RGB565ToUVRow_LASX;
    }
  }
#endif
// Other platforms do intermediate conversion from RGB565 to ARGB.
#if defined(HAS_RGB565TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      RGB565ToARGBRow = RGB565ToARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_RGB565TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToARGBRow = RGB565ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
    }
  }
#endif
  {
#if !(defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA) || \
      defined(HAS_RGB565TOYROW_LSX) || defined(HAS_RGB565TOYROW_LASX))
    // Allocate 2 rows of ARGB.
    const int row_size = (width * 4 + 31) & ~31;
    align_buffer_64(row, row_size * 2);
    if (!row)
      return 1;
#endif
    for (y = 0; y < height - 1; y += 2) {
#if (defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA) || \
     defined(HAS_RGB565TOYROW_LSX) || defined(HAS_RGB565TOYROW_LASX))
      RGB565ToUVRow(src_rgb565, src_stride_rgb565, dst_u, dst_v, width);
      RGB565ToYRow(src_rgb565, dst_y, width);
      RGB565ToYRow(src_rgb565 + src_stride_rgb565, dst_y + dst_stride_y, width);
#else
      RGB565ToARGBRow(src_rgb565, row, width);
      RGB565ToARGBRow(src_rgb565 + src_stride_rgb565, row + row_size, width);
      ARGBToUVRow(row, row_size, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + row_size, dst_y + dst_stride_y, width);
#endif
      src_rgb565 += src_stride_rgb565 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if (defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA) || \
     defined(HAS_RGB565TOYROW_LSX) || defined(HAS_RGB565TOYROW_LASX))
      RGB565ToUVRow(src_rgb565, 0, dst_u, dst_v, width);
      RGB565ToYRow(src_rgb565, dst_y, width);
#else
      RGB565ToARGBRow(src_rgb565, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !(defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA) || \
      defined(HAS_RGB565TOYROW_LSX) || defined(HAS_RGB565TOYROW_LASX))
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}

// Convert ARGB1555 to I420.
LIBYUV_API
int ARGB1555ToI420(const uint8_t* src_argb1555,
                   int src_stride_argb1555,
                   uint8_t* dst_y,
                   int dst_stride_y,
                   uint8_t* dst_u,
                   int dst_stride_u,
                   uint8_t* dst_v,
                   int dst_stride_v,
                   int width,
                   int height) {
  int y;
#if (defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA) || \
     defined(HAS_ARGB1555TOYROW_LSX) || defined(HAS_ARGB1555TOYROW_LASX))
  void (*ARGB1555ToUVRow)(const uint8_t* src_argb1555, int src_stride_argb1555,
                          uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGB1555ToUVRow_C;
  void (*ARGB1555ToYRow)(const uint8_t* src_argb1555, uint8_t* dst_y,
                         int width) = ARGB1555ToYRow_C;
#else
  void (*ARGB1555ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb,
                            int width) = ARGB1555ToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_argb1555 || !dst_y || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb1555 = src_argb1555 + (height - 1) * src_stride_argb1555;
    src_stride_argb1555 = -src_stride_argb1555;
  }

// Neon version does direct ARGB1555 to YUV.
#if defined(HAS_ARGB1555TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGB1555ToYRow = ARGB1555ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGB1555ToYRow = ARGB1555ToYRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGB1555TOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGB1555ToUVRow = ARGB1555ToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToUVRow = ARGB1555ToUVRow_NEON;
    }
  }
#endif

// MSA version does direct ARGB1555 to YUV.
#if defined(HAS_ARGB1555TOYROW_MSA) && defined(HAS_ARGB1555TOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGB1555ToUVRow = ARGB1555ToUVRow_Any_MSA;
    ARGB1555ToYRow = ARGB1555ToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToYRow = ARGB1555ToYRow_MSA;
      ARGB1555ToUVRow = ARGB1555ToUVRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGB1555TOYROW_LSX) && defined(HAS_ARGB1555TOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGB1555ToUVRow = ARGB1555ToUVRow_Any_LSX;
    ARGB1555ToYRow = ARGB1555ToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToYRow = ARGB1555ToYRow_LSX;
      ARGB1555ToUVRow = ARGB1555ToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGB1555TOYROW_LASX) && defined(HAS_ARGB1555TOUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGB1555ToUVRow = ARGB1555ToUVRow_Any_LASX;
    ARGB1555ToYRow = ARGB1555ToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGB1555ToYRow = ARGB1555ToYRow_LASX;
      ARGB1555ToUVRow = ARGB1555ToUVRow_LASX;
    }
  }
#endif

// Other platforms do intermediate conversion from ARGB1555 to ARGB.
#if defined(HAS_ARGB1555TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGB1555TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
    }
  }
#endif
  {
#if !(defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA) || \
      defined(HAS_ARGB1555TOYROW_LSX) || defined(HAS_ARGB1555TOYROW_LASX))
    // Allocate 2 rows of ARGB.
    const int row_size = (width * 4 + 31) & ~31;
    align_buffer_64(row, row_size * 2);
    if (!row)
      return 1;
#endif

    for (y = 0; y < height - 1; y += 2) {
#if (defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA) || \
     defined(HAS_ARGB1555TOYROW_LSX) || defined(HAS_ARGB1555TOYROW_LASX))
      ARGB1555ToUVRow(src_argb1555, src_stride_argb1555, dst_u, dst_v, width);
      ARGB1555ToYRow(src_argb1555, dst_y, width);
      ARGB1555ToYRow(src_argb1555 + src_stride_argb1555, dst_y + dst_stride_y,
                     width);
#else
      ARGB1555ToARGBRow(src_argb1555, row, width);
      ARGB1555ToARGBRow(src_argb1555 + src_stride_argb1555, row + row_size,
                        width);
      ARGBToUVRow(row, row_size, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + row_size, dst_y + dst_stride_y, width);
#endif
      src_argb1555 += src_stride_argb1555 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if (defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA) || \
     defined(HAS_ARGB1555TOYROW_LSX) || defined(HAS_ARGB1555TOYROW_LASX))
      ARGB1555ToUVRow(src_argb1555, 0, dst_u, dst_v, width);
      ARGB1555ToYRow(src_argb1555, dst_y, width);
#else
      ARGB1555ToARGBRow(src_argb1555, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !(defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA) || \
      defined(HAS_ARGB1555TOYROW_LSX) || defined(HAS_ARGB1555TOYROW_LASX))
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}

// Convert ARGB4444 to I420.
LIBYUV_API
int ARGB4444ToI420(const uint8_t* src_argb4444,
                   int src_stride_argb4444,
                   uint8_t* dst_y,
                   int dst_stride_y,
                   uint8_t* dst_u,
                   int dst_stride_u,
                   uint8_t* dst_v,
                   int dst_stride_v,
                   int width,
                   int height) {
  int y;
#if defined(HAS_ARGB4444TOYROW_NEON)
  void (*ARGB4444ToUVRow)(const uint8_t* src_argb4444, int src_stride_argb4444,
                          uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGB4444ToUVRow_C;
  void (*ARGB4444ToYRow)(const uint8_t* src_argb4444, uint8_t* dst_y,
                         int width) = ARGB4444ToYRow_C;
#else
  void (*ARGB4444ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb,
                            int width) = ARGB4444ToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_argb4444 || !dst_y || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb4444 = src_argb4444 + (height - 1) * src_stride_argb4444;
    src_stride_argb4444 = -src_stride_argb4444;
  }

// Neon version does direct ARGB4444 to YUV.
#if defined(HAS_ARGB4444TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGB4444ToYRow = ARGB4444ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGB4444ToYRow = ARGB4444ToYRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGB4444TOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGB4444ToUVRow = ARGB4444ToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGB4444ToUVRow = ARGB4444ToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGB4444TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGB4444TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGB4444TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGB4444TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGB4444TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_MSA) && defined(HAS_ARGBTOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToUVRow = ARGBToUVRow_Any_MSA;
    ARGBToYRow = ARGBToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_MSA;
      if (IS_ALIGNED(width, 32)) {
        ARGBToUVRow = ARGBToUVRow_MSA;
      }
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToYRow = ARGBToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LSX) && defined(HAS_ARGBTOUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToYRow = ARGBToYRow_Any_LSX;
    ARGBToUVRow = ARGBToUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_LSX;
      ARGBToUVRow = ARGBToUVRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_LASX) && defined(HAS_ARGBTOUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToYRow = ARGBToYRow_Any_LASX;
    ARGBToUVRow = ARGBToUVRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYRow = ARGBToYRow_LASX;
      ARGBToUVRow = ARGBToUVRow_LASX;
    }
  }
#endif

  {
#if !(defined(HAS_ARGB4444TOYROW_NEON))
    // Allocate 2 rows of ARGB.
    const int row_size = (width * 4 + 31) & ~31;
    align_buffer_64(row, row_size * 2);
    if (!row)
      return 1;
#endif

    for (y = 0; y < height - 1; y += 2) {
#if defined(HAS_ARGB4444TOYROW_NEON)
      ARGB4444ToUVRow(src_argb4444, src_stride_argb4444, dst_u, dst_v, width);
      ARGB4444ToYRow(src_argb4444, dst_y, width);
      ARGB4444ToYRow(src_argb4444 + src_stride_argb4444, dst_y + dst_stride_y,
                     width);
#else
      ARGB4444ToARGBRow(src_argb4444, row, width);
      ARGB4444ToARGBRow(src_argb4444 + src_stride_argb4444, row + row_size,
                        width);
      ARGBToUVRow(row, row_size, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + row_size, dst_y + dst_stride_y, width);
#endif
      src_argb4444 += src_stride_argb4444 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if defined(HAS_ARGB4444TOYROW_NEON)
      ARGB4444ToUVRow(src_argb4444, 0, dst_u, dst_v, width);
      ARGB4444ToYRow(src_argb4444, dst_y, width);
#else
      ARGB4444ToARGBRow(src_argb4444, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !(defined(HAS_ARGB4444TOYROW_NEON))
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}

// Convert RGB24 to J400.
LIBYUV_API
int RGB24ToJ400(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_yj,
                int dst_stride_yj,
                int width,
                int height) {
  int y;
  void (*RGB24ToYJRow)(const uint8_t* src_rgb24, uint8_t* dst_yj, int width) =
      RGB24ToYJRow_C;
  if (!src_rgb24 || !dst_yj || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_rgb24 = src_rgb24 + (height - 1) * src_stride_rgb24;
    src_stride_rgb24 = -src_stride_rgb24;
  }
  // Coalesce rows.
  if (src_stride_rgb24 == width * 3 && dst_stride_yj == width) {
    width *= height;
    height = 1;
    src_stride_rgb24 = dst_stride_yj = 0;
  }
#if defined(HAS_RGB24TOYJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGB24ToYJRow = RGB24ToYJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYJRow = RGB24ToYJRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RGB24ToYJRow = RGB24ToYJRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      RGB24ToYJRow = RGB24ToYJRow_AVX2;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB24ToYJRow = RGB24ToYJRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYJRow = RGB24ToYJRow_NEON;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGB24ToYJRow = RGB24ToYJRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYJRow = RGB24ToYJRow_MSA;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RGB24ToYJRow = RGB24ToYJRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYJRow = RGB24ToYJRow_LSX;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RGB24ToYJRow = RGB24ToYJRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RGB24ToYJRow = RGB24ToYJRow_LASX;
    }
  }
#endif
#if defined(HAS_RGB24TOYJROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RGB24ToYJRow = RGB24ToYJRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    RGB24ToYJRow(src_rgb24, dst_yj, width);
    src_rgb24 += src_stride_rgb24;
    dst_yj += dst_stride_yj;
  }
  return 0;
}

// Convert RAW to J400.
LIBYUV_API
int RAWToJ400(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_yj,
              int dst_stride_yj,
              int width,
              int height) {
  int y;
  void (*RAWToYJRow)(const uint8_t* src_raw, uint8_t* dst_yj, int width) =
      RAWToYJRow_C;
  if (!src_raw || !dst_yj || width <= 0 || height == 0) {
    return -1;
  }

  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }
  // Coalesce rows.
  if (src_stride_raw == width * 3 && dst_stride_yj == width) {
    width *= height;
    height = 1;
    src_stride_raw = dst_stride_yj = 0;
  }

#if defined(HAS_RAWTOYJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToYJRow = RAWToYJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToYJRow = RAWToYJRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RAWToYJRow = RAWToYJRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      RAWToYJRow = RAWToYJRow_AVX2;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToYJRow = RAWToYJRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RAWToYJRow = RAWToYJRow_NEON;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RAWToYJRow = RAWToYJRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RAWToYJRow = RAWToYJRow_MSA;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RAWToYJRow = RAWToYJRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RAWToYJRow = RAWToYJRow_LSX;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RAWToYJRow = RAWToYJRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RAWToYJRow = RAWToYJRow_LASX;
    }
  }
#endif
#if defined(HAS_RAWTOYJROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RAWToYJRow = RAWToYJRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    RAWToYJRow(src_raw, dst_yj, width);
    src_raw += src_stride_raw;
    dst_yj += dst_stride_yj;
  }
  return 0;
}

// Convert Android420 to I420.
LIBYUV_API
int Android420ToI420(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     int src_pixel_stride_uv,
                     uint8_t* dst_y,
                     int dst_stride_y,
                     uint8_t* dst_u,
                     int dst_stride_u,
                     uint8_t* dst_v,
                     int dst_stride_v,
                     int width,
                     int height) {
  return Android420ToI420Rotate(src_y, src_stride_y, src_u, src_stride_u, src_v,
                                src_stride_v, src_pixel_stride_uv, dst_y,
                                dst_stride_y, dst_u, dst_stride_u, dst_v,
                                dst_stride_v, width, height, kRotate0);
}

// depth is source bits measured from lsb; For msb use 16
static int Biplanar16bitTo8bit(const uint16_t* src_y,
                               int src_stride_y,
                               const uint16_t* src_uv,
                               int src_stride_uv,
                               uint8_t* dst_y,
                               int dst_stride_y,
                               uint8_t* dst_uv,
                               int dst_stride_uv,
                               int width,
                               int height,
                               int subsample_x,
                               int subsample_y,
                               int depth) {
  int uv_width = SUBSAMPLE(width, subsample_x, subsample_x);
  int uv_height = SUBSAMPLE(height, subsample_y, subsample_y);
  int scale = 1 << (24 - depth);
  if ((!src_y && dst_y) || !src_uv || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    uv_height = -uv_height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_uv = src_uv + (uv_height - 1) * src_stride_uv;
    src_stride_y = -src_stride_y;
    src_stride_uv = -src_stride_uv;
  }

  // Convert Y plane.
  if (dst_y) {
    Convert16To8Plane(src_y, src_stride_y, dst_y, dst_stride_y, scale, width,
                      height);
  }
  // Convert UV planes.
  Convert16To8Plane(src_uv, src_stride_uv, dst_uv, dst_stride_uv, scale,
                    uv_width * 2, uv_height);
  return 0;
}

// Convert 10 bit P010 to 8 bit NV12.
// Depth set to 16 because P010 uses 10 msb and this function keeps the upper 8
// bits of the specified number of bits.
LIBYUV_API
int P010ToNV12(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  return Biplanar16bitTo8bit(src_y, src_stride_y, src_uv, src_stride_uv, dst_y,
                             dst_stride_y, dst_uv, dst_stride_uv, width, height,
                             1, 1, 16);
}

static int Planar8bitTo8bit(const uint8_t* src_y,
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
                            int height,
                            int subsample_x,
                            int subsample_y,
                            int scale_y,
                            int bias_y,
                            int scale_uv,
                            int bias_uv) {
  int uv_width = SUBSAMPLE(width, subsample_x, subsample_x);
  int uv_height = SUBSAMPLE(height, subsample_y, subsample_y);
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    uv_height = -uv_height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (uv_height - 1) * src_stride_u;
    src_v = src_v + (uv_height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  // Convert Y plane.
  if (dst_y) {
    Convert8To8Plane(src_y, src_stride_y, dst_y, dst_stride_y, scale_y, bias_y,
                     width, height);
  }
  // Convert UV planes.
  Convert8To8Plane(src_u, src_stride_u, dst_u, dst_stride_u, scale_uv, bias_uv,
                   uv_width, uv_height);
  Convert8To8Plane(src_v, src_stride_v, dst_v, dst_stride_v, scale_uv, bias_uv,
                   uv_width, uv_height);
  return 0;
}

LIBYUV_API
int J420ToI420(const uint8_t* src_y,
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
               int height) {
  return Planar8bitTo8bit(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_y, dst_stride_y, dst_u,
                          dst_stride_u, dst_v, dst_stride_v, width, height, 1,
                          1, 220, 16, 225, 16);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
