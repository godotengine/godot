/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>

#include "libyuv/convert.h"

#include "libyuv/video_common.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Convert camera sample to I420 with cropping, rotation and vertical flip.
// src_width is used for source stride computation
// src_height is used to compute location of planes, and indicate inversion
// sample_size is measured in bytes and is the size of the frame.
//   With MJPEG it is the compressed size of the frame.
LIBYUV_API
int ConvertToI420(const uint8_t* sample,
                  size_t sample_size,
                  uint8_t* dst_y,
                  int dst_stride_y,
                  uint8_t* dst_u,
                  int dst_stride_u,
                  uint8_t* dst_v,
                  int dst_stride_v,
                  int crop_x,
                  int crop_y,
                  int src_width,
                  int src_height,
                  int crop_width,
                  int crop_height,
                  enum RotationMode rotation,
                  uint32_t fourcc) {
  uint32_t format = CanonicalFourCC(fourcc);
  int aligned_src_width = (src_width + 1) & ~1;
  const uint8_t* src;
  const uint8_t* src_uv;
  const int abs_src_height = (src_height < 0) ? -src_height : src_height;
  // TODO(nisse): Why allow crop_height < 0?
  const int abs_crop_height = (crop_height < 0) ? -crop_height : crop_height;
  int r = 0;
  LIBYUV_BOOL need_buf =
      (rotation && format != FOURCC_I420 && format != FOURCC_NV12 &&
       format != FOURCC_NV21 && format != FOURCC_YV12) ||
      dst_y == sample;
  uint8_t* tmp_y = dst_y;
  uint8_t* tmp_u = dst_u;
  uint8_t* tmp_v = dst_v;
  int tmp_y_stride = dst_stride_y;
  int tmp_u_stride = dst_stride_u;
  int tmp_v_stride = dst_stride_v;
  uint8_t* rotate_buffer = NULL;
  const int inv_crop_height =
      (src_height < 0) ? -abs_crop_height : abs_crop_height;

  if (!dst_y || !dst_u || !dst_v || !sample || src_width <= 0 ||
      crop_width <= 0 || src_height == 0 || crop_height == 0) {
    return -1;
  }

  // One pass rotation is available for some formats. For the rest, convert
  // to I420 (with optional vertical flipping) into a temporary I420 buffer,
  // and then rotate the I420 to the final destination buffer.
  // For in-place conversion, if destination dst_y is same as source sample,
  // also enable temporary buffer.
  if (need_buf) {
    int y_size = crop_width * abs_crop_height;
    int uv_size = ((crop_width + 1) / 2) * ((abs_crop_height + 1) / 2);
    rotate_buffer = (uint8_t*)malloc(y_size + uv_size * 2); /* NOLINT */
    if (!rotate_buffer) {
      return 1;  // Out of memory runtime error.
    }
    dst_y = rotate_buffer;
    dst_u = dst_y + y_size;
    dst_v = dst_u + uv_size;
    dst_stride_y = crop_width;
    dst_stride_u = dst_stride_v = ((crop_width + 1) / 2);
  }

  switch (format) {
    // Single plane formats
    case FOURCC_YUY2:
      src = sample + (aligned_src_width * crop_y + crop_x) * 2;
      r = YUY2ToI420(src, aligned_src_width * 2, dst_y, dst_stride_y, dst_u,
                     dst_stride_u, dst_v, dst_stride_v, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_UYVY:
      src = sample + (aligned_src_width * crop_y + crop_x) * 2;
      r = UYVYToI420(src, aligned_src_width * 2, dst_y, dst_stride_y, dst_u,
                     dst_stride_u, dst_v, dst_stride_v, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_RGBP:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = RGB565ToI420(src, src_width * 2, dst_y, dst_stride_y, dst_u,
                       dst_stride_u, dst_v, dst_stride_v, crop_width,
                       inv_crop_height);
      break;
    case FOURCC_RGBO:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = ARGB1555ToI420(src, src_width * 2, dst_y, dst_stride_y, dst_u,
                         dst_stride_u, dst_v, dst_stride_v, crop_width,
                         inv_crop_height);
      break;
    case FOURCC_R444:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = ARGB4444ToI420(src, src_width * 2, dst_y, dst_stride_y, dst_u,
                         dst_stride_u, dst_v, dst_stride_v, crop_width,
                         inv_crop_height);
      break;
    case FOURCC_24BG:
      src = sample + (src_width * crop_y + crop_x) * 3;
      r = RGB24ToI420(src, src_width * 3, dst_y, dst_stride_y, dst_u,
                      dst_stride_u, dst_v, dst_stride_v, crop_width,
                      inv_crop_height);
      break;
    case FOURCC_RAW:
      src = sample + (src_width * crop_y + crop_x) * 3;
      r = RAWToI420(src, src_width * 3, dst_y, dst_stride_y, dst_u,
                    dst_stride_u, dst_v, dst_stride_v, crop_width,
                    inv_crop_height);
      break;
    case FOURCC_ARGB:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = ARGBToI420(src, src_width * 4, dst_y, dst_stride_y, dst_u,
                     dst_stride_u, dst_v, dst_stride_v, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_BGRA:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = BGRAToI420(src, src_width * 4, dst_y, dst_stride_y, dst_u,
                     dst_stride_u, dst_v, dst_stride_v, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_ABGR:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = ABGRToI420(src, src_width * 4, dst_y, dst_stride_y, dst_u,
                     dst_stride_u, dst_v, dst_stride_v, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_RGBA:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = RGBAToI420(src, src_width * 4, dst_y, dst_stride_y, dst_u,
                     dst_stride_u, dst_v, dst_stride_v, crop_width,
                     inv_crop_height);
      break;
    // TODO(fbarchard): Add AR30 and AB30
    case FOURCC_I400:
      src = sample + src_width * crop_y + crop_x;
      r = I400ToI420(src, src_width, dst_y, dst_stride_y, dst_u, dst_stride_u,
                     dst_v, dst_stride_v, crop_width, inv_crop_height);
      break;
    // Biplanar formats
    case FOURCC_NV12:
      src = sample + (src_width * crop_y + crop_x);
      src_uv = sample + (src_width * abs_src_height) +
               ((crop_y / 2) * aligned_src_width) + ((crop_x / 2) * 2);
      r = NV12ToI420Rotate(src, src_width, src_uv, aligned_src_width, dst_y,
                           dst_stride_y, dst_u, dst_stride_u, dst_v,
                           dst_stride_v, crop_width, inv_crop_height, rotation);
      break;
    case FOURCC_NV21:
      src = sample + (src_width * crop_y + crop_x);
      src_uv = sample + (src_width * abs_src_height) +
               ((crop_y / 2) * aligned_src_width) + ((crop_x / 2) * 2);
      // Call NV12 but with dst_u and dst_v parameters swapped.
      r = NV12ToI420Rotate(src, src_width, src_uv, aligned_src_width, dst_y,
                           dst_stride_y, dst_v, dst_stride_v, dst_u,
                           dst_stride_u, crop_width, inv_crop_height, rotation);
      break;
    case FOURCC_M420:
      src = sample + (src_width * crop_y) * 12 / 8 + crop_x;
      r = M420ToI420(src, src_width, dst_y, dst_stride_y, dst_u, dst_stride_u,
                     dst_v, dst_stride_v, crop_width, inv_crop_height);
      break;
    // Triplanar formats
    case FOURCC_I420:
    case FOURCC_YV12: {
      const uint8_t* src_y = sample + (src_width * crop_y + crop_x);
      const uint8_t* src_u;
      const uint8_t* src_v;
      int halfwidth = (src_width + 1) / 2;
      int halfheight = (abs_src_height + 1) / 2;
      if (format == FOURCC_YV12) {
        src_v = sample + src_width * abs_src_height +
                (halfwidth * crop_y + crop_x) / 2;
        src_u = sample + src_width * abs_src_height +
                halfwidth * (halfheight + crop_y / 2) + crop_x / 2;
      } else {
        src_u = sample + src_width * abs_src_height +
                (halfwidth * crop_y + crop_x) / 2;
        src_v = sample + src_width * abs_src_height +
                halfwidth * (halfheight + crop_y / 2) + crop_x / 2;
      }
      r = I420Rotate(src_y, src_width, src_u, halfwidth, src_v, halfwidth,
                     dst_y, dst_stride_y, dst_u, dst_stride_u, dst_v,
                     dst_stride_v, crop_width, inv_crop_height, rotation);
      break;
    }
    case FOURCC_I422:
    case FOURCC_YV16: {
      const uint8_t* src_y = sample + src_width * crop_y + crop_x;
      const uint8_t* src_u;
      const uint8_t* src_v;
      int halfwidth = (src_width + 1) / 2;
      if (format == FOURCC_YV16) {
        src_v = sample + src_width * abs_src_height + halfwidth * crop_y +
                crop_x / 2;
        src_u = sample + src_width * abs_src_height +
                halfwidth * (abs_src_height + crop_y) + crop_x / 2;
      } else {
        src_u = sample + src_width * abs_src_height + halfwidth * crop_y +
                crop_x / 2;
        src_v = sample + src_width * abs_src_height +
                halfwidth * (abs_src_height + crop_y) + crop_x / 2;
      }
      r = I422ToI420(src_y, src_width, src_u, halfwidth, src_v, halfwidth,
                     dst_y, dst_stride_y, dst_u, dst_stride_u, dst_v,
                     dst_stride_v, crop_width, inv_crop_height);
      break;
    }
    case FOURCC_I444:
    case FOURCC_YV24: {
      const uint8_t* src_y = sample + src_width * crop_y + crop_x;
      const uint8_t* src_u;
      const uint8_t* src_v;
      if (format == FOURCC_YV24) {
        src_v = sample + src_width * (abs_src_height + crop_y) + crop_x;
        src_u = sample + src_width * (abs_src_height * 2 + crop_y) + crop_x;
      } else {
        src_u = sample + src_width * (abs_src_height + crop_y) + crop_x;
        src_v = sample + src_width * (abs_src_height * 2 + crop_y) + crop_x;
      }
      r = I444ToI420(src_y, src_width, src_u, src_width, src_v, src_width,
                     dst_y, dst_stride_y, dst_u, dst_stride_u, dst_v,
                     dst_stride_v, crop_width, inv_crop_height);
      break;
    }
#ifdef HAVE_JPEG
    case FOURCC_MJPG:
      r = MJPGToI420(sample, sample_size, dst_y, dst_stride_y, dst_u,
                     dst_stride_u, dst_v, dst_stride_v, src_width,
                     abs_src_height, crop_width, inv_crop_height);
      break;
#endif
    default:
      r = -1;  // unknown fourcc - return failure code.
  }

  if (need_buf) {
    if (!r) {
      r = I420Rotate(dst_y, dst_stride_y, dst_u, dst_stride_u, dst_v,
                     dst_stride_v, tmp_y, tmp_y_stride, tmp_u, tmp_u_stride,
                     tmp_v, tmp_v_stride, crop_width, abs_crop_height,
                     rotation);
    }
    free(rotate_buffer);
  }

  return r;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
