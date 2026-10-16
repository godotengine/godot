/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/convert_argb.h"

#include "libyuv/cpu_id.h"
#ifdef HAVE_JPEG
#include "libyuv/mjpeg_decoder.h"
#endif
#include "libyuv/rotate_argb.h"
#include "libyuv/row.h"
#include "libyuv/video_common.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Convert camera sample to ARGB with cropping, rotation and vertical flip.
// src_width is used for source stride computation
// src_height is used to compute location of planes, and indicate inversion
// sample_size is measured in bytes and is the size of the frame.
//   With MJPEG it is the compressed size of the frame.

// TODO(fbarchard): Add the following:
// H010ToARGB
// H420ToARGB
// H422ToARGB
// I010ToARGB
// J400ToARGB
// J422ToARGB
// J444ToARGB

LIBYUV_API
int ConvertToARGB(const uint8_t* sample,
                  size_t sample_size,
                  uint8_t* dst_argb,
                  int dst_stride_argb,
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
  int abs_src_height = (src_height < 0) ? -src_height : src_height;
  int inv_crop_height = (crop_height < 0) ? -crop_height : crop_height;
  int r = 0;

  // One pass rotation is available for some formats. For the rest, convert
  // to ARGB (with optional vertical flipping) into a temporary ARGB buffer,
  // and then rotate the ARGB to the final destination buffer.
  // For in-place conversion, if destination dst_argb is same as source sample,
  // also enable temporary buffer.
  LIBYUV_BOOL need_buf =
      (rotation && format != FOURCC_ARGB) || dst_argb == sample;
  uint8_t* dest_argb = dst_argb;
  int dest_dst_stride_argb = dst_stride_argb;
  uint8_t* rotate_buffer = NULL;
  int abs_crop_height = (crop_height < 0) ? -crop_height : crop_height;

  if (dst_argb == NULL || sample == NULL || src_width <= 0 || crop_width <= 0 ||
      src_height == 0 || crop_height == 0) {
    return -1;
  }
  if (src_height < 0) {
    inv_crop_height = -inv_crop_height;
  }

  if (need_buf) {
    int argb_size = crop_width * 4 * abs_crop_height;
    rotate_buffer = (uint8_t*)malloc(argb_size); /* NOLINT */
    if (!rotate_buffer) {
      return 1;  // Out of memory runtime error.
    }
    dst_argb = rotate_buffer;
    dst_stride_argb = crop_width * 4;
  }

  switch (format) {
    // Single plane formats
    case FOURCC_YUY2:
      src = sample + (aligned_src_width * crop_y + crop_x) * 2;
      r = YUY2ToARGB(src, aligned_src_width * 2, dst_argb, dst_stride_argb,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_UYVY:
      src = sample + (aligned_src_width * crop_y + crop_x) * 2;
      r = UYVYToARGB(src, aligned_src_width * 2, dst_argb, dst_stride_argb,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_24BG:
      src = sample + (src_width * crop_y + crop_x) * 3;
      r = RGB24ToARGB(src, src_width * 3, dst_argb, dst_stride_argb, crop_width,
                      inv_crop_height);
      break;
    case FOURCC_RAW:
      src = sample + (src_width * crop_y + crop_x) * 3;
      r = RAWToARGB(src, src_width * 3, dst_argb, dst_stride_argb, crop_width,
                    inv_crop_height);
      break;
    case FOURCC_ARGB:
      if (!need_buf && !rotation) {
        src = sample + (src_width * crop_y + crop_x) * 4;
        r = ARGBToARGB(src, src_width * 4, dst_argb, dst_stride_argb,
                       crop_width, inv_crop_height);
      }
      break;
    case FOURCC_BGRA:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = BGRAToARGB(src, src_width * 4, dst_argb, dst_stride_argb, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_ABGR:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = ABGRToARGB(src, src_width * 4, dst_argb, dst_stride_argb, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_RGBA:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = RGBAToARGB(src, src_width * 4, dst_argb, dst_stride_argb, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_AR30:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = AR30ToARGB(src, src_width * 4, dst_argb, dst_stride_argb, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_AB30:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = AB30ToARGB(src, src_width * 4, dst_argb, dst_stride_argb, crop_width,
                     inv_crop_height);
      break;
    case FOURCC_RGBP:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = RGB565ToARGB(src, src_width * 2, dst_argb, dst_stride_argb,
                       crop_width, inv_crop_height);
      break;
    case FOURCC_RGBO:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = ARGB1555ToARGB(src, src_width * 2, dst_argb, dst_stride_argb,
                         crop_width, inv_crop_height);
      break;
    case FOURCC_R444:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = ARGB4444ToARGB(src, src_width * 2, dst_argb, dst_stride_argb,
                         crop_width, inv_crop_height);
      break;
    case FOURCC_I400:
      src = sample + src_width * crop_y + crop_x;
      r = I400ToARGB(src, src_width, dst_argb, dst_stride_argb, crop_width,
                     inv_crop_height);
      break;

    // Biplanar formats
    case FOURCC_NV12:
      src = sample + (src_width * crop_y + crop_x);
      src_uv = sample + aligned_src_width * (abs_src_height + crop_y / 2) + crop_x;
      r = NV12ToARGB(src, src_width, src_uv, aligned_src_width, dst_argb,
                     dst_stride_argb, crop_width, inv_crop_height);
      break;
    case FOURCC_NV21:
      src = sample + (src_width * crop_y + crop_x);
      src_uv = sample + aligned_src_width * (abs_src_height + crop_y / 2) + crop_x;
      // Call NV12 but with u and v parameters swapped.
      r = NV21ToARGB(src, src_width, src_uv, aligned_src_width, dst_argb,
                     dst_stride_argb, crop_width, inv_crop_height);
      break;
    case FOURCC_M420:
      src = sample + (src_width * crop_y) * 12 / 8 + crop_x;
      r = M420ToARGB(src, src_width, dst_argb, dst_stride_argb, crop_width,
                     inv_crop_height);
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
      r = I420ToARGB(src_y, src_width, src_u, halfwidth, src_v, halfwidth,
                     dst_argb, dst_stride_argb, crop_width, inv_crop_height);
      break;
    }

    case FOURCC_J420: {
      const uint8_t* src_y = sample + (src_width * crop_y + crop_x);
      const uint8_t* src_u;
      const uint8_t* src_v;
      int halfwidth = (src_width + 1) / 2;
      int halfheight = (abs_src_height + 1) / 2;
      src_u = sample + src_width * abs_src_height +
              (halfwidth * crop_y + crop_x) / 2;
      src_v = sample + src_width * abs_src_height +
              halfwidth * (halfheight + crop_y / 2) + crop_x / 2;
      r = J420ToARGB(src_y, src_width, src_u, halfwidth, src_v, halfwidth,
                     dst_argb, dst_stride_argb, crop_width, inv_crop_height);
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
      r = I422ToARGB(src_y, src_width, src_u, halfwidth, src_v, halfwidth,
                     dst_argb, dst_stride_argb, crop_width, inv_crop_height);
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
      r = I444ToARGB(src_y, src_width, src_u, src_width, src_v, src_width,
                     dst_argb, dst_stride_argb, crop_width, inv_crop_height);
      break;
    }
#ifdef HAVE_JPEG
    case FOURCC_MJPG:
      r = MJPGToARGB(sample, sample_size, dst_argb, dst_stride_argb, src_width,
                     abs_src_height, crop_width, inv_crop_height);
      break;
#endif
    default:
      r = -1;  // unknown fourcc - return failure code.
  }

  if (need_buf) {
    if (!r) {
      r = ARGBRotate(dst_argb, dst_stride_argb, dest_argb, dest_dst_stride_argb,
                     crop_width, abs_crop_height, rotation);
    }
    free(rotate_buffer);
  } else if (rotation) {
    src = sample + (src_width * crop_y + crop_x) * 4;
    r = ARGBRotate(src, src_width * 4, dst_argb, dst_stride_argb, crop_width,
                   inv_crop_height, rotation);
  }

  return r;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
