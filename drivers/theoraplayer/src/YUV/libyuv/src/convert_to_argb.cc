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
#include "libyuv/format_conversion.h"
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

// Convert camera sample to I420 with cropping, rotation and vertical flip.
// src_width is used for source stride computation
// src_height is used to compute location of planes, and indicate inversion
// sample_size is measured in bytes and is the size of the frame.
//   With MJPEG it is the compressed size of the frame.
LIBYUV_API
int ConvertToARGB(const uint8* sample, size_t sample_size,
                  uint8* crop_argb, int argb_stride,
                  int crop_x, int crop_y,
                  int src_width, int src_height,
                  int crop_width, int crop_height,
                  enum RotationMode rotation,
                  uint32 fourcc) {
  uint32 format = CanonicalFourCC(fourcc);
  int aligned_src_width = (src_width + 1) & ~1;
  const uint8* src;
  const uint8* src_uv;
  int abs_src_height = (src_height < 0) ? -src_height : src_height;
  int inv_crop_height = (crop_height < 0) ? -crop_height : crop_height;
  int r = 0;

  // One pass rotation is available for some formats. For the rest, convert
  // to I420 (with optional vertical flipping) into a temporary I420 buffer,
  // and then rotate the I420 to the final destination buffer.
  // For in-place conversion, if destination crop_argb is same as source sample,
  // also enable temporary buffer.
  LIBYUV_BOOL need_buf = (rotation && format != FOURCC_ARGB) ||
      crop_argb == sample;
  uint8* tmp_argb = crop_argb;
  int tmp_argb_stride = argb_stride;
  uint8* rotate_buffer = NULL;
  int abs_crop_height = (crop_height < 0) ? -crop_height : crop_height;

  if (crop_argb == NULL || sample == NULL ||
      src_width <= 0 || crop_width <= 0 ||
      src_height == 0 || crop_height == 0) {
    return -1;
  }
  if (src_height < 0) {
    inv_crop_height = -inv_crop_height;
  }

  if (need_buf) {
    int argb_size = crop_width * abs_crop_height * 4;
    rotate_buffer = (uint8*)malloc(argb_size);
    if (!rotate_buffer) {
      return 1;  // Out of memory runtime error.
    }
    crop_argb = rotate_buffer;
    argb_stride = crop_width;
  }

  switch (format) {
    // Single plane formats
    case FOURCC_YUY2:
      src = sample + (aligned_src_width * crop_y + crop_x) * 2;
      r = YUY2ToARGB(src, aligned_src_width * 2,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_UYVY:
      src = sample + (aligned_src_width * crop_y + crop_x) * 2;
      r = UYVYToARGB(src, aligned_src_width * 2,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_24BG:
      src = sample + (src_width * crop_y + crop_x) * 3;
      r = RGB24ToARGB(src, src_width * 3,
                      crop_argb, argb_stride,
                      crop_width, inv_crop_height);
      break;
    case FOURCC_RAW:
      src = sample + (src_width * crop_y + crop_x) * 3;
      r = RAWToARGB(src, src_width * 3,
                    crop_argb, argb_stride,
                    crop_width, inv_crop_height);
      break;
    case FOURCC_ARGB:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = ARGBToARGB(src, src_width * 4,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_BGRA:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = BGRAToARGB(src, src_width * 4,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_ABGR:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = ABGRToARGB(src, src_width * 4,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_RGBA:
      src = sample + (src_width * crop_y + crop_x) * 4;
      r = RGBAToARGB(src, src_width * 4,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_RGBP:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = RGB565ToARGB(src, src_width * 2,
                       crop_argb, argb_stride,
                       crop_width, inv_crop_height);
      break;
    case FOURCC_RGBO:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = ARGB1555ToARGB(src, src_width * 2,
                         crop_argb, argb_stride,
                         crop_width, inv_crop_height);
      break;
    case FOURCC_R444:
      src = sample + (src_width * crop_y + crop_x) * 2;
      r = ARGB4444ToARGB(src, src_width * 2,
                         crop_argb, argb_stride,
                         crop_width, inv_crop_height);
      break;
    // TODO(fbarchard): Support cropping Bayer by odd numbers
    // by adjusting fourcc.
    case FOURCC_BGGR:
      src = sample + (src_width * crop_y + crop_x);
      r = BayerBGGRToARGB(src, src_width,
                          crop_argb, argb_stride,
                          crop_width, inv_crop_height);
      break;

    case FOURCC_GBRG:
      src = sample + (src_width * crop_y + crop_x);
      r = BayerGBRGToARGB(src, src_width,
                          crop_argb, argb_stride,
                          crop_width, inv_crop_height);
      break;

    case FOURCC_GRBG:
      src = sample + (src_width * crop_y + crop_x);
      r = BayerGRBGToARGB(src, src_width,
                          crop_argb, argb_stride,
                          crop_width, inv_crop_height);
      break;

    case FOURCC_RGGB:
      src = sample + (src_width * crop_y + crop_x);
      r = BayerRGGBToARGB(src, src_width,
                          crop_argb, argb_stride,
                          crop_width, inv_crop_height);
      break;

    case FOURCC_I400:
      src = sample + src_width * crop_y + crop_x;
      r = I400ToARGB(src, src_width,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;

    // Biplanar formats
    case FOURCC_NV12:
      src = sample + (src_width * crop_y + crop_x);
      src_uv = sample + aligned_src_width * (src_height + crop_y / 2) + crop_x;
      r = NV12ToARGB(src, src_width,
                     src_uv, aligned_src_width,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_NV21:
      src = sample + (src_width * crop_y + crop_x);
      src_uv = sample + aligned_src_width * (src_height + crop_y / 2) + crop_x;
      // Call NV12 but with u and v parameters swapped.
      r = NV21ToARGB(src, src_width,
                     src_uv, aligned_src_width,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    case FOURCC_M420:
      src = sample + (src_width * crop_y) * 12 / 8 + crop_x;
      r = M420ToARGB(src, src_width,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
//    case FOURCC_Q420:
//      src = sample + (src_width + aligned_src_width * 2) * crop_y + crop_x;
//      src_uv = sample + (src_width + aligned_src_width * 2) * crop_y +
//               src_width + crop_x * 2;
//      r = Q420ToARGB(src, src_width * 3,
//                    src_uv, src_width * 3,
//                    crop_argb, argb_stride,
//                    crop_width, inv_crop_height);
//      break;
    // Triplanar formats
    case FOURCC_I420:
    case FOURCC_YU12:
    case FOURCC_YV12: {
      const uint8* src_y = sample + (src_width * crop_y + crop_x);
      const uint8* src_u;
      const uint8* src_v;
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
      r = I420ToARGB(src_y, src_width,
                     src_u, halfwidth,
                     src_v, halfwidth,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    }
    case FOURCC_I422:
    case FOURCC_YV16: {
      const uint8* src_y = sample + src_width * crop_y + crop_x;
      const uint8* src_u;
      const uint8* src_v;
      int halfwidth = (src_width + 1) / 2;
      if (format == FOURCC_YV16) {
        src_v = sample + src_width * abs_src_height +
            halfwidth * crop_y + crop_x / 2;
        src_u = sample + src_width * abs_src_height +
            halfwidth * (abs_src_height + crop_y) + crop_x / 2;
      } else {
        src_u = sample + src_width * abs_src_height +
            halfwidth * crop_y + crop_x / 2;
        src_v = sample + src_width * abs_src_height +
            halfwidth * (abs_src_height + crop_y) + crop_x / 2;
      }
      r = I422ToARGB(src_y, src_width,
                     src_u, halfwidth,
                     src_v, halfwidth,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    }
    case FOURCC_I444:
    case FOURCC_YV24: {
      const uint8* src_y = sample + src_width * crop_y + crop_x;
      const uint8* src_u;
      const uint8* src_v;
      if (format == FOURCC_YV24) {
        src_v = sample + src_width * (abs_src_height + crop_y) + crop_x;
        src_u = sample + src_width * (abs_src_height * 2 + crop_y) + crop_x;
      } else {
        src_u = sample + src_width * (abs_src_height + crop_y) + crop_x;
        src_v = sample + src_width * (abs_src_height * 2 + crop_y) + crop_x;
      }
      r = I444ToARGB(src_y, src_width,
                     src_u, src_width,
                     src_v, src_width,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    }
    case FOURCC_I411: {
      int quarterwidth = (src_width + 3) / 4;
      const uint8* src_y = sample + src_width * crop_y + crop_x;
      const uint8* src_u = sample + src_width * abs_src_height +
          quarterwidth * crop_y + crop_x / 4;
      const uint8* src_v = sample + src_width * abs_src_height +
          quarterwidth * (abs_src_height + crop_y) + crop_x / 4;
      r = I411ToARGB(src_y, src_width,
                     src_u, quarterwidth,
                     src_v, quarterwidth,
                     crop_argb, argb_stride,
                     crop_width, inv_crop_height);
      break;
    }
#ifdef HAVE_JPEG
    case FOURCC_MJPG:
      r = MJPGToARGB(sample, sample_size,
                     crop_argb, argb_stride,
                     src_width, abs_src_height, crop_width, inv_crop_height);
      break;
#endif
    default:
      r = -1;  // unknown fourcc - return failure code.
  }

  if (need_buf) {
    if (!r) {
      r = ARGBRotate(crop_argb, argb_stride,
                     tmp_argb, tmp_argb_stride,
                     crop_width, abs_crop_height, rotation);
    }
    free(rotate_buffer);
  }

  return r;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
