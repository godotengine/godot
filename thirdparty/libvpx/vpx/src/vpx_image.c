/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>
#include <string.h>

#include "vpx/vpx_image.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"

static vpx_image_t *img_alloc_helper(vpx_image_t *img,
                                     vpx_img_fmt_t fmt,
                                     unsigned int d_w,
                                     unsigned int d_h,
                                     unsigned int buf_align,
                                     unsigned int stride_align,
                                     unsigned char *img_data) {
  unsigned int h, w, s, xcs, ycs, bps;
  unsigned int stride_in_bytes;
  int align;

  /* Treat align==0 like align==1 */
  if (!buf_align)
    buf_align = 1;

  /* Validate alignment (must be power of 2) */
  if (buf_align & (buf_align - 1))
    goto fail;

  /* Treat align==0 like align==1 */
  if (!stride_align)
    stride_align = 1;

  /* Validate alignment (must be power of 2) */
  if (stride_align & (stride_align - 1))
    goto fail;

  /* Get sample size for this format */
  switch (fmt) {
    case VPX_IMG_FMT_RGB32:
    case VPX_IMG_FMT_RGB32_LE:
    case VPX_IMG_FMT_ARGB:
    case VPX_IMG_FMT_ARGB_LE:
      bps = 32;
      break;
    case VPX_IMG_FMT_RGB24:
    case VPX_IMG_FMT_BGR24:
      bps = 24;
      break;
    case VPX_IMG_FMT_RGB565:
    case VPX_IMG_FMT_RGB565_LE:
    case VPX_IMG_FMT_RGB555:
    case VPX_IMG_FMT_RGB555_LE:
    case VPX_IMG_FMT_UYVY:
    case VPX_IMG_FMT_YUY2:
    case VPX_IMG_FMT_YVYU:
      bps = 16;
      break;
    case VPX_IMG_FMT_I420:
    case VPX_IMG_FMT_YV12:
    case VPX_IMG_FMT_VPXI420:
    case VPX_IMG_FMT_VPXYV12:
      bps = 12;
      break;
    case VPX_IMG_FMT_I422:
    case VPX_IMG_FMT_I440:
      bps = 16;
      break;
    case VPX_IMG_FMT_I444:
      bps = 24;
      break;
    case VPX_IMG_FMT_I42016:
      bps = 24;
      break;
    case VPX_IMG_FMT_I42216:
    case VPX_IMG_FMT_I44016:
      bps = 32;
      break;
    case VPX_IMG_FMT_I44416:
      bps = 48;
      break;
    default:
      bps = 16;
      break;
  }

  /* Get chroma shift values for this format */
  switch (fmt) {
    case VPX_IMG_FMT_I420:
    case VPX_IMG_FMT_YV12:
    case VPX_IMG_FMT_VPXI420:
    case VPX_IMG_FMT_VPXYV12:
    case VPX_IMG_FMT_I422:
    case VPX_IMG_FMT_I42016:
    case VPX_IMG_FMT_I42216:
      xcs = 1;
      break;
    default:
      xcs = 0;
      break;
  }

  switch (fmt) {
    case VPX_IMG_FMT_I420:
    case VPX_IMG_FMT_I440:
    case VPX_IMG_FMT_YV12:
    case VPX_IMG_FMT_VPXI420:
    case VPX_IMG_FMT_VPXYV12:
    case VPX_IMG_FMT_I42016:
    case VPX_IMG_FMT_I44016:
      ycs = 1;
      break;
    default:
      ycs = 0;
      break;
  }

  /* Calculate storage sizes given the chroma subsampling */
  align = (1 << xcs) - 1;
  w = (d_w + align) & ~align;
  align = (1 << ycs) - 1;
  h = (d_h + align) & ~align;
  s = (fmt & VPX_IMG_FMT_PLANAR) ? w : bps * w / 8;
  s = (s + stride_align - 1) & ~(stride_align - 1);
  stride_in_bytes = (fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? s * 2 : s;

  /* Allocate the new image */
  if (!img) {
    img = (vpx_image_t *)calloc(1, sizeof(vpx_image_t));

    if (!img)
      goto fail;

    img->self_allocd = 1;
  } else {
    memset(img, 0, sizeof(vpx_image_t));
  }

  img->img_data = img_data;

  if (!img_data) {
    const uint64_t alloc_size = (fmt & VPX_IMG_FMT_PLANAR) ?
                                (uint64_t)h * s * bps / 8 : (uint64_t)h * s;

    if (alloc_size != (size_t)alloc_size)
      goto fail;

    img->img_data = (uint8_t *)vpx_memalign(buf_align, (size_t)alloc_size);
    img->img_data_owner = 1;
  }

  if (!img->img_data)
    goto fail;

  img->fmt = fmt;
  img->bit_depth = (fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 16 : 8;
  img->w = w;
  img->h = h;
  img->x_chroma_shift = xcs;
  img->y_chroma_shift = ycs;
  img->bps = bps;

  /* Calculate strides */
  img->stride[VPX_PLANE_Y] = img->stride[VPX_PLANE_ALPHA] = stride_in_bytes;
  img->stride[VPX_PLANE_U] = img->stride[VPX_PLANE_V] = stride_in_bytes >> xcs;

  /* Default viewport to entire image */
  if (!vpx_img_set_rect(img, 0, 0, d_w, d_h))
    return img;

fail:
  vpx_img_free(img);
  return NULL;
}

vpx_image_t *vpx_img_alloc(vpx_image_t  *img,
                           vpx_img_fmt_t fmt,
                           unsigned int  d_w,
                           unsigned int  d_h,
                           unsigned int  align) {
  return img_alloc_helper(img, fmt, d_w, d_h, align, align, NULL);
}

vpx_image_t *vpx_img_wrap(vpx_image_t  *img,
                          vpx_img_fmt_t fmt,
                          unsigned int  d_w,
                          unsigned int  d_h,
                          unsigned int  stride_align,
                          unsigned char       *img_data) {
  /* By setting buf_align = 1, we don't change buffer alignment in this
   * function. */
  return img_alloc_helper(img, fmt, d_w, d_h, 1, stride_align, img_data);
}

int vpx_img_set_rect(vpx_image_t  *img,
                     unsigned int  x,
                     unsigned int  y,
                     unsigned int  w,
                     unsigned int  h) {
  unsigned char      *data;

  if (x + w <= img->w && y + h <= img->h) {
    img->d_w = w;
    img->d_h = h;

    /* Calculate plane pointers */
    if (!(img->fmt & VPX_IMG_FMT_PLANAR)) {
      img->planes[VPX_PLANE_PACKED] =
        img->img_data + x * img->bps / 8 + y * img->stride[VPX_PLANE_PACKED];
    } else {
      const int bytes_per_sample =
          (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;
      data = img->img_data;

      if (img->fmt & VPX_IMG_FMT_HAS_ALPHA) {
        img->planes[VPX_PLANE_ALPHA] =
            data + x * bytes_per_sample + y * img->stride[VPX_PLANE_ALPHA];
        data += img->h * img->stride[VPX_PLANE_ALPHA];
      }

      img->planes[VPX_PLANE_Y] = data + x * bytes_per_sample +
          y * img->stride[VPX_PLANE_Y];
      data += img->h * img->stride[VPX_PLANE_Y];

      if (!(img->fmt & VPX_IMG_FMT_UV_FLIP)) {
        img->planes[VPX_PLANE_U] =
            data + (x >> img->x_chroma_shift) * bytes_per_sample +
            (y >> img->y_chroma_shift) * img->stride[VPX_PLANE_U];
        data += (img->h >> img->y_chroma_shift) * img->stride[VPX_PLANE_U];
        img->planes[VPX_PLANE_V] =
            data + (x >> img->x_chroma_shift) * bytes_per_sample +
            (y >> img->y_chroma_shift) * img->stride[VPX_PLANE_V];
      } else {
        img->planes[VPX_PLANE_V] =
            data + (x >> img->x_chroma_shift) * bytes_per_sample +
            (y >> img->y_chroma_shift) * img->stride[VPX_PLANE_V];
        data += (img->h >> img->y_chroma_shift) * img->stride[VPX_PLANE_V];
        img->planes[VPX_PLANE_U] =
            data + (x >> img->x_chroma_shift) * bytes_per_sample +
            (y >> img->y_chroma_shift) * img->stride[VPX_PLANE_U];
      }
    }
    return 0;
  }
  return -1;
}

void vpx_img_flip(vpx_image_t *img) {
  /* Note: In the calculation pointer adjustment calculation, we want the
   * rhs to be promoted to a signed type. Section 6.3.1.8 of the ISO C99
   * standard indicates that if the adjustment parameter is unsigned, the
   * stride parameter will be promoted to unsigned, causing errors when
   * the lhs is a larger type than the rhs.
   */
  img->planes[VPX_PLANE_Y] += (signed)(img->d_h - 1) * img->stride[VPX_PLANE_Y];
  img->stride[VPX_PLANE_Y] = -img->stride[VPX_PLANE_Y];

  img->planes[VPX_PLANE_U] += (signed)((img->d_h >> img->y_chroma_shift) - 1)
                              * img->stride[VPX_PLANE_U];
  img->stride[VPX_PLANE_U] = -img->stride[VPX_PLANE_U];

  img->planes[VPX_PLANE_V] += (signed)((img->d_h >> img->y_chroma_shift) - 1)
                              * img->stride[VPX_PLANE_V];
  img->stride[VPX_PLANE_V] = -img->stride[VPX_PLANE_V];

  img->planes[VPX_PLANE_ALPHA] += (signed)(img->d_h - 1) * img->stride[VPX_PLANE_ALPHA];
  img->stride[VPX_PLANE_ALPHA] = -img->stride[VPX_PLANE_ALPHA];
}

void vpx_img_free(vpx_image_t *img) {
  if (img) {
    if (img->img_data && img->img_data_owner)
      vpx_free(img->img_data);

    if (img->self_allocd)
      free(img);
  }
}
