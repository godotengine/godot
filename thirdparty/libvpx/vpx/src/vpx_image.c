/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vpx_image.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"

static vpx_image_t *img_alloc_helper(vpx_image_t *img, vpx_img_fmt_t fmt,
                                     unsigned int d_w, unsigned int d_h,
                                     unsigned int buf_align,
                                     unsigned int stride_align,
                                     unsigned char *img_data) {
  unsigned int h, w, xcs, ycs, bps;
  uint64_t s;
  int stride_in_bytes;
  unsigned int align;

  if (img != NULL) memset(img, 0, sizeof(vpx_image_t));

  if (fmt == VPX_IMG_FMT_NONE) goto fail;

  /* Impose maximum values on input parameters so that this function can
   * perform arithmetic operations without worrying about overflows.
   */
  if (d_w > 0x08000000 || d_h > 0x08000000 || buf_align > 65536 ||
      stride_align > 65536) {
    goto fail;
  }

  /* Treat align==0 like align==1 */
  if (!buf_align) buf_align = 1;

  /* Validate alignment (must be power of 2) */
  if (buf_align & (buf_align - 1)) goto fail;

  /* Treat align==0 like align==1 */
  if (!stride_align) stride_align = 1;

  /* Validate alignment (must be power of 2) */
  if (stride_align & (stride_align - 1)) goto fail;

  /* Get sample size for this format */
  switch (fmt) {
    case VPX_IMG_FMT_I420:
    case VPX_IMG_FMT_YV12:
    case VPX_IMG_FMT_NV12: bps = 12; break;
    case VPX_IMG_FMT_I422:
    case VPX_IMG_FMT_I440: bps = 16; break;
    case VPX_IMG_FMT_I444: bps = 24; break;
    case VPX_IMG_FMT_I42016: bps = 24; break;
    case VPX_IMG_FMT_I42216:
    case VPX_IMG_FMT_I44016: bps = 32; break;
    case VPX_IMG_FMT_I44416: bps = 48; break;
    default: bps = 16; break;
  }

  /* Get chroma shift values for this format */
  // For VPX_IMG_FMT_NV12, xcs needs to be 0 such that UV data is all read at
  // once.
  switch (fmt) {
    case VPX_IMG_FMT_I420:
    case VPX_IMG_FMT_YV12:
    case VPX_IMG_FMT_I422:
    case VPX_IMG_FMT_I42016:
    case VPX_IMG_FMT_I42216: xcs = 1; break;
    default: xcs = 0; break;
  }

  switch (fmt) {
    case VPX_IMG_FMT_I420:
    case VPX_IMG_FMT_NV12:
    case VPX_IMG_FMT_I440:
    case VPX_IMG_FMT_YV12:
    case VPX_IMG_FMT_I42016:
    case VPX_IMG_FMT_I44016: ycs = 1; break;
    default: ycs = 0; break;
  }

  /* Calculate storage sizes. */
  if (img_data) {
    /* If the buffer was allocated externally, the width and height shouldn't
     * be adjusted. */
    w = d_w;
    h = d_h;
  } else {
    /* Calculate storage sizes given the chroma subsampling */
    align = (1 << xcs) - 1;
    w = (d_w + align) & ~align;
    assert(d_w <= w);
    align = (1 << ycs) - 1;
    h = (d_h + align) & ~align;
    assert(d_h <= h);
  }

  s = (fmt & VPX_IMG_FMT_PLANAR) ? w : (uint64_t)bps * w / 8;
  s = (fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? s * 2 : s;
  s = (s + stride_align - 1) & ~((uint64_t)stride_align - 1);
  if (s > INT_MAX) goto fail;
  stride_in_bytes = (int)s;
  s = (fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? s / 2 : s;

  /* Allocate the new image */
  if (!img) {
    img = (vpx_image_t *)calloc(1, sizeof(vpx_image_t));

    if (!img) goto fail;

    img->self_allocd = 1;
  }

  img->img_data = img_data;

  if (!img_data) {
    uint64_t alloc_size;
    alloc_size = (fmt & VPX_IMG_FMT_PLANAR) ? (uint64_t)h * s * bps / 8
                                            : (uint64_t)h * s;

    if (alloc_size != (size_t)alloc_size) goto fail;

    img->img_data = (uint8_t *)vpx_memalign(buf_align, (size_t)alloc_size);
    img->img_data_owner = 1;
  }

  if (!img->img_data) goto fail;

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

  /* Default viewport to entire image. (This vpx_img_set_rect call always
   * succeeds.) */
  int ret = vpx_img_set_rect(img, 0, 0, d_w, d_h);
  assert(ret == 0);
  (void)ret;
  return img;

fail:
  vpx_img_free(img);
  return NULL;
}

vpx_image_t *vpx_img_alloc(vpx_image_t *img, vpx_img_fmt_t fmt,
                           unsigned int d_w, unsigned int d_h,
                           unsigned int align) {
  return img_alloc_helper(img, fmt, d_w, d_h, align, align, NULL);
}

vpx_image_t *vpx_img_wrap(vpx_image_t *img, vpx_img_fmt_t fmt, unsigned int d_w,
                          unsigned int d_h, unsigned int stride_align,
                          unsigned char *img_data) {
  /* Set buf_align = 1. It is ignored by img_alloc_helper because img_data is
   * not NULL. */
  return img_alloc_helper(img, fmt, d_w, d_h, 1, stride_align, img_data);
}

int vpx_img_set_rect(vpx_image_t *img, unsigned int x, unsigned int y,
                     unsigned int w, unsigned int h) {
  if (x <= UINT_MAX - w && x + w <= img->w && y <= UINT_MAX - h &&
      y + h <= img->h) {
    img->d_w = w;
    img->d_h = h;

    /* Calculate plane pointers */
    if (!(img->fmt & VPX_IMG_FMT_PLANAR)) {
      img->planes[VPX_PLANE_PACKED] =
          img->img_data + x * img->bps / 8 + y * img->stride[VPX_PLANE_PACKED];
    } else {
      const int bytes_per_sample =
          (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;
      unsigned char *data = img->img_data;

      if (img->fmt & VPX_IMG_FMT_HAS_ALPHA) {
        img->planes[VPX_PLANE_ALPHA] =
            data + x * bytes_per_sample + y * img->stride[VPX_PLANE_ALPHA];
        data += (size_t)img->h * img->stride[VPX_PLANE_ALPHA];
      }

      img->planes[VPX_PLANE_Y] =
          data + x * bytes_per_sample + y * img->stride[VPX_PLANE_Y];
      data += (size_t)img->h * img->stride[VPX_PLANE_Y];

      unsigned int uv_x = x >> img->x_chroma_shift;
      unsigned int uv_y = y >> img->y_chroma_shift;
      if (img->fmt == VPX_IMG_FMT_NV12) {
        img->planes[VPX_PLANE_U] =
            data + uv_x + uv_y * img->stride[VPX_PLANE_U];
        img->planes[VPX_PLANE_V] = img->planes[VPX_PLANE_U] + 1;
      } else if (!(img->fmt & VPX_IMG_FMT_UV_FLIP)) {
        img->planes[VPX_PLANE_U] =
            data + uv_x * bytes_per_sample + uv_y * img->stride[VPX_PLANE_U];
        data +=
            (size_t)(img->h >> img->y_chroma_shift) * img->stride[VPX_PLANE_U];
        img->planes[VPX_PLANE_V] =
            data + uv_x * bytes_per_sample + uv_y * img->stride[VPX_PLANE_V];
      } else {
        img->planes[VPX_PLANE_V] =
            data + uv_x * bytes_per_sample + uv_y * img->stride[VPX_PLANE_V];
        data +=
            (size_t)(img->h >> img->y_chroma_shift) * img->stride[VPX_PLANE_V];
        img->planes[VPX_PLANE_U] =
            data + uv_x * bytes_per_sample + uv_y * img->stride[VPX_PLANE_U];
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

  img->planes[VPX_PLANE_U] += (signed)((img->d_h >> img->y_chroma_shift) - 1) *
                              img->stride[VPX_PLANE_U];
  img->stride[VPX_PLANE_U] = -img->stride[VPX_PLANE_U];

  img->planes[VPX_PLANE_V] += (signed)((img->d_h >> img->y_chroma_shift) - 1) *
                              img->stride[VPX_PLANE_V];
  img->stride[VPX_PLANE_V] = -img->stride[VPX_PLANE_V];

  img->planes[VPX_PLANE_ALPHA] +=
      (signed)(img->d_h - 1) * img->stride[VPX_PLANE_ALPHA];
  img->stride[VPX_PLANE_ALPHA] = -img->stride[VPX_PLANE_ALPHA];
}

void vpx_img_free(vpx_image_t *img) {
  if (img) {
    if (img->img_data && img->img_data_owner) vpx_free(img->img_data);

    if (img->self_allocd) free(img);
  }
}
