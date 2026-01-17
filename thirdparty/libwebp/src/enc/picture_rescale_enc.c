// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// WebPPicture tools: copy, crop, rescaling and view.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/webp/encode.h"

#include <assert.h>
#include <stdlib.h>

#include "src/webp/types.h"
#include "src/dsp/dsp.h"
#include "src/enc/vp8i_enc.h"

#if !defined(WEBP_REDUCE_SIZE)
#include "src/utils/rescaler_utils.h"
#include "src/utils/utils.h"
#endif  // !defined(WEBP_REDUCE_SIZE)

#define HALVE(x) (((x) + 1) >> 1)

// Grab the 'specs' (writer, *opaque, width, height...) from 'src' and copy them
// into 'dst'. Mark 'dst' as not owning any memory.
static void PictureGrabSpecs(const WebPPicture* const src,
                             WebPPicture* const dst) {
  assert(src != NULL && dst != NULL);
  *dst = *src;
  WebPPictureResetBuffers(dst);
}

//------------------------------------------------------------------------------

// Adjust top-left corner to chroma sample position.
static void SnapTopLeftPosition(const WebPPicture* const pic,
                                int* const left, int* const top) {
  if (!pic->use_argb) {
    *left &= ~1;
    *top &= ~1;
  }
}

// Adjust top-left corner and verify that the sub-rectangle is valid.
static int AdjustAndCheckRectangle(const WebPPicture* const pic,
                                   int* const left, int* const top,
                                   int width, int height) {
  SnapTopLeftPosition(pic, left, top);
  if ((*left) < 0 || (*top) < 0) return 0;
  if (width <= 0 || height <= 0) return 0;
  if ((*left) + width > pic->width) return 0;
  if ((*top) + height > pic->height) return 0;
  return 1;
}

#if !defined(WEBP_REDUCE_SIZE)
int WebPPictureCopy(const WebPPicture* src, WebPPicture* dst) {
  if (src == NULL || dst == NULL) return 0;
  if (src == dst) return 1;

  PictureGrabSpecs(src, dst);
  if (!WebPPictureAlloc(dst)) return 0;

  if (!src->use_argb) {
    WebPCopyPlane(src->y, src->y_stride,
                  dst->y, dst->y_stride, dst->width, dst->height);
    WebPCopyPlane(src->u, src->uv_stride, dst->u, dst->uv_stride,
                  HALVE(dst->width), HALVE(dst->height));
    WebPCopyPlane(src->v, src->uv_stride, dst->v, dst->uv_stride,
                  HALVE(dst->width), HALVE(dst->height));
    if (dst->a != NULL)  {
      WebPCopyPlane(src->a, src->a_stride,
                    dst->a, dst->a_stride, dst->width, dst->height);
    }
  } else {
    WebPCopyPlane((const uint8_t*)src->argb, 4 * src->argb_stride,
                  (uint8_t*)dst->argb, 4 * dst->argb_stride,
                  4 * dst->width, dst->height);
  }
  return 1;
}
#endif  // !defined(WEBP_REDUCE_SIZE)

int WebPPictureIsView(const WebPPicture* picture) {
  if (picture == NULL) return 0;
  if (picture->use_argb) {
    return (picture->memory_argb_ == NULL);
  }
  return (picture->memory_ == NULL);
}

int WebPPictureView(const WebPPicture* src,
                    int left, int top, int width, int height,
                    WebPPicture* dst) {
  if (src == NULL || dst == NULL) return 0;

  // verify rectangle position.
  if (!AdjustAndCheckRectangle(src, &left, &top, width, height)) return 0;

  if (src != dst) {  // beware of aliasing! We don't want to leak 'memory_'.
    PictureGrabSpecs(src, dst);
  }
  dst->width = width;
  dst->height = height;
  if (!src->use_argb) {
    dst->y = src->y + top * src->y_stride + left;
    dst->u = src->u + (top >> 1) * src->uv_stride + (left >> 1);
    dst->v = src->v + (top >> 1) * src->uv_stride + (left >> 1);
    dst->y_stride = src->y_stride;
    dst->uv_stride = src->uv_stride;
    if (src->a != NULL) {
      dst->a = src->a + top * src->a_stride + left;
      dst->a_stride = src->a_stride;
    }
  } else {
    dst->argb = src->argb + top * src->argb_stride + left;
    dst->argb_stride = src->argb_stride;
  }
  return 1;
}

#if !defined(WEBP_REDUCE_SIZE)
//------------------------------------------------------------------------------
// Picture cropping

int WebPPictureCrop(WebPPicture* pic,
                    int left, int top, int width, int height) {
  WebPPicture tmp;

  if (pic == NULL) return 0;
  if (!AdjustAndCheckRectangle(pic, &left, &top, width, height)) return 0;

  PictureGrabSpecs(pic, &tmp);
  tmp.width = width;
  tmp.height = height;
  if (!WebPPictureAlloc(&tmp)) {
    return WebPEncodingSetError(pic, tmp.error_code);
  }

  if (!pic->use_argb) {
    const int y_offset = top * pic->y_stride + left;
    const int uv_offset = (top / 2) * pic->uv_stride + left / 2;
    WebPCopyPlane(pic->y + y_offset, pic->y_stride,
                  tmp.y, tmp.y_stride, width, height);
    WebPCopyPlane(pic->u + uv_offset, pic->uv_stride,
                  tmp.u, tmp.uv_stride, HALVE(width), HALVE(height));
    WebPCopyPlane(pic->v + uv_offset, pic->uv_stride,
                  tmp.v, tmp.uv_stride, HALVE(width), HALVE(height));

    if (tmp.a != NULL) {
      const int a_offset = top * pic->a_stride + left;
      WebPCopyPlane(pic->a + a_offset, pic->a_stride,
                    tmp.a, tmp.a_stride, width, height);
    }
  } else {
    const uint8_t* const src =
        (const uint8_t*)(pic->argb + top * pic->argb_stride + left);
    WebPCopyPlane(src, pic->argb_stride * 4, (uint8_t*)tmp.argb,
                  tmp.argb_stride * 4, width * 4, height);
  }
  WebPPictureFree(pic);
  *pic = tmp;
  return 1;
}

//------------------------------------------------------------------------------
// Simple picture rescaler

static int RescalePlane(const uint8_t* src,
                        int src_width, int src_height, int src_stride,
                        uint8_t* dst,
                        int dst_width, int dst_height, int dst_stride,
                        rescaler_t* const work,
                        int num_channels) {
  WebPRescaler rescaler;
  int y = 0;
  if (!WebPRescalerInit(&rescaler, src_width, src_height,
                        dst, dst_width, dst_height, dst_stride,
                        num_channels, work)) {
    return 0;
  }
  while (y < src_height) {
    y += WebPRescalerImport(&rescaler, src_height - y,
                            src + y * src_stride, src_stride);
    WebPRescalerExport(&rescaler);
  }
  return 1;
}

static void AlphaMultiplyARGB(WebPPicture* const pic, int inverse) {
  assert(pic->argb != NULL);
  WebPMultARGBRows((uint8_t*)pic->argb, pic->argb_stride * sizeof(*pic->argb),
                   pic->width, pic->height, inverse);
}

static void AlphaMultiplyY(WebPPicture* const pic, int inverse) {
  if (pic->a != NULL) {
    WebPMultRows(pic->y, pic->y_stride, pic->a, pic->a_stride,
                 pic->width, pic->height, inverse);
  }
}

int WebPPictureRescale(WebPPicture* picture, int width, int height) {
  WebPPicture tmp;
  int prev_width, prev_height;
  rescaler_t* work;

  if (picture == NULL) return 0;
  prev_width = picture->width;
  prev_height = picture->height;
  if (!WebPRescalerGetScaledDimensions(
          prev_width, prev_height, &width, &height)) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_BAD_DIMENSION);
  }

  PictureGrabSpecs(picture, &tmp);
  tmp.width = width;
  tmp.height = height;
  if (!WebPPictureAlloc(&tmp)) {
    return WebPEncodingSetError(picture, tmp.error_code);
  }

  if (!picture->use_argb) {
    work = (rescaler_t*)WebPSafeMalloc(2ULL * width, sizeof(*work));
    if (work == NULL) {
      WebPPictureFree(&tmp);
      return WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
    }
    // If present, we need to rescale alpha first (for AlphaMultiplyY).
    if (picture->a != NULL) {
      WebPInitAlphaProcessing();
      if (!RescalePlane(picture->a, prev_width, prev_height, picture->a_stride,
                        tmp.a, width, height, tmp.a_stride, work, 1)) {
        return WebPEncodingSetError(picture, VP8_ENC_ERROR_BAD_DIMENSION);
      }
    }

    // We take transparency into account on the luma plane only. That's not
    // totally exact blending, but still is a good approximation.
    AlphaMultiplyY(picture, 0);
    if (!RescalePlane(picture->y, prev_width, prev_height, picture->y_stride,
                      tmp.y, width, height, tmp.y_stride, work, 1) ||
        !RescalePlane(picture->u, HALVE(prev_width), HALVE(prev_height),
                      picture->uv_stride, tmp.u, HALVE(width), HALVE(height),
                      tmp.uv_stride, work, 1) ||
        !RescalePlane(picture->v, HALVE(prev_width), HALVE(prev_height),
                      picture->uv_stride, tmp.v, HALVE(width), HALVE(height),
                      tmp.uv_stride, work, 1)) {
      return WebPEncodingSetError(picture, VP8_ENC_ERROR_BAD_DIMENSION);
    }
    AlphaMultiplyY(&tmp, 1);
  } else {
    work = (rescaler_t*)WebPSafeMalloc(2ULL * width * 4, sizeof(*work));
    if (work == NULL) {
      WebPPictureFree(&tmp);
      return WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
    }
    // In order to correctly interpolate colors, we need to apply the alpha
    // weighting first (black-matting), scale the RGB values, and remove
    // the premultiplication afterward (while preserving the alpha channel).
    WebPInitAlphaProcessing();
    AlphaMultiplyARGB(picture, 0);
    if (!RescalePlane((const uint8_t*)picture->argb, prev_width, prev_height,
                      picture->argb_stride * 4, (uint8_t*)tmp.argb, width,
                      height, tmp.argb_stride * 4, work, 4)) {
      return WebPEncodingSetError(picture, VP8_ENC_ERROR_BAD_DIMENSION);
    }
    AlphaMultiplyARGB(&tmp, 1);
  }
  WebPPictureFree(picture);
  WebPSafeFree(work);
  *picture = tmp;
  return 1;
}

#else  // defined(WEBP_REDUCE_SIZE)

int WebPPictureCopy(const WebPPicture* src, WebPPicture* dst) {
  (void)src;
  (void)dst;
  return 0;
}

int WebPPictureCrop(WebPPicture* pic,
                    int left, int top, int width, int height) {
  (void)pic;
  (void)left;
  (void)top;
  (void)width;
  (void)height;
  return 0;
}

int WebPPictureRescale(WebPPicture* pic, int width, int height) {
  (void)pic;
  (void)width;
  (void)height;
  return 0;
}
#endif  // !defined(WEBP_REDUCE_SIZE)
