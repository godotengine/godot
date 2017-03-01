// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// WebPPicture tools: alpha handling, etc.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>

#include "./vp8i_enc.h"
#include "../dsp/yuv.h"

static WEBP_INLINE uint32_t MakeARGB32(int r, int g, int b) {
  return (0xff000000u | (r << 16) | (g << 8) | b);
}

//------------------------------------------------------------------------------
// Helper: clean up fully transparent area to help compressibility.

#define SIZE 8
#define SIZE2 (SIZE / 2)
static int is_transparent_area(const uint8_t* ptr, int stride, int size) {
  int y, x;
  for (y = 0; y < size; ++y) {
    for (x = 0; x < size; ++x) {
      if (ptr[x]) {
        return 0;
      }
    }
    ptr += stride;
  }
  return 1;
}

static int is_transparent_argb_area(const uint32_t* ptr, int stride, int size) {
  int y, x;
  for (y = 0; y < size; ++y) {
    for (x = 0; x < size; ++x) {
      if (ptr[x] & 0xff000000u) {
        return 0;
      }
    }
    ptr += stride;
  }
  return 1;
}

static void flatten(uint8_t* ptr, int v, int stride, int size) {
  int y;
  for (y = 0; y < size; ++y) {
    memset(ptr, v, size);
    ptr += stride;
  }
}

static void flatten_argb(uint32_t* ptr, uint32_t v, int stride, int size) {
  int x, y;
  for (y = 0; y < size; ++y) {
    for (x = 0; x < size; ++x) ptr[x] = v;
    ptr += stride;
  }
}

void WebPCleanupTransparentArea(WebPPicture* pic) {
  int x, y, w, h;
  if (pic == NULL) return;
  w = pic->width / SIZE;
  h = pic->height / SIZE;

  // note: we ignore the left-overs on right/bottom
  if (pic->use_argb) {
    uint32_t argb_value = 0;
    for (y = 0; y < h; ++y) {
      int need_reset = 1;
      for (x = 0; x < w; ++x) {
        const int off = (y * pic->argb_stride + x) * SIZE;
        if (is_transparent_argb_area(pic->argb + off, pic->argb_stride, SIZE)) {
          if (need_reset) {
            argb_value = pic->argb[off];
            need_reset = 0;
          }
          flatten_argb(pic->argb + off, argb_value, pic->argb_stride, SIZE);
        } else {
          need_reset = 1;
        }
      }
    }
  } else {
    const uint8_t* const a_ptr = pic->a;
    int values[3] = { 0 };
    if (a_ptr == NULL) return;    // nothing to do
    for (y = 0; y < h; ++y) {
      int need_reset = 1;
      for (x = 0; x < w; ++x) {
        const int off_a = (y * pic->a_stride + x) * SIZE;
        const int off_y = (y * pic->y_stride + x) * SIZE;
        const int off_uv = (y * pic->uv_stride + x) * SIZE2;
        if (is_transparent_area(a_ptr + off_a, pic->a_stride, SIZE)) {
          if (need_reset) {
            values[0] = pic->y[off_y];
            values[1] = pic->u[off_uv];
            values[2] = pic->v[off_uv];
            need_reset = 0;
          }
          flatten(pic->y + off_y, values[0], pic->y_stride, SIZE);
          flatten(pic->u + off_uv, values[1], pic->uv_stride, SIZE2);
          flatten(pic->v + off_uv, values[2], pic->uv_stride, SIZE2);
        } else {
          need_reset = 1;
        }
      }
    }
  }
}

#undef SIZE
#undef SIZE2

void WebPCleanupTransparentAreaLossless(WebPPicture* const pic) {
  int x, y, w, h;
  uint32_t* argb;
  assert(pic != NULL && pic->use_argb);
  w = pic->width;
  h = pic->height;
  argb = pic->argb;

  for (y = 0; y < h; ++y) {
    for (x = 0; x < w; ++x) {
      if ((argb[x] & 0xff000000) == 0) {
        argb[x] = 0x00000000;
      }
    }
    argb += pic->argb_stride;
  }
}

//------------------------------------------------------------------------------
// Blend color and remove transparency info

#define BLEND(V0, V1, ALPHA) \
    ((((V0) * (255 - (ALPHA)) + (V1) * (ALPHA)) * 0x101) >> 16)
#define BLEND_10BIT(V0, V1, ALPHA) \
    ((((V0) * (1020 - (ALPHA)) + (V1) * (ALPHA)) * 0x101) >> 18)

void WebPBlendAlpha(WebPPicture* pic, uint32_t background_rgb) {
  const int red = (background_rgb >> 16) & 0xff;
  const int green = (background_rgb >> 8) & 0xff;
  const int blue = (background_rgb >> 0) & 0xff;
  int x, y;
  if (pic == NULL) return;
  if (!pic->use_argb) {
    const int uv_width = (pic->width >> 1);  // omit last pixel during u/v loop
    const int Y0 = VP8RGBToY(red, green, blue, YUV_HALF);
    // VP8RGBToU/V expects the u/v values summed over four pixels
    const int U0 = VP8RGBToU(4 * red, 4 * green, 4 * blue, 4 * YUV_HALF);
    const int V0 = VP8RGBToV(4 * red, 4 * green, 4 * blue, 4 * YUV_HALF);
    const int has_alpha = pic->colorspace & WEBP_CSP_ALPHA_BIT;
    if (!has_alpha || pic->a == NULL) return;    // nothing to do
    for (y = 0; y < pic->height; ++y) {
      // Luma blending
      uint8_t* const y_ptr = pic->y + y * pic->y_stride;
      uint8_t* const a_ptr = pic->a + y * pic->a_stride;
      for (x = 0; x < pic->width; ++x) {
        const int alpha = a_ptr[x];
        if (alpha < 0xff) {
          y_ptr[x] = BLEND(Y0, y_ptr[x], a_ptr[x]);
        }
      }
      // Chroma blending every even line
      if ((y & 1) == 0) {
        uint8_t* const u = pic->u + (y >> 1) * pic->uv_stride;
        uint8_t* const v = pic->v + (y >> 1) * pic->uv_stride;
        uint8_t* const a_ptr2 =
            (y + 1 == pic->height) ? a_ptr : a_ptr + pic->a_stride;
        for (x = 0; x < uv_width; ++x) {
          // Average four alpha values into a single blending weight.
          // TODO(skal): might lead to visible contouring. Can we do better?
          const int alpha =
              a_ptr[2 * x + 0] + a_ptr[2 * x + 1] +
              a_ptr2[2 * x + 0] + a_ptr2[2 * x + 1];
          u[x] = BLEND_10BIT(U0, u[x], alpha);
          v[x] = BLEND_10BIT(V0, v[x], alpha);
        }
        if (pic->width & 1) {   // rightmost pixel
          const int alpha = 2 * (a_ptr[2 * x + 0] + a_ptr2[2 * x + 0]);
          u[x] = BLEND_10BIT(U0, u[x], alpha);
          v[x] = BLEND_10BIT(V0, v[x], alpha);
        }
      }
      memset(a_ptr, 0xff, pic->width);
    }
  } else {
    uint32_t* argb = pic->argb;
    const uint32_t background = MakeARGB32(red, green, blue);
    for (y = 0; y < pic->height; ++y) {
      for (x = 0; x < pic->width; ++x) {
        const int alpha = (argb[x] >> 24) & 0xff;
        if (alpha != 0xff) {
          if (alpha > 0) {
            int r = (argb[x] >> 16) & 0xff;
            int g = (argb[x] >>  8) & 0xff;
            int b = (argb[x] >>  0) & 0xff;
            r = BLEND(red, r, alpha);
            g = BLEND(green, g, alpha);
            b = BLEND(blue, b, alpha);
            argb[x] = MakeARGB32(r, g, b);
          } else {
            argb[x] = background;
          }
        }
      }
      argb += pic->argb_stride;
    }
  }
}

#undef BLEND
#undef BLEND_10BIT

//------------------------------------------------------------------------------
