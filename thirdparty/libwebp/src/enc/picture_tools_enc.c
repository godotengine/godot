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

#include "src/enc/vp8i_enc.h"
#include "src/dsp/yuv.h"

//------------------------------------------------------------------------------
// Helper: clean up fully transparent area to help compressibility.

#define SIZE 8
#define SIZE2 (SIZE / 2)
static int IsTransparentARGBArea(const uint32_t* ptr, int stride, int size) {
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

static void Flatten(uint8_t* ptr, int v, int stride, int size) {
  int y;
  for (y = 0; y < size; ++y) {
    memset(ptr, v, size);
    ptr += stride;
  }
}

static void FlattenARGB(uint32_t* ptr, uint32_t v, int stride, int size) {
  int x, y;
  for (y = 0; y < size; ++y) {
    for (x = 0; x < size; ++x) ptr[x] = v;
    ptr += stride;
  }
}

// Smoothen the luma components of transparent pixels. Return true if the whole
// block is transparent.
static int SmoothenBlock(const uint8_t* a_ptr, int a_stride, uint8_t* y_ptr,
                         int y_stride, int width, int height) {
  int sum = 0, count = 0;
  int x, y;
  const uint8_t* alpha_ptr = a_ptr;
  uint8_t* luma_ptr = y_ptr;
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      if (alpha_ptr[x] != 0) {
        ++count;
        sum += luma_ptr[x];
      }
    }
    alpha_ptr += a_stride;
    luma_ptr += y_stride;
  }
  if (count > 0 && count < width * height) {
    const uint8_t avg_u8 = (uint8_t)(sum / count);
    alpha_ptr = a_ptr;
    luma_ptr = y_ptr;
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        if (alpha_ptr[x] == 0) luma_ptr[x] = avg_u8;
      }
      alpha_ptr += a_stride;
      luma_ptr += y_stride;
    }
  }
  return (count == 0);
}

void WebPReplaceTransparentPixels(WebPPicture* const pic, uint32_t color) {
  if (pic != NULL && pic->use_argb) {
    int y = pic->height;
    uint32_t* argb = pic->argb;
    color &= 0xffffffu;   // force alpha=0
    WebPInitAlphaProcessing();
    while (y-- > 0) {
      WebPAlphaReplace(argb, pic->width, color);
      argb += pic->argb_stride;
    }
  }
}

void WebPCleanupTransparentArea(WebPPicture* pic) {
  int x, y, w, h;
  if (pic == NULL) return;
  w = pic->width / SIZE;
  h = pic->height / SIZE;

  // note: we ignore the left-overs on right/bottom, except for SmoothenBlock().
  if (pic->use_argb) {
    uint32_t argb_value = 0;
    for (y = 0; y < h; ++y) {
      int need_reset = 1;
      for (x = 0; x < w; ++x) {
        const int off = (y * pic->argb_stride + x) * SIZE;
        if (IsTransparentARGBArea(pic->argb + off, pic->argb_stride, SIZE)) {
          if (need_reset) {
            argb_value = pic->argb[off];
            need_reset = 0;
          }
          FlattenARGB(pic->argb + off, argb_value, pic->argb_stride, SIZE);
        } else {
          need_reset = 1;
        }
      }
    }
  } else {
    const int width = pic->width;
    const int height = pic->height;
    const int y_stride = pic->y_stride;
    const int uv_stride = pic->uv_stride;
    const int a_stride = pic->a_stride;
    uint8_t* y_ptr = pic->y;
    uint8_t* u_ptr = pic->u;
    uint8_t* v_ptr = pic->v;
    const uint8_t* a_ptr = pic->a;
    int values[3] = { 0 };
    if (a_ptr == NULL || y_ptr == NULL || u_ptr == NULL || v_ptr == NULL) {
      return;
    }
    for (y = 0; y + SIZE <= height; y += SIZE) {
      int need_reset = 1;
      for (x = 0; x + SIZE <= width; x += SIZE) {
        if (SmoothenBlock(a_ptr + x, a_stride, y_ptr + x, y_stride,
                          SIZE, SIZE)) {
          if (need_reset) {
            values[0] = y_ptr[x];
            values[1] = u_ptr[x >> 1];
            values[2] = v_ptr[x >> 1];
            need_reset = 0;
          }
          Flatten(y_ptr + x,        values[0], y_stride,  SIZE);
          Flatten(u_ptr + (x >> 1), values[1], uv_stride, SIZE2);
          Flatten(v_ptr + (x >> 1), values[2], uv_stride, SIZE2);
        } else {
          need_reset = 1;
        }
      }
      if (x < width) {
        SmoothenBlock(a_ptr + x, a_stride, y_ptr + x, y_stride,
                      width - x, SIZE);
      }
      a_ptr += SIZE * a_stride;
      y_ptr += SIZE * y_stride;
      u_ptr += SIZE2 * uv_stride;
      v_ptr += SIZE2 * uv_stride;
    }
    if (y < height) {
      const int sub_height = height - y;
      for (x = 0; x + SIZE <= width; x += SIZE) {
        SmoothenBlock(a_ptr + x, a_stride, y_ptr + x, y_stride,
                      SIZE, sub_height);
      }
      if (x < width) {
        SmoothenBlock(a_ptr + x, a_stride, y_ptr + x, y_stride,
                      width - x, sub_height);
      }
    }
  }
}

#undef SIZE
#undef SIZE2

//------------------------------------------------------------------------------
// Blend color and remove transparency info

#define BLEND(V0, V1, ALPHA) \
    ((((V0) * (255 - (ALPHA)) + (V1) * (ALPHA)) * 0x101 + 256) >> 16)
#define BLEND_10BIT(V0, V1, ALPHA) \
    ((((V0) * (1020 - (ALPHA)) + (V1) * (ALPHA)) * 0x101 + 1024) >> 18)

static WEBP_INLINE uint32_t MakeARGB32(int r, int g, int b) {
  return (0xff000000u | (r << 16) | (g << 8) | b);
}

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
    uint8_t* y_ptr = pic->y;
    uint8_t* u_ptr = pic->u;
    uint8_t* v_ptr = pic->v;
    uint8_t* a_ptr = pic->a;
    if (!has_alpha || a_ptr == NULL) return;    // nothing to do
    for (y = 0; y < pic->height; ++y) {
      // Luma blending
      for (x = 0; x < pic->width; ++x) {
        const uint8_t alpha = a_ptr[x];
        if (alpha < 0xff) {
          y_ptr[x] = BLEND(Y0, y_ptr[x], alpha);
        }
      }
      // Chroma blending every even line
      if ((y & 1) == 0) {
        uint8_t* const a_ptr2 =
            (y + 1 == pic->height) ? a_ptr : a_ptr + pic->a_stride;
        for (x = 0; x < uv_width; ++x) {
          // Average four alpha values into a single blending weight.
          // TODO(skal): might lead to visible contouring. Can we do better?
          const uint32_t alpha =
              a_ptr[2 * x + 0] + a_ptr[2 * x + 1] +
              a_ptr2[2 * x + 0] + a_ptr2[2 * x + 1];
          u_ptr[x] = BLEND_10BIT(U0, u_ptr[x], alpha);
          v_ptr[x] = BLEND_10BIT(V0, v_ptr[x], alpha);
        }
        if (pic->width & 1) {   // rightmost pixel
          const uint32_t alpha = 2 * (a_ptr[2 * x + 0] + a_ptr2[2 * x + 0]);
          u_ptr[x] = BLEND_10BIT(U0, u_ptr[x], alpha);
          v_ptr[x] = BLEND_10BIT(V0, v_ptr[x], alpha);
        }
      } else {
        u_ptr += pic->uv_stride;
        v_ptr += pic->uv_stride;
      }
      memset(a_ptr, 0xff, pic->width);  // reset alpha value to opaque
      a_ptr += pic->a_stride;
      y_ptr += pic->y_stride;
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
