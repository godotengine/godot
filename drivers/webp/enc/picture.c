// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// WebPPicture utils: colorspace conversion, crop, ...
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "./vp8enci.h"
#include "../utils/alpha_processing.h"
#include "../utils/random.h"
#include "../utils/rescaler.h"
#include "../utils/utils.h"
#include "../dsp/dsp.h"
#include "../dsp/yuv.h"

// Uncomment to disable gamma-compression during RGB->U/V averaging
#define USE_GAMMA_COMPRESSION

#define HALVE(x) (((x) + 1) >> 1)
#define IS_YUV_CSP(csp, YUV_CSP) (((csp) & WEBP_CSP_UV_MASK) == (YUV_CSP))

static const union {
  uint32_t argb;
  uint8_t  bytes[4];
} test_endian = { 0xff000000u };
#define ALPHA_IS_LAST (test_endian.bytes[3] == 0xff)

static WEBP_INLINE uint32_t MakeARGB32(int r, int g, int b) {
  return (0xff000000u | (r << 16) | (g << 8) | b);
}

//------------------------------------------------------------------------------
// WebPPicture
//------------------------------------------------------------------------------

int WebPPictureAlloc(WebPPicture* picture) {
  if (picture != NULL) {
    const WebPEncCSP uv_csp = picture->colorspace & WEBP_CSP_UV_MASK;
    const int has_alpha = picture->colorspace & WEBP_CSP_ALPHA_BIT;
    const int width = picture->width;
    const int height = picture->height;

    if (!picture->use_argb) {
      const int y_stride = width;
      const int uv_width = HALVE(width);
      const int uv_height = HALVE(height);
      const int uv_stride = uv_width;
      int uv0_stride = 0;
      int a_width, a_stride;
      uint64_t y_size, uv_size, uv0_size, a_size, total_size;
      uint8_t* mem;

      // U/V
      switch (uv_csp) {
        case WEBP_YUV420:
          break;
#ifdef WEBP_EXPERIMENTAL_FEATURES
        case WEBP_YUV400:    // for now, we'll just reset the U/V samples
          break;
        case WEBP_YUV422:
          uv0_stride = uv_width;
          break;
        case WEBP_YUV444:
          uv0_stride = width;
          break;
#endif
        default:
          return 0;
      }
      uv0_size = height * uv0_stride;

      // alpha
      a_width = has_alpha ? width : 0;
      a_stride = a_width;
      y_size = (uint64_t)y_stride * height;
      uv_size = (uint64_t)uv_stride * uv_height;
      a_size =  (uint64_t)a_stride * height;

      total_size = y_size + a_size + 2 * uv_size + 2 * uv0_size;

      // Security and validation checks
      if (width <= 0 || height <= 0 ||         // luma/alpha param error
          uv_width < 0 || uv_height < 0) {     // u/v param error
        return 0;
      }
      // Clear previous buffer and allocate a new one.
      WebPPictureFree(picture);   // erase previous buffer
      mem = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*mem));
      if (mem == NULL) return 0;

      // From now on, we're in the clear, we can no longer fail...
      picture->memory_ = (void*)mem;
      picture->y_stride  = y_stride;
      picture->uv_stride = uv_stride;
      picture->a_stride  = a_stride;
      picture->uv0_stride = uv0_stride;
      // TODO(skal): we could align the y/u/v planes and adjust stride.
      picture->y = mem;
      mem += y_size;

      picture->u = mem;
      mem += uv_size;
      picture->v = mem;
      mem += uv_size;

      if (a_size) {
        picture->a = mem;
        mem += a_size;
      }
      if (uv0_size) {
        picture->u0 = mem;
        mem += uv0_size;
        picture->v0 = mem;
        mem += uv0_size;
      }
      (void)mem;  // makes the static analyzer happy
    } else {
      void* memory;
      const uint64_t argb_size = (uint64_t)width * height;
      if (width <= 0 || height <= 0) {
        return 0;
      }
      // Clear previous buffer and allocate a new one.
      WebPPictureFree(picture);   // erase previous buffer
      memory = WebPSafeMalloc(argb_size, sizeof(*picture->argb));
      if (memory == NULL) return 0;

      // TODO(skal): align plane to cache line?
      picture->memory_argb_ = memory;
      picture->argb = (uint32_t*)memory;
      picture->argb_stride = width;
    }
  }
  return 1;
}

// Remove reference to the ARGB buffer (doesn't free anything).
static void PictureResetARGB(WebPPicture* const picture) {
  picture->memory_argb_ = NULL;
  picture->argb = NULL;
  picture->argb_stride = 0;
}

// Remove reference to the YUVA buffer (doesn't free anything).
static void PictureResetYUVA(WebPPicture* const picture) {
  picture->memory_ = NULL;
  picture->y = picture->u = picture->v = picture->a = NULL;
  picture->u0 = picture->v0 = NULL;
  picture->y_stride = picture->uv_stride = 0;
  picture->a_stride = 0;
  picture->uv0_stride = 0;
}

// Grab the 'specs' (writer, *opaque, width, height...) from 'src' and copy them
// into 'dst'. Mark 'dst' as not owning any memory.
static void WebPPictureGrabSpecs(const WebPPicture* const src,
                                 WebPPicture* const dst) {
  assert(src != NULL && dst != NULL);
  *dst = *src;
  PictureResetYUVA(dst);
  PictureResetARGB(dst);
}

// Allocate a new argb buffer, discarding any existing one and preserving
// the other YUV(A) buffer.
static int PictureAllocARGB(WebPPicture* const picture) {
  WebPPicture tmp;
  free(picture->memory_argb_);
  PictureResetARGB(picture);
  picture->use_argb = 1;
  WebPPictureGrabSpecs(picture, &tmp);
  if (!WebPPictureAlloc(&tmp)) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }
  picture->memory_argb_ = tmp.memory_argb_;
  picture->argb = tmp.argb;
  picture->argb_stride = tmp.argb_stride;
  return 1;
}

// Release memory owned by 'picture' (both YUV and ARGB buffers).
void WebPPictureFree(WebPPicture* picture) {
  if (picture != NULL) {
    free(picture->memory_);
    free(picture->memory_argb_);
    PictureResetYUVA(picture);
    PictureResetARGB(picture);
  }
}

//------------------------------------------------------------------------------
// Picture copying

// Not worth moving to dsp/enc.c (only used here).
static void CopyPlane(const uint8_t* src, int src_stride,
                      uint8_t* dst, int dst_stride, int width, int height) {
  while (height-- > 0) {
    memcpy(dst, src, width);
    src += src_stride;
    dst += dst_stride;
  }
}

// Adjust top-left corner to chroma sample position.
static void SnapTopLeftPosition(const WebPPicture* const pic,
                                int* const left, int* const top) {
  if (!pic->use_argb) {
    const int is_yuv422 = IS_YUV_CSP(pic->colorspace, WEBP_YUV422);
    if (IS_YUV_CSP(pic->colorspace, WEBP_YUV420) || is_yuv422) {
      *left &= ~1;
      if (!is_yuv422) *top &= ~1;
    }
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

int WebPPictureCopy(const WebPPicture* src, WebPPicture* dst) {
  if (src == NULL || dst == NULL) return 0;
  if (src == dst) return 1;

  WebPPictureGrabSpecs(src, dst);
  if (!WebPPictureAlloc(dst)) return 0;

  if (!src->use_argb) {
    CopyPlane(src->y, src->y_stride,
              dst->y, dst->y_stride, dst->width, dst->height);
    CopyPlane(src->u, src->uv_stride,
              dst->u, dst->uv_stride, HALVE(dst->width), HALVE(dst->height));
    CopyPlane(src->v, src->uv_stride,
              dst->v, dst->uv_stride, HALVE(dst->width), HALVE(dst->height));
    if (dst->a != NULL)  {
      CopyPlane(src->a, src->a_stride,
                dst->a, dst->a_stride, dst->width, dst->height);
    }
#ifdef WEBP_EXPERIMENTAL_FEATURES
    if (dst->u0 != NULL)  {
      int uv0_width = src->width;
      if (IS_YUV_CSP(dst->colorspace, WEBP_YUV422)) {
        uv0_width = HALVE(uv0_width);
      }
      CopyPlane(src->u0, src->uv0_stride,
                dst->u0, dst->uv0_stride, uv0_width, dst->height);
      CopyPlane(src->v0, src->uv0_stride,
                dst->v0, dst->uv0_stride, uv0_width, dst->height);
    }
#endif
  } else {
    CopyPlane((const uint8_t*)src->argb, 4 * src->argb_stride,
              (uint8_t*)dst->argb, 4 * dst->argb_stride,
              4 * dst->width, dst->height);
  }
  return 1;
}

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
    WebPPictureGrabSpecs(src, dst);
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
#ifdef WEBP_EXPERIMENTAL_FEATURES
    if (src->u0 != NULL) {
      const int left_pos =
          IS_YUV_CSP(dst->colorspace, WEBP_YUV422) ? (left >> 1) : left;
      dst->u0 = src->u0 + top * src->uv0_stride + left_pos;
      dst->v0 = src->v0 + top * src->uv0_stride + left_pos;
      dst->uv0_stride = src->uv0_stride;
    }
#endif
  } else {
    dst->argb = src->argb + top * src->argb_stride + left;
    dst->argb_stride = src->argb_stride;
  }
  return 1;
}

//------------------------------------------------------------------------------
// Picture cropping

int WebPPictureCrop(WebPPicture* pic,
                    int left, int top, int width, int height) {
  WebPPicture tmp;

  if (pic == NULL) return 0;
  if (!AdjustAndCheckRectangle(pic, &left, &top, width, height)) return 0;

  WebPPictureGrabSpecs(pic, &tmp);
  tmp.width = width;
  tmp.height = height;
  if (!WebPPictureAlloc(&tmp)) return 0;

  if (!pic->use_argb) {
    const int y_offset = top * pic->y_stride + left;
    const int uv_offset = (top / 2) * pic->uv_stride + left / 2;
    CopyPlane(pic->y + y_offset, pic->y_stride,
              tmp.y, tmp.y_stride, width, height);
    CopyPlane(pic->u + uv_offset, pic->uv_stride,
              tmp.u, tmp.uv_stride, HALVE(width), HALVE(height));
    CopyPlane(pic->v + uv_offset, pic->uv_stride,
              tmp.v, tmp.uv_stride, HALVE(width), HALVE(height));

    if (tmp.a != NULL) {
      const int a_offset = top * pic->a_stride + left;
      CopyPlane(pic->a + a_offset, pic->a_stride,
                tmp.a, tmp.a_stride, width, height);
    }
#ifdef WEBP_EXPERIMENTAL_FEATURES
    if (tmp.u0 != NULL) {
      int w = width;
      int left_pos = left;
      if (IS_YUV_CSP(tmp.colorspace, WEBP_YUV422)) {
        w = HALVE(w);
        left_pos = HALVE(left_pos);
      }
      CopyPlane(pic->u0 + top * pic->uv0_stride + left_pos, pic->uv0_stride,
                tmp.u0, tmp.uv0_stride, w, height);
      CopyPlane(pic->v0 + top * pic->uv0_stride + left_pos, pic->uv0_stride,
                tmp.v0, tmp.uv0_stride, w, height);
    }
#endif
  } else {
    const uint8_t* const src =
        (const uint8_t*)(pic->argb + top * pic->argb_stride + left);
    CopyPlane(src, pic->argb_stride * 4,
              (uint8_t*)tmp.argb, tmp.argb_stride * 4,
              width * 4, height);
  }
  WebPPictureFree(pic);
  *pic = tmp;
  return 1;
}

//------------------------------------------------------------------------------
// Simple picture rescaler

static void RescalePlane(const uint8_t* src,
                         int src_width, int src_height, int src_stride,
                         uint8_t* dst,
                         int dst_width, int dst_height, int dst_stride,
                         int32_t* const work,
                         int num_channels) {
  WebPRescaler rescaler;
  int y = 0;
  WebPRescalerInit(&rescaler, src_width, src_height,
                   dst, dst_width, dst_height, dst_stride,
                   num_channels,
                   src_width, dst_width,
                   src_height, dst_height,
                   work);
  memset(work, 0, 2 * dst_width * num_channels * sizeof(*work));
  while (y < src_height) {
    y += WebPRescalerImport(&rescaler, src_height - y,
                            src + y * src_stride, src_stride);
    WebPRescalerExport(&rescaler);
  }
}

static void AlphaMultiplyARGB(WebPPicture* const pic, int inverse) {
  uint32_t* ptr = pic->argb;
  int y;
  for (y = 0; y < pic->height; ++y) {
    WebPMultARGBRow(ptr, pic->width, inverse);
    ptr += pic->argb_stride;
  }
}

static void AlphaMultiplyY(WebPPicture* const pic, int inverse) {
  const uint8_t* ptr_a = pic->a;
  if (ptr_a != NULL) {
    uint8_t* ptr_y = pic->y;
    int y;
    for (y = 0; y < pic->height; ++y) {
      WebPMultRow(ptr_y, ptr_a, pic->width, inverse);
      ptr_y += pic->y_stride;
      ptr_a += pic->a_stride;
    }
  }
}

int WebPPictureRescale(WebPPicture* pic, int width, int height) {
  WebPPicture tmp;
  int prev_width, prev_height;
  int32_t* work;

  if (pic == NULL) return 0;
  prev_width = pic->width;
  prev_height = pic->height;
  // if width is unspecified, scale original proportionally to height ratio.
  if (width == 0) {
    width = (prev_width * height + prev_height / 2) / prev_height;
  }
  // if height is unspecified, scale original proportionally to width ratio.
  if (height == 0) {
    height = (prev_height * width + prev_width / 2) / prev_width;
  }
  // Check if the overall dimensions still make sense.
  if (width <= 0 || height <= 0) return 0;

  WebPPictureGrabSpecs(pic, &tmp);
  tmp.width = width;
  tmp.height = height;
  if (!WebPPictureAlloc(&tmp)) return 0;

  if (!pic->use_argb) {
    work = (int32_t*)WebPSafeMalloc(2ULL * width, sizeof(*work));
    if (work == NULL) {
      WebPPictureFree(&tmp);
      return 0;
    }
    // If present, we need to rescale alpha first (for AlphaMultiplyY).
    if (pic->a != NULL) {
      RescalePlane(pic->a, prev_width, prev_height, pic->a_stride,
                   tmp.a, width, height, tmp.a_stride, work, 1);
    }

    // We take transparency into account on the luma plane only. That's not
    // totally exact blending, but still is a good approximation.
    AlphaMultiplyY(pic, 0);
    RescalePlane(pic->y, prev_width, prev_height, pic->y_stride,
                 tmp.y, width, height, tmp.y_stride, work, 1);
    AlphaMultiplyY(&tmp, 1);

    RescalePlane(pic->u,
                 HALVE(prev_width), HALVE(prev_height), pic->uv_stride,
                 tmp.u,
                 HALVE(width), HALVE(height), tmp.uv_stride, work, 1);
    RescalePlane(pic->v,
                 HALVE(prev_width), HALVE(prev_height), pic->uv_stride,
                 tmp.v,
                 HALVE(width), HALVE(height), tmp.uv_stride, work, 1);

#ifdef WEBP_EXPERIMENTAL_FEATURES
    if (tmp.u0 != NULL) {
      const int s = IS_YUV_CSP(tmp.colorspace, WEBP_YUV422) ? 2 : 1;
      RescalePlane(
          pic->u0, (prev_width + s / 2) / s, prev_height, pic->uv0_stride,
          tmp.u0, (width + s / 2) / s, height, tmp.uv0_stride, work, 1);
      RescalePlane(
          pic->v0, (prev_width + s / 2) / s, prev_height, pic->uv0_stride,
          tmp.v0, (width + s / 2) / s, height, tmp.uv0_stride, work, 1);
    }
#endif
  } else {
    work = (int32_t*)WebPSafeMalloc(2ULL * width * 4, sizeof(*work));
    if (work == NULL) {
      WebPPictureFree(&tmp);
      return 0;
    }
    // In order to correctly interpolate colors, we need to apply the alpha
    // weighting first (black-matting), scale the RGB values, and remove
    // the premultiplication afterward (while preserving the alpha channel).
    AlphaMultiplyARGB(pic, 0);
    RescalePlane((const uint8_t*)pic->argb, prev_width, prev_height,
                 pic->argb_stride * 4,
                 (uint8_t*)tmp.argb, width, height,
                 tmp.argb_stride * 4,
                 work, 4);
    AlphaMultiplyARGB(&tmp, 1);
  }
  WebPPictureFree(pic);
  free(work);
  *pic = tmp;
  return 1;
}

//------------------------------------------------------------------------------
// WebPMemoryWriter: Write-to-memory

void WebPMemoryWriterInit(WebPMemoryWriter* writer) {
  writer->mem = NULL;
  writer->size = 0;
  writer->max_size = 0;
}

int WebPMemoryWrite(const uint8_t* data, size_t data_size,
                    const WebPPicture* picture) {
  WebPMemoryWriter* const w = (WebPMemoryWriter*)picture->custom_ptr;
  uint64_t next_size;
  if (w == NULL) {
    return 1;
  }
  next_size = (uint64_t)w->size + data_size;
  if (next_size > w->max_size) {
    uint8_t* new_mem;
    uint64_t next_max_size = 2ULL * w->max_size;
    if (next_max_size < next_size) next_max_size = next_size;
    if (next_max_size < 8192ULL) next_max_size = 8192ULL;
    new_mem = (uint8_t*)WebPSafeMalloc(next_max_size, 1);
    if (new_mem == NULL) {
      return 0;
    }
    if (w->size > 0) {
      memcpy(new_mem, w->mem, w->size);
    }
    free(w->mem);
    w->mem = new_mem;
    // down-cast is ok, thanks to WebPSafeMalloc
    w->max_size = (size_t)next_max_size;
  }
  if (data_size > 0) {
    memcpy(w->mem + w->size, data, data_size);
    w->size += data_size;
  }
  return 1;
}

//------------------------------------------------------------------------------
// Detection of non-trivial transparency

// Returns true if alpha[] has non-0xff values.
static int CheckNonOpaque(const uint8_t* alpha, int width, int height,
                          int x_step, int y_step) {
  if (alpha == NULL) return 0;
  while (height-- > 0) {
    int x;
    for (x = 0; x < width * x_step; x += x_step) {
      if (alpha[x] != 0xff) return 1;  // TODO(skal): check 4/8 bytes at a time.
    }
    alpha += y_step;
  }
  return 0;
}

// Checking for the presence of non-opaque alpha.
int WebPPictureHasTransparency(const WebPPicture* picture) {
  if (picture == NULL) return 0;
  if (!picture->use_argb) {
    return CheckNonOpaque(picture->a, picture->width, picture->height,
                          1, picture->a_stride);
  } else {
    int x, y;
    const uint32_t* argb = picture->argb;
    if (argb == NULL) return 0;
    for (y = 0; y < picture->height; ++y) {
      for (x = 0; x < picture->width; ++x) {
        if (argb[x] < 0xff000000u) return 1;   // test any alpha values != 0xff
      }
      argb += picture->argb_stride;
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// RGB -> YUV conversion

static int RGBToY(int r, int g, int b, VP8Random* const rg) {
  return VP8RGBToY(r, g, b, VP8RandomBits(rg, YUV_FIX));
}

static int RGBToU(int r, int g, int b, VP8Random* const rg) {
  return VP8RGBToU(r, g, b, VP8RandomBits(rg, YUV_FIX + 2));
}

static int RGBToV(int r, int g, int b, VP8Random* const rg) {
  return VP8RGBToV(r, g, b, VP8RandomBits(rg, YUV_FIX + 2));
}

//------------------------------------------------------------------------------

#if defined(USE_GAMMA_COMPRESSION)

// gamma-compensates loss of resolution during chroma subsampling
#define kGamma 0.80
#define kGammaFix 12     // fixed-point precision for linear values
#define kGammaScale ((1 << kGammaFix) - 1)
#define kGammaTabFix 7   // fixed-point fractional bits precision
#define kGammaTabScale (1 << kGammaTabFix)
#define kGammaTabRounder (kGammaTabScale >> 1)
#define kGammaTabSize (1 << (kGammaFix - kGammaTabFix))

static int kLinearToGammaTab[kGammaTabSize + 1];
static uint16_t kGammaToLinearTab[256];
static int kGammaTablesOk = 0;

static void InitGammaTables(void) {
  if (!kGammaTablesOk) {
    int v;
    const double scale = 1. / kGammaScale;
    for (v = 0; v <= 255; ++v) {
      kGammaToLinearTab[v] =
          (uint16_t)(pow(v / 255., kGamma) * kGammaScale + .5);
    }
    for (v = 0; v <= kGammaTabSize; ++v) {
      const double x = scale * (v << kGammaTabFix);
      kLinearToGammaTab[v] = (int)(pow(x, 1. / kGamma) * 255. + .5);
    }
    kGammaTablesOk = 1;
  }
}

static WEBP_INLINE uint32_t GammaToLinear(uint8_t v) {
  return kGammaToLinearTab[v];
}

// Convert a linear value 'v' to YUV_FIX+2 fixed-point precision
// U/V value, suitable for RGBToU/V calls.
static WEBP_INLINE int LinearToGamma(uint32_t base_value, int shift) {
  const int v = base_value << shift;              // final uplifted value
  const int tab_pos = v >> (kGammaTabFix + 2);    // integer part
  const int x = v & ((kGammaTabScale << 2) - 1);  // fractional part
  const int v0 = kLinearToGammaTab[tab_pos];
  const int v1 = kLinearToGammaTab[tab_pos + 1];
  const int y = v1 * x + v0 * ((kGammaTabScale << 2) - x);   // interpolate
  return (y + kGammaTabRounder) >> kGammaTabFix;             // descale
}

#else

static void InitGammaTables(void) {}
static WEBP_INLINE uint32_t GammaToLinear(uint8_t v) { return v; }
static WEBP_INLINE int LinearToGamma(uint32_t base_value, int shift) {
  (void)shift;
  return v;
}

#endif    // USE_GAMMA_COMPRESSION

//------------------------------------------------------------------------------

#define SUM4(ptr) LinearToGamma(                         \
    GammaToLinear((ptr)[0]) +                            \
    GammaToLinear((ptr)[step]) +                         \
    GammaToLinear((ptr)[rgb_stride]) +                   \
    GammaToLinear((ptr)[rgb_stride + step]), 0)          \

#define SUM2H(ptr) \
    LinearToGamma(GammaToLinear((ptr)[0]) + GammaToLinear((ptr)[step]), 1)
#define SUM2V(ptr) \
    LinearToGamma(GammaToLinear((ptr)[0]) + GammaToLinear((ptr)[rgb_stride]), 1)
#define SUM1(ptr)  \
    LinearToGamma(GammaToLinear((ptr)[0]), 2)

#define RGB_TO_UV(x, y, SUM) {                           \
  const int src = (2 * (step * (x) + (y) * rgb_stride)); \
  const int dst = (x) + (y) * picture->uv_stride;        \
  const int r = SUM(r_ptr + src);                        \
  const int g = SUM(g_ptr + src);                        \
  const int b = SUM(b_ptr + src);                        \
  picture->u[dst] = RGBToU(r, g, b, &rg);                \
  picture->v[dst] = RGBToV(r, g, b, &rg);                \
}

#define RGB_TO_UV0(x_in, x_out, y, SUM) {                \
  const int src = (step * (x_in) + (y) * rgb_stride);    \
  const int dst = (x_out) + (y) * picture->uv0_stride;   \
  const int r = SUM(r_ptr + src);                        \
  const int g = SUM(g_ptr + src);                        \
  const int b = SUM(b_ptr + src);                        \
  picture->u0[dst] = RGBToU(r, g, b, &rg);               \
  picture->v0[dst] = RGBToV(r, g, b, &rg);               \
}

static void MakeGray(WebPPicture* const picture) {
  int y;
  const int uv_width = HALVE(picture->width);
  const int uv_height = HALVE(picture->height);
  for (y = 0; y < uv_height; ++y) {
    memset(picture->u + y * picture->uv_stride, 128, uv_width);
    memset(picture->v + y * picture->uv_stride, 128, uv_width);
  }
}

static int ImportYUVAFromRGBA(const uint8_t* const r_ptr,
                              const uint8_t* const g_ptr,
                              const uint8_t* const b_ptr,
                              const uint8_t* const a_ptr,
                              int step,         // bytes per pixel
                              int rgb_stride,   // bytes per scanline
                              float dithering,
                              WebPPicture* const picture) {
  const WebPEncCSP uv_csp = picture->colorspace & WEBP_CSP_UV_MASK;
  int x, y;
  const int width = picture->width;
  const int height = picture->height;
  const int has_alpha = CheckNonOpaque(a_ptr, width, height, step, rgb_stride);
  VP8Random rg;

  picture->colorspace = uv_csp;
  picture->use_argb = 0;
  if (has_alpha) {
    picture->colorspace |= WEBP_CSP_ALPHA_BIT;
  }
  if (!WebPPictureAlloc(picture)) return 0;

  VP8InitRandom(&rg, dithering);
  InitGammaTables();

  // Import luma plane
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      const int offset = step * x + y * rgb_stride;
      picture->y[x + y * picture->y_stride] =
          RGBToY(r_ptr[offset], g_ptr[offset], b_ptr[offset], &rg);
    }
  }

  // Downsample U/V plane
  if (uv_csp != WEBP_YUV400) {
    for (y = 0; y < (height >> 1); ++y) {
      for (x = 0; x < (width >> 1); ++x) {
        RGB_TO_UV(x, y, SUM4);
      }
      if (width & 1) {
        RGB_TO_UV(x, y, SUM2V);
      }
    }
    if (height & 1) {
      for (x = 0; x < (width >> 1); ++x) {
        RGB_TO_UV(x, y, SUM2H);
      }
      if (width & 1) {
        RGB_TO_UV(x, y, SUM1);
      }
    }

#ifdef WEBP_EXPERIMENTAL_FEATURES
    // Store original U/V samples too
    if (uv_csp == WEBP_YUV422) {
      for (y = 0; y < height; ++y) {
        for (x = 0; x < (width >> 1); ++x) {
          RGB_TO_UV0(2 * x, x, y, SUM2H);
        }
        if (width & 1) {
          RGB_TO_UV0(2 * x, x, y, SUM1);
        }
      }
    } else if (uv_csp == WEBP_YUV444) {
      for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
          RGB_TO_UV0(x, x, y, SUM1);
        }
      }
    }
#endif
  } else {
    MakeGray(picture);
  }

  if (has_alpha) {
    assert(step >= 4);
    assert(picture->a != NULL);
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        picture->a[x + y * picture->a_stride] =
            a_ptr[step * x + y * rgb_stride];
      }
    }
  }
  return 1;
}

static int Import(WebPPicture* const picture,
                  const uint8_t* const rgb, int rgb_stride,
                  int step, int swap_rb, int import_alpha) {
  const uint8_t* const r_ptr = rgb + (swap_rb ? 2 : 0);
  const uint8_t* const g_ptr = rgb + 1;
  const uint8_t* const b_ptr = rgb + (swap_rb ? 0 : 2);
  const uint8_t* const a_ptr = import_alpha ? rgb + 3 : NULL;
  const int width = picture->width;
  const int height = picture->height;

  if (!picture->use_argb) {
    return ImportYUVAFromRGBA(r_ptr, g_ptr, b_ptr, a_ptr, step, rgb_stride,
                              0.f /* no dithering */, picture);
  }
  if (import_alpha) {
    picture->colorspace |= WEBP_CSP_ALPHA_BIT;
  } else {
    picture->colorspace &= ~WEBP_CSP_ALPHA_BIT;
  }
  if (!WebPPictureAlloc(picture)) return 0;

  if (!import_alpha) {
    int x, y;
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        const int offset = step * x + y * rgb_stride;
        const uint32_t argb =
            MakeARGB32(r_ptr[offset], g_ptr[offset], b_ptr[offset]);
        picture->argb[x + y * picture->argb_stride] = argb;
      }
    }
  } else {
    int x, y;
    assert(step >= 4);
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        const int offset = step * x + y * rgb_stride;
        const uint32_t argb = ((uint32_t)a_ptr[offset] << 24) |
                              (r_ptr[offset] << 16) |
                              (g_ptr[offset] <<  8) |
                              (b_ptr[offset]);
        picture->argb[x + y * picture->argb_stride] = argb;
      }
    }
  }
  return 1;
}
#undef SUM4
#undef SUM2V
#undef SUM2H
#undef SUM1
#undef RGB_TO_UV

int WebPPictureImportRGB(WebPPicture* picture,
                         const uint8_t* rgb, int rgb_stride) {
  return Import(picture, rgb, rgb_stride, 3, 0, 0);
}

int WebPPictureImportBGR(WebPPicture* picture,
                         const uint8_t* rgb, int rgb_stride) {
  return Import(picture, rgb, rgb_stride, 3, 1, 0);
}

int WebPPictureImportRGBA(WebPPicture* picture,
                          const uint8_t* rgba, int rgba_stride) {
  return Import(picture, rgba, rgba_stride, 4, 0, 1);
}

int WebPPictureImportBGRA(WebPPicture* picture,
                          const uint8_t* rgba, int rgba_stride) {
  return Import(picture, rgba, rgba_stride, 4, 1, 1);
}

int WebPPictureImportRGBX(WebPPicture* picture,
                          const uint8_t* rgba, int rgba_stride) {
  return Import(picture, rgba, rgba_stride, 4, 0, 0);
}

int WebPPictureImportBGRX(WebPPicture* picture,
                          const uint8_t* rgba, int rgba_stride) {
  return Import(picture, rgba, rgba_stride, 4, 1, 0);
}

//------------------------------------------------------------------------------
// Automatic YUV <-> ARGB conversions.

int WebPPictureYUVAToARGB(WebPPicture* picture) {
  if (picture == NULL) return 0;
  if (picture->y == NULL || picture->u == NULL || picture->v == NULL) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_NULL_PARAMETER);
  }
  if ((picture->colorspace & WEBP_CSP_ALPHA_BIT) && picture->a == NULL) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_NULL_PARAMETER);
  }
  if ((picture->colorspace & WEBP_CSP_UV_MASK) != WEBP_YUV420) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_INVALID_CONFIGURATION);
  }
  // Allocate a new argb buffer (discarding the previous one).
  if (!PictureAllocARGB(picture)) return 0;

  // Convert
  {
    int y;
    const int width = picture->width;
    const int height = picture->height;
    const int argb_stride = 4 * picture->argb_stride;
    uint8_t* dst = (uint8_t*)picture->argb;
    const uint8_t *cur_u = picture->u, *cur_v = picture->v, *cur_y = picture->y;
    WebPUpsampleLinePairFunc upsample = WebPGetLinePairConverter(ALPHA_IS_LAST);

    // First row, with replicated top samples.
    upsample(cur_y, NULL, cur_u, cur_v, cur_u, cur_v, dst, NULL, width);
    cur_y += picture->y_stride;
    dst += argb_stride;
    // Center rows.
    for (y = 1; y + 1 < height; y += 2) {
      const uint8_t* const top_u = cur_u;
      const uint8_t* const top_v = cur_v;
      cur_u += picture->uv_stride;
      cur_v += picture->uv_stride;
      upsample(cur_y, cur_y + picture->y_stride, top_u, top_v, cur_u, cur_v,
               dst, dst + argb_stride, width);
      cur_y += 2 * picture->y_stride;
      dst += 2 * argb_stride;
    }
    // Last row (if needed), with replicated bottom samples.
    if (height > 1 && !(height & 1)) {
      upsample(cur_y, NULL, cur_u, cur_v, cur_u, cur_v, dst, NULL, width);
    }
    // Insert alpha values if needed, in replacement for the default 0xff ones.
    if (picture->colorspace & WEBP_CSP_ALPHA_BIT) {
      for (y = 0; y < height; ++y) {
        uint32_t* const argb_dst = picture->argb + y * picture->argb_stride;
        const uint8_t* const src = picture->a + y * picture->a_stride;
        int x;
        for (x = 0; x < width; ++x) {
          argb_dst[x] = (argb_dst[x] & 0x00ffffffu) | ((uint32_t)src[x] << 24);
        }
      }
    }
  }
  return 1;
}

int WebPPictureARGBToYUVADithered(WebPPicture* picture, WebPEncCSP colorspace,
                                  float dithering) {
  if (picture == NULL) return 0;
  if (picture->argb == NULL) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_NULL_PARAMETER);
  } else {
    const uint8_t* const argb = (const uint8_t*)picture->argb;
    const uint8_t* const r = ALPHA_IS_LAST ? argb + 2 : argb + 1;
    const uint8_t* const g = ALPHA_IS_LAST ? argb + 1 : argb + 2;
    const uint8_t* const b = ALPHA_IS_LAST ? argb + 0 : argb + 3;
    const uint8_t* const a = ALPHA_IS_LAST ? argb + 3 : argb + 0;
    // We work on a tmp copy of 'picture', because ImportYUVAFromRGBA()
    // would be calling WebPPictureFree(picture) otherwise.
    WebPPicture tmp = *picture;
    PictureResetARGB(&tmp);  // reset ARGB buffer so that it's not free()'d.
    tmp.use_argb = 0;
    tmp.colorspace = colorspace & WEBP_CSP_UV_MASK;
    if (!ImportYUVAFromRGBA(r, g, b, a, 4, 4 * picture->argb_stride, dithering,
                            &tmp)) {
      return WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
    }
    // Copy back the YUV specs into 'picture'.
    tmp.argb = picture->argb;
    tmp.argb_stride = picture->argb_stride;
    tmp.memory_argb_ = picture->memory_argb_;
    *picture = tmp;
  }
  return 1;
}

int WebPPictureARGBToYUVA(WebPPicture* picture, WebPEncCSP colorspace) {
  return WebPPictureARGBToYUVADithered(picture, colorspace, 0.f);
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

static WEBP_INLINE void flatten(uint8_t* ptr, int v, int stride, int size) {
  int y;
  for (y = 0; y < size; ++y) {
    memset(ptr, v, size);
    ptr += stride;
  }
}

void WebPCleanupTransparentArea(WebPPicture* pic) {
  int x, y, w, h;
  const uint8_t* a_ptr;
  int values[3] = { 0 };

  if (pic == NULL) return;

  a_ptr = pic->a;
  if (a_ptr == NULL) return;    // nothing to do

  w = pic->width / SIZE;
  h = pic->height / SIZE;
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
    // ignore the left-overs on right/bottom
  }
}

#undef SIZE
#undef SIZE2

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
  VP8Random rg;
  int x, y;
  if (pic == NULL) return;
  VP8InitRandom(&rg, 0.f);
  if (!pic->use_argb) {
    const int uv_width = (pic->width >> 1);  // omit last pixel during u/v loop
    const int Y0 = RGBToY(red, green, blue, &rg);
    // VP8RGBToU/V expects the u/v values summed over four pixels
    const int U0 = RGBToU(4 * red, 4 * green, 4 * blue, &rg);
    const int V0 = RGBToV(4 * red, 4 * green, 4 * blue, &rg);
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
// local-min distortion
//
// For every pixel in the *reference* picture, we search for the local best
// match in the compressed image. This is not a symmetrical measure.

// search radius. Shouldn't be too large.
#define RADIUS 2

static float AccumulateLSIM(const uint8_t* src, int src_stride,
                            const uint8_t* ref, int ref_stride,
                            int w, int h) {
  int x, y;
  double total_sse = 0.;
  for (y = 0; y < h; ++y) {
    const int y_0 = (y - RADIUS < 0) ? 0 : y - RADIUS;
    const int y_1 = (y + RADIUS + 1 >= h) ? h : y + RADIUS + 1;
    for (x = 0; x < w; ++x) {
      const int x_0 = (x - RADIUS < 0) ? 0 : x - RADIUS;
      const int x_1 = (x + RADIUS + 1 >= w) ? w : x + RADIUS + 1;
      double best_sse = 255. * 255.;
      const double value = (double)ref[y * ref_stride + x];
      int i, j;
      for (j = y_0; j < y_1; ++j) {
        const uint8_t* s = src + j * src_stride;
        for (i = x_0; i < x_1; ++i) {
          const double sse = (double)(s[i] - value) * (s[i] - value);
          if (sse < best_sse) best_sse = sse;
        }
      }
      total_sse += best_sse;
    }
  }
  return (float)total_sse;
}
#undef RADIUS

//------------------------------------------------------------------------------
// Distortion

// Max value returned in case of exact similarity.
static const double kMinDistortion_dB = 99.;
static float GetPSNR(const double v) {
  return (float)((v > 0.) ? -4.3429448 * log(v / (255 * 255.))
                          : kMinDistortion_dB);
}

int WebPPictureDistortion(const WebPPicture* src, const WebPPicture* ref,
                          int type, float result[5]) {
  DistoStats stats[5];
  int has_alpha;
  int uv_w, uv_h;

  if (src == NULL || ref == NULL ||
      src->width != ref->width || src->height != ref->height ||
      src->y == NULL || ref->y == NULL ||
      src->u == NULL || ref->u == NULL ||
      src->v == NULL || ref->v == NULL ||
      result == NULL) {
    return 0;
  }
  // TODO(skal): provide distortion for ARGB too.
  if (src->use_argb == 1 || src->use_argb != ref->use_argb) {
    return 0;
  }

  has_alpha = !!(src->colorspace & WEBP_CSP_ALPHA_BIT);
  if (has_alpha != !!(ref->colorspace & WEBP_CSP_ALPHA_BIT) ||
      (has_alpha && (src->a == NULL || ref->a == NULL))) {
    return 0;
  }

  memset(stats, 0, sizeof(stats));

  uv_w = HALVE(src->width);
  uv_h = HALVE(src->height);
  if (type >= 2) {
    float sse[4];
    sse[0] = AccumulateLSIM(src->y, src->y_stride,
                            ref->y, ref->y_stride, src->width, src->height);
    sse[1] = AccumulateLSIM(src->u, src->uv_stride,
                            ref->u, ref->uv_stride, uv_w, uv_h);
    sse[2] = AccumulateLSIM(src->v, src->uv_stride,
                            ref->v, ref->uv_stride, uv_w, uv_h);
    sse[3] = has_alpha ? AccumulateLSIM(src->a, src->a_stride,
                                        ref->a, ref->a_stride,
                                        src->width, src->height)
                       : 0.f;
    result[0] = GetPSNR(sse[0] / (src->width * src->height));
    result[1] = GetPSNR(sse[1] / (uv_w * uv_h));
    result[2] = GetPSNR(sse[2] / (uv_w * uv_h));
    result[3] = GetPSNR(sse[3] / (src->width * src->height));
    {
      double total_sse = sse[0] + sse[1] + sse[2];
      int total_pixels = src->width * src->height + 2 * uv_w * uv_h;
      if (has_alpha) {
        total_pixels += src->width * src->height;
        total_sse += sse[3];
      }
      result[4] = GetPSNR(total_sse / total_pixels);
    }
  } else {
    int c;
    VP8SSIMAccumulatePlane(src->y, src->y_stride,
                           ref->y, ref->y_stride,
                           src->width, src->height, &stats[0]);
    VP8SSIMAccumulatePlane(src->u, src->uv_stride,
                           ref->u, ref->uv_stride,
                           uv_w, uv_h, &stats[1]);
    VP8SSIMAccumulatePlane(src->v, src->uv_stride,
                           ref->v, ref->uv_stride,
                           uv_w, uv_h, &stats[2]);
    if (has_alpha) {
      VP8SSIMAccumulatePlane(src->a, src->a_stride,
                             ref->a, ref->a_stride,
                             src->width, src->height, &stats[3]);
    }
    for (c = 0; c <= 4; ++c) {
      if (type == 1) {
        const double v = VP8SSIMGet(&stats[c]);
        result[c] = (float)((v < 1.) ? -10.0 * log10(1. - v)
                                     : kMinDistortion_dB);
      } else {
        const double v = VP8SSIMGetSquaredError(&stats[c]);
        result[c] = GetPSNR(v);
      }
      // Accumulate forward
      if (c < 4) VP8SSIMAddStats(&stats[c], &stats[4]);
    }
  }
  return 1;
}

//------------------------------------------------------------------------------
// Simplest high-level calls:

typedef int (*Importer)(WebPPicture* const, const uint8_t* const, int);

static size_t Encode(const uint8_t* rgba, int width, int height, int stride,
                     Importer import, float quality_factor, int lossless,
                     uint8_t** output) {
  WebPPicture pic;
  WebPConfig config;
  WebPMemoryWriter wrt;
  int ok;

  if (!WebPConfigPreset(&config, WEBP_PRESET_DEFAULT, quality_factor) ||
      !WebPPictureInit(&pic)) {
    return 0;  // shouldn't happen, except if system installation is broken
  }

  config.lossless = !!lossless;
  pic.use_argb = !!lossless;
  pic.width = width;
  pic.height = height;
  pic.writer = WebPMemoryWrite;
  pic.custom_ptr = &wrt;
  WebPMemoryWriterInit(&wrt);

  ok = import(&pic, rgba, stride) && WebPEncode(&config, &pic);
  WebPPictureFree(&pic);
  if (!ok) {
    free(wrt.mem);
    *output = NULL;
    return 0;
  }
  *output = wrt.mem;
  return wrt.size;
}

#define ENCODE_FUNC(NAME, IMPORTER)                                     \
size_t NAME(const uint8_t* in, int w, int h, int bps, float q,          \
            uint8_t** out) {                                            \
  return Encode(in, w, h, bps, IMPORTER, q, 0, out);                    \
}

ENCODE_FUNC(WebPEncodeRGB, WebPPictureImportRGB)
ENCODE_FUNC(WebPEncodeBGR, WebPPictureImportBGR)
ENCODE_FUNC(WebPEncodeRGBA, WebPPictureImportRGBA)
ENCODE_FUNC(WebPEncodeBGRA, WebPPictureImportBGRA)

#undef ENCODE_FUNC

#define LOSSLESS_DEFAULT_QUALITY 70.
#define LOSSLESS_ENCODE_FUNC(NAME, IMPORTER)                                 \
size_t NAME(const uint8_t* in, int w, int h, int bps, uint8_t** out) {       \
  return Encode(in, w, h, bps, IMPORTER, LOSSLESS_DEFAULT_QUALITY, 1, out);  \
}

LOSSLESS_ENCODE_FUNC(WebPEncodeLosslessRGB, WebPPictureImportRGB)
LOSSLESS_ENCODE_FUNC(WebPEncodeLosslessBGR, WebPPictureImportBGR)
LOSSLESS_ENCODE_FUNC(WebPEncodeLosslessRGBA, WebPPictureImportRGBA)
LOSSLESS_ENCODE_FUNC(WebPEncodeLosslessBGRA, WebPPictureImportBGRA)

#undef LOSSLESS_ENCODE_FUNC

//------------------------------------------------------------------------------

