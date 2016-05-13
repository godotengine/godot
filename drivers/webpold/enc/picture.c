// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// WebPPicture utils: colorspace conversion, crop, ...
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "./vp8enci.h"
#include "../utils/rescaler.h"
#include "../utils/utils.h"
#include "../dsp/dsp.h"
#include "../dsp/yuv.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define HALVE(x) (((x) + 1) >> 1)
#define IS_YUV_CSP(csp, YUV_CSP) (((csp) & WEBP_CSP_UV_MASK) == (YUV_CSP))

static const union {
  uint32_t argb;
  uint8_t  bytes[4];
} test_endian = { 0xff000000u };
#define ALPHA_IS_LAST (test_endian.bytes[3] == 0xff)

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
    if (src->a != NULL) {
      dst->a = src->a + top * src->a_stride + left;
    }
#ifdef WEBP_EXPERIMENTAL_FEATURES
    if (src->u0 != NULL) {
      const int left_pos =
          IS_YUV_CSP(dst->colorspace, WEBP_YUV422) ? (left >> 1) : left;
      dst->u0 = src->u0 + top * src->uv0_stride + left_pos;
      dst->v0 = src->v0 + top * src->uv0_stride + left_pos;
    }
#endif
  } else {
    dst->argb = src->argb + top * src->argb_stride + left;
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

    RescalePlane(pic->y, prev_width, prev_height, pic->y_stride,
                 tmp.y, width, height, tmp.y_stride, work, 1);
    RescalePlane(pic->u,
                 HALVE(prev_width), HALVE(prev_height), pic->uv_stride,
                 tmp.u,
                 HALVE(width), HALVE(height), tmp.uv_stride, work, 1);
    RescalePlane(pic->v,
                 HALVE(prev_width), HALVE(prev_height), pic->uv_stride,
                 tmp.v,
                 HALVE(width), HALVE(height), tmp.uv_stride, work, 1);

    if (tmp.a != NULL) {
      RescalePlane(pic->a, prev_width, prev_height, pic->a_stride,
                   tmp.a, width, height, tmp.a_stride, work, 1);
    }
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

    RescalePlane((const uint8_t*)pic->argb, prev_width, prev_height,
                 pic->argb_stride * 4,
                 (uint8_t*)tmp.argb, width, height,
                 tmp.argb_stride * 4,
                 work, 4);

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

// TODO: we can do better than simply 2x2 averaging on U/V samples.
#define SUM4(ptr) ((ptr)[0] + (ptr)[step] + \
                   (ptr)[rgb_stride] + (ptr)[rgb_stride + step])
#define SUM2H(ptr) (2 * (ptr)[0] + 2 * (ptr)[step])
#define SUM2V(ptr) (2 * (ptr)[0] + 2 * (ptr)[rgb_stride])
#define SUM1(ptr)  (4 * (ptr)[0])
#define RGB_TO_UV(x, y, SUM) {                           \
  const int src = (2 * (step * (x) + (y) * rgb_stride)); \
  const int dst = (x) + (y) * picture->uv_stride;        \
  const int r = SUM(r_ptr + src);                        \
  const int g = SUM(g_ptr + src);                        \
  const int b = SUM(b_ptr + src);                        \
  picture->u[dst] = VP8RGBToU(r, g, b);                  \
  picture->v[dst] = VP8RGBToV(r, g, b);                  \
}

#define RGB_TO_UV0(x_in, x_out, y, SUM) {                \
  const int src = (step * (x_in) + (y) * rgb_stride);    \
  const int dst = (x_out) + (y) * picture->uv0_stride;   \
  const int r = SUM(r_ptr + src);                        \
  const int g = SUM(g_ptr + src);                        \
  const int b = SUM(b_ptr + src);                        \
  picture->u0[dst] = VP8RGBToU(r, g, b);                 \
  picture->v0[dst] = VP8RGBToV(r, g, b);                 \
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
                              WebPPicture* const picture) {
  const WebPEncCSP uv_csp = picture->colorspace & WEBP_CSP_UV_MASK;
  int x, y;
  const int width = picture->width;
  const int height = picture->height;
  const int has_alpha = CheckNonOpaque(a_ptr, width, height, step, rgb_stride);

  picture->colorspace = uv_csp;
  picture->use_argb = 0;
  if (has_alpha) {
    picture->colorspace |= WEBP_CSP_ALPHA_BIT;
  }
  if (!WebPPictureAlloc(picture)) return 0;

  // Import luma plane
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      const int offset = step * x + y * rgb_stride;
      picture->y[x + y * picture->y_stride] =
          VP8RGBToY(r_ptr[offset], g_ptr[offset], b_ptr[offset]);
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
                              picture);
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
            0xff000000u |
            (r_ptr[offset] << 16) |
            (g_ptr[offset] <<  8) |
            (b_ptr[offset]);
        picture->argb[x + y * picture->argb_stride] = argb;
      }
    }
  } else {
    int x, y;
    assert(step >= 4);
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        const int offset = step * x + y * rgb_stride;
        const uint32_t argb = (a_ptr[offset] << 24) |
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
  if (picture->memory_ == NULL || picture->y == NULL ||
      picture->u == NULL || picture->v == NULL) {
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
    upsample(NULL, cur_y, cur_u, cur_v, cur_u, cur_v, NULL, dst, width);
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
        uint32_t* const dst = picture->argb + y * picture->argb_stride;
        const uint8_t* const src = picture->a + y * picture->a_stride;
        int x;
        for (x = 0; x < width; ++x) {
          dst[x] = (dst[x] & 0x00ffffffu) | (src[x] << 24);
        }
      }
    }
  }
  return 1;
}

int WebPPictureARGBToYUVA(WebPPicture* picture, WebPEncCSP colorspace) {
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
    if (!ImportYUVAFromRGBA(r, g, b, a, 4, 4 * picture->argb_stride, &tmp)) {
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
// Distortion

// Max value returned in case of exact similarity.
static const double kMinDistortion_dB = 99.;

int WebPPictureDistortion(const WebPPicture* pic1, const WebPPicture* pic2,
                          int type, float result[5]) {
  int c;
  DistoStats stats[5];
  int has_alpha;

  if (pic1 == NULL || pic2 == NULL ||
      pic1->width != pic2->width || pic1->height != pic2->height ||
      pic1->y == NULL || pic2->y == NULL ||
      pic1->u == NULL || pic2->u == NULL ||
      pic1->v == NULL || pic2->v == NULL ||
      result == NULL) {
    return 0;
  }
  // TODO(skal): provide distortion for ARGB too.
  if (pic1->use_argb == 1 || pic1->use_argb != pic2->use_argb) {
    return 0;
  }

  has_alpha = !!(pic1->colorspace & WEBP_CSP_ALPHA_BIT);
  if (has_alpha != !!(pic2->colorspace & WEBP_CSP_ALPHA_BIT) ||
      (has_alpha && (pic1->a == NULL || pic2->a == NULL))) {
    return 0;
  }

  memset(stats, 0, sizeof(stats));
  VP8SSIMAccumulatePlane(pic1->y, pic1->y_stride,
                         pic2->y, pic2->y_stride,
                         pic1->width, pic1->height, &stats[0]);
  VP8SSIMAccumulatePlane(pic1->u, pic1->uv_stride,
                         pic2->u, pic2->uv_stride,
                         (pic1->width + 1) >> 1, (pic1->height + 1) >> 1,
                         &stats[1]);
  VP8SSIMAccumulatePlane(pic1->v, pic1->uv_stride,
                         pic2->v, pic2->uv_stride,
                         (pic1->width + 1) >> 1, (pic1->height + 1) >> 1,
                         &stats[2]);
  if (has_alpha) {
    VP8SSIMAccumulatePlane(pic1->a, pic1->a_stride,
                           pic2->a, pic2->a_stride,
                           pic1->width, pic1->height, &stats[3]);
  }
  for (c = 0; c <= 4; ++c) {
    if (type == 1) {
      const double v = VP8SSIMGet(&stats[c]);
      result[c] = (float)((v < 1.) ? -10.0 * log10(1. - v)
                                   : kMinDistortion_dB);
    } else {
      const double v = VP8SSIMGetSquaredError(&stats[c]);
      result[c] = (float)((v > 0.) ? -4.3429448 * log(v / (255 * 255.))
                                   : kMinDistortion_dB);
    }
    // Accumulate forward
    if (c < 4) VP8SSIMAddStats(&stats[c], &stats[4]);
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

ENCODE_FUNC(WebPEncodeRGB, WebPPictureImportRGB);
ENCODE_FUNC(WebPEncodeBGR, WebPPictureImportBGR);
ENCODE_FUNC(WebPEncodeRGBA, WebPPictureImportRGBA);
ENCODE_FUNC(WebPEncodeBGRA, WebPPictureImportBGRA);

#undef ENCODE_FUNC

#define LOSSLESS_DEFAULT_QUALITY 70.
#define LOSSLESS_ENCODE_FUNC(NAME, IMPORTER)                                 \
size_t NAME(const uint8_t* in, int w, int h, int bps, uint8_t** out) {       \
  return Encode(in, w, h, bps, IMPORTER, LOSSLESS_DEFAULT_QUALITY, 1, out);  \
}

LOSSLESS_ENCODE_FUNC(WebPEncodeLosslessRGB, WebPPictureImportRGB);
LOSSLESS_ENCODE_FUNC(WebPEncodeLosslessBGR, WebPPictureImportBGR);
LOSSLESS_ENCODE_FUNC(WebPEncodeLosslessRGBA, WebPPictureImportRGBA);
LOSSLESS_ENCODE_FUNC(WebPEncodeLosslessBGRA, WebPPictureImportBGRA);

#undef LOSSLESS_ENCODE_FUNC

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
