// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Everything about WebPDecBuffer
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>

#include "./vp8i.h"
#include "./webpi.h"
#include "../utils/utils.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// WebPDecBuffer

// Number of bytes per pixel for the different color-spaces.
static const int kModeBpp[MODE_LAST] = {
  3, 4, 3, 4, 4, 2, 2,
  4, 4, 4, 2,    // pre-multiplied modes
  1, 1 };

// Check that webp_csp_mode is within the bounds of WEBP_CSP_MODE.
// Convert to an integer to handle both the unsigned/signed enum cases
// without the need for casting to remove type limit warnings.
static int IsValidColorspace(int webp_csp_mode) {
  return (webp_csp_mode >= MODE_RGB && webp_csp_mode < MODE_LAST);
}

static VP8StatusCode CheckDecBuffer(const WebPDecBuffer* const buffer) {
  int ok = 1;
  const WEBP_CSP_MODE mode = buffer->colorspace;
  const int width = buffer->width;
  const int height = buffer->height;
  if (!IsValidColorspace(mode)) {
    ok = 0;
  } else if (!WebPIsRGBMode(mode)) {   // YUV checks
    const WebPYUVABuffer* const buf = &buffer->u.YUVA;
    const uint64_t y_size = (uint64_t)buf->y_stride * height;
    const uint64_t u_size = (uint64_t)buf->u_stride * ((height + 1) / 2);
    const uint64_t v_size = (uint64_t)buf->v_stride * ((height + 1) / 2);
    const uint64_t a_size = (uint64_t)buf->a_stride * height;
    ok &= (y_size <= buf->y_size);
    ok &= (u_size <= buf->u_size);
    ok &= (v_size <= buf->v_size);
    ok &= (buf->y_stride >= width);
    ok &= (buf->u_stride >= (width + 1) / 2);
    ok &= (buf->v_stride >= (width + 1) / 2);
    ok &= (buf->y != NULL);
    ok &= (buf->u != NULL);
    ok &= (buf->v != NULL);
    if (mode == MODE_YUVA) {
      ok &= (buf->a_stride >= width);
      ok &= (a_size <= buf->a_size);
      ok &= (buf->a != NULL);
    }
  } else {    // RGB checks
    const WebPRGBABuffer* const buf = &buffer->u.RGBA;
    const uint64_t size = (uint64_t)buf->stride * height;
    ok &= (size <= buf->size);
    ok &= (buf->stride >= width * kModeBpp[mode]);
    ok &= (buf->rgba != NULL);
  }
  return ok ? VP8_STATUS_OK : VP8_STATUS_INVALID_PARAM;
}

static VP8StatusCode AllocateBuffer(WebPDecBuffer* const buffer) {
  const int w = buffer->width;
  const int h = buffer->height;
  const WEBP_CSP_MODE mode = buffer->colorspace;

  if (w <= 0 || h <= 0 || !IsValidColorspace(mode)) {
    return VP8_STATUS_INVALID_PARAM;
  }

  if (!buffer->is_external_memory && buffer->private_memory == NULL) {
    uint8_t* output;
    int uv_stride = 0, a_stride = 0;
    uint64_t uv_size = 0, a_size = 0, total_size;
    // We need memory and it hasn't been allocated yet.
    // => initialize output buffer, now that dimensions are known.
    const int stride = w * kModeBpp[mode];
    const uint64_t size = (uint64_t)stride * h;

    if (!WebPIsRGBMode(mode)) {
      uv_stride = (w + 1) / 2;
      uv_size = (uint64_t)uv_stride * ((h + 1) / 2);
      if (mode == MODE_YUVA) {
        a_stride = w;
        a_size = (uint64_t)a_stride * h;
      }
    }
    total_size = size + 2 * uv_size + a_size;

    // Security/sanity checks
    output = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*output));
    if (output == NULL) {
      return VP8_STATUS_OUT_OF_MEMORY;
    }
    buffer->private_memory = output;

    if (!WebPIsRGBMode(mode)) {   // YUVA initialization
      WebPYUVABuffer* const buf = &buffer->u.YUVA;
      buf->y = output;
      buf->y_stride = stride;
      buf->y_size = (size_t)size;
      buf->u = output + size;
      buf->u_stride = uv_stride;
      buf->u_size = (size_t)uv_size;
      buf->v = output + size + uv_size;
      buf->v_stride = uv_stride;
      buf->v_size = (size_t)uv_size;
      if (mode == MODE_YUVA) {
        buf->a = output + size + 2 * uv_size;
      }
      buf->a_size = (size_t)a_size;
      buf->a_stride = a_stride;
    } else {  // RGBA initialization
      WebPRGBABuffer* const buf = &buffer->u.RGBA;
      buf->rgba = output;
      buf->stride = stride;
      buf->size = (size_t)size;
    }
  }
  return CheckDecBuffer(buffer);
}

VP8StatusCode WebPAllocateDecBuffer(int w, int h,
                                    const WebPDecoderOptions* const options,
                                    WebPDecBuffer* const out) {
  if (out == NULL || w <= 0 || h <= 0) {
    return VP8_STATUS_INVALID_PARAM;
  }
  if (options != NULL) {    // First, apply options if there is any.
    if (options->use_cropping) {
      const int cw = options->crop_width;
      const int ch = options->crop_height;
      const int x = options->crop_left & ~1;
      const int y = options->crop_top & ~1;
      if (x < 0 || y < 0 || cw <= 0 || ch <= 0 || x + cw > w || y + ch > h) {
        return VP8_STATUS_INVALID_PARAM;   // out of frame boundary.
      }
      w = cw;
      h = ch;
    }
    if (options->use_scaling) {
      if (options->scaled_width <= 0 || options->scaled_height <= 0) {
        return VP8_STATUS_INVALID_PARAM;
      }
      w = options->scaled_width;
      h = options->scaled_height;
    }
  }
  out->width = w;
  out->height = h;

  // Then, allocate buffer for real
  return AllocateBuffer(out);
}

//------------------------------------------------------------------------------
// constructors / destructors

int WebPInitDecBufferInternal(WebPDecBuffer* buffer, int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_DECODER_ABI_VERSION)) {
    return 0;  // version mismatch
  }
  if (buffer == NULL) return 0;
  memset(buffer, 0, sizeof(*buffer));
  return 1;
}

void WebPFreeDecBuffer(WebPDecBuffer* buffer) {
  if (buffer != NULL) {
    if (!buffer->is_external_memory)
      free(buffer->private_memory);
    buffer->private_memory = NULL;
  }
}

void WebPCopyDecBuffer(const WebPDecBuffer* const src,
                       WebPDecBuffer* const dst) {
  if (src != NULL && dst != NULL) {
    *dst = *src;
    if (src->private_memory != NULL) {
      dst->is_external_memory = 1;   // dst buffer doesn't own the memory.
      dst->private_memory = NULL;
    }
  }
}

// Copy and transfer ownership from src to dst (beware of parameter order!)
void WebPGrabDecBuffer(WebPDecBuffer* const src, WebPDecBuffer* const dst) {
  if (src != NULL && dst != NULL) {
    *dst = *src;
    if (src->private_memory != NULL) {
      src->is_external_memory = 1;   // src relinquishes ownership
      src->private_memory = NULL;
    }
  }
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
