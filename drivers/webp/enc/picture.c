// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// WebPPicture class basis
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>

#include "./vp8enci.h"
#include "../dsp/dsp.h"
#include "../utils/utils.h"

//------------------------------------------------------------------------------
// WebPPicture
//------------------------------------------------------------------------------

static int DummyWriter(const uint8_t* data, size_t data_size,
                       const WebPPicture* const picture) {
  // The following are to prevent 'unused variable' error message.
  (void)data;
  (void)data_size;
  (void)picture;
  return 1;
}

int WebPPictureInitInternal(WebPPicture* picture, int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_ENCODER_ABI_VERSION)) {
    return 0;   // caller/system version mismatch!
  }
  if (picture != NULL) {
    memset(picture, 0, sizeof(*picture));
    picture->writer = DummyWriter;
    WebPEncodingSetError(picture, VP8_ENC_OK);
  }
  return 1;
}

//------------------------------------------------------------------------------

static void WebPPictureResetBufferARGB(WebPPicture* const picture) {
  picture->memory_argb_ = NULL;
  picture->argb = NULL;
  picture->argb_stride = 0;
}

static void WebPPictureResetBufferYUVA(WebPPicture* const picture) {
  picture->memory_ = NULL;
  picture->y = picture->u = picture->v = picture->a = NULL;
  picture->y_stride = picture->uv_stride = 0;
  picture->a_stride = 0;
}

void WebPPictureResetBuffers(WebPPicture* const picture) {
  WebPPictureResetBufferARGB(picture);
  WebPPictureResetBufferYUVA(picture);
}

int WebPPictureAllocARGB(WebPPicture* const picture, int width, int height) {
  void* memory;
  const uint64_t argb_size = (uint64_t)width * height;

  assert(picture != NULL);

  WebPSafeFree(picture->memory_argb_);
  WebPPictureResetBufferARGB(picture);

  if (width <= 0 || height <= 0) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_BAD_DIMENSION);
  }
  // allocate a new buffer.
  memory = WebPSafeMalloc(argb_size, sizeof(*picture->argb));
  if (memory == NULL) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }
  // TODO(skal): align plane to cache line?
  picture->memory_argb_ = memory;
  picture->argb = (uint32_t*)memory;
  picture->argb_stride = width;
  return 1;
}

int WebPPictureAllocYUVA(WebPPicture* const picture, int width, int height) {
  const WebPEncCSP uv_csp = picture->colorspace & WEBP_CSP_UV_MASK;
  const int has_alpha = picture->colorspace & WEBP_CSP_ALPHA_BIT;
  const int y_stride = width;
  const int uv_width = (width + 1) >> 1;
  const int uv_height = (height + 1) >> 1;
  const int uv_stride = uv_width;
  int a_width, a_stride;
  uint64_t y_size, uv_size, a_size, total_size;
  uint8_t* mem;

  assert(picture != NULL);

  WebPSafeFree(picture->memory_);
  WebPPictureResetBufferYUVA(picture);

  if (uv_csp != WEBP_YUV420) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_INVALID_CONFIGURATION);
  }

  // alpha
  a_width = has_alpha ? width : 0;
  a_stride = a_width;
  y_size = (uint64_t)y_stride * height;
  uv_size = (uint64_t)uv_stride * uv_height;
  a_size =  (uint64_t)a_stride * height;

  total_size = y_size + a_size + 2 * uv_size;

  // Security and validation checks
  if (width <= 0 || height <= 0 ||         // luma/alpha param error
      uv_width < 0 || uv_height < 0) {     // u/v param error
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_BAD_DIMENSION);
  }
  // allocate a new buffer.
  mem = (uint8_t*)WebPSafeMalloc(total_size, sizeof(*mem));
  if (mem == NULL) {
    return WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }

  // From now on, we're in the clear, we can no longer fail...
  picture->memory_ = (void*)mem;
  picture->y_stride  = y_stride;
  picture->uv_stride = uv_stride;
  picture->a_stride  = a_stride;

  // TODO(skal): we could align the y/u/v planes and adjust stride.
  picture->y = mem;
  mem += y_size;

  picture->u = mem;
  mem += uv_size;
  picture->v = mem;
  mem += uv_size;

  if (a_size > 0) {
    picture->a = mem;
    mem += a_size;
  }
  (void)mem;  // makes the static analyzer happy
  return 1;
}

int WebPPictureAlloc(WebPPicture* picture) {
  if (picture != NULL) {
    const int width = picture->width;
    const int height = picture->height;

    WebPPictureFree(picture);   // erase previous buffer

    if (!picture->use_argb) {
      return WebPPictureAllocYUVA(picture, width, height);
    } else {
      return WebPPictureAllocARGB(picture, width, height);
    }
  }
  return 1;
}

void WebPPictureFree(WebPPicture* picture) {
  if (picture != NULL) {
    WebPSafeFree(picture->memory_);
    WebPSafeFree(picture->memory_argb_);
    WebPPictureResetBuffers(picture);
  }
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
    WebPSafeFree(w->mem);
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

void WebPMemoryWriterClear(WebPMemoryWriter* writer) {
  if (writer != NULL) {
    WebPSafeFree(writer->mem);
    writer->mem = NULL;
    writer->size = 0;
    writer->max_size = 0;
  }
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
    WebPMemoryWriterClear(&wrt);
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
