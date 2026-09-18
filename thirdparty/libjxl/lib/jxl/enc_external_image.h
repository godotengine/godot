// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_EXTERNAL_IMAGE_H_
#define LIB_JXL_ENC_EXTERNAL_IMAGE_H_

// Interleaved image for color transforms and Codec.

#include <jxl/types.h>
#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {
Status ConvertFromExternalNoSizeCheck(const uint8_t* data, size_t xsize,
                                      size_t ysize, size_t stride,
                                      size_t bits_per_sample,
                                      JxlPixelFormat format, size_t c,
                                      ThreadPool* pool, ImageF* channel);

Status ConvertFromExternalNoSizeCheck(const uint8_t* data, size_t xsize,
                                      size_t ysize, size_t stride,
                                      const ColorEncoding& c_current,
                                      size_t color_channels,
                                      size_t bits_per_sample,
                                      JxlPixelFormat format, ThreadPool* pool,
                                      ImageBundle* ib);

Status ConvertFromExternal(const uint8_t* data, size_t size, size_t xsize,
                           size_t ysize, size_t bits_per_sample,
                           JxlPixelFormat format, size_t c, ThreadPool* pool,
                           ImageF* channel);

// Convert an interleaved pixel buffer to the internal ImageBundle
// representation. This is the opposite of ConvertToExternal().
Status ConvertFromExternal(Span<const uint8_t> bytes, size_t xsize,
                           size_t ysize, const ColorEncoding& c_current,
                           size_t color_channels, size_t bits_per_sample,
                           JxlPixelFormat format, ThreadPool* pool,
                           ImageBundle* ib);
Status ConvertFromExternal(Span<const uint8_t> bytes, size_t xsize,
                           size_t ysize, const ColorEncoding& c_current,
                           size_t bits_per_sample, JxlPixelFormat format,
                           ThreadPool* pool, ImageBundle* ib);
Status BufferToImageF(const JxlPixelFormat& pixel_format, size_t xsize,
                      size_t ysize, const void* buffer, size_t size,
                      ThreadPool* pool, ImageF* channel);
Status BufferToImageBundle(const JxlPixelFormat& pixel_format, uint32_t xsize,
                           uint32_t ysize, const void* buffer, size_t size,
                           jxl::ThreadPool* pool,
                           const jxl::ColorEncoding& c_current,
                           jxl::ImageBundle* ib);

}  // namespace jxl

#endif  // LIB_JXL_ENC_EXTERNAL_IMAGE_H_
