// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_external_image.h"

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <cstdint>
#include <cstring>
#include <utility>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/float.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"

namespace jxl {
namespace {

size_t JxlDataTypeBytes(JxlDataType data_type) {
  switch (data_type) {
    case JXL_TYPE_UINT8:
      return 1;
    case JXL_TYPE_UINT16:
      return 2;
    case JXL_TYPE_FLOAT16:
      return 2;
    case JXL_TYPE_FLOAT:
      return 4;
    default:
      return 0;
  }
}

}  // namespace

Status ConvertFromExternalPlaneNoSizeCheck(const uint8_t* data, size_t xsize,
                                           size_t ysize, size_t stride,
                                           size_t bits_per_sample,
                                           JxlPixelFormat format, size_t c,
                                           ThreadPool* pool, ImageF* channel) {
  if (format.data_type == JXL_TYPE_UINT8) {
    JXL_RETURN_IF_ERROR(bits_per_sample > 0 && bits_per_sample <= 8);
  } else if (format.data_type == JXL_TYPE_UINT16) {
    JXL_RETURN_IF_ERROR(bits_per_sample > 8 && bits_per_sample <= 16);
  } else if (format.data_type != JXL_TYPE_FLOAT16 &&
             format.data_type != JXL_TYPE_FLOAT) {
    return JXL_FAILURE("unsupported pixel format data type %d",
                       format.data_type);
  }

  JXL_ENSURE(channel->xsize() == xsize);
  JXL_ENSURE(channel->ysize() == ysize);

  size_t bytes_per_channel = JxlDataTypeBytes(format.data_type);
  size_t bytes_per_pixel = format.num_channels * bytes_per_channel;
  size_t bytes_per_row;
  if (!SafeMul(xsize, bytes_per_pixel, bytes_per_row)) {
    return JXL_FAILURE("Image dimensions are too large");
  }
  JXL_ENSURE(bytes_per_row <= stride);
  size_t pixel_offset = c * bytes_per_channel;
  // Only for uint8/16.
  float scale = 1.0f / ((1ull << bits_per_sample) - 1);

  const bool little_endian =
      format.endianness == JXL_LITTLE_ENDIAN ||
      (format.endianness == JXL_NATIVE_ENDIAN && IsLittleEndian());

  const auto convert_row = [&](const uint32_t task,
                               size_t /*thread*/) -> Status {
    const size_t y = task;
    size_t offset = y * stride + pixel_offset;
    float* JXL_RESTRICT row_out = channel->Row(y);
    const auto save_value = [&](size_t index, float value) {
      row_out[index] = value;
    };
    JXL_RETURN_IF_ERROR(LoadFloatRow(data + offset, xsize, bytes_per_pixel,
                                     format.data_type, little_endian, scale,
                                     save_value));
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, static_cast<uint32_t>(ysize),
                                ThreadPool::NoInit, convert_row,
                                "ConvertExtraChannel"));
  return true;
}

Status ConvertFromExternal(const uint8_t* data, size_t size, size_t xsize,
                           size_t ysize, size_t bits_per_sample,
                           JxlPixelFormat format, size_t c, ThreadPool* pool,
                           ImageF* channel) {
  size_t bytes_per_channel = JxlDataTypeBytes(format.data_type);
  size_t bytes_per_pixel;
  if (!SafeMul(static_cast<size_t>(format.num_channels), bytes_per_channel,
               bytes_per_pixel)) {
    return JXL_FAILURE("Invalid format");
  }
  size_t last_row_size;
  if (!SafeMul(xsize, bytes_per_pixel, last_row_size)) {
    return JXL_FAILURE("Image dimensions are too large");
  }
  size_t row_size = last_row_size;
  if (!SafeRoundUpTo(row_size, format.align, row_size)) {
    return JXL_FAILURE("Image dimensions are too large");
  }
  if (xsize == 0 || ysize == 0) return JXL_FAILURE("Empty image");
  size_t bytes_to_read;
  if (!SafeMul(row_size, ysize - 1, bytes_to_read) ||
      !SafeAdd(bytes_to_read, last_row_size, bytes_to_read)) {
    return JXL_FAILURE("Image dimensions are too large");
  }
  if (size > 0 && size < bytes_to_read) {
    return JXL_FAILURE("Buffer size is too small, expected: %" PRIuS
                       " got: %" PRIuS " (Image: %" PRIuS "x%" PRIuS
                       "x%u, bytes_per_channel: %" PRIuS ")",
                       bytes_to_read, size, xsize, ysize, format.num_channels,
                       bytes_per_channel);
  }
  // Too large buffer is likely an application bug, so also fail for that.
  // Do allow padding to stride in last row though.
  if (size > row_size * ysize) {
    return JXL_FAILURE("Buffer size is too large");
  }
  return ConvertFromExternalPlaneNoSizeCheck(
      data, xsize, ysize, row_size, bits_per_sample, format, c, pool, channel);
}
Status ConvertFromExternal(Span<const uint8_t> bytes, size_t xsize,
                           size_t ysize, const ColorEncoding& c_current,
                           size_t bits_per_sample, JxlPixelFormat format,
                           ThreadPool* pool, ImageBundle* ib, bool set_alpha) {
  JxlMemoryManager* memory_manager = ib->memory_manager();
  size_t color_channels = c_current.Channels();
  bool has_alpha = format.num_channels == 2 || format.num_channels == 4;
  if (format.num_channels < color_channels) {
    return JXL_FAILURE("Expected %" PRIuS
                       " color channels, received only %u channels",
                       color_channels, format.num_channels);
  }

  JXL_ASSIGN_OR_RETURN(Image3F color,
                       Image3F::Create(memory_manager, xsize, ysize));
  for (size_t c = 0; c < color_channels; ++c) {
    JXL_RETURN_IF_ERROR(ConvertFromExternal(bytes.data(), bytes.size(), xsize,
                                            ysize, bits_per_sample, format, c,
                                            pool, &color.Plane(c)));
  }
  if (color_channels == 1) {
    JXL_RETURN_IF_ERROR(CopyImageTo(color.Plane(0), &color.Plane(1)));
    JXL_RETURN_IF_ERROR(CopyImageTo(color.Plane(0), &color.Plane(2)));
  }
  JXL_RETURN_IF_ERROR(ib->SetFromImage(std::move(color), c_current));

  // Passing an interleaved image with an alpha channel to an image that doesn't
  // have alpha channel just discards the passed alpha channel.
  if (set_alpha) {
    JXL_ASSIGN_OR_RETURN(ImageF alpha,
                         ImageF::Create(memory_manager, xsize, ysize));
    if (has_alpha) {
      JXL_RETURN_IF_ERROR(ConvertFromExternal(
          bytes.data(), bytes.size(), xsize, ysize, bits_per_sample, format,
          format.num_channels - 1, pool, &alpha));
    } else {
      // if alpha is not passed, but it is expected, then assume
      // it is all-opaque
      FillImage(1.0f, &alpha);
    }
    JXL_RETURN_IF_ERROR(ib->SetAlpha(std::move(alpha)));
  }

  return true;
}

Status BufferToImageF(const JxlPixelFormat& pixel_format, size_t xsize,
                      size_t ysize, const void* buffer, size_t size,
                      ThreadPool* pool, ImageF* channel) {
  size_t bitdepth = JxlDataTypeBytes(pixel_format.data_type) * kBitsPerByte;
  return ConvertFromExternal(reinterpret_cast<const uint8_t*>(buffer), size,
                             xsize, ysize, bitdepth, pixel_format, 0, pool,
                             channel);
}

Status BufferToImageBundle(const JxlPixelFormat& pixel_format, uint32_t xsize,
                           uint32_t ysize, const void* buffer, size_t size,
                           jxl::ThreadPool* pool,
                           const jxl::ColorEncoding& c_current,
                           jxl::ImageBundle* ib, bool set_alpha) {
  size_t bitdepth = JxlDataTypeBytes(pixel_format.data_type) * kBitsPerByte;
  JXL_RETURN_IF_ERROR(ConvertFromExternal(
      jxl::Bytes(static_cast<const uint8_t*>(buffer), size), xsize, ysize,
      c_current, bitdepth, pixel_format, pool, ib, set_alpha));
  JXL_RETURN_IF_ERROR(ib->VerifyMetadata());

  return true;
}

}  // namespace jxl
