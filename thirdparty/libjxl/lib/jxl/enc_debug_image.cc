// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_debug_image.h"

#include <jxl/color_encoding.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_external_image.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

namespace {
template <typename From>
StatusOr<Image3F> ConvertToFloat(const Image3<From>& from) {
  float factor = 1.0f / std::numeric_limits<From>::max();
  if (std::is_same<From, double>::value || std::is_same<From, float>::value) {
    factor = 1.0f;
  }
  JxlMemoryManager* memory_manager = from.memory_manager();
  JXL_ASSIGN_OR_RETURN(
      Image3F to, Image3F::Create(memory_manager, from.xsize(), from.ysize()));
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < from.ysize(); ++y) {
      const From* const JXL_RESTRICT row_from = from.ConstPlaneRow(c, y);
      float* const JXL_RESTRICT row_to = to.PlaneRow(c, y);
      for (size_t x = 0; x < from.xsize(); ++x) {
        row_to[x] = row_from[x] * factor;
      }
    }
  }
  return to;
}

template <typename T>
Status DumpImageT(const CompressParams& cparams, const char* label,
                  const ColorEncoding& color_encoding, const Image3<T>& image) {
  if (!cparams.debug_image) return true;
  JXL_ASSIGN_OR_RETURN(Image3F float_image, ConvertToFloat(image));
  JxlColorEncoding color = color_encoding.ToExternal();
  size_t num_pixels = 3 * image.xsize() * image.ysize();
  std::vector<uint16_t> pixels(num_pixels);
  const ImageF* channels[3];
  for (int c = 0; c < 3; ++c) {
    channels[c] = &float_image.Plane(c);
  }
  JXL_RETURN_IF_ERROR(ConvertChannelsToExternal(
      channels, 3, 16, false, JXL_BIG_ENDIAN, 6 * image.xsize(), nullptr,
      pixels.data(), 2 * num_pixels, PixelCallback(), Orientation::kIdentity));
  (*cparams.debug_image)(cparams.debug_image_opaque, label, image.xsize(),
                         image.ysize(), &color, pixels.data());
  return true;
}

template <typename T>
Status DumpPlaneNormalizedT(const CompressParams& cparams, const char* label,
                            const Plane<T>& image) {
  T min;
  T max;
  ImageMinMax(image, &min, &max);
  JxlMemoryManager* memory_manager = image.memory_manager();

  JXL_ASSIGN_OR_RETURN(
      Image3B normalized,
      Image3B::Create(memory_manager, image.xsize(), image.ysize()));
  for (size_t c = 0; c < 3; ++c) {
    float mul = min == max ? 0 : (255.0f / (max - min));
    for (size_t y = 0; y < image.ysize(); ++y) {
      const T* JXL_RESTRICT row_in = image.ConstRow(y);
      uint8_t* JXL_RESTRICT row_out = normalized.PlaneRow(c, y);
      for (size_t x = 0; x < image.xsize(); ++x) {
        row_out[x] = static_cast<uint8_t>((row_in[x] - min) * mul);
      }
    }
  }
  return DumpImageT(cparams, label, ColorEncoding::SRGB(), normalized);
}

}  // namespace

Status DumpImage(const CompressParams& cparams, const char* label,
                 const Image3<float>& image) {
  return DumpImageT(cparams, label, ColorEncoding::SRGB(), image);
}

Status DumpImage(const CompressParams& cparams, const char* label,
                 const Image3<uint8_t>& image) {
  return DumpImageT(cparams, label, ColorEncoding::SRGB(), image);
}

Status DumpXybImage(const CompressParams& cparams, const char* label,
                    const Image3F& image) {
  if (!cparams.debug_image) return true;
  JxlMemoryManager* memory_manager = image.memory_manager();

  JXL_ASSIGN_OR_RETURN(
      Image3F linear,
      Image3F::Create(memory_manager, image.xsize(), image.ysize()));
  OpsinParams opsin_params;
  opsin_params.Init(kDefaultIntensityTarget);
  JXL_RETURN_IF_ERROR(
      OpsinToLinear(image, Rect(linear), nullptr, &linear, opsin_params));

  return DumpImageT(cparams, label, ColorEncoding::LinearSRGB(), linear);
}

Status DumpPlaneNormalized(const CompressParams& cparams, const char* label,
                           const Plane<float>& image) {
  return DumpPlaneNormalizedT(cparams, label, image);
}

Status DumpPlaneNormalized(const CompressParams& cparams, const char* label,
                           const Plane<uint8_t>& image) {
  return DumpPlaneNormalizedT(cparams, label, image);
}

}  // namespace jxl
