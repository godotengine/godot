// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_TRANSFORM_H_
#define LIB_JXL_MODULAR_TRANSFORM_TRANSFORM_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

enum class TransformId : uint32_t {
  // G, R-G, B-G and variants (including YCoCg).
  kRCT = 0,

  // Color palette. Parameters are: [begin_c] [end_c] [nb_colors]
  kPalette = 1,

  // Squeezing (Haar-style)
  kSqueeze = 2,

  // Invalid for now.
  kInvalid = 3,
};

struct SqueezeParams : public Fields {
  JXL_FIELDS_NAME(SqueezeParams)
  bool horizontal;
  bool in_place;
  uint32_t begin_c;
  uint32_t num_c;
  SqueezeParams();
  Status VisitFields(Visitor *JXL_RESTRICT visitor) override {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &horizontal));
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &in_place));
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Bits(3), BitsOffset(6, 8),
                                           BitsOffset(10, 72),
                                           BitsOffset(13, 1096), 0, &begin_c));
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->U32(Val(1), Val(2), Val(3), BitsOffset(4, 4), 2, &num_c));
    return true;
  }
};

class Transform : public Fields {
 public:
  TransformId id;
  // for Palette and RCT.
  uint32_t begin_c;
  // for RCT. 42 possible values starting from 0.
  uint32_t rct_type;
  // Only for Palette and NearLossless.
  uint32_t num_c;
  // Only for Palette.
  uint32_t nb_colors;
  uint32_t nb_deltas;
  // for Squeeze. Default squeeze if empty.
  std::vector<SqueezeParams> squeezes;
  // for NearLossless, not serialized.
  int max_delta_error;
  // Serialized for Palette.
  Predictor predictor;
  // for Palette, not serialized.
  bool ordered_palette = true;
  bool lossy_palette = false;

  explicit Transform(TransformId id);
  // default constructor for bundles.
  Transform() : Transform(TransformId::kInvalid) {}

  Status VisitFields(Visitor *JXL_RESTRICT visitor) override {
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->U32(Val(static_cast<uint32_t>(TransformId::kRCT)),
                     Val(static_cast<uint32_t>(TransformId::kPalette)),
                     Val(static_cast<uint32_t>(TransformId::kSqueeze)),
                     Val(static_cast<uint32_t>(TransformId::kInvalid)),
                     static_cast<uint32_t>(TransformId::kRCT),
                     reinterpret_cast<uint32_t *>(&id)));
    if (id == TransformId::kInvalid) {
      return JXL_FAILURE("Invalid transform ID");
    }
    if (visitor->Conditional(id == TransformId::kRCT ||
                             id == TransformId::kPalette)) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->U32(Bits(3), BitsOffset(6, 8), BitsOffset(10, 72),
                       BitsOffset(13, 1096), 0, &begin_c));
    }
    if (visitor->Conditional(id == TransformId::kRCT)) {
      // 0-41, default YCoCg.
      JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(6), Bits(2), BitsOffset(4, 2),
                                             BitsOffset(6, 10), 6, &rct_type));
      if (rct_type >= 42) {
        return JXL_FAILURE("Invalid transform RCT type");
      }
    }
    if (visitor->Conditional(id == TransformId::kPalette)) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->U32(Val(1), Val(3), Val(4), BitsOffset(13, 1), 3, &num_c));
      JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
          BitsOffset(8, 0), BitsOffset(10, 256), BitsOffset(12, 1280),
          BitsOffset(16, 5376), 256, &nb_colors));
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->U32(Val(0), BitsOffset(8, 1), BitsOffset(10, 257),
                       BitsOffset(16, 1281), 0, &nb_deltas));
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Bits(4, static_cast<uint32_t>(Predictor::Zero),
                        reinterpret_cast<uint32_t *>(&predictor)));
      if (predictor >= Predictor::Best) {
        return JXL_FAILURE("Invalid predictor");
      }
    }

    if (visitor->Conditional(id == TransformId::kSqueeze)) {
      uint32_t num_squeezes = static_cast<uint32_t>(squeezes.size());
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->U32(Val(0), BitsOffset(4, 1), BitsOffset(6, 9),
                       BitsOffset(8, 41), 0, &num_squeezes));
      if (visitor->IsReading()) squeezes.resize(num_squeezes);
      for (size_t i = 0; i < num_squeezes; i++) {
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&squeezes[i]));
      }
    }
    return true;
  }

  JXL_FIELDS_NAME(Transform)

  Status Inverse(Image &input, const weighted::Header &wp_header,
                 ThreadPool *pool = nullptr) const;
  Status MetaApply(Image &input);
};

Status CheckEqualChannels(const Image &image, uint32_t c1, uint32_t c2);

static inline pixel_type PixelAdd(pixel_type a, pixel_type b) {
  return static_cast<pixel_type>(static_cast<uint32_t>(a) +
                                 static_cast<uint32_t>(b));
}

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_TRANSFORM_H_
