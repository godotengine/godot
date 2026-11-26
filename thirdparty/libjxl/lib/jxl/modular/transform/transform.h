// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_TRANSFORM_H_
#define LIB_JXL_MODULAR_TRANSFORM_TRANSFORM_H_

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
#include "lib/jxl/modular/transform/squeeze_params.h"

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
  Transform();

  Status VisitFields(Visitor *JXL_RESTRICT visitor) override;

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
