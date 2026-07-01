// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/transform.h"

#include <cinttypes>  // PRIu32

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/palette.h"
#include "lib/jxl/modular/transform/rct.h"
#include "lib/jxl/modular/transform/squeeze.h"

namespace jxl {

SqueezeParams::SqueezeParams() { Bundle::Init(this); }
Transform::Transform(TransformId id) {
  Bundle::Init(this);
  this->id = id;
}

Status Transform::Inverse(Image &input, const weighted::Header &wp_header,
                          ThreadPool *pool) const {
  JXL_DEBUG_V(6, "Input channels (%" PRIuS ", %" PRIuS " meta): ",
              input.channel.size(), input.nb_meta_channels);
  switch (id) {
    case TransformId::kRCT:
      return InvRCT(input, begin_c, rct_type, pool);
    case TransformId::kSqueeze:
      return InvSqueeze(input, squeezes, pool);
    case TransformId::kPalette:
      return InvPalette(input, begin_c, nb_colors, nb_deltas, predictor,
                        wp_header, pool);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)",
                         static_cast<unsigned int>(id));
  }
}

Status Transform::MetaApply(Image &input) {
  JXL_DEBUG_V(6, "MetaApply input: %s", input.DebugString().c_str());
  switch (id) {
    case TransformId::kRCT:
      JXL_DEBUG_V(2, "Transform: kRCT, rct_type=%" PRIu32, rct_type);
      return CheckEqualChannels(input, begin_c, begin_c + 2);
    case TransformId::kSqueeze:
      JXL_DEBUG_V(2, "Transform: kSqueeze:");
#if JXL_DEBUG_V_LEVEL >= 2
      {
        auto squeezes_copy = squeezes;
        if (squeezes_copy.empty()) {
          DefaultSqueezeParameters(&squeezes_copy, input);
        }
        for (const auto &params : squeezes_copy) {
          JXL_DEBUG_V(
              2,
              "  squeeze params: horizontal=%d, in_place=%d, begin_c=%" PRIu32
              ", num_c=%" PRIu32,
              params.horizontal, params.in_place, params.begin_c, params.num_c);
        }
      }
#endif
      return MetaSqueeze(input, &squeezes);
    case TransformId::kPalette:
      JXL_DEBUG_V(2,
                  "Transform: kPalette, begin_c=%" PRIu32 ", num_c=%" PRIu32
                  ", nb_colors=%" PRIu32 ", nb_deltas=%" PRIu32,
                  begin_c, num_c, nb_colors, nb_deltas);
      return MetaPalette(input, begin_c, begin_c + num_c - 1, nb_colors,
                         nb_deltas, lossy_palette);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)",
                         static_cast<unsigned int>(id));
  }
}

Status CheckEqualChannels(const Image &image, uint32_t c1, uint32_t c2) {
  if (c1 > image.channel.size() || c2 >= image.channel.size() || c2 < c1) {
    return JXL_FAILURE("Invalid channel range: %u..%u (there are only %" PRIuS
                       " channels)",
                       c1, c2, image.channel.size());
  }
  if (c1 < image.nb_meta_channels && c2 >= image.nb_meta_channels) {
    return JXL_FAILURE("Invalid: transforming mix of meta and nonmeta");
  }
  const auto &ch1 = image.channel[c1];
  for (size_t c = c1 + 1; c <= c2; c++) {
    const auto &ch2 = image.channel[c];
    if (ch1.w != ch2.w || ch1.h != ch2.h || ch1.hshift != ch2.hshift ||
        ch1.vshift != ch2.vshift) {
      return false;
    }
  }
  return true;
}

}  // namespace jxl
