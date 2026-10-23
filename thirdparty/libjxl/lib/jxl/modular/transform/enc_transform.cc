// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/enc_transform.h"

#include <cstddef>
#include <limits>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/enc_palette.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include "lib/jxl/modular/transform/enc_squeeze.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

Status TransformForward(Transform &t, Image &input,
                        const weighted::Header &wp_header, ThreadPool *pool) {
  switch (t.id) {
    case TransformId::kRCT:
      return FwdRct(input, t.begin_c, t.rct_type, pool);
    case TransformId::kSqueeze:
      return FwdSqueeze(input, t.squeezes, pool);
    case TransformId::kPalette:
      return FwdPalette(input, t.begin_c, t.begin_c + t.num_c - 1, t.nb_colors,
                        t.nb_deltas, t.ordered_palette, t.lossy_palette,
                        t.predictor, wp_header);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)",
                         static_cast<unsigned int>(t.id));
  }
}

void compute_minmax(const Channel &ch, pixel_type *min, pixel_type *max) {
  pixel_type realmin = std::numeric_limits<pixel_type>::max();
  pixel_type realmax = std::numeric_limits<pixel_type>::min();
  for (size_t y = 0; y < ch.h; y++) {
    const pixel_type *JXL_RESTRICT p = ch.Row(y);
    for (size_t x = 0; x < ch.w; x++) {
      if (p[x] < realmin) realmin = p[x];
      if (p[x] > realmax) realmax = p[x];
    }
  }

  if (min) *min = realmin;
  if (max) *max = realmax;
}

}  // namespace jxl
