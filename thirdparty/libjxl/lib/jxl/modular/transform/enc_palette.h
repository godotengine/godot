// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_H_

#include "lib/jxl/fields.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"

namespace jxl {

Status FwdPalette(Image &input, uint32_t begin_c, uint32_t end_c,
                  uint32_t &nb_colors, uint32_t &nb_deltas, bool ordered,
                  bool lossy, Predictor &predictor,
                  const weighted::Header &wp_header);

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_H_
