// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_TRANSFORM_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_TRANSFORM_H_

#include "lib/jxl/fields.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

Status TransformForward(Transform &t, Image &input,
                        const weighted::Header &wp_header, ThreadPool *pool);

void compute_minmax(const Channel &ch, pixel_type *min, pixel_type *max);

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_ENC_TRANSFORM_H_
