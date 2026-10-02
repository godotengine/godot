// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_SQUEEZE_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_SQUEEZE_H_

#include "lib/jxl/fields.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

Status FwdSqueeze(Image &input, std::vector<SqueezeParams> parameters,
                  ThreadPool *pool);

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_ENC_SQUEEZE_H_
