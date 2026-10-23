// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_SQUEEZE_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_SQUEEZE_H_

#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/squeeze_params.h"

namespace jxl {

Status FwdSqueeze(Image &input, std::vector<SqueezeParams> parameters,
                  ThreadPool *pool);

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_ENC_SQUEEZE_H_
