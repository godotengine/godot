// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_RCT_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_RCT_H_

#include "lib/jxl/modular/modular_image.h"

namespace jxl {

Status FwdRCT(Image &input, size_t begin_c, size_t rct_type, ThreadPool *pool);

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_ENC_RCT_H_
