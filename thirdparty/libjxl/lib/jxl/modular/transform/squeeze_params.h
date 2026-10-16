// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_SQUEEZE_PARAMS_H_
#define LIB_JXL_MODULAR_TRANSFORM_SQUEEZE_PARAMS_H_

#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"

namespace jxl {

struct SqueezeParams : public Fields {
  JXL_FIELDS_NAME(SqueezeParams)
  bool horizontal;
  bool in_place;
  uint32_t begin_c;
  uint32_t num_c;
  SqueezeParams();
  Status VisitFields(Visitor *JXL_RESTRICT visitor) override;
};

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_SQUEEZE_PARAMS_H_
