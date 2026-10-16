// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/squeeze_params.h"

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"

namespace jxl {

Status SqueezeParams::VisitFields(Visitor *JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &horizontal));
  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &in_place));
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Bits(3), BitsOffset(6, 8),
                                         BitsOffset(10, 72),
                                         BitsOffset(13, 1096), 0, &begin_c));
  JXL_QUIET_RETURN_IF_ERROR(
      visitor->U32(Val(1), Val(2), Val(3), BitsOffset(4, 4), 2, &num_c));
  return true;
}

}  // namespace jxl
