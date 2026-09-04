// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_OPSIN_PARAMS_H_
#define LIB_JXL_OPSIN_PARAMS_H_

// Constants that define the XYB color space.

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/matrix_ops.h"

namespace jxl {

// Returns 3x3 row-major matrix inverse of kOpsinAbsorbanceMatrix.
// opsin_image_test verifies this is actually the inverse.
const Matrix3x3& GetOpsinAbsorbanceInverseMatrix();

void InitSIMDInverseMatrix(const Matrix3x3& inverse,
                           float* JXL_RESTRICT simd_inverse,
                           float intensity_target);

}  // namespace jxl

#endif  // LIB_JXL_OPSIN_PARAMS_H_
