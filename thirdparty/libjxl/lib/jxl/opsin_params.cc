// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/opsin_params.h"

#include "lib/jxl/cms/opsin_params.h"

#define INVERSE_OPSIN_FROM_SPEC 1

#include "lib/jxl/base/matrix_ops.h"

namespace jxl {

const Matrix3x3& GetOpsinAbsorbanceInverseMatrix() {
#if INVERSE_OPSIN_FROM_SPEC
  return jxl::cms::DefaultInverseOpsinAbsorbanceMatrix();
#else   // INVERSE_OPSIN_FROM_SPEC
  // Compute the inverse opsin matrix from the forward matrix. Less precise
  // than taking the values from the specification, but must be used if the
  // forward transform is changed and the spec will require updating.
  static const Matrix3x3 const kInverse = [] {
    static Matrix3x3 inverse = kOpsinAbsorbanceMatrix;
    Inv3x3Matrix(inverse);
    return inverse;
  }();
  return kInverse;
#endif  // INVERSE_OPSIN_FROM_SPEC
}

void InitSIMDInverseMatrix(const Matrix3x3& inverse,
                           float* JXL_RESTRICT simd_inverse,
                           float intensity_target) {
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      size_t idx = (j * 3 + i) * 4;
      simd_inverse[idx] = simd_inverse[idx + 1] = simd_inverse[idx + 2] =
          simd_inverse[idx + 3] = inverse[j][i] * (255.0f / intensity_target);
    }
  }
}

}  // namespace jxl
