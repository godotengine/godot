// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_LINALG_H_
#define LIB_JXL_LINALG_H_

// Linear algebra.

#include <array>

namespace jxl {

using Vector2 = std::array<double, 2>;
// NB: matrix2x2[row][column]
using Matrix2x2 = std::array<Vector2, 2>;

// A is symmetric, U is orthogonal, and A = U * Diagonal(diag) * Transpose(U).
void ConvertToDiagonal(const Matrix2x2& A, Vector2& diag, Matrix2x2& U);

}  // namespace jxl

#endif  // LIB_JXL_LINALG_H_
