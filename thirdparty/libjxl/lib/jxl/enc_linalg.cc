// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_linalg.h"

#include <cmath>

#include "lib/jxl/base/status.h"

namespace jxl {

void ConvertToDiagonal(const Matrix2x2& A, Vector2& diag, Matrix2x2& U) {
  // Check A is symmetric.
  JXL_DASSERT(std::abs(A[0][1] - A[1][0]) < 1e-15);

  double b = -(A[0][0] + A[1][1]);
  double c = A[0][0] * A[1][1] - A[0][1] * A[0][1];
  double d = b * b - 4.0 * c;

  if (std::abs(A[0][1]) < 1e-10 || d < 0) {
    // Already diagonal.
    diag[0] = A[0][0];
    diag[1] = A[1][1];
    U[0][0] = U[1][1] = 1.0;
    U[0][1] = U[1][0] = 0.0;
    return;
  }

  double sqd = std::sqrt(d);
  double l1 = (-b - sqd) * 0.5;
  double l2 = (-b + sqd) * 0.5;

  Vector2 v1 = {A[0][0] - l1, A[1][0]};
  double v1n = 1.0 / std::hypot(v1[0], v1[1]);
  v1[0] = v1[0] * v1n;
  v1[1] = v1[1] * v1n;

  diag[0] = l1;
  diag[1] = l2;

  U[0][0] = v1[1];
  U[0][1] = -v1[0];
  U[1][0] = v1[0];
  U[1][1] = v1[1];
}

}  // namespace jxl
