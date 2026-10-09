// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_MATRIX_OPS_H_
#define LIB_JXL_BASE_MATRIX_OPS_H_

// 3x3 matrix operations.

#include <array>
#include <cmath>  // abs
#include <cstddef>

#include "lib/jxl/base/status.h"

namespace jxl {

typedef std::array<float, 3> Vector3;
typedef std::array<double, 3> Vector3d;
typedef std::array<Vector3, 3> Matrix3x3;
typedef std::array<Vector3d, 3> Matrix3x3d;

// Computes C = A * B, where A, B, C are 3x3 matrices.
template <typename Matrix>
void Mul3x3Matrix(const Matrix& a, const Matrix& b, Matrix& c) {
  for (size_t x = 0; x < 3; x++) {
    alignas(16) Vector3d temp{b[0][x], b[1][x], b[2][x]};  // transpose
    for (size_t y = 0; y < 3; y++) {
      c[y][x] = a[y][0] * temp[0] + a[y][1] * temp[1] + a[y][2] * temp[2];
    }
  }
}

// Computes C = A * B, where A is 3x3 matrix and B is vector.
template <typename Matrix, typename Vector>
void Mul3x3Vector(const Matrix& a, const Vector& b, Vector& c) {
  for (size_t y = 0; y < 3; y++) {
    double e = 0;
    for (size_t x = 0; x < 3; x++) {
      e += a[y][x] * b[x];
    }
    c[y] = e;
  }
}

// Inverts a 3x3 matrix in place.
template <typename Matrix>
Status Inv3x3Matrix(Matrix& matrix) {
  // Intermediate computation is done in double precision.
  Matrix3x3d temp;
  temp[0][0] = static_cast<double>(matrix[1][1]) * matrix[2][2] -
               static_cast<double>(matrix[1][2]) * matrix[2][1];
  temp[0][1] = static_cast<double>(matrix[0][2]) * matrix[2][1] -
               static_cast<double>(matrix[0][1]) * matrix[2][2];
  temp[0][2] = static_cast<double>(matrix[0][1]) * matrix[1][2] -
               static_cast<double>(matrix[0][2]) * matrix[1][1];
  temp[1][0] = static_cast<double>(matrix[1][2]) * matrix[2][0] -
               static_cast<double>(matrix[1][0]) * matrix[2][2];
  temp[1][1] = static_cast<double>(matrix[0][0]) * matrix[2][2] -
               static_cast<double>(matrix[0][2]) * matrix[2][0];
  temp[1][2] = static_cast<double>(matrix[0][2]) * matrix[1][0] -
               static_cast<double>(matrix[0][0]) * matrix[1][2];
  temp[2][0] = static_cast<double>(matrix[1][0]) * matrix[2][1] -
               static_cast<double>(matrix[1][1]) * matrix[2][0];
  temp[2][1] = static_cast<double>(matrix[0][1]) * matrix[2][0] -
               static_cast<double>(matrix[0][0]) * matrix[2][1];
  temp[2][2] = static_cast<double>(matrix[0][0]) * matrix[1][1] -
               static_cast<double>(matrix[0][1]) * matrix[1][0];
  double det = matrix[0][0] * temp[0][0] + matrix[0][1] * temp[1][0] +
               matrix[0][2] * temp[2][0];
  if (std::abs(det) < 1e-10) {
    return JXL_FAILURE("Matrix determinant is too close to 0");
  }
  double idet = 1.0 / det;
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      matrix[j][i] = temp[j][i] * idet;
    }
  }
  return true;
}

}  // namespace jxl

#endif  // LIB_JXL_BASE_MATRIX_OPS_H_
