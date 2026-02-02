// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "external/linalg.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "math-util.inc"
#include "pprinter.hh"
#include "value-pprint.hh"
#include "prim-types.hh"
#include "tiny-format.hh"
#include "value-types.hh"
#include "xform.hh"
#include "common-macros.inc"

// Use pxrUSD approach to generate rotation matrix.
// This will give (probably) identical xformOps matrix operation, but the resulting matrix contains some numerical error.
// https://github.com/PixarAnimationStudios/USD/issues/2136
//#define PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION

namespace tinyusdz {

using matrix44d = linalg::aliases::double4x4;
using matrix33d = linalg::aliases::double3x3;
using matrix22d = linalg::aliases::double2x2;
using double3x3 = linalg::aliases::double3x3;
using double3 = linalg::aliases::double3;
using double4 = linalg::aliases::double4;

constexpr uint32_t kIdentityMaxUlps = 1;

bool is_identity(const value::matrix2f &m) {
  return math::almost_equals_by_ulps(m.m[0][0], 1.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[0][1], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][0], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][1], 0.0f, kIdentityMaxUlps);
}

bool is_identity(const value::matrix3f &m) {
  return math::almost_equals_by_ulps(m.m[0][0], 1.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[0][1], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[0][2], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][0], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][1], 1.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][2], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[2][0], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[2][1], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[2][2], 1.0f, kIdentityMaxUlps);
}

bool is_identity(const value::matrix4f &m) {
  return math::almost_equals_by_ulps(m.m[0][0], 1.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[0][1], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[0][2], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[0][3], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][0], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][1], 1.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][2], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[1][3], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[2][0], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[2][1], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[2][2], 1.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[2][3], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[3][0], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[3][1], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[3][2], 0.0f, kIdentityMaxUlps) &&
         math::almost_equals_by_ulps(m.m[3][3], 1.0f, kIdentityMaxUlps);
}

bool is_identity(const value::matrix2d &m) {
  return math::almost_equals_by_ulps(m.m[0][0], 1.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[0][1], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][0], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][1], 0.0, uint64_t(kIdentityMaxUlps));
}

bool is_identity(const value::matrix3d &m) {
  return math::almost_equals_by_ulps(m.m[0][0], 1.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[0][1], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[0][2], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][0], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][1], 1.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][2], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[2][0], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[2][1], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[2][2], 1.0, uint64_t(kIdentityMaxUlps));
}

bool is_identity(const value::matrix4d &m) {
  return math::almost_equals_by_ulps(m.m[0][0], 1.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[0][1], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[0][2], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[0][3], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][0], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][1], 1.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][2], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[1][3], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[2][0], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[2][1], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[2][2], 1.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[2][3], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[3][0], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[3][1], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[3][2], 0.0, uint64_t(kIdentityMaxUlps)) &&
         math::almost_equals_by_ulps(m.m[3][3], 1.0, uint64_t(kIdentityMaxUlps));
}

bool is_close(const value::matrix2f &a, const value::matrix2f &b, const float eps) {
  return math::is_close(a.m[0][0], b.m[0][0], eps) &&
         math::is_close(a.m[0][1], b.m[0][1], eps) &&
         math::is_close(a.m[1][0], b.m[1][0], eps) &&
         math::is_close(a.m[1][1], b.m[1][1], eps);
}

bool is_close(const value::matrix3f &a, const value::matrix3f &b, const float eps) {
  return math::is_close(a.m[0][0], b.m[0][0], eps) &&
         math::is_close(a.m[0][1], b.m[0][1], eps) &&
         math::is_close(a.m[0][2], b.m[0][2], eps) &&
         math::is_close(a.m[1][0], b.m[1][0], eps) &&
         math::is_close(a.m[1][1], b.m[1][1], eps) &&
         math::is_close(a.m[1][2], b.m[1][2], eps) &&
         math::is_close(a.m[2][0], b.m[2][0], eps) &&
         math::is_close(a.m[2][1], b.m[2][1], eps) &&
         math::is_close(a.m[2][2], b.m[2][2], eps);
}

bool is_close(const value::matrix4f &a, const value::matrix4f &b, const float eps) {
  return math::is_close(a.m[0][0], b.m[0][0], eps) &&
         math::is_close(a.m[0][1], b.m[0][1], eps) &&
         math::is_close(a.m[0][2], b.m[0][2], eps) &&
         math::is_close(a.m[0][3], b.m[0][3], eps) &&
         math::is_close(a.m[1][0], b.m[1][0], eps) &&
         math::is_close(a.m[1][1], b.m[1][1], eps) &&
         math::is_close(a.m[1][2], b.m[1][2], eps) &&
         math::is_close(a.m[1][3], b.m[1][3], eps) &&
         math::is_close(a.m[2][0], b.m[2][0], eps) &&
         math::is_close(a.m[2][1], b.m[2][1], eps) &&
         math::is_close(a.m[2][2], b.m[2][2], eps) &&
         math::is_close(a.m[2][3], b.m[2][3], eps) &&
         math::is_close(a.m[3][0], b.m[3][0], eps) &&
         math::is_close(a.m[3][1], b.m[3][1], eps) &&
         math::is_close(a.m[3][2], b.m[3][2], eps) &&
         math::is_close(a.m[3][3], b.m[3][3], eps);
}


bool is_close(const value::matrix2d &a, const value::matrix2d &b, const double eps) {
  return math::is_close(a.m[0][0], b.m[0][0], eps) &&
         math::is_close(a.m[0][1], b.m[0][1], eps) &&
         math::is_close(a.m[1][0], b.m[1][0], eps) &&
         math::is_close(a.m[1][1], b.m[1][1], eps);
}

bool is_close(const value::matrix3d &a, const value::matrix3d &b, const double eps) {
  return math::is_close(a.m[0][0], b.m[0][0], eps) &&
         math::is_close(a.m[0][1], b.m[0][1], eps) &&
         math::is_close(a.m[0][2], b.m[0][2], eps) &&
         math::is_close(a.m[1][0], b.m[1][0], eps) &&
         math::is_close(a.m[1][1], b.m[1][1], eps) &&
         math::is_close(a.m[1][2], b.m[1][2], eps) &&
         math::is_close(a.m[2][0], b.m[2][0], eps) &&
         math::is_close(a.m[2][1], b.m[2][1], eps) &&
         math::is_close(a.m[2][2], b.m[2][2], eps);
}

bool is_close(const value::matrix4d &a, const value::matrix4d &b, const double eps) {
  return math::is_close(a.m[0][0], b.m[0][0], eps) &&
         math::is_close(a.m[0][1], b.m[0][1], eps) &&
         math::is_close(a.m[0][2], b.m[0][2], eps) &&
         math::is_close(a.m[0][3], b.m[0][3], eps) &&
         math::is_close(a.m[1][0], b.m[1][0], eps) &&
         math::is_close(a.m[1][1], b.m[1][1], eps) &&
         math::is_close(a.m[1][2], b.m[1][2], eps) &&
         math::is_close(a.m[1][3], b.m[1][3], eps) &&
         math::is_close(a.m[2][0], b.m[2][0], eps) &&
         math::is_close(a.m[2][1], b.m[2][1], eps) &&
         math::is_close(a.m[2][2], b.m[2][2], eps) &&
         math::is_close(a.m[2][3], b.m[2][3], eps) &&
         math::is_close(a.m[3][0], b.m[3][0], eps) &&
         math::is_close(a.m[3][1], b.m[3][1], eps) &&
         math::is_close(a.m[3][2], b.m[3][2], eps) &&
         math::is_close(a.m[3][3], b.m[3][3], eps);
}

value::quatf to_quaternion(const value::float3 &axis, const float angle) {

  // Use sin_pi and cos_pi for better accuracy.
  float s = float(math::sin_pi(double(angle)/2.0/180.0));
  float c = float(math::cos_pi(double(angle)/2.0/180.0));

  value::quatf q;
  q.imag[0] = axis[0] * s;
  q.imag[1] = axis[1] * s;
  q.imag[2] = axis[2] * s;
  q.real = c;

  return q;
}

value::quatd to_quaternion(const value::double3 &axis, const double angle) {

  // Use sin_pi and cos_pi for better accuracy.
  double s = math::sin_pi(angle/2.0/180.0);
  double c = math::cos_pi(angle/2.0/180.0);

  value::quatd q;
  q.imag[0] = axis[0] * s;
  q.imag[1] = axis[1] * s;
  q.imag[2] = axis[2] * s;
  q.real = c;

  return q;
}


// linalg quat memory layout: (x, y, z, w)
// value::quat memory layout: (imag[0], imag[1], imag[2], real)

value::matrix3d to_matrix3x3(const value::quath &q) {
  double3x3 m33 = linalg::qmat<double>(
      {double(half_to_float(q.imag[0])), double(half_to_float(q.imag[1])),
       double(half_to_float(q.imag[2])), double(half_to_float(q.real))});

  value::matrix3d m;
  Identity(&m);

  memcpy(m.m, &m33[0][0], sizeof(double) * 3 * 3);

  return m;
}

value::matrix3d to_matrix3x3(const value::quatf &q) {
  double3x3 m33 = linalg::qmat<double>({double(q.imag[0]), double(q.imag[1]),
                                        double(q.imag[2]), double(q.real)});

  value::matrix3d m;
  Identity(&m);

  memcpy(m.m, &m33[0][0], sizeof(double) * 3 * 3);

  return m;
}

value::matrix3d to_matrix3x3(const value::quatd &q) {
  double3x3 m33 =
      linalg::qmat<double>({q.imag[0], q.imag[1], q.imag[2], q.real});

  value::matrix3d m;
  Identity(&m);

  memcpy(m.m, &m33[0][0], sizeof(double) * 3 * 3);

  return m;
}

value::matrix4d to_matrix(const value::matrix3d &m33,
                          const value::double3 &tx) {
  value::matrix4d m;
  Identity(&m);

  m.m[0][0] = m33.m[0][0];
  m.m[0][1] = m33.m[0][1];
  m.m[0][2] = m33.m[0][2];
  m.m[1][0] = m33.m[1][0];
  m.m[1][1] = m33.m[1][1];
  m.m[1][2] = m33.m[1][2];
  m.m[2][0] = m33.m[2][0];
  m.m[2][1] = m33.m[2][1];
  m.m[2][2] = m33.m[2][2];

  m.m[3][0] = tx[0];
  m.m[3][1] = tx[1];
  m.m[3][2] = tx[2];

  return m;
}

value::matrix3d to_matrix3x3(const value::matrix4d &m44, value::double3 *tx) {
  value::matrix3d m;
  Identity(&m);

  m.m[0][0] = m44.m[0][0];
  m.m[0][1] = m44.m[0][1];
  m.m[0][2] = m44.m[0][2];
  m.m[1][0] = m44.m[1][0];
  m.m[1][1] = m44.m[1][1];
  m.m[1][2] = m44.m[1][2];
  m.m[2][0] = m44.m[2][0];
  m.m[2][1] = m44.m[2][1];
  m.m[2][2] = m44.m[2][2];

  if (tx) {
    (*tx)[0] = m44.m[3][0];
    (*tx)[1] = m44.m[3][1];
    (*tx)[2] = m44.m[3][2];
  }

  return m;
}

value::matrix4d to_matrix(const value::quath &q) {
  // using double4 = linalg::aliases::double4;

  double3x3 m33 = linalg::qmat<double>(
      {double(half_to_float(q.imag[0])), double(half_to_float(q.imag[1])),
       double(half_to_float(q.imag[2])), double(half_to_float(q.real))});

  value::matrix4d m;
  Identity(&m);

  m.m[0][0] = m33[0][0];
  m.m[0][1] = m33[0][1];
  m.m[0][2] = m33[0][2];
  m.m[1][0] = m33[1][0];
  m.m[1][1] = m33[1][1];
  m.m[1][2] = m33[1][2];
  m.m[2][0] = m33[2][0];
  m.m[2][1] = m33[2][1];
  m.m[2][2] = m33[2][2];

  return m;
}

value::matrix4d to_matrix(const value::quatf &q) {
  double3x3 m33 = linalg::qmat<double>({double(q.imag[0]), double(q.imag[1]),
                                        double(q.imag[2]), double(q.real)});

  value::matrix4d m;
  Identity(&m);

  m.m[0][0] = m33[0][0];
  m.m[0][1] = m33[0][1];
  m.m[0][2] = m33[0][2];
  m.m[1][0] = m33[1][0];
  m.m[1][1] = m33[1][1];
  m.m[1][2] = m33[1][2];
  m.m[2][0] = m33[2][0];
  m.m[2][1] = m33[2][1];
  m.m[2][2] = m33[2][2];

  return m;
}

value::matrix4d to_matrix(const value::quatd &q) {
  double3x3 m33 =
      linalg::qmat<double>({q.imag[0], q.imag[1], q.imag[2], q.real});

  value::matrix4d m;
  Identity(&m);

  m.m[0][0] = m33[0][0];
  m.m[0][1] = m33[0][1];
  m.m[0][2] = m33[0][2];
  m.m[1][0] = m33[1][0];
  m.m[1][1] = m33[1][1];
  m.m[1][2] = m33[1][2];
  m.m[2][0] = m33[2][0];
  m.m[2][1] = m33[2][1];
  m.m[2][2] = m33[2][2];

  return m;
}

value::matrix4d inverse(const value::matrix4d &_m) {
  matrix44d m;
  // memory layout is same
  memcpy(&m[0][0], _m.m, sizeof(double) * 4 * 4);

  matrix44d inv_m = linalg::inverse(m);

  value::matrix4d outm;

  memcpy(outm.m, &inv_m[0][0], sizeof(double) * 4 * 4);

  return outm;
}

value::matrix3d inverse(const value::matrix3d &_m) {
  matrix33d m;
  // memory layout is same
  memcpy(&m[0][0], _m.m, sizeof(double) * 3 * 3);

  matrix33d inv_m = linalg::inverse(m);

  value::matrix3d outm;

  memcpy(outm.m, &inv_m[0][0], sizeof(double) * 3 * 3);

  return outm;
}

double determinant(const value::matrix4d &_m) {
  matrix44d m;
  // memory layout is same
  memcpy(&m[0][0], _m.m, sizeof(double) * 4 * 4);

  double det = linalg::determinant(m);

  return det;
}

double determinant(const value::matrix3d &_m) {
  matrix33d m;
  // memory layout is same
  memcpy(&m[0][0], _m.m, sizeof(double) * 3 * 3);

  double det = linalg::determinant(m);

  return det;
}

bool inverse(const value::matrix4d &_m, value::matrix4d &inv_m, double eps) {
  double det = determinant(_m);

  if (math::is_close(std::fabs(det), 0.0, eps)) {
    return false;
  }

  inv_m = inverse(_m);
  return true;
}

bool inverse(const value::matrix3d &_m, value::matrix3d &inv_m, double eps) {
  double det = determinant(_m);

  if (math::is_close(std::fabs(det), 0.0, eps)) {
    return false;
  }

  inv_m = inverse(_m);
  return true;
}

value::matrix2d transpose(const value::matrix2d &_m) {
  matrix22d m;
  matrix22d tm;
  // memory layout is same
  memcpy(&m[0][0], _m.m, sizeof(double) * 2 * 2);
  tm = linalg::transpose(m);

  value::matrix2d dst;

  // memory layout is same
  memcpy(&dst.m[0][0], &tm[0][0], sizeof(double) * 2 * 2);

  return dst;
}

value::matrix3d transpose(const value::matrix3d &_m) {
  matrix33d m;
  matrix33d tm;
  // memory layout is same
  memcpy(&m[0][0], _m.m, sizeof(double) * 3 * 3);
  tm = linalg::transpose(m);

  value::matrix3d dst;

  // memory layout is same
  memcpy(&dst.m[0][0], &tm[0][0], sizeof(double) * 3 * 3);

  return dst;
}

value::matrix4d transpose(const value::matrix4d &_m) {
  matrix44d m;
  matrix44d tm;
  // memory layout is same
  memcpy(&m[0][0], _m.m, sizeof(double) * 4 * 4);
  tm = linalg::transpose(m);

  value::matrix4d dst;

  // memory layout is same
  memcpy(&dst.m[0][0], &tm[0][0], sizeof(double) * 4 * 4);

  return dst;
}

value::float4 matmul(const value::matrix4d &m, const value::float4 &p) {
  return value::MultV<value::matrix4d, value::float4, double, float, 4>(m, p);
}

value::double4 matmul(const value::matrix4d &m, const value::double4 &p) {
  return value::MultV<value::matrix4d, value::double4, double, double, 4>(m, p);
}

namespace {

///
/// Xform evaluation with method chain style.
/// so if you want to get RotateXYZ,
///
/// xRot * yRot * zRot
///
/// this is implemented in C++ as
///
/// XformEvaluator xe
/// xe.RotateX()
/// xe.RotateY()
/// xe.RotateZ()
///
/// NOTE: Matrix multiplication order is post-multiply in XformEvaluator for C++ readabilty
/// (otherwise we need to invoke xe.RotateZ(), xe.RotateY() then xe.RotateX())
///
class XformEvaluator {
 public:
  XformEvaluator() { Identity(&m); }

  XformEvaluator &RotateX(const double angle) {  // in degrees

    value::matrix4d rm = value::matrix4d::identity();

    double k = angle / 180.0;
    double c = math::cos_pi(k);
    double s = math::sin_pi(k);

    rm.m[1][1] = c;
    rm.m[1][2] = s;
    rm.m[2][1] = -s;
    rm.m[2][2] = c;

    m = m * rm;

    return (*this);
  }

  XformEvaluator &RotateY(const double angle) {  // in degrees

    value::matrix4d rm = value::matrix4d::identity();

    double k = angle / 180.0;
    double c = math::cos_pi(k);
    double s = math::sin_pi(k);

    rm.m[0][0] = c;
    rm.m[0][2] = -s;
    rm.m[2][0] = s;
    rm.m[2][2] = c;

    m = m * rm;

    return (*this);
  }

  XformEvaluator &RotateZ(const double angle) {  // in degrees

    //double rad = math::radian(angle);

    value::matrix4d rm = value::matrix4d::identity();

    double k = angle / 180.0;
    double c = math::cos_pi(k);
    double s = math::sin_pi(k);

    rm.m[0][0] = c;
    rm.m[0][1] = s;
    rm.m[1][0] = -s;
    rm.m[1][1] = c;

    m = m * rm;

    return (*this);
  }

  // From arbitrary rotation axis
  XformEvaluator &Rotation(const double3 &axis, const double angle) {  // in degrees

    // linalg uses radians
    //double4 q = linalg::rotation_quat(axis, math::radian(angle));

    value::quatd q = to_quaternion(axis, angle);

    double3x3 m33 =
        linalg::qmat<double>({q[0], q[1], q[2], q[3]});

    value::matrix4d rm;

    rm.m[0][0] = m33[0][0];
    rm.m[0][1] = m33[0][1];
    rm.m[0][2] = m33[0][2];
    rm.m[0][3] = 0.0;

    rm.m[1][0] = m33[1][0];
    rm.m[1][1] = m33[1][1];
    rm.m[1][2] = m33[1][2];
    rm.m[1][3] = 0.0;

    rm.m[2][0] = m33[2][0];
    rm.m[2][1] = m33[2][1];
    rm.m[2][2] = m33[2][2];
    rm.m[2][3] = 0.0;

    rm.m[3][0] = 0.0;
    rm.m[3][1] = 0.0;
    rm.m[3][2] = 0.0;
    rm.m[3][3] = 1.0;

    m = m * rm;

    return (*this);
  }

  std::string error() const { return err; }

  nonstd::expected<value::matrix4d, std::string> result() const {
    if (err.empty()) {
      return m;
    }

    return nonstd::make_unexpected(err);
  }

  std::string err;
  value::matrix4d m;
};

}  // namespace

bool Xformable::EvaluateXformOps(double t,
                                 value::TimeSampleInterpolationType tinterp,
                                 value::matrix4d *out_matrix,
                                 bool *resetXformStack,
                                 std::string *err) const {
  const auto RotateABC =
      [t, tinterp](const XformOp &x) -> nonstd::expected<value::matrix4d, std::string> {
    value::double3 v;
    if (auto h = x.get_value<value::half3>(t, tinterp)) {
      v[0] = double(half_to_float(h.value()[0]));
      v[1] = double(half_to_float(h.value()[1]));
      v[2] = double(half_to_float(h.value()[2]));
    } else if (auto f = x.get_value<value::float3>(t, tinterp)) {
      v[0] = double(f.value()[0]);
      v[1] = double(f.value()[1]);
      v[2] = double(f.value()[2]);
    } else if (auto d = x.get_value<value::double3>(t, tinterp)) {
      v = d.value();
    } else {
      if (x.suffix.empty()) {
        return nonstd::make_unexpected(
            fmt::format("`{}` is not half3, float3 or double3 type.\n",
                        to_string(x.op_type)));
      } else {
        return nonstd::make_unexpected(
            fmt::format("`{}:{}` is not half3, float3 or double3 type.\n",
                        to_string(x.op_type), x.suffix));
      }
    }

    // invert input, and compute concatenated matrix
    // inv(ABC) = inv(A) x inv(B) x inv(C)
    // as done in pxrUSD.

    if (x.inverted) {
      v[0] = -v[0];
      v[1] = -v[1];
      v[2] = -v[2];
    }

    double xAngle = v[0];
    double yAngle = v[1];
    double zAngle = v[2];

    XformEvaluator eval;

    DCOUT("angles = " << xAngle << ", " << yAngle << ", " << zAngle);
    if (x.inverted) {
      DCOUT("!inverted!\n");
      if (x.op_type == XformOp::OpType::RotateXYZ) {
        // TODO: Apply defined switch for all Rotate*** op.
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
#else
        eval.RotateZ(zAngle);
        eval.RotateY(yAngle);
        eval.RotateX(xAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateXZY) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
#else
        eval.RotateY(yAngle);
        eval.RotateZ(zAngle);
        eval.RotateX(xAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateYXZ) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
#else
        eval.RotateZ(zAngle);
        eval.RotateX(xAngle);
        eval.RotateY(yAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateYZX) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
#else
        eval.RotateX(xAngle);
        eval.RotateZ(zAngle);
        eval.RotateY(yAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateZYX) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
#else
        eval.RotateX(xAngle);
        eval.RotateY(yAngle);
        eval.RotateZ(zAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateZXY) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
#else
        eval.RotateY(yAngle);
        eval.RotateX(xAngle);
        eval.RotateZ(zAngle);
#endif
      } else {
        /// ???
        return nonstd::make_unexpected("[InternalError] RotateABC");
      }
    } else {
      if (x.op_type == XformOp::OpType::RotateXYZ) {

#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
#else
        eval.RotateX(xAngle);
        eval.RotateY(yAngle);
        eval.RotateZ(zAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateXZY) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
#else
        eval.RotateX(xAngle);
        eval.RotateZ(zAngle);
        eval.RotateY(yAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateYXZ) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
#else
        eval.RotateY(yAngle);
        eval.RotateX(xAngle);
        eval.RotateZ(zAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateYZX) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
#else
        eval.RotateY(yAngle);
        eval.RotateZ(zAngle);
        eval.RotateX(xAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateZYX) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
#else
        eval.RotateZ(zAngle);
        eval.RotateY(yAngle);
        eval.RotateX(xAngle);
#endif
      } else if (x.op_type == XformOp::OpType::RotateZXY) {
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        eval.Rotation({0.0, 0.0, 1.0}, zAngle);
        eval.Rotation({1.0, 0.0, 0.0}, xAngle);
        eval.Rotation({0.0, 1.0, 0.0}, yAngle);
#else
        eval.RotateZ(zAngle);
        eval.RotateX(xAngle);
        eval.RotateY(yAngle);
#endif
      } else {
        /// ???
        return nonstd::make_unexpected("[InternalError] RotateABC");
      }
    }

    return eval.result();
  };

  // Concat matrices
  //
  // Matrix concatenation ordering is its appearance order(right to left)
  // This is same with a notation in math equation: i.e,
  //
  // xformOpOrder = [A, B, C]
  //
  // M = A x B x C
  //
  // p' = A x B x C x p
  //
  // in post-multiply order.
  //
  // But in pre-multiply order system(pxrUSD and TinyUSDZ),
  // C++ code is
  //
  // p' = p x C x B x A
  //
  //
  value::matrix4d cm;
  Identity(&cm);

  for (size_t i = 0; i < xformOps.size(); i++) {
    const auto x = xformOps[i];

    value::matrix4d m;  // local matrix
    Identity(&m);

    switch (x.op_type) {
      case XformOp::OpType::ResetXformStack: {
        if (i != 0) {
          if (err) {
            (*err) +=
                "!resetXformStack! should only appear at the first element of "
                "xformOps\n";
          }
          return false;
        }

        // Notify resetting previous(parent node's) matrices
        if (resetXformStack) {
          (*resetXformStack) = true;
        }
        break;
      }
      case XformOp::OpType::Transform: {
        if (auto sxf = x.get_value<value::matrix4f>(t, tinterp)) {
          value::matrix4f mf = sxf.value();
          for (size_t j = 0; j < 4; j++) {
            for (size_t k = 0; k < 4; k++) {
              m.m[j][k] = double(mf.m[j][k]);
            }
          }
        } else if (auto sxd = x.get_value<value::matrix4d>(t, tinterp)) {
          m = sxd.value();
        } else {
          if (err) {
            (*err) += "`xformOp:transform` is not matrix4f or matrix4d type.\n";
          }
          return false;
        }

        if (x.inverted) {
          // Singular check.
          // pxrUSD uses 1e-9
          double det = determinant(m);

          if (std::fabs(det) < 1e-9) {
            if (err) {
              if (x.suffix.empty()) {
                (*err) +=
                    "`xformOp:transform` is singular matrix and cannot be "
                    "inverted.\n";
              } else {
                (*err) += fmt::format(
                    "`xformOp:transform:{}` is singular matrix and cannot be "
                    "inverted.\n",
                    x.suffix);
              }
            }

            return false;
          }

          m = inverse(m);
        }

        break;
      }
      case XformOp::OpType::Scale: {
        double sx, sy, sz;

        if (auto sxh = x.get_value<value::half3>(t, tinterp)) {
          sx = double(half_to_float(sxh.value()[0]));
          sy = double(half_to_float(sxh.value()[1]));
          sz = double(half_to_float(sxh.value()[2]));
        } else if (auto sxf = x.get_value<value::float3>(t, tinterp)) {
          sx = double(sxf.value()[0]);
          sy = double(sxf.value()[1]);
          sz = double(sxf.value()[2]);
        } else if (auto sxd = x.get_value<value::double3>(t, tinterp)) {
          sx = sxd.value()[0];
          sy = sxd.value()[1];
          sz = sxd.value()[2];
        } else {
          if (err) {
            (*err) += "`xformOp:scale` is not half3, float3 or double3 type.\n";
          }
          return false;
        }

        if (x.inverted) {
          // FIXME: Safe division
          sx = 1.0 / sx;
          sy = 1.0 / sy;
          sz = 1.0 / sz;
        }

        m.m[0][0] = sx;
        m.m[1][1] = sy;
        m.m[2][2] = sz;

        break;
      }
      case XformOp::OpType::Translate: {
        double tx, ty, tz;
        if (auto txh = x.get_value<value::half3>(t, tinterp)) {
          tx = double(half_to_float(txh.value()[0]));
          ty = double(half_to_float(txh.value()[1]));
          tz = double(half_to_float(txh.value()[2]));
        } else if (auto txf = x.get_value<value::float3>(t, tinterp)) {
          tx = double(txf.value()[0]);
          ty = double(txf.value()[1]);
          tz = double(txf.value()[2]);
        } else if (auto txd = x.get_value<value::double3>(t, tinterp)) {
          tx = txd.value()[0];
          ty = txd.value()[1];
          tz = txd.value()[2];
        } else {
          if (err) {
            (*err) +=
                "`xformOp:translate` is not half3, float3 or double3 type.\n";
          }
          return false;
        }

        if (x.inverted) {
          tx = -tx;
          ty = -ty;
          tz = -tz;
        }

        m.m[3][0] = tx;
        m.m[3][1] = ty;
        m.m[3][2] = tz;

        break;
      }
      // FIXME: Validate ROTATE_X, _Y, _Z implementation
      case XformOp::OpType::RotateX: {
        double angle;  // in degrees
        if (auto h = x.get_value<value::half>(t, tinterp)) {
          angle = double(half_to_float(h.value()));
        } else if (auto f = x.get_value<float>(t, tinterp)) {
          angle = double(f.value());
        } else if (auto d = x.get_value<double>(t, tinterp)) {
          angle = d.value();
        } else {
          if (err) {
            if (x.suffix.empty()) {
              (*err) +=
                  "`xformOp:rotateX` is not half, float or double type.\n";
            } else {
              (*err) += fmt::format(
                  "`xformOp:rotateX:{}` is not half, float or double type.\n",
                  x.suffix);
            }
          }
          return false;
        }

        XformEvaluator xe;
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        xe.Rotation({1.0, 0.0, 0.0}, angle);
#else
        xe.RotateX(angle);
#endif
        auto ret = xe.result();

        if (ret) {
          m = ret.value();
        } else {
          if (err) {
            (*err) += ret.error();
          }
          return false;
        }
        break;
      }
      case XformOp::OpType::RotateY: {
        double angle;  // in degrees
        if (auto h = x.get_value<value::half>(t, tinterp)) {
          angle = double(half_to_float(h.value()));
        } else if (auto f = x.get_value<float>(t, tinterp)) {
          angle = double(f.value());
        } else if (auto d = x.get_value<double>(t, tinterp)) {
          angle = d.value();
        } else {
          if (err) {
            if (x.suffix.empty()) {
              (*err) +=
                  "`xformOp:rotateY` is not half, float or double type.\n";
            } else {
              (*err) += fmt::format(
                  "`xformOp:rotateY:{}` is not half, float or double type.\n",
                  x.suffix);
            }
          }
          return false;
        }

        XformEvaluator xe;
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        xe.Rotation({0.0, 1.0, 0.0}, angle);
#else
        xe.RotateY(angle);
#endif
        auto ret = xe.result();

        if (ret) {
          m = ret.value();
        } else {
          if (err) {
            (*err) += ret.error();
          }
          return false;
        }
        break;
      }
      case XformOp::OpType::RotateZ: {
        double angle;  // in degrees
        if (auto h = x.get_value<value::half>(t, tinterp)) {
          angle = double(half_to_float(h.value()));
        } else if (auto f = x.get_value<float>(t, tinterp)) {
          angle = double(f.value());
        } else if (auto d = x.get_value<double>(t, tinterp)) {
          angle = d.value();
        } else {
          if (err) {
            if (x.suffix.empty()) {
              (*err) +=
                  "`xformOp:rotateZ` is not half, float or double type.\n";
            } else {
              (*err) += fmt::format(
                  "`xformOp:rotateZ:{}` is not half, float or double type.\n",
                  x.suffix);
            }
          }
          return false;
        }

        XformEvaluator xe;
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
        xe.Rotation({0.0, 0.0, 1.0}, angle);
#else
        xe.RotateZ(angle);
#endif
        auto ret = xe.result();

        if (ret) {
          m = ret.value();
        } else {
          if (err) {
            (*err) += ret.error();
          }
          return false;
        }
        break;
      }
      case XformOp::OpType::Orient: {
        // value::quat stores elements in (x, y, z, w)
        // linalg::quat also stores elements in (x, y, z, w)

        value::matrix3d rm;
        if (auto h = x.get_value<value::quath>(t, tinterp)) {
          rm = to_matrix3x3(h.value());
        } else if (auto f = x.get_value<value::quatf>(t, tinterp)) {
          rm = to_matrix3x3(f.value());
        } else if (auto d = x.get_value<value::quatd>(t, tinterp)) {
          rm = to_matrix3x3(d.value());
        } else {
          if (err) {
            if (x.suffix.empty()) {
              (*err) += "`xformOp:orient` is not quath, quatf or quatd type.\n";
            } else {
              (*err) += fmt::format(
                  "`xformOp:orient:{}` is not quath, quatf or quatd type.\n",
                  x.suffix);
            }
          }
          return false;
        }

        // FIXME: invert before getting matrix.
        if (x.inverted) {
          value::matrix3d inv_rm;
          if (inverse(rm, inv_rm)) {
          } else {
            if (err) {
              if (x.suffix.empty()) {
                (*err) +=
                    "`xformOp:orient` is singular and cannot be inverted.\n";
              } else {
                (*err) += fmt::format(
                    "`xformOp:orient:{}` is singular and cannot be inverted.\n",
                    x.suffix);
              }
            }
          }

          rm = inv_rm;
        }

        m = to_matrix(rm, {0.0, 0.0, 0.0});

        break;
      }

      case XformOp::OpType::RotateXYZ:
      case XformOp::OpType::RotateXZY:
      case XformOp::OpType::RotateYXZ:
      case XformOp::OpType::RotateYZX:
      case XformOp::OpType::RotateZXY:
      case XformOp::OpType::RotateZYX: {
        auto ret = RotateABC(x);

        if (!ret) {
          (*err) += ret.error();
          return false;
        }

        m = ret.value();
      }
    }

    cm = m * cm;  // `m` fist for pre-multiply system.
  }

  (*out_matrix) = cm;

  return true;
}

std::vector<value::token> Xformable::xformOpOrder() const {
  std::vector<value::token> toks;

  for (size_t i = 0; i < xformOps.size(); i++) {
    std::string ss;

    auto xformOp = xformOps[i];

    if (xformOp.inverted) {
      ss += "!invert!";
    }
    ss += to_string(xformOp.op_type);
    if (!xformOp.suffix.empty()) {
      ss += ":" + xformOp.suffix;
    }

    toks.push_back(value::token(ss));
  }

  return toks;
}

value::float3 transform(const value::matrix4d &m, const value::float3 &p) {
  value::float3 tx{float(m.m[3][0]), float(m.m[3][1]), float(m.m[3][2])};
  // MatTy, VecTy, VecBaseTy, vecN
  return value::MultV<value::matrix4d, value::float3, double, float, 3>(m, p) + tx;
}

value::vector3f transform(const value::matrix4d &m, const value::vector3f &p) {
  value::vector3f tx{float(m.m[3][0]), float(m.m[3][1]), float(m.m[3][2])};
  return value::MultV<value::matrix4d, value::vector3f, double, float, 3>(m, p) + tx;
}

value::normal3f transform(const value::matrix4d &m, const value::normal3f &p) {
  value::normal3f tx{float(m.m[3][0]), float(m.m[3][1]), float(m.m[3][2])};
  return value::MultV<value::matrix4d, value::normal3f, double, float, 3>(m, p) + tx;
}
value::point3f transform(const value::matrix4d &m, const value::point3f &p) {
  value::point3f tx{float(m.m[3][0]), float(m.m[3][1]), float(m.m[3][2])};
  return value::MultV<value::matrix4d, value::point3f, double, float, 3>(m, p) + tx;
}
value::double3 transform(const value::matrix4d &m, const value::double3 &p) {
  value::double3 tx{m.m[3][0], m.m[3][1], m.m[3][2]};
  return value::MultV<value::matrix4d, value::double3, double, double, 3>(m, p) + tx;
}
value::vector3d transform(const value::matrix4d &m, const value::vector3d &p) {
  value::vector3d tx{m.m[3][0], m.m[3][1], m.m[3][2]};
  value::vector3d v =
      value::MultV<value::matrix4d, value::vector3d, double, double, 3>(m, p);
  v.x += tx.x;
  v.y += tx.y;
  v.z += tx.z;
  return v;
}
value::normal3d transform(const value::matrix4d &m, const value::normal3d &p) {
  value::normal3d tx{m.m[3][0], m.m[3][1], m.m[3][2]};
  value::normal3d v =
      value::MultV<value::matrix4d, value::normal3d, double, double, 3>(m, p);
  v.x += tx.x;
  v.y += tx.y;
  v.z += tx.z;
  return v;
}
value::point3d transform(const value::matrix4d &m, const value::point3d &p) {
  value::point3d tx{m.m[3][0], m.m[3][1], m.m[3][2]};
  value::point3d v =
      value::MultV<value::matrix4d, value::point3d, double, double, 3>(m, p);
  v.x += tx.x;
  v.y += tx.y;
  v.z += tx.z;
  return v;
}

value::float3 transform_dir(const value::matrix4d &m, const value::float3 &p) {
  // MatTy, VecTy, VecBaseTy, vecN
  return value::MultV<value::matrix4d, value::float3, double, float, 3>(m, p);
}

value::vector3f transform_dir(const value::matrix4d &m,
                              const value::vector3f &p) {
  return value::MultV<value::matrix4d, value::vector3f, double, float, 3>(m, p);
}

value::normal3f transform_dir(const value::matrix4d &m,
                              const value::normal3f &p) {
  return value::MultV<value::matrix4d, value::normal3f, double, float, 3>(m, p);
}
value::point3f transform_dir(const value::matrix4d &m,
                             const value::point3f &p) {
  return value::MultV<value::matrix4d, value::point3f, double, float, 3>(m, p);
}
value::double3 transform_dir(const value::matrix4d &m,
                             const value::double3 &p) {
  return value::MultV<value::matrix4d, value::double3, double, double, 3>(m, p);
}
value::vector3d transform_dir(const value::matrix4d &m,
                              const value::vector3d &p) {
  return value::MultV<value::matrix4d, value::vector3d, double, double, 3>(m, p);
}
value::normal3d transform_dir(const value::matrix4d &m,
                              const value::normal3d &p) {
  return value::MultV<value::matrix4d, value::normal3d, double, double, 3>(m, p);
}
value::point3d transform_dir(const value::matrix4d &m,
                             const value::point3d &p) {
  return value::MultV<value::matrix4d, value::point3d, double, double, 3>(m, p);
}

value::matrix4d upper_left_3x3_only(const value::matrix4d &m) {
  value::matrix4d dst;

  memcpy(dst.m, m.m, sizeof(double) * 4 * 4);

  dst.m[0][3] = 0.0;
  dst.m[0][3] = 0.0;
  dst.m[0][3] = 0.0;

  dst.m[3][0] = 0.0;
  dst.m[3][1] = 0.0;
  dst.m[3][2] = 0.0;

  dst.m[3][3] = 1.0;

  return dst;
}

// -------------------------------------------------------------------------------
// From pxrUSD
//

//
// Copyright 2016 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
////////////////////////////////////////////////////////////////////////

value::matrix3d inverse_pxr(const value::matrix3d &m, double *detp,
                            double eps) {
  double a00, a01, a02, a10, a11, a12, a20, a21, a22;
  double det, rcp;

  a00 = m.m[0][0];
  a01 = m.m[0][1];
  a02 = m.m[0][2];
  a10 = m.m[1][0];
  a11 = m.m[1][1];
  a12 = m.m[1][2];
  a20 = m.m[2][0];
  a21 = m.m[2][1];
  a22 = m.m[2][2];
  det = -(a02 * a11 * a20) + a01 * a12 * a20 + a02 * a10 * a21 -
        a00 * a12 * a21 - a01 * a10 * a22 + a00 * a11 * a22;

  if (detp) {
    *detp = det;
  }

  value::matrix3d inv;

  if (std::fabs(det) > eps) {
    rcp = 1.0 / det;
    inv.m[0][0] = (-(a12 * a21) + a11 * a22) * rcp;
    inv.m[0][1] = (a02 * a21 - a01 * a22) * rcp;
    inv.m[0][2] = (-(a02 * a11) + a01 * a12) * rcp;
    inv.m[1][0] = (a12 * a20 - a10 * a22) * rcp;
    inv.m[1][1] = (-(a02 * a20) + a00 * a22) * rcp;
    inv.m[1][2] = (a02 * a10 - a00 * a12) * rcp;
    inv.m[2][0] = (-(a11 * a20) + a10 * a21) * rcp;
    inv.m[2][1] = (a01 * a20 - a00 * a21) * rcp;
    inv.m[2][2] = (-(a01 * a10) + a00 * a11) * rcp;

  } else {
    inv = value::matrix3d::identity();
    // scale = FLT_MAX
    inv.m[0][0] = double((std::numeric_limits<float>::max)());
    inv.m[1][1] = double((std::numeric_limits<float>::max)());
    inv.m[2][2] = double((std::numeric_limits<float>::max)());
  }

  return inv;
}

value::matrix4d inverse_pxr(const value::matrix4d &m, double *detp,
                            double eps) {
  double x00, x01, x02, x03;
  double x10, x11, x12, x13;
  double x20, x21, x22, x23;
  double x30, x31, x32, x33;
  double y01, y02, y03, y12, y13, y23;
  double z00, z10, z20, z30;
  double z01, z11, z21, z31;
  double z02, z03, z12, z13, z22, z23, z32, z33;

  // Pickle 1st two columns of matrix into registers
  x00 = m.m[0][0];
  x01 = m.m[0][1];
  x10 = m.m[1][0];
  x11 = m.m[1][1];
  x20 = m.m[2][0];
  x21 = m.m[2][1];
  x30 = m.m[3][0];
  x31 = m.m[3][1];

  // Compute all six 2x2 determinants of 1st two columns
  y01 = x00 * x11 - x10 * x01;
  y02 = x00 * x21 - x20 * x01;
  y03 = x00 * x31 - x30 * x01;
  y12 = x10 * x21 - x20 * x11;
  y13 = x10 * x31 - x30 * x11;
  y23 = x20 * x31 - x30 * x21;

  // Pickle 2nd two columns of matrix into registers
  x02 = m.m[0][2];
  x03 = m.m[0][3];
  x12 = m.m[1][2];
  x13 = m.m[1][3];
  x22 = m.m[2][2];
  x23 = m.m[2][3];
  x32 = m.m[3][2];
  x33 = m.m[3][3];

  // Compute all 3x3 cofactors for 2nd two columns */
  z33 = x02 * y12 - x12 * y02 + x22 * y01;
  z23 = x12 * y03 - x32 * y01 - x02 * y13;
  z13 = x02 * y23 - x22 * y03 + x32 * y02;
  z03 = x22 * y13 - x32 * y12 - x12 * y23;
  z32 = x13 * y02 - x23 * y01 - x03 * y12;
  z22 = x03 * y13 - x13 * y03 + x33 * y01;
  z12 = x23 * y03 - x33 * y02 - x03 * y23;
  z02 = x13 * y23 - x23 * y13 + x33 * y12;

  // Compute all six 2x2 determinants of 2nd two columns
  y01 = x02 * x13 - x12 * x03;
  y02 = x02 * x23 - x22 * x03;
  y03 = x02 * x33 - x32 * x03;
  y12 = x12 * x23 - x22 * x13;
  y13 = x12 * x33 - x32 * x13;
  y23 = x22 * x33 - x32 * x23;

  // Compute all 3x3 cofactors for 1st two columns
  z30 = x11 * y02 - x21 * y01 - x01 * y12;
  z20 = x01 * y13 - x11 * y03 + x31 * y01;
  z10 = x21 * y03 - x31 * y02 - x01 * y23;
  z00 = x11 * y23 - x21 * y13 + x31 * y12;
  z31 = x00 * y12 - x10 * y02 + x20 * y01;
  z21 = x10 * y03 - x30 * y01 - x00 * y13;
  z11 = x00 * y23 - x20 * y03 + x30 * y02;
  z01 = x20 * y13 - x30 * y12 - x10 * y23;

  // compute 4x4 determinant & its reciprocal
  double det = x30 * z30 + x20 * z20 + x10 * z10 + x00 * z00;
  if (detp) {
    *detp = det;
  }

  value::matrix4d inv;

  if (std::fabs(det) > eps) {
    double rcp = 1.0 / det;
    // Multiply all 3x3 cofactors by reciprocal & transpose
    inv.m[0][0] = z00 * rcp;
    inv.m[0][1] = z10 * rcp;
    inv.m[1][0] = z01 * rcp;
    inv.m[0][2] = z20 * rcp;
    inv.m[2][0] = z02 * rcp;
    inv.m[0][3] = z30 * rcp;
    inv.m[3][0] = z03 * rcp;
    inv.m[1][1] = z11 * rcp;
    inv.m[1][2] = z21 * rcp;
    inv.m[2][1] = z12 * rcp;
    inv.m[1][3] = z31 * rcp;
    inv.m[3][1] = z13 * rcp;
    inv.m[2][2] = z22 * rcp;
    inv.m[2][3] = z32 * rcp;
    inv.m[3][2] = z23 * rcp;
    inv.m[3][3] = z33 * rcp;

  } else {
    inv = value::matrix4d::identity();
    // scale = FLT_MAX
    inv.m[0][0] = double((std::numeric_limits<float>::max)());
    inv.m[1][1] = double((std::numeric_limits<float>::max)());
    inv.m[2][2] = double((std::numeric_limits<float>::max)());
    // [3][3] = 1.0
  }

  return inv;
}

/*
 * Given 3 basis vectors tx, ty, tz, orthogonalize and optionally normalize
 * them.
 *
 * This uses an iterative method that is very stable even when the vectors
 * are far from orthogonal (close to colinear).  The number of iterations
 * and thus the computation time does increase as the vectors become
 * close to colinear, however.
 *
 * If the iteration fails to converge, returns false with vectors as close to
 * orthogonal as possible.
 */
bool orthonormalize_basis(value::double3 &tx, value::double3 &ty,
                          value::double3 &tz, const bool normalize,
                          const double eps) {
  value::double3 ax, bx, cx, ay, by, cy, az, bz, cz;

  if (normalize) {
    tx = vnormalize(tx);
    ty = vnormalize(ty);
    tz = vnormalize(tz);
    ax = tx;
    ay = ty;
    az = tz;
  } else {
    ax = vnormalize(tx);
    ay = vnormalize(ty);
    az = vnormalize(tz);
  }

  /* Check for colinear vectors. This is not only a quick-out: the
   * error computation below will evaluate to zero if there's no change
   * after an iteration, which can happen either because we have a good
   * solution or because the vectors are colinear.   So we have to check
   * the colinear case beforehand, or we'll get fooled in the error
   * computation.
   */
  if (math::is_close(ax, ay, eps) || math::is_close(ax, az, eps) ||
      math::is_close(ay, az, eps)) {
    return false;
  }

  constexpr int kMAX_ITERS = 20;
  int iter;
  for (iter = 0; iter < kMAX_ITERS; ++iter) {
    bx = tx;
    by = ty;
    bz = tz;

    bx = bx - vdot(ay, bx) * ay;
    bx = bx - vdot(az, bx) * az;

    by = by - vdot(ax, by) * ax;
    by = by - vdot(az, by) * az;

    bz = bz - vdot(ax, bz) * ax;
    bz = bz - vdot(ay, bz) * ay;

    cx = 0.5 * (tx + bx);
    cy = 0.5 * (ty + by);
    cz = 0.5 * (tz + bz);

    if (normalize) {
      cx = vnormalize(cx);
      cy = vnormalize(cy);
      cz = vnormalize(cz);
    }

    value::double3 xDiff = tx - cx;
    value::double3 yDiff = ty - cy;
    value::double3 zDiff = tz - cz;

    double error = vdot(xDiff, xDiff) + vdot(yDiff, yDiff) + vdot(zDiff, zDiff);

    // error is squared, so compare to squared tolerance
    if (error < (eps * eps)) {
      break;
    }

    tx = cx;
    ty = cy;
    tz = cz;

    ax = tx;
    ay = ty;
    az = tz;

    if (!normalize) {
      ax = vnormalize(ax);
      ax = vnormalize(ay);
      ax = vnormalize(az);
    }
  }

  return iter < kMAX_ITERS;
}

/*
 * Return the matrix orthonormal using an iterative method.
 * It is potentially slower if the matrix is far from orthonormal (i.e. if
 * the row basis vectors are close to colinear) but in the common case
 * of near-orthonormality it should be just as fast.
 *
 * For 4f, The translation part is left intact.  If the translation is
 * represented as a homogenous coordinate (i.e. a non-unity lower right corner),
 * it is divided out.
 */
value::matrix3d orthonormalize(const value::matrix3d &m, bool *result_valid) {
  value::matrix3d ret = value::matrix3d::identity();

  // orthogonalize and normalize row vectors
  value::double3 r0{m.m[0][0], m.m[0][1], m.m[0][2]};
  value::double3 r1{m.m[1][0], m.m[1][1], m.m[1][2]};
  value::double3 r2{m.m[2][0], m.m[2][1], m.m[2][2]};
  bool result = orthonormalize_basis(r0, r1, r2, true);
  ret.m[0][0] = r0[0];
  ret.m[0][1] = r0[1];
  ret.m[0][2] = r0[2];
  ret.m[1][0] = r1[0];
  ret.m[1][1] = r1[1];
  ret.m[1][2] = r1[2];
  ret.m[2][0] = r2[0];
  ret.m[2][1] = r2[1];
  ret.m[2][2] = r2[2];

  if (result_valid) {
    (*result_valid) = result;
    // TF_WARN("OrthogonalizeBasis did not converge, matrix may not be "
    //               "orthonormal.");
  }

  return ret;
}

value::matrix4d orthonormalize(const value::matrix4d &m, bool *result_valid) {
  value::matrix4d ret = value::matrix4d::identity();

  // orthogonalize and normalize row vectors
  value::double3 r0{m.m[0][0], m.m[0][1], m.m[0][2]};
  value::double3 r1{m.m[1][0], m.m[1][1], m.m[1][2]};
  value::double3 r2{m.m[2][0], m.m[2][1], m.m[2][2]};
  bool result = orthonormalize_basis(r0, r1, r2, true);
  ret.m[0][0] = r0[0];
  ret.m[0][1] = r0[1];
  ret.m[0][2] = r0[2];
  ret.m[1][0] = r1[0];
  ret.m[1][1] = r1[1];
  ret.m[1][2] = r1[2];
  ret.m[2][0] = r2[0];
  ret.m[2][1] = r2[1];
  ret.m[2][2] = r2[2];

  // divide out any homogeneous coordinate - unless it's zero
  const double min_vector_length = 1e-10;  //
  if (!math::is_close(ret.m[3][3], 1.0,
                      std::numeric_limits<double>::epsilon()) &&
      !math::is_close(ret.m[3][3], 0.0, min_vector_length)) {
    ret.m[3][0] /= ret.m[3][3];
    ret.m[3][1] /= ret.m[3][3];
    ret.m[3][2] /= ret.m[3][3];
    ret.m[3][3] = 1.0;
  }

  if (result_valid) {
    (*result_valid) = result;
    // TF_WARN("OrthogonalizeBasis did not converge, matrix may not be "
    //               "orthonormal.");
  }

  return ret;
}

// End pxrUSD
// -------------------------------------------------------------------------------

value::matrix4d trs_angle_xyz(const value::double3 &translation,
                              const value::double3 &rotation_angles_xyz,
                              const value::double3 &scale) {
  value::matrix4d m{value::matrix4d::identity()};

  XformEvaluator eval;
#if defined(PXR_COMPATIBLE_ROTATE_MATRIX_GENERATION)
  eval.Rotation({1.0, 0.0, 0.0}, rotation_angles_xyz[0]);
  eval.Rotation({0.0, 1.0, 0.0}, rotation_angles_xyz[1]);
  eval.Rotation({0.0, 0.0, 1.0}, rotation_angles_xyz[2]);
#else
  eval.RotateX(rotation_angles_xyz[0]);
  eval.RotateY(rotation_angles_xyz[1]);
  eval.RotateZ(rotation_angles_xyz[2]);
#endif

  auto ret = eval.result();
  if (!ret) {
    // This should not happend though.
    return m;
  }
  value::matrix4d rMat = ret.value();

  value::matrix4d tMat{value::matrix4d::identity()};
  tMat.m[3][0] = translation[0];
  tMat.m[3][1] = translation[1];
  tMat.m[3][2] = translation[2];

  value::matrix4d sMat{value::matrix4d::identity()};
  sMat.m[0][0] = scale[0];
  sMat.m[1][1] = scale[1];
  sMat.m[2][2] = scale[2];

  m = sMat * rMat * tMat;

  return m;
}

//
// Build matrix from T R S.
//
// Rotation is given by 3 vectors axis(orthonormalized inside trs()).
//
value::matrix4d trs_rot_axis(const value::double3 &translation,
                             const value::double3 &rotation_x_axis,
                             const value::double3 &rotation_y_axis,
                             const value::double3 &rotation_z_axis,
                             const value::double3 &scale) {
  value::matrix4d m{value::matrix4d::identity()};

  value::matrix4d rMat{value::matrix4d::identity()};
  rMat.m[0][0] = rotation_x_axis[0];
  rMat.m[0][1] = rotation_x_axis[1];
  rMat.m[0][2] = rotation_x_axis[2];
  rMat.m[1][0] = rotation_y_axis[0];
  rMat.m[1][1] = rotation_y_axis[1];
  rMat.m[1][2] = rotation_y_axis[2];
  rMat.m[2][0] = rotation_z_axis[0];
  rMat.m[2][1] = rotation_z_axis[1];
  rMat.m[2][2] = rotation_z_axis[2];

  bool result_valid{true};
  value::matrix4d orMat = orthonormalize(rMat, &result_valid);
  // TODO: report error when orthonormalize failed.

  value::matrix4d tMat{value::matrix4d::identity()};
  tMat.m[3][0] = translation[0];
  tMat.m[3][1] = translation[1];
  tMat.m[3][2] = translation[2];

  value::matrix4d sMat{value::matrix4d::identity()};
  sMat.m[0][0] = scale[0];
  sMat.m[1][1] = scale[1];
  sMat.m[2][2] = scale[2];

  m = sMat * orMat * tMat;

  return m;
}

}  // namespace tinyusdz
