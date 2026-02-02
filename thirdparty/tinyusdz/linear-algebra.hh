// SPDX-License-Identifier: Apache 2.0
// Copyright 2022-Present Light Transport Entertainment, Inc.
#pragma once

#include "value-types.hh"

namespace tinyusdz {

constexpr float kFloatNormalizeEps = std::numeric_limits<float>::epsilon();
constexpr double kDoubleNormalizeEps = std::numeric_limits<double>::epsilon();

// GF_MIN_VECTOR_LENGTH in pxrUSD
constexpr double kPXRNormalizeEps = 1.0e-10;

value::quath slerp(const value::quath &a, const value::quath &b, const float t);
value::quatf slerp(const value::quatf &a, const value::quatf &b, const float t);
value::quatd slerp(const value::quatd &a, const value::quatd &b, const double t);

float vlength(const value::float3 &a);
float vlength(const value::normal3f &a);
float vlength(const value::vector3f &a);
float vlength(const value::point3f &a);
double vlength(const value::double3 &a);
double vlength(const value::normal3d &a);
double vlength(const value::vector3d &a);
double vlength(const value::point3d &a);

value::float3 vnormalize(const value::float3 &a, const float eps = kFloatNormalizeEps);
value::double3 vnormalize(const value::double3 &a, const double eps = kDoubleNormalizeEps);
value::normal3f vnormalize(const value::normal3f &a, const float eps = kFloatNormalizeEps);
value::normal3d vnormalize(const value::normal3d &a, const double eps = kDoubleNormalizeEps);
value::vector3f vnormalize(const value::vector3f &a, const float eps = kFloatNormalizeEps);
value::vector3d vnormalize(const value::vector3d &a, const double eps = kDoubleNormalizeEps);
value::point3f vnormalize(const value::point3f &a, const float eps = kFloatNormalizeEps);
value::point3d vnormalize(const value::point3d &a, const double eps = kDoubleNormalizeEps);

// Assume CCW(Counter ClockWise)
value::float3 vcross(const value::float3 &a, const value::float3 &b);
value::double3 vcross(const value::double3 &a, const value::double3 &b);
value::normal3f vcross(const value::normal3f &a, const value::normal3f &b);
value::normal3d vcross(const value::normal3d &a, const value::normal3d &b);
value::vector3f vcross(const value::vector3f &a, const value::vector3f &b);
value::vector3d vcross(const value::vector3d &a, const value::vector3d &b);
value::point3f vcross(const value::point3f &a, const value::point3f &b);
value::point3d vcross(const value::point3d &a, const value::point3d &b);

value::float3 geometric_normal(const value::float3 &p0, const value::float3 &p1, const value::float3 &p2);
value::double3 geometric_normal(const value::double3 &p0, const value::double3 &p1, const value::double3 &p2);
value::point3f geometric_normal(const value::point3f &p0, const value::point3f &p1, const value::point3f &p2);
value::point3d geometric_normal(const value::point3d &p0, const value::point3d &p1, const value::point3d &p2);



inline float vdot(const value::float3 &a, const value::float3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double vdot(const value::double3 &a, const value::double3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float vdot(const value::vector3f &a, const value::vector3f &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double vdot(const value::vector3d &a, const value::vector3d &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float vdot(const value::normal3f &a, const value::normal3f &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double vdot(const value::normal3d &a, const value::normal3d &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

} // namespace tinyusdz
