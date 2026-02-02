// SPDX-License-Identifier: Apache 2.0
// Copyright 2022-Present Light Transport Entertainment, Inc.
#include "linear-algebra.hh"
#include "value-eval-util.hh"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "external/linalg.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace tinyusdz {

value::quath slerp(const value::quath &_a, const value::quath &_b, const float t) {
  value::quatf a;
  value::quatf b;

  a[0] = value::half_to_float(_a[0]);
  a[1] = value::half_to_float(_a[1]);
  a[2] = value::half_to_float(_a[2]);
  a[3] = value::half_to_float(_a[3]);

  b[0] = value::half_to_float(_b[0]);
  b[1] = value::half_to_float(_b[1]);
  b[2] = value::half_to_float(_b[2]);
  b[3] = value::half_to_float(_b[3]);

  value::quatf _c = slerp(a, b, t);

  value::quath c;
  c[0] = value::float_to_half_full(_c[0]);
  c[1] = value::float_to_half_full(_c[1]);
  c[2] = value::float_to_half_full(_c[2]);
  c[3] = value::float_to_half_full(_c[3]);

  return c;
}

value::quatf slerp(const value::quatf &a, const value::quatf &b, const float t) {
  linalg::vec<float, 4> qa;    
  linalg::vec<float, 4> qb;    
  linalg::vec<float, 4> qret;    

  memcpy(reinterpret_cast<value::quatf *>(&qa), &a, sizeof(float) * 4);
  memcpy(reinterpret_cast<value::quatf *>(&qb), &b, sizeof(float) * 4);

  qret = linalg::slerp(qa, qb, t);
  
  value::quatf ret;
  memcpy(&ret, reinterpret_cast<value::quatf *>(&qret), sizeof(float) * 4);
  return ret;
}

value::quatd slerp(const value::quatd &a, const value::quatd &b, const double t) {

  linalg::vec<double, 4> qa;    
  linalg::vec<double, 4> qb;    
  linalg::vec<double, 4> qret;    

  memcpy(reinterpret_cast<value::quatd *>(&qa), &a, sizeof(double) * 4);
  memcpy(reinterpret_cast<value::quatd *>(&qb), &b, sizeof(double) * 4);

  qret = linalg::slerp(qa, qb, t);

  value::quatd ret;
  memcpy(&ret, reinterpret_cast<value::quatd *>(&qret), sizeof(double) * 4);
  return ret;
  
}

float vlength(const value::float3 &a) {
  float d2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  if (d2 > std::numeric_limits<float>::epsilon()) {
    return std::sqrt(d2);
  }
  return 0.0f;
}

float vlength(const value::normal3f &a) {
  return vlength(value::float3{a.x, a.y, a.z});
}

float vlength(const value::vector3f &a) {
  return vlength(value::float3{a.x, a.y, a.z});
}

float vlength(const value::point3f &a) {
  return vlength(value::float3{a.x, a.y, a.z});
}

value::float3 vnormalize(const value::float3 &a, const float eps) {
  float len = vlength(a);
  len = (len > eps) ? len : eps;
  return value::float3({a[0] / len, a[1] / len, a[2] / len});
}

value::normal3f vnormalize(const value::normal3f &a, const float eps) {
  float len = vlength(a);
  len = (len > eps) ? len : eps;
  return value::normal3f({a[0] / len, a[1] / len, a[2] / len});
}

value::vector3f vnormalize(const value::vector3f &a, const float eps) {
  float len = vlength(a);
  len = (len > eps) ? len : eps;
  return value::vector3f({a[0] / len, a[1] / len, a[2] / len});
}

value::point3f vnormalize(const value::point3f &a, const float eps) {
  float len = vlength(a);
  len = (len > eps) ? len : eps;
  return value::point3f({a[0] / len, a[1] / len, a[2] / len});
}

double vlength(const value::double3 &a) {
  double d2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  if (d2 > std::numeric_limits<double>::epsilon()) {
    return std::sqrt(d2);
  }
  return 0.0;
}

double vlength(const value::normal3d &a) {
  return vlength(value::double3{a.x, a.y, a.z});
}

double vlength(const value::vector3d &a) {
  return vlength(value::double3{a.x, a.y, a.z});
}

double vlength(const value::point3d &a) {
  return vlength(value::double3{a.x, a.y, a.z});
}

value::double3 vnormalize(const value::double3 &a, const double eps) {
  double len = vlength(a);
  len = (len > eps) ? len : eps;
  return value::double3({a[0] / len, a[1] / len, a[2] / len});
}

value::normal3d vnormalize(const value::normal3d &a, const double eps) {
  double len = vlength(a);
  len = (len > eps) ? len : eps;
  return value::normal3d({a[0] / len, a[1] / len, a[2] / len});
}

value::vector3d vnormalize(const value::vector3d &a, const double eps) {
  double len = vlength(a);
  len = (len > eps) ? len : eps;
  return value::vector3d({a[0] / len, a[1] / len, a[2] / len});
}

value::point3d vnormalize(const value::point3d &a, const double eps) {
  double len = vlength(a);
  len = (len > eps) ? len : eps;
  return value::point3d({a[0] / len, a[1] / len, a[2] / len});
}

value::float3 vcross(const value::float3 &a, const value::float3 &b)
{
  value::float3 n;
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return n;
}

value::double3 vcross(const value::double3 &a, const value::double3 &b)
{
  value::double3 n;
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return n;
}

value::normal3f vcross(const value::normal3f &a, const value::normal3f &b)
{
  value::normal3f n;
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return n;
}

value::normal3d vcross(const value::normal3d &a, const value::normal3d &b)
{
  value::normal3d n;
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return n;
}

value::vector3f vcross(const value::vector3f &a, const value::vector3f &b)
{
  value::vector3f n;
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return n;
}

value::vector3d vcross(const value::vector3d &a, const value::vector3d &b)
{
  value::vector3d n;
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return n;
}

value::point3f vcross(const value::point3f &a, const value::point3f &b)
{
  value::point3f n;
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return n;
}

value::point3d vcross(const value::point3d &a, const value::point3d &b)
{
  value::point3d n;
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];

  return n;
}

// CCW

value::float3 geometric_normal(const value::float3 &p0, const value::float3 &p1, const value::float3 &p2)
{
  value::float3 n = vcross(p1 - p0, p2 - p0);

  return vnormalize(n);
}

value::double3 geometric_normal(const value::double3 &p0, const value::double3 &p1, const value::double3 &p2)
{
  value::double3 n = vcross(p1 - p0, p2 - p0);

  return vnormalize(n);
}

value::point3f geometric_normal(const value::point3f &p0, const value::point3f &p1, const value::point3f &p2)
{
  value::point3f n = vcross(p1 - p0, p2 - p0);

  return vnormalize(n);
}

value::point3d geometric_normal(const value::point3d &p0, const value::point3d &p1, const value::point3d &p2)
{
  value::point3d n = vcross(p1 - p0, p2 - p0);

  return vnormalize(n);
}

} // namespace tinyusdz
