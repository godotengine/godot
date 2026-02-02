#pragma once

#include "value-types.hh"
#include "linear-algebra.hh"

namespace tinyusdz {

#define FOUR_ARITH_OP_2(__ty, __basety) \
inline __ty operator+(const __ty &a, const __ty &b) { \
  return {a[0] + b[0], a[1] + b[1]}; \
} \
\
inline __ty operator+(const __basety a, const __ty &b) { \
  return {a + b[0], a + b[1]}; \
}\
inline __ty operator+(const __ty &a, const __basety b) { \
  return {a[0] + b, a[1] + b}; \
} \
\
inline __ty operator-(const __ty &a, const __ty &b) { \
  return {a[0] - b[0], a[1] - b[1]}; \
} \
\
inline __ty operator-(const __basety a, const __ty &b) { \
  return {a - b[0], a - b[1]}; \
}\
inline __ty operator-(const __ty &a, const __basety b) { \
  return {a[0] - b, a[1] - b}; \
}\
\
inline __ty operator*(const __ty &a, const __ty &b) { \
  return {a[0] * b[0], a[1] * b[1]}; \
} \
\
inline __ty operator*(const __basety a, const __ty &b) { \
  return {a * b[0], a * b[1]}; \
}\
inline __ty operator*(const __ty &a, const __basety b) { \
  return {a[0] * b, a[1] * b}; \
} \
\
inline __ty operator/(const __ty &a, const __ty &b) { \
  return {a[0] / b[0], a[1] / b[1]}; \
} \
\
inline __ty operator/(const __basety a, const __ty &b) { \
  return {a / b[0], a / b[1]}; \
}\
inline __ty operator/(const __ty &a, const __basety b) { \
  return {a[0] / b, a[1] / b}; \
}

#define FOUR_ARITH_OP_3(__ty, __basety) \
inline __ty operator+(const __ty &a, const __ty &b) { \
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; \
} \
\
inline __ty operator+(const __basety a, const __ty &b) { \
  return {a + b[0], a + b[1], a + b[2]}; \
}\
inline __ty operator+(const __ty &a, const __basety b) { \
  return {a[0] + b, a[1] + b, a[2] + b}; \
} \
\
inline __ty operator-(const __ty &a, const __ty &b) { \
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; \
} \
\
inline __ty operator-(const __basety a, const __ty &b) { \
  return {a - b[0], a - b[1], a - b[2]}; \
}\
inline __ty operator-(const __ty &a, const __basety b) { \
  return {a[0] - b, a[1] - b, a[2] - b}; \
}\
\
inline __ty operator*(const __ty &a, const __ty &b) { \
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]}; \
} \
\
inline __ty operator*(const __basety a, const __ty &b) { \
  return {a * b[0], a * b[1], a * b[2]}; \
}\
inline __ty operator*(const __ty &a, const __basety b) { \
  return {a[0] * b, a[1] * b, a[2] * b}; \
} \
\
inline __ty operator/(const __ty &a, const __ty &b) { \
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2]}; \
} \
\
inline __ty operator/(const __basety a, const __ty &b) { \
  return {a / b[0], a / b[1], a / b[2]}; \
}\
inline __ty operator/(const __ty &a, const __basety b) { \
  return {a[0] / b, a[1] / b, a[2] / b}; \
}

#define FOUR_ARITH_OP_4(__ty, __basety) \
inline __ty operator+(const __ty &a, const __ty &b) { \
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}; \
} \
\
inline __ty operator+(const __basety a, const __ty &b) { \
  return {a + b[0], a + b[1], a + b[2], a + b[3]}; \
}\
inline __ty operator+(const __ty &a, const __basety b) { \
  return {a[0] + b, a[1] + b, a[2] + b, a[3] + b}; \
} \
\
inline __ty operator-(const __ty &a, const __ty &b) { \
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]}; \
} \
\
inline __ty operator-(const __basety a, const __ty &b) { \
  return {a - b[0], a - b[1], a - b[2], a - b[3]}; \
}\
inline __ty operator-(const __ty &a, const __basety b) { \
  return {a[0] - b, a[1] - b, a[2] - b, a[3] - b}; \
}\
\
inline __ty operator*(const __ty &a, const __ty &b) { \
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]}; \
} \
\
inline __ty operator*(const __basety a, const __ty &b) { \
  return {a * b[0], a * b[1], a * b[2], a * b[3]}; \
}\
inline __ty operator*(const __ty &a, const __basety b) { \
  return {a[0] * b, a[1] * b, a[2] * b, a[3] * b}; \
} \
\
inline __ty operator/(const __ty &a, const __ty &b) { \
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]}; \
} \
\
inline __ty operator/(const __basety a, const __ty &b) { \
  return {a / b[0], a / b[1], a / b[2], a / b[3]}; \
}\
inline __ty operator/(const __ty &a, const __basety b) { \
  return {a[0] / b, a[1] / b, a[2] / b, a[3] / b}; \
}

#define ARITH_ASSIGN_OP_2(__ty, __basety) \
inline __ty &operator+=(__ty &a, const __ty &b) { \
  a[0] += b[0]; a[1] += b[1]; \
  return a; \
} \
inline __ty &operator-=(__ty &a, const __ty &b) { \
  a[0] -= b[0]; a[1] -= b[1]; \
  return a; \
} \
inline __ty &operator*=(__ty &a, const __ty &b) { \
  a[0] *= b[0]; a[1] *= b[1]; \
  return a; \
} \
inline __ty &operator/=(__ty &a, const __ty &b) { \
  a[0] /= b[0]; a[1] /= b[1]; \
  return a; \
} \

#define ARITH_ASSIGN_OP_3(__ty, __basety) \
inline __ty &operator+=(__ty &a, const __ty &b) { \
  a[0] += b[0]; a[1] += b[1]; a[2] += b[2]; \
  return a; \
} \
inline __ty &operator-=(__ty &a, const __ty &b) { \
  a[0] -= b[0]; a[1] -= b[1]; a[2] -= b[2]; \
  return a; \
} \
inline __ty &operator*=(__ty &a, const __ty &b) { \
  a[0] *= b[0]; a[1] *= b[1]; a[2] *= b[2]; \
  return a; \
} \
inline __ty &operator/=(__ty &a, const __ty &b) { \
  a[0] /= b[0]; a[1] /= b[1]; a[2] /= b[2]; \
  return a; \
} \

#define ARITH_ASSIGN_OP_4(__ty, __basety) \
inline __ty &operator+=(__ty &a, const __ty &b) { \
  a[0] += b[0]; a[1] += b[1]; a[2] += b[2]; a[3] += b[3]; \
  return a; \
} \
inline __ty &operator-=(__ty &a, const __ty &b) { \
  a[0] -= b[0]; a[1] -= b[1]; a[2] -= b[2]; a[3] -= b[3]; \
  return a; \
} \
inline __ty &operator*=(__ty &a, const __ty &b) { \
  a[0] *= b[0]; a[1] *= b[1]; a[2] *= b[2]; a[3] *= b[3]; \
  return a; \
} \
inline __ty &operator/=(__ty &a, const __ty &b) { \
  a[0] /= b[0]; a[1] /= b[1]; a[2] /= b[2]; a[3] /= b[3]; \
  return a; \
} \

// TODO: half op scalar_half
FOUR_ARITH_OP_2(value::half2, float)
FOUR_ARITH_OP_3(value::half3, float)
FOUR_ARITH_OP_4(value::half4, float)
ARITH_ASSIGN_OP_2(value::half2, float)
ARITH_ASSIGN_OP_3(value::half3, float)
ARITH_ASSIGN_OP_4(value::half4, float)

FOUR_ARITH_OP_2(value::int2, int)
FOUR_ARITH_OP_3(value::int3, int)
FOUR_ARITH_OP_4(value::int4, int)
ARITH_ASSIGN_OP_2(value::int2, int)
ARITH_ASSIGN_OP_3(value::int3, int)
ARITH_ASSIGN_OP_4(value::int4, int)

FOUR_ARITH_OP_2(value::uint2, uint32_t)
FOUR_ARITH_OP_3(value::uint3, uint32_t)
FOUR_ARITH_OP_4(value::uint4, uint32_t)
ARITH_ASSIGN_OP_2(value::uint2, uint32_t)
ARITH_ASSIGN_OP_3(value::uint3, uint32_t)
ARITH_ASSIGN_OP_4(value::uint4, uint32_t)

FOUR_ARITH_OP_2(value::float2, float)
FOUR_ARITH_OP_3(value::float3, float)
FOUR_ARITH_OP_4(value::float4, float)
ARITH_ASSIGN_OP_2(value::float2, float)
ARITH_ASSIGN_OP_3(value::float3, float)
ARITH_ASSIGN_OP_4(value::float4, float)

FOUR_ARITH_OP_2(value::double2, double)
FOUR_ARITH_OP_3(value::double3, double)
FOUR_ARITH_OP_4(value::double4, double)
ARITH_ASSIGN_OP_2(value::double2, double)
ARITH_ASSIGN_OP_3(value::double3, double)
ARITH_ASSIGN_OP_4(value::double4, double)

FOUR_ARITH_OP_3(value::normal3h, float)
FOUR_ARITH_OP_3(value::normal3f, float)
FOUR_ARITH_OP_3(value::normal3d, double)
ARITH_ASSIGN_OP_3(value::normal3h, float)
ARITH_ASSIGN_OP_3(value::normal3f, float)
ARITH_ASSIGN_OP_3(value::normal3d, double)

FOUR_ARITH_OP_3(value::vector3h, float)
FOUR_ARITH_OP_3(value::vector3f, float)
FOUR_ARITH_OP_3(value::vector3d, double)
ARITH_ASSIGN_OP_3(value::vector3h, float)
ARITH_ASSIGN_OP_3(value::vector3f, float)
ARITH_ASSIGN_OP_3(value::vector3d, double)

FOUR_ARITH_OP_3(value::point3h, float)
FOUR_ARITH_OP_3(value::point3f, float)
FOUR_ARITH_OP_3(value::point3d, double)
ARITH_ASSIGN_OP_3(value::point3h, float)
ARITH_ASSIGN_OP_3(value::point3f, float)
ARITH_ASSIGN_OP_3(value::point3d, double)

FOUR_ARITH_OP_3(value::color3h, float)
FOUR_ARITH_OP_3(value::color3f, float)
FOUR_ARITH_OP_3(value::color3d, double)
ARITH_ASSIGN_OP_3(value::color3h, float)
ARITH_ASSIGN_OP_3(value::color3f, float)
ARITH_ASSIGN_OP_3(value::color3d, double)

FOUR_ARITH_OP_4(value::color4h, float)
FOUR_ARITH_OP_4(value::color4f, float)
FOUR_ARITH_OP_4(value::color4d, double)
ARITH_ASSIGN_OP_4(value::color4h, float)
ARITH_ASSIGN_OP_4(value::color4f, float)
ARITH_ASSIGN_OP_4(value::color4d, double)

FOUR_ARITH_OP_2(value::texcoord2h, float)
FOUR_ARITH_OP_2(value::texcoord2f, float)
FOUR_ARITH_OP_2(value::texcoord2d, double)
ARITH_ASSIGN_OP_2(value::texcoord2h, float)
ARITH_ASSIGN_OP_2(value::texcoord2f, float)
ARITH_ASSIGN_OP_2(value::texcoord2d, double)

FOUR_ARITH_OP_3(value::texcoord3h, float)
FOUR_ARITH_OP_3(value::texcoord3f, float)
FOUR_ARITH_OP_3(value::texcoord3d, double)
ARITH_ASSIGN_OP_3(value::texcoord3h, float)
ARITH_ASSIGN_OP_3(value::texcoord3f, float)
ARITH_ASSIGN_OP_3(value::texcoord3d, double)

#undef FOUR_ARITH_OP_2
#undef FOUR_ARITH_OP_3
#undef FOUR_ARITH_OP_4

inline value::matrix2f operator+(const value::matrix2f &a, const double b) {
  value::matrix2f dst;
  dst.m[0][0] = float(double(a.m[0][0]) + b);
  dst.m[0][1] = float(double(a.m[0][1]) + b);
  dst.m[1][0] = float(double(a.m[1][0]) + b);
  dst.m[1][1] = float(double(a.m[1][1]) + b);

  return dst;
}

inline value::matrix2f operator+(const double a, const value::matrix2f &b) {
  value::matrix2f dst;
  dst.m[0][0] = float(a + double(b.m[0][0]));
  dst.m[0][1] = float(a + double(b.m[0][1]));
  dst.m[1][0] = float(a + double(b.m[1][0]));
  dst.m[1][1] = float(a + double(b.m[1][1]));

  return dst;
}


inline value::matrix2f operator-(const value::matrix2f &a, const double b) {
  value::matrix2f dst;
  dst.m[0][0] = float(double(a.m[0][0]) - b);
  dst.m[0][1] = float(double(a.m[0][1]) - b);
  dst.m[1][0] = float(double(a.m[1][0]) - b);
  dst.m[1][1] = float(double(a.m[1][1]) - b);

  return dst;
}

inline value::matrix2f operator-(const double a, const value::matrix2f &b) {
  value::matrix2f dst;
  dst.m[0][0] = float(a - double(b.m[0][0]));
  dst.m[0][1] = float(a - double(b.m[0][1]));
  dst.m[1][0] = float(a - double(b.m[1][0]));
  dst.m[1][1] = float(a - double(b.m[1][1]));

  return dst;
}

inline value::matrix2f operator*(const value::matrix2f &a, const double b) {
  value::matrix2f dst;
  dst.m[0][0] = float(double(a.m[0][0]) * b);
  dst.m[0][1] = float(double(a.m[0][1]) * b);
  dst.m[1][0] = float(double(a.m[1][0]) * b);
  dst.m[1][1] = float(double(a.m[1][1]) * b);

  return dst;
}

inline value::matrix2f operator*(const double a, const value::matrix2f &b) {
  value::matrix2f dst;
  dst.m[0][0] = float(a * double(b.m[0][0]));
  dst.m[0][1] = float(a * double(b.m[0][1]));
  dst.m[1][0] = float(a * double(b.m[1][0]));
  dst.m[1][1] = float(a * double(b.m[1][1]));

  return dst;
}

inline value::matrix2f operator/(const value::matrix2f &a, const double b) {
  value::matrix2f dst;
  dst.m[0][0] = float(double(a.m[0][0]) / b);
  dst.m[0][1] = float(double(a.m[0][1]) / b);
  dst.m[1][0] = float(double(a.m[1][0]) / b);
  dst.m[1][1] = float(double(a.m[1][1]) / b);

  return dst;
}

inline value::matrix2f operator/(const double a, const value::matrix2f &b) {
  value::matrix2f dst;
  dst.m[0][0] = float(a / double(b.m[0][0]));
  dst.m[0][1] = float(a / double(b.m[0][1]));
  dst.m[1][0] = float(a / double(b.m[1][0]));
  dst.m[1][1] = float(a / double(b.m[1][1]));

  return dst;
}

inline value::matrix3f operator+(const value::matrix3f &a, const double b) {
  value::matrix3f dst;
  dst.m[0][0] = float(double(a.m[0][0]) + b);
  dst.m[0][1] = float(double(a.m[0][1]) + b);
  dst.m[0][2] = float(double(a.m[0][2]) + b);
  dst.m[1][0] = float(double(a.m[1][0]) + b);
  dst.m[1][1] = float(double(a.m[1][1]) + b);
  dst.m[1][2] = float(double(a.m[1][2]) + b);
  dst.m[2][0] = float(double(a.m[2][0]) + b);
  dst.m[2][1] = float(double(a.m[2][1]) + b);
  dst.m[2][2] = float(double(a.m[2][2]) + b);

  return dst;
}

inline value::matrix3f operator+(const double a, const value::matrix3f &b) {
  value::matrix3f dst;
  dst.m[0][0] = float(a + double(b.m[0][0]));
  dst.m[0][1] = float(a + double(b.m[0][1]));
  dst.m[0][2] = float(a + double(b.m[0][2]));
  dst.m[1][0] = float(a + double(b.m[1][0]));
  dst.m[1][1] = float(a + double(b.m[1][1]));
  dst.m[1][2] = float(a + double(b.m[1][2]));
  dst.m[2][0] = float(a + double(b.m[2][0]));
  dst.m[2][1] = float(a + double(b.m[2][1]));
  dst.m[2][2] = float(a + double(b.m[2][2]));

  return dst;
}


inline value::matrix3f operator-(const value::matrix3f &a, const double b) {
  value::matrix3f dst;
  dst.m[0][0] = float(double(a.m[0][0] )- b);
  dst.m[0][1] = float(double(a.m[0][1] )- b);
  dst.m[0][2] = float(double(a.m[0][2] )- b);
  dst.m[1][0] = float(double(a.m[1][0] )- b);
  dst.m[1][1] = float(double(a.m[1][1] )- b);
  dst.m[1][2] = float(double(a.m[1][2] )- b);
  dst.m[2][0] = float(double(a.m[2][0] )- b);
  dst.m[2][1] = float(double(a.m[2][1] )- b);
  dst.m[2][2] = float(double(a.m[2][2] )- b);

  return dst;
}

inline value::matrix3f operator-(const double a, const value::matrix3f &b) {
  value::matrix3f dst;
  dst.m[0][0] = float(a - double(b.m[0][0]));
  dst.m[0][1] = float(a - double(b.m[0][1]));
  dst.m[0][2] = float(a - double(b.m[0][2]));
  dst.m[1][0] = float(a - double(b.m[1][0]));
  dst.m[1][1] = float(a - double(b.m[1][1]));
  dst.m[1][2] = float(a - double(b.m[1][2]));
  dst.m[2][0] = float(a - double(b.m[2][0]));
  dst.m[2][1] = float(a - double(b.m[2][1]));
  dst.m[2][2] = float(a - double(b.m[2][2]));

  return dst;
}

inline value::matrix3f operator*(const value::matrix3f &a, const double b) {
  value::matrix3f dst;
  dst.m[0][0] = float(double(a.m[0][0]) * b);
  dst.m[0][1] = float(double(a.m[0][1]) * b);
  dst.m[0][2] = float(double(a.m[0][2]) * b);
  dst.m[1][0] = float(double(a.m[1][0]) * b);
  dst.m[1][1] = float(double(a.m[1][1]) * b);
  dst.m[1][2] = float(double(a.m[1][2]) * b);
  dst.m[2][0] = float(double(a.m[2][0]) * b);
  dst.m[2][1] = float(double(a.m[2][1]) * b);
  dst.m[2][2] = float(double(a.m[2][2]) * b);

  return dst;
}

inline value::matrix3f operator*(const double a, const value::matrix3f &b) {
  value::matrix3f dst;
  dst.m[0][0] = float(a * double(b.m[0][0]));
  dst.m[0][1] = float(a * double(b.m[0][1]));
  dst.m[0][2] = float(a * double(b.m[0][2]));
  dst.m[1][0] = float(a * double(b.m[1][0]));
  dst.m[1][1] = float(a * double(b.m[1][1]));
  dst.m[1][2] = float(a * double(b.m[1][2]));
  dst.m[2][0] = float(a * double(b.m[2][0]));
  dst.m[2][1] = float(a * double(b.m[2][1]));
  dst.m[2][2] = float(a * double(b.m[2][2]));

  return dst;
}

inline value::matrix3f operator/(const value::matrix3f &a, const double b) {
  value::matrix3f dst;
  dst.m[0][0] = float(double(a.m[0][0]) / b);
  dst.m[0][1] = float(double(a.m[0][1]) / b);
  dst.m[0][2] = float(double(a.m[0][2]) / b);
  dst.m[1][0] = float(double(a.m[1][0]) / b);
  dst.m[1][1] = float(double(a.m[1][1]) / b);
  dst.m[1][2] = float(double(a.m[1][2]) / b);
  dst.m[2][0] = float(double(a.m[2][0]) / b);
  dst.m[2][1] = float(double(a.m[2][1]) / b);
  dst.m[2][2] = float(double(a.m[2][2]) / b);

  return dst;
}

inline value::matrix3f operator/(const double a, const value::matrix3f &b) {
  value::matrix3f dst;
  dst.m[0][0] = float(a / double(b.m[0][0]));
  dst.m[0][1] = float(a / double(b.m[0][1]));
  dst.m[0][2] = float(a / double(b.m[0][2]));
  dst.m[1][0] = float(a / double(b.m[1][0]));
  dst.m[1][1] = float(a / double(b.m[1][1]));
  dst.m[1][2] = float(a / double(b.m[1][2]));
  dst.m[2][0] = float(a / double(b.m[2][0]));
  dst.m[2][1] = float(a / double(b.m[2][1]));
  dst.m[2][2] = float(a / double(b.m[2][2]));

  return dst;
}

inline value::matrix4f operator+(const value::matrix4f &a, const double b) {
  value::matrix4f dst;
  dst.m[0][0] = float(double(a.m[0][0]) + b);
  dst.m[0][1] = float(double(a.m[0][1]) + b);
  dst.m[0][2] = float(double(a.m[0][2]) + b);
  dst.m[0][3] = float(double(a.m[0][3]) + b);
  dst.m[1][0] = float(double(a.m[1][0]) + b);
  dst.m[1][1] = float(double(a.m[1][1]) + b);
  dst.m[1][2] = float(double(a.m[1][2]) + b);
  dst.m[1][3] = float(double(a.m[1][3]) + b);
  dst.m[2][0] = float(double(a.m[2][0]) + b);
  dst.m[2][1] = float(double(a.m[2][1]) + b);
  dst.m[2][2] = float(double(a.m[2][2]) + b);
  dst.m[2][3] = float(double(a.m[2][3]) + b);
  dst.m[3][0] = float(double(a.m[3][0]) + b);
  dst.m[3][1] = float(double(a.m[3][1]) + b);
  dst.m[3][2] = float(double(a.m[3][2]) + b);
  dst.m[3][3] = float(double(a.m[3][3]) + b);

  return dst;
}

inline value::matrix4f operator+(const double a, const value::matrix4f &b) {
  value::matrix4f dst;
  dst.m[0][0] = float(a + double(b.m[0][0]));
  dst.m[0][1] = float(a + double(b.m[0][1]));
  dst.m[0][2] = float(a + double(b.m[0][2]));
  dst.m[0][3] = float(a + double(b.m[0][3]));
  dst.m[1][0] = float(a + double(b.m[1][0]));
  dst.m[1][1] = float(a + double(b.m[1][1]));
  dst.m[1][2] = float(a + double(b.m[1][2]));
  dst.m[1][3] = float(a + double(b.m[1][3]));
  dst.m[2][0] = float(a + double(b.m[2][0]));
  dst.m[2][1] = float(a + double(b.m[2][1]));
  dst.m[2][2] = float(a + double(b.m[2][2]));
  dst.m[2][3] = float(a + double(b.m[2][3]));
  dst.m[3][0] = float(a + double(b.m[3][0]));
  dst.m[3][1] = float(a + double(b.m[3][1]));
  dst.m[3][2] = float(a + double(b.m[3][2]));
  dst.m[3][3] = float(a + double(b.m[3][3]));

  return dst;
}

inline value::matrix4f operator-(const value::matrix4f &a, const double b) {
  value::matrix4f dst;
  dst.m[0][0] = float(double(a.m[0][0]) - b);
  dst.m[0][1] = float(double(a.m[0][1]) - b);
  dst.m[0][2] = float(double(a.m[0][2]) - b);
  dst.m[0][3] = float(double(a.m[0][3]) - b);
  dst.m[1][0] = float(double(a.m[1][0]) - b);
  dst.m[1][1] = float(double(a.m[1][1]) - b);
  dst.m[1][2] = float(double(a.m[1][2]) - b);
  dst.m[1][3] = float(double(a.m[1][3]) - b);
  dst.m[2][0] = float(double(a.m[2][0]) - b);
  dst.m[2][1] = float(double(a.m[2][1]) - b);
  dst.m[2][2] = float(double(a.m[2][2]) - b);
  dst.m[2][3] = float(double(a.m[2][3]) - b);
  dst.m[3][0] = float(double(a.m[3][0]) - b);
  dst.m[3][1] = float(double(a.m[3][1]) - b);
  dst.m[3][2] = float(double(a.m[3][2]) - b);
  dst.m[3][3] = float(double(a.m[3][3]) - b);

  return dst;
}

inline value::matrix4f operator-(const double a, const value::matrix4f &b) {
  value::matrix4f dst;
  dst.m[0][0] = float(a - double(b.m[0][0]));
  dst.m[0][1] = float(a - double(b.m[0][1]));
  dst.m[0][2] = float(a - double(b.m[0][2]));
  dst.m[0][3] = float(a - double(b.m[0][3]));
  dst.m[1][0] = float(a - double(b.m[1][0]));
  dst.m[1][1] = float(a - double(b.m[1][1]));
  dst.m[1][2] = float(a - double(b.m[1][2]));
  dst.m[1][3] = float(a - double(b.m[1][3]));
  dst.m[2][0] = float(a - double(b.m[2][0]));
  dst.m[2][1] = float(a - double(b.m[2][1]));
  dst.m[2][2] = float(a - double(b.m[2][2]));
  dst.m[2][3] = float(a - double(b.m[2][3]));
  dst.m[3][0] = float(a - double(b.m[3][0]));
  dst.m[3][1] = float(a - double(b.m[3][1]));
  dst.m[3][2] = float(a - double(b.m[3][2]));
  dst.m[3][3] = float(a - double(b.m[3][3]));

  return dst;
}
inline value::matrix4f operator*(const value::matrix4f &a, const double b) {
  value::matrix4f dst;
  dst.m[0][0] = float(double(a.m[0][0]) * b);
  dst.m[0][1] = float(double(a.m[0][1]) * b);
  dst.m[0][2] = float(double(a.m[0][2]) * b);
  dst.m[0][3] = float(double(a.m[0][3]) * b);
  dst.m[1][0] = float(double(a.m[1][0]) * b);
  dst.m[1][1] = float(double(a.m[1][1]) * b);
  dst.m[1][2] = float(double(a.m[1][2]) * b);
  dst.m[1][3] = float(double(a.m[1][3]) * b);
  dst.m[2][0] = float(double(a.m[2][0]) * b);
  dst.m[2][1] = float(double(a.m[2][1]) * b);
  dst.m[2][2] = float(double(a.m[2][2]) * b);
  dst.m[2][3] = float(double(a.m[2][3]) * b);
  dst.m[3][0] = float(double(a.m[3][0]) * b);
  dst.m[3][1] = float(double(a.m[3][1]) * b);
  dst.m[3][2] = float(double(a.m[3][2]) * b);
  dst.m[3][3] = float(double(a.m[3][3]) * b);

  return dst;
}

inline value::matrix4f operator*(const double a, const value::matrix4f &b) {
  value::matrix4f dst;
  dst.m[0][0] = float(a * double(b.m[0][0]));
  dst.m[0][1] = float(a * double(b.m[0][1]));
  dst.m[0][2] = float(a * double(b.m[0][2]));
  dst.m[0][3] = float(a * double(b.m[0][3]));
  dst.m[1][0] = float(a * double(b.m[1][0]));
  dst.m[1][1] = float(a * double(b.m[1][1]));
  dst.m[1][2] = float(a * double(b.m[1][2]));
  dst.m[1][3] = float(a * double(b.m[1][3]));
  dst.m[2][0] = float(a * double(b.m[2][0]));
  dst.m[2][1] = float(a * double(b.m[2][1]));
  dst.m[2][2] = float(a * double(b.m[2][2]));
  dst.m[2][3] = float(a * double(b.m[2][3]));
  dst.m[3][0] = float(a * double(b.m[3][0]));
  dst.m[3][1] = float(a * double(b.m[3][1]));
  dst.m[3][2] = float(a * double(b.m[3][2]));
  dst.m[3][3] = float(a * double(b.m[3][3]));

  return dst;
}

inline value::matrix4f operator/(const value::matrix4f &a, const double b) {
  value::matrix4f dst;
  dst.m[0][0] = float(double(a.m[0][0]) / b);
  dst.m[0][1] = float(double(a.m[0][1]) / b);
  dst.m[0][2] = float(double(a.m[0][2]) / b);
  dst.m[0][3] = float(double(a.m[0][3]) / b);
  dst.m[1][0] = float(double(a.m[1][0]) / b);
  dst.m[1][1] = float(double(a.m[1][1]) / b);
  dst.m[1][2] = float(double(a.m[1][2]) / b);
  dst.m[1][3] = float(double(a.m[1][3]) / b);
  dst.m[2][0] = float(double(a.m[2][0]) / b);
  dst.m[2][1] = float(double(a.m[2][1]) / b);
  dst.m[2][2] = float(double(a.m[2][2]) / b);
  dst.m[2][3] = float(double(a.m[2][3]) / b);
  dst.m[3][0] = float(double(a.m[3][0]) / b);
  dst.m[3][1] = float(double(a.m[3][1]) / b);
  dst.m[3][2] = float(double(a.m[3][2]) / b);
  dst.m[3][3] = float(double(a.m[3][3]) / b);

  return dst;
}

inline value::matrix4f operator/(const double a, const value::matrix4f &b) {
  value::matrix4f dst;
  dst.m[0][0] = float(a / double(b.m[0][0]));
  dst.m[0][1] = float(a / double(b.m[0][1]));
  dst.m[0][2] = float(a / double(b.m[0][2]));
  dst.m[0][3] = float(a / double(b.m[0][3]));
  dst.m[1][0] = float(a / double(b.m[1][0]));
  dst.m[1][1] = float(a / double(b.m[1][1]));
  dst.m[1][2] = float(a / double(b.m[1][2]));
  dst.m[1][3] = float(a / double(b.m[1][3]));
  dst.m[2][0] = float(a / double(b.m[2][0]));
  dst.m[2][1] = float(a / double(b.m[2][1]));
  dst.m[2][2] = float(a / double(b.m[2][2]));
  dst.m[2][3] = float(a / double(b.m[2][3]));
  dst.m[3][0] = float(a / double(b.m[3][0]));
  dst.m[3][1] = float(a / double(b.m[3][1]));
  dst.m[3][2] = float(a / double(b.m[3][2]));
  dst.m[3][3] = float(a / double(b.m[3][3]));

  return dst;
}

inline value::matrix2d operator+(const value::matrix2d &a, const double b) {
  value::matrix2d dst;
  dst.m[0][0] = a.m[0][0] + b;
  dst.m[0][1] = a.m[0][1] + b;
  dst.m[1][0] = a.m[1][0] + b;
  dst.m[1][1] = a.m[1][1] + b;

  return dst;
}

inline value::matrix2d operator+(const double a, const value::matrix2d &b) {
  value::matrix2d dst;
  dst.m[0][0] = a + b.m[0][0];
  dst.m[0][1] = a + b.m[0][1];
  dst.m[1][0] = a + b.m[1][0];
  dst.m[1][1] = a + b.m[1][1];

  return dst;
}


inline value::matrix2d operator-(const value::matrix2d &a, const double b) {
  value::matrix2d dst;
  dst.m[0][0] = a.m[0][0] - b;
  dst.m[0][1] = a.m[0][1] - b;
  dst.m[1][0] = a.m[1][0] - b;
  dst.m[1][1] = a.m[1][1] - b;

  return dst;
}

inline value::matrix2d operator-(const double a, const value::matrix2d &b) {
  value::matrix2d dst;
  dst.m[0][0] = a - b.m[0][0];
  dst.m[0][1] = a - b.m[0][1];
  dst.m[1][0] = a - b.m[1][0];
  dst.m[1][1] = a - b.m[1][1];

  return dst;
}

inline value::matrix2d operator*(const value::matrix2d &a, const double b) {
  value::matrix2d dst;
  dst.m[0][0] = a.m[0][0] * b;
  dst.m[0][1] = a.m[0][1] * b;
  dst.m[1][0] = a.m[1][0] * b;
  dst.m[1][1] = a.m[1][1] * b;

  return dst;
}

inline value::matrix2d operator*(const double a, const value::matrix2d &b) {
  value::matrix2d dst;
  dst.m[0][0] = a * b.m[0][0];
  dst.m[0][1] = a * b.m[0][1];
  dst.m[1][0] = a * b.m[1][0];
  dst.m[1][1] = a * b.m[1][1];

  return dst;
}

inline value::matrix2d operator/(const value::matrix2d &a, const double b) {
  value::matrix2d dst;
  dst.m[0][0] = a.m[0][0] / b;
  dst.m[0][1] = a.m[0][1] / b;
  dst.m[1][0] = a.m[1][0] / b;
  dst.m[1][1] = a.m[1][1] / b;

  return dst;
}

inline value::matrix2d operator/(const double a, const value::matrix2d &b) {
  value::matrix2d dst;
  dst.m[0][0] = a / b.m[0][0];
  dst.m[0][1] = a / b.m[0][1];
  dst.m[1][0] = a / b.m[1][0];
  dst.m[1][1] = a / b.m[1][1];

  return dst;
}

inline value::matrix3d operator+(const value::matrix3d &a, const double b) {
  value::matrix3d dst;
  dst.m[0][0] = a.m[0][0] + b;
  dst.m[0][1] = a.m[0][1] + b;
  dst.m[0][2] = a.m[0][2] + b;
  dst.m[1][0] = a.m[1][0] + b;
  dst.m[1][1] = a.m[1][1] + b;
  dst.m[1][2] = a.m[1][2] + b;
  dst.m[2][0] = a.m[2][0] + b;
  dst.m[2][1] = a.m[2][1] + b;
  dst.m[2][2] = a.m[2][2] + b;

  return dst;
}

inline value::matrix3d operator+(const double a, const value::matrix3d &b) {
  value::matrix3d dst;
  dst.m[0][0] = a + b.m[0][0];
  dst.m[0][1] = a + b.m[0][1];
  dst.m[0][2] = a + b.m[0][2];
  dst.m[1][0] = a + b.m[1][0];
  dst.m[1][1] = a + b.m[1][1];
  dst.m[1][2] = a + b.m[1][2];
  dst.m[2][0] = a + b.m[2][0];
  dst.m[2][1] = a + b.m[2][1];
  dst.m[2][2] = a + b.m[2][2];

  return dst;
}


inline value::matrix3d operator-(const value::matrix3d &a, const double b) {
  value::matrix3d dst;
  dst.m[0][0] = a.m[0][0] - b;
  dst.m[0][1] = a.m[0][1] - b;
  dst.m[0][2] = a.m[0][2] - b;
  dst.m[1][0] = a.m[1][0] - b;
  dst.m[1][1] = a.m[1][1] - b;
  dst.m[1][2] = a.m[1][2] - b;
  dst.m[2][0] = a.m[2][0] - b;
  dst.m[2][1] = a.m[2][1] - b;
  dst.m[2][2] = a.m[2][2] - b;

  return dst;
}

inline value::matrix3d operator-(const double a, const value::matrix3d &b) {
  value::matrix3d dst;
  dst.m[0][0] = a - b.m[0][0];
  dst.m[0][1] = a - b.m[0][1];
  dst.m[0][2] = a - b.m[0][2];
  dst.m[1][0] = a - b.m[1][0];
  dst.m[1][1] = a - b.m[1][1];
  dst.m[1][2] = a - b.m[1][2];
  dst.m[2][0] = a - b.m[2][0];
  dst.m[2][1] = a - b.m[2][1];
  dst.m[2][2] = a - b.m[2][2];

  return dst;
}

inline value::matrix3d operator*(const value::matrix3d &a, const double b) {
  value::matrix3d dst;
  dst.m[0][0] = a.m[0][0] * b;
  dst.m[0][1] = a.m[0][1] * b;
  dst.m[0][2] = a.m[0][2] * b;
  dst.m[1][0] = a.m[1][0] * b;
  dst.m[1][1] = a.m[1][1] * b;
  dst.m[1][2] = a.m[1][2] * b;
  dst.m[2][0] = a.m[2][0] * b;
  dst.m[2][1] = a.m[2][1] * b;
  dst.m[2][2] = a.m[2][2] * b;

  return dst;
}

inline value::matrix3d operator*(const double a, const value::matrix3d &b) {
  value::matrix3d dst;
  dst.m[0][0] = a * b.m[0][0];
  dst.m[0][1] = a * b.m[0][1];
  dst.m[0][2] = a * b.m[0][2];
  dst.m[1][0] = a * b.m[1][0];
  dst.m[1][1] = a * b.m[1][1];
  dst.m[1][2] = a * b.m[1][2];
  dst.m[2][0] = a * b.m[2][0];
  dst.m[2][1] = a * b.m[2][1];
  dst.m[2][2] = a * b.m[2][2];

  return dst;
}

inline value::matrix3d operator/(const value::matrix3d &a, const double b) {
  value::matrix3d dst;
  dst.m[0][0] = a.m[0][0] / b;
  dst.m[0][1] = a.m[0][1] / b;
  dst.m[0][2] = a.m[0][2] / b;
  dst.m[1][0] = a.m[1][0] / b;
  dst.m[1][1] = a.m[1][1] / b;
  dst.m[1][2] = a.m[1][2] / b;
  dst.m[2][0] = a.m[2][0] / b;
  dst.m[2][1] = a.m[2][1] / b;
  dst.m[2][2] = a.m[2][2] / b;

  return dst;
}

inline value::matrix3d operator/(const double a, const value::matrix3d &b) {
  value::matrix3d dst;
  dst.m[0][0] = a / b.m[0][0];
  dst.m[0][1] = a / b.m[0][1];
  dst.m[0][2] = a / b.m[0][2];
  dst.m[1][0] = a / b.m[1][0];
  dst.m[1][1] = a / b.m[1][1];
  dst.m[1][2] = a / b.m[1][2];
  dst.m[2][0] = a / b.m[2][0];
  dst.m[2][1] = a / b.m[2][1];
  dst.m[2][2] = a / b.m[2][2];

  return dst;
}

inline value::matrix4d operator+(const value::matrix4d &a, const double b) {
  value::matrix4d dst;
  dst.m[0][0] = a.m[0][0] + b;
  dst.m[0][1] = a.m[0][1] + b;
  dst.m[0][2] = a.m[0][2] + b;
  dst.m[0][3] = a.m[0][3] + b;
  dst.m[1][0] = a.m[1][0] + b;
  dst.m[1][1] = a.m[1][1] + b;
  dst.m[1][2] = a.m[1][2] + b;
  dst.m[1][3] = a.m[1][3] + b;
  dst.m[2][0] = a.m[2][0] + b;
  dst.m[2][1] = a.m[2][1] + b;
  dst.m[2][2] = a.m[2][2] + b;
  dst.m[2][3] = a.m[2][3] + b;
  dst.m[3][0] = a.m[3][0] + b;
  dst.m[3][1] = a.m[3][1] + b;
  dst.m[3][2] = a.m[3][2] + b;
  dst.m[3][3] = a.m[3][3] + b;

  return dst;
}

inline value::matrix4d operator+(const double a, const value::matrix4d &b) {
  value::matrix4d dst;
  dst.m[0][0] = a + b.m[0][0];
  dst.m[0][1] = a + b.m[0][1];
  dst.m[0][2] = a + b.m[0][2];
  dst.m[0][3] = a + b.m[0][3];
  dst.m[1][0] = a + b.m[1][0];
  dst.m[1][1] = a + b.m[1][1];
  dst.m[1][2] = a + b.m[1][2];
  dst.m[1][3] = a + b.m[1][3];
  dst.m[2][0] = a + b.m[2][0];
  dst.m[2][1] = a + b.m[2][1];
  dst.m[2][2] = a + b.m[2][2];
  dst.m[2][3] = a + b.m[2][3];
  dst.m[3][0] = a + b.m[3][0];
  dst.m[3][1] = a + b.m[3][1];
  dst.m[3][2] = a + b.m[3][2];
  dst.m[3][3] = a + b.m[3][3];

  return dst;
}

inline value::matrix4d operator-(const value::matrix4d &a, const double b) {
  value::matrix4d dst;
  dst.m[0][0] = a.m[0][0] - b;
  dst.m[0][1] = a.m[0][1] - b;
  dst.m[0][2] = a.m[0][2] - b;
  dst.m[0][3] = a.m[0][3] - b;
  dst.m[1][0] = a.m[1][0] - b;
  dst.m[1][1] = a.m[1][1] - b;
  dst.m[1][2] = a.m[1][2] - b;
  dst.m[1][3] = a.m[1][3] - b;
  dst.m[2][0] = a.m[2][0] - b;
  dst.m[2][1] = a.m[2][1] - b;
  dst.m[2][2] = a.m[2][2] - b;
  dst.m[2][3] = a.m[2][3] - b;
  dst.m[3][0] = a.m[3][0] - b;
  dst.m[3][1] = a.m[3][1] - b;
  dst.m[3][2] = a.m[3][2] - b;
  dst.m[3][3] = a.m[3][3] - b;

  return dst;
}

inline value::matrix4d operator-(const double a, const value::matrix4d &b) {
  value::matrix4d dst;
  dst.m[0][0] = a - b.m[0][0];
  dst.m[0][1] = a - b.m[0][1];
  dst.m[0][2] = a - b.m[0][2];
  dst.m[0][3] = a - b.m[0][3];
  dst.m[1][0] = a - b.m[1][0];
  dst.m[1][1] = a - b.m[1][1];
  dst.m[1][2] = a - b.m[1][2];
  dst.m[1][3] = a - b.m[1][3];
  dst.m[2][0] = a - b.m[2][0];
  dst.m[2][1] = a - b.m[2][1];
  dst.m[2][2] = a - b.m[2][2];
  dst.m[2][3] = a - b.m[2][3];
  dst.m[3][0] = a - b.m[3][0];
  dst.m[3][1] = a - b.m[3][1];
  dst.m[3][2] = a - b.m[3][2];
  dst.m[3][3] = a - b.m[3][3];

  return dst;
}
inline value::matrix4d operator*(const value::matrix4d &a, const double b) {
  value::matrix4d dst;
  dst.m[0][0] = a.m[0][0] * b;
  dst.m[0][1] = a.m[0][1] * b;
  dst.m[0][2] = a.m[0][2] * b;
  dst.m[0][3] = a.m[0][3] * b;
  dst.m[1][0] = a.m[1][0] * b;
  dst.m[1][1] = a.m[1][1] * b;
  dst.m[1][2] = a.m[1][2] * b;
  dst.m[1][3] = a.m[1][3] * b;
  dst.m[2][0] = a.m[2][0] * b;
  dst.m[2][1] = a.m[2][1] * b;
  dst.m[2][2] = a.m[2][2] * b;
  dst.m[2][3] = a.m[2][3] * b;
  dst.m[3][0] = a.m[3][0] * b;
  dst.m[3][1] = a.m[3][1] * b;
  dst.m[3][2] = a.m[3][2] * b;
  dst.m[3][3] = a.m[3][3] * b;

  return dst;
}

inline value::matrix4d operator*(const double a, const value::matrix4d &b) {
  value::matrix4d dst;
  dst.m[0][0] = a * b.m[0][0];
  dst.m[0][1] = a * b.m[0][1];
  dst.m[0][2] = a * b.m[0][2];
  dst.m[0][3] = a * b.m[0][3];
  dst.m[1][0] = a * b.m[1][0];
  dst.m[1][1] = a * b.m[1][1];
  dst.m[1][2] = a * b.m[1][2];
  dst.m[1][3] = a * b.m[1][3];
  dst.m[2][0] = a * b.m[2][0];
  dst.m[2][1] = a * b.m[2][1];
  dst.m[2][2] = a * b.m[2][2];
  dst.m[2][3] = a * b.m[2][3];
  dst.m[3][0] = a * b.m[3][0];
  dst.m[3][1] = a * b.m[3][1];
  dst.m[3][2] = a * b.m[3][2];
  dst.m[3][3] = a * b.m[3][3];

  return dst;
}

inline value::matrix4d operator/(const value::matrix4d &a, const double b) {
  value::matrix4d dst;
  dst.m[0][0] = a.m[0][0] / b;
  dst.m[0][1] = a.m[0][1] / b;
  dst.m[0][2] = a.m[0][2] / b;
  dst.m[0][3] = a.m[0][3] / b;
  dst.m[1][0] = a.m[1][0] / b;
  dst.m[1][1] = a.m[1][1] / b;
  dst.m[1][2] = a.m[1][2] / b;
  dst.m[1][3] = a.m[1][3] / b;
  dst.m[2][0] = a.m[2][0] / b;
  dst.m[2][1] = a.m[2][1] / b;
  dst.m[2][2] = a.m[2][2] / b;
  dst.m[2][3] = a.m[2][3] / b;
  dst.m[3][0] = a.m[3][0] / b;
  dst.m[3][1] = a.m[3][1] / b;
  dst.m[3][2] = a.m[3][2] / b;
  dst.m[3][3] = a.m[3][3] / b;

  return dst;
}

inline value::matrix4d operator/(const double a, const value::matrix4d &b) {
  value::matrix4d dst;
  dst.m[0][0] = a / b.m[0][0];
  dst.m[0][1] = a / b.m[0][1];
  dst.m[0][2] = a / b.m[0][2];
  dst.m[0][3] = a / b.m[0][3];
  dst.m[1][0] = a / b.m[1][0];
  dst.m[1][1] = a / b.m[1][1];
  dst.m[1][2] = a / b.m[1][2];
  dst.m[1][3] = a / b.m[1][3];
  dst.m[2][0] = a / b.m[2][0];
  dst.m[2][1] = a / b.m[2][1];
  dst.m[2][2] = a / b.m[2][2];
  dst.m[2][3] = a / b.m[2][3];
  dst.m[3][0] = a / b.m[3][0];
  dst.m[3][1] = a / b.m[3][1];
  dst.m[3][2] = a / b.m[3][2];
  dst.m[3][3] = a / b.m[3][3];

  return dst;
}

inline value::frame4d operator+(const value::frame4d &a, const value::frame4d &b) {
  value::frame4d dst;
  dst.m[0][0] = a.m[0][0] + b.m[0][0];
  dst.m[0][1] = a.m[0][1] + b.m[0][1];
  dst.m[0][2] = a.m[0][2] + b.m[0][2];
  dst.m[0][3] = a.m[0][3] + b.m[0][3];
  dst.m[1][0] = a.m[1][0] + b.m[1][0];
  dst.m[1][1] = a.m[1][1] + b.m[1][1];
  dst.m[1][2] = a.m[1][2] + b.m[1][2];
  dst.m[1][3] = a.m[1][3] + b.m[1][3];
  dst.m[2][0] = a.m[2][0] + b.m[2][0];
  dst.m[2][1] = a.m[2][1] + b.m[2][1];
  dst.m[2][2] = a.m[2][2] + b.m[2][2];
  dst.m[2][3] = a.m[2][3] + b.m[2][3];
  dst.m[3][0] = a.m[3][0] + b.m[3][0];
  dst.m[3][1] = a.m[3][1] + b.m[3][1];
  dst.m[3][2] = a.m[3][2] + b.m[3][2];
  dst.m[3][3] = a.m[3][3] + b.m[3][3];

  return dst;
}

inline value::frame4d operator+(const value::frame4d &a, const double b) {
  value::frame4d dst;
  dst.m[0][0] = a.m[0][0] + b;
  dst.m[0][1] = a.m[0][1] + b;
  dst.m[0][2] = a.m[0][2] + b;
  dst.m[0][3] = a.m[0][3] + b;
  dst.m[1][0] = a.m[1][0] + b;
  dst.m[1][1] = a.m[1][1] + b;
  dst.m[1][2] = a.m[1][2] + b;
  dst.m[1][3] = a.m[1][3] + b;
  dst.m[2][0] = a.m[2][0] + b;
  dst.m[2][1] = a.m[2][1] + b;
  dst.m[2][2] = a.m[2][2] + b;
  dst.m[2][3] = a.m[2][3] + b;
  dst.m[3][0] = a.m[3][0] + b;
  dst.m[3][1] = a.m[3][1] + b;
  dst.m[3][2] = a.m[3][2] + b;
  dst.m[3][3] = a.m[3][3] + b;

  return dst;
}

inline value::frame4d operator+(const double a, const value::frame4d &b) {
  value::frame4d dst;
  dst.m[0][0] = a + b.m[0][0];
  dst.m[0][1] = a + b.m[0][1];
  dst.m[0][2] = a + b.m[0][2];
  dst.m[0][3] = a + b.m[0][3];
  dst.m[1][0] = a + b.m[1][0];
  dst.m[1][1] = a + b.m[1][1];
  dst.m[1][2] = a + b.m[1][2];
  dst.m[1][3] = a + b.m[1][3];
  dst.m[2][0] = a + b.m[2][0];
  dst.m[2][1] = a + b.m[2][1];
  dst.m[2][2] = a + b.m[2][2];
  dst.m[2][3] = a + b.m[2][3];
  dst.m[3][0] = a + b.m[3][0];
  dst.m[3][1] = a + b.m[3][1];
  dst.m[3][2] = a + b.m[3][2];
  dst.m[3][3] = a + b.m[3][3];

  return dst;
}

inline value::frame4d operator-(const value::frame4d &a, const double b) {
  value::frame4d dst;
  dst.m[0][0] = a.m[0][0] - b;
  dst.m[0][1] = a.m[0][1] - b;
  dst.m[0][2] = a.m[0][2] - b;
  dst.m[0][3] = a.m[0][3] - b;
  dst.m[1][0] = a.m[1][0] - b;
  dst.m[1][1] = a.m[1][1] - b;
  dst.m[1][2] = a.m[1][2] - b;
  dst.m[1][3] = a.m[1][3] - b;
  dst.m[2][0] = a.m[2][0] - b;
  dst.m[2][1] = a.m[2][1] - b;
  dst.m[2][2] = a.m[2][2] - b;
  dst.m[2][3] = a.m[2][3] - b;
  dst.m[3][0] = a.m[3][0] - b;
  dst.m[3][1] = a.m[3][1] - b;
  dst.m[3][2] = a.m[3][2] - b;
  dst.m[3][3] = a.m[3][3] - b;

  return dst;
}

inline value::frame4d operator-(const double a, const value::frame4d &b) {
  value::frame4d dst;
  dst.m[0][0] = a - b.m[0][0];
  dst.m[0][1] = a - b.m[0][1];
  dst.m[0][2] = a - b.m[0][2];
  dst.m[0][3] = a - b.m[0][3];
  dst.m[1][0] = a - b.m[1][0];
  dst.m[1][1] = a - b.m[1][1];
  dst.m[1][2] = a - b.m[1][2];
  dst.m[1][3] = a - b.m[1][3];
  dst.m[2][0] = a - b.m[2][0];
  dst.m[2][1] = a - b.m[2][1];
  dst.m[2][2] = a - b.m[2][2];
  dst.m[2][3] = a - b.m[2][3];
  dst.m[3][0] = a - b.m[3][0];
  dst.m[3][1] = a - b.m[3][1];
  dst.m[3][2] = a - b.m[3][2];
  dst.m[3][3] = a - b.m[3][3];

  return dst;
}
inline value::frame4d operator*(const value::frame4d &a, const double b) {
  value::frame4d dst;
  dst.m[0][0] = a.m[0][0] * b;
  dst.m[0][1] = a.m[0][1] * b;
  dst.m[0][2] = a.m[0][2] * b;
  dst.m[0][3] = a.m[0][3] * b;
  dst.m[1][0] = a.m[1][0] * b;
  dst.m[1][1] = a.m[1][1] * b;
  dst.m[1][2] = a.m[1][2] * b;
  dst.m[1][3] = a.m[1][3] * b;
  dst.m[2][0] = a.m[2][0] * b;
  dst.m[2][1] = a.m[2][1] * b;
  dst.m[2][2] = a.m[2][2] * b;
  dst.m[2][3] = a.m[2][3] * b;
  dst.m[3][0] = a.m[3][0] * b;
  dst.m[3][1] = a.m[3][1] * b;
  dst.m[3][2] = a.m[3][2] * b;
  dst.m[3][3] = a.m[3][3] * b;

  return dst;
}

inline value::frame4d operator*(const double a, const value::frame4d &b) {
  value::frame4d dst;
  dst.m[0][0] = a * b.m[0][0];
  dst.m[0][1] = a * b.m[0][1];
  dst.m[0][2] = a * b.m[0][2];
  dst.m[0][3] = a * b.m[0][3];
  dst.m[1][0] = a * b.m[1][0];
  dst.m[1][1] = a * b.m[1][1];
  dst.m[1][2] = a * b.m[1][2];
  dst.m[1][3] = a * b.m[1][3];
  dst.m[2][0] = a * b.m[2][0];
  dst.m[2][1] = a * b.m[2][1];
  dst.m[2][2] = a * b.m[2][2];
  dst.m[2][3] = a * b.m[2][3];
  dst.m[3][0] = a * b.m[3][0];
  dst.m[3][1] = a * b.m[3][1];
  dst.m[3][2] = a * b.m[3][2];
  dst.m[3][3] = a * b.m[3][3];

  return dst;
}

inline value::frame4d operator/(const value::frame4d &a, const double b) {
  value::frame4d dst;
  dst.m[0][0] = a.m[0][0] / b;
  dst.m[0][1] = a.m[0][1] / b;
  dst.m[0][2] = a.m[0][2] / b;
  dst.m[0][3] = a.m[0][3] / b;
  dst.m[1][0] = a.m[1][0] / b;
  dst.m[1][1] = a.m[1][1] / b;
  dst.m[1][2] = a.m[1][2] / b;
  dst.m[1][3] = a.m[1][3] / b;
  dst.m[2][0] = a.m[2][0] / b;
  dst.m[2][1] = a.m[2][1] / b;
  dst.m[2][2] = a.m[2][2] / b;
  dst.m[2][3] = a.m[2][3] / b;
  dst.m[3][0] = a.m[3][0] / b;
  dst.m[3][1] = a.m[3][1] / b;
  dst.m[3][2] = a.m[3][2] / b;
  dst.m[3][3] = a.m[3][3] / b;

  return dst;
}

inline value::frame4d operator/(const double a, const value::frame4d &b) {
  value::frame4d dst;
  dst.m[0][0] = a / b.m[0][0];
  dst.m[0][1] = a / b.m[0][1];
  dst.m[0][2] = a / b.m[0][2];
  dst.m[0][3] = a / b.m[0][3];
  dst.m[1][0] = a / b.m[1][0];
  dst.m[1][1] = a / b.m[1][1];
  dst.m[1][2] = a / b.m[1][2];
  dst.m[1][3] = a / b.m[1][3];
  dst.m[2][0] = a / b.m[2][0];
  dst.m[2][1] = a / b.m[2][1];
  dst.m[2][2] = a / b.m[2][2];
  dst.m[2][3] = a / b.m[2][3];
  dst.m[3][0] = a / b.m[3][0];
  dst.m[3][1] = a / b.m[3][1];
  dst.m[3][2] = a / b.m[3][2];
  dst.m[3][3] = a / b.m[3][3];

  return dst;
}

// half

#if 0
inline value::half2 operator+(const value::half2 &a, const value::half2 &b) {
  return {a[0] + b[0], a[1] + b[1]};
}

inline value::half2 operator+(const float a, const value::half2 &b) {
  return {a + b[0], a + b[1]};
}

inline value::half2 operator+(const value::half2 &a, const float b) {
  return {a[0] + b, a[1] + b};
}

inline value::half2 operator-(const value::half2 &a, const value::half2 &b) {
  return {a[0] - b[0], a[1] - b[1]};
}

inline value::half2 operator-(const float a, const value::half2 &b) {
  return {a - b[0], a - b[1]};
}

inline value::half2 operator-(const value::half2 &a, const float b) {
  return {a[0] - b, a[1] - b};
}

inline value::half2 operator*(const value::half2 &a, const value::half2 &b) {
  return {a[0] * b[0], a[1] * b[1]};
}

inline value::half2 operator*(const float a, const value::half2 &b) {
  return {a * b[0], a * b[1]};
}

inline value::half2 operator*(const value::half2 &a, const float b) {
  return {a[0] * b, a[1] * b};
}

inline value::half2 operator/(const value::half2 &a, const value::half2 &b) {
  return {a[0] / b[0], a[1] / b[1]};
}

inline value::half2 operator/(const float a, const value::half2 &b) {
  return {a / b[0], a / b[1]};
}

inline value::half2 operator/(const value::half2 &a, const float b) {
  return {a[0] / b, a[1] / b};
}

inline value::half3 operator+(const value::half3 &a, const value::half3 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline value::half3 operator+(const float a, const value::half3 &b) {
  return {a + b[0], a + b[1], a + b[2]};
}

inline value::half3 operator+(const value::half3 &a, const float b) {
  return {a[0] + b, a[1] + b, a[2] + b};
}

inline value::half3 operator*(const value::half3 &a, const value::half3 &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

inline value::half3 operator*(const float a, const value::half3 &b) {
  return {a * b[0], a * b[1], a * b[2]};
}

inline value::half3 operator*(const value::half3 &a, const float b) {
  return {a[0] * b, a[1] * b, a[2] * b};
}

inline value::half3 operator/(const value::half3 &a, const value::half3 &b) {
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2]};
}

inline value::half3 operator/(const float a, const value::half3 &b) {
  return {a / b[0], a / b[1], a / b[2]};
}

inline value::half3 operator/(const value::half3 &a, const float b) {
  return {a[0] / b, a[1] / b, a[2] / b};
}

inline value::half4 operator+(const value::half4 &a, const value::half4 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] - b[2], a[3] - b[3]};
}

inline value::half4 operator+(const float a, const value::half4 &b) {
  return {a + b[0], a + b[1], a + b[2], a + b[3]};
}

inline value::half4 operator+(const value::half4 &a, const float b) {
  return {a[0] + b, a[1] + b, a[2] + b, a[3] + b};
}

inline value::half4 operator-(const value::half4 &a, const value::half4 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}

inline value::half4 operator-(const float a, const value::half4 &b) {
  return {a - b[0], a - b[1], a - b[2], a - b[3]};
}

inline value::half4 operator-(const value::half4 &a, const float b) {
  return {a[0] - b, a[1] - b, a[2] - b, a[3] - b};
}

inline value::half4 operator*(const value::half4 &a, const value::half4 &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]};
}

inline value::half4 operator*(const float a, const value::half4 &b) {
  return {a * b[0], a * b[1], a * b[2], a * b[3]};
}

inline value::half4 operator*(const value::half4 &a, const float b) {

  return {a[0] * b, a[1] * b, a[2] * b, a[3] * b};
}

inline value::half4 operator/(const value::half4 &a, const value::half4 &b) {
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]};
}

inline value::half4 operator/(const float a, const value::half4 &b) {
  return {a / b[0], a / b[1], a / b[2], a / b[3]};
}

inline value::half4 operator/(const value::half4 &a, const float b) {

  return {a[0] / b, a[1] / b, a[2] / b, a[3] / b};
}

// float

inline value::float2 operator+(const value::float2 &a, const value::float2 &b) {
  return {a[0] + b[0], a[1] + b[1]};
}

inline value::float2 operator+(const float a, const value::float2 &b) {
  return {a + b[0], a + b[1]};
}

inline value::float2 operator+(const value::float2 &a, const float b) {
  return {a[0] + b, a[1] + b};
}


inline value::float2 operator-(const value::float2 &a, const value::float2 &b) {
  return {a[0] - b[0], a[1] - b[1]};
}

inline value::float2 operator-(const float a, const value::float2 &b) {
  return {a - b[0], a - b[1]};
}

inline value::float2 operator-(const value::float2 &a, const float b) {
  return {a[0] - b, a[1] - b};
}

inline value::float2 operator*(const value::float2 &a, const value::float2 &b) {
  return {a[0] * b[0], a[1] * b[1]};
}

inline value::float2 operator*(const float a, const value::float2 &b) {
  return {a * b[0], a * b[1]};
}

inline value::float2 operator*(const value::float2 &a, const float b) {
  return {a[0] * b, a[1] * b};
}

inline value::float2 operator/(const value::float2 &a, const value::float2 &b) {
  return {a[0] / b[0], a[1] / b[1]};
}

inline value::float2 operator/(const float a, const value::float2 &b) {
  return {a / b[0], a / b[1]};
}

inline value::float2 operator/(const value::float2 &a, const float b) {
  return {a[0] / b, a[1] / b};
}

inline value::float3 operator+(const value::float3 &a, const value::float3 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline value::float3 operator+(const float a, const value::float3 &b) {
  return {a + b[0], a + b[1], a + b[2]};
}

inline value::float3 operator+(const value::float3 &a, const float b) {
  return {a[0] + b, a[1] + b, a[2] + b};
}

inline value::float3 operator-(const value::float3 &a, const value::float3 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline value::float3 operator-(const float a, const value::float3 &b) {
  return {a - b[0], a - b[1], a - b[2]};
}

inline value::float3 operator-(const value::float3 &a, const float b) {
  return {a[0] - b, a[1] - b, a[2] - b};
}

inline value::float3 operator*(const value::float3 &a, const value::float3 &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

inline value::float3 operator*(const float a, const value::float3 &b) {
  return {a * b[0], a * b[1], a * b[2]};
}

inline value::float3 operator*(const value::float3 &a, const float b) {
  return {a[0] * b, a[1] * b, a[2] * b};
}

inline value::float3 operator/(const value::float3 &a, const value::float3 &b) {
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2]};
}

inline value::float3 operator/(const float a, const value::float3 &b) {
  return {a / b[0], a / b[1], a / b[2]};
}

inline value::float3 operator/(const value::float3 &a, const float b) {
  return {a[0] / b, a[1] / b, a[2] / b};
}

inline value::float4 operator+(const value::float4 &a, const value::float4 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

inline value::float4 operator+(const float a, const value::float4 &b) {
  return {a + b[0], a + b[1], a + b[2], a + b[3]};
}

inline value::float4 operator+(const value::float4 &a, const float b) {
  return {a[0] + b, a[1] + b, a[2] + b, a[3] + b};
}

inline value::float4 operator-(const value::float4 &a, const value::float4 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}

inline value::float4 operator-(const float a, const value::float4 &b) {
  return {a - b[0], a - b[1], a - b[2], a - b[3]};
}

inline value::float4 operator-(const value::float4 &a, const float b) {
  return {a[0] - b, a[1] - b, a[2] - b, a[3] - b};
}



inline value::float4 operator*(const value::float4 &a, const value::float4 &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]};
}

inline value::float4 operator*(const float a, const value::float4 &b) {
  return {a * b[0], a * b[1], a * b[2], a * b[3]};
}

inline value::float4 operator*(const value::float4 &a, const float b) {

  return {a[0] * b, a[1] * b, a[2] * b, a[3] * b};
}

// double

inline value::double2 operator+(const value::double2 &a, const value::double2 &b) {
  return {a[0] + b[0], a[1] + b[1]};
}

inline value::double2 operator+(const double a, const value::double2 &b) {
  return {a + b[0], a + b[1]};
}

inline value::double2 operator+(const value::double2 &a, const double b) {
  return {a[0] + b, a[1] + b};
}


inline value::double2 operator-(const value::double2 &a, const value::double2 &b) {
  return {a[0] - b[0], a[1] - b[1]};
}

inline value::double2 operator-(const double a, const value::double2 &b) {
  return {a - b[0], a - b[1]};
}

inline value::double2 operator-(const value::double2 &a, const double b) {
  return {a[0] - b, a[1] - b};
}

inline value::double2 operator*(const value::double2 &a, const value::double2 &b) {
  return {a[0] * b[0], a[1] * b[1]};
}

inline value::double2 operator*(const double a, const value::double2 &b) {
  return {a * b[0], a * b[1]};
}

inline value::double2 operator*(const value::double2 &a, const double b) {
  return {a[0] * b, a[1] * b};
}

inline value::double2 operator/(const value::double2 &a, const value::double2 &b) {
  return {a[0] / b[0], a[1] / b[1]};
}

inline value::double2 operator/(const double a, const value::double2 &b) {
  return {a / b[0], a / b[1]};
}

inline value::double2 operator/(const value::double2 &a, const double b) {
  return {a[0] / b, a[1] / b};
}

inline value::double3 operator+(const value::double3 &a, const value::double3 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline value::double3 operator+(const double a, const value::double3 &b) {
  return {a + b[0], a + b[1], a + b[2]};
}

inline value::double3 operator+(const value::double3 &a, const double b) {
  return {a[0] + b, a[1] + b, a[2] + b};
}

inline value::double3 operator-(const value::double3 &a, const value::double3 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline value::double3 operator-(const double a, const value::double3 &b) {
  return {a - b[0], a - b[1], a - b[2]};
}

inline value::double3 operator-(const value::double3 &a, const double b) {
  return {a[0] - b, a[1] - b, a[2] - b};
}

inline value::double3 operator*(const value::double3 &a, const value::double3 &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

inline value::double3 operator*(const double a, const value::double3 &b) {
  return {a * b[0], a * b[1], a * b[2]};
}

inline value::double3 operator*(const value::double3 &a, const double b) {
  return {a[0] * b, a[1] * b, a[2] * b};
}

inline value::double3 operator/(const value::double3 &a, const value::double3 &b) {
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2]};
}

inline value::double3 operator/(const double a, const value::double3 &b) {
  return {a / b[0], a / b[1], a / b[2]};
}

inline value::double3 operator/(const value::double3 &a, const double b) {
  return {a[0] / b, a[1] / b, a[2] / b};
}

inline value::double4 operator+(const value::double4 &a, const value::double4 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

inline value::double4 operator+(const double a, const value::double4 &b) {
  return {a + b[0], a + b[1], a + b[2], a + b[3]};
}

inline value::double4 operator+(const value::double4 &a, const double b) {
  return {a[0] + b, a[1] + b, a[2] + b, a[3] + b};
}

inline value::double4 operator-(const value::double4 &a, const value::double4 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}

inline value::double4 operator-(const double a, const value::double4 &b) {
  return {a - b[0], a - b[1], a - b[2], a - b[3]};
}

inline value::double4 operator-(const value::double4 &a, const double b) {
  return {a[0] - b, a[1] - b, a[2] - b, a[3] - b};
}



inline value::double4 operator*(const value::double4 &a, const value::double4 &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]};
}

inline value::double4 operator*(const double a, const value::double4 &b) {
  return {a * b[0], a * b[1], a * b[2], a * b[3]};
}

inline value::double4 operator*(const value::double4 &a, const double b) {

  return {a[0] * b, a[1] * b, a[2] * b, a[3] * b};
}

// normal
inline value::normal3f operator+(const value::normal3f &a, const value::normal3f &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline value::normal3f operator+(const float a, const value::normal3f &b) {
  return {a + b[0], a + b[1], a + b[2]};
}

inline value::normal3f operator+(const value::normal3f &a, const float b) {
  return {a[0] + b, a[1] + b, a[2] + b};
}

inline value::normal3f operator-(const value::normal3f &a, const value::normal3f &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline value::normal3f operator-(const float a, const value::normal3f &b) {
  return {a - b[0], a - b[1], a - b[2]};
}

inline value::normal3f operator-(const value::normal3f &a, const float b) {
  return {a[0] - b, a[1] - b, a[2] - b};
}

inline value::normal3f operator*(const value::normal3f &a, const value::normal3f &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

inline value::normal3f operator*(const float a, const value::normal3f &b) {
  return {a * b[0], a * b[1], a * b[2]};
}

inline value::normal3f operator*(const value::normal3f &a, const float b) {
  return {a[0] * b, a[1] * b, a[2] * b};
}

inline value::normal3f operator/(const value::normal3f &a, const value::normal3f &b) {
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2]};
}

inline value::normal3f operator/(const float a, const value::normal3f &b) {
  return {a / b[0], a / b[1], a / b[2]};
}

inline value::normal3f operator/(const value::normal3f &a, const float b) {
  return {a[0] / b, a[1] / b, a[2] / b};
}

// normal
inline value::normal3d operator+(const value::normal3d &a, const value::normal3d &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline value::normal3d operator+(const double a, const value::normal3d &b) {
  return {a + b[0], a + b[1], a + b[2]};
}

inline value::normal3d operator+(const value::normal3d &a, const double b) {
  return {a[0] + b, a[1] + b, a[2] + b};
}

inline value::normal3d operator-(const value::normal3d &a, const value::normal3d &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline value::normal3d operator-(const double a, const value::normal3d &b) {
  return {a - b[0], a - b[1], a - b[2]};
}

inline value::normal3d operator-(const value::normal3d &a, const double b) {
  return {a[0] - b, a[1] - b, a[2] - b};
}

inline value::normal3d operator*(const value::normal3d &a, const value::normal3d &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

inline value::normal3d operator*(const double a, const value::normal3d &b) {
  return {a * b[0], a * b[1], a * b[2]};
}

inline value::normal3d operator*(const value::normal3d &a, const double b) {
  return {a[0] * b, a[1] * b, a[2] * b};
}

inline value::normal3d operator/(const value::normal3d &a, const value::normal3d &b) {
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2]};
}

inline value::normal3d operator/(const double a, const value::normal3d &b) {
  return {a / b[0], a / b[1], a / b[2]};
}

inline value::normal3d operator/(const value::normal3d &a, const double b) {
  return {a[0] / b, a[1] / b, a[2] / b};
}

// vector
inline value::vector3f operator+(const value::vector3f &a, const value::vector3f &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline value::vector3f operator+(const float a, const value::vector3f &b) {
  return {a + b[0], a + b[1], a + b[2]};
}

inline value::vector3f operator+(const value::vector3f &a, const float b) {
  return {a[0] + b, a[1] + b, a[2] + b};
}

inline value::vector3f operator-(const value::vector3f &a, const value::vector3f &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline value::vector3f operator-(const float a, const value::vector3f &b) {
  return {a - b[0], a - b[1], a - b[2]};
}

inline value::vector3f operator-(const value::vector3f &a, const float b) {
  return {a[0] - b, a[1] - b, a[2] - b};
}

inline value::vector3f operator*(const value::vector3f &a, const value::vector3f &b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

inline value::vector3f operator*(const float a, const value::vector3f &b) {
  return {a * b[0], a * b[1], a * b[2]};
}

inline value::vector3f operator*(const value::vector3f &a, const float b) {
  return {a[0] * b, a[1] * b, a[2] * b};
}

inline value::vector3f operator/(const value::vector3f &a, const value::vector3f &b) {
  return {a[0] / b[0], a[1] / b[1], a[2] / b[2]};
}

inline value::vector3f operator/(const float a, const value::vector3f &b) {
  return {a / b[0], a / b[1], a / b[2]};
}

inline value::vector3f operator/(const value::vector3f &a, const float b) {
  return {a[0] / b, a[1] / b, a[2] / b};
}
#endif

//
// Inlined lerp
// (Use prim-types.hh::Lerp if you do not need the performance)

// no lerp by default
template <typename T>
inline T lerp(const T &a, const T &b, const double t) {
  (void)b;
  (void)t;
  return a;
}

#define TUSD_INLINED_LERP(__ty, __interp_ty) \
template <> \
inline __ty lerp(const __ty &a, const __ty &b, const double t) { \
  return __interp_ty(1.0 - t) * a + __interp_ty(t) * b; \
}

TUSD_INLINED_LERP(value::half, float)
TUSD_INLINED_LERP(value::half2, float)
TUSD_INLINED_LERP(value::half3, float)
TUSD_INLINED_LERP(value::half4, float)
TUSD_INLINED_LERP(float, float)
TUSD_INLINED_LERP(value::float2, float)
TUSD_INLINED_LERP(value::float3, float)
TUSD_INLINED_LERP(value::float4, float)
TUSD_INLINED_LERP(double, double)
TUSD_INLINED_LERP(value::double2, double)
TUSD_INLINED_LERP(value::double3, double)
TUSD_INLINED_LERP(value::double4, double)
TUSD_INLINED_LERP(value::normal3h, float)
TUSD_INLINED_LERP(value::normal3f, float)
TUSD_INLINED_LERP(value::normal3d, double)
TUSD_INLINED_LERP(value::vector3h, float)
TUSD_INLINED_LERP(value::vector3f, float)
TUSD_INLINED_LERP(value::vector3d, double)
TUSD_INLINED_LERP(value::point3h, float)
TUSD_INLINED_LERP(value::point3f, float)
TUSD_INLINED_LERP(value::point3d, double)
TUSD_INLINED_LERP(value::color3h, float)
TUSD_INLINED_LERP(value::color3f, float)
TUSD_INLINED_LERP(value::color3d, double)
TUSD_INLINED_LERP(value::color4h, float)
TUSD_INLINED_LERP(value::color4f, float)
TUSD_INLINED_LERP(value::color4d, double)
TUSD_INLINED_LERP(value::texcoord2h, float)
TUSD_INLINED_LERP(value::texcoord2f, float)
TUSD_INLINED_LERP(value::texcoord2d, double)
TUSD_INLINED_LERP(value::texcoord3h, float)
TUSD_INLINED_LERP(value::texcoord3f, float)
TUSD_INLINED_LERP(value::texcoord3d, double)

// TODO: robust arithmetic for matrix add/sub/mul/div
TUSD_INLINED_LERP(value::matrix2f, double)
TUSD_INLINED_LERP(value::matrix3f, double)
TUSD_INLINED_LERP(value::matrix4f, double)
TUSD_INLINED_LERP(value::matrix2d, double)
TUSD_INLINED_LERP(value::matrix3d, double)
TUSD_INLINED_LERP(value::matrix4d, double)
TUSD_INLINED_LERP(value::frame4d, double)
//TUSD_INLINED_LERP(value::timecode, double)

#undef TUSD_INLINED_LERP



// for generic vector data.
template <typename T>
inline std::vector<T> lerp(const std::vector<T> &a, const std::vector<T> &b,
                           const double t) {
  std::vector<T> dst;

  // Choose shorter one
  size_t n = std::min(a.size(), b.size());
  if (n == 0) {
    return dst;
  }

  dst.resize(n);

  if (a.size() != b.size()) {
    return dst;
  }
  for (size_t i = 0; i < n; i++) {
    dst[i] = lerp(a[i], b[i], t);
  }

  return dst;
}

template <>
inline value::quath lerp(const value::quath &a, const value::quath &b, const double t) {
  // to float.
  value::quatf af;
  value::quatf bf;
  af.real = half_to_float(a.real);
  af.imag[0] = half_to_float(a.imag[0]);
  af.imag[1] = half_to_float(a.imag[1]);
  af.imag[2] = half_to_float(a.imag[2]);

  bf.real = half_to_float(b.real);
  bf.imag[0] = half_to_float(b.imag[0]);
  bf.imag[1] = half_to_float(b.imag[1]);
  bf.imag[2] = half_to_float(b.imag[2]);

  value::quatf ret =  slerp(af, bf, float(t));
  value::quath h;
  h.real = value::float_to_half_full(ret.real);
  h.imag[0] = value::float_to_half_full(ret.imag[0]);
  h.imag[1] = value::float_to_half_full(ret.imag[1]);
  h.imag[2] = value::float_to_half_full(ret.imag[2]);

  return h;
}

template <>
inline value::quatf lerp(const value::quatf &a, const value::quatf &b, const double t) {
  // slerp
  return slerp(a, b, float(t));
}

template <>
inline value::quatd lerp(const value::quatd &a, const value::quatd &b, const double t) {
  // slerp
  return slerp(a, b, t);
}

#if 0
// specializations for non-lerp-able types
template <>
inline value::AssetPath lerp(const value::AssetPath &a,
                             const value::AssetPath &b, const double t) {
  (void)b;
  (void)t;
  // no interpolation
  return a;
}

template <>
inline std::vector<value::AssetPath> lerp(
    const std::vector<value::AssetPath> &a,
    const std::vector<value::AssetPath> &b, const double t) {
  (void)b;
  (void)t;
  // no interpolation
  return a;
}
#endif


} // namespace tinyusdz
