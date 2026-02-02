// SPDX-License-Identifier: Apache 2.0
// Copyright 2021 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.

///
/// Type-erasure technique for Value, a Value class which can represent USD's
/// mandatory and frequently used types(e.g. `float3`, `token`, `asset`) and its
/// array and compound-types(1D/2D array, dictionary). Neigher std::any nor
/// std::variant is applicable for such usecases, so write our own.
///
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// TODO(syoyo): Use C++17 std::optional when compiled with C++-17 compiler

// clang and gcc
#if defined(__EXCEPTIONS) || defined(__cpp_exceptions)

#ifdef nsel_CONFIG_NO_EXCEPTIONS
#undef nsel_CONFIG_NO_EXCEPTIONS
#endif
#ifdef nssv_CONFIG_NO_EXCEPTIONS
#undef nssv_CONFIG_NO_EXCEPTIONS
#endif

#define nsel_CONFIG_NO_EXCEPTIONS 0
#define nssv_CONFIG_NO_EXCEPTIONS 0
#else
// -fno-exceptions
#if !defined(nsel_CONFIG_NO_EXCEPTIONS)
#define nsel_CONFIG_NO_EXCEPTIONS 1
#endif

#define nssv_CONFIG_NO_EXCEPTIONS 1
#endif
//#include "../../src/nonstd/expected.hpp"
#include "nonstd/optional.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "token-type.hh"
#include "common-macros.inc"

// forward decl
namespace linb {
class any;
};

namespace tinyusdz {

namespace value {

// Identifier is the one used in USDA(Ascii)
// See: https://graphics.pixar.com/usd/release/api/_usd__page__datatypes.html
// NOTE: There are some TinyUSDZ specific types(e.g., short, char)
constexpr auto kToken = "token";
constexpr auto kString = "string";
constexpr auto kPath =
    "Path";  // generic(usually prim) path. internal representation.
constexpr auto kAssetPath = "asset";  // `asset` in USDA
constexpr auto kDictionary = "dictionary";
constexpr auto kTimeCode = "timecode";

constexpr auto kBool = "bool";
constexpr auto kChar = "char";
constexpr auto kChar2 = "char2";
constexpr auto kChar3 = "char3";
constexpr auto kChar4 = "char4";
constexpr auto kUChar = "uchar";
constexpr auto kUChar2 = "uchar2";
constexpr auto kUChar3 = "uchar3";
constexpr auto kUChar4 = "uchar4";
constexpr auto kHalf = "half";
constexpr auto kInt = "int";
constexpr auto kUInt = "uint";
constexpr auto kInt64 = "int64";
constexpr auto kUInt64 = "uint64";

constexpr auto kShort = "short";
constexpr auto kShort2 = "short2";
constexpr auto kShort3 = "short3";
constexpr auto kShort4 = "short4";

constexpr auto kUShort = "ushort";
constexpr auto kUShort2 = "ushort2";
constexpr auto kUShort3 = "ushort3";
constexpr auto kUShort4 = "ushort4";

constexpr auto kInt2 = "int2";
constexpr auto kInt3 = "int3";
constexpr auto kInt4 = "int4";

constexpr auto kUInt2 = "uint2";
constexpr auto kUInt3 = "uint3";
constexpr auto kUInt4 = "uint4";

constexpr auto kHalf2 = "half2";
constexpr auto kHalf3 = "half3";
constexpr auto kHalf4 = "half4";

// Seems primarily used in usdSkel.
// float precision matrix is not directly used in XformOp
constexpr auto kMatrix2f = "matrix2f";
constexpr auto kMatrix3f = "matrix3f";
constexpr auto kMatrix4f = "matrix4f";

constexpr auto kMatrix2d = "matrix2d";
constexpr auto kMatrix3d = "matrix3d";
constexpr auto kMatrix4d = "matrix4d";

constexpr auto kFloat = "float";
constexpr auto kFloat2 = "float2";
constexpr auto kFloat3 = "float3";
constexpr auto kFloat4 = "float4";

constexpr auto kDouble = "double";
constexpr auto kDouble2 = "double2";
constexpr auto kDouble3 = "double3";
constexpr auto kDouble4 = "double4";

constexpr auto kQuath = "quath";
constexpr auto kQuatf = "quatf";
constexpr auto kQuatd = "quatd";

constexpr auto kVector3h = "vector3h";
constexpr auto kVector3f = "vector3f";
constexpr auto kVector3d = "vector3d";

constexpr auto kVector4h = "vector4h";
constexpr auto kVector4f = "vector4f";
constexpr auto kVector4d = "vector4d";

constexpr auto kPoint3h = "point3h";
constexpr auto kPoint3f = "point3f";
constexpr auto kPoint3d = "point3d";

constexpr auto kNormal3h = "normal3h";
constexpr auto kNormal3f = "normal3f";
constexpr auto kNormal3d = "normal3d";

constexpr auto kColor3h = "color3h";
constexpr auto kColor3f = "color3f";
constexpr auto kColor3d = "color3d";
constexpr auto kColor4h = "color4h";
constexpr auto kColor4f = "color4f";
constexpr auto kColor4d = "color4d";

constexpr auto kFrame4d = "frame4d";

constexpr auto kTexCoord2h = "texCoord2h";
constexpr auto kTexCoord2f = "texCoord2f";
constexpr auto kTexCoord2d = "texCoord2d";

constexpr auto kTexCoord3h = "texCoord3h";
constexpr auto kTexCoord3f = "texCoord3f";
constexpr auto kTexCoord3d = "texCoord3d";

constexpr auto kTexCoord4h = "texCoord4h";
constexpr auto kTexCoord4f = "texCoord4f";
constexpr auto kTexCoord4d = "texCoord4d";

constexpr auto kRelationship = "rel";

inline std::string Add1DArraySuffix(const std::string &c) { return c + "[]"; }

using token = tinyusdz::Token;

// single or triple-quoted('"""' or ''') string
struct StringData {

  StringData() = default;
  StringData(const std::string &v) : value(v) {}
  StringData &operator=(const std::string &v) {
    value = v;
    return (*this);
  }

  std::string value;
  bool is_triple_quoted{false};
  bool single_quote{false};  // true for ', false for "

  // optional(for USDA)
  int line_row{0};
  int line_col{0};
};

// SdfAssetPath
class AssetPath {
 public:
  AssetPath() = default;
  AssetPath(const std::string &a) : asset_path_(a) {}
  AssetPath(const std::string &a, const std::string &r)
      : asset_path_(a), resolved_path_(r) {}

  bool Resolve() {
    // TODO;
    return false;
  }

  const std::string &GetAssetPath() const { return asset_path_; }

  const std::string GetResolvedPath() const { return resolved_path_; }

 private:
  std::string asset_path_;
  std::string resolved_path_;
};

class TimeCode {
 public:
  TimeCode(const double d) : time_(d) {}

  static constexpr double Default() {
    // Return qNaN. same in pxrUSD
    return std::numeric_limits<double>::quiet_NaN();
  }

  // TODO: Deprecate `Get` and use `get`
  double Get(bool *is_default_timecode) {
    if (is_default_timecode) {
      (*is_default_timecode) = is_default();
    }
    return time_;
  }

  double get(bool *is_default_timecode) {
    if (is_default_timecode) {
      (*is_default_timecode) = is_default();
    }
    return time_;
  }

  bool is_default() {
    // TODO: Bitwise comparison
    return std::isnan(time_);
  }

 private:
  double time_{std::numeric_limits<double>::quiet_NaN()};
};

static_assert(sizeof(TimeCode) == 8, "Size of TimeCode must be 8.");

//
// Type ID for TypeTraits<T>::type_id.
//
// These type IDs are internally used and can be changed arbitrary.
//
// These ID assignment won't affect Crate binary serialization.
// (See `crate-format.hh` for Type ID used in Crate binary)
//
// TODO(syoyo): Support 3D and 4D?
constexpr uint32_t TYPE_ID_1D_ARRAY_BIT = 1 << 20;  // 1024
// constexpr uint32_t TYPE_ID_2D_ARRAY_BIT = 1 << 21;  // 2048
//  constexpr uint32_t TYPE_ID_3D_ARRAY_BIT = 1 << 22;
//  constexpr uint32_t TYPE_ID_4D_ARRAY_BIT = 1 << 23;
constexpr uint32_t TYPE_ID_TERMINATOR_BIT = 1 << 24;

enum TypeId {
  TYPE_ID_INVALID,  // = 0
  TYPE_ID_NULL,
  TYPE_ID_VOID,
  TYPE_ID_MONOSTATE,
  TYPE_ID_VALUEBLOCK,  // Value block. `None` in ascii.

  // -- begin value type
  TYPE_ID_VALUE_BEGIN,
  
  TYPE_ID_TOKEN,
  TYPE_ID_STRING,
  TYPE_ID_STRING_DATA,  // String for primvar and metadata. Includes multi-line
                        // string

  TYPE_ID_BOOL,

  TYPE_ID_CHAR,
  TYPE_ID_CHAR2,
  TYPE_ID_CHAR3,
  TYPE_ID_CHAR4,

  // TYPE_ID_INT8,
  TYPE_ID_HALF,
  TYPE_ID_INT32,
  TYPE_ID_INT64,

  TYPE_ID_HALF2,
  TYPE_ID_HALF3,
  TYPE_ID_HALF4,

  TYPE_ID_INT2,  // int32 x 2
  TYPE_ID_INT3,
  TYPE_ID_INT4,

  TYPE_ID_UCHAR,  // uint8
  TYPE_ID_UCHAR2,
  TYPE_ID_UCHAR3,
  TYPE_ID_UCHAR4,

  TYPE_ID_UINT32,
  TYPE_ID_UINT64,

  TYPE_ID_SHORT,
  TYPE_ID_SHORT2,
  TYPE_ID_SHORT3,
  TYPE_ID_SHORT4,

  TYPE_ID_USHORT,
  TYPE_ID_USHORT2,
  TYPE_ID_USHORT3,
  TYPE_ID_USHORT4,

  TYPE_ID_UINT2,
  TYPE_ID_UINT3,
  TYPE_ID_UINT4,

  TYPE_ID_FLOAT,
  TYPE_ID_FLOAT2,
  TYPE_ID_FLOAT3,
  TYPE_ID_FLOAT4,

  TYPE_ID_DOUBLE,
  TYPE_ID_DOUBLE2,
  TYPE_ID_DOUBLE3,
  TYPE_ID_DOUBLE4,

  TYPE_ID_QUATH,
  TYPE_ID_QUATF,
  TYPE_ID_QUATD,

  TYPE_ID_MATRIX2F,
  TYPE_ID_MATRIX3F,
  TYPE_ID_MATRIX4F,

  TYPE_ID_MATRIX2D,
  TYPE_ID_MATRIX3D,
  TYPE_ID_MATRIX4D,

  TYPE_ID_COLOR3H,
  TYPE_ID_COLOR3F,
  TYPE_ID_COLOR3D,

  TYPE_ID_COLOR4H,
  TYPE_ID_COLOR4F,
  TYPE_ID_COLOR4D,

  TYPE_ID_POINT3H,
  TYPE_ID_POINT3F,
  TYPE_ID_POINT3D,

  TYPE_ID_NORMAL3H,
  TYPE_ID_NORMAL3F,
  TYPE_ID_NORMAL3D,

  TYPE_ID_VECTOR3H,
  TYPE_ID_VECTOR3F,
  TYPE_ID_VECTOR3D,

  TYPE_ID_FRAME4D,

  TYPE_ID_TEXCOORD2H,
  TYPE_ID_TEXCOORD2F,
  TYPE_ID_TEXCOORD2D,

  TYPE_ID_TEXCOORD3H,
  TYPE_ID_TEXCOORD3F,
  TYPE_ID_TEXCOORD3D,

  TYPE_ID_EXTENT,  // float3[2]

  TYPE_ID_TIMECODE,

  // TYPE_ID_ASSET,
  TYPE_ID_ASSET_PATH,

  TYPE_ID_DICT,        // Generic dict type. TODO: remove?
  TYPE_ID_CUSTOMDATA,  // similar to `dictionary`, but limited types are allowed
                       // to use. for metadatum(e.g. `customData` in Prim Meta)
                       
  TYPE_ID_VALUE_END,

  // -- end value type
  
  TYPE_ID_LAYER_OFFSET,
  TYPE_ID_PAYLOAD,

  // Types in prim-types.hh
  TYPE_ID_REFERENCE,
  TYPE_ID_SPECIFIER,
  TYPE_ID_PERMISSION,
  TYPE_ID_VARIABILITY,
  TYPE_ID_LIST_OP_TOKEN,
  TYPE_ID_LIST_OP_STRING,
  TYPE_ID_LIST_OP_PATH,
  TYPE_ID_LIST_OP_REFERENCE,
  TYPE_ID_LIST_OP_INT,
  TYPE_ID_LIST_OP_INT64,
  TYPE_ID_LIST_OP_UINT,
  TYPE_ID_LIST_OP_UINT64,
  TYPE_ID_LIST_OP_PAYLOAD,

  TYPE_ID_PATH,
  TYPE_ID_PATH_VECTOR,
  TYPE_ID_TOKEN_VECTOR,
  TYPE_ID_RELATIONSHIP,

  // -- end of base type for Property.

  TYPE_ID_TIMESAMPLES,
  TYPE_ID_VARIANT_SELECION_MAP,

  // Types in crate-format.hh
  TYPE_ID_CRATE_BEGIN = 256,
  TYPE_ID_CRATE_VALUE,
  TYPE_ID_CRATE_UNREGISTERED_VALUE,
  TYPE_ID_CRATE_LIST_OP_UNREGISTERED_VALUE,
  TYPE_ID_CRATE_END,

  // Types for Model and GPrim
  TYPE_ID_MODEL_BEGIN = (1 << 10),
  TYPE_ID_MODEL,  // internally used class
  // TYPE_ID_GROUP,
  TYPE_ID_SCOPE,
  TYPE_ID_GPRIM,
  TYPE_ID_GEOM_XFORM,
  TYPE_ID_GEOM_MESH,
  TYPE_ID_GEOM_BASIS_CURVES,
  TYPE_ID_GEOM_NURBS_CURVES,
  TYPE_ID_GEOM_SPHERE,
  TYPE_ID_GEOM_CUBE,
  TYPE_ID_GEOM_CYLINDER,
  TYPE_ID_GEOM_CONE,
  TYPE_ID_GEOM_CAPSULE,
  TYPE_ID_GEOM_POINTS,
  TYPE_ID_GEOM_GEOMSUBSET,
  TYPE_ID_GEOM_POINT_INSTANCER,
  TYPE_ID_GEOM_CAMERA,
  TYPE_ID_GEOM_END,

  // Types for usdLux
  TYPE_ID_LUX_BEGIN = (1 << 10) + (1 << 9),
  TYPE_ID_LUX_SPHERE,
  TYPE_ID_LUX_DOME,
  TYPE_ID_LUX_CYLINDER,
  TYPE_ID_LUX_DISK,
  TYPE_ID_LUX_RECT,
  TYPE_ID_LUX_DISTANT,
  TYPE_ID_LUX_GEOMETRY,
  TYPE_ID_LUX_PORTAL,
  TYPE_ID_LUX_PLUGIN,
  TYPE_ID_LUX_END,

  // Types for usdShader
  TYPE_ID_SHADER_BEGIN = 1 << 11,
  TYPE_ID_SHADER,
  TYPE_ID_MATERIAL,
  TYPE_ID_NODEGRAPH,
  TYPE_ID_SHADER_END,

  // Types for usdImaging and usdMtlx
  // See <pxrUSD>/pxr/usdImaging/usdImaging/tokens.h
  TYPE_ID_IMAGING_BEGIN = (1 << 11) + (1 << 10),
  TYPE_ID_IMAGING_SHADER_NODE,
  TYPE_ID_IMAGING_PREVIEWSURFACE,
  TYPE_ID_IMAGING_UVTEXTURE,
  TYPE_ID_IMAGING_PRIMVAR_READER_FLOAT,
  TYPE_ID_IMAGING_PRIMVAR_READER_FLOAT2,
  TYPE_ID_IMAGING_PRIMVAR_READER_FLOAT3,
  TYPE_ID_IMAGING_PRIMVAR_READER_FLOAT4,
  TYPE_ID_IMAGING_PRIMVAR_READER_INT,
  TYPE_ID_IMAGING_PRIMVAR_READER_STRING,
  TYPE_ID_IMAGING_PRIMVAR_READER_NORMAL, // float3
  TYPE_ID_IMAGING_PRIMVAR_READER_POINT, // float3
  TYPE_ID_IMAGING_PRIMVAR_READER_VECTOR, // float3
  TYPE_ID_IMAGING_PRIMVAR_READER_MATRIX, // float3
  TYPE_ID_IMAGING_TRANSFORM_2D,

  TYPE_ID_IMAGING_MTLX_PREVIEWSURFACE,
  TYPE_ID_IMAGING_MTLX_STANDARDSURFACE,

  TYPE_ID_IMAGING_END,

  // Types for usdVol
  TYPE_ID_VOL_BEGIN = 1 << 12,
  TYPE_ID_VOL_END,

  // Types for usdSkel
  TYPE_ID_SKEL_BEGIN = 1 << 13,
  TYPE_ID_SKEL_ROOT,
  TYPE_ID_SKELETON,
  TYPE_ID_SKELANIMATION,
  TYPE_ID_BLENDSHAPE,
  TYPE_ID_SKEL_END,

  TYPE_ID_MODEL_END,

  
  // Types for API
  TYPE_ID_API_BEGIN = 1 << 14,
  TYPE_ID_COLLECTION,
  TYPE_ID_COLLECTION_INSTANCE,
  TYPE_ID_MATERIAL_BINDING,
  TYPE_ID_API_END,

  // Base ID for user data type(less than `TYPE_ID_1D_ARRAY_BIT-1`)
  TYPE_ID_USER_BEGIN = 1 << 16,

  TYPE_ID_ALL = (TYPE_ID_TERMINATOR_BIT - 1)  // terminator.
};

struct timecode {
  double value;
};

struct half {
  uint16_t value;
};

using half2 = std::array<half, 2>;
using half3 = std::array<half, 3>;
using half4 = std::array<half, 4>;

float half_to_float(value::half h);
half float_to_half_full(float f);

inline half operator+(const half &a, const half &b) {
  return float_to_half_full(half_to_float(a) + half_to_float(b));
}

inline half operator-(const half &a, const half &b) {
  return float_to_half_full(half_to_float(a) - half_to_float(b));
}

inline half operator*(const half &a, const half &b) {
  return float_to_half_full(half_to_float(a) * half_to_float(b));
}

// TODO: save div
inline half operator/(const half &a, const half &b) {
  return float_to_half_full(half_to_float(a) / half_to_float(b));
}

inline half& operator+=(half &a, const half &b) {
  a = float_to_half_full(half_to_float(a) + half_to_float(b));
  return a;
}

inline half& operator-=(half &a, const half &b) {
  a = float_to_half_full(half_to_float(a) - half_to_float(b));
  return a;
}

inline half& operator*=(half &a, const half &b) {
  a = float_to_half_full(half_to_float(a) * half_to_float(b));
  return a;
}

// TODO: save div
inline half& operator/=(half &a, const half &b) {
  a = float_to_half_full(half_to_float(a) / half_to_float(b));
  return a;
}

inline half operator+(const half &a, float b) {
  return float_to_half_full(half_to_float(a) + b);
}

inline half operator-(const half &a, float b) {
  return float_to_half_full(half_to_float(a) - b);
}

inline half operator*(const half &a, float b) {
  return float_to_half_full(half_to_float(a) * b);
}

inline half operator/(const half &a, float b) {
  return float_to_half_full(half_to_float(a) / b);
}

inline half operator+(float a, const half &b) {
  return float_to_half_full(a + half_to_float(b));
}

inline half operator-(float a, const half &b) {
  return float_to_half_full(a - half_to_float(b));
}

inline half operator*(float a, const half &b) {
  return float_to_half_full(a * half_to_float(b));
}

inline half operator/(float a, const half &b) {
  return float_to_half_full(a / half_to_float(b));
}

using char2 = std::array<char, 2>;
using char3 = std::array<char, 3>;
using char4 = std::array<char, 4>;

using uchar2 = std::array<uint8_t, 2>;
using uchar3 = std::array<uint8_t, 3>;
using uchar4 = std::array<uint8_t, 4>;

using short2 = std::array<int16_t, 2>;
using short3 = std::array<int16_t, 3>;
using short4 = std::array<int16_t, 4>;

using ushort2 = std::array<uint16_t, 2>;
using ushort3 = std::array<uint16_t, 3>;
using ushort4 = std::array<uint16_t, 4>;

using int2 = std::array<int32_t, 2>;
using int3 = std::array<int32_t, 3>;
using int4 = std::array<int32_t, 4>;

using uint2 = std::array<uint32_t, 2>;
using uint3 = std::array<uint32_t, 3>;
using uint4 = std::array<uint32_t, 4>;

using float2 = std::array<float, 2>;
using float3 = std::array<float, 3>;
using float4 = std::array<float, 4>;

using double2 = std::array<double, 2>;
using double3 = std::array<double, 3>;
using double4 = std::array<double, 4>;

//
// Matrix is represented as row-major order as done in pxrUSD.
// m[i][j] is read as: i'th row, j'th column
// memory layout is same both for column-major and row-major.
// (e.g. m[3][0], m[3][1], m[3][2] or a[13], a[14], a[15] are translation components for 4x4 matrix)
//

struct matrix2d;
struct matrix3d;
struct matrix4d;

struct matrix2f {
  matrix2f() {
    m[0][0] = 1.0f;
    m[0][1] = 0.0f;

    m[1][0] = 0.0f;
    m[1][1] = 1.0f;
  }

  matrix2f(const std::array<float, 4> &arr) {
    m[0][0] = arr[0];
    m[0][1] = arr[1];
    m[1][0] = arr[2];
    m[1][1] = arr[3];
  }

  inline void set_row(uint32_t row, float x, float y) {
    if (row < 2) {
      m[row][0] = x;
      m[row][1] = y;
    }
  }

  inline void set_scale(float sx, float sy) {
    m[0][0] = sx;
    m[0][1] = 0.0f;

    m[1][0] = 0.0f;
    m[1][1] = sy;
  }

  static matrix2f identity() {
    matrix2f m;

    m.m[0][0] = 1.0f;
    m.m[0][1] = 0.0f;

    m.m[1][0] = 0.0f;
    m.m[1][1] = 1.0f;

    return m;
  }

  matrix2f(const matrix2d &rhs);
  matrix2f &operator=(const matrix2d &rhs);
  
  float m[2][2];
};

struct matrix3f {
  matrix3f() {
    m[0][0] = 1.0f;
    m[0][1] = 0.0f;
    m[0][2] = 0.0f;

    m[1][0] = 0.0f;
    m[1][1] = 1.0f;
    m[1][2] = 0.0f;

    m[2][0] = 0.0f;
    m[2][1] = 0.0f;
    m[2][2] = 1.0f;
  }

  matrix3f(const std::array<float, 9> &arr) {
    m[0][0] = arr[0];
    m[0][1] = arr[1];
    m[0][2] = arr[2];
    m[1][0] = arr[3];
    m[1][1] = arr[4];
    m[1][2] = arr[5];
    m[2][0] = arr[6];
    m[2][1] = arr[7];
    m[2][2] = arr[8];
  }

  inline void set_row(uint32_t row, float x, float y, float z) {
    if (row < 3) {
      m[row][0] = x;
      m[row][1] = y;
      m[row][2] = z;
    }
  }

  inline void set_scale(float sx, float sy, float sz) {
    m[0][0] = sx;
    m[0][1] = 0.0f;
    m[0][2] = 0.0f;

    m[1][0] = 0.0f;
    m[1][1] = sy;
    m[1][2] = 0.0f;

    m[2][0] = 0.0f;
    m[2][1] = 0.0f;
    m[2][2] = sz;

  }

  inline void set_translation(float tx, float ty, float tz) {
    m[2][0] = tx;
    m[2][1] = ty;
    m[2][2] = tz;
  }

  static matrix3f identity() {
    matrix3f m;

    m.m[0][0] = 1.0f;
    m.m[0][1] = 0.0f;
    m.m[0][2] = 0.0f;

    m.m[1][0] = 0.0f;
    m.m[1][1] = 1.0f;
    m.m[1][2] = 0.0f;

    m.m[2][0] = 0.0f;
    m.m[2][1] = 0.0f;
    m.m[2][2] = 1.0f;

    return m;
  }

  matrix3f(const matrix3d &rhs);
  matrix3f &operator=(const matrix3d &rhs);

  float m[3][3];
};

struct matrix4f {
  matrix4f() {
    m[0][0] = 1.0f;
    m[0][1] = 0.0f;
    m[0][2] = 0.0f;
    m[0][3] = 0.0f;

    m[1][0] = 0.0f;
    m[1][1] = 1.0f;
    m[1][2] = 0.0f;
    m[1][3] = 0.0f;

    m[2][0] = 0.0f;
    m[2][1] = 0.0f;
    m[2][2] = 1.0f;
    m[2][3] = 0.0f;

    m[3][0] = 0.0f;
    m[3][1] = 0.0f;
    m[3][2] = 0.0f;
    m[3][3] = 1.0f;
  }

  matrix4f(const std::array<float, 16> &arr) {
    m[0][0] = arr[0];
    m[0][1] = arr[1];
    m[0][2] = arr[2];
    m[0][3] = arr[3];
    m[1][0] = arr[4];
    m[1][1] = arr[5];
    m[1][2] = arr[6];
    m[1][3] = arr[7];
    m[2][0] = arr[8];
    m[2][1] = arr[9];
    m[2][2] = arr[10];
    m[2][3] = arr[11];
    m[3][0] = arr[12];
    m[3][1] = arr[13];
    m[3][2] = arr[14];
    m[3][3] = arr[15];
  }

  inline void set_row(uint32_t row, float x, float y, float z, float w) {
    if (row < 4) {
      m[row][0] = x;
      m[row][1] = y;
      m[row][2] = z;
      m[row][3] = w;
    }
  }

  inline void set_scale(float sx, float sy, float sz) {
    m[0][0] = sx;
    m[0][1] = 0.0f;
    m[0][2] = 0.0f;
    m[0][3] = 0.0f;

    m[1][0] = 0.0f;
    m[1][1] = sy;
    m[1][2] = 0.0f;
    m[1][3] = 0.0f;

    m[2][0] = 0.0f;
    m[2][1] = 0.0f;
    m[2][2] = sz;
    m[2][3] = 0.0f; 

    m[3][0] = 0.0f;
    m[3][1] = 0.0f;
    m[3][2] = 0.0f;
    m[3][3] = 1.0f;
  }

  inline void set_translation(float tx, float ty, float tz) {
    m[3][0] = tx;
    m[3][1] = ty;
    m[3][2] = tz;
  }

  static matrix4f identity() {
    matrix4f m;

    m.m[0][0] = 1.0f;
    m.m[0][1] = 0.0f;
    m.m[0][2] = 0.0f;
    m.m[0][3] = 0.0f;

    m.m[1][0] = 0.0f;
    m.m[1][1] = 1.0f;
    m.m[1][2] = 0.0f;
    m.m[1][3] = 0.0f;

    m.m[2][0] = 0.0f;
    m.m[2][1] = 0.0f;
    m.m[2][2] = 1.0f;
    m.m[2][3] = 0.0f;

    m.m[3][0] = 0.0f;
    m.m[3][1] = 0.0f;
    m.m[3][2] = 0.0f;
    m.m[3][3] = 1.0f;

    return m;
  }

  matrix4f(const matrix4d &rhs);

  matrix4f &operator=(const matrix4d &rhs);

  float m[4][4];
};

struct matrix2d {
  matrix2d() {
    m[0][0] = 1.0;
    m[0][1] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = 1.0;
  }

  matrix2d(const std::array<double, 4> &arr) {
    m[0][0] = arr[0];
    m[0][1] = arr[1];
    m[1][0] = arr[2];
    m[1][1] = arr[3];
  }

  inline void set_row(uint32_t row, double x, double y) {
    if (row < 2) {
      m[row][0] = x;
      m[row][1] = y;
    }
  }

  inline void set_scale(double sx, double sy) {
    m[0][0] = sx;
    m[0][1] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = sy;
  }

  static matrix2d identity() {
    matrix2d m;

    m.m[0][0] = 1.0;
    m.m[0][1] = 0.0;

    m.m[1][0] = 0.0;
    m.m[1][1] = 1.0;

    return m;
  }

  matrix2d &operator=(const matrix2f &rhs);

  double m[2][2];
};

struct matrix3d {
  matrix3d() {
    m[0][0] = 1.0;
    m[0][1] = 0.0;
    m[0][2] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = 1.0;
    m[1][2] = 0.0;

    m[2][0] = 0.0;
    m[2][1] = 0.0;
    m[2][2] = 1.0;
  }

  matrix3d(const std::array<double, 9> &arr) {
    m[0][0] = arr[0];
    m[0][1] = arr[1];
    m[0][2] = arr[2];
    m[1][0] = arr[3];
    m[1][1] = arr[4];
    m[1][2] = arr[5];
    m[2][0] = arr[6];
    m[2][1] = arr[7];
    m[2][2] = arr[8];
  }

  inline void set_row(uint32_t row, double x, double y, double z) {
    if (row < 3) {
      m[row][0] = x;
      m[row][1] = y;
      m[row][2] = z;
    }
  }

  inline void set_scale(double sx, double sy, double sz) {
    m[0][0] = sx;
    m[0][1] = 0.0;
    m[0][2] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = sy;
    m[1][2] = 0.0;

    m[2][0] = 0.0;
    m[2][1] = 0.0;
    m[2][2] = sz;
  }

  static matrix3d identity() {
    matrix3d m;

    m.m[0][0] = 1.0;
    m.m[0][1] = 0.0;
    m.m[0][2] = 0.0;

    m.m[1][0] = 0.0;
    m.m[1][1] = 1.0;
    m.m[1][2] = 0.0;

    m.m[2][0] = 0.0;
    m.m[2][1] = 0.0;
    m.m[2][2] = 1.0;

    return m;
  }

  matrix3d &operator=(const matrix3f &rhs);

  double m[3][3];
};

struct matrix4d {
  matrix4d() {
    m[0][0] = 1.0;
    m[0][1] = 0.0;
    m[0][2] = 0.0;
    m[0][3] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = 1.0;
    m[1][2] = 0.0;
    m[1][3] = 0.0;

    m[2][0] = 0.0;
    m[2][1] = 0.0;
    m[2][2] = 1.0;
    m[2][3] = 0.0;

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;
  }

  matrix4d(const std::array<double, 16> &arr) {
    m[0][0] = arr[0];
    m[0][1] = arr[1];
    m[0][2] = arr[2];
    m[0][3] = arr[3];
    m[1][0] = arr[4];
    m[1][1] = arr[5];
    m[1][2] = arr[6];
    m[1][3] = arr[7];
    m[2][0] = arr[8];
    m[2][1] = arr[9];
    m[2][2] = arr[10];
    m[2][3] = arr[11];
    m[3][0] = arr[12];
    m[3][1] = arr[13];
    m[3][2] = arr[14];
    m[3][3] = arr[15];
  }

  inline void set_row(uint32_t row, double x, double y, double z, double w) {
    if (row < 4) {
      m[row][0] = x;
      m[row][1] = y;
      m[row][2] = z;
      m[row][3] = w;
    }
  }

  inline void set_scale(double sx, double sy, double sz) {
    m[0][0] = sx;
    m[0][1] = 0.0;
    m[0][2] = 0.0;
    m[0][3] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = sy;
    m[1][2] = 0.0;
    m[1][3] = 0.0;

    m[2][0] = 0.0;
    m[2][1] = 0.0;
    m[2][2] = sz;
    m[2][3] = 0.0; 

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;
  }

  static matrix4d identity() {
    matrix4d m;

    m.m[0][0] = 1.0;
    m.m[0][1] = 0.0;
    m.m[0][2] = 0.0;
    m.m[0][3] = 0.0;

    m.m[1][0] = 0.0;
    m.m[1][1] = 1.0;
    m.m[1][2] = 0.0;
    m.m[1][3] = 0.0;

    m.m[2][0] = 0.0;
    m.m[2][1] = 0.0;
    m.m[2][2] = 1.0;
    m.m[2][3] = 0.0;

    m.m[3][0] = 0.0;
    m.m[3][1] = 0.0;
    m.m[3][2] = 0.0;
    m.m[3][3] = 1.0;

    return m;
  }

  matrix4d &operator=(const matrix4f &rhs);

  double m[4][4];
};

// = matrix4d
struct frame4d {
  frame4d() {
    m[0][0] = 1.0;
    m[0][1] = 0.0;
    m[0][2] = 0.0;
    m[0][3] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = 1.0;
    m[1][2] = 0.0;
    m[1][3] = 0.0;

    m[2][0] = 0.0;
    m[2][1] = 0.0;
    m[2][2] = 1.0;
    m[2][3] = 0.0;

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;
  }
  double m[4][4];
};

// ret = m x n in row-major(n x m in column-major)
// i.e. You can express TRS transform as
//
// p * S * R * T = p'
// p' = Mult(Mult(S, R), T)
// 
// you can express world matrix as
//
// node.world = parent.world * node.local
//            = Mult(parent.world, node.local)
template <typename MTy, typename STy, size_t N>
MTy Mult(const MTy &m, const MTy &n) {
  MTy ret;
  //memset(ret.m, 0, sizeof(MTy)); 

  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < N; i++) {
      STy value = static_cast<STy>(0);
      for (size_t k = 0; k < N; k++) {
        value += m.m[j][k] * n.m[k][i];
      }
      ret.m[j][i] = value;
    }
  }

  return ret;
}

#if 0
// Deprecated.
// TODO: remove column-major functions.
template <typename MTy, typename STy, size_t N>
MTy MultColumnMajor(const MTy &m, const MTy &n) {
  MTy ret;
  memset(ret.m, 0, sizeof(MTy));

  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < N; i++) {
      STy value = static_cast<STy>(0);
      for (size_t k = 0; k < N; k++) {
        value += m.m[j][k] * n.m[k][i];
      }
      ret.m[j][i] = value;
    }
  }

  return ret;
}
#endif

// ret = matrix x vector
// Assume matrixN >= vecN
template <typename MTy, typename VTy, typename MBaseTy, typename VBaseTy, size_t N>
VTy MultV(const MTy &m, const VTy &v) {
  // MBaseTy must be float or double
  // TODO: use std::enable_if?
  static_assert(std::is_same<MBaseTy, double>::value || std::is_same<MBaseTy, float>::value,
    "Matrix element type must be `float` or `double`");

  // Intermediate type. Choose higher precision based on its size.
  typedef typename std::conditional<sizeof(MBaseTy) >= sizeof(VBaseTy), MBaseTy, VBaseTy>::type Ty;

  VTy ret;

  for (size_t j = 0; j < N; j++) {
    Ty value = static_cast<Ty>(0);
    for (size_t i = 0; i < N; i++) {
      value += static_cast<Ty>(m.m[i][j]) * static_cast<Ty>(v[i]);
    }
    ret[j] = static_cast<VBaseTy>(value);
  }

  return ret;
}

template <typename MTy, typename STy, size_t N>
MTy MatAdd(const MTy &m, const MTy &n) {
  MTy ret;
  memset(ret.m, 0, sizeof(MTy));

  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < N; i++) {
      ret.m[j][i] = m.m[j][i] + n.m[j][i];
    }
  }

  return ret;
}

template <typename MTy, typename STy, size_t N>
MTy MatSub(const MTy &m, const MTy &n) {
  MTy ret;
  memset(ret.m, 0, sizeof(MTy));

  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < N; i++) {
      ret.m[j][i] = m.m[j][i] - n.m[j][i];
    }
  }

  return ret;
}

// TODO: division

inline matrix2f operator+(const matrix2f &a, const matrix2f &b) {
  matrix2f ret = MatAdd<matrix2f, float, 2>(a, b);
  return ret;
}

inline matrix2f operator-(const matrix2f &a, const matrix2f &b) {
  matrix2f ret = MatSub<matrix2f, float, 2>(a, b);
  return ret;
}

inline matrix2f operator*(const matrix2f &a, const matrix2f &b) {
  matrix2f ret = Mult<matrix2f, float, 2>(a, b);
  return ret;
}

inline matrix3f operator+(const matrix3f &a, const matrix3f &b) {
  matrix3f ret = MatAdd<matrix3f, float, 3>(a, b);
  return ret;
}

inline matrix3f operator-(const matrix3f &a, const matrix3f &b) {
  matrix3f ret = MatSub<matrix3f, float, 3>(a, b);
  return ret;
}

inline matrix3f operator*(const matrix3f &a, const matrix3f &b) {
  matrix3f ret = Mult<matrix3f, float, 3>(a, b);
  return ret;
}

inline matrix4f operator+(const matrix4f &a, const matrix4f &b) {
  matrix4f ret = MatAdd<matrix4f, float, 4>(a, b);
  return ret;
}

inline matrix4f operator-(const matrix4f &a, const matrix4f &b) {
  matrix4f ret = MatSub<matrix4f, float, 4>(a, b);
  return ret;
}

inline matrix4f operator*(const matrix4f &a, const matrix4f &b) {
  matrix4f ret = Mult<matrix4f, float, 4>(a, b);
  return ret;
}

inline matrix2d operator+(const matrix2d &a, const matrix2d &b) {
  matrix2d ret = MatAdd<matrix2d, double, 2>(a, b);
  return ret;
}

inline matrix2d operator-(const matrix2d &a, const matrix2d &b) {
  matrix2d ret = MatSub<matrix2d, double, 2>(a, b);
  return ret;
}

inline matrix2d operator*(const matrix2d &a, const matrix2d &b) {
  matrix2d ret = Mult<matrix2d, double, 2>(a, b);
  return ret;
}

inline matrix3d operator+(const matrix3d &a, const matrix3d &b) {
  matrix3d ret = MatAdd<matrix3d, double, 3>(a, b);
  return ret;
}

inline matrix3d operator-(const matrix3d &a, const matrix3d &b) {
  matrix3d ret = MatSub<matrix3d, double, 3>(a, b);
  return ret;
}

inline matrix3d operator*(const matrix3d &a, const matrix3d &b) {
  matrix3d ret = Mult<matrix3d, double, 3>(a, b);
  return ret;
}

inline matrix4d operator+(const matrix4d &a, const matrix4d &b) {
  matrix4d ret = MatAdd<matrix4d, double, 4>(a, b);
  return ret;
}

inline matrix4d operator-(const matrix4d &a, const matrix4d &b) {
  matrix4d ret = MatSub<matrix4d, double, 4>(a, b);
  return ret;
}

inline matrix4d operator*(const matrix4d &a, const matrix4d &b) {
  matrix4d ret = Mult<matrix4d, double, 4>(a, b);
  return ret;
}

// Quaternion has memory layout of [x, y, z, w] in Crate(Binary)
// and QfQuat class in pxrUSD.
// https://github.com/PixarAnimationStudios/USD/blob/3abc46452b1271df7650e9948fef9f0ce602e3b2/pxr/base/gf/quatf.h#L287
// NOTE: ASCII uses [w, x, y, z] ordering
struct quath {
  half3 imag;
  half real;
  half operator[](size_t idx) const { return *(&imag[0] + idx); }
  half &operator[](size_t idx) { return *(&imag[0] + idx); }
};

struct quatf {
  float3 imag;
  float real;
  float operator[](size_t idx) const { return *(&imag[0] + idx); }
  float &operator[](size_t idx) { return *(&imag[0] + idx); }
};

struct quatd {
  double3 imag;
  double real;
  double operator[](size_t idx) const { return *(&imag[0] + idx); }
  double &operator[](size_t idx) { return *(&imag[0] + idx); }
};

struct vector3h {
  half x, y, z;

  half operator[](size_t idx) const { return *(&x + idx); }
  half &operator[](size_t idx) { return *(&x + idx); }
};

struct vector3f {
  float x, y, z;

  float operator[](size_t idx) const { return *(&x + idx); }
  float &operator[](size_t idx) { return *(&x + idx); }
};

struct vector3d {
  double x, y, z;

  double operator[](size_t idx) const { return *(&x + idx); }
  double &operator[](size_t idx) { return *(&x + idx); }
};

struct normal3h {
  half x, y, z;

  half operator[](size_t idx) const { return *(&x + idx); }
  half &operator[](size_t idx) { return *(&x + idx); }
};

struct normal3f {
  float x, y, z;

  float operator[](size_t idx) const { return *(&x + idx); }
  float &operator[](size_t idx) { return *(&x + idx); }
};

struct normal3d {
  double x, y, z;

  double operator[](size_t idx) const { return *(&x + idx); }
  double &operator[](size_t idx) { return *(&x + idx); }
};

struct point3h {
  half x, y, z;

  half operator[](size_t idx) const { return *(&x + idx); }
  half &operator[](size_t idx) { return *(&x + idx); }
};

#if 0 // move to value-eval-util.hh

inline point3h operator+(const float a, const point3h &b) {
  return {a + b.x, a + b.y, a + b.z};
}

inline point3h operator-(const float a, const point3h &b) {
  return {a - b.x, a - b.y, a - b.z};
}

inline point3h operator*(const float a, const point3h &b) {
  return {a * b.x, a * b.y, a * b.z};
}

// TODO: safe div
inline point3h operator/(const float a, const point3h &b) {
  return {a / b.x, a / b.y, a / b.z};
}

inline point3h operator+(const double a, const point3h &b) {
  return {float(a) + b.x, float(a) + b.y, float(a) + b.z};
}

inline point3h operator-(const double a, const point3h &b) {
  return {float(a) - b.x, float(a) - b.y, float(a) - b.z};
}

inline point3h operator*(const double a, const point3h &b) {
  return {float(a) * b.x, float(a) * b.y, float(a) * b.z};
}

inline point3h operator/(const double a, const point3h &b) {
  return {float(a) / b.x, float(a) / b.y, float(a) / b.z};
}

inline point3h operator+(const point3h &a, const float b) {
  return {a.x + b, a.y + b, a.z + b};
}

inline point3h operator-(const point3h &a, const float b) {
  return {a.x - b, a.y - b, a.z - b};
}

inline point3h operator*(const point3h &a, const float b) {
  return {a.x * b, a.y * b, a.z * b};
}

inline point3h operator/(const point3h &a, const float b) {
  return {a.x / b, a.y / b, a.z / b};
}

inline point3h operator+(const point3h &a, const double b) {
  return {a.x + float(b), a.y + float(b), a.z + float(b)};
}

inline point3h operator-(const point3h &a, const double b) {
  return {a.x - float(b), a.y - float(b), a.z - float(b)};
}

inline point3h operator*(const point3h &a, const double b) {
  return {a.x * float(b), a.y * float(b), a.z * float(b)};
}

inline point3h operator/(const point3h &a, const double b) {
  return {a.x / float(b), a.y / float(b), a.z / float(b)};
}

inline point3h operator+(const point3h &a, const point3h &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline point3h operator-(const point3h &a, const point3h &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline point3h operator*(const point3h &a, const point3h &b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline point3h operator/(const point3h &a, const point3h &b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}
#endif

struct point3f {
  float x, y, z;

  float operator[](size_t idx) const { return *(&x + idx); }
  float &operator[](size_t idx) { return *(&x + idx); }
};

#if 0
inline point3f operator+(const float a, const point3f &b) {
  return {a + b.x, a + b.y, a + b.z};
}

inline point3f operator-(const float a, const point3f &b) {
  return {a - b.x, a - b.y, a - b.z};
}

inline point3f operator*(const float a, const point3f &b) {
  return {a * b.x, a * b.y, a * b.z};
}

// TODO: safe div
inline point3f operator/(const float a, const point3f &b) {
  return {a / b.x, a / b.y, a / b.z};
}

inline point3f operator+(const double a, const point3f &b) {
  return {float(a) + b.x, float(a) + b.y, float(a) + b.z};
}

inline point3f operator-(const double a, const point3f &b) {
  return {float(a) - b.x, float(a) - b.y, float(a) - b.z};
}

inline point3f operator*(const double a, const point3f &b) {
  return {float(a) * b.x, float(a) * b.y, float(a) * b.z};
}

inline point3f operator/(const double a, const point3f &b) {
  return {float(a) / b.x, float(a) / b.y, float(a) / b.z};
}

inline point3f operator+(const point3f &a, const float b) {
  return {a.x + b, a.y + b, a.z + b};
}

inline point3f operator-(const point3f &a, const float b) {
  return {a.x - b, a.y - b, a.z - b};
}

inline point3f operator*(const point3f &a, const float b) {
  return {a.x * b, a.y * b, a.z * b};
}

inline point3f operator/(const point3f &a, const float b) {
  return {a.x / b, a.y / b, a.z / b};
}

inline point3f operator+(const point3f &a, const double b) {
  return {a.x + float(b), a.y + float(b), a.z + float(b)};
}

inline point3f operator-(const point3f &a, const double b) {
  return {a.x - float(b), a.y - float(b), a.z - float(b)};
}

inline point3f operator*(const point3f &a, const double b) {
  return {a.x * float(b), a.y * float(b), a.z * float(b)};
}

inline point3f operator/(const point3f &a, const double b) {
  return {a.x / float(b), a.y / float(b), a.z / float(b)};
}

inline point3f operator+(const point3f &a, const point3f &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline point3f operator-(const point3f &a, const point3f &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline point3f operator*(const point3f &a, const point3f &b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline point3f operator/(const point3f &a, const point3f &b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}
#endif

struct point3d {
  double x, y, z;

  double operator[](size_t idx) const { return *(&x + idx); }
  double &operator[](size_t idx) { return *(&x + idx); }
};

#if 0
inline point3d operator+(const double a, const point3d &b) {
  return {a + b.x, a + b.y, a + b.z};
}

inline point3d operator-(const double a, const point3d &b) {
  return {a - b.x, a - b.y, a - b.z};
}

inline point3d operator*(const double a, const point3d &b) {
  return {a * b.x, a * b.y, a * b.z};
}

// TODO: safe div
inline point3d operator/(const double a, const point3d &b) {
  return {a / b.x, a / b.y, a / b.z};
}

inline point3d operator+(const float a, const point3d &b) {
  return {double(a) + b.x, double(a) + b.y, double(a) + b.z};
}

inline point3d operator-(const float a, const point3d &b) {
  return {double(a) - b.x, double(a) - b.y, double(a) - b.z};
}

inline point3d operator*(const float a, const point3d &b) {
  return {double(a) * b.x, double(a) * b.y, double(a) * b.z};
}

inline point3d operator/(const float a, const point3d &b) {
  return {double(a) / b.x, double(a) / b.y, double(a) / b.z};
}

inline point3d operator+(const point3d &a, const double b) {
  return {a.x + b, a.y + b, a.z + b};
}

inline point3d operator-(const point3d &a, const double b) {
  return {a.x - b, a.y - b, a.z - b};
}

inline point3d operator*(const point3d &a, const double b) {
  return {a.x * b, a.y * b, a.z * b};
}

inline point3d operator/(const point3d &a, const double b) {
  return {a.x / b, a.y / b, a.z / b};
}

inline point3d operator+(const point3d &a, const float b) {
  return {a.x + double(b), a.y + double(b), a.z + double(b)};
}

inline point3d operator-(const point3d &a, const float b) {
  return {a.x - double(b), a.y - double(b), a.z - double(b)};
}

inline point3d operator*(const point3d &a, const float b) {
  return {a.x * double(b), a.y * double(b), a.z * double(b)};
}

inline point3d operator/(const point3d &a, const float b) {
  return {a.x / double(b), a.y / double(b), a.z / double(b)};
}

inline point3d operator+(const point3d &a, const point3d &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline point3d operator-(const point3d &a, const point3d &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline point3d operator*(const point3d &a, const point3d &b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline point3d operator/(const point3d &a, const point3d &b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}
#endif

struct color3h {
  half r, g, b;

  half operator[](size_t idx) const { return *(&r + idx); }
  half &operator[](size_t idx) { return *(&r + idx); }
};

struct color3f {
  float r, g, b;

  float operator[](size_t idx) const { return *(&r + idx); }
  float &operator[](size_t idx) { return *(&r + idx); }
};

struct color4h {
  half r, g, b, a;

  half operator[](size_t idx) const { return *(&r + idx); }
  half &operator[](size_t idx) { return *(&r + idx); }
};

struct color4f {
  float r, g, b, a;

  float operator[](size_t idx) const { return *(&r + idx); }
  float &operator[](size_t idx) { return *(&r + idx); }
};

struct color3d {
  double r, g, b;

  double operator[](size_t idx) const { return *(&r + idx); }
  double &operator[](size_t idx) { return *(&r + idx); }
};

struct color4d {
  double r, g, b, a;

  double operator[](size_t idx) const { return *(&r + idx); }
  double &operator[](size_t idx) { return *(&r + idx); }
};

struct texcoord2h {
  half s, t;
  half operator[](size_t idx) const { return *(&s + idx); }
  half &operator[](size_t idx) { return *(&s + idx); }
};

struct texcoord2f {
  float s, t;
  float operator[](size_t idx) const { return *(&s + idx); }
  float &operator[](size_t idx) { return *(&s + idx); }
};

struct texcoord2d {
  double s, t;
  double operator[](size_t idx) const { return *(&s + idx); }
  double &operator[](size_t idx) { return *(&s + idx); }
};

struct texcoord3h {
  half s, t, r;
  half operator[](size_t idx) const { return *(&s + idx); }
  half &operator[](size_t idx) { return *(&s + idx); }
};

struct texcoord3f {
  float s, t, r;
  float operator[](size_t idx) const { return *(&s + idx); }
  float &operator[](size_t idx) { return *(&s + idx); }
};

struct texcoord3d {
  double s, t, r;
  double operator[](size_t idx) const { return *(&s + idx); }
  double &operator[](size_t idx) { return *(&s + idx); }
};



// Attribute value Block(`None`)
struct ValueBlock {};

using double2 = std::array<double, 2>;
using double3 = std::array<double, 3>;
using double4 = std::array<double, 4>;

// struct any_value;
// using dict = std::map<std::string, any_value>;
using dict = std::map<std::string, linb::any>;

template <class dtype>
struct TypeTraits;

// import DEFINE_TYPE_TRAIT and DEFINE_ROLE_TYPE_TRAIT
#include "define-type-trait.inc"

// `void` hash no sizeof(void), so define it manually.
template <>
struct TypeTraits<void> {
  using value_type = void;
  using value_underlying_type = void;
  static constexpr uint32_t ndim() { return 0; } /* array dim */
  static constexpr uint32_t size = 0; /* zero for void  */
  static constexpr uint32_t ncomp() { return 0; }
  static constexpr uint32_t type_id() { return TYPE_ID_VOID; }
  static constexpr uint32_t get_type_id() { return TYPE_ID_VOID; }
  static constexpr uint32_t underlying_type_id() { return TYPE_ID_VOID; }
  static std::string type_name() { return "void"; }
  static std::string underlying_type_name() { return "void"; }
  static bool is_role_type() { return false; }
  static bool is_array() { return false; }
};

DEFINE_TYPE_TRAIT(std::nullptr_t, "null", TYPE_ID_NULL, 1);
// DEFINE_TYPE_TRAIT(void, "void", TYPE_ID_VOID, 1);
DEFINE_TYPE_TRAIT(ValueBlock, "None", TYPE_ID_VALUEBLOCK, 1);

DEFINE_TYPE_TRAIT(bool, kBool, TYPE_ID_BOOL, 1);
DEFINE_TYPE_TRAIT(uint8_t, kUChar, TYPE_ID_UCHAR, 1);
DEFINE_TYPE_TRAIT(half, kHalf, TYPE_ID_HALF, 1);

DEFINE_TYPE_TRAIT(int16_t, kShort, TYPE_ID_SHORT, 1);
DEFINE_TYPE_TRAIT(uint16_t, kUShort, TYPE_ID_USHORT, 1);

DEFINE_TYPE_TRAIT(int32_t, kInt, TYPE_ID_INT32, 1);
DEFINE_TYPE_TRAIT(uint32_t, kUInt, TYPE_ID_UINT32, 1);

DEFINE_TYPE_TRAIT(int64_t, kInt64, TYPE_ID_INT64, 1);
DEFINE_TYPE_TRAIT(uint64_t, kUInt64, TYPE_ID_UINT64, 1);

DEFINE_TYPE_TRAIT(char, kChar, TYPE_ID_CHAR, 1);
DEFINE_TYPE_TRAIT(char2, kChar2, TYPE_ID_CHAR2, 2);
DEFINE_TYPE_TRAIT(char3, kChar3, TYPE_ID_CHAR3, 3);
DEFINE_TYPE_TRAIT(char4, kChar4, TYPE_ID_CHAR4, 4);

DEFINE_TYPE_TRAIT(uchar2, kUChar2, TYPE_ID_UCHAR2, 2);
DEFINE_TYPE_TRAIT(uchar3, kUChar3, TYPE_ID_UCHAR3, 3);
DEFINE_TYPE_TRAIT(uchar4, kUChar4, TYPE_ID_UCHAR4, 4);

DEFINE_TYPE_TRAIT(short2, kShort2, TYPE_ID_SHORT2, 2);
DEFINE_TYPE_TRAIT(short3, kShort3, TYPE_ID_SHORT3, 3);
DEFINE_TYPE_TRAIT(short4, kShort4, TYPE_ID_SHORT4, 4);

DEFINE_TYPE_TRAIT(ushort2, kUShort2, TYPE_ID_USHORT2, 2);
DEFINE_TYPE_TRAIT(ushort3, kUShort3, TYPE_ID_USHORT3, 3);
DEFINE_TYPE_TRAIT(ushort4, kUShort4, TYPE_ID_USHORT4, 4);

DEFINE_TYPE_TRAIT(int2, kInt2, TYPE_ID_INT2, 2);
DEFINE_TYPE_TRAIT(int3, kInt3, TYPE_ID_INT3, 3);
DEFINE_TYPE_TRAIT(int4, kInt4, TYPE_ID_INT4, 4);

DEFINE_TYPE_TRAIT(uint2, kUInt2, TYPE_ID_UINT2, 2);
DEFINE_TYPE_TRAIT(uint3, kUInt3, TYPE_ID_UINT3, 3);
DEFINE_TYPE_TRAIT(uint4, kUInt4, TYPE_ID_UINT4, 4);

DEFINE_TYPE_TRAIT(half2, kHalf2, TYPE_ID_HALF2, 2);
DEFINE_TYPE_TRAIT(half3, kHalf3, TYPE_ID_HALF3, 3);
DEFINE_TYPE_TRAIT(half4, kHalf4, TYPE_ID_HALF4, 4);

DEFINE_TYPE_TRAIT(float, kFloat, TYPE_ID_FLOAT, 1);
DEFINE_TYPE_TRAIT(float2, kFloat2, TYPE_ID_FLOAT2, 2);
DEFINE_TYPE_TRAIT(float3, kFloat3, TYPE_ID_FLOAT3, 3);
DEFINE_TYPE_TRAIT(float4, kFloat4, TYPE_ID_FLOAT4, 4);

DEFINE_TYPE_TRAIT(double, kDouble, TYPE_ID_DOUBLE, 1);
DEFINE_TYPE_TRAIT(double2, kDouble2, TYPE_ID_DOUBLE2, 2);
DEFINE_TYPE_TRAIT(double3, kDouble3, TYPE_ID_DOUBLE3, 3);
DEFINE_TYPE_TRAIT(double4, kDouble4, TYPE_ID_DOUBLE4, 4);

DEFINE_TYPE_TRAIT(quath, kQuath, TYPE_ID_QUATH, 1);
DEFINE_TYPE_TRAIT(quatf, kQuatf, TYPE_ID_QUATF, 1);
DEFINE_TYPE_TRAIT(quatd, kQuatd, TYPE_ID_QUATD, 1);

DEFINE_TYPE_TRAIT(matrix2f, kMatrix2f, TYPE_ID_MATRIX2F, 1);
DEFINE_TYPE_TRAIT(matrix3f, kMatrix3f, TYPE_ID_MATRIX3F, 1);
DEFINE_TYPE_TRAIT(matrix4f, kMatrix4f, TYPE_ID_MATRIX4F, 1);

DEFINE_TYPE_TRAIT(matrix2d, kMatrix2d, TYPE_ID_MATRIX2D, 1);
DEFINE_TYPE_TRAIT(matrix3d, kMatrix3d, TYPE_ID_MATRIX3D, 1);
DEFINE_TYPE_TRAIT(matrix4d, kMatrix4d, TYPE_ID_MATRIX4D, 1);

DEFINE_TYPE_TRAIT(timecode, kTimeCode, TYPE_ID_TIMECODE, 1);

//
// Role types
//
DEFINE_ROLE_TYPE_TRAIT(vector3h, kVector3h, TYPE_ID_VECTOR3H, half3);
DEFINE_ROLE_TYPE_TRAIT(vector3f, kVector3f, TYPE_ID_VECTOR3F, float3);
DEFINE_ROLE_TYPE_TRAIT(vector3d, kVector3d, TYPE_ID_VECTOR3D, double3);

DEFINE_ROLE_TYPE_TRAIT(normal3h, kNormal3h, TYPE_ID_NORMAL3H, half3);
DEFINE_ROLE_TYPE_TRAIT(normal3f, kNormal3f, TYPE_ID_NORMAL3F, float3);
DEFINE_ROLE_TYPE_TRAIT(normal3d, kNormal3d, TYPE_ID_NORMAL3D, double3);

DEFINE_ROLE_TYPE_TRAIT(point3h, kPoint3h, TYPE_ID_POINT3H, half3);
DEFINE_ROLE_TYPE_TRAIT(point3f, kPoint3f, TYPE_ID_POINT3F, float3);
DEFINE_ROLE_TYPE_TRAIT(point3d, kPoint3d, TYPE_ID_POINT3D, double3);

DEFINE_ROLE_TYPE_TRAIT(frame4d, kFrame4d, TYPE_ID_FRAME4D, matrix4d);

DEFINE_ROLE_TYPE_TRAIT(color3h, kColor3h, TYPE_ID_COLOR3H, half3);
DEFINE_ROLE_TYPE_TRAIT(color4h, kColor4h, TYPE_ID_COLOR4H, half4);
DEFINE_ROLE_TYPE_TRAIT(color3f, kColor3f, TYPE_ID_COLOR3F, float3);
DEFINE_ROLE_TYPE_TRAIT(color4f, kColor4f, TYPE_ID_COLOR4F, float4);
DEFINE_ROLE_TYPE_TRAIT(color3d, kColor3d, TYPE_ID_COLOR3D, double3);
DEFINE_ROLE_TYPE_TRAIT(color4d, kColor4d, TYPE_ID_COLOR4D, double4);

DEFINE_ROLE_TYPE_TRAIT(texcoord2h, kTexCoord2h, TYPE_ID_TEXCOORD2H, half2);
DEFINE_ROLE_TYPE_TRAIT(texcoord2f, kTexCoord2f, TYPE_ID_TEXCOORD2F, float2);
DEFINE_ROLE_TYPE_TRAIT(texcoord2d, kTexCoord2d, TYPE_ID_TEXCOORD2D, double2);

DEFINE_ROLE_TYPE_TRAIT(texcoord3h, kTexCoord3h, TYPE_ID_TEXCOORD3H, half3);
DEFINE_ROLE_TYPE_TRAIT(texcoord3f, kTexCoord3f, TYPE_ID_TEXCOORD3F, float3);
DEFINE_ROLE_TYPE_TRAIT(texcoord3d, kTexCoord3d, TYPE_ID_TEXCOORD3D, double3);

//
//
//

DEFINE_TYPE_TRAIT(token, kToken, TYPE_ID_TOKEN, 1);
DEFINE_TYPE_TRAIT(std::string, kString, TYPE_ID_STRING, 1);
DEFINE_TYPE_TRAIT(StringData, kString, TYPE_ID_STRING_DATA, 1);
DEFINE_TYPE_TRAIT(dict, kDictionary, TYPE_ID_DICT, 1);

DEFINE_TYPE_TRAIT(AssetPath, kAssetPath, TYPE_ID_ASSET_PATH, 1);

//
// Other types(e.g. TYPE_ID_REFERENCE) are defined in corresponding header
// files(e.g. `prim-types.hh`, `crate-format.hh`(Data types used in Crate data))
//

#undef DEFINE_TYPE_TRAIT
#undef DEFINE_ROLE_TYPE_TRAIT

// 1D Array
template <typename T>
struct TypeTraits<std::vector<T>> {
  using value_type = std::vector<T>;
  static constexpr uint32_t ndim() { return 1; } /* array dim */
  static constexpr uint32_t ncomp() { return TypeTraits<T>::ncomp(); }
  // Return the size of base type
  static constexpr size_t size() { return TypeTraits<T>::size(); }
  static constexpr uint32_t type_id() { return
      TypeTraits<T>::type_id() | TYPE_ID_1D_ARRAY_BIT; }
  static constexpr uint32_t get_type_id() {
      return TypeTraits<T>::type_id() | TYPE_ID_1D_ARRAY_BIT; }
  static constexpr uint32_t underlying_type_id() {
      return TypeTraits<T>::underlying_type_id() | TYPE_ID_1D_ARRAY_BIT; }
  static std::string type_name() { return TypeTraits<T>::type_name() + "[]"; }
  static std::string underlying_type_name() {
    return TypeTraits<T>::underlying_type_name() + "[]";
  }
  static constexpr bool is_role_type() { return TypeTraits<T>::is_role_type(); }
  static constexpr bool is_array() { return true; }
};

#if 0  // Current pxrUSD does not support 2D array
// 2D Array
// TODO(syoyo): support 3D array?
template <typename T>
struct TypeTraits<std::vector<std::vector<T>>> {
  using value_type = std::vector<std::vector<T>>;
  static constexpr uint32_t ndim = 2; /* array dim */
  static constexpr uint32_t ncomp = TypeTraits<T>::ncomp;
  static constexpr uint32_t type_id =
      TypeTraits<T>::type_id | TYPE_ID_2D_ARRAY_BIT;
  static constexpr uint32_t underlying_type_id =
      TypeTraits<T>::underlying_type_id | TYPE_ID_2D_ARRAY_BIT;
  static std::string type_name() { return TypeTraits<T>::type_name() + "[][]"; }
  static std::string underlying_type_name() {
    return TypeTraits<T>::underlying_type_name() + "[][]";
  }
};
#endif

// Lookup TypeTraits<T>::type_name from type_id
// Return nullopt when the input is invalid type id.
nonstd::optional<std::string> TryGetTypeName(uint32_t tyid);

// Return error string when the input is invalid type id
std::string GetTypeName(uint32_t tyid);

// Lookup TypeTraits<T>::type_id from string
// Return nullopt when the input is invalid type name.
nonstd::optional<uint32_t> TryGetTypeId(const std::string &tyname);

// Return TYPE_ID_INVALID when the input is invalid type name
uint32_t GetTypeId(const std::string &tyname);

// For Role type.
// Get underlying type name(e.g. return type "float4" for role type "color4f"),
// or return nullopt/invalid string for invalid input type id. For non-Role
// type, the behavior is same with TryGetTypeName/GetTypeName(i.e, return
// "float4" for type `float4`)
nonstd::optional<std::string> TryGetUnderlyingTypeName(uint32_t tyid);
std::string GetUnderlyingTypeName(uint32_t tyid);

// Get underlying type id(e.g. return type "float4" for role type "color4f"), or
// return nullopt/TYPE_ID_INVALID for invalid input type name For non-Role type,
// the behavior is same with TryGetTypeId/GetTypeId(i.e, return `float4` for
// name "float4")
nonstd::optional<uint32_t> TryGetUnderlyingTypeId(const std::string &tyname);
uint32_t GetUnderlyingTypeId(const std::string &tyname);

// TODO: uint32_t GetUnderlyingTypeId(const uint32_t tyid)

/// @brief Check if given typeName string is a role-type(e.g. "vector3f")
/// @param[in] tyname typeName string
/// @return true if a type is role-type.
bool IsRoleType(const std::string &tyname);

/// @brief Check if given type_id is a role-type(e.g. "vector3f")
/// @param[in] tyid type id(value::TYPE_ID_***)
/// @return true if a type is role-type.
bool IsRoleType(const uint32_t tyid);

}  // namespace value
}  // namespace tinyusdz

#include "tiny-any.inc"

namespace tinyusdz {
namespace value {

///
/// Generic Value class using any
/// TODO: Type-check when casting with underlying_type(Need to modify linb::any
/// class)
///
class Value {
 public:
  Value() = default;

  template <class T>
  Value(const T &v) : v_(v) {}

  // template <class T>
  // Value(T &&v) : v_(v) {}

  const std::string type_name() const { return v_.type_name(); }
  const std::string underlying_type_name() const {
    return v_.underlying_type_name();
  }

  uint32_t type_id() const { return v_.type_id(); }
  uint32_t underlying_type_id() const { return v_.underlying_type_id(); }

  //
  // Cast value to given type.
  //
  // when `strict_cast` is false(default behavior), it supports casting type among role type and underlying type.
  // (e.g. "float3" -> "color3f", "color3f" -> "vector3f", "normal3f[]" -> "float3[]")
  //
  // Return nullptr when type conversion failed.
  template <class T>
  const T *as(bool strict_cast = false) const {
    if (TypeTraits<T>::type_id() == v_.type_id()) {
      return linb::any_cast<const T>(&v_);
    } else if (!strict_cast) {
      // NOTE: linb::any_cast does type_id check, so use linb::cast(~= reinterpret_cast) here
      if (TypeTraits<T>::is_array() && (v_.type_id() & value::TYPE_ID_1D_ARRAY_BIT)) { // both are array type
        if ((TypeTraits<T>::underlying_type_id() & (~value::TYPE_ID_1D_ARRAY_BIT)) == (v_.underlying_type_id() & (~value::TYPE_ID_1D_ARRAY_BIT))) {
          return linb::cast<const T>(&v_);
        }
      } else if (!TypeTraits<T>::is_array() && !(v_.type_id() & value::TYPE_ID_1D_ARRAY_BIT)) { // both are scalar type.
        if (TypeTraits<T>::underlying_type_id() == v_.underlying_type_id()) {
          return linb::cast<const T>(&v_);
        }
      }
    }

    return nullptr;
  }

  // Non const version of `as`.
  //
  // Return nullptr when type conversion failed.
  template <class T>
  T *as(bool strict_cast = false) {
    if (TypeTraits<T>::type_id() == v_.type_id()) {
      return linb::any_cast<T>(&v_);
    } else if (!strict_cast) {
      if (TypeTraits<T>::is_array() && (v_.type_id() & value::TYPE_ID_1D_ARRAY_BIT)) { // both are array type
        if ((TypeTraits<T>::underlying_type_id() & (~value::TYPE_ID_1D_ARRAY_BIT)) == (v_.underlying_type_id() & (~value::TYPE_ID_1D_ARRAY_BIT))) {
          return linb::cast<T>(&v_);
        }
      } else if (!TypeTraits<T>::is_array() && !(v_.type_id() & value::TYPE_ID_1D_ARRAY_BIT)) { // both are scalar type.
        if (TypeTraits<T>::underlying_type_id() == v_.underlying_type_id()) {
          return linb::cast<T>(&v_);
        }
      }
    }

    return nullptr;
  }


#if 0
  // Useful function to retrieve concrete value with type T.
  // Undefined behavior(usually will triger segmentation fault) when
  // type-mismatch. (We don't throw exception)
  template <class T>
  const T value() const {
    //return (*reinterpret_cast<const T *>(v_.value()));
    return linb::any_cast<const T>(v_);
  }
#endif

  // Type-safe way to get concrete value.
  template <class T>
  nonstd::optional<T> get_value(bool strict_cast = false) const {
    if (TypeTraits<T>::type_id() == v_.type_id()) {
      const T *pv = linb::any_cast<const T>(&v_);
      if (!pv) {
        // ???
        return nonstd::nullopt;
      }

      return std::move(*pv);
    } else if (!strict_cast) {

      if (TypeTraits<T>::is_array() && (v_.type_id() & value::TYPE_ID_1D_ARRAY_BIT)) { // both are array type
        if ((TypeTraits<T>::underlying_type_id() & (~value::TYPE_ID_1D_ARRAY_BIT)) == (v_.underlying_type_id() & (~value::TYPE_ID_1D_ARRAY_BIT))) {
          return std::move(*linb::cast<const T>(&v_));
        }
      } else if (!TypeTraits<T>::is_array() && !(v_.type_id() & value::TYPE_ID_1D_ARRAY_BIT)) { // both are scalar type.
        if (TypeTraits<T>::underlying_type_id() == v_.underlying_type_id()) {
          return std::move(*linb::cast<const T>(&v_));
        }
      }
    }
    return nonstd::nullopt;
  }


  template <class T>
  Value &operator=(const T &v) {
    v_ = v;
    return (*this);
  }

  const linb::any &get_raw() const { return v_; }

  bool is_array() const { return (v_.type_id() & value::TYPE_ID_1D_ARRAY_BIT); }

  // return 0 for non array type.
  // This method is primaliry for Primvar types(`float[]`, `color3f[]`, ...)
  // It does not report non-Primvar types(e.g. `Reference`, `Xform`, `GeomMesh`,
  // ...)
  size_t array_size() const;

  bool is_empty() const { return v_.type_id() == value::TYPE_ID_NULL; }

  bool is_none() const { return v_.type_id() == value::TYPE_ID_VALUEBLOCK; }

 private:
  // any_value v_;
  linb::any v_{nullptr};
};

// TimeSample interpolation type.
//
// Held = something like numpy.digitize(right=False)
// https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
//
// Returns `values[i-1]` for `times[i-1] <= t < times[i]`
//
// Linear = linear interpolation
//
// example:
// { 0 : 0.0
//   10 : 1.0
// }
//
// - Held
//   - time 5 = returns 0.0
//   - time 9.99 = returns 0.0
//   - time 10 = returns 1.0
// - Linear
//   - time 5 = returns 0.5
//   - time 9.99 = nearly 1.0
//   - time 10 = 1.0
//
enum class TimeSampleInterpolationType {
  Held,  // something like nearest-neighbor.
  Linear,
};

//
// Supported type for `Linear` interpolation
//
// half, float, double, TimeCode(double)
// matrix2d, matrix3d, matrix4d,
// float2h, float3h, float4h
// float2f, float3f, float4f
// float2d, float3d, float4d
// quath, quatf, quatd
// (use slerp for quaternion type)

bool IsLerpSupportedType(uint32_t tyid);

template<class T>
struct LerpTraits
{
  static constexpr bool supported() {
    return false;
  }
};

#define DEFINE_LERP_TRAIT(ty) \
template <> \
struct LerpTraits<ty> { \
  static constexpr bool supported() { \
    return true; \
  } \
};  \
template <> \
struct LerpTraits<std::vector<ty>> { \
  static constexpr bool supported() { \
    return true; \
  } \
}; 

DEFINE_LERP_TRAIT(value::half)
DEFINE_LERP_TRAIT(value::half2)
DEFINE_LERP_TRAIT(value::half3)
DEFINE_LERP_TRAIT(value::half4)
DEFINE_LERP_TRAIT(float)
DEFINE_LERP_TRAIT(value::float2)
DEFINE_LERP_TRAIT(value::float3)
DEFINE_LERP_TRAIT(value::float4)
DEFINE_LERP_TRAIT(double)
DEFINE_LERP_TRAIT(value::double2)
DEFINE_LERP_TRAIT(value::double3)
DEFINE_LERP_TRAIT(value::double4)
DEFINE_LERP_TRAIT(value::quath)
DEFINE_LERP_TRAIT(value::quatf)
DEFINE_LERP_TRAIT(value::quatd)
DEFINE_LERP_TRAIT(value::matrix2f)
DEFINE_LERP_TRAIT(value::matrix3f)
DEFINE_LERP_TRAIT(value::matrix4f)
DEFINE_LERP_TRAIT(value::matrix2d)
DEFINE_LERP_TRAIT(value::matrix3d)
DEFINE_LERP_TRAIT(value::matrix4d)
DEFINE_LERP_TRAIT(value::timecode)
DEFINE_LERP_TRAIT(value::normal3h)
DEFINE_LERP_TRAIT(value::normal3f)
DEFINE_LERP_TRAIT(value::normal3d)
DEFINE_LERP_TRAIT(value::vector3h)
DEFINE_LERP_TRAIT(value::vector3f)
DEFINE_LERP_TRAIT(value::vector3d)
DEFINE_LERP_TRAIT(value::point3h)
DEFINE_LERP_TRAIT(value::point3f)
DEFINE_LERP_TRAIT(value::point3d)
DEFINE_LERP_TRAIT(value::color3h)
DEFINE_LERP_TRAIT(value::color3f)
DEFINE_LERP_TRAIT(value::color3d)
DEFINE_LERP_TRAIT(value::color4h)
DEFINE_LERP_TRAIT(value::color4f)
DEFINE_LERP_TRAIT(value::color4d)
DEFINE_LERP_TRAIT(value::texcoord2h)
DEFINE_LERP_TRAIT(value::texcoord2f)
DEFINE_LERP_TRAIT(value::texcoord2d)
DEFINE_LERP_TRAIT(value::texcoord3h)
DEFINE_LERP_TRAIT(value::texcoord3f)
DEFINE_LERP_TRAIT(value::texcoord3d)
DEFINE_LERP_TRAIT(value::frame4d)

#undef DEFINE_LERP_TRAIT

///
/// @param[in] dt interpolator [0.0, 1.0)
///
bool Lerp(const value::Value &a, const value::Value &b, double dt,
          value::Value *dst);



// Handy, but may not be efficient for large time samples(e.g. 1M samples or
// more)
//
// For the runtime speed, with "-O2 -g" optimization, adding 10M `double`
// samples to linb::any takes roughly 1.8 ms on Threadripper 1950X, whereas
// simple vector<double> push_back takes 390 us(roughly x4 times faster). (Build
// benchmarks to see the numbers on your CPU)
//
// We assume having large time samples is rare situlation, and above benchmark
// speed is acceptable in general  usecases.
//
// `None`(ValueBlock) is represented by setting `Sample::blocked` true.
//
struct TimeSamples {
  struct Sample {
    double t;
    value::Value value;
    bool blocked{false};
  };

  bool empty() const { return _samples.empty(); }

  size_t size() const { return _samples.size(); }

  void clear() {
    _samples.clear();
    _dirty = true;
  }

  void update() const {
    std::sort(_samples.begin(), _samples.end(),
              [](const Sample &a, const Sample &b) { return a.t < b.t; });

    _dirty = false;
  }

  bool has_sample_at(const double t) const;
  bool get_sample_at(const double t, Sample **s);

  nonstd::optional<double> get_time(size_t idx) const {
    if (idx >= _samples.size()) {
      return nonstd::nullopt;
    }

    if (_dirty) {
      update();
    }

    return _samples[idx].t;
  }

  nonstd::optional<value::Value> get_value(size_t idx) const {
    if (idx >= _samples.size()) {
      return nonstd::nullopt;
    }

    if (_dirty) {
      update();
    }

    return _samples[idx].value;
  }

  uint32_t type_id() const {
    if (_samples.size()) {
      if (_dirty) {
        update();
      }
      return _samples[0].value.type_id();
    } else {
      return value::TypeId::TYPE_ID_INVALID;
    }
  }

  std::string type_name() const {
    if (_samples.size()) {
      if (_dirty) {
        update();
      }
      return _samples[0].value.type_name();
    } else {
      return std::string();
    }
  }

  void add_sample(const Sample &s) {
    _samples.push_back(s);
    _dirty = true;
  }

  // Value may be None(ValueBlock)
  void add_sample(double t, const value::Value &v) {
    Sample s;
    s.t = t;
    s.value = v;
    s.blocked = v.is_none();
    _samples.push_back(s);
    _dirty = true;
  }

  // We still need "dummy" value for type_name() and type_id()
  void add_blocked_sample(double t, const value::Value &v) {
    Sample s;
    s.t = t;
    s.value = v;
    s.blocked = true;

    _samples.emplace_back(s);
    _dirty = true;
  }

  const std::vector<Sample> &get_samples() const {
    if (_dirty) {
      update();
    }
    return _samples;
  }

  std::vector<Sample> &samples() {
    if (_dirty) {
      update();
    }
    return _samples;
  }

#if 1  // TODO: Write implementation in .cc

    // Get value at specified time.
    // For non-interpolatable types(includes enums and unknown types)
    //
    // Return `Held` value even when TimeSampleInterpolationType is
    // Linear. Returns nullopt when specified time is out-of-range.
    template<typename T, std::enable_if_t<!value::LerpTraits<T>::supported(), std::nullptr_t> = nullptr>
    bool get(T *dst, double t = value::TimeCode::Default(),
             value::TimeSampleInterpolationType interp =
                 value::TimeSampleInterpolationType::Linear) const {

      (void)interp;

      if (!dst) {
        return false;
      }

      if (empty()) {
        return false;
      }

      if (_dirty) {
        update();
      }

      if (value::TimeCode(t).is_default()) {
        // TODO: Handle bloked
        if (const auto pv = _samples[0].value.as<T>()) {
          (*dst) = *pv;
          return true;
        }
        return false;
      } else {

        if (_samples.size() == 1) {
          if (const auto pv = _samples[0].value.as<T>()) {
            (*dst) = *pv;
            return true;
          }
          return false;
        }

        auto it = std::upper_bound(
          _samples.begin(), _samples.end(), t,
          [](double tval, const Sample &a) { return tval < a.t; });

        const auto it_minus_1 = (it == _samples.begin()) ? _samples.begin() : (it - 1);

        const value::Value &v = it_minus_1->value;

        if (const T *pv = v.as<T>()) {
          (*dst) = *pv;
          return true;
        }
        return false;
      }
  }

  // Get value at specified time.
  // Return linearly interpolated value when TimeSampleInterpolationType is
  // Linear. Returns false when samples is empty or some internal error.
  template<typename T, std::enable_if_t<value::LerpTraits<T>::supported(), std::nullptr_t> = nullptr>
  bool get(T *dst, double t = value::TimeCode::Default(),
           TimeSampleInterpolationType interp =
               TimeSampleInterpolationType::Linear) const {
    if (!dst) {
      return false;
    }

    if (empty()) {
      return false;
    }

    if (_dirty) {
      update();
    }

    if (value::TimeCode(t).is_default()) {
      // FIXME: Use the first item for now.
      // TODO: Handle bloked
      if (const auto pv = _samples[0].value.as<T>()) {
        (*dst) = *pv;
        return true;
      }
      return false;
    } else {

      if (_samples.size() == 1) {
        if (const auto pv = _samples[0].value.as<T>()) {
          (*dst) = *pv;
          return true;
        }
        return true;
      }

      if (interp == TimeSampleInterpolationType::Linear) {
        auto it = std::lower_bound(
            _samples.begin(), _samples.end(), t,
            [](const Sample &a, double tval) { return a.t < tval; });


        // MS STL does not allow seek vector iterator before begin
        // Issue #110
        const auto it_minus_1 = (it == _samples.begin()) ? _samples.begin() : (it - 1);

        size_t idx0 = size_t(std::max(
            int64_t(0),
            std::min(int64_t(_samples.size() - 1),
                     int64_t(std::distance(_samples.begin(), it_minus_1)))));
        size_t idx1 =
            size_t(std::max(int64_t(0), std::min(int64_t(_samples.size() - 1),
                                                 int64_t(idx0) + 1)));

        double tl = _samples[idx0].t;
        double tu = _samples[idx1].t;

        double dt = (t - tl);
        if (std::fabs(tu - tl) < std::numeric_limits<double>::epsilon()) {
          // slope is zero.
          dt = 0.0;
        } else {
          dt /= (tu - tl);
        }

        // Just in case.
        dt = std::max(0.0, std::min(1.0, dt));

        const value::Value &p0 = _samples[idx0].value;
        const value::Value &p1 = _samples[idx1].value;

        value::Value p;
        if (!Lerp(p0, p1, dt, &p)) {
          return false;
        }

        if (const auto pv = p.as<T>()) {
          (*dst) = *pv;
          return true;
        }
        return false;
      } else {
        // Held
        auto it = std::upper_bound(
          _samples.begin(), _samples.end(), t,
          [](double tval, const Sample &a) { return tval < a.t; });

        const auto it_minus_1 = (it == _samples.begin()) ? _samples.begin() : (it - 1);

        const value::Value &v = it_minus_1->value;

        if (const T *pv = v.as<T>()) {
          (*dst) = *pv;
          return true;
        }
        
        return false;
      }
    }

    return false;
  }
#endif

 private:
  mutable std::vector<Sample> _samples;
  mutable bool _dirty{false};
};



///
/// Try to cast the value with src type to dest type as much as possible.
/// When src type is scalar type and dest type is vector, the value of src type is scattered to the value of dest type.
///
/// No lexical cast feature involved.
/// TODO: overflow check
///
/// Considers role type.
/// example:
/// - float3 -> normal3 : OK
/// - float3 -> normal3 : OK
/// - normal3 -> float : OK
/// - normal3 -> int : OK(use normal3[0])
/// - float2 -> normal3 : OK
/// - float3 -> texcoord2 : OK(use float3[0] and float3[1])
/// - string -> float : NG
/// - float -> string : NG
/// - float -> string : NG
///

bool FlexibleValueConvert(const value::Value &src, value::Value &dst);

template<typename SrcT, typename DestT>
bool FlexibleTypeCast(const SrcT &src, DestT &dst) {
  value::Value srcv(src);
  value::Value dstv(dst);

  return FlexibleValueConvert(srcv, dstv);
}

///
/// Cast input value's type to Role type
/// Return true: cast success.
///
bool RoleTypeCast(const uint32_t roleTyId, value::Value &inout);

///
/// Upcast value to specified type(e.g. `half` -> `float`)
/// Return true: Upcast success.
///
bool UpcastType(const std::string &toType, value::Value &inout);

#if 0
// simple linear interpolator
template <typename T>
struct LinearInterpolator {
  static T interpolate(const T *values, const size_t n, const double _t) {
    if (n == 0) {
      return static_cast<T>(0);
    } else if (n == 1) {
      return values[0];
    }

    // [0.0, 1.0]
    double t = std::fmin(0.0, std::fmax(_t, 1.0));

    size_t idx0 = std::max(n - 1, size_t(t * double(n)));
    size_t idx1 = std::max(n - 1, idx0 + 1);

    return (1.0 - t) * values[idx0] + t * values[idx1];
  }
};

// Explicitly typed version of `TimeSamples`
//
// `None` value and `deleted` items are omitted in this data struct.
// e.g.
//
// double radius.timeSamples = { 0: 1.0, 1: None, 2: 3.0 }
//
// in .usd(or `TimeSamples` class), are stored as
//
// radius = { 0: 1.0, 2: 3.0 }
//
template <typename T>
struct AnimatableValue {
  std::vector<double> times;  // Assume sorted
  std::vector<T> values;

  bool is_scalar() const { return (times.size() == 0) && (values.size() == 1); }

  bool is_timesample() const {
    return (times.size() > 0) && (times.size() == values.size());
  }

  template <class Interpolator>
  T Get(double time = 0.0) {
    std::vector<double>::iterator it =
        std::lower_bound(times.begin(), times.end(), time);

    size_t idx0, idx1;
    if (it != times.end()) {
      idx0 = std::distance(times.begin(), it);
      idx1 = std::min(idx0 + 1, times.size() - 1);
    } else {
      idx0 = idx1 = times.size() - 1;
    }
    double slope = times[idx1] - times[idx0];
    if (slope < std::numeric_limits<double>::epsilon()) {
      slope = 1.0;
    }

    const double t = (times[idx1] - time) / slope;

    T val = Interpolator::interpolate(values.data(), values.size(), t);
    return val;
  }
};
#endif

#if 0  // TODO: Remove? since not used so frequently at the moment.
//
// typecast from type_id
// It does not throw exception.
//
template <uint32_t tid>
struct typecast {};

#define TYPECAST_BASETYPE(__tid, __ty)                   \
  template <>                                            \
  struct typecast<__tid> {                               \
    static __ty to(const any_value &v) {                 \
      return *reinterpret_cast<const __ty *>(v.value()); \
    }                                                    \
  }

TYPECAST_BASETYPE(TYPE_ID_BOOL, bool);
TYPECAST_BASETYPE(TYPE_ID_UCHAR, uint8_t);
TYPECAST_BASETYPE(TYPE_ID_HALF, half);
TYPECAST_BASETYPE(TYPE_ID_HALF2, half2);
TYPECAST_BASETYPE(TYPE_ID_HALF3, half3);
TYPECAST_BASETYPE(TYPE_ID_HALF4, half4);

TYPECAST_BASETYPE(TYPE_ID_UINT32, uint32_t);
TYPECAST_BASETYPE(TYPE_ID_FLOAT, float);
TYPECAST_BASETYPE(TYPE_ID_DOUBLE, double);

TYPECAST_BASETYPE(TYPE_ID_FLOAT | TYPE_ID_1D_ARRAY_BIT, std::vector<float>);

// TODO(syoyo): Implement more types...

#undef TYPECAST_BASETYPE
#endif

#if 0
struct AttribMap {
  std::map<std::string, Value> attribs;
};
#endif

}  // namespace value

}  // namespace tinyusdz

namespace tinyusdz {
namespace value {

static_assert(sizeof(quath) == 8, "sizeof(quath) must be 8");
static_assert(sizeof(quatf) == 16, "sizeof(quatf) must be 16");
static_assert(sizeof(quatd) == 32, "sizeof(quatd) must be 32");
static_assert(sizeof(half) == 2, "sizeof(half) must be 2");
static_assert(sizeof(half2) == 4, "sizeof(half2) must be 4");
static_assert(sizeof(half3) == 6, "sizeof(half3) must be 6");
static_assert(sizeof(half4) == 8, "sizeof(half4) must be 8");
static_assert(sizeof(float3) == 12, "sizeof(float3) must be 12");
static_assert(sizeof(color3f) == 12, "sizeof(color3f) must be 12");
static_assert(sizeof(color4f) == 16, "sizeof(color4f) must be 16");

}  // namespace value

}  // namespace tinyusdz
