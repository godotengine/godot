// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
#include "value-types.hh"

#include "str-util.hh"
#include "value-pprint.hh"
#include "value-eval-util.hh"

//
#include "common-macros.inc"
#include "math-util.inc"

// For compile-time map
// Another candidate is frozen: https://github.com/serge-sans-paille/frozen
//
#include "external/mapbox/eternal/include/mapbox/eternal.hpp"

namespace tinyusdz {
namespace value {

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
bool IsLerpSupportedType(uint32_t tyid) {

  // TODO: Directly get underlying_typeid
  bool has_underlying_tyid{false};
  uint32_t underlying_tyid{TYPE_ID_INVALID};

  if (auto pv = TryGetUnderlyingTypeName(tyid)) {
    underlying_tyid = GetTypeId(pv.value());
    has_underlying_tyid = true;
  } 

  // See also for underlying_type_id to simplify check for Role types(e.g. color3f)
#define IS_SUPPORTED_TYPE(__tyid, __ty) \
  if (__tyid == value::TypeTraits<__ty>::type_id()) { \
    return true; \
  } else if (__tyid == value::TypeTraits<__ty>::underlying_type_id()) { \
    return true; \
  } else if (__tyid & value::TYPE_ID_1D_ARRAY_BIT) { \
    if ((__tyid & (~value::TYPE_ID_1D_ARRAY_BIT)) == (value::TypeTraits<__ty>::type_id())) { \
      return true; \
    } else if ((__tyid & (~value::TYPE_ID_1D_ARRAY_BIT)) == (value::TypeTraits<__ty>::underlying_type_id())) { \
      return true; \
    } \
  }

  // Assume __uty is underlying_type.
#define IS_SUPPORTED_UNDERLYING_TYPE(__utyid, __uty) \
  if (__utyid == value::TypeTraits<__uty>::type_id()) { \
    return true; \
  } else if (__utyid & value::TYPE_ID_1D_ARRAY_BIT) { \
    if ((__utyid & (~value::TYPE_ID_1D_ARRAY_BIT)) == (value::TypeTraits<__uty>::type_id())) { \
      return true; \
    } \
  }

  IS_SUPPORTED_TYPE(tyid, value::half)
  IS_SUPPORTED_TYPE(tyid, value::half2)
  IS_SUPPORTED_TYPE(tyid, value::half3)
  IS_SUPPORTED_TYPE(tyid, value::half4)
  IS_SUPPORTED_TYPE(tyid, float)
  IS_SUPPORTED_TYPE(tyid, value::float2)
  IS_SUPPORTED_TYPE(tyid, value::float3)
  IS_SUPPORTED_TYPE(tyid, value::float4)
  IS_SUPPORTED_TYPE(tyid, double)
  IS_SUPPORTED_TYPE(tyid, value::double2)
  IS_SUPPORTED_TYPE(tyid, value::double3)
  IS_SUPPORTED_TYPE(tyid, value::double4)
  IS_SUPPORTED_TYPE(tyid, value::quath)
  IS_SUPPORTED_TYPE(tyid, value::quatf)
  IS_SUPPORTED_TYPE(tyid, value::quatd)
  IS_SUPPORTED_TYPE(tyid, value::matrix2d)
  IS_SUPPORTED_TYPE(tyid, value::matrix3d)
  IS_SUPPORTED_TYPE(tyid, value::matrix4d)

  if (has_underlying_tyid) {
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::half)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::half2)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::half3)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::half4)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, float)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::float2)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::float3)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::float4)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, double)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::double2)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::double3)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::double4)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::quath)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::quatf)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::quatd)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::matrix2d)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::matrix3d)
    IS_SUPPORTED_UNDERLYING_TYPE(underlying_tyid, value::matrix4d)
  }

#undef IS_SUPPORTED_TYPE
#undef IS_SUPPORTED_UNDERLYING_TYPE

  return false;

}

bool Lerp(const value::Value &a, const value::Value &b, double dt, value::Value *dst) {
  if (!dst) {
    return false;
  }

  if (a.type_id() != b.type_id()) {
    return false;
  }

  uint32_t tyid = a.type_id();

  if (!IsLerpSupportedType(tyid)) {
    return false;
  }

  bool ok{false};
  value::Value result;

#define DO_LERP(__ty) \
  if (tyid == value::TypeTraits<__ty>::type_id()) { \
    const __ty *v0 = a.as<__ty>(); \
    const __ty *v1 = b.as<__ty>(); \
    __ty c; \
    if (v0 && v1) { \
      c = lerp(*v0, *v1, dt); \
      result = c; \
      ok = true; \
    } \
  } else if (tyid == value::TypeTraits<std::vector<__ty>>::type_id()) { \
    const std::vector<__ty> *v0 = a.as<std::vector<__ty>>(); \
    const std::vector<__ty> *v1 = b.as<std::vector<__ty>>(); \
    std::vector<__ty> c; \
    if (v0 && v1) { \
      c = lerp(*v0, *v1, dt); \
      result = c; \
      ok = true; \
    } \
  } else

  DO_LERP(value::half)
  DO_LERP(value::half2)
  DO_LERP(value::half3)
  DO_LERP(value::half4)
  DO_LERP(float)
  DO_LERP(value::float2)
  DO_LERP(value::float3)
  DO_LERP(value::float4)
  DO_LERP(double)
  DO_LERP(value::double2)
  DO_LERP(value::double3)
  DO_LERP(value::double4)
  DO_LERP(value::quath)
  DO_LERP(value::quatf)
  DO_LERP(value::quatd)
  DO_LERP(value::color3h)
  DO_LERP(value::color3f)
  DO_LERP(value::color3d)
  DO_LERP(value::color4h)
  DO_LERP(value::color4f)
  DO_LERP(value::color4d)
  DO_LERP(value::point3h)
  DO_LERP(value::point3f)
  DO_LERP(value::point3d)
  DO_LERP(value::normal3h)
  DO_LERP(value::normal3f)
  DO_LERP(value::normal3d)
  DO_LERP(value::vector3h)
  DO_LERP(value::vector3f)
  DO_LERP(value::vector3d)
  DO_LERP(value::texcoord2h)
  DO_LERP(value::texcoord2f)
  DO_LERP(value::texcoord2d)
  DO_LERP(value::texcoord3h)
  DO_LERP(value::texcoord3f)
  DO_LERP(value::texcoord3d)
  {
    DCOUT("TODO: type " << GetTypeName(tyid));
  }

#undef DO_LERP

  if (ok) {
    (*dst) = result;
  }

  return ok;
}

nonstd::optional<std::string> TryGetTypeName(uint32_t tyid) {
  MAPBOX_ETERNAL_CONSTEXPR const auto tynamemap =
      mapbox::eternal::map<uint32_t, mapbox::eternal::string>({
          {TYPE_ID_TOKEN, kToken},
          {TYPE_ID_STRING, kString},
          {TYPE_ID_STRING, kPath},
          {TYPE_ID_ASSET_PATH, kAssetPath},
          {TYPE_ID_DICT, kDictionary},
          {TYPE_ID_TIMECODE, kTimeCode},
          {TYPE_ID_BOOL, kBool},
          {TYPE_ID_UCHAR, kUChar},
          {TYPE_ID_HALF, kHalf},
          {TYPE_ID_INT32, kInt},
          {TYPE_ID_UINT32, kUInt},
          {TYPE_ID_INT64, kInt64},
          {TYPE_ID_UINT64, kUInt64},
          {TYPE_ID_INT2, kInt2},
          {TYPE_ID_INT3, kInt3},
          {TYPE_ID_INT4, kInt4},
          {TYPE_ID_UINT2, kUInt2},
          {TYPE_ID_UINT3, kUInt3},
          {TYPE_ID_UINT4, kUInt4},
          {TYPE_ID_HALF2, kHalf2},
          {TYPE_ID_HALF3, kHalf3},
          {TYPE_ID_HALF4, kHalf4},
          {TYPE_ID_MATRIX2D, kMatrix2d},
          {TYPE_ID_MATRIX3D, kMatrix3d},
          {TYPE_ID_MATRIX4D, kMatrix4d},
          {TYPE_ID_FLOAT, kFloat},
          {TYPE_ID_FLOAT2, kFloat2},
          {TYPE_ID_FLOAT3, kFloat3},
          {TYPE_ID_FLOAT4, kFloat4},
          {TYPE_ID_DOUBLE, kDouble},
          {TYPE_ID_DOUBLE2, kDouble2},
          {TYPE_ID_DOUBLE3, kDouble3},
          {TYPE_ID_DOUBLE4, kDouble4},
          {TYPE_ID_QUATH, kQuath},
          {TYPE_ID_QUATF, kQuatf},
          {TYPE_ID_QUATD, kQuatd},
          {TYPE_ID_VECTOR3H, kVector3h},
          {TYPE_ID_VECTOR3F, kVector3f},
          {TYPE_ID_VECTOR3D, kVector3d},
          {TYPE_ID_POINT3H, kPoint3h},
          {TYPE_ID_POINT3F, kPoint3f},
          {TYPE_ID_POINT3D, kPoint3d},
          {TYPE_ID_NORMAL3H, kNormal3h},
          {TYPE_ID_NORMAL3F, kNormal3f},
          {TYPE_ID_NORMAL3D, kNormal3d},
          {TYPE_ID_COLOR3F, kColor3f},
          {TYPE_ID_COLOR3D, kColor3d},
          {TYPE_ID_COLOR4F, kColor4f},
          {TYPE_ID_COLOR4D, kColor4d},
          {TYPE_ID_FRAME4D, kFrame4d},
          {TYPE_ID_TEXCOORD2H, kTexCoord2h},
          {TYPE_ID_TEXCOORD2F, kTexCoord2f},
          {TYPE_ID_TEXCOORD2D, kTexCoord2d},
          {TYPE_ID_TEXCOORD3H, kTexCoord3h},
          {TYPE_ID_TEXCOORD3F, kTexCoord3f},
          {TYPE_ID_TEXCOORD3D, kTexCoord3d},
          {TYPE_ID_RELATIONSHIP, kRelationship},
      });

  bool array_bit = (TYPE_ID_1D_ARRAY_BIT & tyid);
  uint32_t scalar_tid = tyid & (~TYPE_ID_1D_ARRAY_BIT);

  auto ret = tynamemap.find(scalar_tid);
  if (ret != tynamemap.end()) {
    std::string s = ret->second.c_str();
    if (array_bit) {
      s += "[]";
    }
    return std::move(s);
  }

  return nonstd::nullopt;
}

std::string GetTypeName(uint32_t tyid) {
  auto ret = TryGetTypeName(tyid);

  if (!ret) {
    return "(GetTypeName) [[Unknown or unimplemented/unsupported type_id: " +
           std::to_string(tyid) + "]]";
  }

  return ret.value();
}

nonstd::optional<uint32_t> TryGetTypeId(const std::string &tyname) {
  MAPBOX_ETERNAL_CONSTEXPR const auto tyidmap =
      mapbox::eternal::hash_map<mapbox::eternal::string, uint32_t>({
          {kToken, TYPE_ID_TOKEN},
          {kString, TYPE_ID_STRING},
          {kPath, TYPE_ID_STRING},
          {kAssetPath, TYPE_ID_ASSET_PATH},
          {kDictionary, TYPE_ID_DICT},
          {kTimeCode, TYPE_ID_TIMECODE},
          {kBool, TYPE_ID_BOOL},
          {kUChar, TYPE_ID_UCHAR},
          {kHalf, TYPE_ID_HALF},
          {kInt, TYPE_ID_INT32},
          {kUInt, TYPE_ID_UINT32},
          {kInt64, TYPE_ID_INT64},
          {kUInt64, TYPE_ID_UINT64},
          {kInt2, TYPE_ID_INT2},
          {kInt3, TYPE_ID_INT3},
          {kInt4, TYPE_ID_INT4},
          {kUInt2, TYPE_ID_UINT2},
          {kUInt3, TYPE_ID_UINT3},
          {kUInt4, TYPE_ID_UINT4},
          {kHalf2, TYPE_ID_HALF2},
          {kHalf3, TYPE_ID_HALF3},
          {kHalf4, TYPE_ID_HALF4},
          {kMatrix2d, TYPE_ID_MATRIX2D},
          {kMatrix3d, TYPE_ID_MATRIX3D},
          {kMatrix4d, TYPE_ID_MATRIX4D},
          {kFloat, TYPE_ID_FLOAT},
          {kFloat2, TYPE_ID_FLOAT2},
          {kFloat3, TYPE_ID_FLOAT3},
          {kFloat4, TYPE_ID_FLOAT4},
          {kDouble, TYPE_ID_DOUBLE},
          {kDouble2, TYPE_ID_DOUBLE2},
          {kDouble3, TYPE_ID_DOUBLE3},
          {kDouble4, TYPE_ID_DOUBLE4},
          {kQuath, TYPE_ID_QUATH},
          {kQuatf, TYPE_ID_QUATF},
          {kQuatd, TYPE_ID_QUATD},
          {kVector3h, TYPE_ID_VECTOR3H},
          {kVector3f, TYPE_ID_VECTOR3F},
          {kVector3d, TYPE_ID_VECTOR3D},
          {kPoint3h, TYPE_ID_POINT3H},
          {kPoint3f, TYPE_ID_POINT3F},
          {kPoint3d, TYPE_ID_POINT3D},
          {kNormal3h, TYPE_ID_NORMAL3H},
          {kNormal3f, TYPE_ID_NORMAL3F},
          {kNormal3d, TYPE_ID_NORMAL3D},
          {kColor3f, TYPE_ID_COLOR3F},
          {kColor3d, TYPE_ID_COLOR3D},
          {kColor4f, TYPE_ID_COLOR4F},
          {kColor4d, TYPE_ID_COLOR4D},
          {kFrame4d, TYPE_ID_FRAME4D},
          {kTexCoord2h, TYPE_ID_TEXCOORD2H},
          {kTexCoord2f, TYPE_ID_TEXCOORD2F},
          {kTexCoord2d, TYPE_ID_TEXCOORD2D},
          {kTexCoord3h, TYPE_ID_TEXCOORD3H},
          {kTexCoord3f, TYPE_ID_TEXCOORD3F},
          {kTexCoord3d, TYPE_ID_TEXCOORD3D},
          {kRelationship, TYPE_ID_RELATIONSHIP},
      });

  std::string s = tyname;
  uint32_t array_bit = 0;
  if (endsWith(tyname, "[]")) {
    s = removeSuffix(s, "[]");
    array_bit |= TYPE_ID_1D_ARRAY_BIT;
  }

  // It looks USD does not support 2D array type, so no further `[]` check

  auto ret = tyidmap.find(s.c_str());
  if (ret != tyidmap.end()) {
    return ret->second | array_bit;
  }

  return nonstd::nullopt;
}

uint32_t GetTypeId(const std::string &tyname) {
  auto ret = TryGetTypeId(tyname);

  if (!ret) {
    return TYPE_ID_INVALID;
  }

  return ret.value();
}

nonstd::optional<uint32_t> TryGetUnderlyingTypeId(const std::string &tyname) {
  MAPBOX_ETERNAL_CONSTEXPR const auto utyidmap =
      mapbox::eternal::hash_map<mapbox::eternal::string, uint32_t>({
        {kPoint3h, TYPE_ID_HALF3},
        {kPoint3f, TYPE_ID_FLOAT3},
        {kPoint3d, TYPE_ID_DOUBLE3},
        {kNormal3h, TYPE_ID_HALF3},
        {kNormal3f, TYPE_ID_FLOAT3},
        {kNormal3d, TYPE_ID_DOUBLE3},
        {kVector3h, TYPE_ID_HALF3},
        {kVector3f, TYPE_ID_FLOAT3},
        {kVector3d, TYPE_ID_DOUBLE3},
        {kColor3h, TYPE_ID_HALF3},
        {kColor3f, TYPE_ID_FLOAT3},
        {kColor3d, TYPE_ID_DOUBLE3},
        {kColor4h, TYPE_ID_HALF4},
        {kColor4f, TYPE_ID_FLOAT4},
        {kColor4d, TYPE_ID_DOUBLE4},
        {kTexCoord2h, TYPE_ID_HALF2},
        {kTexCoord2f, TYPE_ID_FLOAT2},
        {kTexCoord2d, TYPE_ID_DOUBLE3},
        {kTexCoord3h, TYPE_ID_HALF3},
        {kTexCoord3f, TYPE_ID_FLOAT3},
        {kTexCoord3d, TYPE_ID_DOUBLE4},
        {kFrame4d, TYPE_ID_MATRIX4D},
  });

  {
    std::string s = tyname;
    uint32_t array_bit = 0;
    if (endsWith(tyname, "[]")) {
      s = removeSuffix(s, "[]");
      array_bit |= TYPE_ID_1D_ARRAY_BIT;
    }

    auto ret = utyidmap.find(s.c_str());
    if (ret != utyidmap.end()) {
      return ret->second | array_bit;
    }
  }

  // Fallback
  return TryGetTypeId(tyname);
}

uint32_t GetUnderlyingTypeId(const std::string &tyname) {
  auto ret = TryGetUnderlyingTypeId(tyname);

  if (!ret) {
    return TYPE_ID_INVALID;
  }

  return ret.value();
}

nonstd::optional<std::string> TryGetUnderlyingTypeName(const uint32_t tyid) {
  MAPBOX_ETERNAL_CONSTEXPR const auto utynamemap =
      mapbox::eternal::map<uint32_t, mapbox::eternal::string>({
        {TYPE_ID_POINT3H, kHalf3},
        {TYPE_ID_POINT3F, kFloat3},
        {TYPE_ID_POINT3D, kDouble3},
        {TYPE_ID_NORMAL3H, kHalf3},
        {TYPE_ID_NORMAL3F, kFloat3},
        {TYPE_ID_NORMAL3D, kDouble3},
        {TYPE_ID_VECTOR3H, kHalf3},
        {TYPE_ID_VECTOR3F, kFloat3},
        {TYPE_ID_VECTOR3D, kDouble3},
        {TYPE_ID_COLOR3H, kHalf3},
        {TYPE_ID_COLOR3F, kFloat3},
        {TYPE_ID_COLOR3D, kDouble3},
        {TYPE_ID_COLOR4H, kHalf4},
        {TYPE_ID_COLOR4F, kFloat4},
        {TYPE_ID_COLOR4D, kDouble4},
        {TYPE_ID_TEXCOORD2H, kHalf2},
        {TYPE_ID_TEXCOORD2F, kFloat2},
        {TYPE_ID_TEXCOORD2D, kDouble2},
        {TYPE_ID_TEXCOORD3H, kHalf3},
        {TYPE_ID_TEXCOORD3F, kFloat3},
        {TYPE_ID_TEXCOORD3D, kDouble3},
        {TYPE_ID_FRAME4D, kMatrix4d},
  });

  {
  bool array_bit = (TYPE_ID_1D_ARRAY_BIT & tyid);
  uint32_t scalar_tid = tyid & (~TYPE_ID_1D_ARRAY_BIT);

  auto ret = utynamemap.find(scalar_tid);
  if (ret != utynamemap.end()) {
    std::string s = ret->second.c_str();
    if (array_bit) {
      s += "[]";
    }
    return std::move(s);
  }
  }

  return TryGetTypeName(tyid);

}

std::string GetUnderlyingTypeName(uint32_t tyid) {
  auto ret = TryGetUnderlyingTypeName(tyid);

  if (!ret) {
    return "(GetUnderlyingTypeName) [[Unknown or unimplemented/unsupported type_id: " +
           std::to_string(tyid) + "]]";
  }

  return ret.value();
}

bool IsRoleType(const std::string &tyname) {
  return GetUnderlyingTypeId(tyname) != value::TYPE_ID_INVALID;
}

bool IsRoleType(const uint32_t tyid) {
  return GetUnderlyingTypeId(GetTypeName(tyid)) != value::TYPE_ID_INVALID;
}

//
// half float
//
namespace {

// https://www.realtime.bc.ca/articles/endian-safe.html
union HostEndianness {
  int i;
  char c[sizeof(int)];

  HostEndianness() : i(1) {}

  bool isBig() const { return c[0] == 0; }
  bool isLittle() const { return c[0] != 0; }
};

// https://gist.github.com/rygorous/2156668
// Little endian
union FP32le {
  unsigned int u;
  float f;
  struct {
    unsigned int Mantissa : 23;
    unsigned int Exponent : 8;
    unsigned int Sign : 1;
  } s;
};

// Big endian
union FP32be {
  unsigned int u;
  float f;
  struct {
    unsigned int Sign : 1;
    unsigned int Exponent : 8;
    unsigned int Mantissa : 23;
  } s;
};

// Little endian
union float16le {
  unsigned short u;
  struct {
    unsigned int Mantissa : 10;
    unsigned int Exponent : 5;
    unsigned int Sign : 1;
  } s;
};

// Big endian
union float16be {
  unsigned short u;
  struct {
    unsigned int Sign : 1;
    unsigned int Exponent : 5;
    unsigned int Mantissa : 10;
  } s;
};

float half_to_float_le(float16le h) {
  static const FP32le magic = {113 << 23};
  static const unsigned int shifted_exp = 0x7c00
                                          << 13;  // exponent mask after shift
  FP32le o;

  o.u = (h.u & 0x7fffU) << 13U;           // exponent/mantissa bits
  unsigned int exp_ = shifted_exp & o.u;  // just the exponent
  o.u += (127 - 15) << 23;                // exponent adjust

  // handle exponent special cases
  if (exp_ == shifted_exp)    // Inf/NaN?
    o.u += (128 - 16) << 23;  // extra exp adjust
  else if (exp_ == 0)         // Zero/Denormal?
  {
    o.u += 1 << 23;  // extra exp adjust
    o.f -= magic.f;  // renormalize
  }

  o.u |= (h.u & 0x8000U) << 16U;  // sign bit
  return o.f;
}

float half_to_float_be(float16be h) {
  static const FP32be magic = {113 << 23};
  static const unsigned int shifted_exp = 0x7c00
                                          << 13;  // exponent mask after shift
  FP32be o;

  o.u = (h.u & 0x7fffU) << 13U;           // exponent/mantissa bits
  unsigned int exp_ = shifted_exp & o.u;  // just the exponent
  o.u += (127 - 15) << 23;                // exponent adjust

  // handle exponent special cases
  if (exp_ == shifted_exp)    // Inf/NaN?
    o.u += (128 - 16) << 23;  // extra exp adjust
  else if (exp_ == 0)         // Zero/Denormal?
  {
    o.u += 1 << 23;  // extra exp adjust
    o.f -= magic.f;  // renormalize
  }

  o.u |= (h.u & 0x8000U) << 16U;  // sign bit
  return o.f;
}

half float_to_half_full_be(float _f) {
  FP32be f;
  f.f = _f;
  float16be o = {0};

  // Based on ISPC reference code (with minor modifications)
  if (f.s.Exponent == 0)  // Signed zero/denormal (which will underflow)
    o.s.Exponent = 0;
  else if (f.s.Exponent == 255)  // Inf or NaN (all exponent bits set)
  {
    o.s.Exponent = 31;
    o.s.Mantissa = f.s.Mantissa ? 0x200 : 0;  // NaN->qNaN and Inf->Inf
  } else                                      // Normalized number
  {
    // Exponent unbias the single, then bias the halfp
    int newexp = f.s.Exponent - 127 + 15;
    if (newexp >= 31)  // Overflow, return signed infinity
      o.s.Exponent = 31;
    else if (newexp <= 0)  // Underflow
    {
      if ((14 - newexp) <= 24)  // Mantissa might be non-zero
      {
        unsigned int mant = f.s.Mantissa | 0x800000;  // Hidden 1 bit
        o.s.Mantissa = mant >> (14 - newexp);
        if ((mant >> (13 - newexp)) & 1)  // Check for rounding
          o.u++;  // Round, might overflow into exp bit, but this is OK
      }
    } else {
      o.s.Exponent = static_cast<unsigned int>(newexp);
      o.s.Mantissa = f.s.Mantissa >> 13;
      if (f.s.Mantissa & 0x1000)  // Check for rounding
        o.u++;                    // Round, might overflow to inf, this is OK
    }
  }

  o.s.Sign = f.s.Sign;

  half ret;
  ret.value = (*reinterpret_cast<const uint16_t *>(&o));

  return ret;
}

half float_to_half_full_le(float _f) {
  FP32le f;
  f.f = _f;
  float16le o = {0};

  // Based on ISPC reference code (with minor modifications)
  if (f.s.Exponent == 0)  // Signed zero/denormal (which will underflow)
    o.s.Exponent = 0;
  else if (f.s.Exponent == 255)  // Inf or NaN (all exponent bits set)
  {
    o.s.Exponent = 31;
    o.s.Mantissa = f.s.Mantissa ? 0x200 : 0;  // NaN->qNaN and Inf->Inf
  } else                                      // Normalized number
  {
    // Exponent unbias the single, then bias the halfp
    int newexp = f.s.Exponent - 127 + 15;
    if (newexp >= 31)  // Overflow, return signed infinity
      o.s.Exponent = 31;
    else if (newexp <= 0)  // Underflow
    {
      if ((14 - newexp) <= 24)  // Mantissa might be non-zero
      {
        unsigned int mant = f.s.Mantissa | 0x800000;  // Hidden 1 bit
        o.s.Mantissa = mant >> (14 - newexp);
        if ((mant >> (13 - newexp)) & 1)  // Check for rounding
          o.u++;  // Round, might overflow into exp bit, but this is OK
      }
    } else {
      o.s.Exponent = static_cast<unsigned int>(newexp);
      o.s.Mantissa = f.s.Mantissa >> 13;
      if (f.s.Mantissa & 0x1000)  // Check for rounding
        o.u++;                    // Round, might overflow to inf, this is OK
    }
  }

  o.s.Sign = f.s.Sign;

  half ret;
  ret.value = (*reinterpret_cast<const uint16_t *>(&o));
  return ret;
}

}  // namespace

float half_to_float(half h) {
  // TODO: Compile time detection of endianness
  HostEndianness endian;

  if (endian.isBig()) {
    float16be f;
    f.u = h.value;
    return half_to_float_be(f);
  } else if (endian.isLittle()) {
    float16le f;
    f.u = h.value;
    return half_to_float_le(f);
  }

  ///???
  return std::numeric_limits<float>::quiet_NaN();
}

half float_to_half_full(float _f) {
  // TODO: Compile time detection of endianness
  HostEndianness endian;

  if (endian.isBig()) {
    return float_to_half_full_be(_f);
  } else if (endian.isLittle()) {
    return float_to_half_full_le(_f);
  }

  ///???
  half fp16{0};  // TODO: Raise exception or return NaN
  return fp16;
}

matrix2f::matrix2f(const matrix2d &src) {
  (*this) = src;
}

matrix2f &matrix2f::operator=(const matrix2d &src) {

  for (size_t j = 0; j < 2; j++) {
    for (size_t i = 0; i < 2; i++) {
      m[j][i] = float(src.m[j][i]);
    }
  }

  return *this;
}

matrix3f::matrix3f(const matrix3d &src) {
  (*this) = src;
}

matrix3f &matrix3f::operator=(const matrix3d &src) {

  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      m[j][i] = float(src.m[j][i]);
    }
  }

  return *this;
}

matrix4f::matrix4f(const matrix4d &src) {
  (*this) = src;
}

matrix4f &matrix4f::operator=(const matrix4d &src) {

  for (size_t j = 0; j < 4; j++) {
    for (size_t i = 0; i < 4; i++) {
      m[j][i] = float(src.m[j][i]);
    }
  }

  return *this;
}

matrix2d &matrix2d::operator=(const matrix2f &src) {

  for (size_t j = 0; j < 2; j++) {
    for (size_t i = 0; i < 2; i++) {
      m[j][i] = double(src.m[j][i]);
    }
  }

  return *this;
}

matrix3d &matrix3d::operator=(const matrix3f &src) {

  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      m[j][i] = double(src.m[j][i]);
    }
  }

  return *this;
}

matrix4d &matrix4d::operator=(const matrix4f &src) {

  for (size_t j = 0; j < 4; j++) {
    for (size_t i = 0; i < 4; i++) {
      m[j][i] = double(src.m[j][i]);
    }
  }

  return *this;
}


size_t Value::array_size() const {
  if (!is_array()) {
    return 0;
  }

  // primvar types only.

#define APPLY_FUNC_TO_TYPES(__FUNC) \
  __FUNC(bool)                 \
  __FUNC(value::token)                 \
  __FUNC(std::string)                 \
  __FUNC(StringData)                 \
  __FUNC(half)                 \
  __FUNC(half2)                \
  __FUNC(half3)                \
  __FUNC(half4)                \
  __FUNC(int32_t)              \
  __FUNC(uint32_t)             \
  __FUNC(int2)                 \
  __FUNC(int3)                 \
  __FUNC(int4)                 \
  __FUNC(uint2)                \
  __FUNC(uint3)                \
  __FUNC(uint4)                \
  __FUNC(int64_t)              \
  __FUNC(uint64_t)             \
  __FUNC(float)                \
  __FUNC(float2)               \
  __FUNC(float3)               \
  __FUNC(float4)               \
  __FUNC(double)               \
  __FUNC(double2)              \
  __FUNC(double3)              \
  __FUNC(double4)              \
  __FUNC(quath)                \
  __FUNC(quatf)                \
  __FUNC(quatd)                \
  __FUNC(normal3h)             \
  __FUNC(normal3f)             \
  __FUNC(normal3d)             \
  __FUNC(vector3h)             \
  __FUNC(vector3f)             \
  __FUNC(vector3d)             \
  __FUNC(point3h)              \
  __FUNC(point3f)              \
  __FUNC(point3d)              \
  __FUNC(color3f)              \
  __FUNC(color3d)              \
  __FUNC(color4h)              \
  __FUNC(color4f)              \
  __FUNC(color4d)              \
  __FUNC(texcoord2h)           \
  __FUNC(texcoord2f)           \
  __FUNC(texcoord2d)           \
  __FUNC(texcoord3h)           \
  __FUNC(texcoord3f)           \
  __FUNC(texcoord3d) \
  __FUNC(matrix2d) \
  __FUNC(matrix3d) \
  __FUNC(matrix4d) \
  __FUNC(frame4d)

#define ARRAY_SIZE_GET(__ty) case value::TypeTraits<__ty>::type_id() | value::TYPE_ID_1D_ARRAY_BIT: { \
    if (auto pv = v_.cast<std::vector<__ty>>()) { \
      return pv->size(); \
    } \
    return 0; \
  }


  switch (v_.type_id()) {
    APPLY_FUNC_TO_TYPES(ARRAY_SIZE_GET)
    default:
      return 0;
  }

#undef ARRAY_SIZE_GET
#undef APPLY_FUNC_TO_TYPES

}

bool RoleTypeCast(const uint32_t roleTyId, value::Value &inout) {
  const uint32_t srcUnderlyingTyId = inout.underlying_type_id();

  DCOUT("input type = " << inout.type_name());

  // scalar and array
#define ROLE_TYPE_CAST(__roleTy, __srcBaseTy)                                  \
  {                                                                            \
    static_assert(value::TypeTraits<__roleTy>::size() ==                       \
                      value::TypeTraits<__srcBaseTy>::size(),                  \
                  "");                                                         \
    if (srcUnderlyingTyId == value::TypeTraits<__srcBaseTy>::type_id()) {      \
      if (roleTyId == value::TypeTraits<__roleTy>::type_id()) {                \
        if (auto pv = inout.get_value<__srcBaseTy>()) {                        \
          __srcBaseTy val = pv.value();                                        \
          __roleTy newval;                                                     \
          memcpy(reinterpret_cast<__srcBaseTy *>(&newval), &val, sizeof(__srcBaseTy));                          \
          inout = newval;                                                      \
          return true;                                                         \
        }                                                                      \
      }                                                                        \
    } else if (srcUnderlyingTyId ==                                            \
               (value::TypeTraits<__srcBaseTy>::type_id() |                    \
                value::TYPE_ID_1D_ARRAY_BIT)) {                                \
      if (roleTyId == value::TypeTraits<std::vector<__roleTy>>::type_id()) {   \
        if (auto pv = inout.get_value<std::vector<__srcBaseTy>>()) {           \
          std::vector<__srcBaseTy> val = pv.value();                           \
          std::vector<__roleTy> newval;                                        \
          newval.resize(val.size());                                           \
          memcpy(reinterpret_cast<__srcBaseTy *>(newval.data()), val.data(), sizeof(__srcBaseTy) * val.size()); \
          inout = newval;                                                      \
          return true;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

  ROLE_TYPE_CAST(value::texcoord2h, value::half2)
  ROLE_TYPE_CAST(value::texcoord2f, value::float2)
  ROLE_TYPE_CAST(value::texcoord2d, value::double2)

  ROLE_TYPE_CAST(value::texcoord3h, value::half3)
  ROLE_TYPE_CAST(value::texcoord3f, value::float3)
  ROLE_TYPE_CAST(value::texcoord3d, value::double3)

  ROLE_TYPE_CAST(value::normal3h, value::half3)
  ROLE_TYPE_CAST(value::normal3f, value::float3)
  ROLE_TYPE_CAST(value::normal3d, value::double3)

  ROLE_TYPE_CAST(value::vector3h, value::half3)
  ROLE_TYPE_CAST(value::vector3f, value::float3)
  ROLE_TYPE_CAST(value::vector3d, value::double3)

  ROLE_TYPE_CAST(value::point3h, value::half3)
  ROLE_TYPE_CAST(value::point3f, value::float3)
  ROLE_TYPE_CAST(value::point3d, value::double3)

  ROLE_TYPE_CAST(value::color3h, value::half3)
  ROLE_TYPE_CAST(value::color3f, value::float3)
  ROLE_TYPE_CAST(value::color3d, value::double3)

  ROLE_TYPE_CAST(value::color4h, value::half4)
  ROLE_TYPE_CAST(value::color4f, value::float4)
  ROLE_TYPE_CAST(value::color4d, value::double4)

  ROLE_TYPE_CAST(value::frame4d, value::matrix4d)

#undef ROLE_TYPE_CAST

  return false;
}

// TODO: Use template
bool UpcastType(const std::string &reqType, value::Value &inout) {
  // `reqType` may be Role type. Get underlying type
  uint32_t tyid;
  if (auto pv = value::TryGetUnderlyingTypeId(reqType)) {
    tyid = pv.value();
  } else {
    // Invalid reqType.
    return false;
  }

  bool reqTypeArray = false;
  //uint32_t baseReqTyId;
  DCOUT("UpcastType trial: reqTy : " << reqType
                                     << ", valtype = " << inout.type_name());

  if (endsWith(reqType, "[]")) {
    reqTypeArray = true;
    //baseReqTyId = value::GetTypeId(removeSuffix(reqType, "[]"));
  } else {
    //baseReqTyId = value::GetTypeId(reqType);
  }
  DCOUT("is array: " << reqTypeArray);

  // For array
  if (reqTypeArray) {
    // TODO
  } else {
    if (tyid == value::TYPE_ID_FLOAT) {
      float dst;
      if (auto pv = inout.get_value<value::half>()) {
        dst = half_to_float(pv.value());
        inout = dst;
        return true;
      }
    } else if (tyid == value::TYPE_ID_FLOAT2) {
      if (auto pv = inout.get_value<value::half2>()) {
        value::float2 dst;
        value::half2 v = pv.value();
        dst[0] = half_to_float(v[0]);
        dst[1] = half_to_float(v[1]);
        inout = dst;
        return true;
      }

    } else if (tyid == value::TYPE_ID_FLOAT3) {
      value::float3 dst;
      if (auto pv = inout.get_value<value::half3>()) {
        value::half3 v = pv.value();
        dst[0] = half_to_float(v[0]);
        dst[1] = half_to_float(v[1]);
        dst[2] = half_to_float(v[2]);
        inout = dst;
        return true;
      }
    } else if (tyid == value::TYPE_ID_FLOAT4) {
      value::float4 dst;
      if (auto pv = inout.get_value<value::half4>()) {
        value::half4 v = pv.value();
        dst[0] = half_to_float(v[0]);
        dst[1] = half_to_float(v[1]);
        dst[2] = half_to_float(v[2]);
        dst[3] = half_to_float(v[3]);
        inout = dst;
        return true;
      }
    } else if (tyid == value::TYPE_ID_DOUBLE) {
      double dst;
      if (auto pv = inout.get_value<value::half>()) {
        dst = double(half_to_float(pv.value()));
        inout = dst;
        return true;
      }
    } else if (tyid == value::TYPE_ID_DOUBLE2) {
      value::double2 dst;
      if (auto pv = inout.get_value<value::half2>()) {
        value::half2 v = pv.value();
        dst[0] = double(half_to_float(v[0]));
        dst[1] = double(half_to_float(v[1]));
        inout = dst;
        return true;
      }
    } else if (tyid == value::TYPE_ID_DOUBLE3) {
      value::double3 dst;
      if (auto pv = inout.get_value<value::half3>()) {
        value::half3 v = pv.value();
        dst[0] = double(half_to_float(v[0]));
        dst[1] = double(half_to_float(v[1]));
        dst[2] = double(half_to_float(v[2]));
        inout = dst;
        return true;
      }
    } else if (tyid == value::TYPE_ID_DOUBLE4) {
      value::double4 dst;
      if (auto pv = inout.get_value<value::half4>()) {
        value::half4 v = pv.value();
        dst[0] = double(half_to_float(v[0]));
        dst[1] = double(half_to_float(v[1]));
        dst[2] = double(half_to_float(v[2]));
        dst[3] = double(half_to_float(v[3]));
        inout = dst;
        return true;
      }
    }
  }

  return false;
}

#if 0
bool FlexibleTypeCast(const value::Value &src, value::Value &dst) {
  uint32_t src_utype_id = src.type_id();
  uint32_t dst_utype_id = src.type_id();

  if (src_utype_id == value::TypeTraits<int32_t>::type_id()) {

  }

  // TODO

  return false;
}
#endif

bool TimeSamples::has_sample_at(const double t) const {
  if (_dirty) {
    update();
  }

  const auto it = std::find_if(_samples.begin(), _samples.end(), [&t](const Sample &s) {
    return math::is_close(t, s.t);
  });

  return (it != _samples.end());
}

bool TimeSamples::get_sample_at(const double t, Sample **dst) {
  if (!dst) {
    return false;
  }

  if (_dirty) {
    update();
  }

  const auto it = std::find_if(_samples.begin(), _samples.end(), [&t](const Sample &sample) {
    return math::is_close(t, sample.t);
  });

  if (it != _samples.end()) {
    (*dst) = &(*it); 
  }
  return false;
}

}  // namespace value
}  // namespace tinyusdz
