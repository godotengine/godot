//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_RENDER_TYPES_H
#define MATERIALX_RENDER_TYPES_H

/// @file
/// Data types for rendering functionality

#include <MaterialXRender/Export.h>

#include <MaterialXCore/Types.h>

MATERIALX_NAMESPACE_BEGIN

/// @class Quaternion
/// A quaternion vector
class MX_RENDER_API Quaternion : public VectorN<Vector4, float, 4>
{
  public:
    using VectorN<Vector4, float, 4>::VectorN;
    Quaternion() = default;
    Quaternion(float x, float y, float z, float w) :
        VectorN(Uninit{})
    {
        _arr = { x, y, z, w };
    }

    Quaternion operator*(const Quaternion& q) const
    {
        return {
            _arr[0] * q._arr[3] + _arr[3] * q._arr[0] + _arr[1] * q._arr[2] - _arr[2] * q._arr[1],
            _arr[1] * q._arr[3] + _arr[3] * q._arr[1] + _arr[2] * q._arr[0] - _arr[0] * q._arr[2],
            _arr[2] * q._arr[3] + _arr[3] * q._arr[2] + _arr[0] * q._arr[1] - _arr[1] * q._arr[0],
            _arr[3] * q._arr[3] - _arr[0] * q._arr[0] - _arr[1] * q._arr[1] - _arr[2] * q._arr[2]
        };
    }

    Quaternion getNormalized() const
    {
        float l = 1.f / getMagnitude() * (_arr[3] < 0 ? -1.f : 1.f); // after normalization, real part will be non-negative
        return { _arr[0] * l, _arr[1] * l, _arr[2] * l, _arr[3] * l };
    }

    static Quaternion createFromAxisAngle(const Vector3& v, float a)
    {
        float s = std::sin(a * 0.5f);
        return Quaternion(v[0] * s, v[1] * s, v[2] * s, std::cos(a * 0.5f));
    }

    Matrix44 toMatrix() const;

  public:
    static const Quaternion IDENTITY;
};

/// @class Vector3d
/// A vector of three floating-point values (double-precision)
class MX_RENDER_API Vector3d : public VectorN<Vector3d, double, 3>
{
  public:
    using VectorN<Vector3d, double, 3>::VectorN;
    Vector3d() = default;
    Vector3d(double x, double y, double z) : VectorN(Uninit{})
    {
        _arr = { x, y, z };
    }
};

/// @class Vector4d
/// A vector of four floating-point values (double-precision)
class MX_RENDER_API Vector4d : public VectorN<Vector4d, double, 4>
{
  public:
    using VectorN<Vector4d, double, 4>::VectorN;
    Vector4d() = default;
    Vector4d(double x, double y, double z, double w) : VectorN(Uninit{})
    {
        _arr = { x, y, z, w };
    }
};

/// @class Color3d
/// A three-component color value (double-precision)
class MX_RENDER_API Color3d : public VectorN<Color3d, double, 3>
{
  public:
    using VectorN<Color3d, double, 3>::VectorN;
    Color3d() = default;
    Color3d(double r, double g, double b) : VectorN(Uninit{})
    {
        _arr = { r, g, b };
    }
};

/// @class Half
/// A lightweight 16-bit half-precision float class.  Based on the public-domain
/// implementation by Paul Tessier.
class MX_RENDER_API Half
{
  public:
    explicit Half(float value) : _data(toFloat16(value)) { }
    operator float() const { return toFloat32(_data); }

    bool operator==(Half rhs) const { return float(*this) == float(rhs); }
    bool operator!=(Half rhs) const { return float(*this) != float(rhs); }
    bool operator<(Half rhs) const { return float(*this) < float(rhs); }
    bool operator>(Half rhs) const { return float(*this) > float(rhs); }
    bool operator<=(Half rhs) const { return float(*this) <= float(rhs); }
    bool operator>=(Half rhs) const { return float(*this) >= float(rhs); }

    Half operator+(Half rhs) const { return Half(float(*this) + float(rhs)); }
    Half operator-(Half rhs) const { return Half(float(*this) - float(rhs)); }
    Half operator*(Half rhs) const { return Half(float(*this) * float(rhs)); }
    Half operator/(Half rhs) const { return Half(float(*this) / float(rhs)); }

    Half& operator+=(Half rhs) { return operator=(*this + rhs); }
    Half& operator-=(Half rhs) { return operator=(*this - rhs); }
    Half& operator*=(Half rhs) { return operator=(*this * rhs); }
    Half& operator/=(Half rhs) { return operator=(*this / rhs); }

    Half operator-() const { return Half(-float(*this)); }

  private:
    union Bits
    {
        float f;
        int32_t si;
        uint32_t ui;
    };

    static constexpr int const shift = 13;
    static constexpr int const shiftSign = 16;

    static constexpr int32_t const infN = 0x7F800000; // flt32 infinity
    static constexpr int32_t const maxN = 0x477FE000; // max flt16 normal as a flt32
    static constexpr int32_t const minN = 0x38800000; // min flt16 normal as a flt32
    static constexpr int32_t const signN = (int32_t) 0x80000000; // flt32 sign bit

    static constexpr int32_t const infC = infN >> shift;
    static constexpr int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
    static constexpr int32_t const maxC = maxN >> shift;
    static constexpr int32_t const minC = minN >> shift;
    static constexpr int32_t const signC = (int32_t) 0x00008000; // flt16 sign bit

    static constexpr int32_t const mulN = 0x52000000; // (1 << 23) / minN
    static constexpr int32_t const mulC = 0x33800000; // minN / (1 << (23 - shift))

    static constexpr int32_t const subC = 0x003FF; // max flt32 subnormal down shifted
    static constexpr int32_t const norC = 0x00400; // min flt32 normal down shifted

    static constexpr int32_t const maxD = infC - maxC - 1;
    static constexpr int32_t const minD = minC - subC - 1;

    static constexpr int32_t const maxF = 0x7FFFFFBF; // max int32 expressible as a flt32

    static uint16_t toFloat16(float value)
    {
        Bits v, s;
        v.f = value;
        uint32_t sign = (uint32_t) (v.si & signN);
        v.si ^= sign;
        sign >>= shiftSign; // logical shift
        s.si = mulN;
        int32_t subN = (int32_t) std::min(s.f * v.f, (float) maxF); // correct subnormals
        v.si ^= (subN ^ v.si) & -(minN > v.si);
        v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
        v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
        v.ui >>= shift; // logical shift
        v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
        v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
        return (uint16_t) (v.ui | sign);
    }

    static float toFloat32(uint16_t value)
    {
        Bits v;
        v.ui = value;
        int32_t sign = v.si & signC;
        v.si ^= sign;
        sign <<= shiftSign;
        v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
        v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
        Bits s;
        s.si = mulC;
        s.f *= float(v.si);
        int32_t mask = (norC > v.si) ? -1 : 1;
        v.si <<= shift;
        v.si ^= (s.si ^ v.si) & mask;
        v.si |= sign;
        return v.f;
    }

  private:
    uint16_t _data;
};

MATERIALX_NAMESPACE_END

#endif
