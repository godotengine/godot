#pragma once

#include "./vec-types.h"
#include "./vec-quat.h"
#include "./vec-matrix.h"

namespace math
{
    struct affineX
    {
        MATH_EMPTYINLINE affineX() {}
        MATH_FORCEINLINE affineX(const affineX& x) { rs = x.rs; t = x.t; }
        MATH_FORCEINLINE affineX(const float3& inT, const float3x3& inRS) : rs(inRS), t(inT) {}
        MATH_FORCEINLINE affineX& operator=(const affineX& x) { rs = x.rs; t = x.t; return *this; }

        float3x3 rs;
        float3 t;
    };

    static MATH_FORCEINLINE affineX affineIdentity()
    {
        return affineX(float3(0, 0, 0), float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1)));
    }

    static MATH_FORCEINLINE float3 mul(const affineX& x, const float3& v)
    {
        return x.t + mul(x.rs, v);
    }

    static MATH_FORCEINLINE affineX mul(const affineX& a, const affineX& b)
    {
        return affineX(mul(a, b.t), mul(a.rs, b.rs));
    }

    static MATH_FORCEINLINE affineX mul(const float3x3& rs, const affineX& b)
    {
        return affineX(mul(rs, b.t), mul(rs, b.rs));
    }

    static MATH_FORCEINLINE affineX mul(const affineX& a, const float3x3& rs)
    {
        return affineX(a.t, mul(rs, a.rs));
    }

    static MATH_FORCEINLINE affineX mulScale(const affineX& x, const float3& v)
    {
        return affineX(x.t * v, mulScale(x.rs, v));
    }

    static MATH_FORCEINLINE affineX affineCompose(const float3& t, const float4& r, const float3& s)
    {
        affineX res;

        quatToMatrix(r, res.rs);
        res.rs = mulScale(res.rs, s);
        res.t = t;

        return res;
    }

    static MATH_FORCEINLINE affineX affineCompose(const float3& t, const float4& q)
    {
        affineX res;

        quatToMatrix(q, res.rs);
        res.t = t;

        return res;
    }

    static MATH_FORCEINLINE float4x4 affineTo4x4(const affineX& a)
    {
        return float4x4(float4(a.rs.m0, ZERO), float4(a.rs.m1, ZERO), float4(a.rs.m2, ZERO), float4(a.t, 1.0F));
    }

    static MATH_FORCEINLINE affineX float4x4ToAffine(const float4x4& a)
    {
        return affineX(a.m3.xyz, float3x3(a.m0.xyz, a.m1.xyz, a.m2.xyz));
    }
}
