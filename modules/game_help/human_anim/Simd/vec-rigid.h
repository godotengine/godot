#pragma once

#include "Runtime/Math/Simd/vec-types.h"
#include "Runtime/Math/Simd/vec-quat.h"

namespace math
{
    struct rigidX
    {
        MATH_EMPTYINLINE rigidX() {}
        MATH_FORCEINLINE rigidX(rigidX const& x) { ts = x.ts; q = x.q; }
        MATH_FORCEINLINE rigidX(float3 const& t, float4 const& quaternion, float1 const& s) : ts(float4(t, s)), q(quaternion) {}
        MATH_FORCEINLINE rigidX &operator=(rigidX const& x)
        { ts = x.ts; q = x.q; return *this; }

        float4 ts; // t = ts.xyz, s = ts.w
        //@TODO: Rename r
        float4 q;
    };

    static MATH_FORCEINLINE rigidX rigidIdentity()
    {
        return rigidX(float3(ZERO), quatIdentity(), float1(1.0F));
    }

    static MATH_FORCEINLINE float3 mul(rigidX const& x, float3 const& v)
    {
        float3 u = cross(x.q.xyz, v + v);
        u = v + x.q.w * u + cross(x.q.xyz, u);
        u = x.ts.w * u + x.ts.xyz;
        return u;
    }

    static MATH_FORCEINLINE rigidX mul(const rigidX& a, const rigidX& b)
    {
        return rigidX(mul(a, b.ts.xyz), normalize(quatMul(a.q, b.q)), a.ts.w * b.ts.w);
    }

    static MATH_FORCEINLINE rigidX inverse(const rigidX& x)
    {
        rigidX out;
        out.q = quatConj(x.q);
        float1 invScale = rcp(float1(x.ts.w));
        out.ts.w = invScale;
        out.ts.xyz = quatMulVec(out.q, -x.ts.xyz * invScale);
        return out;
    }

    static MATH_FORCEINLINE rigidX inverseNS(const rigidX& x)
    {
        float4 invQ = quatConj(x.q);
        return rigidX(quatMulVec(invQ, -x.ts.xyz), invQ, float1(1.0F));
    }

    static MATH_FORCEINLINE rigidX rigidCompose(const float3& t, const float4& q, const float1& s)
    {
        return rigidX(t, q, s);
    }

    static MATH_FORCEINLINE float4x4 rigidTo4x4(const rigidX& rx)
    {
        float3x3 rs;
        quatToMatrix(rx.q, rs);
        float1 s = rx.ts.w;
        rs.m0 *= s;
        rs.m1 *= s;
        rs.m2 *= s;

        return float4x4(float4(rs.m0, ZERO), float4(rs.m1, ZERO), float4(rs.m2, ZERO), float4(rx.ts.xyz, 1));
    }
}
