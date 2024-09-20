#pragma once

#include "Runtime/Math/Simd/vec-types.h"
#include "Runtime/Math/Simd/vec-math.h"

namespace math
{
    static MATH_FORCEINLINE float1 uniformScaleSquared(float3x3 const &x)
    {
        return float1(0.333333f * (dot(x.m0) + dot(x.m1) + dot(x.m2)));
    }

    static MATH_FORCEINLINE float3 inverseScale(float3 const &s, float epsilon = epsilon_scale())
    {
        return select(rcp(s), float3(0.0f), abs(s) < float3(epsilon));
    }

    static MATH_FORCEINLINE float3x3 mulScale(float3x3 const &x, const float3 &s)
    {
        return float3x3(x.m0 * s.x, x.m1 * s.y, x.m2 * s.z);
    }

    static MATH_FORCEINLINE float3x3 scaleMul(const float3 &s, float3x3 const &x)
    {
        return float3x3(x.m0 * s, x.m1 * s, x.m2 * s);
    }

    static MATH_FORCEINLINE float3x3 transpose(float3x3 const &x)
    {
        return float3x3(float3(x.m0.x, x.m1.x, x.m2.x),
            float3(x.m0.y, x.m1.y, x.m2.y),
            float3(x.m0.z, x.m1.z, x.m2.z));
    }

    static MATH_FORCEINLINE float det(float3x3 const &x)
    {
        return dot(cross(x.m0, x.m1), x.m2);
    }

// returns adjoint matrix and computes determinant
    static MATH_FORCEINLINE float3x3 adj(float3x3 const &x, float &det)
    {
        float3x3 adjT;

        adjT.m0 = cross(x.m1, x.m2);
        adjT.m1 = cross(x.m2, x.m0);
        adjT.m2 = cross(x.m0, x.m1);

        det = dot(x.m0, adjT.m0);

        return transpose(adjT);
    }

// inverts a non singular matrix. returns false if matrix is singular and i is set to adjoint
// fastest inverse, when you know you deal with a non singular matrix
    static MATH_FORCEINLINE bool adjInverse(float3x3 const &x, float3x3 &i, float epsilon = epsilon_normal())
    {
        float det = 0;
        i = adj(x, det);
        bool c = abs(det) > epsilon;
        float3 detInv = cond(c, float3(rcp(det)), float3(1.0f));
        i = scaleMul(detInv, i);
        return c;
    }

    static MATH_FORCEINLINE float3 mul(const float3x3& x, const float3& v)
    {
        return x.m0 * v.x + mad(x.m1, v.y, x.m2 * v.z);
    }

    static MATH_FORCEINLINE float3x3 mul(const float3x3& a, const float3x3& b)
    {
        return float3x3(mul(a, b.m0), mul(a, b.m1), mul(a, b.m2));
    }

    static MATH_FORCEINLINE float4 mul(const float4x4& x, const float4& v)
    {
        return mad(x.m0, v.x, x.m1 * v.y) + mad(x.m2, v.z, x.m3 * v.w);
    }

    static MATH_FORCEINLINE float4x4 mul(const float4x4& a, const float4x4& b)
    {
        return float4x4(mul(a, b.m0), mul(a, b.m1), mul(a, b.m2), mul(a, b.m3));
    }
}
