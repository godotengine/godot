#pragma once

#include "Runtime/Math/Simd/vec-matrix.h"
#include "Runtime/Math/Simd/vec-svd.h"
#include "Runtime/Math/Simd/vec-trs.h"
#include "Runtime/Math/Simd/vec-rigid.h"
#include "Runtime/Math/Simd/vec-affine.h"

namespace math
{
    static MATH_FORCEINLINE void convert(const trsX& from, trsX& to)
    {
        to = from;
    }

    static MATH_FORCEINLINE void convert(const affineX& from, affineX& to)
    {
        to = from;
    }

    static MATH_FORCEINLINE void convert(const trsX& from, affineX& to)
    {
        to = affineCompose(from.t, from.q, from.s);
    }

    static MATH_FORCEINLINE void convert(const rigidX& from, affineX& to)
    {
        to = affineCompose(from.ts.xyz, from.q, float3(from.ts.w));
    }

    static MATH_FORCEINLINE void inverse(const float4& fromR, const float3& fromS, float3x3 &to)
    {
        quatToMatrix(quatConj(fromR), to);
        to = scaleMul(inverseScale(fromS), to);
    }

    static MATH_FORCEINLINE void inverse(const trsX& from, affineX& to)
    {
        inverse(from.q, from.s, to.rs);
        to.t = mul(to.rs, -from.t);
    }

// trsX
    static MATH_FORCEINLINE float3 translation(const trsX& x)
    {
        return x.t;
    }

    static MATH_FORCEINLINE float4 rotation(trsX const &x)
    {
        return x.q;
    }

    static MATH_FORCEINLINE float3 scale(const trsX& x)
    {
        return x.s;
    }

// affineX
    static MATH_FORCEINLINE float3 translation(const affineX& x)
    {
        return x.t;
    }

// rigidX
    static MATH_FORCEINLINE float3 translation(const rigidX& x)
    {
        return x.ts.xyz;
    }

    static MATH_FORCEINLINE float4 rotation(const rigidX& x)
    {
        return x.q;
    }

    static MATH_FORCEINLINE float3 scale(const rigidX& x)
    {
        return float3(x.ts.w);
    }

    static MATH_FORCEINLINE float3x3 inverse(const float3x3& x)
    {
        float3x3 i;
        float1 scaleSquared = uniformScaleSquared(x);
        if (scaleSquared < float1(epsilon_normal()))
            return float3x3(float3(ZERO), float3(ZERO), float3(ZERO));
        float3 scaleInv(rsqrte(scaleSquared));
        float3x3 xs = mulScale(x, scaleInv);
        if (!adjInverse(xs, i, epsilon_determinant()))
        {
            i = svdInverse(xs);
        }
        i = mulScale(i, scaleInv);
        return i;
    }

    static MATH_FORCEINLINE affineX inverse(const affineX& x)
    {
        affineX i;
        i.rs = inverse(x.rs);
        i.t = mul(i.rs, -x.t);
        return i;
    }

// This can be used to extract the rotation matrix
// from a matrix that potentially has scale in it
    static MATH_FORCEINLINE float3x3 rotation(float3x3 const &i)
    {
        float detI = det(i);

        if (abs(1.0f - detI) < epsilon_determinant())
        {
            return i;
        }
        else if (abs(detI) > epsilon_determinant())
        {
            float3x3 is = mulScale(i, rsqrt(float3(dot(i.m0), dot(i.m1), dot(i.m2))));

            if (abs(1.0f - det(is)) < epsilon_determinant())
            {
                return is;
            }
        }

        float3x3 ret;
        math::quatToMatrix(svdRotation(i), ret);
        return ret;
    }
}
