#include "UnityPrefix.h"
#include "vec-svd.h"
#include "vec-quat.h"

namespace math
{
    static MATH_FORCEINLINE void condSwap(bool c, float1 &x, float1 &y)
    {
        float1 z = x;
        x = cond(c, y, x);
        y = cond(c, z, y);
    }

    static MATH_FORCEINLINE void condSwap(bool c, float3 &x, float3 &y)
    {
        float3 z = x;
        x = cond(c, y, x);
        y = cond(c, z, y);
    }

    static MATH_FORCEINLINE void condNegSwap(bool c, float3 &x, float3 &y)
    {
        float3 z = -x;
        x = cond(c, y, x);
        y = cond(c, z, y);
    }

    static MATH_FORCEINLINE float4 condNegSwapQuat(bool c, const float4 &q, const float4& mask)
    {
        const float halfSqrt2 = 0.707106781186548f;
        float4 qr = cond(c, float4(halfSqrt2) * mask, quatIdentity());
        return quatMul(q, qr);
    }

    static MATH_FORCEINLINE void sortSigularValues(float3x3 &b, float4 &v)
    {
        float1 l0 = dot(b.m0);
        float1 l1 = dot(b.m1);
        float1 l2 = dot(b.m2);

        bool c = l0 < l1;
        condNegSwap(c, b.m0, b.m1);
        v = condNegSwapQuat(c, v, float4(0, 0, 1, 1));
        condSwap(c, l0, l1);

        c = l0 < l2;
        condNegSwap(c, b.m0, b.m2);
        v = condNegSwapQuat(c, v, float4(0, -1, 0, 1));
        condSwap(c, l0, l2);

        c = l1 < l2;
        condNegSwap(c, b.m1, b.m2);
        v = condNegSwapQuat(c, v, float4(1, 0, 0, 1));
    }

    static MATH_FORCEINLINE float4 approxGivensQuat(const float3 &pq, const float4 &mask)
    {
        const float c8 = 0.923879532511287f; // cos(pi/8)
        const float s8 = 0.38268343236509f; // sin(pi/8)
        const float1 g = 5.82842712474619f; // 3 + 2 * sqrt(2)

        float1 ch = float1(2) * (pq.x - pq.y); // approx cos(a/2)
        float1 sh = pq.z;                   // approx sin(a/2)
        float4 r = cond(g * sh * sh < ch * ch, float4(sh, sh, sh, ch), float4(s8, s8, s8, c8)) * mask;
        return normalize(r);
    }

    static MATH_FORCEINLINE float4 jacobiIteration(float3x3 &s, uint32_t count = 5)
    {
        float3x3 qm;
        float4 q, v = quatIdentity();

        for (uint32_t iter = 0; iter < count; iter++)
        {
            q = approxGivensQuat(float3(s.m0.x, s.m1.y, s.m0.y), float4(0, 0, 1, 1));
            v = quatMul(v, q);
            quatToMatrix(q, qm);
            s = mul(mul(transpose(qm), s), qm);

            q = approxGivensQuat(float3(s.m1.y, s.m2.z, s.m1.z), float4(1, 0, 0, 1));
            v = quatMul(v, q);
            quatToMatrix(q, qm);
            s = mul(mul(transpose(qm), s), qm);

            q = approxGivensQuat(float3(s.m2.z, s.m0.x, s.m2.x), float4(0, 1, 0, 1));
            v = quatMul(v, q);
            quatToMatrix(q, qm);
            s = mul(mul(transpose(qm), s), qm);
        }

        return v;
    }

    static MATH_FORCEINLINE float4 qrGivensQuat(const float2 &pq, const float4 &mask)
    {
        float1 a1 = pq.x;
        float1 a2 = pq.y;
        float1 l = sqrt(float1(a1 * a1 + a2 * a2));
        float1 sh = cond(l > float1(epsilon_normal_sqrt()), a2, float1(ZERO));
        float1 ch = abs(a1) + max(l, float1(epsilon_normal_sqrt()));
        condSwap(a1 < float1(ZERO), sh, ch);
        float4 r = float4(sh, sh, sh, ch) * mask;
        return normalize(r);
    }

    static MATH_FORCEINLINE float4 givensQRFactorization(const float3x3 &b, float3x3 &r)
    {
        float4 u;
        float3x3 qmt;

        u = qrGivensQuat(float2(b.m0.x, b.m0.y), float4(0, 0, 1, 1));
        quatToMatrix(quatConj(u), qmt);
        r = mul(qmt, b);

        float4 q = qrGivensQuat(float2(r.m0.x, r.m0.z), float4(0, -1, 0, 1));
        u = quatMul(u, q);
        quatToMatrix(quatConj(q), qmt);
        r = mul(qmt, r);

        q = qrGivensQuat(float2(r.m1.y, r.m1.z), float4(1, 0, 0, 1));
        u = quatMul(u, q);
        quatToMatrix(quatConj(q), qmt);
        r = mul(qmt, r);

        return u;
    }

    static MATH_FORCEINLINE float3 singularValuesDecomposition(const float3x3 &a, float4 &u, float4 &v)
    {
        float3x3 e;
        float3x3 b;
        float3x3 s = mul(transpose(a), a);
        v = jacobiIteration(s);
        quatToMatrix(v, b);
        b = mul(a, b);
        sortSigularValues(b, v);
        u = givensQRFactorization(b, e);

        return float3(e.m0.x, e.m1.y, e.m2.z);
    }

    float3x3 svdInverse(const float3x3 &a)
    {
        float4 u;
        float4 v;

        float3 e = singularValuesDecomposition(a, u, v);

        float3x3 um, vm;
        quatToMatrix(u, um);
        quatToMatrix(v, vm);

        return mul(vm, scaleMul(inverseScale(e, epsilon_determinant()), transpose(um)));
    }

    float4 svdRotation(const float3x3 &a)
    {
        float4 u;
        float4 v;

        singularValuesDecomposition(a, u, v);

        return quatMul(u, quatConj(v));
    }
}
