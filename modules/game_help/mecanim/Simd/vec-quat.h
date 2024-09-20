#pragma once

#ifndef _MATH_SIMD2_VEC_TYPES_H_
#   include "vec-types.h"
#endif

#ifndef _MATH_SIMD2_VEC_MATH_H_
#   include "vec-math.h"
#endif

#include "RotationOrder.h"
#include "vec-trig.h"

namespace math
{
    // Returns the identity quaternion.
    static MATH_FORCEINLINE float4 quatIdentity()
    {
        return float4(0.f, 0.f, 0.f, 1.f);
    }

    // returns the inverse of q.
    // * if q is normalized. The inverse quaternions is also normalized
    // * it is legal to call quatConj with non-normalized quaternion, the output will also be non-normalized
    static MATH_FORCEINLINE float4 quatConj(const float4& q)
    {
        //@TODO: If we switch to quaternion type instead of float4 type
        //       then we could rename this to inverse
        return chgsign(q, float4(-1.f, -1.f, -1.f, 1.f));
    }

    // normalizes a quaternion.
    // if q is 0, then it will return identity quaternion.
    // for this reason it is more expensive & safer than normalize
    static MATH_FORCEINLINE float4 quatNormalize(const float4& q)
    {
        float1 len = dot(q, q);

        //@TODO: We should use rsqrt here, but we have issues with rsqrt precision on ppc platforms.
        // Sonny will adress that first, before we can optimize this.

        // note we use float4 comparison here because this gives us -1 / 0 which is necessary for select.
        //return select(quatIdentity(), q*rsqrt(len), len > float4(epsilon_normal()));
        return select(quatIdentity(), q / sqrt(len), len > float4(epsilon_normal()));
    }

    // combines the two rotations
    // * if q1 and q2 are normalized. The output will also be normalized.
    // * it is legal to call quatMul with non-normalized quaternions, the output will also be non-normalized
    static MATH_FORCEINLINE float4 quatMul(const float4& q1, const float4& q2)
    {
        return chgsign((q1.ywzx * q2.xwyz - q1.wxyz * q2.zxzx - q1.zzww * q2.wzxy - q1.xyxy * q2.yyww).zwxy, float4(-1.f, -1.f, -1.f, 1.f));
    }

    // rotates a point/direction by the quaternion
    // q must be normalized
    static MATH_FORCEINLINE float3 quatMulVec(const float4& q, const float3& u)
    {
        const float3 c0 = float3(-2, +2, -2);
        const float3 c1 = float3(-2, -2, +2);
        const float3 c2 = float3(+2, -2, -2);

        float3 qyxw = q.yxw;
        float3 qzwx = q.zwx;
        float3 qwzy = q.wzy;

        float3 m0 = (c0 * q.y) * qyxw - (c2 * q.z) * qzwx;
        float3 m1 = (c1 * q.z) * qwzy - (c0 * q.x) * qyxw;
        float3 m2 = (c2 * q.x) * qzwx - (c1 * q.y) * qwzy;

        return float3((u + u.x * m0) + (u.y * m1 + u.z * m2));
    }

    // rotates a point/direction by the inverse(q)
    // q must be normalized
    static MATH_FORCEINLINE float3 quatInvMulVec(const float4 &q, const float3 &v)
    {
        return quatMulVec(quatConj(q), v);
    }

    // Interpolates two quaternions. blend must be in 0...1 range.
    // p and q input quaternions must be normalized.
    // Returns a normalized quaternion.
    static MATH_FORCEINLINE float4 quatLerp(const float4& p, const float4& q, const float1& blend)
    {
        return normalize(p + blend * (chgsign(q, dot(p, q)) - p));
    }

    // Creates a rotation which rotates from a to b.
    // a and b can be either normalized or unnormalized vector
    // Returns a normalized quaternion
    //  note: using two parallel vector pointing in opposite side yields NAN as there is an infinity of solution.
    static MATH_FORCEINLINE float4 quatArcRotate(const float3& a, const float3& b)
    {
        float4 aa = float4(a, ZERO);
        float4 bb = float4(b, ZERO);

        float4 q = cross(aa, bb);
        q.w = dot(aa, bb) + sqrt(float1(dot(aa, aa) * dot(bb, bb)));
        return normalize(q);
    }

    static MATH_FORCEINLINE float4 quatArcRotateX(const float4& n)
    {
        return float4(0.f, -n.z, n.y, n.x + 1.f);
    }

    static MATH_FORCEINLINE float3 quatXcos(const float4& qn)
    {
        const float4 qw = qn.w * qn - float4(0.f, 0.f, 0.f, .5f);
        const float4 u = qn.x * qn + float4(1.f, 1.f, -1.f, -1.f) * qw.wzyx;
        return (u + u).xyz;
    }

    static MATH_FORCEINLINE float3 quatYcos(const float4& qn)
    {
        const float4 qw = qn.w * qn - float4(0.f, 0.f, 0.f, .5f);
        const float4 v = qn.y * qn + float4(-1.f, 1.f, 1.f, -1.f) * qw.zwxy;
        return (v + v).xyz;
    }

    static MATH_FORCEINLINE float3 quatZcos(const float4& qn)
    {
        const float4 qw = qn.w * qn - float4(0.f, 0.f, 0.f, .5f);
        const float4 w = qn.z * qn + float4(1.f, -1.f, 1.f, -1.f) * qw.yxwz;

        return (w + w).xyz;
    }

    static const float4 kRotationOrderLUT[] =
    {
        float4(1.0f, 1.0f, 1.0f, 1.0f), float4(-1.0f, 1.0f, -1.0f, 1.0f), //XYZ
        float4(1.0f, 1.0f, 1.0f, 1.0f), float4(1.0f, 1.0f, -1.0f, -1.0f), //XZY
        float4(1.0f, -1.0f, 1.0f, 1.0f), float4(-1.0f, 1.0f, 1.0f, 1.0f), //YZX
        float4(1.0f, 1.0f, 1.0f, 1.0f), float4(-1.0f, 1.0f, 1.0f, -1.0f), //YXZ
        float4(1.0f, -1.0f, 1.0f, 1.0f), float4(1.0f, 1.0f, -1.0f, 1.0f), //ZXY
        float4(1.0f, -1.0f, 1.0f, 1.0f), float4(1.0f, 1.0f, 1.0f, -1.0f) //ZYX
    };

    static MATH_FORCEINLINE float4 eulerToQuat(const float3& euler, RotationOrder order = kOrderXYZ)
    {
        float3 c, s;
        sincos(euler * .5f, s, c);

        const float4 t = float4(s.x * c.z, s.x * s.z, c.x * s.z, c.x * c.z);

        return c.y * t * kRotationOrderLUT[2 * order] + s.y * kRotationOrderLUT[2 * order + 1] * t.zwxy;
    }

    float3 quatToEuler(const float4& q, RotationOrder order = kOrderXYZ);


    static MATH_FORCEINLINE float4 quatProjOnYPlane(const float4& q)
    {
        const float4 lQAlignUp = quatArcRotate(quatYcos(q), float3(0.f, 1.f, 0.f));
        return quatNormalize(quatMul(lQAlignUp, q) * float4(0.f, 1.f, 0.f, 1.f));
    }

    static MATH_FORCEINLINE float4 quatClampSmooth(const float4& q, float maxAngle)
    {
        float4 ret = q;

        float1 halfCosMaxAngle = cos(float1(0.5f * maxAngle));

        float4 qn = normalize(q);
        // faster (if the resulting sign would be not be cared about for qn.w == -0.f): qn = chgsign(qn, qn.w);
        qn = select(qn, -qn, qn.wwww < 0.f);

        if (qn.w < halfCosMaxAngle)
        {
            // The division is safe. q.w cannot be 1 (if it is 1, it cannot be smaller than halfCosMaxAngle, whose max value is 1)
            // in addition, halfCosMaxAngle cannot be 0, since it must be larger than q.w, and q.w's min value is 0.
            float1 fact_tmp = 1.f - (q.w * (1.f - halfCosMaxAngle)) / (halfCosMaxAngle * (1.f - q.w));
            // We cube the factor here, this smoothes the transitions at the edges of the clamp (when fact is close to 0).
            float1 fact = fact_tmp * fact_tmp * fact_tmp;

            ret = quatLerp(qn, quatIdentity(), fact);
        }

        return ret;
    }

    // returns the weighted quaternion. This is very useful for blending.
    // additive blending = quatMul(quatWeight(q1, 0.5F), quatWeight(q1, 0.5F))
    // q must be normalized
    // returns a normalized
    static MATH_FORCEINLINE float4 quatWeight(const float4& q, const float1& w)
    {
        return normalize(float4(q.x * w, q.y * w, q.z * w, q.w));
    }

    // a Qtan is a float3 parametric representation of a quaternion
    // used for blending rotations and other operations on rotations
    static MATH_FORCEINLINE float3 quat2Qtan(float4 const& q)
    {
        return select(q.xyz / q.w, q.xyz / float3(epsilon_radian()), abs(q.w) < epsilon_radian());
    }

    static MATH_FORCEINLINE float4 qtan2Quat(const float3& qxyz)
    {
        return normalize(float4(qxyz, 1.f));
    }

    static MATH_FORCEINLINE float4 ZYRoll2Quat(const float3& zyroll)
    {
        return normalize(float4(zyroll.x, zyroll.y + zyroll.x * zyroll.z, zyroll.z - zyroll.x * zyroll.y, 1.f));
    }

    static MATH_FORCEINLINE float3 quat2ZYRoll(const float4& q)
    {
        const float3 qtan = quat2Qtan(q);
        const float1 qtanx = qtan.x;
        const float1 x2p1 = 1.f + qtanx * qtanx;
        return float3(qtanx, (qtan.y - qtanx * qtan.z) / x2p1, (qtan.z + qtanx * qtan.y) / x2p1);
    }

    static MATH_FORCEINLINE float4 RollZY2Quat(const float3& zyroll)
    {
        return normalize(float4(zyroll.x, zyroll.y - zyroll.x * zyroll.z, zyroll.z + zyroll.x * zyroll.y, 1.f));
    }

    static MATH_FORCEINLINE float3 quat2RollZY(const float4& q)
    {
        const float3 qtan = quat2Qtan(q);
        const float1 qtanx = qtan.x;
        const float1 x2p1 = 1.f + qtanx * qtanx;
        return float3(qtanx, (qtan.y + qtanx * qtan.z) / x2p1, (qtan.z - qtanx * qtan.y) / x2p1);
    }

    static MATH_FORCEINLINE float4 axisAngleToQuat(const float3& axis, const float1& angle, const float1& axisLength = 1.0f)
    {
        DebugAssert((axisLength <= epsilon()) || (abs(dot(axis / axisLength) - 1.0f) <= epsilon()));

        float1 s, c;
        float1 halfLen = 0.5f * axisLength;
        sincos(angle * halfLen, s, c);

        return float4(s * axis, c);
    }

    static MATH_FORCEINLINE float3 quatToAngularDisplacement(const float4& q)
    {
        const float4 qn = normalize(q);
        const float1 len = length(qn.xyz);
        const float1 angle = 2.0f * asin(len);

        return select(qn.xyz * angle / len, math::float3(ZERO), float3(len) == 0.f);
    }

    static MATH_FORCEINLINE float4 angularDisplacementToQuat(const float3& d)
    {
        float1 len = length(d);
        float4 q = axisAngleToQuat(d, 1.0f, len);
        return select(float4(q.xyz / len, q.w), float4(0.f, 0.f, 0.f, 1.f), float4(len) == 0.f);
    }

    // Converts the quaternion to a 3x3 matrix
    // Matrix is column major
    //@TODO: Test if quatToMatrix requires a normalized quaternion
    static MATH_FORCEINLINE void quatToMatrix(const float4& q, float3x3& m)
    {
        float3 yxw = q.yxw;
        float3 zwx = q.zwx;
        float3 wzy = q.wzy;

        m.m0 = float3(-2, +2, -2) * q.y * yxw + float3(-2, +2, +2) * q.z * zwx + float3(1, 0, 0);
        m.m1 = float3(-2, -2, +2) * q.z * wzy + float3(+2, -2, +2) * q.x * yxw + float3(0, 1, 0);
        m.m2 = float3(+2, -2, -2) * q.x * zwx + float3(+2, +2, -2) * q.y * wzy + float3(0, 0, 1);
    }

    // get unit quaternion from rotation matrix
    // u, v, w must be ortho-normal.
    static float4 matrixToQuat(const float3 & u, const float3& v, const float3& w)
    {
        float4 q;
        if (u.x >= 0.f)
        {
            const float1 t = v.y + w.z;
            if (t >= 0.f)
            {
                float1 x(v.z - w.y);
                float1 y(w.x - u.z);
                float1 z(u.y - v.x);
                float1 ww(1.f + u.x + t);
                q = float4(x, y, z, ww);
                // Android doesn't like this expression, it does generate the wrong assembly
                //q = float4(v.z() - w.y(), w.x() - u.z(), u.y() - v.x(), float1(1.f) + u.x() + t);
            }
            else
            {
                float1 x(1.f + u.x - t);
                float1 y(u.y + v.x);
                float1 z(w.x + u.z);
                float1 ww(v.z - w.y);
                q = float4(x, y, z, ww);
                // Android doesn't like this expression, it does generate the wrong assembly
                //q = float4(float1(1.f) + u.x() - t, u.y() + v.x(), w.x() + u.z(), v.z() - w.y());
            }
        }
        else
        {
            const float1 t = v.y - w.z;
            if (t >= 0.f)
            {
                float1 x(u.y + v.x);
                float1 y(1.f - u.x + t);
                float1 z(v.z + w.y);
                float1 ww(w.x - u.z);
                q = float4(x, y, z, ww);
                // Android doesn't like this expression, it generates the wrong assembly
                //q = float4(u.y() + v.x(), float1(1.f) - u.x() + t, v.z() + w.y(), w.x() - u.z());
            }
            else
            {
                float1 x(w.x + u.z);
                float1 y(v.z + w.y);
                float1 z(1.f - u.x - t);
                float1 ww(u.y - v.x);
                q = float4(x, y, z, ww);
                // Android doesn't like this expression, it generates the wrong assembly
                //q = float4(w.x() + u.z(), v.z() + w.y(), float1(1.f) - u.x() - t, u.y() - v.x());
            }
        }
        return normalize(q);
    }

    // get unit quaternion from rotation matrix
    // matrix r must be ortho-normal.
    // @TODO: Write unit test for ortho-normal requirement...
    static MATH_FORCEINLINE float4 matrixToQuat(const float3x3& r)
    {
        return matrixToQuat(r.m0, r.m1, r.m2);
    }

    // quaternion will be adjusted for negative scale
    static MATH_FORCEINLINE float4 scaleMulQuat(const float3 &scale, const float4 &q)
    {
        float3 s = chgsign(float3(1.0f), scale);
        return chgsign(q, float4(s.yxx * s.zzy, 0));
    }

    // returns the angle between two quaternions
    static MATH_FORCEINLINE float1 quatDiff(const float4 &a, const float4 &b)
    {
        float1 diff = asin(length(normalize(quatMul(quatConj(a), b)).xyz));
        return diff + diff;
    }

    math::float3 closestEuler(const float3 &euler, const float3 &eulerHint, math::RotationOrder rotationOrder);
    math::float3 closestEuler(const float4 &q, const float3 &eulerHint, math::RotationOrder rotationOrder);
}
