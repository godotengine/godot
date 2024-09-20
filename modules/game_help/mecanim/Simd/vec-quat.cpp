#include "UnityPrefix.h"
#include "vec-quat.h"

namespace math
{
    static MATH_FORCEINLINE float3 eulerReorder(const float3& euler, RotationOrder order)
    {
        switch (order)
        {
            case kOrderXYZ: //XYZ
                return euler;
            case kOrderXZY: //XZY
                return euler.xzy;
            case kOrderYZX: //YZX
                return euler.yzx;
            case kOrderYXZ: //YXZ
                return euler.yxz;
            case kOrderZXY: //ZXY
                return euler.zxy;
            case kOrderZYX: //ZYX
                return euler.zyx;
        }

        AssertString("invalid rotationOrder");
        return euler;
    }

    static MATH_FORCEINLINE float3 eulerReorderBack(const float3& euler, RotationOrder order)
    {
        switch (order)
        {
            case kOrderXYZ: //XYZ
                return euler;
            case kOrderXZY: //XZY
                return euler.xzy;
            case kOrderYZX: //YZX
                return euler.zxy;
            case kOrderYXZ: //YXZ
                return euler.yxz;
            case kOrderZXY: //ZXY
                return euler.yzx;
            case kOrderZYX: //ZYX
                return euler.zyx;
        }

        AssertString("invalid rotationOrder");
        return euler;
    }

    float3 quatToEuler(const float4& q, RotationOrder order)
    {
        //prepare the data
        float4 d1 = q * q.wwww * float4(2.f); //xw, yw, zw, ww
        float4 d2 = q * q.yzxw * float4(2.f); //xy, yz, zx, ww
        float4 d3 = q * q;
        float3 euler(ZERO);

        const float1 CUTOFF = (1.0f - 2.0f * math::epsilon()) * (1.0f - 2.0f * math::epsilon());

        switch (order)
        {
            case kOrderZYX: //ZYX
            {
                float1 y1 = d2.z + d1.y;
                if (y1 * y1 < CUTOFF)
                {
                    float1 x1 = -d2.x + d1.z;
                    float1 x2 = d3.x + d3.w - d3.y - d3.z;
                    float1 z1 = -d2.y + d1.x;
                    float1 z2 = d3.z + d3.w - d3.y - d3.x;
                    euler = float3(atan2(x1, x2), asin(y1), atan2(z1, z2));
                }
                else //zxz
                {
                    y1 = clamp(y1, float1(-1.0f), float1(1.0f));
                    float4 abcd = float4(d2.z, d1.y, d2.y, d1.x);
                    float1 x1 = 2.0f * (abcd.x * abcd.w + abcd.y * abcd.z); //2(ad+bc)
                    float1 x2 = csum(abcd * abcd * float4(-1.0f, 1.0f, -1.0f, 1.0f));
                    euler = float3(atan2(x1, x2), asin(y1), float1(0.f));
                }
                break;
            }
            case kOrderZXY: //ZXY
            {
                float1 y1 = d2.y - d1.x;
                if (y1 * y1 < CUTOFF)
                {
                    float1 x1 = d2.x + d1.z;
                    float1 x2 = d3.y + d3.w - d3.x - d3.z;
                    float1 z1 = d2.z + d1.y;
                    float1 z2 = d3.z + d3.w - d3.x - d3.y;
                    euler = float3(atan2(x1, x2), -asin(y1), atan2(z1, z2));
                }
                else //zxz
                {
                    y1 = clamp(y1, float1(-1.0f), float1(1.0f));
                    float4 abcd = float4(d2.z, d1.y, d2.y, d1.x);
                    float1 x1 = 2.0f * (abcd.x * abcd.w + abcd.y * abcd.z); //2(ad+bc)
                    float1 x2 = csum(abcd * abcd * float4(-1.0f, 1.0f, -1.0f, 1.0f));
                    euler = float3(atan2(x1, x2), -asin(y1), float1(0.f));
                }
                break;
            }
            case kOrderYXZ: //YXZ
            {
                float1 y1 = d2.y + d1.x;
                if (y1 * y1 < CUTOFF)
                {
                    float1 x1 = -d2.z + d1.y;
                    float1 x2 = d3.z + d3.w - d3.x - d3.y;
                    float1 z1 = -d2.x + d1.z;
                    float1 z2 = d3.y + d3.w - d3.z - d3.x;
                    euler = float3(atan2(x1, x2), asin(y1), atan2(z1, z2));
                }
                else //yzy
                {
                    y1 = clamp(y1, float1(-1.0f), float1(1.0f));
                    float4 abcd = float4(d2.x, d1.z, d2.y, d1.x);
                    float1 x1 = 2.0f * (abcd.x * abcd.w + abcd.y * abcd.z); //2(ad+bc)
                    float1 x2 = csum(abcd * abcd * float4(-1.0f, 1.0f, -1.0f, 1.0f));
                    euler = float3(atan2(x1, x2), asin(y1), float1(0.f));
                }
                break;
            }
            case kOrderYZX: //YZX
            {
                float1 y1 = d2.x - d1.z;
                if (y1 * y1 < CUTOFF)
                {
                    float1 x1 = d2.z + d1.y;
                    float1 x2 = d3.x + d3.w - d3.z - d3.y;
                    float1 z1 = d2.y + d1.x;
                    float1 z2 = d3.y + d3.w - d3.x - d3.z;
                    euler = float3(atan2(x1, x2), -asin(y1), atan2(z1, z2));
                }
                else //yxy
                {
                    y1 = clamp(y1, float1(-1.0f), float1(1.0f));
                    float4 abcd = float4(d2.x, d1.z, d2.y, d1.x);
                    float1 x1 = 2.0f * (abcd.x * abcd.w + abcd.y * abcd.z); //2(ad+bc)
                    float1 x2 = csum(abcd * abcd * float4(-1.0f, 1.0f, -1.0f, 1.0f));
                    euler = float3(atan2(x1, x2), -asin(y1), float1(0.f));
                }
                break;
            }

            case kOrderXZY: //XZY
            {
                float1 y1 = d2.x + d1.z;
                if (y1 * y1 < CUTOFF)
                {
                    float1 x1 = -d2.y + d1.x;
                    float1 x2 = d3.y + d3.w - d3.z - d3.x;
                    float1 z1 = -d2.z + d1.y;
                    float1 z2 = d3.x + d3.w - d3.y - d3.z;
                    euler = float3(atan2(x1, x2), asin(y1), atan2(z1, z2));
                }
                else //xyx
                {
                    y1 = clamp(y1, float1(-1.0f), float1(1.0f));
                    float4 abcd = float4(d2.x, d1.z, d2.z, d1.y);
                    float1 x1 = 2.0f * (abcd.x * abcd.w + abcd.y * abcd.z); //2(ad+bc)
                    float1 x2 = csum(abcd * abcd * float4(-1.0f, 1.0f, -1.0f, 1.0f));
                    euler = float3(atan2(x1, x2), asin(y1), float1(0.0f));
                }
                break;
            }
            case kOrderXYZ: //XYZ
            {
                float1 y1 = d2.z - d1.y;
                if (y1 * y1 < CUTOFF)
                {
                    float1 x1 = d2.y + d1.x;
                    float1 x2 = d3.z + d3.w - d3.y - d3.x;
                    float1 z1 = d2.x + d1.z;
                    float1 z2 = d3.x + d3.w - d3.y - d3.z;
                    euler = float3(atan2(x1, x2), -asin(y1), atan2(z1, z2));
                }
                else //xzx
                {
                    y1 = clamp(y1, float1(-1.0f), float1(1.0f));
                    float4 abcd = float4(d2.z, d1.y, d2.x, d1.z);
                    float1 x1 = 2.0f * (abcd.x * abcd.w + abcd.y * abcd.z); //2(ad+bc)
                    float1 x2 = csum(abcd * abcd * float4(-1.0f, 1.0f, -1.0f, 1.0f));
                    euler = float3(atan2(x1, x2), -asin(y1), float1(0.0f));
                }
                break;
            }
        }

        return eulerReorderBack(euler, order);
    }

    static MATH_FORCEINLINE float3 alternateEuler(const float3 &euler, math::RotationOrder rotationOrder)
    {
        float3 eulerAlt = eulerReorder(euler, rotationOrder);
        eulerAlt += float3(180.0f);
        eulerAlt = chgsign(eulerAlt, float3(1, -1, 1));
        return eulerReorderBack(eulerAlt, rotationOrder);
    }

    static MATH_FORCEINLINE float3 syncEuler(const float3 &euler, const float3 &eulerHint)
    {
        return euler + round((eulerHint - euler) / float3(360.f)) * float3(360.0f);
    }

    math::float3 closestEuler(const float3 &euler, const float3 &eulerHint, math::RotationOrder rotationOrder)
    {
        float3 eulerSynced = syncEuler(euler, eulerHint);
        float3 altEulerSynced = syncEuler(alternateEuler(euler, rotationOrder), eulerHint);

        float3 diff = eulerSynced - eulerHint;
        float3 altDiff = altEulerSynced - eulerHint;

        return select(altEulerSynced, eulerSynced, -(dot(diff, diff) <= dot(altDiff, altDiff)));
    }

    math::float3 closestEuler(const float4 &q, const float3 &eulerHint, math::RotationOrder rotationOrder)
    {
        float3 euler = degrees(quatToEuler(q, rotationOrder));
        // Previously this used round(euler / float3(1e-3f)) * float3(1e-3f) but 1e-3f cannot be accurately represented by
        // a floating point value. This led to rounding errors.
        // Switching the equation round and using 1000.0f (1e3f) instead avoids this issue.
        euler = round(euler * float3(1000.0f)) / float3(1000.0f);
        float4 qHint = eulerToQuat(radians(eulerHint), rotationOrder);
        float1 angleDiff = degrees(quatDiff(q, qHint));

        return select(closestEuler(euler, eulerHint, rotationOrder), eulerHint, -(angleDiff < 1e-3f));
    }
} // namespace math
