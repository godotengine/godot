#pragma once

#include "vec-affine.h"

namespace math
{
    struct aabb
    {
        aabb() {}
        aabb(const aabb& rhs) : center(rhs.center), extents(rhs.extents) {}
        aabb(const float3& c, const float3& e) : center(c), extents(e) {}
        aabb& operator=(const aabb& rhs) { center = rhs.center; extents = rhs.extents; return *this; }

        float3 center;
        float3 extents;
    };

    static MATH_FORCEINLINE aabb aabbLoad(const void* value);
    static MATH_FORCEINLINE void aabbStore(void* out, const aabb& value);

    struct aabb_storage
    {
        float center[3];
        float extents[3];

        MATH_FORCEINLINE operator aabb() const
        {
            return aabbLoad(this);
        }

        MATH_FORCEINLINE aabb_storage& operator=(const aabb& v)
        {
            aabbStore(this, v);
            return *this;
        }
    };

    static MATH_FORCEINLINE aabb aabbLoad(const void* value)
    {
        aabb b;
        b.center = vload3f(reinterpret_cast<const aabb_storage*>(value)->center);
        b.extents = vload3f(reinterpret_cast<const aabb_storage*>(value)->extents);
        return b;
    }

    static MATH_FORCEINLINE void aabbStore(void* out, const aabb& value)
    {
        vstore3f(reinterpret_cast<aabb_storage*>(out)->center, value.center);
        vstore3f(reinterpret_cast<aabb_storage*>(out)->extents, value.extents);
    }

    static MATH_FORCEINLINE aabb aabbTransform(const affineX& transform, const aabb& localAABB)
    {
        aabb b;
        b.center = mul(transform, localAABB.center);
        b.extents =
            abs(transform.rs.m0 * localAABB.extents.x) +
            abs(transform.rs.m1 * localAABB.extents.y) +
            abs(transform.rs.m2 * localAABB.extents.z);
        return b;
    }

    static MATH_FORCEINLINE void aabbMinMaxEncapsulate(float3& bbMin, float3& bbMax, const float3& point)
    {
        bbMin = min(bbMin, point);
        bbMax = max(bbMax, point);
    }

    static MATH_FORCEINLINE aabb aabbFromMinMax(const float3& bbMin, const float3& bbMax)
    {
        return aabb((bbMin + bbMax) * 0.5f, (bbMax - bbMin) * 0.5f);
    }

    static MATH_FORCEINLINE void aabbCalculateVertices(const float3& min, const float3& max, float3* outVertices)
    {
        //    7-----6
        //   /     /|
        //  3-----2 |
        //  | 4   | 5
        //  |     |/
        //  0-----1
        outVertices[0] = min;
        outVertices[1] = float3(max.x, min.y, min.z);
        outVertices[2] = float3(max.x, max.y, min.z);
        outVertices[3] = float3(min.x, max.y, min.z);
        outVertices[4] = float3(min.x, min.y, max.z);
        outVertices[5] = float3(max.x, min.y, max.z);
        outVertices[6] = float3(max.x, max.y, max.z);
        outVertices[7] = float3(min.x, max.y, max.z);
    }

    static MATH_FORCEINLINE aabb aabbTransformNonUniform(const affineX& transform, const aabb& localAABB)
    {
        float3 v[8];
        aabbCalculateVertices(localAABB.center - localAABB.extents, localAABB.center + localAABB.extents, v);

        float3 max;
        float3 min = max = mul(transform, v[0]);
        for (size_t i = 1; i < 8; i++)
            aabbMinMaxEncapsulate(min, max, mul(transform, v[i]));

        aabb b;
        b.extents = (max - min) * 0.5F;
        b.center = min + b.extents;

        return b;
    }
}
