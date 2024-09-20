#include "UnityPrefix.h"
#include "TestHelpers.h"

#if ENABLE_UNIT_TESTS
#include "External/UnitTest++/src/MemoryOutStream.h"

namespace math
{
#if (defined(MATH_HAS_NATIVE_SIMD) && defined(MATH_HAS_SIMD_INT)) || defined(MATH_HAS_SIMD_FLOAT)
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const int1& vec)
    {
        int1_storage store = vec;
        stream << "{x: " << store.x << "}";
        return stream;
    }

#endif

    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const int2& vec)
    {
        int2_storage store = vec;
        stream << "{x: " << store.x << ", y: " << store.y << "}";
        return stream;
    }

    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const int3& vec)
    {
        int3_storage store = vec;
        stream << "{x: " << store.x << ", y: " << store.y << ", z: " << store.z << "}";
        return stream;
    }

    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const int4& vec)
    {
        int4_storage store = vec;
        stream << "{x: " << store.x << ", y: " << store.y << ", z: " << store.z << ", w: " << store.w << "}";
        return stream;
    }

#if defined(MATH_HAS_SIMD_FLOAT)
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const float1& vec)
    {
        float1_storage store = vec;
        stream << "{x: " << store.x << "}";
        return stream;
    }

#endif

    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const float2& vec)
    {
        float2_storage store = vec;
        stream << "{x: " << store.x << ", y: " << store.y << "}";
        return stream;
    }

    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const float3& vec)
    {
        float3_storage store = vec;
        stream << "{x: " << store.x << ", y: " << store.y << ", z: " << store.z << "}";
        return stream;
    }

    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const float4& vec)
    {
        float4_storage store = vec;
        stream << "{x: " << store.x << ", y: " << store.y << ", z: " << store.z << ", w: " << store.w << "}";
        return stream;
    }
}
#endif
