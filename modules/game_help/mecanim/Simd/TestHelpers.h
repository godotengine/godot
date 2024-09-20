#pragma once

#if ENABLE_UNIT_TESTS
#include "Runtime/Testing/TestingForwardDecls.h"
#include "Runtime/Math/Simd/vec-math.h"

namespace math
{
#if (defined(MATH_HAS_NATIVE_SIMD) && defined(MATH_HAS_SIMD_INT)) || defined(MATH_HAS_SIMD_FLOAT)
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const int1& vec);
#endif
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const int2& vec);
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const int3& vec);
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const int4& vec);
#if defined(MATH_HAS_SIMD_FLOAT)
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const float1& vec);
#endif
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const float2& vec);
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const float3& vec);
    UnitTest::MemoryOutStream & operator<<(UnitTest::MemoryOutStream& stream, const float4& vec);
}

namespace UnitTest
{
    template<> inline bool AreClose(math::float3 const& expected, math::float3 const& actual, float const& tolerance)
    {
        return math::compare_approx(expected, actual, tolerance);
    }

    template<>
    inline bool AreEqual(math::int3 const& expected, math::int3 const& actual)
    {
        return math::all(expected == actual);
    }
}

#endif
