#include "UnityPrefix.h"
#if ENABLE_UNIT_TESTS
#include "Runtime/Testing/Testing.h"
#include "Runtime/Profiler/TimeHelper.h"

#if PLATFORM_WIN
#pragma warning(push)
#pragma warning(disable: 4723) // potential division by zero
#endif

#include "vec-trig.h"

REGRESSION_TEST_SUITE(SIMDMath_trigonometricOps)
{
    using namespace math;
    const float epsilonHighPrecision = 2e-6f;
    const float epsilonMediumPrecision = 1e-2f;

    TEST(sin_float4_CompareHighPrecision)
    {
        int degree;
        for (degree = -1000; degree < 1000; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           sin_stl   = std::sin(rad);
            math::float4    sin_unity = math::highp::sin(math::float4(rad));

            CHECK_CLOSE(sin_stl, (float)sin_unity.x, epsilonHighPrecision);
        }
    }

    TEST(sin_float4_CompareMediumPrecision)
    {
        int degree;
        for (degree = -1000; degree < 1000; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           sin_stl   = std::sin(rad);
            math::float4    sin_unity = math::mediump::sin(math::float4(rad));

            CHECK_CLOSE(sin_stl, (float)sin_unity.x, epsilonMediumPrecision);
        }
    }

    TEST(sin_float3_HighPrecisionCriticalValuesAreExact)
    {
        math::float3 a = float3(0.f, math::pi_over_two(), math::pi());
        math::float3 b = math::highp::sin(a);

        CHECK_EQUAL(0.0f, (float)b.x);
        CHECK_EQUAL(1.0f, (float)b.y);
        CHECK_EQUAL(0.0f, (float)b.z);
    }

#if PLATFORM_GAMECORE_XBOXSERIES || PLATFORM_PS5
    TEST(sin_float2_HighPrecisionCriticalValuesAreExact, TestAttributes::KnownFailure(1342272, "AVX2 compiler setting appears to make a few tests fail."))
#else
    TEST(sin_float2_HighPrecisionCriticalValuesAreExact)
#endif
    {
        math::float2 a = float2(0.f, math::pi_over_two());
        math::float2 b = math::highp::sin(a);

        CHECK_EQUAL(0.0f, (float)b.x);
        CHECK_EQUAL(1.0f, (float)b.y);

        a = float2(math::pi(), math::pi_over_six());
        b = math::highp::sin(a);
        CHECK_EQUAL(0.0f, (float)b.x);
        CHECK_EQUAL(0.5f, (float)b.y);
    }

// float1 results for sin() & cos() only precise when using MATH_HAS_SIMD_FLOAT
#if MATH_HAS_SIMD_FLOAT
#if PLATFORM_GAMECORE_XBOXSERIES || PLATFORM_PS5
    TEST(sin_float1_HighPrecisionCriticalValuesAreExact, TestAttributes::KnownFailure(1342272, "AVX2 compiler setting appears to make a few tests fail."))
#else
    TEST(sin_float1_HighPrecisionCriticalValuesAreExact)
#endif
    {
        math::float1 b = math::highp::sin(float1(0.f));
        CHECK_EQUAL(0.0f, (float)b);

        b = math::highp::sin(float1(math::pi_over_two()));
        CHECK_EQUAL(1.0f, (float)b);

        b = math::highp::sin(float1(math::pi()));
        CHECK_EQUAL(0.0f, (float)b);

        b = math::highp::sin(float1(math::pi_over_six()));
        CHECK_EQUAL(0.5f, (float)b);
    }
#endif // #if MATH_HAS_SIMD_FLOAT

    TEST(cos_float4_CompareHighPrecision)
    {
        int degree;
        for (degree = -1000; degree < 1000; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           cos_stl   = std::cos(rad);
            math::float4    cos_unity = math::highp::cos(math::float4(rad));

            CHECK_CLOSE(cos_stl, (float)cos_unity.x, epsilonHighPrecision);
        }
    }

    TEST(cos_float4_CompareMediumPrecision)
    {
        int degree;
        for (degree = -1000; degree < 1000; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           cos_stl   = std::cos(rad);
            math::float4    cos_unity = math::mediump::cos(math::float4(rad));

            CHECK_CLOSE(cos_stl, (float)cos_unity.x, epsilonMediumPrecision);
        }
    }

    TEST(cos_float3_HighPrecisionCriticalValuesAreExact)
    {
        math::float3 a = float3(0.f, math::pi_over_two(), math::pi());
        math::float3 b = math::highp::cos(a);

        CHECK_EQUAL(1.0f, (float)b.x);
        CHECK_EQUAL(0.0f, (float)b.y);
        CHECK_EQUAL(-1.0f, (float)b.z);
    }

    TEST(cos_float2_HighPrecisionCriticalValuesAreExact)
    {
        math::float2 a = float2(0.f, math::pi_over_two());
        math::float2 b = math::highp::cos(a);

        CHECK_EQUAL(1.0f, (float)b.x);
        CHECK_EQUAL(0.0f, (float)b.y);

        a = float2(math::pi());
        b = math::highp::cos(a);
        CHECK_EQUAL(-1.0f, (float)b.x);
    }

// float1 results for sin() & cos() only precise when using MATH_HAS_SIMD_FLOAT
#if MATH_HAS_SIMD_FLOAT
    TEST(cos_float1_HighPrecisionCriticalValuesAreExact)
    {
        math::float1 b = math::highp::cos(float1(0.f));
        CHECK_EQUAL(1.0f, (float)b);

        b = math::highp::cos(float1(math::pi_over_two()));
        CHECK_EQUAL(0.0f, (float)b);

        b = math::highp::cos(float1(math::pi()));
        CHECK_EQUAL(-1.0f, (float)b);
    }
#endif // #if MATH_HAS_SIMD_FLOAT

    TEST(tan_float4_CompareHighPrecision)
    {
        // Tan is -inf and inf at -90 and 90 degrees
        int degree;

        for (degree = -89; degree < -81; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           tan_stl = std::tan(rad);
            math::float4    tan_unity = math::highp::tan(math::float4(rad));

            CHECK_CLOSE(tan_stl, (float)tan_unity.x, epsilonMediumPrecision);
        }

        for (degree = -81; degree < 82; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           tan_stl   = std::tan(rad);
            math::float4    tan_unity = math::highp::tan(math::float4(rad));

            CHECK_CLOSE(tan_stl, (float)tan_unity.x, epsilonHighPrecision);
        }

        for (degree = 82; degree <= 89; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           tan_stl = std::tan(rad);
            math::float4    tan_unity = math::highp::tan(math::float4(rad));

            CHECK_CLOSE(tan_stl, (float)tan_unity.x, epsilonMediumPrecision);
        }
    }

    TEST(tan_float4_CompareMediumPrecision)
    {
        // Tan is -inf and inf at -90 and 90 degrees
        int degree;
        for (degree = -89; degree < 89; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           tan_stl   = std::tan(rad);
            math::float4    tan_unity = math::mediump::tan(math::float4(rad));

            CHECK_CLOSE(tan_stl, (float)tan_unity.x, epsilonMediumPrecision);
        }
    }

    TEST(tan_float3_HighPrecisionCriticalValuesAreExact)
    {
        math::float3 a = float3(0.f, math::pi_over_four(), math::pi_over_two());
        math::float3 b = math::highp::tan(a);

        CHECK_EQUAL(0.0f, (float)b.x);
        CHECK_EQUAL(1.0f, (float)b.y);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)b.z);
    }

    TEST(tan_float2_HighPrecisionCriticalValuesAreExact)
    {
        math::float2 a = float2(0.f, math::pi_over_four());
        math::float2 b = math::highp::tan(a);

        CHECK_EQUAL(0.0f, (float)b.x);
        CHECK_EQUAL(1.0f, (float)b.y);

        a = float2(math::pi(), math::pi_over_two());
        b = math::highp::tan(a);
        CHECK_EQUAL(0.0f, (float)b.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)b.y);
    }

    TEST(tan_float1_HighPrecisionCriticalValuesAreExact)
    {
        math::float1 b = math::highp::tan(float1(0.f));
        CHECK_EQUAL(0.0f, (float)b);

        b = math::highp::tan(float1(math::pi_over_two()));
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)b);

        b = math::highp::tan(float1(math::pi()));
        CHECK_EQUAL(0.0f, (float)b);

        b = math::highp::tan(float1(math::pi_over_four()));
        CHECK_EQUAL(1.0f, (float)b);
    }


    TEST(sincos_float4_CompareHighPrecision)
    {
        int degree;
        for (degree = -180; degree < 180; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           sin_stl;
            float           cos_stl;

            sin_stl = std::sin(rad);
            cos_stl = std::cos(rad);


            math::float4 sin_unity, cos_unity;
            math::highp::sincos(float4(rad), sin_unity, cos_unity);

            CHECK_CLOSE(sin_stl, (float)sin_unity.x, epsilonHighPrecision);
            CHECK_CLOSE(cos_stl, (float)cos_unity.x, epsilonHighPrecision);
        }
    }

    TEST(sincos_float4_CompareMediumPrecision)
    {
        int degree;
        for (degree = -180; degree < 180; degree++)
        {
            float rad = radians(static_cast<float>(degree));

            float           sin_stl;
            float           cos_stl;

            sin_stl = std::sin(rad);
            cos_stl = std::cos(rad);

            math::float4 sin_unity, cos_unity;
            math::mediump::sincos(float4(rad), sin_unity, cos_unity);

            CHECK_CLOSE(sin_stl, (float)sin_unity.x, epsilonMediumPrecision);
            CHECK_CLOSE(cos_stl, (float)cos_unity.x, epsilonMediumPrecision);
        }
    }

    TEST(sincos_float3_Works)
    {
        math::float3 a = float3(0.f, math::pi_over_two(), math::pi());
        math::float3 c = float3(ZERO);
        math::float3 s = float3(ZERO);

        math::highp::sincos(a, s, c);

        CHECK_EQUAL(0.0f, (float)s.x);
        CHECK_EQUAL(1.0f, (float)s.y);
        CHECK_EQUAL(0.0f, (float)s.z);

        CHECK_EQUAL(1.0f, (float)c.x);
        CHECK_EQUAL(0.0f, (float)c.y);
        CHECK_EQUAL(-1.0f, (float)c.z);
    }

    TEST(sincos_float2_Works)
    {
        math::float2 a = float2(0.f, math::pi_over_two());
        math::float2 c = float2(ZERO);
        math::float2 s = float2(ZERO);

        math::highp::sincos(a, s, c);

        CHECK_EQUAL(0.0f, (float)s.x);
        CHECK_EQUAL(1.0f, (float)s.y);

        CHECK_EQUAL(1.0f, (float)c.x);
        CHECK_EQUAL(0.0f, (float)c.y);
    }

// float1 results for sin() & cos() only precise when using MATH_HAS_SIMD_FLOAT
#if MATH_HAS_SIMD_FLOAT
    TEST(sincos_float1_Works)
    {
        math::float1 a = float1(ZERO);
        math::float1 c = float1(ZERO);
        math::float1 s = float1(ZERO);

        math::highp::sincos(a, s, c);

        CHECK_EQUAL(0.0f, (float)s);
        CHECK_EQUAL(1.0f, (float)c);

        a = float1(math::pi_over_two());
        math::highp::sincos(a, s, c);
        CHECK_EQUAL(1.0f, (float)s);
        CHECK_EQUAL(0.0f, (float)c);
    }
#endif // #if MATH_HAS_SIMD_FLOAT

    TEST(acos_float4_CompareHighPrecision)
    {
        float x;
        for (x = -1.0f; x <= 1.05f; x += 0.05f)
        {
            float clampX = math::clamp(x, -1.0f, 1.0f);
            float           acos_stl   = std::acos(clampX);
            math::float4    acos_unity = math::highp::acos(math::float4(clampX));

            CHECK_CLOSE(acos_stl, (float)acos_unity.x, epsilonHighPrecision);
        }
    }

    TEST(acos_float4_CompareMediumPrecision)
    {
        float x;
        for (x = -1.0f; x <= 1.05f; x += 0.05f)
        {
            float clampX = math::clamp(x, -1.0f, 1.0f);
            float           acos_stl = std::acos(clampX);
            math::float4    acos_unity = math::mediump::acos(math::float4(clampX));

            CHECK_CLOSE(acos_stl, (float)acos_unity.x, epsilonMediumPrecision);
        }
    }

    TEST(acos_float3_Works)
    {
        math::float3 a = math::highp::acos(math::float3(-1.0f, 0.f, 1.0f));

        CHECK_CLOSE(math::pi(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(math::pi_over_two(), (float)a.y, epsilonHighPrecision);
        CHECK_CLOSE(0.f, (float)a.z, epsilonHighPrecision);
    }

    TEST(acos_float2_Works)
    {
        math::float2 a = math::highp::acos(math::float2(-1.0f, 1.0f));

        CHECK_CLOSE(math::pi(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(0.f, (float)a.y, epsilonHighPrecision);
    }

    TEST(acos_float1_Works)
    {
        math::float1 a = math::highp::acos(math::float1(-1.0f));
        CHECK_CLOSE(math::pi(), (float)a, epsilonHighPrecision);

        a = math::highp::acos(math::float1(0.f));
        CHECK_CLOSE(math::pi_over_two(), (float)a, epsilonHighPrecision);

        a = math::highp::acos(math::float1(1.0f));
        CHECK_CLOSE(0.f, (float)a, epsilonHighPrecision);
    }

    TEST(asin_float4_CompareHighPrecision)
    {
        float x;
        for (x = -1.0f; x <= 1.05f; x += 0.05f)
        {
            float clampX = math::clamp(x, -1.0f, 1.0f);
            float           asin_stl   = std::asin(clampX);
            math::float4    asin_unity = math::highp::asin(math::float4(clampX));

            CHECK_CLOSE(asin_stl, (float)asin_unity.x, epsilonHighPrecision);
        }
    }

    TEST(asin_float4_CompareMediumPrecision)
    {
        float x;
        for (x = -1.0f; x <= 1.05f; x += 0.05f)
        {
            float clampX = math::clamp(x, -1.0f, 1.0f);
            float           asin_stl = std::asin(clampX);
            math::float4    asin_unity = math::mediump::asin(math::float4(clampX));

            CHECK_CLOSE(asin_stl, (float)asin_unity.x, epsilonMediumPrecision);
        }
    }

    TEST(asin_float3_Works)
    {
        math::float3 a = math::highp::asin(math::float3(-1.0f, 0.f, 1.0f));

        CHECK_CLOSE(-math::pi_over_two(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(0.f, (float)a.y, epsilonHighPrecision);
        CHECK_CLOSE(math::pi_over_two(), (float)a.z, epsilonHighPrecision);
    }

    TEST(asin_float2_Works)
    {
        math::float2 a = math::highp::asin(math::float2(-1.0f, 1.0f));

        CHECK_CLOSE(-math::pi_over_two(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(math::pi_over_two(), (float)a.y, epsilonHighPrecision);
    }

    TEST(asin_float1_Works)
    {
        math::float1 a = math::highp::asin(math::float1(-1.0f));
        CHECK_CLOSE(-math::pi_over_two(), (float)a, epsilonHighPrecision);

        a = math::highp::asin(math::float1(0.f));
        CHECK_CLOSE(0.f, (float)a, epsilonHighPrecision);

        a = math::highp::asin(math::float1(1.0f));
        CHECK_CLOSE(math::pi_over_two(), (float)a, epsilonHighPrecision);
    }

    TEST(atan_float4_CompareHighPrecision)
    {
        float minRange = -1.0f;
        float maxRange = +1.05f;

        float x;
        for (x = minRange; x <= maxRange; x += 0.05f)
        {
            float clampX = math::clamp(x, -1.0f, 1.0f);
            float           atan_stl = std::atan(clampX);
            math::float4    atan_unity = math::highp::atan(math::float4(clampX));
            CHECK_CLOSE(atan_stl, (float)atan_unity.x, epsilonHighPrecision);
        }
    }

    TEST(atan_float4_CompareMediumPrecision)
    {
        float minRange = -1.0f;
        float maxRange = +1.05f;

        float x;
        for (x = minRange; x <= maxRange; x += 0.05f)
        {
            float clampX = math::clamp(x, -1.0f, 1.0f);
            float           atan_stl = std::atan(clampX);
            math::float4    atan_unity = math::mediump::atan(math::float4(clampX));
            CHECK_CLOSE(atan_stl, (float)atan_unity.x, epsilonMediumPrecision);
        }
    }

    TEST(atan_float3_Works)
    {
        math::float3 a = math::highp::atan(math::float3(1.0f, 0.f, std::numeric_limits<float>::infinity()));

        CHECK_CLOSE(math::pi_over_four(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(0.f, (float)a.y, epsilonHighPrecision);
        CHECK_CLOSE(math::pi_over_two(), (float)a.z, epsilonHighPrecision);
    }

    TEST(atan_float2_Works)
    {
        math::float2 a = math::highp::atan(math::float2(1.0f, 0.0f));

        CHECK_CLOSE(math::pi_over_four(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(0.f, (float)a.y, epsilonHighPrecision);

        a = math::highp::atan(math::float2(-1.0f, -std::numeric_limits<float>::infinity()));

        CHECK_CLOSE(-math::pi_over_four(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(-math::pi_over_two(), (float)a.y, epsilonHighPrecision);
    }

    TEST(atan_float1_Works)
    {
        math::float1 a = math::highp::atan(math::float1(-1.0f));
        CHECK_CLOSE(-math::pi_over_four(), (float)a, epsilonHighPrecision);

        a = math::highp::atan(math::float1(0.f));
        CHECK_CLOSE(0.f, (float)a, epsilonHighPrecision);

        a = math::highp::atan(math::float1(1.0f));
        CHECK_CLOSE(math::pi_over_four(), (float)a, epsilonHighPrecision);

        a = math::highp::atan(math::float1(-std::numeric_limits<float>::infinity()));
        CHECK_CLOSE(-math::pi_over_two(), (float)a, epsilonHighPrecision);

        a = math::highp::atan(math::float1(std::numeric_limits<float>::infinity()));
        CHECK_CLOSE(math::pi_over_two(), (float)a, epsilonHighPrecision);
    }


    TEST(atan2_float4_CompareHighPrecision)
    {
        float minRange = -1.0f;
        float maxRange = +1.05f;

        float x;
        for (x = minRange; x <= maxRange; x += 0.05f)
        {
            float y;
            for (y = minRange; y <= maxRange; y += 0.05f)
            {
                float clampX = math::clamp(x, -1.0f, 1.0f);
                float clampY = math::clamp(y, -1.0f, 1.0f);
                float           atan2_stl   = std::atan2(clampX, clampY);
                math::float4    atan2_unity = math::highp::atan2(math::float4(clampX), math::float4(clampY));
                CHECK_CLOSE(atan2_stl, (float)atan2_unity.x, epsilonHighPrecision);
            }
        }
    }

    TEST(atan2_float4_CompareMediumPrecision)
    {
        float minRange = -1.0f;
        float maxRange = +1.05f;

        float x;
        for (x = minRange; x <= maxRange; x += 0.05f)
        {
            float y;
            for (y = minRange; y <= maxRange; y += 0.05f)
            {
                float clampX = math::clamp(x, -1.0f, 1.0f);
                float clampY = math::clamp(y, -1.0f, 1.0f);
                float           atan2_stl = std::atan2(clampX, clampY);
                math::float4    atan2_unity = math::mediump::atan2(math::float4(clampX), math::float4(clampY));
                CHECK_CLOSE(atan2_stl, (float)atan2_unity.x, epsilonMediumPrecision);
            }
        }
    }

    TEST(atan2_float3_Works)
    {
        math::float3 a = math::highp::atan2(math::float3(1.0f, -1.f, 1.0f), math::float3(1.0f, 1.f, -1.0f));

        CHECK_CLOSE(math::pi_over_four(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(-math::pi_over_four(), (float)a.y, epsilonHighPrecision);
        CHECK_CLOSE(3 * math::pi_over_four(), (float)a.z, epsilonHighPrecision);
    }

    TEST(atan2_float2_Works)
    {
        math::float2 a = math::highp::atan2(math::float2(1.0f, -1.0f), math::float2(0.0f, 0.0f));

        CHECK_CLOSE(math::pi_over_two(), (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(-math::pi_over_two(), (float)a.y, epsilonHighPrecision);

        a = math::highp::atan2(math::float2(0.0f, 0.0f), math::float2(1.0f, -1.0f));

        CHECK_CLOSE(0.f, (float)a.x, epsilonHighPrecision);
        CHECK_CLOSE(math::pi(), (float)a.y, epsilonHighPrecision);
    }

    TEST(atan2_float1_Works)
    {
        math::float1 a = math::highp::atan2(math::float1(1.0f), math::float1(0.0f));
        CHECK_CLOSE(math::pi_over_two(), (float)a, epsilonHighPrecision);

        a = math::highp::atan2(math::float1(-1.0f), math::float1(0.0f));
        CHECK_CLOSE(-math::pi_over_two(), (float)a, epsilonHighPrecision);
    }
}

#if PLATFORM_WIN
#pragma warning(pop)
#endif

#endif
