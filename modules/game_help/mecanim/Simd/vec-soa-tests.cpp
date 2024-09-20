#include "UnityPrefix.h"

#include "Runtime/Math/Simd/vec-soa.h"
#include "Runtime/Math/Simd/vec-math.h"
#include "Runtime/Math/ColorSpaceConversion.h"
#include "Runtime/Math/FloatConversion.h"
#include "Runtime/Math/Matrix3x3.h"

#if ENABLE_UNIT_TESTS || ENABLE_PERFORMANCE_TESTS
// Reference Implementation using scalar code
static math::pixN reference_convert_pixN(const math::floatNx4& hdr)
{
    math::pixN result;

    math::floatNx4 scaled = math::mul(math::saturate(hdr), math::floatN(255.0f));

    const float* src = reinterpret_cast<const float*>(&scaled);
    UInt8* dst = reinterpret_cast<UInt8*>(&result);

    for (int i = 0; i < math::kSimdWidth; i++)
    {
        for (int j = 0; j < 4; j++)
            dst[i * 4 + j] = RoundfToIntPos(src[j * math::kSimdWidth + i]);
    }

    return result;
}

static math::floatNx4 reference_convert_floatNx4(const math::pixN& ldr)
{
    math::floatNx4 result;

    const UInt8* src = reinterpret_cast<const UInt8*>(&ldr);
    float* dst = reinterpret_cast<float*>(&result);

    for (int i = 0; i < math::kSimdWidth; i++)
    {
        for (int j = 0; j < 4; j++)
            dst[i * 4 + j] = ((float)src[j * math::kSimdWidth + i] / 255.0f);
    }

    return result;
}

#endif

#if ENABLE_UNIT_TESTS
#include "Runtime/Testing/Testing.h"

UNIT_TEST_SUITE(SIMDMath_SoAOps)
{
#define MAKE_PIX(a, b, c, d) (d<<24)|(c<<16)|(b<<8)|(a)

    // This test is a premise to convert_pixN as we need to know if all platform can correctly
    // convert the full supported range from 0 to 1
    TEST(CanEmulate_RoundfToIntPos_Between0and1)
    {
        const float end = 1.0f;
        float v = 0.0f;
        float inc = std::numeric_limits<float>::epsilon();
        do
        {
            float scaled = math::saturate(v) * 255.0f;

            int expectedResult = RoundfToIntPos(scaled);
            math::int1 result = math::convert_int1(math::float1(scaled + 0.5f));

            CHECK_EQUAL(expectedResult, (int)result);

            v += inc;
        }
        while (v <= end);
    }

    TEST(convert_pixN_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx4 colors = math::floatNx4(math::floatN(math::ZERO), math::floatN(0.333f), math::floatN(0.666f), math::floatN(1.0f));
        math::pixN pixColors = math::convert_pixN(colors);
        math::pixN expectedPixColors = reference_convert_pixN(colors);

        CHECK(math::all(expectedPixColors == pixColors));
        CHECK(math::all(pixColors == math::pixN(MAKE_PIX(0, 85, 170, 255), MAKE_PIX(0, 85, 170, 255), MAKE_PIX(0, 85, 170, 255), MAKE_PIX(0, 85, 170, 255))));
    }

    TEST(convert_floatNx4_GivesSameResultsAs_ReferenceImpl)
    {
        math::pixN colors = math::pixN(MAKE_PIX(0, 85, 170, 255), MAKE_PIX(0, 85, 170, 255), MAKE_PIX(0, 85, 170, 255), MAKE_PIX(0, 85, 170, 255));
        math::floatNx4 floatColors = math::convert_floatNx4(colors);
        math::floatNx4 expectedFloatColors = reference_convert_floatNx4(colors);

        CHECK(math::all(expectedFloatColors.x == floatColors.x) && math::all(expectedFloatColors.y == floatColors.y) && math::all(expectedFloatColors.z == floatColors.z) && math::all(expectedFloatColors.w == floatColors.w));
        CHECK(math::compare_approx(floatColors.x, math::floatN(math::ZERO), 0.01f) && math::compare_approx(floatColors.y, math::floatN(0.333f), 0.01f) && math::compare_approx(floatColors.z, math::floatN(0.666f), 0.01f) && math::compare_approx(floatColors.w, math::floatN(1.0f), 0.01f));
    }

    // Reference function is math::abs(float4)
    TEST(abs_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 absValues = math::abs(values);
        math::floatNx3 expectedAbsValues = math::floatNx3(math::abs(values.x), math::abs(values.y), math::abs(values.z));

        CHECK(math::all(expectedAbsValues.x == absValues.x) && math::all(expectedAbsValues.y == absValues.y) && math::all(expectedAbsValues.z == absValues.z));
    }

    // Reference function is math::dot(float3, float3)
    TEST(dot3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 valuesB = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatN dotValues = math::dot(valuesA, valuesB);
        math::float1 expectedDotValue0 = math::dot(math::float3(valuesA.x.x, valuesA.y.x, valuesA.z.x), math::float3(valuesB.x.x, valuesB.y.x, valuesB.z.x));
        math::float1 expectedDotValue1 = math::dot(math::float3(valuesA.x.y, valuesA.y.y, valuesA.z.y), math::float3(valuesB.x.y, valuesB.y.y, valuesB.z.y));
        math::float1 expectedDotValue2 = math::dot(math::float3(valuesA.x.z, valuesA.y.z, valuesA.z.z), math::float3(valuesB.x.z, valuesB.y.z, valuesB.z.z));
        math::float1 expectedDotValue3 = math::dot(math::float3(valuesA.x.w, valuesA.y.w, valuesA.z.w), math::float3(valuesB.x.w, valuesB.y.w, valuesB.z.w));

        CHECK(math::compare_approx(expectedDotValue0, (math::float1)dotValues.x, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedDotValue1, (math::float1)dotValues.y, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedDotValue2, (math::float1)dotValues.z, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedDotValue3, (math::float1)dotValues.w, (math::float1)math::epsilon_scale()));
    }

    // Reference function is math::dot(float2, float2)
    TEST(dot2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 valuesA = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 valuesB = math::floatNx2(math::floatN(-345.5f), math::floatN(100.0f));
        math::floatN dotValues = math::dot(valuesA, valuesB);
        math::float1 expectedDotValue0 = math::dot(math::float2(valuesA.x.x, valuesA.y.x), math::float2(valuesB.x.x, valuesB.y.x));
        math::float1 expectedDotValue1 = math::dot(math::float2(valuesA.x.y, valuesA.y.y), math::float2(valuesB.x.y, valuesB.y.y));
        math::float1 expectedDotValue2 = math::dot(math::float2(valuesA.x.z, valuesA.y.z), math::float2(valuesB.x.z, valuesB.y.z));
        math::float1 expectedDotValue3 = math::dot(math::float2(valuesA.x.w, valuesA.y.w), math::float2(valuesB.x.w, valuesB.y.w));

        CHECK(math::compare_approx(expectedDotValue0, (math::float1)dotValues.x, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedDotValue1, (math::float1)dotValues.y, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedDotValue2, (math::float1)dotValues.z, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedDotValue3, (math::float1)dotValues.w, (math::float1)math::epsilon_scale()));
    }

    // Reference function is math::cross(float3, float3)
    TEST(cross_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 valuesB = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatNx3 crossValues = math::cross(valuesA, valuesB);
        math::float3 expectedCrossValue0 = math::cross(math::float3(valuesA.x.x, valuesA.y.x, valuesA.z.x), math::float3(valuesB.x.x, valuesB.y.x, valuesB.z.x));
        math::float3 expectedCrossValue1 = math::cross(math::float3(valuesA.x.y, valuesA.y.y, valuesA.z.y), math::float3(valuesB.x.y, valuesB.y.y, valuesB.z.y));
        math::float3 expectedCrossValue2 = math::cross(math::float3(valuesA.x.z, valuesA.y.z, valuesA.z.z), math::float3(valuesB.x.z, valuesB.y.z, valuesB.z.z));
        math::float3 expectedCrossValue3 = math::cross(math::float3(valuesA.x.w, valuesA.y.w, valuesA.z.w), math::float3(valuesB.x.w, valuesB.y.w, valuesB.z.w));

        CHECK(math::compare_approx(expectedCrossValue0, math::float3(crossValues.x.x, crossValues.y.x, crossValues.z.x), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedCrossValue1, math::float3(crossValues.x.y, crossValues.y.y, crossValues.z.y), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedCrossValue2, math::float3(crossValues.x.z, crossValues.y.z, crossValues.z.z), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedCrossValue3, math::float3(crossValues.x.w, crossValues.y.w, crossValues.z.w), math::epsilon_scale()));
    }

    // Reference function is math::length(float3)
    TEST(length3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatN lengthValues = math::length(values);
        math::float1 expectedLengthValue0 = math::length(math::float3(values.x.x, values.y.x, values.z.x));
        math::float1 expectedLengthValue1 = math::length(math::float3(values.x.y, values.y.y, values.z.y));
        math::float1 expectedLengthValue2 = math::length(math::float3(values.x.z, values.y.z, values.z.z));
        math::float1 expectedLengthValue3 = math::length(math::float3(values.x.w, values.y.w, values.z.w));

        CHECK(math::compare_approx(expectedLengthValue0, (math::float1)lengthValues.x, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedLengthValue1, (math::float1)lengthValues.y, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedLengthValue2, (math::float1)lengthValues.z, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedLengthValue3, (math::float1)lengthValues.w, (math::float1)math::epsilon_scale()));
    }

    // Reference function is math::length(float2)
    TEST(length2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 values = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatN lengthValues = math::length(values);
        math::float1 expectedLengthValue0 = math::length(math::float2(values.x.x, values.y.x));
        math::float1 expectedLengthValue1 = math::length(math::float2(values.x.y, values.y.y));
        math::float1 expectedLengthValue2 = math::length(math::float2(values.x.z, values.y.z));
        math::float1 expectedLengthValue3 = math::length(math::float2(values.x.w, values.y.w));

        CHECK(math::compare_approx(expectedLengthValue0, (math::float1)lengthValues.x, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedLengthValue1, (math::float1)lengthValues.y, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedLengthValue2, (math::float1)lengthValues.z, (math::float1)math::epsilon_scale()));
        CHECK(math::compare_approx(expectedLengthValue3, (math::float1)lengthValues.w, (math::float1)math::epsilon_scale()));
    }

    // Reference function is math::min(float4, float4)
    TEST(min3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 valuesB = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatNx3 minValues = math::min(valuesA, valuesB);
        math::floatNx3 expectedMinValues = math::floatNx3(math::min(valuesA.x, valuesB.x), math::min(valuesA.y, valuesB.y), math::min(valuesA.z, valuesB.z));

        CHECK(math::all(expectedMinValues.x == minValues.x) && math::all(expectedMinValues.y == minValues.y) && math::all(expectedMinValues.z == minValues.z));
    }

    // Reference function is math::min(float4, float4)
    TEST(min2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 valuesA = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 valuesB = math::floatNx2(math::floatN(-345.5f), math::floatN(100.0f));
        math::floatNx2 minValues = math::min(valuesA, valuesB);
        math::floatNx2 expectedMinValues = math::floatNx2(math::min(valuesA.x, valuesB.x), math::min(valuesA.y, valuesB.y));

        CHECK(math::all(expectedMinValues.x == minValues.x) && math::all(expectedMinValues.y == minValues.y));
    }

    // Reference function is math::max(float4, float4)
    TEST(max3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 valuesB = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatNx3 maxValues = math::max(valuesA, valuesB);
        math::floatNx3 expectedMaxValues = math::floatNx3(math::max(valuesA.x, valuesB.x), math::max(valuesA.y, valuesB.y), math::max(valuesA.z, valuesB.z));

        CHECK(math::all(expectedMaxValues.x == maxValues.x) && math::all(expectedMaxValues.y == maxValues.y) && math::all(expectedMaxValues.z == maxValues.z));
    }

    // Reference function is math::max(float4, float4)
    TEST(max2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 valuesA = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 valuesB = math::floatNx2(math::floatN(-345.5f), math::floatN(100.0f));
        math::floatNx2 maxValues = math::max(valuesA, valuesB);
        math::floatNx2 expectedMaxValues = math::floatNx2(math::max(valuesA.x, valuesB.x), math::max(valuesA.y, valuesB.y));

        CHECK(math::all(expectedMaxValues.x == maxValues.x) && math::all(expectedMaxValues.y == maxValues.y));
    }

    // Reference function is math::cmin(float3)
    TEST(cmin3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::float3 minValues = math::cmin(values);
        math::float3 expectedMinValues = math::float3(math::cmin(values.x), math::cmin(values.y), math::cmin(values.z));

        CHECK(math::all(expectedMinValues == minValues));
    }

    // Reference function is math::cmin(float2)
    TEST(cmin2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 values = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::float2 minValues = math::cmin(values);
        math::float2 expectedMinValues = math::float2(math::cmin(values.x), math::cmin(values.y));

        CHECK(math::all(expectedMinValues == minValues));
    }

    // Reference function is math::cmax(float3)
    TEST(cmax3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::float3 maxValues = math::cmax(values);
        math::float3 expectedMaxValues = math::float3(math::cmax(values.x), math::cmax(values.y), math::cmax(values.z));

        CHECK(math::all(expectedMaxValues == maxValues));
    }

    // Reference function is math::cmax(float2)
    TEST(cmax2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 values = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::float2 maxValues = math::cmax(values);
        math::float2 expectedMaxValues = math::float2(math::cmax(values.x), math::cmax(values.y));

        CHECK(math::all(expectedMaxValues == maxValues));
    }

    // Reference function is math::mul(float4, float4)
    TEST(mul4_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx4 valuesA = math::floatNx4(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f), math::floatN(-456641.565f));
        math::floatNx4 valuesB = math::floatNx4(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f), math::floatN(0.9f));
        math::floatNx4 mulValues = math::mul(valuesA, valuesB);
        math::floatNx4 expectedMulValues = math::floatNx4(valuesA.x * valuesB.x, valuesA.y * valuesB.y, valuesA.z * valuesB.z, valuesA.w * valuesB.w);

        CHECK(math::all(expectedMulValues.x == mulValues.x) && math::all(expectedMulValues.y == mulValues.y) && math::all(expectedMulValues.z == mulValues.z) && math::all(expectedMulValues.w == mulValues.w));
    }

    // Reference function is math::mul(float4, float4)
    TEST(mul3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 valuesB = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatNx3 mulValues = math::mul(valuesA, valuesB);
        math::floatNx3 expectedMulValues = math::floatNx3(valuesA.x * valuesB.x, valuesA.y * valuesB.y, valuesA.z * valuesB.z);

        CHECK(math::all(expectedMulValues.x == mulValues.x) && math::all(expectedMulValues.y == mulValues.y) && math::all(expectedMulValues.z == mulValues.z));
    }

    // Reference function is math::mul(float4, float4)
    TEST(mul2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 valuesA = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 valuesB = math::floatNx2(math::floatN(-345.5f), math::floatN(100.0f));
        math::floatNx2 mulValues = math::mul(valuesA, valuesB);
        math::floatNx2 expectedMulValues = math::floatNx2(valuesA.x * valuesB.x, valuesA.y * valuesB.y);

        CHECK(math::all(expectedMulValues.x == mulValues.x) && math::all(expectedMulValues.y == mulValues.y));
    }

    // Reference function is math::div(float4, float4)
    TEST(div3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatN valuesB = math::floatN(-35.5f);
        math::floatNx3 divValues = math::div(valuesA, valuesB);
        math::floatNx3 expectedDivValues = math::floatNx3(valuesA.x / valuesB, valuesA.y / valuesB, valuesA.z / valuesB);

        CHECK(math::all(expectedDivValues.x == divValues.x) && math::all(expectedDivValues.y == divValues.y) && math::all(expectedDivValues.z == divValues.z));
    }

    // Reference function is math::div(float4, float4)
    TEST(div2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 valuesA = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatN valuesB = math::floatN(-35.5f);
        math::floatNx2 divValues = math::div(valuesA, valuesB);
        math::floatNx2 expectedDivValues = math::floatNx2(valuesA.x / valuesB, valuesA.y / valuesB);

        CHECK(math::all(expectedDivValues.x == divValues.x) && math::all(expectedDivValues.y == divValues.y));
    }

    // Reference function is math::mad(float4, float4, float4)
#if PLATFORM_GAMECORE_XBOXSERIES || PLATFORM_PS5
    TEST(mad3_GivesSameResultsAs_ReferenceImpl, TestAttributes::KnownFailure(1342272, "AVX2 compiler setting appears to make a few tests fail."))
#else
    TEST(mad3_GivesSameResultsAs_ReferenceImpl)
#endif
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatN valuesB = math::floatN(-35.5f);
        math::floatN valuesC = math::floatN(0.2f);
        math::floatNx3 madValues = math::mad(valuesA, valuesB, valuesC);
        math::floatNx3 expectedMadValues = math::floatNx3(valuesA.x * valuesB + valuesC, valuesA.y * valuesB + valuesC, valuesA.z * valuesB + valuesC);

        CHECK(math::all(expectedMadValues.x == madValues.x) && math::all(expectedMadValues.y == madValues.y) && math::all(expectedMadValues.z == madValues.z));
    }

    // Reference function is math::mad(float4, float4, float4)
#if PLATFORM_GAMECORE_XBOXSERIES || PLATFORM_PS5
    TEST(mad2_GivesSameResultsAs_ReferenceImpl, TestAttributes::KnownFailure(1342272, "AVX2 compiler setting appears to make a few tests fail."))
#else
    TEST(mad2_GivesSameResultsAs_ReferenceImpl)
#endif
    {
        math::floatNx2 valuesA = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatN valuesB = math::floatN(-35.5f);
        math::floatN valuesC = math::floatN(0.2f);
        math::floatNx2 madValues = math::mad(valuesA, valuesB, valuesC);
        math::floatNx2 expectedMadValues = math::floatNx2(valuesA.x * valuesB + valuesC, valuesA.y * valuesB + valuesC);

        CHECK(math::all(expectedMadValues.x == madValues.x) && math::all(expectedMadValues.y == madValues.y));
    }

    // Reference function is math::select(float4, float4, int4)
    TEST(select3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 valuesB = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::intN valuesC = math::int4(0xffffffff, 0, 0xffffffff, 0);
        math::floatNx3 selectValues = math::select(valuesA, valuesB, valuesC);
        math::floatNx3 expectedSelectValues = math::floatNx3(math::select(valuesA.x, valuesB.x, valuesC), math::select(valuesA.y, valuesB.y, valuesC), math::select(valuesA.z, valuesB.z, valuesC));

        CHECK(math::all(expectedSelectValues.x == selectValues.x) && math::all(expectedSelectValues.y == selectValues.y) && math::all(expectedSelectValues.z == selectValues.z));
    }

    // Reference function is math::select(float4, float4, int4)
    TEST(select2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 valuesA = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 valuesB = math::floatNx2(math::floatN(-345.5f), math::floatN(100.0f));
        math::intN valuesC = math::int4(0xffffffff, 0, 0xffffffff, 0);
        math::floatNx2 selectValues = math::select(valuesA, valuesB, valuesC);
        math::floatNx2 expectedSelectValues = math::floatNx2(math::select(valuesA.x, valuesB.x, valuesC), math::select(valuesA.y, valuesB.y, valuesC));

        CHECK(math::all(expectedSelectValues.x == selectValues.x) && math::all(expectedSelectValues.y == selectValues.y));
    }

    // Reference function is math::normalize(float3)
    TEST(normalize3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(0.1f), math::floatN(0.02f), math::floatN(-1.5f));
        math::floatNx3 normalizeValues = math::normalize(values);
        math::float3 expectedNormalizeValue0 = math::normalize(math::float3(values.x.x, values.y.x, values.z.x));
        math::float3 expectedNormalizeValue1 = math::normalize(math::float3(values.x.y, values.y.y, values.z.y));
        math::float3 expectedNormalizeValue2 = math::normalize(math::float3(values.x.z, values.y.z, values.z.z));
        math::float3 expectedNormalizeValue3 = math::normalize(math::float3(values.x.w, values.y.w, values.z.w));

        CHECK(math::compare_approx(expectedNormalizeValue0, math::float3(normalizeValues.x.x, normalizeValues.y.x, normalizeValues.z.x), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue1, math::float3(normalizeValues.x.y, normalizeValues.y.y, normalizeValues.z.y), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue2, math::float3(normalizeValues.x.z, normalizeValues.y.z, normalizeValues.z.z), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue3, math::float3(normalizeValues.x.w, normalizeValues.y.w, normalizeValues.z.w), math::epsilon_normal_sqrt()));
    }

    // Reference function is math::normalize(float2)
    TEST(normalize2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 values = math::floatNx2(math::floatN(0.1f), math::floatN(0.02f));
        math::floatNx2 normalizeValues = math::normalize(values);
        math::float2 expectedNormalizeValue0 = math::normalize(math::float2(values.x.x, values.y.x));
        math::float2 expectedNormalizeValue1 = math::normalize(math::float2(values.x.y, values.y.y));
        math::float2 expectedNormalizeValue2 = math::normalize(math::float2(values.x.z, values.y.z));
        math::float2 expectedNormalizeValue3 = math::normalize(math::float2(values.x.w, values.y.w));

        CHECK(math::compare_approx(expectedNormalizeValue0, math::float2(normalizeValues.x.x, normalizeValues.y.x), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue1, math::float2(normalizeValues.x.y, normalizeValues.y.y), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue2, math::float2(normalizeValues.x.z, normalizeValues.y.z), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue3, math::float2(normalizeValues.x.w, normalizeValues.y.w), math::epsilon_normal_sqrt()));
    }

    // Reference function is math::normalizeSafe(float3)
    TEST(normalizeSafe3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 normalizeValues = math::normalizeSafe(values);
        math::float3 expectedNormalizeValue0 = math::normalizeSafe(math::float3(values.x.x, values.y.x, values.z.x));
        math::float3 expectedNormalizeValue1 = math::normalizeSafe(math::float3(values.x.y, values.y.y, values.z.y));
        math::float3 expectedNormalizeValue2 = math::normalizeSafe(math::float3(values.x.z, values.y.z, values.z.z));
        math::float3 expectedNormalizeValue3 = math::normalizeSafe(math::float3(values.x.w, values.y.w, values.z.w));

        CHECK(math::compare_approx(expectedNormalizeValue0, math::float3(normalizeValues.x.x, normalizeValues.y.x, normalizeValues.z.x), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue1, math::float3(normalizeValues.x.y, normalizeValues.y.y, normalizeValues.z.y), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue2, math::float3(normalizeValues.x.z, normalizeValues.y.z, normalizeValues.z.z), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue3, math::float3(normalizeValues.x.w, normalizeValues.y.w, normalizeValues.z.w), math::epsilon_normal_sqrt()));
    }

    // Reference function is math::normalizeSafe(float2)
    TEST(normalizeSafe2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 values = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 normalizeValues = math::normalizeSafe(values);
        math::float2 expectedNormalizeValue0 = math::normalizeSafe(math::float2(values.x.x, values.y.x));
        math::float2 expectedNormalizeValue1 = math::normalizeSafe(math::float2(values.x.y, values.y.y));
        math::float2 expectedNormalizeValue2 = math::normalizeSafe(math::float2(values.x.z, values.y.z));
        math::float2 expectedNormalizeValue3 = math::normalizeSafe(math::float2(values.x.w, values.y.w));

        CHECK(math::compare_approx(expectedNormalizeValue0, math::float2(normalizeValues.x.x, normalizeValues.y.x), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue1, math::float2(normalizeValues.x.y, normalizeValues.y.y), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue2, math::float2(normalizeValues.x.z, normalizeValues.y.z), math::epsilon_normal_sqrt()));
        CHECK(math::compare_approx(expectedNormalizeValue3, math::float2(normalizeValues.x.w, normalizeValues.y.w), math::epsilon_normal_sqrt()));
    }

    // Reference function is math::floor(float4)
    TEST(floor3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 floorValues = math::floor(values);
        math::floatNx3 expectedFloorValues = math::floatNx3(math::floor(values.x), math::floor(values.y), math::floor(values.z));

        CHECK(math::all(expectedFloorValues.x == floorValues.x) && math::all(expectedFloorValues.y == floorValues.y) && math::all(expectedFloorValues.z == floorValues.z));
    }

    // Reference function is math::floor(float4)
    TEST(floor2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 values = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 floorValues = math::floor(values);
        math::floatNx2 expectedFloorValues = math::floatNx2(math::floor(values.x), math::floor(values.y));

        CHECK(math::all(expectedFloorValues.x == floorValues.x) && math::all(expectedFloorValues.y == floorValues.y));
    }

    // Reference function is math::saturate(float4)
    TEST(saturate4_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx4 values = math::floatNx4(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f), math::floatN(356322.1f));
        math::floatNx4 saturateValues = math::saturate(values);
        math::floatNx4 expectedSaturateValues = math::floatNx4(math::saturate(values.x), math::saturate(values.y), math::saturate(values.z), math::saturate(values.w));

        CHECK(math::all(expectedSaturateValues.x == saturateValues.x) && math::all(expectedSaturateValues.y == saturateValues.y) && math::all(expectedSaturateValues.z == saturateValues.z) && math::all(expectedSaturateValues.w == saturateValues.w));
    }

    // Reference function is math::saturate(float4)
    TEST(saturate3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 saturateValues = math::saturate(values);
        math::floatNx3 expectedSaturateValues = math::floatNx3(math::saturate(values.x), math::saturate(values.y), math::saturate(values.z));

        CHECK(math::all(expectedSaturateValues.x == saturateValues.x) && math::all(expectedSaturateValues.y == saturateValues.y) && math::all(expectedSaturateValues.z == saturateValues.z));
    }

    // Reference function is math::saturate(float4)
    TEST(saturate2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 values = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 saturateValues = math::saturate(values);
        math::floatNx2 expectedSaturateValues = math::floatNx2(math::saturate(values.x), math::saturate(values.y));

        CHECK(math::all(expectedSaturateValues.x == saturateValues.x) && math::all(expectedSaturateValues.y == saturateValues.y));
    }

    // Reference function is math::lerp(float4)
    TEST(lerp4_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx4 valuesA = math::floatNx4(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f), math::floatN(4564.0f));
        math::floatNx4 valuesB = math::floatNx4(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f), math::floatN(-6.0f));
        math::floatN valuesC = math::float4(0.0f, 0.4f, 2.4f, -0.7f);
        math::floatNx4 lerpValues = math::lerp(valuesA, valuesB, valuesC);
        math::floatNx4 expectedLerpValues = math::floatNx4(math::lerp(valuesA.x, valuesB.x, valuesC), math::lerp(valuesA.y, valuesB.y, valuesC), math::lerp(valuesA.z, valuesB.z, valuesC), math::lerp(valuesA.w, valuesB.w, valuesC));

        CHECK(math::all(expectedLerpValues.x == lerpValues.x) && math::all(expectedLerpValues.y == lerpValues.y) && math::all(expectedLerpValues.z == lerpValues.z));
    }

    // Reference function is math::lerp(float4)
    TEST(lerp3_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 valuesA = math::floatNx3(math::floatN(0.1f), math::floatN(0.0f), math::floatN(-1.5f));
        math::floatNx3 valuesB = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatN valuesC = math::float4(0.0f, 0.4f, 2.4f, -0.7f);
        math::floatNx3 lerpValues = math::lerp(valuesA, valuesB, valuesC);
        math::floatNx3 expectedLerpValues = math::floatNx3(math::lerp(valuesA.x, valuesB.x, valuesC), math::lerp(valuesA.y, valuesB.y, valuesC), math::lerp(valuesA.z, valuesB.z, valuesC));

        CHECK(math::all(expectedLerpValues.x == lerpValues.x) && math::all(expectedLerpValues.y == lerpValues.y) && math::all(expectedLerpValues.z == lerpValues.z));
    }

    // Reference function is math::lerp(float4)
    TEST(lerp2_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx2 valuesA = math::floatNx2(math::floatN(0.1f), math::floatN(0.0f));
        math::floatNx2 valuesB = math::floatNx2(math::floatN(-345.5f), math::floatN(100.0f));
        math::floatN valuesC = math::float4(0.0f, 0.4f, 2.4f, -0.7f);
        math::floatNx2 lerpValues = math::lerp(valuesA, valuesB, valuesC);
        math::floatNx2 expectedLerpValues = math::floatNx2(math::lerp(valuesA.x, valuesB.x, valuesC), math::lerp(valuesA.y, valuesB.y, valuesC));

        CHECK(math::all(expectedLerpValues.x == lerpValues.x) && math::all(expectedLerpValues.y == lerpValues.y));
    }

    // Reference function is math::mul(affineX, float3)
    TEST(mul3_affine_GivesSameResultsAs_ReferenceImpl)
    {
        math::affineX affine = math::affineCompose(math::float3(1.0f, 2.0f, 3.0f), math::eulerToQuat(math::float3(1.0f, 2.0f, 3.0f)));
        math::floatNx3 values = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatNx3 mulValues = math::mul(affine, values);
        math::float3 expectedMulValue0 = math::mul(affine, math::float3(values.x.x, values.y.x, values.z.x));
        math::float3 expectedMulValue1 = math::mul(affine, math::float3(values.x.y, values.y.y, values.z.y));
        math::float3 expectedMulValue2 = math::mul(affine, math::float3(values.x.z, values.y.z, values.z.z));
        math::float3 expectedMulValue3 = math::mul(affine, math::float3(values.x.w, values.y.w, values.z.w));

        CHECK(math::compare_approx(expectedMulValue0, math::float3(mulValues.x.x, mulValues.y.x, mulValues.z.x), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedMulValue1, math::float3(mulValues.x.y, mulValues.y.y, mulValues.z.y), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedMulValue2, math::float3(mulValues.x.z, mulValues.y.z, mulValues.z.z), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedMulValue3, math::float3(mulValues.x.w, mulValues.y.w, mulValues.z.w), math::epsilon_scale()));
    }

    // Reference function is math::mul(float3x3, float3)
    TEST(mul3_float3x3_GivesSameResultsAs_ReferenceImpl)
    {
        math::float3x3 rotation;
        math::quatToMatrix(math::eulerToQuat(math::float3(1.0f, 2.0f, 3.0f)), rotation);
        math::floatNx3 values = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatNx3 mulValues = math::mul(rotation, values);
        math::float3 expectedMulValue0 = math::mul(rotation, math::float3(values.x.x, values.y.x, values.z.x));
        math::float3 expectedMulValue1 = math::mul(rotation, math::float3(values.x.y, values.y.y, values.z.y));
        math::float3 expectedMulValue2 = math::mul(rotation, math::float3(values.x.z, values.y.z, values.z.z));
        math::float3 expectedMulValue3 = math::mul(rotation, math::float3(values.x.w, values.y.w, values.z.w));

        CHECK(math::compare_approx(expectedMulValue0, math::float3(mulValues.x.x, mulValues.y.x, mulValues.z.x), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedMulValue1, math::float3(mulValues.x.y, mulValues.y.y, mulValues.z.y), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedMulValue2, math::float3(mulValues.x.z, mulValues.y.z, mulValues.z.z), math::epsilon_scale()));
        CHECK(math::compare_approx(expectedMulValue3, math::float3(mulValues.x.w, mulValues.y.w, mulValues.z.w), math::epsilon_scale()));
    }

    // Reference function is EulerToMatrix(Vector3f, Matrix3x3f)
    TEST(eulerToMatrix_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatNx3 values = math::floatNx3(math::floatN(-345.5f), math::floatN(100.0f), math::floatN(-1543.9f));
        math::floatNx3 matrixValues[3];
        math::eulerToMatrix(values, matrixValues);
        Matrix3x3f expectedMatrixValues[4];
        EulerToMatrix(Vector3f(values.x.x, values.y.x, values.z.x), expectedMatrixValues[0]);
        EulerToMatrix(Vector3f(values.x.y, values.y.y, values.z.y), expectedMatrixValues[1]);
        EulerToMatrix(Vector3f(values.x.z, values.y.z, values.z.z), expectedMatrixValues[2]);
        EulerToMatrix(Vector3f(values.x.w, values.y.w, values.z.w), expectedMatrixValues[3]);

        for (int m = 0; m < 4; m++)
        {
            for (int x = 0; x < 3; x++)
            {
                for (int y = 0; y < 3; y++)
                {
                    CHECK(CompareApproximately(expectedMatrixValues[m].Get(x, y), math::extract(*((&matrixValues[y].x) + x), m), 0.01f)); // Tolerance needs to be very large. May want to investigate why.
                }
            }
        }
    }

    // Reference function is GammaToLinearSpace(float)
    TEST(GammaToLinearSpaceApprox_GivesSameResultsAs_ReferenceImpl)
    {
        math::floatN values = math::float4(0.01f, 0.8f, 1.7f, 0.2f);
        math::floatN linearValues = math::GammaToLinearSpaceApprox(values);
        math::floatN expectedLinearValues = math::float4(GammaToLinearSpace(values.x), GammaToLinearSpace(values.y), GammaToLinearSpace(values.z), GammaToLinearSpace(values.w));

        CHECK(math::compare_approx(expectedLinearValues, linearValues, 0.01f)); // Tolerance needs to be very large. May want to investigate why.
    }
}
#endif

#if ENABLE_PERFORMANCE_TESTS

#include "Runtime/Testing/PerformanceTesting.h"
PERFORMANCE_TEST_SUITE(SIMDMath_SoAOps)
{
    template<int ISize, int IIterations>
    struct convert_pixN_PerfFixture
    {
        convert_pixN_PerfFixture()
        {
            FillPerformanceTestData(reinterpret_cast<float*>(&data[0].x), kSize * math::kSimdWidth * 4, 0, 1.0f);
        }

        enum { kSize = ISize, kIterations = IIterations };

        math::floatNx4 data[kSize];
        math::pixN output[kSize];
    };

    struct convert_pixN_Fixture : public convert_pixN_PerfFixture<1000, 100000> {};

    TEST_FIXTURE(convert_pixN_Fixture, reference_convert_pixN_Perf)
    {
        PERFORMANCE_TEST_LOOP(kIterations)
        {
            OPTIMIZER_PREVENT(data);
            for (int i = 0; i < kSize; ++i)
                output[i] = reference_convert_pixN(data[i]);
            OPTIMIZER_PREVENT(output);
        }
    }

    TEST_FIXTURE(convert_pixN_Fixture, convert_pixN_Perf)
    {
        PERFORMANCE_TEST_LOOP(kIterations)
        {
            OPTIMIZER_PREVENT(data);
            for (int i = 0; i < kSize; ++i)
                output[i] = math::convert_pixN(data[i]);
            OPTIMIZER_PREVENT(output);
        }
    }

    template<int ISize, int IIterations>
    struct convert_floatNx4_PerfFixture
    {
        convert_floatNx4_PerfFixture()
        {
            FillPerformanceTestData(reinterpret_cast<UInt32*>(&data[0].i), kSize * 4, 0u, 0xffffffff);
        }

        enum { kSize = ISize, kIterations = IIterations };

        math::pixN data[kSize];
        math::floatNx4 output[kSize];
    };

    struct convert_floatNx4_Fixture : public convert_floatNx4_PerfFixture<1000, 100000> {};

    TEST_FIXTURE(convert_floatNx4_Fixture, reference_convert_floatNx4_Perf)
    {
        PERFORMANCE_TEST_LOOP(kIterations)
        {
            OPTIMIZER_PREVENT(data);
            for (int i = 0; i < kSize; ++i)
                output[i] = reference_convert_floatNx4(data[i]);
            OPTIMIZER_PREVENT(output);
        }
    }

    TEST_FIXTURE(convert_floatNx4_Fixture, convert_floatNx4_Perf)
    {
        PERFORMANCE_TEST_LOOP(kIterations)
        {
            OPTIMIZER_PREVENT(data);
            for (int i = 0; i < kSize; ++i)
                output[i] = math::convert_floatNx4(data[i]);
            OPTIMIZER_PREVENT(output);
        }
    }
}
#endif // #if ENABLE_PERFORMANCE_TESTS
