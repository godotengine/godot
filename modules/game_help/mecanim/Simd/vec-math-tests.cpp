#include "UnityPrefix.h"

#if ENABLE_UNIT_TESTS

#if PLATFORM_WIN
#pragma warning(push)
#pragma warning(disable: 4723) // potential division by zero
#endif


#include "Runtime/Testing/Testing.h"
#include "Runtime/Profiler/TimeHelper.h"

#include "vec-math.h"

UNIT_TEST_SUITE(SIMDMath_BaseOps)
{
    using namespace math;
    const float epsilon = 1e-5f;
    const float approximationEpsilon = 2e-4f;

    const SInt32 floatOne = 0x3F800000;
    const SInt32 floatNegativeOne = 0xBF800000;

    TEST(as_int4_WithZero_ReturnsZero)
    {
        float4 zero = float4(ZERO);
        int4 isZero = as_int4(zero);

        CHECK(all(isZero == int4(ZERO)));
    }

    TEST(as_int4_WithOne_Returns0X3F800000)
    {
        float4 one = float4(1.0f);
        int4 is0X3F800000 = as_int4(one);

        CHECK(all(is0X3F800000 == int4(floatOne)));
    }

    TEST(as_int4_WithNegativeOne_Returns0XBF800000)
    {
        float4 one = float4(-1.0f);
        int4 is0XBF800000 = as_int4(one);

        CHECK(all(is0XBF800000 == int4(floatNegativeOne)));
    }

    TEST(as_int3_WithZero_ReturnsZero)
    {
        float3 zero = float3(ZERO);
        int3 isZero = as_int3(zero);

        CHECK(all(isZero == int3(ZERO)));
    }

    TEST(as_int3_WithOne_Returns0X3F800000)
    {
        float3 one = float3(1.0f);
        int3 is0X3F800000 = as_int3(one);

        CHECK(all(is0X3F800000 == int3(floatOne)));
    }

    TEST(as_int3_WithNegativeOne_Returns0XBF800000)
    {
        float3 one = float3(-1.0f);
        int3 is0XBF800000 = as_int3(one);

        CHECK(all(is0XBF800000 == int3(floatNegativeOne)));
    }

    TEST(as_int2_WithZero_ReturnsZero)
    {
        float2 zero = float2(ZERO);
        int2 isZero = as_int2(zero);

        CHECK(all(isZero == int2(ZERO)));
    }

    TEST(as_int2_WithOne_Returns0X3F800000)
    {
        float2 one = float2(1.0f);
        int2 is0X3F800000 = as_int2(one);

        CHECK(all(is0X3F800000 == int2(floatOne)));
    }

    TEST(as_int2_WithNegativeOne_Returns0XBF800000)
    {
        float2 one = float2(-1.0f);
        int2 is0XBF800000 = as_int2(one);

        CHECK(all(is0XBF800000 == int2(floatNegativeOne)));
    }

    TEST(as_float4_WithZero_ReturnsZero)
    {
        int4 zero = int4(ZERO);
        float4 isZero = as_float4(zero);

        CHECK(all(isZero == float4(ZERO)));
    }

    TEST(as_float4_With0X3F800000_ReturnsOne)
    {
        int4 value = int4(floatOne);
        float4 isOne = as_float4(value);

        CHECK(all(isOne == float4(1.0f)));
    }

    TEST(as_float4_With0XBF800000_ReturnsNegativeOne)
    {
        int4 value = int4(floatNegativeOne);
        float4 isNegativeOne = as_float4(value);

        CHECK(all(isNegativeOne == float4(-1.0f)));
    }

    TEST(as_float3_WithZero_ReturnsZero)
    {
        int3 zero = int3(ZERO);
        float3 isZero = as_float3(zero);

        CHECK(all(isZero == float3(ZERO)));
    }

    TEST(as_float3_With0X3F800000_ReturnsOne)
    {
        int3 value = int3(floatOne);
        float3 isOne = as_float3(value);

        CHECK(all(isOne == float3(1.0f)));
    }

    TEST(as_float3_With0XBF800000_ReturnsNegativeOne)
    {
        int3 value = int3(floatNegativeOne);
        float3 isNegativeOne = as_float3(value);

        CHECK(all(isNegativeOne == float3(-1.0f)));
    }

    TEST(as_float2_WithZero_ReturnsZero)
    {
        int2 zero = int2(ZERO);
        float2 isZero = as_float2(zero);

        CHECK(all(isZero == float2(ZERO)));
    }

    TEST(as_float2_With0X3F800000_ReturnsOne)
    {
        int2 value = int2(floatOne);
        float2 isOne = as_float2(value);

        CHECK(all(isOne == float2(1.0f)));
    }

    TEST(as_float2_With0XBF800000_ReturnsNegativeOne)
    {
        int2 value = int2(floatNegativeOne);
        float2 isNegativeOne = as_float2(value);

        CHECK(all(isNegativeOne == float2(-1.0f)));
    }

    TEST(convert_int4_WithZero_ReturnsZero)
    {
        float4 zero = float4(ZERO);
        int4 isZero = convert_int4(zero);

        CHECK(all(isZero == int4(ZERO)));
    }

    TEST(convert_int4_WithOne_ReturnsOne)
    {
        float4 one = float4(1.0f);
        int4 isOne = convert_int4(one);

        CHECK(all(isOne == int4(1)));
    }

    TEST(convert_int4_WithNegativeOne_ReturnsNegativeOne)
    {
        float4 negativeOne = float4(-1.0f);
        int4 isNegativeOne = convert_int4(negativeOne);

        CHECK(all(isNegativeOne == int4(-1)));
    }

    TEST(convert_int4_WithFractionnalPart_ReturnsTruncatedValue)
    {
        CHECK(all(convert_int4(float4(-9.0f / 4.0f)) == int4(-9 / 4)));
        CHECK(all(convert_int4(float4(14.0f / 3.0f)) == int4(14 / 3)));
    }

    TEST(convert_int3_WithOne_ReturnsOne)
    {
        float3 one = float3(1.0f);
        int3 isOne = convert_int3(one);

        CHECK(all(isOne == int3(1)));
    }

    TEST(convert_int2_WithOne_ReturnsOne)
    {
        float2 one = float2(1.0f);
        int2 isOne = convert_int2(one);

        CHECK(all(isOne == int2(1)));
    }

    TEST(convert_float4_WithZero_ReturnsZero)
    {
        int4 zero = int4(ZERO);
        float4 isZero = convert_float4(zero);

        CHECK(all(isZero == float4(ZERO)));
    }

    TEST(convert_float4_WithOne_ReturnsOne)
    {
        int4 one = int4(1);
        float4 isOne = convert_float4(one);

        CHECK(all(isOne == float4(1.0f)));
    }

    TEST(convert_float4_WithNegativeOne_ReturnsNegativeOne)
    {
        int4 negativeOne = int4(-1);
        float4 isNegativeOne = convert_float4(negativeOne);

        CHECK(all(isNegativeOne == float4(-1.0f)));
    }

    TEST(convert_float3_WithOne_ReturnsOne)
    {
        int3 one = int3(1);
        float3 isOne = convert_float3(one);

        CHECK(all(isOne == float3(1.0f)));
    }

    TEST(convert_float2_WithOne_ReturnsOne)
    {
        int2 one = int2(1);
        float2 isOne = convert_float2(one);

        CHECK(all(isOne == float2(1.0f)));
    }

    TEST(bitselect_int4_Works)
    {
        int4 a = int4(1, 2, 3, 4);
        int4 b = int4(5, 6, 7, 8);

        int4 selectA = int4(0, 0, 0, 0);
        int4 selectB = int4(~0, ~0, ~0, ~0);
        int4 mix = int4(0, ~0, 0, ~0);

        int4 result = bitselect(a, b, selectA);
        CHECK(all(result == a));

        result = bitselect(a, b, selectB);
        CHECK(all(result == b));

        result = bitselect(a, b, mix);
        CHECK(all(result == int4(1, 6, 3, 8)));
    }

    TEST(bitselect_int3_Works)
    {
        int3 a = int3(1, 2, 3);
        int3 b = int3(5, 6, 7);

        int3 selectA = int3(0, 0, 0);
        int3 selectB = int3(~0, ~0, ~0);
        int3 mix = int3(0, ~0, 0);

        int3 result = bitselect(a, b, selectA);
        CHECK(all(result == a));

        result = bitselect(a, b, selectB);
        CHECK(all(result == b));

        result = bitselect(a, b, mix);
        CHECK(all(result == int3(1, 6, 3)));
    }

    TEST(bitselect_int2_Works)
    {
        int2 a = int2(1, 2);
        int2 b = int2(5, 6);

        int2 selectA = int2(0, 0);
        int2 selectB = int2(~0, ~0);
        int2 mix = int2(0, ~0);

        int2 result = bitselect(a, b, selectA);
        CHECK(all(result == a));

        result = bitselect(a, b, selectB);
        CHECK(all(result == b));

        result = bitselect(a, b, mix);
        CHECK(all(result == int2(1, 6)));
    }

    TEST(bitselect_int1_Works)
    {
        int1 a = int1(1);
        int1 b = int1(5);

        int1 selectA = int1(0);
        int1 selectB = int1(~0);

        int1 result = bitselect(a, b, selectA);
        CHECK(result == a);

        result = bitselect(a, b, selectB);
        CHECK(result == b);
    }

    TEST(bitselect_int_Works)
    {
        int a = 1;
        int b = 5;

        int selectA = 0;
        int selectB = ~0;

        int result = bitselect(a, b, selectA);
        CHECK(result == a);

        result = bitselect(a, b, selectB);
        CHECK(result == b);
    }

    TEST(bitselect_float4_Works)
    {
        float4 a = float4(1.f, 2.f, 3.f, 4.f);
        float4 b = float4(5.f, 6.f, 7.f, 8.f);

        int4 selectA = int4(0, 0, 0, 0);
        int4 selectB = int4(~0, ~0, ~0, ~0);
        int4 mix = int4(0, ~0, 0, ~0);

        float4 result = bitselect(a, b, selectA);
        CHECK(all(result == a));

        result = bitselect(a, b, selectB);
        CHECK(all(result == b));

        result = bitselect(a, b, mix);
        CHECK(all(result == float4(1.f, 6.f, 3.f, 8.f)));
    }

    TEST(bitselect_float3_Works)
    {
        float3 a = float3(1.f, 2.f, 3.f);
        float3 b = float3(5.f, 6.f, 7.f);

        int3 selectA = int3(0, 0, 0);
        int3 selectB = int3(~0, ~0, ~0);
        int3 mix = int3(0, ~0, 0);

        float3 result = bitselect(a, b, selectA);
        CHECK(all(result == a));

        result = bitselect(a, b, selectB);
        CHECK(all(result == b));

        result = bitselect(a, b, mix);
        CHECK(all(result == float3(1.f, 6.f, 3.f)));
    }

    TEST(bitselect_float2_Works)
    {
        float2 a = float2(1.f, 2.f);
        float2 b = float2(5.f, 6.f);

        int2 selectA = int2(0, 0);
        int2 selectB = int2(~0, ~0);
        int2 mix = int2(0, ~0);

        float2 result = bitselect(a, b, selectA);
        CHECK(all(result == a));

        result = bitselect(a, b, selectB);
        CHECK(all(result == b));

        result = bitselect(a, b, mix);
        CHECK(all(result == float2(1.f, 6.f)));
    }

    TEST(bitselect_float1_Works)
    {
        float1 a = float1(1);
        float1 b = float1(5);

        int1 selectA = int1(0);
        int1 selectB = int1(~0);

        float1 result = bitselect(a, b, selectA);
        CHECK(result == a);

        result = bitselect(a, b, selectB);
        CHECK(result == b);
    }

    TEST(bitselect_float_Works)
    {
        float a = 1.0f;
        float b = 5.0f;

        int selectA = 0;
        int selectB = ~0;

        float result = bitselect(a, b, selectA);
        CHECK(result == a);

        result = bitselect(a, b, selectB);
        CHECK(result == b);
    }

    TEST(select_int4_Works)
    {
        int4 a = int4(1, 2, 3, 4);
        int4 b = int4(5, 6, 7, 8);

        int4 selectA = int4(0);
        int4 selectB = int4(-1);
        int4 mix = int4(0, -1, 0, -1);

        int4 result = select(a, b, selectA);
        CHECK(all(result == a));

        result = select(a, b, selectB);
        CHECK(all(result == b));

        result = select(a, b, mix);
        CHECK(all(result == int4(1, 6, 3, 8)));
    }

    TEST(select_int_Works)
    {
        int a = 1;
        int b = 5;

        int selectA = 0;
        int selectB = -1;

        int result = select(a, b, selectA);
        CHECK(result == a);

        result = select(a, b, selectB);
        CHECK(result == b);
    }

    TEST(select_float4_Works)
    {
        const float4 a = float4(-1.f, -.263f, 345.f, 0.f);
        const float4 b = float4(5.f, 2.34f, -12.76f, 54.f);
        float4 c;

        // res = msb(c) ? b : a;
        c = select(a, b, int4(0));
        CHECK(all(c == a));

        // (and the result of a compare is -1 -> true)
        c = select(a, b, int4(-1));
        CHECK(all(c == b));

        // Same as when doing a compare
        c = select(a, b, float4(0.0F) < float4(1.0F));
        CHECK(all(c == b));

        // Somewhat counter intuitive but select works against most msb only
        c = select(a, b, int4(1));
        CHECK(all(c == a));

        c = select(a, b, int4(53));
        CHECK(all(c == a));

        c = select(a, b, int4(-53));
        CHECK(all(c == b));

        c = select(a, b, int4(~0));
        CHECK(all(c == b));

        c = select(a, b, float4(0.0F) > float4(1.0F));
        CHECK(all(c == a));

        c = select(a, b, a < b);
        CHECK(all(c == float4(5.F, 2.34f, 345.f, 54.f)));

        c = select(a, b, a > b);
        CHECK(all(c == float4(-1.f, -.263f, -12.76f, 0.f)));

        float4 d;

        int index = -1;
        d = select(float4(3.f, 3.f, 3.f, 3.f), float4(2.f, 2.f, 2.f, 2.f),  int4(-(index != -1)));
        CHECK(all(d == float4(3.f, 3.f, 3.f, 3.f)));

        index = 1;
        d = select(float4(3.f, 3.f, 3.f, 3.f), float4(2.f, 2.f, 2.f, 2.f),  int4(-(index != -1)));
        CHECK(all(d == float4(2.f, 2.f, 2.f, 2.f)));

        d = select(float4(0.0f), float4(1.0f), float4(0.0f) != float4(0.0f));
        CHECK(all(d == float4(0.f)));

        d = select(float4(0.0f), float4(1.0f), float4(0.3f) != float4(0.0f));
        CHECK(all(d == float4(1.f)));
    }

    TEST(select_float3_Works)
    {
        float3 a = float3(1.f, 2.f, 3.f);
        float3 b = float3(5.f, 6.f, 7.f);

        int3 selectA = int3(0);
        int3 selectB = int3(-1);
        int3 mix = int3(0, -1, 0);

        float3 result = select(a, b, selectA);
        CHECK(all(result == a));

        result = select(a, b, selectB);
        CHECK(all(result == b));

        result = select(a, b, mix);
        CHECK(all(result == float3(1.f, 6.f, 3.f)));
    }

    TEST(select_float2_Works)
    {
        float2 a = float2(1.f, 2.f);
        float2 b = float2(5.f, 6.f);

        int2 selectA = int2(0);
        int2 selectB = int2(-1);
        int2 mix = int2(0, -1);

        float2 result = select(a, b, selectA);
        CHECK(all(result == a));

        result = select(a, b, selectB);
        CHECK(all(result == b));

        result = select(a, b, mix);
        CHECK(all(result == float2(1.f, 6.f)));
    }

    TEST(select_float1_Works)
    {
        float1 a = float1(1.f);
        float1 b = float1(5.f);

        int1 selectA = int1(0);
        int1 selectB = int1(-1);

        float1 result = select(a, b, selectA);
        CHECK(result == a);

        result = select(a, b, selectB);
        CHECK(result == b);
    }

    TEST(select_float_Works)
    {
        float a = 1.f;
        float b = 5.f;

        int selectA = 0;
        int selectB = -1;

        float result = select(a, b, selectA);
        CHECK(result == a);

        result = select(a, b, selectB);
        CHECK(result == b);
    }

    TEST(cond_float4_Work)
    {
        // float4 cond(const int4 &c, const float4 &a, const float4 &b)
        const float4 a4 = float4(1.f, -2.f, 3.f, -4.f);
        const float4 b4 = float4(-5.f, 6.f, -7.f, 8.f);
        float4 res4;

        res4 = cond(true, a4, b4);
        CHECK(all(res4 == a4));

        res4 = cond(false, a4, b4);
        CHECK(all(res4 == b4));
    }

    TEST(cond_float3_Work)
    {
        // float3 cond(const int3 &c, const float3 &a, const float3 &b)
        const float3 a3 = float3(1.f, -2.f, 3.f);
        const float3 b3 = float3(-5.f, 6.f, -7.f);
        float3 res3;

        res3 = cond(true, a3, b3);
        CHECK(all(res3 == a3));

        res3 = cond(false, a3, b3);
        CHECK(all(res3 == b3));
    }

    TEST(cond_float2_Work)
    {
        // float2 cond(bool c, const float2 &a, const float2 &b)
        const float2 a2 = float2(1.f, -2.f);
        const float2 b2 = float2(-5.f, 6.f);
        float2 res2;

        res2 = cond(true, a2, b2);
        CHECK(all(res2 == a2));

        res2 = cond(false, a2, b2);
        CHECK(all(res2 == b2));
    }

    TEST(cond_float1_Work)
    {
        // float1 cond(const int1 &c, const float1 &a, const float1 &b)
        const float1 a1 = float1(1.f);
        const float1 b1 = float1(0.f);
        float1 res1;

        res1 = cond(true, a1, b1);
        CHECK(res1 == a1);

        res1 = cond(false, a1, b1);
        CHECK(res1 == b1);

        res1 = cond(a1 > b1, a1, b1);
        CHECK(res1 == a1);

        res1 = cond(a1 < b1, a1, b1);
        CHECK(res1 == b1);

        // float cond(int c, float a, float b)
        float res = cond(true, 1.f, 2.f);
        CHECK(res == 1.f);

        res = cond(false, 1.f, 2.f);
        CHECK(res == 2.f);

        // test type promotion from bool to int
        res = cond(1.f > 2.f, 1.f, 2.f);
        CHECK(res == 2.f);

        res = cond(1.f < 2.f, 1.f, 2.f);
        CHECK(res == 1.f);
    }

    TEST(sign_int4_Works)
    {
        int4 c = sign(int4(-25, 0, 23, -0));
        CHECK(all(c == int4(-1, 0, 1, 0)));
    }

    TEST(sign_int3_Works)
    {
        int3 c = sign(int3(-25, 0, 23));
        CHECK(all(c == int3(-1, 0, 1)));

        c = sign(int3(-0));
        CHECK(all(c == int3(0)));
    }

    TEST(sign_int2_Works)
    {
        int2 c = sign(int2(-25, 0));
        CHECK(all(c == int2(-1, 0)));

        c = sign(int2(23, -0));
        CHECK(all(c == int2(1, 0)));
    }

    TEST(sign_int1_Works)
    {
        int1 c = sign(int1(-25));
        CHECK(c == int1(-1));

        c = sign(int1(23));
        CHECK(c == int1(1));

        c = sign(int1(0));
        CHECK(c == int1(0));

        c = sign(int1(-0));
        CHECK(c == int1(0));
    }

    TEST(sign_int_Works)
    {
        int c = sign(-25);
        CHECK(c == -1);

        c = sign(23);
        CHECK(c == 1);

        c = sign(0);
        CHECK(c == 0);

        c = sign(-0);
        CHECK(c == 0);
    }

    TEST(sign_float4_Works)
    {
        float4 c = sign(float4(-25.f, 0.f, 23.5f, -0.f));
        CHECK(all(c == float4(-1, 0.f, 1.f, 0.f)));
    }

    TEST(sign_float3_Works)
    {
        float3 c = sign(float3(-25.f, 0.f, 23.5f));
        CHECK(all(c == float3(-1.f, 0.f, 1.f)));

        c = sign(float3(-0.f));
        CHECK(all(c == float3(0.f)));
    }

    TEST(sign_float2_Works)
    {
        float2 c = sign(float2(-25.f, 0.f));
        CHECK(all(c == float2(-1.f, 0.f)));

        c = sign(float2(23.f, -0.f));
        CHECK(all(c == float2(1.f, 0.f)));
    }

    TEST(sign_float1_Works)
    {
        float1 c = sign(float1(-25.f));
        CHECK(c == float1(-1.f));

        c = sign(float1(23.f));
        CHECK(c == float1(1.f));

        c = sign(float1(0.f));
        CHECK(c == float1(0.f));

        c = sign(float1(-0.f));
        CHECK(c == float1(0.f));
    }

    TEST(sign_float_Works)
    {
        float c = sign(-25.f);
        CHECK(c == -1.f);

        c = sign(23.5f);
        CHECK(c == 1.f);

        c = sign(0.f);
        CHECK(c == 0.f);

        c = sign(-0.f);
        CHECK(c == 0.f);
    }

    TEST(floor_float4_Works)
    {
        float4 c = floor(float4(1.5f, -1.5f, 0.f, 2.99f));
        CHECK(all(c == float4(1.f, -2.f, 0.f, 2.f)));
    }

    TEST(floor_float3_Works)
    {
        float3 c = floor(float3(1.5f, -1.5f, 0.f));
        CHECK(all(c == float3(1.f, -2.f, 0.f)));
    }

    TEST(floor_float2_Works)
    {
        float2 c = floor(float2(1.5f, -1.5f));
        CHECK(all(c == float2(1.f, -2.f)));

        c = floor(float2(0.f, 2.99f));
        CHECK(all(c == float2(0.f, 2.f)));
    }

    TEST(floor_float1_Works)
    {
        float1 c = math::floor(float1(1.5f));
        CHECK(c == float1(1.f));

        c = math::floor(float1(0.f));
        CHECK(c == float1(0.f));

        c = math::floor(float1(-1.5f));
        CHECK(c == float1(-2.f));

        c = math::floor(float1(2.99f));
        CHECK(c == float1(2.f));
    }

    TEST(floor_float_Works)
    {
        float c = math::floor(1.5f);
        CHECK(c == 1.f);

        c = math::floor(-1.5f);
        CHECK(c == -2.f);

        c = math::floor(3.f);
        CHECK(c == 3.f);

        c = math::floor(3.99f);
        CHECK(c == 3.f);

        c = math::floor(0.f);
        CHECK(c == 0.f);
    }

    TEST(ceil_float4_Works)
    {
        float4 c = ceil(float4(1.5f, -1.5f, 0.f, 2.99f));
        CHECK(all(c == float4(2.f, -1.f, 0.f, 3.f)));
    }

    TEST(ceil_float3_Works)
    {
        float3 c = ceil(float3(1.5f, -1.5f, 0.f));
        CHECK(all(c == float3(2.f, -1.f, 0.f)));
    }

    TEST(ceil_float2_Works)
    {
        float2 c = ceil(float2(1.5f, -1.5f));
        CHECK(all(c == float2(2.f, -1.f)));

        c = ceil(float2(0.f, 2.99f));
        CHECK(all(c == float2(0.f, 3.f)));
    }

    TEST(ceil_float1_Works)
    {
        float1 c = math::ceil(float1(1.5f));
        CHECK(c == float1(2.f));

        c = math::ceil(float1(0.f));
        CHECK(c == float1(0.f));

        c = math::ceil(float1(-1.5f));
        CHECK(c == float1(-1.f));

        c = math::ceil(float1(2.99f));
        CHECK(c == float1(3.f));
    }

    TEST(ceil_float_Works)
    {
        float c = math::ceil(1.5f);
        CHECK(c == 2.f);

        c = math::ceil(-1.5f);
        CHECK(c == -1.f);

        c = math::ceil(3.f);
        CHECK(c == 3.f);

        c = math::ceil(3.99f);
        CHECK(c == 4.f);

        c = math::ceil(0.f);
        CHECK(c == 0.f);
    }

    TEST(trunc_float4_Works)
    {
        float4 c = trunc(float4(1.5f, -1.5f, 0.f, 2.99f));
        CHECK(all(c == float4(1.f, -1.f, 0.f, 2.f)));
    }

    TEST(trunc_float3_Works)
    {
        float3 c = trunc(float3(1.5f, -1.5f, 0.f));
        CHECK(all(c == float3(1.f, -1.f, 0.f)));
    }

    TEST(trunc_float2_Works)
    {
        float2 c = trunc(float2(1.5f, -1.5f));
        CHECK(all(c == float2(1.f, -1.f)));

        c = trunc(float2(0.f, 2.99f));
        CHECK(all(c == float2(0.f, 2.f)));
    }

    TEST(trunc_float1_Works)
    {
        float1 c = math::trunc(float1(1.5f));
        CHECK(c == float1(1.f));

        c = math::trunc(float1(0.f));
        CHECK(c == float1(0.f));

        c = math::trunc(float1(-1.5f));
        CHECK(c == float1(-1.f));

        c = math::trunc(float1(2.99f));
        CHECK(c == float1(2.f));
    }

    TEST(trunc_float_Works)
    {
        float c = math::trunc(1.5f);
        CHECK(c == 1.f);

        c = math::trunc(-1.5f);
        CHECK(c == -1.f);

        c = math::trunc(3.f);
        CHECK(c == 3.f);

        c = math::trunc(3.99f);
        CHECK(c == 3.f);

        c = math::trunc(0.f);
        CHECK(c == 0.f);
    }

    TEST(chgsign_float4_Works)
    {
        float4 c = chgsign(float4(1.f, 2.f, 3.f, -4.f), float4(-1.f, 0.f, -0.f, -23.56f));
        CHECK(all(c == float4(-1.f, 2.f, -3.f, 4.f)));
    }

    TEST(chgsign_float3_Works)
    {
        float3 c = chgsign(float3(1.f, -2.f, 3.f), float3(-1.f, 23.548f, -0.f));
        CHECK(all(c == float3(-1.f, -2.f, -3.f)));
    }

    TEST(chgsign_float2_Works)
    {
        float2 c = chgsign(float2(1.f, -2.f), float2(-1.f, 23.548f));
        CHECK(all(c == float2(-1.f, -2.f)));
    }

    TEST(chgsign_float1_Works)
    {
        float1 c = chgsign(float1(1.f), float1(-1.f));
        CHECK(c == float1(-1.f));

        c = chgsign(float1(1.f), float1(1.f));
        CHECK(c == float1(1.f));

        c = chgsign(float1(-1.f), float1(-1.f));
        CHECK(c == float1(1.f));

        c = chgsign(float1(-1.f), float1(1.f));
        CHECK(c == float1(-1.f));

        c = chgsign(float1(-1.f), float1(0.f));
        CHECK(c == float1(-1.f));

        c = chgsign(float1(-1.f), float1(-0.f));
        CHECK(c == float1(1.f));
    }

    TEST(chgsign_float_Works)
    {
        float c = chgsign(1.f, -1.f);
        CHECK(c == -1.f);

        c = chgsign(1.f, 1.f);
        CHECK(c == 1.f);

        c = chgsign(-1.f, -1.f);
        CHECK(c == 1.f);

        c = chgsign(-1.f, 1.f);
        CHECK(c == -1.f);

        c = chgsign(-1.f, 0.f);
        CHECK(c == -1.f);

        c = chgsign(-1.f, -0.f);
        CHECK(c == 1.f);
    }

    TEST(round_float4_Works)
    {
        float4 c = round(float4(1.2f, 4.52f, -1.02f, -100.90f));
        CHECK(all(c == float4(1.f, 5.f, -1.f, -101.f)));

        c = round(float4(1.f, 0.f, -1.0f, -0.f));
        CHECK(all(c == float4(1.f, 0.f, -1.f, 0.f)));

        c = round(float4(-0.5f - epsilon));
        CHECK(all(c == float4(-1.f)));

        c = round(float4(-0.5f + epsilon));
        CHECK(all(c == float4(0.f)));
    }

    TEST(round_float3_Works)
    {
        float3 c = round(float3(1.2f, 4.52f, -1.02f));
        CHECK(all(c == float3(1.f, 5.f, -1.f)));

        c = round(float3(1.f, 0.f, -1.0f));
        CHECK(all(c == float3(1.f, 0.f, -1.f)));

        c = round(float3(-0.5f + epsilon));
        CHECK(all(c == float3(0.f)));
    }

    TEST(round_float2_Works)
    {
        float2 c = round(float2(1.2f, -1.02f));
        CHECK(all(c == float2(1.f, -1.f)));

        c = round(float2(1.f, 0.f));
        CHECK(all(c == float2(1.f, 0.f)));

        c = round(float2(-0.5f + epsilon));
        CHECK(all(c == float2(0.f)));
    }

    TEST(round_float1_Works)
    {
        float1 c = math::round(float1(1.2f));
        CHECK(c == float1(1.f));

        c = math::round(float1(-4.6f));
        CHECK(c == float1(-5.f));

        c = math::round(float1(3.f));
        CHECK(c == float1(3.f));

        c = math::round(float1(3.9f));
        CHECK(c == float1(4.f));

        c = math::round(float1(-0.5f + epsilon));
        CHECK(c == float1(0.f));
    }

    TEST(round_float_Works)
    {
        float c = math::round(1.2f);
        CHECK(c == 1.f);

        c = math::round(-4.6f);
        CHECK(c == -5.f);

        c = math::round(3.f);
        CHECK(c == 3.f);

        c = math::round(3.9f);
        CHECK(c == 4.f);

        c = math::round(-0.5f - epsilon);
        CHECK(c == -1.f);

        c = math::round(-0.5f + epsilon);
        CHECK(c == 0.f);
    }

    TEST(copysign_float4_Works)
    {
        float4 c = copysign(float4(1.f, 2.f, 3.f, -4.f), float4(-1.f, 0.f, -0.f, -23.56f));
        CHECK(all(c == float4(-1.f, 2.f, -3.f, -4.f)));
    }

    TEST(copysign_float3_Works)
    {
        float3 c = copysign(float3(1.f, -2.f, 3.f), float3(-1.f, 23.548f, -0.f));
        CHECK(all(c == float3(-1.f, 2.f, -3.f)));
    }

    TEST(copysign_float2_Works)
    {
        float2 c = copysign(float2(1.f, -2.f), float2(-1.f, 23.548f));
        CHECK(all(c == float2(-1.f, 2.f)));
    }

    // On UWP, writing simply copysign (without namespace) will cause an error:
    //2>G:\Projects\platform - metro - unit - tests\Runtime / Math / Simd / vec - test.cpp(870) : error C2668 : 'copysign' : ambiguous call to overloaded function
    //  2>  C:\Program Files(x86)\Microsoft Visual Studio 14.0\vc\include\cmath(391) : note : could be 'long double copysign(long double,long double) noexcept'
    //  2>  C:\Program Files(x86)\Microsoft Visual Studio 14.0\vc\include\cmath(95) : note : or 'float copysign(float,float) noexcept'
    TEST(copysign_float1_Works)
    {
        float1 c = math::copysign(float1(1.f), float1(-1.f));
        CHECK(c == float1(-1.f));

        c = math::copysign(float1(1.f), float1(1.f));
        CHECK(c == float1(1.f));

        c = math::copysign(float1(-1.f), float1(-1.f));
        CHECK(c == float1(-1.f));

        c = math::copysign(float1(-1.f), float1(1.f));
        CHECK(c == float1(1.f));

        c = math::copysign(float1(-1.f), float1(0.f));
        CHECK(c == float1(1.f));

        c = math::copysign(float1(-1.f), float1(-0.f));
        CHECK(c == float1(-1.f));
    }

    TEST(copysign_float_Works)
    {
        float c = math::copysign(1.f, -1.f);
        CHECK(c == -1.f);

        c = math::copysign(1.f, 1.f);
        CHECK(c == 1.f);

        c = math::copysign(-1.f, -1.f);
        CHECK(c == -1.f);

        c = math::copysign(-1.f, 1.f);
        CHECK(c == 1.f);

        c = math::copysign(-1.f, 0.f);
        CHECK(c == 1.f);

        c = math::copysign(-1.f, -0.f);
        CHECK(c == -1.f);
    }

    TEST(isfinite_float4_Works)
    {
        // With clang, quiet_NaN produces a negative NaN (http://comments.gmane.org/gmane.comp.compilers.llvm.bugs/24542).
        // Use fabs to work around.
        int4 c = isfinite(float4(0.f, 1.f, std::numeric_limits<float>::infinity(), std::fabs(std::numeric_limits<float>::quiet_NaN())));
        CHECK(all(c == int4(~0, ~0, 0, 0)));
    }

    TEST(isfinite_float3_Works)
    {
        // With clang, quiet_NaN produces a negative NaN (http://comments.gmane.org/gmane.comp.compilers.llvm.bugs/24542).
        // Use fabs to work around.
        int3 c = isfinite(float3(1.f, std::numeric_limits<float>::infinity(), std::fabs(std::numeric_limits<float>::quiet_NaN())));
        CHECK(all(c == int3(~0, 0, 0)));
    }

    TEST(isfinite_float2_Works)
    {
        int2 c = isfinite(float2(1.f, 0.f));
        CHECK(all(c == int2(~0, ~0)));

        // With clang, quiet_NaN produces a negative NaN (http://comments.gmane.org/gmane.comp.compilers.llvm.bugs/24542).
        // Use fabs to work around.
        c = isfinite(float2(std::numeric_limits<float>::infinity(), std::fabs(std::numeric_limits<float>::quiet_NaN())));
        CHECK(all(c == int2(0, 0)));
    }

    TEST(isfinite_float1_Works)
    {
        int1 c = isfinite(float1(1.f));
        CHECK(c == 1);

        c = isfinite(float1(std::numeric_limits<float>::infinity()));
        CHECK(c == 0);
    }

    TEST(isfinite_float_Works)
    {
        int c = math::isfinite(1.f);
        CHECK(c == 1);

        c = math::isfinite(std::numeric_limits<float>::infinity());
        CHECK(c == 0);
    }

    TEST(rcpe_float4_Works)
    {
        float4 c = rcpe(float4(1.f, 0.f, 10.f, 999999999.f));
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
        CHECK_CLOSE(0.1f, (float)c.z, approximationEpsilon);
        CHECK_CLOSE(0.f, (float)c.w, approximationEpsilon);
    }

    TEST(rcpe_float3_Works)
    {
        float3 c = rcpe(float3(1.f, 0.f, 999999999.f));
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
        CHECK_CLOSE(0.f, (float)c.z, approximationEpsilon);
    }

#if PLATFORM_PS4 || (defined(__APPLE__) && defined(__clang__))
    // yes that looks awful, but it seems that apple clang (xcode 7.3) sometimes optimize over-agressively
    // and in this particluar case it will generate RCPSS instead of RCPPS because of zero
    // it might be that we need optnone on float2 ctor, but it seems it really is isolated to this particular test idiom
    //
    // The same code generation issue occurs on PS4 with ORBIS SDK 3.5 - which upgrades Clang from 3.6.1 to 3.7.1
    // Testing shows that float4/float3/float2 types are all affected when constants with trailing zeros are used.
    // e.g.
    //      float4(XX, 0.f, 0.f, 0.f)
    //      float3(XX, 0.f, 0.f)
    //      float2(XX, 0.f)
    //
    // ... where  (XX is a non-zero value) - the RCPPS instruction is replaced with RCPSS
    //
    __attribute__((optnone))
#endif
    static void test_rcpe_float2_with_zero()
    {
        float2 c = rcpe(float2(1.f, 0.f));
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
    }

    TEST(rcpe_float2_Works)
    {
        test_rcpe_float2_with_zero();

        float2 c = rcpe(float2(10.f, 999999999.f));
        CHECK_CLOSE(0.1f, (float)c.x, approximationEpsilon);
        CHECK_CLOSE(0.f, (float)c.y, approximationEpsilon);
    }

    TEST(rcpe_float1_Works)
    {
        float1 c = rcpe(float1(1.f));
        CHECK_EQUAL(1.f, (float)c);

        c = rcpe(float1(ZERO));
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c);

        c = rcpe(float1(10.f));
        CHECK_CLOSE(0.1f, (float)c, approximationEpsilon);

        c = rcpe(float1(999999999.f));
        CHECK_CLOSE(0.f, (float)c, approximationEpsilon);
    }

    TEST(rcpe_float_Works)
    {
        float c = rcpe(1.f);
        CHECK_EQUAL(1.f, (float)c);

        c = rcpe(10.f);
        CHECK_CLOSE(0.1f, (float)c, approximationEpsilon);

        c = rcpe(999999999.f);
        CHECK_CLOSE(0.f, (float)c, approximationEpsilon);
    }

    TEST(rcp_float4_Works)
    {
        float4 c = rcp(float4(1.f, 0.f, 10.f, std::numeric_limits<float>::infinity()));
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
        CHECK_CLOSE(0.1f, (float)c.z, epsilon);
        CHECK_CLOSE(0.f, (float)c.w, epsilon);
    }

    TEST(rcp_float3_Works)
    {
        float3 c = rcp(float3(1.f, 0.f, std::numeric_limits<float>::infinity()));
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
        CHECK_CLOSE(0.f, (float)c.z, epsilon);
    }

    TEST(rcp_float2_Works)
    {
        float2 c = rcp(float2(1.f, 0.f));
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);

        c = rcp(float2(10.f, std::numeric_limits<float>::infinity()));
        CHECK_CLOSE(0.1f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
    }

    TEST(rcp_float1_Works)
    {
        float1 c = math::rcp(float1(1.f));
        CHECK_EQUAL(1.f, (float)c);

        c = math::rcp(float1(ZERO));
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c);

        c = math::rcp(float1(10.f));
        CHECK_CLOSE(0.1f, (float)c, epsilon);

        c = math::rcp(float1(std::numeric_limits<float>::infinity()));
        CHECK_CLOSE(0.f, (float)c, epsilon);
    }

    TEST(rcp_float_Works)
    {
        float c = math::rcp(1.f);
        CHECK_EQUAL(1.f, (float)c);

        c = math::rcp(0.f);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c);

        c = math::rcp(10.f);
        CHECK_CLOSE(0.1f, (float)c, epsilon);

        c = math::rcp(999999999.f);
        CHECK_CLOSE(0.f, (float)c, epsilon);
    }

    TEST(rsqrte_float4_Works)
    {
        float4 c = rsqrte(float4(1.f, 0.f, 16.f, 999999999999.f));

        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
        CHECK_CLOSE(0.25f, (float)c.z, epsilon);
        CHECK_CLOSE(0.f, (float)c.w, epsilon);
    }

    TEST(rsqrte_float3_Works)
    {
        float3 c = rsqrte(float3(1.f, 0.f, 16.f));

        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
        CHECK_CLOSE(0.25f, (float)c.z, epsilon);
    }

#if PLATFORM_PS4 || (defined(__APPLE__) && defined(__clang__))
    // yes that looks awful, but it seems that apple clang (xcode 7.3) sometimes optimize over-agressively
    // and in this particluar case it will generate RSQRTSS instead of RSQRTPS because of zero
    // it might be that we need optnone on float2 ctor, but it seems it really is isolated to this particular test idiom
    //
    // The same code generation issue occurs on PS4 with ORBIS SDK 3.5 - which upgrades Clang from 3.6.1 to 3.7.1
    // Testing shows that float4/float3/float2 types are all affected when constants with trailing zeros are used.
    // e.g.
    //      float4(XX, 0.f, 0.f, 0.f)
    //      float3(XX, 0.f, 0.f)
    //      float2(XX, 0.f)
    //
    // ... where  (XX is a non-zero value) - the RSQRTPS instruction is replaced with RSQRTSS
    //
    __attribute__((optnone))
#endif
    static void test_rsqrte_float2_with_zero()
    {
        float2 c = rsqrte(float2(1.f, 0.f));
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
    }

    TEST(rsqrte_float2_Works)
    {
        test_rsqrte_float2_with_zero();

        float2 c = rsqrte(float2(16.f , 999999999999.f));
        CHECK_CLOSE(0.25f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
    }

    TEST(rsqrte_float1_Works)
    {
        float1 c = rsqrte(float1(1.f));
        CHECK_EQUAL(1.f, (float)c);

        c = rsqrte(float1(0.f));
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c);

        c = rsqrte(float1(16.f));
        CHECK_CLOSE(0.25f, (float)c, epsilon);

        c = rsqrte(float1(999999999999.f));
        CHECK_CLOSE(0.f, (float)c, epsilon);
    }

    TEST(rsqrte_float_Works)
    {
        float c = rsqrte(1.f);
        CHECK_CLOSE(1.f, c, approximationEpsilon);

        c = rsqrte(16.f);
        CHECK_CLOSE(0.25f, c, approximationEpsilon);

        c = rsqrte(999999999999.f);
        CHECK_CLOSE(0.f, c, approximationEpsilon);
    }

    TEST(rsqrt_float4_Works)
    {
        float4 c = rsqrt(float4(1.f, 0.f, 16.f, 999999999999.f));

        // rsqrt(1) must return 1 otherwise normalization of a already normalized vector will denormalize it
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
        CHECK_CLOSE(0.25f, (float)c.z, epsilon);
        CHECK_CLOSE(0.f, (float)c.w, epsilon);
    }

    TEST(rsqrt_float3_Works)
    {
        float3 c = rsqrt(float3(1.f, 0.f, 16.f));

        // rsqrt(1) must return 1 otherwise normalization of a already normalized vector will denormalize it
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);
        CHECK_CLOSE(0.25f, (float)c.z, epsilon);
    }

    TEST(rsqrt_float2_Works)
    {
        float2 c = rsqrt(float2(1.f, 0.f));

        // rsqrt(1) must return 1 otherwise normalization of a already normalized vector will denormalize it
        CHECK_EQUAL(1.f, (float)c.x);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c.y);

        c = rsqrt(float2(16.f , 999999999999.f));
        CHECK_CLOSE(0.25f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
    }

    TEST(rsqrt_float1_Works)
    {
        // rsqrt(1) must return 1 otherwise normalization of a already normalized vector will denormalize it
        float1 c = math::rsqrt(float1(1.f));
        CHECK_EQUAL(1.f, (float)c);

        c = math::rsqrt(float1(0.f));
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), (float)c);

        c = math::rsqrt(float1(16.f));
        CHECK_CLOSE(0.25f, (float)c, epsilon);

        c = math::rsqrt(float1(999999999999.f));
        CHECK_CLOSE(0.f, (float)c, epsilon);
    }

    TEST(rsqrt_float_Works)
    {
        // rsqrt(1) must return 1 otherwise normalization of a already normalized vector will denormalize it
        float c = math::rsqrt(1.f);
        CHECK_EQUAL(1.f, (float)c);

        c = math::rsqrt(0.f);
        CHECK_EQUAL(std::numeric_limits<float>::infinity(), c);

        c = math::rsqrt(16.f);
        CHECK_CLOSE(0.25f, c, epsilon);

        c = math::rsqrt(999999999999.f);
        CHECK_CLOSE(0.f, c, epsilon);
    }

    TEST(sqrt_float4_Works)
    {
        float4 c = sqrt(float4(1.f, 0.f, 16.f, 456.234f));

        CHECK_CLOSE(1.f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
        CHECK_CLOSE(4.f, (float)c.z, epsilon);
        CHECK_CLOSE(21.35963482f, (float)c.w, epsilon);
    }

    TEST(sqrt_float3_Works)
    {
        float3 c = sqrt(float3(1.f, 0.f, 16.f));

        CHECK_CLOSE(1.f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
        CHECK_CLOSE(4.f, (float)c.z, epsilon);
    }

#if defined(__APPLE__) && defined(__clang__)
    // yes that looks awful, but it seems that apple clang (xcode 7.3) sometimes optimize over-agressively
    // and in this particluar case it will generate RSQRTSS instead of RSQRTPS because of zero
    // it might be that we need optnone on float2 ctor, but it seems it really is isolated to this particular test idiom
    __attribute__((optnone))
#endif
    static void test_sqrt_float2_with_zero()
    {
        float2 c = sqrt(float2(1.f, 0.f));

        CHECK_CLOSE(1.f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
    }

    TEST(sqrt_float2_Works)
    {
        test_sqrt_float2_with_zero();

        float2 c = sqrt(float2(16.f , 456.234f));
        CHECK_CLOSE(4.f, (float)c.x, epsilon);
        CHECK_CLOSE(21.35963482f, (float)c.y, epsilon);
    }

    TEST(sqrt_float1_Works)
    {
        float1 c = math::sqrt(float1(1.f));
        CHECK_CLOSE(1.f, (float)c, epsilon);

        c = math::sqrt(float1(0.f));
        CHECK_CLOSE(0.f, (float)c, epsilon);

        c = math::sqrt(float1(16.f));
        CHECK_CLOSE(4.f, (float)c, epsilon);

        c = math::sqrt(float1(456.234f));
        CHECK_CLOSE(21.35963482f, (float)c, epsilon);
    }

    TEST(sqrt_float_Works)
    {
        float c = math::sqrt(1.f);
        CHECK_CLOSE(1.f, c, epsilon);

        c = math::sqrt(0.f);
        CHECK_CLOSE(0.f, c, epsilon);

        c = math::sqrt(16.f);
        CHECK_CLOSE(4.f, c, epsilon);

        c = math::sqrt(456.234f);
        CHECK_CLOSE(21.35963482f, c, epsilon);
    }

    TEST(saturate_float4_Works)
    {
        float4 c  = saturate(float4(-1.345f, 0.f, 0.345f, 1.345f));
        CHECK_CLOSE(0.f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
        CHECK_CLOSE(0.345f, (float)c.z, epsilon);
        CHECK_CLOSE(1.f, (float)c.w, epsilon);
    }

    TEST(saturate_float3_Works)
    {
        float3 c  = saturate(float3(-1.345f, 0.f, 0.345f));
        CHECK_CLOSE(0.f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
        CHECK_CLOSE(0.345f, (float)c.z, epsilon);

        c  = saturate(float3(1.345f));
        CHECK(all(c == float3(1.f)));
    }

    TEST(saturate_float2_Works)
    {
        float2 c  = saturate(float2(-1.345f, 0.f));
        CHECK_CLOSE(0.f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);

        c  = saturate(float2(0.345f, 1.345f));
        CHECK_CLOSE(0.345f, (float)c.x, epsilon);
        CHECK_CLOSE(1.f, (float)c.y, epsilon);
    }

    TEST(saturate_float1_Works)
    {
        float1 c = saturate(float1(-1.345f));
        CHECK_CLOSE(0.f, (float)c, epsilon);

        c = saturate(float1(0.0f));
        CHECK_CLOSE(0.f, (float)c, epsilon);

        c = saturate(float1(0.345f));
        CHECK_CLOSE(0.345f, (float)c, epsilon);

        c = saturate(float1(1.345f));
        CHECK_CLOSE(1.f, (float)c, epsilon);
    }

    TEST(saturate_float_Works)
    {
        float c = saturate(-1.345f);
        CHECK_CLOSE(0.f, c, epsilon);

        c = saturate(0.0f);
        CHECK_CLOSE(0.f, c, epsilon);

        c = saturate(0.345f);
        CHECK_CLOSE(0.345f, c, epsilon);

        c = saturate(1.345f);
        CHECK_CLOSE(1.f, c, epsilon);
    }

    TEST(clamp_float4_Works)
    {
        float4 c = clamp(float4(1.f, 0.f, 350.f, -100.f), float4(0.f, 1.f, 100.f, -10.f), float4(2.f, 3.f, 200.f, -2.f));
        CHECK(all(c == float4(1.f, 1.f, 200.f, -10.f)));
    }

    TEST(clamp_float3_Works)
    {
        float3 c = clamp(float3(1.f, 0.f, 350.f), float3(0.f, 1.f, 100.f), float3(2.f, 3.f, 200.f));
        CHECK(all(c == float3(1.f, 1.f, 200.f)));
    }

    TEST(clamp_float2_Works)
    {
        float2 c = clamp(float2(1.f, 0.f), float2(0.f, 1.f), float2(2.f, 3.f));
        CHECK(all(c == float2(1.f, 1.f)));
    }

    TEST(clamp_float1_Works)
    {
        float1 c = clamp(float1(1.f), float1(0.f), float1(2.f));
        CHECK(c == float1(1.f));

        c = clamp(float1(0.f), float1(1.f), float1(3.f));
        CHECK(c == float1(1.f));

        c = clamp(float1(350.f), float1(100.f), float1(200.f));
        CHECK(c == float1(200.f));

        c = clamp(float1(-100.f), float1(-10.f), float1(-2.f));
        CHECK(c == float1(-10.f));
    }

    TEST(clamp_float_Works)
    {
        float c = clamp(1.f, 0.f, 2.f);
        CHECK(c == 1.f);

        c = clamp(0.f, 1.f, 3.f);
        CHECK(c == 1.f);

        c = clamp(350.f, 100.f, 200.f);
        CHECK(c == 200.f);

        c = clamp(-100.f, -10.f, -2.f);
        CHECK(c == -10.f);
    }

    TEST(clamp_int4_Works)
    {
        int4 c = clamp(int4(1, 0, 350, -100), int4(0, 1, 100, -10), int4(2, 3, 200, -2));
        CHECK(all(c == int4(1, 1, 200, -10)));
    }

    TEST(clamp_int3_Works)
    {
        int3 c = clamp(int3(1, 0, 350), int3(0, 1, 100), int3(2, 3, 200));
        CHECK(all(c == int3(1, 1, 200)));
    }

    TEST(clamp_int2_Works)
    {
        int2 c = clamp(int2(1, 0), int2(0, 1), int2(2, 3));
        CHECK(all(c == int2(1, 1)));
    }

    TEST(clamp_int1_Works)
    {
        int1 c = clamp(int1(1), int1(0), int1(2));
        CHECK(c == int1(1));

        c = clamp(int1(0), int1(1), int1(3));
        CHECK(c == int1(1));

        c = clamp(int1(350), int1(100), int1(200));
        CHECK(c == int1(200));

        c = clamp(int1(-100), int1(-10), int1(-2));
        CHECK(c == int1(-10));
    }

    TEST(clamp_int_Works)
    {
        int c = clamp(1, 0, 2);
        CHECK(c == 1);

        c = clamp(0, 1, 3);
        CHECK(c == 1);

        c = clamp(350, 100, 200);
        CHECK(c == 200);

        c = clamp(-100, -10, -2);
        CHECK(c == -10);
    }

    TEST(csum_float4_Works)
    {
        float1 c = csum(float4(9.f, 81.f, 49.f, 74.f));
        CHECK_CLOSE(213.f, (float)c, epsilon);

        c = csum(float4(0.f, 0.f, 0.f, 0.f));
        CHECK_CLOSE(0.f, (float)c, epsilon);
    }

    TEST(csum_float3_Works)
    {
        float1 c = csum(float3(1.f, 2.f, 3.f));
        CHECK_CLOSE(6.f, (float)c, epsilon);
    }

    TEST(csum_float2_Works)
    {
        float1 c = csum(float2(10.f, 20.f));
        CHECK_CLOSE(30.f, (float)c, epsilon);
    }

    TEST(cmax_float4_Works)
    {
        float1 c = cmax(float4(-1.f, -.263f, 345.f, 0.f));
        CHECK_CLOSE(345.f, (float)c, epsilon);
    }

    TEST(cmax_float3_Works)
    {
        float1 c = cmax(float3(-1.f, -.263f, -10.f));
        CHECK_CLOSE(-0.263f, (float)c, epsilon);
    }

    TEST(cmax_float2_Works)
    {
        float1 c = cmax(float2(-1.f, -10.f));
        CHECK_CLOSE(-1.f, (float)c, epsilon);
    }

    TEST(cmax_int4_Works)
    {
        int1 c = cmax(int4(-1, 0, 345, 10.f));
        CHECK(345 == (int)c);
    }

    TEST(cmax_int3_Works)
    {
        int1 c = cmax(int3(-1, 0, 345));
        CHECK(345 == (int)c);
    }

    TEST(cmax_int2_Works)
    {
        int1 c = cmax(int2(-1, 0));
        CHECK(0 == (int)c);
    }


    TEST(cmin_float4_Works)
    {
        float1 c = cmin(float4(-1.f, -.263f, 345.f, 0.f));
        CHECK_CLOSE(-1.f, (float)c, epsilon);
    }

    TEST(cmin_float3_Works)
    {
        float1 c = cmin(float3(-1.f, -.263f, -10.f));
        CHECK_CLOSE(-10.f, (float)c, epsilon);
    }

    TEST(cmin_float2_Works)
    {
        float1 c = cmin(float2(-1.f, -10.f));
        CHECK_CLOSE(-10.f, (float)c, epsilon);
    }

    TEST(cmin_int4_Works)
    {
        int1 c = cmin(int4(-1, 0, 345, 10.f));
        CHECK(-1 == (int)c);
    }

    TEST(cmin_int3_Works)
    {
        int1 c = cmin(int3(-1, 0, 345));
        CHECK(-1 == (int)c);
    }

    TEST(cmin_int2_Works)
    {
        int1 c = cmin(int2(-1, 0));
        CHECK(-1 == (int)c);
    }

    TEST(dot_float4_Works)
    {
        float1 teta = dot(float4(1, 0, 0, 0), float4(0, 1, 0, 0));
        CHECK_CLOSE(0.f, (float)teta, epsilon);

        teta = dot(float4(1, 0, 0, 0), float4(1, 0, 0, 0));
        CHECK_CLOSE(1.f, (float)teta, epsilon);

        teta = dot(float4(1, 0, 0, 0), normalize(float4(1, 1, 0, 0)));
        CHECK_CLOSE(0.70710f, (float)teta, epsilon);

        float4 v = float4(10.f, 5.f, 2.f, 0.f);
        float1 s = dot(v, v);
        CHECK_CLOSE(129.0f, (float)s, epsilon);
    }

    TEST(dot_float3_Works)
    {
        float1 teta = dot(float3(1, 0, 0), float3(0, 1, 0));
        CHECK_CLOSE(0.f, (float)teta, epsilon);

        teta = dot(float3(1, 0, 0), float3(1, 0, 0));
        CHECK_CLOSE(1.f, (float)teta, epsilon);

        teta = dot(float3(1, 0, 0), normalize(float3(1, 1, 0)));
        CHECK_CLOSE(0.70710f, (float)teta, epsilon);

        float3 v = float3(10.f, 5.f, 2.f);
        float1 s = dot(v, v);
        CHECK_CLOSE(129.0f, (float)s, epsilon);
    }

    TEST(dot_float2_Works)
    {
        float1 teta = dot(float2(1, 0), float2(0, 1));
        CHECK_CLOSE(0.f, (float)teta, epsilon);

        teta = dot(float2(1, 0), float2(1, 0));
        CHECK_CLOSE(1.f, (float)teta, epsilon);

        teta = dot(float2(1, 0), normalize(float2(1, 1)));
        CHECK_CLOSE(0.70710f, (float)teta, epsilon);
    }

    TEST(length_float4_Works)
    {
        float1 s = length(float4(1.f, 0.f, 0.f, 0.f));
        CHECK_CLOSE(1.f, (float)s, epsilon);

        s = length(float4(10.f, 5.f, 2.f, 0.f));
        CHECK_CLOSE(11.357816f, (float)s, epsilon);

        s = length(float4(0.f, 0.f, 0.f, 0.f));
        CHECK_CLOSE(0.f, (float)s, epsilon);
    }

    TEST(length_float3_Works)
    {
        float1 s = length(float3(1.f, 0.f, 0.f));
        CHECK_CLOSE(1.f, (float)s, epsilon);

        s = length(float3(10.f, 5.f, 2.f));
        CHECK_CLOSE(11.357816f, (float)s, epsilon);

        s = length(float3(0.f, 0.f, 0.f));
        CHECK_CLOSE(0.f, (float)s, epsilon);
    }

    TEST(length_float2_Works)
    {
        float1 s = length(float2(1.f, 0.f));
        CHECK_CLOSE(1.f, (float)s, epsilon);

        s = length(float2(10.f, 5.f));
        CHECK_CLOSE(11.1803398, (float)s, epsilon);

        s = length(float2(0.f, 0.f));
        CHECK_CLOSE(0.f, (float)s, epsilon);
    }

    TEST(normalize_float4_Works)
    {
        float4 c = normalize(float4(0.f, 0.f, 0.f, 1.f));
        CHECK_CLOSE(0.f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
        CHECK_CLOSE(0.f, (float)c.z, epsilon);
        CHECK_CLOSE(1.f, (float)c.w, epsilon);

        c = normalize(float4(10.f, 5.f, 2.f, 0.f));
        CHECK_CLOSE(0.880451f, (float)c.x, epsilon);
        CHECK_CLOSE(0.440225f, (float)c.y, epsilon);
        CHECK_CLOSE(0.176090f, (float)c.z, epsilon);
        CHECK_CLOSE(0.f, (float)c.w, epsilon);
    }

    TEST(normalize_float3_Works)
    {
        float3 c = normalize(float3(0.f, 0.f, 1.f));
        CHECK_CLOSE(0.f, (float)c.x, epsilon);
        CHECK_CLOSE(0.f, (float)c.y, epsilon);
        CHECK_CLOSE(1.f, (float)c.z, epsilon);

        c = normalize(float3(10.f, 5.f, 2.f));
        CHECK_CLOSE(0.880451f, (float)c.x, epsilon);
        CHECK_CLOSE(0.440225f, (float)c.y, epsilon);
        CHECK_CLOSE(0.176090f, (float)c.z, epsilon);
    }

    TEST(normalize_float2_Works)
    {
        float2 c = normalize(float2(0.f, 1.f));
        CHECK_CLOSE(0.f, (float)c.x, epsilon);
        CHECK_CLOSE(1.f, (float)c.y, epsilon);

        c = normalize(float2(10.f, 5.f));
        CHECK_CLOSE(0.894427f, (float)c.x, epsilon);
        CHECK_CLOSE(0.447214f, (float)c.y, epsilon);
    }

    TEST(normalizeToByte_float4_Works)
    {
        int4 c;
        c = normalizeToByte(float4(-FLT_MAX, -1.0f, -0.001f, 0.499f));
        CHECK_EQUAL(0, (int)c.x);
        CHECK_EQUAL(0, (int)c.y);
        CHECK_EQUAL(0, (int)c.z);
        CHECK_EQUAL(127, (int)c.w);

        c = normalizeToByte(float4(0.501f, 1.0f, 1.001f, 2.0f));
        CHECK_EQUAL(128, (int)c.x);
        CHECK_EQUAL(255, (int)c.y);
        CHECK_EQUAL(255, (int)c.z);
        CHECK_EQUAL(255, (int)c.w);

        c = normalizeToByte(float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX));
        CHECK_EQUAL(255, (int)c.x);
    }

    TEST(cross_float4_Works)
    {
        float4 c = cross(float4(-1.f, 0.f, 0.f, 0.f), float4(0.f, 1.f, 0.f, 0.f));
        CHECK(all(c == float4(0.f, 0.f, -1.f, 0.f)));

        c = cross(float4(-1.f, 2.f, -4.f, 1.f), float4(4.f, 1.f, -3.f, 1.f));
        CHECK(all(c == float4(-2.f, -19.f, -9.f, 0.f)));
    }

    TEST(cross_float3_Works)
    {
        float3 c = cross(float3(-1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f));
        CHECK(all(c == float3(0.f, 0.f, -1.f)));

        c = cross(float3(-1.f, 2.f, -4.f), float3(4.f, 1.f, -3.f));
        CHECK(all(c == float3(-2.f, -19.f, -9.f)));
    }

    TEST(degrees_float4_Works)
    {
        float4 c = degrees(float4(math::pi(), math::pi_over_two(), math::pi_over_four(), 0.f));
        CHECK(all(c == float4(180.f, 90.f, 45.f, 0.f)));
    }

    TEST(degrees_float3_Works)
    {
        float3 c = degrees(float3(math::pi(), math::pi_over_two(), math::pi_over_four()));
        CHECK(all(c == float3(180.f, 90.f, 45.f)));
    }

    TEST(degrees_float2_Works)
    {
        float2 c = degrees(float2(math::pi(), math::pi_over_two()));
        CHECK(all(c == float2(180.f, 90.f)));
    }

    TEST(degrees_float1_Works)
    {
        float1 c = degrees(float1(math::pi()));
        CHECK(c == float1(180.f));

        c = degrees(float1(math::pi_over_two()));
        CHECK(c == float1(90.f));

        c = degrees(float1(math::pi_over_four()));
        CHECK(c == float1(45.f));
    }

    TEST(degrees_float_Works)
    {
        float c = degrees(math::pi());
        CHECK_CLOSE(180.f, c, epsilon);

        c = degrees(math::pi_over_two());
        CHECK_CLOSE(90.f, c, epsilon);

        c = degrees(math::pi_over_four());
        CHECK_CLOSE(45.f, c, epsilon);
    }

    TEST(radians_float4_Works)
    {
        float4 c = radians(float4(180.f, 90.f, 45.f, 0.f));
        CHECK(all(c == float4(math::pi(), math::pi_over_two(), math::pi_over_four(), 0.f)));
    }

    TEST(radians_float3_Works)
    {
        float3 c = radians(float3(180.f, 90.f, 45.f));
        CHECK(all(c == float3(math::pi(), math::pi_over_two(), math::pi_over_four())));
    }

    TEST(radians_float2_Works)
    {
        float2 c = radians(float2(180.f, 90.f));
        CHECK(all(c == float2(math::pi(), math::pi_over_two())));
    }

    TEST(radians_float1_Works)
    {
        float1 c = radians(float1(180.f));
        CHECK(c == float1(math::pi()));

        c = radians(float1(90.f));
        CHECK(c == float1(math::pi_over_two()));

        c = radians(float1(45.f));
        CHECK(c == float1(math::pi_over_four()));
    }

    TEST(radians_float_Works)
    {
        float c = radians(180.f);
        CHECK_CLOSE(math::pi(), c, epsilon);

        c = radians(90.f);
        CHECK_CLOSE(math::pi_over_two(), c, epsilon);

        c = radians(45.f);
        CHECK_CLOSE(math::pi_over_four(), c, epsilon);
    }

    TEST(powr_float4_Works)
    {
        float4 s = math::powr(math::float4(0.0f, 1.0f, 0.5f, 2.0f), math::float4(4.f, 0.f, 4.f, 4.f));
        CHECK_CLOSE(0.f, (float)s.x, epsilon);
        CHECK_CLOSE(1.f, (float)s.y, epsilon);
        CHECK_CLOSE(0.0625f, (float)s.z, epsilon);
        CHECK_CLOSE(16.f, (float)s.w, epsilon);
    }

    TEST(powr_float3_Works)
    {
        float3 s = math::powr(math::float3(0.0f, 1.0f, 0.5f), math::float3(4.f, 0.f, 4.f));
        CHECK_CLOSE(0.f, (float)s.x, epsilon);
        CHECK_CLOSE(1.f, (float)s.y, epsilon);
        CHECK_CLOSE(0.0625f, (float)s.z, epsilon);
    }

    TEST(powr_float2_Works)
    {
        float2 s = math::powr(math::float2(0.0f, 1.0f), math::float2(4.f, 0.f));
        CHECK_CLOSE(0.f, (float)s.x, epsilon);
        CHECK_CLOSE(1.f, (float)s.y, epsilon);
    }

    TEST(powr_float1_Works)
    {
        float1 s = math::powr(math::float1(math::ZERO), math::float1(4.f));
        CHECK_CLOSE(0.f, (float)s, epsilon);

        s = math::powr(math::float1(1.f), math::float1(math::ZERO));
        CHECK_CLOSE(1.f, (float)s, epsilon);

        s = math::powr(math::float1(0.5f), math::float1(4.f));
        CHECK_CLOSE(0.0625f, (float)s, epsilon);

        s = math::powr(math::float1(2.0f), math::float1(4.f));
        CHECK_CLOSE(16.f, (float)s, epsilon);

        s = math::powr(math::float1(1.f), math::float1(4.f));
        CHECK_CLOSE(1.f, (float)s, epsilon);
    }

    TEST(powr_float_Works)
    {
        float s = math::powr(float(0.f), float(4.f));
        CHECK_CLOSE(0.f, s, epsilon);

        s = math::powr(float(1.f),  float(0.f));
        CHECK_CLOSE(1.f, s, epsilon);

        s = math::powr(float(0.5f), float(4.f));
        CHECK_CLOSE(0.0625f, s, epsilon);

        s = math::powr(float(2.0f), float(4.f));
        CHECK_CLOSE(16.f, s, epsilon);

        s = math::powr(float(1.f), float(4.f));
        CHECK_CLOSE(1.f, s, epsilon);
    }

    TEST(fmod_float4_Works)
    {
        float4 c = fmod(float4(9.45f), float4(1.f, 2.f, 5.f, 10.f));
        CHECK_CLOSE(0.45f, (float)c.x, epsilon);
        CHECK_CLOSE(1.45f, (float)c.y, epsilon);
        CHECK_CLOSE(4.45f, (float)c.z, epsilon);
        CHECK_CLOSE(9.45f, (float)c.w, epsilon);
    }

    TEST(fmod_float3_Works)
    {
        float3 c = fmod(float3(9.45f), float3(1.f, 2.f, 5.f));
        CHECK_CLOSE(0.45f, (float)c.x, epsilon);
        CHECK_CLOSE(1.45f, (float)c.y, epsilon);
        CHECK_CLOSE(4.45f, (float)c.z, epsilon);
    }

    TEST(fmod_float2_Works)
    {
        float2 c = fmod(float2(9.45f), float2(1.f, 2.f));
        CHECK_CLOSE(0.45f, (float)c.x, epsilon);
        CHECK_CLOSE(1.45f, (float)c.y, epsilon);
    }

    TEST(fmod_float1_Works)
    {
        float1 c = math::fmod(float1(9.45f), float1(1.f));
        CHECK_CLOSE(0.45f, (float)c, epsilon);

        c = math::fmod(float1(9.45f), float1(2.f));
        CHECK_CLOSE(1.45f, (float)c, epsilon);

        c = math::fmod(float1(9.45f), float1(5.f));
        CHECK_CLOSE(4.45f, (float)c, epsilon);

        c = math::fmod(float1(9.45f), float1(10.f));
        CHECK_CLOSE(9.45f, (float)c, epsilon);
    }

    TEST(lerp_float4_Works)
    {
        float4 c = lerp(float4(1, 2, 3, 4), float4(3, 4, 5, 6), float1(.5f));
        CHECK_CLOSE(2.f, (float)c.x, epsilon);
        CHECK_CLOSE(3.f, (float)c.y, epsilon);
        CHECK_CLOSE(4.f, (float)c.z, epsilon);
        CHECK_CLOSE(5.f, (float)c.w, epsilon);

        c = lerp(float4(1, 2, 3, 4), float4(3, 4, 5, 6), float4(-.5f, 0, 1.0, 1.5f));
        CHECK_CLOSE(0.f, (float)c.x, epsilon);
        CHECK_CLOSE(2.f, (float)c.y, epsilon);
        CHECK_CLOSE(5.f, (float)c.z, epsilon);
        CHECK_CLOSE(7.f, (float)c.w, epsilon);
    }

    TEST(lerp_float3_Works)
    {
        float3 c = lerp(float3(1, 2, 3), float3(3, 4, 5), float1(.5f));
        CHECK_CLOSE(2.f, (float)c.x, epsilon);
        CHECK_CLOSE(3.f, (float)c.y, epsilon);
        CHECK_CLOSE(4.f, (float)c.z, epsilon);

        c = lerp(float3(1, 2, 3), float3(3, 4, 5), float3(-.5f, 0, 1.0));
        CHECK_CLOSE(0.f, (float)c.x, epsilon);
        CHECK_CLOSE(2.f, (float)c.y, epsilon);
        CHECK_CLOSE(5.f, (float)c.z, epsilon);
    }

    TEST(lerp_float1_Works)
    {
        float1 c = lerp(float1(1), float1(3), float1(.5f));
        CHECK_CLOSE(2.f, (float)c, epsilon);
    }

    TEST(lerp_float_Works)
    {
        float c = lerp(1.f, 3.f, .5f);
        CHECK_CLOSE(2.f, c, epsilon);
    }

    TEST(vector_float4_Works)
    {
        float4 c = math::vector(float4(-25.f, 0.f, .5f, 1.5f));
        CHECK(all(c == float4(-25.f, 0.f, .5f, 0.f)));
    }

    TEST(extract_float4_Works)
    {
        float4 c = float4(-25.f, 0.f, .5f, 1.5f);
        CHECK_EQUAL(-25.f, extract(c, 0));
        CHECK_EQUAL(0.f, extract(c, 1));
        CHECK_EQUAL(.5f, extract(c, 2));
        CHECK_EQUAL(1.5f, extract(c, 3));
    }

    TEST(extract_float3_Works)
    {
        float3 c = float3(-25.f, 0.f, .5f);
        CHECK_EQUAL(-25.f, extract(c, 0U));
        CHECK_EQUAL(0.f, extract(c, 1U));
        CHECK_EQUAL(.5f, extract(c, 2U));
    }

    TEST(extract_float2_Works)
    {
        float2 c = float2(-25.f, 0.f);
        CHECK_EQUAL(-25.f, extract(c, 0U));
        CHECK_EQUAL(0.f, extract(c, 1U));
    }

    TEST(insert_float4_Works)
    {
        float4 c;
        insert(c, 0, -25.f);
        insert(c, 1, 0.f);
        insert(c, 2, .5f);
        insert(c, 3, 1.5f);

        CHECK(all(c == float4(-25.f, 0.f, .5f, 1.5f)));
    }

    TEST(insert_float3_Works)
    {
        float3 c;
        insert(c, 0, -25.f);
        insert(c, 1, 0.f);
        insert(c, 2, .5f);

        CHECK(all(c == float3(-25.f, 0.f, .5f)));
    }

    TEST(insert_float2_Works)
    {
        float2 c;
        insert(c, 0, -25.f);
        insert(c, 1, 0.f);

        CHECK(all(c == float2(-25.f, 0.f)));
    }
}
#if PLATFORM_WIN
#pragma warning(pop)
#endif

#endif
