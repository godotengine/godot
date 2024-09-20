#include "UnityPrefix.h"
#if ENABLE_UNIT_TESTS
#include "Runtime/Testing/Testing.h"
#include "Runtime/Profiler/TimeHelper.h"

#if PLATFORM_WIN
#pragma warning(push)
#pragma warning(disable: 4723) // potential division by zero
#endif

#include "vec-types.h"
#include "vec-quat.h"
#include "Modules/Animation/mecanim/math/axes.h"

UNIT_TEST_SUITE(SIMDMath_floatOps)
{
    using namespace math;

    template<typename TYPE> TYPE* test_cast(TYPE& data)
    {
        return &data;
    }

    const float epsilon = 1e-5f;

    TEST(float_swizzle)
    {
        float4 value = float4(1, 2, 3, 4);

        float4 a = float4(0, 0, 0, 0);

        // 1 component swizzles
        a.x = 3.f;
        CHECK_CLOSE(3, (float)a.x, epsilon);
        CHECK_CLOSE(0, (float)a.y, epsilon);
        CHECK_CLOSE(0, (float)a.z, epsilon);
        CHECK_CLOSE(0, (float)a.w, epsilon);

        a.x = value.x;
        CHECK_CLOSE(1, (float)a.x, epsilon);
        CHECK_CLOSE(0, (float)a.y, epsilon);
        CHECK_CLOSE(0, (float)a.z, epsilon);
        CHECK_CLOSE(0, (float)a.w, epsilon);

        a.y = value.y;
        CHECK_CLOSE(1, (float)a.x, epsilon);
        CHECK_CLOSE(2, (float)a.y, epsilon);
        CHECK_CLOSE(0, (float)a.z, epsilon);
        CHECK_CLOSE(0, (float)a.w, epsilon);

        a.z = value.z;
        CHECK_CLOSE(1, (float)a.x, epsilon);
        CHECK_CLOSE(2, (float)a.y, epsilon);
        CHECK_CLOSE(3, (float)a.z, epsilon);
        CHECK_CLOSE(0, (float)a.w, epsilon);

        a.w = value.w;
        CHECK_CLOSE(1, (float)a.x, epsilon);
        CHECK_CLOSE(2, (float)a.y, epsilon);
        CHECK_CLOSE(3, (float)a.z, epsilon);
        CHECK_CLOSE(4, (float)a.w, epsilon);

        a.z = value.y;
        CHECK_CLOSE(1, (float)a.x, epsilon);
        CHECK_CLOSE(2, (float)a.y, epsilon);
        CHECK_CLOSE(2, (float)a.z, epsilon);
        CHECK_CLOSE(4, (float)a.w, epsilon);

        float1 f = value.w;
        CHECK_CLOSE(4, f, epsilon);

        // 4 component swizzles
        CHECK(all(value.xyzw == float4(1, 2, 3, 4)));
        CHECK(all(value.yzwx == float4(2, 3, 4, 1)));
        CHECK(all(value.zwxy == float4(3, 4, 1, 2)));
        CHECK(all(value.wxyz == float4(4, 1, 2, 3)));
        CHECK(all(value.wzyx == float4(4, 3, 2, 1)));
        CHECK(all(value.yzxw == float4(2, 3, 1, 4)));
        CHECK(all(value.ywzx == float4(2, 4, 3, 1)));
        CHECK(all(value.zxyw == float4(3, 1, 2, 4)));
        CHECK(all(value.xwyz == float4(1, 4, 2, 3)));
        CHECK(all(value.wzxy == float4(4, 3, 1, 2)));
        CHECK(all(value.xzyw == float4(1, 3, 2, 4)));
        CHECK(all(value.yxzw == float4(2, 1, 3, 4)));
        CHECK(all(value.zyxw == float4(3, 2, 1, 4)));
        CHECK(all(value.xyxy == float4(1, 2, 1, 2)));
        CHECK(all(value.yyww == float4(2, 2, 4, 4)));
        CHECK(all(value.zxzx == float4(3, 1, 3, 1)));
        CHECK(all(value.zzww == float4(3, 3, 4, 4)));
        CHECK(all(value.yzxy == float4(2, 3, 1, 2)));
        CHECK(all(value.wwwx == float4(4, 4, 4, 1)));
        CHECK(all(value.yxxw == float4(2, 1, 1, 4)));
        CHECK(all(value.zzyw == float4(3, 3, 2, 4)));
        CHECK(all(value.xxyy == float4(1, 1, 2, 2)));
        CHECK(all(value.xyxy == float4(1, 2, 1, 2)));

        // 2 component swizzles
        CHECK(all(value.xy == float2(1, 2)));
        CHECK(all(value.yx == float2(2, 1)));
        CHECK(all(value.yz == float2(2, 3)));
        CHECK(all(value.zx == float2(3, 1)));
        CHECK(all(value.zy == float2(3, 2)));
        CHECK(all(value.zw == float2(3, 4)));
        CHECK(all(value.xz == float2(1, 3)));
        CHECK(all(value.yw == float2(2, 4)));

        // 3 component swizzles
        CHECK(all(value.xyz == float3(1, 2, 3)));
        CHECK(all(value.xzy == float3(1, 3, 2)));
        CHECK(all(value.yzw == float3(2, 3, 4)));
        CHECK(all(value.yzx == float3(2, 3, 1)));
        CHECK(all(value.zxy == float3(3, 1, 2)));
        CHECK(all(value.zyx == float3(3, 2, 1)));
        CHECK(all(value.yxw == float3(2, 1, 4)));
        CHECK(all(value.zwx == float3(3, 4, 1)));
        CHECK(all(value.wzy == float3(4, 3, 2)));
    }

    TEST(float1_operator)
    {
        float1 b(3.f);
        float1 c(2.f);
        float4 cx;
        float1 e;

        {
            CHECK_CLOSE(3.f, (float)b, epsilon);

            CHECK_CLOSE(2.f, (float)c, epsilon);

            cx = float4(10.f, 2.f, 3.f, 4.f);

            float1 d(cx.x);

            CHECK_CLOSE(10.f, (float)d, epsilon);

            e = cx.y;

            CHECK_CLOSE(2.f, (float)e, epsilon);

            e = float1(4.f);
            CHECK_CLOSE(4.f, (float)e, epsilon);

            e = float1(cx.x);
            CHECK_CLOSE(10.f, (float)e, epsilon);

            e = cx.z;
            CHECK_CLOSE(3.f, (float)e, epsilon);

            e += cx.w;
            CHECK_CLOSE(7.f, (float)e, epsilon);

            e -= cx.x;
            CHECK_CLOSE(-3.f, (float)e, epsilon);

            e *= cx.y;
            CHECK_CLOSE(-6.f, (float)e, epsilon);

            e = float1(-6.f);
            e /= cx.z;
            CHECK_CLOSE(-2.f, (float)e, epsilon);
        }

        e = float1(-2.f);

        {
            float1 f = e++;
            CHECK_CLOSE(-2.f, (float)f, epsilon);

            CHECK_CLOSE(-1.f, (float)e, epsilon);

            float1 g = ++e;
            CHECK_CLOSE(0.f, (float)g, epsilon);

            CHECK_CLOSE(0.f, (float)e, epsilon);

            float1 j(4.f);
            float1 l(3.f);

            float1 m = j + l;
            CHECK_CLOSE(7.f, (float)m, epsilon);

            float1 n = j - l;
            CHECK_CLOSE(1.f, (float)n, epsilon);

            float1 o = j * l;
            CHECK_CLOSE(12.f, (float)o, epsilon);

            float1 p = j / l;
            CHECK_CLOSE(4.f / 3.f, (float)p, epsilon);
        }

        int1 ivalue = float1(4.f) < float1(3.f);
        CHECK(ivalue == 0);

        ivalue = float1(4.f) < float1(4.f);
        CHECK(ivalue == 0);

        ivalue = float1(4.f) < float1(5.f);
        CHECK(ivalue == 1);

        ivalue = float1(4.f) <= float1(3.f);
        CHECK(ivalue == 0);

        ivalue = float1(4.f) <= float1(4.f);
        CHECK(ivalue == 1);

        ivalue = float1(4.f) <= float1(5.f);
        CHECK(ivalue == 1);

        ivalue = float1(4.f) > float1(3.f);
        CHECK(ivalue == 1);

        ivalue = float1(4.f) > float1(4.f);
        CHECK(ivalue == 0);

        ivalue = float1(4.f) > float1(5.f);
        CHECK(ivalue == 0);

        ivalue = float1(4.f) >= float1(3.f);
        CHECK(ivalue == 1);

        ivalue = float1(4.f) >= float1(4.f);
        CHECK(ivalue == 1);

        ivalue = float1(4.f) >= float1(5.f);
        CHECK(ivalue == 0);

        ivalue = float1(10.f) == float1(5.f);
        CHECK(ivalue == 0);

        ivalue = float1(10.f) == float1(10.f);
        CHECK(ivalue == 1);

        ivalue = float1(10.f) != float1(5.f);
        CHECK(ivalue == 1);

        ivalue = float1(10.f) != float1(10.f);
        CHECK(ivalue == 0);
    }

    TEST(float4_operator)
    {
        float4 a = float4(1.f, 2.f, 3.f, 4.f);
        float4 b = float4(4.f, 3.f, 2.f, 1.f);
        float4 e = float4(54.f, 3.f, 42.f, 2.f);

        float4 c = a + b;
        CHECK_CLOSE(5.f, (float)c.x, epsilon);
        CHECK_CLOSE(5.f, (float)c.y, epsilon);
        CHECK_CLOSE(5.f, (float)c.z, epsilon);
        CHECK_CLOSE(5.f, (float)c.w, epsilon);

        c = a + b.wwwx;
        CHECK_CLOSE(2.f, (float)c.x, epsilon);
        CHECK_CLOSE(3.f, (float)c.y, epsilon);
        CHECK_CLOSE(4.f, (float)c.z, epsilon);
        CHECK_CLOSE(8.f, (float)c.w, epsilon);

        c = a + b.z;
        CHECK_CLOSE(3.f, (float)c.x, epsilon);
        CHECK_CLOSE(4.f, (float)c.y, epsilon);
        CHECK_CLOSE(5.f, (float)c.z, epsilon);
        CHECK_CLOSE(6.f, (float)c.w, epsilon);

        c = a + b.wwwx + e.y;
        CHECK_CLOSE(5.f, (float)c.x, epsilon);
        CHECK_CLOSE(6.f, (float)c.y, epsilon);
        CHECK_CLOSE(7.f, (float)c.z, epsilon);
        CHECK_CLOSE(11.f, (float)c.w, epsilon);

        float4 d = a;
        CHECK_CLOSE(1.f, (float)d.x, epsilon);
        CHECK_CLOSE(2.f, (float)d.y, epsilon);
        CHECK_CLOSE(3.f, (float)d.z, epsilon);
        CHECK_CLOSE(4.f, (float)d.w, epsilon);

        float1 a1 = float1(10.f);

        d = a + a1;
        CHECK_CLOSE(11.f, (float)d.x, epsilon);
        CHECK_CLOSE(12.f, (float)d.y, epsilon);
        CHECK_CLOSE(13.f, (float)d.z, epsilon);
        CHECK_CLOSE(14.f, (float)d.w, epsilon);

        a.x = 0.f;
        CHECK(all(a == float4(0.f, 2.f, 3.f, 4.f)));

        a.y = float1(12.f);
        CHECK(all(a == float4(0.f, 12.f, 3.f, 4.f)));

        a = float4(1.f, 2.f, 3.f, 4.f);

        c = a + b;
        CHECK(all(c == float4(5.f, 5.f, 5.f, 5.f)));

        c = a * b;
        CHECK(all(c == float4(4.f, 6.f, 6.f, 4.f)));

        c = a / b;
        CHECK(all(c == float4(1.f / 4.f, 2.f / 3.f, 3.f / 2.f, 4.f / 1.f)));

        a += float4(1.f);
        c = a;
        CHECK(all(c == float4(2.f, 3.f, 4.f, 5.f)));
        CHECK(all(a == float4(2.f, 3.f, 4.f, 5.f)));

        a += float4(1.f);
        CHECK(all(c == float4(2.f, 3.f, 4.f, 5.f)));
        CHECK(all(a == float4(3.f, 4.f, 5.f, 6.f)));

        c += b;
        CHECK(all(c == float4(6.f, 6.f, 6.f, 6.f)));

        c -= a;
        CHECK(all(c == float4(3.f, 2.f, 1.f, 0.f)));

        c += 5.f;
        CHECK(all(c == float4(8.f, 7.f, 6.f, 5.f)));

        c *= b;
        CHECK(all(c == float4(32.f, 21.f, 12.f, 5.f)));

        c /= b;
        CHECK(all(c == float4(8.f, 7.f, 6.f, 5.f)));

        c = -c;
        CHECK(all(c == float4(-8.f, -7.f, -6.f, -5.f)));

        c -= .5f;
        CHECK(all(c == float4(-8.5f, -7.5f, -6.5f, -5.5f)));

        c *= 2.f;
        CHECK(all(c == float4(-17.f, -15.f, -13.f, -11.f)));

        c /= 3.f;
        CHECK_CLOSE(-17.f / 3.f, (float)c.x, epsilon);
        CHECK_CLOSE(-15.f / 3.f, (float)c.y, epsilon);
        CHECK_CLOSE(-13.f / 3.f, (float)c.z, epsilon);
        CHECK_CLOSE(-11.f / 3.f, (float)c.w, epsilon);
    }

    TEST(vec1_operator)
    {
        float4 c = float4(-1.f, 2.f, -3.f, 4.f);
        float4 t = float4(5.f, 6.f, 7.f, 8.f);

        t.x *= float1(-1.f);
        CHECK(all(t == float4(-5.f, 6.f, 7.f, 8.f)));

        t.y += float1(4.f);
        CHECK(all(t == float4(-5.f, 10.f, 7.f, 8.f)));

        t.z -= float1(-2.f);
        CHECK(all(t == float4(-5.f, 10.f, 9.f, 8.f)));

        t.w /= float1(-2.f);
        CHECK(all(t == float4(-5.f, 10.f, 9.f, -4.f)));

        t.x *= c.w;
        CHECK(all(t == float4(-20.f, 10.f, 9.f, -4.f)));

        t.y /= c.z;
        CHECK_CLOSE(-20.f, (float)t.x, epsilon);
        CHECK_CLOSE(10.f / -3.f, (float)t.y, epsilon);
        CHECK_CLOSE(9.f, (float)t.z, epsilon);
        CHECK_CLOSE(-4.f, (float)t.w, epsilon);

        t.w += c.y;
        CHECK_CLOSE(-20.f, (float)t.x, epsilon);
        CHECK_CLOSE(10.f / -3.f, (float)t.y, epsilon);
        CHECK_CLOSE(9.f, (float)t.z, epsilon);
        CHECK_CLOSE(-2.f, (float)t.w, epsilon);

        t.z -= c.x;
        CHECK_CLOSE(-20.f, (float)t.x, epsilon);
        CHECK_CLOSE(10.f / -3.f, (float)t.y, epsilon);
        CHECK_CLOSE(10.f, (float)t.z, epsilon);
        CHECK_CLOSE(-2.f, (float)t.w, epsilon);

        float x = -(float)c.x;
        CHECK(x == 1.f);
    }

    TEST(float_compare)
    {
        float4 a4 = float4(1.f,  2.f, 3.f, 4.f);
        float4 b4 = float4(-1.f, -2.f, 6.f, 4.f);
        int4 c4;

        // float4 comparison returns 0(false) or -1(true)
        c4 = a4 < b4;
        CHECK(all(c4 == int4(0, 0, -1, 0)));

        c4 = a4 > b4;
        CHECK(all(c4 == int4(-1, -1, 0, 0)));

        c4 = a4 == b4;
        CHECK(all(c4 == int4(0, 0, 0, -1)));

        c4 = a4 != b4;
        CHECK(all(c4 == int4(-1, -1, -1, 0)));


        // float1 comparison returns 0(false) or 1(true)!!!
        // THIS IS INCONSISTENT WITH ALL OTHER FLOAT4 / FLOAT3 / FLOAT2
        // This behaviour is defined by the standard,
        // and when you think hard enough about compatibility with existing scalar code this inconsistency even makes sense!
        float1 a1 = float1(1.f);
        float1 b1 = float1(-1.f);
        int1 c1;

        c1 = a1 < b1;
        CHECK_EQUAL(0, c1);

        c1 = a1 > b1;
        CHECK_EQUAL(1, c1);

        c1 = a1 == b1;
        CHECK_EQUAL(0, c1);

        c1 = a1 != b1;
        CHECK_EQUAL(1, c1);

        // float3  comparison returns 0(false) or -1(true)
        float3 a3 = float3(1.f,  2.f, 3.f);
        float3 b3 = float3(-1.f, -2.f, 6.f);
        int3 c3;

        c3 = a3 < b3;
        CHECK(all(c3 == int3(0, 0, -1)));

        c3 = a3 > b3;
        CHECK(all(c3 == int3(-1, -1, 0)));

        c3 = a3 == b3;
        CHECK(all(c3 == int3(0, 0, 0)));

        c3 = a3 != b3;
        CHECK(all(c3 == int3(-1, -1, -1)));

        // float2
        float2 a2 = float2(-1, 2);
        float2 b2 = float2(1, -2);
        int2 c2;

        c2 = a2 < b2;
        CHECK(all(c2 == int2(-1, 0)));

        c2 = a2 > b2;
        CHECK(all(c2 == int2(0, -1)));

        c2 = a2 == b2;
        CHECK(all(c2 == int2(0, 0)));

        c2 = a2 != b2;
        CHECK(all(c2 == int2(-1, -1)));
    }

    TEST(vec2_operator)
    {
        float2 c = float2(1.0f, 2.0f);

        CHECK(all(c == float2(1.0f, 2.0f)));
        CHECK(!all(c == float2(0.0f, 2.0f)));

        CHECK(any(c == float2(0.0f, 2.0f)));
        CHECK(!any(c == float2(0.0f, 0.0f)));
    }


    TEST(float_generic)
    {
        float4 a = float4(-1.f, -.263f, 345.f, 0.f);

        float4* ptrA = test_cast(a);
        CHECK(all(a == *ptrA));
    }

    TEST(any_float4_Works)
    {
        int4 bv = int4(0, 0, 0, 0);
        CHECK(any(bv) == 0);

        bv = int4(0, ~0, 0, 0);
        CHECK(any(bv) == 1);

        bv = int4(~0, ~0, ~0, ~0);
        CHECK(any(bv) == 1);
    }

    TEST(all_float4_Works)
    {
        int4 bv = int4(0, 0, 0, 0);
        CHECK(all(bv) == 0);

        bv = int4(0, ~0, 0, 0);
        CHECK(all(bv) == 0);

        bv = int4(~0, ~0, ~0, ~0);
        CHECK(all(bv) == 1);
    }

    TEST(float4_storage_aligned_cast)
    {
        math::float4_storage_aligned temp[2];

        temp[0] = math::float4(1.0F);
        temp[1] = math::float4(4.0F);

        math::float4* casted = reinterpret_cast<math::float4*>(temp);

        CHECK(math::all(casted[0] == math::float4(1.0F)));
        CHECK(math::all(casted[1] == math::float4(4.0F)));
    }

    TEST(float_aligned)
    {
        alignas(16) float s[5];

        {
            float4 v = float4(1.f, 2.f, 3.f, 4.f);
            float4 u;
            for (int j = 0; j < 5; ++j)
                s[j] = 5.f;
            vstore4f_aligned(&s[0], v);
            CHECK_EQUAL(1.f, s[0]);
            CHECK_EQUAL(2.f, s[1]);
            CHECK_EQUAL(3.f, s[2]);
            CHECK_EQUAL(4.f, s[3]);
            CHECK_EQUAL(5.f, s[4]);
            u = vload4f_aligned(&s[0]);
            CHECK(all(u == v));
        }
    }

    TEST(float_unaligned)
    {
        float s[8];

        {
            float4 v = float4(1.f, 2.f, 3.f, 4.f);
            float4 u;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 8; ++j)
                    s[j] = 5.f;
                vstore4f(&s[i], v);
                for (int j = 0; j < i; ++j)
                    CHECK(s[j] == 5.f);
                CHECK(s[i] == 1.f);
                CHECK(s[i + 1] == 2.f);
                CHECK(s[i + 2] == 3.f);
                CHECK(s[i + 3] == 4.f);
                for (int j = i + 4; j < 8; ++j)
                    CHECK(s[j] == 5.f);
                u = vload4f(&s[i]);
                CHECK(all(u == v));
            }
        }

        {
            float3 v = float3(1.f, 2.f, 3.f);
            float3 u;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 8; ++j)
                    s[j] = 5.f;
                vstore3f(&s[i], v);
                for (int j = 0; j < i; ++j)
                    CHECK(s[j] == 5.f);
                CHECK(s[i] == 1.f);
                CHECK(s[i + 1] == 2.f);
                CHECK(s[i + 2] == 3.f);
                for (int j = i + 3; j < 8; ++j)
                    CHECK(s[j] == 5.f);
                u = vload3f(&s[i]);
                CHECK(all(u == v));
            }
        }

        {
            float2 v = float2(1.f, 2.f);
            float2 u;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 8; ++j)
                    s[j] = 5.f;
                vstore2f(&s[i], v);
                for (int j = 0; j < i; ++j)
                    CHECK(s[j] == 5.f);
                CHECK(s[i] == 1.f);
                CHECK(s[i + 1] == 2.f);
                for (int j = i + 2; j < 8; ++j)
                    CHECK(s[j] == 5.f);
                u = vload2f(&s[i]);
                CHECK(all(u == v));
            }
        }

        {
            float1 v = float1(1.f);
            float1 u;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 8; ++j)
                    s[j] = 5.f;
                vstore1f(&s[i], v);
                for (int j = 0; j < i; ++j)
                    CHECK(s[j] == 5.f);
                CHECK(s[i] == 1.f);
                for (int j = i + 1; j < 8; ++j)
                    CHECK(s[j] == 5.f);
                u = vload1f(&s[i]);
                CHECK(u == v);
            }
        }
    }

    TEST(test_division_by_itself_produces_one)
    {
        float4 value1 = float4(7.0f, 31.0f, 127.0f, 255.0f);
        float4 value2 = float4(7.0f, 31.0f, 127.0f, 255.0f);

        float4 value3 = value1 / value2;

        CHECK(all(value3 == float4(1.0f, 1.0f, 1.0f, 1.0f)));
    }
}

#if PLATFORM_WIN
#pragma warning(pop)
#endif

#endif
