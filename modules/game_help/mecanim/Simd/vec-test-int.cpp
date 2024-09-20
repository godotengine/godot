#include "UnityPrefix.h"
#if ENABLE_UNIT_TESTS
#include "Runtime/Testing/Testing.h"
#include "Runtime/Testing/OptimizerPrevent.h"
#include "Runtime/Profiler/TimeHelper.h"

#include "vec-types.h"
#include "vec-quat.h"
#include "Modules/Animation/mecanim/math/axes.h"

UNIT_TEST_SUITE(SIMDMath_intOps)
{
    using namespace math;

    template<typename TYPE> TYPE* test_cast(TYPE& data)
    {
        return &data;
    }

    TEST(int_swizzle)
    {
        int4 value = int4(1, 2, 3, 4);

        int4 a = int4(0, 0, 0, 0);

        a.x = 3.f;
        CHECK_EQUAL(3, (int)a.x);
        CHECK_EQUAL(0, (int)a.y);
        CHECK_EQUAL(0, (int)a.z);
        CHECK_EQUAL(0, (int)a.w);

        a.x = value.x;
        CHECK_EQUAL(1, (int)a.x);
        CHECK_EQUAL(0, (int)a.y);
        CHECK_EQUAL(0, (int)a.z);
        CHECK_EQUAL(0, (int)a.w);

        a.y = value.y;
        CHECK_EQUAL(1, (int)a.x);
        CHECK_EQUAL(2, (int)a.y);
        CHECK_EQUAL(0, (int)a.z);
        CHECK_EQUAL(0, (int)a.w);

        a.z = value.z;
        CHECK_EQUAL(1, (int)a.x);
        CHECK_EQUAL(2, (int)a.y);
        CHECK_EQUAL(3, (int)a.z);
        CHECK_EQUAL(0, (int)a.w);

        a.w = value.w;
        CHECK_EQUAL(1, (int)a.x);
        CHECK_EQUAL(2, (int)a.y);
        CHECK_EQUAL(3, (int)a.z);
        CHECK_EQUAL(4, (int)a.w);

        a.z = value.y;
        CHECK_EQUAL(1, (int)a.x);
        CHECK_EQUAL(2, (int)a.y);
        CHECK_EQUAL(2, (int)a.z);
        CHECK_EQUAL(4, (int)a.w);

        int1 f = value.w;
        CHECK_EQUAL(4, f);

        a = value.xyzw;
        CHECK(all(a == int4(1, 2, 3, 4)));

        int4 g(value.yzwx);
        CHECK(all(g == int4(2, 3, 4, 1)));

        a = value.zwxy;
        CHECK(all(a == int4(3, 4, 1, 2)));

        a = value.wxyz;
        CHECK(all(a == int4(4, 1, 2, 3)));

        a = value.wzyx;
        CHECK(all(a == int4(4, 3, 2, 1)));

        a = value.yzxw;
        CHECK(all(a == int4(2, 3, 1, 4)));

        a = value.ywzx;
        CHECK(all(a == int4(2, 4, 3, 1)));

        a = value.zxyw;
        CHECK(all(a == int4(3, 1, 2, 4)));

        a = value.xwyz;
        CHECK(all(a == int4(1, 4, 2, 3)));

        a = value.wzxy;
        CHECK(all(a == int4(4, 3, 1, 2)));

        a = value.xzyw;
        CHECK(all(a == int4(1, 3, 2, 4)));

        a = value.yxzw;
        CHECK(all(a == int4(2, 1, 3, 4)));

        a = value.zyxw;
        CHECK(all(a == int4(3, 2, 1, 4)));

        a = value.xyxy;
        CHECK(all(a == int4(1, 2, 1, 2)));

        a = value.yyww;
        CHECK(all(a == int4(2, 2, 4, 4)));

        a = value.zxzx;
        CHECK(all(a == int4(3, 1, 3, 1)));

        a = value.zzww;
        CHECK(all(a == int4(3, 3, 4, 4)));

        a = value.yzxy;
        CHECK(all(a == int4(2, 3, 1, 2)));

        a = value.wwwx;
        CHECK(all(a == int4(4, 4, 4, 1)));

        a = value.yxxw;
        CHECK(all(a == int4(2, 1, 1, 4)));

        a = value.zzyw;
        CHECK(all(a == int4(3, 3, 2, 4)));
    }

    TEST(int1_operator)
    {
        int1 b(3);
        int1 c(2);
        int4 cx;
        int1 e;

        {
            CHECK_EQUAL(3, (int)b);

            CHECK_EQUAL(2, (int)c);

            cx = int4(10, 2, 3, 4);

            int1 d(cx.x);

            CHECK_EQUAL(10, (int)d);

            e = cx.y;

            CHECK_EQUAL(2, (int)e);

            e = int1(4.f);
            CHECK_EQUAL(4, (int)e);

            e = int1(cx.x);
            CHECK_EQUAL(10, (int)e);

            e = cx.z;
            CHECK_EQUAL(3, (int)e);

            e += cx.w;
            CHECK_EQUAL(7, (int)e);

            e -= cx.x;
            CHECK_EQUAL(-3, (int)e);

            e *= cx.y;
            CHECK_EQUAL(-6, (int)e);

            e = int1(-6);
            e /= cx.z;
            CHECK_EQUAL(-2, (int)e);
        }
        e = int1(-2);
        {
            int1 f = e++;
            CHECK_EQUAL(-2, (int)f);

            CHECK_EQUAL(-1, (int)e);

            int1 g = ++e;
            CHECK_EQUAL(0, (int)g);

            CHECK_EQUAL(0, (int)e);

            int1 j(4);
            int1 l(3);

            int1 m = j + l;
            CHECK_EQUAL(7, (int)m);

            int1 n = j - l;
            CHECK_EQUAL(1, (int)n);

            int1 o = j * l;
            CHECK_EQUAL(12, (int)o);

            int1 p = j / l;
            CHECK_EQUAL(4 / 3, (int)p);
        }

        int1 ivalue = int1(4) < int1(3);
        CHECK(ivalue == 0);

        ivalue = int1(4) < int1(4);
        CHECK(ivalue == 0);

        ivalue = int1(4) < int1(5);
        CHECK(ivalue == 1);

        ivalue = int1(4) <= int1(3);
        CHECK(ivalue == 0);

        ivalue = int1(4) <= int1(4);
        CHECK(ivalue == 1);

        ivalue = int1(4) <= int1(5);
        CHECK(ivalue == 1);

        ivalue = int1(4) > int1(3);
        CHECK(ivalue == 1);

        ivalue = int1(4) > int1(4);
        CHECK(ivalue == 0);

        ivalue = int1(4) > int1(5);
        CHECK(ivalue == 0);

        ivalue = int1(4) >= int1(3);
        CHECK(ivalue == 1);

        ivalue = int1(4) >= int1(4);
        CHECK(ivalue == 1);

        ivalue = int1(4) >= int1(5);
        CHECK(ivalue == 0);

        ivalue = int1(10) == int1(5);
        CHECK(ivalue == 0);

        ivalue = int1(10) == int1(10);
        CHECK(ivalue == 1);

        ivalue = int1(10) != int1(5);
        CHECK(ivalue == 1);

        ivalue = int1(10) != int1(10);
        CHECK(ivalue == 0);
    }

    TEST(int4_operator)
    {
        int4 a = int4(1, 2, 3, 4);
        int4 b = int4(4, 3, 2, 1);
        int4 e = int4(54, 3, 42, 2);

        int4 c = a + b;
        CHECK_EQUAL(5, (int)c.x);
        CHECK_EQUAL(5, (int)c.y);
        CHECK_EQUAL(5, (int)c.z);
        CHECK_EQUAL(5, (int)c.w);

        c = a + b.wwwx;
        CHECK_EQUAL(2, (int)c.x);
        CHECK_EQUAL(3, (int)c.y);
        CHECK_EQUAL(4, (int)c.z);
        CHECK_EQUAL(8, (int)c.w);

        c = a + b.z;
        CHECK_EQUAL(3, (int)c.x);
        CHECK_EQUAL(4, (int)c.y);
        CHECK_EQUAL(5, (int)c.z);
        CHECK_EQUAL(6, (int)c.w);

        c = a + b.wwwx + e.y;
        CHECK_EQUAL(5, (int)c.x);
        CHECK_EQUAL(6, (int)c.y);
        CHECK_EQUAL(7, (int)c.z);
        CHECK_EQUAL(11, (int)c.w);

        int4 d = a;
        CHECK_EQUAL(1, (int)d.x);
        CHECK_EQUAL(2, (int)d.y);
        CHECK_EQUAL(3, (int)d.z);
        CHECK_EQUAL(4, (int)d.w);

        int1 a1 = int1(10);

        d = a + a1;
        CHECK_EQUAL(11, (int)d.x);
        CHECK_EQUAL(12, (int)d.y);
        CHECK_EQUAL(13, (int)d.z);
        CHECK_EQUAL(14, (int)d.w);

        a.x = 0;
        CHECK(all(a == int4(0, 2, 3, 4)));

        a.y = int1(12);
        CHECK(all(a == int4(0, 12, 3, 4)));

        a = int4(1, 2, 3, 4);

        c = a + b;
        CHECK(all(c == int4(5, 5, 5, 5)));

        c = a * b;
        CHECK(all(c == int4(4, 6, 6, 4)));

        c = a / b;
        CHECK(all(c == int4(1 / 4, 2 / 3, 3 / 2, 4 / 1)));

        a += int4(1);
        c = a;
        CHECK(all(c == int4(2, 3, 4, 5)));
        CHECK(all(a == int4(2, 3, 4, 5)));

        a += int4(1);
        CHECK(all(c == int4(2, 3, 4, 5)));
        CHECK(all(a == int4(3, 4, 5, 6)));

        c += b;
        CHECK(all(c == int4(6, 6, 6, 6)));

        c -= a;
        CHECK(all(c == int4(3, 2, 1, 0)));

        c += 5;
        CHECK(all(c == int4(8, 7, 6, 5)));

        c *= b;
        CHECK(all(c == int4(32, 21, 12, 5)));

        c /= b;
        CHECK(all(c == int4(8, 7, 6, 5)));

        c = -c;
        CHECK(all(c == int4(-8, -7, -6, -5)));

        c -= 1;
        CHECK(all(c == int4(-9, -8, -7, -6)));

        c *= 2;
        CHECK(all(c == int4(-18, -16, -14, -12)));

        c /= 3;
        CHECK_EQUAL(-18 / 3, (int)c.x);
        CHECK_EQUAL(-16 / 3, (int)c.y);
        CHECK_EQUAL(-14 / 3, (int)c.z);
        CHECK_EQUAL(-12 / 3, (int)c.w);
    }

    TEST(ivec1_operator)
    {
        int4 c = int4(-1, 2, -3, 4);
        int4 t = int4(5, 6, 7, 8);

        t.x *= int1(-1);
        CHECK(all(t == int4(-5, 6, 7, 8)));

        t.y += int1(4);
        CHECK(all(t == int4(-5, 10, 7, 8)));

        t.z -= int1(-2);
        CHECK(all(t == int4(-5, 10, 9, 8)));

        t.w /= int1(-2);
        CHECK(all(t == int4(-5, 10, 9, -4)));

        t.x *= c.w;
        CHECK(all(t == int4(-20, 10, 9, -4)));

        t.y /= c.z;
        CHECK_EQUAL(-20, (int)t.x);
        CHECK_EQUAL(10 / -3, (int)t.y);
        CHECK_EQUAL(9, (int)t.z);
        CHECK_EQUAL(-4, (int)t.w);

        t.w += c.y;
        CHECK_EQUAL(-20, (int)t.x);
        CHECK_EQUAL(10 / -3, (int)t.y);
        CHECK_EQUAL(9, (int)t.z);
        CHECK_EQUAL(-2, (int)t.w);

        t.z -= c.x;
        CHECK_EQUAL(-20, (int)t.x);
        CHECK_EQUAL(10 / -3, (int)t.y);
        CHECK_EQUAL(10, (int)t.z);
        CHECK_EQUAL(-2, (int)t.w);

        int x = -(int)c.x;
        CHECK(x == 1);
    }

    TEST(int_compare)
    {
        int4 a4 = int4(1,  2, 3, 4);
        int4 b4 = int4(-1, -2, 6, 4);
        int4 c4;

        // int4 comparison returns 0(false) or -1(true)
        c4 = a4 < b4;
        CHECK(all(c4 == int4(0, 0, -1, 0)));

        c4 = a4 > b4;
        CHECK(all(c4 == int4(-1, -1, 0, 0)));

        c4 = a4 == b4;
        CHECK(all(c4 == int4(0, 0, 0, -1)));

        c4 = a4 != b4;
        CHECK(all(c4 == int4(-1, -1, -1, 0)));


        // int1 comparison returns 0(false) or 1(true)!!!
        // THIS IS INCONSISTENT WITH ALL OTHER FLOAT4 / FLOAT3 / FLOAT2
        // This behaviour is defined by the standard,
        // and when you think hard enough about compatibility with existing scalar code this inconsistency even makes sense!
        int1 a1 = int1(1);
        int1 b1 = int1(-1);
        int1 c1;

        c1 = a1 < b1;
        CHECK_EQUAL(0, c1);

        c1 = a1 > b1;
        CHECK_EQUAL(1, c1);

        c1 = a1 == b1;
        CHECK_EQUAL(0, c1);

        c1 = a1 != b1;
        CHECK_EQUAL(1, c1);

        // int3  comparison returns 0(false) or -1(true)
        int3 a3 = int3(1,  2, 3);
        int3 b3 = int3(-1, -2, 6);
        int3 c3;

        c3 = a3 < b3;
        CHECK(all(c3 == int3(0, 0, -1)));

        c3 = a3 > b3;
        CHECK(all(c3 == int3(-1, -1, 0)));

        c3 = a3 == b3;
        CHECK(all(c3 == int3(0, 0, 0)));

        c3 = a3 != b3;
        CHECK(all(c3 == int3(-1, -1, -1)));

        // int2
        int2 a2 = int2(1, -2);
        int2 b2 = int2(-1, -2);
        int2 c2;

        c2 = a2 < b2;
        CHECK(all(c2 == int2(0, 0)));

        c2 = a2 > b2;
        CHECK(all(c2 == int2(-1, 0)));

        c2 = a2 == b2;
        CHECK(all(c2 == int2(0, -1)));

        c2 = a2 != b2;
        CHECK(all(c2 == int2(-1, 0)));
    }

    TEST(ivec2_operator)
    {
        int2 c = int2(1, 2);

        CHECK(all(c == int2(1, 2)));
        CHECK(!all(c == int2(0, 2)));

        CHECK(any(c == int2(0, 2)));
        CHECK(!any(c == int2(0, 0)));
    }

    TEST(select)
    {
        const int4 a = int4(-1, 0, 345, 0);
        const int4 b = int4(5, 2, -12, 54);
        int4 c;

        // res = msb(c) ? b : a;
        c = select(a, b, int4(0));
        CHECK(all(c == a));

        // (and the result of a compare is -1 -> true)
        c = select(a, b, int4(-1));
        CHECK(all(c == b));

        // Same as when doing a compare
        c = select(a, b, int4(0) < int4(1));
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

        c = select(a, b, int4(0) > int4(1));
        CHECK(all(c == a));


        c = select(a, b, a < b);
        CHECK(all(c == int4(5, 2, 345, 54)));

        c = select(a, b, a > b);
        CHECK(all(c == int4(-1, 0, -12, 0)));


        int4 d;

        int index = -1;
        d = select(int4(3, 3, 3, 3), int4(2, 2, 2, 2),  int4(-(index != -1)));
        CHECK(all(d == int4(3, 3, 3, 3)));

        index = 1;
        d = select(int4(3, 3, 3, 3), int4(2, 2, 2, 2),  int4(-(index != -1)));
        CHECK(all(d == int4(2, 2, 2, 2)));

        d = select(int4(0), int4(1), int4(0) != int4(0));
        CHECK(all(d == int4(0)));

        d = select(int4(0), int4(1), int4(1) != int4(0));
        CHECK(all(d == int4(1)));
    }

    TEST(abs_int4_Works)
    {
        int4 c = math::abs(int4(-1, -263, 345, -0));
        CHECK(all(c == int4(1, 263, 345, 0)));
    }

    TEST(abs_int3_Works)
    {
        int3 c = math::abs(int3(-1, -263, 345));
        CHECK(all(c == int3(1, 263, 345)));
    }

    TEST(abs_int2_Works)
    {
        int2 c = math::abs(int2(-1, 263));
        CHECK(all(c == int2(1, 263)));
    }

    TEST(abs_int1_Works)
    {
        int1 c = math::abs(int1(-1));
        CHECK_EQUAL(int1(1), c);
    }

    TEST(abs_int_Works)
    {
        int c = math::abs(-1);
        CHECK_EQUAL(1, c);
    }

    TEST(clamp_int4_Works)
    {
        int4 a = int4(0, 1, 100, -2);
        int4 b = int4(2, 3, 200, -10);

        int4 c = clamp(int4(0, 1, 100, -2), a, b);
        CHECK(all(c == int4(0, 1, 100, -10)));

        c = clamp(int4(-10, 2, 300, 20), a, b);
        CHECK(all(c == int4(0, 2, 200, -10)));
    }

    TEST(clamp_int3_Works)
    {
        int3 a = int3(0, 1, 100);
        int3 b = int3(2, 3, 200);

        int3 c = clamp(int3(0, 1, 100), a, b);
        CHECK(all(c == int3(0, 1, 100)));

        c = clamp(int3(-10, 2, 300), a, b);
        CHECK(all(c == int3(0, 2, 200)));
    }

    TEST(clamp_int2_Works)
    {
        int2 a = int2(0, 1);
        int2 b = int2(2, 3);

        int2 c = clamp(int2(0, 1), a, b);
        CHECK(all(c == int2(0, 1)));

        c = clamp(int2(-10, 2), a, b);
        CHECK(all(c == int2(0, 2)));
    }

    TEST(clamp_int1_Works)
    {
        int1 a = int1(0);
        int1 b = int1(2);

        int1 c = clamp(int1(0), a, b);
        CHECK_EQUAL(int1(0), c);

        c = clamp(int1(1), a, b);
        CHECK_EQUAL(int1(1), c);

        c = clamp(int1(3), a, b);
        CHECK_EQUAL(int1(2), c);
    }

    TEST(clamp_int_Works)
    {
        int a = int(0);
        int b = int(2);

        int c = math::clamp(0, a, b);
        CHECK_EQUAL(int(0), c);

        c = math::clamp(int(1), a, b);
        CHECK_EQUAL(int(1), c);

        c = math::clamp(int(3), a, b);
        CHECK_EQUAL(int(2), c);
    }

    TEST(int_generic)
    {
        int4 a = int4(-1, -263, 345, 0);
        int4 b = int4(5, 234, -12, 54);
        int4 c;

        int1 s;

        int4* ptrA = test_cast(a);
        CHECK(all(a == *ptrA));

        s = cmax(int4(-1, -263, 345, 0));
        CHECK_EQUAL(345, (int)s);

        s = cmin(int4(-1, -263, 345, 0));
        CHECK_EQUAL(-263, (int)s);

        int i = sign(-25);
        CHECK_EQUAL(-1, i);

        i = sign(0);
        CHECK_EQUAL(0, i);

        i = sign(3);
        CHECK_EQUAL(1, i);

        a = int4(-1, -4, 8, 1);
        b = int4(5, 2, -2, 54);
        c = int4(-25, 0, 1, 2);
        int4 d = a * b + c;
        CHECK(all(d == int4(-30, -8, -15, 56)));

        d = a * b - c;
        CHECK(all(d == int4(20, -8, -17, 52)));

        int4 bv = int4(0, 0, 0, 0);
        CHECK(any(bv) == 0);
        CHECK(all(bv) == 0);

        bv = int4(0, ~0, 0, 0);
        CHECK(any(bv) == 1);
        CHECK(all(bv) == 0);

        bv = int4(~0, ~0, ~0, ~0);
        CHECK(any(bv) == 1);
        CHECK(all(bv) == 1);

        int index = -1;
        d = math::select(math::int4(3, 3, 3, 3), math::int4(2, 2, 2, 2),  math::int4(-(index != -1)));
        CHECK(all(d == int4(3, 3, 3, 3)));

        index = 1;
        d = math::select(math::int4(3, 3, 3, 3), math::int4(2, 2, 2, 2),  math::int4(-(index != -1)));
        CHECK(all(d == int4(2, 2, 2, 2)));

        d = math::select(math::int4(0), math::int4(1), math::int4(0) != math::int4(0));
        CHECK(all(d == int4(0)));
    }

    TEST(int_aligned)
    {
        alignas(16) int s[5];

        {
            int4 v = int4(1, 2, 3, 4);
            int4 u;
            for (int j = 0; j < 5; ++j)
                s[j] = 5;
            vstore4i_aligned(&s[0], v);
            CHECK_EQUAL(1, s[0]);
            CHECK_EQUAL(2, s[1]);
            CHECK_EQUAL(3, s[2]);
            CHECK_EQUAL(4, s[3]);
            CHECK_EQUAL(5, s[4]);
            u = vload4i_aligned(&s[0]);
            CHECK(all(u == v));
        }
    }

    TEST(int_unaligned)
    {
        int s[8];
        {
            int4 v = int4(1, 2, 3, 4);
            int4 u;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 8; ++j)
                    s[j] = 5;
                vstore4i(&s[i], v);
                for (int j = 0; j < i; ++j)
                    CHECK(s[j] == 5);
                CHECK(s[i] == 1);
                CHECK(s[i + 1] == 2);
                CHECK(s[i + 2] == 3);
                CHECK(s[i + 3] == 4);
                for (int j = i + 4; j < 8; ++j)
                    CHECK(s[j] == 5);
                u = vload4i(&s[i]);
                CHECK(all(u == v));
            }
        }

        {
            int3 v = int3(1, 2, 3);
            int3 u;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 8; ++j)
                    s[j] = 5;
                vstore3i(&s[i], v);
                for (int j = 0; j < i; ++j)
                    CHECK(s[j] == 5);
                CHECK(s[i] == 1);
                CHECK(s[i + 1] == 2);
                CHECK(s[i + 2] == 3);
                for (int j = i + 3; j < 8; ++j)
                    CHECK(s[j] == 5);
                u = vload3i(&s[i]);
                CHECK(all(u == v));
            }
        }

        {
            int2 v = int2(1, 2);
            int2 u;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 8; ++j)
                    s[j] = 5;
                vstore2i(&s[i], v);
                for (int j = 0; j < i; ++j)
                    CHECK(s[j] == 5);
                CHECK(s[i] == 1);
                CHECK(s[i + 1] == 2);
                for (int j = i + 2; j < 8; ++j)
                    CHECK(s[j] == 5);
                u = vload2i(&s[i]);
                CHECK(all(u == v));
            }
        }

        {
            int1 v = int1(1);
            int1 u;

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 8; ++j)
                    s[j] = 5;
                vstore1i(&s[i], v);
                for (int j = 0; j < i; ++j)
                    CHECK(s[j] == 5);
                CHECK(s[i] == 1);
                for (int j = i + 1; j < 8; ++j)
                    CHECK(s[j] == 5);
                u = vload1i(&s[i]);
                CHECK(u == v);
            }
        }
    }

    TEST(int_bitops)
    {
        int4 a = int4(999, -263, 345, 0);
        int4 b = int4(5, 234, 12, 54);
        int4 c;

        c = a | b;
        CHECK(all(c == int4(999, -261, 349, 54)));

        c = a & b;
        CHECK(all(c == int4(5, 232, 8, 0)));

        c = a ^ b;
        CHECK(all(c == int4(994, -493, 341, 54)));

        c = a << 5;
        CHECK(all(c == int4(31968, -8416, 11040, 0)));

        c = b >> 2;
        CHECK(all(c == int4(1, 58, 3, 13)));
    }

    TEST(shiftRightLogical_works)
    {
        const int4 a = int4(-1, 0, 1 << 25, 253897987);

        CHECK(all(shiftRightLogical(a, 4) == int4(0x0fffffff, 0, 0x00200000, 15868624)));
        CHECK(all(shiftRightLogical(a, 12) == int4(0x000fffff, 0, 0x00002000, 61986)));
        CHECK(all(shiftRightLogical(a, 31) == int4(1, 0, 0, 0)));
    }

    TEST(shiftLeftLogical_works)
    {
        const int4 a = int4(-1, 0, 1 , 253897987);

        CHECK(all(shiftLeftLogical(a, 4) == int4(0xfffffff0, 0, 0x00000010, -232599504)));
        CHECK(all(shiftLeftLogical(a, 12) == int4(0xfffff000, 0, 0x00001000, 584069120)));
        CHECK(all(shiftLeftLogical(a, 31) == int4(0x80000000, 0, 0x80000000, 0x80000000)));
    }

    TEST(charLoad_SignedAndUnsignedWorkTheSame)
    {
        unsigned char data[16] =
        {
            0, 20, 30, 40,
            110, 120, 130, 140,
            150, 160, 170, 180,
            210, 220, 230, 255
        };

        int expectedContents[4] =
        {
            0 | 20 << 8 | 30 << 16 | 40 << 24,
            110 | 120 << 8 | 130 << 16 | 140 << 24,
            150 | 160 << 8 | 170 << 16 | 180 << 24,
            210 | 220 << 8 | 230 << 16 | 255 << 24,
        };

        // force the compiler to issue SIMD loads instead of resolving
        // everything at compile time
        unsigned char* unsignedDataPtr = OPTIMIZER_PREVENT(data);
        char* signedDataPtr = reinterpret_cast<char*>(OPTIMIZER_PREVENT(data));

        int4 expected = vload4i(expectedContents);

        CHECK(all(expected == vload16uc(unsignedDataPtr)));
        CHECK(all(expected == vload16c(signedDataPtr)));
    }

    TEST(charUnalignedLoadDoesNotCrash)
    {
        alignas(16) unsigned char data[17] =
        {
            0,
            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 2, 3, 4,
            5, 6, 7, 8
        };

        alignas(16) int expected[4] =
        {
            0x04030201, 0x08070605, 0x04030201, 0x08070605
        };

        // force the compiler to issue SIMD loads instead of resolving
        // everything at compile time
        unsigned char* dataPtr = OPTIMIZER_PREVENT(data + 1);
        int* expectedPtr = OPTIMIZER_PREVENT(expected + 0);
        CHECK(all(vload16uc(dataPtr) == vload4i(expectedPtr)));
    }

    TEST(charUnalignedStoreDoesNotCrash)
    {
        alignas(16) char data[17] = { 0 };

        alignas(16) int expected[4] =
        {
            0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d
        };

        // force the compiler to issue SIMD loads instead of resolving
        // everything at compile time
        int4 a = vload4i(OPTIMIZER_PREVENT(expected));
        vstore16c(OPTIMIZER_PREVENT(data + 1), a);

        for (int i = 0; i < 17; ++i)
            CHECK_EQUAL((int)data[i], i);
    }

    TEST(char4byteUnalignedLoadDoesNotCrash)
    {
        alignas(16) char data[17] =
        {
            0,
            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 2, 3, 4,
            5, 6, 7, 8
        };

        alignas(16) int expected[4] =
        {
            0x04030201, 0x08070605, 0x04030201, 0x08070605
        };

        // force the compiler to issue SIMD loads instead of resolving
        // everything at compile time
        char* dataPtr = OPTIMIZER_PREVENT(data + 1);
        int* expectedPtr = OPTIMIZER_PREVENT(expected + 0);
        CHECK_EQUAL(vload1i(expectedPtr), vload4c(dataPtr));
    }

    TEST(char4byteUnalignedStoreDoesNotCrash)
    {
        alignas(16) char data[17] = { 0 };

        alignas(16) int expected[4] =
        {
            0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d
        };

        // force the compiler to issue SIMD loads instead of resolving
        // everything at compile time
        int1 a = vload1i(OPTIMIZER_PREVENT(expected));
        vstore4c(OPTIMIZER_PREVENT(data + 1), a);

        for (int i = 0; i < 5; ++i)
            CHECK_EQUAL(i, (int)data[i]);
        for (int i = 5; i < 17; ++i)
            CHECK_EQUAL(0, (int)data[i]);
    }
}

#endif
