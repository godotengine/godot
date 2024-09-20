#pragma once

#include "vec-pix-types.h"

namespace math
{
    enum { kSimdWidth = 4 };
    typedef float4 floatN;
    typedef int4 intN;
    typedef pix4 pixN;
    typedef int4_storage intN_storage;

    struct floatNx4
    {
        floatN x, y, z, w;

        MATH_EMPTYINLINE floatNx4() {}

        MATH_FORCEINLINE floatNx4(const floatNx4 &a)
            :   x(a.x)
            ,   y(a.y)
            ,   z(a.z)
            ,   w(a.w)
        {
        }

        MATH_FORCEINLINE floatNx4(const floatN& a, const floatN& b, const floatN& c, const floatN& d)
            :   x(a)
            ,   y(b)
            ,   z(c)
            ,   w(d)
        {
        }

        MATH_FORCEINLINE floatNx4(const float4& v)
            :   x(v.x)
            ,   y(v.y)
            ,   z(v.z)
            ,   w(v.w)
        {
        }

        MATH_FORCEINLINE floatNx4 &operator=(const floatNx4& m)
        {
            x = m.x; y = m.y; z = m.z; w = m.w;
            return *this;
        }

        MATH_FORCEINLINE floatNx4 operator+(const float1& v) const
        {
            return floatNx4(x + v, y + v, z + v, w + v);
        }

        MATH_FORCEINLINE floatNx4 &operator+=(const float1& v)
        {
            x += v;
            y += v;
            z += v;
            w += v;
            return *this;
        }
    };

    struct floatNx3
    {
        floatN x, y, z;

        MATH_EMPTYINLINE floatNx3() {}

        MATH_FORCEINLINE floatNx3(const floatNx3 &a)
            :   x(a.x)
            ,   y(a.y)
            ,   z(a.z)
        {
        }

        MATH_FORCEINLINE floatNx3(const floatN &a, const floatN &b, const floatN &c)
            :   x(a)
            ,   y(b)
            ,   z(c)
        {
        }

        MATH_FORCEINLINE floatNx3(const float3 &v)
            :   x(v.x)
            ,   y(v.y)
            ,   z(v.z)
        {
        }

        MATH_FORCEINLINE floatNx3 &operator=(const floatNx3 &m)
        {
            x = m.x; y = m.y; z = m.z;
            return *this;
        }

        MATH_FORCEINLINE floatNx3 operator+(const floatNx3& m) const
        {
            return floatNx3(x + m.x, y + m.y, z + m.z);
        }

        MATH_FORCEINLINE floatNx3 operator+(const floatN& v) const
        {
            return floatNx3(x + v, y + v, z + v);
        }

        MATH_FORCEINLINE floatNx3 operator+(const float1& v) const
        {
            return floatNx3(x + v, y + v, z + v);
        }

        MATH_FORCEINLINE floatNx3 operator-(const floatNx3& m) const
        {
            return floatNx3(x - m.x, y - m.y, z - m.z);
        }

        MATH_FORCEINLINE floatNx3 operator-(const floatN& v) const
        {
            return floatNx3(x - v, y - v, z - v);
        }

        MATH_FORCEINLINE floatNx3 operator-(const float1& v) const
        {
            return floatNx3(x - v, y - v, z - v);
        }

        MATH_FORCEINLINE floatNx3 &operator+=(const floatNx3& m)
        {
            x += m.x;
            y += m.y;
            z += m.z;
            return *this;
        }

        MATH_FORCEINLINE floatNx3 &operator-=(const floatNx3& m)
        {
            x -= m.x;
            y -= m.y;
            z -= m.z;
            return *this;
        }

        MATH_FORCEINLINE floatNx3 &operator+=(const floatN& v)
        {
            x += v;
            y += v;
            z += v;
            return *this;
        }

        MATH_FORCEINLINE floatNx3 &operator-=(const floatN& v)
        {
            x -= v;
            y -= v;
            z -= v;
            return *this;
        }

        MATH_FORCEINLINE floatNx3 operator-() const
        {
            return floatNx3(-x, -y, -z);
        }
    };

    struct floatNx2
    {
        floatN x, y;

        MATH_EMPTYINLINE floatNx2() {}

        MATH_FORCEINLINE floatNx2(const floatNx2 &a)
            :   x(a.x)
            ,   y(a.y)
        {
        }

        MATH_FORCEINLINE floatNx2(const floatN &a, const floatN &b)
            :   x(a)
            ,   y(b)
        {
        }

        MATH_FORCEINLINE floatNx2(const float2 &v)
            :   x(v.x)
            ,   y(v.y)
        {
        }

        MATH_FORCEINLINE floatNx2 &operator=(const floatNx2 &m)
        {
            x = m.x; y = m.y;
            return *this;
        }

        MATH_FORCEINLINE floatNx2 operator+(const floatNx2& m) const
        {
            return floatNx2(x + m.x, y + m.y);
        }

        MATH_FORCEINLINE floatNx2 operator-(const floatNx2& m) const
        {
            return floatNx2(x - m.x, y - m.y);
        }

        MATH_FORCEINLINE floatNx2 operator-(const floatN& v) const
        {
            return floatNx2(x - v, y - v);
        }

        MATH_FORCEINLINE floatNx2 &operator+=(const floatNx2& m)
        {
            x += m.x;
            y += m.y;
            return *this;
        }

        MATH_FORCEINLINE floatNx2 &operator-=(const floatNx2& m)
        {
            x -= m.x;
            y -= m.y;
            return *this;
        }

        MATH_FORCEINLINE floatNx2 &operator+=(const floatN& v)
        {
            x += v;
            y += v;
            return *this;
        }

        MATH_FORCEINLINE floatNx2 &operator-=(const floatN& v)
        {
            x -= v;
            y -= v;
            return *this;
        }

        MATH_FORCEINLINE floatNx2 operator-() const
        {
            return floatNx2(-x, -y);
        }
    };
}
