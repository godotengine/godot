#pragma once

#include "vec-scalar.h"

#if defined(MATH_HAS_NATIVE_SIMD)

namespace math
{
#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int1 &operator+=(int1 &a, int1 b)
    {
        a.p += b.p;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator+=(int1 &a, int x)
    {
        a.p += x;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator-=(int1 &a, int1 b)
    {
        a.p -= b.p;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator-=(int1 &a, int x)
    {
        a.p -= x;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator*=(int1 &a, int1 b)
    {
        a.p *= b.p;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator*=(int1 &a, int x)
    {
        a.p *= x;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator/=(int1 &a, int1 b)
    {
        a.p /= b.p;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator/=(int1 &a, int x)
    {
        a.p /= x;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator%=(int1 &a, int1 b)
    {
        a.p %= b.p;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator%=(int1 &a, int x)
    {
        a.p %= x;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator&=(int1 &a, int1 b)
    {
        a.p &= b.p;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator&=(int1 &a, int x)
    {
        a.p &= x;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator|=(int1 &a, int1 b)
    {
        a.p |= b.p;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator|=(int1 &a, int x)
    {
        a.p |= x;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator^=(int1 &a, int1 b)
    {
        a.p ^= b.p;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator^=(int1 &a, int x)
    {
        a.p ^= x;
        return a;
    }

    static MATH_FORCEINLINE int1 &operator++(int1 &a)
    {
        a.p += 1;
        return a;
    }

    static MATH_FORCEINLINE int1 operator++(int1 &a, int)
    {
        int1 b = a;
        a.p += 1;
        return b;
    }

    static MATH_FORCEINLINE int1 &operator--(int1 &a)
    {
        a.p -= 1;
        return a;
    }

    static MATH_FORCEINLINE int1 operator--(int1 &a, int)
    {
        int1 b = a;
        a.p -= 1;
        return b;
    }

    static MATH_FORCEINLINE int1 operator-(int1 a)
    {
        return int1(-a.p);
    }

    static MATH_FORCEINLINE const int1 &operator+(const int1 &a)
    {
        return a;
    }

    static MATH_FORCEINLINE bool operator!(int1 a)
    {
        return (a.p == 0).x;
    }

    static MATH_FORCEINLINE int1 operator~(int1 a)
    {
        return int1(~a.p);
    }

    static MATH_FORCEINLINE int1 operator+(int1 a, int1 b)
    {
        return int1(a.p + b.p);
    }

    static MATH_FORCEINLINE int1 operator+(int1 a, int b)
    {
        return int1(a.p + b);
    }

    static MATH_FORCEINLINE int1 operator+(int a, int1 b)
    {
        return int1(a + b.p);
    }

    static MATH_FORCEINLINE int4 operator+(int4 a, int1 b)
    {
        return a + b.p;
    }

    static MATH_FORCEINLINE int3 operator+(int3 a, int1 b)
    {
        return a + b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator+(int2 a, int1 b)
    {
        return a + b.p.xy;
    }

    static MATH_FORCEINLINE int4 &operator+=(int4 &a, int1 b)
    {
        return a += b.p;
    }

    static MATH_FORCEINLINE int3 &operator+=(int3 &a, int1 b)
    {
        return a += b.p.xyz;
    }

    static MATH_FORCEINLINE int2 &operator+=(int2 &a, int1 b)
    {
        return a += b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator+(int1 a, int4 b)
    {
        return a.p + b;
    }

    static MATH_FORCEINLINE int3 operator+(int1 a, int3 b)
    {
        return a.p.xyz + b;
    }

    static MATH_FORCEINLINE int2 operator+(int1 a, int2 b)
    {
        return a.p.xy + b;
    }

    static MATH_FORCEINLINE int1 operator-(int1 a, int1 b)
    {
        return int1(a.p - b.p);
    }

    static MATH_FORCEINLINE int1 operator-(int1 a, int b)
    {
        return int1(a.p - b);
    }

    static MATH_FORCEINLINE int1 operator-(int a, int1 b)
    {
        return int1(a - b.p);
    }

    static MATH_FORCEINLINE int4 operator-(int4 a, int1 b)
    {
        return a - b.p;
    }

    static MATH_FORCEINLINE int3 operator-(int3 a, int1 b)
    {
        return a - b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator-(int2 a, int1 b)
    {
        return a - b.p.xy;
    }

    static MATH_FORCEINLINE int4 &operator-=(int4 &a, int1 b)
    {
        return a -= b.p;
    }

    static MATH_FORCEINLINE int3 &operator-=(int3 &a, int1 b)
    {
        return a -= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 &operator-=(int2 &a, int1 b)
    {
        return a -= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator-(int1 a, int4 b)
    {
        return a.p - b;
    }

    static MATH_FORCEINLINE int3 operator-(int1 a, int3 b)
    {
        return a.p.xyz - b;
    }

    static MATH_FORCEINLINE int2 operator-(int1 a, int2 b)
    {
        return a.p.xy - b;
    }

    static MATH_FORCEINLINE int1 operator*(int1 a, int1 b)
    {
        return int1(a.p * b.p);
    }

    static MATH_FORCEINLINE int1 operator*(int1 a, int b)
    {
        return int1(a.p * b);
    }

    static MATH_FORCEINLINE int1 operator*(int a, int1 b)
    {
        return int1(a * b.p);
    }

    static MATH_FORCEINLINE int4 operator*(int4 a, int1 b)
    {
        return a * b.p;
    }

    static MATH_FORCEINLINE int3 operator*(int3 a, int1 b)
    {
        return a * b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator*(int2 a, int1 b)
    {
        return a * b.p.xy;
    }

    static MATH_FORCEINLINE int4 &operator*=(int4 &a, int1 b)
    {
        return a *= b.p;
    }

    static MATH_FORCEINLINE int3 &operator*=(int3 &a, int1 b)
    {
        return a *= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 &operator*=(int2 &a, int1 b)
    {
        return a *= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator*(int1 a, int4 b)
    {
        return a.p * b;
    }

    static MATH_FORCEINLINE int3 operator*(int1 a, int3 b)
    {
        return a.p.xyz * b;
    }

    static MATH_FORCEINLINE int2 operator*(int1 a, int2 b)
    {
        return a.p.xy * b;
    }

    static MATH_FORCEINLINE int1 operator/(int1 a, int1 b)
    {
        return int1(a.p / b.p);
    }

    static MATH_FORCEINLINE int1 operator/(int1 a, int b)
    {
        return int1(a.p / b);
    }

    static MATH_FORCEINLINE int1 operator/(int a, int1 b)
    {
        return int1(a / b.p);
    }

    static MATH_FORCEINLINE int4 operator/(int4 a, int1 b)
    {
        return a / b.p;
    }

    static MATH_FORCEINLINE int3 operator/(int3 a, int1 b)
    {
        return a / b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator/(int2 a, int1 b)
    {
        return a / b.p.xy;
    }

    static MATH_FORCEINLINE int4 &operator/=(int4 &a, int1 b)
    {
        return a /= b.p;
    }

    static MATH_FORCEINLINE int3 &operator/=(int3 &a, int1 b)
    {
        return a /= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 &operator/=(int2 &a, int1 b)
    {
        return a /= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator/(int1 a, int4 b)
    {
        return a.p / b;
    }

    static MATH_FORCEINLINE int3 operator/(int1 a, int3 b)
    {
        return a.p.xyz / b;
    }

    static MATH_FORCEINLINE int2 operator/(int1 a, int2 b)
    {
        return a.p.xy / b;
    }

    static MATH_FORCEINLINE int1 operator%(int1 a, int1 b)
    {
        return int1(a.p % b.p);
    }

    static MATH_FORCEINLINE int1 operator%(int1 a, int b)
    {
        return int1(a.p % b);
    }

    static MATH_FORCEINLINE int1 operator%(int a, int1 b)
    {
        return int1(a % b.p);
    }

    static MATH_FORCEINLINE int4 operator%(int4 a, int1 b)
    {
        return a % b.p;
    }

    static MATH_FORCEINLINE int3 operator%(int3 a, int1 b)
    {
        return a % b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator%(int2 a, int1 b)
    {
        return a % b.p.xy;
    }

    static MATH_FORCEINLINE int4 &operator%=(int4 &a, int1 b)
    {
        return a %= b.p;
    }

    static MATH_FORCEINLINE int3 &operator%=(int3 &a, int1 b)
    {
        return a %= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 &operator%=(int2 &a, int1 b)
    {
        return a %= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator%(int1 a, int4 b)
    {
        return a.p % b;
    }

    static MATH_FORCEINLINE int3 operator%(int1 a, int3 b)
    {
        return a.p.xyz % b;
    }

    static MATH_FORCEINLINE int2 operator%(int1 a, int2 b)
    {
        return a.p.xy % b;
    }

    static MATH_FORCEINLINE int1 operator&(int1 a, int1 b)
    {
        return int1(a.p & b.p);
    }

    static MATH_FORCEINLINE int1 operator&(int1 a, int b)
    {
        return int1(a.p & b);
    }

    static MATH_FORCEINLINE int1 operator&(int a, int1 b)
    {
        return int1(a & b.p);
    }

    static MATH_FORCEINLINE int4 operator&(int4 a, int1 b)
    {
        return a & b.p;
    }

    static MATH_FORCEINLINE int3 operator&(int3 a, int1 b)
    {
        return a & b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator&(int2 a, int1 b)
    {
        return a & b.p.xy;
    }

    static MATH_FORCEINLINE int4 &operator&=(int4 &a, int1 b)
    {
        return a &= b.p;
    }

    static MATH_FORCEINLINE int3 &operator&=(int3 &a, int1 b)
    {
        return a &= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 &operator&=(int2 &a, int1 b)
    {
        return a &= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator&(int1 a, int4 b)
    {
        return a.p & b;
    }

    static MATH_FORCEINLINE int3 operator&(int1 a, int3 b)
    {
        return a.p.xyz & b;
    }

    static MATH_FORCEINLINE int2 operator&(int1 a, int2 b)
    {
        return a.p.xy & b;
    }

    static MATH_FORCEINLINE int1 operator|(int1 a, int1 b)
    {
        return int1(a.p | b.p);
    }

    static MATH_FORCEINLINE int1 operator|(int1 a, int b)
    {
        return int1(a.p | b);
    }

    static MATH_FORCEINLINE int1 operator|(int a, int1 b)
    {
        return int1(a | b.p);
    }

    static MATH_FORCEINLINE int4 operator|(int4 a, int1 b)
    {
        return a | b.p;
    }

    static MATH_FORCEINLINE int3 operator|(int3 a, int1 b)
    {
        return a | b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator|(int2 a, int1 b)
    {
        return a | b.p.xy;
    }

    static MATH_FORCEINLINE int4 &operator|=(int4 &a, int1 b)
    {
        return a |= b.p;
    }

    static MATH_FORCEINLINE int3 &operator|=(int3 &a, int1 b)
    {
        return a |= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 &operator|=(int2 &a, int1 b)
    {
        return a |= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator|(int1 a, int4 b)
    {
        return a.p | b;
    }

    static MATH_FORCEINLINE int3 operator|(int1 a, int3 b)
    {
        return a.p.xyz | b;
    }

    static MATH_FORCEINLINE int2 operator|(int1 a, int2 b)
    {
        return a.p.xy | b;
    }

    static MATH_FORCEINLINE int1 operator^(int1 a, int1 b)
    {
        return int1(a.p ^ b.p);
    }

    static MATH_FORCEINLINE int1 operator^(int1 a, int b)
    {
        return int1(a.p ^ b);
    }

    static MATH_FORCEINLINE int1 operator^(int a, int1 b)
    {
        return int1(a ^ b.p);
    }

    static MATH_FORCEINLINE int4 operator^(int4 a, int1 b)
    {
        return a ^ b.p;
    }

    static MATH_FORCEINLINE int3 operator^(int3 a, int1 b)
    {
        return a ^ b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator^(int2 a, int1 b)
    {
        return a ^ b.p.xy;
    }

    static MATH_FORCEINLINE int4 &operator^=(int4 &a, int1 b)
    {
        return a ^= b.p;
    }

    static MATH_FORCEINLINE int3 &operator^=(int3 &a, int1 b)
    {
        return a ^= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 &operator^=(int2 &a, int1 b)
    {
        return a ^= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator^(int1 a, int4 b)
    {
        return a.p ^ b;
    }

    static MATH_FORCEINLINE int3 operator^(int1 a, int3 b)
    {
        return a.p.xyz ^ b;
    }

    static MATH_FORCEINLINE int2 operator^(int1 a, int2 b)
    {
        return a.p.xy ^ b;
    }

    static MATH_FORCEINLINE int1 operator<=(int1 a, int1 b)
    {
        return int1(a.p <= b.p) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator<=(int1 a, int b)
    {
        return int1(a.p <= b) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator<=(int a, int1 b)
    {
        return int1(a <= b.p) & int1(1);
    }

    static MATH_FORCEINLINE int4 operator<=(int4 a, int1 b)
    {
        return a <= b.p;
    }

    static MATH_FORCEINLINE int3 operator<=(int3 a, int1 b)
    {
        return a <= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator<=(int2 a, int1 b)
    {
        return a <= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator<=(int1 a, int4 b)
    {
        return a.p <= b;
    }

    static MATH_FORCEINLINE int3 operator<=(int1 a, int3 b)
    {
        return a.p.xyz <= b;
    }

    static MATH_FORCEINLINE int2 operator<=(int1 a, int2 b)
    {
        return a.p.xy <= b;
    }

    static MATH_FORCEINLINE int1 operator<(int1 a, int1 b)
    {
        return int1(a.p < b.p) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator<(int1 a, int b)
    {
        return int1(a.p < b) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator<(int a, int1 b)
    {
        return int1(a < b.p) & int1(1);
    }

    static MATH_FORCEINLINE int4 operator<(int4 a, int1 b)
    {
        return a < b.p;
    }

    static MATH_FORCEINLINE int3 operator<(int3 a, int1 b)
    {
        return a < b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator<(int2 a, int1 b)
    {
        return a < b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator<(int1 a, int4 b)
    {
        return a.p < b;
    }

    static MATH_FORCEINLINE int3 operator<(int1 a, int3 b)
    {
        return a.p.xyz < b;
    }

    static MATH_FORCEINLINE int2 operator<(int1 a, int2 b)
    {
        return a.p.xy < b;
    }

    static MATH_FORCEINLINE int1 operator==(int1 a, int1 b)
    {
        return int1(a.p == b.p) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator==(int1 a, int b)
    {
        return int1(a.p == b) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator==(int a, int1 b)
    {
        return int1(a == b.p) & int1(1);
    }

    static MATH_FORCEINLINE int4 operator==(int4 a, int1 b)
    {
        return a == b.p;
    }

    static MATH_FORCEINLINE int3 operator==(int3 a, int1 b)
    {
        return a == b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator==(int2 a, int1 b)
    {
        return a == b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator==(int1 a, int4 b)
    {
        return a.p == b;
    }

    static MATH_FORCEINLINE int3 operator==(int1 a, int3 b)
    {
        return a.p.xyz == b;
    }

    static MATH_FORCEINLINE int2 operator==(int1 a, int2 b)
    {
        return a.p.xy == b;
    }

    static MATH_FORCEINLINE int1 operator!=(int1 a, int1 b)
    {
        return int1(a.p != b.p) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator!=(int1 a, int b)
    {
        return int1(a.p != b) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator!=(int a, int1 b)
    {
        return int1(a <= b.p) & int1(1);
    }

    static MATH_FORCEINLINE int4 operator!=(int4 a, int1 b)
    {
        return a != b.p;
    }

    static MATH_FORCEINLINE int3 operator!=(int3 a, int1 b)
    {
        return a != b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator!=(int2 a, int1 b)
    {
        return a != b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator!=(int1 a, int4 b)
    {
        return a.p != b;
    }

    static MATH_FORCEINLINE int3 operator!=(int1 a, int3 b)
    {
        return a.p.xyz != b;
    }

    static MATH_FORCEINLINE int2 operator!=(int1 a, int2 b)
    {
        return a.p.xy != b;
    }

    static MATH_FORCEINLINE int1 operator>(int1 a, int1 b)
    {
        return int1(a.p > b.p) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator>(int1 a, int b)
    {
        return int1(a.p > b) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator>(int a, int1 b)
    {
        return int1(a > b.p) & int1(1);
    }

    static MATH_FORCEINLINE int4 operator>(int4 a, int1 b)
    {
        return a > b.p;
    }

    static MATH_FORCEINLINE int3 operator>(int3 a, int1 b)
    {
        return a > b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator>(int2 a, int1 b)
    {
        return a > b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator>(int1 a, int4 b)
    {
        return a.p > b;
    }

    static MATH_FORCEINLINE int3 operator>(int1 a, int3 b)
    {
        return a.p.xyz > b;
    }

    static MATH_FORCEINLINE int2 operator>(int1 a, int2 b)
    {
        return a.p.xy > b;
    }

    static MATH_FORCEINLINE int1 operator>=(int1 a, int1 b)
    {
        return int1(a.p >= b.p) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator>=(int1 a, int b)
    {
        return int1(a.p >= b) & int1(1);
    }

    static MATH_FORCEINLINE int1 operator>=(int a, int1 b)
    {
        return int1(a >= b.p) & int1(1);
    }

    static MATH_FORCEINLINE int4 operator>=(int4 a, int1 b)
    {
        return a >= b.p;
    }

    static MATH_FORCEINLINE int3 operator>=(int3 a, int1 b)
    {
        return a >= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator>=(int2 a, int1 b)
    {
        return a >= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator>=(int1 a, int4 b)
    {
        return a.p >= b;
    }

    static MATH_FORCEINLINE int3 operator>=(int1 a, int3 b)
    {
        return a.p.xyz >= b;
    }

    static MATH_FORCEINLINE int2 operator>=(int1 a, int2 b)
    {
        return a.p.xy >= b;
    }

#endif

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float1 &operator+=(float1 &a, float1 b)
    {
        a.p += b.p;
        return a;
    }

    static MATH_FORCEINLINE float1 &operator+=(float1 &a, float x)
    {
        a.p += x;
        return a;
    }

    static MATH_FORCEINLINE float1 &operator-=(float1 &a, float1 b)
    {
        a.p -= b.p;
        return a;
    }

    static MATH_FORCEINLINE float1 &operator-=(float1 &a, float x)
    {
        a.p -= x;
        return a;
    }

    static MATH_FORCEINLINE float1 &operator*=(float1 &a, float1 b)
    {
        a.p *= b.p;
        return a;
    }

    static MATH_FORCEINLINE float1 &operator*=(float1 &a, float x)
    {
        a.p *= x;
        return a;
    }

    static MATH_FORCEINLINE float1 &operator/=(float1 &a, float1 b)
    {
        a.p /= b.p;
        return a;
    }

    static MATH_FORCEINLINE float1 &operator/=(float1 &a, float x)
    {
        a.p /= x;
        return a;
    }

    static MATH_FORCEINLINE float1 &operator++(float1 &a)
    {
        a.p += 1.f;
        return a;
    }

    static MATH_FORCEINLINE float1 operator++(float1 &a, int)
    {
        float1 b = a;
        a.p += 1.f;
        return b;
    }

    static MATH_FORCEINLINE float1 &operator--(float1 &a)
    {
        a.p -= 1.f;
        return a;
    }

    static MATH_FORCEINLINE float1 operator--(float1 &a, int)
    {
        float1 b = a;
        a.p -= 1.f;
        return b;
    }

    static MATH_FORCEINLINE const float1 &operator+(const float1 &a)
    {
        return a;
    }

    static MATH_FORCEINLINE float1 operator-(float1 a)
    {
        return float1(-a.p);
    }

    static MATH_FORCEINLINE bool operator!(float1 a)
    {
        return (a.p == 0.f).x;
    }

    static MATH_FORCEINLINE float1 operator+(float1 a, float1 b)
    {
        return float1(a.p + b.p);
    }

    static MATH_FORCEINLINE float1 operator+(float1 a, float b)
    {
        return float1(a.p + b);
    }

    static MATH_FORCEINLINE float1 operator+(float a, float1 b)
    {
        return float1(a + b.p);
    }

    static MATH_FORCEINLINE float4 operator+(float4 a, float1 b)
    {
        return a + b.p;
    }

    static MATH_FORCEINLINE float3 operator+(float3 a, float1 b)
    {
        return a + b.p.xyz;
    }

    static MATH_FORCEINLINE float2 operator+(float2 a, float1 b)
    {
        return a + b.p.xy;
    }

    static MATH_FORCEINLINE float4 &operator+=(float4 &a, float1 b)
    {
        return a += b.p;
    }

    static MATH_FORCEINLINE float3 &operator+=(float3 &a, float1 b)
    {
        return a += b.p.xyz;
    }

    static MATH_FORCEINLINE float2 &operator+=(float2 &a, float1 b)
    {
        return a += b.p.xy;
    }

    static MATH_FORCEINLINE float4 operator+(float1 a, float4 b)
    {
        return a.p + b;
    }

    static MATH_FORCEINLINE float3 operator+(float1 a, float3 b)
    {
        return a.p.xyz + b;
    }

    static MATH_FORCEINLINE float2 operator+(float1 a, float2 b)
    {
        return a.p.xy + b;
    }

    static MATH_FORCEINLINE float1 operator-(float1 a, float1 b)
    {
        return float1(a.p - b.p);
    }

    static MATH_FORCEINLINE float1 operator-(float1 a, float b)
    {
        return float1(a.p - b);
    }

    static MATH_FORCEINLINE float1 operator-(float a, float1 b)
    {
        return float1(a - b.p);
    }

    static MATH_FORCEINLINE float4 operator-(float4 a, float1 b)
    {
        return a - b.p;
    }

    static MATH_FORCEINLINE float3 operator-(float3 a, float1 b)
    {
        return a - b.p.xyz;
    }

    static MATH_FORCEINLINE float2 operator-(float2 a, float1 b)
    {
        return a - b.p.xy;
    }

    static MATH_FORCEINLINE float4 &operator-=(float4 &a, float1 b)
    {
        return a -= b.p;
    }

    static MATH_FORCEINLINE float3 &operator-=(float3 &a, float1 b)
    {
        return a -= b.p.xyz;
    }

    static MATH_FORCEINLINE float2 &operator-=(float2 &a, float1 b)
    {
        return a -= b.p.xy;
    }

    static MATH_FORCEINLINE float4 operator-(float1 a, float4 b)
    {
        return a.p - b;
    }

    static MATH_FORCEINLINE float3 operator-(float1 a, float3 b)
    {
        return a.p.xyz - b;
    }

    static MATH_FORCEINLINE float2 operator-(float1 a, float2 b)
    {
        return a.p.xy - b;
    }

    static MATH_FORCEINLINE float1 operator*(float1 a, float1 b)
    {
        return float1(a.p * b.p);
    }

    static MATH_FORCEINLINE float1 operator*(float1 a, float b)
    {
        return float1(a.p * b);
    }

    static MATH_FORCEINLINE float1 operator*(float a, float1 b)
    {
        return float1(a * b.p);
    }

    static MATH_FORCEINLINE float4 operator*(float4 a, float1 b)
    {
        return a * b.p;
    }

    static MATH_FORCEINLINE float3 operator*(float3 a, float1 b)
    {
        return a * b.p.xyz;
    }

    static MATH_FORCEINLINE float2 operator*(float2 a, float1 b)
    {
        return a * b.p.xy;
    }

    static MATH_FORCEINLINE float4 &operator*=(float4 &a, float1 b)
    {
        return a *= b.p;
    }

    static MATH_FORCEINLINE float3 &operator*=(float3 &a, float1 b)
    {
        return a *= b.p.xyz;
    }

    static MATH_FORCEINLINE float2 &operator*=(float2 &a, float1 b)
    {
        return a *= b.p.xy;
    }

    static MATH_FORCEINLINE float4 operator*(float1 a, float4 b)
    {
        return a.p * b;
    }

    static MATH_FORCEINLINE float3 operator*(float1 a, float3 b)
    {
        return a.p.xyz * b;
    }

    static MATH_FORCEINLINE float2 operator*(float1 a, float2 b)
    {
        return a.p.xy * b;
    }

    static MATH_FORCEINLINE float1 operator/(float1 a, float1 b)
    {
        return float1(a.p / b.p);
    }

    static MATH_FORCEINLINE float1 operator/(float1 a, float b)
    {
        return float1(a.p / b);
    }

    static MATH_FORCEINLINE float1 operator/(float a, float1 b)
    {
        return float1(a / b.p);
    }

    static MATH_FORCEINLINE float4 operator/(float4 a, float1 b)
    {
        return a / b.p;
    }

    static MATH_FORCEINLINE float3 operator/(float3 a, float1 b)
    {
        return a / b.p.xyz;
    }

    static MATH_FORCEINLINE float2 operator/(float2 a, float1 b)
    {
        return a / b.p.xy;
    }

    static MATH_FORCEINLINE float4 &operator/=(float4 &a, float1 b)
    {
        return a /= b.p;
    }

    static MATH_FORCEINLINE float3 &operator/=(float3 &a, float1 b)
    {
        return a /= b.p.xyz;
    }

    static MATH_FORCEINLINE float2 &operator/=(float2 &a, float1 b)
    {
        return a /= b.p.xy;
    }

    static MATH_FORCEINLINE float4 operator/(float1 a, float4 b)
    {
        return a.p / b;
    }

    static MATH_FORCEINLINE float3 operator/(float1 a, float3 b)
    {
        return a.p.xyz / b;
    }

    static MATH_FORCEINLINE float2 operator/(float1 a, float2 b)
    {
        return a.p.xy / b;
    }

    static MATH_FORCEINLINE int1 operator<=(float1 a, float1 b)
    {
        return int1((a.p <= b.p) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator<=(float1 a, float b)
    {
        return int1((a.p <= b) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator<=(float a, float1 b)
    {
        return int1((a <= b.p) & int1(1));
    }

    static MATH_FORCEINLINE int4 operator<=(float4 a, float1 b)
    {
        return a <= b.p;
    }

    static MATH_FORCEINLINE int3 operator<=(float3 a, float1 b)
    {
        return a <= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator<=(float2 a, float1 b)
    {
        return a <= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator<=(float1 a, float4 b)
    {
        return a.p <= b;
    }

    static MATH_FORCEINLINE int3 operator<=(float1 a, float3 b)
    {
        return a.p.xyz <= b;
    }

    static MATH_FORCEINLINE int2 operator<=(float1 a, float2 b)
    {
        return a.p.xy <= b;
    }

    static MATH_FORCEINLINE int1 operator<(float1 a, float1 b)
    {
        return int1((a.p < b.p) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator<(float1 a, float b)
    {
        return int1((a.p < b) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator<(float a, float1 b)
    {
        return int1((a < b.p) & int1(1));
    }

    static MATH_FORCEINLINE int4 operator<(float4 a, float1 b)
    {
        return a < b.p;
    }

    static MATH_FORCEINLINE int3 operator<(float3 a, float1 b)
    {
        return a < b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator<(float2 a, float1 b)
    {
        return a < b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator<(float1 a, float4 b)
    {
        return a.p < b;
    }

    static MATH_FORCEINLINE int3 operator<(float1 a, float3 b)
    {
        return a.p.xyz < b;
    }

    static MATH_FORCEINLINE int2 operator<(float1 a, float2 b)
    {
        return a.p.xy < b;
    }

    static MATH_FORCEINLINE int1 operator==(float1 a, float1 b)
    {
        return int1((a.p == b.p) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator==(float1 a, float b)
    {
        return int1((a.p == b) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator==(float a, float1 b)
    {
        return int1((a == b.p) & int1(1));
    }

    static MATH_FORCEINLINE int4 operator==(float4 a, float1 b)
    {
        return a == b.p;
    }

    static MATH_FORCEINLINE int3 operator==(float3 a, float1 b)
    {
        return a == b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator==(float2 a, float1 b)
    {
        return a == b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator==(float1 a, float4 b)
    {
        return a.p == b;
    }

    static MATH_FORCEINLINE int3 operator==(float1 a, float3 b)
    {
        return a.p.xyz == b;
    }

    static MATH_FORCEINLINE int2 operator==(float1 a, float2 b)
    {
        return a.p.xy == b;
    }

    static MATH_FORCEINLINE int1 operator!=(float1 a, float1 b)
    {
        return int1((a.p != b.p) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator!=(float1 a, float b)
    {
        return int1((a.p != b) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator!=(float a, float1 b)
    {
        return int1((a <= b.p) & int1(1));
    }

    static MATH_FORCEINLINE int4 operator!=(float4 a, float1 b)
    {
        return a != b.p;
    }

    static MATH_FORCEINLINE int3 operator!=(float3 a, float1 b)
    {
        return a != b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator!=(float2 a, float1 b)
    {
        return a != b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator!=(float1 a, float4 b)
    {
        return a.p != b;
    }

    static MATH_FORCEINLINE int3 operator!=(float1 a, float3 b)
    {
        return a.p.xyz != b;
    }

    static MATH_FORCEINLINE int2 operator!=(float1 a, float2 b)
    {
        return a.p.xy != b;
    }

    static MATH_FORCEINLINE int1 operator>(float1 a, float1 b)
    {
        return int1((a.p > b.p) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator>(float1 a, float b)
    {
        return int1((a.p > b) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator>(float a, float1 b)
    {
        return int1((a > b.p) & int1(1));
    }

    static MATH_FORCEINLINE int4 operator>(float4 a, float1 b)
    {
        return a > b.p;
    }

    static MATH_FORCEINLINE int3 operator>(float3 a, float1 b)
    {
        return a > b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator>(float2 a, float1 b)
    {
        return a > b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator>(float1 a, float4 b)
    {
        return a.p > b;
    }

    static MATH_FORCEINLINE int3 operator>(float1 a, float3 b)
    {
        return a.p.xyz > b;
    }

    static MATH_FORCEINLINE int2 operator>(float1 a, float2 b)
    {
        return a.p.xy > b;
    }

    static MATH_FORCEINLINE int1 operator>=(float1 a, float1 b)
    {
        return int1((a.p >= b.p) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator>=(float1 a, float b)
    {
        return int1((a.p >= b) & int1(1));
    }

    static MATH_FORCEINLINE int1 operator>=(float a, float1 b)
    {
        return int1((a >= b.p) & int1(1));
    }

    static MATH_FORCEINLINE int4 operator>=(float4 a, float1 b)
    {
        return a >= b.p;
    }

    static MATH_FORCEINLINE int3 operator>=(float3 a, float1 b)
    {
        return a >= b.p.xyz;
    }

    static MATH_FORCEINLINE int2 operator>=(float2 a, float1 b)
    {
        return a >= b.p.xy;
    }

    static MATH_FORCEINLINE int4 operator>=(float1 a, float4 b)
    {
        return a.p >= b;
    }

    static MATH_FORCEINLINE int3 operator>=(float1 a, float3 b)
    {
        return a.p.xyz >= b;
    }

    static MATH_FORCEINLINE int2 operator>=(float1 a, float2 b)
    {
        return a.p.xy >= b;
    }

#endif

    //
    //  mad: multiply add
    //  return value: a*b + c
    //  note: NEON floating-point vmla has a large latency and should be avoided
    //
    static MATH_FORCEINLINE float4 mad(float4 a, float4 b, float4 c)
    {
#   if defined(__ARM_NEON) && defined(MATH_HAS_FAST_MADD)
        return vmlaq_f32((float32x4_t)c, (float32x4_t)a, (float32x4_t)b);
#   elif defined(__FMA__)
        return _mm_fmadd_ps((__m128)a, (__m128)b, (__m128)c);
#   else
        return a * b + c;
#   endif
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float3 mad(const float3 &a, const float3 &b, const float3 &c)
    {
        return mad(float4(as_native(a)), float4(as_native(b)), float4(as_native(c))).xyz;
    }

    static MATH_FORCEINLINE float2 mad(const float2 &a, const float2 &b, const float2 &c)
    {
        return mad(float4(as_native(a)), float4(as_native(b)), float4(as_native(c))).xy;
    }

    static MATH_FORCEINLINE float1 mad(float1 a, float1 b, float1 c)
    {
        return float1((float1::packed)mad(float4((float1::packed)a), float4((float1::packed)b), float4((float1::packed)c)));
    }

#else

    static MATH_FORCEINLINE float3 mad(float3 a, float3 b, float3 c)
    {
        return a * b + c;
    }

#endif

    /*
     *  msub
     */
    static MATH_FORCEINLINE float4 msub(float4 a, float4 b, float4 c)
    {
#   if defined(__ARM_NEON) && defined(MATH_HAS_FAST_MADD)
        return vmlsq_f32((float32x4_t)c, (float32x4_t)a, (float32x4_t)b);
#   elif defined(__FMA__)
        return _mm_fmsub_ps((__m128)a, (__m128)b, (__m128)c);
#   else
        return a * b - c;
#   endif
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float3 msub(float3 a, float3 b, float3 c)
    {
        return msub(float4(as_native(a)), float4(as_native(b)), float4(as_native(c))).xyz;
    }

    static MATH_FORCEINLINE float2 msub(float2 a, float2 b, float2 c)
    {
        return msub(float4(as_native(a)), float4(as_native(b)), float4(as_native(c))).xy;
    }

    static MATH_FORCEINLINE float1 msub(float1 a, float1 b, float1 c)
    {
        return float1((float1::packed)msub(float4((float1::packed)a), float4((float1::packed)b), float4((float1::packed)c)));
    }

#else

    static MATH_FORCEINLINE float3 msub(float3 a, float3 b, float3 c)
    {
        return a * b - c;
    }

#endif

    //
    //  all: all non-zero
    //
    static MATH_FORCEINLINE int all(const int4 &p)
    {
#   if defined(__ARM_NEON)
        int32x2_t r = vpmax_s32(vget_low_s32((int32x4_t)p), vget_high_s32((int32x4_t)p));
        return vget_lane_s32(vpmax_s32(r, r), 0) < 0;
#   elif defined(__SSE__)
        return _mm_movemask_ps((__m128)(__m128i)p) == 0xf;
#   else
        return (p.x & p.y & p.z & p.w) < 0;
#   endif
    }

    static MATH_FORCEINLINE int all(const int3 &p)
    {
#   if defined(__ARM_NEON)
        int32x2_t r = vpmax_s32(vget_low_s32(as_native(p)), vget_low_s32(as_native(p)));
        return vget_lane_s32(vmax_s32(vget_high_s32((as_native(p))), r), 0) < 0;
#   elif defined(__SSE__)
        return (_mm_movemask_ps(as_native(p)) & 0x7) == 0x7;
#   else
        return (p.x & p.y & p.z) < 0;
#   endif
    }

    static MATH_FORCEINLINE int all(const int2 &p)
    {
#   if defined(__ARM_NEON)
        int32x2_t r = vpmax_s32(vget_low_s32(as_native(p)), vget_low_s32(as_native(p)));
        return vget_lane_s32(r, 0) < 0;
#   elif defined(__SSE__)
        return (_mm_movemask_ps(as_native(p)) & 0x3) == 0x3;
#   else
        return (p.x & p.y) < 0;
#   endif
    }

    //
    //  any: any non-zero
    //
    static MATH_FORCEINLINE int any(const int4 &p)
    {
#   if defined(__ARM_NEON)
        int32x2_t r = vpmin_s32(vget_low_s32((int32x4_t)p), vget_high_s32((int32x4_t)p));
        return vget_lane_s32(vpmin_s32(r, r), 0) < 0;
#   elif defined(__SSE__)
        return _mm_movemask_ps((__m128)(__m128i)p) != 0;
#   else
        return (p.x | p.y | p.z | p.w) < 0;
#   endif
    }

    static MATH_FORCEINLINE int any(const int3 &p)
    {
#   if defined(__ARM_NEON)
        int32x2_t r = vpmin_s32(vget_low_s32(as_native(p)), vget_low_s32(as_native(p)));
        return vget_lane_s32(vmin_s32(vget_high_s32(as_native(p)), r), 0) < 0;
#   elif defined(__SSE__)
        return (_mm_movemask_ps(as_native(p)) & 0x7) != 0;
#   else
        return (p.x | p.y | p.z) < 0;
#   endif
    }

    static MATH_FORCEINLINE int any(const int2 &p)
    {
#   if defined(__ARM_NEON)
        int32x2_t r = vpmin_s32(vget_low_s32(as_native(p)), vget_low_s32(as_native(p)));
        return vget_lane_s32(r, 0) < 0;
#   elif defined(__SSE__)
        return (_mm_movemask_ps(as_native(p)) & 0x3) != 0;
#   else
        return (p.x | p.y) < 0;
#   endif
    }

    static MATH_FORCEINLINE int4 abs(const int4 &x)
    {
#   if defined(__ARM_NEON)
        return vabsq_s32((int32x4_t)x);
#   elif defined(__SSSE3__)
        return _mm_abs_epi32((__m128i)x);
#   elif defined(__SSE__)
        __m128i c = _mm_srai_epi32((__m128i)x, 31);
        return _mm_sub_epi32(_mm_xor_si128((__m128i)x, c), c);
#   else
        return int4(abs(x.x), abs(x.y), abs(x.z), abs(x.w));
#   endif
    }

#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int3 abs(const int3 &x)
    {
        return abs(int4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE int2 abs(const int2 &x)
    {
        return abs(int4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE int1 abs(const int1 &x)
    {
        return int1((int1::packed)abs(int4((int1::packed)x)));
    }

#else

    static MATH_FORCEINLINE int3 abs(const int3 &x)
    {
        return int3(abs(x.x), abs(x.y), abs(x.z));
    }

    static MATH_FORCEINLINE int2 abs(const int2 &x)
    {
        return int3(abs(x.x), abs(x.y));
    }

#endif

    static MATH_FORCEINLINE float4 abs(const float4 &x)
    {
#   if defined(__ARM_NEON)
        return vabsq_f32((float32x4_t)x);
#   elif defined(__SSE__)
        return _mm_and_ps((__m128)x, _mm_castsi128_ps(cv4i(0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff)));
#   else
        return float4(abs(x.x), abs(x.y), abs(x.z), abs(x.w));
#   endif
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float3 abs(const float3 &x)
    {
        return abs(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 abs(const float2 &x)
    {
        return abs(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 abs(const float1 &x)
    {
        return float1((float1::packed)abs(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float3 abs(const float3 &x)
    {
        return float3(abs(x.x), abs(x.y), abs(x.z));
    }

    static MATH_FORCEINLINE float2 abs(const float2 &x)
    {
        return float2(abs(x.x), abs(x.y));
    }

#endif

    static MATH_FORCEINLINE int4 min(const int4 &x, const int4 &y)
    {
#   if defined(__ARM_NEON)
        return vminq_s32((int32x4_t)x, (int32x4_t)y);
#   elif defined(__SSE4_1__)
        return _mm_min_epi32((__m128i)x, (__m128i)y);
#   else
        return y - ((x < y) & (y - x));
#   endif
    }

#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int3 min(const int3 &x, const int3 &y)
    {
        return min(int4(as_native(x)), int4(as_native(y))).xyz;
    }

    static MATH_FORCEINLINE int2 min(const int2 &x, const int2 &y)
    {
        return min(int4(as_native(x)), int4(as_native(y))).xy;
    }

    static MATH_FORCEINLINE int1 min(const int1 &x, const int1 &y)
    {
        return int1((int1::packed)min(int4((int1::packed)x), int4((int1::packed)y)));
    }

#else

    static MATH_FORCEINLINE int3 min(const int3 &x, const int3 &y)
    {
        return y - ((x < y) & (y - x));
    }

    static MATH_FORCEINLINE int2 min(const int2 &x, const int2 &y)
    {
        return y - ((x < y) & (y - x));
    }

#endif

    static MATH_FORCEINLINE float4 min(const float4 &x, const float4 &y)
    {
#   if defined(__ARM_NEON)
        return vminq_f32((float32x4_t)x, (float32x4_t)y);
#   elif defined(__SSE__)
        return _mm_min_ps((__m128)x, (__m128)y);
#   else
        return float4(x.x < y.x ? x.x : y.x, x.y < y.y ? x.y : y.y, x.z < y.z ? x.z : y.z, x.w < y.w ? x.w : y.w);
#   endif
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float3 min(const float3 &x, const float3 &y)
    {
        return min(float4(as_native(x)), float4(as_native(y))).xyz;
    }

    static MATH_FORCEINLINE float2 min(const float2 &x, const float2 &y)
    {
        return min(float4(as_native(x)), float4(as_native(y))).xy;
    }

    static MATH_FORCEINLINE float1 min(const float1 &x, const float1 &y)
    {
        return float1((float1::packed)min(float4((float1::packed)x), float4((float1::packed)y)));
    }

#else

    static MATH_FORCEINLINE float3 min(const float3 &x, const float3 &y)
    {
        return float3(x.x < y.x ? x.x : y.x, x.y < y.y ? x.y : y.y, x.z < y.z ? x.z : y.z);
    }

    static MATH_FORCEINLINE float2 min(const float2 &x, const float2 &y)
    {
        return float3(x.x < y.x ? x.x : y.x, x.y < y.y ? x.y : y.y);
    }

#endif

    static MATH_FORCEINLINE int4 max(const int4 &x, const int4 &y)
    {
#   if defined(__ARM_NEON)
        return vmaxq_s32((int32x4_t)x, (int32x4_t)y);
#   elif defined(__SSE4_1__)
        return _mm_max_epi32((__m128i)x, (__m128i)y);
#   else
        return x + ((x < y) & (y - x));
#   endif
    }

#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int3 max(const int3 &x, const int3 &y)
    {
        return max(int4(as_native(x)), int4(as_native(y))).xyz;
    }

    static MATH_FORCEINLINE int2 max(const int2 &x, const int2 &y)
    {
        return max(int4(as_native(x)), int4(as_native(y))).xy;
    }

    static MATH_FORCEINLINE int1 max(const int1 &x, const int1 &y)
    {
        return int1((int1::packed)max(int4((int1::packed)x), int4((int1::packed)y)));
    }

#else

    static MATH_FORCEINLINE int3 max(const int3 &x, const int3 &y)
    {
        return x + ((x < y) & (y - x));
    }

    static MATH_FORCEINLINE int2 max(const int2 &x, const int2 &y)
    {
        return x + ((x < y) & (y - x));
    }

#endif

    static MATH_FORCEINLINE float4 max(const float4 &x, const float4 &y)
    {
#   if defined(__ARM_NEON)
        return vmaxq_f32((float32x4_t)x, (float32x4_t)y);
#   elif defined(__SSE__)
        return _mm_max_ps((__m128)x, (__m128)y);
#   else
        return float4(x.x > y.x ? x.x : y.x, x.y > y.y ? x.y : y.y, x.z > y.z ? x.z : y.z, x.w > y.w ? x.w : y.w);
#   endif
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float3 max(const float3 &x, const float3 &y)
    {
        return max(float4(as_native(x)), float4(as_native(y))).xyz;
    }

    static MATH_FORCEINLINE float2 max(const float2 &x, const float2 &y)
    {
        return max(float4(as_native(x)), float4(as_native(y))).xy;
    }

    static MATH_FORCEINLINE float1 max(const float1 &x, const float1 &y)
    {
        return float1((float1::packed)max(float4((float1::packed)x), float4((float1::packed)y)));
    }

#else

    static MATH_FORCEINLINE float3 max(const float3 &x, const float3 &y)
    {
        return float3(x.x > y.x ? x.x : y.x, x.y > y.y ? x.y : y.y, x.z > y.z ? x.z : y.z);
    }

    static MATH_FORCEINLINE float2 max(const float2 &x, const float2 &y)
    {
        return float2(x.x > y.x ? x.x : y.x, x.y > y.y ? x.y : y.y);
    }

#endif
}

#elif defined(MATH_HAS_SIMD_FLOAT)

namespace math
{
namespace meta
{
#   define SPECIFIC1_T(T, OP, RHS, N)                   v<typename T::template OP<RHS>::type, sp<typename T::template OP<RHS>::type, T::template OP<RHS>::S, N>, N>
#   define SPECIFIC1_F(T, OP, RHS, rhs, N)              SPECIFIC1_T(T, OP, RHS, N)::pack(T::template OP<RHS>::f(rhs))
#   define SPECIFIC1I_F(T, OP, RHS, lhs, rhs, N)        SPECIFIC1_T(T, OP, RHS, N)::pack(T::template OP<RHS>::f(lhs, rhs))

#   define SPECIFIC2_T(T, OP, LHS, RHS, N)              v<typename T::template OP<LHS, RHS>::type, sp<typename T::template OP<LHS, RHS>::type, T::template OP<LHS, RHS>::S, N>, N>
#   define SPECIFIC2_F(T, OP, LHS, lhs, RHS, rhs, N)    SPECIFIC2_T(T, OP, LHS, RHS, N)::pack(T::template OP<LHS, RHS>::f(lhs, rhs))
#   define SPECIFIC2_FB(T, OP, LHS, lhs, RHS, rhs, N)   SPECIFIC2_T(T, OP, LHS, RHS, N)::pack(T::template OP<LHS, RHS>::type::BOOL(T::template OP<LHS, RHS>::f(lhs, rhs)))

    /*
     *  unary plus
     */
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE const v<T, RHS, N> &operator+(const v<T, RHS, N> &rhs)
    {
        return rhs;
    }

    /*
     *  unary minus
     */
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC1_T(T, NEG, RHS::SWZ, N) operator-(const v<T, RHS, N> &rhs)
    {
        return SPECIFIC1_F(T, NEG, RHS::SWZ, rhs.p, N);
    }

    /*
     *  unary complement
     */
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC1_T(T, NOT, RHS::SWZ, N) operator~(const v<T, RHS, N> &rhs)
    {
        return SPECIFIC1_F(T, NOT, RHS::SWZ, rhs.p, N);
    }

    /*
     *  binary add
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, ADD, LHS::SWZ, RHS::SWZ, 1) operator+(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, ADD, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, ADD, BC(LHS::SWZ, N), RHS::SWZ, N) operator+(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, ADD, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, ADD, LHS::SWZ, BC(RHS::SWZ, N), N) operator+(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, ADD, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, ADD, LHS::SWZ, RHS::SWZ, N) operator+(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, ADD, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, ADD, LHS::SWZ, SWZ_ANY, N) operator+(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, ADD, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, ADD, SWZ_ANY, RHS::SWZ, N) operator+(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, ADD, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator+=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, ADD, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator+=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, ADD, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator+=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, ADD, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator+=(vec_attr lv<T, LHS, N> &lhs, const typename T::type rhs)
    {
        lhs = SPECIFIC2_F(T, ADD, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
        return lhs;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, RHS, 1> &operator++(vec_attr lv<T, RHS, 1> &rhs)
    {
        enum
        {
            U = USED(RHS::SWZ) & SWZ_XYZW,
            X = (U & MSK_X) != 0,
            Y = (U & MSK_Y) != 0,
            Z = (U & MSK_Z) != 0,
            W = (U & MSK_W) != 0
        };
        rhs.p = T::template ADD<U, U>::f(rhs.p, T::CTOR(X, Y, Z, W));
        return rhs;
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE v<T, sp<T, LHS::SWZ, 1>, 1> operator++(vec_attr lv<T, LHS, 1> &lhs, int)
    {
        enum
        {
            U = USED(LHS::SWZ) & SWZ_XYZW,
            X = (U & MSK_X) != 0,
            Y = (U & MSK_Y) != 0,
            Z = (U & MSK_Z) != 0,
            W = (U & MSK_W) != 0
        };
        typename T::packed p = lhs.p;
        lhs.p = T::template ADD<U, U>::f(lhs.p, T::CTOR(X, Y, Z, W));
        return v<T, sp<T, LHS::SWZ, 1>, 1>::pack(p);
    }

    /*
     *  binary subtract
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, SUB, LHS::SWZ, RHS::SWZ, 1) operator-(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, SUB, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SUB, BC(LHS::SWZ, N), RHS::SWZ, N) operator-(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SUB, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SUB, LHS::SWZ, BC(RHS::SWZ, N), N) operator-(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, SUB, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SUB, LHS::SWZ, RHS::SWZ, N) operator-(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SUB, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SUB, LHS::SWZ, SWZ_ANY, N) operator-(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, SUB, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SUB, SWZ_ANY, RHS::SWZ, N) operator-(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SUB, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator-=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, SUB, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator-=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, SUB, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator-=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, SUB, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator-=(vec_attr lv<T, LHS, N> &lhs, const typename T::type rhs)
    {
        lhs = SPECIFIC2_F(T, SUB, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
        return lhs;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, RHS, 1> &operator--(vec_attr lv<T, RHS, 1> &rhs)
    {
        enum
        {
            U = USED(RHS::SWZ) & SWZ_XYZW,
            X = (U & MSK_X) != 0,
            Y = (U & MSK_Y) != 0,
            Z = (U & MSK_Z) != 0,
            W = (U & MSK_W) != 0
        };
        rhs.p = T::template SUB<U, U>::f(rhs.p, T::CTOR(X, Y, Z, W));
        return rhs;
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE v<T, sp<T, LHS::SWZ, 1>, 1> operator--(vec_attr lv<T, LHS, 1> &lhs, int)
    {
        enum
        {
            U = USED(LHS::SWZ) & SWZ_XYZW,
            X = (U & MSK_X) != 0,
            Y = (U & MSK_Y) != 0,
            Z = (U & MSK_Z) != 0,
            W = (U & MSK_W) != 0
        };
        typename T::packed p = lhs.p;
        lhs.p = T::template SUB<U, U>::f(lhs.p, T::CTOR(X, Y, Z, W));
        return v<T, sp<T, LHS::SWZ, 1>, 1>::pack(p);
    }

    /*
     *  binary multiply
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, MUL, LHS::SWZ, RHS::SWZ, 1) operator*(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, MUL, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MUL, BC(LHS::SWZ, N), RHS::SWZ, N) operator*(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MUL, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MUL, LHS::SWZ, BC(RHS::SWZ, N), N) operator*(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, MUL, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MUL, LHS::SWZ, RHS::SWZ, N) operator*(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MUL, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MUL, LHS::SWZ, SWZ_ANY, N) operator*(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, MUL, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MUL, SWZ_ANY, RHS::SWZ, N) operator*(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MUL, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator*=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, MUL, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator*=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, MUL, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator*=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, MUL, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator*=(vec_attr lv<T, LHS, N> &lhs, const typename T::type rhs)
    {
        lhs = SPECIFIC2_F(T, MUL, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
        return lhs;
    }

    /*
     *  binary divide
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, DIV, LHS::SWZ, RHS::SWZ, 1) operator/(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, DIV, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, DIV, BC(LHS::SWZ, N), RHS::SWZ, N) operator/(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, DIV, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, DIV, LHS::SWZ, BC(RHS::SWZ, N), N) operator/(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, DIV, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, DIV, LHS::SWZ, RHS::SWZ, N) operator/(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, DIV, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, DIV, LHS::SWZ, SWZ_ANY, N) operator/(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, DIV, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, DIV, SWZ_ANY, RHS::SWZ, N) operator/(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, DIV, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator/=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, DIV, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator/=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, DIV, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator/=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, DIV, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator/=(vec_attr lv<T, LHS, N> &lhs, const typename T::type rhs)
    {
        lhs = SPECIFIC2_F(T, DIV, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
        return lhs;
    }

    /*
     *  binary modulus
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, LHS::SWZ, RHS::SWZ, 1) operator%(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, BC(LHS::SWZ, N), RHS::SWZ, N) operator%(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, AND, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, LHS::SWZ, BC(RHS::SWZ, N), N) operator%(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, LHS::SWZ, RHS::SWZ, N) operator%(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, LHS::SWZ, SWZ_ANY, N) operator%(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, SWZ_ANY, RHS::SWZ, N) operator%(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, AND, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator%=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator%=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator%=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator%=(vec_attr lv<T, LHS, N> &lhs, const typename T::type rhs)
    {
        lhs = SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
        return lhs;
    }

    /*
     *  binary and
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, LHS::SWZ, RHS::SWZ, 1) operator&(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, BC(LHS::SWZ, N), RHS::SWZ, N) operator&(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, AND, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, LHS::SWZ, BC(RHS::SWZ, N), N) operator&(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, LHS::SWZ, RHS::SWZ, N) operator&(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, LHS::SWZ, SWZ_ANY, N) operator&(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, AND, SWZ_ANY, RHS::SWZ, N) operator&(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, AND, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator&=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator&=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator&=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator&=(vec_attr lv<T, LHS, N> &lhs, const typename T::type rhs)
    {
        lhs = SPECIFIC2_F(T, AND, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
        return lhs;
    }

    /*
     *  binary or
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, OR, LHS::SWZ, RHS::SWZ, 1) operator|(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, OR, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, OR, BC(LHS::SWZ, N), RHS::SWZ, N) operator|(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, OR, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, OR, LHS::SWZ, BC(RHS::SWZ, N), N) operator|(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, OR, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, OR, LHS::SWZ, RHS::SWZ, N) operator|(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, OR, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, OR, LHS::SWZ, SWZ_ANY, N) operator|(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, OR, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, OR, SWZ_ANY, RHS::SWZ, N) operator|(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, OR, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator|=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, OR, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator|=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, OR, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator|=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, OR, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator|=(vec_attr lv<T, LHS, N> &lhs, const typename T::type rhs)
    {
        lhs = SPECIFIC2_F(T, OR, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
        return lhs;
    }

    /*
     *  binary xor
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, XOR, LHS::SWZ, RHS::SWZ, 1) operator^(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, XOR, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, XOR, BC(LHS::SWZ, N), RHS::SWZ, N) operator^(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, XOR, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, XOR, LHS::SWZ, BC(RHS::SWZ, N), N) operator^(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, XOR, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, XOR, LHS::SWZ, RHS::SWZ, N) operator^(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, XOR, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, XOR, LHS::SWZ, SWZ_ANY, N) operator^(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, XOR, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, XOR, SWZ_ANY, RHS::SWZ, N) operator^(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, XOR, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator^=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, XOR, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator^=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, XOR, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator^=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, XOR, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator^=(vec_attr lv<T, LHS, N> &lhs, const typename T::type rhs)
    {
        lhs = SPECIFIC2_F(T, XOR, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
        return lhs;
    }

    /*
     *  binary shift left logical
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, SLL, LHS::SWZ, RHS::SWZ, 1) operator<<(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, SLL, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SLL, BC(LHS::SWZ, N), RHS::SWZ, N) operator<<(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SLL, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SLL, LHS::SWZ, BC(RHS::SWZ, N), N) operator<<(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, SLL, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SLL, LHS::SWZ, RHS::SWZ, N) operator<<(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SLL, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC1_T(T, SLLI, LHS::SWZ, N) operator<<(const v<T, LHS, N> &lhs, int rhs)
    {
        return SPECIFIC1I_F(T, SLLI, LHS::SWZ, lhs.p, rhs, N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SLL, SWZ_ANY, RHS::SWZ, N) operator<<(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SLL, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator<<=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, SLL, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator<<=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, SLL, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator<<=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, SLL, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator<<=(vec_attr lv<T, LHS, N> &lhs, int rhs)
    {
        lhs = SPECIFIC1I_F(T, SLLI, LHS::SWZ, lhs.p, rhs, N);
        return lhs;
    }

    /*
     *  binary shift right arithmetical
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, SRA, LHS::SWZ, RHS::SWZ, 1) operator>>(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, SRA, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SRA, BC(LHS::SWZ, N), RHS::SWZ, N) operator>>(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SRA, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SRA, LHS::SWZ, BC(RHS::SWZ, N), N) operator>>(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, SRA, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SRA, LHS::SWZ, RHS::SWZ, N) operator>>(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SRA, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC1_T(T, SRAI, LHS::SWZ, N) operator>>(const v<T, LHS, N> &lhs, int rhs)
    {
        return SPECIFIC1I_F(T, SRAI, LHS::SWZ, lhs.p, rhs, N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, SRA, SWZ_ANY, RHS::SWZ, N) operator>>(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, SRA, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 1> &operator>>=(vec_attr lv<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, SRA, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator>>=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        lhs = SPECIFIC2_F(T, SRA, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator>>=(vec_attr lv<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        lhs = SPECIFIC2_F(T, SRA, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
        return lhs;
    }

    template<typename T, typename LHS, int N> static MATH_FORCEINLINE vec_attr lv<T, LHS, N> &operator>>=(vec_attr lv<T, LHS, N> &lhs, int rhs)
    {
        lhs = SPECIFIC1I_F(T, SRAI, LHS::SWZ, lhs.p, rhs, N);
        return lhs;
    }

    /*
     * binary compare equal
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPEQ, LHS::SWZ, RHS::SWZ, 1) operator==(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPEQ, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPEQ, BC(LHS::SWZ, N), RHS::SWZ, N) operator==(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPEQ, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPEQ, LHS::SWZ, BC(RHS::SWZ, N), N) operator==(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, CMPEQ, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPEQ, LHS::SWZ, RHS::SWZ, N) operator==(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPEQ, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPEQ, LHS::SWZ, SWZ_ANY, N) operator==(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, CMPEQ, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPEQ, SWZ_ANY, RHS::SWZ, N) operator==(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPEQ, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPEQ, LHS::SWZ, SWZ_ANY, 1) operator==(const v<T, LHS, 1> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_FB(T, CMPEQ, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), 1);
    }
    template<typename T, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPEQ, SWZ_ANY, RHS::SWZ, 1) operator==(const typename T::type lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPEQ, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, 1);
    }

    /*
     * binary compare not equal
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPNE, LHS::SWZ, RHS::SWZ, 1) operator!=(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPNE, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPNE, BC(LHS::SWZ, N), RHS::SWZ, N) operator!=(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPNE, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPNE, LHS::SWZ, BC(RHS::SWZ, N), N) operator!=(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, CMPNE, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPNE, LHS::SWZ, RHS::SWZ, N) operator!=(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPNE, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPNE, LHS::SWZ, SWZ_ANY, N) operator!=(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, CMPNE, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPNE, SWZ_ANY, RHS::SWZ, N) operator!=(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPNE, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPNE, LHS::SWZ, SWZ_ANY, 1) operator!=(const v<T, LHS, 1> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_FB(T, CMPNE, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), 1);
    }
    template<typename T, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPNE, SWZ_ANY, RHS::SWZ, 1) operator!=(const typename T::type lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPNE, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, 1);
    }

    /*
     * binary compare less than or equal
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLE, LHS::SWZ, RHS::SWZ, 1) operator<=(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPLE, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLE, BC(LHS::SWZ, N), RHS::SWZ, N) operator<=(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPLE, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLE, LHS::SWZ, BC(RHS::SWZ, N), N) operator<=(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, CMPLE, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLE, LHS::SWZ, RHS::SWZ, N) operator<=(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPLE, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLE, LHS::SWZ, SWZ_ANY, N) operator<=(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, CMPLE, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLE, SWZ_ANY, RHS::SWZ, N) operator<=(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPLE, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLE, LHS::SWZ, SWZ_ANY, 1) operator<=(const v<T, LHS, 1> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_FB(T, CMPLE, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), 1);
    }
    template<typename T, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLE, SWZ_ANY, RHS::SWZ, 1) operator<=(const typename T::type lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPLE, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, 1);
    }

    /*
     * binary compare less than
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLT, LHS::SWZ, RHS::SWZ, 1) operator<(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPLT, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLT, BC(LHS::SWZ, N), RHS::SWZ, N) operator<(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPLT, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLT, LHS::SWZ, BC(RHS::SWZ, N), N) operator<(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, CMPLT, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLT, LHS::SWZ, RHS::SWZ, N) operator<(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPLT, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLT, LHS::SWZ, SWZ_ANY, N) operator<(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, CMPLT, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLT, SWZ_ANY, RHS::SWZ, N) operator<(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPLT, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLT, LHS::SWZ, SWZ_ANY, 1) operator<(const v<T, LHS, 1> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_FB(T, CMPLT, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), 1);
    }
    template<typename T, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPLT, SWZ_ANY, RHS::SWZ, 1) operator<(const typename T::type lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPLT, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, 1);
    }

    /*
     * binary compare greater than or equal
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGE, LHS::SWZ, RHS::SWZ, 1) operator>=(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPGE, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGE, BC(LHS::SWZ, N), RHS::SWZ, N) operator>=(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPGE, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGE, LHS::SWZ, BC(RHS::SWZ, N), N) operator>=(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, CMPGE, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGE, LHS::SWZ, RHS::SWZ, N) operator>=(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPGE, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGE, LHS::SWZ, SWZ_ANY, N) operator>=(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, CMPGE, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGE, SWZ_ANY, RHS::SWZ, N) operator>=(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPGE, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGE, LHS::SWZ, SWZ_ANY, 1) operator>=(const v<T, LHS, 1> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_FB(T, CMPGE, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), 1);
    }
    template<typename T, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGE, SWZ_ANY, RHS::SWZ, 1) operator>=(const typename T::type lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPGE, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, 1);
    }

    /*
     * binary compare greater than
     */

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGT, LHS::SWZ, RHS::SWZ, 1) operator>(const v<T, LHS, 1> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPGT, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGT, BC(LHS::SWZ, N), RHS::SWZ, N) operator>(const v<T, LHS, 1> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPGT, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGT, LHS::SWZ, BC(RHS::SWZ, N), N) operator>(const v<T, LHS, N> &lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, CMPGT, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGT, LHS::SWZ, RHS::SWZ, N) operator>(const v<T, LHS, N> &lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPGT, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGT, LHS::SWZ, SWZ_ANY, N) operator>(const v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, CMPGT, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGT, SWZ_ANY, RHS::SWZ, N) operator>(const typename T::type lhs, const v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, CMPGT, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGT, LHS::SWZ, SWZ_ANY, 1) operator>(const v<T, LHS, 1> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_FB(T, CMPGT, LHS::SWZ, lhs.p, SWZ_ANY, T::CTOR(rhs), 1);
    }
    template<typename T, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, CMPGT, SWZ_ANY, RHS::SWZ, 1) operator>(const typename T::type lhs, const v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_FB(T, CMPGT, SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, 1);
    }

#   undef SPECIFIC2_FB
#   undef SPECIFIC2_F
#   undef SPECIFIC2_T

#   undef SPECIFIC1I_F
#   undef SPECIFIC1_F
#   undef SPECIFIC1_T
}

#   define SPECIFIC1_T(T, OP, RHS, N)                           meta::v<typename T::template OP<RHS>::type, meta::sp<typename T::template OP<RHS>::type, T::template OP<RHS>::S, N>, N>
#   define SPECIFIC1_F(T, OP, RHS, rhs, N)                      SPECIFIC1_T(T, OP, RHS, N)::pack(T::template OP<RHS>::f(rhs))

#   define SPECIFIC2_T(T, OP, LHS, RHS, N)                      meta::v<typename T::template OP<LHS, RHS>::type, meta::sp<typename T::template OP<LHS, RHS>::type, T::template OP<LHS, RHS>::S, N>, N>
#   define SPECIFIC2_F(T, OP, LHS, lhs, RHS, rhs, N)            SPECIFIC2_T(T, OP, LHS, RHS, N)::pack(T::template OP<LHS, RHS>::f(lhs, rhs))

#   define SPECIFIC3_T(T, OP, LHS, CHS, RHS, N)                 meta::v<typename T::template OP<LHS, CHS, RHS>::type, meta::sp<typename T::template OP<LHS, CHS, RHS>::type, T::template OP<LHS, CHS, RHS>::S, N>, N>
#   define SPECIFIC3_F(T, OP, LHS, lhs, CHS, chs, RHS, rhs, N)  SPECIFIC3_T(T, OP, LHS, CHS, RHS, N)::pack(T::template OP<LHS, CHS, RHS>::f(lhs, chs, rhs))

    /*
     * ternary mad
     */

    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, LHS::SWZ, CHS::SWZ, RHS::SWZ, N) mad(const meta::v<T, LHS, N> &lhs, const meta::v<T, CHS, N> &chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MADD, LHS::SWZ, lhs.p, CHS::SWZ, chs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename CHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, LHS::SWZ, CHS::SWZ, meta::SWZ_ANY, N) mad(const meta::v<T, LHS, N> &lhs, const meta::v<T, CHS, N> &chs, const typename T::type rhs)
    {
        return SPECIFIC3_F(T, MADD, LHS::SWZ, lhs.p, CHS::SWZ, chs.p, meta::SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, BC(LHS::SWZ, N), CHS::SWZ, RHS::SWZ, N) mad(const meta::v<T, LHS, 1> &lhs, const meta::v<T, CHS, N> &chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MADD, BC(LHS::SWZ, N), lhs.p, CHS::SWZ, chs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, meta::SWZ_ANY, CHS::SWZ, RHS::SWZ, N) mad(const typename T::type lhs, const meta::v<T, CHS, N> &chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MADD, meta::SWZ_ANY, T::CTOR(lhs), CHS::SWZ, chs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, LHS::SWZ, BC(CHS::SWZ, N), RHS::SWZ, N) mad(const meta::v<T, LHS, N> &lhs, const meta::v<T, CHS, 1> &chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MADD, LHS::SWZ, lhs.p, BC(CHS::SWZ, N), chs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, LHS::SWZ, meta::SWZ_ANY, RHS::SWZ, N) mad(const meta::v<T, LHS, N> &lhs, const typename T::type chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MADD, LHS::SWZ, lhs.p, meta::SWZ_ANY, T::CTOR(chs), RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, BC(LHS::SWZ, N), CHS::SWZ, BC(RHS::SWZ, N), N) mad(const meta::v<T, LHS, 1> &lhs, const meta::v<T, CHS, N> &chs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC3_F(T, MADD, BC(LHS::SWZ, N), lhs.p, CHS::SWZ, chs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename CHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, meta::SWZ_ANY, CHS::SWZ, meta::SWZ_ANY, N) mad(const typename T::type lhs, const meta::v<T, CHS, N> &chs, const typename T::type rhs)
    {
        return SPECIFIC3_F(T, MADD, meta::SWZ_ANY, T::CTOR(lhs), CHS::SWZ, chs.p, meta::SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, LHS::SWZ, BC(CHS::SWZ, N), BC(RHS::SWZ, N), N) mad(const meta::v<T, LHS, N> &lhs, const meta::v<T, CHS, 1> &chs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC3_F(T, MADD, LHS::SWZ, lhs.p, BC(CHS::SWZ, N), chs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, LHS::SWZ, meta::SWZ_ANY, meta::SWZ_ANY, N) mad(const meta::v<T, LHS, N> &lhs, const typename T::type chs, const typename T::type rhs)
    {
        return SPECIFIC3_F(T, MADD, LHS::SWZ, lhs.p, meta::SWZ_ANY, T::CTOR(chs), meta::SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS> static MATH_FORCEINLINE SPECIFIC3_T(T, MADD, LHS::SWZ, CHS::SWZ, RHS::SWZ, 1) mad(const meta::v<T, LHS, 1> &lhs, const meta::v<T, CHS, 1> &chs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC3_F(T, MADD, LHS::SWZ, lhs.p, CHS::SWZ, chs.p, RHS::SWZ, rhs.p, 1);
    }

    /*
     * ternary msub
     */

    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, LHS::SWZ, CHS::SWZ, RHS::SWZ, N) msub(const meta::v<T, LHS, N> &lhs, const meta::v<T, CHS, N> &chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MSUB, LHS::SWZ, lhs.p, CHS::SWZ, chs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename CHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, LHS::SWZ, CHS::SWZ, meta::SWZ_ANY, N) msub(const meta::v<T, LHS, N> &lhs, const meta::v<T, CHS, N> &chs, const typename T::type rhs)
    {
        return SPECIFIC3_F(T, MSUB, LHS::SWZ, lhs.p, CHS::SWZ, chs.p, meta::SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, BC(LHS::SWZ, N), CHS::SWZ, RHS::SWZ, N) msub(const meta::v<T, LHS, 1> &lhs, const meta::v<T, CHS, N> &chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MSUB, BC(LHS::SWZ, N), lhs.p, CHS::SWZ, chs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, meta::SWZ_ANY, CHS::SWZ, RHS::SWZ, N) msub(const typename T::type lhs, const meta::v<T, CHS, N> &chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MSUB, meta::SWZ_ANY, T::CTOR(lhs), CHS::SWZ, chs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, LHS::SWZ, BC(CHS::SWZ, N), RHS::SWZ, N) msub(const meta::v<T, LHS, N> &lhs, const meta::v<T, CHS, 1> &chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MSUB, LHS::SWZ, lhs.p, BC(CHS::SWZ, N), chs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, LHS::SWZ, meta::SWZ_ANY, RHS::SWZ, N) msub(const meta::v<T, LHS, N> &lhs, const typename T::type chs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC3_F(T, MSUB, LHS::SWZ, lhs.p, meta::SWZ_ANY, T::CTOR(chs), RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, BC(LHS::SWZ, N), CHS::SWZ, BC(RHS::SWZ, N), N) msub(const meta::v<T, LHS, 1> &lhs, const meta::v<T, CHS, N> &chs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC3_F(T, MSUB, BC(LHS::SWZ, N), lhs.p, CHS::SWZ, chs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename CHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, meta::SWZ_ANY, CHS::SWZ, meta::SWZ_ANY, N) msub(const typename T::type lhs, const meta::v<T, CHS, N> &chs, const typename T::type rhs)
    {
        return SPECIFIC3_F(T, MSUB, meta::SWZ_ANY, T::CTOR(lhs), CHS::SWZ, chs.p, meta::SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, LHS::SWZ, BC(CHS::SWZ, N), BC(RHS::SWZ, N), N) msub(const meta::v<T, LHS, N> &lhs, const meta::v<T, CHS, 1> &chs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC3_F(T, MSUB, LHS::SWZ, lhs.p, BC(CHS::SWZ, N), chs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, LHS::SWZ, meta::SWZ_ANY, meta::SWZ_ANY, N) msub(const meta::v<T, LHS, N> &lhs, const typename T::type chs, const typename T::type rhs)
    {
        return SPECIFIC3_F(T, MSUB, LHS::SWZ, lhs.p, meta::SWZ_ANY, T::CTOR(chs), meta::SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename LHS, typename CHS, typename RHS> static MATH_FORCEINLINE SPECIFIC3_T(T, MSUB, LHS::SWZ, CHS::SWZ, RHS::SWZ, 1) msub(const meta::v<T, LHS, 1> &lhs, const meta::v<T, CHS, 1> &chs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC3_F(T, MSUB, LHS::SWZ, lhs.p, CHS::SWZ, chs.p, RHS::SWZ, rhs.p, 1);
    }

    /*
     *  abs
     */
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC1_T(T, ABS, RHS::SWZ, N) abs(const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC1_F(T, ABS, RHS::SWZ, rhs.p, N);
    }

    /*
     *  any
     */
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE int any(const meta::v<T, RHS, N> &rhs)
    {
        return T::template ANY<RHS::SWZ>::f(rhs.p);
    }

    /*
     *  all
     */
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE int all(const meta::v<T, RHS, N> &rhs)
    {
        return T::template ALL<RHS::SWZ>::f(rhs.p);
    }

    /*
     *  min
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, MIN, LHS::SWZ, RHS::SWZ, 1) min(const meta::v<T, LHS, 1> &lhs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, MIN, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MIN, BC(LHS::SWZ, N), RHS::SWZ, N) min(const meta::v<T, LHS, 1> &lhs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MIN, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MIN, LHS::SWZ, BC(RHS::SWZ, N), N) min(const meta::v<T, LHS, N> &lhs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, MIN, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MIN, LHS::SWZ, RHS::SWZ, N) min(const meta::v<T, LHS, N> &lhs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MIN, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MIN, LHS::SWZ, meta::SWZ_ANY, N) min(const meta::v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, MIN, LHS::SWZ, lhs.p, meta::SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MIN, meta::SWZ_ANY, RHS::SWZ, N) min(const typename T::type lhs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MIN, meta::SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

    /*
     *  max
     */
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE SPECIFIC2_T(T, MAX, LHS::SWZ, RHS::SWZ, 1) max(const meta::v<T, LHS, 1> &lhs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, MAX, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, 1);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MAX, BC(LHS::SWZ, N), RHS::SWZ, N) max(const meta::v<T, LHS, 1> &lhs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MAX, BC(LHS::SWZ, N), lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MAX, LHS::SWZ, BC(RHS::SWZ, N), N) max(const meta::v<T, LHS, N> &lhs, const meta::v<T, RHS, 1> &rhs)
    {
        return SPECIFIC2_F(T, MAX, LHS::SWZ, lhs.p, BC(RHS::SWZ, N), rhs.p, N);
    }
    template<typename T, typename LHS, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MAX, LHS::SWZ, RHS::SWZ, N) max(const meta::v<T, LHS, N> &lhs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MAX, LHS::SWZ, lhs.p, RHS::SWZ, rhs.p, N);
    }
    template<typename T, typename LHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MAX, LHS::SWZ, meta::SWZ_ANY, N) max(const meta::v<T, LHS, N> &lhs, const typename T::type rhs)
    {
        return SPECIFIC2_F(T, MAX, LHS::SWZ, lhs.p, meta::SWZ_ANY, T::CTOR(rhs), N);
    }
    template<typename T, typename RHS, int N> static MATH_FORCEINLINE SPECIFIC2_T(T, MAX, meta::SWZ_ANY, RHS::SWZ, N) max(const typename T::type lhs, const meta::v<T, RHS, N> &rhs)
    {
        return SPECIFIC2_F(T, MAX, meta::SWZ_ANY, T::CTOR(lhs), RHS::SWZ, rhs.p, N);
    }

#   undef SPECIFIC3_F
#   undef SPECIFIC3_T

#   undef SPECIFIC2_F
#   undef SPECIFIC2_T

#   undef SPECIFIC1_F
#   undef SPECIFIC1_T
}

#else // defined(MATH_HAS_NATIVE_SIMD)

namespace math
{
namespace meta
{
    template<typename T> struct bool_of
    {
        typedef signed int type;
    };
    template<> struct bool_of<signed char>
    {
        typedef signed char type;
    };
    template<> struct bool_of<unsigned char>
    {
        typedef signed char type;
    };
    template<> struct bool_of<signed short>
    {
        typedef signed short type;
    };
    template<> struct bool_of<unsigned short>
    {
        typedef signed short type;
    };
    template<> struct bool_of<signed long>
    {
        typedef signed long type;
    };
    template<> struct bool_of<unsigned long>
    {
        typedef signed long type;
    };
    template<> struct bool_of<signed long long>
    {
        typedef signed long long type;
    };
    template<> struct bool_of<unsigned long long>
    {
        typedef signed long long type;
    };
    template<> struct bool_of<float>
    {
        typedef int type;
    };
#if defined(__LP64__)
    template<> struct bool_of<double>
    {
        typedef signed long type;
    };
#else
    template<> struct bool_of<double>
    {
        typedef signed long long type;
    };
#endif

    template<int condition, typename T> struct constraint;
    template<typename T> struct constraint<1, T>
    {
        typedef T type;
    };

    // vec<2, T> unary operators
    template<typename T, typename RHS> static MATH_FORCEINLINE rv<T, RHS, 2U> operator+(const rv<T, RHS, 2U> &rhs)
    {
        return rhs;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator-(const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(-rhs.x, -rhs.y);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator~(const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(~rhs.x, ~rhs.y);
    }

    // vec<2, T> binary operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator+(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x + rhs.x, lhs.y + rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator+(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<T, 2U>(lhs.x + rhs, lhs.y + rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator+(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs + rhs.x, lhs + rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator-(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator-(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<T, 2U>(lhs.x - rhs, lhs.y - rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator-(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs - rhs.x, lhs - rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator*(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x * rhs.x, lhs.y * rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator*(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<T, 2U>(lhs.x * rhs, lhs.y * rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator*(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs * rhs.x, lhs * rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator/(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x / rhs.x, lhs.y / rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator/(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<T, 2U>(lhs.x / rhs, lhs.y / rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator/(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs / rhs.x, lhs / rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator%(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x % rhs.x, lhs.y % rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator%(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<T, 2U>(lhs.x % rhs, lhs.y % rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator%(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs % rhs.x, lhs % rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator&(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x & rhs.x, lhs.y & rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator&(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<T, 2U>(lhs.x & rhs, lhs.y & rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator&(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs & rhs.x, lhs & rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator|(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x | rhs.x, lhs.y | rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator|(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<T, 2U>(lhs.x | rhs, lhs.y | rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator|(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs | rhs.x, lhs | rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator^(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x ^ rhs.x, lhs.y ^ rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator^(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<T, 2U>(lhs.x ^ rhs, lhs.y ^ rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator^(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs ^ rhs.x, lhs ^ rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator<<(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x << rhs.x, lhs.y << rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator<<(const rv<T, LHS, 2U> &lhs, int rhs)
    {
        return vec<T, 2U>(lhs.x << rhs, lhs.y << rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator<<(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs << rhs.x, lhs << rhs.y);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator>>(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs.x >> rhs.x, lhs.y >> rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 2U> operator>>(const rv<T, LHS, 2U> &lhs, int rhs)
    {
        return vec<T, 2U>(lhs.x >> rhs, lhs.y >> rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 2U> operator>>(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<T, 2U>(lhs >> rhs.x, lhs >> rhs.y);
    }

    // vec<2, T> binary assignment operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator+=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x + rhs.x, lhs.y + rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator+=(const lv<T, LHS, 2U> &lhs, T rhs)
    {
        return lhs.set(lhs.x + rhs, lhs.y + rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator-=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator-=(const lv<T, LHS, 2U> &lhs, T rhs)
    {
        return lhs.set(lhs.x - rhs, lhs.y - rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator*=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x * rhs.x, lhs.y * rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator*=(const lv<T, LHS, 2U> &lhs, T rhs)
    {
        return lhs.set(lhs.x * rhs, lhs.y * rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator/=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x / rhs.x, lhs.y / rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator/=(const lv<T, LHS, 2U> &lhs, T rhs)
    {
        return lhs.set(lhs.x / rhs, lhs.y / rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator%=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x % rhs.x, lhs.y % rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator%=(const lv<T, LHS, 2U> &lhs, T rhs)
    {
        return lhs.set(lhs.x % rhs, lhs.y % rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator&=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x & rhs.x, lhs.y & rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator&=(const lv<T, LHS, 2U> &lhs, T rhs)
    {
        return lhs.set(lhs.x & rhs, lhs.y & rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator|=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x | rhs.x, lhs.y | rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator|=(const lv<T, LHS, 2U> &lhs, T rhs)
    {
        return lhs.set(lhs.x | rhs, lhs.y | rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator^=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x ^ rhs.x, lhs.y ^ rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator^=(const lv<T, LHS, 2U> &lhs, T rhs)
    {
        return lhs.set(lhs.x ^ rhs, lhs.y ^ rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator<<=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x << rhs.x, lhs.y << rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator<<=(const lv<T, LHS, 2U> &lhs, int rhs)
    {
        return lhs.set(lhs.x << rhs, lhs.y << rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator>>=(vec_attr lv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return lhs.set(lhs.x >> rhs.x, lhs.y >> rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 2U> &operator>>=(const lv<T, LHS, 2U> &lhs, int rhs)
    {
        return lhs.set(lhs.x >> rhs, lhs.y >> rhs);
    }

    // vec<2, T> comparison operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator==(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x == rhs.x), -(lhs.y == rhs.y));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator==(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x == rhs), -(lhs.y == rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator==(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs == rhs.x), -(lhs == rhs.y));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator!=(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x != rhs.x), -(lhs.y != rhs.y));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator!=(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x != rhs), -(lhs.y != rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator!=(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs != rhs.x), -(lhs != rhs.y));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator<(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x < rhs.x), -(lhs.y < rhs.y));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator<(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x < rhs), -(lhs.y < rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator<(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs < rhs.x), -(lhs < rhs.y));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator<=(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x <= rhs.x), -(lhs.y <= rhs.y));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator<=(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x <= rhs), -(lhs.y <= rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator<=(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs <= rhs.x), -(lhs <= rhs.y));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator>(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x > rhs.x), -(lhs.y > rhs.y));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator>(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x > rhs), -(lhs.y > rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator>(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs > rhs.x), -(lhs > rhs.y));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator>=(const rv<T, LHS, 2U> &lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x >= rhs.x), -(lhs.y >= rhs.y));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator>=(const rv<T, LHS, 2U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs.x >= rhs), -(lhs.y >= rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 2U> operator>=(T lhs, const rv<T, RHS, 2U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 2U>(-(lhs >= rhs.x), -(lhs >= rhs.y));
    }

    // vec<3, T> unary operators
    template<typename T, typename RHS> static MATH_FORCEINLINE rv<T, RHS, 3U> operator+(const rv<T, RHS, 3U> &rhs)
    {
        return rhs;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator-(const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(-rhs.x, -rhs.y, -rhs.z);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator~(const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(~rhs.x, ~rhs.y, ~rhs.z);
    }

    // vec<3, T> binary operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator+(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator+(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<T, 3U>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator+(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator-(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator-(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<T, 3U>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator-(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator*(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator*(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<T, 3U>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator*(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator/(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator/(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<T, 3U>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator/(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator%(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator%(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<T, 3U>(lhs.x % rhs, lhs.y % rhs, lhs.z % rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator%(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs % rhs.x, lhs % rhs.y, lhs % rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator&(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator&(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<T, 3U>(lhs.x & rhs, lhs.y & rhs, lhs.z & rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator&(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs & rhs.x, lhs & rhs.y, lhs & rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator|(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator|(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<T, 3U>(lhs.x | rhs, lhs.y | rhs, lhs.z | rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator|(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs | rhs.x, lhs | rhs.y, lhs | rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator^(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator^(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<T, 3U>(lhs.x ^ rhs, lhs.y ^ rhs, lhs.z ^ rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator^(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs ^ rhs.x, lhs ^ rhs.y, lhs ^ rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator<<(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator<<(const rv<T, LHS, 3U> &lhs, int rhs)
    {
        return vec<T, 3U>(lhs.x << rhs, lhs.y << rhs, lhs.z << rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator<<(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs << rhs.x, lhs << rhs.y, lhs << rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator>>(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 3U> operator>>(const rv<T, LHS, 3U> &lhs, int rhs)
    {
        return vec<T, 3U>(lhs.x >> rhs, lhs.y >> rhs, lhs.z >> rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 3U> operator>>(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<T, 3U>(lhs >> rhs.x, lhs >> rhs.y, lhs >> rhs.z);
    }

    // vec<3, T> binary assignment operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator+=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator+=(const lv<T, LHS, 3U> &lhs, T rhs)
    {
        return lhs.set(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator-=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator-=(const lv<T, LHS, 3U> &lhs, T rhs)
    {
        return lhs.set(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator*=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator*=(const lv<T, LHS, 3U> &lhs, T rhs)
    {
        return lhs.set(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator/=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator/=(const lv<T, LHS, 3U> &lhs, T rhs)
    {
        return lhs.set(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator%=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator%=(const lv<T, LHS, 3U> &lhs, T rhs)
    {
        return lhs.set(lhs.x % rhs, lhs.y % rhs, lhs.z % rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator&=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator&=(const lv<T, LHS, 3U> &lhs, T rhs)
    {
        return lhs.set(lhs.x & rhs, lhs.y & rhs, lhs.z & rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator|=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator|=(const lv<T, LHS, 3U> &lhs, T rhs)
    {
        return lhs.set(lhs.x | rhs, lhs.y | rhs, lhs.z | rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator^=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator^=(const lv<T, LHS, 3U> &lhs, T rhs)
    {
        return lhs.set(lhs.x ^ rhs, lhs.y ^ rhs, lhs.z ^ rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator<<=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator<<=(const lv<T, LHS, 3U> &lhs, int rhs)
    {
        return lhs.set(lhs.x << rhs, lhs.y << rhs, lhs.z << rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator>>=(vec_attr lv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return lhs.set(lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 3U> &operator>>=(const lv<T, LHS, 3U> &lhs, int rhs)
    {
        return lhs.set(lhs.x >> rhs, lhs.y >> rhs, lhs.z >> rhs);
    }

    // vec<3, T> comparison operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator==(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x == rhs.x), -(lhs.y == rhs.y), -(lhs.z == rhs.z));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator==(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x == rhs), -(lhs.y == rhs), -(lhs.z == rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator==(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs == rhs.x), -(lhs == rhs.y), -(lhs == rhs.z));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator!=(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x != rhs.x), -(lhs.y != rhs.y), -(lhs.z != rhs.z));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator!=(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x != rhs), -(lhs.y != rhs), -(lhs.z != rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator!=(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs != rhs.x), -(lhs != rhs.y), -(lhs != rhs.z));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator<(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x < rhs.x), -(lhs.y < rhs.y), -(lhs.z < rhs.z));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator<(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x < rhs), -(lhs.y < rhs), -(lhs.z < rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator<(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs < rhs.x), -(lhs < rhs.y), -(lhs < rhs.z));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator<=(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x <= rhs.x), -(lhs.y <= rhs.y), -(lhs.z <= rhs.z));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator<=(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x <= rhs), -(lhs.y <= rhs), -(lhs.z <= rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator<=(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs <= rhs.x), -(lhs <= rhs.y), -(lhs <= rhs.z));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator>(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x > rhs.x), -(lhs.y > rhs.y), -(lhs.z > rhs.z));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator>(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x > rhs), -(lhs.y > rhs), -(lhs.z > rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator>(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs > rhs.x), -(lhs > rhs.y), -(lhs > rhs.z));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator>=(const rv<T, LHS, 3U> &lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x >= rhs.x), -(lhs.y >= rhs.y), -(lhs.z >= rhs.z));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator>=(const rv<T, LHS, 3U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs.x >= rhs), -(lhs.y >= rhs), -(lhs.z >= rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 3U> operator>=(T lhs, const rv<T, RHS, 3U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 3U>(-(lhs >= rhs.x), -(lhs >= rhs.y), -(lhs >= rhs.z));
    }

    // vec<4, T> unary operators
    template<typename T, typename RHS> static MATH_FORCEINLINE rv<T, RHS, 4U> operator+(const rv<T, RHS, 4U> &rhs)
    {
        return rhs;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator-(const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(-rhs.x, -rhs.y, -rhs.z, -rhs.w);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator~(const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(~rhs.x, ~rhs.y, ~rhs.z, ~rhs.w);
    }

    // vec<4, T> binary operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator+(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator+(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<T, 4U>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator+(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator-(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator-(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<T, 4U>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator-(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator*(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator*(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<T, 4U>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator*(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator/(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator/(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<T, 4U>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator/(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator%(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z, lhs.w % rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator%(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<T, 4U>(lhs.x % rhs, lhs.y % rhs, lhs.z % rhs, lhs.w % rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator%(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs % rhs.x, lhs % rhs.y, lhs % rhs.z, lhs % rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator&(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z, lhs.w & rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator&(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<T, 4U>(lhs.x & rhs, lhs.y & rhs, lhs.z & rhs, lhs.w & rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator&(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs & rhs.x, lhs & rhs.y, lhs & rhs.z, lhs & rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator|(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z, lhs.w | rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator|(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<T, 4U>(lhs.x | rhs, lhs.y | rhs, lhs.z | rhs, lhs.w | rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator|(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs | rhs.x, lhs | rhs.y, lhs | rhs.z, lhs | rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator^(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator^(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<T, 4U>(lhs.x ^ rhs, lhs.y ^ rhs, lhs.z ^ rhs, lhs.w ^ rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator^(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs ^ rhs.x, lhs ^ rhs.y, lhs ^ rhs.z, lhs ^ rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator<<(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z, lhs.w << rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator<<(const rv<T, LHS, 4U> &lhs, int rhs)
    {
        return vec<T, 4U>(lhs.x << rhs, lhs.y << rhs, lhs.z << rhs, lhs.w << rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator<<(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs << rhs.x, lhs << rhs.y, lhs << rhs.z, lhs << rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator>>(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z, lhs.w >> rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<T, 4U> operator>>(const rv<T, LHS, 4U> &lhs, int rhs)
    {
        return vec<T, 4U>(lhs.x >> rhs, lhs.y >> rhs, lhs.z >> rhs, lhs.w >> rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<T, 4U> operator>>(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<T, 4U>(lhs >> rhs.x, lhs >> rhs.y, lhs >> rhs.z, lhs >> rhs.w);
    }

    // vec<4, T> binary assignment operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator+=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator+=(const lv<T, LHS, 4U> &lhs, T rhs)
    {
        return lhs.set(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator-=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator-=(const lv<T, LHS, 4U> &lhs, T rhs)
    {
        return lhs.set(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator*=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator*=(const lv<T, LHS, 4U> &lhs, T rhs)
    {
        return lhs.set(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator/=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator/=(const lv<T, LHS, 4U> &lhs, T rhs)
    {
        return lhs.set(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator%=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z, lhs.w % rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator%=(const lv<T, LHS, 4U> &lhs, T rhs)
    {
        return lhs.set(lhs.x % rhs, lhs.y % rhs, lhs.z % rhs, lhs.w % rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator&=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z, lhs.w & rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator&=(const lv<T, LHS, 4U> &lhs, T rhs)
    {
        return lhs.set(lhs.x & rhs, lhs.y & rhs, lhs.z & rhs, lhs.w & rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator|=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z, lhs.w | rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator|=(const lv<T, LHS, 4U> &lhs, T rhs)
    {
        return lhs.set(lhs.x | rhs, lhs.y | rhs, lhs.z | rhs, lhs.w | rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator^=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator^=(const lv<T, LHS, 4U> &lhs, T rhs)
    {
        return lhs.set(lhs.x ^ rhs, lhs.y ^ rhs, lhs.z ^ rhs, lhs.w ^ rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator<<=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z, lhs.w << rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator<<=(const lv<T, LHS, 4U> &lhs, int rhs)
    {
        return lhs.set(lhs.x << rhs, lhs.y << rhs, lhs.z << rhs, lhs.w << rhs);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator>>=(vec_attr lv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return lhs.set(lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z, lhs.w >> rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec_attr lv<T, LHS, 4U> &operator>>=(const lv<T, LHS, 4U> &lhs, int rhs)
    {
        return lhs.set(lhs.x >> rhs, lhs.y >> rhs, lhs.z >> rhs, lhs.w >> rhs);
    }

    // vec<4, T> comparison operators
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator==(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x == rhs.x), -(lhs.y == rhs.y), -(lhs.z == rhs.z), -(lhs.w == rhs.w));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator==(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x == rhs), -(lhs.y == rhs), -(lhs.z == rhs), -(lhs.w == rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator==(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs == rhs.x), -(lhs == rhs.y), -(lhs == rhs.z), -(lhs == rhs.w));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator!=(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x != rhs.x), -(lhs.y != rhs.y), -(lhs.z != rhs.z), -(lhs.w != rhs.w));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator!=(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x != rhs), -(lhs.y != rhs), -(lhs.z != rhs), -(lhs.w != rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator!=(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs != rhs.x), -(lhs != rhs.y), -(lhs != rhs.z), -(lhs != rhs.w));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator<(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x < rhs.x), -(lhs.y < rhs.y), -(lhs.z < rhs.z), -(lhs.w < rhs.w));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator<(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x < rhs), -(lhs.y < rhs), -(lhs.z < rhs), -(lhs.w < rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator<(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs < rhs.x), -(lhs < rhs.y), -(lhs < rhs.z), -(lhs < rhs.w));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator<=(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x <= rhs.x), -(lhs.y <= rhs.y), -(lhs.z <= rhs.z), -(lhs.w <= rhs.w));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator<=(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x <= rhs), -(lhs.y <= rhs), -(lhs.z <= rhs), -(lhs.w <= rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator<=(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs <= rhs.x), -(lhs <= rhs.y), -(lhs <= rhs.z), -(lhs <= rhs.w));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator>(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x > rhs.x), -(lhs.y > rhs.y), -(lhs.z > rhs.z), -(lhs.w > rhs.w));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator>(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x > rhs), -(lhs.y > rhs), -(lhs.z > rhs), -(lhs.w > rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator>(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs > rhs.x), -(lhs > rhs.y), -(lhs > rhs.z), -(lhs > rhs.w));
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator>=(const rv<T, LHS, 4U> &lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x >= rhs.x), -(lhs.y >= rhs.y), -(lhs.z >= rhs.z), -(lhs.w >= rhs.w));
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator>=(const rv<T, LHS, 4U> &lhs, T rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs.x >= rhs), -(lhs.y >= rhs), -(lhs.z >= rhs), -(lhs.w >= rhs));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE vec<explicit_typename bool_of<T>::type, 4U> operator>=(T lhs, const rv<T, RHS, 4U> &rhs)
    {
        return vec<explicit_typename bool_of<T>::type, 4U>(-(lhs >= rhs.x), -(lhs >= rhs.y), -(lhs >= rhs.z), -(lhs >= rhs.w));
    }
}

    /*
     *  mad
     */
    static MATH_FORCEINLINE float mad(float a, float b, float c)
    {
        return a * b + c;
    }

    template<typename T, typename LHS, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> mad(const meta::rv<T, LHS, 2U> &lhs, const meta::rv<T, CHS, 2U> &chs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> mad(T lhs, const meta::rv<T, CHS, 2U> &chs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> mad(const meta::rv<T, LHS, 2U> &lhs, T chs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename CHS> static MATH_FORCEINLINE meta::vec<T, 2U> mad(T lhs, const meta::rv<T, CHS, 2U> &chs, T rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 2U> mad(const meta::rv<T, LHS, 2U> &lhs, T chs, T rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename LHS, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> mad(const meta::rv<T, LHS, 3U> &lhs, const meta::rv<T, CHS, 3U> &chs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> mad(T lhs, const meta::rv<T, CHS, 3U> &chs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> mad(const meta::rv<T, LHS, 3U> &lhs, T chs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename CHS> static MATH_FORCEINLINE meta::vec<T, 3U> mad(T lhs, const meta::rv<T, CHS, 3U> &chs, T rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 3U> mad(const meta::rv<T, LHS, 3U> &lhs, T chs, T rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename LHS, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> mad(const meta::rv<T, LHS, 4U> &lhs, const meta::rv<T, CHS, 4U> &chs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> mad(T lhs, const meta::rv<T, CHS, 4U> &chs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> mad(const meta::rv<T, LHS, 4U> &lhs, T chs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename CHS> static MATH_FORCEINLINE meta::vec<T, 4U> mad(T lhs, const meta::rv<T, CHS, 4U> &chs, T rhs)
    {
        return lhs * chs + rhs;
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 4U> mad(const meta::rv<T, LHS, 4U> &lhs, T chs, T rhs)
    {
        return lhs * chs + rhs;
    }

    /*
     *  msub
     */
    static MATH_FORCEINLINE float msub(float a, float b, float c)
    {
        return a * b - c;
    }

    template<typename T, typename LHS, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> msub(const meta::rv<T, LHS, 2U> &lhs, const meta::rv<T, CHS, 2U> &chs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> msub(T lhs, const meta::rv<T, CHS, 2U> &chs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> msub(const meta::rv<T, LHS, 2U> &lhs, T chs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename CHS> static MATH_FORCEINLINE meta::vec<T, 2U> msub(T lhs, const meta::rv<T, CHS, 2U> &chs, T rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 2U> msub(const meta::rv<T, LHS, 2U> &lhs, T chs, T rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename LHS, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> msub(const meta::rv<T, LHS, 3U> &lhs, const meta::rv<T, CHS, 3U> &chs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> msub(T lhs, const meta::rv<T, CHS, 3U> &chs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> msub(const meta::rv<T, LHS, 3U> &lhs, T chs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename CHS> static MATH_FORCEINLINE meta::vec<T, 3U> msub(T lhs, const meta::rv<T, CHS, 3U> &chs, T rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 3U> msub(const meta::rv<T, LHS, 3U> &lhs, T chs, T rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename LHS, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> msub(const meta::rv<T, LHS, 4U> &lhs, const meta::rv<T, CHS, 4U> &chs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename CHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> msub(T lhs, const meta::rv<T, CHS, 4U> &chs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> msub(const meta::rv<T, LHS, 4U> &lhs, T chs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename CHS> static MATH_FORCEINLINE meta::vec<T, 4U> msub(T lhs, const meta::rv<T, CHS, 4U> &chs, T rhs)
    {
        return lhs * chs - rhs;
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 4U> msub(const meta::rv<T, LHS, 4U> &lhs, T chs, T rhs)
    {
        return lhs * chs - rhs;
    }

    /*
     *  abs
     */
    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> abs(const meta::rv<T, RHS, 2U> &rhs)
    {
        return meta::vec<T, 2U>(abs(rhs.x), abs(rhs.y));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> abs(const meta::rv<T, RHS, 3U> &rhs)
    {
        return meta::vec<T, 3U>(abs(rhs.x), abs(rhs.y), abs(rhs.z));
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> abs(const meta::rv<T, RHS, 4U> &rhs)
    {
        return meta::vec<T, 4U>(abs(rhs.x), abs(rhs.y), abs(rhs.z), abs(rhs.w));
    }

    //
    //  all
    //
    template<typename T, typename RHS> static MATH_FORCEINLINE int all(const meta::rv<T, RHS, 4U> &rhs)
    {
        return (rhs.x & rhs.y & rhs.z & rhs.w) < 0;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE int all(const meta::rv<T, RHS, 3U> &rhs)
    {
        return (rhs.x & rhs.y & rhs.z) < 0;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE int all(const meta::rv<T, RHS, 2U> &rhs)
    {
        return (rhs.x & rhs.y) < 0;
    }

    //
    //  any
    //
    template<typename T, typename RHS> static MATH_FORCEINLINE int any(const meta::rv<T, RHS, 4U> &rhs)
    {
        return (rhs.x | rhs.y | rhs.z | rhs.w) < 0;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE int any(const meta::rv<T, RHS, 3U> &rhs)
    {
        return (rhs.x | rhs.y | rhs.z) < 0;
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE int any(const meta::rv<T, RHS, 2U> &rhs)
    {
        return (rhs.x | rhs.y) < 0;
    }

    //
    //  min
    //
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> min(const meta::rv<T, LHS, 4U> &lhs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return meta::vec<T, 4U>(lhs.x < rhs.x ? lhs.x : rhs.x, lhs.y < rhs.y ? lhs.y : rhs.y, lhs.z < rhs.z ? lhs.z : rhs.z, lhs.w < rhs.w ? lhs.w : rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 4U> min(const meta::rv<T, LHS, 4U> &lhs, T rhs)
    {
        return meta::vec<T, 4U>(lhs.x < rhs ? lhs.x : rhs, lhs.y < rhs ? lhs.y : rhs, lhs.z < rhs ? lhs.z : rhs, lhs.w < rhs ? lhs.w : rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> min(T lhs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return meta::vec<T, 4U>(lhs < rhs.x ? lhs : rhs.x, lhs < rhs.y ? lhs : rhs.y, lhs < rhs.z ? lhs : rhs.z, lhs < rhs.w ? lhs : rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> min(const meta::rv<T, LHS, 3U> &lhs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return meta::vec<T, 3U>(lhs.x < rhs.x ? lhs.x : rhs.x, lhs.y < rhs.y ? lhs.y : rhs.y, lhs.z < rhs.z ? lhs.z : rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 3U> min(const meta::rv<T, LHS, 3U> &lhs, T rhs)
    {
        return meta::vec<T, 3U>(lhs.x < rhs ? lhs.x : rhs, lhs.y < rhs ? lhs.y : rhs, lhs.z < rhs ? lhs.z : rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> min(T lhs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return meta::vec<T, 3U>(lhs < rhs.x ? lhs : rhs.x, lhs < rhs.y ? lhs : rhs.y, lhs < rhs.z ? lhs : rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> min(const meta::rv<T, LHS, 2U> &lhs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return meta::vec<T, 2U>(lhs.x < rhs.x ? lhs.x : rhs.x, lhs.y < rhs.y ? lhs.y : rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 2U> min(const meta::rv<T, LHS, 2U> &lhs, T rhs)
    {
        return meta::vec<T, 2U>(lhs.x < rhs ? lhs.x : rhs, lhs.y < rhs ? lhs.y : rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> min(T lhs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return meta::vec<T, 2U>(lhs < rhs.x ? lhs : rhs.x, lhs < rhs.y ? lhs : rhs.y);
    }

    //
    //  max
    //
    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> max(const meta::rv<T, LHS, 4U> &lhs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return meta::vec<T, 4U>(lhs.x > rhs.x ? lhs.x : rhs.x, lhs.y > rhs.y ? lhs.y : rhs.y, lhs.z > rhs.z ? lhs.z : rhs.z, lhs.w > rhs.w ? lhs.w : rhs.w);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 4U> max(const meta::rv<T, LHS, 4U> &lhs, T rhs)
    {
        return meta::vec<T, 4U>(lhs.x > rhs ? lhs.x : rhs, lhs.y > rhs ? lhs.y : rhs, lhs.z > rhs ? lhs.z : rhs, lhs.w > rhs ? lhs.w : rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 4U> max(T lhs, const meta::rv<T, RHS, 4U> &rhs)
    {
        return meta::vec<T, 4U>(lhs > rhs.x ? lhs : rhs.x, lhs > rhs.y ? lhs : rhs.y, lhs > rhs.z ? lhs : rhs.z, lhs > rhs.w ? lhs : rhs.w);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> max(const meta::rv<T, LHS, 3U> &lhs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return meta::vec<T, 3U>(lhs.x > rhs.x ? lhs.x : rhs.x, lhs.y > rhs.y ? lhs.y : rhs.y, lhs.z > rhs.z ? lhs.z : rhs.z);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 3U> max(const meta::rv<T, LHS, 3U> &lhs, T rhs)
    {
        return meta::vec<T, 3U>(lhs.x > rhs ? lhs.x : rhs, lhs.y > rhs ? lhs.y : rhs, lhs.z > rhs ? lhs.z : rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 3U> max(T lhs, const meta::rv<T, RHS, 3U> &rhs)
    {
        return meta::vec<T, 3U>(lhs > rhs.x ? lhs : rhs.x, lhs > rhs.y ? lhs : rhs.y, lhs > rhs.z ? lhs : rhs.z);
    }

    template<typename T, typename LHS, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> max(const meta::rv<T, LHS, 2U> &lhs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return meta::vec<T, 2U>(lhs.x > rhs.x ? lhs.x : rhs.x, lhs.y > rhs.y ? lhs.y : rhs.y);
    }

    template<typename T, typename LHS> static MATH_FORCEINLINE meta::vec<T, 2U> max(const meta::rv<T, LHS, 2U> &lhs, T rhs)
    {
        return meta::vec<T, 2U>(lhs.x > rhs ? lhs.x : rhs, lhs.y > rhs ? lhs.y : rhs);
    }

    template<typename T, typename RHS> static MATH_FORCEINLINE meta::vec<T, 2U> max(T lhs, const meta::rv<T, RHS, 2U> &rhs)
    {
        return meta::vec<T, 2U>(lhs > rhs.x ? lhs : rhs.x, lhs > rhs.y ? lhs : rhs.y);
    }
}

#endif //defined(MATH_HAS_NATIVE_SIMD)
