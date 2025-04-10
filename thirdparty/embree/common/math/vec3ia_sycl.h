// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../sys/alloc.h"
#include "emath.h"
#include "../simd/sse.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// SSE Vec3ia Type
  ////////////////////////////////////////////////////////////////////////////////

  struct __aligned(16) Vec3ia
  {
    ALIGNED_STRUCT_(16);

    struct { int x,y,z; };

    typedef int Scalar;
    enum { N = 3 };

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3ia( ) {}
    //__forceinline Vec3ia( const __m128i a ) : m128(a) {}

    __forceinline Vec3ia( const Vec3ia& other ) : x(other.x), y(other.y), z(other.z) {}
    __forceinline Vec3ia& operator =(const Vec3ia& other) { x = other.x; y = other.y; z = other.z; return *this; }

    __forceinline explicit Vec3ia( const int a ) : x(a), y(a), z(a) {}
    __forceinline          Vec3ia( const int x, const int y, const int z) : x(x), y(y), z(z) {}
    //__forceinline explicit Vec3ia( const __m128 a ) : m128(_mm_cvtps_epi32(a)) {}
    __forceinline explicit Vec3ia(const vint4& a) : x(a[0]), y(a[1]), z(a[2]) {}

    __forceinline explicit Vec3ia( const Vec3fa& a );

    //__forceinline operator const __m128i&() const { return m128; }
    //__forceinline operator       __m128i&()       { return m128; }
    __forceinline operator vint4() const { return vint4(x,y,z,z); }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3ia( ZeroTy   ) : x(0), y(0), z(0) {}
    __forceinline Vec3ia( OneTy    ) : x(1), y(1), z(1) {}
    __forceinline Vec3ia( PosInfTy ) : x(0x7FFFFFFF), y(0x7FFFFFFF), z(0x7FFFFFFF) {}
    __forceinline Vec3ia( NegInfTy ) : x(0x80000000), y(0x80000000), z(0x80000000) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const int& operator []( const size_t index ) const { assert(index < 3); return (&x)[index]; }
    __forceinline       int& operator []( const size_t index )       { assert(index < 3); return (&x)[index]; }
  };


  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3ia operator +( const Vec3ia& a ) { return Vec3ia(+a.x,+a.y,+a.z); }
  __forceinline Vec3ia operator -( const Vec3ia& a ) { return Vec3ia(-a.x,-a.y,-a.z); }
  __forceinline Vec3ia abs       ( const Vec3ia& a ) { return Vec3ia(sycl::abs(a.x),sycl::abs(a.y),sycl::abs(a.z)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3ia operator +( const Vec3ia& a, const Vec3ia& b ) { return Vec3ia(a.x+b.x, a.y+b.y, a.z+b.z); }
  __forceinline Vec3ia operator +( const Vec3ia& a, const int     b ) { return a+Vec3ia(b); }
  __forceinline Vec3ia operator +( const int     a, const Vec3ia& b ) { return Vec3ia(a)+b; }

  __forceinline Vec3ia operator -( const Vec3ia& a, const Vec3ia& b ) { return Vec3ia(a.x-b.x, a.y-b.y, a.z-b.z); }
  __forceinline Vec3ia operator -( const Vec3ia& a, const int     b ) { return a-Vec3ia(b); }
  __forceinline Vec3ia operator -( const int     a, const Vec3ia& b ) { return Vec3ia(a)-b; }

  __forceinline Vec3ia operator *( const Vec3ia& a, const Vec3ia& b ) { return Vec3ia(a.x*b.x, a.y*b.y, a.z*b.z); }
  __forceinline Vec3ia operator *( const Vec3ia& a, const int     b ) { return a * Vec3ia(b); }
  __forceinline Vec3ia operator *( const int     a, const Vec3ia& b ) { return Vec3ia(a) * b; }

  __forceinline Vec3ia operator &( const Vec3ia& a, const Vec3ia& b ) { return Vec3ia(a.x&b.x, a.y&b.y, a.z&b.z); }
  __forceinline Vec3ia operator &( const Vec3ia& a, const int     b ) { return a & Vec3ia(b); }
  __forceinline Vec3ia operator &( const int     a, const Vec3ia& b ) { return Vec3ia(a) & b; }

  __forceinline Vec3ia operator |( const Vec3ia& a, const Vec3ia& b ) { return Vec3ia(a.x|b.x, a.y|b.y, a.z|b.z); }
  __forceinline Vec3ia operator |( const Vec3ia& a, const int     b ) { return a | Vec3ia(b); }
  __forceinline Vec3ia operator |( const int     a, const Vec3ia& b ) { return Vec3ia(a) | b; }

  __forceinline Vec3ia operator ^( const Vec3ia& a, const Vec3ia& b ) { return Vec3ia(a.x^b.x, a.y^b.y, a.z^b.z); }
  __forceinline Vec3ia operator ^( const Vec3ia& a, const int     b ) { return a ^ Vec3ia(b); }
  __forceinline Vec3ia operator ^( const int     a, const Vec3ia& b ) { return Vec3ia(a) ^ b; }

  __forceinline Vec3ia operator <<( const Vec3ia& a, const int n ) { return Vec3ia(a.x<<n, a.y<<n, a.z<<n); }
  __forceinline Vec3ia operator >>( const Vec3ia& a, const int n ) { return Vec3ia(a.x>>n, a.y>>n, a.z>>n); }

  __forceinline Vec3ia sll ( const Vec3ia& a, const int b ) { return Vec3ia(a.x<<b, a.y<<b, a.z<<b); }
  __forceinline Vec3ia sra ( const Vec3ia& a, const int b ) { return Vec3ia(a.x>>b, a.y>>b, a.z>>b); }
  __forceinline Vec3ia srl ( const Vec3ia& a, const int b ) { return Vec3ia(unsigned(a.x)>>b, unsigned(a.y)>>b, unsigned(a.z)>>b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3ia& operator +=( Vec3ia& a, const Vec3ia& b ) { return a = a + b; }
  __forceinline Vec3ia& operator +=( Vec3ia& a, const int&   b ) { return a = a + b; }
  
  __forceinline Vec3ia& operator -=( Vec3ia& a, const Vec3ia& b ) { return a = a - b; }
  __forceinline Vec3ia& operator -=( Vec3ia& a, const int&   b ) { return a = a - b; }
  
  __forceinline Vec3ia& operator *=( Vec3ia& a, const Vec3ia& b ) { return a = a * b; }
  __forceinline Vec3ia& operator *=( Vec3ia& a, const int&    b ) { return a = a * b; }
  
  __forceinline Vec3ia& operator &=( Vec3ia& a, const Vec3ia& b ) { return a = a & b; }
  __forceinline Vec3ia& operator &=( Vec3ia& a, const int&    b ) { return a = a & b; }
  
  __forceinline Vec3ia& operator |=( Vec3ia& a, const Vec3ia& b ) { return a = a | b; }
  __forceinline Vec3ia& operator |=( Vec3ia& a, const int&    b ) { return a = a | b; }
  
  __forceinline Vec3ia& operator <<=( Vec3ia& a, const int& b ) { return a = a << b; }
  __forceinline Vec3ia& operator >>=( Vec3ia& a, const int& b ) { return a = a >> b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline int reduce_add(const Vec3ia& v) { return v.x+v.y+v.z; }
  __forceinline int reduce_mul(const Vec3ia& v) { return v.x*v.y*v.z; }
  __forceinline int reduce_min(const Vec3ia& v) { return sycl::min(sycl::min(v.x,v.y),v.z); }
  __forceinline int reduce_max(const Vec3ia& v) { return sycl::max(sycl::max(v.x,v.y),v.z); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool operator ==( const Vec3ia& a, const Vec3ia& b ) { return a.x == b.x & a.y == b.y & a.z == b.z; }
  __forceinline bool operator !=( const Vec3ia& a, const Vec3ia& b ) { return a.x != b.x & a.y != b.y & a.z != b.z; }

/*
  __forceinline bool operator < ( const Vec3ia& a, const Vec3ia& b ) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    return false;
  }
*/
  __forceinline Vec3ba eq_mask( const Vec3ia& a, const Vec3ia& b ) { return Vec3ba(a.x == b.x, a.y == b.y, a.z == b.z); }
  __forceinline Vec3ba lt_mask( const Vec3ia& a, const Vec3ia& b ) { return Vec3ba(a.x <  b.x, a.y <  b.y, a.z <  b.z); }
  __forceinline Vec3ba gt_mask( const Vec3ia& a, const Vec3ia& b ) { return Vec3ba(a.x >  b.x, a.y >  b.y, a.z >  b.z); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline Vec3ia select( const Vec3ba& m, const Vec3ia& t, const Vec3ia& f ) {
    const int x = m.x ? t.x : f.x;
    const int y = m.y ? t.y : f.y;
    const int z = m.z ? t.z : f.z;
    return Vec3ia(x,y,z);
  }
  
  __forceinline Vec3ia min( const Vec3ia& a, const Vec3ia& b ) { return Vec3ia(sycl::min(a.x,b.x), sycl::min(a.y,b.y), sycl::min(a.z,b.z)); }
  __forceinline Vec3ia max( const Vec3ia& a, const Vec3ia& b ) { return Vec3ia(sycl::max(a.x,b.x), sycl::max(a.y,b.y), sycl::max(a.z,b.z)); }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////
  
  inline embree_ostream operator<<(embree_ostream cout, const Vec3ia& a) {
    return cout;
  }
}
