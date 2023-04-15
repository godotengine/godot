// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "math.h"
#include "vec3.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// Generic 4D vector Class
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> struct Vec4
  {
    enum { N = 4 };    
    union {
      struct { T x, y, z, w; };
#if !(defined(__WIN32__) && _MSC_VER == 1800) // workaround for older VS 2013 compiler
      T components[N];
#endif
    };

    typedef T Scalar;

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec4( ) {}
    __forceinline explicit Vec4( const T& a                                     ) : x(a), y(a), z(a), w(a) {}
    __forceinline          Vec4( const T& x, const T& y, const T& z, const T& w ) : x(x), y(y), z(z), w(w) {}
    __forceinline          Vec4( const Vec3<T>& xyz, const T& w ) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}

    __forceinline Vec4( const Vec4& other ) { x = other.x; y = other.y; z = other.z; w = other.w; }
    __forceinline Vec4( const Vec3fx& other );

    template<typename T1> __forceinline Vec4( const Vec4<T1>& a ) : x(T(a.x)), y(T(a.y)), z(T(a.z)), w(T(a.w)) {}
    template<typename T1> __forceinline Vec4& operator =(const Vec4<T1>& other) { x = other.x; y = other.y; z = other.z; w = other.w; return *this; }

    __forceinline Vec4& operator =(const Vec4& other) { x = other.x; y = other.y; z = other.z; w = other.w; return *this; }

    __forceinline operator Vec3<T> () const { return Vec3<T>(x,y,z); }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec4( ZeroTy   ) : x(zero), y(zero), z(zero), w(zero) {}
    __forceinline Vec4( OneTy    ) : x(one),  y(one),  z(one),  w(one) {}
    __forceinline Vec4( PosInfTy ) : x(pos_inf), y(pos_inf), z(pos_inf), w(pos_inf) {}
    __forceinline Vec4( NegInfTy ) : x(neg_inf), y(neg_inf), z(neg_inf), w(neg_inf) {}

#if defined(__WIN32__) && (_MSC_VER == 1800) // workaround for older VS 2013 compiler
	__forceinline const T& operator [](const size_t axis) const { assert(axis < 4); return (&x)[axis]; }
	__forceinline       T& operator [](const size_t axis)       { assert(axis < 4); return (&x)[axis]; }
#else
	__forceinline const T& operator [](const size_t axis ) const { assert(axis < 4); return components[axis]; }
	__forceinline       T& operator [](const size_t axis)        { assert(axis < 4); return components[axis]; }
#endif

    ////////////////////////////////////////////////////////////////////////////////
    /// Swizzles
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3<T> xyz() const { return Vec3<T>(x, y, z); }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec4<T> operator +( const Vec4<T>& a ) { return Vec4<T>(+a.x, +a.y, +a.z, +a.w); }
  template<typename T> __forceinline Vec4<T> operator -( const Vec4<T>& a ) { return Vec4<T>(-a.x, -a.y, -a.z, -a.w); }
  template<typename T> __forceinline Vec4<T> abs       ( const Vec4<T>& a ) { return Vec4<T>(abs  (a.x), abs  (a.y), abs  (a.z), abs  (a.w)); }
  template<typename T> __forceinline Vec4<T> rcp       ( const Vec4<T>& a ) { return Vec4<T>(rcp  (a.x), rcp  (a.y), rcp  (a.z), rcp  (a.w)); }
  template<typename T> __forceinline Vec4<T> rsqrt     ( const Vec4<T>& a ) { return Vec4<T>(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z), rsqrt(a.w)); }
  template<typename T> __forceinline Vec4<T> sqrt      ( const Vec4<T>& a ) { return Vec4<T>(sqrt (a.x), sqrt (a.y), sqrt (a.z), sqrt (a.w)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec4<T> operator +( const Vec4<T>& a, const Vec4<T>& b ) { return Vec4<T>(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
  template<typename T> __forceinline Vec4<T> operator -( const Vec4<T>& a, const Vec4<T>& b ) { return Vec4<T>(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
  template<typename T> __forceinline Vec4<T> operator *( const Vec4<T>& a, const Vec4<T>& b ) { return Vec4<T>(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
  template<typename T> __forceinline Vec4<T> operator *( const       T& a, const Vec4<T>& b ) { return Vec4<T>(a   * b.x, a   * b.y, a   * b.z, a   * b.w); }
  template<typename T> __forceinline Vec4<T> operator *( const Vec4<T>& a, const       T& b ) { return Vec4<T>(a.x * b  , a.y * b  , a.z * b  , a.w * b  ); }
  template<typename T> __forceinline Vec4<T> operator /( const Vec4<T>& a, const Vec4<T>& b ) { return Vec4<T>(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
  template<typename T> __forceinline Vec4<T> operator /( const Vec4<T>& a, const       T& b ) { return Vec4<T>(a.x / b  , a.y / b  , a.z / b  , a.w / b  ); }
  template<typename T> __forceinline Vec4<T> operator /( const       T& a, const Vec4<T>& b ) { return Vec4<T>(a   / b.x, a   / b.y, a   / b.z, a   / b.w); }

  template<typename T> __forceinline Vec4<T> min(const Vec4<T>& a, const Vec4<T>& b) { return Vec4<T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)); }
  template<typename T> __forceinline Vec4<T> max(const Vec4<T>& a, const Vec4<T>& b) { return Vec4<T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec4<T> madd  ( const Vec4<T>& a, const Vec4<T>& b, const Vec4<T>& c) { return Vec4<T>( madd(a.x,b.x,c.x), madd(a.y,b.y,c.y), madd(a.z,b.z,c.z), madd(a.w,b.w,c.w)); }
  template<typename T> __forceinline Vec4<T> msub  ( const Vec4<T>& a, const Vec4<T>& b, const Vec4<T>& c) { return Vec4<T>( msub(a.x,b.x,c.x), msub(a.y,b.y,c.y), msub(a.z,b.z,c.z), msub(a.w,b.w,c.w)); }
  template<typename T> __forceinline Vec4<T> nmadd ( const Vec4<T>& a, const Vec4<T>& b, const Vec4<T>& c) { return Vec4<T>(nmadd(a.x,b.x,c.x),nmadd(a.y,b.y,c.y),nmadd(a.z,b.z,c.z),nmadd(a.w,b.w,c.w)); }
  template<typename T> __forceinline Vec4<T> nmsub ( const Vec4<T>& a, const Vec4<T>& b, const Vec4<T>& c) { return Vec4<T>(nmsub(a.x,b.x,c.x),nmsub(a.y,b.y,c.y),nmsub(a.z,b.z,c.z),nmsub(a.w,b.w,c.w)); }

  template<typename T> __forceinline Vec4<T> madd  ( const T& a, const Vec4<T>& b, const Vec4<T>& c) { return Vec4<T>( madd(a,b.x,c.x), madd(a,b.y,c.y), madd(a,b.z,c.z), madd(a,b.w,c.w)); }
  template<typename T> __forceinline Vec4<T> msub  ( const T& a, const Vec4<T>& b, const Vec4<T>& c) { return Vec4<T>( msub(a,b.x,c.x), msub(a,b.y,c.y), msub(a,b.z,c.z), msub(a,b.w,c.w)); }
  template<typename T> __forceinline Vec4<T> nmadd ( const T& a, const Vec4<T>& b, const Vec4<T>& c) { return Vec4<T>(nmadd(a,b.x,c.x),nmadd(a,b.y,c.y),nmadd(a,b.z,c.z),nmadd(a,b.w,c.w)); }
  template<typename T> __forceinline Vec4<T> nmsub ( const T& a, const Vec4<T>& b, const Vec4<T>& c) { return Vec4<T>(nmsub(a,b.x,c.x),nmsub(a,b.y,c.y),nmsub(a,b.z,c.z),nmsub(a,b.w,c.w)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec4<T>& operator +=( Vec4<T>& a, const Vec4<T>& b ) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
  template<typename T> __forceinline Vec4<T>& operator -=( Vec4<T>& a, const Vec4<T>& b ) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
  template<typename T> __forceinline Vec4<T>& operator *=( Vec4<T>& a, const       T& b ) { a.x *= b  ; a.y *= b  ; a.z *= b  ; a.w *= b  ; return a; }
  template<typename T> __forceinline Vec4<T>& operator /=( Vec4<T>& a, const       T& b ) { a.x /= b  ; a.y /= b  ; a.z /= b  ; a.w /= b  ; return a; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline T reduce_add( const Vec4<T>& a ) { return a.x + a.y + a.z + a.w; }
  template<typename T> __forceinline T reduce_mul( const Vec4<T>& a ) { return a.x * a.y * a.z * a.w; }
  template<typename T> __forceinline T reduce_min( const Vec4<T>& a ) { return min(a.x, a.y, a.z, a.w); }
  template<typename T> __forceinline T reduce_max( const Vec4<T>& a ) { return max(a.x, a.y, a.z, a.w); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline bool operator ==( const Vec4<T>& a, const Vec4<T>& b ) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
  template<typename T> __forceinline bool operator !=( const Vec4<T>& a, const Vec4<T>& b ) { return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w; }
  template<typename T> __forceinline bool operator < ( const Vec4<T>& a, const Vec4<T>& b ) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    if (a.w != b.w) return a.w < b.w;
    return false;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Shift Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec4<T> shift_right_1( const Vec4<T>& a ) {
    return Vec4<T>(shift_right_1(a.x),shift_right_1(a.y),shift_right_1(a.z),shift_right_1(a.w));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidean Space Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline T       dot      ( const Vec4<T>& a, const Vec4<T>& b ) { return madd(a.x,b.x,madd(a.y,b.y,madd(a.z,b.z,a.w*b.w))); }

  template<typename T> __forceinline T       length   ( const Vec4<T>& a )                   { return sqrt(dot(a,a)); }
  template<typename T> __forceinline Vec4<T> normalize( const Vec4<T>& a )                   { return a*rsqrt(dot(a,a)); }
  template<typename T> __forceinline T       distance ( const Vec4<T>& a, const Vec4<T>& b ) { return length(a-b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec4<T> select ( bool s, const Vec4<T>& t, const Vec4<T>& f ) {
    return Vec4<T>(select(s,t.x,f.x),select(s,t.y,f.y),select(s,t.z,f.z),select(s,t.w,f.w));
  }

  template<typename T> __forceinline Vec4<T> select ( const Vec4<bool>& s, const Vec4<T>& t, const Vec4<T>& f ) {
    return Vec4<T>(select(s.x,t.x,f.x),select(s.y,t.y,f.y),select(s.z,t.z,f.z),select(s.w,t.w,f.w));
  }

  template<typename T> __forceinline Vec4<T> select ( const typename T::Bool& s, const Vec4<T>& t, const Vec4<T>& f ) {
    return Vec4<T>(select(s,t.x,f.x),select(s,t.y,f.y),select(s,t.z,f.z),select(s,t.w,f.w));
  }

  template<typename T>
    __forceinline Vec4<T> lerp(const Vec4<T>& v0, const Vec4<T>& v1, const T& t) {
    return madd(Vec4<T>(T(1.0f)-t),v0,t*v1);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline embree_ostream operator<<(embree_ostream cout, const Vec4<T>& a) {
    return cout << "(" << a.x << ", " << a.y << ", " << a.z << ", " << a.w << ")";
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Default template instantiations
  ////////////////////////////////////////////////////////////////////////////////

  typedef Vec4<bool         > Vec4b;
  typedef Vec4<unsigned char> Vec4uc;
  typedef Vec4<int          > Vec4i;
  typedef Vec4<float        > Vec4f;
}

#include "vec3ba.h"
#include "vec3ia.h"
#include "vec3fa.h"

////////////////////////////////////////////////////////////////////////////////
/// SSE / AVX / MIC specializations
////////////////////////////////////////////////////////////////////////////////

#if defined(__SSE__) || defined(__ARM_NEON)
#include "../simd/sse.h"
#endif

#if defined __AVX__
#include "../simd/avx.h"
#endif

#if defined __AVX512F__
#include "../simd/avx512.h"
#endif

namespace embree
{
  template<> __forceinline Vec4<float>::Vec4( const Vec3fx& a ) { x = a.x; y = a.y; z = a.z; w = a.w; }

#if defined(__AVX__)
  template<> __forceinline Vec4<vfloat4>::Vec4( const Vec3fx& a ) {
    x = a.x; y = a.y; z = a.z; w = a.w;
  }
#elif defined(__SSE__) || defined(__ARM_NEON)
  template<> __forceinline Vec4<vfloat4>::Vec4( const Vec3fx& a ) {
    const vfloat4 v = vfloat4(a.m128); x = shuffle<0,0,0,0>(v); y = shuffle<1,1,1,1>(v); z = shuffle<2,2,2,2>(v); w = shuffle<3,3,3,3>(v);
  }
#endif

#if defined(__AVX__)
  template<> __forceinline Vec4<vfloat8>::Vec4( const Vec3fx& a ) {
    x = a.x; y = a.y; z = a.z; w = a.w;
  }
#endif

#if defined(__AVX512F__)
  template<> __forceinline Vec4<vfloat16>::Vec4( const Vec3fx& a ) : x(a.x), y(a.y), z(a.z), w(a.w) {}
#endif
}
