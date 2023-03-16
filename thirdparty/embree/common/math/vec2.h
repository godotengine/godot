// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "math.h"

namespace embree
{
  struct Vec2fa;
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Generic 2D vector Class
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> struct Vec2
  {
    enum { N = 2 };
    union {
      struct { T x, y; };
#if !(defined(__WIN32__) && _MSC_VER == 1800) // workaround for older VS 2013 compiler
      T components[N];
#endif
    };

    typedef T Scalar;

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec2( ) {}
    __forceinline explicit Vec2( const T& a             ) : x(a), y(a) {}
    __forceinline          Vec2( const T& x, const T& y ) : x(x), y(y) {}

    __forceinline Vec2( const Vec2& other ) { x = other.x; y = other.y; }
    __forceinline Vec2( const Vec2fa& other );

    template<typename T1> __forceinline Vec2( const Vec2<T1>& a ) : x(T(a.x)), y(T(a.y)) {}
    template<typename T1> __forceinline Vec2& operator =( const Vec2<T1>& other ) { x = other.x; y = other.y; return *this; }

    __forceinline Vec2& operator =( const Vec2& other ) { x = other.x; y = other.y; return *this; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec2( ZeroTy   ) : x(zero), y(zero) {}
    __forceinline Vec2( OneTy    ) : x(one),  y(one) {}
    __forceinline Vec2( PosInfTy ) : x(pos_inf), y(pos_inf) {}
    __forceinline Vec2( NegInfTy ) : x(neg_inf), y(neg_inf) {}

#if defined(__WIN32__) && _MSC_VER == 1800 // workaround for older VS 2013 compiler
	__forceinline const T& operator [](const size_t axis) const { assert(axis < 2); return (&x)[axis]; }
	__forceinline       T& operator [](const size_t axis)       { assert(axis < 2); return (&x)[axis]; }
#else
	__forceinline const T& operator [](const size_t axis) const { assert(axis < 2); return components[axis]; }
	__forceinline       T& operator [](const size_t axis )      { assert(axis < 2); return components[axis]; }
#endif
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec2<T> operator +( const Vec2<T>& a ) { return Vec2<T>(+a.x, +a.y); }
  template<typename T> __forceinline Vec2<T> operator -( const Vec2<T>& a ) { return Vec2<T>(-a.x, -a.y); }
  template<typename T> __forceinline Vec2<T> abs       ( const Vec2<T>& a ) { return Vec2<T>(abs  (a.x), abs  (a.y)); }
  template<typename T> __forceinline Vec2<T> rcp       ( const Vec2<T>& a ) { return Vec2<T>(rcp  (a.x), rcp  (a.y)); }
  template<typename T> __forceinline Vec2<T> rsqrt     ( const Vec2<T>& a ) { return Vec2<T>(rsqrt(a.x), rsqrt(a.y)); }
  template<typename T> __forceinline Vec2<T> sqrt      ( const Vec2<T>& a ) { return Vec2<T>(sqrt (a.x), sqrt (a.y)); }
  template<typename T> __forceinline Vec2<T> frac      ( const Vec2<T>& a ) { return Vec2<T>(frac (a.x), frac (a.y)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec2<T> operator +( const Vec2<T>& a, const Vec2<T>& b ) { return Vec2<T>(a.x + b.x, a.y + b.y); }
  template<typename T> __forceinline Vec2<T> operator +( const Vec2<T>& a, const       T& b ) { return Vec2<T>(a.x + b  , a.y + b  ); }
  template<typename T> __forceinline Vec2<T> operator +( const       T& a, const Vec2<T>& b ) { return Vec2<T>(a   + b.x, a   + b.y); }
  template<typename T> __forceinline Vec2<T> operator -( const Vec2<T>& a, const Vec2<T>& b ) { return Vec2<T>(a.x - b.x, a.y - b.y); }
  template<typename T> __forceinline Vec2<T> operator -( const Vec2<T>& a, const       T& b ) { return Vec2<T>(a.x - b  , a.y - b  ); }
  template<typename T> __forceinline Vec2<T> operator -( const       T& a, const Vec2<T>& b ) { return Vec2<T>(a   - b.x, a   - b.y); }
  template<typename T> __forceinline Vec2<T> operator *( const Vec2<T>& a, const Vec2<T>& b ) { return Vec2<T>(a.x * b.x, a.y * b.y); }
  template<typename T> __forceinline Vec2<T> operator *( const       T& a, const Vec2<T>& b ) { return Vec2<T>(a   * b.x, a   * b.y); }
  template<typename T> __forceinline Vec2<T> operator *( const Vec2<T>& a, const       T& b ) { return Vec2<T>(a.x * b  , a.y * b  ); }
  template<typename T> __forceinline Vec2<T> operator /( const Vec2<T>& a, const Vec2<T>& b ) { return Vec2<T>(a.x / b.x, a.y / b.y); }
  template<typename T> __forceinline Vec2<T> operator /( const Vec2<T>& a, const       T& b ) { return Vec2<T>(a.x / b  , a.y / b  ); }
  template<typename T> __forceinline Vec2<T> operator /( const       T& a, const Vec2<T>& b ) { return Vec2<T>(a   / b.x, a   / b.y); }

  template<typename T> __forceinline Vec2<T> min(const Vec2<T>& a, const Vec2<T>& b) { return Vec2<T>(min(a.x, b.x), min(a.y, b.y)); }
  template<typename T> __forceinline Vec2<T> max(const Vec2<T>& a, const Vec2<T>& b) { return Vec2<T>(max(a.x, b.x), max(a.y, b.y)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec2<T> madd  ( const Vec2<T>& a, const Vec2<T>& b, const Vec2<T>& c) { return Vec2<T>( madd(a.x,b.x,c.x), madd(a.y,b.y,c.y) ); }
  template<typename T> __forceinline Vec2<T> msub  ( const Vec2<T>& a, const Vec2<T>& b, const Vec2<T>& c) { return Vec2<T>( msub(a.x,b.x,c.x), msub(a.y,b.y,c.y) ); }
  template<typename T> __forceinline Vec2<T> nmadd ( const Vec2<T>& a, const Vec2<T>& b, const Vec2<T>& c) { return Vec2<T>(nmadd(a.x,b.x,c.x),nmadd(a.y,b.y,c.y) ); }
  template<typename T> __forceinline Vec2<T> nmsub ( const Vec2<T>& a, const Vec2<T>& b, const Vec2<T>& c) { return Vec2<T>(nmsub(a.x,b.x,c.x),nmsub(a.y,b.y,c.y) ); }

  template<typename T> __forceinline Vec2<T> madd  ( const T& a, const Vec2<T>& b, const Vec2<T>& c) { return Vec2<T>( madd(a,b.x,c.x), madd(a,b.y,c.y) ); }
  template<typename T> __forceinline Vec2<T> msub  ( const T& a, const Vec2<T>& b, const Vec2<T>& c) { return Vec2<T>( msub(a,b.x,c.x), msub(a,b.y,c.y) ); }
  template<typename T> __forceinline Vec2<T> nmadd ( const T& a, const Vec2<T>& b, const Vec2<T>& c) { return Vec2<T>(nmadd(a,b.x,c.x),nmadd(a,b.y,c.y) ); }
  template<typename T> __forceinline Vec2<T> nmsub ( const T& a, const Vec2<T>& b, const Vec2<T>& c) { return Vec2<T>(nmsub(a,b.x,c.x),nmsub(a,b.y,c.y) ); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec2<T>& operator +=( Vec2<T>& a, const Vec2<T>& b ) { a.x += b.x; a.y += b.y; return a; }
  template<typename T> __forceinline Vec2<T>& operator -=( Vec2<T>& a, const Vec2<T>& b ) { a.x -= b.x; a.y -= b.y; return a; }
  template<typename T> __forceinline Vec2<T>& operator *=( Vec2<T>& a, const       T& b ) { a.x *= b  ; a.y *= b  ; return a; }
  template<typename T> __forceinline Vec2<T>& operator /=( Vec2<T>& a, const       T& b ) { a.x /= b  ; a.y /= b  ; return a; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline T reduce_add( const Vec2<T>& a ) { return a.x + a.y; }
  template<typename T> __forceinline T reduce_mul( const Vec2<T>& a ) { return a.x * a.y; }
  template<typename T> __forceinline T reduce_min( const Vec2<T>& a ) { return min(a.x, a.y); }
  template<typename T> __forceinline T reduce_max( const Vec2<T>& a ) { return max(a.x, a.y); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline bool operator ==( const Vec2<T>& a, const Vec2<T>& b ) { return a.x == b.x && a.y == b.y; }
  template<typename T> __forceinline bool operator !=( const Vec2<T>& a, const Vec2<T>& b ) { return a.x != b.x || a.y != b.y; }
  template<typename T> __forceinline bool operator < ( const Vec2<T>& a, const Vec2<T>& b ) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return false;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Shift Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec2<T> shift_right_1( const Vec2<T>& a ) {
    return Vec2<T>(shift_right_1(a.x),shift_right_1(a.y));
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidean Space Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline T       dot      ( const Vec2<T>& a, const Vec2<T>& b ) { return madd(a.x,b.x,a.y*b.y); }
  template<typename T> __forceinline Vec2<T> cross    ( const Vec2<T>& a )                   { return Vec2<T>(-a.y,a.x); } 
  template<typename T> __forceinline T       length   ( const Vec2<T>& a )                   { return sqrt(dot(a,a)); }
  template<typename T> __forceinline Vec2<T> normalize( const Vec2<T>& a )                   { return a*rsqrt(dot(a,a)); }
  template<typename T> __forceinline T       distance ( const Vec2<T>& a, const Vec2<T>& b ) { return length(a-b); }
  template<typename T> __forceinline T       det      ( const Vec2<T>& a, const Vec2<T>& b ) { return a.x*b.y - a.y*b.x; }

  template<typename T> __forceinline Vec2<T> normalize_safe( const Vec2<T>& a ) {
    const T d = dot(a,a); return select(d == T( zero ),a, a*rsqrt(d) );
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec2<T> select ( bool s, const Vec2<T>& t, const Vec2<T>& f ) {
    return Vec2<T>(select(s,t.x,f.x),select(s,t.y,f.y));
  }

  template<typename T> __forceinline Vec2<T> select ( const Vec2<bool>& s, const Vec2<T>& t, const Vec2<T>& f ) {
    return Vec2<T>(select(s.x,t.x,f.x),select(s.y,t.y,f.y));
  }

  template<typename T> __forceinline Vec2<T> select ( const typename T::Bool& s, const Vec2<T>& t, const Vec2<T>& f ) {
    return Vec2<T>(select(s,t.x,f.x),select(s,t.y,f.y));
  }

  template<typename T>
    __forceinline Vec2<T> lerp(const Vec2<T>& v0, const Vec2<T>& v1, const T& t) {
    return madd(Vec2<T>(T(1.0f)-t),v0,t*v1);
  }

  template<typename T> __forceinline int maxDim ( const Vec2<T>& a )
  {
    const Vec2<T> b = abs(a);
    if (b.x > b.y) return 0;
    else return 1;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline embree_ostream operator<<(embree_ostream cout, const Vec2<T>& a) {
    return cout << "(" << a.x << ", " << a.y << ")";
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Default template instantiations
  ////////////////////////////////////////////////////////////////////////////////

  typedef Vec2<bool > Vec2b;
  typedef Vec2<int  > Vec2i;
  typedef Vec2<float> Vec2f;
}

#include "vec2fa.h"

#if defined(__SSE__) || defined(__ARM_NEON)
#include "../simd/sse.h"
#endif

#if defined(__AVX__)
#include "../simd/avx.h"
#endif

#if defined(__AVX512F__)
#include "../simd/avx512.h"
#endif

namespace embree
{
  template<> __forceinline Vec2<float>::Vec2(const Vec2fa& a) : x(a.x), y(a.y) {}

#if defined(__SSE__) || defined(__ARM_NEON)
  template<> __forceinline Vec2<vfloat4>::Vec2(const Vec2fa& a) : x(a.x), y(a.y) {}
#endif

#if defined(__AVX__)
  template<> __forceinline Vec2<vfloat8>::Vec2(const Vec2fa& a) : x(a.x), y(a.y) {}
#endif

#if defined(__AVX512F__)
  template<> __forceinline Vec2<vfloat16>::Vec2(const Vec2fa& a) : x(a.x), y(a.y) {}
#endif
}
