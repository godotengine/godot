// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "constants.h"
#include "col3.h"
#include "col4.h"

#include "../simd/sse.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// SSE RGBA Color Class
  ////////////////////////////////////////////////////////////////////////////////

  struct Color4
  {
    union {
      __m128 m128;
      struct { float r,g,b,a; };
    };

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Color4 () {}
    __forceinline Color4 ( const __m128 a ) : m128(a) {}

    __forceinline explicit Color4 (const float v) : m128(_mm_set1_ps(v)) {}
    __forceinline          Color4 (const float r, const float g, const float b, const float a) : m128(_mm_set_ps(a,b,g,r)) {}

    __forceinline explicit Color4 ( const Col3uc& other ) { m128 = _mm_mul_ps(_mm_set_ps(255.0f,other.b,other.g,other.r),_mm_set1_ps(one_over_255)); }
    __forceinline explicit Color4 ( const Col3f&  other ) { m128 = _mm_set_ps(1.0f,other.b,other.g,other.r); }
    __forceinline explicit Color4 ( const Col4uc& other ) { m128 = _mm_mul_ps(_mm_set_ps(other.a,other.b,other.g,other.r),_mm_set1_ps(one_over_255)); }
    __forceinline explicit Color4 ( const Col4f&  other ) { m128 = _mm_set_ps(other.a,other.b,other.g,other.r); }

    __forceinline Color4           ( const Color4& other ) : m128(other.m128) {}
    __forceinline Color4& operator=( const Color4& other ) { m128 = other.m128; return *this; }

    __forceinline operator const __m128&() const { return m128; }
    __forceinline operator       __m128&()       { return m128; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Set
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline void set(Col3f& d) const { d.r = r; d.g = g; d.b = b; }
    __forceinline void set(Col4f& d) const { d.r = r; d.g = g; d.b = b; d.a = a; }
    __forceinline void set(Col3uc& d) const 
    {
      vfloat4 s = clamp(vfloat4(m128))*255.0f;
      d.r = (unsigned char)(s[0]); 
      d.g = (unsigned char)(s[1]); 
      d.b = (unsigned char)(s[2]); 
    }
    __forceinline void set(Col4uc& d) const 
    {
      vfloat4 s = clamp(vfloat4(m128))*255.0f;
      d.r = (unsigned char)(s[0]); 
      d.g = (unsigned char)(s[1]); 
      d.b = (unsigned char)(s[2]); 
      d.a = (unsigned char)(s[3]); 
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Color4( ZeroTy   ) : m128(_mm_set1_ps(0.0f)) {}
    __forceinline Color4( OneTy    ) : m128(_mm_set1_ps(1.0f)) {}
    __forceinline Color4( PosInfTy ) : m128(_mm_set1_ps(pos_inf)) {}
    __forceinline Color4( NegInfTy ) : m128(_mm_set1_ps(neg_inf)) {}
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// SSE RGB Color Class
  ////////////////////////////////////////////////////////////////////////////////

  struct Color
  {
    union {
      __m128 m128;
      struct { float r,g,b; };
    };

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Color () {}
    __forceinline Color ( const __m128 a ) : m128(a) {}

    __forceinline explicit Color  (const float v)                               : m128(_mm_set1_ps(v)) {}
    __forceinline          Color  (const float r, const float g, const float b) : m128(_mm_set_ps(0.0f,b,g,r)) {}

    __forceinline Color           ( const Color& other ) : m128(other.m128) {}
    __forceinline Color& operator=( const Color& other ) { m128 = other.m128; return *this; }

    __forceinline Color           ( const Color4& other ) : m128(other.m128) {}
    __forceinline Color& operator=( const Color4& other ) { m128 = other.m128; return *this; }

    __forceinline operator const __m128&() const { return m128; }
    __forceinline operator       __m128&()       { return m128; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Set
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline void set(Col3f& d) const { d.r = r; d.g = g; d.b = b; }
    __forceinline void set(Col4f& d) const { d.r = r; d.g = g; d.b = b; d.a = 1.0f; }
    __forceinline void set(Col3uc& d) const 
    { 
      vfloat4 s = clamp(vfloat4(m128))*255.0f;
      d.r = (unsigned char)(s[0]); 
      d.g = (unsigned char)(s[1]); 
      d.b = (unsigned char)(s[2]); 
    }
    __forceinline void set(Col4uc& d) const 
    { 
      vfloat4 s = clamp(vfloat4(m128))*255.0f;
      d.r = (unsigned char)(s[0]); 
      d.g = (unsigned char)(s[1]); 
      d.b = (unsigned char)(s[2]); 
      d.a = 255; 
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Color( ZeroTy   ) : m128(_mm_set1_ps(0.0f)) {}
    __forceinline Color( OneTy    ) : m128(_mm_set1_ps(1.0f)) {}
    __forceinline Color( PosInfTy ) : m128(_mm_set1_ps(pos_inf)) {}
    __forceinline Color( NegInfTy ) : m128(_mm_set1_ps(neg_inf)) {}
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const Color operator +( const Color& a ) { return a; }
  __forceinline const Color operator -( const Color& a ) {
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_xor_ps(a.m128, mask);
  }
  __forceinline const Color abs  ( const Color& a ) {
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    return _mm_and_ps(a.m128, mask);
  }
  __forceinline const Color rcp  ( const Color& a )
  {
#if defined(__aarch64__)
    __m128 reciprocal = _mm_rcp_ps(a.m128);
    reciprocal = vmulq_f32(vrecpsq_f32(a.m128, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(a.m128, reciprocal), reciprocal);
    return (const Color)reciprocal;
#else
#if defined(__AVX512VL__)
    const Color r = _mm_rcp14_ps(a.m128);
#else
    const Color r = _mm_rcp_ps(a.m128);
#endif
    return _mm_add_ps(r,_mm_mul_ps(r, _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(a, r))));   // computes r + r * (1 - a * r)

#endif  //defined(__aarch64__)
  }
  __forceinline const Color rsqrt( const Color& a )
  {
#if defined(__aarch64__)
    __m128 r = _mm_rsqrt_ps(a.m128);
    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(a.m128, r), r));
    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(a.m128, r), r));
    return r;
#else

#if defined(__AVX512VL__)
    __m128 r = _mm_rsqrt14_ps(a.m128);
#else
    __m128 r = _mm_rsqrt_ps(a.m128);
#endif
    return _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f),r), _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));

#endif  //defined(__aarch64__)
  }
  __forceinline const Color sqrt ( const Color& a ) { return _mm_sqrt_ps(a.m128); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const Color operator +( const Color& a, const Color& b ) { return _mm_add_ps(a.m128, b.m128); }
  __forceinline const Color operator -( const Color& a, const Color& b ) { return _mm_sub_ps(a.m128, b.m128); }
  __forceinline const Color operator *( const Color& a, const Color& b ) { return _mm_mul_ps(a.m128, b.m128); }
  __forceinline const Color operator *( const Color& a, const float  b ) { return a * Color(b); }
  __forceinline const Color operator *( const float  a, const Color& b ) { return Color(a) * b; }
  __forceinline const Color operator /( const Color& a, const Color& b ) { return a * rcp(b); }
  __forceinline const Color operator /( const Color& a, const float  b ) { return a * rcp(b); }

  __forceinline const Color min( const Color& a, const Color& b ) { return _mm_min_ps(a.m128,b.m128); }
  __forceinline const Color max( const Color& a, const Color& b ) { return _mm_max_ps(a.m128,b.m128); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const Color operator+=(Color& a, const Color& b) { return a = a + b; }
  __forceinline const Color operator-=(Color& a, const Color& b) { return a = a - b; }
  __forceinline const Color operator*=(Color& a, const Color& b) { return a = a * b; }
  __forceinline const Color operator/=(Color& a, const Color& b) { return a = a / b; }
  __forceinline const Color operator*=(Color& a, const float b      ) { return a = a * b; }
  __forceinline const Color operator/=(Color& a, const float b      ) { return a = a / b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline float reduce_add(const Color& v) { return v.r+v.g+v.b; }
  __forceinline float reduce_mul(const Color& v) { return v.r*v.g*v.b; }
  __forceinline float reduce_min(const Color& v) { return min(v.r,v.g,v.b); }
  __forceinline float reduce_max(const Color& v) { return max(v.r,v.g,v.b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool operator ==( const Color& a, const Color& b ) { return (_mm_movemask_ps(_mm_cmpeq_ps (a.m128, b.m128)) & 7) == 7; }
  __forceinline bool operator !=( const Color& a, const Color& b ) { return (_mm_movemask_ps(_mm_cmpneq_ps(a.m128, b.m128)) & 7) != 0; }
  __forceinline bool operator < ( const Color& a, const Color& b ) {
    if (a.r != b.r) return a.r < b.r;
    if (a.g != b.g) return a.g < b.g;
    if (a.b != b.b) return a.b < b.b;
    return false;
  }

   ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const Color select( bool s, const Color& t, const Color& f ) {
    __m128 mask = s ? _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128())) : _mm_setzero_ps();
    return blendv_ps(f, t, mask);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Special Operators
  ////////////////////////////////////////////////////////////////////////////////

  /*! computes luminance of a color */
  __forceinline float luminance (const Color& a) { return madd(0.212671f,a.r,madd(0.715160f,a.g,0.072169f*a.b)); }

  /*! output operator */
  __forceinline embree_ostream operator<<(embree_ostream cout, const Color& a) {
    return cout << "(" << a.r << ", " << a.g << ", " << a.b << ")";
  }
}
