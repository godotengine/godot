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
    struct { float r,g,b,a; };

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Color4 () {}
    //__forceinline Color4 ( const __m128 a ) : m128(a) {}

    __forceinline explicit Color4 (const float v) : r(v), g(v), b(v), a(v) {}
    __forceinline          Color4 (const float r, const float g, const float b, const float a) : r(r), g(g), b(b), a(a) {}

    __forceinline explicit Color4 ( const Col3uc& other ) : r(other.r/255.0f), g(other.g/255.0f), b(other.b/255.0f), a(1.0f) {}
    __forceinline explicit Color4 ( const Col3f&  other ) : r(other.r), g(other.g), b(other.b), a(1.0f) {}
    __forceinline explicit Color4 ( const Col4uc& other ) : r(other.r/255.0f), g(other.g/255.0f), b(other.b/255.0f), a(other.a/255.0f) {}
    __forceinline explicit Color4 ( const Col4f&  other ) : r(other.r), g(other.g), b(other.b), a(other.a) {}

    //__forceinline Color4           ( const Color4& other ) : m128(other.m128) {}
    //__forceinline Color4& operator=( const Color4& other ) { m128 = other.m128; return *this; }

    //__forceinline operator const __m128&() const { return m128; }
    //__forceinline operator       __m128&()       { return m128; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Set
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline void set(Col3f& d) const { d.r = r; d.g = g; d.b = b; }
    __forceinline void set(Col4f& d) const { d.r = r; d.g = g; d.b = b; d.a = a; }

    __forceinline void set(Col3uc& d) const 
    {
      d.r = (unsigned char)(clamp(r)*255.0f); 
      d.g = (unsigned char)(clamp(g)*255.0f); 
      d.b = (unsigned char)(clamp(b)*255.0f);
    }
    
    __forceinline void set(Col4uc& d) const 
    {
      d.r = (unsigned char)(clamp(r)*255.0f); 
      d.g = (unsigned char)(clamp(g)*255.0f); 
      d.b = (unsigned char)(clamp(b)*255.0f); 
      d.a = (unsigned char)(clamp(a)*255.0f);
    }
    __forceinline void set(float &f) const
    {
      f = 0.2126f*r+0.7125f*g+0.0722f*b; // sRGB luminance.
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Color4( ZeroTy   ) : r(0.0f), g(0.0f), b(0.0f), a(0.0f) {}
    __forceinline Color4( OneTy    ) : r(1.0f), g(1.0f), b(1.0f), a(1.0f) {}
    //__forceinline Color4( PosInfTy ) : m128(_mm_set1_ps(pos_inf)) {}
    //__forceinline Color4( NegInfTy ) : m128(_mm_set1_ps(neg_inf)) {}
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// SSE RGB Color Class
  ////////////////////////////////////////////////////////////////////////////////

  struct Color
  {
    struct { float r,g,b; };
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Color () {}
    //__forceinline Color ( const __m128 a ) : m128(a) {}

    __forceinline explicit Color  (const float v)  : r(v), g(v), b(v) {}
    __forceinline          Color  (const float r, const float g, const float b) : r(r), g(g), b(b) {}

    //__forceinline Color           ( const Color& other ) : m128(other.m128) {}
    //__forceinline Color& operator=( const Color& other ) { m128 = other.m128; return *this; }

    //__forceinline Color           ( const Color4& other ) : m128(other.m128) {}
    //__forceinline Color& operator=( const Color4& other ) { m128 = other.m128; return *this; }

    //__forceinline operator const __m128&() const { return m128; }
    //__forceinline operator       __m128&()       { return m128; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Set
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline void set(Col3f& d) const { d.r = r; d.g = g; d.b = b; }
    __forceinline void set(Col4f& d) const { d.r = r; d.g = g; d.b = b; d.a = 1.0f; }

#if 0
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
#endif

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Color( ZeroTy   ) : r(0.0f), g(0.0f), b(0.0f) {}
    __forceinline Color( OneTy    ) : r(1.0f), g(1.0f), b(1.0f) {}
    //__forceinline Color( PosInfTy ) : m128(_mm_set1_ps(pos_inf)) {}
    //__forceinline Color( NegInfTy ) : m128(_mm_set1_ps(neg_inf)) {}
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const Color operator +( const Color& a ) { return a; }
  __forceinline const Color operator -( const Color& a ) { return Color(-a.r, -a.g, -a.b); }
  __forceinline const Color abs  ( const Color& a ) { return Color(abs(a.r), abs(a.g), abs(a.b)); }
  __forceinline const Color rcp  ( const Color& a ) { return Color(1.0f/a.r, 1.0f/a.g, 1.0f/a.b); }
  __forceinline const Color rsqrt( const Color& a ) { return Color(1.0f/sqrt(a.r), 1.0f/sqrt(a.g), 1.0f/sqrt(a.b)); }
  __forceinline const Color sqrt ( const Color& a ) { return Color(sqrt(a.r), sqrt(a.g), sqrt(a.b)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const Color operator +( const Color& a, const Color& b ) { return Color(a.r+b.r, a.g+b.g, a.b+b.b); }
  __forceinline const Color operator -( const Color& a, const Color& b ) { return Color(a.r-b.r, a.g-b.g, a.b-b.b); }
  __forceinline const Color operator *( const Color& a, const Color& b ) { return Color(a.r*b.r, a.g*b.g, a.b*b.b); }
  __forceinline const Color operator *( const Color& a, const float  b ) { return a * Color(b); }
  __forceinline const Color operator *( const float  a, const Color& b ) { return Color(a) * b; }
  __forceinline const Color operator /( const Color& a, const Color& b ) { return a * rcp(b); }
  __forceinline const Color operator /( const Color& a, const float  b ) { return a * rcp(b); }

  __forceinline const Color min( const Color& a, const Color& b ) { return Color(min(a.r,b.r), min(a.g,b.g), min(a.b,b.b)); }
  __forceinline const Color max( const Color& a, const Color& b ) { return Color(max(a.r,b.r), max(a.g,b.g), max(a.b,b.b)); }

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

  __forceinline bool operator ==( const Color& a, const Color& b ) { return a.r == b.r && a.g == b.g && a.b == b.b; }
  __forceinline bool operator !=( const Color& a, const Color& b ) { return a.r != b.r || a.g != b.g || a.b != b.b; }
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
    return s ? t : f;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Special Operators
  ////////////////////////////////////////////////////////////////////////////////

  /*! computes luminance of a color */
  __forceinline float luminance (const Color& a) { return madd(0.212671f,a.r,madd(0.715160f,a.g,0.072169f*a.b)); }

  /*! output operator */
  inline std::ostream& operator<<(std::ostream& cout, const Color& a) {
    return cout << "(" << a.r << ", " << a.g << ", " << a.b << ")";
  }
}
