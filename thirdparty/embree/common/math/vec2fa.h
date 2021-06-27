// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../sys/alloc.h"
#include "math.h"
#include "../simd/sse.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// SSE Vec2fa Type
  ////////////////////////////////////////////////////////////////////////////////

  struct __aligned(16) Vec2fa
  {
    ALIGNED_STRUCT_(16);

    typedef float Scalar;
    enum { N = 2 };
    union {
      __m128 m128;
      struct { float x,y,az,aw; };
    };

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec2fa( ) {}
    __forceinline Vec2fa( const __m128 a ) : m128(a) {}

    __forceinline Vec2fa            ( const Vec2<float>& other  ) { x = other.x; y = other.y; }
    __forceinline Vec2fa& operator =( const Vec2<float>& other ) { x = other.x; y = other.y; return *this; }

    __forceinline Vec2fa            ( const Vec2fa& other ) { m128 = other.m128; }
    __forceinline Vec2fa& operator =( const Vec2fa& other ) { m128 = other.m128; return *this; }

    __forceinline explicit Vec2fa( const float a ) : m128(_mm_set1_ps(a)) {}
    __forceinline          Vec2fa( const float x, const float y) : m128(_mm_set_ps(y, y, y, x)) {}

    __forceinline explicit Vec2fa( const __m128i a ) : m128(_mm_cvtepi32_ps(a)) {}

    __forceinline operator const __m128&() const { return m128; }
    __forceinline operator       __m128&()       { return m128; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline Vec2fa load( const void* const a ) {
      return Vec2fa(_mm_and_ps(_mm_load_ps((float*)a),_mm_castsi128_ps(_mm_set_epi32(0, 0, -1, -1))));
    }

    static __forceinline Vec2fa loadu( const void* const a ) {
      return Vec2fa(_mm_and_ps(_mm_loadu_ps((float*)a),_mm_castsi128_ps(_mm_set_epi32(0, 0, -1, -1))));
    }

    static __forceinline void storeu ( void* ptr, const Vec2fa& v ) {
      _mm_storeu_ps((float*)ptr,v);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec2fa( ZeroTy   ) : m128(_mm_setzero_ps()) {}
    __forceinline Vec2fa( OneTy    ) : m128(_mm_set1_ps(1.0f)) {}
    __forceinline Vec2fa( PosInfTy ) : m128(_mm_set1_ps(pos_inf)) {}
    __forceinline Vec2fa( NegInfTy ) : m128(_mm_set1_ps(neg_inf)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const float& operator []( const size_t index ) const { assert(index < 2); return (&x)[index]; }
    __forceinline       float& operator []( const size_t index )       { assert(index < 2); return (&x)[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec2fa operator +( const Vec2fa& a ) { return a; }
  __forceinline Vec2fa operator -( const Vec2fa& a ) {
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_xor_ps(a.m128, mask);
  }
  __forceinline Vec2fa abs  ( const Vec2fa& a ) {
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    return _mm_and_ps(a.m128, mask);
  }
  __forceinline Vec2fa sign ( const Vec2fa& a ) {
    return blendv_ps(Vec2fa(one), -Vec2fa(one), _mm_cmplt_ps (a,Vec2fa(zero)));
  }

  __forceinline Vec2fa rcp  ( const Vec2fa& a )
  {
#if defined(__AVX512VL__)
    const Vec2fa r = _mm_rcp14_ps(a.m128);
#else
    const Vec2fa r = _mm_rcp_ps(a.m128);
#endif

#if defined(__AVX2__)
    const Vec2fa res = _mm_mul_ps(r,_mm_fnmadd_ps(r, a, vfloat4(2.0f)));
#else
    const Vec2fa res = _mm_mul_ps(r,_mm_sub_ps(vfloat4(2.0f), _mm_mul_ps(r, a)));
    //return _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), a));
#endif

    return res;
  }

  __forceinline Vec2fa sqrt ( const Vec2fa& a ) { return _mm_sqrt_ps(a.m128); }
  __forceinline Vec2fa sqr  ( const Vec2fa& a ) { return _mm_mul_ps(a,a); }

  __forceinline Vec2fa rsqrt( const Vec2fa& a )
  {
#if defined(__AVX512VL__)
    __m128 r = _mm_rsqrt14_ps(a.m128);
#else
    __m128 r = _mm_rsqrt_ps(a.m128);
#endif
    return _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f),r), _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
  }

  __forceinline Vec2fa zero_fix(const Vec2fa& a) {
    return blendv_ps(a, _mm_set1_ps(min_rcp_input), _mm_cmplt_ps (abs(a).m128, _mm_set1_ps(min_rcp_input)));
  }
  __forceinline Vec2fa rcp_safe(const Vec2fa& a) {
    return rcp(zero_fix(a));
  }
  __forceinline Vec2fa log ( const Vec2fa& a ) {
    return Vec2fa(logf(a.x),logf(a.y));
  }

  __forceinline Vec2fa exp ( const Vec2fa& a ) {
    return Vec2fa(expf(a.x),expf(a.y));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec2fa operator +( const Vec2fa& a, const Vec2fa& b ) { return _mm_add_ps(a.m128, b.m128); }
  __forceinline Vec2fa operator -( const Vec2fa& a, const Vec2fa& b ) { return _mm_sub_ps(a.m128, b.m128); }
  __forceinline Vec2fa operator *( const Vec2fa& a, const Vec2fa& b ) { return _mm_mul_ps(a.m128, b.m128); }
  __forceinline Vec2fa operator *( const Vec2fa& a, const float b ) { return a * Vec2fa(b); }
  __forceinline Vec2fa operator *( const float a, const Vec2fa& b ) { return Vec2fa(a) * b; }
  __forceinline Vec2fa operator /( const Vec2fa& a, const Vec2fa& b ) { return _mm_div_ps(a.m128,b.m128); }
  __forceinline Vec2fa operator /( const Vec2fa& a, const float b        ) { return _mm_div_ps(a.m128,_mm_set1_ps(b)); }
  __forceinline Vec2fa operator /( const        float a, const Vec2fa& b ) { return _mm_div_ps(_mm_set1_ps(a),b.m128); }

  __forceinline Vec2fa min( const Vec2fa& a, const Vec2fa& b ) { return _mm_min_ps(a.m128,b.m128); }
  __forceinline Vec2fa max( const Vec2fa& a, const Vec2fa& b ) { return _mm_max_ps(a.m128,b.m128); }

#if defined(__SSE4_1__)
    __forceinline Vec2fa mini(const Vec2fa& a, const Vec2fa& b) {
      const vint4 ai = _mm_castps_si128(a);
      const vint4 bi = _mm_castps_si128(b);
      const vint4 ci = _mm_min_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }
#endif

#if defined(__SSE4_1__)
    __forceinline Vec2fa maxi(const Vec2fa& a, const Vec2fa& b) {
      const vint4 ai = _mm_castps_si128(a);
      const vint4 bi = _mm_castps_si128(b);
      const vint4 ci = _mm_max_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }
#endif

    __forceinline Vec2fa pow ( const Vec2fa& a, const float& b ) {
      return Vec2fa(powf(a.x,b),powf(a.y,b));
    }

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX2__)
  __forceinline Vec2fa madd  ( const Vec2fa& a, const Vec2fa& b, const Vec2fa& c) { return _mm_fmadd_ps(a,b,c); }
  __forceinline Vec2fa msub  ( const Vec2fa& a, const Vec2fa& b, const Vec2fa& c) { return _mm_fmsub_ps(a,b,c); }
  __forceinline Vec2fa nmadd ( const Vec2fa& a, const Vec2fa& b, const Vec2fa& c) { return _mm_fnmadd_ps(a,b,c); }
  __forceinline Vec2fa nmsub ( const Vec2fa& a, const Vec2fa& b, const Vec2fa& c) { return _mm_fnmsub_ps(a,b,c); }
#else
  __forceinline Vec2fa madd  ( const Vec2fa& a, const Vec2fa& b, const Vec2fa& c) { return a*b+c; }
  __forceinline Vec2fa msub  ( const Vec2fa& a, const Vec2fa& b, const Vec2fa& c) { return a*b-c; }
  __forceinline Vec2fa nmadd ( const Vec2fa& a, const Vec2fa& b, const Vec2fa& c) { return -a*b+c;}
  __forceinline Vec2fa nmsub ( const Vec2fa& a, const Vec2fa& b, const Vec2fa& c) { return -a*b-c; }
#endif

  __forceinline Vec2fa madd  ( const float a, const Vec2fa& b, const Vec2fa& c) { return madd(Vec2fa(a),b,c); }
  __forceinline Vec2fa msub  ( const float a, const Vec2fa& b, const Vec2fa& c) { return msub(Vec2fa(a),b,c); }
  __forceinline Vec2fa nmadd ( const float a, const Vec2fa& b, const Vec2fa& c) { return nmadd(Vec2fa(a),b,c); }
  __forceinline Vec2fa nmsub ( const float a, const Vec2fa& b, const Vec2fa& c) { return nmsub(Vec2fa(a),b,c); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec2fa& operator +=( Vec2fa& a, const Vec2fa& b ) { return a = a + b; }
  __forceinline Vec2fa& operator -=( Vec2fa& a, const Vec2fa& b ) { return a = a - b; }
  __forceinline Vec2fa& operator *=( Vec2fa& a, const Vec2fa& b ) { return a = a * b; }
  __forceinline Vec2fa& operator *=( Vec2fa& a, const float   b ) { return a = a * b; }
  __forceinline Vec2fa& operator /=( Vec2fa& a, const Vec2fa& b ) { return a = a / b; }
  __forceinline Vec2fa& operator /=( Vec2fa& a, const float   b ) { return a = a / b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline float reduce_add(const Vec2fa& v) { return v.x+v.y; }
  __forceinline float reduce_mul(const Vec2fa& v) { return v.x*v.y; }
  __forceinline float reduce_min(const Vec2fa& v) { return min(v.x,v.y); }
  __forceinline float reduce_max(const Vec2fa& v) { return max(v.x,v.y); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool operator ==( const Vec2fa& a, const Vec2fa& b ) { return (_mm_movemask_ps(_mm_cmpeq_ps (a.m128, b.m128)) & 3) == 3; }
  __forceinline bool operator !=( const Vec2fa& a, const Vec2fa& b ) { return (_mm_movemask_ps(_mm_cmpneq_ps(a.m128, b.m128)) & 3) != 0; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidian Space Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__SSE4_1__)
  __forceinline float dot ( const Vec2fa& a, const Vec2fa& b ) {
    return _mm_cvtss_f32(_mm_dp_ps(a,b,0x3F));
  }
#else
  __forceinline float dot ( const Vec2fa& a, const Vec2fa& b ) {
    return reduce_add(a*b);
  }
#endif

  __forceinline Vec2fa cross ( const Vec2fa& a ) {
    return Vec2fa(-a.y,a.x);
  }

  __forceinline float  sqr_length ( const Vec2fa& a )                { return dot(a,a); }
  __forceinline float  rcp_length ( const Vec2fa& a )                { return rsqrt(dot(a,a)); }
  __forceinline float  rcp_length2( const Vec2fa& a )                { return rcp(dot(a,a)); }
  __forceinline float  length   ( const Vec2fa& a )                  { return sqrt(dot(a,a)); }
  __forceinline Vec2fa normalize( const Vec2fa& a )                  { return a*rsqrt(dot(a,a)); }
  __forceinline float  distance ( const Vec2fa& a, const Vec2fa& b ) { return length(a-b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec2fa select( bool s, const Vec2fa& t, const Vec2fa& f ) {
    __m128 mask = s ? _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128())) : _mm_setzero_ps();
    return blendv_ps(f, t, mask);
  }

  __forceinline Vec2fa lerp(const Vec2fa& v0, const Vec2fa& v1, const float t) {
    return madd(1.0f-t,v0,t*v1);
  }

  __forceinline int maxDim ( const Vec2fa& a )
  {
    const Vec2fa b = abs(a);
    if (b.x > b.y) return 0;
    else return 1;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Rounding Functions
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__aarch64__)
  //__forceinline Vec2fa trunc(const Vec2fa& a) { return vrndq_f32(a); }
  __forceinline Vec2fa floor(const Vec2fa& a) { return vrndmq_f32(a); }
  __forceinline Vec2fa ceil (const Vec2fa& a) { return vrndpq_f32(a); }
#elif defined (__SSE4_1__)
  //__forceinline Vec2fa trunc( const Vec2fa& a ) { return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT); }
  __forceinline Vec2fa floor( const Vec2fa& a ) { return _mm_round_ps(a, _MM_FROUND_TO_NEG_INF    ); }
  __forceinline Vec2fa ceil ( const Vec2fa& a ) { return _mm_round_ps(a, _MM_FROUND_TO_POS_INF    ); }
#else
  //__forceinline Vec2fa trunc( const Vec2fa& a ) { return Vec2fa(truncf(a.x),truncf(a.y),truncf(a.z)); }
  __forceinline Vec2fa floor( const Vec2fa& a ) { return Vec2fa(floorf(a.x),floorf(a.y)); }
  __forceinline Vec2fa ceil ( const Vec2fa& a ) { return Vec2fa(ceilf (a.x),ceilf (a.y)); }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline embree_ostream operator<<(embree_ostream cout, const Vec2fa& a) {
    return cout << "(" << a.x << ", " << a.y << ")";
  }

  typedef Vec2fa Vec2fa_t;
}
