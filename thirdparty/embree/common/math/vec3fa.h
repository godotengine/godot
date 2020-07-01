// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "../sys/alloc.h"
#include "math.h"
#include "../simd/sse.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// SSE Vec3fa Type
  ////////////////////////////////////////////////////////////////////////////////

  struct __aligned(16) Vec3fa
  {
    ALIGNED_STRUCT_(16);

    typedef float Scalar;
    enum { N = 3 };
    union {
      __m128 m128;
      struct { float x,y,z; union { int a; unsigned u; float w; }; };
    };

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3fa( ) {}
    __forceinline Vec3fa( const __m128 a ) : m128(a) {}

    __forceinline Vec3fa            ( const Vec3<float>& other  ) { x = other.x; y = other.y; z = other.z; }
    __forceinline Vec3fa& operator =( const Vec3<float>& other ) { x = other.x; y = other.y; z = other.z; return *this; }

    __forceinline Vec3fa            ( const Vec3fa& other ) { m128 = other.m128; }

    __forceinline Vec3fa& operator =( const Vec3fa& other ) { m128 = other.m128; return *this; }

    __forceinline explicit Vec3fa( const float a ) : m128(_mm_set1_ps(a)) {}
    __forceinline          Vec3fa( const float x, const float y, const float z) : m128(_mm_set_ps(z, z, y, x)) {}

    __forceinline Vec3fa( const Vec3fa& other, const int      a1) { m128 = other.m128; a = a1; }
    __forceinline Vec3fa( const Vec3fa& other, const unsigned a1) { m128 = other.m128; u = a1; }
    __forceinline Vec3fa( const Vec3fa& other, const float    w1) {      
#if defined (__SSE4_1__)
      m128 = _mm_insert_ps(other.m128, _mm_set_ss(w1),3 << 4);
#else
      const vint4 mask(-1,-1,-1,0);
      m128 = select(vboolf4(_mm_castsi128_ps(mask)),vfloat4(other.m128),vfloat4(w1));
#endif
    }
    //__forceinline Vec3fa( const float x, const float y, const float z, const int      a) : x(x), y(y), z(z), a(a) {} // not working properly!
    //__forceinline Vec3fa( const float x, const float y, const float z, const unsigned a) : x(x), y(y), z(z), u(a) {} // not working properly!
    __forceinline Vec3fa( const float x, const float y, const float z, const float w) : m128(_mm_set_ps(w, z, y, x)) {}

    __forceinline explicit Vec3fa( const __m128i a ) : m128(_mm_cvtepi32_ps(a)) {}

    __forceinline operator const __m128&() const { return m128; }
    __forceinline operator       __m128&()       { return m128; }

#if defined (__SSE4_1__)
    friend __forceinline Vec3fa copy_a( const Vec3fa& a, const Vec3fa& b ) { return _mm_insert_ps(a, b, (3 << 4) | (3 << 6)); }
#else
    friend __forceinline Vec3fa copy_a( const Vec3fa& a, const Vec3fa& b ) { Vec3fa c = a; c.a = b.a; return c; }
#endif

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline Vec3fa load( const void* const a ) {
      return Vec3fa(_mm_and_ps(_mm_load_ps((float*)a),_mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1))));
    }

    static __forceinline Vec3fa loadu( const void* const a ) {
      return Vec3fa(_mm_loadu_ps((float*)a));
    }

    static __forceinline void storeu ( void* ptr, const Vec3fa& v ) {
      _mm_storeu_ps((float*)ptr,v);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3fa( ZeroTy   ) : m128(_mm_setzero_ps()) {}
    __forceinline Vec3fa( OneTy    ) : m128(_mm_set1_ps(1.0f)) {}
    __forceinline Vec3fa( PosInfTy ) : m128(_mm_set1_ps(pos_inf)) {}
    __forceinline Vec3fa( NegInfTy ) : m128(_mm_set1_ps(neg_inf)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const float& operator []( const size_t index ) const { assert(index < 3); return (&x)[index]; }
    __forceinline       float& operator []( const size_t index )       { assert(index < 3); return (&x)[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3fa operator +( const Vec3fa& a ) { return a; }
  __forceinline Vec3fa operator -( const Vec3fa& a ) {
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_xor_ps(a.m128, mask);
  }
  __forceinline Vec3fa abs  ( const Vec3fa& a ) {
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    return _mm_and_ps(a.m128, mask);
  }
  __forceinline Vec3fa sign ( const Vec3fa& a ) {
    return blendv_ps(Vec3fa(one), -Vec3fa(one), _mm_cmplt_ps (a,Vec3fa(zero)));
  }

  __forceinline Vec3fa rcp  ( const Vec3fa& a )
  {
#if defined(__AVX512VL__)
    const Vec3fa r = _mm_rcp14_ps(a.m128);
#else
    const Vec3fa r = _mm_rcp_ps(a.m128);
#endif

#if defined(__AVX2__)
    const Vec3fa res = _mm_mul_ps(r,_mm_fnmadd_ps(r, a, vfloat4(2.0f)));
#else
    const Vec3fa res = _mm_mul_ps(r,_mm_sub_ps(vfloat4(2.0f), _mm_mul_ps(r, a)));
    //return _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), a));
#endif

    return res;
  }

  __forceinline Vec3fa sqrt ( const Vec3fa& a ) { return _mm_sqrt_ps(a.m128); }
  __forceinline Vec3fa sqr  ( const Vec3fa& a ) { return _mm_mul_ps(a,a); }

  __forceinline Vec3fa rsqrt( const Vec3fa& a )
  {
#if defined(__AVX512VL__)
    __m128 r = _mm_rsqrt14_ps(a.m128);
#else
    __m128 r = _mm_rsqrt_ps(a.m128);
#endif
    return _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f),r), _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
  }

  __forceinline Vec3fa zero_fix(const Vec3fa& a) {
    return blendv_ps(a, _mm_set1_ps(min_rcp_input), _mm_cmplt_ps (abs(a).m128, _mm_set1_ps(min_rcp_input)));
  }
  __forceinline Vec3fa rcp_safe(const Vec3fa& a) {
    return rcp(zero_fix(a));
  }
  __forceinline Vec3fa log ( const Vec3fa& a ) {
    return Vec3fa(logf(a.x),logf(a.y),logf(a.z));
  }

  __forceinline Vec3fa exp ( const Vec3fa& a ) {
    return Vec3fa(expf(a.x),expf(a.y),expf(a.z));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3fa operator +( const Vec3fa& a, const Vec3fa& b ) { return _mm_add_ps(a.m128, b.m128); }
  __forceinline Vec3fa operator -( const Vec3fa& a, const Vec3fa& b ) { return _mm_sub_ps(a.m128, b.m128); }
  __forceinline Vec3fa operator *( const Vec3fa& a, const Vec3fa& b ) { return _mm_mul_ps(a.m128, b.m128); }
  __forceinline Vec3fa operator *( const Vec3fa& a, const float b ) { return a * Vec3fa(b); }
  __forceinline Vec3fa operator *( const float a, const Vec3fa& b ) { return Vec3fa(a) * b; }
  __forceinline Vec3fa operator /( const Vec3fa& a, const Vec3fa& b ) { return _mm_div_ps(a.m128,b.m128); }
  __forceinline Vec3fa operator /( const Vec3fa& a, const float b        ) { return _mm_div_ps(a.m128,_mm_set1_ps(b)); }
  __forceinline Vec3fa operator /( const        float a, const Vec3fa& b ) { return _mm_div_ps(_mm_set1_ps(a),b.m128); }

  __forceinline Vec3fa min( const Vec3fa& a, const Vec3fa& b ) { return _mm_min_ps(a.m128,b.m128); }
  __forceinline Vec3fa max( const Vec3fa& a, const Vec3fa& b ) { return _mm_max_ps(a.m128,b.m128); }

#if defined(__SSE4_1__)
    __forceinline Vec3fa mini(const Vec3fa& a, const Vec3fa& b) {
      const vint4 ai = _mm_castps_si128(a);
      const vint4 bi = _mm_castps_si128(b);
      const vint4 ci = _mm_min_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }
#endif

#if defined(__SSE4_1__)
    __forceinline Vec3fa maxi(const Vec3fa& a, const Vec3fa& b) {
      const vint4 ai = _mm_castps_si128(a);
      const vint4 bi = _mm_castps_si128(b);
      const vint4 ci = _mm_max_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }
#endif

    __forceinline Vec3fa pow ( const Vec3fa& a, const float& b ) {
      return Vec3fa(powf(a.x,b),powf(a.y,b),powf(a.z,b));
    }

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX2__)
  __forceinline Vec3fa madd  ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return _mm_fmadd_ps(a,b,c); }
  __forceinline Vec3fa msub  ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return _mm_fmsub_ps(a,b,c); }
  __forceinline Vec3fa nmadd ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return _mm_fnmadd_ps(a,b,c); }
  __forceinline Vec3fa nmsub ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return _mm_fnmsub_ps(a,b,c); }
#else
  __forceinline Vec3fa madd  ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return a*b+c; }
  __forceinline Vec3fa msub  ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return a*b-c; }
  __forceinline Vec3fa nmadd ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return -a*b+c;}
  __forceinline Vec3fa nmsub ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return -a*b-c; }
#endif

  __forceinline Vec3fa madd  ( const float a, const Vec3fa& b, const Vec3fa& c) { return madd(Vec3fa(a),b,c); }
  __forceinline Vec3fa msub  ( const float a, const Vec3fa& b, const Vec3fa& c) { return msub(Vec3fa(a),b,c); }
  __forceinline Vec3fa nmadd ( const float a, const Vec3fa& b, const Vec3fa& c) { return nmadd(Vec3fa(a),b,c); }
  __forceinline Vec3fa nmsub ( const float a, const Vec3fa& b, const Vec3fa& c) { return nmsub(Vec3fa(a),b,c); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3fa& operator +=( Vec3fa& a, const Vec3fa& b ) { return a = a + b; }
  __forceinline Vec3fa& operator -=( Vec3fa& a, const Vec3fa& b ) { return a = a - b; }
  __forceinline Vec3fa& operator *=( Vec3fa& a, const Vec3fa& b ) { return a = a * b; }
  __forceinline Vec3fa& operator *=( Vec3fa& a, const float   b ) { return a = a * b; }
  __forceinline Vec3fa& operator /=( Vec3fa& a, const Vec3fa& b ) { return a = a / b; }
  __forceinline Vec3fa& operator /=( Vec3fa& a, const float   b ) { return a = a / b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline float reduce_add(const Vec3fa& v) { 
    const vfloat4 a(v);
    const vfloat4 b = shuffle<1>(a);
    const vfloat4 c = shuffle<2>(a);
    return _mm_cvtss_f32(a+b+c); 
  }

  __forceinline float reduce_mul(const Vec3fa& v) { return v.x*v.y*v.z; }
  __forceinline float reduce_min(const Vec3fa& v) { return min(v.x,v.y,v.z); }
  __forceinline float reduce_max(const Vec3fa& v) { return max(v.x,v.y,v.z); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool operator ==( const Vec3fa& a, const Vec3fa& b ) { return (_mm_movemask_ps(_mm_cmpeq_ps (a.m128, b.m128)) & 7) == 7; }
  __forceinline bool operator !=( const Vec3fa& a, const Vec3fa& b ) { return (_mm_movemask_ps(_mm_cmpneq_ps(a.m128, b.m128)) & 7) != 0; }

  __forceinline Vec3ba eq_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmpeq_ps (a.m128, b.m128); }
  __forceinline Vec3ba neq_mask(const Vec3fa& a, const Vec3fa& b ) { return _mm_cmpneq_ps(a.m128, b.m128); }
  __forceinline Vec3ba lt_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmplt_ps (a.m128, b.m128); }
  __forceinline Vec3ba le_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmple_ps (a.m128, b.m128); }
  __forceinline Vec3ba gt_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmpnle_ps(a.m128, b.m128); }
  __forceinline Vec3ba ge_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmpnlt_ps(a.m128, b.m128); }

  __forceinline bool isvalid ( const Vec3fa& v ) {
    return all(gt_mask(v,Vec3fa(-FLT_LARGE)) & lt_mask(v,Vec3fa(+FLT_LARGE)));
  }

  __forceinline bool is_finite ( const Vec3fa& a ) {
    return all(ge_mask(a,Vec3fa(-FLT_MAX)) & le_mask(a,Vec3fa(+FLT_MAX)));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidian Space Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__SSE4_1__)
  __forceinline float dot ( const Vec3fa& a, const Vec3fa& b ) {
    return _mm_cvtss_f32(_mm_dp_ps(a,b,0x7F));
  }
#else
  __forceinline float dot ( const Vec3fa& a, const Vec3fa& b ) {
    return reduce_add(a*b);
  }
#endif

  __forceinline Vec3fa cross ( const Vec3fa& a, const Vec3fa& b )
  {
    vfloat4 a0 = vfloat4(a);
    vfloat4 b0 = shuffle<1,2,0,3>(vfloat4(b));
    vfloat4 a1 = shuffle<1,2,0,3>(vfloat4(a));
    vfloat4 b1 = vfloat4(b);
    return Vec3fa(shuffle<1,2,0,3>(msub(a0,b0,a1*b1)));
  }

  __forceinline float  sqr_length ( const Vec3fa& a )                { return dot(a,a); }
  __forceinline float  rcp_length ( const Vec3fa& a )                { return rsqrt(dot(a,a)); }
  __forceinline float  rcp_length2( const Vec3fa& a )                { return rcp(dot(a,a)); }
  __forceinline float  length   ( const Vec3fa& a )                  { return sqrt(dot(a,a)); }
  __forceinline Vec3fa normalize( const Vec3fa& a )                  { return a*rsqrt(dot(a,a)); }
  __forceinline float  distance ( const Vec3fa& a, const Vec3fa& b ) { return length(a-b); }
  __forceinline float  halfArea ( const Vec3fa& d )                  { return madd(d.x,(d.y+d.z),d.y*d.z); }
  __forceinline float  area     ( const Vec3fa& d )                  { return 2.0f*halfArea(d); }

  __forceinline Vec3fa normalize_safe( const Vec3fa& a ) {
    const float d = dot(a,a); if (unlikely(d == 0.0f)) return a; else return a*rsqrt(d);
  }

  /*! differentiated normalization */
  __forceinline Vec3fa dnormalize(const Vec3fa& p, const Vec3fa& dp)
  {
    const float pp  = dot(p,p);
    const float pdp = dot(p,dp);
    return (pp*dp-pdp*p)*rcp(pp)*rsqrt(pp);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3fa select( bool s, const Vec3fa& t, const Vec3fa& f ) {
    __m128 mask = s ? _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128())) : _mm_setzero_ps();
    return blendv_ps(f, t, mask);
  }

  __forceinline Vec3fa select( const Vec3ba& s, const Vec3fa& t, const Vec3fa& f ) {
    return blendv_ps(f, t, s);
  }

  __forceinline Vec3fa lerp(const Vec3fa& v0, const Vec3fa& v1, const float t) {
    return madd(1.0f-t,v0,t*v1);
  }

  __forceinline int maxDim ( const Vec3fa& a )
  {
    const Vec3fa b = abs(a);
    if (b.x > b.y) {
      if (b.x > b.z) return 0; else return 2;
    } else {
      if (b.y > b.z) return 1; else return 2;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Rounding Functions
  ////////////////////////////////////////////////////////////////////////////////

#if defined (__SSE4_1__)
  __forceinline Vec3fa trunc( const Vec3fa& a ) { return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT); }
  __forceinline Vec3fa floor( const Vec3fa& a ) { return _mm_round_ps(a, _MM_FROUND_TO_NEG_INF    ); }
  __forceinline Vec3fa ceil ( const Vec3fa& a ) { return _mm_round_ps(a, _MM_FROUND_TO_POS_INF    ); }
#else
  __forceinline Vec3fa trunc( const Vec3fa& a ) { return Vec3fa(truncf(a.x),truncf(a.y),truncf(a.z)); }
  __forceinline Vec3fa floor( const Vec3fa& a ) { return Vec3fa(floorf(a.x),floorf(a.y),floorf(a.z)); }
  __forceinline Vec3fa ceil ( const Vec3fa& a ) { return Vec3fa(ceilf (a.x),ceilf (a.y),ceilf (a.z)); }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  inline std::ostream& operator<<(std::ostream& cout, const Vec3fa& a) {
    return cout << "(" << a.x << ", " << a.y << ", " << a.z << ")";
  }

  typedef Vec3fa Vec3fa_t;
}
