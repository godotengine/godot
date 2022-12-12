// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
      struct { float x,y,z; };
    };

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3fa( ) {}
    __forceinline Vec3fa( const __m128 a ) : m128(a) {}

    __forceinline Vec3fa            ( const Vec3<float>& other ) { m128  = _mm_set_ps(0, other.z, other.y, other.x); }
    //__forceinline Vec3fa& operator =( const Vec3<float>& other ) { m128  = _mm_set_ps(0, other.z, other.y, other.x); return *this; }

    __forceinline Vec3fa            ( const Vec3fa& other ) { m128 = other.m128; }
    __forceinline Vec3fa& operator =( const Vec3fa& other ) { m128 = other.m128; return *this; }

    __forceinline explicit Vec3fa( const float a ) : m128(_mm_set1_ps(a)) {}
    __forceinline          Vec3fa( const float x, const float y, const float z) : m128(_mm_set_ps(0, z, y, x)) {}

    __forceinline explicit Vec3fa( const __m128i a ) : m128(_mm_cvtepi32_ps(a)) {}

    __forceinline explicit operator const vfloat4() const { return vfloat4(m128); }
    __forceinline explicit operator const   vint4() const { return vint4(_mm_cvtps_epi32(m128)); }
    __forceinline explicit operator const  Vec2fa() const { return Vec2fa(m128); }
    __forceinline explicit operator const  Vec3ia() const { return Vec3ia(_mm_cvtps_epi32(m128)); }
    
    //__forceinline operator const __m128&() const { return m128; }
    //__forceinline operator       __m128&()       { return m128; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline Vec3fa load( const void* const a ) {
#if defined(__aarch64__)
        __m128 t = _mm_load_ps((float*)a);
        t[3] = 0.0f;
        return Vec3fa(t);
#else
      return Vec3fa(_mm_and_ps(_mm_load_ps((float*)a),_mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1))));
#endif
    }

    static __forceinline Vec3fa loadu( const void* const a ) {
      return Vec3fa(_mm_loadu_ps((float*)a));
    }

    static __forceinline void storeu ( void* ptr, const Vec3fa& v ) {
      _mm_storeu_ps((float*)ptr,v.m128);
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
#if defined(__aarch64__)
    return vnegq_f32(a.m128);
#else
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_xor_ps(a.m128, mask);
#endif
  }
  __forceinline Vec3fa abs  ( const Vec3fa& a ) {
#if defined(__aarch64__)
    return _mm_abs_ps(a.m128);
#else
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    return _mm_and_ps(a.m128, mask);
#endif
  }
  __forceinline Vec3fa sign ( const Vec3fa& a ) {
    return blendv_ps(Vec3fa(one).m128, (-Vec3fa(one)).m128, _mm_cmplt_ps (a.m128,Vec3fa(zero).m128));
  }

  __forceinline Vec3fa rcp  ( const Vec3fa& a )
  {
#if defined(__aarch64__)
  return vdivq_f32(vdupq_n_f32(1.0f),a.m128);
#else

#if defined(__AVX512VL__)
    const Vec3fa r = _mm_rcp14_ps(a.m128);
#else
    const Vec3fa r = _mm_rcp_ps(a.m128);
#endif

#if defined(__AVX2__)
    const Vec3fa h_n = _mm_fnmadd_ps(a.m128, r.m128, vfloat4(1.0));  // First, compute 1 - a * r (which will be very close to 0)
    const Vec3fa res = _mm_fmadd_ps(r.m128, h_n.m128, r.m128);       // Then compute r + r * h_n
#else
    const Vec3fa h_n = _mm_sub_ps(vfloat4(1.0f), _mm_mul_ps(a.m128, r.m128));  // First, compute 1 - a * r (which will be very close to 0)
    const Vec3fa res = _mm_add_ps(r.m128,_mm_mul_ps(r.m128, h_n.m128));        // Then compute r + r * h_n  
#endif

    return res;
#endif  //defined(__aarch64__)
  }

  __forceinline Vec3fa sqrt ( const Vec3fa& a ) { return _mm_sqrt_ps(a.m128); }
  __forceinline Vec3fa sqr  ( const Vec3fa& a ) { return _mm_mul_ps(a.m128,a.m128); }

  __forceinline Vec3fa rsqrt( const Vec3fa& a )
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
    return _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f),r), _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a.m128, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
#endif
  }

  __forceinline Vec3fa zero_fix(const Vec3fa& a) {
    return blendv_ps(a.m128, _mm_set1_ps(min_rcp_input), _mm_cmplt_ps (abs(a).m128, _mm_set1_ps(min_rcp_input)));
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

#if defined(__aarch64__) || defined(__SSE4_1__)
    __forceinline Vec3fa mini(const Vec3fa& a, const Vec3fa& b) {
      const vint4 ai = _mm_castps_si128(a.m128);
      const vint4 bi = _mm_castps_si128(b.m128);
      const vint4 ci = _mm_min_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }
#endif

#if defined(__aarch64__) || defined(__SSE4_1__)
    __forceinline Vec3fa maxi(const Vec3fa& a, const Vec3fa& b) {
      const vint4 ai = _mm_castps_si128(a.m128);
      const vint4 bi = _mm_castps_si128(b.m128);
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

#if defined(__AVX2__) || defined(__ARM_NEON)
  __forceinline Vec3fa madd  ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return _mm_fmadd_ps(a.m128,b.m128,c.m128); }
  __forceinline Vec3fa msub  ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return _mm_fmsub_ps(a.m128,b.m128,c.m128); }
  __forceinline Vec3fa nmadd ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return _mm_fnmadd_ps(a.m128,b.m128,c.m128); }
  __forceinline Vec3fa nmsub ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return _mm_fnmsub_ps(a.m128,b.m128,c.m128); }
#else
  __forceinline Vec3fa madd  ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return a*b+c; }
  __forceinline Vec3fa nmadd ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return -a*b+c;}
  __forceinline Vec3fa nmsub ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return -a*b-c; }
  __forceinline Vec3fa msub  ( const Vec3fa& a, const Vec3fa& b, const Vec3fa& c) { return a*b-c; }
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
#if defined(__aarch64__)
  __forceinline float reduce_add(const Vec3fa& v) {
    float32x4_t t = v.m128;
    t[3] = 0.0f;
    return vaddvq_f32(t);
  }

  __forceinline float reduce_mul(const Vec3fa& v) { return v.x*v.y*v.z; }
  __forceinline float reduce_min(const Vec3fa& v) {
    float32x4_t t = v.m128;
      t[3] = t[2];
    return vminvq_f32(t);
  }
  __forceinline float reduce_max(const Vec3fa& v) {
    float32x4_t t = v.m128;
      t[3] = t[2];
    return vmaxvq_f32(t);
  }
#else
  __forceinline float reduce_add(const Vec3fa& v) {
    const vfloat4 a(v.m128);
    const vfloat4 b = shuffle<1>(a);
    const vfloat4 c = shuffle<2>(a);
    return _mm_cvtss_f32(a+b+c); 
  }

  __forceinline float reduce_mul(const Vec3fa& v) { return v.x*v.y*v.z; }
  __forceinline float reduce_min(const Vec3fa& v) { return min(v.x,v.y,v.z); }
  __forceinline float reduce_max(const Vec3fa& v) { return max(v.x,v.y,v.z); }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool operator ==( const Vec3fa& a, const Vec3fa& b ) { return (_mm_movemask_ps(_mm_cmpeq_ps (a.m128, b.m128)) & 7) == 7; }
  __forceinline bool operator !=( const Vec3fa& a, const Vec3fa& b ) { return (_mm_movemask_ps(_mm_cmpneq_ps(a.m128, b.m128)) & 7) != 0; }

  __forceinline Vec3ba eq_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmpeq_ps (a.m128, b.m128); }
  __forceinline Vec3ba neq_mask(const Vec3fa& a, const Vec3fa& b ) { return _mm_cmpneq_ps(a.m128, b.m128); }
  __forceinline Vec3ba lt_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmplt_ps (a.m128, b.m128); }
  __forceinline Vec3ba le_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmple_ps (a.m128, b.m128); }
 #if defined(__aarch64__)
  __forceinline Vec3ba gt_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmpgt_ps (a.m128, b.m128); }
  __forceinline Vec3ba ge_mask( const Vec3fa& a, const Vec3fa& b ) { return _mm_cmpge_ps (a.m128, b.m128); }
#else
  __forceinline Vec3ba gt_mask(const Vec3fa& a, const Vec3fa& b) { return _mm_cmpnle_ps(a.m128, b.m128); }
  __forceinline Vec3ba ge_mask(const Vec3fa& a, const Vec3fa& b) { return _mm_cmpnlt_ps(a.m128, b.m128); }
#endif

  __forceinline bool isvalid ( const Vec3fa& v ) {
    return all(gt_mask(v,Vec3fa(-FLT_LARGE)) & lt_mask(v,Vec3fa(+FLT_LARGE)));
  }

  __forceinline bool is_finite ( const Vec3fa& a ) {
    return all(ge_mask(a,Vec3fa(-FLT_MAX)) & le_mask(a,Vec3fa(+FLT_MAX)));
  }

  __forceinline bool isvalid4 ( const Vec3fa& v ) {
    return all((vfloat4(v.m128) > vfloat4(-FLT_LARGE)) & (vfloat4(v.m128) < vfloat4(+FLT_LARGE)));
  }

  __forceinline bool is_finite4 ( const Vec3fa& a ) {
    return all((vfloat4(a.m128) >= vfloat4(-FLT_MAX)) & (vfloat4(a.m128) <= vfloat4(+FLT_MAX)));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidean Space Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__SSE4_1__)
  __forceinline float dot ( const Vec3fa& a, const Vec3fa& b ) {
    return _mm_cvtss_f32(_mm_dp_ps(a.m128,b.m128,0x7F));
  }
#else
  __forceinline float dot ( const Vec3fa& a, const Vec3fa& b ) {
    return reduce_add(a*b);
  }
#endif

  __forceinline Vec3fa cross ( const Vec3fa& a, const Vec3fa& b )
  {
    vfloat4 a0 = vfloat4(a.m128);
    vfloat4 b0 = shuffle<1,2,0,3>(vfloat4(b.m128));
    vfloat4 a1 = shuffle<1,2,0,3>(vfloat4(a.m128));
    vfloat4 b1 = vfloat4(b.m128);
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
    return blendv_ps(f.m128, t.m128, mask);
  }

  __forceinline Vec3fa select( const Vec3ba& s, const Vec3fa& t, const Vec3fa& f ) {
    return blendv_ps(f.m128, t.m128, s);
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

#if defined(__aarch64__)
  __forceinline Vec3fa floor(const Vec3fa& a) { return vrndmq_f32(a.m128); }
  __forceinline Vec3fa ceil (const Vec3fa& a) { return vrndpq_f32(a.m128); }
  __forceinline Vec3fa trunc(const Vec3fa& a) { return vrndq_f32(a.m128); }
#elif defined (__SSE4_1__)
  __forceinline Vec3fa trunc( const Vec3fa& a ) { return _mm_round_ps(a.m128, _MM_FROUND_TO_NEAREST_INT); }
  __forceinline Vec3fa floor( const Vec3fa& a ) { return _mm_round_ps(a.m128, _MM_FROUND_TO_NEG_INF    ); }
  __forceinline Vec3fa ceil ( const Vec3fa& a ) { return _mm_round_ps(a.m128, _MM_FROUND_TO_POS_INF    ); }
#else
  __forceinline Vec3fa trunc( const Vec3fa& a ) { return Vec3fa(truncf(a.x),truncf(a.y),truncf(a.z)); }
  __forceinline Vec3fa floor( const Vec3fa& a ) { return Vec3fa(floorf(a.x),floorf(a.y),floorf(a.z)); }
  __forceinline Vec3fa ceil ( const Vec3fa& a ) { return Vec3fa(ceilf (a.x),ceilf (a.y),ceilf (a.z)); }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline embree_ostream operator<<(embree_ostream cout, const Vec3fa& a) {
    return cout << "(" << a.x << ", " << a.y << ", " << a.z << ")";
  }

  typedef Vec3fa Vec3fa_t;


  ////////////////////////////////////////////////////////////////////////////////
  /// SSE Vec3fx Type
  ////////////////////////////////////////////////////////////////////////////////

  struct __aligned(16) Vec3fx
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

    __forceinline Vec3fx( ) {}
    __forceinline Vec3fx( const __m128 a ) : m128(a) {}

    __forceinline explicit Vec3fx(const Vec3fa& v) : m128(v.m128) {}
    __forceinline operator Vec3fa () const { return Vec3fa(m128); }
        
    __forceinline explicit Vec3fx            ( const Vec3<float>& other ) { m128  = _mm_set_ps(0, other.z, other.y, other.x); }
    //__forceinline Vec3fx& operator =( const Vec3<float>& other ) { m128  = _mm_set_ps(0, other.z, other.y, other.x); return *this; }

    __forceinline Vec3fx            ( const Vec3fx& other ) { m128 = other.m128; }

    __forceinline Vec3fx& operator =( const Vec3fx& other ) { m128 = other.m128; return *this; }

    __forceinline explicit Vec3fx( const float a ) : m128(_mm_set1_ps(a)) {}
    __forceinline          Vec3fx( const float x, const float y, const float z) : m128(_mm_set_ps(0, z, y, x)) {}

    __forceinline Vec3fx( const Vec3fa& other, const int      a1) { m128 = other.m128; a = a1; }
    __forceinline Vec3fx( const Vec3fa& other, const unsigned a1) { m128 = other.m128; u = a1; }
    __forceinline Vec3fx( const Vec3fa& other, const float    w1) {
#if defined (__aarch64__)
      m128 = other.m128; m128[3] = w1;
#elif defined (__SSE4_1__)
      m128 = _mm_insert_ps(other.m128, _mm_set_ss(w1),3 << 4);
#else
      const vint4 mask(-1,-1,-1,0);
      m128 = select(vboolf4(_mm_castsi128_ps(mask)),vfloat4(other.m128),vfloat4(w1));
#endif
    }
    //__forceinline Vec3fx( const float x, const float y, const float z, const int      a) : x(x), y(y), z(z), a(a) {} // not working properly!
    //__forceinline Vec3fx( const float x, const float y, const float z, const unsigned a) : x(x), y(y), z(z), u(a) {} // not working properly!
    __forceinline Vec3fx( const float x, const float y, const float z, const float w) : m128(_mm_set_ps(w, z, y, x)) {}
    
    //__forceinline explicit Vec3fx( const __m128i a ) : m128(_mm_cvtepi32_ps(a)) {}

    __forceinline explicit operator const vfloat4() const { return vfloat4(m128); }
    __forceinline explicit operator const   vint4() const { return vint4(_mm_cvtps_epi32(m128)); }
    __forceinline explicit operator const  Vec2fa() const { return Vec2fa(m128); }
    __forceinline explicit operator const  Vec3ia() const { return Vec3ia(_mm_cvtps_epi32(m128)); }
    
    //__forceinline operator const __m128&() const { return m128; }
    //__forceinline operator       __m128&()       { return m128; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline Vec3fx load( const void* const a ) {
      return Vec3fx(_mm_and_ps(_mm_load_ps((float*)a),_mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1))));
    }

    static __forceinline Vec3fx loadu( const void* const a ) {
      return Vec3fx(_mm_loadu_ps((float*)a));
    }

    static __forceinline void storeu ( void* ptr, const Vec3fx& v ) {
      _mm_storeu_ps((float*)ptr,v.m128);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3fx( ZeroTy   ) : m128(_mm_setzero_ps()) {}
    __forceinline Vec3fx( OneTy    ) : m128(_mm_set1_ps(1.0f)) {}
    __forceinline Vec3fx( PosInfTy ) : m128(_mm_set1_ps(pos_inf)) {}
    __forceinline Vec3fx( NegInfTy ) : m128(_mm_set1_ps(neg_inf)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const float& operator []( const size_t index ) const { assert(index < 3); return (&x)[index]; }
    __forceinline       float& operator []( const size_t index )       { assert(index < 3); return (&x)[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3fx operator +( const Vec3fx& a ) { return a; }
  __forceinline Vec3fx operator -( const Vec3fx& a ) {
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_xor_ps(a.m128, mask);
  }
  __forceinline Vec3fx abs  ( const Vec3fx& a ) {
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    return _mm_and_ps(a.m128, mask);
  }
  __forceinline Vec3fx sign ( const Vec3fx& a ) {
    return blendv_ps(Vec3fx(one).m128, (-Vec3fx(one)).m128, _mm_cmplt_ps (a.m128,Vec3fx(zero).m128));
  }

  __forceinline Vec3fx rcp  ( const Vec3fx& a )
  {
#if defined(__AVX512VL__)
    const Vec3fx r = _mm_rcp14_ps(a.m128);
#else
    const Vec3fx r = _mm_rcp_ps(a.m128);
#endif

#if defined(__AVX2__)
    const Vec3fx res = _mm_mul_ps(r.m128,_mm_fnmadd_ps(r.m128, a.m128, vfloat4(2.0f)));
#else
    const Vec3fx res = _mm_mul_ps(r.m128,_mm_sub_ps(vfloat4(2.0f), _mm_mul_ps(r.m128, a.m128)));
    //return _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), a));
#endif

    return res;
  }

  __forceinline Vec3fx sqrt ( const Vec3fx& a ) { return _mm_sqrt_ps(a.m128); }
  __forceinline Vec3fx sqr  ( const Vec3fx& a ) { return _mm_mul_ps(a.m128,a.m128); }

  __forceinline Vec3fx rsqrt( const Vec3fx& a )
  {
#if defined(__AVX512VL__)
    __m128 r = _mm_rsqrt14_ps(a.m128);
#else
    __m128 r = _mm_rsqrt_ps(a.m128);
#endif
    return _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f),r), _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a.m128, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
  }

  __forceinline Vec3fx zero_fix(const Vec3fx& a) {
    return blendv_ps(a.m128, _mm_set1_ps(min_rcp_input), _mm_cmplt_ps (abs(a).m128, _mm_set1_ps(min_rcp_input)));
  }
  __forceinline Vec3fx rcp_safe(const Vec3fx& a) {
    return rcp(zero_fix(a));
  }
  __forceinline Vec3fx log ( const Vec3fx& a ) {
    return Vec3fx(logf(a.x),logf(a.y),logf(a.z));
  }

  __forceinline Vec3fx exp ( const Vec3fx& a ) {
    return Vec3fx(expf(a.x),expf(a.y),expf(a.z));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3fx operator +( const Vec3fx& a, const Vec3fx& b ) { return _mm_add_ps(a.m128, b.m128); }
  __forceinline Vec3fx operator -( const Vec3fx& a, const Vec3fx& b ) { return _mm_sub_ps(a.m128, b.m128); }
  __forceinline Vec3fx operator *( const Vec3fx& a, const Vec3fx& b ) { return _mm_mul_ps(a.m128, b.m128); }
  __forceinline Vec3fx operator *( const Vec3fx& a, const float b ) { return a * Vec3fx(b); }
  __forceinline Vec3fx operator *( const float a, const Vec3fx& b ) { return Vec3fx(a) * b; }
  __forceinline Vec3fx operator /( const Vec3fx& a, const Vec3fx& b ) { return _mm_div_ps(a.m128,b.m128); }
  __forceinline Vec3fx operator /( const Vec3fx& a, const float b        ) { return _mm_div_ps(a.m128,_mm_set1_ps(b)); }
  __forceinline Vec3fx operator /( const        float a, const Vec3fx& b ) { return _mm_div_ps(_mm_set1_ps(a),b.m128); }

  __forceinline Vec3fx min( const Vec3fx& a, const Vec3fx& b ) { return _mm_min_ps(a.m128,b.m128); }
  __forceinline Vec3fx max( const Vec3fx& a, const Vec3fx& b ) { return _mm_max_ps(a.m128,b.m128); }

#if defined(__SSE4_1__) || defined(__aarch64__)
    __forceinline Vec3fx mini(const Vec3fx& a, const Vec3fx& b) {
      const vint4 ai = _mm_castps_si128(a.m128);
      const vint4 bi = _mm_castps_si128(b.m128);
      const vint4 ci = _mm_min_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }
#endif

#if defined(__SSE4_1__) || defined(__aarch64__)
    __forceinline Vec3fx maxi(const Vec3fx& a, const Vec3fx& b) {
      const vint4 ai = _mm_castps_si128(a.m128);
      const vint4 bi = _mm_castps_si128(b.m128);
      const vint4 ci = _mm_max_epi32(ai,bi);
      return _mm_castsi128_ps(ci);
    }
#endif

    __forceinline Vec3fx pow ( const Vec3fx& a, const float& b ) {
      return Vec3fx(powf(a.x,b),powf(a.y,b),powf(a.z,b));
    }

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX2__)
  __forceinline Vec3fx madd  ( const Vec3fx& a, const Vec3fx& b, const Vec3fx& c) { return _mm_fmadd_ps(a.m128,b.m128,c.m128); }
  __forceinline Vec3fx msub  ( const Vec3fx& a, const Vec3fx& b, const Vec3fx& c) { return _mm_fmsub_ps(a.m128,b.m128,c.m128); }
  __forceinline Vec3fx nmadd ( const Vec3fx& a, const Vec3fx& b, const Vec3fx& c) { return _mm_fnmadd_ps(a.m128,b.m128,c.m128); }
  __forceinline Vec3fx nmsub ( const Vec3fx& a, const Vec3fx& b, const Vec3fx& c) { return _mm_fnmsub_ps(a.m128,b.m128,c.m128); }
#else
  __forceinline Vec3fx madd  ( const Vec3fx& a, const Vec3fx& b, const Vec3fx& c) { return a*b+c; }
  __forceinline Vec3fx msub  ( const Vec3fx& a, const Vec3fx& b, const Vec3fx& c) { return a*b-c; }
  __forceinline Vec3fx nmadd ( const Vec3fx& a, const Vec3fx& b, const Vec3fx& c) { return -a*b+c;}
  __forceinline Vec3fx nmsub ( const Vec3fx& a, const Vec3fx& b, const Vec3fx& c) { return -a*b-c; }
#endif

  __forceinline Vec3fx madd  ( const float a, const Vec3fx& b, const Vec3fx& c) { return madd(Vec3fx(a),b,c); }
  __forceinline Vec3fx msub  ( const float a, const Vec3fx& b, const Vec3fx& c) { return msub(Vec3fx(a),b,c); }
  __forceinline Vec3fx nmadd ( const float a, const Vec3fx& b, const Vec3fx& c) { return nmadd(Vec3fx(a),b,c); }
  __forceinline Vec3fx nmsub ( const float a, const Vec3fx& b, const Vec3fx& c) { return nmsub(Vec3fx(a),b,c); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3fx& operator +=( Vec3fx& a, const Vec3fx& b ) { return a = a + b; }
  __forceinline Vec3fx& operator -=( Vec3fx& a, const Vec3fx& b ) { return a = a - b; }
  __forceinline Vec3fx& operator *=( Vec3fx& a, const Vec3fx& b ) { return a = a * b; }
  __forceinline Vec3fx& operator *=( Vec3fx& a, const float   b ) { return a = a * b; }
  __forceinline Vec3fx& operator /=( Vec3fx& a, const Vec3fx& b ) { return a = a / b; }
  __forceinline Vec3fx& operator /=( Vec3fx& a, const float   b ) { return a = a / b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline float reduce_add(const Vec3fx& v) { 
    const vfloat4 a(v.m128);
    const vfloat4 b = shuffle<1>(a);
    const vfloat4 c = shuffle<2>(a);
    return _mm_cvtss_f32(a+b+c); 
  }

  __forceinline float reduce_mul(const Vec3fx& v) { return v.x*v.y*v.z; }
  __forceinline float reduce_min(const Vec3fx& v) { return min(v.x,v.y,v.z); }
  __forceinline float reduce_max(const Vec3fx& v) { return max(v.x,v.y,v.z); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool operator ==( const Vec3fx& a, const Vec3fx& b ) { return (_mm_movemask_ps(_mm_cmpeq_ps (a.m128, b.m128)) & 7) == 7; }
  __forceinline bool operator !=( const Vec3fx& a, const Vec3fx& b ) { return (_mm_movemask_ps(_mm_cmpneq_ps(a.m128, b.m128)) & 7) != 0; }

  __forceinline Vec3ba eq_mask( const Vec3fx& a, const Vec3fx& b ) { return _mm_cmpeq_ps (a.m128, b.m128); }
  __forceinline Vec3ba neq_mask(const Vec3fx& a, const Vec3fx& b ) { return _mm_cmpneq_ps(a.m128, b.m128); }
  __forceinline Vec3ba lt_mask( const Vec3fx& a, const Vec3fx& b ) { return _mm_cmplt_ps (a.m128, b.m128); }
  __forceinline Vec3ba le_mask( const Vec3fx& a, const Vec3fx& b ) { return _mm_cmple_ps (a.m128, b.m128); }
  __forceinline Vec3ba gt_mask( const Vec3fx& a, const Vec3fx& b ) { return _mm_cmpnle_ps(a.m128, b.m128); }
  __forceinline Vec3ba ge_mask( const Vec3fx& a, const Vec3fx& b ) { return _mm_cmpnlt_ps(a.m128, b.m128); }

  __forceinline bool isvalid ( const Vec3fx& v ) {
    return all(gt_mask(v,Vec3fx(-FLT_LARGE)) & lt_mask(v,Vec3fx(+FLT_LARGE)));
  }

  __forceinline bool is_finite ( const Vec3fx& a ) {
    return all(ge_mask(a,Vec3fx(-FLT_MAX)) & le_mask(a,Vec3fx(+FLT_MAX)));
  }

  __forceinline bool isvalid4 ( const Vec3fx& v ) {
    return all((vfloat4(v.m128) > vfloat4(-FLT_LARGE)) & (vfloat4(v.m128) < vfloat4(+FLT_LARGE)));
  }

  __forceinline bool is_finite4 ( const Vec3fx& a ) {
    return all((vfloat4(a.m128) >= vfloat4(-FLT_MAX)) & (vfloat4(a.m128) <= vfloat4(+FLT_MAX)));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidean Space Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__SSE4_1__)
  __forceinline float dot ( const Vec3fx& a, const Vec3fx& b ) {
    return _mm_cvtss_f32(_mm_dp_ps(a.m128,b.m128,0x7F));
  }
#else
  __forceinline float dot ( const Vec3fx& a, const Vec3fx& b ) {
    return reduce_add(a*b);
  }
#endif

  __forceinline Vec3fx cross ( const Vec3fx& a, const Vec3fx& b )
  {
    vfloat4 a0 = vfloat4(a.m128);
    vfloat4 b0 = shuffle<1,2,0,3>(vfloat4(b.m128));
    vfloat4 a1 = shuffle<1,2,0,3>(vfloat4(a.m128));
    vfloat4 b1 = vfloat4(b.m128);
    return Vec3fx(shuffle<1,2,0,3>(msub(a0,b0,a1*b1)));
  }

  __forceinline float  sqr_length ( const Vec3fx& a )                { return dot(a,a); }
  __forceinline float  rcp_length ( const Vec3fx& a )                { return rsqrt(dot(a,a)); }
  __forceinline float  rcp_length2( const Vec3fx& a )                { return rcp(dot(a,a)); }
  __forceinline float  length   ( const Vec3fx& a )                  { return sqrt(dot(a,a)); }
  __forceinline Vec3fx normalize( const Vec3fx& a )                  { return a*rsqrt(dot(a,a)); }
  __forceinline float  distance ( const Vec3fx& a, const Vec3fx& b ) { return length(a-b); }
  __forceinline float  halfArea ( const Vec3fx& d )                  { return madd(d.x,(d.y+d.z),d.y*d.z); }
  __forceinline float  area     ( const Vec3fx& d )                  { return 2.0f*halfArea(d); }

  __forceinline Vec3fx normalize_safe( const Vec3fx& a ) {
    const float d = dot(a,a); if (unlikely(d == 0.0f)) return a; else return a*rsqrt(d);
  }

  /*! differentiated normalization */
  __forceinline Vec3fx dnormalize(const Vec3fx& p, const Vec3fx& dp)
  {
    const float pp  = dot(p,p);
    const float pdp = dot(p,dp);
    return (pp*dp-pdp*p)*rcp(pp)*rsqrt(pp);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline Vec3fx select( bool s, const Vec3fx& t, const Vec3fx& f ) {
    __m128 mask = s ? _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128())) : _mm_setzero_ps();
    return blendv_ps(f.m128, t.m128, mask);
  }

  __forceinline Vec3fx select( const Vec3ba& s, const Vec3fx& t, const Vec3fx& f ) {
    return blendv_ps(f.m128, t.m128, s);
  }

  __forceinline Vec3fx lerp(const Vec3fx& v0, const Vec3fx& v1, const float t) {
    return madd(1.0f-t,v0,t*v1);
  }

  __forceinline int maxDim ( const Vec3fx& a )
  {
    const Vec3fx b = abs(a);
    if (b.x > b.y) {
      if (b.x > b.z) return 0; else return 2;
    } else {
      if (b.y > b.z) return 1; else return 2;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Rounding Functions
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__aarch64__)
  __forceinline Vec3fx trunc(const Vec3fx& a) { return vrndq_f32(a.m128); }
  __forceinline Vec3fx floor(const Vec3fx& a) { return vrndmq_f32(a.m128); }
  __forceinline Vec3fx ceil (const Vec3fx& a) { return vrndpq_f32(a.m128); }
#elif defined (__SSE4_1__)
  __forceinline Vec3fx trunc( const Vec3fx& a ) { return _mm_round_ps(a.m128, _MM_FROUND_TO_NEAREST_INT); }
  __forceinline Vec3fx floor( const Vec3fx& a ) { return _mm_round_ps(a.m128, _MM_FROUND_TO_NEG_INF    ); }
  __forceinline Vec3fx ceil ( const Vec3fx& a ) { return _mm_round_ps(a.m128, _MM_FROUND_TO_POS_INF    ); }
#else
  __forceinline Vec3fx trunc( const Vec3fx& a ) { return Vec3fx(truncf(a.x),truncf(a.y),truncf(a.z)); }
  __forceinline Vec3fx floor( const Vec3fx& a ) { return Vec3fx(floorf(a.x),floorf(a.y),floorf(a.z)); }
  __forceinline Vec3fx ceil ( const Vec3fx& a ) { return Vec3fx(ceilf (a.x),ceilf (a.y),ceilf (a.z)); }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline embree_ostream operator<<(embree_ostream cout, const Vec3fx& a) {
    return cout << "(" << a.x << ", " << a.y << ", " << a.z << ")";
  }

  
  typedef Vec3fx Vec3ff;
}
