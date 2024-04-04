// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../sys/platform.h"
#include "../sys/intrinsics.h"
#include "constants.h"
#include <cmath>

namespace embree
{
  __forceinline bool isvalid ( const float& v ) {
    return (v > -FLT_LARGE) & (v < +FLT_LARGE);
  }

  __forceinline int cast_f2i(float f) {
    return __builtin_bit_cast(int,f);
  }

  __forceinline float cast_i2f(int i) {
    return __builtin_bit_cast(float,i);
  }

  __forceinline int   toInt  (const float& a) { return int(a); }
  __forceinline float toFloat(const int&   a) { return float(a); }

  __forceinline float asFloat(const int   a) { return __builtin_bit_cast(float,a); }
  __forceinline int   asInt  (const float a) { return __builtin_bit_cast(int,a); }
  
  //__forceinline bool finite ( const float x ) { return _finite(x) != 0; }
  __forceinline float sign ( const float x ) { return x<0?-1.0f:1.0f; }
  __forceinline float sqr  ( const float x ) { return x*x; }

  __forceinline float rcp  ( const float x ) {
    return sycl::native::recip(x);
  }

  __forceinline float signmsk(const float a) { return asFloat(asInt(a) & 0x80000000); }
  //__forceinline float signmsk ( const float x ) {
  //  return _mm_cvtss_f32(_mm_and_ps(_mm_set_ss(x),_mm_castsi128_ps(_mm_set1_epi32(0x80000000))));
  //}
  //__forceinline float xorf( const float x, const float y ) {
  //  return _mm_cvtss_f32(_mm_xor_ps(_mm_set_ss(x),_mm_set_ss(y)));
  //}
  //__forceinline float andf( const float x, const unsigned y ) {
  //  return _mm_cvtss_f32(_mm_and_ps(_mm_set_ss(x),_mm_castsi128_ps(_mm_set1_epi32(y))));
  //}
  
  __forceinline float rsqrt( const float x ) {
    return sycl::rsqrt(x);
  }

  //__forceinline float nextafter(float x, float y) { if ((x<y) == (x>0)) return x*(1.1f+float(ulp)); else return x*(0.9f-float(ulp)); }
  //__forceinline double nextafter(double x, double y) { return _nextafter(x, y); }
  //__forceinline int roundf(float f) { return (int)(f + 0.5f); }

  __forceinline float abs  ( const float x ) { return sycl::fabs(x); }
  __forceinline float acos ( const float x ) { return sycl::acos(x); }
  __forceinline float asin ( const float x ) { return sycl::asin(x); }
  __forceinline float atan ( const float x ) { return sycl::atan(x); }
  __forceinline float atan2( const float y, const float x ) { return sycl::atan2(y, x); }
  __forceinline float cos  ( const float x ) { return sycl::cos(x); }
  __forceinline float cosh ( const float x ) { return sycl::cosh(x); }
  __forceinline float exp  ( const float x ) { return sycl::exp(x); }
  __forceinline float fmod ( const float x, const float y ) { return sycl::fmod(x, y); }
  __forceinline float log  ( const float x ) { return sycl::log(x); }
  __forceinline float log10( const float x ) { return sycl::log10(x); }
  __forceinline float pow  ( const float x, const float y ) { return sycl::pow(x, y); }
  __forceinline float sin  ( const float x ) { return sycl::sin(x); }
  __forceinline float sinh ( const float x ) { return sycl::sinh(x); }
  __forceinline float sqrt ( const float x ) { return sycl::sqrt(x); }
  __forceinline float tan  ( const float x ) { return sycl::tan(x); }
  __forceinline float tanh ( const float x ) { return sycl::tanh(x); }
  __forceinline float floor( const float x ) { return sycl::floor(x); }
  __forceinline float ceil ( const float x ) { return sycl::ceil(x); }
  __forceinline float frac ( const float x ) { return x-floor(x); }

  //__forceinline double abs  ( const double x ) { return ::fabs(x); }
  //__forceinline double sign ( const double x ) { return x<0?-1.0:1.0; }
  //__forceinline double acos ( const double x ) { return ::acos (x); }
  //__forceinline double asin ( const double x ) { return ::asin (x); }
  //__forceinline double atan ( const double x ) { return ::atan (x); }
  //__forceinline double atan2( const double y, const double x ) { return ::atan2(y, x); }
  //__forceinline double cos  ( const double x ) { return ::cos  (x); }
  //__forceinline double cosh ( const double x ) { return ::cosh (x); }
  //__forceinline double exp  ( const double x ) { return ::exp  (x); }
  //__forceinline double fmod ( const double x, const double y ) { return ::fmod (x, y); }
  //__forceinline double log  ( const double x ) { return ::log  (x); }
  //__forceinline double log10( const double x ) { return ::log10(x); }
  //__forceinline double pow  ( const double x, const double y ) { return ::pow  (x, y); }
  //__forceinline double rcp  ( const double x ) { return 1.0/x; }
  //__forceinline double rsqrt( const double x ) { return 1.0/::sqrt(x); }
  //__forceinline double sin  ( const double x ) { return ::sin  (x); }
  //__forceinline double sinh ( const double x ) { return ::sinh (x); }
  //__forceinline double sqr  ( const double x ) { return x*x; }
  //__forceinline double sqrt ( const double x ) { return ::sqrt (x); }
  //__forceinline double tan  ( const double x ) { return ::tan  (x); }
  //__forceinline double tanh ( const double x ) { return ::tanh (x); }
  //__forceinline double floor( const double x ) { return ::floor (x); }
  //__forceinline double ceil ( const double x ) { return ::ceil (x); }

/*
#if defined(__SSE4_1__)
  __forceinline float mini(float a, float b) {
    const __m128i ai = _mm_castps_si128(_mm_set_ss(a));
    const __m128i bi = _mm_castps_si128(_mm_set_ss(b));
    const __m128i ci = _mm_min_epi32(ai,bi);
    return _mm_cvtss_f32(_mm_castsi128_ps(ci));
  }
#endif

#if defined(__SSE4_1__)
  __forceinline float maxi(float a, float b) {
    const __m128i ai = _mm_castps_si128(_mm_set_ss(a));
    const __m128i bi = _mm_castps_si128(_mm_set_ss(b));
    const __m128i ci = _mm_max_epi32(ai,bi);
    return _mm_cvtss_f32(_mm_castsi128_ps(ci));
  }
#endif
*/
  
  template<typename T>
    __forceinline T twice(const T& a) { return a+a; }

  __forceinline      int min(int      a, int      b) { return sycl::min(a,b); }
  __forceinline unsigned min(unsigned a, unsigned b) { return sycl::min(a,b); }
  __forceinline  int64_t min(int64_t  a, int64_t  b) { return sycl::min(a,b); }
  __forceinline    float min(float    a, float    b) { return sycl::fmin(a,b); }
  __forceinline   double min(double   a, double   b) { return sycl::fmin(a,b); }
#if defined(__X86_64__)
  __forceinline   size_t min(size_t   a, size_t   b) { return sycl::min(a,b); }
#endif

  template<typename T> __forceinline T min(const T& a, const T& b, const T& c) { return min(min(a,b),c); }
  template<typename T> __forceinline T min(const T& a, const T& b, const T& c, const T& d) { return min(min(a,b),min(c,d)); }
  template<typename T> __forceinline T min(const T& a, const T& b, const T& c, const T& d, const T& e) { return min(min(min(a,b),min(c,d)),e); }

//  template<typename T> __forceinline T mini(const T& a, const T& b, const T& c) { return mini(mini(a,b),c); }
//  template<typename T> __forceinline T mini(const T& a, const T& b, const T& c, const T& d) { return mini(mini(a,b),mini(c,d)); }
//  template<typename T> __forceinline T mini(const T& a, const T& b, const T& c, const T& d, const T& e) { return mini(mini(mini(a,b),mini(c,d)),e); }

  __forceinline      int max(int      a, int      b) { return sycl::max(a,b); }
  __forceinline unsigned max(unsigned a, unsigned b) { return sycl::max(a,b); }
  __forceinline  int64_t max(int64_t  a, int64_t  b) { return sycl::max(a,b); }
  __forceinline    float max(float    a, float    b) { return sycl::fmax(a,b); }
  __forceinline   double max(double   a, double   b) { return sycl::fmax(a,b); }
#if defined(__X86_64__)
  __forceinline   size_t max(size_t   a, size_t   b) { return sycl::max(a,b); }
#endif

  template<typename T> __forceinline T max(const T& a, const T& b, const T& c) { return max(max(a,b),c); }
  template<typename T> __forceinline T max(const T& a, const T& b, const T& c, const T& d) { return max(max(a,b),max(c,d)); }
  template<typename T> __forceinline T max(const T& a, const T& b, const T& c, const T& d, const T& e) { return max(max(max(a,b),max(c,d)),e); }

//  template<typename T> __forceinline T maxi(const T& a, const T& b, const T& c) { return maxi(maxi(a,b),c); }
//  template<typename T> __forceinline T maxi(const T& a, const T& b, const T& c, const T& d) { return maxi(maxi(a,b),maxi(c,d)); }
//  template<typename T> __forceinline T maxi(const T& a, const T& b, const T& c, const T& d, const T& e) { return maxi(maxi(maxi(a,b),maxi(c,d)),e); }

  template<typename T> __forceinline T clamp(const T& x, const T& lower = T(zero), const T& upper = T(one)) { return max(min(x,upper),lower); }
  template<typename T> __forceinline T clampz(const T& x, const T& upper) { return max(T(zero), min(x,upper)); }

  template<typename T> __forceinline T  deg2rad ( const T& x )  { return x * T(1.74532925199432957692e-2f); }
  template<typename T> __forceinline T  rad2deg ( const T& x )  { return x * T(5.72957795130823208768e1f); }
  template<typename T> __forceinline T  sin2cos ( const T& x )  { return sqrt(max(T(zero),T(one)-x*x)); }
  template<typename T> __forceinline T  cos2sin ( const T& x )  { return sin2cos(x); }

  __forceinline float madd  ( const float a, const float b, const float c) { return +sycl::fma(+a,b,+c); }
  __forceinline float msub  ( const float a, const float b, const float c) { return +sycl::fma(+a,b,-c); }
  __forceinline float nmadd ( const float a, const float b, const float c) { return +sycl::fma(-a,b,+c); }
  __forceinline float nmsub ( const float a, const float b, const float c) { return -sycl::fma(+a,b,+c); }

  /*! random functions */
/*
  template<typename T> T random() { return T(0); }
  template<> __forceinline int      random() { return int(rand()); }
  template<> __forceinline uint32_t random() { return uint32_t(rand()) ^ (uint32_t(rand()) << 16); }
  template<> __forceinline float  random() { return rand()/float(RAND_MAX); }
  template<> __forceinline double random() { return rand()/double(RAND_MAX); }
*/
  
  /*! selects */
  __forceinline bool  select(bool s, bool  t , bool f) { return s ? t : f; }
  __forceinline int   select(bool s, int   t,   int f) { return s ? t : f; }
  __forceinline float select(bool s, float t, float f) { return s ? t : f; }

  __forceinline bool none(bool s) { return !s; }
  __forceinline bool all (bool s) { return s; }
  __forceinline bool any (bool s) { return s; }

  __forceinline unsigned movemask (bool s) { return (unsigned)s; }

  __forceinline float lerp(const float v0, const float v1, const float t) {
    return madd(1.0f-t,v0,t*v1);
  }

  template<typename T>
    __forceinline T lerp2(const float x0, const float x1, const float x2, const float x3, const T& u, const T& v) {
    return madd((1.0f-u),madd((1.0f-v),T(x0),v*T(x2)),u*madd((1.0f-v),T(x1),v*T(x3)));
  }

  /*! exchange */
  template<typename T> __forceinline void xchg ( T& a, T& b ) { const T tmp = a; a = b; b = tmp; }

   /*  load/store */
  template<typename Ty> struct mem;
 
  template<> struct mem<float> {
    static __forceinline float load (bool mask, const void* ptr) { return mask ? *(float*)ptr : 0.0f; }
    static __forceinline float loadu(bool mask, const void* ptr) { return mask ? *(float*)ptr : 0.0f; }
  
    static __forceinline void store (bool mask, void* ptr, const float v) { if (mask) *(float*)ptr = v; }
    static __forceinline void storeu(bool mask, void* ptr, const float v) { if (mask) *(float*)ptr = v; }
  };
  
  /*! bit reverse operation */
  template<class T>
    __forceinline T bitReverse(const T& vin)
  {
    T v = vin;
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    v = ( v >> 16             ) | ( v               << 16);
    return v;
  }

  /*! bit interleave operation */
  template<class T>
    __forceinline T bitInterleave(const T& xin, const T& yin, const T& zin)
  {
	T x = xin, y = yin, z = zin;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;

    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y <<  8)) & 0x0300F00F;
    y = (y | (y <<  4)) & 0x030C30C3;
    y = (y | (y <<  2)) & 0x09249249;

    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z <<  8)) & 0x0300F00F;
    z = (z | (z <<  4)) & 0x030C30C3;
    z = (z | (z <<  2)) & 0x09249249;

    return x | (y << 1) | (z << 2);
  }

  /*! bit interleave operation for 64bit data types*/
  template<class T>
    __forceinline T bitInterleave64(const T& xin, const T& yin, const T& zin){
    T x = xin & 0x1fffff;
    T y = yin & 0x1fffff;
    T z = zin & 0x1fffff;

    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;

    y = (y | y << 32) & 0x1f00000000ffff;
    y = (y | y << 16) & 0x1f0000ff0000ff;
    y = (y | y << 8) & 0x100f00f00f00f00f;
    y = (y | y << 4) & 0x10c30c30c30c30c3;
    y = (y | y << 2) & 0x1249249249249249;

    z = (z | z << 32) & 0x1f00000000ffff;
    z = (z | z << 16) & 0x1f0000ff0000ff;
    z = (z | z << 8) & 0x100f00f00f00f00f;
    z = (z | z << 4) & 0x10c30c30c30c30c3;
    z = (z | z << 2) & 0x1249249249249249;

    return x | (y << 1) | (z << 2);
  }
}
