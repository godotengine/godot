// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_SSE_H
#define EIGEN_PACKET_MATH_SSE_H

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#endif

#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS (2*sizeof(void*))
#endif

#ifdef __FMA__
#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD 1
#endif
#endif

#if (defined EIGEN_VECTORIZE_AVX) && (EIGEN_COMP_GNUC_STRICT || EIGEN_COMP_MINGW) && (__GXX_ABI_VERSION < 1004)
// With GCC's default ABI version, a __m128 or __m256 are the same types and therefore we cannot
// have overloads for both types without linking error.
// One solution is to increase ABI version using -fabi-version=4 (or greater).
// Otherwise, we workaround this inconvenience by wrapping 128bit types into the following helper
// structure:
template<typename T>
struct eigen_packet_wrapper
{
  EIGEN_ALWAYS_INLINE operator T&() { return m_val; }
  EIGEN_ALWAYS_INLINE operator const T&() const { return m_val; }
  EIGEN_ALWAYS_INLINE eigen_packet_wrapper() {}
  EIGEN_ALWAYS_INLINE eigen_packet_wrapper(const T &v) : m_val(v) {}
  EIGEN_ALWAYS_INLINE eigen_packet_wrapper& operator=(const T &v) {
    m_val = v;
    return *this;
  }

  T m_val;
};
typedef eigen_packet_wrapper<__m128>  Packet4f;
typedef eigen_packet_wrapper<__m128i> Packet4i;
typedef eigen_packet_wrapper<__m128d> Packet2d;
#else
typedef __m128  Packet4f;
typedef __m128i Packet4i;
typedef __m128d Packet2d;
#endif

template<> struct is_arithmetic<__m128>  { enum { value = true }; };
template<> struct is_arithmetic<__m128i> { enum { value = true }; };
template<> struct is_arithmetic<__m128d> { enum { value = true }; };

#define vec4f_swizzle1(v,p,q,r,s) \
  (_mm_castsi128_ps(_mm_shuffle_epi32( _mm_castps_si128(v), ((s)<<6|(r)<<4|(q)<<2|(p)))))

#define vec4i_swizzle1(v,p,q,r,s) \
  (_mm_shuffle_epi32( v, ((s)<<6|(r)<<4|(q)<<2|(p))))

#define vec2d_swizzle1(v,p,q) \
  (_mm_castsi128_pd(_mm_shuffle_epi32( _mm_castpd_si128(v), ((q*2+1)<<6|(q*2)<<4|(p*2+1)<<2|(p*2)))))

#define vec4f_swizzle2(a,b,p,q,r,s) \
  (_mm_shuffle_ps( (a), (b), ((s)<<6|(r)<<4|(q)<<2|(p))))

#define vec4i_swizzle2(a,b,p,q,r,s) \
  (_mm_castps_si128( (_mm_shuffle_ps( _mm_castsi128_ps(a), _mm_castsi128_ps(b), ((s)<<6|(r)<<4|(q)<<2|(p))))))

#define _EIGEN_DECLARE_CONST_Packet4f(NAME,X) \
  const Packet4f p4f_##NAME = pset1<Packet4f>(X)

#define _EIGEN_DECLARE_CONST_Packet2d(NAME,X) \
  const Packet2d p2d_##NAME = pset1<Packet2d>(X)

#define _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(NAME,X) \
  const Packet4f p4f_##NAME = _mm_castsi128_ps(pset1<Packet4i>(X))

#define _EIGEN_DECLARE_CONST_Packet4i(NAME,X) \
  const Packet4i p4i_##NAME = pset1<Packet4i>(X)


// Use the packet_traits defined in AVX/PacketMath.h instead if we're going
// to leverage AVX instructions.
#ifndef EIGEN_VECTORIZE_AVX
template<> struct packet_traits<float>  : default_packet_traits
{
  typedef Packet4f type;
  typedef Packet4f half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=4,
    HasHalfPacket = 0,

    HasDiv  = 1,
    HasSin  = EIGEN_FAST_MATH,
    HasCos  = EIGEN_FAST_MATH,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasTanh  = EIGEN_FAST_MATH,
    HasBlend = 1

#ifdef EIGEN_VECTORIZE_SSE4_1
    ,
    HasRound = 1,
    HasFloor = 1,
    HasCeil = 1
#endif
  };
};
template<> struct packet_traits<double> : default_packet_traits
{
  typedef Packet2d type;
  typedef Packet2d half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=2,
    HasHalfPacket = 0,

    HasDiv  = 1,
    HasExp  = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasBlend = 1

#ifdef EIGEN_VECTORIZE_SSE4_1
    ,
    HasRound = 1,
    HasFloor = 1,
    HasCeil = 1
#endif
  };
};
#endif
template<> struct packet_traits<int>    : default_packet_traits
{
  typedef Packet4i type;
  typedef Packet4i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=4,

    HasBlend = 1
  };
};

template<> struct unpacket_traits<Packet4f> { typedef float  type; enum {size=4, alignment=Aligned16}; typedef Packet4f half; };
template<> struct unpacket_traits<Packet2d> { typedef double type; enum {size=2, alignment=Aligned16}; typedef Packet2d half; };
template<> struct unpacket_traits<Packet4i> { typedef int    type; enum {size=4, alignment=Aligned16}; typedef Packet4i half; };

#ifndef EIGEN_VECTORIZE_AVX
template<> struct scalar_div_cost<float,true> { enum { value = 7 }; };
template<> struct scalar_div_cost<double,true> { enum { value = 8 }; };
#endif

#if EIGEN_COMP_MSVC==1500
// Workaround MSVC 9 internal compiler error.
// TODO: It has been detected with win64 builds (amd64), so let's check whether it also happens in 32bits+SSE mode
// TODO: let's check whether there does not exist a better fix, like adding a pset0() function. (it crashed on pset1(0)).
template<> EIGEN_STRONG_INLINE Packet4f pset1<Packet4f>(const float&  from) { return _mm_set_ps(from,from,from,from); }
template<> EIGEN_STRONG_INLINE Packet2d pset1<Packet2d>(const double& from) { return _mm_set_pd(from,from); }
template<> EIGEN_STRONG_INLINE Packet4i pset1<Packet4i>(const int&    from) { return _mm_set_epi32(from,from,from,from); }
#else
template<> EIGEN_STRONG_INLINE Packet4f pset1<Packet4f>(const float&  from) { return _mm_set_ps1(from); }
template<> EIGEN_STRONG_INLINE Packet2d pset1<Packet2d>(const double& from) { return _mm_set1_pd(from); }
template<> EIGEN_STRONG_INLINE Packet4i pset1<Packet4i>(const int&    from) { return _mm_set1_epi32(from); }
#endif

// GCC generates a shufps instruction for _mm_set1_ps/_mm_load1_ps instead of the more efficient pshufd instruction.
// However, using inrinsics for pset1 makes gcc to generate crappy code in some cases (see bug 203)
// Using inline assembly is also not an option because then gcc fails to reorder properly the instructions.
// Therefore, we introduced the pload1 functions to be used in product kernels for which bug 203 does not apply.
// Also note that with AVX, we want it to generate a vbroadcastss.
#if EIGEN_COMP_GNUC_STRICT && (!defined __AVX__)
template<> EIGEN_STRONG_INLINE Packet4f pload1<Packet4f>(const float *from) {
  return vec4f_swizzle1(_mm_load_ss(from),0,0,0,0);
}
#endif

template<> EIGEN_STRONG_INLINE Packet4f plset<Packet4f>(const float& a) { return _mm_add_ps(pset1<Packet4f>(a), _mm_set_ps(3,2,1,0)); }
template<> EIGEN_STRONG_INLINE Packet2d plset<Packet2d>(const double& a) { return _mm_add_pd(pset1<Packet2d>(a),_mm_set_pd(1,0)); }
template<> EIGEN_STRONG_INLINE Packet4i plset<Packet4i>(const int& a) { return _mm_add_epi32(pset1<Packet4i>(a),_mm_set_epi32(3,2,1,0)); }

template<> EIGEN_STRONG_INLINE Packet4f padd<Packet4f>(const Packet4f& a, const Packet4f& b) { return _mm_add_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d padd<Packet2d>(const Packet2d& a, const Packet2d& b) { return _mm_add_pd(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i padd<Packet4i>(const Packet4i& a, const Packet4i& b) { return _mm_add_epi32(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f psub<Packet4f>(const Packet4f& a, const Packet4f& b) { return _mm_sub_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d psub<Packet2d>(const Packet2d& a, const Packet2d& b) { return _mm_sub_pd(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i psub<Packet4i>(const Packet4i& a, const Packet4i& b) { return _mm_sub_epi32(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f pnegate(const Packet4f& a)
{
  const Packet4f mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000,0x80000000,0x80000000,0x80000000));
  return _mm_xor_ps(a,mask);
}
template<> EIGEN_STRONG_INLINE Packet2d pnegate(const Packet2d& a)
{
  const Packet2d mask = _mm_castsi128_pd(_mm_setr_epi32(0x0,0x80000000,0x0,0x80000000));
  return _mm_xor_pd(a,mask);
}
template<> EIGEN_STRONG_INLINE Packet4i pnegate(const Packet4i& a)
{
  return psub(Packet4i(_mm_setr_epi32(0,0,0,0)), a);
}

template<> EIGEN_STRONG_INLINE Packet4f pconj(const Packet4f& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet2d pconj(const Packet2d& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4i pconj(const Packet4i& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet4f pmul<Packet4f>(const Packet4f& a, const Packet4f& b) { return _mm_mul_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d pmul<Packet2d>(const Packet2d& a, const Packet2d& b) { return _mm_mul_pd(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pmul<Packet4i>(const Packet4i& a, const Packet4i& b)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_mullo_epi32(a,b);
#else
  // this version is slightly faster than 4 scalar products
  return vec4i_swizzle1(
            vec4i_swizzle2(
              _mm_mul_epu32(a,b),
              _mm_mul_epu32(vec4i_swizzle1(a,1,0,3,2),
                            vec4i_swizzle1(b,1,0,3,2)),
              0,2,0,2),
            0,2,1,3);
#endif
}

template<> EIGEN_STRONG_INLINE Packet4f pdiv<Packet4f>(const Packet4f& a, const Packet4f& b) { return _mm_div_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d pdiv<Packet2d>(const Packet2d& a, const Packet2d& b) { return _mm_div_pd(a,b); }

// for some weird raisons, it has to be overloaded for packet of integers
template<> EIGEN_STRONG_INLINE Packet4i pmadd(const Packet4i& a, const Packet4i& b, const Packet4i& c) { return padd(pmul(a,b), c); }
#ifdef __FMA__
template<> EIGEN_STRONG_INLINE Packet4f pmadd(const Packet4f& a, const Packet4f& b, const Packet4f& c) { return _mm_fmadd_ps(a,b,c); }
template<> EIGEN_STRONG_INLINE Packet2d pmadd(const Packet2d& a, const Packet2d& b, const Packet2d& c) { return _mm_fmadd_pd(a,b,c); }
#endif

template<> EIGEN_STRONG_INLINE Packet4f pmin<Packet4f>(const Packet4f& a, const Packet4f& b) {
#if EIGEN_COMP_GNUC
  // There appears to be a bug in GCC, by which the optimizer may
  // flip the argument order in calls to _mm_min_ps, so we have to
  // resort to inline ASM here. This is supposed to be fixed in gcc6.3,
  // see also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72867
  Packet4f res = b;
  asm("minps %[a], %[res]" : [res] "+x" (res) : [a] "x" (a));
  return res;
#else
  // Arguments are reversed to match NaN propagation behavior of std::min.
  return _mm_min_ps(b, a);
#endif
}
template<> EIGEN_STRONG_INLINE Packet2d pmin<Packet2d>(const Packet2d& a, const Packet2d& b) {
#if EIGEN_COMP_GNUC
  // There appears to be a bug in GCC, by which the optimizer may
  // flip the argument order in calls to _mm_min_pd, so we have to
  // resort to inline ASM here. This is supposed to be fixed in gcc6.3,
  // see also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72867
  Packet2d res = b;
  asm("minpd %[a], %[res]" : [res] "+x" (res) : [a] "x" (a));
  return res;
#else
  // Arguments are reversed to match NaN propagation behavior of std::min.
  return _mm_min_pd(b, a);
#endif
}
template<> EIGEN_STRONG_INLINE Packet4i pmin<Packet4i>(const Packet4i& a, const Packet4i& b)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_min_epi32(a,b);
#else
  // after some bench, this version *is* faster than a scalar implementation
  Packet4i mask = _mm_cmplt_epi32(a,b);
  return _mm_or_si128(_mm_and_si128(mask,a),_mm_andnot_si128(mask,b));
#endif
}

template<> EIGEN_STRONG_INLINE Packet4f pmax<Packet4f>(const Packet4f& a, const Packet4f& b) {
#if EIGEN_COMP_GNUC
  // There appears to be a bug in GCC, by which the optimizer may
  // flip the argument order in calls to _mm_max_ps, so we have to
  // resort to inline ASM here. This is supposed to be fixed in gcc6.3,
  // see also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72867
  Packet4f res = b;
  asm("maxps %[a], %[res]" : [res] "+x" (res) : [a] "x" (a));
  return res;
#else
  // Arguments are reversed to match NaN propagation behavior of std::max.
  return _mm_max_ps(b, a);
#endif
}
template<> EIGEN_STRONG_INLINE Packet2d pmax<Packet2d>(const Packet2d& a, const Packet2d& b) {
#if EIGEN_COMP_GNUC
  // There appears to be a bug in GCC, by which the optimizer may
  // flip the argument order in calls to _mm_max_pd, so we have to
  // resort to inline ASM here. This is supposed to be fixed in gcc6.3,
  // see also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72867
  Packet2d res = b;
  asm("maxpd %[a], %[res]" : [res] "+x" (res) : [a] "x" (a));
  return res;
#else
  // Arguments are reversed to match NaN propagation behavior of std::max.
  return _mm_max_pd(b, a);
#endif
}
template<> EIGEN_STRONG_INLINE Packet4i pmax<Packet4i>(const Packet4i& a, const Packet4i& b)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_max_epi32(a,b);
#else
  // after some bench, this version *is* faster than a scalar implementation
  Packet4i mask = _mm_cmpgt_epi32(a,b);
  return _mm_or_si128(_mm_and_si128(mask,a),_mm_andnot_si128(mask,b));
#endif
}

#ifdef EIGEN_VECTORIZE_SSE4_1
template<> EIGEN_STRONG_INLINE Packet4f pround<Packet4f>(const Packet4f& a) { return _mm_round_ps(a, 0); }
template<> EIGEN_STRONG_INLINE Packet2d pround<Packet2d>(const Packet2d& a) { return _mm_round_pd(a, 0); }

template<> EIGEN_STRONG_INLINE Packet4f pceil<Packet4f>(const Packet4f& a) { return _mm_ceil_ps(a); }
template<> EIGEN_STRONG_INLINE Packet2d pceil<Packet2d>(const Packet2d& a) { return _mm_ceil_pd(a); }

template<> EIGEN_STRONG_INLINE Packet4f pfloor<Packet4f>(const Packet4f& a) { return _mm_floor_ps(a); }
template<> EIGEN_STRONG_INLINE Packet2d pfloor<Packet2d>(const Packet2d& a) { return _mm_floor_pd(a); }
#endif

template<> EIGEN_STRONG_INLINE Packet4f pand<Packet4f>(const Packet4f& a, const Packet4f& b) { return _mm_and_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d pand<Packet2d>(const Packet2d& a, const Packet2d& b) { return _mm_and_pd(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pand<Packet4i>(const Packet4i& a, const Packet4i& b) { return _mm_and_si128(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f por<Packet4f>(const Packet4f& a, const Packet4f& b) { return _mm_or_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d por<Packet2d>(const Packet2d& a, const Packet2d& b) { return _mm_or_pd(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i por<Packet4i>(const Packet4i& a, const Packet4i& b) { return _mm_or_si128(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f pxor<Packet4f>(const Packet4f& a, const Packet4f& b) { return _mm_xor_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d pxor<Packet2d>(const Packet2d& a, const Packet2d& b) { return _mm_xor_pd(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pxor<Packet4i>(const Packet4i& a, const Packet4i& b) { return _mm_xor_si128(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f pandnot<Packet4f>(const Packet4f& a, const Packet4f& b) { return _mm_andnot_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d pandnot<Packet2d>(const Packet2d& a, const Packet2d& b) { return _mm_andnot_pd(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pandnot<Packet4i>(const Packet4i& a, const Packet4i& b) { return _mm_andnot_si128(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f pload<Packet4f>(const float*   from) { EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_ps(from); }
template<> EIGEN_STRONG_INLINE Packet2d pload<Packet2d>(const double*  from) { EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_pd(from); }
template<> EIGEN_STRONG_INLINE Packet4i pload<Packet4i>(const int*     from) { EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(reinterpret_cast<const __m128i*>(from)); }

#if EIGEN_COMP_MSVC
  template<> EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float*  from) {
    EIGEN_DEBUG_UNALIGNED_LOAD
    #if (EIGEN_COMP_MSVC==1600)
    // NOTE Some version of MSVC10 generates bad code when using _mm_loadu_ps
    // (i.e., it does not generate an unaligned load!!
    __m128 res = _mm_loadl_pi(_mm_set1_ps(0.0f), (const __m64*)(from));
    res = _mm_loadh_pi(res, (const __m64*)(from+2));
    return res;
    #else
    return _mm_loadu_ps(from);
    #endif
  }
#else
// NOTE: with the code below, MSVC's compiler crashes!

template<> EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float* from)
{
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_ps(from);
}
#endif

template<> EIGEN_STRONG_INLINE Packet2d ploadu<Packet2d>(const double* from)
{
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_pd(from);
}
template<> EIGEN_STRONG_INLINE Packet4i ploadu<Packet4i>(const int* from)
{
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(from));
}


template<> EIGEN_STRONG_INLINE Packet4f ploaddup<Packet4f>(const float*   from)
{
  return vec4f_swizzle1(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(from))), 0, 0, 1, 1);
}
template<> EIGEN_STRONG_INLINE Packet2d ploaddup<Packet2d>(const double*  from)
{ return pset1<Packet2d>(from[0]); }
template<> EIGEN_STRONG_INLINE Packet4i ploaddup<Packet4i>(const int*     from)
{
  Packet4i tmp;
  tmp = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(from));
  return vec4i_swizzle1(tmp, 0, 0, 1, 1);
}

template<> EIGEN_STRONG_INLINE void pstore<float>(float*   to, const Packet4f& from) { EIGEN_DEBUG_ALIGNED_STORE _mm_store_ps(to, from); }
template<> EIGEN_STRONG_INLINE void pstore<double>(double* to, const Packet2d& from) { EIGEN_DEBUG_ALIGNED_STORE _mm_store_pd(to, from); }
template<> EIGEN_STRONG_INLINE void pstore<int>(int*       to, const Packet4i& from) { EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to), from); }

template<> EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const Packet2d& from) { EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_pd(to, from); }
template<> EIGEN_STRONG_INLINE void pstoreu<float>(float*   to, const Packet4f& from) { EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_ps(to, from); }
template<> EIGEN_STRONG_INLINE void pstoreu<int>(int*       to, const Packet4i& from) { EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to), from); }

template<> EIGEN_DEVICE_FUNC inline Packet4f pgather<float, Packet4f>(const float* from, Index stride)
{
 return _mm_set_ps(from[3*stride], from[2*stride], from[1*stride], from[0*stride]);
}
template<> EIGEN_DEVICE_FUNC inline Packet2d pgather<double, Packet2d>(const double* from, Index stride)
{
 return _mm_set_pd(from[1*stride], from[0*stride]);
}
template<> EIGEN_DEVICE_FUNC inline Packet4i pgather<int, Packet4i>(const int* from, Index stride)
{
 return _mm_set_epi32(from[3*stride], from[2*stride], from[1*stride], from[0*stride]);
 }

template<> EIGEN_DEVICE_FUNC inline void pscatter<float, Packet4f>(float* to, const Packet4f& from, Index stride)
{
  to[stride*0] = _mm_cvtss_f32(from);
  to[stride*1] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 1));
  to[stride*2] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 2));
  to[stride*3] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 3));
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<double, Packet2d>(double* to, const Packet2d& from, Index stride)
{
  to[stride*0] = _mm_cvtsd_f64(from);
  to[stride*1] = _mm_cvtsd_f64(_mm_shuffle_pd(from, from, 1));
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<int, Packet4i>(int* to, const Packet4i& from, Index stride)
{
  to[stride*0] = _mm_cvtsi128_si32(from);
  to[stride*1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 1));
  to[stride*2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 2));
  to[stride*3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 3));
}

// some compilers might be tempted to perform multiple moves instead of using a vector path.
template<> EIGEN_STRONG_INLINE void pstore1<Packet4f>(float* to, const float& a)
{
  Packet4f pa = _mm_set_ss(a);
  pstore(to, Packet4f(vec4f_swizzle1(pa,0,0,0,0)));
}
// some compilers might be tempted to perform multiple moves instead of using a vector path.
template<> EIGEN_STRONG_INLINE void pstore1<Packet2d>(double* to, const double& a)
{
  Packet2d pa = _mm_set_sd(a);
  pstore(to, Packet2d(vec2d_swizzle1(pa,0,0)));
}

#ifndef EIGEN_VECTORIZE_AVX
template<> EIGEN_STRONG_INLINE void prefetch<float>(const float*   addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }
template<> EIGEN_STRONG_INLINE void prefetch<double>(const double* addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }
template<> EIGEN_STRONG_INLINE void prefetch<int>(const int*       addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }
#endif

#if EIGEN_COMP_MSVC_STRICT && EIGEN_OS_WIN64
// The temporary variable fixes an internal compilation error in vs <= 2008 and a wrong-result bug in vs 2010
// Direct of the struct members fixed bug #62.
template<> EIGEN_STRONG_INLINE float  pfirst<Packet4f>(const Packet4f& a) { return a.m128_f32[0]; }
template<> EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) { return a.m128d_f64[0]; }
template<> EIGEN_STRONG_INLINE int    pfirst<Packet4i>(const Packet4i& a) { int x = _mm_cvtsi128_si32(a); return x; }
#elif EIGEN_COMP_MSVC_STRICT
// The temporary variable fixes an internal compilation error in vs <= 2008 and a wrong-result bug in vs 2010
template<> EIGEN_STRONG_INLINE float  pfirst<Packet4f>(const Packet4f& a) { float x = _mm_cvtss_f32(a); return x; }
template<> EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) { double x = _mm_cvtsd_f64(a); return x; }
template<> EIGEN_STRONG_INLINE int    pfirst<Packet4i>(const Packet4i& a) { int x = _mm_cvtsi128_si32(a); return x; }
#else
template<> EIGEN_STRONG_INLINE float  pfirst<Packet4f>(const Packet4f& a) { return _mm_cvtss_f32(a); }
template<> EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) { return _mm_cvtsd_f64(a); }
template<> EIGEN_STRONG_INLINE int    pfirst<Packet4i>(const Packet4i& a) { return _mm_cvtsi128_si32(a); }
#endif

template<> EIGEN_STRONG_INLINE Packet4f preverse(const Packet4f& a)
{ return _mm_shuffle_ps(a,a,0x1B); }
template<> EIGEN_STRONG_INLINE Packet2d preverse(const Packet2d& a)
{ return _mm_shuffle_pd(a,a,0x1); }
template<> EIGEN_STRONG_INLINE Packet4i preverse(const Packet4i& a)
{ return _mm_shuffle_epi32(a,0x1B); }

template<> EIGEN_STRONG_INLINE Packet4f pabs(const Packet4f& a)
{
  const Packet4f mask = _mm_castsi128_ps(_mm_setr_epi32(0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF));
  return _mm_and_ps(a,mask);
}
template<> EIGEN_STRONG_INLINE Packet2d pabs(const Packet2d& a)
{
  const Packet2d mask = _mm_castsi128_pd(_mm_setr_epi32(0xFFFFFFFF,0x7FFFFFFF,0xFFFFFFFF,0x7FFFFFFF));
  return _mm_and_pd(a,mask);
}
template<> EIGEN_STRONG_INLINE Packet4i pabs(const Packet4i& a)
{
  #ifdef EIGEN_VECTORIZE_SSSE3
  return _mm_abs_epi32(a);
  #else
  Packet4i aux = _mm_srai_epi32(a,31);
  return _mm_sub_epi32(_mm_xor_si128(a,aux),aux);
  #endif
}

// with AVX, the default implementations based on pload1 are faster
#ifndef __AVX__
template<> EIGEN_STRONG_INLINE void
pbroadcast4<Packet4f>(const float *a,
                      Packet4f& a0, Packet4f& a1, Packet4f& a2, Packet4f& a3)
{
  a3 = pload<Packet4f>(a);
  a0 = vec4f_swizzle1(a3, 0,0,0,0);
  a1 = vec4f_swizzle1(a3, 1,1,1,1);
  a2 = vec4f_swizzle1(a3, 2,2,2,2);
  a3 = vec4f_swizzle1(a3, 3,3,3,3);
}
template<> EIGEN_STRONG_INLINE void
pbroadcast4<Packet2d>(const double *a,
                      Packet2d& a0, Packet2d& a1, Packet2d& a2, Packet2d& a3)
{
#ifdef EIGEN_VECTORIZE_SSE3
  a0 = _mm_loaddup_pd(a+0);
  a1 = _mm_loaddup_pd(a+1);
  a2 = _mm_loaddup_pd(a+2);
  a3 = _mm_loaddup_pd(a+3);
#else
  a1 = pload<Packet2d>(a);
  a0 = vec2d_swizzle1(a1, 0,0);
  a1 = vec2d_swizzle1(a1, 1,1);
  a3 = pload<Packet2d>(a+2);
  a2 = vec2d_swizzle1(a3, 0,0);
  a3 = vec2d_swizzle1(a3, 1,1);
#endif
}
#endif

EIGEN_STRONG_INLINE void punpackp(Packet4f* vecs)
{
  vecs[1] = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vecs[0]), 0x55));
  vecs[2] = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vecs[0]), 0xAA));
  vecs[3] = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vecs[0]), 0xFF));
  vecs[0] = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vecs[0]), 0x00));
}

#ifdef EIGEN_VECTORIZE_SSE3
template<> EIGEN_STRONG_INLINE Packet4f preduxp<Packet4f>(const Packet4f* vecs)
{
  return _mm_hadd_ps(_mm_hadd_ps(vecs[0], vecs[1]),_mm_hadd_ps(vecs[2], vecs[3]));
}

template<> EIGEN_STRONG_INLINE Packet2d preduxp<Packet2d>(const Packet2d* vecs)
{
  return _mm_hadd_pd(vecs[0], vecs[1]);
}

#else
template<> EIGEN_STRONG_INLINE Packet4f preduxp<Packet4f>(const Packet4f* vecs)
{
  Packet4f tmp0, tmp1, tmp2;
  tmp0 = _mm_unpacklo_ps(vecs[0], vecs[1]);
  tmp1 = _mm_unpackhi_ps(vecs[0], vecs[1]);
  tmp2 = _mm_unpackhi_ps(vecs[2], vecs[3]);
  tmp0 = _mm_add_ps(tmp0, tmp1);
  tmp1 = _mm_unpacklo_ps(vecs[2], vecs[3]);
  tmp1 = _mm_add_ps(tmp1, tmp2);
  tmp2 = _mm_movehl_ps(tmp1, tmp0);
  tmp0 = _mm_movelh_ps(tmp0, tmp1);
  return _mm_add_ps(tmp0, tmp2);
}

template<> EIGEN_STRONG_INLINE Packet2d preduxp<Packet2d>(const Packet2d* vecs)
{
  return _mm_add_pd(_mm_unpacklo_pd(vecs[0], vecs[1]), _mm_unpackhi_pd(vecs[0], vecs[1]));
}
#endif  // SSE3

template<> EIGEN_STRONG_INLINE float predux<Packet4f>(const Packet4f& a)
{
  // Disable SSE3 _mm_hadd_pd that is extremely slow on all existing Intel's architectures
  // (from Nehalem to Haswell)
// #ifdef EIGEN_VECTORIZE_SSE3
//   Packet4f tmp = _mm_add_ps(a, vec4f_swizzle1(a,2,3,2,3));
//   return pfirst<Packet4f>(_mm_hadd_ps(tmp, tmp));
// #else
  Packet4f tmp = _mm_add_ps(a, _mm_movehl_ps(a,a));
  return pfirst<Packet4f>(_mm_add_ss(tmp, _mm_shuffle_ps(tmp,tmp, 1)));
// #endif
}

template<> EIGEN_STRONG_INLINE double predux<Packet2d>(const Packet2d& a)
{
  // Disable SSE3 _mm_hadd_pd that is extremely slow on all existing Intel's architectures
  // (from Nehalem to Haswell)
// #ifdef EIGEN_VECTORIZE_SSE3
//   return pfirst<Packet2d>(_mm_hadd_pd(a, a));
// #else
  return pfirst<Packet2d>(_mm_add_sd(a, _mm_unpackhi_pd(a,a)));
// #endif
}

#ifdef EIGEN_VECTORIZE_SSSE3
template<> EIGEN_STRONG_INLINE Packet4i preduxp<Packet4i>(const Packet4i* vecs)
{
  return _mm_hadd_epi32(_mm_hadd_epi32(vecs[0], vecs[1]),_mm_hadd_epi32(vecs[2], vecs[3]));
}
template<> EIGEN_STRONG_INLINE int predux<Packet4i>(const Packet4i& a)
{
  Packet4i tmp0 = _mm_hadd_epi32(a,a);
  return pfirst<Packet4i>(_mm_hadd_epi32(tmp0,tmp0));
}
#else
template<> EIGEN_STRONG_INLINE int predux<Packet4i>(const Packet4i& a)
{
  Packet4i tmp = _mm_add_epi32(a, _mm_unpackhi_epi64(a,a));
  return pfirst(tmp) + pfirst<Packet4i>(_mm_shuffle_epi32(tmp, 1));
}

template<> EIGEN_STRONG_INLINE Packet4i preduxp<Packet4i>(const Packet4i* vecs)
{
  Packet4i tmp0, tmp1, tmp2;
  tmp0 = _mm_unpacklo_epi32(vecs[0], vecs[1]);
  tmp1 = _mm_unpackhi_epi32(vecs[0], vecs[1]);
  tmp2 = _mm_unpackhi_epi32(vecs[2], vecs[3]);
  tmp0 = _mm_add_epi32(tmp0, tmp1);
  tmp1 = _mm_unpacklo_epi32(vecs[2], vecs[3]);
  tmp1 = _mm_add_epi32(tmp1, tmp2);
  tmp2 = _mm_unpacklo_epi64(tmp0, tmp1);
  tmp0 = _mm_unpackhi_epi64(tmp0, tmp1);
  return _mm_add_epi32(tmp0, tmp2);
}
#endif
// Other reduction functions:

// mul
template<> EIGEN_STRONG_INLINE float predux_mul<Packet4f>(const Packet4f& a)
{
  Packet4f tmp = _mm_mul_ps(a, _mm_movehl_ps(a,a));
  return pfirst<Packet4f>(_mm_mul_ss(tmp, _mm_shuffle_ps(tmp,tmp, 1)));
}
template<> EIGEN_STRONG_INLINE double predux_mul<Packet2d>(const Packet2d& a)
{
  return pfirst<Packet2d>(_mm_mul_sd(a, _mm_unpackhi_pd(a,a)));
}
template<> EIGEN_STRONG_INLINE int predux_mul<Packet4i>(const Packet4i& a)
{
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (eg., reusing pmul is very slow !)
  // TODO try to call _mm_mul_epu32 directly
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  return  (aux[0] * aux[1]) * (aux[2] * aux[3]);;
}

// min
template<> EIGEN_STRONG_INLINE float predux_min<Packet4f>(const Packet4f& a)
{
  Packet4f tmp = _mm_min_ps(a, _mm_movehl_ps(a,a));
  return pfirst<Packet4f>(_mm_min_ss(tmp, _mm_shuffle_ps(tmp,tmp, 1)));
}
template<> EIGEN_STRONG_INLINE double predux_min<Packet2d>(const Packet2d& a)
{
  return pfirst<Packet2d>(_mm_min_sd(a, _mm_unpackhi_pd(a,a)));
}
template<> EIGEN_STRONG_INLINE int predux_min<Packet4i>(const Packet4i& a)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  Packet4i tmp = _mm_min_epi32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0,0,3,2)));
  return pfirst<Packet4i>(_mm_min_epi32(tmp,_mm_shuffle_epi32(tmp, 1)));
#else
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (eg., it does not like using std::min after the pstore !!)
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  int aux0 = aux[0]<aux[1] ? aux[0] : aux[1];
  int aux2 = aux[2]<aux[3] ? aux[2] : aux[3];
  return aux0<aux2 ? aux0 : aux2;
#endif // EIGEN_VECTORIZE_SSE4_1
}

// max
template<> EIGEN_STRONG_INLINE float predux_max<Packet4f>(const Packet4f& a)
{
  Packet4f tmp = _mm_max_ps(a, _mm_movehl_ps(a,a));
  return pfirst<Packet4f>(_mm_max_ss(tmp, _mm_shuffle_ps(tmp,tmp, 1)));
}
template<> EIGEN_STRONG_INLINE double predux_max<Packet2d>(const Packet2d& a)
{
  return pfirst<Packet2d>(_mm_max_sd(a, _mm_unpackhi_pd(a,a)));
}
template<> EIGEN_STRONG_INLINE int predux_max<Packet4i>(const Packet4i& a)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  Packet4i tmp = _mm_max_epi32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0,0,3,2)));
  return pfirst<Packet4i>(_mm_max_epi32(tmp,_mm_shuffle_epi32(tmp, 1)));
#else
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (eg., it does not like using std::min after the pstore !!)
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  int aux0 = aux[0]>aux[1] ? aux[0] : aux[1];
  int aux2 = aux[2]>aux[3] ? aux[2] : aux[3];
  return aux0>aux2 ? aux0 : aux2;
#endif // EIGEN_VECTORIZE_SSE4_1
}

#if EIGEN_COMP_GNUC
// template <> EIGEN_STRONG_INLINE Packet4f pmadd(const Packet4f&  a, const Packet4f&  b, const Packet4f&  c)
// {
//   Packet4f res = b;
//   asm("mulps %[a], %[b] \n\taddps %[c], %[b]" : [b] "+x" (res) : [a] "x" (a), [c] "x" (c));
//   return res;
// }
// EIGEN_STRONG_INLINE Packet4i _mm_alignr_epi8(const Packet4i&  a, const Packet4i&  b, const int i)
// {
//   Packet4i res = a;
//   asm("palignr %[i], %[a], %[b] " : [b] "+x" (res) : [a] "x" (a), [i] "i" (i));
//   return res;
// }
#endif

#ifdef EIGEN_VECTORIZE_SSSE3
// SSSE3 versions
template<int Offset>
struct palign_impl<Offset,Packet4f>
{
  static EIGEN_STRONG_INLINE void run(Packet4f& first, const Packet4f& second)
  {
    if (Offset!=0)
      first = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(second), _mm_castps_si128(first), Offset*4));
  }
};

template<int Offset>
struct palign_impl<Offset,Packet4i>
{
  static EIGEN_STRONG_INLINE void run(Packet4i& first, const Packet4i& second)
  {
    if (Offset!=0)
      first = _mm_alignr_epi8(second,first, Offset*4);
  }
};

template<int Offset>
struct palign_impl<Offset,Packet2d>
{
  static EIGEN_STRONG_INLINE void run(Packet2d& first, const Packet2d& second)
  {
    if (Offset==1)
      first = _mm_castsi128_pd(_mm_alignr_epi8(_mm_castpd_si128(second), _mm_castpd_si128(first), 8));
  }
};
#else
// SSE2 versions
template<int Offset>
struct palign_impl<Offset,Packet4f>
{
  static EIGEN_STRONG_INLINE void run(Packet4f& first, const Packet4f& second)
  {
    if (Offset==1)
    {
      first = _mm_move_ss(first,second);
      first = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(first),0x39));
    }
    else if (Offset==2)
    {
      first = _mm_movehl_ps(first,first);
      first = _mm_movelh_ps(first,second);
    }
    else if (Offset==3)
    {
      first = _mm_move_ss(first,second);
      first = _mm_shuffle_ps(first,second,0x93);
    }
  }
};

template<int Offset>
struct palign_impl<Offset,Packet4i>
{
  static EIGEN_STRONG_INLINE void run(Packet4i& first, const Packet4i& second)
  {
    if (Offset==1)
    {
      first = _mm_castps_si128(_mm_move_ss(_mm_castsi128_ps(first),_mm_castsi128_ps(second)));
      first = _mm_shuffle_epi32(first,0x39);
    }
    else if (Offset==2)
    {
      first = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(first),_mm_castsi128_ps(first)));
      first = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(first),_mm_castsi128_ps(second)));
    }
    else if (Offset==3)
    {
      first = _mm_castps_si128(_mm_move_ss(_mm_castsi128_ps(first),_mm_castsi128_ps(second)));
      first = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(first),_mm_castsi128_ps(second),0x93));
    }
  }
};

template<int Offset>
struct palign_impl<Offset,Packet2d>
{
  static EIGEN_STRONG_INLINE void run(Packet2d& first, const Packet2d& second)
  {
    if (Offset==1)
    {
      first = _mm_castps_pd(_mm_movehl_ps(_mm_castpd_ps(first),_mm_castpd_ps(first)));
      first = _mm_castps_pd(_mm_movelh_ps(_mm_castpd_ps(first),_mm_castpd_ps(second)));
    }
  }
};
#endif

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet4f,4>& kernel) {
  _MM_TRANSPOSE4_PS(kernel.packet[0], kernel.packet[1], kernel.packet[2], kernel.packet[3]);
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet2d,2>& kernel) {
  __m128d tmp = _mm_unpackhi_pd(kernel.packet[0], kernel.packet[1]);
  kernel.packet[0] = _mm_unpacklo_pd(kernel.packet[0], kernel.packet[1]);
  kernel.packet[1] = tmp;
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet4i,4>& kernel) {
  __m128i T0 = _mm_unpacklo_epi32(kernel.packet[0], kernel.packet[1]);
  __m128i T1 = _mm_unpacklo_epi32(kernel.packet[2], kernel.packet[3]);
  __m128i T2 = _mm_unpackhi_epi32(kernel.packet[0], kernel.packet[1]);
  __m128i T3 = _mm_unpackhi_epi32(kernel.packet[2], kernel.packet[3]);

  kernel.packet[0] = _mm_unpacklo_epi64(T0, T1);
  kernel.packet[1] = _mm_unpackhi_epi64(T0, T1);
  kernel.packet[2] = _mm_unpacklo_epi64(T2, T3);
  kernel.packet[3] = _mm_unpackhi_epi64(T2, T3);
}

template<> EIGEN_STRONG_INLINE Packet4i pblend(const Selector<4>& ifPacket, const Packet4i& thenPacket, const Packet4i& elsePacket) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i select = _mm_set_epi32(ifPacket.select[3], ifPacket.select[2], ifPacket.select[1], ifPacket.select[0]);
  __m128i false_mask = _mm_cmpeq_epi32(select, zero);
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_blendv_epi8(thenPacket, elsePacket, false_mask);
#else
  return _mm_or_si128(_mm_andnot_si128(false_mask, thenPacket), _mm_and_si128(false_mask, elsePacket));
#endif
}
template<> EIGEN_STRONG_INLINE Packet4f pblend(const Selector<4>& ifPacket, const Packet4f& thenPacket, const Packet4f& elsePacket) {
  const __m128 zero = _mm_setzero_ps();
  const __m128 select = _mm_set_ps(ifPacket.select[3], ifPacket.select[2], ifPacket.select[1], ifPacket.select[0]);
  __m128 false_mask = _mm_cmpeq_ps(select, zero);
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_blendv_ps(thenPacket, elsePacket, false_mask);
#else
  return _mm_or_ps(_mm_andnot_ps(false_mask, thenPacket), _mm_and_ps(false_mask, elsePacket));
#endif
}
template<> EIGEN_STRONG_INLINE Packet2d pblend(const Selector<2>& ifPacket, const Packet2d& thenPacket, const Packet2d& elsePacket) {
  const __m128d zero = _mm_setzero_pd();
  const __m128d select = _mm_set_pd(ifPacket.select[1], ifPacket.select[0]);
  __m128d false_mask = _mm_cmpeq_pd(select, zero);
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_blendv_pd(thenPacket, elsePacket, false_mask);
#else
  return _mm_or_pd(_mm_andnot_pd(false_mask, thenPacket), _mm_and_pd(false_mask, elsePacket));
#endif
}

template<> EIGEN_STRONG_INLINE Packet4f pinsertfirst(const Packet4f& a, float b)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_blend_ps(a,pset1<Packet4f>(b),1);
#else
  return _mm_move_ss(a, _mm_load_ss(&b));
#endif
}

template<> EIGEN_STRONG_INLINE Packet2d pinsertfirst(const Packet2d& a, double b)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_blend_pd(a,pset1<Packet2d>(b),1);
#else
  return _mm_move_sd(a, _mm_load_sd(&b));
#endif
}

template<> EIGEN_STRONG_INLINE Packet4f pinsertlast(const Packet4f& a, float b)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_blend_ps(a,pset1<Packet4f>(b),(1<<3));
#else
  const Packet4f mask = _mm_castsi128_ps(_mm_setr_epi32(0x0,0x0,0x0,0xFFFFFFFF));
  return _mm_or_ps(_mm_andnot_ps(mask, a), _mm_and_ps(mask, pset1<Packet4f>(b)));
#endif
}

template<> EIGEN_STRONG_INLINE Packet2d pinsertlast(const Packet2d& a, double b)
{
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_blend_pd(a,pset1<Packet2d>(b),(1<<1));
#else
  const Packet2d mask = _mm_castsi128_pd(_mm_setr_epi32(0x0,0x0,0xFFFFFFFF,0xFFFFFFFF));
  return _mm_or_pd(_mm_andnot_pd(mask, a), _mm_and_pd(mask, pset1<Packet2d>(b)));
#endif
}

// Scalar path for pmadd with FMA to ensure consistency with vectorized path.
#ifdef __FMA__
template<> EIGEN_STRONG_INLINE float pmadd(const float& a, const float& b, const float& c) {
  return ::fmaf(a,b,c);
}
template<> EIGEN_STRONG_INLINE double pmadd(const double& a, const double& b, const double& c) {
  return ::fma(a,b,c);
}
#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKET_MATH_SSE_H
