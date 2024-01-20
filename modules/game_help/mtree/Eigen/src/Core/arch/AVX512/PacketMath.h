// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner (benoit.steiner.goog@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_AVX512_H
#define EIGEN_PACKET_MATH_AVX512_H

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
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif
#endif

typedef __m512 Packet16f;
typedef __m512i Packet16i;
typedef __m512d Packet8d;

template <>
struct is_arithmetic<__m512> {
  enum { value = true };
};
template <>
struct is_arithmetic<__m512i> {
  enum { value = true };
};
template <>
struct is_arithmetic<__m512d> {
  enum { value = true };
};

template<> struct packet_traits<float>  : default_packet_traits
{
  typedef Packet16f type;
  typedef Packet8f half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 16,
    HasHalfPacket = 1,
#if EIGEN_GNUC_AT_LEAST(5, 3)
#ifdef EIGEN_VECTORIZE_AVX512DQ
    HasLog = 1,
#endif
    HasExp = 1,
    HasSqrt = EIGEN_FAST_MATH,
    HasRsqrt = EIGEN_FAST_MATH,
#endif
    HasDiv = 1
  };
 };
template<> struct packet_traits<double> : default_packet_traits
{
  typedef Packet8d type;
  typedef Packet4d half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 8,
    HasHalfPacket = 1,
#if EIGEN_GNUC_AT_LEAST(5, 3)
    HasSqrt = EIGEN_FAST_MATH,
    HasRsqrt = EIGEN_FAST_MATH,
#endif
    HasDiv = 1
  };
};

/* TODO Implement AVX512 for integers
template<> struct packet_traits<int>    : default_packet_traits
{
  typedef Packet16i type;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=8
  };
};
*/

template <>
struct unpacket_traits<Packet16f> {
  typedef float type;
  typedef Packet8f half;
  enum { size = 16, alignment=Aligned64 };
};
template <>
struct unpacket_traits<Packet8d> {
  typedef double type;
  typedef Packet4d half;
  enum { size = 8, alignment=Aligned64 };
};
template <>
struct unpacket_traits<Packet16i> {
  typedef int type;
  typedef Packet8i half;
  enum { size = 16, alignment=Aligned64 };
};

template <>
EIGEN_STRONG_INLINE Packet16f pset1<Packet16f>(const float& from) {
  return _mm512_set1_ps(from);
}
template <>
EIGEN_STRONG_INLINE Packet8d pset1<Packet8d>(const double& from) {
  return _mm512_set1_pd(from);
}
template <>
EIGEN_STRONG_INLINE Packet16i pset1<Packet16i>(const int& from) {
  return _mm512_set1_epi32(from);
}

template <>
EIGEN_STRONG_INLINE Packet16f pload1<Packet16f>(const float* from) {
  return _mm512_broadcastss_ps(_mm_load_ps1(from));
}
template <>
EIGEN_STRONG_INLINE Packet8d pload1<Packet8d>(const double* from) {
  return _mm512_broadcastsd_pd(_mm_load_pd1(from));
}

template <>
EIGEN_STRONG_INLINE Packet16f plset<Packet16f>(const float& a) {
  return _mm512_add_ps(
      _mm512_set1_ps(a),
      _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f,
                    4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
}
template <>
EIGEN_STRONG_INLINE Packet8d plset<Packet8d>(const double& a) {
  return _mm512_add_pd(_mm512_set1_pd(a),
                       _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0));
}

template <>
EIGEN_STRONG_INLINE Packet16f padd<Packet16f>(const Packet16f& a,
                                              const Packet16f& b) {
  return _mm512_add_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8d padd<Packet8d>(const Packet8d& a,
                                            const Packet8d& b) {
  return _mm512_add_pd(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet16f psub<Packet16f>(const Packet16f& a,
                                              const Packet16f& b) {
  return _mm512_sub_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8d psub<Packet8d>(const Packet8d& a,
                                            const Packet8d& b) {
  return _mm512_sub_pd(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet16f pnegate(const Packet16f& a) {
  return _mm512_sub_ps(_mm512_set1_ps(0.0), a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pnegate(const Packet8d& a) {
  return _mm512_sub_pd(_mm512_set1_pd(0.0), a);
}

template <>
EIGEN_STRONG_INLINE Packet16f pconj(const Packet16f& a) {
  return a;
}
template <>
EIGEN_STRONG_INLINE Packet8d pconj(const Packet8d& a) {
  return a;
}
template <>
EIGEN_STRONG_INLINE Packet16i pconj(const Packet16i& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE Packet16f pmul<Packet16f>(const Packet16f& a,
                                              const Packet16f& b) {
  return _mm512_mul_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8d pmul<Packet8d>(const Packet8d& a,
                                            const Packet8d& b) {
  return _mm512_mul_pd(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet16f pdiv<Packet16f>(const Packet16f& a,
                                              const Packet16f& b) {
  return _mm512_div_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8d pdiv<Packet8d>(const Packet8d& a,
                                            const Packet8d& b) {
  return _mm512_div_pd(a, b);
}

#ifdef __FMA__
template <>
EIGEN_STRONG_INLINE Packet16f pmadd(const Packet16f& a, const Packet16f& b,
                                    const Packet16f& c) {
  return _mm512_fmadd_ps(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet8d pmadd(const Packet8d& a, const Packet8d& b,
                                   const Packet8d& c) {
  return _mm512_fmadd_pd(a, b, c);
}
#endif

template <>
EIGEN_STRONG_INLINE Packet16f pmin<Packet16f>(const Packet16f& a,
                                              const Packet16f& b) {
  // Arguments are reversed to match NaN propagation behavior of std::min.
  return _mm512_min_ps(b, a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pmin<Packet8d>(const Packet8d& a,
                                            const Packet8d& b) {
  // Arguments are reversed to match NaN propagation behavior of std::min.
  return _mm512_min_pd(b, a);
}

template <>
EIGEN_STRONG_INLINE Packet16f pmax<Packet16f>(const Packet16f& a,
                                              const Packet16f& b) {
  // Arguments are reversed to match NaN propagation behavior of std::max.
  return _mm512_max_ps(b, a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pmax<Packet8d>(const Packet8d& a,
                                            const Packet8d& b) {
  // Arguments are reversed to match NaN propagation behavior of std::max.
  return _mm512_max_pd(b, a);
}

template <>
EIGEN_STRONG_INLINE Packet16f pand<Packet16f>(const Packet16f& a,
                                              const Packet16f& b) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return _mm512_and_ps(a, b);
#else
  Packet16f res = _mm512_undefined_ps();
  Packet4f lane0_a = _mm512_extractf32x4_ps(a, 0);
  Packet4f lane0_b = _mm512_extractf32x4_ps(b, 0);
  res = _mm512_insertf32x4(res, _mm_and_ps(lane0_a, lane0_b), 0);

  Packet4f lane1_a = _mm512_extractf32x4_ps(a, 1);
  Packet4f lane1_b = _mm512_extractf32x4_ps(b, 1);
  res = _mm512_insertf32x4(res, _mm_and_ps(lane1_a, lane1_b), 1);

  Packet4f lane2_a = _mm512_extractf32x4_ps(a, 2);
  Packet4f lane2_b = _mm512_extractf32x4_ps(b, 2);
  res = _mm512_insertf32x4(res, _mm_and_ps(lane2_a, lane2_b), 2);

  Packet4f lane3_a = _mm512_extractf32x4_ps(a, 3);
  Packet4f lane3_b = _mm512_extractf32x4_ps(b, 3);
  res = _mm512_insertf32x4(res, _mm_and_ps(lane3_a, lane3_b), 3);

  return res;
#endif
}
template <>
EIGEN_STRONG_INLINE Packet8d pand<Packet8d>(const Packet8d& a,
                                            const Packet8d& b) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return _mm512_and_pd(a, b);
#else
  Packet8d res = _mm512_undefined_pd();
  Packet4d lane0_a = _mm512_extractf64x4_pd(a, 0);
  Packet4d lane0_b = _mm512_extractf64x4_pd(b, 0);
  res = _mm512_insertf64x4(res, _mm256_and_pd(lane0_a, lane0_b), 0);

  Packet4d lane1_a = _mm512_extractf64x4_pd(a, 1);
  Packet4d lane1_b = _mm512_extractf64x4_pd(b, 1);
  res = _mm512_insertf64x4(res, _mm256_and_pd(lane1_a, lane1_b), 1);

  return res;
#endif
}
template <>
EIGEN_STRONG_INLINE Packet16f por<Packet16f>(const Packet16f& a,
                                             const Packet16f& b) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return _mm512_or_ps(a, b);
#else
  Packet16f res = _mm512_undefined_ps();
  Packet4f lane0_a = _mm512_extractf32x4_ps(a, 0);
  Packet4f lane0_b = _mm512_extractf32x4_ps(b, 0);
  res = _mm512_insertf32x4(res, _mm_or_ps(lane0_a, lane0_b), 0);

  Packet4f lane1_a = _mm512_extractf32x4_ps(a, 1);
  Packet4f lane1_b = _mm512_extractf32x4_ps(b, 1);
  res = _mm512_insertf32x4(res, _mm_or_ps(lane1_a, lane1_b), 1);

  Packet4f lane2_a = _mm512_extractf32x4_ps(a, 2);
  Packet4f lane2_b = _mm512_extractf32x4_ps(b, 2);
  res = _mm512_insertf32x4(res, _mm_or_ps(lane2_a, lane2_b), 2);

  Packet4f lane3_a = _mm512_extractf32x4_ps(a, 3);
  Packet4f lane3_b = _mm512_extractf32x4_ps(b, 3);
  res = _mm512_insertf32x4(res, _mm_or_ps(lane3_a, lane3_b), 3);

  return res;
#endif
}

template <>
EIGEN_STRONG_INLINE Packet8d por<Packet8d>(const Packet8d& a,
                                           const Packet8d& b) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return _mm512_or_pd(a, b);
#else
  Packet8d res = _mm512_undefined_pd();
  Packet4d lane0_a = _mm512_extractf64x4_pd(a, 0);
  Packet4d lane0_b = _mm512_extractf64x4_pd(b, 0);
  res = _mm512_insertf64x4(res, _mm256_or_pd(lane0_a, lane0_b), 0);

  Packet4d lane1_a = _mm512_extractf64x4_pd(a, 1);
  Packet4d lane1_b = _mm512_extractf64x4_pd(b, 1);
  res = _mm512_insertf64x4(res, _mm256_or_pd(lane1_a, lane1_b), 1);

  return res;
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16f pxor<Packet16f>(const Packet16f& a,
                                              const Packet16f& b) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return _mm512_xor_ps(a, b);
#else
  Packet16f res = _mm512_undefined_ps();
  Packet4f lane0_a = _mm512_extractf32x4_ps(a, 0);
  Packet4f lane0_b = _mm512_extractf32x4_ps(b, 0);
  res = _mm512_insertf32x4(res, _mm_xor_ps(lane0_a, lane0_b), 0);

  Packet4f lane1_a = _mm512_extractf32x4_ps(a, 1);
  Packet4f lane1_b = _mm512_extractf32x4_ps(b, 1);
  res = _mm512_insertf32x4(res, _mm_xor_ps(lane1_a, lane1_b), 1);

  Packet4f lane2_a = _mm512_extractf32x4_ps(a, 2);
  Packet4f lane2_b = _mm512_extractf32x4_ps(b, 2);
  res = _mm512_insertf32x4(res, _mm_xor_ps(lane2_a, lane2_b), 2);

  Packet4f lane3_a = _mm512_extractf32x4_ps(a, 3);
  Packet4f lane3_b = _mm512_extractf32x4_ps(b, 3);
  res = _mm512_insertf32x4(res, _mm_xor_ps(lane3_a, lane3_b), 3);

  return res;
#endif
}
template <>
EIGEN_STRONG_INLINE Packet8d pxor<Packet8d>(const Packet8d& a,
                                            const Packet8d& b) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return _mm512_xor_pd(a, b);
#else
  Packet8d res = _mm512_undefined_pd();
  Packet4d lane0_a = _mm512_extractf64x4_pd(a, 0);
  Packet4d lane0_b = _mm512_extractf64x4_pd(b, 0);
  res = _mm512_insertf64x4(res, _mm256_xor_pd(lane0_a, lane0_b), 0);

  Packet4d lane1_a = _mm512_extractf64x4_pd(a, 1);
  Packet4d lane1_b = _mm512_extractf64x4_pd(b, 1);
  res = _mm512_insertf64x4(res, _mm256_xor_pd(lane1_a, lane1_b), 1);

  return res;
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16f pandnot<Packet16f>(const Packet16f& a,
                                                 const Packet16f& b) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return _mm512_andnot_ps(a, b);
#else
  Packet16f res = _mm512_undefined_ps();
  Packet4f lane0_a = _mm512_extractf32x4_ps(a, 0);
  Packet4f lane0_b = _mm512_extractf32x4_ps(b, 0);
  res = _mm512_insertf32x4(res, _mm_andnot_ps(lane0_a, lane0_b), 0);

  Packet4f lane1_a = _mm512_extractf32x4_ps(a, 1);
  Packet4f lane1_b = _mm512_extractf32x4_ps(b, 1);
  res = _mm512_insertf32x4(res, _mm_andnot_ps(lane1_a, lane1_b), 1);

  Packet4f lane2_a = _mm512_extractf32x4_ps(a, 2);
  Packet4f lane2_b = _mm512_extractf32x4_ps(b, 2);
  res = _mm512_insertf32x4(res, _mm_andnot_ps(lane2_a, lane2_b), 2);

  Packet4f lane3_a = _mm512_extractf32x4_ps(a, 3);
  Packet4f lane3_b = _mm512_extractf32x4_ps(b, 3);
  res = _mm512_insertf32x4(res, _mm_andnot_ps(lane3_a, lane3_b), 3);

  return res;
#endif
}
template <>
EIGEN_STRONG_INLINE Packet8d pandnot<Packet8d>(const Packet8d& a,
                                               const Packet8d& b) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return _mm512_andnot_pd(a, b);
#else
  Packet8d res = _mm512_undefined_pd();
  Packet4d lane0_a = _mm512_extractf64x4_pd(a, 0);
  Packet4d lane0_b = _mm512_extractf64x4_pd(b, 0);
  res = _mm512_insertf64x4(res, _mm256_andnot_pd(lane0_a, lane0_b), 0);

  Packet4d lane1_a = _mm512_extractf64x4_pd(a, 1);
  Packet4d lane1_b = _mm512_extractf64x4_pd(b, 1);
  res = _mm512_insertf64x4(res, _mm256_andnot_pd(lane1_a, lane1_b), 1);

  return res;
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16f pload<Packet16f>(const float* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_ps(from);
}
template <>
EIGEN_STRONG_INLINE Packet8d pload<Packet8d>(const double* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_pd(from);
}
template <>
EIGEN_STRONG_INLINE Packet16i pload<Packet16i>(const int* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}

template <>
EIGEN_STRONG_INLINE Packet16f ploadu<Packet16f>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_ps(from);
}
template <>
EIGEN_STRONG_INLINE Packet8d ploadu<Packet8d>(const double* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_pd(from);
}
template <>
EIGEN_STRONG_INLINE Packet16i ploadu<Packet16i>(const int* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}

// Loads 8 floats from memory a returns the packet
// {a0, a0  a1, a1, a2, a2, a3, a3, a4, a4, a5, a5, a6, a6, a7, a7}
template <>
EIGEN_STRONG_INLINE Packet16f ploaddup<Packet16f>(const float* from) {
  __m256i low_half = _mm256_load_si256(reinterpret_cast<const __m256i*>(from));
  __m512 even_elements = _mm512_castsi512_ps(_mm512_cvtepu32_epi64(low_half));
  __m512 pairs = _mm512_permute_ps(even_elements, _MM_SHUFFLE(2, 2, 0, 0));
  return pairs;
}
// Loads 4 doubles from memory a returns the packet {a0, a0  a1, a1, a2, a2, a3,
// a3}
template <>
EIGEN_STRONG_INLINE Packet8d ploaddup<Packet8d>(const double* from) {
 __m512d x = _mm512_setzero_pd();
  x = _mm512_insertf64x2(x, _mm_loaddup_pd(&from[0]), 0);
  x = _mm512_insertf64x2(x, _mm_loaddup_pd(&from[1]), 1);
  x = _mm512_insertf64x2(x, _mm_loaddup_pd(&from[2]), 2);
  x = _mm512_insertf64x2(x, _mm_loaddup_pd(&from[3]), 3);
  return x;
}

// Loads 4 floats from memory a returns the packet
// {a0, a0  a0, a0, a1, a1, a1, a1, a2, a2, a2, a2, a3, a3, a3, a3}
template <>
EIGEN_STRONG_INLINE Packet16f ploadquad<Packet16f>(const float* from) {
  Packet16f tmp = _mm512_undefined_ps();
  tmp = _mm512_insertf32x4(tmp, _mm_load_ps1(from), 0);
  tmp = _mm512_insertf32x4(tmp, _mm_load_ps1(from + 1), 1);
  tmp = _mm512_insertf32x4(tmp, _mm_load_ps1(from + 2), 2);
  tmp = _mm512_insertf32x4(tmp, _mm_load_ps1(from + 3), 3);
  return tmp;
}
// Loads 2 doubles from memory a returns the packet
// {a0, a0  a0, a0, a1, a1, a1, a1}
template <>
EIGEN_STRONG_INLINE Packet8d ploadquad<Packet8d>(const double* from) {
  __m128d tmp0 = _mm_load_pd1(from);
  __m256d lane0 = _mm256_broadcastsd_pd(tmp0);
  __m128d tmp1 = _mm_load_pd1(from + 1);
  __m256d lane1 = _mm256_broadcastsd_pd(tmp1);
  __m512d tmp = _mm512_undefined_pd();
  tmp = _mm512_insertf64x4(tmp, lane0, 0);
  return _mm512_insertf64x4(tmp, lane1, 1);
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet16f& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_ps(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstore<double>(double* to, const Packet8d& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_pd(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstore<int>(int* to, const Packet16i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm512_storeu_si512(reinterpret_cast<__m512i*>(to),
                                                from);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet16f& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_ps(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const Packet8d& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_pd(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<int>(int* to, const Packet16i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from);
}

template <>
EIGEN_DEVICE_FUNC inline Packet16f pgather<float, Packet16f>(const float* from,
                                                             Index stride) {
  Packet16i stride_vector = _mm512_set1_epi32(stride);
  Packet16i stride_multiplier =
      _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier);

  return _mm512_i32gather_ps(indices, from, 4);
}
template <>
EIGEN_DEVICE_FUNC inline Packet8d pgather<double, Packet8d>(const double* from,
                                                            Index stride) {
  Packet8i stride_vector = _mm256_set1_epi32(stride);
  Packet8i stride_multiplier = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier);

  return _mm512_i32gather_pd(indices, from, 8);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<float, Packet16f>(float* to,
                                                         const Packet16f& from,
                                                         Index stride) {
  Packet16i stride_vector = _mm512_set1_epi32(stride);
  Packet16i stride_multiplier =
      _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier);
  _mm512_i32scatter_ps(to, indices, from, 4);
}
template <>
EIGEN_DEVICE_FUNC inline void pscatter<double, Packet8d>(double* to,
                                                         const Packet8d& from,
                                                         Index stride) {
  Packet8i stride_vector = _mm256_set1_epi32(stride);
  Packet8i stride_multiplier = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier);
  _mm512_i32scatter_pd(to, indices, from, 8);
}

template <>
EIGEN_STRONG_INLINE void pstore1<Packet16f>(float* to, const float& a) {
  Packet16f pa = pset1<Packet16f>(a);
  pstore(to, pa);
}
template <>
EIGEN_STRONG_INLINE void pstore1<Packet8d>(double* to, const double& a) {
  Packet8d pa = pset1<Packet8d>(a);
  pstore(to, pa);
}
template <>
EIGEN_STRONG_INLINE void pstore1<Packet16i>(int* to, const int& a) {
  Packet16i pa = pset1<Packet16i>(a);
  pstore(to, pa);
}

template<> EIGEN_STRONG_INLINE void prefetch<float>(const float*   addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }
template<> EIGEN_STRONG_INLINE void prefetch<double>(const double* addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }
template<> EIGEN_STRONG_INLINE void prefetch<int>(const int*       addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }

template <>
EIGEN_STRONG_INLINE float pfirst<Packet16f>(const Packet16f& a) {
  return _mm_cvtss_f32(_mm512_extractf32x4_ps(a, 0));
}
template <>
EIGEN_STRONG_INLINE double pfirst<Packet8d>(const Packet8d& a) {
  return _mm_cvtsd_f64(_mm256_extractf128_pd(_mm512_extractf64x4_pd(a, 0), 0));
}
template <>
EIGEN_STRONG_INLINE int pfirst<Packet16i>(const Packet16i& a) {
  return _mm_extract_epi32(_mm512_extracti32x4_epi32(a, 0), 0);
}

template<> EIGEN_STRONG_INLINE Packet16f preverse(const Packet16f& a)
{
  return _mm512_permutexvar_ps(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), a);
}

template<> EIGEN_STRONG_INLINE Packet8d preverse(const Packet8d& a)
{
  return _mm512_permutexvar_pd(_mm512_set_epi32(0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7), a);
}

template<> EIGEN_STRONG_INLINE Packet16f pabs(const Packet16f& a)
{
  // _mm512_abs_ps intrinsic not found, so hack around it
  return (__m512)_mm512_and_si512((__m512i)a, _mm512_set1_epi32(0x7fffffff));
}
template <>
EIGEN_STRONG_INLINE Packet8d pabs(const Packet8d& a) {
  // _mm512_abs_ps intrinsic not found, so hack around it
  return (__m512d)_mm512_and_si512((__m512i)a,
                                   _mm512_set1_epi64(0x7fffffffffffffff));
}

#ifdef EIGEN_VECTORIZE_AVX512DQ
// AVX512F does not define _mm512_extractf32x8_ps to extract _m256 from _m512
#define EIGEN_EXTRACT_8f_FROM_16f(INPUT, OUTPUT)                           \
  __m256 OUTPUT##_0 = _mm512_extractf32x8_ps(INPUT, 0);                    \
  __m256 OUTPUT##_1 = _mm512_extractf32x8_ps(INPUT, 1)
#else
#define EIGEN_EXTRACT_8f_FROM_16f(INPUT, OUTPUT)                \
  __m256 OUTPUT##_0 = _mm256_insertf128_ps(                     \
      _mm256_castps128_ps256(_mm512_extractf32x4_ps(INPUT, 0)), \
      _mm512_extractf32x4_ps(INPUT, 1), 1);                     \
  __m256 OUTPUT##_1 = _mm256_insertf128_ps(                     \
      _mm256_castps128_ps256(_mm512_extractf32x4_ps(INPUT, 2)), \
      _mm512_extractf32x4_ps(INPUT, 3), 1);
#endif

#ifdef EIGEN_VECTORIZE_AVX512DQ
#define EIGEN_INSERT_8f_INTO_16f(OUTPUT, INPUTA, INPUTB) \
  OUTPUT = _mm512_insertf32x8(OUTPUT, INPUTA, 0);        \
  OUTPUT = _mm512_insertf32x8(OUTPUT, INPUTB, 1);
#else
#define EIGEN_INSERT_8f_INTO_16f(OUTPUT, INPUTA, INPUTB)                    \
  OUTPUT = _mm512_insertf32x4(OUTPUT, _mm256_extractf128_ps(INPUTA, 0), 0); \
  OUTPUT = _mm512_insertf32x4(OUTPUT, _mm256_extractf128_ps(INPUTA, 1), 1); \
  OUTPUT = _mm512_insertf32x4(OUTPUT, _mm256_extractf128_ps(INPUTB, 0), 2); \
  OUTPUT = _mm512_insertf32x4(OUTPUT, _mm256_extractf128_ps(INPUTB, 1), 3);
#endif
template<> EIGEN_STRONG_INLINE Packet16f preduxp<Packet16f>(const Packet16f*
vecs)
{
  EIGEN_EXTRACT_8f_FROM_16f(vecs[0], vecs0);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[1], vecs1);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[2], vecs2);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[3], vecs3);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[4], vecs4);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[5], vecs5);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[6], vecs6);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[7], vecs7);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[8], vecs8);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[9], vecs9);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[10], vecs10);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[11], vecs11);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[12], vecs12);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[13], vecs13);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[14], vecs14);
  EIGEN_EXTRACT_8f_FROM_16f(vecs[15], vecs15);

  __m256 hsum1 = _mm256_hadd_ps(vecs0_0, vecs1_0);
  __m256 hsum2 = _mm256_hadd_ps(vecs2_0, vecs3_0);
  __m256 hsum3 = _mm256_hadd_ps(vecs4_0, vecs5_0);
  __m256 hsum4 = _mm256_hadd_ps(vecs6_0, vecs7_0);

  __m256 hsum5 = _mm256_hadd_ps(hsum1, hsum1);
  __m256 hsum6 = _mm256_hadd_ps(hsum2, hsum2);
  __m256 hsum7 = _mm256_hadd_ps(hsum3, hsum3);
  __m256 hsum8 = _mm256_hadd_ps(hsum4, hsum4);

  __m256 perm1 = _mm256_permute2f128_ps(hsum5, hsum5, 0x23);
  __m256 perm2 = _mm256_permute2f128_ps(hsum6, hsum6, 0x23);
  __m256 perm3 = _mm256_permute2f128_ps(hsum7, hsum7, 0x23);
  __m256 perm4 = _mm256_permute2f128_ps(hsum8, hsum8, 0x23);

  __m256 sum1 = _mm256_add_ps(perm1, hsum5);
  __m256 sum2 = _mm256_add_ps(perm2, hsum6);
  __m256 sum3 = _mm256_add_ps(perm3, hsum7);
  __m256 sum4 = _mm256_add_ps(perm4, hsum8);

  __m256 blend1 = _mm256_blend_ps(sum1, sum2, 0xcc);
  __m256 blend2 = _mm256_blend_ps(sum3, sum4, 0xcc);

  __m256 final = _mm256_blend_ps(blend1, blend2, 0xf0);

  hsum1 = _mm256_hadd_ps(vecs0_1, vecs1_1);
  hsum2 = _mm256_hadd_ps(vecs2_1, vecs3_1);
  hsum3 = _mm256_hadd_ps(vecs4_1, vecs5_1);
  hsum4 = _mm256_hadd_ps(vecs6_1, vecs7_1);

  hsum5 = _mm256_hadd_ps(hsum1, hsum1);
  hsum6 = _mm256_hadd_ps(hsum2, hsum2);
  hsum7 = _mm256_hadd_ps(hsum3, hsum3);
  hsum8 = _mm256_hadd_ps(hsum4, hsum4);

  perm1 = _mm256_permute2f128_ps(hsum5, hsum5, 0x23);
  perm2 = _mm256_permute2f128_ps(hsum6, hsum6, 0x23);
  perm3 = _mm256_permute2f128_ps(hsum7, hsum7, 0x23);
  perm4 = _mm256_permute2f128_ps(hsum8, hsum8, 0x23);

  sum1 = _mm256_add_ps(perm1, hsum5);
  sum2 = _mm256_add_ps(perm2, hsum6);
  sum3 = _mm256_add_ps(perm3, hsum7);
  sum4 = _mm256_add_ps(perm4, hsum8);

  blend1 = _mm256_blend_ps(sum1, sum2, 0xcc);
  blend2 = _mm256_blend_ps(sum3, sum4, 0xcc);

  final = _mm256_add_ps(final, _mm256_blend_ps(blend1, blend2, 0xf0));

  hsum1 = _mm256_hadd_ps(vecs8_0, vecs9_0);
  hsum2 = _mm256_hadd_ps(vecs10_0, vecs11_0);
  hsum3 = _mm256_hadd_ps(vecs12_0, vecs13_0);
  hsum4 = _mm256_hadd_ps(vecs14_0, vecs15_0);

  hsum5 = _mm256_hadd_ps(hsum1, hsum1);
  hsum6 = _mm256_hadd_ps(hsum2, hsum2);
  hsum7 = _mm256_hadd_ps(hsum3, hsum3);
  hsum8 = _mm256_hadd_ps(hsum4, hsum4);

  perm1 = _mm256_permute2f128_ps(hsum5, hsum5, 0x23);
  perm2 = _mm256_permute2f128_ps(hsum6, hsum6, 0x23);
  perm3 = _mm256_permute2f128_ps(hsum7, hsum7, 0x23);
  perm4 = _mm256_permute2f128_ps(hsum8, hsum8, 0x23);

  sum1 = _mm256_add_ps(perm1, hsum5);
  sum2 = _mm256_add_ps(perm2, hsum6);
  sum3 = _mm256_add_ps(perm3, hsum7);
  sum4 = _mm256_add_ps(perm4, hsum8);

  blend1 = _mm256_blend_ps(sum1, sum2, 0xcc);
  blend2 = _mm256_blend_ps(sum3, sum4, 0xcc);

  __m256 final_1 = _mm256_blend_ps(blend1, blend2, 0xf0);

  hsum1 = _mm256_hadd_ps(vecs8_1, vecs9_1);
  hsum2 = _mm256_hadd_ps(vecs10_1, vecs11_1);
  hsum3 = _mm256_hadd_ps(vecs12_1, vecs13_1);
  hsum4 = _mm256_hadd_ps(vecs14_1, vecs15_1);

  hsum5 = _mm256_hadd_ps(hsum1, hsum1);
  hsum6 = _mm256_hadd_ps(hsum2, hsum2);
  hsum7 = _mm256_hadd_ps(hsum3, hsum3);
  hsum8 = _mm256_hadd_ps(hsum4, hsum4);

  perm1 = _mm256_permute2f128_ps(hsum5, hsum5, 0x23);
  perm2 = _mm256_permute2f128_ps(hsum6, hsum6, 0x23);
  perm3 = _mm256_permute2f128_ps(hsum7, hsum7, 0x23);
  perm4 = _mm256_permute2f128_ps(hsum8, hsum8, 0x23);

  sum1 = _mm256_add_ps(perm1, hsum5);
  sum2 = _mm256_add_ps(perm2, hsum6);
  sum3 = _mm256_add_ps(perm3, hsum7);
  sum4 = _mm256_add_ps(perm4, hsum8);

  blend1 = _mm256_blend_ps(sum1, sum2, 0xcc);
  blend2 = _mm256_blend_ps(sum3, sum4, 0xcc);

  final_1 = _mm256_add_ps(final_1, _mm256_blend_ps(blend1, blend2, 0xf0));

  __m512 final_output;

  EIGEN_INSERT_8f_INTO_16f(final_output, final, final_1);
  return final_output;
}

template<> EIGEN_STRONG_INLINE Packet8d preduxp<Packet8d>(const Packet8d* vecs)
{
  Packet4d vecs0_0 = _mm512_extractf64x4_pd(vecs[0], 0);
  Packet4d vecs0_1 = _mm512_extractf64x4_pd(vecs[0], 1);

  Packet4d vecs1_0 = _mm512_extractf64x4_pd(vecs[1], 0);
  Packet4d vecs1_1 = _mm512_extractf64x4_pd(vecs[1], 1);

  Packet4d vecs2_0 = _mm512_extractf64x4_pd(vecs[2], 0);
  Packet4d vecs2_1 = _mm512_extractf64x4_pd(vecs[2], 1);

  Packet4d vecs3_0 = _mm512_extractf64x4_pd(vecs[3], 0);
  Packet4d vecs3_1 = _mm512_extractf64x4_pd(vecs[3], 1);

  Packet4d vecs4_0 = _mm512_extractf64x4_pd(vecs[4], 0);
  Packet4d vecs4_1 = _mm512_extractf64x4_pd(vecs[4], 1);

  Packet4d vecs5_0 = _mm512_extractf64x4_pd(vecs[5], 0);
  Packet4d vecs5_1 = _mm512_extractf64x4_pd(vecs[5], 1);

  Packet4d vecs6_0 = _mm512_extractf64x4_pd(vecs[6], 0);
  Packet4d vecs6_1 = _mm512_extractf64x4_pd(vecs[6], 1);

  Packet4d vecs7_0 = _mm512_extractf64x4_pd(vecs[7], 0);
  Packet4d vecs7_1 = _mm512_extractf64x4_pd(vecs[7], 1);

  Packet4d tmp0, tmp1;

  tmp0 = _mm256_hadd_pd(vecs0_0, vecs1_0);
  tmp0 = _mm256_add_pd(tmp0, _mm256_permute2f128_pd(tmp0, tmp0, 1));

  tmp1 = _mm256_hadd_pd(vecs2_0, vecs3_0);
  tmp1 = _mm256_add_pd(tmp1, _mm256_permute2f128_pd(tmp1, tmp1, 1));

  __m256d final_0 = _mm256_blend_pd(tmp0, tmp1, 0xC);

  tmp0 = _mm256_hadd_pd(vecs0_1, vecs1_1);
  tmp0 = _mm256_add_pd(tmp0, _mm256_permute2f128_pd(tmp0, tmp0, 1));

  tmp1 = _mm256_hadd_pd(vecs2_1, vecs3_1);
  tmp1 = _mm256_add_pd(tmp1, _mm256_permute2f128_pd(tmp1, tmp1, 1));

  final_0 = _mm256_add_pd(final_0, _mm256_blend_pd(tmp0, tmp1, 0xC));

  tmp0 = _mm256_hadd_pd(vecs4_0, vecs5_0);
  tmp0 = _mm256_add_pd(tmp0, _mm256_permute2f128_pd(tmp0, tmp0, 1));

  tmp1 = _mm256_hadd_pd(vecs6_0, vecs7_0);
  tmp1 = _mm256_add_pd(tmp1, _mm256_permute2f128_pd(tmp1, tmp1, 1));

  __m256d final_1 = _mm256_blend_pd(tmp0, tmp1, 0xC);

  tmp0 = _mm256_hadd_pd(vecs4_1, vecs5_1);
  tmp0 = _mm256_add_pd(tmp0, _mm256_permute2f128_pd(tmp0, tmp0, 1));

  tmp1 = _mm256_hadd_pd(vecs6_1, vecs7_1);
  tmp1 = _mm256_add_pd(tmp1, _mm256_permute2f128_pd(tmp1, tmp1, 1));

  final_1 = _mm256_add_pd(final_1, _mm256_blend_pd(tmp0, tmp1, 0xC));

  __m512d final_output = _mm512_insertf64x4(final_output, final_0, 0);

  return _mm512_insertf64x4(final_output, final_1, 1);
}

template <>
EIGEN_STRONG_INLINE float predux<Packet16f>(const Packet16f& a) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  __m256 lane0 = _mm512_extractf32x8_ps(a, 0);
  __m256 lane1 = _mm512_extractf32x8_ps(a, 1);
  Packet8f x = _mm256_add_ps(lane0, lane1);
  return predux<Packet8f>(x);
#else
  __m128 lane0 = _mm512_extractf32x4_ps(a, 0);
  __m128 lane1 = _mm512_extractf32x4_ps(a, 1);
  __m128 lane2 = _mm512_extractf32x4_ps(a, 2);
  __m128 lane3 = _mm512_extractf32x4_ps(a, 3);
  __m128 sum = _mm_add_ps(_mm_add_ps(lane0, lane1), _mm_add_ps(lane2, lane3));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, _mm_permute_ps(sum, 1));
  return _mm_cvtss_f32(sum);
#endif
}
template <>
EIGEN_STRONG_INLINE double predux<Packet8d>(const Packet8d& a) {
  __m256d lane0 = _mm512_extractf64x4_pd(a, 0);
  __m256d lane1 = _mm512_extractf64x4_pd(a, 1);
  __m256d sum = _mm256_add_pd(lane0, lane1);
  __m256d tmp0 = _mm256_hadd_pd(sum, _mm256_permute2f128_pd(sum, sum, 1));
  return _mm_cvtsd_f64(_mm256_castpd256_pd128(_mm256_hadd_pd(tmp0, tmp0)));
}

template <>
EIGEN_STRONG_INLINE Packet8f predux_downto4<Packet16f>(const Packet16f& a) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  __m256 lane0 = _mm512_extractf32x8_ps(a, 0);
  __m256 lane1 = _mm512_extractf32x8_ps(a, 1);
  return _mm256_add_ps(lane0, lane1);
#else
  __m128 lane0 = _mm512_extractf32x4_ps(a, 0);
  __m128 lane1 = _mm512_extractf32x4_ps(a, 1);
  __m128 lane2 = _mm512_extractf32x4_ps(a, 2);
  __m128 lane3 = _mm512_extractf32x4_ps(a, 3);
  __m128 sum0 = _mm_add_ps(lane0, lane2);
  __m128 sum1 = _mm_add_ps(lane1, lane3);
  return _mm256_insertf128_ps(_mm256_castps128_ps256(sum0), sum1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet4d predux_downto4<Packet8d>(const Packet8d& a) {
  __m256d lane0 = _mm512_extractf64x4_pd(a, 0);
  __m256d lane1 = _mm512_extractf64x4_pd(a, 1);
  __m256d res = _mm256_add_pd(lane0, lane1);
  return res;
}

template <>
EIGEN_STRONG_INLINE float predux_mul<Packet16f>(const Packet16f& a) {
//#ifdef EIGEN_VECTORIZE_AVX512DQ
#if 0
  Packet8f lane0 = _mm512_extractf32x8_ps(a, 0);
  Packet8f lane1 = _mm512_extractf32x8_ps(a, 1);
  Packet8f res = pmul(lane0, lane1);
  res = pmul(res, _mm256_permute2f128_ps(res, res, 1));
  res = pmul(res, _mm_permute_ps(res, _MM_SHUFFLE(0, 0, 3, 2)));
  return pfirst(pmul(res, _mm_permute_ps(res, _MM_SHUFFLE(0, 0, 0, 1))));
#else
  __m128 lane0 = _mm512_extractf32x4_ps(a, 0);
  __m128 lane1 = _mm512_extractf32x4_ps(a, 1);
  __m128 lane2 = _mm512_extractf32x4_ps(a, 2);
  __m128 lane3 = _mm512_extractf32x4_ps(a, 3);
  __m128 res = pmul(pmul(lane0, lane1), pmul(lane2, lane3));
  res = pmul(res, _mm_permute_ps(res, _MM_SHUFFLE(0, 0, 3, 2)));
  return pfirst(pmul(res, _mm_permute_ps(res, _MM_SHUFFLE(0, 0, 0, 1))));
#endif
}
template <>
EIGEN_STRONG_INLINE double predux_mul<Packet8d>(const Packet8d& a) {
  __m256d lane0 = _mm512_extractf64x4_pd(a, 0);
  __m256d lane1 = _mm512_extractf64x4_pd(a, 1);
  __m256d res = pmul(lane0, lane1);
  res = pmul(res, _mm256_permute2f128_pd(res, res, 1));
  return pfirst(pmul(res, _mm256_shuffle_pd(res, res, 1)));
}

template <>
EIGEN_STRONG_INLINE float predux_min<Packet16f>(const Packet16f& a) {
  __m128 lane0 = _mm512_extractf32x4_ps(a, 0);
  __m128 lane1 = _mm512_extractf32x4_ps(a, 1);
  __m128 lane2 = _mm512_extractf32x4_ps(a, 2);
  __m128 lane3 = _mm512_extractf32x4_ps(a, 3);
  __m128 res = _mm_min_ps(_mm_min_ps(lane0, lane1), _mm_min_ps(lane2, lane3));
  res = _mm_min_ps(res, _mm_permute_ps(res, _MM_SHUFFLE(0, 0, 3, 2)));
  return pfirst(_mm_min_ps(res, _mm_permute_ps(res, _MM_SHUFFLE(0, 0, 0, 1))));
}
template <>
EIGEN_STRONG_INLINE double predux_min<Packet8d>(const Packet8d& a) {
  __m256d lane0 = _mm512_extractf64x4_pd(a, 0);
  __m256d lane1 = _mm512_extractf64x4_pd(a, 1);
  __m256d res = _mm256_min_pd(lane0, lane1);
  res = _mm256_min_pd(res, _mm256_permute2f128_pd(res, res, 1));
  return pfirst(_mm256_min_pd(res, _mm256_shuffle_pd(res, res, 1)));
}

template <>
EIGEN_STRONG_INLINE float predux_max<Packet16f>(const Packet16f& a) {
  __m128 lane0 = _mm512_extractf32x4_ps(a, 0);
  __m128 lane1 = _mm512_extractf32x4_ps(a, 1);
  __m128 lane2 = _mm512_extractf32x4_ps(a, 2);
  __m128 lane3 = _mm512_extractf32x4_ps(a, 3);
  __m128 res = _mm_max_ps(_mm_max_ps(lane0, lane1), _mm_max_ps(lane2, lane3));
  res = _mm_max_ps(res, _mm_permute_ps(res, _MM_SHUFFLE(0, 0, 3, 2)));
  return pfirst(_mm_max_ps(res, _mm_permute_ps(res, _MM_SHUFFLE(0, 0, 0, 1))));
}

template <>
EIGEN_STRONG_INLINE double predux_max<Packet8d>(const Packet8d& a) {
  __m256d lane0 = _mm512_extractf64x4_pd(a, 0);
  __m256d lane1 = _mm512_extractf64x4_pd(a, 1);
  __m256d res = _mm256_max_pd(lane0, lane1);
  res = _mm256_max_pd(res, _mm256_permute2f128_pd(res, res, 1));
  return pfirst(_mm256_max_pd(res, _mm256_shuffle_pd(res, res, 1)));
}

template <int Offset>
struct palign_impl<Offset, Packet16f> {
  static EIGEN_STRONG_INLINE void run(Packet16f& first,
                                      const Packet16f& second) {
    if (Offset != 0) {
      __m512i first_idx = _mm512_set_epi32(
          Offset + 15, Offset + 14, Offset + 13, Offset + 12, Offset + 11,
          Offset + 10, Offset + 9, Offset + 8, Offset + 7, Offset + 6,
          Offset + 5, Offset + 4, Offset + 3, Offset + 2, Offset + 1, Offset);

      __m512i second_idx =
          _mm512_set_epi32(Offset - 1, Offset - 2, Offset - 3, Offset - 4,
                           Offset - 5, Offset - 6, Offset - 7, Offset - 8,
                           Offset - 9, Offset - 10, Offset - 11, Offset - 12,
                           Offset - 13, Offset - 14, Offset - 15, Offset - 16);

      unsigned short mask = 0xFFFF;
      mask <<= (16 - Offset);

      first = _mm512_permutexvar_ps(first_idx, first);
      Packet16f tmp = _mm512_permutexvar_ps(second_idx, second);
      first = _mm512_mask_blend_ps(mask, first, tmp);
    }
  }
};
template <int Offset>
struct palign_impl<Offset, Packet8d> {
  static EIGEN_STRONG_INLINE void run(Packet8d& first, const Packet8d& second) {
    if (Offset != 0) {
      __m512i first_idx = _mm512_set_epi32(
          0, Offset + 7, 0, Offset + 6, 0, Offset + 5, 0, Offset + 4, 0,
          Offset + 3, 0, Offset + 2, 0, Offset + 1, 0, Offset);

      __m512i second_idx = _mm512_set_epi32(
          0, Offset - 1, 0, Offset - 2, 0, Offset - 3, 0, Offset - 4, 0,
          Offset - 5, 0, Offset - 6, 0, Offset - 7, 0, Offset - 8);

      unsigned char mask = 0xFF;
      mask <<= (8 - Offset);

      first = _mm512_permutexvar_pd(first_idx, first);
      Packet8d tmp = _mm512_permutexvar_pd(second_idx, second);
      first = _mm512_mask_blend_pd(mask, first, tmp);
    }
  }
};


#define PACK_OUTPUT(OUTPUT, INPUT, INDEX, STRIDE) \
  EIGEN_INSERT_8f_INTO_16f(OUTPUT[INDEX], INPUT[INDEX], INPUT[INDEX + STRIDE]);

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet16f, 16>& kernel) {
  __m512 T0 = _mm512_unpacklo_ps(kernel.packet[0], kernel.packet[1]);
  __m512 T1 = _mm512_unpackhi_ps(kernel.packet[0], kernel.packet[1]);
  __m512 T2 = _mm512_unpacklo_ps(kernel.packet[2], kernel.packet[3]);
  __m512 T3 = _mm512_unpackhi_ps(kernel.packet[2], kernel.packet[3]);
  __m512 T4 = _mm512_unpacklo_ps(kernel.packet[4], kernel.packet[5]);
  __m512 T5 = _mm512_unpackhi_ps(kernel.packet[4], kernel.packet[5]);
  __m512 T6 = _mm512_unpacklo_ps(kernel.packet[6], kernel.packet[7]);
  __m512 T7 = _mm512_unpackhi_ps(kernel.packet[6], kernel.packet[7]);
  __m512 T8 = _mm512_unpacklo_ps(kernel.packet[8], kernel.packet[9]);
  __m512 T9 = _mm512_unpackhi_ps(kernel.packet[8], kernel.packet[9]);
  __m512 T10 = _mm512_unpacklo_ps(kernel.packet[10], kernel.packet[11]);
  __m512 T11 = _mm512_unpackhi_ps(kernel.packet[10], kernel.packet[11]);
  __m512 T12 = _mm512_unpacklo_ps(kernel.packet[12], kernel.packet[13]);
  __m512 T13 = _mm512_unpackhi_ps(kernel.packet[12], kernel.packet[13]);
  __m512 T14 = _mm512_unpacklo_ps(kernel.packet[14], kernel.packet[15]);
  __m512 T15 = _mm512_unpackhi_ps(kernel.packet[14], kernel.packet[15]);
  __m512 S0 = _mm512_shuffle_ps(T0, T2, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S1 = _mm512_shuffle_ps(T0, T2, _MM_SHUFFLE(3, 2, 3, 2));
  __m512 S2 = _mm512_shuffle_ps(T1, T3, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S3 = _mm512_shuffle_ps(T1, T3, _MM_SHUFFLE(3, 2, 3, 2));
  __m512 S4 = _mm512_shuffle_ps(T4, T6, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S5 = _mm512_shuffle_ps(T4, T6, _MM_SHUFFLE(3, 2, 3, 2));
  __m512 S6 = _mm512_shuffle_ps(T5, T7, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S7 = _mm512_shuffle_ps(T5, T7, _MM_SHUFFLE(3, 2, 3, 2));
  __m512 S8 = _mm512_shuffle_ps(T8, T10, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S9 = _mm512_shuffle_ps(T8, T10, _MM_SHUFFLE(3, 2, 3, 2));
  __m512 S10 = _mm512_shuffle_ps(T9, T11, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S11 = _mm512_shuffle_ps(T9, T11, _MM_SHUFFLE(3, 2, 3, 2));
  __m512 S12 = _mm512_shuffle_ps(T12, T14, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S13 = _mm512_shuffle_ps(T12, T14, _MM_SHUFFLE(3, 2, 3, 2));
  __m512 S14 = _mm512_shuffle_ps(T13, T15, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S15 = _mm512_shuffle_ps(T13, T15, _MM_SHUFFLE(3, 2, 3, 2));

  EIGEN_EXTRACT_8f_FROM_16f(S0, S0);
  EIGEN_EXTRACT_8f_FROM_16f(S1, S1);
  EIGEN_EXTRACT_8f_FROM_16f(S2, S2);
  EIGEN_EXTRACT_8f_FROM_16f(S3, S3);
  EIGEN_EXTRACT_8f_FROM_16f(S4, S4);
  EIGEN_EXTRACT_8f_FROM_16f(S5, S5);
  EIGEN_EXTRACT_8f_FROM_16f(S6, S6);
  EIGEN_EXTRACT_8f_FROM_16f(S7, S7);
  EIGEN_EXTRACT_8f_FROM_16f(S8, S8);
  EIGEN_EXTRACT_8f_FROM_16f(S9, S9);
  EIGEN_EXTRACT_8f_FROM_16f(S10, S10);
  EIGEN_EXTRACT_8f_FROM_16f(S11, S11);
  EIGEN_EXTRACT_8f_FROM_16f(S12, S12);
  EIGEN_EXTRACT_8f_FROM_16f(S13, S13);
  EIGEN_EXTRACT_8f_FROM_16f(S14, S14);
  EIGEN_EXTRACT_8f_FROM_16f(S15, S15);

  PacketBlock<Packet8f, 32> tmp;

  tmp.packet[0] = _mm256_permute2f128_ps(S0_0, S4_0, 0x20);
  tmp.packet[1] = _mm256_permute2f128_ps(S1_0, S5_0, 0x20);
  tmp.packet[2] = _mm256_permute2f128_ps(S2_0, S6_0, 0x20);
  tmp.packet[3] = _mm256_permute2f128_ps(S3_0, S7_0, 0x20);
  tmp.packet[4] = _mm256_permute2f128_ps(S0_0, S4_0, 0x31);
  tmp.packet[5] = _mm256_permute2f128_ps(S1_0, S5_0, 0x31);
  tmp.packet[6] = _mm256_permute2f128_ps(S2_0, S6_0, 0x31);
  tmp.packet[7] = _mm256_permute2f128_ps(S3_0, S7_0, 0x31);

  tmp.packet[8] = _mm256_permute2f128_ps(S0_1, S4_1, 0x20);
  tmp.packet[9] = _mm256_permute2f128_ps(S1_1, S5_1, 0x20);
  tmp.packet[10] = _mm256_permute2f128_ps(S2_1, S6_1, 0x20);
  tmp.packet[11] = _mm256_permute2f128_ps(S3_1, S7_1, 0x20);
  tmp.packet[12] = _mm256_permute2f128_ps(S0_1, S4_1, 0x31);
  tmp.packet[13] = _mm256_permute2f128_ps(S1_1, S5_1, 0x31);
  tmp.packet[14] = _mm256_permute2f128_ps(S2_1, S6_1, 0x31);
  tmp.packet[15] = _mm256_permute2f128_ps(S3_1, S7_1, 0x31);

  // Second set of _m256 outputs
  tmp.packet[16] = _mm256_permute2f128_ps(S8_0, S12_0, 0x20);
  tmp.packet[17] = _mm256_permute2f128_ps(S9_0, S13_0, 0x20);
  tmp.packet[18] = _mm256_permute2f128_ps(S10_0, S14_0, 0x20);
  tmp.packet[19] = _mm256_permute2f128_ps(S11_0, S15_0, 0x20);
  tmp.packet[20] = _mm256_permute2f128_ps(S8_0, S12_0, 0x31);
  tmp.packet[21] = _mm256_permute2f128_ps(S9_0, S13_0, 0x31);
  tmp.packet[22] = _mm256_permute2f128_ps(S10_0, S14_0, 0x31);
  tmp.packet[23] = _mm256_permute2f128_ps(S11_0, S15_0, 0x31);

  tmp.packet[24] = _mm256_permute2f128_ps(S8_1, S12_1, 0x20);
  tmp.packet[25] = _mm256_permute2f128_ps(S9_1, S13_1, 0x20);
  tmp.packet[26] = _mm256_permute2f128_ps(S10_1, S14_1, 0x20);
  tmp.packet[27] = _mm256_permute2f128_ps(S11_1, S15_1, 0x20);
  tmp.packet[28] = _mm256_permute2f128_ps(S8_1, S12_1, 0x31);
  tmp.packet[29] = _mm256_permute2f128_ps(S9_1, S13_1, 0x31);
  tmp.packet[30] = _mm256_permute2f128_ps(S10_1, S14_1, 0x31);
  tmp.packet[31] = _mm256_permute2f128_ps(S11_1, S15_1, 0x31);

  // Pack them into the output
  PACK_OUTPUT(kernel.packet, tmp.packet, 0, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 1, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 2, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 3, 16);

  PACK_OUTPUT(kernel.packet, tmp.packet, 4, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 5, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 6, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 7, 16);

  PACK_OUTPUT(kernel.packet, tmp.packet, 8, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 9, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 10, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 11, 16);

  PACK_OUTPUT(kernel.packet, tmp.packet, 12, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 13, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 14, 16);
  PACK_OUTPUT(kernel.packet, tmp.packet, 15, 16);
}
#define PACK_OUTPUT_2(OUTPUT, INPUT, INDEX, STRIDE)         \
  EIGEN_INSERT_8f_INTO_16f(OUTPUT[INDEX], INPUT[2 * INDEX], \
                           INPUT[2 * INDEX + STRIDE]);

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet16f, 4>& kernel) {
  __m512 T0 = _mm512_unpacklo_ps(kernel.packet[0], kernel.packet[1]);
  __m512 T1 = _mm512_unpackhi_ps(kernel.packet[0], kernel.packet[1]);
  __m512 T2 = _mm512_unpacklo_ps(kernel.packet[2], kernel.packet[3]);
  __m512 T3 = _mm512_unpackhi_ps(kernel.packet[2], kernel.packet[3]);

  __m512 S0 = _mm512_shuffle_ps(T0, T2, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S1 = _mm512_shuffle_ps(T0, T2, _MM_SHUFFLE(3, 2, 3, 2));
  __m512 S2 = _mm512_shuffle_ps(T1, T3, _MM_SHUFFLE(1, 0, 1, 0));
  __m512 S3 = _mm512_shuffle_ps(T1, T3, _MM_SHUFFLE(3, 2, 3, 2));

  EIGEN_EXTRACT_8f_FROM_16f(S0, S0);
  EIGEN_EXTRACT_8f_FROM_16f(S1, S1);
  EIGEN_EXTRACT_8f_FROM_16f(S2, S2);
  EIGEN_EXTRACT_8f_FROM_16f(S3, S3);

  PacketBlock<Packet8f, 8> tmp;

  tmp.packet[0] = _mm256_permute2f128_ps(S0_0, S1_0, 0x20);
  tmp.packet[1] = _mm256_permute2f128_ps(S2_0, S3_0, 0x20);
  tmp.packet[2] = _mm256_permute2f128_ps(S0_0, S1_0, 0x31);
  tmp.packet[3] = _mm256_permute2f128_ps(S2_0, S3_0, 0x31);

  tmp.packet[4] = _mm256_permute2f128_ps(S0_1, S1_1, 0x20);
  tmp.packet[5] = _mm256_permute2f128_ps(S2_1, S3_1, 0x20);
  tmp.packet[6] = _mm256_permute2f128_ps(S0_1, S1_1, 0x31);
  tmp.packet[7] = _mm256_permute2f128_ps(S2_1, S3_1, 0x31);

  PACK_OUTPUT_2(kernel.packet, tmp.packet, 0, 1);
  PACK_OUTPUT_2(kernel.packet, tmp.packet, 1, 1);
  PACK_OUTPUT_2(kernel.packet, tmp.packet, 2, 1);
  PACK_OUTPUT_2(kernel.packet, tmp.packet, 3, 1);
}

#define PACK_OUTPUT_SQ_D(OUTPUT, INPUT, INDEX, STRIDE)                \
  OUTPUT[INDEX] = _mm512_insertf64x4(OUTPUT[INDEX], INPUT[INDEX], 0); \
  OUTPUT[INDEX] = _mm512_insertf64x4(OUTPUT[INDEX], INPUT[INDEX + STRIDE], 1);

#define PACK_OUTPUT_D(OUTPUT, INPUT, INDEX, STRIDE)                         \
  OUTPUT[INDEX] = _mm512_insertf64x4(OUTPUT[INDEX], INPUT[(2 * INDEX)], 0); \
  OUTPUT[INDEX] =                                                           \
      _mm512_insertf64x4(OUTPUT[INDEX], INPUT[(2 * INDEX) + STRIDE], 1);

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet8d, 4>& kernel) {
  __m512d T0 = _mm512_shuffle_pd(kernel.packet[0], kernel.packet[1], 0);
  __m512d T1 = _mm512_shuffle_pd(kernel.packet[0], kernel.packet[1], 0xff);
  __m512d T2 = _mm512_shuffle_pd(kernel.packet[2], kernel.packet[3], 0);
  __m512d T3 = _mm512_shuffle_pd(kernel.packet[2], kernel.packet[3], 0xff);

  PacketBlock<Packet4d, 8> tmp;

  tmp.packet[0] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T0, 0),
                                         _mm512_extractf64x4_pd(T2, 0), 0x20);
  tmp.packet[1] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T1, 0),
                                         _mm512_extractf64x4_pd(T3, 0), 0x20);
  tmp.packet[2] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T0, 0),
                                         _mm512_extractf64x4_pd(T2, 0), 0x31);
  tmp.packet[3] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T1, 0),
                                         _mm512_extractf64x4_pd(T3, 0), 0x31);

  tmp.packet[4] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T0, 1),
                                         _mm512_extractf64x4_pd(T2, 1), 0x20);
  tmp.packet[5] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T1, 1),
                                         _mm512_extractf64x4_pd(T3, 1), 0x20);
  tmp.packet[6] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T0, 1),
                                         _mm512_extractf64x4_pd(T2, 1), 0x31);
  tmp.packet[7] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T1, 1),
                                         _mm512_extractf64x4_pd(T3, 1), 0x31);

  PACK_OUTPUT_D(kernel.packet, tmp.packet, 0, 1);
  PACK_OUTPUT_D(kernel.packet, tmp.packet, 1, 1);
  PACK_OUTPUT_D(kernel.packet, tmp.packet, 2, 1);
  PACK_OUTPUT_D(kernel.packet, tmp.packet, 3, 1);
}

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet8d, 8>& kernel) {
  __m512d T0 = _mm512_unpacklo_pd(kernel.packet[0], kernel.packet[1]);
  __m512d T1 = _mm512_unpackhi_pd(kernel.packet[0], kernel.packet[1]);
  __m512d T2 = _mm512_unpacklo_pd(kernel.packet[2], kernel.packet[3]);
  __m512d T3 = _mm512_unpackhi_pd(kernel.packet[2], kernel.packet[3]);
  __m512d T4 = _mm512_unpacklo_pd(kernel.packet[4], kernel.packet[5]);
  __m512d T5 = _mm512_unpackhi_pd(kernel.packet[4], kernel.packet[5]);
  __m512d T6 = _mm512_unpacklo_pd(kernel.packet[6], kernel.packet[7]);
  __m512d T7 = _mm512_unpackhi_pd(kernel.packet[6], kernel.packet[7]);

  PacketBlock<Packet4d, 16> tmp;

  tmp.packet[0] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T0, 0),
                                         _mm512_extractf64x4_pd(T2, 0), 0x20);
  tmp.packet[1] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T1, 0),
                                         _mm512_extractf64x4_pd(T3, 0), 0x20);
  tmp.packet[2] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T0, 0),
                                         _mm512_extractf64x4_pd(T2, 0), 0x31);
  tmp.packet[3] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T1, 0),
                                         _mm512_extractf64x4_pd(T3, 0), 0x31);

  tmp.packet[4] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T0, 1),
                                         _mm512_extractf64x4_pd(T2, 1), 0x20);
  tmp.packet[5] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T1, 1),
                                         _mm512_extractf64x4_pd(T3, 1), 0x20);
  tmp.packet[6] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T0, 1),
                                         _mm512_extractf64x4_pd(T2, 1), 0x31);
  tmp.packet[7] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T1, 1),
                                         _mm512_extractf64x4_pd(T3, 1), 0x31);

  tmp.packet[8] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T4, 0),
                                         _mm512_extractf64x4_pd(T6, 0), 0x20);
  tmp.packet[9] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T5, 0),
                                         _mm512_extractf64x4_pd(T7, 0), 0x20);
  tmp.packet[10] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T4, 0),
                                          _mm512_extractf64x4_pd(T6, 0), 0x31);
  tmp.packet[11] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T5, 0),
                                          _mm512_extractf64x4_pd(T7, 0), 0x31);

  tmp.packet[12] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T4, 1),
                                          _mm512_extractf64x4_pd(T6, 1), 0x20);
  tmp.packet[13] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T5, 1),
                                          _mm512_extractf64x4_pd(T7, 1), 0x20);
  tmp.packet[14] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T4, 1),
                                          _mm512_extractf64x4_pd(T6, 1), 0x31);
  tmp.packet[15] = _mm256_permute2f128_pd(_mm512_extractf64x4_pd(T5, 1),
                                          _mm512_extractf64x4_pd(T7, 1), 0x31);

  PACK_OUTPUT_SQ_D(kernel.packet, tmp.packet, 0, 8);
  PACK_OUTPUT_SQ_D(kernel.packet, tmp.packet, 1, 8);
  PACK_OUTPUT_SQ_D(kernel.packet, tmp.packet, 2, 8);
  PACK_OUTPUT_SQ_D(kernel.packet, tmp.packet, 3, 8);

  PACK_OUTPUT_SQ_D(kernel.packet, tmp.packet, 4, 8);
  PACK_OUTPUT_SQ_D(kernel.packet, tmp.packet, 5, 8);
  PACK_OUTPUT_SQ_D(kernel.packet, tmp.packet, 6, 8);
  PACK_OUTPUT_SQ_D(kernel.packet, tmp.packet, 7, 8);
}
template <>
EIGEN_STRONG_INLINE Packet16f pblend(const Selector<16>& /*ifPacket*/,
                                     const Packet16f& /*thenPacket*/,
                                     const Packet16f& /*elsePacket*/) {
  assert(false && "To be implemented");
  return Packet16f();
}
template <>
EIGEN_STRONG_INLINE Packet8d pblend(const Selector<8>& /*ifPacket*/,
                                    const Packet8d& /*thenPacket*/,
                                    const Packet8d& /*elsePacket*/) {
  assert(false && "To be implemented");
  return Packet8d();
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKET_MATH_AVX512_H
