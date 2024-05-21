// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Pedro Gonnet (pedro.gonnet@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_
#define THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_

namespace Eigen {

namespace internal {

// Disable the code for older versions of gcc that don't support many of the required avx512 instrinsics.
#if EIGEN_GNUC_AT_LEAST(5, 3)

#define _EIGEN_DECLARE_CONST_Packet16f(NAME, X) \
  const Packet16f p16f_##NAME = pset1<Packet16f>(X)

#define _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(NAME, X) \
  const Packet16f p16f_##NAME = (__m512)pset1<Packet16i>(X)

#define _EIGEN_DECLARE_CONST_Packet8d(NAME, X) \
  const Packet8d p8d_##NAME = pset1<Packet8d>(X)

#define _EIGEN_DECLARE_CONST_Packet8d_FROM_INT64(NAME, X) \
  const Packet8d p8d_##NAME = _mm512_castsi512_pd(_mm512_set1_epi64(X))

// Natural logarithm
// Computes log(x) as log(2^e * m) = C*e + log(m), where the constant C =log(2)
// and m is in the range [sqrt(1/2),sqrt(2)). In this range, the logarithm can
// be easily approximated by a polynomial centered on m=1 for stability.
#if defined(EIGEN_VECTORIZE_AVX512DQ)
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet16f
plog<Packet16f>(const Packet16f& _x) {
  Packet16f x = _x;
  _EIGEN_DECLARE_CONST_Packet16f(1, 1.0f);
  _EIGEN_DECLARE_CONST_Packet16f(half, 0.5f);
  _EIGEN_DECLARE_CONST_Packet16f(126f, 126.0f);

  _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(inv_mant_mask, ~0x7f800000);

  // The smallest non denormalized float number.
  _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(min_norm_pos, 0x00800000);
  _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(minus_inf, 0xff800000);
  _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(nan, 0x7fc00000);

  // Polynomial coefficients.
  _EIGEN_DECLARE_CONST_Packet16f(cephes_SQRTHF, 0.707106781186547524f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p0, 7.0376836292E-2f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p1, -1.1514610310E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p2, 1.1676998740E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p3, -1.2420140846E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p4, +1.4249322787E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p5, -1.6668057665E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p6, +2.0000714765E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p7, -2.4999993993E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_p8, +3.3333331174E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_q1, -2.12194440e-4f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_log_q2, 0.693359375f);

  // invalid_mask is set to true when x is NaN
  __mmask16 invalid_mask =
      _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_NGE_UQ);
  __mmask16 iszero_mask =
      _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_EQ_UQ);

  // Truncate input values to the minimum positive normal.
  x = pmax(x, p16f_min_norm_pos);

  // Extract the shifted exponents.
  Packet16f emm0 = _mm512_cvtepi32_ps(_mm512_srli_epi32((__m512i)x, 23));
  Packet16f e = _mm512_sub_ps(emm0, p16f_126f);

  // Set the exponents to -1, i.e. x are in the range [0.5,1).
  x = _mm512_and_ps(x, p16f_inv_mant_mask);
  x = _mm512_or_ps(x, p16f_half);

  // part2: Shift the inputs from the range [0.5,1) to [sqrt(1/2),sqrt(2))
  // and shift by -1. The values are then centered around 0, which improves
  // the stability of the polynomial evaluation.
  //   if( x < SQRTHF ) {
  //     e -= 1;
  //     x = x + x - 1.0;
  //   } else { x = x - 1.0; }
  __mmask16 mask = _mm512_cmp_ps_mask(x, p16f_cephes_SQRTHF, _CMP_LT_OQ);
  Packet16f tmp = _mm512_mask_blend_ps(mask, x, _mm512_setzero_ps());
  x = psub(x, p16f_1);
  e = psub(e, _mm512_mask_blend_ps(mask, p16f_1, _mm512_setzero_ps()));
  x = padd(x, tmp);

  Packet16f x2 = pmul(x, x);
  Packet16f x3 = pmul(x2, x);

  // Evaluate the polynomial approximant of degree 8 in three parts, probably
  // to improve instruction-level parallelism.
  Packet16f y, y1, y2;
  y = pmadd(p16f_cephes_log_p0, x, p16f_cephes_log_p1);
  y1 = pmadd(p16f_cephes_log_p3, x, p16f_cephes_log_p4);
  y2 = pmadd(p16f_cephes_log_p6, x, p16f_cephes_log_p7);
  y = pmadd(y, x, p16f_cephes_log_p2);
  y1 = pmadd(y1, x, p16f_cephes_log_p5);
  y2 = pmadd(y2, x, p16f_cephes_log_p8);
  y = pmadd(y, x3, y1);
  y = pmadd(y, x3, y2);
  y = pmul(y, x3);

  // Add the logarithm of the exponent back to the result of the interpolation.
  y1 = pmul(e, p16f_cephes_log_q1);
  tmp = pmul(x2, p16f_half);
  y = padd(y, y1);
  x = psub(x, tmp);
  y2 = pmul(e, p16f_cephes_log_q2);
  x = padd(x, y);
  x = padd(x, y2);

  // Filter out invalid inputs, i.e. negative arg will be NAN, 0 will be -INF.
  return _mm512_mask_blend_ps(iszero_mask, p16f_minus_inf,
                              _mm512_mask_blend_ps(invalid_mask, p16f_nan, x));
}
#endif

// Exponential function. Works by writing "x = m*log(2) + r" where
// "m = floor(x/log(2)+1/2)" and "r" is the remainder. The result is then
// "exp(x) = 2^m*exp(r)" where exp(r) is in the range [-1,1).
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet16f
pexp<Packet16f>(const Packet16f& _x) {
  _EIGEN_DECLARE_CONST_Packet16f(1, 1.0f);
  _EIGEN_DECLARE_CONST_Packet16f(half, 0.5f);
  _EIGEN_DECLARE_CONST_Packet16f(127, 127.0f);

  _EIGEN_DECLARE_CONST_Packet16f(exp_hi, 88.3762626647950f);
  _EIGEN_DECLARE_CONST_Packet16f(exp_lo, -88.3762626647949f);

  _EIGEN_DECLARE_CONST_Packet16f(cephes_LOG2EF, 1.44269504088896341f);

  _EIGEN_DECLARE_CONST_Packet16f(cephes_exp_p0, 1.9875691500E-4f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_exp_p1, 1.3981999507E-3f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_exp_p2, 8.3334519073E-3f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_exp_p3, 4.1665795894E-2f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_exp_p4, 1.6666665459E-1f);
  _EIGEN_DECLARE_CONST_Packet16f(cephes_exp_p5, 5.0000001201E-1f);

  // Clamp x.
  Packet16f x = pmax(pmin(_x, p16f_exp_hi), p16f_exp_lo);

  // Express exp(x) as exp(m*ln(2) + r), start by extracting
  // m = floor(x/ln(2) + 0.5).
  Packet16f m = _mm512_floor_ps(pmadd(x, p16f_cephes_LOG2EF, p16f_half));

  // Get r = x - m*ln(2). Note that we can do this without losing more than one
  // ulp precision due to the FMA instruction.
  _EIGEN_DECLARE_CONST_Packet16f(nln2, -0.6931471805599453f);
  Packet16f r = _mm512_fmadd_ps(m, p16f_nln2, x);
  Packet16f r2 = pmul(r, r);

  // TODO(gonnet): Split into odd/even polynomials and try to exploit
  //               instruction-level parallelism.
  Packet16f y = p16f_cephes_exp_p0;
  y = pmadd(y, r, p16f_cephes_exp_p1);
  y = pmadd(y, r, p16f_cephes_exp_p2);
  y = pmadd(y, r, p16f_cephes_exp_p3);
  y = pmadd(y, r, p16f_cephes_exp_p4);
  y = pmadd(y, r, p16f_cephes_exp_p5);
  y = pmadd(y, r2, r);
  y = padd(y, p16f_1);

  // Build emm0 = 2^m.
  Packet16i emm0 = _mm512_cvttps_epi32(padd(m, p16f_127));
  emm0 = _mm512_slli_epi32(emm0, 23);

  // Return 2^m * exp(r).
  return pmax(pmul(y, _mm512_castsi512_ps(emm0)), _x);
}

/*template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet8d
pexp<Packet8d>(const Packet8d& _x) {
  Packet8d x = _x;

  _EIGEN_DECLARE_CONST_Packet8d(1, 1.0);
  _EIGEN_DECLARE_CONST_Packet8d(2, 2.0);

  _EIGEN_DECLARE_CONST_Packet8d(exp_hi, 709.437);
  _EIGEN_DECLARE_CONST_Packet8d(exp_lo, -709.436139303);

  _EIGEN_DECLARE_CONST_Packet8d(cephes_LOG2EF, 1.4426950408889634073599);

  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_p0, 1.26177193074810590878e-4);
  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_p1, 3.02994407707441961300e-2);
  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_p2, 9.99999999999999999910e-1);

  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_q0, 3.00198505138664455042e-6);
  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_q1, 2.52448340349684104192e-3);
  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_q2, 2.27265548208155028766e-1);
  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_q3, 2.00000000000000000009e0);

  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_C1, 0.693145751953125);
  _EIGEN_DECLARE_CONST_Packet8d(cephes_exp_C2, 1.42860682030941723212e-6);

  // clamp x
  x = pmax(pmin(x, p8d_exp_hi), p8d_exp_lo);

  // Express exp(x) as exp(g + n*log(2)).
  const Packet8d n =
      _mm512_mul_round_pd(p8d_cephes_LOG2EF, x, _MM_FROUND_TO_NEAREST_INT);

  // Get the remainder modulo log(2), i.e. the "g" described above. Subtract
  // n*log(2) out in two steps, i.e. n*C1 + n*C2, C1+C2=log2 to get the last
  // digits right.
  const Packet8d nC1 = pmul(n, p8d_cephes_exp_C1);
  const Packet8d nC2 = pmul(n, p8d_cephes_exp_C2);
  x = psub(x, nC1);
  x = psub(x, nC2);

  const Packet8d x2 = pmul(x, x);

  // Evaluate the numerator polynomial of the rational interpolant.
  Packet8d px = p8d_cephes_exp_p0;
  px = pmadd(px, x2, p8d_cephes_exp_p1);
  px = pmadd(px, x2, p8d_cephes_exp_p2);
  px = pmul(px, x);

  // Evaluate the denominator polynomial of the rational interpolant.
  Packet8d qx = p8d_cephes_exp_q0;
  qx = pmadd(qx, x2, p8d_cephes_exp_q1);
  qx = pmadd(qx, x2, p8d_cephes_exp_q2);
  qx = pmadd(qx, x2, p8d_cephes_exp_q3);

  // I don't really get this bit, copied from the SSE2 routines, so...
  // TODO(gonnet): Figure out what is going on here, perhaps find a better
  // rational interpolant?
  x = _mm512_div_pd(px, psub(qx, px));
  x = pmadd(p8d_2, x, p8d_1);

  // Build e=2^n.
  const Packet8d e = _mm512_castsi512_pd(_mm512_slli_epi64(
      _mm512_add_epi64(_mm512_cvtpd_epi64(n), _mm512_set1_epi64(1023)), 52));

  // Construct the result 2^n * exp(g) = e * x. The max is used to catch
  // non-finite values in the input.
  return pmax(pmul(x, e), _x);
  }*/

// Functions for sqrt.
// The EIGEN_FAST_MATH version uses the _mm_rsqrt_ps approximation and one step
// of Newton's method, at a cost of 1-2 bits of precision as opposed to the
// exact solution. The main advantage of this approach is not just speed, but
// also the fact that it can be inlined and pipelined with other computations,
// further reducing its effective latency.
#if EIGEN_FAST_MATH
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet16f
psqrt<Packet16f>(const Packet16f& _x) {
  _EIGEN_DECLARE_CONST_Packet16f(one_point_five, 1.5f);
  _EIGEN_DECLARE_CONST_Packet16f(minus_half, -0.5f);
  _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(flt_min, 0x00800000);

  Packet16f neg_half = pmul(_x, p16f_minus_half);

  // select only the inverse sqrt of positive normal inputs (denormals are
  // flushed to zero and cause infs as well).
  __mmask16 non_zero_mask = _mm512_cmp_ps_mask(_x, p16f_flt_min, _CMP_GE_OQ);
  Packet16f x = _mm512_mask_blend_ps(non_zero_mask, _mm512_rsqrt14_ps(_x),
                                     _mm512_setzero_ps());

  // Do a single step of Newton's iteration.
  x = pmul(x, pmadd(neg_half, pmul(x, x), p16f_one_point_five));

  // Multiply the original _x by it's reciprocal square root to extract the
  // square root.
  return pmul(_x, x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet8d
psqrt<Packet8d>(const Packet8d& _x) {
  _EIGEN_DECLARE_CONST_Packet8d(one_point_five, 1.5);
  _EIGEN_DECLARE_CONST_Packet8d(minus_half, -0.5);
  _EIGEN_DECLARE_CONST_Packet8d_FROM_INT64(dbl_min, 0x0010000000000000LL);

  Packet8d neg_half = pmul(_x, p8d_minus_half);

  // select only the inverse sqrt of positive normal inputs (denormals are
  // flushed to zero and cause infs as well).
  __mmask8 non_zero_mask = _mm512_cmp_pd_mask(_x, p8d_dbl_min, _CMP_GE_OQ);
  Packet8d x = _mm512_mask_blend_pd(non_zero_mask, _mm512_rsqrt14_pd(_x),
                                    _mm512_setzero_pd());

  // Do a first step of Newton's iteration.
  x = pmul(x, pmadd(neg_half, pmul(x, x), p8d_one_point_five));

  // Do a second step of Newton's iteration.
  x = pmul(x, pmadd(neg_half, pmul(x, x), p8d_one_point_five));

  // Multiply the original _x by it's reciprocal square root to extract the
  // square root.
  return pmul(_x, x);
}
#else
template <>
EIGEN_STRONG_INLINE Packet16f psqrt<Packet16f>(const Packet16f& x) {
  return _mm512_sqrt_ps(x);
}
template <>
EIGEN_STRONG_INLINE Packet8d psqrt<Packet8d>(const Packet8d& x) {
  return _mm512_sqrt_pd(x);
}
#endif

// Functions for rsqrt.
// Almost identical to the sqrt routine, just leave out the last multiplication
// and fill in NaN/Inf where needed. Note that this function only exists as an
// iterative version for doubles since there is no instruction for diretly
// computing the reciprocal square root in AVX-512.
#ifdef EIGEN_FAST_MATH
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet16f
prsqrt<Packet16f>(const Packet16f& _x) {
  _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(inf, 0x7f800000);
  _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(nan, 0x7fc00000);
  _EIGEN_DECLARE_CONST_Packet16f(one_point_five, 1.5f);
  _EIGEN_DECLARE_CONST_Packet16f(minus_half, -0.5f);
  _EIGEN_DECLARE_CONST_Packet16f_FROM_INT(flt_min, 0x00800000);

  Packet16f neg_half = pmul(_x, p16f_minus_half);

  // select only the inverse sqrt of positive normal inputs (denormals are
  // flushed to zero and cause infs as well).
  __mmask16 le_zero_mask = _mm512_cmp_ps_mask(_x, p16f_flt_min, _CMP_LT_OQ);
  Packet16f x = _mm512_mask_blend_ps(le_zero_mask, _mm512_setzero_ps(),
                                     _mm512_rsqrt14_ps(_x));

  // Fill in NaNs and Infs for the negative/zero entries.
  __mmask16 neg_mask = _mm512_cmp_ps_mask(_x, _mm512_setzero_ps(), _CMP_LT_OQ);
  Packet16f infs_and_nans = _mm512_mask_blend_ps(
      neg_mask, p16f_nan,
      _mm512_mask_blend_ps(le_zero_mask, p16f_inf, _mm512_setzero_ps()));

  // Do a single step of Newton's iteration.
  x = pmul(x, pmadd(neg_half, pmul(x, x), p16f_one_point_five));

  // Insert NaNs and Infs in all the right places.
  return _mm512_mask_blend_ps(le_zero_mask, infs_and_nans, x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet8d
prsqrt<Packet8d>(const Packet8d& _x) {
  _EIGEN_DECLARE_CONST_Packet8d_FROM_INT64(inf, 0x7ff0000000000000LL);
  _EIGEN_DECLARE_CONST_Packet8d_FROM_INT64(nan, 0x7ff1000000000000LL);
  _EIGEN_DECLARE_CONST_Packet8d(one_point_five, 1.5);
  _EIGEN_DECLARE_CONST_Packet8d(minus_half, -0.5);
  _EIGEN_DECLARE_CONST_Packet8d_FROM_INT64(dbl_min, 0x0010000000000000LL);

  Packet8d neg_half = pmul(_x, p8d_minus_half);

  // select only the inverse sqrt of positive normal inputs (denormals are
  // flushed to zero and cause infs as well).
  __mmask8 le_zero_mask = _mm512_cmp_pd_mask(_x, p8d_dbl_min, _CMP_LT_OQ);
  Packet8d x = _mm512_mask_blend_pd(le_zero_mask, _mm512_setzero_pd(),
                                    _mm512_rsqrt14_pd(_x));

  // Fill in NaNs and Infs for the negative/zero entries.
  __mmask8 neg_mask = _mm512_cmp_pd_mask(_x, _mm512_setzero_pd(), _CMP_LT_OQ);
  Packet8d infs_and_nans = _mm512_mask_blend_pd(
      neg_mask, p8d_nan,
      _mm512_mask_blend_pd(le_zero_mask, p8d_inf, _mm512_setzero_pd()));

  // Do a first step of Newton's iteration.
  x = pmul(x, pmadd(neg_half, pmul(x, x), p8d_one_point_five));

  // Do a second step of Newton's iteration.
  x = pmul(x, pmadd(neg_half, pmul(x, x), p8d_one_point_five));

  // Insert NaNs and Infs in all the right places.
  return _mm512_mask_blend_pd(le_zero_mask, infs_and_nans, x);
}
#else
template <>
EIGEN_STRONG_INLINE Packet16f prsqrt<Packet16f>(const Packet16f& x) {
  return _mm512_rsqrt28_ps(x);
}
#endif
#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_
