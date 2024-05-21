// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATHFUNCTIONSIMPL_H
#define EIGEN_MATHFUNCTIONSIMPL_H

namespace Eigen {

namespace internal {

/** \internal \returns the hyperbolic tan of \a a (coeff-wise)
    Doesn't do anything fancy, just a 13/6-degree rational interpolant which
    is accurate up to a couple of ulp in the range [-9, 9], outside of which
    the tanh(x) = +/-1.

    This implementation works on both scalars and packets.
*/
template<typename T>
T generic_fast_tanh_float(const T& a_x)
{
  // Clamp the inputs to the range [-9, 9] since anything outside
  // this range is +/-1.0f in single-precision.
  const T plus_9 = pset1<T>(9.f);
  const T minus_9 = pset1<T>(-9.f);
  const T x = pmax(pmin(a_x, plus_9), minus_9);
  // The monomial coefficients of the numerator polynomial (odd).
  const T alpha_1 = pset1<T>(4.89352455891786e-03f);
  const T alpha_3 = pset1<T>(6.37261928875436e-04f);
  const T alpha_5 = pset1<T>(1.48572235717979e-05f);
  const T alpha_7 = pset1<T>(5.12229709037114e-08f);
  const T alpha_9 = pset1<T>(-8.60467152213735e-11f);
  const T alpha_11 = pset1<T>(2.00018790482477e-13f);
  const T alpha_13 = pset1<T>(-2.76076847742355e-16f);

  // The monomial coefficients of the denominator polynomial (even).
  const T beta_0 = pset1<T>(4.89352518554385e-03f);
  const T beta_2 = pset1<T>(2.26843463243900e-03f);
  const T beta_4 = pset1<T>(1.18534705686654e-04f);
  const T beta_6 = pset1<T>(1.19825839466702e-06f);

  // Since the polynomials are odd/even, we need x^2.
  const T x2 = pmul(x, x);

  // Evaluate the numerator polynomial p.
  T p = pmadd(x2, alpha_13, alpha_11);
  p = pmadd(x2, p, alpha_9);
  p = pmadd(x2, p, alpha_7);
  p = pmadd(x2, p, alpha_5);
  p = pmadd(x2, p, alpha_3);
  p = pmadd(x2, p, alpha_1);
  p = pmul(x, p);

  // Evaluate the denominator polynomial p.
  T q = pmadd(x2, beta_6, beta_4);
  q = pmadd(x2, q, beta_2);
  q = pmadd(x2, q, beta_0);

  // Divide the numerator by the denominator.
  return pdiv(p, q);
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_MATHFUNCTIONSIMPL_H
