// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_CUDA_H
#define EIGEN_COMPLEX_CUDA_H

// clang-format off

namespace Eigen {

namespace internal {

#if defined(EIGEN_CUDACC) && defined(EIGEN_USE_GPU)

// Many std::complex methods such as operator+, operator-, operator* and
// operator/ are not constexpr. Due to this, clang does not treat them as device
// functions and thus Eigen functors making use of these operators fail to
// compile. Here, we manually specialize these functors for complex types when
// building for CUDA to avoid non-constexpr methods.

// Sum
template<typename T> struct scalar_sum_op<const std::complex<T>, const std::complex<T> > : binary_op_base<const std::complex<T>, const std::complex<T> > {
  typedef typename std::complex<T> result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_sum_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator() (const std::complex<T>& a, const std::complex<T>& b) const {
    return std::complex<T>(numext::real(a) + numext::real(b),
                           numext::imag(a) + numext::imag(b));
  }
};

template<typename T> struct scalar_sum_op<std::complex<T>, std::complex<T> > : scalar_sum_op<const std::complex<T>, const std::complex<T> > {};


// Difference
template<typename T> struct scalar_difference_op<const std::complex<T>, const std::complex<T> >  : binary_op_base<const std::complex<T>, const std::complex<T> > {
  typedef typename std::complex<T> result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_difference_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator() (const std::complex<T>& a, const std::complex<T>& b) const {
    return std::complex<T>(numext::real(a) - numext::real(b),
                           numext::imag(a) - numext::imag(b));
  }
};

template<typename T> struct scalar_difference_op<std::complex<T>, std::complex<T> > : scalar_difference_op<const std::complex<T>, const std::complex<T> > {};


// Product
template<typename T> struct scalar_product_op<const std::complex<T>, const std::complex<T> >  : binary_op_base<const std::complex<T>, const std::complex<T> > {
  enum {
    Vectorizable = packet_traits<std::complex<T> >::HasMul
  };
  typedef typename std::complex<T> result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator() (const std::complex<T>& a, const std::complex<T>& b) const {
    const T a_real = numext::real(a);
    const T a_imag = numext::imag(a);
    const T b_real = numext::real(b);
    const T b_imag = numext::imag(b);
    return std::complex<T>(a_real * b_real - a_imag * b_imag,
                           a_real * b_imag + a_imag * b_real);
  }
};

template<typename T> struct scalar_product_op<std::complex<T>, std::complex<T> > : scalar_product_op<const std::complex<T>, const std::complex<T> > {};


// Quotient
template<typename T> struct scalar_quotient_op<const std::complex<T>, const std::complex<T> > : binary_op_base<const std::complex<T>, const std::complex<T> > {
  enum {
    Vectorizable = packet_traits<std::complex<T> >::HasDiv
  };
  typedef typename std::complex<T> result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_quotient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator() (const std::complex<T>& a, const std::complex<T>& b) const {
    const T a_real = numext::real(a);
    const T a_imag = numext::imag(a);
    const T b_real = numext::real(b);
    const T b_imag = numext::imag(b);
    const T norm = T(1) / (b_real * b_real + b_imag * b_imag);
    return std::complex<T>((a_real * b_real + a_imag * b_imag) * norm,
                           (a_imag * b_real - a_real * b_imag) * norm);
  }
};

template<typename T> struct scalar_quotient_op<std::complex<T>, std::complex<T> > : scalar_quotient_op<const std::complex<T>, const std::complex<T> > {};

#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX_CUDA_H
