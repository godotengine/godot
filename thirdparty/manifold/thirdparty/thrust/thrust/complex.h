/*
 *  Copyright 2008-2019 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file complex.h
 *  \brief Complex numbers
 */

#pragma once

#include <thrust/detail/config.h>

#include <cmath>
#include <complex>
#include <sstream>
#include <thrust/detail/type_traits.h>

#if THRUST_CPP_DIALECT >= 2011
#  define THRUST_STD_COMPLEX_REAL(z) \
    reinterpret_cast< \
      const typename thrust::detail::remove_reference<decltype(z)>::type::value_type (&)[2] \
    >(z)[0]
#  define THRUST_STD_COMPLEX_IMAG(z) \
    reinterpret_cast< \
      const typename thrust::detail::remove_reference<decltype(z)>::type::value_type (&)[2] \
    >(z)[1]
#  define THRUST_STD_COMPLEX_DEVICE __device__
#else
#  define THRUST_STD_COMPLEX_REAL(z) (z).real()
#  define THRUST_STD_COMPLEX_IMAG(z) (z).imag()
#  define THRUST_STD_COMPLEX_DEVICE
#endif

THRUST_NAMESPACE_BEGIN

/*
 *  Calls to the standard math library from inside the thrust namespace
 *  with real arguments require explicit scope otherwise they will fail
 *  to resolve as it will find the equivalent complex function but then
 *  fail to match the template, and give up looking for other scopes.
 */


/*! \addtogroup numerics
 *  \{
 */

/*! \addtogroup complex_numbers Complex Numbers
 *  \{
 */

/*! \cond
 */

namespace detail
{

template <typename T, std::size_t Align>
struct complex_storage;

#if THRUST_CPP_DIALECT >= 2011                                                    \
  && (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC)                       \
  && (THRUST_GCC_VERSION >= 40800)
  // C++11 implementation, excluding GCC 4.7, which doesn't have `alignas`.
  template <typename T, std::size_t Align>
  struct complex_storage
  {
    struct alignas(Align) type { T x; T y; };
  };
#elif  (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC)                    \
    || (   (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC)                 \
        && (THRUST_GCC_VERSION < 40600))
  // C++03 implementation for MSVC and GCC <= 4.5.
  //
  // We have to implement `aligned_type` with specializations for MSVC
  // and GCC 4.2 and older because they require literals as arguments to
  // their alignment attribute.

  #if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC)
    // MSVC implementation.
    #define THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(X)                   \
      template <typename T>                                                   \
      struct complex_storage<T, X>                                            \
      {                                                                       \
        __declspec(align(X)) struct type { T x; T y; };                       \
      };                                                                      \
      /**/
  #else
    // GCC <= 4.2 implementation.
    #define THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(X)                   \
      template <typename T>                                                   \
      struct complex_storage<T, X>                                            \
      {                                                                       \
        struct type { T x; T y; } __attribute__((aligned(X)));                \
      };                                                                      \
      /**/
  #endif

  // The primary template is a fallback, which doesn't specify any alignment.
  // It's only used when T is very large and we're using an older compilers
  // which we have to fully specialize each alignment case.
  template <typename T, std::size_t Align>
  struct complex_storage
  {
    T x; T y;
  };

  THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(1);
  THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(2);
  THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(4);
  THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(8);
  THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(16);
  THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(32);
  THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(64);
  THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(128);

  #undef THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION
#else
  // C++03 implementation for GCC > 4.5, Clang, PGI, ICPC, and xlC.
  template <typename T, std::size_t Align>
  struct complex_storage
  {
    struct type { T x; T y; } __attribute__((aligned(Align)));
  };
#endif

} // end namespace detail

/*! \endcond
 */

/*! \p complex is the Thrust equivalent to <tt>std::complex</tt>. It is
 *  functionally identical to it, but can also be used in device code which
 *  <tt>std::complex</tt> currently cannot.
 *
 *  \tparam T The type used to hold the real and imaginary parts. Should be
 *  <tt>float</tt> or <tt>double</tt>. Others types are not supported.
 *
 */
template <typename T>
struct complex
{
public:

  /*! \p value_type is the type of \p complex's real and imaginary parts.
   */
  typedef T value_type;



  /* --- Constructors --- */

  /*! Construct a complex number with an imaginary part of 0.
   *
   *  \param re The real part of the number.
   */
  __host__ __device__
  complex(const T& re);

  /*! Construct a complex number from its real and imaginary parts.
   *
   *  \param re The real part of the number.
   *  \param im The imaginary part of the number.
   */
  __host__ __device__
  complex(const T& re, const T& im);

#if THRUST_CPP_DIALECT >= 2011
  /*! Default construct a complex number.
   */
  complex() = default;

  /*! This copy constructor copies from a \p complex with a type that is
   *  convertible to this \p complex's \c value_type.
   *
   *  \param z The \p complex to copy from.
   */
  complex(const complex<T>& z) = default;
#else
  /*! Default construct a complex number.
   */
  __host__ __device__
  complex();

  /*! This copy constructor copies from a \p complex with a type that is
   *  convertible to this \p complex's \c value_type.
   *
   *  \param z The \p complex to copy from.
   */
  __host__ __device__
  complex(const complex<T>& z);
#endif

  /*! This converting copy constructor copies from a \p complex with a type
   *  that is convertible to this \p complex's \c value_type.
   *
   *  \param z The \p complex to copy from.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex(const complex<U>& z);

  /*! This converting copy constructor copies from a <tt>std::complex</tt> with
   *  a type that is convertible to this \p complex's \c value_type.
   *
   *  \param z The \p complex to copy from.
   */
  __host__ THRUST_STD_COMPLEX_DEVICE
  complex(const std::complex<T>& z);

  /*! This converting copy constructor copies from a <tt>std::complex</tt> with
   *  a type that is convertible to this \p complex's \c value_type.
   *
   *  \param z The \p complex to copy from.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ THRUST_STD_COMPLEX_DEVICE
  complex(const std::complex<U>& z);



  /* --- Assignment Operators --- */

  /*! Assign `re` to the real part of this \p complex and set the imaginary part
   *  to 0.
   *
   *  \param re The real part of the number.
   */
  __host__ __device__
  complex& operator=(const T& re);

#if THRUST_CPP_DIALECT >= 2011
  /*! Assign `z.real()` and `z.imag()` to the real and imaginary parts of this
   *  \p complex respectively.
   *
   *  \param z The \p complex to copy from.
   */
  complex& operator=(const complex<T>& z) = default;
#else
  /*! Assign `z.real()` and `z.imag()` to the real and imaginary parts of this
   *  \p complex respectively.
   *
   *  \param z The \p complex to copy from.
   */
  __host__ __device__
  complex& operator=(const complex<T>& z);
#endif

  /*! Assign `z.real()` and `z.imag()` to the real and imaginary parts of this
   *  \p complex respectively.
   *
   *  \param z The \p complex to copy from.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex& operator=(const complex<U>& z);

  /*! Assign `z.real()` and `z.imag()` to the real and imaginary parts of this
   *  \p complex respectively.
   *
   *  \param z The \p complex to copy from.
   */
  __host__ THRUST_STD_COMPLEX_DEVICE
  complex& operator=(const std::complex<T>& z);

  /*! Assign `z.real()` and `z.imag()` to the real and imaginary parts of this
   *  \p complex respectively.
   *
   *  \param z The \p complex to copy from.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ THRUST_STD_COMPLEX_DEVICE
  complex& operator=(const std::complex<U>& z);


  /* --- Compound Assignment Operators --- */

  /*! Adds a \p complex to this \p complex and assigns the result to this
   *  \p complex.
   *
   *  \param z The \p complex to be added.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex<T>& operator+=(const complex<U>& z);

  /*! Subtracts a \p complex from this \p complex and assigns the result to
   *  this \p complex.
   *
   *  \param z The \p complex to be subtracted.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex<T>& operator-=(const complex<U>& z);

  /*! Multiplies this \p complex by another \p complex and assigns the result
   *  to this \p complex.
   *
   *  \param z The \p complex to be multiplied.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex<T>& operator*=(const complex<U>& z);

  /*! Divides this \p complex by another \p complex and assigns the result to
   *  this \p complex.
   *
   *  \param z The \p complex to be divided.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex<T>& operator/=(const complex<U>& z);

  /*! Adds a scalar to this \p complex and assigns the result to this
   *  \p complex.
   *
   *  \param z The \p complex to be added.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex<T>& operator+=(const U& z);

  /*! Subtracts a scalar from this \p complex and assigns the result to
   *  this \p complex.
   *
   *  \param z The scalar to be subtracted.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex<T>& operator-=(const U& z);

  /*! Multiplies this \p complex by a scalar and assigns the result
   *  to this \p complex.
   *
   *  \param z The scalar to be multiplied.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex<T>& operator*=(const U& z);

  /*! Divides this \p complex by a scalar and assigns the result to
   *  this \p complex.
   *
   *  \param z The scalar to be divided.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  __host__ __device__
  complex<T>& operator/=(const U& z);



  /* --- Getter functions ---
   * The volatile ones are there to help for example
   * with certain reductions optimizations
   */

  /*! Returns the real part of this \p complex.
   */
  __host__ __device__
  T real() const volatile { return data.x; }

  /*! Returns the imaginary part of this \p complex.
   */
  __host__ __device__
  T imag() const volatile { return data.y; }

  /*! Returns the real part of this \p complex.
   */
  __host__ __device__
  T real() const { return data.x; }

  /*! Returns the imaginary part of this \p complex.
   */
  __host__ __device__
  T imag() const { return data.y; }



  /* --- Setter functions ---
   * The volatile ones are there to help for example
   * with certain reductions optimizations
   */

  /*! Sets the real part of this \p complex.
   *
   *  \param re The new real part of this \p complex.
   */
  __host__ __device__
  void real(T re) volatile { data.x = re; }

  /*! Sets the imaginary part of this \p complex.
   *
   *  \param im The new imaginary part of this \p complex.e
   */
  __host__ __device__
  void imag(T im) volatile { data.y = im; }

  /*! Sets the real part of this \p complex.
   *
   *  \param re The new real part of this \p complex.
   */
  __host__ __device__
  void real(T re) { data.x = re; }

  /*! Sets the imaginary part of this \p complex.
   *
   *  \param im The new imaginary part of this \p complex.
   */
  __host__ __device__
  void imag(T im) { data.y = im; }



  /* --- Casting functions --- */

  /*! Casts this \p complex to a <tt>std::complex</tt> of the same type.
   */
  __host__
  operator std::complex<T>() const { return std::complex<T>(real(), imag()); }

private:
  typename detail::complex_storage<T, sizeof(T) * 2>::type data;
};


/* --- General Functions --- */

/*! Returns the magnitude (also known as absolute value) of a \p complex.
 *
 *  \param z The \p complex from which to calculate the absolute value.
 */
template<typename T>
__host__ __device__
T abs(const complex<T>& z);

/*! Returns the phase angle (also known as argument) in radians of a \p complex.
 *
 *  \param z The \p complex from which to calculate the phase angle.
 */
template <typename T>
__host__ __device__
T arg(const complex<T>& z);

/*! Returns the square of the magnitude of a \p complex.
 *
 *  \param z The \p complex from which to calculate the norm.
 */
template <typename T>
__host__ __device__
T norm(const complex<T>& z);

/*! Returns the complex conjugate of a \p complex.
 *
 *  \param z The \p complex from which to calculate the complex conjugate.
 */
template <typename T>
__host__ __device__
complex<T> conj(const complex<T>& z);

/*! Returns a \p complex with the specified magnitude and phase.
 *
 *  \param m The magnitude of the returned \p complex.
 *  \param theta The phase of the returned \p complex in radians.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
polar(const T0& m, const T1& theta = T1());

/*! Returns the projection of a \p complex on the Riemann sphere.
 *  For all finite \p complex it returns the argument. For \p complexs
 *  with a non finite part returns (INFINITY,+/-0) where the sign of
 *  the zero matches the sign of the imaginary part of the argument.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> proj(const T& z);



/* --- Binary Arithmetic operators --- */

/*! Adds two \p complex numbers.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const complex<T0>& x, const complex<T1>& y);

/*! Adds a scalar to a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const complex<T0>& x, const T1& y);

/*! Adds a \p complex number to a scalar.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The scalar.
 *  \param y The \p complex.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const T0& x, const complex<T1>& y);

/*! Subtracts two \p complex numbers.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The first \p complex (minuend).
 *  \param y The second \p complex (subtrahend).
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const complex<T0>& x, const complex<T1>& y);

/*! Subtracts a scalar from a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The \p complex (minuend).
 *  \param y The scalar (subtrahend).
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const complex<T0>& x, const T1& y);

/*! Subtracts a \p complex number from a scalar.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The scalar (minuend).
 *  \param y The \p complex (subtrahend).
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const T0& x, const complex<T1>& y);

/*! Multiplies two \p complex numbers.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const complex<T0>& x, const complex<T1>& y);

/*! Multiplies a \p complex number by a scalar.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const complex<T0>& x, const T1& y);

/*! Multiplies a scalar by a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The scalar.
 *  \param y The \p complex.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const T0& x, const complex<T1>& y);

/*! Divides two \p complex numbers.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The numerator (dividend).
 *  \param y The denomimator (divisor).
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const complex<T0>& x, const complex<T1>& y);

/*! Divides a \p complex number by a scalar.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The complex numerator (dividend).
 *  \param y The scalar denomimator (divisor).
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const complex<T0>& x, const T1& y);

/*! Divides a scalar by a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The scalar numerator (dividend).
 *  \param y The complex denomimator (divisor).
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const T0& x, const complex<T1>& y);



/* --- Unary Arithmetic operators --- */

/*! Unary plus, returns its \p complex argument.
 *
 *  \param y The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T>
operator+(const complex<T>& y);

/*! Unary minus, returns the additive inverse (negation) of its \p complex
 * argument.
 *
 *  \param y The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T>
operator-(const complex<T>& y);



/* --- Exponential Functions --- */

/*! Returns the complex exponential of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> exp(const complex<T>& z);

/*! Returns the complex natural logarithm of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> log(const complex<T>& z);

/*! Returns the complex base 10 logarithm of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> log10(const complex<T>& z);



/* --- Power Functions --- */

/*! Returns a \p complex number raised to another.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The base.
 *  \param y The exponent.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const complex<T1>& y);

/*! Returns a \p complex number raised to a scalar.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The base.
 *  \param y The exponent.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const T1& y);

/*! Returns a scalar raised to a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The base.
 *  \param y The exponent.
 */
template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const T0& x, const complex<T1>& y);

/*! Returns the complex square root of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> sqrt(const complex<T>& z);


/* --- Trigonometric Functions --- */

/*! Returns the complex cosine of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> cos(const complex<T>& z);

/*! Returns the complex sine of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> sin(const complex<T>& z);

/*! Returns the complex tangent of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> tan(const complex<T>& z);



/* --- Hyperbolic Functions --- */

/*! Returns the complex hyperbolic cosine of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> cosh(const complex<T>& z);

/*! Returns the complex hyperbolic sine of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> sinh(const complex<T>& z);

/*! Returns the complex hyperbolic tangent of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> tanh(const complex<T>& z);



/* --- Inverse Trigonometric Functions --- */

/*! Returns the complex arc cosine of a \p complex number.
 *
 *  The range of the real part of the result is [0, Pi] and
 *  the range of the imaginary part is [-inf, +inf]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> acos(const complex<T>& z);

/*! Returns the complex arc sine of a \p complex number.
 *
 *  The range of the real part of the result is [-Pi/2, Pi/2] and
 *  the range of the imaginary part is [-inf, +inf]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> asin(const complex<T>& z);

/*! Returns the complex arc tangent of a \p complex number.
 *
 *  The range of the real part of the result is [-Pi/2, Pi/2] and
 *  the range of the imaginary part is [-inf, +inf]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> atan(const complex<T>& z);



/* --- Inverse Hyperbolic Functions --- */

/*! Returns the complex inverse hyperbolic cosine of a \p complex number.
 *
 *  The range of the real part of the result is [0, +inf] and
 *  the range of the imaginary part is [-Pi, Pi]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> acosh(const complex<T>& z);

/*! Returns the complex inverse hyperbolic sine of a \p complex number.
 *
 *  The range of the real part of the result is [-inf, +inf] and
 *  the range of the imaginary part is [-Pi/2, Pi/2]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> asinh(const complex<T>& z);

/*! Returns the complex inverse hyperbolic tangent of a \p complex number.
 *
 *  The range of the real part of the result is [-inf, +inf] and
 *  the range of the imaginary part is [-Pi/2, Pi/2]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__
complex<T> atanh(const complex<T>& z);



/* --- Stream Operators --- */

/*! Writes to an output stream a \p complex number in the form (real, imaginary).
 *
 *  \param os The output stream.
 *  \param z The \p complex number to output.
 */
template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const complex<T>& z);

/*! Reads a \p complex number from an input stream.
 *
 *  The recognized formats are:
 * - real
 * - (real)
 * - (real, imaginary)
 *
 * The values read must be convertible to the \p complex's \c value_type
 *
 *  \param is The input stream.
 *  \param z The \p complex number to set.
 */
template <typename T, typename CharT, typename Traits>
__host__
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& is, complex<T>& z);



/* --- Equality Operators --- */

/*! Returns true if two \p complex numbers are equal and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 */
template <typename T0, typename T1>
__host__ __device__
bool operator==(const complex<T0>& x, const complex<T1>& y);

/*! Returns true if two \p complex numbers are equal and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 */
template <typename T0, typename T1>
__host__ THRUST_STD_COMPLEX_DEVICE
bool operator==(const complex<T0>& x, const std::complex<T1>& y);

/*! Returns true if two \p complex numbers are equal and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 */
template <typename T0, typename T1>
__host__ THRUST_STD_COMPLEX_DEVICE
bool operator==(const std::complex<T0>& x, const complex<T1>& y);

/*! Returns true if the imaginary part of the \p complex number is zero and
 *  the real part is equal to the scalar. Returns false otherwise.
 *
 *  \param x The scalar.
 *  \param y The \p complex.
 */
template <typename T0, typename T1>
__host__ __device__
bool operator==(const T0& x, const complex<T1>& y);

/*! Returns true if the imaginary part of the \p complex number is zero and
 *  the real part is equal to the scalar. Returns false otherwise.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 */
template <typename T0, typename T1>
__host__ __device__
bool operator==(const complex<T0>& x, const T1& y);

/*! Returns true if two \p complex numbers are different and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 */
template <typename T0, typename T1>
__host__ __device__
bool operator!=(const complex<T0>& x, const complex<T1>& y);

/*! Returns true if two \p complex numbers are different and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 */
template <typename T0, typename T1>
__host__ THRUST_STD_COMPLEX_DEVICE
bool operator!=(const complex<T0>& x, const std::complex<T1>& y);

/*! Returns true if two \p complex numbers are different and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 */
template <typename T0, typename T1>
__host__ THRUST_STD_COMPLEX_DEVICE
bool operator!=(const std::complex<T0>& x, const complex<T1>& y);

/*! Returns true if the imaginary part of the \p complex number is not zero or
 *  the real part is different from the scalar. Returns false otherwise.
 *
 *  \param x The scalar.
 *  \param y The \p complex.
 */
template <typename T0, typename T1>
__host__ __device__
bool operator!=(const T0& x, const complex<T1>& y);

/*! Returns true if the imaginary part of the \p complex number is not zero or
 *  the real part is different from the scalar. Returns false otherwise.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 */
template <typename T0, typename T1>
__host__ __device__
bool operator!=(const complex<T0>& x, const T1& y);

THRUST_NAMESPACE_END

#include <thrust/detail/complex/complex.inl>

#undef THRUST_STD_COMPLEX_REAL
#undef THRUST_STD_COMPLEX_IMAG
#undef THRUST_STD_COMPLEX_DEVICE

/*! \} // complex_numbers
 */

/*! \} // numerics
 */

