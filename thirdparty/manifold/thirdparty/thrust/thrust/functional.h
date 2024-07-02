/*
 *  Copyright 2008-2018 NVIDIA Corporation
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


/*! \file functional.h
 *  \brief Function objects and tools for manipulating them
 */

#pragma once

#include <thrust/detail/config.h>
#include <functional>
#include <thrust/detail/functional/placeholder.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup function_objects Function Objects
 */

template<typename Operation> struct unary_traits;

template<typename Operation> struct binary_traits;

/*! \addtogroup function_object_adaptors Function Object Adaptors
 *  \ingroup function_objects
 *  \{
 */

/*! \p unary_function is an empty base class: it contains no member functions
 *  or member variables, but only type information. The only reason it exists
 *  is to make it more convenient to define types that are models of the
 *  concept Adaptable Unary Function. Specifically, any model of Adaptable
 *  Unary Function must define nested \c typedefs. Those \c typedefs are
 *  provided by the base class \p unary_function.
 *
 *  The following code snippet demonstrates how to construct an
 *  Adaptable Unary Function using \p unary_function.
 *
 *  \code
 *  struct sine : public thrust::unary_function<float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x) { return sinf(x); }
 *  };
 *  \endcode
 *
 *  \note Because C++11 language support makes the functionality of
 *        \c unary_function obsolete, its use is optional if C++11 language
 *        features are enabled.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/unary_function
 *  \see binary_function
 */
template<typename Argument,
         typename Result>
struct unary_function
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  typedef Argument argument_type;

  /*! \typedef result_type;
   *  \brief The type of the function object's result.
   */
  typedef Result   result_type;
}; // end unary_function

/*! \p binary_function is an empty base class: it contains no member functions
 *  or member variables, but only type information. The only reason it exists
 *  is to make it more convenient to define types that are models of the
 *  concept Adaptable Binary Function. Specifically, any model of Adaptable
 *  Binary Function must define nested \c typedefs. Those \c typedefs are
 *  provided by the base class \p binary_function.
 *
 *  The following code snippet demonstrates how to construct an
 *  Adaptable Binary Function using \p binary_function.
 *
 *  \code
 *  struct exponentiate : public thrust::binary_function<float,float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x, float y) { return powf(x,y); }
 *  };
 *  \endcode
 *
 *  \note Because C++11 language support makes the functionality of
 *        \c binary_function obsolete, its use is optional if C++11 language
 *        features are enabled.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/binary_function
 *  \see unary_function
 */
template<typename Argument1,
         typename Argument2,
         typename Result>
struct binary_function
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef Argument1 first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef Argument2 second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef Result    result_type;
}; // end binary_function

/*! \}
 */


/*! \addtogroup predefined_function_objects Predefined Function Objects
 *  \ingroup function_objects
 */

/*! \addtogroup arithmetic_operations Arithmetic Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

#define THRUST_UNARY_FUNCTOR_VOID_SPECIALIZATION(func, impl)                   \
  template <>                                                                  \
  struct func<void>                                                            \
  {                                                                            \
    using is_transparent = void;                                               \
    __thrust_exec_check_disable__                                              \
    template <typename T>                                                      \
    __host__ __device__                                                        \
    constexpr auto operator()(T&& x) const                                     \
      noexcept(noexcept(impl)) THRUST_TRAILING_RETURN(decltype(impl))          \
    {                                                                          \
      return impl;                                                             \
    }                                                                          \
  }

#define THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION(func, impl)                  \
  template <>                                                                  \
  struct func<void>                                                            \
  {                                                                            \
    using is_transparent = void;                                               \
    __thrust_exec_check_disable__                                              \
    template <typename T1, typename T2>                                        \
    __host__ __device__                                                        \
    constexpr auto operator()(T1&& t1, T2&& t2) const                          \
      noexcept(noexcept(impl)) THRUST_TRAILING_RETURN(decltype(impl))          \
    {                                                                          \
      return impl;                                                             \
    }                                                                          \
  }

#define THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(func, op)                 \
  THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION(                                   \
    func, THRUST_FWD(t1) op THRUST_FWD(t2))


/*! \p plus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>plus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x+y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x+y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>plus</tt> to sum two
 *  device_vectors of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::plus<float>());
 *  // V3 is now {76, 77, 78, ..., 1075}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/plus
 *  \see binary_function
 */
template<typename T = void>
struct plus
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs + rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs + rhs;
  }
}; // end plus

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(plus, +);

/*! \p minus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>minus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x-y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x-y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>minus</tt> to subtract
 *  a device_vector of \c floats from another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::minus<float>());
 *  // V3 is now {-74, -73, -72, ..., 925}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/minus
 *  \see binary_function
 */
template<typename T = void>
struct minus
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs - rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs - rhs;
  }
}; // end minus

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(minus, -);

/*! \p multiplies is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>multiplies<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x*y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x*y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>multiplies</tt> to multiply
 *  two device_vectors of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::multiplies<float>());
 *  // V3 is now {75, 150, 225, ..., 75000}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/multiplies
 *  \see binary_function
 */
template<typename T = void>
struct multiplies
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs * rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs * rhs;
  }
}; // end multiplies

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(multiplies, *);

/*! \p divides is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>divides<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x/y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x/y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>divides</tt> to divide
 *  one device_vectors of \c floats by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::divides<float>());
 *  // V3 is now {1/75, 2/75, 3/75, ..., 1000/75}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/divides
 *  \see binary_function
 */
template<typename T = void>
struct divides
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs / rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs / rhs;
  }
}; // end divides

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(divides, /);

/*! \p modulus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>modulus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x \% y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x \% y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>modulus</tt> to take
 *  the modulus of one device_vectors of \c floats by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::modulus<int>());
 *  // V3 is now {1%75, 2%75, 3%75, ..., 1000%75}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/modulus
 *  \see binary_function
 */
template<typename T = void>
struct modulus
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs % rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs % rhs;
  }
}; // end modulus

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(modulus, %);

/*! \p negate is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f is an object of class <tt>negate<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>-x</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>-x</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>negate</tt> to negate
 *  the elements of a device_vector of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(),
 *                    thrust::negate<float>());
 *  // V2 is now {-1, -2, -3, ..., -1000}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/negate
 *  \see unary_function
 */
template<typename T = void>
struct negate
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  typedef T argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>-x</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &x) const
  {
    return -x;
  }
}; // end negate

THRUST_UNARY_FUNCTOR_VOID_SPECIALIZATION(negate, -THRUST_FWD(x));

/*! \p square is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f is an object of class <tt>square<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x*x</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x*x</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>square</tt> to square
 *  the elements of a device_vector of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(),
 *                    thrust::square<float>());
 *  // V2 is now {1, 4, 9, ..., 1000000}
 *  \endcode
 *
 *  \see unary_function
 */
template<typename T = void>
struct square
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  typedef T argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>x*x</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &x) const
  {
    return x*x;
  }
}; // end square

THRUST_UNARY_FUNCTOR_VOID_SPECIALIZATION(square, x*x);

/*! \}
 */

/*! \addtogroup comparison_operations Comparison Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p equal_to is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>equal_to<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x == y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/equal_to
 *  \see binary_function
 */
template<typename T = void>
struct equal_to
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs == rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs == rhs;
  }
}; // end equal_to

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(equal_to, ==);

/*! \p not_equal_to is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>not_equal_to<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x != y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/not_equal_to
 *  \see binary_function
 */
template<typename T = void>
struct not_equal_to
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs != rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs != rhs;
  }
}; // end not_equal_to

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(not_equal_to, !=);

/*! \p greater is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>greater<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x > y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/greater
 *  \see binary_function
 */
template<typename T = void>
struct greater
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs > rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs > rhs;
  }
}; // end greater

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(greater, >);

/*! \p less is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>less<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x < y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/less
 *  \see binary_function
 */
template<typename T = void>
struct less
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs < rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs < rhs;
  }
}; // end less

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(less, <);

/*! \p greater_equal is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>greater_equal<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x >= y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/greater_equal
 *  \see binary_function
 */
template<typename T = void>
struct greater_equal
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs >= rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs >= rhs;
  }
}; // end greater_equal

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(greater_equal, >=);

/*! \p less_equal is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>less_equal<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x <= y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/less_equal
 *  \see binary_function
 */
template<typename T = void>
struct less_equal
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs <= rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs <= rhs;
  }
}; // end less_equal

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(less_equal, <=);

/*! \}
 */


/*! \addtogroup logical_operations Logical Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p logical_and is a function object. Specifically, it is an Adaptable Binary Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_and<T></tt> and \c x and \c y are objects of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x,y)</tt> returns \c true
 *  if and only if both \c x and \c y are \c true.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_and
 *  \see binary_function
 */
template<typename T = void>
struct logical_and
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs && rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs && rhs;
  }
}; // end logical_and

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(logical_and, &&);

/*! \p logical_or is a function object. Specifically, it is an Adaptable Binary Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_or<T></tt> and \c x and \c y are objects of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x,y)</tt> returns \c true
 *  if and only if either \c x or \c y are \c true.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_or
 *  \see binary_function
 */
template<typename T = void>
struct logical_or
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs || rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return lhs || rhs;
  }
}; // end logical_or

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(logical_or, ||);

/*! \p logical_not is a function object. Specifically, it is an Adaptable Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_not<T></tt> and \c x is an object of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x)</tt> returns \c true
 *  if and only if \c x is \c false.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  The following code snippet demonstrates how to use \p logical_not to transform
 *  a device_vector of \c bools into its logical complement.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<bool> V;
 *  ...
 *  thrust::transform(V.begin(), V.end(), V.begin(), thrust::logical_not<bool>());
 *  // The elements of V are now the logical complement of what they were prior
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_not
 *  \see unary_function
 */
template<typename T = void>
struct logical_not
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>!x</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &x) const
  {
    return !x;
  }
}; // end logical_not

THRUST_UNARY_FUNCTOR_VOID_SPECIALIZATION(logical_not, !THRUST_FWD(x));

/*! \}
 */

/*! \addtogroup bitwise_operations Bitwise Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p bit_and is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x&y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x&y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_and</tt> to take
 *  the bitwise AND of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_and<int>());
 *  // V3 is now {1&13, 2&13, 3&13, ..., 1000%13}
 *  \endcode
 *
 *  \see binary_function
 */
template<typename T = void>
struct bit_and
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs & rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs & rhs;
  }
}; // end bit_and

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(bit_and, &);

/*! \p bit_or is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x|y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x|y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_or</tt> to take
 *  the bitwise OR of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_or<int>());
 *  // V3 is now {1|13, 2|13, 3|13, ..., 1000|13}
 *  \endcode
 *
 *  \see binary_function
 */
template<typename T = void>
struct bit_or
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs | rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs | rhs;
  }
}; // end bit_or

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(bit_or, |);

/*! \p bit_xor is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x^y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x^y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_xor</tt> to take
 *  the bitwise XOR of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_xor<int>());
 *  // V3 is now {1^13, 2^13, 3^13, ..., 1000^13}
 *  \endcode
 *
 *  \see binary_function
 */
template<typename T = void>
struct bit_xor
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs ^ rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs ^ rhs;
  }
}; // end bit_xor

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(bit_xor, ^);

/*! \}
 */

/*! \addtogroup generalized_identity_operations Generalized Identity Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p identity is a Unary Function that represents the identity function: it takes
 *  a single argument \c x, and returns \c x.
 *
 *  \tparam T No requirements on \p T.
 *
 *  The following code snippet demonstrates that \p identity returns its
 *  argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x = 137;
 *  thrust::identity<int> id;
 *  assert(x == id(x));
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/identity
 *  \see unary_function
 */
template<typename T = void>
struct identity
{
  /*! \typedef argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>x</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr const T &operator()(const T &x) const
  {
    return x;
  }
}; // end identity

THRUST_UNARY_FUNCTOR_VOID_SPECIALIZATION(identity, THRUST_FWD(x));

/*! \p maximum is a function object that takes two arguments and returns the greater
 *  of the two. Specifically, it is an Adaptable Binary Function. If \c f is an
 *  object of class <tt>maximum<T></tt> and \c x and \c y are objects of class \c T
 *  <tt>f(x,y)</tt> returns \c x if <tt>x > y</tt> and \c y, otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  The following code snippet demonstrates that \p maximum returns its
 *  greater argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::maximum<int> mx;
 *  assert(x == mx(x,y));
 *  \endcode
 *
 *  \see minimum
 *  \see min
 *  \see binary_function
 */
template<typename T = void>
struct maximum
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>rhs < lhs ? lhs : rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs < rhs ? rhs : lhs;
  }
}; // end maximum

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION(maximum,
                                          t1 < t2 ? THRUST_FWD(t2)
                                                  : THRUST_FWD(t1));

/*! \p minimum is a function object that takes two arguments and returns the lesser
 *  of the two. Specifically, it is an Adaptable Binary Function. If \c f is an
 *  object of class <tt>minimum<T></tt> and \c x and \c y are objects of class \c T
 *  <tt>f(x,y)</tt> returns \c x if <tt>x < y</tt> and \c y, otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  The following code snippet demonstrates that \p minimum returns its
 *  lesser argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::minimum<int> mn;
 *  assert(y == mn(x,y));
 *  \endcode
 *
 *  \see maximum
 *  \see max
 *  \see binary_function
 */
template<typename T = void>
struct minimum
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs < rhs ? lhs : rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return lhs < rhs ? lhs : rhs;
  }
}; // end minimum

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION(minimum,
                                          t1 < t2 ? THRUST_FWD(t1)
                                                  : THRUST_FWD(t2));

/*! \p project1st is a function object that takes two arguments and returns
 *  its first argument; the second argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project1st<int> pj1;
 *  assert(x == pj1(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project2nd
 *  \see binary_function
 */
template<typename T1 = void, typename T2 = void>
struct project1st
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T1 first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T2 second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T1 result_type;

  /*! Function call operator. The return value is <tt>lhs</tt>.
   */
  __host__ __device__
  constexpr const T1 &operator()(const T1 &lhs, const T2 & /*rhs*/) const
  {
    return lhs;
  }
}; // end project1st

template <>
struct project1st<void, void>
{
  using is_transparent = void;
  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&&) const
    noexcept(noexcept(THRUST_FWD(t1)))
    THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1)))
  {
    return THRUST_FWD(t1);
  }
};

/*! \p project2nd is a function object that takes two arguments and returns
 *  its second argument; the first argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project2nd<int> pj2;
 *  assert(y == pj2(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project1st
 *  \see binary_function
 */
template<typename T1 = void, typename T2 = void>
struct project2nd
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T1 first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T2 second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T2 result_type;

  /*! Function call operator. The return value is <tt>rhs</tt>.
   */
  __host__ __device__
  constexpr const T2 &operator()(const T1 &/*lhs*/, const T2 &rhs) const
  {
    return rhs;
  }
}; // end project2nd

template <>
struct project2nd<void, void>
{
  using is_transparent = void;
  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&&, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t2)))
  {
    return THRUST_FWD(t2);
  }
};

/*! \}
 */

// odds and ends

/*! \addtogroup function_object_adaptors
 *  \{
 */

/*! \p unary_negate is a function object adaptor: it is an Adaptable Predicate
 *  that represents the logical negation of some other Adaptable Predicate.
 *  That is: if \c f is an object of class <tt>unary_negate<AdaptablePredicate></tt>,
 *  then there exists an object \c pred of class \c AdaptablePredicate such
 *  that <tt>f(x)</tt> always returns the same value as <tt>!pred(x)</tt>.
 *  There is rarely any reason to construct a <tt>unary_negate</tt> directly;
 *  it is almost always easier to use the helper function not1.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/unary_negate
 *  \see not1
 */
template<typename Predicate>
struct unary_negate
    : public thrust::unary_function<typename Predicate::argument_type, bool>
{
  /*! Constructor takes a \p Predicate object to negate.
   *  \param p The \p Predicate object to negate.
   */
  __host__ __device__
  explicit unary_negate(Predicate p) : pred(p){}

  /*! Function call operator. The return value is <tt>!pred(x)</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  bool operator()(const typename Predicate::argument_type& x) { return !pred(x); }

  /*! \cond
   */
  Predicate pred;
  /*! \endcond
   */
}; // end unary_negate

/*! \p not1 is a helper function to simplify the creation of Adaptable Predicates:
 *  it takes an Adaptable Predicate \p pred as an argument and returns a new Adaptable
 *  Predicate that represents the negation of \p pred. That is: if \c pred is an object
 *  of a type which models Adaptable Predicate, then the the type of the result
 *  \c npred of <tt>not1(pred)</tt> is also a model of Adaptable Predicate and
 *  <tt>npred(x)</tt> always returns the same value as <tt>!pred(x)</tt>.
 *
 *  \param pred The Adaptable Predicate to negate.
 *  \return A new object, <tt>npred</tt> such that <tt>npred(x)</tt> always returns
 *          the same value as <tt>!pred(x)</tt>.
 *
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_negate">Adaptable Predicate</a>.
 *
 *  \see unary_negate
 *  \see not2
 */
template<typename Predicate>
  __host__ __device__
  unary_negate<Predicate> not1(const Predicate &pred);

/*! \p binary_negate is a function object adaptor: it is an Adaptable Binary
 *  Predicate that represents the logical negation of some other Adaptable
 *  Binary Predicate. That is: if \c f is an object of class <tt>binary_negate<AdaptablePredicate></tt>,
 *  then there exists an object \c pred of class \c AdaptableBinaryPredicate
 *  such that <tt>f(x,y)</tt> always returns the same value as <tt>!pred(x,y)</tt>.
 *  There is rarely any reason to construct a <tt>binary_negate</tt> directly;
 *  it is almost always easier to use the helper function not2.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/binary_negate
 */
template<typename Predicate>
struct binary_negate
    : public thrust::binary_function<typename Predicate::first_argument_type,
                                     typename Predicate::second_argument_type,
                                     bool>
{
  /*! Constructor takes a \p Predicate object to negate.
   *  \param p The \p Predicate object to negate.
   */
  __host__ __device__
  explicit binary_negate(Predicate p) : pred(p){}

  /*! Function call operator. The return value is <tt>!pred(x,y)</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  bool operator()(const typename Predicate::first_argument_type& x, const typename Predicate::second_argument_type& y)
  {
      return !pred(x,y);
  }

  /*! \cond
   */
  Predicate pred;
  /*! \endcond
   */
}; // end binary_negate

/*! \p not2 is a helper function to simplify the creation of Adaptable Binary Predicates:
 *  it takes an Adaptable Binary Predicate \p pred as an argument and returns a new Adaptable
 *  Binary Predicate that represents the negation of \p pred. That is: if \c pred is an object
 *  of a type which models Adaptable Binary Predicate, then the the type of the result
 *  \c npred of <tt>not2(pred)</tt> is also a model of Adaptable Binary Predicate and
 *  <tt>npred(x,y)</tt> always returns the same value as <tt>!pred(x,y)</tt>.
 *
 *  \param pred The Adaptable Binary Predicate to negate.
 *  \return A new object, <tt>npred</tt> such that <tt>npred(x,y)</tt> always returns
 *          the same value as <tt>!pred(x,y)</tt>.
 *
 *  \tparam Binary Predicate is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/AdaptableBinaryPredicate">Adaptable Binary Predicate</a>.
 *
 *  \see binary_negate
 *  \see not1
 */
template<typename BinaryPredicate>
  __host__ __device__
  binary_negate<BinaryPredicate> not2(const BinaryPredicate &pred);

/*! \}
 */


/*! \addtogroup placeholder_objects Placeholder Objects
 *  \ingroup function_objects
 *  \{
 */


/*! \namespace thrust::placeholders
 *  \brief Facilities for constructing simple functions inline.
 *
 *  Objects in the \p thrust::placeholders namespace may be used to create simple arithmetic functions inline
 *  in an algorithm invocation. Combining placeholders such as \p _1 and \p _2 with arithmetic operations such as \c +
 *  creates an unnamed function object which applies the operation to their arguments.
 *
 *  The type of placeholder objects is implementation-defined.
 *
 *  The following code snippet demonstrates how to use the placeholders \p _1 and \p _2 with \p thrust::transform
 *  to implement the SAXPY computation:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *
 *  int main()
 *  {
 *    thrust::device_vector<float> x(4), y(4);
 *    x[0] = 1;
 *    x[1] = 2;
 *    x[2] = 3;
 *    x[3] = 4;
 *
 *    y[0] = 1;
 *    y[1] = 1;
 *    y[2] = 1;
 *    y[3] = 1;
 *
 *    float a = 2.0f;
 *
 *    using namespace thrust::placeholders;
 *
 *    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),
 *      a * _1 + _2
 *    );
 *
 *    // y is now {3, 5, 7, 9}
 *  }
 *  \endcode
 */
namespace placeholders
{


/*! \p thrust::placeholders::_1 is the placeholder for the first function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<0>::type _1;


/*! \p thrust::placeholders::_2 is the placeholder for the second function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<1>::type _2;


/*! \p thrust::placeholders::_3 is the placeholder for the third function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<2>::type _3;


/*! \p thrust::placeholders::_4 is the placeholder for the fourth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<3>::type _4;


/*! \p thrust::placeholders::_5 is the placeholder for the fifth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<4>::type _5;


/*! \p thrust::placeholders::_6 is the placeholder for the sixth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<5>::type _6;


/*! \p thrust::placeholders::_7 is the placeholder for the seventh function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<6>::type _7;


/*! \p thrust::placeholders::_8 is the placeholder for the eighth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<7>::type _8;


/*! \p thrust::placeholders::_9 is the placeholder for the ninth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<8>::type _9;


/*! \p thrust::placeholders::_10 is the placeholder for the tenth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<9>::type _10;


} // end placeholders


/*! \} // placeholder_objects
 */

#undef THRUST_UNARY_FUNCTOR_VOID_SPECIALIZATION
#undef THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION
#undef THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP

THRUST_NAMESPACE_END

#include <thrust/detail/functional.inl>
#include <thrust/detail/functional/operators.h>
