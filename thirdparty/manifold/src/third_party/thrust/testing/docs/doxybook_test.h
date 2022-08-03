/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

/*! \file
 *  \brief Test case for Doxybook rendering.
 */

#pragma once

namespace thrust
{

/*! \addtogroup test Test
 *  \{
 */

/*! \brief \c test_predefined_friend_struct is a class intended to exercise and
 *  test Doxybook rendering.
 */
template <typename... Z>
struct test_predefined_friend_struct {};

/*! \brief \c test_predefined_friend_function is a function intended to
 *  exercise and test Doxybook rendering.
 */
template <typename Z>
void test_predefined_friend_function();

/*! \brief \c test_class is a class intended to exercise and test Doxybook
 *  rendering.
 *
 *  It does many things.
 *
 *  \tparam T A template parameter.
 *  \tparam U Another template parameter.
 *
 *  \see test_function
 */
template <typename T, typename U>
class test_class
{
public:
  template <typename Z>
  struct test_nested_struct {};

  int test_member_variable = 0; ///< A test member variable.

  [[deprecated]] static constexpr int test_member_constant = 42; ///< A test member constant.

  template <typename X, typename Y>
  using test_type_alias = test_class<X, Y>;

  enum class test_enum_class {
    A = 15, ///< An enumerator. It is equal to 15.
    B,
    C
  };

  /*! \brief Construct an empty test class.
   */
  test_class() = default;

  /*! \brief Construct a test class.
   */
  __host__ __device__ constexpr
  test_class(int);

  /*! \brief \c test_member_function is a function intended to exercise
   *  and test Doxybook rendering.
   */
  __host__ __device__ constexpr
  int test_member_function() = 0;

  /*! \brief \c test_virtual_member_function is a function intended to exercise
   *  and test Doxybook rendering.
   */
  __host__ __device__
  virtual int test_virtual_member_function() = 0;

  /*! \brief \c test_parameter_overflow_member_function is a function intended
   *  to test Doxybook's rendering of function and template parameters that exceed
   *  the length of a line.
   */
  template <typename A = test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int>,
            typename B = test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int>,
            typename C = test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int>>
  test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int>
  test_parameter_overflow_member_function(test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int> a,
                                          test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int> b,
                                          test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int> c);

  template <typename Z>
  friend void test_friend_function() {}

  template <typename Z>
  friend void test_predefined_friend_function();

  template <typename... Z>
  friend struct thrust::test_predefined_friend_struct;

protected:

  template <typename Z>
  class test_protected_nested_class {};

  /*! \brief \c test_protected_member_function is a function intended to
   *  exercise and test Doxybook rendering.
   */
  __device__
  auto test_protected_member_function();
};

/*! \brief \c test_derived_class is a derived class intended to exercise and
 *  test Doxybook rendering.
 */
class test_derived_class : test_class<int, double>
{
  template <typename Z>
  struct test_derived_nested_struct {};

  double test_derived_member_variable = 3.14; ///< A test member variable.

  typedef double test_typedef;

  /*! \brief \c test_derived_member_function is a function intended to exercise
   *  and test Doxybook rendering.
   */
  __host__ __device__ constexpr
  double test_derived_member_function(int, int);
};

/*! \brief \c test_function is a function intended to exercise and test Doxybook
 *  rendering.
 *
 *  \tparam T A template parameter.
 *
 *  \param a A function parameter.
 *  \param b A function parameter.
 */
template <typename T>
void test_function(T const& a, test_class<T, T const>&& b);

/*! \brief \c test_parameter_overflow_function is a function intended to test
 *  Doxybook's rendering of function and template parameters that exceed the
 *  length of a line.
 */
template <typename T = test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int>,
  typename U = test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int>,
  typename V = test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int>
>
test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int>
test_parameter_overflow_function(test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int> t,
  test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int> u,
  test_predefined_friend_struct<int, int, int, int, int, int, int, int, int, int, int, int> v);

/*! \brief \c test_enum is an enum namespace intended to exercise and test
 *  Doxybook rendering.
 */
enum class test_enum {
  X = 1, ///< An enumerator. It is equal to 1.
  Y = X,
  Z = 2
};

/*! \brief \c test_alias is a type alias intended to exercise and test Doxybook
 * rendering.
 */
using test_alias = test_class<int, double>;

/*! \brief \c test_namespace is a namespace intended to exercise and test
 *  Doxybook rendering.
 */
namespace test_namespace {

inline constexpr int test_constant = 12;

/*! \brief \c nested_function is a function intended to exercise and test
 *  Doxybook rendering.
 */
template <typename T, typename U>
auto test_nested_function(T t, U u) noexcept(noexcept(t + u)) -> decltype(t + u)
{ return t + u; }

/*! \brief \c test_struct is a struct intended to exercise and test Doxybook
 *  rendering.
 */
template <typename Z>
struct test_struct
{
  test_struct& operator=(test_struct const&) = default;

  /*! \brief \c operator< is a function intended to exercise and test Doxybook
   *  rendering.
   */
  bool operator<(test_struct const& t);
};

} // namespace test_namespace

/*! \brief \c THRUST_TEST_MACRO is a macro intended to exercise and test
 *  Doxybook rendering.
 */
#define THRUST_TEST_MACRO(x, y) thrust::test_namespace::nested_function(x, y)

/*! \} // test
 */

} // namespace thrust

