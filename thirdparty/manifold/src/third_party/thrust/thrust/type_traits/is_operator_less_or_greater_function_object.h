/*
 *  Copyright 2008-2021 NVIDIA Corporation
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
 *  \brief Type traits for determining if a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>
 *  is equivalent to either \c operator< or \c operator>.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

/*! \cond
 */

namespace detail
{

template <typename T>
struct is_operator_less_function_object_impl;

template <typename T>
struct is_operator_greater_function_object_impl;

} // namespace detail

/*! \endcond
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c T is a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>
 *  equivalent to \c operator<, and \c false_type otherwise.
 *
 *  \see is_operator_less_function_object_v
 *  \see is_operator_greater_function_object
 *  \see is_operator_less_or_greater_function_object
 *  \see is_operator_plus_function_object
 */
template <typename T>
#if THRUST_CPP_DIALECT >= 2011
using is_operator_less_function_object =
#else
struct is_operator_less_function_object :
#endif
  detail::is_operator_less_function_object_impl<T>
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/*! \brief <tt>constexpr bool</tt> that is \c true if \c T is a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>
 *  equivalent to \c operator<, and \c false otherwise.
 *
 *  \see is_operator_less_function_object
 *  \see is_operator_greater_function_object
 *  \see is_operator_less_or_greater_function_object
 *  \see is_operator_plus_function_object
 */
template <typename T>
constexpr bool is_operator_less_function_object_v
  = is_operator_less_function_object<T>::value;
#endif

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c T is a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>
 *  equivalent to \c operator>, and \c false_type otherwise.
 *
 *  \see is_operator_greater_function_object_v
 *  \see is_operator_less_function_object
 *  \see is_operator_less_or_greater_function_object
 *  \see is_operator_plus_function_object
 */
template <typename T>
#if THRUST_CPP_DIALECT >= 2011
using is_operator_greater_function_object =
#else
struct is_operator_greater_function_object :
#endif
  detail::is_operator_greater_function_object_impl<T>
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/*! \brief <tt>constexpr bool</tt> that is \c true if \c T is a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>
 *  equivalent to \c operator>, and \c false otherwise.
 *
 *  \see is_operator_greater_function_object
 *  \see is_operator_less_function_object
 *  \see is_operator_less_or_greater_function_object
 *  \see is_operator_plus_function_object
 */
template <typename T>
constexpr bool is_operator_greater_function_object_v
  = is_operator_greater_function_object<T>::value;
#endif

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c T is a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>
 *  equivalent to \c operator< or \c operator>, and \c false_type otherwise.
 *
 *  \see is_operator_less_or_greater_function_object_v
 *  \see is_operator_less_function_object
 *  \see is_operator_greater_function_object
 *  \see is_operator_plus_function_object
 */
template <typename T>
#if THRUST_CPP_DIALECT >= 2011
using is_operator_less_or_greater_function_object =
#else
struct is_operator_less_or_greater_function_object :
#endif
  integral_constant<
    bool
  ,    detail::is_operator_less_function_object_impl<T>::value
    || detail::is_operator_greater_function_object_impl<T>::value
  >
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/*! \brief <tt>constexpr bool</tt> that is \c true if \c T is a
 *  <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>
 *  equivalent to \c operator< or \c operator>, and \c false otherwise.
 *
 *  \see is_operator_less_or_greater_function_object
 *  \see is_operator_less_function_object
 *  \see is_operator_greater_function_object
 *  \see is_operator_plus_function_object
 */
template <typename T>
constexpr bool is_operator_less_or_greater_function_object_v
  = is_operator_less_or_greater_function_object<T>::value;
#endif

///////////////////////////////////////////////////////////////////////////////

/*! \cond
 */

namespace detail
{

template <typename T>
struct is_operator_less_function_object_impl                   : false_type {};
template <typename T>
struct is_operator_less_function_object_impl<thrust::less<T> > : true_type {};
template <typename T>
struct is_operator_less_function_object_impl<std::less<T>    > : true_type {};

template <typename T>
struct is_operator_greater_function_object_impl                      : false_type {};
template <typename T>
struct is_operator_greater_function_object_impl<thrust::greater<T> > : true_type {};
template <typename T>
struct is_operator_greater_function_object_impl<std::greater<T>    > : true_type {};

} // namespace detail

/*! \endcond
 */

///////////////////////////////////////////////////////////////////////////////

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

