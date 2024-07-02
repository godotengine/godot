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
 *  \brief A type trait that determines if a type is an \a ExecutionPolicy.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/type_traits.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c T is an \a ExecutionPolicy and \c false_type
 *  otherwise.
 */
template <typename T>
#if THRUST_CPP_DIALECT >= 2011
using is_execution_policy =
#else
struct is_execution_policy :
#endif
  detail::is_base_of<detail::execution_policy_marker, T>
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/*! \brief <tt>constexpr bool</tt> that is \c true if \c T is an
 *  \a ExecutionPolicy and \c false otherwise.
 */
template <typename T>
constexpr bool is_execution_policy_v = is_execution_policy<T>::value;
#endif

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

