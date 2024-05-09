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
 *  \brief C++17's
 *  <a href="https://en.cppreference.com/w/cpp/types/conjunction"><tt>std::conjunction</tt></a>,
 *  <a href="https://en.cppreference.com/w/cpp/types/disjunction"><tt>std::disjunction</tt></a>,
 *  and <a href="https://en.cppreference.com/w/cpp/types/negation"><tt>std::negation</tt></a>
 *  metafunctions and related extensions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <type_traits>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
 *  whose value is <tt>(... && Ts::value)</tt>.
 *
 *  \see conjunction_v
 *  \see conjunction_value
 *  \see <a href="https://en.cppreference.com/w/cpp/types/conjunction"><tt>std::conjunction</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2017
template <typename... Ts>
using conjunction = std::conjunction<Ts...>;
#else // Older than C++17.
template <typename... Ts>
struct conjunction;

/*! \cond
 */

template <>
struct conjunction<> : std::true_type {};

template <typename T>
struct conjunction<T> : T {};

template <typename T0, typename T1>
struct conjunction<T0, T1> : std::conditional<T0::value, T1, T0>::type {};

template<typename T0, typename T1, typename T2, typename... TN>
struct conjunction<T0, T1, T2, TN...>
  : std::conditional<T0::value, conjunction<T1, T2, TN...>, T0>::type {};

/*! \endcond
 */
#endif

/*! \brief <tt>constexpr bool</tt> whose value is <tt>(... && Ts::value)</tt>.
 *
 *  \see conjunction
 *  \see conjunction_value
 *  \see <a href="https://en.cppreference.com/w/cpp/types/conjunction"><tt>std::conjunction</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2014
template <typename... Ts>
constexpr bool conjunction_v = conjunction<Ts...>::value;
#endif

/*! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
 *  whose value is <tt>(... || Ts::value)</tt>.
 *
 *  \see disjunction_v
 *  \see disjunction_value
 *  \see <a href="https://en.cppreference.com/w/cpp/types/disjunction"><tt>std::disjunction</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2017
template <typename... Ts>
using disjunction = std::disjunction<Ts...>;
#else // Older than C++17.
template <typename... Ts>
struct disjunction;

/*! \cond
 */

template <>
struct disjunction<> : std::false_type {};

template <typename T>
struct disjunction<T> : T {};

template <typename T0, typename... TN>
struct disjunction<T0, TN...>
  : std::conditional<T0::value != false, T0, disjunction<TN...> >::type {};

/*! \endcond
 */
#endif

/*! \brief <tt>constexpr bool</tt> whose value is <tt>(... || Ts::value)</tt>.
 *
 *  \see disjunction
 *  \see disjunction_value
 *  \see <a href="https://en.cppreference.com/w/cpp/types/disjunction"><tt>std::disjunction</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2014
template <typename... Ts>
constexpr bool disjunction_v = disjunction<Ts...>::value;
#endif

/*! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
 *  whose value is <tt>!Ts::value</tt>.
 *
 *  \see negation_v
 *  \see negation_value
 *  \see <a href="https://en.cppreference.com/w/cpp/types/negation"><tt>std::negation</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2017
template <typename T>
using negation = std::negation<T>;
#else // Older than C++17.
template <typename T>
struct negation;

/*! \cond
 */

template <typename T>
struct negation : std::integral_constant<bool, !T::value> {};

/*! \endcond
 */
#endif

/*! \brief <tt>constexpr bool</tt> whose value is <tt>!Ts::value</tt>.
 *
 *  \see negation
 *  \see negation_value
 *  \see <a href="https://en.cppreference.com/w/cpp/types/negation"><tt>std::negation</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2014
template <typename T>
constexpr bool negation_v = negation<T>::value;
#endif

///////////////////////////////////////////////////////////////////////////////

/*! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
 *  whose value is <tt>(... && Bs)</tt>.
 *
 *  \see conjunction_value_v
 *  \see conjunction
 *  \see <a href="https://en.cppreference.com/w/cpp/types/conjunction"><tt>std::conjunction</tt></a>
 */
template <bool... Bs>
struct conjunction_value;

#if THRUST_CPP_DIALECT >= 2014
/*! \brief <tt>constexpr bool</tt> whose value is <tt>(... && Bs)</tt>.
 *
 *  \see conjunction_value
 *  \see conjunction
 *  \see <a href="https://en.cppreference.com/w/cpp/types/conjunction"><tt>std::conjunction</tt></a>
 */
template <bool... Bs>
constexpr bool conjunction_value_v = conjunction_value<Bs...>::value;
#endif

/*! \cond
 */

template <>
struct conjunction_value<> : std::true_type {};

template <bool B>
struct conjunction_value<B> : std::integral_constant<bool, B> {};

template <bool B, bool... Bs>
struct conjunction_value<B, Bs...>
  : std::integral_constant<bool, B && conjunction_value<Bs...>::value> {};

/*! \endcond
 */

///////////////////////////////////////////////////////////////////////////////

/*! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
 *  whose value is <tt>(... || Bs)</tt>.
 *
 *  \see disjunction_value_v
 *  \see disjunction
 *  \see <a href="https://en.cppreference.com/w/cpp/types/disjunction"><tt>std::disjunction</tt></a>
 */
template <bool... Bs>
struct disjunction_value;

#if THRUST_CPP_DIALECT >= 2014
/*! \brief <tt>constexpr bool</tt> whose value is <tt>(... || Bs)</tt>.
 *
 *  \see disjunction_value
 *  \see disjunction
 *  \see <a href="https://en.cppreference.com/w/cpp/types/disjunction"><tt>std::disjunction</tt></a>
 */
template <bool... Bs>
constexpr bool disjunction_value_v = disjunction_value<Bs...>::value;
#endif

/*! \cond
 */

template <>
struct disjunction_value<> : std::false_type {};

template <bool B>
struct disjunction_value<B> : std::integral_constant<bool, B> {};

template <bool B, bool... Bs>
struct disjunction_value<B, Bs...>
  : std::integral_constant<bool, B || disjunction_value<Bs...>::value> {};

/*! \endcond
 */

///////////////////////////////////////////////////////////////////////////////

/*! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
 *  whose value is <tt>!Bs</tt>.
 *
 *  \see negation_value_v
 *  \see negation
 *  \see <a href="https://en.cppreference.com/w/cpp/types/negation"><tt>std::negation</tt></a>
 */
template <bool B>
struct negation_value;

#if THRUST_CPP_DIALECT >= 2014
/*! \brief <tt>constexpr bool</tt> whose value is <tt>!Ts::value</tt>.
 *
 *  \see negation_value
 *  \see negation
 *  \see <a href="https://en.cppreference.com/w/cpp/types/negation"><tt>std::negation</tt></a>
 */
template <bool B>
constexpr bool negation_value_v = negation_value<B>::value;
#endif

/*! \cond
 */

template <bool B>
struct negation_value : std::integral_constant<bool, !B> {};

/*! \endcond
 */

///////////////////////////////////////////////////////////////////////////////

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

