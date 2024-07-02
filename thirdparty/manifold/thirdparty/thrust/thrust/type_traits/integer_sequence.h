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
 *  \brief C++14's
 *  <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><tt>std::index_sequence</tt></a>,
 *  associated helper aliases, and some related extensions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <type_traits>
#include <utility>
#include <cstdint>
#include <utility>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

/*! \brief A compile-time sequence of
 *  <a href="https://en.cppreference.com/w/cpp/language/constant_expression#Integral_constant_expression"><i>integral constants</i></a>
 *  of type \c T with values <tt>Is...</tt>.
 *
 *  \see <a href="https://en.cppreference.com/w/cpp/language/constant_expression#Integral_constant_expression"><i>integral constants</i></a>
 *  \see index_sequence
 *  \see make_integer_sequence
 *  \see make_reversed_integer_sequence
 *  \see make_index_sequence
 *  \see make_reversed_index_sequence
 *  \see integer_sequence_push_front
 *  \see integer_sequence_push_back
 *  \see <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><tt>std::integer_sequence</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2014
template <typename T, T... Is>
using integer_sequence = std::integer_sequence<T, Is...>;
#else
template <typename T, T... Is>
struct integer_sequence
{
  using type = integer_sequence;
  using value_type = T;
  using size_type = std::size_t;

  __host__ __device__
  static constexpr size_type size() noexcept
  {
    return sizeof...(Is);
  }
};
#endif

///////////////////////////////////////////////////////////////////////////////

/*! \brief A compile-time sequence of type
 *  <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a>
 *  with values <tt>Is...</tt>.
 *
 *  \see integer_sequence
 *  \see make_integer_sequence
 *  \see make_reversed_integer_sequence
 *  \see make_index_sequence
 *  \see make_reversed_index_sequence
 *  \see integer_sequence_push_front
 *  \see integer_sequence_push_back
 *  \see <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><tt>std::index_sequence</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2014
template <std::size_t... Is>
using index_sequence = std::index_sequence<Is...>;
#else
template <std::size_t... Is>
using index_sequence = integer_sequence<std::size_t, Is...>;
#endif

#if THRUST_CPP_DIALECT < 2014
/*! \cond
 */

namespace detail
{

/*! \brief Create a new \c integer_sequence containing the elements of \c
 * Sequence0 followed by the elements of \c Sequence1. \c Sequence0::size() is
 * added to each element from \c Sequence1 in the new sequence.
 *
 *  \see integer_sequence
 *  \see index_sequence
 *  \see make_reversed_integer_sequence
 *  \see make_index_sequence
 *  \see make_reversed_index_sequence
 *  \see merge_and_renumber_reversed_integer_sequences_impl
 */
template <typename Sequence0, typename Sequence1>
  struct merge_and_renumber_integer_sequences_impl;
template <typename Sequence0, typename Sequence1>
  using merge_and_renumber_integer_sequences =
      typename merge_and_renumber_integer_sequences_impl<
          Sequence0, Sequence1
      >::type;

template <typename T, std::size_t N>
  struct make_integer_sequence_impl;

} // namespace detail

/*! \endcond
 */
#endif

/*! \brief Create a new \c integer_sequence with elements
 *  <tt>0, 1, 2, ..., N - 1</tt> of type \c T.
 *
 *  \see integer_sequence
 *  \see index_sequence
 *  \see make_reversed_integer_sequence
 *  \see make_index_sequence
 *  \see make_reversed_index_sequence
 *  \see <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><tt>std::make_integer_sequence</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2014
template <typename T, std::size_t N>
using make_integer_sequence = std::make_integer_sequence<T, N>;
#else
template <typename T, std::size_t N>
using make_integer_sequence =
  typename detail::make_integer_sequence_impl<T, N>::type;

/*! \cond
 */

namespace detail
{

template <typename T, T... Is0, T... Is1>
struct merge_and_renumber_integer_sequences_impl<
  integer_sequence<T, Is0...>, integer_sequence<T, Is1...>
>
{
  using type = integer_sequence<T, Is0..., (sizeof...(Is0) + Is1)...>;
};

template <typename T, std::size_t N>
struct make_integer_sequence_impl
{
  using type = merge_and_renumber_integer_sequences<
    make_integer_sequence<T, N / 2>
  , make_integer_sequence<T, N - N / 2>
  >;
};

template <typename T>
struct make_integer_sequence_impl<T, 0>
{
  using type = integer_sequence<T>;
};

template <typename T>
struct make_integer_sequence_impl<T, 1>
{
  using type = integer_sequence<T, 0>;
};

} // namespace detail

/*! \endcond
 */
#endif

///////////////////////////////////////////////////////////////////////////////

/*! \brief Create a new \c integer_sequence with elements
 *  <tt>0, 1, 2, ..., N - 1</tt> of type
 *  <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a>.
 *
 *  \see integer_sequence
 *  \see index_sequence
 *  \see make_integer_sequence
 *  \see make_reversed_integer_sequence
 *  \see make_reversed_index_sequence
 *  \see <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><tt>std::make_index_sequence</tt></a>
 */
#if THRUST_CPP_DIALECT >= 2014
template <std::size_t N>
using make_index_sequence = std::make_index_sequence<N>;
#else
template <std::size_t N>
using make_index_sequence =
  make_integer_sequence<std::size_t, N>;
#endif

///////////////////////////////////////////////////////////////////////////////

/*! \cond
 */

namespace detail
{

/*! \brief Create a new \c integer_sequence containing the elements of \c
 *  Sequence0 followed by the elements of \c Sequence1. \c Sequence1::size() is
 *  added to each element from \c Sequence0 in the new sequence.
 *
 *  \see make_reversed_integer_sequence
 *  \see integer_sequence
 *  \see index_sequence
 *  \see make_integer_sequence
 *  \see make_reversed_integer_sequence
 *  \see make_index_sequence
 *  \see make_reversed_index_sequence
 *  \see merge_and_renumber_integer_sequences_impl
 */
template <typename Sequence0, typename Sequence1>
  struct merge_and_renumber_reversed_integer_sequences_impl;
template <typename Sequence0, typename Sequence1>
  using merge_and_renumber_reversed_integer_sequences =
      typename merge_and_renumber_reversed_integer_sequences_impl<
          Sequence0, Sequence1
      >::type;

template <typename T, std::size_t N>
struct make_reversed_integer_sequence_impl;

template <typename T, T Value, typename Sequence>
struct integer_sequence_push_front_impl;

template <typename T, T Value, typename Sequence>
struct integer_sequence_push_back_impl;

template <typename T, T... Is0, T... Is1>
struct merge_and_renumber_reversed_integer_sequences_impl<
  integer_sequence<T, Is0...>, integer_sequence<T, Is1...>
>
{
  using type = integer_sequence<T, (sizeof...(Is1) + Is0)..., Is1...>;
};

} // namespace detail

/*! \endcond
 */

///////////////////////////////////////////////////////////////////////////////

/*! \brief Create a new \c integer_sequence with elements
 *  <tt>N - 1, N - 2, N - 3, ..., 0</tt>.
 *
 *  \see integer_sequence
 *  \see index_sequence
 *  \see make_integer_sequence
 *  \see make_index_sequence
 *  \see make_reversed_index_sequence
 */
template <typename T, std::size_t N>
using make_reversed_integer_sequence =
  typename detail::make_reversed_integer_sequence_impl<T, N>::type;

/*! \brief Create a new \c index_sequence with elements
 *  <tt>N - 1, N - 2, N - 3, ..., 0</tt>.
 *
 *  \see integer_sequence
 *  \see index_sequence
 *  \see make_integer_sequence
 *  \see make_reversed_integer_sequence
 *  \see make_reversed_index_sequence
 */
template <std::size_t N>
using make_reversed_index_sequence =
  make_reversed_integer_sequence<std::size_t, N>;

/*! \brief Add a new element to the front of an \c integer_sequence.
 *
 *  \see integer_sequence
 *  \see index_sequence
 *  \see make_integer_sequence
 *  \see make_index_sequence
 */
template <typename T, T Value, typename Sequence>
using integer_sequence_push_front =
  typename detail::integer_sequence_push_front_impl<T, Value, Sequence>::type;

/*! \brief Add a new element to the back of an \c integer_sequence.
 *
 *  \see integer_sequence
 *  \see index_sequence
 *  \see make_integer_sequence
 *  \see make_index_sequence
 */
template <typename T, T Value, typename Sequence>
using integer_sequence_push_back =
  typename detail::integer_sequence_push_back_impl<T, Value, Sequence>::type;

///////////////////////////////////////////////////////////////////////////////

/*! \cond
 */

namespace detail
{

template <typename T, std::size_t N>
struct make_reversed_integer_sequence_impl
{
  using type = merge_and_renumber_reversed_integer_sequences<
      make_reversed_integer_sequence<T, N / 2>
    , make_reversed_integer_sequence<T, N - N / 2>
  >;
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct make_reversed_integer_sequence_impl<T, 0>
{
  using type = integer_sequence<T>;
};

template <typename T>
struct make_reversed_integer_sequence_impl<T, 1>
{
  using type = integer_sequence<T, 0>;
};

///////////////////////////////////////////////////////////////////////////////

template <typename T, T I0, T... Is>
struct integer_sequence_push_front_impl<T, I0, integer_sequence<T, Is...> >
{
  using type = integer_sequence<T, I0, Is...>;
};

///////////////////////////////////////////////////////////////////////////////

template <typename T, T I0, T... Is>
struct integer_sequence_push_back_impl<T, I0, integer_sequence<T, Is...> >
{
  using type = integer_sequence<T, Is..., I0>;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace detail

/*! \endcond
 */

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

