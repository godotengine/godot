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
 *  \brief An extensible type trait for determining if an iterator satisifies the
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>
 *  requirements (aka is pointer-like).
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>

#include <iterator>
#include <type_traits>
#include <utility>

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC && _MSC_VER < 1916 // MSVC 2017 version 15.9
  #include <vector>
  #include <string>
  #include <array>

  #if THRUST_CPP_DIALECT >= 2017
    #include <string_view>
  #endif
#endif

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

template <typename Iterator>
struct is_contiguous_iterator_impl;

} // namespace detail

/*! \endcond
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory, and \c false_type
 *  otherwise.
 *
 * \see is_contiguous_iterator_v
 * \see proclaim_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
#if THRUST_CPP_DIALECT >= 2011
using is_contiguous_iterator =
#else
struct is_contiguous_iterator :
#endif
  detail::is_contiguous_iterator_impl<Iterator>
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/*! \brief <tt>constexpr bool</tt> that is \c true if \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory, and \c false
 *  otherwise.
 *
 * \see is_contiguous_iterator
 * \see proclaim_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
constexpr bool is_contiguous_iterator_v = is_contiguous_iterator<Iterator>::value;
#endif

/*! \brief Customization point that can be customized to indicate that an
 *  iterator type \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory.
 *
 * \see is_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
struct proclaim_contiguous_iterator : false_type {};

/*! \brief Declares that the iterator \c Iterator is
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>
 *  by specializing \c proclaim_contiguous_iterator.
 *
 * \see is_contiguous_iterator
 * \see proclaim_contiguous_iterator
 */
#define THRUST_PROCLAIM_CONTIGUOUS_ITERATOR(Iterator)                         \
  THRUST_NAMESPACE_BEGIN                                                      \
  template <>                                                                 \
  struct proclaim_contiguous_iterator<Iterator>                               \
      : THRUST_NS_QUALIFIER::true_type {};                                    \
  THRUST_NAMESPACE_END                                                        \
  /**/

/*! \cond
 */

namespace detail
{

template <typename Iterator>
struct is_libcxx_wrap_iter : false_type {};

#if defined(_LIBCPP_VERSION)
template <typename Iterator>
struct is_libcxx_wrap_iter<
  _VSTD::__wrap_iter<Iterator>
> : true_type {};
#endif

template <typename Iterator>
struct is_libstdcxx_normal_iterator : false_type {};

#if defined(__GLIBCXX__)
template <typename Iterator, typename Container>
struct is_libstdcxx_normal_iterator<
  ::__gnu_cxx::__normal_iterator<Iterator, Container>
> : true_type {};
#endif

#if   _MSC_VER >= 1916 // MSVC 2017 version 15.9.
template <typename Iterator>
struct is_msvc_contiguous_iterator
  : is_pointer<::std::_Unwrapped_t<Iterator> > {};
#elif _MSC_VER >= 1700 // MSVC 2012.
template <typename Iterator>
struct is_msvc_contiguous_iterator : false_type {};

template <typename Vector>
struct is_msvc_contiguous_iterator<
  ::std::_Vector_const_iterator<Vector>
> : true_type {};

template <typename Vector>
struct is_msvc_contiguous_iterator<
  ::std::_Vector_iterator<Vector>
> : true_type {};

template <typename String>
struct is_msvc_contiguous_iterator<
  ::std::_String_const_iterator<String>
> : true_type {};

template <typename String>
struct is_msvc_contiguous_iterator<
  ::std::_String_iterator<String>
> : true_type {};

template <typename T, std::size_t N>
struct is_msvc_contiguous_iterator<
  ::std::_Array_const_iterator<T, N>
> : true_type {};

template <typename T, std::size_t N>
struct is_msvc_contiguous_iterator<
  ::std::_Array_iterator<T, N>
> : true_type {};

#if THRUST_CPP_DIALECT >= 2017
template <typename Traits>
struct is_msvc_contiguous_iterator<
  ::std::_String_view_iterator<Traits>
> : true_type {};
#endif
#else
template <typename Iterator>
struct is_msvc_contiguous_iterator : false_type {};
#endif

template <typename Iterator>
struct is_contiguous_iterator_impl
  : integral_constant<
      bool
    ,    is_pointer<Iterator>::value
      || is_thrust_pointer<Iterator>::value
      || is_libcxx_wrap_iter<Iterator>::value
      || is_libstdcxx_normal_iterator<Iterator>::value
      || is_msvc_contiguous_iterator<Iterator>::value
      || proclaim_contiguous_iterator<Iterator>::value
    >
{};

// Type traits for contiguous iterators:
template <typename Iterator>
struct contiguous_iterator_traits
{
  static_assert(thrust::is_contiguous_iterator<Iterator>::value,
                "contiguous_iterator_traits requires a contiguous iterator.");

  using raw_pointer = typename thrust::detail::pointer_traits<
    decltype(&*std::declval<Iterator>())>::raw_pointer;
};

template <typename Iterator>
using contiguous_iterator_raw_pointer_t =
  typename contiguous_iterator_traits<Iterator>::raw_pointer;

// Converts a contiguous iterator to a raw pointer:
template <typename Iterator>
__host__ __device__
contiguous_iterator_raw_pointer_t<Iterator>
contiguous_iterator_raw_pointer_cast(Iterator it)
{
  static_assert(thrust::is_contiguous_iterator<Iterator>::value,
                "contiguous_iterator_raw_pointer_cast called with "
                "non-contiguous iterator.");
  return thrust::raw_pointer_cast(&*it);
}

// Implementation for non-contiguous iterators -- passthrough.
template <typename Iterator,
          bool IsContiguous = thrust::is_contiguous_iterator<Iterator>::value>
struct try_unwrap_contiguous_iterator_impl
{
  using type = Iterator;

  static __host__ __device__ type get(Iterator it) { return it; }
};

// Implementation for contiguous iterators -- unwraps to raw pointer.
template <typename Iterator>
struct try_unwrap_contiguous_iterator_impl<Iterator, true /*is_contiguous*/>
{
  using type = contiguous_iterator_raw_pointer_t<Iterator>;

  static __host__ __device__ type get(Iterator it)
  {
    return contiguous_iterator_raw_pointer_cast(it);
  }
};

template <typename Iterator>
using try_unwrap_contiguous_iterator_return_t =
  typename try_unwrap_contiguous_iterator_impl<Iterator>::type;

// Casts to a raw pointer if iterator is marked as contiguous, otherwise returns
// the input iterator.
template <typename Iterator>
__host__ __device__
try_unwrap_contiguous_iterator_return_t<Iterator>
try_unwrap_contiguous_iterator(Iterator it)
{
  return try_unwrap_contiguous_iterator_impl<Iterator>::get(it);
}

} // namespace detail

/*! \endcond
 */

///////////////////////////////////////////////////////////////////////////////

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

