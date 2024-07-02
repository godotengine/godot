/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file general_copy.h
 *  \brief Sequential copy algorithms for general iterators.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{
namespace general_copy_detail
{


template<typename T1, typename T2>
struct lazy_is_assignable
  : thrust::detail::is_assignable<
      typename T1::type,
      typename T2::type
    >
{};


// sometimes OutputIterator's reference type is reported as void
// in that case, just assume that we're able to assign to it OK
template<typename InputIterator, typename OutputIterator>
struct reference_is_assignable
  : thrust::detail::eval_if<
      thrust::detail::is_same<
        typename thrust::iterator_reference<OutputIterator>::type, void
      >::value,
      thrust::detail::true_type,
      lazy_is_assignable<
        thrust::iterator_reference<OutputIterator>,
        thrust::iterator_reference<InputIterator>
      >
    >::type
{};


// introduce an iterator assign helper to deal with assignments from
// a wrapped reference

__thrust_exec_check_disable__
template<typename OutputIterator, typename InputIterator>
inline __host__ __device__
typename thrust::detail::enable_if<
  reference_is_assignable<InputIterator,OutputIterator>::value
>::type
iter_assign(OutputIterator dst, InputIterator src)
{
  *dst = *src;
}


__thrust_exec_check_disable__
template<typename OutputIterator, typename InputIterator>
inline __host__ __device__
typename thrust::detail::disable_if<
  reference_is_assignable<InputIterator,OutputIterator>::value
>::type
iter_assign(OutputIterator dst, InputIterator src)
{
  typedef typename thrust::iterator_value<InputIterator>::type value_type;

  // insert a temporary and hope for the best
  *dst = static_cast<value_type>(*src);
}


} // end general_copy_detail


__thrust_exec_check_disable__
template<typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator general_copy(InputIterator first,
                              InputIterator last,
                              OutputIterator result)
{
  for(; first != last; ++first, ++result)
  {
    // gcc 4.2 crashes while instantiating iter_assign
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) && (THRUST_GCC_VERSION < 40300)
    *result = *first;
#else
    general_copy_detail::iter_assign(result, first);
#endif
  }

  return result;
} // end general_copy()


__thrust_exec_check_disable__
template<typename InputIterator,
         typename Size,
         typename OutputIterator>
__host__ __device__
  OutputIterator general_copy_n(InputIterator first,
                                Size n,
                                OutputIterator result)
{
  for(; n > Size(0); ++first, ++result, --n)
  {
    // gcc 4.2 crashes while instantiating iter_assign
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) && (THRUST_GCC_VERSION < 40300)
    *result = *first;
#else
    general_copy_detail::iter_assign(result, first);
#endif
  }

  return result;
} // end general_copy_n()


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

