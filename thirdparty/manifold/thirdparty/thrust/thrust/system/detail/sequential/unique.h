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


/*! \file unique.h
 *  \brief Sequential implementations of unique algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/sequential/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator unique_copy(sequential::execution_policy<DerivedPolicy> &,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type T;

  if(first != last)
  {
    T prev = *first;

    for(++first; first != last; ++first)
    {
      T temp = *first;

      if (!binary_pred(prev, temp))
      {
        *output = prev;

        ++output;

        prev = temp;
      }
    }

    *output = prev;
    ++output;
  }

  return output;
} // end unique_copy()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
  ForwardIterator unique(sequential::execution_policy<DerivedPolicy> &exec,
                         ForwardIterator first,
                         ForwardIterator last,
                         BinaryPredicate binary_pred)
{
  // sequential unique_copy permits in-situ operation
  return sequential::unique_copy(exec, first, last, first, binary_pred);
} // end unique()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
  typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(sequential::execution_policy<DerivedPolicy> &,
                 ForwardIterator first,
                 ForwardIterator last,
                 BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type T;
  typename thrust::iterator_traits<ForwardIterator>::difference_type count{};

  if(first != last)
  {
    count++;
    T prev = *first;

    for(++first; first != last; ++first)
    {
      T temp = *first;

      if (!binary_pred(prev, temp))
      {
        count++;
        prev = temp;
      }
    }
  }

  return count;
} // end unique()


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

