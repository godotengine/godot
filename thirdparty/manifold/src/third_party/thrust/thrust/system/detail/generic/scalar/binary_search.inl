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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/detail/function.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

namespace system
{

namespace detail
{

namespace generic
{

namespace scalar
{

template<typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound_n(RandomAccessIterator first,
                                   Size n,
                                   const T &val,
                                   BinaryPredicate comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    BinaryPredicate,
    bool
  > wrapped_comp(comp);

  Size start = 0, i;
  while(start < n)
  {
    i = start + (n - start) / 2;  // Overflow-safe variant of (a+b)/2
    if(wrapped_comp(first[i], val))
    {
      start = i + 1;
    }
    else
    {
      n = i;
    }
  } // end while

  return first + start;
}

// XXX generalize these upon implementation of scalar::distance & scalar::advance

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound(RandomAccessIterator first, RandomAccessIterator last,
                                 const T &val,
                                 BinaryPredicate comp)
{
  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;
  return lower_bound_n(first, n, val, comp);
}

template<typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound_n(RandomAccessIterator first,
                                   Size n,
                                   const T &val,
                                   BinaryPredicate comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    BinaryPredicate,
    bool
  > wrapped_comp(comp);

  Size start = 0, i;
  while(start < n)
  {
    i = start + (n - start) / 2;  // Overflow-safe variant of (a+b)/2
    if(wrapped_comp(val, first[i]))
    {
      n = i;
    }
    else
    {
      start = i + 1;
    }
  } // end while

  return first + start;
}

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound(RandomAccessIterator first, RandomAccessIterator last,
                                 const T &val,
                                 BinaryPredicate comp)
{
  typename thrust::iterator_difference<RandomAccessIterator>::type n = last - first;
  return upper_bound_n(first, n, val, comp);
}

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
  pair<RandomAccessIterator,RandomAccessIterator>
    equal_range(RandomAccessIterator first, RandomAccessIterator last,
                const T &val,
                BinaryPredicate comp)
{
  RandomAccessIterator lb = thrust::system::detail::generic::scalar::lower_bound(first, last, val, comp);
  return thrust::make_pair(lb, thrust::system::detail::generic::scalar::upper_bound(lb, last, val, comp));
}


template<typename RandomAccessIterator, typename T, typename Compare>
__host__ __device__
bool binary_search(RandomAccessIterator first, RandomAccessIterator last, const T &value, Compare comp)
{
  RandomAccessIterator iter = thrust::system::detail::generic::scalar::lower_bound(first, last, value, comp);

  // wrap comp
  thrust::detail::wrapped_function<
    Compare,
    bool
  > wrapped_comp(comp);

  return iter != last && !wrapped_comp(value,*iter);
}

} // end scalar

} // end generic

} // end detail

} // end system

THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/scalar/binary_search.inl>
