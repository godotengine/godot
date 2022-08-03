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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/detail/copy.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cpp/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator sequential_copy(InputIterator first,
                                 InputIterator last,
                                 OutputIterator result)
{
  for(; first != last; ++first, ++result)
  {
    *result = *first;
  } // end for

  return result;
} // end sequential_copy()


template<typename BidirectionalIterator1,
         typename BidirectionalIterator2>
  BidirectionalIterator2 sequential_copy_backward(BidirectionalIterator1 first,
                                                  BidirectionalIterator1 last,
                                                  BidirectionalIterator2 result)
{
  // yes, we preincrement
  // the ranges are open on the right, i.e. [first, last)
  while(first != last)
  {
    *--result = *--last;
  } // end while

  return result;
} // end sequential_copy_backward()


namespace dispatch
{


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 overlapped_copy(thrust::system::cpp::detail::execution_policy<DerivedPolicy> &,
                                        RandomAccessIterator1 first,
                                        RandomAccessIterator1 last,
                                        RandomAccessIterator2 result)
{
  if(first < last && first <= result && result < last)
  {
    // result lies in [first, last)
    // it's safe to use std::copy_backward here
    thrust::detail::sequential_copy_backward(first, last, result + (last - first));
    result += (last - first);
  } // end if
  else
  {
    // result + (last - first) lies in [first, last)
    // it's safe to use sequential_copy here
    result = thrust::detail::sequential_copy(first, last, result);
  } // end else

  return result;
} // end overlapped_copy()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 overlapped_copy(thrust::execution_policy<DerivedPolicy> &exec,
                                        RandomAccessIterator1 first,
                                        RandomAccessIterator1 last,
                                        RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

  // make a temporary copy of [first,last), and copy into it first
  thrust::detail::temporary_array<value_type, DerivedPolicy> temp(exec, first, last);
  return thrust::copy(exec, temp.begin(), temp.end(), result);
} // end overlapped_copy()

} // end dispatch


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 overlapped_copy(RandomAccessIterator1 first,
                                        RandomAccessIterator1 last,
                                        RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_system<RandomAccessIterator2>::type System1;
  typedef typename thrust::iterator_system<RandomAccessIterator2>::type System2;

  typedef typename thrust::detail::minimum_system<System1, System2>::type System;

  // XXX presumes System is default constructible
  System system;

  return thrust::detail::dispatch::overlapped_copy(system, first, last, result);
} // end overlapped_copy()

} // end detail

THRUST_NAMESPACE_END
