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


/*! \file binary_search.h
 *  \brief Sequential implementation of binary search algorithms.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/function.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T,
         typename StrictWeakOrdering>
__host__ __device__
ForwardIterator lower_bound(sequential::execution_policy<DerivedPolicy> &,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T& val,
                            StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  typedef typename thrust::iterator_difference<ForwardIterator>::type difference_type;

  difference_type len = thrust::distance(first, last);

  while(len > 0)
  {
    difference_type half = len >> 1;
    ForwardIterator middle = first;

    thrust::advance(middle, half);

    if(wrapped_comp(*middle, val))
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
    else
    {
      len = half;
    }
  }

  return first;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T,
         typename StrictWeakOrdering>
__host__ __device__
ForwardIterator upper_bound(sequential::execution_policy<DerivedPolicy> &,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T& val, 
                            StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  typedef typename thrust::iterator_difference<ForwardIterator>::type difference_type;

  difference_type len = thrust::distance(first, last);

  while(len > 0)
  {
    difference_type half = len >> 1;
    ForwardIterator middle = first;

    thrust::advance(middle, half);

    if(wrapped_comp(val, *middle))
    {
      len = half;
    }
    else
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
  }

  return first;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T,
         typename StrictWeakOrdering>
__host__ __device__
bool binary_search(sequential::execution_policy<DerivedPolicy> &exec,
                   ForwardIterator first,
                   ForwardIterator last,
                   const T& val, 
                   StrictWeakOrdering comp)
{
  ForwardIterator iter = sequential::lower_bound(exec, first, last, val, comp);

  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  return iter != last && !wrapped_comp(val,*iter);
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

