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


/*! \file set_operations.h
 *  \brief Sequential implementation of set operation functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/sequential/execution_policy.h>
#include <thrust/detail/copy.h>
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
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
__host__ __device__
  OutputIterator set_difference(sequential::execution_policy<DerivedPolicy> &exec,
                                InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result,
                                StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first1,*first2))
    {
      *result = *first1;
      ++first1;
      ++result;
    } // end if
    else if(wrapped_comp(*first2,*first1))
    {
      ++first2;
    } // end else if
    else
    {
      ++first1;
      ++first2;
    } // end else
  } // end while

  return thrust::copy(exec, first1, last1, result);
} // end set_difference()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
__host__ __device__
  OutputIterator set_intersection(sequential::execution_policy<DerivedPolicy> &,
                                  InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first1,*first2))
    {
      ++first1;
    } // end if
    else if(wrapped_comp(*first2,*first1))
    {
      ++first2;
    } // end else if
    else
    {
      *result = *first1;
      ++first1;
      ++first2;
      ++result;
    } // end else
  } // end while

  return result;
} // end set_intersection()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
__host__ __device__
  OutputIterator set_symmetric_difference(sequential::execution_policy<DerivedPolicy> &exec,
                                          InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result,
                                          StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first1,*first2))
    {
      *result = *first1;
      ++first1;
      ++result;
    } // end if
    else if(wrapped_comp(*first2,*first1))
    {
      *result = *first2;
      ++first2;
      ++result;
    } // end else if
    else
    {
      ++first1;
      ++first2;
    } // end else
  } // end while

  return thrust::copy(exec, first2, last2, thrust::copy(exec, first1, last1, result));
} // end set_symmetric_difference()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
__host__ __device__
  OutputIterator set_union(sequential::execution_policy<DerivedPolicy> &exec,
                           InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result,
                           StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first1,*first2))
    {
      *result = *first1;
      ++first1;
    } // end if
    else if(wrapped_comp(*first2,*first1))
    {
      *result = *first2;
      ++first2;
    } // end else if
    else
    {
      *result = *first1;
      ++first1;
      ++first2;
    } // end else

    ++result;
  } // end while

  return thrust::copy(exec, first2, last2, thrust::copy(exec, first1, last1, result));
} // end set_union()


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

