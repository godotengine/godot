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


/*! \file remove.h
 *  \brief Sequential implementations of remove functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/function.h>
#include <thrust/system/detail/sequential/execution_policy.h>

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
         typename Predicate>
__host__ __device__
  ForwardIterator remove_if(sequential::execution_policy<DerivedPolicy> &,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  // advance iterators until wrapped_pred(*first) is true or we reach the end of input
  while(first != last && !wrapped_pred(*first))
    ++first;

  if(first == last)
    return first;

  // result always trails first 
  ForwardIterator result = first;

  ++first;

  while(first != last)
  {
    if(!wrapped_pred(*first))
    {
      *result = *first;
      ++result;
    }
    ++first;
  }

  return result;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator remove_if(sequential::execution_policy<DerivedPolicy> &,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  // advance iterators until wrapped_pred(*stencil) is true or we reach the end of input
  while(first != last && !wrapped_pred(*stencil))
  {
    ++first;
    ++stencil;
  }

  if(first == last)
    return first;

  // result always trails first 
  ForwardIterator result = first;

  ++first;
  ++stencil;

  while(first != last)
  {
    if(!wrapped_pred(*stencil))
    {
      *result = *first;
      ++result;
    }
    ++first;
    ++stencil;
  }

  return result;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
  OutputIterator remove_copy_if(sequential::execution_policy<DerivedPolicy> &,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  while (first != last)
  {
    if (!wrapped_pred(*first))
    {
      *result = *first;
      ++result;
    }

    ++first;
  }

  return result;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
  OutputIterator remove_copy_if(sequential::execution_policy<DerivedPolicy> &,
                                InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  while (first != last)
  {
    if (!wrapped_pred(*stencil))
    {
      *result = *first;
      ++result;
    }

    ++first;
    ++stencil;
  }

  return result;
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

