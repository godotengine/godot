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


/*! \file partition.h
 *  \brief Sequential implementations of partition functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/function.h>
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{


// XXX WAR an unfortunate circular #inclusion problem
template<typename,typename> class temporary_array;


} // end detail

namespace system
{
namespace detail
{
namespace sequential
{


__thrust_exec_check_disable__
template<typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
void iter_swap(ForwardIterator1 iter1, ForwardIterator2 iter2)
{
  // XXX this isn't correct because it doesn't use thrust::swap
  using namespace thrust::detail;

  typedef typename thrust::iterator_value<ForwardIterator1>::type T;

  T temp = *iter1;
  *iter1 = *iter2;
  *iter2 = temp;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(sequential::execution_policy<DerivedPolicy> &,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  if(first == last)
    return first;

  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  while(wrapped_pred(*first))
  {
    if(++first == last)
      return first;
  }

  ForwardIterator next = first;

  while(++next != last)
  {
    if(wrapped_pred(*next))
    {
      // Fully qualify name to disambiguate overloads found via ADL.
      THRUST_NS_QUALIFIER::system::detail::sequential::iter_swap(first, next);
      ++first;
    }
  }

  return first;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(sequential::execution_policy<DerivedPolicy> &,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil_first,
                            Predicate pred)
{
  if(first == last)
    return first;

  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  while(wrapped_pred(*stencil_first))
  {
    ++stencil_first;
    if(++first == last)
    {
      return first;
    }
  }

  ForwardIterator next = first;

  // advance stencil to next element as well
  ++stencil_first;

  while(++next != last)
  {
    if(wrapped_pred(*stencil_first))
    {
      // Fully qualify name to disambiguate overloads found via ADL.
      THRUST_NS_QUALIFIER::system::detail::sequential::iter_swap(first, next);
      ++first;
    }

    ++stencil_first;
  }

  return first;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(sequential::execution_policy<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  typedef typename thrust::iterator_value<ForwardIterator>::type T;

  typedef thrust::detail::temporary_array<T,DerivedPolicy> TempRange;
  typedef typename TempRange::iterator                     TempIterator;

  TempRange temp(exec, first, last);

  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter)
  {
    if(wrapped_pred(*iter))
    {
      *first = *iter;
      ++first;
    }
  }

  ForwardIterator middle = first;

  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter)
  {
    if(!wrapped_pred(*iter))
    {
      *first = *iter;
      ++first;
    }
  }

  return middle;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(sequential::execution_policy<DerivedPolicy> &exec,
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

  typedef typename thrust::iterator_value<ForwardIterator>::type T;

  typedef thrust::detail::temporary_array<T,DerivedPolicy> TempRange;
  typedef typename TempRange::iterator                     TempIterator;

  TempRange temp(exec, first, last);

  InputIterator stencil_iter = stencil;
  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter, ++stencil_iter)
  {
    if(wrapped_pred(*stencil_iter))
    {
      *first = *iter;
      ++first;
    }
  }

  ForwardIterator middle = first;
  stencil_iter = stencil;

  for(TempIterator iter = temp.begin(); iter != temp.end(); ++iter, ++stencil_iter)
  {
    if(!wrapped_pred(*stencil_iter))
    {
      *first = *iter;
      ++first;
    }
  }

  return middle;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(sequential::execution_policy<DerivedPolicy> &,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  for(; first != last; ++first)
  {
    if(wrapped_pred(*first))
    {
      *out_true = *first;
      ++out_true;
    } // end if
    else
    {
      *out_false = *first;
      ++out_false;
    } // end else
  }

  return thrust::make_pair(out_true, out_false);
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(sequential::execution_policy<DerivedPolicy> &,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  for(; first != last; ++first, ++stencil)
  {
    if(wrapped_pred(*stencil))
    {
      *out_true = *first;
      ++out_true;
    } // end if
    else
    {
      *out_false = *first;
      ++out_false;
    } // end else
  }

  return thrust::make_pair(out_true, out_false);
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

