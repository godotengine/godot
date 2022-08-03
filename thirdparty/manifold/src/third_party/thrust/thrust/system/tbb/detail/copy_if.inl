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
#include <thrust/detail/function.h>
#include <thrust/system/tbb/detail/copy_if.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{
namespace detail
{
namespace copy_if_detail
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate,
         typename Size>
struct body
{

  InputIterator1 first;
  InputIterator2 stencil;
  OutputIterator result;
  thrust::detail::wrapped_function<Predicate,bool> pred;
  Size sum;

  body(InputIterator1 first, InputIterator2 stencil, OutputIterator result, Predicate pred)
    : first(first), stencil(stencil), result(result), pred(pred), sum(0)
  {}

  body(body& b, ::tbb::split)
    : first(b.first), stencil(b.stencil), result(b.result), pred(b.pred), sum(0)
  {}

  void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::pre_scan_tag)
  {
    InputIterator2 iter = stencil + r.begin();

    for (Size i = r.begin(); i != r.end(); ++i, ++iter)
    {
      if (pred(*iter))
        ++sum;
    }
  }
  
  void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::final_scan_tag)
  {
    InputIterator1  iter1 = first   + r.begin();
    InputIterator2  iter2 = stencil + r.begin();
    OutputIterator  iter3 = result  + sum;
      
    for (Size i = r.begin(); i != r.end(); ++i, ++iter1, ++iter2)
    {
      if (pred(*iter2))
      {
        *iter3 = *iter1;
        ++sum;
        ++iter3;
      }
    }
  }

  void reverse_join(body& b)
  {
    sum = b.sum + sum;
  } 

  void assign(body& b)
  {
    sum = b.sum;
  } 
}; // end body

} // end copy_if_detail

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(tag,
                         InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type Size; 
  typedef typename copy_if_detail::body<InputIterator1,InputIterator2,OutputIterator,Predicate,Size> Body;
  
  Size n = thrust::distance(first, last);

  if (n != 0)
  {
    Body body(first, stencil, result, pred);
    ::tbb::parallel_scan(::tbb::blocked_range<Size>(0,n), body);
    thrust::advance(result, body.sum);
  }

  return result;
} // end copy_if()

} // end detail
} // end tbb
} // end system
THRUST_NAMESPACE_END

