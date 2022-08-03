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
#include <thrust/detail/static_assert.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{
namespace detail
{
namespace reduce_detail
{

template<typename RandomAccessIterator,
         typename OutputType,
         typename BinaryFunction>
struct body
{
  RandomAccessIterator first;
  OutputType sum;
  bool first_call;  // TBB can invoke operator() multiple times on the same body
  thrust::detail::wrapped_function<BinaryFunction,OutputType> binary_op;

  // note: we only initalize sum with init to avoid calling OutputType's default constructor
  body(RandomAccessIterator first, OutputType init, BinaryFunction binary_op)
    : first(first), sum(init), first_call(true), binary_op(binary_op)
  {}

  // note: we only initalize sum with b.sum to avoid calling OutputType's default constructor
  body(body& b, ::tbb::split)
    : first(b.first), sum(b.sum), first_call(true), binary_op(b.binary_op)
  {}

  template <typename Size>
  void operator()(const ::tbb::blocked_range<Size> &r)
  {
    // we assume that blocked_range specifies a contiguous range of integers
    
    if (r.empty()) return; // nothing to do

    RandomAccessIterator iter = first + r.begin();

    OutputType temp = thrust::raw_reference_cast(*iter);

    ++iter;

    for (Size i = r.begin() + 1; i != r.end(); ++i, ++iter)
      temp = binary_op(temp, *iter);


    if (first_call)
    {
      // first time body has been invoked
      first_call = false;
      sum = temp;
    }
    else
    {
      // body has been previously invoked, accumulate temp into sum
      sum = binary_op(sum, temp);
    }
  } // end operator()()
  
  void join(body& b)
  {
    sum = binary_op(sum, b.sum);
  }
}; // end body

} // end reduce_detail


template<typename DerivedPolicy,
         typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(execution_policy<DerivedPolicy> &,
                    InputIterator begin,
                    InputIterator end,
                    OutputType init,
                    BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<InputIterator>::type Size; 

  Size n = thrust::distance(begin, end);

  if (n == 0)
  {
    return init;
  }
  else
  {
    typedef typename reduce_detail::body<InputIterator,OutputType,BinaryFunction> Body;
    Body reduce_body(begin, init, binary_op);
    ::tbb::parallel_reduce(::tbb::blocked_range<Size>(0,n), reduce_body);
    return binary_op(init, reduce_body.sum);
  }
}


} // end namespace detail
} // end namespace tbb
} // end namespace system
THRUST_NAMESPACE_END

