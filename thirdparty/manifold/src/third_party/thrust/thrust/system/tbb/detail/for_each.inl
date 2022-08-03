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
#include <thrust/detail/static_assert.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/system/detail/sequential/execution_policy.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{
namespace detail
{
namespace for_each_detail
{

template<typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
  struct body
{
  RandomAccessIterator m_first;
  UnaryFunction m_f;

  body(RandomAccessIterator first, UnaryFunction f)
    : m_first(first), m_f(f)
  {}

  void operator()(const ::tbb::blocked_range<Size> &r) const
  {
    // we assume that blocked_range specifies a contiguous range of integers
    thrust::for_each_n(thrust::system::detail::sequential::seq, m_first + r.begin(), r.size(), m_f);
  } // end operator()()
}; // end body


template<typename Size, typename RandomAccessIterator, typename UnaryFunction>
  body<RandomAccessIterator,Size,UnaryFunction>
    make_body(RandomAccessIterator first, UnaryFunction f)
{
  return body<RandomAccessIterator,Size,UnaryFunction>(first, f);
} // end make_body()


} // end for_each_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
RandomAccessIterator for_each_n(execution_policy<DerivedPolicy> &,
                                RandomAccessIterator first,
                                Size n,
                                UnaryFunction f)
{
  ::tbb::parallel_for(::tbb::blocked_range<Size>(0,n), for_each_detail::make_body<Size>(first,f));

  // return the end of the range
  return first + n;
} // end for_each_n


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename UnaryFunction>
  RandomAccessIterator for_each(execution_policy<DerivedPolicy> &s,
                                RandomAccessIterator first,
                                RandomAccessIterator last,
                                UnaryFunction f)
{
  return tbb::detail::for_each_n(s, first, thrust::distance(first,last), f);
} // end for_each()


} // end namespace detail
} // end namespace tbb
} // end namespace system
THRUST_NAMESPACE_END

