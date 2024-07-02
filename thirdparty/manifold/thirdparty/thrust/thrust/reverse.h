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


/*! \file reverse.h
 *  \brief Reverses the order of a range
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup reordering
 *  \ingroup algorithms
 */


/*! \p reverse reverses a range. That is: for every <tt>i</tt> such that
 *  <tt>0 <= i <= (last - first) / 2</tt>, it exchanges <tt>*(first + i)</tt>
 *  and <tt>*(last - (i + 1))</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range to reverse.
 *  \param last The end of the range to reverse.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam BidirectionalIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/bidirectional_iterator">Bidirectional Iterator</a> and
 *          \p BidirectionalIterator is mutable.
 *
 *  The following code snippet demonstrates how to use \p reverse to reverse a
 *  \p device_vector of integers using the \p thrust::device execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/reverse.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int data[N] = {0, 1, 2, 3, 4, 5};
 *  thrust::device_vector<int> v(data, data + N);
 *  thrust::reverse(thrust::device, v.begin(), v.end());
 *  // v is now {5, 4, 3, 2, 1, 0}
 *  \endcode
 *  
 *  \see https://en.cppreference.com/w/cpp/algorithm/reverse
 *  \see \p reverse_copy
 *  \see \p reverse_iterator
 */
template<typename DerivedPolicy, typename BidirectionalIterator>
__host__ __device__
  void reverse(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               BidirectionalIterator first,
               BidirectionalIterator last);


/*! \p reverse reverses a range. That is: for every <tt>i</tt> such that
 *  <tt>0 <= i <= (last - first) / 2</tt>, it exchanges <tt>*(first + i)</tt>
 *  and <tt>*(last - (i + 1))</tt>.
 *
 *  \param first The beginning of the range to reverse.
 *  \param last The end of the range to reverse.
 *
 *  \tparam BidirectionalIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/bidirectional_iterator">Bidirectional Iterator</a> and
 *          \p BidirectionalIterator is mutable.
 *
 *  The following code snippet demonstrates how to use \p reverse to reverse a
 *  \p device_vector of integers.
 *
 *  \code
 *  #include <thrust/reverse.h>
 *  ...
 *  const int N = 6;
 *  int data[N] = {0, 1, 2, 3, 4, 5};
 *  thrust::device_vector<int> v(data, data + N);
 *  thrust::reverse(v.begin(), v.end());
 *  // v is now {5, 4, 3, 2, 1, 0}
 *  \endcode
 *  
 *  \see https://en.cppreference.com/w/cpp/algorithm/reverse
 *  \see \p reverse_copy
 *  \see \p reverse_iterator
 */
template<typename BidirectionalIterator>
  void reverse(BidirectionalIterator first,
               BidirectionalIterator last);


/*! \p reverse_copy differs from \p reverse only in that the reversed range
 *  is written to a different output range, rather than inplace.
 *
 *  \p reverse_copy copies elements from the range <tt>[first, last)</tt> to the
 *  range <tt>[result, result + (last - first))</tt> such that the copy is a 
 *  reverse of the original range. Specifically: for every <tt>i</tt> such that
 *  <tt>0 <= i < (last - first)</tt>, \p reverse_copy performs the assignment
 *  <tt>*(result + (last - first) - i) = *(first + i)</tt>.
 *
 *  The return value is <tt>result + (last - first))</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range to reverse.
 *  \param last The end of the range to reverse.
 *  \param result The beginning of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam BidirectionalIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/bidirectional_iterator">Bidirectional Iterator</a>,
 *          and \p BidirectionalIterator's \p value_type is convertible to \p OutputIterator's \p value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p reverse_copy to reverse
 *  an input \p device_vector of integers to an output \p device_vector using the \p thrust::device
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/reverse.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int data[N] = {0, 1, 2, 3, 4, 5};
 *  thrust::device_vector<int> input(data, data + N);
 *  thrust::device_vector<int> output(N);
 *  thrust::reverse_copy(thrust::device, v.begin(), v.end(), output.begin());
 *  // input is still {0, 1, 2, 3, 4, 5}
 *  // output is now  {5, 4, 3, 2, 1, 0}
 *  \endcode
 *  
 *  \see https://en.cppreference.com/w/cpp/algorithm/reverse_copy
 *  \see \p reverse
 *  \see \p reverse_iterator
 */
template<typename DerivedPolicy, typename BidirectionalIterator, typename OutputIterator>
__host__ __device__
  OutputIterator reverse_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result);


/*! \p reverse_copy differs from \p reverse only in that the reversed range
 *  is written to a different output range, rather than inplace.
 *
 *  \p reverse_copy copies elements from the range <tt>[first, last)</tt> to the
 *  range <tt>[result, result + (last - first))</tt> such that the copy is a 
 *  reverse of the original range. Specifically: for every <tt>i</tt> such that
 *  <tt>0 <= i < (last - first)</tt>, \p reverse_copy performs the assignment
 *  <tt>*(result + (last - first) - i) = *(first + i)</tt>.
 *
 *  The return value is <tt>result + (last - first))</tt>.
 *
 *  \param first The beginning of the range to reverse.
 *  \param last The end of the range to reverse.
 *  \param result The beginning of the output range.
 *
 *  \tparam BidirectionalIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/bidirectional_iterator">Bidirectional Iterator</a>,
 *          and \p BidirectionalIterator's \p value_type is convertible to \p OutputIterator's \p value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p reverse_copy to reverse
 *  an input \p device_vector of integers to an output \p device_vector.
 *
 *  \code
 *  #include <thrust/reverse.h>
 *  ...
 *  const int N = 6;
 *  int data[N] = {0, 1, 2, 3, 4, 5};
 *  thrust::device_vector<int> input(data, data + N);
 *  thrust::device_vector<int> output(N);
 *  thrust::reverse_copy(v.begin(), v.end(), output.begin());
 *  // input is still {0, 1, 2, 3, 4, 5}
 *  // output is now  {5, 4, 3, 2, 1, 0}
 *  \endcode
 *  
 *  \see https://en.cppreference.com/w/cpp/algorithm/reverse_copy
 *  \see \p reverse
 *  \see \p reverse_iterator
 */
template<typename BidirectionalIterator, typename OutputIterator>
  OutputIterator reverse_copy(BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result);


/*! \} // end reordering
 */

THRUST_NAMESPACE_END

#include <thrust/detail/reverse.inl>
