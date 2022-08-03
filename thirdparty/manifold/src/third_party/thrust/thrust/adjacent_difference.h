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


/*! \file adjacent_difference.h
 *  \brief Compute difference between consecutive elements of a range
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup transformations Transformations
 *  \{
 */


/*! \p adjacent_difference calculates the differences of adjacent elements in the
 *  range <tt>[first, last)</tt>. That is, <tt>\*first</tt> is assigned to
 *  <tt>\*result</tt>, and, for each iterator \p i in the range
 *  <tt>[first + 1, last)</tt>, the difference of <tt>\*i</tt> and <tt>*(i - 1)</tt>
 *  is assigned to <tt>\*(result + (i - first))</tt>.
 *
 *  This version of \p adjacent_difference uses <tt>operator-</tt> to calculate
 *  differences.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param result The beginning of the output range.
 *  \return The iterator <tt>result + (last - first)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \c x and \c y are objects of \p InputIterator's \c value_type, then \c x - \c is defined,
 *          and \p InputIterator's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types,
 *          and the return type of <tt>x - y</tt> is convertible to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \remark Note that \p result is permitted to be the same iterator as \p first. This is
 *          useful for computing differences "in place".
 *
 *  The following code snippet demonstrates how to use \p adjacent_difference to compute
 *  the difference between adjacent elements of a range using the \p thrust::device execution policy:
 *
 *  \code
 *  #include <thrust/adjacent_difference.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
 *  thrust::device_vector<int> d_data(h_data, h_data + 8);
 *  thrust::device_vector<int> d_result(8);
 *
 *  thrust::adjacent_difference(thrust::device, d_data.begin(), d_data.end(), d_result.begin());
 *
 *  // d_result is now [1, 1, -1, 1, -1, 1, -1, 1]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/adjacent_difference
 *  \see inclusive_scan
 */
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator adjacent_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   InputIterator first, InputIterator last, 
                                   OutputIterator result);

/*! \p adjacent_difference calculates the differences of adjacent elements in the
 *  range <tt>[first, last)</tt>. That is, <tt>*first</tt> is assigned to
 *  <tt>\*result</tt>, and, for each iterator \p i in the range
 *  <tt>[first + 1, last)</tt>, <tt>binary_op(\*i, \*(i - 1))</tt> is assigned to
 *  <tt>\*(result + (i - first))</tt>.
 *  
 *  This version of \p adjacent_difference uses the binary function \p binary_op to
 *  calculate differences.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param result The beginning of the output range.
 *  \param binary_op The binary function used to compute differences.
 *  \return The iterator <tt>result + (last - first)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p BinaryFunction's \c first_argument_type and \c second_argument_type,
 *          and \p InputIterator's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam BinaryFunction's \c result_type is convertible to a type in \p OutputIterator's set of \c value_types.
 *
 *  \remark Note that \p result is permitted to be the same iterator as \p first. This is
 *          useful for computing differences "in place".
 *
 *  The following code snippet demonstrates how to use \p adjacent_difference to compute
 *  the sum between adjacent elements of a range using the \p thrust::device execution policy:
 *
 *  \code
 *  #include <thrust/adjacent_difference.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
 *  thrust::device_vector<int> d_data(h_data, h_data + 8);
 *  thrust::device_vector<int> d_result(8);
 *
 *  thrust::adjacent_difference(thrust::device, d_data.begin(), d_data.end(), d_result.begin(), thrust::plus<int>());
 *
 *  // d_result is now [1, 3, 3, 3, 3, 3, 3, 3]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/adjacent_difference
 *  \see inclusive_scan
 */
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
__host__ __device__
OutputIterator adjacent_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op);

/*! \p adjacent_difference calculates the differences of adjacent elements in the
 *  range <tt>[first, last)</tt>. That is, <tt>\*first</tt> is assigned to
 *  <tt>\*result</tt>, and, for each iterator \p i in the range
 *  <tt>[first + 1, last)</tt>, the difference of <tt>\*i</tt> and <tt>*(i - 1)</tt>
 *  is assigned to <tt>\*(result + (i - first))</tt>.
 *
 *  This version of \p adjacent_difference uses <tt>operator-</tt> to calculate
 *  differences.
 *
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param result The beginning of the output range.
 *  \return The iterator <tt>result + (last - first)</tt>
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \c x and \c y are objects of \p InputIterator's \c value_type, then \c x - \c is defined,
 *          and \p InputIterator's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types,
 *          and the return type of <tt>x - y</tt> is convertible to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \remark Note that \p result is permitted to be the same iterator as \p first. This is
 *          useful for computing differences "in place".
 *
 *  The following code snippet demonstrates how to use \p adjacent_difference to compute
 *  the difference between adjacent elements of a range.
 *
 *  \code
 *  #include <thrust/adjacent_difference.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
 *  thrust::device_vector<int> d_data(h_data, h_data + 8);
 *  thrust::device_vector<int> d_result(8);
 *
 *  thrust::adjacent_difference(d_data.begin(), d_data.end(), d_result.begin());
 *
 *  // d_result is now [1, 1, -1, 1, -1, 1, -1, 1]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/adjacent_difference
 *  \see inclusive_scan
 */
template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(InputIterator first, InputIterator last, 
                                   OutputIterator result);

/*! \p adjacent_difference calculates the differences of adjacent elements in the
 *  range <tt>[first, last)</tt>. That is, <tt>*first</tt> is assigned to
 *  <tt>\*result</tt>, and, for each iterator \p i in the range
 *  <tt>[first + 1, last)</tt>, <tt>binary_op(\*i, \*(i - 1))</tt> is assigned to
 *  <tt>\*(result + (i - first))</tt>.
 *  
 *  This version of \p adjacent_difference uses the binary function \p binary_op to
 *  calculate differences.
 *
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param result The beginning of the output range.
 *  \param binary_op The binary function used to compute differences.
 *  \return The iterator <tt>result + (last - first)</tt>
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p BinaryFunction's \c first_argument_type and \c second_argument_type,
 *          and \p InputIterator's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam BinaryFunction's \c result_type is convertible to a type in \p OutputIterator's set of \c value_types.
 *
 *  \remark Note that \p result is permitted to be the same iterator as \p first. This is
 *          useful for computing differences "in place".
 *
 *  The following code snippet demonstrates how to use \p adjacent_difference to compute
 *  the sum between adjacent elements of a range.
 *
 *  \code
 *  #include <thrust/adjacent_difference.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
 *  thrust::device_vector<int> d_data(h_data, h_data + 8);
 *  thrust::device_vector<int> d_result(8);
 *
 *  thrust::adjacent_difference(d_data.begin(), d_data.end(), d_result.begin(), thrust::plus<int>());
 *
 *  // d_result is now [1, 3, 3, 3, 3, 3, 3, 3]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/adjacent_difference
 *  \see inclusive_scan
 */
template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op);

/*! \}
 */

THRUST_NAMESPACE_END

#include <thrust/detail/adjacent_difference.inl>

