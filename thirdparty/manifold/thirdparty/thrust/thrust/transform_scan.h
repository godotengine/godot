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


/*! \file transform_scan.h
 *  \brief Fused transform / prefix-sum
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup prefixsums Prefix Sums
 *  \ingroup algorithms
 *  \{
 */
	
/*! \addtogroup transformed_prefixsums Transformed Prefix Sums
 *  \ingroup prefixsums
 *  \{
 */


/*! \p transform_inclusive_scan fuses the \p transform and \p inclusive_scan
 *  operations.  \p transform_inclusive_scan is equivalent to performing a
 *  tranformation defined by \p unary_op into a temporary sequence and then
 *  performing an \p inclusive_scan on the tranformed sequence.  In most
 *  cases, fusing these two operations together is more efficient, since
 *  fewer memory reads and writes are required. In \p transform_inclusive_scan,
 *  <tt>unary_op(\*first)</tt> is assigned to <tt>\*result</tt> and the result
 *  of <tt>binary_op(unary_op(\*first), unary_op(\*(first + 1)))</tt> is
 *  assigned to <tt>\*(result + 1)</tt>, and so on.  The transform scan
 *  operation is permitted to be in-place.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param unary_op The function used to tranform the input sequence.
 *  \param binary_op The associatve operator used to 'sum' transformed values.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                               and \c InputIterator's \c value_type is convertible to \c unary_op's input type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                               and accepts inputs of \c InputIterator's \c value_type.  \c UnaryFunction's result_type
 *                               is convertable to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_inclusive_scan using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/transform_scan.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::negate<int> unary_op;
 *  thrust::plus<int> binary_op;
 *
 *  thrust::transform_inclusive_scan(thrust::host, data, data + 6, data, unary_op, binary_op); // in-place scan
 *
 *  // data is now {-1, -1, -3, -5, -6, -9}
 *  \endcode
 *
 *  \see \p transform
 *  \see \p inclusive_scan
 *
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator transform_inclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          AssociativeOperator binary_op);


/*! \p transform_inclusive_scan fuses the \p transform and \p inclusive_scan
 *  operations.  \p transform_inclusive_scan is equivalent to performing a
 *  tranformation defined by \p unary_op into a temporary sequence and then
 *  performing an \p inclusive_scan on the tranformed sequence.  In most
 *  cases, fusing these two operations together is more efficient, since
 *  fewer memory reads and writes are required. In \p transform_inclusive_scan,
 *  <tt>unary_op(\*first)</tt> is assigned to <tt>\*result</tt> and the result
 *  of <tt>binary_op(unary_op(\*first), unary_op(\*(first + 1)))</tt> is
 *  assigned to <tt>\*(result + 1)</tt>, and so on.  The transform scan
 *  operation is permitted to be in-place.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param unary_op The function used to tranform the input sequence.
 *  \param binary_op The associatve operator used to 'sum' transformed values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                               and \c InputIterator's \c value_type is convertible to \c unary_op's input type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                               and accepts inputs of \c InputIterator's \c value_type.  \c UnaryFunction's result_type
 *                               is convertable to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_inclusive_scan
 *
 *  \code
 *  #include <thrust/transform_scan.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::negate<int> unary_op;
 *  thrust::plus<int> binary_op;
 *
 *  thrust::transform_inclusive_scan(data, data + 6, data, unary_op, binary_op); // in-place scan
 *
 *  // data is now {-1, -1, -3, -5, -6, -9}
 *  \endcode
 *
 *  \see \p transform
 *  \see \p inclusive_scan
 *
 */
template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
  OutputIterator transform_inclusive_scan(InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          AssociativeOperator binary_op);


/*! \p transform_exclusive_scan fuses the \p transform and \p exclusive_scan
 *  operations.  \p transform_exclusive_scan is equivalent to performing a
 *  tranformation defined by \p unary_op into a temporary sequence and then
 *  performing an \p exclusive_scan on the tranformed sequence.  In most
 *  cases, fusing these two operations together is more efficient, since
 *  fewer memory reads and writes are required. In 
 *  \p transform_exclusive_scan, \p init is assigned to <tt>\*result</tt> 
 *  and the result of <tt>binary_op(init, unary_op(\*first))</tt> is assigned
 *  to <tt>\*(result + 1)</tt>, and so on.  The transform scan operation is 
 *  permitted to be in-place.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param unary_op The function used to tranform the input sequence.
 *  \param init The initial value of the \p exclusive_scan
 *  \param binary_op The associatve operator used to 'sum' transformed values.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                               and \c InputIterator's \c value_type is convertible to \c unary_op's input type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                               and accepts inputs of \c InputIterator's \c value_type.  \c UnaryFunction's result_type
 *                               is convertable to \c OutputIterator's \c value_type.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_exclusive_scan using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/transform_scan.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::negate<int> unary_op;
 *  thrust::plus<int> binary_op;
 *
 *  thrust::transform_exclusive_scan(thrust::host, data, data + 6, data, unary_op, 4, binary_op); // in-place scan
 *
 *  // data is now {4, 3, 3, 1, -1, -2}
 *  \endcode
 *
 *  \see \p transform
 *  \see \p exclusive_scan
 *
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator transform_exclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          T init,
                                          AssociativeOperator binary_op);


/*! \p transform_exclusive_scan fuses the \p transform and \p exclusive_scan
 *  operations.  \p transform_exclusive_scan is equivalent to performing a
 *  tranformation defined by \p unary_op into a temporary sequence and then
 *  performing an \p exclusive_scan on the tranformed sequence.  In most
 *  cases, fusing these two operations together is more efficient, since
 *  fewer memory reads and writes are required. In 
 *  \p transform_exclusive_scan, \p init is assigned to <tt>\*result</tt> 
 *  and the result of <tt>binary_op(init, unary_op(\*first))</tt> is assigned
 *  to <tt>\*(result + 1)</tt>, and so on.  The transform scan operation is 
 *  permitted to be in-place.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param unary_op The function used to tranform the input sequence.
 *  \param init The initial value of the \p exclusive_scan
 *  \param binary_op The associatve operator used to 'sum' transformed values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                               and \c InputIterator's \c value_type is convertible to \c unary_op's input type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                               and accepts inputs of \c InputIterator's \c value_type.  \c UnaryFunction's result_type
 *                               is convertable to \c OutputIterator's \c value_type.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_exclusive_scan
 *
 *  \code
 *  #include <thrust/transform_scan.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::negate<int> unary_op;
 *  thrust::plus<int> binary_op;
 *
 *  thrust::transform_exclusive_scan(data, data + 6, data, unary_op, 4, binary_op); // in-place scan
 *
 *  // data is now {4, 3, 3, 1, -1, -2}
 *  \endcode
 *
 *  \see \p transform
 *  \see \p exclusive_scan
 *
 */
template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
  OutputIterator transform_exclusive_scan(InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          T init,
                                          AssociativeOperator binary_op);


/*! \} // end transformed_prefixsums
 */


/*! \} // end prefixsums
 */

THRUST_NAMESPACE_END

#include <thrust/detail/transform_scan.inl>
