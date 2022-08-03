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


/*! \file scan.h
 *  \brief Functions for computing prefix sums
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


/*! \p inclusive_scan computes an inclusive prefix sum operation. The
 *  term 'inclusive' means that each result includes the corresponding
 *  input operand in the partial sum. More precisely, <tt>*first</tt> is 
 *  assigned to <tt>*result</tt> and the sum of <tt>*first</tt> and 
 *  <tt>*(first + 1)</tt> is assigned to <tt>*(result + 1)</tt>, and so on. 
 *  This version of \p inclusive_scan assumes plus as the associative operator.  
 *  When the input and output sequences are the same, the scan is performed 
 *  in-place.
 *
 *  \p inclusive_scan is similar to \c std::partial_sum in the STL.  The primary
 *  difference between the two functions is that \c std::partial_sum guarantees
 *  a serial summation order, while \p inclusive_scan requires associativity of 
 *  the binary operation to parallelize the prefix sum.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *    
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan to compute an in-place
 *  prefix sum using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::inclusive_scan(thrust::host, data, data + 6, data); // in-place scan
 *
 *  // data is now {1, 1, 3, 5, 6, 9}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 *
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator inclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result);


/*! \p inclusive_scan computes an inclusive prefix sum operation. The
 *  term 'inclusive' means that each result includes the corresponding
 *  input operand in the partial sum. More precisely, <tt>*first</tt> is 
 *  assigned to <tt>*result</tt> and the sum of <tt>*first</tt> and 
 *  <tt>*(first + 1)</tt> is assigned to <tt>*(result + 1)</tt>, and so on. 
 *  This version of \p inclusive_scan assumes plus as the associative operator.  
 *  When the input and output sequences are the same, the scan is performed 
 *  in-place.
 *
 *  \p inclusive_scan is similar to \c std::partial_sum in the STL.  The primary
 *  difference between the two functions is that \c std::partial_sum guarantees
 *  a serial summation order, while \p inclusive_scan requires associativity of 
 *  the binary operation to parallelize the prefix sum.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan
 *
 *  \code
 *  #include <thrust/scan.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::inclusive_scan(data, data + 6, data); // in-place scan
 *
 *  // data is now {1, 1, 3, 5, 6, 9}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 *
 */
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result);


/*! \p inclusive_scan computes an inclusive prefix sum operation. The
 *  term 'inclusive' means that each result includes the corresponding
 *  input operand in the partial sum.  When the input and output sequences 
 *  are the same, the scan is performed in-place.
 *
 *  \p inclusive_scan is similar to \c std::partial_sum in the STL.  The primary
 *  difference between the two functions is that \c std::partial_sum guarantees
 *  a serial summation order, while \p inclusive_scan requires associativity of 
 *  the binary operation to parallelize the prefix sum.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>
 *                         and \c OutputIterator's \c value_type is convertible to
 *                         both \c AssociativeOperator's \c first_argument_type and
 *                         \c second_argument_type.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan to compute an in-place
 *  prefix sum using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 * 
 *  thrust::maximum<int> binary_op;
 *
 *  thrust::inclusive_scan(thrust::host, data, data + 10, data, binary_op); // in-place scan
 *
 *  // data is now {-5, 0, 2, 2, 2, 4, 4, 4, 4, 8}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator inclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op);


/*! \p inclusive_scan computes an inclusive prefix sum operation. The
 *  term 'inclusive' means that each result includes the corresponding
 *  input operand in the partial sum.  When the input and output sequences 
 *  are the same, the scan is performed in-place.
 *    
 *  \p inclusive_scan is similar to \c std::partial_sum in the STL.  The primary
 *  difference between the two functions is that \c std::partial_sum guarantees
 *  a serial summation order, while \p inclusive_scan requires associativity of 
 *  the binary operation to parallelize the prefix sum.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>
 *                         and \c OutputIterator's \c value_type is convertible to
 *                         both \c AssociativeOperator's \c first_argument_type and
 *                         \c second_argument_type.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan
 *
 *  \code
 *  int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 * 
 *  thrust::maximum<int> binary_op;
 *
 *  thrust::inclusive_scan(data, data + 10, data, binary_op); // in-place scan
 *
 *  // data is now {-5, 0, 2, 2, 2, 4, 4, 4, 4, 8}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 */
template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op);


/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  <tt>0</tt> is assigned to <tt>*result</tt> and the sum of 
 *  <tt>0</tt> and <tt>*first</tt> is assigned to <tt>*(result + 1)</tt>,
 *  and so on. This version of \p exclusive_scan assumes plus as the 
 *  associative operator and \c 0 as the initial value.  When the input and 
 *  output sequences are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *    
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan to compute an in-place
 *  prefix sum using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::exclusive_scan(thrust::host, data, data + 6, data); // in-place scan
 *
 *  // data is now {0, 1, 1, 3, 5, 6}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator exclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result);


/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  <tt>0</tt> is assigned to <tt>*result</tt> and the sum of 
 *  <tt>0</tt> and <tt>*first</tt> is assigned to <tt>*(result + 1)</tt>,
 *  and so on. This version of \p exclusive_scan assumes plus as the 
 *  associative operator and \c 0 as the initial value.  When the input and 
 *  output sequences are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan
 *
 *  \code
 *  #include <thrust/scan.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::exclusive_scan(data, data + 6, data); // in-place scan
 *
 *  // data is now {0, 1, 1, 3, 5, 6}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 */
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result);


/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  \p init is assigned to <tt>*result</tt> and the sum of \p init and 
 *  <tt>*first</tt> is assigned to <tt>*(result + 1)</tt>, and so on. 
 *  This version of \p exclusive_scan assumes plus as the associative 
 *  operator but requires an initial value \p init.  When the input and 
 *  output sequences are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan to compute an in-place
 *  prefix sum using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/execution_policy.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::exclusive_scan(thrust::host, data, data + 6, data, 4); // in-place scan
 *
 *  // data is now {4, 5, 5, 7, 9, 10}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T>
__host__ __device__
  OutputIterator exclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init);


/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  \p init is assigned to <tt>*result</tt> and the sum of \p init and 
 *  <tt>*first</tt> is assigned to <tt>*(result + 1)</tt>, and so on. 
 *  This version of \p exclusive_scan assumes plus as the associative 
 *  operator but requires an initial value \p init.  When the input and 
 *  output sequences are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan
 *
 *  \code
 *  #include <thrust/scan.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::exclusive_scan(data, data + 6, data, 4); // in-place scan
 *
 *  // data is now {4, 5, 5, 7, 9, 10}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 */
template<typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init);


/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  \p init is assigned to <tt>\*result</tt> and the value
 *  <tt>binary_op(init, \*first)</tt> is assigned to <tt>\*(result + 1)</tt>,
 *  and so on. This version of the function requires both an associative 
 *  operator and an initial value \p init.  When the input and output
 *  sequences are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *    
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>
 *                         and \c OutputIterator's \c value_type is convertible to
 *                         both \c AssociativeOperator's \c first_argument_type and
 *                         \c second_argument_type.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan to compute an in-place
 *  prefix sum using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 * 
 *  thrust::maximum<int> binary_op;
 *
 *  thrust::exclusive_scan(thrust::host, data, data + 10, data, 1, binary_op); // in-place scan
 *
 *  // data is now {1, 1, 1, 2, 2, 2, 4, 4, 4, 4 }
 *  \endcode
 *  
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator exclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op);


/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  \p init is assigned to <tt>\*result</tt> and the value
 *  <tt>binary_op(init, \*first)</tt> is assigned to <tt>\*(result + 1)</tt>,
 *  and so on. This version of the function requires both an associative 
 *  operator and an initial value \p init.  When the input and output
 *  sequences are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>
 *                         and \c OutputIterator's \c value_type is convertible to
 *                         both \c AssociativeOperator's \c first_argument_type and
 *                         \c second_argument_type.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 * 
 *  thrust::maximum<int> binary_op;
 *
 *  thrust::exclusive_scan(data, data + 10, data, 1, binary_op); // in-place scan
 *
 *  // data is now {1, 1, 1, 2, 2, 2, 4, 4, 4, 4 }
 *  \endcode
 *  
 *  \see https://en.cppreference.com/w/cpp/algorithm/partial_sum
 */
template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op);


/*! \addtogroup segmentedprefixsums Segmented Prefix Sums
 *  \ingroup prefixsums
 *  \{
 */


/*! \p inclusive_scan_by_key computes an inclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_scan_by_key assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first1, last1)</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to
 *  different segments otherwise.
 *
 *  This version of \p inclusive_scan_by_key assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's \c value_type, then 
 *                         <tt>binary_op(x,y)</tt> is defined.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan_by_key using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::inclusive_scan_by_key(thrust::host, keys, keys + 10, data, data); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *  \see exclusive_scan_by_key
 *
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator inclusive_scan_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result);
 

/*! \p inclusive_scan_by_key computes an inclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_scan_by_key assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first1, last1)</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to
 *  different segments otherwise.
 *
 *  This version of \p inclusive_scan_by_key assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's \c value_type, then 
 *                         <tt>binary_op(x,y)</tt> is defined.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan_by_key
 *
 *  \code
 *  #include <thrust/scan.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::inclusive_scan_by_key(keys, keys + 10, data, data); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *  \see exclusive_scan_by_key
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result);


/*! \p inclusive_scan_by_key computes an inclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_scan_by_key uses the binary predicate 
 *  \c pred to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first1, last1)</tt>
 *  belong to the same segment if <tt>binary_pred(*i, *(i+1))</tt> is true, and belong to 
 *  different segments otherwise.
 *
 *  This version of \p inclusive_scan_by_key assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec. 
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param binary_pred  The binary predicate used to determine equality of keys.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's \c value_type, then 
 *                         <tt>binary_op(x,y)</tt> is defined.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan_by_key using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::equal_to<int> binary_pred;
 *
 *  thrust::inclusive_scan_by_key(thrust::host, keys, keys + 10, data, data, binary_pred); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *  \see exclusive_scan_by_key
 *
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator inclusive_scan_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred);


/*! \p inclusive_scan_by_key computes an inclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_scan_by_key uses the binary predicate 
 *  \c pred to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first1, last1)</tt>
 *  belong to the same segment if <tt>binary_pred(*i, *(i+1))</tt> is true, and belong to 
 *  different segments otherwise.
 *
 *  This version of \p inclusive_scan_by_key assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param binary_pred  The binary predicate used to determine equality of keys.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's \c value_type, then 
 *                         <tt>binary_op(x,y)</tt> is defined.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan_by_key
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::equal_to<int> binary_pred;
 *
 *  thrust::inclusive_scan_by_key(keys, keys + 10, data, data, binary_pred); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *  \see exclusive_scan_by_key
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred);


/*! \p inclusive_scan_by_key computes an inclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_scan_by_key uses the binary predicate 
 *  \c pred to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first1, last1)</tt>
 *  belong to the same segment if <tt>binary_pred(*i, *(i+1))</tt> is true, and belong to 
 *  different segments otherwise.
 *
 *  This version of \p inclusive_scan_by_key uses the associative operator 
 *  \c binary_op to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param binary_pred  The binary predicate used to determine equality of keys.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's \c value_type, then 
 *                         <tt>binary_op(x,y)</tt> is defined.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan_by_key using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::equal_to<int> binary_pred;
 *  thrust::plus<int>     binary_op;
 *
 *  thrust::inclusive_scan_by_key(thrust::host, keys, keys + 10, data, data, binary_pred, binary_op); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *  \see exclusive_scan_by_key
 *
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator inclusive_scan_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op);


/*! \p inclusive_scan_by_key computes an inclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_scan_by_key uses the binary predicate 
 *  \c pred to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first1, last1)</tt>
 *  belong to the same segment if <tt>binary_pred(*i, *(i+1))</tt> is true, and belong to 
 *  different segments otherwise.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  This version of \p inclusive_scan_by_key uses the associative operator 
 *  \c binary_op to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 *
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param binary_pred  The binary predicate used to determine equality of keys.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's \c value_type, then 
 *                         <tt>binary_op(x,y)</tt> is defined.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan_by_key
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::equal_to<int> binary_pred;
 *  thrust::plus<int>     binary_op;
 *
 *  thrust::inclusive_scan_by_key(keys, keys + 10, data, data, binary_pred, binary_op); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *  \see exclusive_scan_by_key
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op);


/*! \p exclusive_scan_by_key computes an exclusive segmented prefix 
 *
 *  This version of \p exclusive_scan_by_key uses the value \c 0 to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_scan_by_key assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p exclusive_scan_by_key assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first1, last1</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to 
 *  different segments otherwise.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  Refer to the most general form of \p exclusive_scan_by_key for additional details.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan_by_key using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *  int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *
 *  thrust::exclusive_scan_by_key(thrust::host, key, key + 10, vals, vals); // in-place scan
 *
 *  // vals is now {0, 1, 2, 0, 1, 0, 0, 1, 2, 3};
 *  \endcode
 *
 *  \see exclusive_scan
 *
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator exclusive_scan_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result);


/*! \p exclusive_scan_by_key computes an exclusive segmented prefix 
 *
 *  This version of \p exclusive_scan_by_key uses the value \c 0 to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_scan_by_key assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p exclusive_scan_by_key assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first1, last1</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to 
 *  different segments otherwise.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  Refer to the most general form of \p exclusive_scan_by_key for additional details.
 *
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan_by_key.
 *
 *  \code
 *  #include <thrust/scan.h>
 *  
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *  int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *
 *  thrust::exclusive_scan_by_key(key, key + 10, vals, vals); // in-place scan
 *
 *  // vals is now {0, 1, 2, 0, 1, 0, 0, 1, 2, 3};
 *  \endcode
 *
 *  \see exclusive_scan
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result);


/*! \p exclusive_scan_by_key computes an exclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_scan_by_key uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param init The initial of the exclusive sum value.
 *  \return The end of the output sequence.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan_by_key using the \p
 *  thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *  int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *
 *  int init = 5;
 *
 *  thrust::exclusive_scan_by_key(thrust::host, key, key + 10, vals, vals, init); // in-place scan
 *
 *  // vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *  \see inclusive_scan_by_key
 *
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
__host__ __device__
  OutputIterator exclusive_scan_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init);


/*! \p exclusive_scan_by_key computes an exclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_scan_by_key uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param init The initial of the exclusive sum value.
 *  \return The end of the output sequence.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan_by_key
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *  int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *
 *  int init = 5;
 *
 *  thrust::exclusive_scan_by_key(key, key + 10, vals, vals, init); // in-place scan
 *
 *  // vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *  \see inclusive_scan_by_key
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init);


/*! \p exclusive_scan_by_key computes an exclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_scan_by_key uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_scan_by_key uses the binary predicate \c binary_pred
 *  to compare adjacent keys.  Specifically, consecutive iterators <tt>i</tt> and
 *  <tt>i+1</tt> in the range <tt>[first1, last1)</tt> belong to the same segment if
 *  <tt>binary_pred(*i, *(i+1))</tt> is true, and belong to different segments otherwise.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param init The initial of the exclusive sum value.
 *  \param binary_pred The binary predicate used to determine equality of keys.
 *  \return The end of the output sequence.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan_by_key using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *  int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *
 *  int init = 5;
 *
 *  thrust::equal_to<int> binary_pred;
 *
 *  thrust::exclusive_scan_by_key(thrust::host, key, key + 10, vals, vals, init, binary_pred); // in-place scan
 *
 *  // vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *  \see inclusive_scan_by_key
 *
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator exclusive_scan_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred);


/*! \p exclusive_scan_by_key computes an exclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_scan_by_key uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_scan_by_key uses the binary predicate \c binary_pred
 *  to compare adjacent keys.  Specifically, consecutive iterators <tt>i</tt> and
 *  <tt>i+1</tt> in the range <tt>[first1, last1)</tt> belong to the same segment if
 *  <tt>binary_pred(*i, *(i+1))</tt> is true, and belong to different segments otherwise.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param init The initial of the exclusive sum value.
 *  \param binary_pred The binary predicate used to determine equality of keys.
 *  \return The end of the output sequence.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan_by_key
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *  int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *
 *  int init = 5;
 *
 *  thrust::equal_to<int> binary_pred;
 *
 *  thrust::exclusive_scan_by_key(key, key + 10, vals, vals, init, binary_pred); // in-place scan
 *
 *  // vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *  \see inclusive_scan_by_key
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred);


/*! \p exclusive_scan_by_key computes an exclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_scan_by_key uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_scan_by_key uses the binary predicate \c binary_pred
 *  to compare adjacent keys.  Specifically, consecutive iterators <tt>i</tt> and
 *  <tt>i+1</tt> in the range <tt>[first1, last1)</tt> belong to the same segment if 
 *  <tt>binary_pred(*i, *(i+1))</tt> is true, and belong to different segments otherwise.
 *
 *  This version of \p exclusive_scan_by_key uses the associative operator 
 *  \c binary_op to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param init The initial of the exclusive sum value.
 *  \param binary_pred The binary predicate used to determine equality of keys.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's \c value_type, then 
 *                         <tt>binary_op(x,y)</tt> is defined.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                         and \c AssociativeOperator's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan_by_key using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *  int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *
 *  int init = 5;
 *
 *  thrust::equal_to<int> binary_pred;
 *  thrust::plus<int>     binary_op;
 *
 *  thrust::exclusive_scan_by_key(thrust::host, key, key + 10, vals, vals, init, binary_pred, binary_op); // in-place scan
 *
 *  // vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *  \see inclusive_scan_by_key
 *
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator exclusive_scan_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op);


/*! \p exclusive_scan_by_key computes an exclusive key-value or 'segmented' prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_scan_by_key uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_scan_by_key uses the binary predicate \c binary_pred
 *  to compare adjacent keys.  Specifically, consecutive iterators <tt>i</tt> and
 *  <tt>i+1</tt> in the range <tt>[first1, last1)</tt> belong to the same segment if 
 *  <tt>binary_pred(*i, *(i+1))</tt> is true, and belong to different segments otherwise.
 *
 *  This version of \p exclusive_scan_by_key uses the associative operator 
 *  \c binary_op to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 *
 *  Results are not deterministic for pseudo-associative operators (e.g.,
 *  addition of floating-point types). Results for pseudo-associative
 *  operators may vary from run to run.
 *
 *  \param first1 The beginning of the key sequence.
 *  \param last1 The end of the key sequence.
 *  \param first2 The beginning of the input value sequence.
 *  \param result The beginning of the output value sequence.
 *  \param init The initial of the exclusive sum value.
 *  \param binary_pred The binary predicate used to determine equality of keys.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's \c value_type, then 
 *                         <tt>binary_op(x,y)</tt> is defined.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *  \tparam AssociativeOperator is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                         and \c AssociativeOperator's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first1 may equal \p result but the range <tt>[first1, last1)</tt> and the range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *  \pre \p first2 may equal \p result but the range <tt>[first2, first2 + (last1 - first1)</tt> and range <tt>[result, result + (last1 - first1))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan_by_key
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *  int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *
 *  int init = 5;
 *
 *  thrust::equal_to<int> binary_pred;
 *  thrust::plus<int>     binary_op;
 *
 *  thrust::exclusive_scan_by_key(key, key + 10, vals, vals, init, binary_pred, binary_op); // in-place scan
 *
 *  // vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *  \see inclusive_scan_by_key
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op);


/*! \} // end segmentedprefixsums
 */


/*! \} // end prefix sums
 */

THRUST_NAMESPACE_END

#include <thrust/detail/scan.inl>
