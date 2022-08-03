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


/*! \file thrust/reduce.h
 *  \brief Functions for reducing a range to a single value
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup reductions
 *  \{
 */


/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \c 0 as the initial value of the
 *  reduction. \p reduce is similar to the C++ Standard Template Library's
 *  <tt>std::accumulate</tt>. The primary difference between the two functions
 *  is that <tt>std::accumulate</tt> guarantees the order of summation, while
 *  \p reduce requires associativity of the binary operation to parallelize
 *  the reduction.
 *
 *  Note that \p reduce also assumes that the binary reduction operator (in this
 *  case operator+) is commutative.  If the reduction operator is not commutative
 *  then \p thrust::reduce should not be used.  Instead, one could use 
 *  \p inclusive_scan (which does not require commutativity) and select the
 *  last element of the output array.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \return The result of the reduction.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and if \c x and \c y are objects of \p InputIterator's \c value_type,
 *          then <tt>x + y</tt> is defined and is convertible to \p InputIterator's
 *          \c value_type. If \c T is \c InputIterator's \c value_type, then
 *          <tt>T(0)</tt> is defined.
 *
 *  The following code snippet demonstrates how to use \p reduce to compute
 *  the sum of a sequence of integers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(thrust::host, data, data + 6);
 *
 *  // result == 9
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/accumulate
 */
template<typename DerivedPolicy, typename InputIterator>
__host__ __device__
  typename thrust::iterator_traits<InputIterator>::value_type
    reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last);


/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \c 0 as the initial value of the
 *  reduction. \p reduce is similar to the C++ Standard Template Library's
 *  <tt>std::accumulate</tt>. The primary difference between the two functions
 *  is that <tt>std::accumulate</tt> guarantees the order of summation, while
 *  \p reduce requires associativity of the binary operation to parallelize
 *  the reduction.
 *
 *  Note that \p reduce also assumes that the binary reduction operator (in this
 *  case operator+) is commutative.  If the reduction operator is not commutative
 *  then \p thrust::reduce should not be used.  Instead, one could use 
 *  \p inclusive_scan (which does not require commutativity) and select the
 *  last element of the output array.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \return The result of the reduction.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and if \c x and \c y are objects of \p InputIterator's \c value_type,
 *          then <tt>x + y</tt> is defined and is convertible to \p InputIterator's
 *          \c value_type. If \c T is \c InputIterator's \c value_type, then
 *          <tt>T(0)</tt> is defined.
 *
 *  The following code snippet demonstrates how to use \p reduce to compute
 *  the sum of a sequence of integers.
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(data, data + 6);
 *
 *  // result == 9
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/accumulate
 */
template<typename InputIterator> typename
  thrust::iterator_traits<InputIterator>::value_type reduce(InputIterator first, InputIterator last);


/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \p init as the initial value of the
 *  reduction. \p reduce is similar to the C++ Standard Template Library's
 *  <tt>std::accumulate</tt>. The primary difference between the two functions
 *  is that <tt>std::accumulate</tt> guarantees the order of summation, while
 *  \p reduce requires associativity of the binary operation to parallelize
 *  the reduction.
 *
 *  Note that \p reduce also assumes that the binary reduction operator (in this
 *  case operator+) is commutative.  If the reduction operator is not commutative
 *  then \p thrust::reduce should not be used.  Instead, one could use 
 *  \p inclusive_scan (which does not require commutativity) and select the
 *  last element of the output array.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param init The initial value.
 *  \return The result of the reduction.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and if \c x and \c y are objects of \p InputIterator's \c value_type,
 *          then <tt>x + y</tt> is defined and is convertible to \p T.
 *  \tparam T is convertible to \p InputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p reduce to compute
 *  the sum of a sequence of integers including an intialization value using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(thrust::host, data, data + 6, 1);
 *
 *  // result == 10
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/accumulate
 */
template<typename DerivedPolicy, typename InputIterator, typename T>
__host__ __device__
  T reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           InputIterator first,
           InputIterator last,
           T init);


/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \p init as the initial value of the
 *  reduction. \p reduce is similar to the C++ Standard Template Library's
 *  <tt>std::accumulate</tt>. The primary difference between the two functions
 *  is that <tt>std::accumulate</tt> guarantees the order of summation, while
 *  \p reduce requires associativity of the binary operation to parallelize
 *  the reduction.
 *
 *  Note that \p reduce also assumes that the binary reduction operator (in this
 *  case operator+) is commutative.  If the reduction operator is not commutative
 *  then \p thrust::reduce should not be used.  Instead, one could use 
 *  \p inclusive_scan (which does not require commutativity) and select the
 *  last element of the output array.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param init The initial value.
 *  \return The result of the reduction.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and if \c x and \c y are objects of \p InputIterator's \c value_type,
 *          then <tt>x + y</tt> is defined and is convertible to \p T.
 *  \tparam T is convertible to \p InputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p reduce to compute
 *  the sum of a sequence of integers including an intialization value.
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(data, data + 6, 1);
 *
 *  // result == 10
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/accumulate
 */
template<typename InputIterator, typename T>
  T reduce(InputIterator first,
           InputIterator last,
           T init);


/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \p init as the initial value of the
 *  reduction and \p binary_op as the binary function used for summation. \p reduce
 *  is similar to the C++ Standard Template Library's <tt>std::accumulate</tt>.
 *  The primary difference between the two functions is that <tt>std::accumulate</tt>
 *  guarantees the order of summation, while \p reduce requires associativity of
 *  \p binary_op to parallelize the reduction.
 *
 *  Note that \p reduce also assumes that the binary reduction operator (in this
 *  case \p binary_op) is commutative.  If the reduction operator is not commutative
 *  then \p thrust::reduce should not be used.  Instead, one could use 
 *  \p inclusive_scan (which does not require commutativity) and select the
 *  last element of the output array.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param init The initial value.
 *  \param binary_op The binary function used to 'sum' values.
 *  \return The result of the reduction.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and \c InputIterator's \c value_type is convertible to \c T.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and is convertible to \p BinaryFunction's \c first_argument_type and \c second_argument_type.
 *  \tparam BinaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>,
 *          and \p BinaryFunction's \c result_type is convertible to \p OutputType.
 *
 *  The following code snippet demonstrates how to use \p reduce to
 *  compute the maximum value of a sequence of integers using the \p thrust::host execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(thrust::host,
 *                              data, data + 6,
 *                              -1,
 *                              thrust::maximum<int>());
 *  // result == 3
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/accumulate
 *  \see transform_reduce
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename T,
         typename BinaryFunction>
__host__ __device__
  T reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           InputIterator first,
           InputIterator last,
           T init,
           BinaryFunction binary_op);


/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \p init as the initial value of the
 *  reduction and \p binary_op as the binary function used for summation. \p reduce
 *  is similar to the C++ Standard Template Library's <tt>std::accumulate</tt>.
 *  The primary difference between the two functions is that <tt>std::accumulate</tt>
 *  guarantees the order of summation, while \p reduce requires associativity of
 *  \p binary_op to parallelize the reduction.
 *
 *  Note that \p reduce also assumes that the binary reduction operator (in this
 *  case \p binary_op) is commutative.  If the reduction operator is not commutative
 *  then \p thrust::reduce should not be used.  Instead, one could use 
 *  \p inclusive_scan (which does not require commutativity) and select the
 *  last element of the output array.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param init The initial value.
 *  \param binary_op The binary function used to 'sum' values.
 *  \return The result of the reduction.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and \c InputIterator's \c value_type is convertible to \c T.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and is convertible to \p BinaryFunction's \c first_argument_type and \c second_argument_type.
 *  \tparam BinaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>,
 *          and \p BinaryFunction's \c result_type is convertible to \p OutputType.
 *
 *  The following code snippet demonstrates how to use \p reduce to
 *  compute the maximum value of a sequence of integers.
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(data, data + 6,
 *                              -1,
 *                              thrust::maximum<int>());
 *  // result == 3
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/accumulate
 *  \see transform_reduce
 */
template<typename InputIterator,
         typename T,
         typename BinaryFunction>
  T reduce(InputIterator first,
           InputIterator last,
           T init,
           BinaryFunction binary_op);


/*! \p reduce_by_key is a generalization of \p reduce to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p reduce_by_key copies the first element of the group to the
 *  \c keys_output. The corresponding values in the range are reduced using the
 *  \c plus and the result copied to \c values_output. 
 *
 *  This version of \p reduce_by_key uses the function object \c equal_to
 *  to test for equality and \c plus to reduce values with equal keys.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_output The beginning of the output key range.
 *  \param values_output The beginning of the output value range.
 *  \return A pair of iterators at end of the ranges <tt>[keys_output, keys_output_last)</tt> and <tt>[values_output, values_output_last)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator1's \c value_type is convertible to \c OutputIterator1's \c value_type.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator2's \c value_type is convertible to \c OutputIterator2's \c value_type.
 *
 *  \pre The input ranges shall not overlap either output range.
 *
 *  The following code snippet demonstrates how to use \p reduce_by_key to
 *  compact a sequence of key/value pairs and sum values with equal keys using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  new_end = thrust::reduce_by_key(thrust::host, A, A + N, B, C, D);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
 *  \endcode
 *  
 *  \see reduce
 *  \see unique_copy
 *  \see unique_by_key
 *  \see unique_by_key_copy
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output);


/*! \p reduce_by_key is a generalization of \p reduce to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p reduce_by_key copies the first element of the group to the
 *  \c keys_output. The corresponding values in the range are reduced using the
 *  \c plus and the result copied to \c values_output. 
 *
 *  This version of \p reduce_by_key uses the function object \c equal_to
 *  to test for equality and \c plus to reduce values with equal keys.
 *
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_output The beginning of the output key range.
 *  \param values_output The beginning of the output value range.
 *  \return A pair of iterators at end of the ranges <tt>[keys_output, keys_output_last)</tt> and <tt>[values_output, values_output_last)</tt>.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator1's \c value_type is convertible to \c OutputIterator1's \c value_type.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator2's \c value_type is convertible to \c OutputIterator2's \c value_type.
 *
 *  \pre The input ranges shall not overlap either output range.
 *
 *  The following code snippet demonstrates how to use \p reduce_by_key to
 *  compact a sequence of key/value pairs and sum values with equal keys.
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  new_end = thrust::reduce_by_key(A, A + N, B, C, D);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
 *  \endcode
 *  
 *  \see reduce
 *  \see unique_copy
 *  \see unique_by_key
 *  \see unique_by_key_copy
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output);


/*! \p reduce_by_key is a generalization of \p reduce to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p reduce_by_key copies the first element of the group to the
 *  \c keys_output. The corresponding values in the range are reduced using the
 *  \c plus and the result copied to \c values_output. 
 *
 *  This version of \p reduce_by_key uses the function object \c binary_pred
 *  to test for equality and \c plus to reduce values with equal keys.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_output The beginning of the output key range.
 *  \param values_output The beginning of the output value range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return A pair of iterators at end of the ranges <tt>[keys_output, keys_output_last)</tt> and <tt>[values_output, values_output_last)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator1's \c value_type is convertible to \c OutputIterator1's \c value_type.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator2's \c value_type is convertible to \c OutputIterator2's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  \pre The input ranges shall not overlap either output range.
 *
 *  The following code snippet demonstrates how to use \p reduce_by_key to
 *  compact a sequence of key/value pairs and sum values with equal keys using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  thrust::equal_to<int> binary_pred;
 *  new_end = thrust::reduce_by_key(thrust::host, A, A + N, B, C, D, binary_pred);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
 *  \endcode
 *  
 *  \see reduce
 *  \see unique_copy
 *  \see unique_by_key
 *  \see unique_by_key_copy
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred);


/*! \p reduce_by_key is a generalization of \p reduce to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p reduce_by_key copies the first element of the group to the
 *  \c keys_output. The corresponding values in the range are reduced using the
 *  \c plus and the result copied to \c values_output. 
 *
 *  This version of \p reduce_by_key uses the function object \c binary_pred
 *  to test for equality and \c plus to reduce values with equal keys.
 *
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_output The beginning of the output key range.
 *  \param values_output The beginning of the output value range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return A pair of iterators at end of the ranges <tt>[keys_output, keys_output_last)</tt> and <tt>[values_output, values_output_last)</tt>.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator1's \c value_type is convertible to \c OutputIterator1's \c value_type.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator2's \c value_type is convertible to \c OutputIterator2's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  \pre The input ranges shall not overlap either output range.
 *
 *  The following code snippet demonstrates how to use \p reduce_by_key to
 *  compact a sequence of key/value pairs and sum values with equal keys.
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  thrust::equal_to<int> binary_pred;
 *  new_end = thrust::reduce_by_key(A, A + N, B, C, D, binary_pred);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
 *  \endcode
 *  
 *  \see reduce
 *  \see unique_copy
 *  \see unique_by_key
 *  \see unique_by_key_copy
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred);


/*! \p reduce_by_key is a generalization of \p reduce to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p reduce_by_key copies the first element of the group to the
 *  \c keys_output. The corresponding values in the range are reduced using the
 *  \c BinaryFunction \c binary_op and the result copied to \c values_output. 
 *  Specifically, if consecutive key iterators \c i and \c (i + 1) are 
 *  such that <tt>binary_pred(*i, *(i+1))</tt> is \c true, then the corresponding
 *  values are reduced to a single value with \c binary_op.
 *
 *  This version of \p reduce_by_key uses the function object \c binary_pred
 *  to test for equality and \c binary_op to reduce values with equal keys.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_output The beginning of the output key range.
 *  \param values_output The beginning of the output value range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \param binary_op The binary function used to accumulate values.
 *  \return A pair of iterators at end of the ranges <tt>[keys_output, keys_output_last)</tt> and <tt>[values_output, values_output_last)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator1's \c value_type is convertible to \c OutputIterator1's \c value_type.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator2's \c value_type is convertible to \c OutputIterator2's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *  \tparam BinaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *          and \c BinaryFunction's \c result_type is convertible to \c OutputIterator2's \c value_type.
 *
 *  \pre The input ranges shall not overlap either output range.
 *
 *  The following code snippet demonstrates how to use \p reduce_by_key to
 *  compact a sequence of key/value pairs and sum values with equal keys using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  thrust::equal_to<int> binary_pred;
 *  thrust::plus<int> binary_op;
 *  new_end = thrust::reduce_by_key(thrust::host, A, A + N, B, C, D, binary_pred, binary_op);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
 *  \endcode
 *  
 *  \see reduce
 *  \see unique_copy
 *  \see unique_by_key
 *  \see unique_by_key_copy
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op);


/*! \p reduce_by_key is a generalization of \p reduce to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p reduce_by_key copies the first element of the group to the
 *  \c keys_output. The corresponding values in the range are reduced using the
 *  \c BinaryFunction \c binary_op and the result copied to \c values_output. 
 *  Specifically, if consecutive key iterators \c i and \c (i + 1) are 
 *  such that <tt>binary_pred(*i, *(i+1))</tt> is \c true, then the corresponding
 *  values are reduced to a single value with \c binary_op.
 *
 *  This version of \p reduce_by_key uses the function object \c binary_pred
 *  to test for equality and \c binary_op to reduce values with equal keys.
 *
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_output The beginning of the output key range.
 *  \param values_output The beginning of the output value range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \param binary_op The binary function used to accumulate values.
 *  \return A pair of iterators at end of the ranges <tt>[keys_output, keys_output_last)</tt> and <tt>[values_output, values_output_last)</tt>.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator1's \c value_type is convertible to \c OutputIterator1's \c value_type.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator2's \c value_type is convertible to \c OutputIterator2's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *  \tparam BinaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *          and \c BinaryFunction's \c result_type is convertible to \c OutputIterator2's \c value_type.
 *
 *  \pre The input ranges shall not overlap either output range.
 *
 *  The following code snippet demonstrates how to use \p reduce_by_key to
 *  compact a sequence of key/value pairs and sum values with equal keys.
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  thrust::equal_to<int> binary_pred;
 *  thrust::plus<int> binary_op;
 *  new_end = thrust::reduce_by_key(A, A + N, B, C, D, binary_pred, binary_op);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
 *  \endcode
 *  
 *  \see reduce
 *  \see unique_copy
 *  \see unique_by_key
 *  \see unique_by_key_copy
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op);


/*! \} // end reductions
 */

THRUST_NAMESPACE_END

#include <thrust/detail/reduce.inl>
