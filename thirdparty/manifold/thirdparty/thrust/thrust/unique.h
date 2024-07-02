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


/*! \file unique.h
 *  \brief Move unique elements to the front of a range
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup stream_compaction
 *  \{
 */


/*! For each group of consecutive elements in the range <tt>[first, last)</tt>
 *  with the same value, \p unique removes all but the first element of 
 *  the group. The return value is an iterator \c new_last such that 
 *  no two consecutive elements in the range <tt>[first, new_last)</tt> are
 *  equal. The iterators in the range <tt>[new_last, last)</tt> are all still
 *  dereferenceable, but the elements that they point to are unspecified.
 *  \p unique is stable, meaning that the relative order of elements that are
 *  not removed is unchanged.
 *
 *  This version of \p unique uses \c operator== to test for equality.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \return The end of the unique range <tt>[first, new_last)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *
 *  The following code snippet demonstrates how to use \p unique to
 *  compact a sequence of numbers to remove consecutive duplicates using the \p thrust::host execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int *new_end = thrust::unique(thrust::host, A, A + N);
 *  // The first four values of A are now {1, 3, 2, 1}
 *  // Values beyond new_end are unspecified.
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/unique
 *  \see unique_copy
 */
template<typename DerivedPolicy,
         typename ForwardIterator>
__host__ __device__
ForwardIterator unique(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       ForwardIterator first,
                       ForwardIterator last);


/*! For each group of consecutive elements in the range <tt>[first, last)</tt>
 *  with the same value, \p unique removes all but the first element of 
 *  the group. The return value is an iterator \c new_last such that 
 *  no two consecutive elements in the range <tt>[first, new_last)</tt> are
 *  equal. The iterators in the range <tt>[new_last, last)</tt> are all still
 *  dereferenceable, but the elements that they point to are unspecified.
 *  \p unique is stable, meaning that the relative order of elements that are
 *  not removed is unchanged.
 *
 *  This version of \p unique uses \c operator== to test for equality.
 *
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \return The end of the unique range <tt>[first, new_last)</tt>.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *
 *  The following code snippet demonstrates how to use \p unique to
 *  compact a sequence of numbers to remove consecutive duplicates.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int *new_end = thrust::unique(A, A + N);
 *  // The first four values of A are now {1, 3, 2, 1}
 *  // Values beyond new_end are unspecified.
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/unique
 *  \see unique_copy
 */
template<typename ForwardIterator>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last);


/*! For each group of consecutive elements in the range <tt>[first, last)</tt>
 *  with the same value, \p unique removes all but the first element of 
 *  the group. The return value is an iterator \c new_last such that 
 *  no two consecutive elements in the range <tt>[first, new_last)</tt> are
 *  equal. The iterators in the range <tt>[new_last, last)</tt> are all still
 *  dereferenceable, but the elements that they point to are unspecified.
 *  \p unique is stable, meaning that the relative order of elements that are
 *  not removed is unchanged.
 *
 *  This version of \p unique uses the function object \p binary_pred to test
 *  for equality.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The end of the unique range <tt>[first, new_last)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type and to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p unique to
 *  compact a sequence of numbers to remove consecutive duplicates using the \p thrust::host execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int *new_end = thrust::unique(thrust::host, A, A + N, thrust::equal_to<int>());
 *  // The first four values of A are now {1, 3, 2, 1}
 *  // Values beyond new_end are unspecified.
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/unique
 *  \see unique_copy
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
ForwardIterator unique(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred);


/*! For each group of consecutive elements in the range <tt>[first, last)</tt>
 *  with the same value, \p unique removes all but the first element of 
 *  the group. The return value is an iterator \c new_last such that 
 *  no two consecutive elements in the range <tt>[first, new_last)</tt> are
 *  equal. The iterators in the range <tt>[new_last, last)</tt> are all still
 *  dereferenceable, but the elements that they point to are unspecified.
 *  \p unique is stable, meaning that the relative order of elements that are
 *  not removed is unchanged.
 *
 *  This version of \p unique uses the function object \p binary_pred to test
 *  for equality.
 *
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The end of the unique range <tt>[first, new_last)</tt>
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type and to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p unique to
 *  compact a sequence of numbers to remove consecutive duplicates.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int *new_end = thrust::unique(A, A + N, thrust::equal_to<int>());
 *  // The first four values of A are now {1, 3, 2, 1}
 *  // Values beyond new_end are unspecified.
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/unique
 *  \see unique_copy
 */
template<typename ForwardIterator,
         typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred);


/*! \p unique_copy copies elements from the range <tt>[first, last)</tt>
 * to a range beginning with \p result, except that in a consecutive group
 * of duplicate elements only the first one is copied. The return value
 * is the end of the range to which the elements are copied. 
 *
 * The reason there are two different versions of unique_copy is that there
 * are two different definitions of what it means for a consecutive group of
 * elements to be duplicates. In the first version, the test is simple
 * equality: the elements in a range <tt>[f, l)</tt> are duplicates if,
 * for every iterator \p i in the range, either <tt>i == f</tt> or else 
 * <tt>*i == *(i-1)</tt>. In the second, the test is an arbitrary 
 * \p BinaryPredicate \p binary_pred: the elements in <tt>[f, l)</tt> are
 * duplicates if, for every iterator \p i in the range, either <tt>i == f</tt>
 * or else <tt>binary_pred(*i, *(i-1))</tt> is \p true.
 *
 * This version of \p unique_copy uses \c operator== to test for equality.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param result The beginning of the output range.
 *  \return The end of the unique range <tt>[result, result_end)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator's \c value_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre The range <tt>[first,last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p unique_copy to
 *  compact a sequence of numbers to remove consecutive duplicates using the \p thrust::host execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int B[N];
 *  int *result_end = thrust::unique_copy(thrust::host, A, A + N, B);
 *  // The first four values of B are now {1, 3, 2, 1} and (result_end - B) is 4
 *  // Values beyond result_end are unspecified
 *  \endcode
 *
 *  \see unique
 *  \see https://en.cppreference.com/w/cpp/algorithm/unique_copy
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
OutputIterator unique_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result);


/*! \p unique_copy copies elements from the range <tt>[first, last)</tt>
 * to a range beginning with \p result, except that in a consecutive group
 * of duplicate elements only the first one is copied. The return value
 * is the end of the range to which the elements are copied. 
 *
 * The reason there are two different versions of unique_copy is that there
 * are two different definitions of what it means for a consecutive group of
 * elements to be duplicates. In the first version, the test is simple
 * equality: the elements in a range <tt>[f, l)</tt> are duplicates if,
 * for every iterator \p i in the range, either <tt>i == f</tt> or else 
 * <tt>*i == *(i-1)</tt>. In the second, the test is an arbitrary 
 * \p BinaryPredicate \p binary_pred: the elements in <tt>[f, l)</tt> are
 * duplicates if, for every iterator \p i in the range, either <tt>i == f</tt>
 * or else <tt>binary_pred(*i, *(i-1))</tt> is \p true.
 *
 * This version of \p unique_copy uses \c operator== to test for equality.
 *
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param result The beginning of the output range.
 *  \return The end of the unique range <tt>[result, result_end)</tt>.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator's \c value_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre The range <tt>[first,last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p unique_copy to
 *  compact a sequence of numbers to remove consecutive duplicates.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int B[N];
 *  int *result_end = thrust::unique_copy(A, A + N, B);
 *  // The first four values of B are now {1, 3, 2, 1} and (result_end - B) is 4
 *  // Values beyond result_end are unspecified
 *  \endcode
 *
 *  \see unique
 *  \see https://en.cppreference.com/w/cpp/algorithm/unique_copy
 */
template<typename InputIterator,
         typename OutputIterator>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator result);


/*! \p unique_copy copies elements from the range <tt>[first, last)</tt>
 * to a range beginning with \p result, except that in a consecutive group
 * of duplicate elements only the first one is copied. The return value
 * is the end of the range to which the elements are copied. 
 *
 * This version of \p unique_copy uses the function object \c binary_pred 
 * to test for equality.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param result The beginning of the output range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The end of the unique range <tt>[result, result_end)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  \pre The range <tt>[first,last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p unique_copy to
 *  compact a sequence of numbers to remove consecutive duplicates using the \p thrust::host execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int B[N];
 *  int *result_end = thrust::unique_copy(thrust::host, A, A + N, B, thrust::equal_to<int>());
 *  // The first four values of B are now {1, 3, 2, 1} and (result_end - B) is 4
 *  // Values beyond result_end are unspecified.
 *  \endcode
 *
 *  \see unique
 *  \see https://en.cppreference.com/w/cpp/algorithm/unique_copy
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
OutputIterator unique_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           BinaryPredicate binary_pred);
                       

/*! \p unique_copy copies elements from the range <tt>[first, last)</tt>
 * to a range beginning with \p result, except that in a consecutive group
 * of duplicate elements only the first one is copied. The return value
 * is the end of the range to which the elements are copied. 
 *
 * This version of \p unique_copy uses the function object \c binary_pred 
 * to test for equality.
 *
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param result The beginning of the output range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The end of the unique range <tt>[result, result_end)</tt>.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and
 *          and \p InputIterator's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  \pre The range <tt>[first,last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p unique_copy to
 *  compact a sequence of numbers to remove consecutive duplicates.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int B[N];
 *  int *result_end = thrust::unique_copy(A, A + N, B, thrust::equal_to<int>());
 *  // The first four values of B are now {1, 3, 2, 1} and (result_end - B) is 4
 *  // Values beyond result_end are unspecified.
 *  \endcode
 *
 *  \see unique
 *  \see https://en.cppreference.com/w/cpp/algorithm/unique_copy
 */
template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           BinaryPredicate binary_pred);


/*! \p unique_by_key is a generalization of \p unique to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p unique_by_key removes all but the first element of 
 *  the group.  Similarly, the corresponding values in the range
 *  <tt>[values_first, values_first + (keys_last - keys_first))</tt> 
 *  are also removed.
 *
 *  The return value is a \p pair of iterators <tt>(new_keys_last,new_values_last)</tt>
 *  such that no two consecutive elements in the range <tt>[keys_first, new_keys_last)</tt>
 *  are equal.
 *
 *  This version of \p unique_by_key uses \c operator== to test for equality and 
 *  \c project1st to reduce values with equal keys.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the key range.
 *  \param keys_last  The end of the key range.
 *  \param values_first The beginning of the value range.
 *  \return A pair of iterators at end of the ranges <tt>[key_first, keys_new_last)</tt> and <tt>[values_first, values_new_last)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator1 is mutable,
 *          and \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator2 is mutable.
 *
 *  \pre The range <tt>[keys_first, keys_last)</tt> and the range <tt>[values_first, values_first + (keys_last - keys_first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p unique_by_key to
 *  compact a sequence of key/value pairs to remove consecutive duplicates using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // values
 *
 *  thrust::pair<int*,int*> new_end;
 *  new_end = thrust::unique_by_key(thrust::host, A, A + N, B);
 *
 *  // The first four keys in A are now {1, 3, 2, 1} and new_end.first - A is 4.
 *  // The first four values in B are now {9, 8, 5, 3} and new_end.second - B is 4.
 *  \endcode
 *
 *  \see unique
 *  \see unique_by_key_copy
 *  \see reduce_by_key
 */
template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first);


/*! \p unique_by_key is a generalization of \p unique to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p unique_by_key removes all but the first element of 
 *  the group.  Similarly, the corresponding values in the range
 *  <tt>[values_first, values_first + (keys_last - keys_first))</tt> 
 *  are also removed.
 *
 *  The return value is a \p pair of iterators <tt>(new_keys_last,new_values_last)</tt>
 *  such that no two consecutive elements in the range <tt>[keys_first, new_keys_last)</tt>
 *  are equal.
 *
 *  This version of \p unique_by_key uses \c operator== to test for equality and 
 *  \c project1st to reduce values with equal keys.
 *
 *  \param keys_first The beginning of the key range.
 *  \param keys_last  The end of the key range.
 *  \param values_first The beginning of the value range.
 *  \return A pair of iterators at end of the ranges <tt>[key_first, keys_new_last)</tt> and <tt>[values_first, values_new_last)</tt>.
 *
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator1 is mutable,
 *          and \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator2 is mutable.
 *
 *  \pre The range <tt>[keys_first, keys_last)</tt> and the range <tt>[values_first, values_first + (keys_last - keys_first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p unique_by_key to
 *  compact a sequence of key/value pairs to remove consecutive duplicates.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // values
 *
 *  thrust::pair<int*,int*> new_end;
 *  new_end = thrust::unique_by_key(A, A + N, B);
 *
 *  // The first four keys in A are now {1, 3, 2, 1} and new_end.first - A is 4.
 *  // The first four values in B are now {9, 8, 5, 3} and new_end.second - B is 4.
 *  \endcode
 *
 *  \see unique
 *  \see unique_by_key_copy
 *  \see reduce_by_key
 */
template<typename ForwardIterator1,
         typename ForwardIterator2>
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first);


/*! \p unique_by_key is a generalization of \p unique to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p unique_by_key removes all but the first element of 
 *  the group.  Similarly, the corresponding values in the range
 *  <tt>[values_first, values_first + (keys_last - keys_first))</tt> 
 *  are also removed.
 *
 *  This version of \p unique_by_key uses the function object \c binary_pred
 *  to test for equality and \c project1st to reduce values with equal keys.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the key range.
 *  \param keys_last  The end of the key range.
 *  \param values_first The beginning of the value range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The end of the unique range <tt>[first, new_last)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator1 is mutable,
 *          and \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator2 is mutable.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  \pre The range <tt>[keys_first, keys_last)</tt> and the range <tt>[values_first, values_first + (keys_last - keys_first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p unique_by_key to
 *  compact a sequence of key/value pairs to remove consecutive duplicates using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // values
 *
 *  thrust::pair<int*,int*> new_end;
 *  thrust::equal_to<int> binary_pred;
 *  new_end = thrust::unique_by_key(thrust::host, keys, keys + N, values, binary_pred);
 *
 *  // The first four keys in A are now {1, 3, 2, 1} and new_end.first - A is 4.
 *  // The first four values in B are now {9, 8, 5, 3} and new_end.second - B is 4.
 *  \endcode
 *
 *  \see unique
 *  \see unique_by_key_copy
 *  \see reduce_by_key
 */
template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                  ForwardIterator1 keys_first, 
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first,
                  BinaryPredicate binary_pred);


/*! \p unique_by_key is a generalization of \p unique to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p unique_by_key removes all but the first element of 
 *  the group.  Similarly, the corresponding values in the range
 *  <tt>[values_first, values_first + (keys_last - keys_first))</tt> 
 *  are also removed.
 *
 *  This version of \p unique_by_key uses the function object \c binary_pred
 *  to test for equality and \c project1st to reduce values with equal keys.
 *
 *  \param keys_first The beginning of the key range.
 *  \param keys_last  The end of the key range.
 *  \param values_first The beginning of the value range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The end of the unique range <tt>[first, new_last)</tt>.
 *
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator1 is mutable,
 *          and \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator2 is mutable.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  \pre The range <tt>[keys_first, keys_last)</tt> and the range <tt>[values_first, values_first + (keys_last - keys_first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p unique_by_key to
 *  compact a sequence of key/value pairs to remove consecutive duplicates.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // values
 *
 *  thrust::pair<int*,int*> new_end;
 *  thrust::equal_to<int> binary_pred;
 *  new_end = thrust::unique_by_key(keys, keys + N, values, binary_pred);
 *
 *  // The first four keys in A are now {1, 3, 2, 1} and new_end.first - A is 4.
 *  // The first four values in B are now {9, 8, 5, 3} and new_end.second - B is 4.
 *  \endcode
 *
 *  \see unique
 *  \see unique_by_key_copy
 *  \see reduce_by_key
 */
template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first,
                BinaryPredicate binary_pred);


/*! \p unique_by_key_copy is a generalization of \p unique_copy to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p unique_by_key_copy copies the first element of the group to
 *  a range beginning with \c keys_result and the corresponding values from the range
 *  <tt>[values_first, values_first + (keys_last - keys_first))</tt> are copied to a range
 *  beginning with \c values_result.
 *
 *  This version of \p unique_by_key_copy uses \c operator== to test for equality and
 *  \c project1st to reduce values with equal keys.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_result The beginning of the output key range.
 *  \param values_result The beginning of the output value range.
 *  \return A pair of iterators at end of the ranges <tt>[keys_result, keys_result_last)</tt> and <tt>[values_result, values_result_last)</tt>.
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
 *  The following code snippet demonstrates how to use \p unique_by_key_copy to
 *  compact a sequence of key/value pairs and with equal keys using the \p thrust::host execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  new_end = thrust::unique_by_key_copy(thrust::host, A, A + N, B, C, D);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 8, 5, 3} and new_end.second - D is 4.
 *  \endcode
 *
 *  \see unique_copy
 *  \see unique_by_key
 *  \see reduce_by_key
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       InputIterator1 keys_first, 
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_result,
                       OutputIterator2 values_result);


/*! \p unique_by_key_copy is a generalization of \p unique_copy to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p unique_by_key_copy copies the first element of the group to
 *  a range beginning with \c keys_result and the corresponding values from the range
 *  <tt>[values_first, values_first + (keys_last - keys_first))</tt> are copied to a range
 *  beginning with \c values_result.
 *
 *  This version of \p unique_by_key_copy uses \c operator== to test for equality and
 *  \c project1st to reduce values with equal keys.
 *
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_result The beginning of the output key range.
 *  \param values_result The beginning of the output value range.
 *  \return A pair of iterators at end of the ranges <tt>[keys_result, keys_result_last)</tt> and <tt>[values_result, values_result_last)</tt>.
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
 *  The following code snippet demonstrates how to use \p unique_by_key_copy to
 *  compact a sequence of key/value pairs and with equal keys.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  new_end = thrust::unique_by_key_copy(A, A + N, B, C, D);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 8, 5, 3} and new_end.second - D is 4.
 *  \endcode
 *
 *  \see unique_copy
 *  \see unique_by_key
 *  \see reduce_by_key
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_result,
                     OutputIterator2 values_result);


/*! \p unique_by_key_copy is a generalization of \p unique_copy to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p unique_by_key_copy copies the first element of the group to
 *  a range beginning with \c keys_result and the corresponding values from the range
 *  <tt>[values_first, values_first + (keys_last - keys_first))</tt> are copied to a range
 *  beginning with \c values_result.
 *
 *  This version of \p unique_by_key_copy uses the function object \c binary_pred
 *  to test for equality and \c project1st to reduce values with equal keys.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_result The beginning of the output key range.
 *  \param values_result The beginning of the output value range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return A pair of iterators at end of the ranges <tt>[keys_result, keys_result_last)</tt> and <tt>[values_result, values_result_last)</tt>.
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
 *  The following code snippet demonstrates how to use \p unique_by_key_copy to
 *  compact a sequence of key/value pairs and with equal keys using the \p thrust::host execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
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
 *  new_end = thrust::unique_by_key_copy(thrust::host, A, A + N, B, C, D, binary_pred);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 8, 5, 3} and new_end.second - D is 4.
 *  \endcode
 *
 *  \see unique_copy
 *  \see unique_by_key
 *  \see reduce_by_key
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       InputIterator1 keys_first, 
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_result,
                       OutputIterator2 values_result,
                       BinaryPredicate binary_pred);


/*! \p unique_by_key_copy is a generalization of \p unique_copy to key-value pairs.
 *  For each group of consecutive keys in the range <tt>[keys_first, keys_last)</tt>
 *  that are equal, \p unique_by_key_copy copies the first element of the group to
 *  a range beginning with \c keys_result and the corresponding values from the range
 *  <tt>[values_first, values_first + (keys_last - keys_first))</tt> are copied to a range
 *  beginning with \c values_result.
 *
 *  This version of \p unique_by_key_copy uses the function object \c binary_pred
 *  to test for equality and \c project1st to reduce values with equal keys.
 *
 *  \param keys_first The beginning of the input key range.
 *  \param keys_last  The end of the input key range.
 *  \param values_first The beginning of the input value range.
 *  \param keys_result The beginning of the output key range.
 *  \param values_result The beginning of the output value range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return A pair of iterators at end of the ranges <tt>[keys_result, keys_result_last)</tt> and <tt>[values_result, values_result_last)</tt>.
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
 *  The following code snippet demonstrates how to use \p unique_by_key_copy to
 *  compact a sequence of key/value pairs and with equal keys.
 *
 *  \code
 *  #include <thrust/unique.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
 *  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
 *  int C[N];                         // output keys
 *  int D[N];                         // output values
 *
 *  thrust::pair<int*,int*> new_end;
 *  thrust::equal_to<int> binary_pred;
 *  new_end = thrust::unique_by_key_copy(A, A + N, B, C, D, binary_pred);
 *
 *  // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
 *  // The first four values in D are now {9, 8, 5, 3} and new_end.second - D is 4.
 *  \endcode
 *
 *  \see unique_copy
 *  \see unique_by_key
 *  \see reduce_by_key
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_result,
                     OutputIterator2 values_result,
                     BinaryPredicate binary_pred);


/*! \p unique_count counts runs of equal elements in the range <tt>[first, last)</tt>
 *  with the same value, 
 *
 *  This version of \p unique_count uses the function object \p binary_pred to test for equality.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The number of runs of equal elements in <tt>[first, new_last)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type and to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p unique_count to
 *  determine a number of runs of equal elements using the \p thrust::host execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int count = thrust::unique_count(thrust::host, A, A + N, thrust::equal_to<int>());
 *  // count is now 4
 *  \endcode
 *
 *  \see unique_copy
 *  \see unique_by_key_copy
 *  \see reduce_by_key_copy
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
  typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 BinaryPredicate binary_pred);


/*! \p unique_count counts runs of equal elements in the range <tt>[first, last)</tt>
 *  with the same value, 
 *
 *  This version of \p unique_count uses \c operator== to test for equality.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The number of runs of equal elements in <tt>[first, new_last)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type and to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p unique_count to
 *  determine the number of runs of equal elements using the \p thrust::host execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int count = thrust::unique_count(thrust::host, A, A + N);
 *  // count is now 4
 *  \endcode
 *
 *  \see unique_copy
 *  \see unique_by_key_copy
 *  \see reduce_by_key_copy
 */
template<typename DerivedPolicy,
         typename ForwardIterator>
__host__ __device__
  typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last);


/*! \p unique_count counts runs of equal elements in the range <tt>[first, last)</tt>
 *  with the same value, 
 *
 *  This version of \p unique_count uses the function object \p binary_pred to test for equality.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The number of runs of equal elements in <tt>[first, new_last)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type and to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p unique_count to
 *  determine the number of runs of equal elements:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int count = thrust::unique_count(A, A + N, thrust::equal_to<int>());
 *  // count is now 4
 *  \endcode
 *
 *  \see unique_copy
 *  \see unique_by_key_copy
 *  \see reduce_by_key_copy
 */
template<typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
  typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(ForwardIterator first,
                 ForwardIterator last,
                 BinaryPredicate binary_pred);


/*! \p unique_count counts runs of equal elements in the range <tt>[first, last)</tt>
 *  with the same value, 
 *
 *  This version of \p unique_count uses \c operator== to test for equality.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input range.
 *  \param last  The end of the input range.
 *  \param binary_pred  The binary predicate used to determine equality.
 *  \return The number of runs of equal elements in <tt>[first, new_last)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type and to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p unique_count to
 *  determine the number of runs of equal elements:
 *
 *  \code
 *  #include <thrust/unique.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 7;
 *  int A[N] = {1, 3, 3, 3, 2, 2, 1};
 *  int count = thrust::unique_count(thrust::host, A, A + N);
 *  // count is now 4
 *  \endcode
 *
 *  \see unique_copy
 *  \see unique_by_key_copy
 *  \see reduce_by_key_copy
 */
template<typename ForwardIterator>
__host__ __device__
  typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(ForwardIterator first,
                 ForwardIterator last);


/*! \} // end stream_compaction
 */

THRUST_NAMESPACE_END

#include <thrust/detail/unique.inl>

