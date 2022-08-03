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


/*! \file thrust/sort.h
 *  \brief Functions for reorganizing ranges into sorted order
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup sorting
 *  \ingroup algorithms
 *  \{
 */


/*! \p sort sorts the elements in <tt>[first, last)</tt> into
 *  ascending order, meaning that if \c i and \c j are any two valid
 *  iterators in <tt>[first, last)</tt> such that \c i precedes \c j,
 *  then \c *j is not less than \c *i. Note: \c sort is not guaranteed
 *  to be stable. That is, suppose that \c *i and \c *j are equivalent:
 *  neither one is less than the other. It is not guaranteed that the
 *  relative order of these two elements will be preserved by \p sort.
 *
 *  This version of \p sort compares objects using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomAccessIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *
 *  The following code snippet demonstrates how to use \p sort to sort
 *  a sequence of integers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sort(thrust::host, A, A + N);
 *  // A is now {1, 2, 4, 5, 7, 8}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p stable_sort
 *  \see \p sort_by_key
 */
template<typename DerivedPolicy, typename RandomAccessIterator>
__host__ __device__
  void sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last);


/*! \p sort sorts the elements in <tt>[first, last)</tt> into
 *  ascending order, meaning that if \c i and \c j are any two valid
 *  iterators in <tt>[first, last)</tt> such that \c i precedes \c j,
 *  then \c *j is not less than \c *i. Note: \c sort is not guaranteed
 *  to be stable. That is, suppose that \c *i and \c *j are equivalent:
 *  neither one is less than the other. It is not guaranteed that the
 *  relative order of these two elements will be preserved by \p sort.
 *
 *  This version of \p sort compares objects using \c operator<.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *
 *  \tparam RandomAccessIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *
 *  The following code snippet demonstrates how to use \p sort to sort
 *  a sequence of integers.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sort(A, A + N);
 *  // A is now {1, 2, 4, 5, 7, 8}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p stable_sort
 *  \see \p sort_by_key
 */
template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last);


/*! \p sort sorts the elements in <tt>[first, last)</tt> into
 *  ascending order, meaning that if \c i and \c j are any two valid
 *  iterators in <tt>[first, last)</tt> such that \c i precedes \c j,
 *  then \c *j is not less than \c *i. Note: \c sort is not guaranteed
 *  to be stable. That is, suppose that \c *i and \c *j are equivalent:
 *  neither one is less than the other. It is not guaranteed that the
 *  relative order of these two elements will be preserved by \p sort.
 *
 *  This version of \p sort compares objects using a function object
 *  \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param comp  Comparison operator.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomAccessIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code demonstrates how to sort integers in descending order
 *  using the greater<int> comparison operator using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sort(thrust::host, A, A + N, thrust::greater<int>());
 *  // A is now {8, 7, 5, 4, 2, 1};
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p stable_sort
 *  \see \p sort_by_key
 */
template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp);


/*! \p sort sorts the elements in <tt>[first, last)</tt> into
 *  ascending order, meaning that if \c i and \c j are any two valid
 *  iterators in <tt>[first, last)</tt> such that \c i precedes \c j,
 *  then \c *j is not less than \c *i. Note: \c sort is not guaranteed
 *  to be stable. That is, suppose that \c *i and \c *j are equivalent:
 *  neither one is less than the other. It is not guaranteed that the
 *  relative order of these two elements will be preserved by \p sort.
 *
 *  This version of \p sort compares objects using a function object
 *  \p comp.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param comp  Comparison operator.
 *
 *  \tparam RandomAccessIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code demonstrates how to sort integers in descending order
 *  using the greater<int> comparison operator.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/functional.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sort(A, A + N, thrust::greater<int>());
 *  // A is now {8, 7, 5, 4, 2, 1};
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p stable_sort
 *  \see \p sort_by_key
 */
template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp);


/*! \p stable_sort is much like \c sort: it sorts the elements in
 *  <tt>[first, last)</tt> into ascending order, meaning that if \c i
 *  and \c j are any two valid iterators in <tt>[first, last)</tt> such
 *  that \c i precedes \c j, then \c *j is not less than \c *i.
 *
 *  As the name suggests, \p stable_sort is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[first, last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort compares objects using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomAccessIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *
 *  The following code snippet demonstrates how to use \p sort to sort
 *  a sequence of integers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::stable_sort(thrust::host, A, A + N);
 *  // A is now {1, 2, 4, 5, 7, 8}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/stable_sort
 *  \see \p sort
 *  \see \p stable_sort_by_key
 */
template<typename DerivedPolicy, typename RandomAccessIterator>
__host__ __device__
  void stable_sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last);


/*! \p stable_sort is much like \c sort: it sorts the elements in
 *  <tt>[first, last)</tt> into ascending order, meaning that if \c i
 *  and \c j are any two valid iterators in <tt>[first, last)</tt> such
 *  that \c i precedes \c j, then \c *j is not less than \c *i.
 *
 *  As the name suggests, \p stable_sort is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[first, last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort compares objects using \c operator<.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *
 *  \tparam RandomAccessIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *
 *  The following code snippet demonstrates how to use \p sort to sort
 *  a sequence of integers.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::stable_sort(A, A + N);
 *  // A is now {1, 2, 4, 5, 7, 8}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/stable_sort
 *  \see \p sort
 *  \see \p stable_sort_by_key
 */
template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last);


/*! \p stable_sort is much like \c sort: it sorts the elements in
 *  <tt>[first, last)</tt> into ascending order, meaning that if \c i
 *  and \c j are any two valid iterators in <tt>[first, last)</tt> such
 *  that \c i precedes \c j, then \c *j is not less than \c *i.
 *
 *  As the name suggests, \p stable_sort is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[first, last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort compares objects using a function object
 *  \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomAccessIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code demonstrates how to sort integers in descending order
 *  using the greater<int> comparison operator using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sort(A, A + N, thrust::greater<int>());
 *  // A is now {8, 7, 5, 4, 2, 1};
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/stable_sort
 *  \see \p sort
 *  \see \p stable_sort_by_key
 */
template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void stable_sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp);


/*! \p stable_sort is much like \c sort: it sorts the elements in
 *  <tt>[first, last)</tt> into ascending order, meaning that if \c i
 *  and \c j are any two valid iterators in <tt>[first, last)</tt> such
 *  that \c i precedes \c j, then \c *j is not less than \c *i.
 *
 *  As the name suggests, \p stable_sort is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[first, last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort compares objects using a function object
 *  \p comp.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam RandomAccessIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator is mutable,
 *          and \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code demonstrates how to sort integers in descending order
 *  using the greater<int> comparison operator.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/functional.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  thrust::sort(A, A + N, thrust::greater<int>());
 *  // A is now {8, 7, 5, 4, 2, 1};
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/stable_sort
 *  \see \p sort
 *  \see \p stable_sort_by_key
 */
template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp);


///////////////
// Key Value //
///////////////


/*! \p sort_by_key performs a key-value sort. That is, \p sort_by_key sorts the
 *  elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  Note: \c sort_by_key is not guaranteed to be stable. That is, suppose that
 *  \c *i and \c *j are equivalent: neither one is less than the other. It is not
 *  guaranteed that the relative order of these two keys or the relative
 *  order of their corresponding values will be preserved by \p sort_by_key.
 *
 *  This version of \p sort_by_key compares key objects using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomAccessIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator1's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *  \tparam RandomAccessIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *
 *  \pre The range <tt>[keys_first, keys_last))</tt> shall not overlap the range <tt>[values_first, values_first + (keys_last - keys_first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p sort_by_key to sort
 *  an array of character values using integers as sorting keys using the \p thrust::host execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::sort_by_key(thrust::host, keys, keys + N, values);
 *  // keys is now   {  1,   2,   4,   5,   7,   8}
 *  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p stable_sort_by_key
 *  \see \p sort
 */
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
  void sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first);


/*! \p sort_by_key performs a key-value sort. That is, \p sort_by_key sorts the
 *  elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  Note: \c sort_by_key is not guaranteed to be stable. That is, suppose that
 *  \c *i and \c *j are equivalent: neither one is less than the other. It is not
 *  guaranteed that the relative order of these two keys or the relative
 *  order of their corresponding values will be preserved by \p sort_by_key.
 *
 *  This version of \p sort_by_key compares key objects using \c operator<.
 *
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *
 *  \tparam RandomAccessIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator1's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *  \tparam RandomAccessIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *
 *  \pre The range <tt>[keys_first, keys_last))</tt> shall not overlap the range <tt>[values_first, values_first + (keys_last - keys_first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p sort_by_key to sort
 *  an array of character values using integers as sorting keys.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::sort_by_key(keys, keys + N, values);
 *  // keys is now   {  1,   2,   4,   5,   7,   8}
 *  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p stable_sort_by_key
 *  \see \p sort
 */
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first);


/*! \p sort_by_key performs a key-value sort. That is, \p sort_by_key sorts the
 *  elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  Note: \c sort_by_key is not guaranteed to be stable. That is, suppose that
 *  \c *i and \c *j are equivalent: neither one is less than the other. It is not
 *  guaranteed that the relative order of these two keys or the relative
 *  order of their corresponding values will be preserved by \p sort_by_key.
 *
 *  This version of \p sort_by_key compares key objects using a function object
 *  \c comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomAccessIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam RandomAccessIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The range <tt>[keys_first, keys_last))</tt> shall not overlap the range <tt>[values_first, values_first + (keys_last - keys_first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p sort_by_key to sort
 *  an array of character values using integers as sorting keys using the \p thrust::host execution policy
 *  for parallelization.The keys are sorted in descending order using the <tt>greater<int></tt> comparison operator.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::sort_by_key(thrust::host, keys, keys + N, values, thrust::greater<int>());
 *  // keys is now   {  8,   7,   5,   4,   2,   1}
 *  // values is now {'d', 'f', 'e', 'b', 'c', 'a'}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p stable_sort_by_key
 *  \see \p sort
 */
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
  void sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp);


/*! \p sort_by_key performs a key-value sort. That is, \p sort_by_key sorts the
 *  elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  Note: \c sort_by_key is not guaranteed to be stable. That is, suppose that
 *  \c *i and \c *j are equivalent: neither one is less than the other. It is not
 *  guaranteed that the relative order of these two keys or the relative
 *  order of their corresponding values will be preserved by \p sort_by_key.
 *
 *  This version of \p sort_by_key compares key objects using a function object
 *  \c comp.
 *
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam RandomAccessIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam RandomAccessIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The range <tt>[keys_first, keys_last))</tt> shall not overlap the range <tt>[values_first, values_first + (keys_last - keys_first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p sort_by_key to sort
 *  an array of character values using integers as sorting keys.  The keys
 *  are sorted in descending order using the greater<int> comparison operator.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::sort_by_key(keys, keys + N, values, thrust::greater<int>());
 *  // keys is now   {  8,   7,   5,   4,   2,   1}
 *  // values is now {'d', 'f', 'e', 'b', 'c', 'a'}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p stable_sort_by_key
 *  \see \p sort
 */
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp);


/*! \p stable_sort_by_key performs a key-value sort. That is, \p stable_sort_by_key
 *  sorts the elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  As the name suggests, \p stable_sort_by_key is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[keys_first, keys_last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort_by_key is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort_by_key compares key objects using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomAccessIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator1's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *  \tparam RandomAccessIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *
 *  \pre The range <tt>[keys_first, keys_last))</tt> shall not overlap the range <tt>[values_first, values_first + (keys_last - keys_first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p stable_sort_by_key to sort
 *  an array of characters using integers as sorting keys using the \p thrust::host execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::stable_sort_by_key(thrust::host, keys, keys + N, values);
 *  // keys is now   {  1,   2,   4,   5,   7,   8}
 *  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p sort_by_key
 *  \see \p stable_sort
 */
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
  void stable_sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first);


/*! \p stable_sort_by_key performs a key-value sort. That is, \p stable_sort_by_key
 *  sorts the elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  As the name suggests, \p stable_sort_by_key is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[keys_first, keys_last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort_by_key is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort_by_key compares key objects using \c operator<.
 *
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *
 *  \tparam RandomAccessIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering relation on \p RandomAccessIterator1's \c value_type is a <em>strict weak ordering</em>, as defined in the
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *  \tparam RandomAccessIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *
 *  \pre The range <tt>[keys_first, keys_last))</tt> shall not overlap the range <tt>[values_first, values_first + (keys_last - keys_first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p stable_sort_by_key to sort
 *  an array of characters using integers as sorting keys.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::stable_sort_by_key(keys, keys + N, values);
 *  // keys is now   {  1,   2,   4,   5,   7,   8}
 *  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p sort_by_key
 *  \see \p stable_sort
 */
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first);


/*! \p stable_sort_by_key performs a key-value sort. That is, \p stable_sort_by_key
 *  sorts the elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  As the name suggests, \p stable_sort_by_key is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[keys_first, keys_last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort_by_key is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort_by_key compares key objects using the function
 *  object \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomAccessIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam RandomAccessIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The range <tt>[keys_first, keys_last))</tt> shall not overlap the range <tt>[values_first, values_first + (keys_last - keys_first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p sort_by_key to sort
 *  an array of character values using integers as sorting keys using the \p thrust::host execution policy for
 *  parallelization. The keys are sorted in descending order using the <tt>greater<int></tt> comparison operator.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::stable_sort_by_key(thrust::host, keys, keys + N, values, thrust::greater<int>());
 *  // keys is now   {  8,   7,   5,   4,   2,   1}
 *  // values is now {'d', 'f', 'e', 'b', 'c', 'a'}
 *  \endcode
 *
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p sort_by_key
 *  \see \p stable_sort
 */
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
  void stable_sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp);


/*! \p stable_sort_by_key performs a key-value sort. That is, \p stable_sort_by_key
 *  sorts the elements in <tt>[keys_first, keys_last)</tt> and <tt>[values_first,
 *  values_first + (keys_last - keys_first))</tt> into ascending key order,
 *  meaning that if \c i and \c j are any two valid iterators in <tt>[keys_first,
 *  keys_last)</tt> such that \c i precedes \c j, and \c p and \c q are iterators
 *  in <tt>[values_first, values_first + (keys_last - keys_first))</tt>
 *  corresponding to \c i and \c j respectively, then \c *j is not less than
 *  \c *i.
 *
 *  As the name suggests, \p stable_sort_by_key is stable: it preserves the
 *  relative ordering of equivalent elements. That is, if \c x and \c y
 *  are elements in <tt>[keys_first, keys_last)</tt> such that \c x precedes \c y,
 *  and if the two elements are equivalent (neither <tt>x < y</tt> nor
 *  <tt>y < x</tt>) then a postcondition of \p stable_sort_by_key is that \c x
 *  still precedes \c y.
 *
 *  This version of \p stable_sort_by_key compares key objects using the function
 *  object \p comp.
 *
 *  \param keys_first The beginning of the key sequence.
 *  \param keys_last The end of the key sequence.
 *  \param values_first The beginning of the value sequence.
 *  \param comp Comparison operator.
 *
 *  \tparam RandomAccessIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>,
 *          \p RandomAccessIterator1 is mutable,
 *          and \p RandomAccessIterator1's \c value_type is convertible to \p StrictWeakOrdering's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam RandomAccessIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>,
 *          and \p RandomAccessIterator2 is mutable.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The range <tt>[keys_first, keys_last))</tt> shall not overlap the range <tt>[values_first, values_first + (keys_last - keys_first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p sort_by_key to sort
 *  an array of character values using integers as sorting keys.  The keys
 *  are sorted in descending order using the greater<int> comparison operator.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  ...
 *  const int N = 6;
 *  int    keys[N] = {  1,   4,   2,   8,   5,   7};
 *  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
 *  thrust::stable_sort_by_key(keys, keys + N, values, thrust::greater<int>());
 *  // keys is now   {  8,   7,   5,   4,   2,   1}
 *  // values is now {'d', 'f', 'e', 'b', 'c', 'a'}
 *  \endcode
 *
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/sort
 *  \see \p sort_by_key
 *  \see \p stable_sort
 */
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp);


/*! \} // end sorting
 */


/*! \addtogroup reductions
 *  \{
 *  \addtogroup predicates
 *  \{
 */


/*! \p is_sorted returns \c true if the range <tt>[first, last)</tt> is
 *  sorted in ascending order, and \c false otherwise.
 *
 *  Specifically, this version of \p is_sorted returns \c false if for
 *  some iterator \c i in the range <tt>[first, last - 1)</tt> the
 *  expression <tt>*(i + 1) < *i</tt> is \c true.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return \c true, if the sequence is sorted; \c false, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering on objects of \p ForwardIterator's \c value_type is a <em>strict weak ordering</em>, as defined
 *          in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *
 *
 *  The following code demonstrates how to use \p is_sorted to test whether the
 *  contents of a \c device_vector are stored in ascending order using the \p thrust::device execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/sort.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> v(6);
 *  v[0] = 1;
 *  v[1] = 4;
 *  v[2] = 2;
 *  v[3] = 8;
 *  v[4] = 5;
 *  v[5] = 7;
 *
 *  bool result = thrust::is_sorted(thrust::device, v.begin(), v.end());
 *
 *  // result == false
 *
 *  thrust::sort(v.begin(), v.end());
 *  result = thrust::is_sorted(thrust::device, v.begin(), v.end());
 *
 *  // result == true
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/is_sorted
 *  \see is_sorted_until
 *  \see \c sort
 *  \see \c stable_sort
 *  \see \c less<T>
 */
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  bool is_sorted(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last);


/*! \p is_sorted returns \c true if the range <tt>[first, last)</tt> is
 *  sorted in ascending order, and \c false otherwise.
 *
 *  Specifically, this version of \p is_sorted returns \c false if for
 *  some iterator \c i in the range <tt>[first, last - 1)</tt> the
 *  expression <tt>*(i + 1) < *i</tt> is \c true.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return \c true, if the sequence is sorted; \c false, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          and the ordering on objects of \p ForwardIterator's \c value_type is a <em>strict weak ordering</em>, as defined
 *          in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.
 *
 *
 *  The following code demonstrates how to use \p is_sorted to test whether the
 *  contents of a \c device_vector are stored in ascending order.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/sort.h>
 *  ...
 *  thrust::device_vector<int> v(6);
 *  v[0] = 1;
 *  v[1] = 4;
 *  v[2] = 2;
 *  v[3] = 8;
 *  v[4] = 5;
 *  v[5] = 7;
 *
 *  bool result = thrust::is_sorted(v.begin(), v.end());
 *
 *  // result == false
 *
 *  thrust::sort(v.begin(), v.end());
 *  result = thrust::is_sorted(v.begin(), v.end());
 *
 *  // result == true
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/is_sorted
 *  \see is_sorted_until
 *  \see \c sort
 *  \see \c stable_sort
 *  \see \c less<T>
 */
template<typename ForwardIterator>
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last);


/*! \p is_sorted returns \c true if the range <tt>[first, last)</tt> is sorted in ascending 
 *  order accoring to a user-defined comparison operation, and \c false otherwise.
 *
 *  Specifically, this version of \p is_sorted returns \c false if for some iterator \c i in
 *  the range <tt>[first, last - 1)</tt> the expression <tt>comp(*(i + 1), *i)</tt> is \c true.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp  Comparison operator.
 *  \return \c true, if the sequence is sorted according to comp; \c false, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \c StrictWeakOrdering's \c first_argument_type
 *          and \c second_argument_type.
 *  \tparam Compare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p is_sorted to test whether the
 *  contents of a \c device_vector are stored in descending order using the \p thrust::device execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> v(6);
 *  v[0] = 1;
 *  v[1] = 4;
 *  v[2] = 2;
 *  v[3] = 8;
 *  v[4] = 5;
 *  v[5] = 7;
 *
 *  thrust::greater<int> comp;
 *  bool result = thrust::is_sorted(thrust::device, v.begin(), v.end(), comp);
 *
 *  // result == false
 *
 *  thrust::sort(v.begin(), v.end(), comp);
 *  result = thrust::is_sorted(thrust::device, v.begin(), v.end(), comp);
 *
 *  // result == true
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/is_sorted
 *  \see \c sort
 *  \see \c stable_sort
 *  \see \c less<T>
 */
template<typename DerivedPolicy, typename ForwardIterator, typename Compare>
__host__ __device__
  bool is_sorted(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 Compare comp);


/*! \p is_sorted returns \c true if the range <tt>[first, last)</tt> is sorted in ascending 
 *  order accoring to a user-defined comparison operation, and \c false otherwise.
 *
 *  Specifically, this version of \p is_sorted returns \c false if for some iterator \c i in
 *  the range <tt>[first, last - 1)</tt> the expression <tt>comp(*(i + 1), *i)</tt> is \c true.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp  Comparison operator.
 *  \return \c true, if the sequence is sorted according to comp; \c false, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \c StrictWeakOrdering's \c first_argument_type
 *          and \c second_argument_type.
 *  \tparam Compare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p is_sorted to test whether the
 *  contents of a \c device_vector are stored in descending order.
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> v(6);
 *  v[0] = 1;
 *  v[1] = 4;
 *  v[2] = 2;
 *  v[3] = 8;
 *  v[4] = 5;
 *  v[5] = 7;
 *
 *  thrust::greater<int> comp;
 *  bool result = thrust::is_sorted(v.begin(), v.end(), comp);
 *
 *  // result == false
 *
 *  thrust::sort(v.begin(), v.end(), comp);
 *  result = thrust::is_sorted(v.begin(), v.end(), comp);
 *
 *  // result == true
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/is_sorted
 *  \see \c sort
 *  \see \c stable_sort
 *  \see \c less<T>
 */
template<typename ForwardIterator, typename Compare>
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last,
                 Compare comp);


/*! This version of \p is_sorted_until returns the last iterator \c i in <tt>[first,last]</tt> for
 *  which the range <tt>[first,last)</tt> is sorted using \c operator<. If <tt>distance(first,last) < 2</tt>,
 *  \p is_sorted_until simply returns \p last.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \return The last iterator in the input range for which it is sorted.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and
 *          \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  The following code snippet demonstrates how to use \p is_sorted_until to find the first position
 *  in an array where the data becomes unsorted using the \p thrust::host execution policy for
 *  parallelization:
 *  
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/execution_policy.h>
 *
 *  ...
 *   
 *  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
 *  
 *  int * B = thrust::is_sorted_until(thrust::host, A, A + 8);
 *  
 *  // B - A is 4
 *  // [A, B) is sorted
 *  \endcode
 *
 *  \see \p is_sorted
 *  \see \p sort
 *  \see \p sort_by_key
 *  \see \p stable_sort
 *  \see \p stable_sort_by_key
 */
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator is_sorted_until(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last);


/*! This version of \p is_sorted_until returns the last iterator \c i in <tt>[first,last]</tt> for
 *  which the range <tt>[first,last)</tt> is sorted using \c operator<. If <tt>distance(first,last) < 2</tt>,
 *  \p is_sorted_until simply returns \p last.
 *
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \return The last iterator in the input range for which it is sorted.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and
 *          \p ForwardIterator's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  The following code snippet demonstrates how to use \p is_sorted_until to find the first position
 *  in an array where the data becomes unsorted:
 *  
 *  \code
 *  #include <thrust/sort.h>
 *
 *  ...
 *   
 *  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
 *  
 *  int * B = thrust::is_sorted_until(A, A + 8);
 *  
 *  // B - A is 4
 *  // [A, B) is sorted
 *  \endcode
 *
 *  \see \p is_sorted
 *  \see \p sort
 *  \see \p sort_by_key
 *  \see \p stable_sort
 *  \see \p stable_sort_by_key
 */
template<typename ForwardIterator>
  ForwardIterator is_sorted_until(ForwardIterator first,
                                  ForwardIterator last);


/*! This version of \p is_sorted_until returns the last iterator \c i in <tt>[first,last]</tt> for
 *  which the range <tt>[first,last)</tt> is sorted using the function object \c comp. If <tt>distance(first,last) < 2</tt>,
 *  \p is_sorted_until simply returns \p last.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization:
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param comp The function object to use for comparison.
 *  \return The last iterator in the input range for which it is sorted.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and
 *          \p ForwardIterator's \c value_type is convertible to \p Compare's \c argument_type.
 *  \tparam Compare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p is_sorted_until to find the first position
 *  in an array where the data becomes unsorted in descending order using the \p thrust::host execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *
 *  ...
 *   
 *  int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};
 *  
 *  thrust::greater<int> comp;
 *  int * B = thrust::is_sorted_until(thrust::host, A, A + 8, comp);
 *  
 *  // B - A is 4
 *  // [A, B) is sorted in descending order
 *  \endcode
 *
 *  \see \p is_sorted
 *  \see \p sort
 *  \see \p sort_by_key
 *  \see \p stable_sort
 *  \see \p stable_sort_by_key
 */
template<typename DerivedPolicy, typename ForwardIterator, typename Compare>
__host__ __device__
  ForwardIterator is_sorted_until(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp);


/*! This version of \p is_sorted_until returns the last iterator \c i in <tt>[first,last]</tt> for
 *  which the range <tt>[first,last)</tt> is sorted using the function object \c comp. If <tt>distance(first,last) < 2</tt>,
 *  \p is_sorted_until simply returns \p last.
 *
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param comp The function object to use for comparison.
 *  \return The last iterator in the input range for which it is sorted.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and
 *          \p ForwardIterator's \c value_type is convertible to \p Compare's \c argument_type.
 *  \tparam Compare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p is_sorted_until to find the first position
 *  in an array where the data becomes unsorted in descending order:
 *
 *  \code
 *  #include <thrust/sort.h>
 *  #include <thrust/functional.h>
 *
 *  ...
 *   
 *  int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};
 *  
 *  thrust::greater<int> comp;
 *  int * B = thrust::is_sorted_until(A, A + 8, comp);
 *  
 *  // B - A is 4
 *  // [A, B) is sorted in descending order
 *  \endcode
 *
 *  \see \p is_sorted
 *  \see \p sort
 *  \see \p sort_by_key
 *  \see \p stable_sort
 *  \see \p stable_sort_by_key
 */
template<typename ForwardIterator, typename Compare>
  ForwardIterator is_sorted_until(ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp);


/*! \} // end predicates
 *  \} // end reductions
 */

THRUST_NAMESPACE_END

#include <thrust/detail/sort.inl>
