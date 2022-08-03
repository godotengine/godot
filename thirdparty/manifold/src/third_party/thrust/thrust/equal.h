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


/*! \file equal.h
 *  \brief Equality between ranges
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup reductions
 *  \{
 *  \addtogroup comparisons
 *  \ingroup reductions
 *  \{
 */


/*! \p equal returns \c true if the two ranges <tt>[first1, last1)</tt>
 *  and <tt>[first2, first2 + (last1 - first1))</tt> are identical when
 *  compared element-by-element, and otherwise returns \c false.
 *
 *  This version of \p equal returns \c true if and only if for every
 *  iterator \c i in <tt>[first1, last1)</tt>, <tt>*i == *(first2 + (i - first1))</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first sequence.
 *  \param last1  The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \return \c true, if the sequences are equal; \c false, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          and \p InputIterator1's \c value_type can be compared for equality with \c InputIterator2's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          and \p InputIterator2's \c value_type can be compared for equality with \c InputIterator1's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p equal to test
 *  two ranges for equality using the \p thrust::host execution policy:
 *
 *  \code
 *  #include <thrust/equal.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[7] = {3, 1, 4, 1, 5, 9, 3};
 *  int A2[7] = {3, 1, 4, 2, 8, 5, 7};
 *  ...
 *  bool result = thrust::equal(thrust::host, A1, A1 + 7, A2);
 *
 *  // result == false
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/equal
 */
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
__host__ __device__
bool equal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2);


/*! \p equal returns \c true if the two ranges <tt>[first1, last1)</tt>
 *  and <tt>[first2, first2 + (last1 - first1))</tt> are identical when
 *  compared element-by-element, and otherwise returns \c false.
 *
 *  This version of \p equal returns \c true if and only if for every
 *  iterator \c i in <tt>[first1, last1)</tt>, <tt>*i == *(first2 + (i - first1))</tt>.
 *
 *  \param first1 The beginning of the first sequence.
 *  \param last1  The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \return \c true, if the sequences are equal; \c false, otherwise.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          and \p InputIterator1's \c value_type can be compared for equality with \c InputIterator2's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          and \p InputIterator2's \c value_type can be compared for equality with \c InputIterator1's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p equal to test
 *  two ranges for equality.
 *
 *  \code
 *  #include <thrust/equal.h>
 *  ...
 *  int A1[7] = {3, 1, 4, 1, 5, 9, 3};
 *  int A2[7] = {3, 1, 4, 2, 8, 5, 7};
 *  ...
 *  bool result = thrust::equal(A1, A1 + 7, A2);
 *
 *  // result == false
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/equal
 */
template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2);


/*! \p equal returns \c true if the two ranges <tt>[first1, last1)</tt>
 *  and <tt>[first2, first2 + (last1 - first1))</tt> are identical when
 *  compared element-by-element, and otherwise returns \c false.
 *
 *  This version of \p equal returns \c true if and only if for every
 *  iterator \c i in <tt>[first1, last1)</tt>,
 *  <tt>binary_pred(*i, *(first2 + (i - first1)))</tt> is \c true.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first sequence.
 *  \param last1  The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \param binary_pred Binary predicate used to test element equality.
 *  \return \c true, if the sequences are equal; \c false, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator1's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is convertible to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p equal to compare the
 *  elements in two ranges modulo 2 using the \p thrust::host execution policy.
 *
 *  \code
 *  #include <thrust/equal.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  struct compare_modulo_two
 *  {
 *    __host__ __device__
 *    bool operator()(int x, int y) const
 *    {
 *      return (x % 2) == (y % 2);
 *    }
 *  };
 *  ...
 *  int x[6] = {0, 2, 4, 6, 8, 10};
 *  int y[6] = {1, 3, 5, 7, 9, 11};
 *
 *  bool result = thrust::equal(x, x + 6, y, compare_modulo_two());
 *
 *  // result is false
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/equal
 */
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
__host__ __device__
bool equal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate binary_pred);


/*! \p equal returns \c true if the two ranges <tt>[first1, last1)</tt>
 *  and <tt>[first2, first2 + (last1 - first1))</tt> are identical when
 *  compared element-by-element, and otherwise returns \c false.
 *
 *  This version of \p equal returns \c true if and only if for every
 *  iterator \c i in <tt>[first1, last1)</tt>,
 *  <tt>binary_pred(*i, *(first2 + (i - first1)))</tt> is \c true.
 *
 *  \param first1 The beginning of the first sequence.
 *  \param last1  The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \param binary_pred Binary predicate used to test element equality.
 *  \return \c true, if the sequences are equal; \c false, otherwise.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator1's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is convertible to \p BinaryPredicate's \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p equal to compare the
 *  elements in two ranges modulo 2.
 *
 *  \code
 *  #include <thrust/equal.h>
 *  
 *  struct compare_modulo_two
 *  {
 *    __host__ __device__
 *    bool operator()(int x, int y) const
 *    {
 *      return (x % 2) == (y % 2);
 *    }
 *  };
 *  ...
 *  int x[6] = {0, 2, 4, 6, 8, 10};
 *  int y[6] = {1, 3, 5, 7, 9, 11};
 *
 *  bool result = thrust::equal(x, x + 5, y, compare_modulo_two());
 *
 *  // result is true
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/equal
 */
template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred);


/*! \} // end comparisons
 *  \} // end reductions
 */

THRUST_NAMESPACE_END

#include <thrust/detail/equal.inl>
