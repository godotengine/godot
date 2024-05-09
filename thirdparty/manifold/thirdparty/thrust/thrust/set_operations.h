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


/*! \file set_operations.h
 *  \brief Set theoretic operations for sorted ranges
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup set_operations Set Operations
 *  \ingroup algorithms
 *  \{
 */


/*! \p set_difference constructs a sorted range that is the set difference of the sorted
 *  ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_difference performs the "difference" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt> and not contained in <tt>[first2, last1)</tt>. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[first1, last1)</tt> range shall be copied to the output range.
 *
 *  This version of \p set_difference compares elements using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_difference to compute the
 *  set difference of two sets of integers sorted in ascending order using the \p thrust::host execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[7] = {0, 1, 3, 4, 5, 6, 9};
 *  int A2[5] = {1, 3, 5, 7, 9};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_difference(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
 *  // result is now {0, 4, 6}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_difference
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator set_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator1                                              first1,
                                InputIterator1                                              last1,
                                InputIterator2                                              first2,
                                InputIterator2                                              last2,
                                OutputIterator                                              result);


/*! \p set_difference constructs a sorted range that is the set difference of the sorted
 *  ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_difference performs the "difference" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt> and not contained in <tt>[first2, last1)</tt>. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[first1, last1)</tt> range shall be copied to the output range.
 *
 *  This version of \p set_difference compares elements using \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_difference to compute the
 *  set difference of two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[7] = {0, 1, 3, 4, 5, 6, 9};
 *  int A2[5] = {1, 3, 5, 7, 9};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_difference(A1, A1 + 7, A2, A2 + 5, result);
 *  // result is now {0, 4, 6}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_difference
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result);


/*! \p set_difference constructs a sorted range that is the set difference of the sorted
 *  ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_difference performs the "difference" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt> and not contained in <tt>[first2, last1)</tt>. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[first1, last1)</tt> range shall be copied to the output range.
 *
 *  This version of \p set_difference compares elements using a function object \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertable to \p StrictWeakCompare's \c first_argument_type.
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2's \c value_type is convertable to \p StrictWeakCompare's \c second_argument_type.
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_difference to compute the
 *  set difference of two sets of integers sorted in descending order using the \p thrust::host execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[7] = {9, 6, 5, 4, 3, 1, 0};
 *  int A2[5] = {9, 7, 5, 3, 1};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_difference(thrust::host, A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
 *  // result is now {6, 4, 0}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_difference
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
  OutputIterator set_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator1                                              first1,
                                InputIterator1                                              last1,
                                InputIterator2                                              first2,
                                InputIterator2                                              last2,
                                OutputIterator                                              result,
                                StrictWeakCompare                                           comp);


/*! \p set_difference constructs a sorted range that is the set difference of the sorted
 *  ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_difference performs the "difference" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt> and not contained in <tt>[first2, last1)</tt>. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[first1, last1)</tt> range shall be copied to the output range.
 *
 *  This version of \p set_difference compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertable to \p StrictWeakCompare's \c first_argument_type.
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2's \c value_type is convertable to \p StrictWeakCompare's \c second_argument_type.
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_difference to compute the
 *  set difference of two sets of integers sorted in descending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A1[7] = {9, 6, 5, 4, 3, 1, 0};
 *  int A2[5] = {9, 7, 5, 3, 1};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_difference(A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
 *  // result is now {6, 4, 0}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_difference
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result,
                                StrictWeakCompare comp);


/*! \p set_intersection constructs a sorted range that is the
 *  intersection of sorted ranges <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt>. The return value is the end of the
 *  output range.
 *
 *  In the simplest case, \p set_intersection performs the
 *  "intersection" operation from set theory: the output range
 *  contains a copy of every element that is contained in both
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The
 *  general case is more complicated, because the input ranges may
 *  contain duplicate elements. The generalization is that if a value
 *  appears \c m times in <tt>[first1, last1)</tt> and \c n times in
 *  <tt>[first2, last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the output range.
 *  \p set_intersection is stable, meaning that both elements are
 *  copied from the first range rather than the second, and that the
 *  relative order of elements in the output range is the same as in
 *  the first input range.
 *
 *  This version of \p set_intersection compares objects using
 *  \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_intersection to compute the
 *  set intersection of two sets of integers sorted in ascending order using the \p thrust::host execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[6] = {1, 3, 5, 7, 9, 11};
 *  int A2[7] = {1, 1, 2, 3, 5,  8, 13};
 *
 *  int result[7];
 *
 *  int *result_end = thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result);
 *  // result is now {1, 3, 5}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_intersection
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator set_intersection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  InputIterator1                                              first1,
                                  InputIterator1                                              last1,
                                  InputIterator2                                              first2,
                                  InputIterator2                                              last2,
                                  OutputIterator                                              result);


/*! \p set_intersection constructs a sorted range that is the
 *  intersection of sorted ranges <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt>. The return value is the end of the
 *  output range.
 *
 *  In the simplest case, \p set_intersection performs the
 *  "intersection" operation from set theory: the output range
 *  contains a copy of every element that is contained in both
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The
 *  general case is more complicated, because the input ranges may
 *  contain duplicate elements. The generalization is that if a value
 *  appears \c m times in <tt>[first1, last1)</tt> and \c n times in
 *  <tt>[first2, last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the output range.
 *  \p set_intersection is stable, meaning that both elements are
 *  copied from the first range rather than the second, and that the
 *  relative order of elements in the output range is the same as in
 *  the first input range.
 *
 *  This version of \p set_intersection compares objects using
 *  \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_intersection to compute the
 *  set intersection of two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[6] = {1, 3, 5, 7, 9, 11};
 *  int A2[7] = {1, 1, 2, 3, 5,  8, 13};
 *
 *  int result[7];
 *
 *  int *result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result);
 *  // result is now {1, 3, 5}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_intersection
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result);


/*! \p set_intersection constructs a sorted range that is the
 *  intersection of sorted ranges <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt>. The return value is the end of the
 *  output range.
 *
 *  In the simplest case, \p set_intersection performs the
 *  "intersection" operation from set theory: the output range
 *  contains a copy of every element that is contained in both
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The
 *  general case is more complicated, because the input ranges may
 *  contain duplicate elements. The generalization is that if a value
 *  appears \c m times in <tt>[first1, last1)</tt> and \c n times in
 *  <tt>[first2, last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the output range.
 *  \p set_intersection is stable, meaning that both elements are
 *  copied from the first range rather than the second, and that the
 *  relative order of elements in the output range is the same as in
 *  the first input range.
 *
 *  This version of \p set_intersection compares elements using a function object \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p set_intersection to compute
 *  the set intersection of sets of integers sorted in descending order using the \p thrust::host execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[6] = {11, 9, 7, 5, 3, 1};
 *  int A2[7] = {13, 8, 5, 3, 2,  1, 1};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
 *  // result is now {5, 3, 1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_intersection
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
  OutputIterator set_intersection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  InputIterator1                                              first1,
                                  InputIterator1                                              last1,
                                  InputIterator2                                              first2,
                                  InputIterator2                                              last2,
                                  OutputIterator                                              result,
                                  StrictWeakCompare                                           comp);


/*! \p set_intersection constructs a sorted range that is the
 *  intersection of sorted ranges <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt>. The return value is the end of the
 *  output range.
 *
 *  In the simplest case, \p set_intersection performs the
 *  "intersection" operation from set theory: the output range
 *  contains a copy of every element that is contained in both
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The
 *  general case is more complicated, because the input ranges may
 *  contain duplicate elements. The generalization is that if a value
 *  appears \c m times in <tt>[first1, last1)</tt> and \c n times in
 *  <tt>[first2, last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the output range.
 *  \p set_intersection is stable, meaning that both elements are
 *  copied from the first range rather than the second, and that the
 *  relative order of elements in the output range is the same as in
 *  the first input range.
 *
 *  This version of \p set_intersection compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p set_intersection to compute
 *  the set intersection of sets of integers sorted in descending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[6] = {11, 9, 7, 5, 3, 1};
 *  int A2[7] = {13, 8, 5, 3, 2,  1, 1};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
 *  // result is now {5, 3, 1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_intersection
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakCompare comp);


/*! \p set_symmetric_difference constructs a sorted range that is the set symmetric
 *  difference of the sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>.
 *  The return value is the end of the output range.
 *
 *  In the simplest case, \p set_symmetric_difference performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[first1, last1)</tt> but not <tt>[first2, last1)</tt>, and a copy of
 *  every element that is contained in <tt>[first2, last2)</tt> but not <tt>[first1, last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[first2, last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[first1, last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[first2, last2)</tt> if <tt>m < n</tt>.
 *
 *  This version of \p set_union compares elements using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference to compute
 *  the symmetric difference of two sets of integers sorted in ascending order using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[7] = {0, 1, 2, 2, 4, 6, 7};
 *  int A2[5] = {1, 1, 2, 5, 8};
 *
 *  int result[6];
 *
 *  int *result_end = thrust::set_symmetric_difference(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
 *  // result = {0, 4, 5, 6, 7, 8}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_difference
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator set_symmetric_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                          InputIterator1                                              first1,
                                          InputIterator1                                              last1,
                                          InputIterator2                                              first2,
                                          InputIterator2                                              last2,
                                          OutputIterator                                              result);


/*! \p set_symmetric_difference constructs a sorted range that is the set symmetric
 *  difference of the sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>.
 *  The return value is the end of the output range.
 *
 *  In the simplest case, \p set_symmetric_difference performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[first1, last1)</tt> but not <tt>[first2, last1)</tt>, and a copy of
 *  every element that is contained in <tt>[first2, last2)</tt> but not <tt>[first1, last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[first2, last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[first1, last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[first2, last2)</tt> if <tt>m < n</tt>.
 *
 *  This version of \p set_union compares elements using \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference to compute
 *  the symmetric difference of two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[7] = {0, 1, 2, 2, 4, 6, 7};
 *  int A2[5] = {1, 1, 2, 5, 8};
 *
 *  int result[6];
 *
 *  int *result_end = thrust::set_symmetric_difference(A1, A1 + 7, A2, A2 + 5, result);
 *  // result = {0, 4, 5, 6, 7, 8}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_difference
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result);


/*! \p set_symmetric_difference constructs a sorted range that is the set symmetric
 *  difference of the sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>.
 *  The return value is the end of the output range.
 *
 *  In the simplest case, \p set_symmetric_difference performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[first1, last1)</tt> but not <tt>[first2, last1)</tt>, and a copy of
 *  every element that is contained in <tt>[first2, last2)</tt> but not <tt>[first1, last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[first2, last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[first1, last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[first2, last2)</tt> if <tt>m < n</tt>.
 *
 *  This version of \p set_union compares elements using a function object \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference to compute
 *  the symmetric difference of two sets of integers sorted in descending order using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[7] = {7, 6, 4, 2, 2, 1, 0};
 *  int A2[5] = {8, 5, 2, 1, 1};
 *
 *  int result[6];
 *
 *  int *result_end = thrust::set_symmetric_difference(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
 *  // result = {8, 7, 6, 5, 4, 0}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_difference
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
  OutputIterator set_symmetric_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                          InputIterator1                                              first1,
                                          InputIterator1                                              last1,
                                          InputIterator2                                              first2,
                                          InputIterator2                                              last2,
                                          OutputIterator                                              result,
                                          StrictWeakCompare                                           comp);


/*! \p set_symmetric_difference constructs a sorted range that is the set symmetric
 *  difference of the sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>.
 *  The return value is the end of the output range.
 *
 *  In the simplest case, \p set_symmetric_difference performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[first1, last1)</tt> but not <tt>[first2, last1)</tt>, and a copy of
 *  every element that is contained in <tt>[first2, last2)</tt> but not <tt>[first1, last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[first2, last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[first1, last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[first2, last2)</tt> if <tt>m < n</tt>.
 *
 *  This version of \p set_union compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference to compute
 *  the symmetric difference of two sets of integers sorted in descending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[7] = {7, 6, 4, 2, 2, 1, 0};
 *  int A2[5] = {8, 5, 2, 1, 1};
 *
 *  int result[6];
 *
 *  int *result_end = thrust::set_symmetric_difference(A1, A1 + 7, A2, A2 + 5, result);
 *  // result = {8, 7, 6, 5, 4, 0}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_difference
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result,
                                          StrictWeakCompare comp);


/*! \p set_union constructs a sorted range that is the union of the sorted ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_union performs the "union" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt>, <tt>[first2, last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  This version of \p set_union compares elements using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_union to compute the union of
 *  two sets of integers sorted in ascending order using the \p thrust::host execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
 *  int A2[5] = {1, 3, 5, 7, 9};
 *
 *  int result[11];
 *
 *  int *result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
 *  // result = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_union
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator set_union(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator1                                              first1,
                           InputIterator1                                              last1,
                           InputIterator2                                              first2,
                           InputIterator2                                              last2,
                           OutputIterator                                              result);


/*! \p set_union constructs a sorted range that is the union of the sorted ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_union performs the "union" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt>, <tt>[first2, last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  This version of \p set_union compares elements using \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_union to compute the union of
 *  two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
 *  int A2[5] = {1, 3, 5, 7, 9};
 *
 *  int result[11];
 *
 *  int *result_end = thrust::set_union(A1, A1 + 7, A2, A2 + 5, result);
 *  // result = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_union
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result);


/*! \p set_union constructs a sorted range that is the union of the sorted ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_union performs the "union" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt>, <tt>[first2, last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  This version of \p set_union compares elements using a function object \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertable to \p StrictWeakCompare's \c first_argument_type.
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2's \c value_type is convertable to \p StrictWeakCompare's \c second_argument_type.
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_union to compute the union of
 *  two sets of integers sorted in ascending order using the \p thrust::host execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[7] = {12, 10, 8, 6, 4, 2, 0};
 *  int A2[5] = {9, 7, 5, 3, 1};
 *
 *  int result[11];
 *
 *  int *result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
 *  // result = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_union
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
  OutputIterator set_union(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator1                                              first1,
                           InputIterator1                                              last1,
                           InputIterator2                                              first2,
                           InputIterator2                                              last2,
                           OutputIterator                                              result,
                           StrictWeakCompare                                           comp);


/*! \p set_union constructs a sorted range that is the union of the sorted ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_union performs the "union" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt>, <tt>[first2, last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  This version of \p set_union compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertable to \p StrictWeakCompare's \c first_argument_type.
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2's \c value_type is convertable to \p StrictWeakCompare's \c second_argument_type.
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting range shall not overlap with either input range.
 *
 *  The following code snippet demonstrates how to use \p set_union to compute the union of
 *  two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A1[7] = {12, 10, 8, 6, 4, 2, 0};
 *  int A2[5] = {9, 7, 5, 3, 1};
 *
 *  int result[11];
 *
 *  int *result_end = thrust::set_union(A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
 *  // result = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/set_union
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result,
                           StrictWeakCompare comp);


/*! \p set_difference_by_key performs a key-value difference operation from set theory.
 *  \p set_difference_by_key constructs a sorted range that is the difference of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_difference_by_key performs the "difference" operation from set
 *  theory: the keys output range contains a copy of every element that is contained in
 *  <tt>[keys_first1, keys_last1)</tt> and not contained in <tt>[keys_first2, keys_last2)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[keys_first2, keys_last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[keys_first1, keys_last1)</tt> range shall be copied to the output range.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_difference_by_key compares key elements using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_difference_by_key to compute the
 *  set difference of two sets of integers sorted in ascending order with their values using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {0, 1, 3, 4, 5, 6, 9};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {1, 3, 5, 7, 9};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[3];
 *  int vals_result[3];
 *
 *  thrust::pair<int*,int*> end = thrust::set_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
 *  // keys_result is now {0, 4, 6}
 *  // vals_result is now {0, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_symmetric_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    set_difference_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator1                                              keys_first1,
                          InputIterator1                                              keys_last1,
                          InputIterator2                                              keys_first2,
                          InputIterator2                                              keys_last2,
                          InputIterator3                                              values_first1,
                          InputIterator4                                              values_first2,
                          OutputIterator1                                             keys_result,
                          OutputIterator2                                             values_result);


/*! \p set_difference_by_key performs a key-value difference operation from set theory.
 *  \p set_difference_by_key constructs a sorted range that is the difference of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_difference_by_key performs the "difference" operation from set
 *  theory: the keys output range contains a copy of every element that is contained in
 *  <tt>[keys_first1, keys_last1)</tt> and not contained in <tt>[keys_first2, keys_last2)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[keys_first2, keys_last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[keys_first1, keys_last1)</tt> range shall be copied to the output range.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_difference_by_key compares key elements using \c operator<.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_difference_by_key to compute the
 *  set difference of two sets of integers sorted in ascending order with their values.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A_keys[6] = {0, 1, 3, 4, 5, 6, 9};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {1, 3, 5, 7, 9};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[3];
 *  int vals_result[3];
 *
 *  thrust::pair<int*,int*> end = thrust::set_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
 *  // keys_result is now {0, 4, 6}
 *  // vals_result is now {0, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_symmetric_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_difference_by_key(InputIterator1                             keys_first1,
                          InputIterator1                             keys_last1,
                          InputIterator2                             keys_first2,
                          InputIterator2                             keys_last2,
                          InputIterator3                             values_first1,
                          InputIterator4                             values_first2,
                          OutputIterator1                            keys_result,
                          OutputIterator2                            values_result);


/*! \p set_difference_by_key performs a key-value difference operation from set theory.
 *  \p set_difference_by_key constructs a sorted range that is the difference of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_difference_by_key performs the "difference" operation from set
 *  theory: the keys output range contains a copy of every element that is contained in
 *  <tt>[keys_first1, keys_last1)</tt> and not contained in <tt>[keys_first2, keys_last2)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[keys_first2, keys_last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[keys_first1, keys_last1)</tt> range shall be copied to the output range.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_difference_by_key compares key elements using a function object \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_difference_by_key to compute the
 *  set difference of two sets of integers sorted in descending order with their values using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {9, 6, 5, 4, 3, 1, 0};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {9, 7, 5, 3, 1};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[3];
 *  int vals_result[3];
 *
 *  thrust::pair<int*,int*> end = thrust::set_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
 *  // keys_result is now {0, 4, 6}
 *  // vals_result is now {0, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_symmetric_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    set_difference_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator1                                              keys_first1,
                          InputIterator1                                              keys_last1,
                          InputIterator2                                              keys_first2,
                          InputIterator2                                              keys_last2,
                          InputIterator3                                              values_first1,
                          InputIterator4                                              values_first2,
                          OutputIterator1                                             keys_result,
                          OutputIterator2                                             values_result,
                          StrictWeakCompare                                           comp);


/*! \p set_difference_by_key performs a key-value difference operation from set theory.
 *  \p set_difference_by_key constructs a sorted range that is the difference of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_difference_by_key performs the "difference" operation from set
 *  theory: the keys output range contains a copy of every element that is contained in
 *  <tt>[keys_first1, keys_last1)</tt> and not contained in <tt>[keys_first2, keys_last2)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[keys_first2, keys_last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[keys_first1, keys_last1)</tt> range shall be copied to the output range.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_difference_by_key compares key elements using a function object \p comp.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_difference_by_key to compute the
 *  set difference of two sets of integers sorted in descending order with their values.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A_keys[6] = {9, 6, 5, 4, 3, 1, 0};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {9, 7, 5, 3, 1};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[3];
 *  int vals_result[3];
 *
 *  thrust::pair<int*,int*> end = thrust::set_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
 *  // keys_result is now {0, 4, 6}
 *  // vals_result is now {0, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_symmetric_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_difference_by_key(InputIterator1                             keys_first1,
                          InputIterator1                             keys_last1,
                          InputIterator2                             keys_first2,
                          InputIterator2                             keys_last2,
                          InputIterator3                             values_first1,
                          InputIterator4                             values_first2,
                          OutputIterator1                            keys_result,
                          OutputIterator2                            values_result,
                          StrictWeakCompare                          comp);


/*! \p set_intersection_by_key performs a key-value intersection operation from set theory.
 *  \p set_intersection_by_key constructs a sorted range that is the intersection of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_intersection_by_key performs the "intersection" operation from set
 *  theory: the keys output range contains a copy of every element that is contained in both
 *  <tt>[keys_first1, keys_last1)</tt> <tt>[keys_first2, keys_last2)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if an element appears \c m times in <tt>[keys_first1, keys_last1)</tt>
 *  and \c n times in <tt>[keys_first2, keys_last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the keys output range.
 *  \p set_intersection_by_key is stable, meaning both that elements are copied from the first
 *  input range rather than the second, and that the relative order of elements in the output range
 *  is the same as the first input range.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> to the keys output range,
 *  the corresponding value element is copied from <tt>[values_first1, values_last1)</tt> to the values
 *  output range.
 *
 *  This version of \p set_intersection_by_key compares objects using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \note Unlike the other key-value set operations, \p set_intersection_by_key is unique in that it has no
 *        \c values_first2 parameter because elements from the second input range are never copied to the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_intersection_by_key to compute the
 *  set intersection of two sets of integers sorted in ascending order with their values using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {1, 3, 5, 7, 9, 11};
 *  int A_vals[6] = {0, 0, 0, 0, 0,  0};
 *  
 *  int B_keys[7] = {1, 1, 2, 3, 5,  8, 13};
 *
 *  int keys_result[7];
 *  int vals_result[7];
 *
 *  thrust::pair<int*,int*> end = thrust::set_intersection_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, keys_result, vals_result);
 *
 *  // keys_result is now {1, 3, 5}
 *  // vals_result is now {0, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_difference_by_key
 *  \see \p set_symmetric_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    set_intersection_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            InputIterator1                                              keys_first1,
                            InputIterator1                                              keys_last1,
                            InputIterator2                                              keys_first2,
                            InputIterator2                                              keys_last2,
                            InputIterator3                                              values_first1,
                            OutputIterator1                                             keys_result,
                            OutputIterator2                                             values_result);


/*! \p set_intersection_by_key performs a key-value intersection operation from set theory.
 *  \p set_intersection_by_key constructs a sorted range that is the intersection of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_intersection_by_key performs the "intersection" operation from set
 *  theory: the keys output range contains a copy of every element that is contained in both
 *  <tt>[keys_first1, keys_last1)</tt> <tt>[keys_first2, keys_last2)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if an element appears \c m times in <tt>[keys_first1, keys_last1)</tt>
 *  and \c n times in <tt>[keys_first2, keys_last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the keys output range.
 *  \p set_intersection_by_key is stable, meaning both that elements are copied from the first
 *  input range rather than the second, and that the relative order of elements in the output range
 *  is the same as the first input range.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> to the keys output range,
 *  the corresponding value element is copied from <tt>[values_first1, values_last1)</tt> to the values
 *  output range.
 *
 *  This version of \p set_intersection_by_key compares objects using \c operator<.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \note Unlike the other key-value set operations, \p set_intersection_by_key is unique in that it has no
 *        \c values_first2 parameter because elements from the second input range are never copied to the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_intersection_by_key to compute the
 *  set intersection of two sets of integers sorted in ascending order with their values.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A_keys[6] = {1, 3, 5, 7, 9, 11};
 *  int A_vals[6] = {0, 0, 0, 0, 0,  0};
 *  
 *  int B_keys[7] = {1, 1, 2, 3, 5,  8, 13};
 *
 *  int keys_result[7];
 *  int vals_result[7];
 *
 *  thrust::pair<int*,int*> end = thrust::set_intersection_by_key(A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, keys_result, vals_result);
 *
 *  // keys_result is now {1, 3, 5}
 *  // vals_result is now {0, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_difference_by_key
 *  \see \p set_symmetric_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_intersection_by_key(InputIterator1                             keys_first1,
                            InputIterator1                             keys_last1,
                            InputIterator2                             keys_first2,
                            InputIterator2                             keys_last2,
                            InputIterator3                             values_first1,
                            OutputIterator1                            keys_result,
                            OutputIterator2                            values_result);


/*! \p set_intersection_by_key performs a key-value intersection operation from set theory.
 *  \p set_intersection_by_key constructs a sorted range that is the intersection of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_intersection_by_key performs the "intersection" operation from set
 *  theory: the keys output range contains a copy of every element that is contained in both
 *  <tt>[keys_first1, keys_last1)</tt> <tt>[keys_first2, keys_last2)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if an element appears \c m times in <tt>[keys_first1, keys_last1)</tt>
 *  and \c n times in <tt>[keys_first2, keys_last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the keys output range.
 *  \p set_intersection_by_key is stable, meaning both that elements are copied from the first
 *  input range rather than the second, and that the relative order of elements in the output range
 *  is the same as the first input range.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> to the keys output range,
 *  the corresponding value element is copied from <tt>[values_first1, values_last1)</tt> to the values
 *  output range.
 *
 *  This version of \p set_intersection_by_key compares objects using a function object \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \note Unlike the other key-value set operations, \p set_intersection_by_key is unique in that it has no
 *        \c values_first2 parameter because elements from the second input range are never copied to the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_intersection_by_key to compute the
 *  set intersection of two sets of integers sorted in descending order with their values using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {11, 9, 7, 5, 3, 1};
 *  int A_vals[6] = { 0, 0, 0, 0, 0, 0};
 *  
 *  int B_keys[7] = {13, 8, 5, 3, 2, 1, 1};
 *
 *  int keys_result[7];
 *  int vals_result[7];
 *
 *  thrust::pair<int*,int*> end = thrust::set_intersection_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, keys_result, vals_result, thrust::greater<int>());
 *
 *  // keys_result is now {5, 3, 1}
 *  // vals_result is now {0, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_difference_by_key
 *  \see \p set_symmetric_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    set_intersection_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            InputIterator1                                              keys_first1,
                            InputIterator1                                              keys_last1,
                            InputIterator2                                              keys_first2,
                            InputIterator2                                              keys_last2,
                            InputIterator3                                              values_first1,
                            OutputIterator1                                             keys_result,
                            OutputIterator2                                             values_result,
                            StrictWeakCompare                                           comp);


/*! \p set_intersection_by_key performs a key-value intersection operation from set theory.
 *  \p set_intersection_by_key constructs a sorted range that is the intersection of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_intersection_by_key performs the "intersection" operation from set
 *  theory: the keys output range contains a copy of every element that is contained in both
 *  <tt>[keys_first1, keys_last1)</tt> <tt>[keys_first2, keys_last2)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if an element appears \c m times in <tt>[keys_first1, keys_last1)</tt>
 *  and \c n times in <tt>[keys_first2, keys_last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the keys output range.
 *  \p set_intersection_by_key is stable, meaning both that elements are copied from the first
 *  input range rather than the second, and that the relative order of elements in the output range
 *  is the same as the first input range.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> to the keys output range,
 *  the corresponding value element is copied from <tt>[values_first1, values_last1)</tt> to the values
 *  output range.
 *
 *  This version of \p set_intersection_by_key compares objects using a function object \p comp.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \note Unlike the other key-value set operations, \p set_intersection_by_key is unique in that it has no
 *        \c values_first2 parameter because elements from the second input range are never copied to the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_intersection_by_key to compute the
 *  set intersection of two sets of integers sorted in descending order with their values.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A_keys[6] = {11, 9, 7, 5, 3, 1};
 *  int A_vals[6] = { 0, 0, 0, 0, 0, 0};
 *  
 *  int B_keys[7] = {13, 8, 5, 3, 2, 1, 1};
 *
 *  int keys_result[7];
 *  int vals_result[7];
 *
 *  thrust::pair<int*,int*> end = thrust::set_intersection_by_key(A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, keys_result, vals_result, thrust::greater<int>());
 *
 *  // keys_result is now {5, 3, 1}
 *  // vals_result is now {0, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_difference_by_key
 *  \see \p set_symmetric_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_intersection_by_key(InputIterator1                             keys_first1,
                            InputIterator1                             keys_last1,
                            InputIterator2                             keys_first2,
                            InputIterator2                             keys_last2,
                            InputIterator3                             values_first1,
                            OutputIterator1                            keys_result,
                            OutputIterator2                            values_result,
                            StrictWeakCompare                          comp);


/*! \p set_symmetric_difference_by_key performs a key-value symmetric difference operation from set theory.
 *  \p set_difference_by_key constructs a sorted range that is the symmetric difference of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_symmetric_difference_by_key performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[keys_first1, keys_last1)</tt> but not <tt>[keys_first2, keys_last1)</tt>, and a copy of
 *  every element that is contained in <tt>[keys_first2, keys_last2)</tt> but not <tt>[keys_first1, keys_last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[keys_first2, keys_last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[keys_first1, keys_last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[keys_first2, keys_last2)</tt> if <tt>m < n</tt>.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_symmetric_difference_by_key compares key elements using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference_by_key to compute the
 *  symmetric difference of two sets of integers sorted in ascending order with their values using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {0, 1, 2, 2, 4, 6, 7};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {1, 1, 2, 5, 8};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[6];
 *  int vals_result[6];
 *
 *  thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
 *  // keys_result is now {0, 4, 5, 6, 7, 8}
 *  // vals_result is now {0, 0, 1, 0, 0, 1}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    set_symmetric_difference_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                    InputIterator1                                              keys_first1,
                                    InputIterator1                                              keys_last1,
                                    InputIterator2                                              keys_first2,
                                    InputIterator2                                              keys_last2,
                                    InputIterator3                                              values_first1,
                                    InputIterator4                                              values_first2,
                                    OutputIterator1                                             keys_result,
                                    OutputIterator2                                             values_result);


/*! \p set_symmetric_difference_by_key performs a key-value symmetric difference operation from set theory.
 *  \p set_difference_by_key constructs a sorted range that is the symmetric difference of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_symmetric_difference_by_key performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[keys_first1, keys_last1)</tt> but not <tt>[keys_first2, keys_last1)</tt>, and a copy of
 *  every element that is contained in <tt>[keys_first2, keys_last2)</tt> but not <tt>[keys_first1, keys_last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[keys_first2, keys_last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[keys_first1, keys_last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[keys_first2, keys_last2)</tt> if <tt>m < n</tt>.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_symmetric_difference_by_key compares key elements using \c operator<.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference_by_key to compute the
 *  symmetric difference of two sets of integers sorted in ascending order with their values.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A_keys[6] = {0, 1, 2, 2, 4, 6, 7};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {1, 1, 2, 5, 8};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[6];
 *  int vals_result[6];
 *
 *  thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
 *  // keys_result is now {0, 4, 5, 6, 7, 8}
 *  // vals_result is now {0, 0, 1, 0, 0, 1}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_symmetric_difference_by_key(InputIterator1                             keys_first1,
                                    InputIterator1                             keys_last1,
                                    InputIterator2                             keys_first2,
                                    InputIterator2                             keys_last2,
                                    InputIterator3                             values_first1,
                                    InputIterator4                             values_first2,
                                    OutputIterator1                            keys_result,
                                    OutputIterator2                            values_result);


/*! \p set_symmetric_difference_by_key performs a key-value symmetric difference operation from set theory.
 *  \p set_difference_by_key constructs a sorted range that is the symmetric difference of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_symmetric_difference_by_key performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[keys_first1, keys_last1)</tt> but not <tt>[keys_first2, keys_last1)</tt>, and a copy of
 *  every element that is contained in <tt>[keys_first2, keys_last2)</tt> but not <tt>[keys_first1, keys_last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[keys_first2, keys_last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[keys_first1, keys_last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[keys_first2, keys_last2)</tt> if <tt>m < n</tt>.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_symmetric_difference_by_key compares key elements using a function object \c comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference_by_key to compute the
 *  symmetric difference of two sets of integers sorted in descending order with their values using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {7, 6, 4, 2, 2, 1, 0};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {8, 5, 2, 1, 1};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[6];
 *  int vals_result[6];
 *
 *  thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
 *  // keys_result is now {8, 7, 6, 5, 4, 0}
 *  // vals_result is now {1, 0, 0, 1, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    set_symmetric_difference_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                    InputIterator1                                              keys_first1,
                                    InputIterator1                                              keys_last1,
                                    InputIterator2                                              keys_first2,
                                    InputIterator2                                              keys_last2,
                                    InputIterator3                                              values_first1,
                                    InputIterator4                                              values_first2,
                                    OutputIterator1                                             keys_result,
                                    OutputIterator2                                             values_result,
                                    StrictWeakCompare                                           comp);


/*! \p set_symmetric_difference_by_key performs a key-value symmetric difference operation from set theory.
 *  \p set_difference_by_key constructs a sorted range that is the symmetric difference of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_symmetric_difference_by_key performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[keys_first1, keys_last1)</tt> but not <tt>[keys_first2, keys_last1)</tt>, and a copy of
 *  every element that is contained in <tt>[keys_first2, keys_last2)</tt> but not <tt>[keys_first1, keys_last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[keys_first2, keys_last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[keys_first1, keys_last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[keys_first2, keys_last2)</tt> if <tt>m < n</tt>.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_symmetric_difference_by_key compares key elements using a function object \c comp.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference_by_key to compute the
 *  symmetric difference of two sets of integers sorted in descending order with their values.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A_keys[6] = {7, 6, 4, 2, 2, 1, 0};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {8, 5, 2, 1, 1};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[6];
 *  int vals_result[6];
 *
 *  thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
 *  // keys_result is now {8, 7, 6, 5, 4, 0}
 *  // vals_result is now {1, 0, 0, 1, 0, 0}
 *  \endcode
 *
 *  \see \p set_union_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_symmetric_difference_by_key(InputIterator1                             keys_first1,
                                    InputIterator1                             keys_last1,
                                    InputIterator2                             keys_first2,
                                    InputIterator2                             keys_last2,
                                    InputIterator3                             values_first1,
                                    InputIterator4                             values_first2,
                                    OutputIterator1                            keys_result,
                                    OutputIterator2                            values_result,
                                    StrictWeakCompare                          comp);


/*! \p set_union_by_key performs a key-value union operation from set theory.
 *  \p set_union_by_key constructs a sorted range that is the union of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_union_by_key performs the "union" operation from set theory:
 *  the output range contains a copy of every element that is contained in
 *  <tt>[keys_first1, keys_last1)</tt>, <tt>[keys_first2, keys_last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[keys_first2, keys_last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_union_by_key compares key elements using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference_by_key to compute the
 *  symmetric difference of two sets of integers sorted in ascending order with their values using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {0, 2, 4, 6, 8, 10, 12};
 *  int A_vals[6] = {0, 0, 0, 0, 0,  0,  0};
 *
 *  int B_keys[5] = {1, 3, 5, 7, 9};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[11];
 *  int vals_result[11];
 *
 *  thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
 *  // keys_result is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
 *  // vals_result is now {0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  0,  0}
 *  \endcode
 *
 *  \see \p set_symmetric_difference_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    set_union_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                     InputIterator1                                              keys_first1,
                     InputIterator1                                              keys_last1,
                     InputIterator2                                              keys_first2,
                     InputIterator2                                              keys_last2,
                     InputIterator3                                              values_first1,
                     InputIterator4                                              values_first2,
                     OutputIterator1                                             keys_result,
                     OutputIterator2                                             values_result);


/*! \p set_union_by_key performs a key-value union operation from set theory.
 *  \p set_union_by_key constructs a sorted range that is the union of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_union_by_key performs the "union" operation from set theory:
 *  the output range contains a copy of every element that is contained in
 *  <tt>[keys_first1, keys_last1)</tt>, <tt>[keys_first2, keys_last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[keys_first2, keys_last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_union_by_key compares key elements using \c operator<.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to <tt>operator<</tt>.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference_by_key to compute the
 *  symmetric difference of two sets of integers sorted in ascending order with their values.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A_keys[6] = {0, 2, 4, 6, 8, 10, 12};
 *  int A_vals[6] = {0, 0, 0, 0, 0,  0,  0};
 *
 *  int B_keys[5] = {1, 3, 5, 7, 9};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[11];
 *  int vals_result[11];
 *
 *  thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
 *  // keys_result is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
 *  // vals_result is now {0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  0,  0}
 *  \endcode
 *
 *  \see \p set_symmetric_difference_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_union_by_key(InputIterator1                             keys_first1,
                     InputIterator1                             keys_last1,
                     InputIterator2                             keys_first2,
                     InputIterator2                             keys_last2,
                     InputIterator3                             values_first1,
                     InputIterator4                             values_first2,
                     OutputIterator1                            keys_result,
                     OutputIterator2                            values_result);


/*! \p set_union_by_key performs a key-value union operation from set theory.
 *  \p set_union_by_key constructs a sorted range that is the union of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_union_by_key performs the "union" operation from set theory:
 *  the output range contains a copy of every element that is contained in
 *  <tt>[keys_first1, keys_last1)</tt>, <tt>[keys_first2, keys_last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[keys_first2, keys_last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_union_by_key compares key elements using a function object \c comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference_by_key to compute the
 *  symmetric difference of two sets of integers sorted in descending order with their values using the
 *  \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {12, 10, 8, 6, 4, 2, 0};
 *  int A_vals[6] = { 0,  0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {9, 7, 5, 3, 1};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[11];
 *  int vals_result[11];
 *
 *  thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
 *  // keys_result is now {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
 *  // vals_result is now { 0,  1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
 *  \endcode
 *
 *  \see \p set_symmetric_difference_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    set_union_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                     InputIterator1                                              keys_first1,
                     InputIterator1                                              keys_last1,
                     InputIterator2                                              keys_first2,
                     InputIterator2                                              keys_last2,
                     InputIterator3                                              values_first1,
                     InputIterator4                                              values_first2,
                     OutputIterator1                                             keys_result,
                     OutputIterator2                                             values_result,
                     StrictWeakCompare                                           comp);


/*! \p set_union_by_key performs a key-value union operation from set theory.
 *  \p set_union_by_key constructs a sorted range that is the union of the sorted
 *  ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt>. Associated
 *  with each element from the input and output key ranges is a value element. The associated input
 *  value ranges need not be sorted.
 *
 *  In the simplest case, \p set_union_by_key performs the "union" operation from set theory:
 *  the output range contains a copy of every element that is contained in
 *  <tt>[keys_first1, keys_last1)</tt>, <tt>[keys_first2, keys_last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[keys_first1, keys_last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[keys_first2, keys_last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  Each time a key element is copied from <tt>[keys_first1, keys_last1)</tt> or
 *  <tt>[keys_first2, keys_last2)</tt> is copied to the keys output range, the
 *  corresponding value element is copied from the corresponding values input range (beginning at
 *  \p values_first1 or \p values_first2) to the values output range.
 *
 *  This version of \p set_union_by_key compares key elements using a function object \c comp.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the output range of keys.
 *  \param values_result The beginning of the output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator3 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator3's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam InputIterator4 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator4's \c value_type is convertible to a type in \p OutputIterator2's set of \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> shall be sorted with respect to \p comp.
 *  \pre The resulting ranges shall not overlap with any input range.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference_by_key to compute the
 *  symmetric difference of two sets of integers sorted in descending order with their values.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A_keys[6] = {12, 10, 8, 6, 4, 2, 0};
 *  int A_vals[6] = { 0,  0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[5] = {9, 7, 5, 3, 1};
 *  int B_vals[5] = {1, 1, 1, 1, 1};
 *
 *  int keys_result[11];
 *  int vals_result[11];
 *
 *  thrust::pair<int*,int*> end = thrust::set_symmetric_difference_by_key(A_keys, A_keys + 6, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
 *  // keys_result is now {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
 *  // vals_result is now { 0,  1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
 *  \endcode
 *
 *  \see \p set_symmetric_difference_by_key
 *  \see \p set_intersection_by_key
 *  \see \p set_difference_by_key
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_union_by_key(InputIterator1                             keys_first1,
                     InputIterator1                             keys_last1,
                     InputIterator2                             keys_first2,
                     InputIterator2                             keys_last2,
                     InputIterator3                             values_first1,
                     InputIterator4                             values_first2,
                     OutputIterator1                            keys_result,
                     OutputIterator2                            values_result,
                     StrictWeakCompare                          comp);


/*! \} // end set_operations
 */

THRUST_NAMESPACE_END

#include <thrust/detail/set_operations.inl>
