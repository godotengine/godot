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

/*! \file merge.h
 *  \brief Merging sorted ranges
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup merging Merging
 *  \ingroup algorithms
 *  \{
 */


/*! \p merge combines two sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>
 *  into a single sorted range. That is, it copies from <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt> into <tt>[result, result + (last1 - first1) + (last2 - first2))</tt>
 *  such that the resulting range is in ascending order. \p merge is stable, meaning both that the
 *  relative order of elements within each input range is preserved, and that for equivalent elements
 *  in both input ranges the element from the first range precedes the element from the second. The
 *  return value is <tt>result + (last1 - first1) + (last2 - first2)</tt>.
 *
 *  This version of \p merge compares elements using \c operator<.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the merged output.
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
 *  The following code snippet demonstrates how to use
 *  \p merge to compute the merger of two sorted sets of integers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/merge.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[6] = {1, 3, 5, 7, 9, 11};
 *  int A2[7] = {1, 1, 2, 3, 5,  8, 13};
 *
 *  int result[13];
 *
 *  int *result_end =
 *    thrust::merge(thrust::host,
 *                  A1, A1 + 6,
 *                  A2, A2 + 7,
 *                  result);
 *  // result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/merge
 *  \see \p set_union
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator merge(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result);


/*! \p merge combines two sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>
 *  into a single sorted range. That is, it copies from <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt> into <tt>[result, result + (last1 - first1) + (last2 - first2))</tt>
 *  such that the resulting range is in ascending order. \p merge is stable, meaning both that the
 *  relative order of elements within each input range is preserved, and that for equivalent elements
 *  in both input ranges the element from the first range precedes the element from the second. The
 *  return value is <tt>result + (last1 - first1) + (last2 - first2)</tt>.
 *
 *  This version of \p merge compares elements using \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the merged output.
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
 *  The following code snippet demonstrates how to use
 *  \p merge to compute the merger of two sorted sets of integers.
 *
 *  \code
 *  #include <thrust/merge.h>
 *  ...
 *  int A1[6] = {1, 3, 5, 7, 9, 11};
 *  int A2[7] = {1, 1, 2, 3, 5,  8, 13};
 *
 *  int result[13];
 *
 *  int *result_end = thrust::merge(A1, A1 + 6, A2, A2 + 7, result);
 *  // result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/merge
 *  \see \p set_union
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator merge(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result);


/*! \p merge combines two sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>
 *  into a single sorted range. That is, it copies from <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt> into <tt>[result, result + (last1 - first1) + (last2 - first2))</tt>
 *  such that the resulting range is in ascending order. \p merge is stable, meaning both that the
 *  relative order of elements within each input range is preserved, and that for equivalent elements
 *  in both input ranges the element from the first range precedes the element from the second. The
 *  return value is <tt>result + (last1 - first1) + (last2 - first2)</tt>.
 *
 *  This version of \p merge compares elements using a function object \p comp.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the merged output.
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
 *  The following code snippet demonstrates how to use
 *  \p merge to compute the merger of two sets of integers sorted in
 *  descending order using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/merge.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A1[6] = {11, 9, 7, 5, 3, 1};
 *  int A2[7] = {13, 8, 5, 3, 2, 1, 1};
 *
 *  int result[13];
 *
 *  int *result_end = thrust::merge(thrust::host,
 *                                  A1, A1 + 6,
 *                                  A2, A2 + 7,
 *                                  result,
 *                                  thrust::greater<int>());
 *  // result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/merge
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
  OutputIterator merge(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result,
                       StrictWeakCompare comp);


/*! \p merge combines two sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>
 *  into a single sorted range. That is, it copies from <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt> into <tt>[result, result + (last1 - first1) + (last2 - first2))</tt>
 *  such that the resulting range is in ascending order. \p merge is stable, meaning both that the
 *  relative order of elements within each input range is preserved, and that for equivalent elements
 *  in both input ranges the element from the first range precedes the element from the second. The
 *  return value is <tt>result + (last1 - first1) + (last2 - first2)</tt>.
 *
 *  This version of \p merge compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the merged output.
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
 *  The following code snippet demonstrates how to use
 *  \p merge to compute the merger of two sets of integers sorted in
 *  descending order.
 *
 *  \code
 *  #include <thrust/merge.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A1[6] = {11, 9, 7, 5, 3, 1};
 *  int A2[7] = {13, 8, 5, 3, 2, 1, 1};
 *
 *  int result[13];
 *
 *  int *result_end = thrust::merge(A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
 *  // result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/merge
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator merge(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result,
                       StrictWeakCompare comp);


/*! \p merge_by_key performs a key-value merge. That is, \p merge_by_key copies elements from
 *  <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> into a single range,
 *  <tt>[keys_result, keys_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt> such that
 *  the resulting range is in ascending key order.
 *
 *  At the same time, \p merge_by_key copies elements from the two associated ranges <tt>[values_first1 + (keys_last1 - keys_first1))</tt>
 *  and <tt>[values_first2 + (keys_last2 - keys_first2))</tt> into a single range,
 *  <tt>[values_result, values_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt> such that
 *  the resulting range is in ascending order implied by each input element's associated key.
 *
 *  \p merge_by_key is stable, meaning both that the relative order of elements within each input range is
 *  preserved, and that for equivalent elements in all input key ranges the element from the first range
 *  precedes the element from the second.
 *
 *  The return value is is <tt>(keys_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt>
 *  and <tt>(values_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt>.
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
 *  \param keys_result The beginning of the merged output range of keys.
 *  \param values_result The beginning of the merged output range of values.
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
 *  The following code snippet demonstrates how to use
 *  \p merge_by_key to compute the merger of two sets of integers sorted in
 *  ascending order using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/merge.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {1, 3, 5, 7, 9, 11};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[7] = {1, 1, 2, 3, 5, 8, 13};
 *  int B_vals[7] = {1, 1, 1, 1, 1, 1, 1};
 *
 *  int keys_result[13];
 *  int vals_result[13];
 *
 *  thrust::pair<int*,int*> end =
 *    thrust::merge_by_key(thrust::host,
 *                         A_keys, A_keys + 6,
 *                         B_keys, B_keys + 7,
 *                         A_vals, B_vals,
 *                         keys_result, vals_result);
 *
 *  // keys_result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
 *  // vals_result = {0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,  0,  1}
 *  \endcode
 *
 *  \see merge
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result);


/*! \p merge_by_key performs a key-value merge. That is, \p merge_by_key copies elements from
 *  <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> into a single range,
 *  <tt>[keys_result, keys_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt> such that
 *  the resulting range is in ascending key order.
 *
 *  At the same time, \p merge_by_key copies elements from the two associated ranges <tt>[values_first1 + (keys_last1 - keys_first1))</tt>
 *  and <tt>[values_first2 + (keys_last2 - keys_first2))</tt> into a single range,
 *  <tt>[values_result, values_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt> such that
 *  the resulting range is in ascending order implied by each input element's associated key.
 *
 *  \p merge_by_key is stable, meaning both that the relative order of elements within each input range is
 *  preserved, and that for equivalent elements in all input key ranges the element from the first range
 *  precedes the element from the second.
 *
 *  The return value is is <tt>(keys_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt>
 *  and <tt>(values_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt>.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the merged output range of keys.
 *  \param values_result The beginning of the merged output range of values.
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
 *  The following code snippet demonstrates how to use
 *  \p merge_by_key to compute the merger of two sets of integers sorted in
 *  ascending order.
 *
 *  \code
 *  #include <thrust/merge.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A_keys[6] = {1, 3, 5, 7, 9, 11};
 *  int A_vals[6] = {0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[7] = {1, 1, 2, 3, 5, 8, 13};
 *  int B_vals[7] = {1, 1, 1, 1, 1, 1, 1};
 *
 *  int keys_result[13];
 *  int vals_result[13];
 *
 *  thrust::pair<int*,int*> end = thrust::merge_by_key(A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, B_vals, keys_result, vals_result);
 *
 *  // keys_result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
 *  // vals_result = {0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,  0,  1}
 *  \endcode
 *
 *  \see merge
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result);


/*! \p merge_by_key performs a key-value merge. That is, \p merge_by_key copies elements from
 *  <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> into a single range,
 *  <tt>[keys_result, keys_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt> such that
 *  the resulting range is in ascending key order.
 *
 *  At the same time, \p merge_by_key copies elements from the two associated ranges <tt>[values_first1 + (keys_last1 - keys_first1))</tt>
 *  and <tt>[values_first2 + (keys_last2 - keys_first2))</tt> into a single range,
 *  <tt>[values_result, values_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt> such that
 *  the resulting range is in ascending order implied by each input element's associated key.
 *
 *  \p merge_by_key is stable, meaning both that the relative order of elements within each input range is
 *  preserved, and that for equivalent elements in all input key ranges the element from the first range
 *  precedes the element from the second.
 *
 *  The return value is is <tt>(keys_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt>
 *  and <tt>(values_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt>.
 *
 *  This version of \p merge_by_key compares key elements using a function object \p comp.
 *
 *  The algorithm's execution is parallelized using \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the merged output range of keys.
 *  \param values_result The beginning of the merged output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertable to \p StrictWeakCompare's \c first_argument_type.
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator1's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2's \c value_type is convertable to \p StrictWeakCompare's \c second_argument_type.
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator1's set of \c value_types.
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
 *  The following code snippet demonstrates how to use
 *  \p merge_by_key to compute the merger of two sets of integers sorted in
 *  descending order using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/merge.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A_keys[6] = {11, 9, 7, 5, 3, 1};
 *  int A_vals[6] = { 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[7] = {13, 8, 5, 3, 2, 1, 1};
 *  int B_vals[7] = { 1, 1, 1, 1, 1, 1, 1};
 *
 *  int keys_result[13];
 *  int vals_result[13];
 *
 *  thrust::pair<int*,int*> end =
 *    thrust::merge_by_key(thrust::host,
 *                         A_keys, A_keys + 6,
 *                         B_keys, B_keys + 7,
 *                         A_vals, B_vals,
 *                         keys_result, vals_result,
 *                         thrust::greater<int>());
 *
 *  // keys_result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
 *  // vals_result = { 1,  0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1}
 *  \endcode
 *
 *  \see merge
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2, typename Compare>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 Compare comp);


/*! \p merge_by_key performs a key-value merge. That is, \p merge_by_key copies elements from
 *  <tt>[keys_first1, keys_last1)</tt> and <tt>[keys_first2, keys_last2)</tt> into a single range,
 *  <tt>[keys_result, keys_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt> such that
 *  the resulting range is in ascending key order.
 *
 *  At the same time, \p merge_by_key copies elements from the two associated ranges <tt>[values_first1 + (keys_last1 - keys_first1))</tt>
 *  and <tt>[values_first2 + (keys_last2 - keys_first2))</tt> into a single range,
 *  <tt>[values_result, values_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt> such that
 *  the resulting range is in ascending order implied by each input element's associated key.
 *
 *  \p merge_by_key is stable, meaning both that the relative order of elements within each input range is
 *  preserved, and that for equivalent elements in all input key ranges the element from the first range
 *  precedes the element from the second.
 *
 *  The return value is is <tt>(keys_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt>
 *  and <tt>(values_result + (keys_last1 - keys_first1) + (keys_last2 - keys_first2))</tt>.
 *
 *  This version of \p merge_by_key compares key elements using a function object \p comp.
 *
 *  \param keys_first1 The beginning of the first input range of keys.
 *  \param keys_last1 The end of the first input range of keys.
 *  \param keys_first2 The beginning of the second input range of keys.
 *  \param keys_last2 The end of the second input range of keys.
 *  \param values_first1 The beginning of the first input range of values.
 *  \param values_first2 The beginning of the first input range of values.
 *  \param keys_result The beginning of the merged output range of keys.
 *  \param values_result The beginning of the merged output range of values.
 *  \param comp Comparison operator.
 *  \return A \p pair \c p such that <tt>p.first</tt> is the end of the output range of keys,
 *          and such that <tt>p.second</tt> is the end of the output range of values.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertable to \p StrictWeakCompare's \c first_argument_type.
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator1's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator2's \c value_type is convertable to \p StrictWeakCompare's \c second_argument_type.
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator1's set of \c value_types.
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
 *  The following code snippet demonstrates how to use
 *  \p merge_by_key to compute the merger of two sets of integers sorted in
 *  descending order.
 *
 *  \code
 *  #include <thrust/merge.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A_keys[6] = {11, 9, 7, 5, 3, 1};
 *  int A_vals[6] = { 0, 0, 0, 0, 0, 0};
 *
 *  int B_keys[7] = {13, 8, 5, 3, 2, 1, 1};
 *  int B_vals[7] = { 1, 1, 1, 1, 1, 1, 1};
 *
 *  int keys_result[13];
 *  int vals_result[13];
 *
 *  thrust::pair<int*,int*> end = thrust::merge_by_key(A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());
 *
 *  // keys_result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
 *  // vals_result = { 1,  0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1}
 *  \endcode
 *
 *  \see merge
 *  \see \p sort_by_key
 *  \see \p is_sorted
 */
template<typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2, typename StrictWeakCompare>
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 StrictWeakCompare comp);


/*! \} // merging
 */

THRUST_NAMESPACE_END

#include <thrust/detail/merge.inl>
