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


/*! \file binary_search.h
 *  \brief Search for values in sorted ranges.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */


/*! \addtogroup searching
 *  \ingroup algorithms
 *  \{
 */


/*! \addtogroup binary_search Binary Search
 *  \ingroup searching
 *  \{
 */


//////////////////////   
// Scalar Functions //
//////////////////////


/*! \p lower_bound is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * Specifically, it returns the first position where value could be
 * inserted without violating the ordering. This version of 
 * \p lower_bound uses <tt>operator<</tt> for comparison and returns
 * the furthermost iterator \c i in <tt>[first, last)</tt> such that,
 * for every iterator \c j in <tt>[first, i)</tt>, <tt>*j < value</tt>. 
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \return The furthermost iterator \c i, such that <tt>*i < value</tt>.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam LessThanComparable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
 *
 *  The following code snippet demonstrates how to use \p lower_bound
 *  to search for values in a ordered range using the \p thrust::device execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::lower_bound(thrust::device, input.begin(), input.end(), 0); // returns input.begin()
 *  thrust::lower_bound(thrust::device, input.begin(), input.end(), 1); // returns input.begin() + 1
 *  thrust::lower_bound(thrust::device, input.begin(), input.end(), 2); // returns input.begin() + 1
 *  thrust::lower_bound(thrust::device, input.begin(), input.end(), 3); // returns input.begin() + 2
 *  thrust::lower_bound(thrust::device, input.begin(), input.end(), 8); // returns input.begin() + 4
 *  thrust::lower_bound(thrust::device, input.begin(), input.end(), 9); // returns input.end()
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template<typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable>
__host__ __device__
ForwardIterator lower_bound(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            const LessThanComparable &value);


/*! \p lower_bound is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * Specifically, it returns the first position where value could be
 * inserted without violating the ordering. This version of 
 * \p lower_bound uses <tt>operator<</tt> for comparison and returns
 * the furthermost iterator \c i in <tt>[first, last)</tt> such that,
 * for every iterator \c j in <tt>[first, i)</tt>, <tt>*j < value</tt>. 
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \return The furthermost iterator \c i, such that <tt>*i < value</tt>.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam LessThanComparable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
 *
 *  The following code snippet demonstrates how to use \p lower_bound
 *  to search for values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::lower_bound(input.begin(), input.end(), 0); // returns input.begin()
 *  thrust::lower_bound(input.begin(), input.end(), 1); // returns input.begin() + 1
 *  thrust::lower_bound(input.begin(), input.end(), 2); // returns input.begin() + 1
 *  thrust::lower_bound(input.begin(), input.end(), 3); // returns input.begin() + 2
 *  thrust::lower_bound(input.begin(), input.end(), 8); // returns input.begin() + 4
 *  thrust::lower_bound(input.begin(), input.end(), 9); // returns input.end()
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <class ForwardIterator, class LessThanComparable>
ForwardIterator lower_bound(ForwardIterator first, 
                            ForwardIterator last,
                            const LessThanComparable& value);


/*! \p lower_bound is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * Specifically, it returns the first position where value could be
 * inserted without violating the ordering. This version of 
 * \p lower_bound uses function object \c comp for comparison 
 * and returns the furthermost iterator \c i in <tt>[first, last)</tt>
 * such that, for every iterator \c j in <tt>[first, i)</tt>, 
 * <tt>comp(*j, value)</tt> is \c true. 
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \param comp The comparison operator.
 *  \return The furthermost iterator \c i, such that <tt>comp(*i, value)</tt> is \c true.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam T is comparable to \p ForwardIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p lower_bound
 *  to search for values in a ordered range using the \p thrust::device execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::lower_bound(input.begin(), input.end(), 0, thrust::less<int>()); // returns input.begin()
 *  thrust::lower_bound(input.begin(), input.end(), 1, thrust::less<int>()); // returns input.begin() + 1
 *  thrust::lower_bound(input.begin(), input.end(), 2, thrust::less<int>()); // returns input.begin() + 1
 *  thrust::lower_bound(input.begin(), input.end(), 3, thrust::less<int>()); // returns input.begin() + 2
 *  thrust::lower_bound(input.begin(), input.end(), 8, thrust::less<int>()); // returns input.begin() + 4
 *  thrust::lower_bound(input.begin(), input.end(), 9, thrust::less<int>()); // returns input.end()
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template<typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__host__ __device__
ForwardIterator lower_bound(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T &value,
                            StrictWeakOrdering comp);


/*! \p lower_bound is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * Specifically, it returns the first position where value could be
 * inserted without violating the ordering. This version of 
 * \p lower_bound uses function object \c comp for comparison 
 * and returns the furthermost iterator \c i in <tt>[first, last)</tt>
 * such that, for every iterator \c j in <tt>[first, i)</tt>, 
 * <tt>comp(*j, value)</tt> is \c true. 
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \param comp The comparison operator.
 *  \return The furthermost iterator \c i, such that <tt>comp(*i, value)</tt> is \c true.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam T is comparable to \p ForwardIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p lower_bound
 *  to search for values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::lower_bound(input.begin(), input.end(), 0, thrust::less<int>()); // returns input.begin()
 *  thrust::lower_bound(input.begin(), input.end(), 1, thrust::less<int>()); // returns input.begin() + 1
 *  thrust::lower_bound(input.begin(), input.end(), 2, thrust::less<int>()); // returns input.begin() + 1
 *  thrust::lower_bound(input.begin(), input.end(), 3, thrust::less<int>()); // returns input.begin() + 2
 *  thrust::lower_bound(input.begin(), input.end(), 8, thrust::less<int>()); // returns input.begin() + 4
 *  thrust::lower_bound(input.begin(), input.end(), 9, thrust::less<int>()); // returns input.end()
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& value, 
                            StrictWeakOrdering comp);


/*! \p upper_bound is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * Specifically, it returns the last position where value could be
 * inserted without violating the ordering. This version of 
 * \p upper_bound uses <tt>operator<</tt> for comparison and returns
 * the furthermost iterator \c i in <tt>[first, last)</tt> such that,
 * for every iterator \c j in <tt>[first, i)</tt>, <tt>value < *j</tt>
 * is \c false.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \return The furthermost iterator \c i, such that <tt>value < *i</tt> is \c false.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam LessThanComparable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
 *
 *  The following code snippet demonstrates how to use \p upper_bound
 *  to search for values in a ordered range using the \p thrust::device execution policy for parallelism:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 0); // returns input.begin() + 1
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 1); // returns input.begin() + 1
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 2); // returns input.begin() + 2
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 3); // returns input.begin() + 2
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 8); // returns input.end()
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 9); // returns input.end()
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 *  \see \p lower_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template<typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable>
__host__ __device__
ForwardIterator upper_bound(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            const LessThanComparable &value);


/*! \p upper_bound is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * Specifically, it returns the last position where value could be
 * inserted without violating the ordering. This version of 
 * \p upper_bound uses <tt>operator<</tt> for comparison and returns
 * the furthermost iterator \c i in <tt>[first, last)</tt> such that,
 * for every iterator \c j in <tt>[first, i)</tt>, <tt>value < *j</tt>
 * is \c false.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \return The furthermost iterator \c i, such that <tt>value < *i</tt> is \c false.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam LessThanComparable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
 *
 *  The following code snippet demonstrates how to use \p upper_bound
 *  to search for values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::upper_bound(input.begin(), input.end(), 0); // returns input.begin() + 1
 *  thrust::upper_bound(input.begin(), input.end(), 1); // returns input.begin() + 1
 *  thrust::upper_bound(input.begin(), input.end(), 2); // returns input.begin() + 2
 *  thrust::upper_bound(input.begin(), input.end(), 3); // returns input.begin() + 2
 *  thrust::upper_bound(input.begin(), input.end(), 8); // returns input.end()
 *  thrust::upper_bound(input.begin(), input.end(), 9); // returns input.end()
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 *  \see \p lower_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <class ForwardIterator, class LessThanComparable>
ForwardIterator upper_bound(ForwardIterator first, 
                            ForwardIterator last,
                            const LessThanComparable& value);


/*! \p upper_bound is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * Specifically, it returns the last position where value could be
 * inserted without violating the ordering. This version of 
 * \p upper_bound uses function object \c comp for comparison and returns
 * the furthermost iterator \c i in <tt>[first, last)</tt> such that,
 * for every iterator \c j in <tt>[first, i)</tt>, <tt>comp(value, *j)</tt>
 * is \c false.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \param comp The comparison operator.
 *  \return The furthermost iterator \c i, such that <tt>comp(value, *i)</tt> is \c false.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam T is comparable to \p ForwardIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p upper_bound
 *  to search for values in a ordered range using the \p thrust::device execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 0, thrust::less<int>()); // returns input.begin() + 1
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 1, thrust::less<int>()); // returns input.begin() + 1
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 2, thrust::less<int>()); // returns input.begin() + 2
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 3, thrust::less<int>()); // returns input.begin() + 2
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 8, thrust::less<int>()); // returns input.end()
 *  thrust::upper_bound(thrust::device, input.begin(), input.end(), 9, thrust::less<int>()); // returns input.end()
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 *  \see \p lower_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template<typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__host__ __device__
ForwardIterator upper_bound(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T &value,
                            StrictWeakOrdering comp);

/*! \p upper_bound is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * Specifically, it returns the last position where value could be
 * inserted without violating the ordering. This version of 
 * \p upper_bound uses function object \c comp for comparison and returns
 * the furthermost iterator \c i in <tt>[first, last)</tt> such that,
 * for every iterator \c j in <tt>[first, i)</tt>, <tt>comp(value, *j)</tt>
 * is \c false.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \param comp The comparison operator.
 *  \return The furthermost iterator \c i, such that <tt>comp(value, *i)</tt> is \c false.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam T is comparable to \p ForwardIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p upper_bound
 *  to search for values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::upper_bound(input.begin(), input.end(), 0, thrust::less<int>()); // returns input.begin() + 1
 *  thrust::upper_bound(input.begin(), input.end(), 1, thrust::less<int>()); // returns input.begin() + 1
 *  thrust::upper_bound(input.begin(), input.end(), 2, thrust::less<int>()); // returns input.begin() + 2
 *  thrust::upper_bound(input.begin(), input.end(), 3, thrust::less<int>()); // returns input.begin() + 2
 *  thrust::upper_bound(input.begin(), input.end(), 8, thrust::less<int>()); // returns input.end()
 *  thrust::upper_bound(input.begin(), input.end(), 9, thrust::less<int>()); // returns input.end()
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 *  \see \p lower_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& value, 
                            StrictWeakOrdering comp);


/*! \p binary_search is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * It returns \c true if an element that is equivalent to \c value 
 * is present in <tt>[first, last)</tt> and \c false if no such element
 * exists.  Specifically, this version returns \c true if and only if 
 * there exists an iterator \c i in <tt>[first, last)</tt> such that 
 * <tt>*i < value</tt> and <tt>value < *i</tt> are both \c false.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \return \c true if an equivalent element exists in <tt>[first, last)</tt>, otherwise \c false.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam LessThanComparable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
 *
 *  The following code snippet demonstrates how to use \p binary_search
 *  to search for values in a ordered range using the \p thrust::device execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 0); // returns true
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 1); // returns false
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 2); // returns true
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 3); // returns false
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 8); // returns true
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 9); // returns false
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/binary_search
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 */
template <typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable>
__host__ __device__
bool binary_search(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   ForwardIterator first, 
                   ForwardIterator last,
                   const LessThanComparable& value);


/*! \p binary_search is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * It returns \c true if an element that is equivalent to \c value 
 * is present in <tt>[first, last)</tt> and \c false if no such element
 * exists.  Specifically, this version returns \c true if and only if 
 * there exists an iterator \c i in <tt>[first, last)</tt> such that 
 * <tt>*i < value</tt> and <tt>value < *i</tt> are both \c false.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \return \c true if an equivalent element exists in <tt>[first, last)</tt>, otherwise \c false.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam LessThanComparable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
 *
 *  The following code snippet demonstrates how to use \p binary_search
 *  to search for values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::binary_search(input.begin(), input.end(), 0); // returns true
 *  thrust::binary_search(input.begin(), input.end(), 1); // returns false
 *  thrust::binary_search(input.begin(), input.end(), 2); // returns true
 *  thrust::binary_search(input.begin(), input.end(), 3); // returns false
 *  thrust::binary_search(input.begin(), input.end(), 8); // returns true
 *  thrust::binary_search(input.begin(), input.end(), 9); // returns false
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/binary_search
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 */
template <class ForwardIterator, class LessThanComparable>
bool binary_search(ForwardIterator first, 
                   ForwardIterator last,
                   const LessThanComparable& value);


/*! \p binary_search is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * It returns \c true if an element that is equivalent to \c value 
 * is present in <tt>[first, last)</tt> and \c false if no such element
 * exists.  Specifically, this version returns \c true if and only if 
 * there exists an iterator \c i in <tt>[first, last)</tt> such that 
 * <tt>comp(*i, value)</tt> and <tt>comp(value, *i)</tt> are both \c false.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \param comp The comparison operator.
 *  \return \c true if an equivalent element exists in <tt>[first, last)</tt>, otherwise \c false.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam T is comparable to \p ForwardIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p binary_search
 *  to search for values in a ordered range using the \p thrust::device execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 0, thrust::less<int>()); // returns true
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 1, thrust::less<int>()); // returns false
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 2, thrust::less<int>()); // returns true
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 3, thrust::less<int>()); // returns false
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 8, thrust::less<int>()); // returns true
 *  thrust::binary_search(thrust::device, input.begin(), input.end(), 9, thrust::less<int>()); // returns false
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/binary_search
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 */
template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__host__ __device__
bool binary_search(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   ForwardIterator first,
                   ForwardIterator last,
                   const T& value, 
                   StrictWeakOrdering comp);


/*! \p binary_search is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. 
 * It returns \c true if an element that is equivalent to \c value 
 * is present in <tt>[first, last)</tt> and \c false if no such element
 * exists.  Specifically, this version returns \c true if and only if 
 * there exists an iterator \c i in <tt>[first, last)</tt> such that 
 * <tt>comp(*i, value)</tt> and <tt>comp(value, *i)</tt> are both \c false.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \param comp The comparison operator.
 *  \return \c true if an equivalent element exists in <tt>[first, last)</tt>, otherwise \c false.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam T is comparable to \p ForwardIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p binary_search
 *  to search for values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::binary_search(input.begin(), input.end(), 0, thrust::less<int>()); // returns true
 *  thrust::binary_search(input.begin(), input.end(), 1, thrust::less<int>()); // returns false
 *  thrust::binary_search(input.begin(), input.end(), 2, thrust::less<int>()); // returns true
 *  thrust::binary_search(input.begin(), input.end(), 3, thrust::less<int>()); // returns false
 *  thrust::binary_search(input.begin(), input.end(), 8, thrust::less<int>()); // returns true
 *  thrust::binary_search(input.begin(), input.end(), 9, thrust::less<int>()); // returns false
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/binary_search
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 */
template <class ForwardIterator, class T, class StrictWeakOrdering>
bool binary_search(ForwardIterator first,
                   ForwardIterator last,
                   const T& value, 
                   StrictWeakOrdering comp);


/*! \p equal_range is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. The 
 * value returned by \p equal_range is essentially a combination of
 * the values returned by \p lower_bound and \p upper_bound: it returns
 * a \p pair of iterators \c i and \c j such that \c i is the first
 * position where value could be inserted without violating the 
 * ordering and \c j is the last position where value could be inserted
 * without violating the ordering. It follows that every element in the
 * range <tt>[i, j)</tt> is equivalent to value, and that 
 * <tt>[i, j)</tt> is the largest subrange of <tt>[first, last)</tt> that
 * has this property. 
 *
 * This version of \p equal_range returns a \p pair of iterators 
 * <tt>[i, j)</tt>, where \c i is the furthermost iterator in 
 * <tt>[first, last)</tt> such that, for every iterator \c k in 
 * <tt>[first, i)</tt>, <tt>*k < value</tt>.  \c j is the furthermost
 * iterator in <tt>[first, last)</tt> such that, for every iterator 
 * \c k in <tt>[first, j)</tt>, <tt>value < *k</tt> is \c false. 
 * For every iterator \c k in <tt>[i, j)</tt>, neither 
 * <tt>value < *k</tt> nor <tt>*k < value</tt> is \c true.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \return A \p pair of iterators <tt>[i, j)</tt> that define the range of equivalent elements.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam LessThanComparable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
 *
 *  The following code snippet demonstrates how to use \p equal_range
 *  to search for values in a ordered range using the \p thrust::device execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 0); // returns [input.begin(), input.begin() + 1)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 1); // returns [input.begin() + 1, input.begin() + 1)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 2); // returns [input.begin() + 1, input.begin() + 2)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 3); // returns [input.begin() + 2, input.begin() + 2)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 8); // returns [input.begin() + 4, input.end)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 9); // returns [input.end(), input.end)
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/equal_range
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p binary_search
 */
template <typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable>
__host__ __device__
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable& value);


/*! \p equal_range is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. The 
 * value returned by \p equal_range is essentially a combination of
 * the values returned by \p lower_bound and \p upper_bound: it returns
 * a \p pair of iterators \c i and \c j such that \c i is the first
 * position where value could be inserted without violating the 
 * ordering and \c j is the last position where value could be inserted
 * without violating the ordering. It follows that every element in the
 * range <tt>[i, j)</tt> is equivalent to value, and that 
 * <tt>[i, j)</tt> is the largest subrange of <tt>[first, last)</tt> that
 * has this property. 
 *
 * This version of \p equal_range returns a \p pair of iterators 
 * <tt>[i, j)</tt>, where \c i is the furthermost iterator in 
 * <tt>[first, last)</tt> such that, for every iterator \c k in 
 * <tt>[first, i)</tt>, <tt>*k < value</tt>.  \c j is the furthermost
 * iterator in <tt>[first, last)</tt> such that, for every iterator 
 * \c k in <tt>[first, j)</tt>, <tt>value < *k</tt> is \c false. 
 * For every iterator \c k in <tt>[i, j)</tt>, neither 
 * <tt>value < *k</tt> nor <tt>*k < value</tt> is \c true.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \return A \p pair of iterators <tt>[i, j)</tt> that define the range of equivalent elements.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam LessThanComparable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>. 
 *
 *  The following code snippet demonstrates how to use \p equal_range
 *  to search for values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::equal_range(input.begin(), input.end(), 0); // returns [input.begin(), input.begin() + 1)
 *  thrust::equal_range(input.begin(), input.end(), 1); // returns [input.begin() + 1, input.begin() + 1)
 *  thrust::equal_range(input.begin(), input.end(), 2); // returns [input.begin() + 1, input.begin() + 2)
 *  thrust::equal_range(input.begin(), input.end(), 3); // returns [input.begin() + 2, input.begin() + 2)
 *  thrust::equal_range(input.begin(), input.end(), 8); // returns [input.begin() + 4, input.end)
 *  thrust::equal_range(input.begin(), input.end(), 9); // returns [input.end(), input.end)
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/equal_range
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p binary_search
 */
template <class ForwardIterator, class LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable& value);


/*! \p equal_range is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. The 
 * value returned by \p equal_range is essentially a combination of
 * the values returned by \p lower_bound and \p upper_bound: it returns
 * a \p pair of iterators \c i and \c j such that \c i is the first
 * position where value could be inserted without violating the 
 * ordering and \c j is the last position where value could be inserted
 * without violating the ordering. It follows that every element in the
 * range <tt>[i, j)</tt> is equivalent to value, and that 
 * <tt>[i, j)</tt> is the largest subrange of <tt>[first, last)</tt> that
 * has this property. 
 *
 * This version of \p equal_range returns a \p pair of iterators 
 * <tt>[i, j)</tt>. \c i is the furthermost iterator in 
 * <tt>[first, last)</tt> such that, for every iterator \c k in 
 * <tt>[first, i)</tt>, <tt>comp(*k, value)</tt> is \c true.
 * \c j is the furthermost iterator in <tt>[first, last)</tt> such
 * that, for every iterator \c k in <tt>[first, last)</tt>, 
 * <tt>comp(value, *k)</tt> is \c false. For every iterator \c k 
 * in <tt>[i, j)</tt>, neither <tt>comp(value, *k)</tt> nor 
 * <tt>comp(*k, value)</tt> is \c true.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \param comp The comparison operator.
 *  \return A \p pair of iterators <tt>[i, j)</tt> that define the range of equivalent elements.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam T is comparable to \p ForwardIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p equal_range
 *  to search for values in a ordered range using the \p thrust::device execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 0, thrust::less<int>()); // returns [input.begin(), input.begin() + 1)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 1, thrust::less<int>()); // returns [input.begin() + 1, input.begin() + 1)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 2, thrust::less<int>()); // returns [input.begin() + 1, input.begin() + 2)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 3, thrust::less<int>()); // returns [input.begin() + 2, input.begin() + 2)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 8, thrust::less<int>()); // returns [input.begin() + 4, input.end)
 *  thrust::equal_range(thrust::device, input.begin(), input.end(), 9, thrust::less<int>()); // returns [input.end(), input.end)
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/equal_range
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p binary_search
 */
template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__host__ __device__
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const T& value,
            StrictWeakOrdering comp);


/*! \p equal_range is a version of binary search: it attempts to find
 * the element value in an ordered range <tt>[first, last)</tt>. The 
 * value returned by \p equal_range is essentially a combination of
 * the values returned by \p lower_bound and \p upper_bound: it returns
 * a \p pair of iterators \c i and \c j such that \c i is the first
 * position where value could be inserted without violating the 
 * ordering and \c j is the last position where value could be inserted
 * without violating the ordering. It follows that every element in the
 * range <tt>[i, j)</tt> is equivalent to value, and that 
 * <tt>[i, j)</tt> is the largest subrange of <tt>[first, last)</tt> that
 * has this property. 
 *
 * This version of \p equal_range returns a \p pair of iterators 
 * <tt>[i, j)</tt>. \c i is the furthermost iterator in 
 * <tt>[first, last)</tt> such that, for every iterator \c k in 
 * <tt>[first, i)</tt>, <tt>comp(*k, value)</tt> is \c true.
 * \c j is the furthermost iterator in <tt>[first, last)</tt> such
 * that, for every iterator \c k in <tt>[first, last)</tt>, 
 * <tt>comp(value, *k)</tt> is \c false. For every iterator \c k 
 * in <tt>[i, j)</tt>, neither <tt>comp(value, *k)</tt> nor 
 * <tt>comp(*k, value)</tt> is \c true.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param value The value to be searched.
 *  \param comp The comparison operator.
 *  \return A \p pair of iterators <tt>[i, j)</tt> that define the range of equivalent elements.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam T is comparable to \p ForwardIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p equal_range
 *  to search for values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::equal_range(input.begin(), input.end(), 0, thrust::less<int>()); // returns [input.begin(), input.begin() + 1)
 *  thrust::equal_range(input.begin(), input.end(), 1, thrust::less<int>()); // returns [input.begin() + 1, input.begin() + 1)
 *  thrust::equal_range(input.begin(), input.end(), 2, thrust::less<int>()); // returns [input.begin() + 1, input.begin() + 2)
 *  thrust::equal_range(input.begin(), input.end(), 3, thrust::less<int>()); // returns [input.begin() + 2, input.begin() + 2)
 *  thrust::equal_range(input.begin(), input.end(), 8, thrust::less<int>()); // returns [input.begin() + 4, input.end)
 *  thrust::equal_range(input.begin(), input.end(), 9, thrust::less<int>()); // returns [input.end(), input.end)
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/equal_range
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p binary_search
 */
template <class ForwardIterator, class T, class StrictWeakOrdering>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const T& value,
            StrictWeakOrdering comp);


/*! \addtogroup vectorized_binary_search Vectorized Searches
 *  \ingroup binary_search
 *  \{
 */


//////////////////////
// Vector Functions //
//////////////////////


/*! \p lower_bound is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * Specifically, it returns the index of first position where value could
 * be inserted without violating the ordering.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and \c ForwardIterator's difference_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p lower_bound
 *  to search for multiple values in a ordered range using the \p thrust::device execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<unsigned int> output(6);
 *
 *  thrust::lower_bound(thrust::device,
 *                      input.begin(), input.end(),
 *                      values.begin(), values.end(),
 *                      output.begin());
 *
 *  // output is now [0, 1, 1, 2, 4, 5]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator lower_bound(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator result);


/*! \p lower_bound is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * Specifically, it returns the index of first position where value could
 * be inserted without violating the ordering.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and \c ForwardIterator's difference_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p lower_bound
 *  to search for multiple values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<unsigned int> output(6);
 *
 *  thrust::lower_bound(input.begin(), input.end(),
 *                      values.begin(), values.end(),
 *                      output.begin());
 *
 *  // output is now [0, 1, 1, 2, 4, 5]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator lower_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator result);


/*! \p lower_bound is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * Specifically, it returns the index of first position where value could
 * be inserted without violating the ordering.  This version of 
 * \p lower_bound uses function object \c comp for comparison.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 *  \param comp The comparison operator.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is comparable to \p ForwardIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and \c ForwardIterator's difference_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p lower_bound
 *  to search for multiple values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<unsigned int> output(6);
 *
 *  thrust::lower_bound(input.begin(), input.end(),
 *                      values.begin(), values.end(), 
 *                      output.begin(),
 *                      thrust::less<int>());
 *
 *  // output is now [0, 1, 1, 2, 4, 5]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__host__ __device__
OutputIterator lower_bound(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator result,
                           StrictWeakOrdering comp);


/*! \p lower_bound is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * Specifically, it returns the index of first position where value could
 * be inserted without violating the ordering.  This version of 
 * \p lower_bound uses function object \c comp for comparison.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 *  \param comp The comparison operator.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is comparable to \p ForwardIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and \c ForwardIterator's difference_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p lower_bound
 *  to search for multiple values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<unsigned int> output(6);
 *
 *  thrust::lower_bound(input.begin(), input.end(),
 *                      values.begin(), values.end(), 
 *                      output.begin(),
 *                      thrust::less<int>());
 *
 *  // output is now [0, 1, 1, 2, 4, 5]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator result,
                           StrictWeakOrdering comp);


/*! \p upper_bound is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * Specifically, it returns the index of last position where value could
 * be inserted without violating the ordering.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and \c ForwardIterator's difference_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p upper_bound
 *  to search for multiple values in a ordered range using the \p thrust::device execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<unsigned int> output(6);
 *
 *  thrust::upper_bound(thrust::device,
 *                      input.begin(), input.end(),
 *                      values.begin(), values.end(),
 *                      output.begin());
 *
 *  // output is now [1, 1, 2, 2, 5, 5]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator upper_bound(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator result);


/*! \p upper_bound is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * Specifically, it returns the index of last position where value could
 * be inserted without violating the ordering.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and \c ForwardIterator's difference_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p upper_bound
 *  to search for multiple values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<unsigned int> output(6);
 *
 *  thrust::upper_bound(input.begin(), input.end(),
 *                      values.begin(), values.end(),
 *                      output.begin());
 *
 *  // output is now [1, 1, 2, 2, 5, 5]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator upper_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator result);


/*! \p upper_bound is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * Specifically, it returns the index of first position where value could
 * be inserted without violating the ordering.  This version of 
 * \p upper_bound uses function object \c comp for comparison.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 *  \param comp The comparison operator.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is comparable to \p ForwardIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and \c ForwardIterator's difference_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p upper_bound
 *  to search for multiple values in a ordered range using the \p thrust::device execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<unsigned int> output(6);
 *
 *  thrust::upper_bound(thrust::device,
 *                      input.begin(), input.end(),
 *                      values.begin(), values.end(), 
 *                      output.begin(),
 *                      thrust::less<int>());
 *
 *  // output is now [1, 1, 2, 2, 5, 5]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 *  \see \p lower_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__host__ __device__
OutputIterator upper_bound(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator result,
                           StrictWeakOrdering comp);


/*! \p upper_bound is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * Specifically, it returns the index of first position where value could
 * be inserted without violating the ordering.  This version of 
 * \p upper_bound uses function object \c comp for comparison.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 *  \param comp The comparison operator.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is comparable to \p ForwardIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and \c ForwardIterator's difference_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p upper_bound
 *  to search for multiple values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<unsigned int> output(6);
 *
 *  thrust::upper_bound(input.begin(), input.end(),
 *                      values.begin(), values.end(), 
 *                      output.begin(),
 *                      thrust::less<int>());
 *
 *  // output is now [1, 1, 2, 2, 5, 5]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/upper_bound
 *  \see \p lower_bound
 *  \see \p equal_range
 *  \see \p binary_search
 */
template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator result,
                           StrictWeakOrdering comp);


/*! \p binary_search is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * It returns \c true if an element that is equivalent to \c value 
 * is present in <tt>[first, last)</tt> and \c false if no such element
 * exists.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and bool is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p binary_search
 *  to search for multiple values in a ordered range using the \p thrust::device execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<bool> output(6);
 *
 *  thrust::binary_search(thrust::device,
 *                        input.begin(), input.end(),
 *                        values.begin(), values.end(),
 *                        output.begin());
 *
 *  // output is now [true, false, true, false, true, false]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/binary_search
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 */
template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator binary_search(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                             ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator result);


/*! \p binary_search is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * It returns \c true if an element that is equivalent to \c value 
 * is present in <tt>[first, last)</tt> and \c false if no such element
 * exists.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and bool is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p binary_search
 *  to search for multiple values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<bool> output(6);
 *
 *  thrust::binary_search(input.begin(), input.end(),
 *                        values.begin(), values.end(),
 *                        output.begin());
 *
 *  // output is now [true, false, true, false, true, false]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/binary_search
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 */
template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator binary_search(ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator result);


/*! \p binary_search is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * It returns \c true if an element that is equivalent to \c value 
 * is present in <tt>[first, last)</tt> and \c false if no such element
 * exists.  This version of \p binary_search uses function object 
 * \c comp for comparison.
 *
 * The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 *  \param comp The comparison operator.
 * 
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and bool is convertible to \c OutputIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p binary_search
 *  to search for multiple values in a ordered range using the \p thrust::device execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<bool> output(6);
 *
 *  thrust::binary_search(thrust::device,
 *                        input.begin(), input.end(),
 *                        values.begin(), values.end(),
 *                        output.begin(),
 *                        thrust::less<T>());
 *
 *  // output is now [true, false, true, false, true, false]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/binary_search
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 */
template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__host__ __device__
OutputIterator binary_search(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                             ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator result,
                             StrictWeakOrdering comp);


/*! \p binary_search is a vectorized version of binary search: for each 
 * iterator \c v in <tt>[values_first, values_last)</tt> it attempts to
 * find the value <tt>*v</tt> in an ordered range <tt>[first, last)</tt>.
 * It returns \c true if an element that is equivalent to \c value 
 * is present in <tt>[first, last)</tt> and \c false if no such element
 * exists.  This version of \p binary_search uses function object 
 * \c comp for comparison.
 *
 *  \param first The beginning of the ordered sequence.
 *  \param last The end of the ordered sequence.
 *  \param values_first The beginning of the search values sequence.
 *  \param values_last The end of the search values sequence.
 *  \param result The beginning of the output sequence.
 *  \param comp The comparison operator.
 * 
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *                        and \c InputIterator's \c value_type is <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThanComparable</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *                        and bool is convertible to \c OutputIterator's \c value_type.
 *  \tparam StrictWeakOrdering is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p binary_search
 *  to search for multiple values in a ordered range.
 *
 *  \code
 *  #include <thrust/binary_search.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<int> input(5);
 *
 *  input[0] = 0;
 *  input[1] = 2;
 *  input[2] = 5;
 *  input[3] = 7;
 *  input[4] = 8;
 *
 *  thrust::device_vector<int> values(6);
 *  values[0] = 0; 
 *  values[1] = 1;
 *  values[2] = 2;
 *  values[3] = 3;
 *  values[4] = 8;
 *  values[5] = 9;
 *
 *  thrust::device_vector<bool> output(6);
 *
 *  thrust::binary_search(input.begin(), input.end(),
 *                        values.begin(), values.end(),
 *                        output.begin(),
 *                        thrust::less<T>());
 *
 *  // output is now [true, false, true, false, true, false]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/binary_search
 *  \see \p lower_bound
 *  \see \p upper_bound
 *  \see \p equal_range
 */
template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator result,
                             StrictWeakOrdering comp);


/*! \} // end vectorized_binary_search
 */


/*! \} // end binary_search
 */


/*! \} // end searching
 */

THRUST_NAMESPACE_END

#include <thrust/detail/binary_search.inl>

