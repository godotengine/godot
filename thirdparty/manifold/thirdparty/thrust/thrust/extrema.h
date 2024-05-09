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

/*! \file extrema.h
 *  \brief Functions for computing computing extremal values
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! This version of \p min returns the smaller of two values, given a comparison operation.
 *  \param lhs The first value to compare.
 *  \param rhs The second value to compare.
 *  \param comp A comparison operation.
 *  \return The smaller element.
 *
 *  \tparam T is convertible to \p BinaryPredicate's first argument type and to its second argument type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>.
 *
 *  The following code snippet demonstrates how to use \p min to compute the smaller of two
 *  key-value objects.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  ...
 *  struct key_value
 *  {
 *    int key;
 *    int value;
 *  };
 *
 *  struct compare_key_value
 *  {
 *    __host__ __device__
 *    bool operator()(key_value lhs, key_value rhs)
 *    {
 *      return lhs.key < rhs.key;
 *    }
 *  };
 *
 *  ...
 *  key_value a = {13, 0};
 *  key_value b = { 7, 1);
 *
 *  key_value smaller = thrust::min(a, b, compare_key_value());
 *
 *  // smaller is {7, 1}
 *  \endcode
 *
 *  \note Returns the first argument when the arguments are equivalent.
 *  \see max
 */
template<typename T, typename BinaryPredicate>
__host__ __device__
  T min THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs, BinaryPredicate comp);


/*! This version of \p min returns the smaller of two values.
 *  \param lhs The first value to compare.
 *  \param rhs The second value to compare.
 *  \return The smaller element.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  The following code snippet demonstrates how to use \p min to compute the smaller of two
 *  integers.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  ...
 *  int a = 13;
 *  int b = 7;
 *
 *  int smaller = thrust::min(a, b);
 *
 *  // smaller is 7
 *  \endcode
 *
 *  \note Returns the first argument when the arguments are equivalent.
 *  \see max
 */
template<typename T>
__host__ __device__
  T min THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs);


/*! This version of \p max returns the larger of two values, given a comparison operation.
 *  \param lhs The first value to compare.
 *  \param rhs The second value to compare.
 *  \param comp A comparison operation.
 *  \return The larger element.
 *
 *  \tparam T is convertible to \p BinaryPredicate's first argument type and to its second argument type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a>.
 *
 *  The following code snippet demonstrates how to use \p max to compute the larger of two
 *  key-value objects.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  ...
 *  struct key_value
 *  {
 *    int key;
 *    int value;
 *  };
 *
 *  struct compare_key_value
 *  {
 *    __host__ __device__
 *    bool operator()(key_value lhs, key_value rhs)
 *    {
 *      return lhs.key < rhs.key;
 *    }
 *  };
 *
 *  ...
 *  key_value a = {13, 0};
 *  key_value b = { 7, 1);
 *
 *  key_value larger = thrust::max(a, b, compare_key_value());
 *
 *  // larger is {13, 0}
 *  \endcode
 *
 *  \note Returns the first argument when the arguments are equivalent.
 *  \see min
 */
template<typename T, typename BinaryPredicate>
__host__ __device__
  T max THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs, BinaryPredicate comp);


/*! This version of \p max returns the larger of two values.
 *  \param lhs The first value to compare.
 *  \param rhs The second value to compare.
 *  \return The larger element.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  The following code snippet demonstrates how to use \p max to compute the larger of two
 *  integers.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  ...
 *  int a = 13;
 *  int b = 7;
 *
 *  int larger = thrust::min(a, b);
 *
 *  // larger is 13
 *  \endcode
 *
 *  \note Returns the first argument when the arguments are equivalent.
 *  \see min
 */
template<typename T>
__host__ __device__
  T max THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs);


/*! \addtogroup reductions
 *  \{
 *  \addtogroup extrema
 *  \ingroup reductions
 *  \{
 */

/*! \p min_element finds the smallest element in the range <tt>[first, last)</tt>.
 *  It returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that no other iterator in <tt>[first, last)</tt> points to a value smaller
 *  than \c *i. The return value is \p last if and only if <tt>[first, last)</tt> is an
 *  empty range.
 *
 *  The two versions of \p min_element differ in how they define whether one element is
 *  less than another. This version compares objects using \c operator<. Specifically,
 *  this version of \p min_element returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that, for every iterator \c j in <tt>[first, last)</tt>, <tt>*j < *i</tt> is
 *  \c false.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return An iterator pointing to the smallest element of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \c ForwardIterator's \c value_type is a model of
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int *result = thrust::min_element(thrust::host, data, data + 6);
 *
 *  // result is data + 1
 *  // *result is 0
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/min_element 
 */
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator min_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last);


/*! \p min_element finds the smallest element in the range <tt>[first, last)</tt>.
 *  It returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that no other iterator in <tt>[first, last)</tt> points to a value smaller
 *  than \c *i. The return value is \p last if and only if <tt>[first, last)</tt> is an
 *  empty range.
 *
 *  The two versions of \p min_element differ in how they define whether one element is
 *  less than another. This version compares objects using \c operator<. Specifically,
 *  this version of \p min_element returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that, for every iterator \c j in <tt>[first, last)</tt>, <tt>*j < *i</tt> is
 *  \c false.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return An iterator pointing to the smallest element of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \c ForwardIterator's \c value_type is a model of
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int *result = thrust::min_element(data, data + 6);
 *
 *  // result is data + 1
 *  // *result is 0
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/min_element 
 */
template <typename ForwardIterator>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last);


/*! \p min_element finds the smallest element in the range <tt>[first, last)</tt>.
 *  It returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that no other iterator in <tt>[first, last)</tt> points to a value smaller
 *  than \c *i. The return value is \p last if and only if <tt>[first, last)</tt> is an
 *  empty range.
 *
 *  The two versions of \p min_element differ in how they define whether one element is
 *  less than another. This version compares objects using a function object \p comp.
 *  Specifically, this version of \p min_element returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that, for every iterator \c j in <tt>[first, last)</tt>, <tt>comp(*j, *i)</tt> is
 *  \c false.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp A binary predicate used for comparison.
 *  \return An iterator pointing to the smallest element of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \p comp's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p min_element to find the smallest element
 *  of a collection of key-value pairs using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  struct key_value
 *  {
 *    int key;
 *    int value;
 *  };
 *
 *  struct compare_key_value
 *  {
 *    __host__ __device__
 *    bool operator()(key_value lhs, key_value rhs)
 *    {
 *      return lhs.key < rhs.key;
 *    }
 *  };
 *
 *  ...
 *  key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };
 *
 *  key_value *smallest = thrust::min_element(thrust::host, data, data + 4, compare_key_value());
 *
 *  // smallest == data + 1
 *  // *smallest == {0,7}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/min_element 
 */
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator min_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp);


/*! \p min_element finds the smallest element in the range <tt>[first, last)</tt>.
 *  It returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that no other iterator in <tt>[first, last)</tt> points to a value smaller
 *  than \c *i. The return value is \p last if and only if <tt>[first, last)</tt> is an
 *  empty range.
 *
 *  The two versions of \p min_element differ in how they define whether one element is
 *  less than another. This version compares objects using a function object \p comp.
 *  Specifically, this version of \p min_element returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that, for every iterator \c j in <tt>[first, last)</tt>, <tt>comp(*j, *i)</tt> is
 *  \c false.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp A binary predicate used for comparison.
 *  \return An iterator pointing to the smallest element of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \p comp's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p min_element to find the smallest element
 *  of a collection of key-value pairs.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *
 *  struct key_value
 *  {
 *    int key;
 *    int value;
 *  };
 *
 *  struct compare_key_value
 *  {
 *    __host__ __device__
 *    bool operator()(key_value lhs, key_value rhs)
 *    {
 *      return lhs.key < rhs.key;
 *    }
 *  };
 *
 *  ...
 *  key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };
 *
 *  key_value *smallest = thrust::min_element(data, data + 4, compare_key_value());
 *
 *  // smallest == data + 1
 *  // *smallest == {0,7}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/min_element 
 */
template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp);


/*! \p max_element finds the largest element in the range <tt>[first, last)</tt>.
 *  It returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that no other iterator in <tt>[first, last)</tt> points to a value larger
 *  than \c *i. The return value is \p last if and only if <tt>[first, last)</tt> is an
 *  empty range.
 *
 *  The two versions of \p max_element differ in how they define whether one element is
 *  greater than another. This version compares objects using \c operator<. Specifically,
 *  this version of \p max_element returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that, for every iterator \c j in <tt>[first, last)</tt>, <tt>*i < *j</tt> is
 *  \c false.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return An iterator pointing to the largest element of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam A Thrust backend system.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \c ForwardIterator's \c value_type is a model of
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int *result = thrust::max_element(thrust::host, data, data + 6);
 *
 *  // *result == 3
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/max_element 
 */
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator max_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last);


/*! \p max_element finds the largest element in the range <tt>[first, last)</tt>.
 *  It returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that no other iterator in <tt>[first, last)</tt> points to a value larger
 *  than \c *i. The return value is \p last if and only if <tt>[first, last)</tt> is an
 *  empty range.
 *
 *  The two versions of \p max_element differ in how they define whether one element is
 *  greater than another. This version compares objects using \c operator<. Specifically,
 *  this version of \p max_element returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that, for every iterator \c j in <tt>[first, last)</tt>, <tt>*i < *j</tt> is
 *  \c false.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return An iterator pointing to the largest element of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \c ForwardIterator's \c value_type is a model of
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int *result = thrust::max_element(data, data + 6);
 *
 *  // *result == 3
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/max_element 
 */
template <typename ForwardIterator>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last);


/*! \p max_element finds the largest element in the range <tt>[first, last)</tt>.
 *  It returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that no other iterator in <tt>[first, last)</tt> points to a value larger
 *  than \c *i. The return value is \p last if and only if <tt>[first, last)</tt> is an
 *  empty range.
 *
 *  The two versions of \p max_element differ in how they define whether one element is
 *  less than another. This version compares objects using a function object \p comp.
 *  Specifically, this version of \p max_element returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that, for every iterator \c j in <tt>[first, last)</tt>, <tt>comp(*i, *j)</tt> is
 *  \c false.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp A binary predicate used for comparison.
 *  \return An iterator pointing to the largest element of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \p comp's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p max_element to find the largest element
 *  of a collection of key-value pairs using the \p thrust::host execution policy for parallelization.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  struct key_value
 *  {
 *    int key;
 *    int value;
 *  };
 *
 *  struct compare_key_value
 *  {
 *    __host__ __device__
 *    bool operator()(key_value lhs, key_value rhs)
 *    {
 *      return lhs.key < rhs.key;
 *    }
 *  };
 *
 *  ...
 *  key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };
 *
 *  key_value *largest = thrust::max_element(thrust::host, data, data + 4, compare_key_value());
 *
 *  // largest == data + 3
 *  // *largest == {6,1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/max_element 
 */
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator max_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp);


/*! \p max_element finds the largest element in the range <tt>[first, last)</tt>.
 *  It returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that no other iterator in <tt>[first, last)</tt> points to a value larger
 *  than \c *i. The return value is \p last if and only if <tt>[first, last)</tt> is an
 *  empty range.
 *
 *  The two versions of \p max_element differ in how they define whether one element is
 *  less than another. This version compares objects using a function object \p comp.
 *  Specifically, this version of \p max_element returns the first iterator \c i in <tt>[first, last)</tt>
 *  such that, for every iterator \c j in <tt>[first, last)</tt>, <tt>comp(*i, *j)</tt> is
 *  \c false.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp A binary predicate used for comparison.
 *  \return An iterator pointing to the largest element of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \p comp's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p max_element to find the largest element
 *  of a collection of key-value pairs.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *
 *  struct key_value
 *  {
 *    int key;
 *    int value;
 *  };
 *
 *  struct compare_key_value
 *  {
 *    __host__ __device__
 *    bool operator()(key_value lhs, key_value rhs)
 *    {
 *      return lhs.key < rhs.key;
 *    }
 *  };
 *
 *  ...
 *  key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };
 *
 *  key_value *largest = thrust::max_element(data, data + 4, compare_key_value());
 *
 *  // largest == data + 3
 *  // *largest == {6,1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/max_element 
 */
template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp);


/*! \p minmax_element finds the smallest and largest elements in the range <tt>[first, last)</tt>.
 *  It returns a pair of iterators <tt>(imin, imax)</tt> where \c imin is the same iterator
 *  returned by \p min_element and \c imax is the same iterator returned by \p max_element.
 *  This function is potentially more efficient than separate calls to \p min_element and \p max_element.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return A pair of iterator pointing to the smallest and largest elements of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \c ForwardIterator's \c value_type is a model of
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);
 *
 *  // result.first is data + 1
 *  // result.second is data + 5
 *  // *result.first is 0
 *  // *result.second is 3
 *  \endcode
 *
 *  \see min_element
 *  \see max_element
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf
 */
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last);


/*! \p minmax_element finds the smallest and largest elements in the range <tt>[first, last)</tt>.
 *  It returns a pair of iterators <tt>(imin, imax)</tt> where \c imin is the same iterator
 *  returned by \p min_element and \c imax is the same iterator returned by \p max_element.
 *  This function is potentially more efficient than separate calls to \p min_element and \p max_element.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \return A pair of iterator pointing to the smallest and largest elements of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \c ForwardIterator's \c value_type is a model of
 *          <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  thrust::pair<int *, int *> result = thrust::minmax_element(data, data + 6);
 *
 *  // result.first is data + 1
 *  // result.second is data + 5
 *  // *result.first is 0
 *  // *result.second is 3
 *  \endcode
 *
 *  \see min_element
 *  \see max_element
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf
 */
template <typename ForwardIterator>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last);


/*! \p minmax_element finds the smallest and largest elements in the range <tt>[first, last)</tt>.
 *  It returns a pair of iterators <tt>(imin, imax)</tt> where \c imin is the same iterator
 *  returned by \p min_element and \c imax is the same iterator returned by \p max_element.
 *  This function is potentially more efficient than separate calls to \p min_element and \p max_element.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp A binary predicate used for comparison.
 *  \return A pair of iterator pointing to the smallest and largest elements of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \p comp's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p minmax_element to find the smallest and largest elements
 *  of a collection of key-value pairs using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  #include <thrust/pair.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  struct key_value
 *  {
 *    int key;
 *    int value;
 *  };
 *
 *  struct compare_key_value
 *  {
 *    __host__ __device__
 *    bool operator()(key_value lhs, key_value rhs)
 *    {
 *      return lhs.key < rhs.key;
 *    }
 *  };
 *
 *  ...
 *  key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };
 *
 *  thrust::pair<key_value*,key_value*> extrema = thrust::minmax_element(thrust::host, data, data + 4, compare_key_value());
 *
 *  // extrema.first   == data + 1
 *  // *extrema.first  == {0,7}
 *  // extrema.second  == data + 3
 *  // *extrema.second == {6,1}
 *  \endcode
 *
 *  \see min_element
 *  \see max_element
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf
 */
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp);


/*! \p minmax_element finds the smallest and largest elements in the range <tt>[first, last)</tt>.
 *  It returns a pair of iterators <tt>(imin, imax)</tt> where \c imin is the same iterator
 *  returned by \p min_element and \c imax is the same iterator returned by \p max_element.
 *  This function is potentially more efficient than separate calls to \p min_element and \p max_element.
 *
 *  \param first The beginning of the sequence.
 *  \param last  The end of the sequence.
 *  \param comp A binary predicate used for comparison.
 *  \return A pair of iterator pointing to the smallest and largest elements of the range <tt>[first, last)</tt>,
 *          if it is not an empty range; \p last, otherwise.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to both \p comp's
 *          \c first_argument_type and \c second_argument_type.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p minmax_element to find the smallest and largest elements
 *  of a collection of key-value pairs.
 *
 *  \code
 *  #include <thrust/extrema.h>
 *  #include <thrust/pair.h>
 *
 *  struct key_value
 *  {
 *    int key;
 *    int value;
 *  };
 *
 *  struct compare_key_value
 *  {
 *    __host__ __device__
 *    bool operator()(key_value lhs, key_value rhs)
 *    {
 *      return lhs.key < rhs.key;
 *    }
 *  };
 *
 *  ...
 *  key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };
 *
 *  thrust::pair<key_value*,key_value*> extrema = thrust::minmax_element(data, data + 4, compare_key_value());
 *
 *  // extrema.first   == data + 1
 *  // *extrema.first  == {0,7}
 *  // extrema.second  == data + 3
 *  // *extrema.second == {6,1}
 *  \endcode
 *
 *  \see min_element
 *  \see max_element
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf
 */
template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp);

/*! \} // end extrema
 *  \} // end reductions
 */

THRUST_NAMESPACE_END

#include <thrust/detail/extrema.inl>
#include <thrust/detail/minmax.h>
