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


/*! \file count.h
 *  \brief Counting elements in a range
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup reductions
 *  \ingroup algorithms
 *  \{
 */

/*! \addtogroup counting
 *  \ingroup reductions
 *  \{
 */


/*! \p count finds the number of elements in <tt>[first,last)</tt> that are equal
 *  to \p value. More precisely, \p count returns the number of iterators \c i in
 *  <tt>[first, last)</tt> such that <tt>*i == value</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param value The value to be counted.
 *  \return The number of elements equal to \p value.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and \c InputIterator's \c value_type must be a model of must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam EqualityComparable must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a> and can be compared for equality with \c InputIterator's \c value_type
 *
 *  The following code snippet demonstrates how to use \p count to 
 *  count the number of instances in a range of a value of interest using the \p thrust::device execution policy:
 *
 *  \code
 *  #include <thrust/count.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  // put 3 1s in a device_vector
 *  thrust::device_vector<int> vec(5,0);
 *  vec[1] = 1;
 *  vec[3] = 1;
 *  vec[4] = 1;
 *  
 *  // count the 1s
 *  int result = thrust::count(thrust::device, vec.begin(), vec.end(), 1);
 *  // result == 3
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/count
 */
template<typename DerivedPolicy, typename InputIterator, typename EqualityComparable>
__host__ __device__
  typename thrust::iterator_traits<InputIterator>::difference_type
    count(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last, const EqualityComparable& value);



/*! \p count finds the number of elements in <tt>[first,last)</tt> that are equal
 *  to \p value. More precisely, \p count returns the number of iterators \c i in
 *  <tt>[first, last)</tt> such that <tt>*i == value</tt>.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param value The value to be counted.
 *  \return The number of elements equal to \p value.
 *
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and \c InputIterator's \c value_type must be a model of must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 *  \tparam EqualityComparable must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a> and can be compared for equality with \c InputIterator's \c value_type
 *
 *  The following code snippet demonstrates how to use \p count to 
 *  count the number of instances in a range of a value of interest.
 *  \code
 *  #include <thrust/count.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  // put 3 1s in a device_vector
 *  thrust::device_vector<int> vec(5,0);
 *  vec[1] = 1;
 *  vec[3] = 1;
 *  vec[4] = 1;
 *  
 *  // count the 1s
 *  int result = thrust::count(vec.begin(), vec.end(), 1);
 *  // result == 3
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/count
 */
template <typename InputIterator, typename EqualityComparable>
  typename thrust::iterator_traits<InputIterator>::difference_type
    count(InputIterator first, InputIterator last, const EqualityComparable& value);


/*! \p count_if finds the number of elements in <tt>[first,last)</tt> for which 
 *  a predicate is \c true. More precisely, \p count_if returns the number of iterators
 *  \c i in <tt>[first, last)</tt> such that <tt>pred(*i) == true</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param pred The predicate.
 *  \return The number of elements where \p pred is \c true.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and \c InputIterator's \c value_type must be convertible to \c Predicate's \c argument_type.
 *  \tparam Predicate must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p count to
 *  count the number of odd numbers in a range using the \p thrust::device execution policy:
 *
 *  \code
 *  #include <thrust/count.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_odd
 *  {
 *    __host__ __device__
 *    bool operator()(int &x)
 *    {
 *      return x & 1;
 *    }
 *  };
 *  ...
 *  // fill a device_vector with even & odd numbers
 *  thrust::device_vector<int> vec(5);
 *  vec[0] = 0;
 *  vec[1] = 1;
 *  vec[2] = 2;
 *  vec[3] = 3;
 *  vec[4] = 4;
 *
 *  // count the odd elements in vec
 *  int result = thrust::count_if(thrust::device, vec.begin(), vec.end(), is_odd());
 *  // result == 2
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/count
 */
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
  typename thrust::iterator_traits<InputIterator>::difference_type
    count_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last, Predicate pred);


/*! \p count_if finds the number of elements in <tt>[first,last)</tt> for which 
 *  a predicate is \c true. More precisely, \p count_if returns the number of iterators
 *  \c i in <tt>[first, last)</tt> such that <tt>pred(*i) == true</tt>.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param pred The predicate.
 *  \return The number of elements where \p pred is \c true.
 *
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and \c InputIterator's \c value_type must be convertible to \c Predicate's \c argument_type.
 *  \tparam Predicate must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p count to
 *  count the number of odd numbers in a range.
 *  \code
 *  #include <thrust/count.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  struct is_odd
 *  {
 *    __host__ __device__
 *    bool operator()(int &x)
 *    {
 *      return x & 1;
 *    }
 *  };
 *  ...
 *  // fill a device_vector with even & odd numbers
 *  thrust::device_vector<int> vec(5);
 *  vec[0] = 0;
 *  vec[1] = 1;
 *  vec[2] = 2;
 *  vec[3] = 3;
 *  vec[4] = 4;
 *
 *  // count the odd elements in vec
 *  int result = thrust::count_if(vec.begin(), vec.end(), is_odd());
 *  // result == 2
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/count
 */
template <typename InputIterator, typename Predicate>
  typename thrust::iterator_traits<InputIterator>::difference_type
    count_if(InputIterator first, InputIterator last, Predicate pred);


/*! \} // end counting
 *  \} // end reductions
 */

THRUST_NAMESPACE_END

#include <thrust/detail/count.h>
