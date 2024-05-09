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


/*! \file uninitialized_copy.h
 *  \brief Copy construction into a range of uninitialized elements from a source range
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup copying
 *  \{
 */


/*! In \c thrust, the function \c thrust::device_new allocates memory for
 *  an object and then creates an object at that location by calling a constructor.
 *  Occasionally, however, it is useful to separate those two operations.
 *  If each iterator in the range <tt>[result, result + (last - first))</tt> points
 *  to uninitialized memory, then \p uninitialized_copy creates a copy of
 *  <tt>[first, last)</tt> in that range. That is, for each iterator \c i in
 *  the input, \p uninitialized_copy creates a copy of \c *i in the location pointed
 *  to by the corresponding iterator in the output range by \p ForwardIterator's
 *  \c value_type's copy constructor with *i as its argument.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The first element of the input range to copy from.
 *  \param last The last element of the input range to copy from.
 *  \param result The first element of the output range to copy to.
 *  \return An iterator pointing to the last element of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator is mutable, and \p ForwardIterator's \c value_type has a constructor that takes
 *          a single argument whose type is \p InputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p uninitialized_copy to initialize
 *  a range of uninitialized memory using the \p thrust::device execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/uninitialized_copy.h>
 *  #include <thrust/device_malloc.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  
 *  struct Int
 *  {
 *    __host__ __device__
 *    Int(int x) : val(x) {}
 *    int val;
 *  };  
 *  ...
 *  const int N = 137;
 *
 *  Int val(46);
 *  thrust::device_vector<Int> input(N, val);
 *  thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
 *  thrust::uninitialized_copy(thrust::device, input.begin(), input.end(), array);
 *
 *  // Int x = array[i];
 *  // x.val == 46 for all 0 <= i < N
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/memory/uninitialized_copy
 *  \see \c copy
 *  \see \c uninitialized_fill
 *  \see \c device_new
 *  \see \c device_malloc
 */
template<typename DerivedPolicy, typename InputIterator, typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                     InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result);


/*! In \c thrust, the function \c thrust::device_new allocates memory for
 *  an object and then creates an object at that location by calling a constructor.
 *  Occasionally, however, it is useful to separate those two operations.
 *  If each iterator in the range <tt>[result, result + (last - first))</tt> points
 *  to uninitialized memory, then \p uninitialized_copy creates a copy of
 *  <tt>[first, last)</tt> in that range. That is, for each iterator \c i in
 *  the input, \p uninitialized_copy creates a copy of \c *i in the location pointed
 *  to by the corresponding iterator in the output range by \p ForwardIterator's
 *  \c value_type's copy constructor with *i as its argument.
 *
 *  \param first The first element of the input range to copy from.
 *  \param last The last element of the input range to copy from.
 *  \param result The first element of the output range to copy to.
 *  \return An iterator pointing to the last element of the output range.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator is mutable, and \p ForwardIterator's \c value_type has a constructor that takes
 *          a single argument whose type is \p InputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> and the range <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p uninitialized_copy to initialize
 *  a range of uninitialized memory.
 *
 *  \code
 *  #include <thrust/uninitialized_copy.h>
 *  #include <thrust/device_malloc.h>
 *  #include <thrust/device_vector.h>
 *  
 *  struct Int
 *  {
 *    __host__ __device__
 *    Int(int x) : val(x) {}
 *    int val;
 *  };  
 *  ...
 *  const int N = 137;
 *
 *  Int val(46);
 *  thrust::device_vector<Int> input(N, val);
 *  thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
 *  thrust::uninitialized_copy(input.begin(), input.end(), array);
 *
 *  // Int x = array[i];
 *  // x.val == 46 for all 0 <= i < N
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/memory/uninitialized_copy
 *  \see \c copy
 *  \see \c uninitialized_fill
 *  \see \c device_new
 *  \see \c device_malloc
 */
template<typename InputIterator, typename ForwardIterator>
  ForwardIterator uninitialized_copy(InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result);


/*! In \c thrust, the function \c thrust::device_new allocates memory for
 *  an object and then creates an object at that location by calling a constructor.
 *  Occasionally, however, it is useful to separate those two operations.
 *  If each iterator in the range <tt>[result, result + n)</tt> points
 *  to uninitialized memory, then \p uninitialized_copy_n creates a copy of
 *  <tt>[first, first + n)</tt> in that range. That is, for each iterator \c i in
 *  the input, \p uninitialized_copy_n creates a copy of \c *i in the location pointed
 *  to by the corresponding iterator in the output range by \p InputIterator's
 *  \c value_type's copy constructor with *i as its argument.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The first element of the input range to copy from.
 *  \param n The number of elements to copy.
 *  \param result The first element of the output range to copy to.
 *  \return An iterator pointing to the last element of the output range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Size is an integral type.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator is mutable, and \p ForwardIterator's \c value_type has a constructor that takes
 *          a single argument whose type is \p InputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, first + n)</tt> and the range <tt>[result, result + n)</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p uninitialized_copy to initialize
 *  a range of uninitialized memory using the \p thrust::device execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/uninitialized_copy.h>
 *  #include <thrust/device_malloc.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  
 *  struct Int
 *  {
 *    __host__ __device__
 *    Int(int x) : val(x) {}
 *    int val;
 *  };  
 *  ...
 *  const int N = 137;
 *
 *  Int val(46);
 *  thrust::device_vector<Int> input(N, val);
 *  thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
 *  thrust::uninitialized_copy_n(thrust::device, input.begin(), N, array);
 *
 *  // Int x = array[i];
 *  // x.val == 46 for all 0 <= i < N
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/memory/uninitialized_copy
 *  \see \c uninitialized_copy
 *  \see \c copy
 *  \see \c uninitialized_fill
 *  \see \c device_new
 *  \see \c device_malloc
 */
template<typename DerivedPolicy, typename InputIterator, typename Size, typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy_n(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator first,
                                       Size n,
                                       ForwardIterator result);


/*! In \c thrust, the function \c thrust::device_new allocates memory for
 *  an object and then creates an object at that location by calling a constructor.
 *  Occasionally, however, it is useful to separate those two operations.
 *  If each iterator in the range <tt>[result, result + n)</tt> points
 *  to uninitialized memory, then \p uninitialized_copy_n creates a copy of
 *  <tt>[first, first + n)</tt> in that range. That is, for each iterator \c i in
 *  the input, \p uninitialized_copy_n creates a copy of \c *i in the location pointed
 *  to by the corresponding iterator in the output range by \p InputIterator's
 *  \c value_type's copy constructor with *i as its argument.
 *
 *  \param first The first element of the input range to copy from.
 *  \param n The number of elements to copy.
 *  \param result The first element of the output range to copy to.
 *  \return An iterator pointing to the last element of the output range.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Size is an integral type.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator is mutable, and \p ForwardIterator's \c value_type has a constructor that takes
 *          a single argument whose type is \p InputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, first + n)</tt> and the range <tt>[result, result + n)</tt> shall not overlap otherwise.
 *
 *  The following code snippet demonstrates how to use \p uninitialized_copy to initialize
 *  a range of uninitialized memory.
 *
 *  \code
 *  #include <thrust/uninitialized_copy.h>
 *  #include <thrust/device_malloc.h>
 *  #include <thrust/device_vector.h>
 *  
 *  struct Int
 *  {
 *    __host__ __device__
 *    Int(int x) : val(x) {}
 *    int val;
 *  };  
 *  ...
 *  const int N = 137;
 *
 *  Int val(46);
 *  thrust::device_vector<Int> input(N, val);
 *  thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
 *  thrust::uninitialized_copy_n(input.begin(), N, array);
 *
 *  // Int x = array[i];
 *  // x.val == 46 for all 0 <= i < N
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/memory/uninitialized_copy
 *  \see \c uninitialized_copy
 *  \see \c copy
 *  \see \c uninitialized_fill
 *  \see \c device_new
 *  \see \c device_malloc
 */
template<typename InputIterator, typename Size, typename ForwardIterator>
  ForwardIterator uninitialized_copy_n(InputIterator first,
                                       Size n,
                                       ForwardIterator result);


/*! \} // copying
 */

THRUST_NAMESPACE_END

#include <thrust/detail/uninitialized_copy.inl>
