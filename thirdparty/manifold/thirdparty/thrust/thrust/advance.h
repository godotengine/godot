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


/*! \file advance.h
 *  \brief Advance an iterator by a given distance.
 */

#pragma once

#include <thrust/detail/config.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \p advance(i, n) increments the iterator \p i by the distance \p n.
 *  If <tt>n > 0</tt> it is equivalent to executing <tt>++i</tt> \p n
 *  times, and if <tt>n < 0</tt> it is equivalent to executing <tt>--i</tt>
 *  \p n times. If <tt>n == 0</tt>, the call has no effect.
 *
 *  \param i The iterator to be advanced.
 *  \param n The distance by which to advance the iterator.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Distance is an integral type that is convertible to \p InputIterator's distance type.
 *
 *  \pre \p n shall be negative only for bidirectional and random access iterators.
 *
 *  The following code snippet demonstrates how to use \p advance to increment
 *  an iterator a given number of times.
 *
 *  \code
 *  #include <thrust/advance.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> vec(13);
 *  thrust::device_vector<int>::iterator iter = vec.begin();
 *
 *  thrust::advance(iter, 7);
 *
 *  // iter - vec.begin() == 7
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/advance
 */
template <typename InputIterator, typename Distance>
__host__ __device__
void advance(InputIterator& i, Distance n);

/*! \p next(i, n) returns the \p n th successor of the iterator \p i.
 *
 *  \param i An iterator.
 *  \param n The number of elements to advance.
 *
 *  \tparam InputIterator must meet the <a href="https://en.cppreference.com/w/cpp/named_req/InputIterator">InputIterator</a>.
 *
 *  \pre \p n shall be negative only for bidirectional and random access iterators.
 *
 *  The following code snippet demonstrates how to use \p next.
 *
 *  \code
 *  #include <thrust/advance.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> vec(13);
 *  thrust::device_vector<int>::iterator i0 = vec.begin();
 *
 *  auto i1 = thrust::next(i0);
 *
 *  // i0 - vec.begin() == 0
 *  // i1 - vec.begin() == 1
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/next
 */
#if 0 // Doxygen only
template <typename InputIterator, typename Distance>
__host__ __device__
InputIterator next(
  InputIterator i
, typename iterator_traits<InputIterator>::difference_type n = 1
);
#endif

/*! \p prev(i, n) returns the \p n th predecessor of the iterator \p i.
 *
 *  \param i An iterator.
 *  \param n The number of elements to descend.
 *
 *  \tparam BidirectionalIterator must meet the <a href="https://en.cppreference.com/w/cpp/named_req/BidirectionalIterator">BidirectionalIterator</a>.
 *
 *  The following code snippet demonstrates how to use \p prev.
 *
 *  \code
 *  #include <thrust/advance.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> vec(13);
 *  thrust::device_vector<int>::iterator i0 = vec.end();
 *
 *  auto i1 = thrust::prev(i0);
 *
 *  // vec.end() - i0 == 0
 *  // vec.end() - i1 == 1
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/prev
 */
#if 0 // Doxygen only
template <typename BidirectionalIterator, typename Distance>
__host__ __device__
BidirectionalIterator prev(
  BidirectionalIterator i
, typename iterator_traits<BidirectionalIterator>::difference_type n = 1
);
#endif

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

#include <thrust/detail/advance.inl>

