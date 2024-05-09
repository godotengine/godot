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


/*! \file distance.h
 *  \brief Computes the size of a range
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \p distance finds the distance between \p first and \p last, i.e. the
 *  number of times that \p first must be incremented until it is equal to
 *  \p last.
 *
 *  \param first The beginning of an input range of interest.
 *  \param last The end of an input range of interest.
 *  \return The distance between the beginning and end of the input range.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *
 *  \pre If \c InputIterator meets the requirements of random access iterator, \p last shall be reachable from \p first or
 *       \p first shall be reachable from \p last; otherwise, \p last shall be reachable from \p first.
 *
 *  The following code snippet demonstrates how to use \p distance to compute
 *  the distance to one iterator from another.
 *
 *  \code
 *  #include <thrust/distance.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> vec(13);
 *  thrust::device_vector<int>::iterator iter1 = vec.begin();
 *  thrust::device_vector<int>::iterator iter2 = iter1 + 7;
 *
 *  int d = thrust::distance(iter1, iter2);
 *
 *  // d is 7
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/distance
 */
template<typename InputIterator>
inline __host__ __device__
  typename thrust::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last);

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

#include <thrust/detail/distance.inl>
