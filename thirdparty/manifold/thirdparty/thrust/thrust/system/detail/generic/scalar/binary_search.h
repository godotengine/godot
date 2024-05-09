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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

namespace system
{

namespace detail
{

namespace generic
{

namespace scalar
{

template<typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound_n(RandomAccessIterator first,
                                   Size n,
                                   const T &val,
                                   BinaryPredicate comp);

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound(RandomAccessIterator first, RandomAccessIterator last,
                                 const T &val,
                                 BinaryPredicate comp);

template<typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound_n(RandomAccessIterator first,
                                   Size n,
                                   const T &val,
                                   BinaryPredicate comp);

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound(RandomAccessIterator first, RandomAccessIterator last,
                                 const T &val,
                                 BinaryPredicate comp);

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
  pair<RandomAccessIterator,RandomAccessIterator>
    equal_range(RandomAccessIterator first, RandomAccessIterator last,
                const T &val,
                BinaryPredicate comp);

template<typename RandomAccessIterator, typename T, typename Compare>
__host__ __device__
bool binary_search(RandomAccessIterator first, RandomAccessIterator last, const T &value, Compare comp);

} // end scalar

} // end generic

} // end detail

} // end system

THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/scalar/binary_search.inl>

