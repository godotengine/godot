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

/*! \file sort.h
 *  \brief Sequential implementations of sort algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort(sequential::execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp);


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering comp);


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/sequential/sort.inl>

