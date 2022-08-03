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
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate, typename T>
__host__ __device__
  OutputIterator replace_copy_if(thrust::execution_policy<DerivedPolicy> &exec,
                                 InputIterator first,
                                 InputIterator last,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value);


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
__host__ __device__
  OutputIterator replace_copy_if(thrust::execution_policy<DerivedPolicy> &exec,
                                 InputIterator1 first,
                                 InputIterator1 last,
                                 InputIterator2 stencil,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value);


template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
__host__ __device__
  OutputIterator replace_copy(thrust::execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              const T &old_value,
                              const T &new_value);


template<typename DerivedPolicy, typename ForwardIterator, typename Predicate, typename T>
__host__ __device__
  void replace_if(thrust::execution_policy<DerivedPolicy> &exec,
                  ForwardIterator first,
                  ForwardIterator last,
                  Predicate pred,
                  const T &new_value);


template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
__host__ __device__
  void replace_if(thrust::execution_policy<DerivedPolicy> &exec,
                  ForwardIterator first,
                  ForwardIterator last,
                  InputIterator stencil,
                  Predicate pred,
                  const T &new_value);


template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void replace(thrust::execution_policy<DerivedPolicy> &exec,
               ForwardIterator first,
               ForwardIterator last,
               const T &old_value,
               const T &new_value);


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/replace.inl>

