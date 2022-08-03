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
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

template<typename System,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator copy(const thrust::detail::execution_policy_base<System> &system,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result);

template<typename System,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
__host__ __device__
  OutputIterator copy_n(const thrust::detail::execution_policy_base<System> &system,
                        InputIterator first,
                        Size n,
                        OutputIterator result);

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result);

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result);


namespace detail
{


template<typename FromSystem,
         typename ToSystem,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator two_system_copy(const thrust::execution_policy<FromSystem> &from_system,
                                 const thrust::execution_policy<ToSystem>   &two_system,
                                 InputIterator first,
                                 InputIterator last,
                                 OutputIterator result);


template<typename FromSystem,
         typename ToSystem,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
__host__ __device__
  OutputIterator two_system_copy_n(const thrust::execution_policy<FromSystem> &from_system,
                                   const thrust::execution_policy<ToSystem>   &two_system,
                                   InputIterator first,
                                   Size n,
                                   OutputIterator result);


} // end detail

THRUST_NAMESPACE_END

#include <thrust/detail/copy.inl>
