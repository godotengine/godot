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
#include <thrust/detail/copy.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/copy.h>
#include <thrust/system/detail/adl/copy.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator>
__host__ __device__
  OutputIterator copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  using thrust::system::detail::generic::copy;
  return copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result);
} // end copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Size, typename OutputIterator>
__host__ __device__
  OutputIterator copy_n(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        InputIterator first,
                        Size n,
                        OutputIterator result)
{
  using thrust::system::detail::generic::copy_n;
  return copy_n(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, n, result);
} // end copy_n()


namespace detail
{


__thrust_exec_check_disable__ // because we might call e.g. std::ostream_iterator's constructor
template<typename System1,
         typename System2,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator two_system_copy(const thrust::execution_policy<System1> &system1,
                                 const thrust::execution_policy<System2> &system2,
                                 InputIterator first,
                                 InputIterator last,
                                 OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  return thrust::copy(select_system(thrust::detail::derived_cast(thrust::detail::strip_const(system1)), thrust::detail::derived_cast(thrust::detail::strip_const(system2))), first, last, result);
} // end two_system_copy()


__thrust_exec_check_disable__ // because we might call e.g. std::ostream_iterator's constructor
template<typename System1,
         typename System2,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
__host__ __device__
  OutputIterator two_system_copy_n(const thrust::execution_policy<System1> &system1,
                                   const thrust::execution_policy<System2> &system2,
                                   InputIterator first,
                                   Size n,
                                   OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  return thrust::copy_n(select_system(thrust::detail::derived_cast(thrust::detail::strip_const(system1)), thrust::detail::derived_cast(thrust::detail::strip_const(system2))), first, n, result);
} // end two_system_copy_n()


} // end detail


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::detail::two_system_copy(system1, system2, first, last, result);
} // end copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result)
{
  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::detail::two_system_copy_n(system1, system2, first, n, result);
} // end copy_n()

THRUST_NAMESPACE_END
