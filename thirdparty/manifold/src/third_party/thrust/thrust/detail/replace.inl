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
#include <thrust/replace.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/replace.h>
#include <thrust/system/detail/adl/replace.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void replace(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               ForwardIterator first, ForwardIterator last,
               const T &old_value,
               const T &new_value)
{
  using thrust::system::detail::generic::replace;
  return replace(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, old_value, new_value);
} // end replace()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename Predicate, typename T>
__host__ __device__
  void replace_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                  ForwardIterator first, ForwardIterator last,
                  Predicate pred,
                  const T &new_value)
{
  using thrust::system::detail::generic::replace_if;
  return replace_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred, new_value);
} // end replace_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
__host__ __device__
  void replace_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                  ForwardIterator first, ForwardIterator last,
                  InputIterator stencil,
                  Predicate pred,
                  const T &new_value)
{
  using thrust::system::detail::generic::replace_if;
  return replace_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, stencil, pred, new_value);
} // end replace_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
__host__ __device__
  OutputIterator replace_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              InputIterator first, InputIterator last,
                              OutputIterator result,
                              const T &old_value,
                              const T &new_value)
{
  using thrust::system::detail::generic::replace_copy;
  return replace_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, old_value, new_value);
} // end replace_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate, typename T>
__host__ __device__
  OutputIterator replace_copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                 InputIterator first, InputIterator last,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value)
{
  using thrust::system::detail::generic::replace_copy_if;
  return replace_copy_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, pred, new_value);
} // end replace_copy_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
__host__ __device__
  OutputIterator replace_copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                 InputIterator1 first, InputIterator1 last,
                                 InputIterator2 stencil,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value)
{
  using thrust::system::detail::generic::replace_copy_if;
  return replace_copy_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, stencil, result, pred, new_value);
} // end replace_copy_if()


template<typename InputIterator, typename OutputIterator, typename Predicate, typename T>
  OutputIterator replace_copy_if(InputIterator first, InputIterator last,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::replace_copy_if(select_system(system1,system2), first, last, result, pred, new_value);
} // end replace_copy_if()


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
  OutputIterator replace_copy_if(InputIterator1 first, InputIterator1 last,
                                 InputIterator2 stencil,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::replace_copy_if(select_system(system1,system2,system3), first, last, stencil, result, pred, new_value);
} // end replace_copy_if()


template<typename InputIterator, typename OutputIterator, typename T>
  OutputIterator replace_copy(InputIterator first, InputIterator last,
                              OutputIterator result,
                              const T &old_value,
                              const T &new_value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::replace_copy(select_system(system1,system2), first, last, result, old_value, new_value);
} // end replace_copy()


template<typename ForwardIterator, typename Predicate, typename T>
  void replace_if(ForwardIterator first, ForwardIterator last,
                  Predicate pred,
                  const T &new_value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::replace_if(select_system(system), first, last, pred, new_value);
} // end replace_if()


template<typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
  void replace_if(ForwardIterator first, ForwardIterator last,
                  InputIterator stencil,
                  Predicate pred,
                  const T &new_value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System1;
  typedef typename thrust::iterator_system<InputIterator>::type   System2;

  System1 system1;
  System2 system2;

  return thrust::replace_if(select_system(system1,system2), first, last, stencil, pred, new_value);
} // end replace_if()


template<typename ForwardIterator, typename T>
  void replace(ForwardIterator first, ForwardIterator last,
               const T &old_value,
               const T &new_value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::replace(select_system(system), first, last, old_value, new_value);
} // end replace()


THRUST_NAMESPACE_END

