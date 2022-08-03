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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/set_operations.h>
#include <thrust/system/detail/adl/set_operations.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
OutputIterator set_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              InputIterator1                                              first1,
                              InputIterator1                                              last1,
                              InputIterator2                                              first2,
                              InputIterator2                                              last2,
                              OutputIterator                                              result)
{
  using thrust::system::detail::generic::set_difference;
  return set_difference(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result);
} // end set_difference()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
OutputIterator set_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              InputIterator1                                              first1,
                              InputIterator1                                              last1,
                              InputIterator2                                              first2,
                              InputIterator2                                              last2,
                              OutputIterator                                              result,
                              StrictWeakCompare                                           comp)
{
  using thrust::system::detail::generic::set_difference;
  return set_difference(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result, comp);
} // end set_difference()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  set_difference_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        InputIterator1                                              keys_first1,
                        InputIterator1                                              keys_last1,
                        InputIterator2                                              keys_first2,
                        InputIterator2                                              keys_last2,
                        InputIterator3                                              values_first1,
                        InputIterator4                                              values_first2,
                        OutputIterator1                                             keys_result,
                        OutputIterator2                                             values_result)
{
  using thrust::system::detail::generic::set_difference_by_key;
  return set_difference_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end set_difference_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  set_difference_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        InputIterator1                                              keys_first1,
                        InputIterator1                                              keys_last1,
                        InputIterator2                                              keys_first2,
                        InputIterator2                                              keys_last2,
                        InputIterator3                                              values_first1,
                        InputIterator4                                              values_first2,
                        OutputIterator1                                             keys_result,
                        OutputIterator2                                             values_result,
                        StrictWeakCompare                                           comp)
{
  using thrust::system::detail::generic::set_difference_by_key;
  return set_difference_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end set_difference_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
OutputIterator set_intersection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator1                                              first1,
                                InputIterator1                                              last1,
                                InputIterator2                                              first2,
                                InputIterator2                                              last2,
                                OutputIterator                                              result)
{
  using thrust::system::detail::generic::set_intersection;
  return set_intersection(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result);
} // end set_intersection()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
OutputIterator set_intersection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator1                                              first1,
                                InputIterator1                                              last1,
                                InputIterator2                                              first2,
                                InputIterator2                                              last2,
                                OutputIterator                                              result,
                                StrictWeakCompare                                           comp)
{
  using thrust::system::detail::generic::set_intersection;
  return set_intersection(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result, comp);
} // end set_intersection()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  set_intersection_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator1                                              keys_first1,
                          InputIterator1                                              keys_last1,
                          InputIterator2                                              keys_first2,
                          InputIterator2                                              keys_last2,
                          InputIterator3                                              values_first1,
                          OutputIterator1                                             keys_result,
                          OutputIterator2                                             values_result)
{
  using thrust::system::detail::generic::set_intersection_by_key;
  return set_intersection_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, keys_result, values_result);
} // end set_intersection_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  set_intersection_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator1                                              keys_first1,
                          InputIterator1                                              keys_last1,
                          InputIterator2                                              keys_first2,
                          InputIterator2                                              keys_last2,
                          InputIterator3                                              values_first1,
                          OutputIterator1                                             keys_result,
                          OutputIterator2                                             values_result,
                          StrictWeakCompare                                           comp)
{
  using thrust::system::detail::generic::set_intersection_by_key;
  return set_intersection_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, keys_result, values_result, comp);
} // end set_intersection_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
OutputIterator set_symmetric_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                        InputIterator1                                              first1,
                                        InputIterator1                                              last1,
                                        InputIterator2                                              first2,
                                        InputIterator2                                              last2,
                                        OutputIterator                                              result)
{
  using thrust::system::detail::generic::set_symmetric_difference;
  return set_symmetric_difference(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result);
} // end set_symmetric_difference()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
OutputIterator set_symmetric_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                        InputIterator1                                              first1,
                                        InputIterator1                                              last1,
                                        InputIterator2                                              first2,
                                        InputIterator2                                              last2,
                                        OutputIterator                                              result,
                                        StrictWeakCompare                                           comp)
{
  using thrust::system::detail::generic::set_symmetric_difference;
  return set_symmetric_difference(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result, comp);
} // end set_symmetric_difference()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  set_symmetric_difference_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  InputIterator1                                              keys_first1,
                                  InputIterator1                                              keys_last1,
                                  InputIterator2                                              keys_first2,
                                  InputIterator2                                              keys_last2,
                                  InputIterator3                                              values_first1,
                                  InputIterator4                                              values_first2,
                                  OutputIterator1                                             keys_result,
                                  OutputIterator2                                             values_result)
{
  using thrust::system::detail::generic::set_symmetric_difference_by_key;
  return set_symmetric_difference_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end set_symmetric_difference_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  set_symmetric_difference_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  InputIterator1                                              keys_first1,
                                  InputIterator1                                              keys_last1,
                                  InputIterator2                                              keys_first2,
                                  InputIterator2                                              keys_last2,
                                  InputIterator3                                              values_first1,
                                  InputIterator4                                              values_first2,
                                  OutputIterator1                                             keys_result,
                                  OutputIterator2                                             values_result,
                                  StrictWeakCompare                                           comp)
{
  using thrust::system::detail::generic::set_symmetric_difference_by_key;
  return set_symmetric_difference_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end set_symmetric_difference_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
OutputIterator set_union(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         InputIterator1                                              first1,
                         InputIterator1                                              last1,
                         InputIterator2                                              first2,
                         InputIterator2                                              last2,
                         OutputIterator                                              result)
{
  using thrust::system::detail::generic::set_union;
  return set_union(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result);
} // end set_union()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
__host__ __device__
OutputIterator set_union(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         InputIterator1                                              first1,
                         InputIterator1                                              last1,
                         InputIterator2                                              first2,
                         InputIterator2                                              last2,
                         OutputIterator                                              result,
                         StrictWeakCompare                                           comp)
{
  using thrust::system::detail::generic::set_union;
  return set_union(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result, comp);
} // end set_union()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  set_union_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator1                                              keys_first1,
                   InputIterator1                                              keys_last1,
                   InputIterator2                                              keys_first2,
                   InputIterator2                                              keys_last2,
                   InputIterator3                                              values_first1,
                   InputIterator4                                              values_first2,
                   OutputIterator1                                             keys_result,
                   OutputIterator2                                             values_result)
{
  using thrust::system::detail::generic::set_union_by_key;
  return set_union_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end set_union_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakCompare>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  set_union_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator1                                              keys_first1,
                   InputIterator1                                              keys_last1,
                   InputIterator2                                              keys_first2,
                   InputIterator2                                              keys_last2,
                   InputIterator3                                              values_first1,
                   InputIterator4                                              values_first2,
                   OutputIterator1                                             keys_result,
                   OutputIterator2                                             values_result,
                   StrictWeakCompare                                           comp)
{
  using thrust::system::detail::generic::set_union_by_key;
  return set_union_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end set_union_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result,
                                StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_difference(select_system(system1,system2,system3), first1, last1, first2, last2, result, comp);
} // end set_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_difference(select_system(system1,system2,system3), first1, last1, first2, last2, result);
} // end set_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakOrdering>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_difference_by_key(InputIterator1 keys_first1,
                          InputIterator1 keys_last1,
                          InputIterator2 keys_first2,
                          InputIterator2 keys_last2,
                          InputIterator3 values_first1,
                          InputIterator4 values_first2,
                          OutputIterator1 keys_result,
                          OutputIterator2 values_result,
                          StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<InputIterator4>::type  System4;
  typedef typename thrust::iterator_system<OutputIterator1>::type System5;
  typedef typename thrust::iterator_system<OutputIterator2>::type System6;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_difference_by_key(select_system(system1,system2,system3,system4,system5,system6), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end set_difference_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_difference_by_key(InputIterator1 keys_first1,
                          InputIterator1 keys_last1,
                          InputIterator2 keys_first2,
                          InputIterator2 keys_last2,
                          InputIterator3 values_first1,
                          InputIterator4 values_first2,
                          OutputIterator1 keys_result,
                          OutputIterator2 values_result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<InputIterator4>::type  System4;
  typedef typename thrust::iterator_system<OutputIterator1>::type System5;
  typedef typename thrust::iterator_system<OutputIterator2>::type System6;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_difference_by_key(select_system(system1,system2,system3,system4,system5,system6), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end set_difference_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_intersection(select_system(system1,system2,system3), first1, last1, first2, last2, result, comp);
} // end set_intersection()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_intersection(select_system(system1,system2,system3), first1, last1, first2, last2, result);
} // end set_intersection()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakOrdering>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_intersection_by_key(InputIterator1 keys_first1,
                            InputIterator1 keys_last1,
                            InputIterator2 keys_first2,
                            InputIterator2 keys_last2,
                            InputIterator3 values_first1,
                            OutputIterator1 keys_result,
                            OutputIterator2 values_result,
                            StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<OutputIterator1>::type System4;
  typedef typename thrust::iterator_system<OutputIterator2>::type System5;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;

  return thrust::set_intersection_by_key(select_system(system1,system2,system3,system4,system5), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, keys_result, values_result, comp);
} // end set_intersection_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_intersection_by_key(InputIterator1 keys_first1,
                            InputIterator1 keys_last1,
                            InputIterator2 keys_first2,
                            InputIterator2 keys_last2,
                            InputIterator3 values_first1,
                            OutputIterator1 keys_result,
                            OutputIterator2 values_result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<OutputIterator1>::type System4;
  typedef typename thrust::iterator_system<OutputIterator2>::type System5;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;

  return thrust::set_intersection_by_key(select_system(system1,system2,system3,system4,system5), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, keys_result, values_result);
} // end set_intersection_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result,
                                          StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_symmetric_difference(select_system(system1,system2,system3), first1, last1, first2, last2, result, comp);
} // end set_symmetric_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_symmetric_difference(select_system(system1,system2,system3), first1, last1, first2, last2, result);
} // end set_symmetric_difference()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakOrdering>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_symmetric_difference_by_key(InputIterator1 keys_first1,
                                    InputIterator1 keys_last1,
                                    InputIterator2 keys_first2,
                                    InputIterator2 keys_last2,
                                    InputIterator3 values_first1,
                                    InputIterator4 values_first2,
                                    OutputIterator1 keys_result,
                                    OutputIterator2 values_result,
                                    StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<InputIterator4>::type  System4;
  typedef typename thrust::iterator_system<OutputIterator1>::type System5;
  typedef typename thrust::iterator_system<OutputIterator2>::type System6;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_symmetric_difference_by_key(select_system(system1,system2,system3,system4,system5,system6), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end set_symmetric_difference_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_symmetric_difference_by_key(InputIterator1 keys_first1,
                                    InputIterator1 keys_last1,
                                    InputIterator2 keys_first2,
                                    InputIterator2 keys_last2,
                                    InputIterator3 values_first1,
                                    InputIterator4 values_first2,
                                    OutputIterator1 keys_result,
                                    OutputIterator2 values_result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<InputIterator4>::type  System4;
  typedef typename thrust::iterator_system<OutputIterator1>::type System5;
  typedef typename thrust::iterator_system<OutputIterator2>::type System6;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_symmetric_difference_by_key(select_system(system1,system2,system3,system4,system5,system6), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end set_symmetric_difference_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result,
                           StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_union(select_system(system1,system2,system3), first1, last1, first2, last2, result, comp);
} // end set_union()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_union(select_system(system1,system2,system3), first1, last1, first2, last2, result);
} // end set_union()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakOrdering>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_union_by_key(InputIterator1 keys_first1,
                     InputIterator1 keys_last1,
                     InputIterator2 keys_first2,
                     InputIterator2 keys_last2,
                     InputIterator3 values_first1,
                     InputIterator4 values_first2,
                     OutputIterator1 keys_result,
                     OutputIterator2 values_result,
                     StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<InputIterator4>::type  System4;
  typedef typename thrust::iterator_system<OutputIterator1>::type System5;
  typedef typename thrust::iterator_system<OutputIterator2>::type System6;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_union_by_key(select_system(system1,system2,system3,system4,system5,system6), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
} // end set_union_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    set_union_by_key(InputIterator1 keys_first1,
                     InputIterator1 keys_last1,
                     InputIterator2 keys_first2,
                     InputIterator2 keys_last2,
                     InputIterator3 values_first1,
                     InputIterator4 values_first2,
                     OutputIterator1 keys_result,
                     OutputIterator2 values_result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<InputIterator3>::type  System3;
  typedef typename thrust::iterator_system<InputIterator4>::type  System4;
  typedef typename thrust::iterator_system<OutputIterator1>::type System5;
  typedef typename thrust::iterator_system<OutputIterator2>::type System6;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_union_by_key(select_system(system1,system2,system3,system4,system5,system6), keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
} // end set_union_by_key()


THRUST_NAMESPACE_END

