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
#include <thrust/detail/type_traits.h>
#include <thrust/system/detail/generic/select_system_exists.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{
namespace select_system_detail
{


// min_system case 1: both systems have the same type, just return the first one
template<typename System>
__host__ __device__
System &min_system(thrust::execution_policy<System> &system1,
                   thrust::execution_policy<System> &)
{
  return thrust::detail::derived_cast(system1);
} // end min_system()


// min_system case 2: systems have differing type and the first type is considered the minimum
template<typename System1, typename System2>
__host__ __device__
typename thrust::detail::enable_if<
  thrust::detail::is_same<
    System1,
    typename thrust::detail::minimum_system<System1,System2>::type
  >::value,
  System1 &
>::type
  min_system(thrust::execution_policy<System1> &system1, thrust::execution_policy<System2> &)
{
  return thrust::detail::derived_cast(system1);
} // end min_system()


// min_system case 3: systems have differing type and the second type is considered the minimum
template<typename System1, typename System2>
__host__ __device__
typename thrust::detail::enable_if<
  thrust::detail::is_same<
    System2,
    typename thrust::detail::minimum_system<System1,System2>::type
  >::value,
    System2 &
  >::type
    min_system(thrust::execution_policy<System1> &, thrust::execution_policy<System2> &system2)
{
  return thrust::detail::derived_cast(system2);
} // end min_system()


} // end select_system_detail


template<typename System>
__host__ __device__
  typename thrust::detail::disable_if<
    select_system1_exists<System>::value,
    System &
  >::type
    select_system(thrust::execution_policy<System> &system)
{
  return thrust::detail::derived_cast(system);
} // end select_system()


template<typename System1, typename System2>
__host__ __device__
  typename thrust::detail::enable_if_defined<
    thrust::detail::minimum_system<System1,System2>
  >::type
    &select_system(thrust::execution_policy<System1> &system1,
                   thrust::execution_policy<System2> &system2)
{
  return select_system_detail::min_system(system1,system2);
} // end select_system()


template<typename System1, typename System2, typename System3>
__host__ __device__
  typename thrust::detail::lazy_disable_if<
    select_system3_exists<System1,System2,System3>::value,
    thrust::detail::minimum_system<System1,System2,System3>
  >::type
    &select_system(thrust::execution_policy<System1> &system1,
                   thrust::execution_policy<System2> &system2,
                   thrust::execution_policy<System3> &system3)
{
  return select_system(select_system(system1,system2), system3);
} // end select_system()


template<typename System1, typename System2, typename System3, typename System4>
__host__ __device__
  typename thrust::detail::lazy_disable_if<
    select_system4_exists<System1,System2,System3,System4>::value,
    thrust::detail::minimum_system<System1,System2,System3,System4>
  >::type
    &select_system(thrust::execution_policy<System1> &system1,
                   thrust::execution_policy<System2> &system2,
                   thrust::execution_policy<System3> &system3,
                   thrust::execution_policy<System4> &system4)
{
  return select_system(select_system(system1,system2,system3), system4);
} // end select_system()


template<typename System1, typename System2, typename System3, typename System4, typename System5>
__host__ __device__
  typename thrust::detail::lazy_disable_if<
    select_system5_exists<System1,System2,System3,System4,System5>::value,
    thrust::detail::minimum_system<System1,System2,System3,System4,System5>
  >::type
    &select_system(thrust::execution_policy<System1> &system1,
                   thrust::execution_policy<System2> &system2,
                   thrust::execution_policy<System3> &system3,
                   thrust::execution_policy<System4> &system4,
                   thrust::execution_policy<System5> &system5)
{
  return select_system(select_system(system1,system2,system3,system4), system5);
} // end select_system()


template<typename System1, typename System2, typename System3, typename System4, typename System5, typename System6>
__host__ __device__
  typename thrust::detail::lazy_disable_if<
    select_system6_exists<System1,System2,System3,System4,System5,System6>::value,
    thrust::detail::minimum_system<System1,System2,System3,System4,System5,System6>
  >::type
    &select_system(thrust::execution_policy<System1> &system1,
                   thrust::execution_policy<System2> &system2,
                   thrust::execution_policy<System3> &system3,
                   thrust::execution_policy<System4> &system4,
                   thrust::execution_policy<System5> &system5,
                   thrust::execution_policy<System6> &system6)
{
  return select_system(select_system(system1,system2,system3,system4,system5), system6);
} // end select_system()


// map a single any_system_tag to device_system_tag
inline __host__ __device__
thrust::device_system_tag select_system(thrust::any_system_tag)
{
  return thrust::device_system_tag();
} // end select_system()


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

