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

#include <thrust/reduce.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/reduce.h>
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <thrust/system/detail/adl/reduce.h>
#include <thrust/system/detail/adl/reduce_by_key.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator>
__host__ __device__
  typename thrust::iterator_traits<InputIterator>::value_type
    reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last)
{
  using thrust::system::detail::generic::reduce;
  return reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end reduce()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename T>
__host__ __device__
  T reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           InputIterator first,
           InputIterator last,
           T init)
{
  using thrust::system::detail::generic::reduce;
  return reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, init);
} // end reduce()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename T,
         typename BinaryFunction>
__host__ __device__
  T reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           InputIterator first,
           InputIterator last,
           T init,
           BinaryFunction binary_op)
{
  using thrust::system::detail::generic::reduce;
  return reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, init, binary_op);
} // end reduce()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, keys_output, values_output);
} // end reduce_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
} // end reduce_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
} // end reduce_by_key()


template<typename InputIterator>
typename thrust::iterator_traits<InputIterator>::value_type
  reduce(InputIterator first,
         InputIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::reduce(select_system(system), first, last);
}


template<typename InputIterator,
         typename T>
   T reduce(InputIterator first,
            InputIterator last,
            T init)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::reduce(select_system(system), first, last, init);
}


template<typename InputIterator,
         typename T,
         typename BinaryFunction>
   T reduce(InputIterator first,
            InputIterator last,
            T init,
            BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::reduce(select_system(system), first, last, init, binary_op);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::reduce_by_key(select_system(system1,system2,system3,system4), keys_first, keys_last, values_first, keys_output, values_output);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::reduce_by_key(select_system(system1,system2,system3,system4), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::reduce_by_key(select_system(system1,system2,system3,system4), keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}


THRUST_NAMESPACE_END

