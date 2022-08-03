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
#include <thrust/system/detail/generic/unique_by_key.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/copy_if.h>
#include <thrust/unique.h>
#include <thrust/detail/range/head_flags.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(thrust::execution_policy<ExecutionPolicy> &exec,
              ForwardIterator1 keys_first,
              ForwardIterator1 keys_last,
              ForwardIterator2 values_first)
{
  typedef typename thrust::iterator_traits<ForwardIterator1>::value_type KeyType;
  return thrust::unique_by_key(exec, keys_first, keys_last, values_first, thrust::equal_to<KeyType>());
} // end unique_by_key()


template<typename ExecutionPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
__host__ __device__
thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(thrust::execution_policy<ExecutionPolicy> &exec,
              ForwardIterator1 keys_first,
              ForwardIterator1 keys_last,
              ForwardIterator2 values_first,
              BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator1>::value_type InputType1;
  typedef typename thrust::iterator_traits<ForwardIterator2>::value_type InputType2;

  ForwardIterator2 values_last = values_first + (keys_last - keys_first);

  thrust::detail::temporary_array<InputType1,ExecutionPolicy> keys(exec, keys_first, keys_last);
  thrust::detail::temporary_array<InputType2,ExecutionPolicy> vals(exec, values_first, values_last);

  return thrust::unique_by_key_copy(exec, keys.begin(), keys.end(), vals.begin(), keys_first, values_first, binary_pred);
} // end unique_by_key()


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(thrust::execution_policy<ExecutionPolicy> &exec,
                   InputIterator1 keys_first,
                   InputIterator1 keys_last,
                   InputIterator2 values_first,
                   OutputIterator1 keys_output,
                   OutputIterator2 values_output)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type KeyType;
  return thrust::unique_by_key_copy(exec, keys_first, keys_last, values_first, keys_output, values_output, thrust::equal_to<KeyType>());
} // end unique_by_key_copy()


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(thrust::execution_policy<ExecutionPolicy> &exec,
                   InputIterator1 keys_first,
                   InputIterator1 keys_last,
                   InputIterator2 values_first,
                   OutputIterator1 keys_output,
                   OutputIterator2 values_output,
                   BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;

  difference_type n = thrust::distance(keys_first, keys_last);

  thrust::detail::head_flags<InputIterator1, BinaryPredicate> stencil(keys_first, keys_last, binary_pred);

  using namespace thrust::placeholders;
  thrust::zip_iterator< thrust::tuple<OutputIterator1, OutputIterator2> > result =
    thrust::copy_if(exec,
                    thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)),
                    thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)) + n,
                    stencil.begin(),
                    thrust::make_zip_iterator(thrust::make_tuple(keys_output, values_output)),
                    _1);

  difference_type output_size = result - thrust::make_zip_iterator(thrust::make_tuple(keys_output, values_output));

  return thrust::make_pair(keys_output + output_size, values_output + output_size);
} // end unique_by_key_copy()


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

