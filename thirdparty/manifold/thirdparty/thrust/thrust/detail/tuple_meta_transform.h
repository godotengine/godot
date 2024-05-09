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

#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

// introduce an intermediate type tuple_meta_transform_WAR_NVCC
// rather than directly specializing tuple_meta_transform with
// default argument IndexSequence = thrust::make_index_sequence<thrust::tuple_size<Tuple>::value>
// to workaround nvcc 11.0 compiler bug
template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename IndexSequence>
  struct tuple_meta_transform_WAR_NVCC;

template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         size_t... Is>
  struct tuple_meta_transform_WAR_NVCC<Tuple, UnaryMetaFunction, thrust::index_sequence<Is...>>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<Is,Tuple>::type>::type...
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform
{
  typedef typename tuple_meta_transform_WAR_NVCC<Tuple, UnaryMetaFunction, thrust::make_index_sequence<thrust::tuple_size<Tuple>::value>>::type type;
};

} // end detail

THRUST_NAMESPACE_END

