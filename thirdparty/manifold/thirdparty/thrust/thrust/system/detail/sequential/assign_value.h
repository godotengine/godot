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
#include <thrust/system/detail/sequential/execution_policy.h>
#include <thrust/detail/raw_pointer_cast.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{

template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
__host__ __device__
  void assign_value(sequential::execution_policy<DerivedPolicy> &, Pointer1 dst, Pointer2 src)
{
  *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
} // end assign_value()

} // end sequential
} // end detail
} // end system
THRUST_NAMESPACE_END

