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
#include <thrust/system/detail/generic/tag.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename MR, typename DerivedPolicy>
__host__
MR * get_per_device_resource(thrust::detail::execution_policy_base<DerivedPolicy>&)
{
    return mr::get_global_resource<MR>();
}


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

