/*
 *  Copyright 2020 NVIDIA Corporation
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
#include <thrust/mr/allocator.h>
#include <thrust/mr/disjoint_tls_pool.h>
#include <thrust/mr/new.h>
#include <thrust/mr/device_memory_resource.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
inline
thrust::mr::allocator<
    char,
    thrust::mr::disjoint_unsynchronized_pool_resource<
        thrust::device_memory_resource,
        thrust::mr::new_delete_resource
    >
> single_device_tls_caching_allocator()
{
    return {
        &thrust::mr::tls_disjoint_pool(
            thrust::mr::get_global_resource<thrust::device_memory_resource>(),
            thrust::mr::get_global_resource<thrust::mr::new_delete_resource>()
        )
    };
}
}

THRUST_NAMESPACE_END
