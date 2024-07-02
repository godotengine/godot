/*
 *  Copyright 2018-2020 NVIDIA Corporation
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

// #include the host system's memory_resource header
#define __THRUST_HOST_SYSTEM_MEMORY_HEADER <__THRUST_HOST_SYSTEM_ROOT/memory_resource.h>
#include __THRUST_HOST_SYSTEM_MEMORY_HEADER
#undef __THRUST_HOST_SYSTEM_MEMORY_HEADER

THRUST_NAMESPACE_BEGIN

typedef thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::memory_resource
    host_memory_resource;

THRUST_NAMESPACE_END

