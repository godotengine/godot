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
#include <thrust/device_delete.h>
#include <thrust/device_free.h>
#include <thrust/detail/allocator/destroy_range.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

// define an empty allocator class to use below
struct device_delete_allocator {};

}

template<typename T>
  void device_delete(device_ptr<T> ptr,
                     const size_t n)
{
  // we can use device_allocator to destroy the range
  thrust::detail::device_delete_allocator a;
  thrust::detail::destroy_range(a, ptr, n);
  thrust::device_free(ptr);
} // end device_delete()

THRUST_NAMESPACE_END
