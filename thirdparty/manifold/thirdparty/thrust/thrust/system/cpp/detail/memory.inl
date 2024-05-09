/*
 *  Copyright 2008-2018 NVIDIA Corporation
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
#include <thrust/system/cpp/memory.h>
#include <thrust/system/cpp/detail/malloc_and_free.h>
#include <limits>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace cpp
{

pointer<void> malloc(std::size_t n)
{
  tag t;
  return pointer<void>(thrust::system::detail::sequential::malloc(t, n));
} // end malloc()

template<typename T>
pointer<T> malloc(std::size_t n)
{
  pointer<void> raw_ptr = thrust::system::cpp::malloc(sizeof(T) * n);
  return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

void free(pointer<void> ptr)
{
  tag t;
  return thrust::system::detail::sequential::free(t, ptr);
} // end free()

} // end cpp
} // end system
THRUST_NAMESPACE_END

