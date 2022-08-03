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
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/tbb/memory.h>
#include <thrust/system/cpp/memory.h>
#include <limits>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{


namespace detail
{

// XXX circular #inclusion problems cause the compiler to believe that cpp::malloc
//     is not defined
//     WAR the problem by using adl to call cpp::malloc, which requires it to depend
//     on a template parameter
template<typename Tag>
  pointer<void> malloc_workaround(Tag t, std::size_t n)
{
  return pointer<void>(malloc(t, n));
} // end malloc_workaround()

// XXX circular #inclusion problems cause the compiler to believe that cpp::free
//     is not defined
//     WAR the problem by using adl to call cpp::free, which requires it to depend
//     on a template parameter
template<typename Tag>
  void free_workaround(Tag t, pointer<void> ptr)
{
  free(t, ptr.get());
} // end free_workaround()

} // end detail

inline pointer<void> malloc(std::size_t n)
{
  // XXX this is how we'd like to implement this function,
  //     if not for circular #inclusion problems:
  //
  // return pointer<void>(thrust::system::cpp::malloc(n))
  //
  return detail::malloc_workaround(cpp::tag(), n);
} // end malloc()

template<typename T>
pointer<T> malloc(std::size_t n)
{
  pointer<void> raw_ptr = thrust::system::tbb::malloc(sizeof(T) * n);
  return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

inline void free(pointer<void> ptr)
{
  // XXX this is how we'd like to implement this function,
  //     if not for circular #inclusion problems:
  //
  // thrust::system::cpp::free(ptr)
  //
  detail::free_workaround(cpp::tag(), ptr);
} // end free()

} // end tbb
} // end system
THRUST_NAMESPACE_END

