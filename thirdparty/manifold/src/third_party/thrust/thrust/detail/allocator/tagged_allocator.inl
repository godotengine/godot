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
#include <thrust/detail/allocator/tagged_allocator.h>
#include <limits>

THRUST_NAMESPACE_BEGIN
namespace detail
{


template<typename T, typename Tag, typename Pointer>
  __host__ __device__
  tagged_allocator<T,Tag,Pointer>
    ::tagged_allocator()
{}


template<typename T, typename Tag, typename Pointer>
  __host__ __device__
  tagged_allocator<T,Tag,Pointer>
    ::tagged_allocator(const tagged_allocator<T,Tag,Pointer> &)
{}


template<typename T, typename Tag, typename Pointer>
  template<typename U, typename OtherPointer>
    __host__ __device__
    tagged_allocator<T,Tag,Pointer>
      ::tagged_allocator(const tagged_allocator<U,Tag,OtherPointer> &)
{}


template<typename T, typename Tag, typename Pointer>
  __host__ __device__
  tagged_allocator<T,Tag,Pointer>
    ::~tagged_allocator()
{}


template<typename T, typename Tag, typename Pointer>
  __host__ __device__
  typename tagged_allocator<T,Tag,Pointer>::pointer
    tagged_allocator<T,Tag,Pointer>
      ::address(reference x) const
{
  return &x;
}


template<typename T, typename Tag, typename Pointer>
  __host__ __device__
  typename tagged_allocator<T,Tag,Pointer>::const_pointer
    tagged_allocator<T,Tag,Pointer>
      ::address(const_reference x) const
{
  return &x;
}


template<typename T, typename Tag, typename Pointer>
  typename tagged_allocator<T,Tag,Pointer>::size_type
    tagged_allocator<T,Tag,Pointer>
      ::max_size() const
{
  return (std::numeric_limits<size_type>::max)() / sizeof(T);
}


template<typename T1, typename Pointer1, typename T2, typename Pointer2, typename Tag>
__host__ __device__
bool operator==(const tagged_allocator<T1,Pointer1,Tag> &, const tagged_allocator<T2,Pointer2,Tag> &)
{
  return true;
}


template<typename T1, typename Pointer1, typename T2, typename Pointer2, typename Tag>
__host__ __device__
bool operator!=(const tagged_allocator<T1,Pointer1,Tag> &, const tagged_allocator<T2,Pointer2,Tag> &)
{
  return false;
}


} // end detail
THRUST_NAMESPACE_END

