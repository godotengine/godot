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
#include <thrust/system/cpp/vector.h>
#include <utility>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace cpp
{

template<typename T, typename Allocator>
  vector<T,Allocator>
    ::vector()
      : super_t()
{}

template<typename T, typename Allocator>
  vector<T,Allocator>
    ::vector(size_type n)
      : super_t(n)
{}

template<typename T, typename Allocator>
  vector<T,Allocator>
    ::vector(size_type n, const value_type &value)
      : super_t(n,value)
{}

template<typename T, typename Allocator>
  vector<T,Allocator>
    ::vector(const vector &x)
      : super_t(x)
{}

#if THRUST_CPP_DIALECT >= 2011
  template<typename T, typename Allocator>
    vector<T,Allocator>
      ::vector(vector &&x)
        : super_t(std::move(x))
  {}
#endif

template<typename T, typename Allocator>
  template<typename OtherT, typename OtherAllocator>
    vector<T,Allocator>
      ::vector(const thrust::detail::vector_base<OtherT,OtherAllocator> &x)
        : super_t(x)
{}

template<typename T, typename Allocator>
  template<typename OtherT, typename OtherAllocator>
    vector<T,Allocator>
      ::vector(const std::vector<OtherT,OtherAllocator> &x)
        : super_t(x)
{}

template<typename T, typename Allocator>
  template<typename InputIterator>
    vector<T,Allocator>
      ::vector(InputIterator first, InputIterator last)
        : super_t(first,last)
{}

template<typename T, typename Allocator>
  vector<T,Allocator> &
    vector<T,Allocator>
      ::operator=(const vector &x)
{
  super_t::operator=(x);
  return *this;
}

#if THRUST_CPP_DIALECT >= 2011
  template<typename T, typename Allocator>
    vector<T,Allocator> &
      vector<T,Allocator>
        ::operator=(vector &&x)
  {
    super_t::operator=(std::move(x));
    return *this;
  }
#endif

template<typename T, typename Allocator>
  template<typename OtherT, typename OtherAllocator>
    vector<T,Allocator> &
      vector<T,Allocator>
        ::operator=(const std::vector<OtherT,OtherAllocator> &x)
{
  super_t::operator=(x);
  return *this;
}

template<typename T, typename Allocator>
  template<typename OtherT, typename OtherAllocator>
    vector<T,Allocator> &
      vector<T,Allocator>
        ::operator=(const thrust::detail::vector_base<OtherT,OtherAllocator> &x)
{
  super_t::operator=(x);
  return *this;
}
      
} // end cpp
} // end system
THRUST_NAMESPACE_END

