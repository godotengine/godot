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

#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

template<typename T>
  __host__ __device__
  device_ptr<T> device_pointer_cast(T *ptr)
{
  return device_ptr<T>(ptr);
} // end device_pointer_cast()

template<typename T>
  __host__ __device__
  device_ptr<T> device_pointer_cast(const device_ptr<T> &ptr)
{
  return ptr;
} // end device_pointer_cast()


namespace detail
{

template<typename T>
  struct is_device_ptr< thrust::device_ptr<T> >
    : public true_type
{
}; // end is_device_ptr

#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && (_MSC_VER <= 1400)
// XXX WAR MSVC 2005 problem with correctly implementing
//     pointer_raw_pointer for device_ptr by specializing it here
template<typename T>
  struct pointer_raw_pointer< thrust::device_ptr<T> >
{
  typedef typename device_ptr<T>::raw_pointer type;
}; // end pointer_raw_pointer
#endif


} // end namespace detail

THRUST_NAMESPACE_END
