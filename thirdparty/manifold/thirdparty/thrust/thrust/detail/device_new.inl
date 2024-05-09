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
#include <thrust/device_new.h>
#include <thrust/device_malloc.h>
#include <thrust/uninitialized_fill.h>

THRUST_NAMESPACE_BEGIN

template<typename T>
  device_ptr<T> device_new(device_ptr<void> p,
                           const size_t n)
{
  // XXX TODO dispatch n null device constructors at p here
  // in the meantime, dispatch 1 null host constructor here
  // and dispatch n copy constructors
  return device_new<T>(p, T(), n);
} // end device_new()

template<typename T>
  device_ptr<T> device_new(device_ptr<void> p,
                           const T &exemplar,
                           const size_t n)
{
  device_ptr<T> result(reinterpret_cast<T*>(p.get()));

  // run copy constructors at p here
  thrust::uninitialized_fill(result, result + n, exemplar);

  return result;
} // end device_new()

template<typename T>
  device_ptr<T> device_new(const size_t n)
{
  // call placement new
  return device_new<T>(thrust::device_malloc<T>(n));
} // end device_new()

THRUST_NAMESPACE_END
