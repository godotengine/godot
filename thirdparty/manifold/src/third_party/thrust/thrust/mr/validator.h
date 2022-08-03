/*
 *  Copyright 2018 NVIDIA Corporation
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

#include <thrust/detail/config/memory_resource.h>
#include <thrust/mr/memory_resource.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

template<typename MR>
struct validator
{
#if THRUST_CPP_DIALECT >= 2011
  static_assert(
    std::is_base_of<memory_resource<typename MR::pointer>, MR>::value,
    "a type used as a memory resource must derive from memory_resource"
  );
#endif
};

template<typename T, typename U>
struct validator2 : private validator<T>, private validator<U>
{
};

template<typename T>
struct validator2<T, T> : private validator<T>
{
};

} // end mr
THRUST_NAMESPACE_END

