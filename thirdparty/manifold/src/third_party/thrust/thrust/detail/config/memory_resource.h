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

#include <cstddef>

#include <thrust/detail/config.h>
#include <thrust/detail/alignment.h>
#include <thrust/detail/config/cpp_compatibility.h>

#define THRUST_MR_DEFAULT_ALIGNMENT THRUST_ALIGNOF(THRUST_NS_QUALIFIER::detail::max_align_t)

#if THRUST_CPP_DIALECT >= 2017
#  if __has_include(<memory_resource>)
#    define THRUST_MR_STD_MR_HEADER <memory_resource>
#    define THRUST_MR_STD_MR_NS std::pmr
#  elif __has_include(<experimental/memory_resource>)
#    define THRUST_MR_STD_MR_HEADER <experimental/memory_resource>
#    define THRUST_MR_STD_MR_NS std::experimental::pmr
#  endif
#endif
