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
#include <thrust/detail/preprocessor.h>

// TODO: Enable this or remove this file once nvGRAPH/CUSP migrates off of it.
//#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
//  #pragma message("warning: The functionality in this header is unsafe, deprecated, and will soon be removed. Use C++11 or C11 atomics instead.")
//#else
//  #warning The functionality in this header is unsafe, deprecated, and will soon be removed. Use C++11 or C11 atomics instead.
//#endif

// msvc case
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC

#ifndef _DEBUG

#include <intrin.h>
#pragma intrinsic(_ReadWriteBarrier)
#define __thrust_compiler_fence() _ReadWriteBarrier()
#else

#define __thrust_compiler_fence() do {} while (0)

#endif // _DEBUG

// gcc case
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC

#if THRUST_GCC_VERSION >= 40200 // atomic built-ins were introduced ~4.2
#define __thrust_compiler_fence() __sync_synchronize()
#else
// allow the code to compile without any guarantees
#define __thrust_compiler_fence() do {} while (0)
#endif // THRUST_GCC_VERSION

// unknown case
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
#define __thrust_compiler_fence() __sync_synchronize()
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_UNKNOWN

// allow the code to compile without any guarantees
#define __thrust_compiler_fence() do {} while (0)

#endif

