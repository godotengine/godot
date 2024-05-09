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

/*! \file exec_check_disable.h
 *  \brief Defines __thrust_exec_check_disable__
 */

#pragma once

#include <thrust/detail/config.h>

// #pragma nv_exec_check_disable is only recognized by NVCC.  Having a macro
// expand to a #pragma (rather than _Pragma) only works with NVCC's compilation
// model, not with other compilers.
#if defined(__CUDACC__) && !defined(_NVHPC_CUDA) && \
    !(defined(__CUDA__) && defined(__clang__))

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#define __thrust_exec_check_disable__ __pragma("nv_exec_check_disable")
#else // MSVC
#define __thrust_exec_check_disable__ _Pragma("nv_exec_check_disable")
#endif // MSVC

#else

#define __thrust_exec_check_disable__

#endif


