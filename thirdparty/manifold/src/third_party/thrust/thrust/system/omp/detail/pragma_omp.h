/******************************************************************************
* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#pragma once

#include <thrust/detail/config.h>

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// MSVC ICEs when using the standard C++11 `_Pragma` operator with OpenMP
// directives.
// WAR this by using the MSVC-extension `__pragma`. See this link for more info:
// https://developercommunity.visualstudio.com/t/Using-C11s-_Pragma-with-OpenMP-dire/1590628
#define THRUST_PRAGMA_OMP_IMPL(directive) __pragma(directive)
#else // Not MSVC:
#define THRUST_PRAGMA_OMP_IMPL(directive) _Pragma(#directive)
#endif

// For internal use only -- THRUST_PRAGMA_OMP is used to switch between
// different flavors of openmp pragmas. Pragmas are not emitted when OpenMP is
// not available.
//
// Usage:
//   Replace: #pragma omp parallel for
//   With   : THRUST_PRAGMA_OMP(parallel for)
//
#if defined(_NVHPC_STDPAR_OPENMP) && _NVHPC_STDPAR_OPENMP == 1
#define THRUST_PRAGMA_OMP(directive) THRUST_PRAGMA_OMP_IMPL(omp_stdpar directive)
#elif defined(_OPENMP)
#define THRUST_PRAGMA_OMP(directive) THRUST_PRAGMA_OMP_IMPL(omp directive)
#else
#define THRUST_PRAGMA_OMP(directive)
#endif
