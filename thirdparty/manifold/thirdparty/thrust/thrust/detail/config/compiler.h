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

/*! \file compiler.h
 *  \brief Compiler-specific configuration
 */

#pragma once

// enumerate host compilers we know about
#define THRUST_HOST_COMPILER_UNKNOWN 0
#define THRUST_HOST_COMPILER_MSVC    1
#define THRUST_HOST_COMPILER_GCC     2
#define THRUST_HOST_COMPILER_CLANG   3
#define THRUST_HOST_COMPILER_INTEL   4

// enumerate device compilers we know about
#define THRUST_DEVICE_COMPILER_UNKNOWN 0
#define THRUST_DEVICE_COMPILER_MSVC    1
#define THRUST_DEVICE_COMPILER_GCC     2
#define THRUST_DEVICE_COMPILER_CLANG   3
#define THRUST_DEVICE_COMPILER_NVCC    4

// figure out which host compiler we're using
// XXX we should move the definition of THRUST_DEPRECATED out of this logic
#if   defined(_MSC_VER)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_MSVC
#define THRUST_MSVC_VERSION _MSC_VER
#define THRUST_MSVC_VERSION_FULL _MSC_FULL_VER
#elif defined(__ICC)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_INTEL
#elif defined(__clang__)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_CLANG
#define THRUST_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_GCC
#define THRUST_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if (THRUST_GCC_VERSION >= 50000)
#define THRUST_MODERN_GCC
#else
#define THRUST_LEGACY_GCC
#endif
#else
#define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_UNKNOWN
#endif // THRUST_HOST_COMPILER

// figure out which device compiler we're using
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_NVCC
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_MSVC
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_GCC
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
// CUDA-capable clang should behave similar to NVCC.
#if defined(__CUDA__)
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_NVCC
#else
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_CLANG
#endif
#else
#define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_UNKNOWN
#endif

// is the device compiler capable of compiling omp?
#if defined(_OPENMP) || defined(_NVHPC_STDPAR_OPENMP)
#define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_TRUE
#else
#define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_FALSE
#endif // _OPENMP


#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && !defined(__CUDA_ARCH__)
  #define THRUST_DISABLE_MSVC_WARNING_BEGIN(x)                                \
    __pragma(warning(push))                                                   \
    __pragma(warning(disable : x))                                            \
    /**/
  #define THRUST_DISABLE_MSVC_WARNING_END(x)                                  \
    __pragma(warning(pop))                                                    \
    /**/
#else
  #define THRUST_DISABLE_MSVC_WARNING_BEGIN(x)
  #define THRUST_DISABLE_MSVC_WARNING_END(x)
#endif

#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG) && !defined(__CUDA_ARCH__)
  #define THRUST_IGNORE_CLANG_WARNING_IMPL(x)                                 \
    THRUST_PP_STRINGIZE(clang diagnostic ignored x)                           \
    /**/
  #define THRUST_IGNORE_CLANG_WARNING(x)                                      \
    THRUST_IGNORE_CLANG_WARNING_IMPL(THRUST_PP_STRINGIZE(x))                  \
    /**/

  #define THRUST_DISABLE_CLANG_WARNING_BEGIN(x)                               \
    _Pragma("clang diagnostic push")                                          \
    _Pragma(THRUST_IGNORE_CLANG_WARNING(x))                                   \
    /**/
  #define THRUST_DISABLE_CLANG_WARNING_END(x)                                 \
    _Pragma("clang diagnostic pop")                                           \
    /**/
#else
  #define THRUST_DISABLE_CLANG_WARNING_BEGIN(x)
  #define THRUST_DISABLE_CLANG_WARNING_END(x)
#endif

#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) && !defined(__CUDA_ARCH__)
  #define THRUST_IGNORE_GCC_WARNING_IMPL(x)                                   \
    THRUST_PP_STRINGIZE(GCC diagnostic ignored x)                             \
    /**/
  #define THRUST_IGNORE_GCC_WARNING(x)                                        \
    THRUST_IGNORE_GCC_WARNING_IMPL(THRUST_PP_STRINGIZE(x))                    \
    /**/

  #define THRUST_DISABLE_GCC_WARNING_BEGIN(x)                                 \
    _Pragma("GCC diagnostic push")                                            \
    _Pragma(THRUST_IGNORE_GCC_WARNING(x))                                     \
    /**/
  #define THRUST_DISABLE_GCC_WARNING_END(x)                                   \
    _Pragma("GCC diagnostic pop")                                             \
    /**/
#else
  #define THRUST_DISABLE_GCC_WARNING_BEGIN(x)
  #define THRUST_DISABLE_GCC_WARNING_END(x)
#endif

#define THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN               \
  THRUST_DISABLE_MSVC_WARNING_BEGIN(4244 4267)                                \
  /**/
#define THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END                 \
  THRUST_DISABLE_MSVC_WARNING_END(4244 4267)                                  \
  /**/
#define THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING(x)                  \
  THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN                     \
  x;                                                                          \
  THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END                       \
  /**/

#define THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN               \
  THRUST_DISABLE_MSVC_WARNING_BEGIN(4800)                                     \
  /**/
#define THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END                 \
  THRUST_DISABLE_MSVC_WARNING_END(4800)                                       \
  /**/
#define THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING(x)                  \
  THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN                     \
  x;                                                                          \
  THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END                       \
  /**/

#define THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_BEGIN                    \
  THRUST_DISABLE_CLANG_WARNING_BEGIN(-Wself-assign)                           \
  /**/
#define THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_END                      \
  THRUST_DISABLE_CLANG_WARNING_END(-Wself-assign)                             \
  /**/
#define THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING(x)                       \
  THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_BEGIN                          \
  x;                                                                          \
  THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING_END                            \
  /**/

#define THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_BEGIN     \
  THRUST_DISABLE_CLANG_WARNING_BEGIN(-Wreorder)                               \
  THRUST_DISABLE_GCC_WARNING_BEGIN(-Wreorder)                                 \
  /**/
#define THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_END       \
  THRUST_DISABLE_CLANG_WARNING_END(-Wreorder)                                 \
  THRUST_DISABLE_GCC_WARNING_END(-Wreorder)                                   \
  /**/
#define THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING(x)        \
  THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_BEGIN           \
  x;                                                                          \
  THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_END             \
  /**/


