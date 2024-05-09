/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#include <thrust/detail/config/cpp_dialect.h>

#include <cstddef>

#ifndef __has_cpp_attribute
#  define __has_cpp_attribute(X) 0
#endif

// Trailing return types seem to confuse Doxygen, and cause it to interpret
// parts of the function's body as new function signatures.
#if defined(THRUST_DOXYGEN)
#  define THRUST_TRAILING_RETURN(...)
#else
#  define THRUST_TRAILING_RETURN(...) -> __VA_ARGS__
#endif

#if THRUST_CPP_DIALECT >= 2014 && __has_cpp_attribute(nodiscard)
#  define THRUST_NODISCARD [[nodiscard]]
#else
#  define THRUST_NODISCARD
#endif

#if THRUST_CPP_DIALECT >= 2017 && __cpp_if_constexpr
#  define THRUST_IF_CONSTEXPR if constexpr
#else
#  define THRUST_IF_CONSTEXPR if
#endif

// FIXME: Combine THRUST_INLINE_CONSTANT and
// THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT into one macro when NVCC properly
// supports `constexpr` globals in host and device code.
#if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
// FIXME: Add this when NVCC supports inline variables.
//#  if   THRUST_CPP_DIALECT >= 2017
//#    define THRUST_INLINE_CONSTANT                 inline constexpr
//#    define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT inline constexpr
#  if THRUST_CPP_DIALECT >= 2011
#    define THRUST_INLINE_CONSTANT                 static const __device__
#    define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr
#  else
#    define THRUST_INLINE_CONSTANT                 static const __device__
#    define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static const
#  endif
#else
// FIXME: Add this when NVCC supports inline variables.
//#  if   THRUST_CPP_DIALECT >= 2017
//#    define THRUST_INLINE_CONSTANT                 inline constexpr
//#    define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT inline constexpr
#  if THRUST_CPP_DIALECT >= 2011
#    define THRUST_INLINE_CONSTANT                 static constexpr
#    define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr
#  else
#    define THRUST_INLINE_CONSTANT                 static const
#    define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static const
#  endif
#endif

// These definitions were intended for internal use only and are now obsolete.
// If you relied on them, consider porting your code to use the functionality
// in libcu++'s <nv/target> header.
// For a temporary workaround, define THRUST_PROVIDE_LEGACY_ARCH_MACROS to make
// them available again. These should be considered deprecated and will be
// fully removed in a future version.
#ifdef THRUST_PROVIDE_LEGACY_ARCH_MACROS
  #ifndef THRUST_IS_DEVICE_CODE
    #if defined(_NVHPC_CUDA)
      #define THRUST_IS_DEVICE_CODE __builtin_is_device_code()
      #define THRUST_IS_HOST_CODE (!__builtin_is_device_code())
      #define THRUST_INCLUDE_DEVICE_CODE 1
      #define THRUST_INCLUDE_HOST_CODE 1
    #elif defined(__CUDA_ARCH__)
      #define THRUST_IS_DEVICE_CODE 1
      #define THRUST_IS_HOST_CODE 0
      #define THRUST_INCLUDE_DEVICE_CODE 1
      #define THRUST_INCLUDE_HOST_CODE 0
    #else
      #define THRUST_IS_DEVICE_CODE 0
      #define THRUST_IS_HOST_CODE 1
      #define THRUST_INCLUDE_DEVICE_CODE 0
      #define THRUST_INCLUDE_HOST_CODE 1
    #endif
  #endif
#endif // THRUST_PROVIDE_LEGACY_ARCH_MACROS
