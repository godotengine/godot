/*
 *  Copyright 2020 NVIDIA Corporation
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

/*! \file cpp_dialect.h
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include <thrust/detail/config/compiler.h>

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - THRUST_IGNORE_DEPRECATED_CPP_DIALECT:
//   Ignore all deprecated C++ dialects and outdated compilers.
// - THRUST_IGNORE_DEPRECATED_CPP_11:
//   Ignore deprecation warnings when compiling with C++11. C++03 and outdated
//   compilers will still issue warnings.
// - THRUST_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with C++03 and C++11 will still issue warnings.

// Check for the CUB opt-outs as well:
#if !defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT) && \
     defined(CUB_IGNORE_DEPRECATED_CPP_DIALECT)
#  define    THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#endif
#if !defined(THRUST_IGNORE_DEPRECATED_CPP_11) && \
     defined(CUB_IGNORE_DEPRECATED_CPP_11)
#  define    THRUST_IGNORE_DEPRECATED_CPP_11
#endif
#if !defined(THRUST_IGNORE_DEPRECATED_COMPILER) && \
     defined(CUB_IGNORE_DEPRECATED_COMPILER)
#  define    THRUST_IGNORE_DEPRECATED_COMPILER
#endif

#ifdef THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#  define THRUST_IGNORE_DEPRECATED_CPP_11
#  define THRUST_IGNORE_DEPRECATED_COMPILER
#endif

// Define this to override the built-in detection.
#ifndef THRUST_CPP_DIALECT

// MSVC does not define __cplusplus correctly. _MSVC_LANG is used instead.
// This macro is only defined in MSVC 2015U3+.
#  ifdef _MSVC_LANG // Do not replace with THRUST_HOST_COMPILER test (see above)
// MSVC2015 reports C++14 but lacks extended constexpr support. Treat as C++11.
#    if THRUST_MSVC_VERSION < 1910 && _MSVC_LANG > 201103L /* MSVC < 2017 && CPP > 2011 */
#      define THRUST_CPLUSPLUS 201103L /* Fix to 2011 */
#    else
#      define THRUST_CPLUSPLUS _MSVC_LANG /* We'll trust this for now. */
#    endif // MSVC 2015 C++14 fix
#  else
#    define THRUST_CPLUSPLUS __cplusplus
#  endif

// Detect current dialect:
#  if THRUST_CPLUSPLUS < 201103L
#    define THRUST_CPP_DIALECT 2003
#  elif THRUST_CPLUSPLUS < 201402L
#    define THRUST_CPP_DIALECT 2011
#  elif THRUST_CPLUSPLUS < 201703L
#    define THRUST_CPP_DIALECT 2014
#  elif THRUST_CPLUSPLUS == 201703L
#    define THRUST_CPP_DIALECT 2017
#  elif THRUST_CPLUSPLUS > 201703L // unknown, but is higher than 2017.
#    define THRUST_CPP_DIALECT 2020
#  endif

#  undef THRUST_CPLUSPLUS // cleanup

#endif // !THRUST_CPP_DIALECT

// Define THRUST_COMPILER_DEPRECATION macro:
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#  define THRUST_COMP_DEPR_IMPL(msg) \
    __pragma(message(__FILE__ ":" THRUST_COMP_DEPR_IMPL0(__LINE__) ": warning: " #msg))
#  define THRUST_COMP_DEPR_IMPL0(x) THRUST_COMP_DEPR_IMPL1(x)
#  define THRUST_COMP_DEPR_IMPL1(x) #x
#else // clang / gcc:
#  define THRUST_COMP_DEPR_IMPL(msg) THRUST_COMP_DEPR_IMPL0(GCC warning #msg)
#  define THRUST_COMP_DEPR_IMPL0(expr) _Pragma(#expr)
#  define THRUST_COMP_DEPR_IMPL1 /* intentionally blank */
#endif

#define THRUST_COMPILER_DEPRECATION(REQ) \
  THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)

#define THRUST_COMPILER_DEPRECATION_SOFT(REQ, CUR) \
  THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)

#ifndef THRUST_IGNORE_DEPRECATED_COMPILER

// Compiler checks:
#  if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC && THRUST_GCC_VERSION < 50000
     THRUST_COMPILER_DEPRECATION(GCC 5.0);
#  elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG && THRUST_CLANG_VERSION < 70000
     THRUST_COMPILER_DEPRECATION(Clang 7.0);
#  elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC && THRUST_MSVC_VERSION < 1910
     // <2017. Hard upgrade message:
     THRUST_COMPILER_DEPRECATION(MSVC 2019 (19.20/16.0/14.20));
#  elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC && THRUST_MSVC_VERSION < 1920
     // >=2017, <2019. Soft deprecation message:
     THRUST_COMPILER_DEPRECATION_SOFT(MSVC 2019 (19.20/16.0/14.20), MSVC 2017);
#  endif

#endif // THRUST_IGNORE_DEPRECATED_COMPILER

#ifndef THRUST_IGNORE_DEPRECATED_DIALECT

// Dialect checks:
#  if THRUST_CPP_DIALECT < 2011
     // <C++11. Hard upgrade message:
     THRUST_COMPILER_DEPRECATION(C++14);
#  elif THRUST_CPP_DIALECT == 2011 && !defined(THRUST_IGNORE_DEPRECATED_CPP_11)
     // =C++11. Soft upgrade message:
     THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
#  endif

#endif // THRUST_IGNORE_DEPRECATED_DIALECT

#undef THRUST_COMPILER_DEPRECATION_SOFT
#undef THRUST_COMPILER_DEPRECATION
#undef THRUST_COMP_DEPR_IMPL
#undef THRUST_COMP_DEPR_IMPL0
#undef THRUST_COMP_DEPR_IMPL1
