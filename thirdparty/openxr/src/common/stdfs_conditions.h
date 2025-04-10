// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT

#ifndef _STDFS_CONDITIONS_H
#define _STDFS_CONDITIONS_H

// If the C++ macro is set to the version containing C++17, it must support
// the final C++17 package
#if __cplusplus >= 201703L
#define USE_EXPERIMENTAL_FS 0
#define USE_FINAL_FS 1

#elif defined(_MSC_VER) && _MSC_VER >= 1900

#if defined(_HAS_CXX17) && _HAS_CXX17
// When MSC supports c++17 use <filesystem> package.
#define USE_EXPERIMENTAL_FS 0
#define USE_FINAL_FS 1
#endif  // !_HAS_CXX17

// GCC supports the experimental filesystem items starting in GCC 6
#elif (__GNUC__ >= 6)
#define USE_EXPERIMENTAL_FS 1
#define USE_FINAL_FS 0

// If Clang, check for feature support
#elif defined(__clang__) && (__cpp_lib_filesystem || __cpp_lib_experimental_filesystem)
#if __cpp_lib_filesystem
#define USE_EXPERIMENTAL_FS 0
#define USE_FINAL_FS 1
#else
#define USE_EXPERIMENTAL_FS 1
#define USE_FINAL_FS 0
#endif

// If all above fails, fall back to standard C++ and OS-specific items
#else
#define USE_EXPERIMENTAL_FS 0
#define USE_FINAL_FS 0
#endif

#endif  // !_STDFS_CONDITIONS_H
