// Copyright (c) 2017-2026 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

#if defined(__GNUC__) && __GNUC__ >= 4
#define PLATFORM_EXPORT __attribute__((visibility("default")))
#elif defined(__SUNPRO_C) && (__SUNPRO_C >= 0x590)
#define PLATFORM_EXPORT __attribute__((visibility("default")))
#elif defined(_WIN32)
#define PLATFORM_EXPORT __declspec(dllexport)
#else
#define PLATFORM_EXPORT
#endif
