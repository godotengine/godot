// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_SANITIZER_DEFINITIONS_H_
#define LIB_JXL_BASE_SANITIZER_DEFINITIONS_H_

#ifdef MEMORY_SANITIZER
#define JXL_MEMORY_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define JXL_MEMORY_SANITIZER 1
#else
#define JXL_MEMORY_SANITIZER 0
#endif
#else
#define JXL_MEMORY_SANITIZER 0
#endif

#ifdef ADDRESS_SANITIZER
#define JXL_ADDRESS_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define JXL_ADDRESS_SANITIZER 1
#else
#define JXL_ADDRESS_SANITIZER 0
#endif
#else
#define JXL_ADDRESS_SANITIZER 0
#endif

#ifdef THREAD_SANITIZER
#define JXL_THREAD_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define JXL_THREAD_SANITIZER 1
#else
#define JXL_THREAD_SANITIZER 0
#endif
#else
#define JXL_THREAD_SANITIZER 0
#endif
#endif  // LIB_JXL_BASE_SANITIZER_DEFINITIONS_H
