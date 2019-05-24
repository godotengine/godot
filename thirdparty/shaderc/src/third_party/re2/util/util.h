// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef UTIL_UTIL_H_
#define UTIL_UTIL_H_

#define arraysize(array) (int)(sizeof(array)/sizeof((array)[0]))

#ifndef ATTRIBUTE_NORETURN
#if defined(__GNUC__)
#define ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define ATTRIBUTE_NORETURN __declspec(noreturn)
#else
#define ATTRIBUTE_NORETURN
#endif
#endif

#ifndef FALLTHROUGH_INTENDED
#if defined(__clang__)
#define FALLTHROUGH_INTENDED [[clang::fallthrough]]
#elif defined(__GNUC__) && __GNUC__ >= 7
#define FALLTHROUGH_INTENDED [[gnu::fallthrough]]
#else
#define FALLTHROUGH_INTENDED do {} while (0)
#endif
#endif

#ifndef NO_THREAD_SAFETY_ANALYSIS
#define NO_THREAD_SAFETY_ANALYSIS
#endif

#endif  // UTIL_UTIL_H_
