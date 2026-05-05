/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_COMPILER_ATTRIBUTES_H_
#define VPX_VPX_PORTS_COMPILER_ATTRIBUTES_H_

#if !defined(__has_feature)
#define __has_feature(x) 0
#endif  // !defined(__has_feature)

#if !defined(__has_attribute)
#define __has_attribute(x) 0
#endif  // !defined(__has_attribute)

//------------------------------------------------------------------------------
// Sanitizer attributes.

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define VPX_WITH_ASAN 1
#else
#define VPX_WITH_ASAN 0
#endif  // __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)

#if defined(__clang__) && __has_attribute(no_sanitize)
// Both of these have defined behavior and are used in certain operations or
// optimizations thereof. There are cases where an overflow may be unintended,
// however, so use of these attributes should be done with care.
#define VPX_NO_UNSIGNED_OVERFLOW_CHECK \
  __attribute__((no_sanitize("unsigned-integer-overflow")))
#if __clang_major__ >= 12
#define VPX_NO_UNSIGNED_SHIFT_CHECK \
  __attribute__((no_sanitize("unsigned-shift-base")))
#endif  // __clang__ >= 12
#endif  // __clang__

#ifndef VPX_NO_UNSIGNED_OVERFLOW_CHECK
#define VPX_NO_UNSIGNED_OVERFLOW_CHECK
#endif
#ifndef VPX_NO_UNSIGNED_SHIFT_CHECK
#define VPX_NO_UNSIGNED_SHIFT_CHECK
#endif

//------------------------------------------------------------------------------
// Variable attributes.

#if __has_attribute(uninitialized)
// Attribute "uninitialized" disables -ftrivial-auto-var-init=pattern for
// the specified variable.
//
// -ftrivial-auto-var-init is security risk mitigation feature, so attribute
// should not be used "just in case", but only to fix real performance
// bottlenecks when other approaches do not work. In general the compiler is
// quite effective at eliminating unneeded initializations introduced by the
// flag, e.g. when they are followed by actual initialization by a program.
// However if compiler optimization fails and code refactoring is hard, the
// attribute can be used as a workaround.
#define VPX_UNINITIALIZED __attribute__((uninitialized))
#else
#define VPX_UNINITIALIZED
#endif  // __has_attribute(uninitialized)

#endif  // VPX_VPX_PORTS_COMPILER_ATTRIBUTES_H_
