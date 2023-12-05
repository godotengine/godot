/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* Copyright 2019-2018 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/* I'm extending this beyond the purpose implied by its name rather than creating
 * a new file to hold the FALLTHROUGH declaration as this
 * file is already included in most places FALLTHROUGH
 * is needed.
 */

#ifndef _UNUSED_H
#define _UNUSED_H

#if (__cplusplus >= 201703L)
#define MAYBE_UNUSED [[maybe_unused]]
#elif __GNUC__ || __clang__
  #define MAYBE_UNUSED __attribute__((unused))
#else
  // Boohoo. VC++ has no equivalent
  #define MAYBE_UNUSED
#endif

#define U_ASSERT_ONLY MAYBE_UNUSED

// For unused parameters of c functions. Portable.
#define UNUSED(x) (void)(x)

#if !__clang__ && __GNUC__ // grumble ... clang ... grumble
#define FALLTHROUGH __attribute__((fallthrough))
#else
#define FALLTHROUGH
#endif

#endif /* UNUSED_H */
