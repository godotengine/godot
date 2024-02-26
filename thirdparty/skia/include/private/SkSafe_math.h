/*
 * Copyright 2016 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkSafe_math_DEFINED
#define SkSafe_math_DEFINED

// This file protects against known bugs in ucrt\math.h.
// Namely, that header defines inline methods without marking them static,
// which makes it very easy to cause ODR violations and ensuing chaos.
//
// TODO: other headers?  Here are some potential problem headers:
// $ grep -R __inline * | grep -v static | cut -f 1 -d: | sort | uniq
//   corecrt.h
//   corecrt_stdio_config.h
//   ctype.h
//   fenv.h
//   locale.h
//   malloc.h
//   math.h
//   tchar.h
//   wchar.h
// I took a quick look through other headers outside math.h.
// Nothing looks anywhere near as likely to be used by Skia as math.h.

#if defined(_MSC_VER) && !defined(_INC_MATH)
    // Our strategy here is to simply inject "static" into the headers
    // where it should have been written, just before __inline.
    //
    // Most inline-but-not-static methods in math.h are 32-bit only,
    // but not all of them (see frexpf, hypothf, ldexpf...).  So to
    // be safe, 32- and 64-bit builds both get this treatment.

    #define __inline static __inline
    #include <math.h>
    #undef __inline

    #if !defined(_INC_MATH)
        #error Hmm.  Looks like math.h has changed its header guards.
    #endif

    #define INC_MATH_IS_SAFE_NOW

#else
    #include <math.h>

#endif

#endif//SkSafe_math_DEFINED
