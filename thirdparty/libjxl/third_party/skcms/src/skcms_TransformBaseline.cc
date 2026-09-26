/*
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "skcms_public.h"     // NO_G3_REWRITE
#include "skcms_internals.h"  // NO_G3_REWRITE
#include "skcms_Transform.h"  // NO_G3_REWRITE
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
    #include <arm_neon.h>
#elif defined(__SSE__)
    #include <immintrin.h>

    #if defined(__clang__)
        // That #include <immintrin.h> is usually enough, but Clang's headers
        // avoid #including the whole kitchen sink when _MSC_VER is defined,
        // because lots of programs on Windows would include that and it'd be
        // a lot slower. But we want all those headers included, so we can use
        // their features (after making runtime checks).
        #include <smmintrin.h>
    #endif
#elif defined(__loongarch_sx)
    #include <lsxintrin.h>
#endif

namespace skcms_private {
namespace baseline {

#if defined(SKCMS_PORTABLE)
    // Build skcms in a portable scalar configuration.
    #define N 1
    template <typename T> using V = T;
#else
    // Build skcms with basic four-line SIMD support. (SSE on Intel, or Neon on ARM)
    #define N 4
    template <typename T> using V = skcms_private::Vec<N,T>;
#endif

#include "Transform_inl.h"

}  // namespace baseline
}  // namespace skcms_private
