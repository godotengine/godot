#pragma once

#include "config.h"

#if defined(_M_X64) && !defined(__SSE2__)
#   define __SSE2__
#endif

#if defined(_M_IX86_FP)
#   if _M_IX86_FP >= 2 && !defined(__SSE2__)
#       define __SSE2__
#   endif
#   if _M_IX86_FP >= 1 && !defined(__SSE__)
#       define __SSE__
#   endif
#endif

#if defined(__ARM_NEON)

#   if defined(_M_ARM64)
#       include <arm64_neon.h>
#   else
#       include <arm_neon.h>
#   endif

typedef float32x4_t cv4f;
typedef int32x4_t cv4i;

#   define MATH_HAS_SIMD_INT    4
#   define MATH_HAS_SIMD_FLOAT  4

#   ifndef _MSC_VER
#       define cv4f(x, y, z, w)     ((const float32x4_t) { (x), (y), (z), (w) })
#       define cv4i(x, y, z, w)     ((const int32x4_t) { (int) (x), (int) (y), (int) (z), (int) (w) })
#   else
// MSVC does not support vector literal declaration with the form (const float32x4_t) { (x), (y), (z), (w) }
#       define cv4f(x, y, z, w) ([](){ static constexpr float s_Value[] = { x, y, z, w }; return vld1q_f32(s_Value); }())
#       define cv4i(x, y, z, w) ([](){ static constexpr int s_Value[] = { x, y, z, w }; return vld1q_s32(s_Value); }())
#   endif

#elif !defined(USE_GENERIC_VEC) && (defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__) || defined(__AVX__))

#   if defined(__AVX__)
#       include <immintrin.h>
/* add missing defines for Visual C */
#       if !defined(__SSE4_2__)
#           define __SSE4_2__
#       endif
#       if !defined(__SSE4_1__)
#           define __SSE4_1__
#       endif
#       if !defined(__SSSE3__)
#           define __SSSE3__
#       endif
#       if !defined(__SSE3__)
#           define __SSE3__
#       endif
#       if !defined(__SSE2__)
#           define __SSE2__
#       endif
#       if !defined(__SSE__)
#           define __SSE__
#       endif
#   elif defined(__SSE4_2__)
#       include <nmmintrin.h>
/* add missing defines for Visual C */
#       if !defined(__SSE4_1__)
#           define __SSE4_1__
#       endif
#       if !defined(__SSSE3__)
#           define __SSSE3__
#       endif
#       if !defined(__SSE3__)
#           define __SSE3__
#       endif
#       if !defined(__SSE2__)
#           define __SSE2__
#       endif
#       if !defined(__SSE__)
#           define __SSE__
#       endif
#   elif defined(__SSE4_1__)
#       include <smmintrin.h>
/* add missing defines for Visual C */
#       if !defined(__SSSE3__)
#           define __SSSE3__
#       endif
#       if !defined(__SSE3__)
#           define __SSE3__
#       endif
#       if !defined(__SSE2__)
#           define __SSE2__
#       endif
#       if !defined(__SSE__)
#           define __SSE__
#       endif
#   elif defined(__SSSE3__)
#       include <tmmintrin.h>
/* add missing defines for Visual C */
#       if !defined(__SSE3__)
#           define __SSE3__
#       endif
#       if !defined(__SSE2__)
#           define __SSE2__
#       endif
#       if !defined(__SSE__)
#           define __SSE__
#       endif
#   elif defined(__SSE3__)
#       include <pmmintrin.h>
/* add missing defines for Visual C */
#       if !defined(__SSE2__)
#           define __SSE2__
#       endif
#       if !defined(__SSE__)
#           define __SSE__
#       endif
#   elif defined(__SSE2__)
#       include <emmintrin.h>
/* add missing defines for Visual C */
#       if !defined(__SSE__)
#           define __SSE__
#       endif
#   else
#       include <xmmintrin.h>
#   endif

#   define MATH_HAS_SIMD_INT    4
#   define MATH_HAS_SIMD_FLOAT  4

typedef __m128              cv4f;
typedef __m128i             cv4i;

#   if defined(__clang__)
#       define cv4f(x, y, z, w) ((const __m128) { (x), (y), (z), (w) })
#       define cv4i(x, y, z, w) ((const int __attribute__ ((ext_vector_type(4)))) { (int) (x), (int) (y), (int) (z), (int) (w) })
#   elif defined(__GNUC__)
#       define cv4f(x, y, z, w) ((const __m128) { (x), (y), (z), (w) })
#       define cv4i(x, y, z, w) ((const int __attribute__ ((vector_size(16)))) { (int) (x), (int) (y), (int) (z), (int) (w) })
#   else
#       define cv4f(x, y, z, w) _mm_set_ps(w, z, y, x)
#       define cv4i(x, y, z, w) _mm_set_epi32(w, z, y, x)
#   endif

#   define _mm_cvtf32_ss(x)     _mm_set_ss(x)

#else
typedef struct { float x, y, z, w; } cv4f;
typedef struct { int x, y, z, w; } cv4i;
#define cv4f(x, y, z, w) ((cv4f) { x, y, z, w })
#define cv4i(x, y, z, w) ((cv4i) { x, y, z, w })
#endif
