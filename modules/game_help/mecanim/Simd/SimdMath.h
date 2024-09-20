#pragma once

#include "Runtime/Utilities/Annotations.h"

// Legacy SIMD math utilities (V4*, used in some Matrix4x4f functions). Only a very small set,
// and only specialized for SSE. Do not use for new code (use vec-math.h and others instead)!


#if UNITY_SUPPORTS_SSE

#   if PLATFORM_WIN
#       include <intrin.h>
#   else
#       include <xmmintrin.h>
#   endif

typedef __m128 Simd128;

// Load / Save
#   define V4LoadUnaligned(base, offset)            _mm_loadu_ps((base)+(offset))
#   define V4StoreUnaligned(value, base, offset)    _mm_storeu_ps((base)+(offset), value)

// Math functions
#   define V4Add(v0, v1)        _mm_add_ps((v0), (v1))
#   define V4Mul(v0, v1)        _mm_mul_ps((v0), (v1))
#   define V4MulAdd(v0, v1, v2) _mm_add_ps(_mm_mul_ps((v0), (v1)), (v2))

// Shuffling / Permuting / Splatting / Merging
#   define V4Splat(v0, i)           _mm_shuffle_ps((v0), (v0), _MM_SHUFFLE(i,i,i,i))

// Attention! : these are done after PPC big-endian specs.
#   define V4MergeL(v0, v1)         _mm_unpackhi_ps((v0), (v1))
#   define V4MergeH(v0, v1)         _mm_unpacklo_ps((v0), (v1))
#else


#include "Runtime/Math/Vector4.h"

typedef Vector4f Simd128;

// Load / Save
#   define V4LoadUnaligned(base, offset)            Vector4f((base)+(offset))
#   define V4StoreUnaligned(value, base, offset)    memcpy ((base)+(offset), value.GetPtr(), sizeof (float) * 4)

// Math functions
#   define V4Add(v0, v1)        (v0 + v1)
#   define V4Mul(v0, v1)        (v0 * v1)
#   define V4MulAdd(v0, v1, v2) (v0 * v1 + v2)

// Shuffling / Permuting / Splatting / Merging
#   define V4Splat(v0, i)           Vector4f(v0[i], v0[i], v0[i], v0[i])

// Attention! : these are done after PPC big-endian specs.
#   define V4MergeL(v0, v1)         (TBD: NOT IMPLEMENTED)
#   define V4MergeH(v0, v1)         (TBD: NOT IMPLEMENTED)


#endif
