

#include "vec-math.h"

namespace math
{
#if defined(MATH_HAS_SIMD_FLOAT)
    static MATH_FORCEINLINE float4 rsqrte_internal(const float4 &x)
    {
#   if defined(__ARM_NEON)
        return vrsqrteq_f32((float32x4_t)x);
#   elif defined(__SSE__)
        return _mm_rsqrt_ps((__m128)x);
#   else
        int4 i = as_int4(x); i = 0x5f37642f - (i >> 1);
        return as_float4(i);
#   endif
    }

    float4 rsqrte_magicnumber = float4(1.f) / rsqrte_internal(float4(1.f));
#endif
}
