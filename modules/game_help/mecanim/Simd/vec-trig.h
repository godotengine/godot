#pragma once

#include "vec-intrin.h"
#include "vec-scalar.h"

#if defined(_MSC_VER)
#   include <cstdlib>
#endif

#if !defined(MATH_HAS_SIMD_FLOAT) || !defined(__GNUC__) || !defined(__OPTIMIZE__)
#   include <cmath>
#endif

#include "vec-types.h"
#include "vec-const.h"
#include "vec-math.h"

namespace math
{
    static MATH_FORCEINLINE float4 poly13_atan(const float4 &x, const float4 &c13, const float4 &c11, const float4 &c9, const float4 &c7, const float4 &c5, const float4 &c3, const float4 &c1)
    {
        float4 x2 = x * x;
        float4 x4 = x2 * x2;
        float4 x6 = x4 * x2;
        float4 x8 = x6 * x2;
        float4 x10 = x8 * x2;
        float4 x12 = x10 * x2;
        return x * (mad(c13, x12, mad(c11, x10, mad(c9, x8, mad(c7, x6, mad(c5, x4, mad(c3, x2, c1)))))));
    }

    static MATH_FORCEINLINE float4 poly7_atan(const float4 &x, const float4 &c7, const float4 &c5, const float4 &c3, const float4 &c1)
    {
        float4 x2 = x * x;
        float4 x4 = x2 * x2;
        float4 x6 = x4 * x2;
        return x * (mad(c7, x6, mad(c5, x4, mad(c3, x2, c1))));
    }

namespace highp
{
        #if defined(VEC_MATH_SIN_COS_POLY_C0)
            #undef VEC_MATH_SIN_COS_POLY_C0
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY_C1)
            #undef VEC_MATH_SIN_COS_POLY_C1
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY_C2)
            #undef VEC_MATH_SIN_COS_POLY_C2
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY_C3)
            #undef VEC_MATH_SIN_COS_POLY_C3
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY_C4)
            #undef VEC_MATH_SIN_COS_POLY_C4
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY)
            #undef VEC_MATH_SIN_COS_POLY
        #endif

        #if defined(VEC_MATH_ATAN_POLY_C1)
            #undef VEC_MATH_ATAN_POLY_C1
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C3)
            #undef VEC_MATH_ATAN_POLY_C3
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C5)
            #undef VEC_MATH_ATAN_POLY_C5
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C7)
            #undef VEC_MATH_ATAN_POLY_C7
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C9)
            #undef VEC_MATH_ATAN_POLY_C9
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C11)
            #undef VEC_MATH_ATAN_POLY_C11
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C13)
            #undef VEC_MATH_ATAN_POLY_C13
        #endif
        #if defined(VEC_MATH_ATAN_POLY)
            #undef VEC_MATH_ATAN_POLY
        #endif

    // 21 bits precision
        #define VEC_MATH_SIN_COS_POLY_C0 6.283185005f
        #define VEC_MATH_SIN_COS_POLY_C1 -41.341659546f
        #define VEC_MATH_SIN_COS_POLY_C2 81.601821899f
        #define VEC_MATH_SIN_COS_POLY_C3 -76.568618774f
        #define VEC_MATH_SIN_COS_POLY_C4 39.657032013f
        #define VEC_MATH_SIN_COS_POLY(x)  x*poly4(x*x, VEC_MATH_SIN_COS_POLY_C4, VEC_MATH_SIN_COS_POLY_C3, VEC_MATH_SIN_COS_POLY_C2, VEC_MATH_SIN_COS_POLY_C1, VEC_MATH_SIN_COS_POLY_C0)

    // 20-21 bits precision
        #define VEC_MATH_ATAN_POLY_C1 0.999999464f
        #define VEC_MATH_ATAN_POLY_C3 -0.333264589f
        #define VEC_MATH_ATAN_POLY_C5 0.198815241f
        #define VEC_MATH_ATAN_POLY_C7 -0.134872660f
        #define VEC_MATH_ATAN_POLY_C9 0.083871357f
        #define VEC_MATH_ATAN_POLY_C11 -0.037013143f
        #define VEC_MATH_ATAN_POLY_C13 0.007862508f
        #define VEC_MATH_ATAN_POLY(x)   poly13_atan(x, VEC_MATH_ATAN_POLY_C13, VEC_MATH_ATAN_POLY_C11, VEC_MATH_ATAN_POLY_C9, VEC_MATH_ATAN_POLY_C7, VEC_MATH_ATAN_POLY_C5, VEC_MATH_ATAN_POLY_C3, VEC_MATH_ATAN_POLY_C1)
        #include "vec-trig-internal.h"
}

namespace mediump
{
        #if defined(VEC_MATH_SIN_COS_POLY_C0)
            #undef VEC_MATH_SIN_COS_POLY_C0
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY_C1)
            #undef VEC_MATH_SIN_COS_POLY_C1
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY_C2)
            #undef VEC_MATH_SIN_COS_POLY_C2
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY_C3)
            #undef VEC_MATH_SIN_COS_POLY_C3
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY_C4)
            #undef VEC_MATH_SIN_COS_POLY_C4
        #endif
        #if defined(VEC_MATH_SIN_COS_POLY)
            #undef VEC_MATH_SIN_COS_POLY
        #endif

        #if defined(VEC_MATH_ATAN_POLY_C1)
            #undef VEC_MATH_ATAN_POLY_C1
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C3)
            #undef VEC_MATH_ATAN_POLY_C3
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C5)
            #undef VEC_MATH_ATAN_POLY_C5
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C7)
            #undef VEC_MATH_ATAN_POLY_C7
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C9)
            #undef VEC_MATH_ATAN_POLY_C9
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C11)
            #undef VEC_MATH_ATAN_POLY_C11
        #endif
        #if defined(VEC_MATH_ATAN_POLY_C13)
            #undef VEC_MATH_ATAN_POLY_C13
        #endif
        #if defined(VEC_MATH_ATAN_POLY)
            #undef VEC_MATH_ATAN_POLY
        #endif

    // 11 bits precision
        #define VEC_MATH_SIN_COS_POLY_C0 6.2831854820251464844f
        #define VEC_MATH_SIN_COS_POLY_C1 -41.283184051513671875f
        #define VEC_MATH_SIN_COS_POLY_C2 76.035461425781250000f
        #define VEC_MATH_SIN_COS_POLY(x)  x*poly2(x*x, VEC_MATH_SIN_COS_POLY_C2, VEC_MATH_SIN_COS_POLY_C1, VEC_MATH_SIN_COS_POLY_C0)

    // 11 bits precision
        #define VEC_MATH_ATAN_POLY_C1 0.999802172f
        #define VEC_MATH_ATAN_POLY_C3 -0.325227708f
        #define VEC_MATH_ATAN_POLY_C5 0.153163940f
        #define VEC_MATH_ATAN_POLY_C7 -0.042340223f
        #define VEC_MATH_ATAN_POLY(x)  poly7_atan(x, VEC_MATH_ATAN_POLY_C7, VEC_MATH_ATAN_POLY_C5, VEC_MATH_ATAN_POLY_C3, VEC_MATH_ATAN_POLY_C1)
        #include "vec-trig-internal.h"
}

    // Default precision is high
    // if you want to use another precision level, please use the full qualified name: mediump::sin()
    using namespace highp;
}
