#pragma once

#include "config.h"

#include "vec-intrin.h"
#include "vec-types.h"

namespace math
{
#if defined(MATH_HAS_CONSTRUCTOR_DEFINE)
    #undef int3
    #undef int2
    #undef int4

    #undef float3
    #undef float2
    #undef float4
#endif

    // represent 4 rgba colors as bytes.
    // All operations must match the byte order defined by ColorRGBA32
    struct pix4
    {
        int4 i;
        MATH_EMPTYINLINE pix4() {}
        MATH_FORCEINLINE pix4(const int4& _i) : i(_i) {}
    #if defined(MATH_HAS_NATIVE_SIMD)
        MATH_FORCEINLINE pix4(const int1& x, const int1& y, const int1& z, const int1& w)
        {
            i = int4_ctor(x, y, z, w);
        }

    #else
        MATH_FORCEINLINE pix4(const int1& x, const int1& y, const int1& z, const int1& w) : i(x, y, z, w) {}
    #endif
        MATH_FORCEINLINE int4 operator==(const pix4& p) { return i == p.i; }
        MATH_FORCEINLINE int4 operator==(const int4& p) { return i == p; }
        MATH_FORCEINLINE pix4& operator=(const pix4& p) { i = p.i; return *this; }
        MATH_FORCEINLINE operator int4() const { return i; }
    };
#if defined(MATH_HAS_CONSTRUCTOR_DEFINE)
    #define int3(...)   int3_ctor(__VA_ARGS__)
    #define int2(...)   int2_ctor(__VA_ARGS__)
    #define int4(...)   int4_ctor(__VA_ARGS__)

    #define float3(...) float3_ctor(__VA_ARGS__)
    #define float2(...) float2_ctor(__VA_ARGS__)
    #define float4(...) float4_ctor(__VA_ARGS__)
#endif
}
