// This code is in the public domain -- castano@gmail.com

#pragma once
#ifndef NV_MATH_FTOI_H
#define NV_MATH_FTOI_H

#include "nvmath/nvmath.h"

#include <math.h>

namespace nv
{
    // Optimized float to int conversions. See:
    // http://cbloomrants.blogspot.com/2009/01/01-17-09-float-to-int.html
    // http://www.stereopsis.com/sree/fpu2006.html
    // http://assemblyrequired.crashworks.org/2009/01/12/why-you-should-never-cast-floats-to-ints/
    // http://chrishecker.com/Miscellaneous_Technical_Articles#Floating_Point


    union DoubleAnd64 {
        uint64    i;
        double    d;
    };

    static const double floatutil_xs_doublemagic = (6755399441055744.0);                            // 2^52 * 1.5
    static const double floatutil_xs_doublemagicdelta = (1.5e-8);                                   // almost .5f = .5f + 1e^(number of exp bit)
    static const double floatutil_xs_doublemagicroundeps = (0.5f - floatutil_xs_doublemagicdelta);  // almost .5f = .5f - 1e^(number of exp bit)

    NV_FORCEINLINE int ftoi_round_xs(double val, double magic) {
#if 1
        DoubleAnd64 dunion;
        dunion.d = val + magic;
        return (int32) dunion.i; // just cast to grab the bottom bits
#else
        val += magic;
        return ((int*)&val)[0]; // @@ Assumes little endian.
#endif
    }

    NV_FORCEINLINE int ftoi_round_xs(float val) {
        return ftoi_round_xs(val, floatutil_xs_doublemagic);
    }

    NV_FORCEINLINE int ftoi_floor_xs(float val) {
        return ftoi_round_xs(val - floatutil_xs_doublemagicroundeps, floatutil_xs_doublemagic);
    }

    NV_FORCEINLINE int ftoi_ceil_xs(float val) {
        return ftoi_round_xs(val + floatutil_xs_doublemagicroundeps, floatutil_xs_doublemagic);
    }

    NV_FORCEINLINE int ftoi_trunc_xs(float val) {
        return (val<0) ? ftoi_ceil_xs(val) : ftoi_floor_xs(val);
    }

// -- GODOT start --
//#if NV_CPU_X86 || NV_CPU_X86_64
#if NV_USE_SSE
// -- GODOT end --

    NV_FORCEINLINE int ftoi_round_sse(float f) {
        return _mm_cvt_ss2si(_mm_set_ss(f));
    }

    NV_FORCEINLINE int ftoi_trunc_sse(float f) {
      return _mm_cvtt_ss2si(_mm_set_ss(f));
    }

#endif



#if NV_USE_SSE

    NV_FORCEINLINE int ftoi_round(float val) {
        return ftoi_round_sse(val);
    }

    NV_FORCEINLINE int ftoi_trunc(float f) {
      return ftoi_trunc_sse(f);
    }

    // We can probably do better than this. See for example:
    // http://dss.stephanierct.com/DevBlog/?p=8
    NV_FORCEINLINE int ftoi_floor(float val) {
        return ftoi_round(floorf(val));
    }

    NV_FORCEINLINE int ftoi_ceil(float val) {
        return ftoi_round(ceilf(val));
    }

#else

    // In theory this should work with any double floating point math implementation, but it appears that MSVC produces incorrect code
    // when SSE2 is targeted and fast math is enabled (/arch:SSE2 & /fp:fast). These problems go away with /fp:precise, which is the default mode.

    NV_FORCEINLINE int ftoi_round(float val) {
        return ftoi_round_xs(val);
    }

    NV_FORCEINLINE int ftoi_floor(float val) {
        return ftoi_floor_xs(val);
    }

    NV_FORCEINLINE int ftoi_ceil(float val) {
        return ftoi_ceil_xs(val);
    }

    NV_FORCEINLINE int ftoi_trunc(float f) {
      return ftoi_trunc_xs(f);
    }

#endif


    inline void test_ftoi() {

        // Round to nearest integer.
        nvCheck(ftoi_round(0.1f) == 0);
        nvCheck(ftoi_round(0.6f) == 1);
        nvCheck(ftoi_round(-0.2f) == 0);
        nvCheck(ftoi_round(-0.7f) == -1);
        nvCheck(ftoi_round(10.1f) == 10);
        nvCheck(ftoi_round(10.6f) == 11);
        nvCheck(ftoi_round(-90.1f) == -90);
        nvCheck(ftoi_round(-90.6f) == -91);

        nvCheck(ftoi_round(0) == 0);
        nvCheck(ftoi_round(1) == 1);
        nvCheck(ftoi_round(-1) == -1);
        
        nvCheck(ftoi_round(0.5f) == 0);  // How are midpoints rounded? Bankers rounding.
        nvCheck(ftoi_round(1.5f) == 2);
        nvCheck(ftoi_round(2.5f) == 2);
        nvCheck(ftoi_round(3.5f) == 4);
        nvCheck(ftoi_round(4.5f) == 4);
        nvCheck(ftoi_round(-0.5f) == 0);
        nvCheck(ftoi_round(-1.5f) == -2);
                

        // Truncation (round down if > 0, round up if < 0).
        nvCheck(ftoi_trunc(0.1f) == 0);
        nvCheck(ftoi_trunc(0.6f) == 0);
        nvCheck(ftoi_trunc(-0.2f) == 0);
        nvCheck(ftoi_trunc(-0.7f) == 0);    // @@ When using /arch:SSE2 in Win32, msvc produce wrong code for this one. It is skipping the addition.
        nvCheck(ftoi_trunc(1.99f) == 1);
        nvCheck(ftoi_trunc(-1.2f) == -1);

        // Floor (round down).
        nvCheck(ftoi_floor(0.1f) == 0);
        nvCheck(ftoi_floor(0.6f) == 0);
        nvCheck(ftoi_floor(-0.2f) == -1);
        nvCheck(ftoi_floor(-0.7f) == -1);
        nvCheck(ftoi_floor(1.99f) == 1);
        nvCheck(ftoi_floor(-1.2f) == -2);

        nvCheck(ftoi_floor(0) == 0);
        nvCheck(ftoi_floor(1) == 1);
        nvCheck(ftoi_floor(-1) == -1);
        nvCheck(ftoi_floor(2) == 2);
        nvCheck(ftoi_floor(-2) == -2);

        // Ceil (round up).
        nvCheck(ftoi_ceil(0.1f) == 1);
        nvCheck(ftoi_ceil(0.6f) == 1);
        nvCheck(ftoi_ceil(-0.2f) == 0);
        nvCheck(ftoi_ceil(-0.7f) == 0);
        nvCheck(ftoi_ceil(1.99f) == 2);
        nvCheck(ftoi_ceil(-1.2f) == -1);

        nvCheck(ftoi_ceil(0) == 0);
        nvCheck(ftoi_ceil(1) == 1);
        nvCheck(ftoi_ceil(-1) == -1);
        nvCheck(ftoi_ceil(2) == 2);
        nvCheck(ftoi_ceil(-2) == -2);
    }





    // Safe versions using standard casts.

    inline int iround(float f)
    {
        return ftoi_round(f);
        //return int(floorf(f + 0.5f));
    }

    inline int iround(double f)
    {
        return int(::floor(f + 0.5));
    }

    inline int ifloor(float f)
    {
        return ftoi_floor(f);
        //return int(floorf(f));
    }

    inline int iceil(float f)
    {
        return int(ceilf(f));
    }



    // I'm always confused about which quantizer to use. I think we should choose a quantizer based on how the values are expanded later and this is generally using the 'exact endpoints' rule.
    // Some notes from cbloom: http://cbloomrants.blogspot.com/2011/07/07-26-11-pixel-int-to-float-options.html

    // Quantize a float in the [0,1] range, using exact end points or uniform bins.
    inline float quantizeFloat(float x, uint bits, bool exactEndPoints = true) {
        nvDebugCheck(bits <= 16);

        float range = float(1 << bits);
        if (exactEndPoints) {
            return floorf(x * (range-1) + 0.5f) / (range-1);
        }
        else {
            return (floorf(x * range) + 0.5f) / range;
        }
    }


    // This is the most common rounding mode:
    // 
    //   0     1       2     3
    // |___|_______|_______|___|
    // 0                       1
    //
    // You get that if you take the unit floating point number multiply by 'N-1' and round to nearest. That is, `i = round(f * (N-1))`.
    // You reconstruct the original float dividing by 'N-1': `f = i / (N-1)`


    //    0     1     2     3
    // |_____|_____|_____|_____|
    // 0                       1

    /*enum BinningMode {
        RoundMode_ExactEndPoints,       
        RoundMode_UniformBins,
    };*/

    template <int N>
    inline uint unitFloatToFixed(float f) {
        return ftoi_round(f * ((1<<N)-1));
    }

    inline uint8 unitFloatToFixed8(float f) {
        return (uint8)unitFloatToFixed<8>(f);
    }

    inline uint16 unitFloatToFixed16(float f) {
        return (uint16)unitFloatToFixed<16>(f);
    }


} // nv

#endif // NV_MATH_FTOI_H
