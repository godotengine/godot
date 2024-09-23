#pragma once
#include <limits>
#   if !defined(M_EPS)
#       define M_EPS        1e-6f
#   endif

#   if !defined(M_DEG_2_RAD)
#       define M_DEG_2_RAD  0.0174532925f
#   endif

#   if !defined(M_RAD_2_DEG)
#       define M_RAD_2_DEG  57.295779513f
#   endif

#   ifndef FLT_MIN_NORMAL
#       define FLT_MIN_NORMAL  1.1754943508222875079687365372222457e-38f
#   endif

#   ifndef M_1_LN2
#       define M_1_LN2  1.44269504088896340735992468100189214
#   endif

namespace math
{
    static MATH_FORCEINLINE float epsilon() { return 1e-6f; }                                           // epsilon
    static MATH_FORCEINLINE float epsilon_second() { return 1e-6f; }
    static MATH_FORCEINLINE float epsilon_scale() { return 1e-9f; }
    static MATH_FORCEINLINE float epsilon_radian() { return 1e-6f; }
    static MATH_FORCEINLINE float epsilon_normal_sqrt() { return 1e-15f; }                              // for testing whether a normal is too small to be normalized (compare with length) sqrt(dot(v))
    static MATH_FORCEINLINE float epsilon_normal() { return 1e-30f; }                                   // for testing whether a normal is too small to be normalized (compare with length squared) dot(v)
    static MATH_FORCEINLINE float epsilon_determinant() { return 1e-6f; }

    static MATH_FORCEINLINE float pi() { return 3.14159265358979323846f; }                              // pi
    static MATH_FORCEINLINE float pi_over_two() { return 1.57079632679489661923f; }                     // pi / 2
    static MATH_FORCEINLINE float pi_over_three() { return 1.0471975511965977461542144610932f; }        // pi / 3
    static MATH_FORCEINLINE float pi_over_four() { return 0.785398163397448309615660845819875721f; }    // pi / 4
    static MATH_FORCEINLINE float pi_over_six() { return 0.52359877559829887307710723054658f; }         // pi / 6
    static MATH_FORCEINLINE float one_over_pi() { return 0.31830988618379067153776752674503f; }         // 1 / pi
    static MATH_FORCEINLINE float one_over_two_pi() { return 0.159154943091895335768883763372514362f; } // 1 / 2*pi

    static MATH_FORCEINLINE float half_sqrt2() { return 0.70710678118654752440084436210485f; } // sqrt(2)/2

    static MATH_FORCEINLINE float infinity() { return std::numeric_limits<float>::infinity(); }

    static MATH_FORCEINLINE float3 xAxis() { return float3(1.0f, 0.0f, 0.0f); }
    static MATH_FORCEINLINE float3 yAxis() { return float3(0.0f, 1.0f, 0.0f); }
    static MATH_FORCEINLINE float3 zAxis() { return float3(0.0f, 0.0f, 1.0f); }

    static MATH_FORCEINLINE float4x4 identity4x4f() { return float4x4(float4(1.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 1.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 1.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 1.0f)); }
    static MATH_FORCEINLINE float3x3 identity3x3f() { return float3x3(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f)); }
}
