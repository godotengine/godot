#pragma once

#include "Runtime/Math/Simd/vec-matrix.h"

namespace math
{
// inverse (and unique pseudo-inverse) of 3x3 matrix, singular or not.
// robust, but way slower than adjInverse
    float3x3 svdInverse(const float3x3 &a);

    float4 svdRotation(const float3x3 &a);
}
