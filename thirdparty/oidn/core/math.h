// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "common/platform.h"

namespace oidn {

  constexpr float minVectorLength    = 1e-10f;
  constexpr float minVectorLengthSqr = minVectorLength * minVectorLength;

  using std::log;
  using std::log2;
  using std::exp;
  using std::exp2;
  using std::pow;
  using std::isfinite;
  using std::isnan;

  __forceinline float sqr(float x)
  {
    return x * x;
  }

  __forceinline float rcp(float x)
  {
    __m128 r = _mm_rcp_ss(_mm_set_ss(x));
    return _mm_cvtss_f32(_mm_sub_ss(_mm_add_ss(r, r), _mm_mul_ss(_mm_mul_ss(r, r), _mm_set_ss(x))));
  }

  __forceinline float rsqrt(float x)
  {
    __m128 r = _mm_rsqrt_ss(_mm_set_ss(x));
    return _mm_cvtss_f32(_mm_add_ss(_mm_mul_ss(_mm_set_ss(1.5f), r),
             _mm_mul_ss(_mm_mul_ss(_mm_mul_ss(_mm_set_ss(x), _mm_set_ss(-0.5f)), r), _mm_mul_ss(r, r))));
  }

  __forceinline float maxSafe(float value, float minValue)
  {
    return isfinite(value) ? max(value, minValue) : minValue;
  }

  __forceinline float clampSafe(float value, float minValue, float maxValue)
  {
    return isfinite(value) ? clamp(value, minValue, maxValue) : minValue;
  }

  // Returns ceil(a / b) for non-negative integers
  template<class Int>
  __forceinline constexpr Int ceilDiv(Int a, Int b)
  {
    //assert(a >= 0);
    //assert(b > 0);
    return (a + b - 1) / b;
  }

  // Returns a rounded up to multiple of b
  template<class Int>
  __forceinline constexpr Int roundUp(Int a, Int b)
  {
    return ceilDiv(a, b) * b;
  }

} // namespace oidn
