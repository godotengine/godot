// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

#include "../sys/platform.h"
#include "../sys/intrinsics.h"
#include "../math/constants.h"
#include "varying.h"

namespace embree 
{
#if defined(__SSE4_1__)
  __forceinline __m128 blendv_ps(__m128 f, __m128 t, __m128 mask) { 
    return _mm_blendv_ps(f,t,mask);
  }
#else
  __forceinline __m128 blendv_ps(__m128 f, __m128 t, __m128 mask) { 
    return _mm_or_ps(_mm_and_ps(mask, t), _mm_andnot_ps(mask, f)); 
  }
#endif

  extern const __m128  mm_lookupmask_ps[16];
  extern const __m128d mm_lookupmask_pd[4];
}

#if defined(__AVX512VL__)
#include "vboolf4_avx512.h"
#else
#include "vboolf4_sse2.h"
#endif
#include "vint4_sse2.h"
#include "vuint4_sse2.h"
#include "vfloat4_sse2.h"
