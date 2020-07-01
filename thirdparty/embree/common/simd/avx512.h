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

#include "vboolf16_avx512.h"
#include "vint16_avx512.h"
#include "vuint16_avx512.h"
#include "vfloat16_avx512.h"

#include "vboold8_avx512.h"
#include "vllong8_avx512.h"
#include "vdouble8_avx512.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// Prefetching
  ////////////////////////////////////////////////////////////////////////////////

#define PFHINT_L1   0
#define PFHINT_L2   1
#define PFHINT_NT   2

  template<const unsigned int mode>
    __forceinline void prefetch(const void * __restrict__ const m)
  {
    if (mode == PFHINT_L1)
      _mm_prefetch((const char*)m,_MM_HINT_T0); 
    else if (mode == PFHINT_L2) 
      _mm_prefetch((const char*)m,_MM_HINT_T1); 
    else if (mode == PFHINT_NT) 
      _mm_prefetch((const char*)m,_MM_HINT_NTA); 
  }
}
