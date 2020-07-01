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

#include "../math/math.h"

/* include SSE wrapper classes */
#if defined(__SSE__)
#  include "sse.h"
#endif

/* include AVX wrapper classes */
#if defined(__AVX__)
#  include "avx.h"
#endif

/* include AVX512 wrapper classes */
#if defined (__AVX512F__)
#  include "avx512.h"
#endif

namespace embree
{
  /* foreach unique */
  template<typename vbool, typename vint, typename Closure>
  __forceinline void foreach_unique(const vbool& valid0, const vint& vi, const Closure& closure)
  {
    vbool valid1 = valid0;
    while (any(valid1)) {
      const int j = int(bsf(movemask(valid1)));
      const int i = vi[j];
      const vbool valid2 = valid1 & (i == vi);
      valid1 = andn(valid1, valid2);
      closure(valid2, i);
    }
  }

  /* returns the next unique value i in vi and the corresponding valid_i mask */
  template<typename vbool, typename vint>
  __forceinline int next_unique(vbool& valid, const vint& vi, /*out*/ vbool& valid_i)
  {
    assert(any(valid));
    const int j = int(bsf(movemask(valid)));
    const int i = vi[j];
    valid_i = valid & (i == vi);
    valid = andn(valid, valid_i);
    return i;
  }

  /* foreach unique index */
  template<typename vbool, typename vint, typename Closure>
  __forceinline void foreach_unique_index(const vbool& valid0, const vint& vi, const Closure& closure)
  {
    vbool valid1 = valid0;
    while (any(valid1)) {
      const int j = int(bsf(movemask(valid1)));
      const int i = vi[j];
      const vbool valid2 = valid1 & (i == vi);
      valid1 = andn(valid1, valid2);
      closure(valid2, i, j);
    }
  }

  /* returns the index of the next unique value i in vi and the corresponding valid_i mask */
  template<typename vbool, typename vint>
  __forceinline int next_unique_index(vbool& valid, const vint& vi, /*out*/ vbool& valid_i)
  {
    assert(any(valid));
    const int j = int(bsf(movemask(valid)));
    const int i = vi[j];
    valid_i = valid & (i == vi);
    valid = andn(valid, valid_i);
    return j;
  }

  template<typename Closure>
  __forceinline void foreach2(int x0, int x1, int y0, int y1, const Closure& closure)
  {
    __aligned(64) int U[2*VSIZEX];
    __aligned(64) int V[2*VSIZEX];
    int index = 0;
    for (int y=y0; y<y1; y++) {
      const bool lasty = y+1>=y1;
      const vintx vy = y;
      for (int x=x0; x<x1; ) { //x+=VSIZEX) {
        const bool lastx = x+VSIZEX >= x1;
        vintx vx = x+vintx(step);
        vintx::storeu(&U[index], vx);
        vintx::storeu(&V[index], vy);
        const int dx = min(x1-x,VSIZEX);
        index += dx;
        x += dx;
        if (index >= VSIZEX || (lastx && lasty)) {
          const vboolx valid = vintx(step) < vintx(index);
          closure(valid, vintx::load(U), vintx::load(V));
          x-= max(0, index-VSIZEX);
          index = 0;
        }
      }
    }
  }
}
