// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sse.h"

namespace embree 
{
  const __m128 mm_lookupmask_ps[16] = {
    _mm_castsi128_ps(_mm_set_epi32( 0, 0, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32( 0, 0, 0,-1)),
    _mm_castsi128_ps(_mm_set_epi32( 0, 0,-1, 0)),
    _mm_castsi128_ps(_mm_set_epi32( 0, 0,-1,-1)),
    _mm_castsi128_ps(_mm_set_epi32( 0,-1, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32( 0,-1, 0,-1)),
    _mm_castsi128_ps(_mm_set_epi32( 0,-1,-1, 0)),
    _mm_castsi128_ps(_mm_set_epi32( 0,-1,-1,-1)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0, 0,-1)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0,-1, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0,-1,-1)),
    _mm_castsi128_ps(_mm_set_epi32(-1,-1, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1,-1, 0,-1)),
    _mm_castsi128_ps(_mm_set_epi32(-1,-1,-1, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1,-1,-1,-1))
  };

  const __m128d mm_lookupmask_pd[4] = {
    _mm_castsi128_pd(_mm_set_epi32( 0, 0, 0, 0)),
    _mm_castsi128_pd(_mm_set_epi32( 0, 0,-1,-1)),
    _mm_castsi128_pd(_mm_set_epi32(-1,-1, 0, 0)),
    _mm_castsi128_pd(_mm_set_epi32(-1,-1,-1,-1))
  };

}
