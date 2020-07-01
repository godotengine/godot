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

#include "catmullclark_coefficients.h"


namespace embree
{
  CatmullClarkPrecomputedCoefficients CatmullClarkPrecomputedCoefficients::table;

  CatmullClarkPrecomputedCoefficients::CatmullClarkPrecomputedCoefficients()
  {
    /* precompute cosf(2.0f*M_PI/n) */
    for (size_t n=0; n<=MAX_RING_FACE_VALENCE; n++)
      table_cos_2PI_div_n[n] = set_cos_2PI_div_n(n);

    /* precompute limit tangents coefficients */
    for (size_t n=0; n<=MAX_RING_FACE_VALENCE; n++)
    {
      table_limittangent_a[n] = new float[n];
      table_limittangent_b[n] = new float[n];

      for (size_t i=0; i<n; i++) {
        table_limittangent_a[n][i] = set_limittangent_a(i,n);
        table_limittangent_b[n][i] = set_limittangent_b(i,n);
      }      
    }

    for (size_t n=0; n<=MAX_RING_FACE_VALENCE; n++)
      table_limittangent_c[n] = set_limittangent_c(n);
  }

  CatmullClarkPrecomputedCoefficients::~CatmullClarkPrecomputedCoefficients()
  {
    for (size_t n=0; n<=MAX_RING_FACE_VALENCE; n++)
    {
      delete [] table_limittangent_a[n];
      delete [] table_limittangent_b[n];
    }
  }
}
