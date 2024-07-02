// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
