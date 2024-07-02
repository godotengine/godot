// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bezier_curve.h"

namespace embree
{
  PrecomputedBezierBasis::PrecomputedBezierBasis(int dj)
  {
    for (size_t i=1; i<=N; i++) 
    {
      for (size_t j=0; j<=N; j++) 
      {
        const float u = float(j+dj)/float(i);
        const Vec4f f = BezierBasis::eval(u);
        c0[i][j] = f.x;
        c1[i][j] = f.y;
        c2[i][j] = f.z;
        c3[i][j] = f.w;
        const Vec4f d = BezierBasis::derivative(u);
        d0[i][j] = d.x;
        d1[i][j] = d.y;
        d2[i][j] = d.z;
        d3[i][j] = d.w;
      }
    }
  }
  PrecomputedBezierBasis bezier_basis0(0);
  PrecomputedBezierBasis bezier_basis1(1);
}
