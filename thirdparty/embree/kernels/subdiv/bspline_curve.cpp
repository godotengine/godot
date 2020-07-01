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

#include "bspline_curve.h"

namespace embree
{
  PrecomputedBSplineBasis::PrecomputedBSplineBasis(int dj)
  {
    for (size_t i=1; i<=N; i++) 
    {
      for (size_t j=0; j<=N; j++) 
      {
        const float u = float(j+dj)/float(i);
        const Vec4f f = BSplineBasis::eval(u);
        c0[i][j] = f.x;
        c1[i][j] = f.y;
        c2[i][j] = f.z;
        c3[i][j] = f.w;
        const Vec4f d = BSplineBasis::derivative(u);
        d0[i][j] = d.x;
        d1[i][j] = d.y;
        d2[i][j] = d.z;
        d3[i][j] = d.w;
      }
    }
  }
  PrecomputedBSplineBasis bspline_basis0(0);
  PrecomputedBSplineBasis bspline_basis1(1);
}
