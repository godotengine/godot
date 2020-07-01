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

#include "../common/default.h"
#include "bezier_curve.h"

namespace embree
{
  template<typename Vertex>
    struct HermiteCurveT : BezierCurveT<Vertex>
    {
      __forceinline HermiteCurveT() {}

      __forceinline HermiteCurveT(const BezierCurveT<Vertex>& curve)
        : BezierCurveT<Vertex>(curve) {}
      
      __forceinline HermiteCurveT(const Vertex& v0, const Vertex& t0, const Vertex& v1, const Vertex& t1)
        : BezierCurveT<Vertex>(v0,madd(1.0f/3.0f,t0,v0),nmadd(1.0f/3.0f,t1,v1),v1) {}

      __forceinline HermiteCurveT<Vec3fa> xfm_pr(const LinearSpace3fa& space, const Vec3fa& p) const
      {
        Vec3fa q0 = xfmVector(space,this->v0-p); q0.w = this->v0.w;
        Vec3fa q1 = xfmVector(space,this->v1-p); q1.w = this->v1.w;
        Vec3fa q2 = xfmVector(space,this->v2-p); q2.w = this->v2.w;
        Vec3fa q3 = xfmVector(space,this->v3-p); q3.w = this->v3.w;
        return BezierCurveT<Vec3fa>(q0,q1,q2,q3);
      }
    };
  
  typedef HermiteCurveT<Vec3fa> HermiteCurve3fa;
}

