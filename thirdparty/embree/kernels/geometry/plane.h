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

#include "../common/ray.h"

namespace embree
{
  namespace isa
  {
    struct HalfPlane
    {
      const Vec3fa P;  //!< plane origin
      const Vec3fa N;  //!< plane normal

      __forceinline HalfPlane(const Vec3fa& P, const Vec3fa& N) 
        : P(P), N(N) {}
      
      __forceinline BBox1f intersect(const Vec3fa& ray_org, const Vec3fa& ray_dir) const
      {
        Vec3fa O = Vec3fa(ray_org) - P;
        Vec3fa D = Vec3fa(ray_dir);
        float ON = dot(O,N);
        float DN = dot(D,N);
        bool eps = abs(DN) < min_rcp_input;
        float t = -ON*rcp(DN);
        float lower = select(eps || DN < 0.0f, float(neg_inf), t);
        float upper = select(eps || DN > 0.0f, float(pos_inf), t);
        return BBox1f(lower,upper);
      }
    };

    template<int M>
      struct HalfPlaneN
      {
        const Vec3vf<M> P;  //!< plane origin
        const Vec3vf<M> N;  //!< plane normal

        __forceinline HalfPlaneN(const Vec3vf<M>& P, const Vec3vf<M>& N)
          : P(P), N(N) {}

        __forceinline BBox<vfloat<M>> intersect(const Vec3fa& ray_org, const Vec3fa& ray_dir) const
        {
          Vec3vf<M> O = Vec3vf<M>(ray_org) - P;
          Vec3vf<M> D = Vec3vf<M>(ray_dir);
          vfloat<M> ON = dot(O,N);
          vfloat<M> DN = dot(D,N);
          vboolx eps = abs(DN) < min_rcp_input;
          vfloat<M> t = -ON*rcp(DN);
          vfloat<M> lower = select(eps | DN < 0.0f, vfloat<M>(neg_inf), t);
          vfloat<M> upper = select(eps | DN > 0.0f, vfloat<M>(pos_inf), t);
          return BBox<vfloat<M>>(lower,upper);
        }
      };
  }
}
