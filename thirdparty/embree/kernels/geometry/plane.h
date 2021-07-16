// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
          Vec3vf<M> O = Vec3vf<M>((Vec3fa)ray_org) - P;
          Vec3vf<M> D = Vec3vf<M>((Vec3fa)ray_dir);
          vfloat<M> ON = dot(O,N);
          vfloat<M> DN = dot(D,N);
          vbool<M> eps = abs(DN) < min_rcp_input;
          vfloat<M> t = -ON*rcp(DN);
          vfloat<M> lower = select(eps | DN < 0.0f, vfloat<M>(neg_inf), t);
          vfloat<M> upper = select(eps | DN > 0.0f, vfloat<M>(pos_inf), t);
          return BBox<vfloat<M>>(lower,upper);
        }
      };
  }
}
