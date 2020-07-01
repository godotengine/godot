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
#include "curve_intersector_precalculations.h"

namespace embree
{
  namespace isa
  {
    template<int M>
    struct SphereIntersectorHitM
    {
      __forceinline SphereIntersectorHitM() {}

      __forceinline SphereIntersectorHitM(const vfloat<M>& u,
                                          const vfloat<M>& v,
                                          const vfloat<M>& t,
                                          const Vec3vf<M>& Ng)
          : vu(u), vv(v), vt(t), vNg(Ng)
      {
      }

      __forceinline void finalize() {}

      __forceinline Vec2f uv(const size_t i) const
      {
        return Vec2f(vu[i], vv[i]);
      }
      __forceinline float t(const size_t i) const
      {
        return vt[i];
      }
      __forceinline Vec3fa Ng(const size_t i) const
      {
        return Vec3fa(vNg.x[i], vNg.y[i], vNg.z[i]);
      }

     public:
      vfloat<M> vu;
      vfloat<M> vv;
      vfloat<M> vt;
      Vec3vf<M> vNg;
    };

    template<int M>
    struct SphereIntersector1
    {
      typedef CurvePrecalculations1 Precalculations;

      template<typename Epilog>
      static __forceinline bool intersect(
          const vbool<M>& valid_i, Ray& ray, const Precalculations& pre, const Vec4vf<M>& v0, const Epilog& epilog)
      {
        vbool<M> valid = valid_i;

        const vfloat<M> rd2    = rcp(dot(ray.dir, ray.dir));
        const Vec3vf<M> ray_org(ray.org.x, ray.org.y, ray.org.z);
        const Vec3vf<M> ray_dir(ray.dir.x, ray.dir.y, ray.dir.z);
        const Vec3vf<M> center = v0.xyz();
        const vfloat<M> radius = v0.w;

        const Vec3vf<M> c0     = center - ray_org;
        const vfloat<M> projC0 = dot(c0, ray_dir) * rd2;
        const Vec3vf<M> perp   = c0 - projC0 * ray_dir;
        const vfloat<M> l2     = dot(perp, perp);
        const vfloat<M> r2     = radius * radius;
        valid &= (l2 <= r2);
        if (unlikely(none(valid)))
          return false;

        vfloat<M> td          = sqrt((r2 - l2) * rd2);
        const vfloat<M> t_in  = projC0 - td;
        const vfloat<M> t_out = projC0 + td;

        const vbool<M> valid_in  = valid & (t_in > ray.tnear()) & (t_in < ray.tfar);
        const vbool<M> valid_out = valid & !valid_in & (t_out > ray.tnear()) & (t_out < ray.tfar);

        td          = select(valid_in, -1.0f * td, td);
        vfloat<M> t = 0.f;
        t           = select(valid_in, t_in, t);
        t           = select(valid_out, t_out, t);
        valid &= valid_in | valid_out;
        if (unlikely(none(valid)))
          return false;

        const Vec3vf<M> Ng = td * ray_dir - perp;

        SphereIntersectorHitM<M> hit(zero, zero, t, Ng);
        return epilog(valid, hit);
      }
    };

    template<int M, int K>
    struct SphereIntersectorK
    {
      typedef CurvePrecalculationsK<K> Precalculations;

      template<typename Epilog>
      static __forceinline bool intersect(const vbool<M>& valid_i,
                                          RayK<K>& ray,
                                          size_t k,
                                          const Precalculations& pre,
                                          const Vec4vf<M>& v0,
                                          const Epilog& epilog)
      {
        vbool<M> valid = valid_i;

        const Vec3vf<M> ray_org(ray.org.x[k], ray.org.y[k], ray.org.z[k]);
        const Vec3vf<M> ray_dir(ray.dir.x[k], ray.dir.y[k], ray.dir.z[k]);
        const vfloat<M> rd2 = rcp(dot(ray_dir, ray_dir));
        const Vec3vf<M> center = v0.xyz();
        const vfloat<M> radius = v0.w;

        const Vec3vf<M> c0     = center - ray_org;
        const vfloat<M> projC0 = dot(c0, ray_dir) * rd2;
        const Vec3vf<M> perp   = c0 - projC0 * ray_dir;
        const vfloat<M> l2     = dot(perp, perp);
        const vfloat<M> r2     = radius * radius;
        valid &= (l2 <= r2);
        if (unlikely(none(valid)))
          return false;

        vfloat<M> td          = sqrt((r2 - l2) * rd2);
        const vfloat<M> t_in  = projC0 - td;
        const vfloat<M> t_out = projC0 + td;

        const vbool<M> valid_in  = valid & (t_in > ray.tnear()[k]) & (t_in < ray.tfar[k]);
        const vbool<M> valid_out = valid & !valid_in & (t_out > ray.tnear()[k]) & (t_out < ray.tfar[k]);

        td          = select(valid_in, -1.0f * td, td);
        vfloat<M> t = 0.f;
        t           = select(valid_in, t_in, t);
        t           = select(valid_out, t_out, t);
        valid &= valid_in | valid_out;
        if (unlikely(none(valid)))
          return false;

        const Vec3vf<M> Ng = td * ray_dir - perp;

        SphereIntersectorHitM<M> hit(zero, zero, t, Ng);
        return epilog(valid, hit);
      }
    };
  }  // namespace isa
}  // namespace embree
