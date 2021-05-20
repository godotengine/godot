// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "../common/scene_points.h"
#include "curve_intersector_precalculations.h"

namespace embree
{
  namespace isa
  {
    template<int M>
    struct SphereIntersectorHitM
    {
      __forceinline SphereIntersectorHitM() {}

      __forceinline SphereIntersectorHitM(const vfloat<M>& t, const Vec3vf<M>& Ng)
        : vt(t), vNg(Ng) {}

      __forceinline void finalize() {}

      __forceinline Vec2f uv(const size_t i) const {
        return Vec2f(0.0f, 0.0f);
      }
      __forceinline float t(const size_t i) const {
        return vt[i];
      }
      __forceinline Vec3fa Ng(const size_t i) const {
        return Vec3fa(vNg.x[i], vNg.y[i], vNg.z[i]);
      }

     public:
      vfloat<M> vt;
      Vec3vf<M> vNg;
    };

    template<int M>
    struct SphereIntersector1
    {
      typedef CurvePrecalculations1 Precalculations;

      template<typename Epilog>
      static __forceinline bool intersect(
          const vbool<M>& valid_i, Ray& ray,
          const Precalculations& pre, const Vec4vf<M>& v0, const Epilog& epilog)
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

        const vfloat<M> td      = sqrt((r2 - l2) * rd2);
        const vfloat<M> t_front = projC0 - td;
        const vfloat<M> t_back  = projC0 + td;

        const vbool<M> valid_front = valid & (ray.tnear() <= t_front) & (t_front <= ray.tfar);
        const vbool<M> valid_back  = valid & (ray.tnear() <= t_back ) & (t_back  <= ray.tfar);

        /* check if there is a first hit */
        const vbool<M> valid_first = valid_front | valid_back;
        if (unlikely(none(valid_first)))
          return false;

        /* construct first hit */
        const vfloat<M> td_front = -td;
        const vfloat<M> td_back  = +td;
        const vfloat<M> t_first  = select(valid_front, t_front, t_back);
        const Vec3vf<M> Ng_first = select(valid_front, td_front, td_back) * ray_dir - perp;
        SphereIntersectorHitM<M> hit(t_first, Ng_first);

        /* invoke intersection filter for first hit */
        const bool is_hit_first = epilog(valid_first, hit);
                
        /* check for possible second hits before potentially accepted hit */
        const vfloat<M> t_second = t_back;
        const vbool<M> valid_second = valid_front & valid_back & (t_second <= ray.tfar);
        if (unlikely(none(valid_second)))
          return is_hit_first;

        /* invoke intersection filter for second hit */
        const Vec3vf<M> Ng_second = td_back * ray_dir - perp;
        hit = SphereIntersectorHitM<M> (t_second, Ng_second);
        const bool is_hit_second = epilog(valid_second, hit);
        
        return is_hit_first | is_hit_second;
      }

      template<typename Epilog>
      static __forceinline bool intersect(
        const vbool<M>& valid_i, Ray& ray, IntersectContext* context, const Points* geom,
        const Precalculations& pre, const Vec4vf<M>& v0i, const Epilog& epilog)
      {
        const Vec3vf<M> ray_org(ray.org.x, ray.org.y, ray.org.z);
        const Vec4vf<M> v0 = enlargeRadiusToMinWidth<M>(context,geom,ray_org,v0i);
        return intersect(valid_i,ray,pre,v0,epilog);
      }
    };

    template<int M, int K>
    struct SphereIntersectorK
    {
      typedef CurvePrecalculationsK<K> Precalculations;

      template<typename Epilog>
      static __forceinline bool intersect(const vbool<M>& valid_i,
                                          RayK<K>& ray, size_t k,
                                          IntersectContext* context,
                                          const Points* geom,
                                          const Precalculations& pre,
                                          const Vec4vf<M>& v0i,
                                          const Epilog& epilog)
      {
        vbool<M> valid = valid_i;

        const Vec3vf<M> ray_org(ray.org.x[k], ray.org.y[k], ray.org.z[k]);
        const Vec3vf<M> ray_dir(ray.dir.x[k], ray.dir.y[k], ray.dir.z[k]);
        const vfloat<M> rd2 = rcp(dot(ray_dir, ray_dir));

        const Vec4vf<M> v0 = enlargeRadiusToMinWidth<M>(context,geom,ray_org,v0i);
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

        const vfloat<M> td      = sqrt((r2 - l2) * rd2);
        const vfloat<M> t_front = projC0 - td;
        const vfloat<M> t_back  = projC0 + td;

        const vbool<M> valid_front = valid & (ray.tnear()[k] <= t_front) & (t_front <= ray.tfar[k]);
        const vbool<M> valid_back  = valid & (ray.tnear()[k] <= t_back ) & (t_back  <= ray.tfar[k]);

        /* check if there is a first hit */
        const vbool<M> valid_first = valid_front | valid_back;
        if (unlikely(none(valid_first)))
          return false;

        /* construct first hit */
        const vfloat<M> td_front = -td;
        const vfloat<M> td_back  = +td;
        const vfloat<M> t_first  = select(valid_front, t_front, t_back);
        const Vec3vf<M> Ng_first = select(valid_front, td_front, td_back) * ray_dir - perp;
        SphereIntersectorHitM<M> hit(t_first, Ng_first);

        /* invoke intersection filter for first hit */
        const bool is_hit_first = epilog(valid_first, hit);
                
        /* check for possible second hits before potentially accepted hit */
        const vfloat<M> t_second = t_back;
        const vbool<M> valid_second = valid_front & valid_back & (t_second <= ray.tfar[k]);
        if (unlikely(none(valid_second)))
          return is_hit_first;

        /* invoke intersection filter for second hit */
        const Vec3vf<M> Ng_second = td_back * ray_dir - perp;
        hit = SphereIntersectorHitM<M> (t_second, Ng_second);
        const bool is_hit_second = epilog(valid_second, hit);
        
        return is_hit_first | is_hit_second;
      }
    };
  }  // namespace isa
}  // namespace embree
