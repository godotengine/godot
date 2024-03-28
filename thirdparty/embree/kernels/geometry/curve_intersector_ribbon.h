// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "quad_intersector.h"
#include "curve_intersector_precalculations.h"

#define Bezier1Intersector1 RibbonCurve1Intersector1
#define Bezier1IntersectorK RibbonCurve1IntersectorK

namespace embree
{
  namespace isa
  {
    template<typename NativeCurve3ff, int M>
    struct RibbonHit
    {
      __forceinline RibbonHit() {}

      __forceinline RibbonHit(const vbool<M>& valid, const vfloat<M>& U, const vfloat<M>& V, const vfloat<M>& T, const int i, const int N,
                              const NativeCurve3ff& curve3D)
        : U(U), V(V), T(T), i(i), N(N), curve3D(curve3D), valid(valid) {}
      
      __forceinline void finalize() 
      {
        vu = (vfloat<M>(step)+U+vfloat<M>(float(i)))*(1.0f/float(N));
        vv = V;
        vt = T;
      }
      
      __forceinline Vec2f uv (const size_t i) const { return Vec2f(vu[i],vv[i]); }
      __forceinline float t  (const size_t i) const { return vt[i]; }
      __forceinline Vec3fa Ng(const size_t i) const { return curve3D.eval_du(vu[i]); }

      __forceinline Vec2vf<M> uv() const { return Vec2vf<M>(vu,vv); }
      __forceinline vfloat<M> t () const { return vt; }
      __forceinline Vec3vf<M> Ng() const { return (Vec3vf<M>) curve3D.template veval_du<M>(vu); }
      
    public:
      vfloat<M> U;
      vfloat<M> V;
      vfloat<M> T;
      int i, N;
      NativeCurve3ff curve3D;
      
    public:
      vbool<M> valid;
      vfloat<M> vu;
      vfloat<M> vv;
      vfloat<M> vt;
    };

    /* calculate squared distance of point p0 to line p1->p2 */
    template<int M>
    __forceinline std::pair<vfloat<M>,vfloat<M>> sqr_point_line_distance(const Vec2vf<M>& p0, const Vec2vf<M>& p1, const Vec2vf<M>& p2)
    {
      const vfloat<M> num = det(p2-p1,p1-p0);
      const vfloat<M> den2 = dot(p2-p1,p2-p1);
      return std::make_pair(num*num,den2);
    }
    
    /* performs culling against a cylinder */
     template<int M>
     __forceinline vbool<M> cylinder_culling_test(const Vec2vf<M>& p0, const Vec2vf<M>& p1, const Vec2vf<M>& p2, const vfloat<M>& r)
    {
      const std::pair<vfloat<M>,vfloat<M>> d = sqr_point_line_distance<M>(p0,p1,p2);
      return d.first <= r*r*d.second;
    }

    template<int M = VSIZEX, typename NativeCurve3ff, typename Epilog>
    __forceinline bool intersect_ribbon(const Vec3fa& ray_org, const Vec3fa& ray_dir, const float ray_tnear, const float& ray_tfar,
                                        const LinearSpace3fa& ray_space, const float& depth_scale,
                                        const NativeCurve3ff& curve3D, const int N,
                                        const Epilog& epilog)
    {
      /* transform control points into ray space */
      const NativeCurve3ff curve2D = curve3D.xfm_pr(ray_space,ray_org);
      float eps = 4.0f*float(ulp)*reduce_max(max(abs(curve2D.v0),abs(curve2D.v1),abs(curve2D.v2),abs(curve2D.v3)));

      int i=0;
      bool ishit = false;
      
#if !defined(__SYCL_DEVICE_ONLY__)
      {
        /* evaluate the bezier curve */
        vbool<M> valid = vfloat<M>(step) < vfloat<M>(float(N));
        const Vec4vf<M> p0 = curve2D.template eval0<M>(0,N);
        const Vec4vf<M> p1 = curve2D.template eval1<M>(0,N);
        valid &= cylinder_culling_test<M>(zero,Vec2vf<M>(p0.x,p0.y),Vec2vf<M>(p1.x,p1.y),max(p0.w,p1.w));
        
        if (any(valid)) 
        {
          Vec3vf<M> dp0dt = curve2D.template derivative0<M>(0,N);
          Vec3vf<M> dp1dt = curve2D.template derivative1<M>(0,N);
          dp0dt = select(reduce_max(abs(dp0dt)) < vfloat<M>(eps),Vec3vf<M>(p1-p0),dp0dt);
          dp1dt = select(reduce_max(abs(dp1dt)) < vfloat<M>(eps),Vec3vf<M>(p1-p0),dp1dt);
          const Vec3vf<M> n0(dp0dt.y,-dp0dt.x,0.0f);
          const Vec3vf<M> n1(dp1dt.y,-dp1dt.x,0.0f);
          const Vec3vf<M> nn0 = normalize(n0);
          const Vec3vf<M> nn1 = normalize(n1);
          const Vec3vf<M> lp0 = madd(p0.w,nn0,Vec3vf<M>(p0));
          const Vec3vf<M> lp1 = madd(p1.w,nn1,Vec3vf<M>(p1));
          const Vec3vf<M> up0 = nmadd(p0.w,nn0,Vec3vf<M>(p0));
          const Vec3vf<M> up1 = nmadd(p1.w,nn1,Vec3vf<M>(p1));
          
          vfloat<M> vu,vv,vt;
          vbool<M> valid0 = intersect_quad_backface_culling<M>(valid,zero,Vec3fa(0,0,1),ray_tnear,ray_tfar,lp0,lp1,up1,up0,vu,vv,vt);
          
          if (any(valid0))
          {
            /* ignore self intersections */
            if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f) {
              vfloat<M> r = lerp(p0.w, p1.w, vu);
              valid0 &= vt > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*depth_scale;
            }
            
            if (any(valid0))
            {
              vv = madd(2.0f,vv,vfloat<M>(-1.0f));
              RibbonHit<NativeCurve3ff,M> bhit(valid0,vu,vv,vt,0,N,curve3D);
              ishit |= epilog(bhit.valid,bhit);
            }
          }
        }
        i += M;
      }
      
      if (unlikely(i < N))
#endif
      {
        /* process SIMD-size many segments per iteration */
        for (; i<N; i+=M)
        {
          /* evaluate the bezier curve */
          vbool<M> valid = vint<M>(i)+vint<M>(step) < vint<M>(N);
          const Vec4vf<M> p0 = curve2D.template eval0<M>(i,N);
          const Vec4vf<M> p1 = curve2D.template eval1<M>(i,N);
          valid &= cylinder_culling_test<M>(zero,Vec2vf<M>(p0.x,p0.y),Vec2vf<M>(p1.x,p1.y),max(p0.w,p1.w));
          if (none(valid)) continue;
          
          Vec3vf<M> dp0dt = curve2D.template derivative0<M>(i,N);
          Vec3vf<M> dp1dt = curve2D.template derivative1<M>(i,N);
          dp0dt = select(reduce_max(abs(dp0dt)) < vfloat<M>(eps),Vec3vf<M>(p1-p0),dp0dt);
          dp1dt = select(reduce_max(abs(dp1dt)) < vfloat<M>(eps),Vec3vf<M>(p1-p0),dp1dt);
          const Vec3vf<M> n0(dp0dt.y,-dp0dt.x,0.0f);
          const Vec3vf<M> n1(dp1dt.y,-dp1dt.x,0.0f);
          const Vec3vf<M> nn0 = normalize(n0);
          const Vec3vf<M> nn1 = normalize(n1);
          const Vec3vf<M> lp0 = madd(p0.w,nn0,Vec3vf<M>(p0));
          const Vec3vf<M> lp1 = madd(p1.w,nn1,Vec3vf<M>(p1));
          const Vec3vf<M> up0 = nmadd(p0.w,nn0,Vec3vf<M>(p0));
          const Vec3vf<M> up1 = nmadd(p1.w,nn1,Vec3vf<M>(p1));
          
          vfloat<M> vu,vv,vt;
          vbool<M> valid0 = intersect_quad_backface_culling<M>(valid,zero,Vec3fa(0,0,1),ray_tnear,ray_tfar,lp0,lp1,up1,up0,vu,vv,vt);

          if (any(valid0))
          {
            /* ignore self intersections */
            if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f) {
              vfloat<M> r = lerp(p0.w, p1.w, vu);
              valid0 &= vt > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*depth_scale;
            }
            
            if (any(valid0))
            {
              vv = madd(2.0f,vv,vfloat<M>(-1.0f));
              RibbonHit<NativeCurve3ff,M> bhit(valid0,vu,vv,vt,i,N,curve3D);
              ishit |= epilog(bhit.valid,bhit);
            }
          }
        }
      }
      return ishit;
    }
        
    template<template<typename Ty> class NativeCurve, int M = VSIZEX>
    struct RibbonCurve1Intersector1
    {
      typedef NativeCurve<Vec3ff> NativeCurve3ff;
      
      template<typename Ray, typename Epilog>
      __forceinline bool intersect(const CurvePrecalculations1& pre, Ray& ray,
                                   RayQueryContext* context,
                                   const CurveGeometry* geom, const unsigned int primID,
                                   const Vec3ff& v0, const Vec3ff& v1, const Vec3ff& v2, const Vec3ff& v3,
                                   const Epilog& epilog)
      {
        const int N = geom->tessellationRate;
        NativeCurve3ff curve(v0,v1,v2,v3);
        curve = enlargeRadiusToMinWidth(context,geom,ray.org,curve);
        return intersect_ribbon<M,NativeCurve3ff>(ray.org,ray.dir,ray.tnear(),ray.tfar,
                                                pre.ray_space,pre.depth_scale,
                                                curve,N,
                                                epilog);
      }
    };
    
    template<template<typename Ty> class NativeCurve, int K, int M = VSIZEX>
    struct RibbonCurve1IntersectorK
    {
      typedef NativeCurve<Vec3ff> NativeCurve3ff;
      
      template<typename Epilog>
      __forceinline bool intersect(const CurvePrecalculationsK<K>& pre, RayK<K>& ray, size_t k,
                                   RayQueryContext* context,
                                   const CurveGeometry* geom, const unsigned int primID,
                                   const Vec3ff& v0, const Vec3ff& v1, const Vec3ff& v2, const Vec3ff& v3,
                                   const Epilog& epilog)
      {
        const int N = geom->tessellationRate;
        const Vec3fa ray_org(ray.org.x[k],ray.org.y[k],ray.org.z[k]);
        const Vec3fa ray_dir(ray.dir.x[k],ray.dir.y[k],ray.dir.z[k]);
        NativeCurve3ff curve(v0,v1,v2,v3);
        curve = enlargeRadiusToMinWidth(context,geom,ray_org,curve);
        return intersect_ribbon<M,NativeCurve3ff>(ray_org,ray_dir,ray.tnear()[k],ray.tfar[k],
                                                pre.ray_space[k],pre.depth_scale[k],
                                                curve,N,
                                                epilog);
      }
    };
  }
}
