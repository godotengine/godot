// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "curve_intersector_precalculations.h"

namespace embree
{
  namespace isa
  {
    template<typename NativeCurve3fa, int M>
    struct DistanceCurveHit
    {
      __forceinline DistanceCurveHit() {}

      __forceinline DistanceCurveHit(const vbool<M>& valid, const vfloat<M>& U, const vfloat<M>& V, const vfloat<M>& T, const int i, const int N,
                                     const NativeCurve3fa& curve3D)
        : U(U), V(V), T(T), i(i), N(N), curve3D(curve3D), valid(valid) {}
      
      __forceinline void finalize() 
      {
        vu = (vfloat<M>(step)+U+vfloat<M>(float(i)))*(1.0f/float(N));
        vv = V;
        vt = T;
      }
      
      __forceinline Vec2f uv (const size_t i) const { return Vec2f(vu[i],vv[i]); }
      __forceinline float t  (const size_t i) const { return vt[i]; }
      __forceinline Vec3fa Ng(const size_t i) const { 
        return curve3D.eval_du(vu[i]);
      }
      
    public:
      vfloat<M> U;
      vfloat<M> V;
      vfloat<M> T;
      int i, N;
      NativeCurve3fa curve3D;
      
    public:
      vbool<M> valid;
      vfloat<M> vu;
      vfloat<M> vv;
      vfloat<M> vt;
    };

    template<typename NativeCurve3fa>
    struct DistanceCurveHit<NativeCurve3fa,1>
    {
      enum { M = 1 };
      
      __forceinline DistanceCurveHit() {}

      __forceinline DistanceCurveHit(const vbool<M>& valid, const vfloat<M>& U, const vfloat<M>& V, const vfloat<M>& T, const int i, const int N,
                                     const NativeCurve3fa& curve3D)
        : U(U), V(V), T(T), i(i), N(N), curve3D(curve3D), valid(valid) {}
      
      __forceinline void finalize() 
      {
        vu = (vfloat<M>(step)+U+vfloat<M>(float(i)))*(1.0f/float(N));
        vv = V;
        vt = T;
      }
      
      __forceinline Vec2f uv () const { return Vec2f(vu,vv); }
      __forceinline float t  () const { return vt; }
      __forceinline Vec3fa Ng() const { return curve3D.eval_du(vu); }
      
    public:
      vfloat<M> U;
      vfloat<M> V;
      vfloat<M> T;
      int i, N;
      NativeCurve3fa curve3D;
      
    public:
      vbool<M> valid;
      vfloat<M> vu;
      vfloat<M> vv;
      vfloat<M> vt;
    };
    
    template<typename NativeCurve3fa, int W = VSIZEX>
    struct DistanceCurve1Intersector1
    {
      using vboolx = vbool<W>;
      using vintx = vint<W>;
      using vfloatx = vfloat<W>;
      using Vec4vfx = Vec4vf<W>;
      
      template<typename Epilog>
      __forceinline bool intersect(const CurvePrecalculations1& pre, Ray& ray,
                                   RayQueryContext* context,
                                   const CurveGeometry* geom, const unsigned int primID,
                                   const Vec3ff& v0, const Vec3ff& v1, const Vec3ff& v2, const Vec3ff& v3,
                                   const Epilog& epilog)
      {
        const int N = geom->tessellationRate;
        
        /* transform control points into ray space */
        const NativeCurve3fa curve3Di(v0,v1,v2,v3);
        const NativeCurve3fa curve3D = enlargeRadiusToMinWidth(context,geom,ray.org,curve3Di);
        const NativeCurve3fa curve2D = curve3D.xfm_pr(pre.ray_space,ray.org);
      
        /* evaluate the bezier curve */
        vboolx valid = vfloatx(step) < vfloatx(float(N));
        const Vec4vfx p0 = curve2D.template eval0<W>(0,N);
        const Vec4vfx p1 = curve2D.template eval1<W>(0,N);

        /* approximative intersection with cone */
        const Vec4vfx v = p1-p0;
        const Vec4vfx w = -p0;
        const vfloatx d0 = madd(w.x,v.x,w.y*v.y);
        const vfloatx d1 = madd(v.x,v.x,v.y*v.y);
        const vfloatx u = clamp(d0*rcp(d1),vfloatx(zero),vfloatx(one));
        const Vec4vfx p = madd(u,v,p0);
        const vfloatx t = p.z*pre.depth_scale;
        const vfloatx d2 = madd(p.x,p.x,p.y*p.y); 
        const vfloatx r = p.w;
        const vfloatx r2 = r*r;
        valid &= (d2 <= r2) & (vfloatx(ray.tnear()) <= t) & (t <= vfloatx(ray.tfar));
        if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f) 
          valid &= t > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*pre.depth_scale; // ignore self intersections

        /* update hit information */
        bool ishit = false;
        if (unlikely(any(valid))) {
          DistanceCurveHit<NativeCurve3fa,W> hit(valid,u,0.0f,t,0,N,curve3D);
          ishit = ishit | epilog(valid,hit);
        }

        if (unlikely(W < N)) 
        {
          /* process SIMD-size many segments per iteration */
          for (int i=W; i<N; i+=W)
          {
            /* evaluate the bezier curve */
            vboolx valid = vintx(i)+vintx(step) < vintx(N);
            const Vec4vfx p0 = curve2D.template eval0<W>(i,N);
            const Vec4vfx p1 = curve2D.template eval1<W>(i,N);
            
            /* approximative intersection with cone */
            const Vec4vfx v = p1-p0;
            const Vec4vfx w = -p0;
            const vfloatx d0 = madd(w.x,v.x,w.y*v.y);
            const vfloatx d1 = madd(v.x,v.x,v.y*v.y);
            const vfloatx u = clamp(d0*rcp(d1),vfloatx(zero),vfloatx(one));
            const Vec4vfx p = madd(u,v,p0);
            const vfloatx t = p.z*pre.depth_scale;
            const vfloatx d2 = madd(p.x,p.x,p.y*p.y); 
            const vfloatx r = p.w;
            const vfloatx r2 = r*r;
            valid &= (d2 <= r2) & (vfloatx(ray.tnear()) <= t) & (t <= vfloatx(ray.tfar));
            if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f)
              valid &= t > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*pre.depth_scale; // ignore self intersections

             /* update hit information */
            if (unlikely(any(valid))) {
              DistanceCurveHit<NativeCurve3fa,W> hit(valid,u,0.0f,t,i,N,curve3D);
              ishit = ishit | epilog(valid,hit);
            }
          }
        }
        return ishit;
      }
    };
  }
}
