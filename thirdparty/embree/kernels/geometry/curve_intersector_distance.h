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
    struct DistanceCurve1Intersector1
    {
      template<typename Epilog>
      __forceinline bool intersect(const CurvePrecalculations1& pre,Ray& ray,
                                   const CurveGeometry* geom, const unsigned int primID,
                                   const Vec3fa& v0, const Vec3fa& v1, const Vec3fa& v2, const Vec3fa& v3,
                                   const Epilog& epilog)
      {
        const int N = geom->tessellationRate;
        
        /* transform control points into ray space */
        const NativeCurve3fa curve3D(v0,v1,v2,v3);
        const NativeCurve3fa curve2D = curve3D.xfm_pr(pre.ray_space,ray.org);
      
        /* evaluate the bezier curve */
        vboolx valid = vfloatx(step) < vfloatx(float(N));
        const Vec4vfx p0 = curve2D.template eval0<VSIZEX>(0,N);
        const Vec4vfx p1 = curve2D.template eval1<VSIZEX>(0,N);

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
        valid &= (d2 <= r2) & (vfloatx(ray.tnear()) < t) & (t <= vfloatx(ray.tfar));
        if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f) 
          valid &= t > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*pre.depth_scale; // ignore self intersections

        /* update hit information */
        bool ishit = false;
        if (unlikely(any(valid))) {
          DistanceCurveHit<NativeCurve3fa,VSIZEX> hit(valid,u,0.0f,t,0,N,curve3D);
          ishit = ishit | epilog(valid,hit);
        }

        if (unlikely(VSIZEX < N)) 
        {
          /* process SIMD-size many segments per iteration */
          for (int i=VSIZEX; i<N; i+=VSIZEX)
          {
            /* evaluate the bezier curve */
            vboolx valid = vintx(i)+vintx(step) < vintx(N);
            const Vec4vfx p0 = curve2D.template eval0<VSIZEX>(i,N);
            const Vec4vfx p1 = curve2D.template eval1<VSIZEX>(i,N);
            
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
            valid &= (d2 <= r2) & (vfloatx(ray.tnear()) < t) & (t <= vfloatx(ray.tfar));
            if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f)
              valid &= t > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*pre.depth_scale; // ignore self intersections

             /* update hit information */
            if (unlikely(any(valid))) {
              DistanceCurveHit<NativeCurve3fa,VSIZEX> hit(valid,u,0.0f,t,i,N,curve3D);
              ishit = ishit | epilog(valid,hit);
            }
          }
        }
        return ishit;
      }
    };
  }
}
