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
#include "quad_intersector.h"
#include "curve_intersector_precalculations.h"

#define Bezier1Intersector1 RibbonCurve1Intersector1
#define Bezier1IntersectorK RibbonCurve1IntersectorK

namespace embree
{
  namespace isa
  {
    template<typename NativeCurve3fa, int M>
    struct RibbonHit
    {
      __forceinline RibbonHit() {}

      __forceinline RibbonHit(const vbool<M>& valid, const vfloat<M>& U, const vfloat<M>& V, const vfloat<M>& T, const int i, const int N,
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

    /* calculate squared distance of point p0 to line p1->p2 */
    __forceinline std::pair<vfloatx,vfloatx> sqr_point_line_distance(const Vec2vfx& p0, const Vec2vfx& p1, const Vec2vfx& p2)
    {
      const vfloatx num = det(p2-p1,p1-p0);
      const vfloatx den2 = dot(p2-p1,p2-p1);
      return std::make_pair(num*num,den2);
    }
    
    /* performs culling against a cylinder */
    __forceinline vboolx cylinder_culling_test(const Vec2vfx& p0, const Vec2vfx& p1, const Vec2vfx& p2, const vfloatx& r)
    {
      const std::pair<vfloatx,vfloatx> d = sqr_point_line_distance(p0,p1,p2);
      return d.first <= r*r*d.second;
    }

    template<typename NativeCurve3fa, typename Epilog>
    __forceinline bool intersect_ribbon(const Vec3fa& ray_org, const Vec3fa& ray_dir, const float ray_tnear, const float& ray_tfar,
                                        const LinearSpace3fa& ray_space, const float& depth_scale,
                                        const NativeCurve3fa& curve3D, const int N,
                                        const Epilog& epilog)
    {
      /* transform control points into ray space */
      const NativeCurve3fa curve2D = curve3D.xfm_pr(ray_space,ray_org);
      float eps = 4.0f*float(ulp)*reduce_max(max(abs(curve2D.v0),abs(curve2D.v1),abs(curve2D.v2),abs(curve2D.v3)));
      
      /* evaluate the bezier curve */
      bool ishit = false;
      vboolx valid = vfloatx(step) < vfloatx(float(N));
      const Vec4vfx p0 = curve2D.template eval0<VSIZEX>(0,N);
      const Vec4vfx p1 = curve2D.template eval1<VSIZEX>(0,N);
      valid &= cylinder_culling_test(zero,Vec2vfx(p0.x,p0.y),Vec2vfx(p1.x,p1.y),max(p0.w,p1.w));
      
      if (any(valid)) 
      {
        Vec3vfx dp0dt = curve2D.template derivative0<VSIZEX>(0,N);
        Vec3vfx dp1dt = curve2D.template derivative1<VSIZEX>(0,N);
        dp0dt = select(reduce_max(abs(dp0dt)) < vfloatx(eps),Vec3vfx(p1-p0),dp0dt);
        dp1dt = select(reduce_max(abs(dp1dt)) < vfloatx(eps),Vec3vfx(p1-p0),dp1dt);
        const Vec3vfx n0(dp0dt.y,-dp0dt.x,0.0f);
        const Vec3vfx n1(dp1dt.y,-dp1dt.x,0.0f);
        const Vec3vfx nn0 = normalize(n0);
        const Vec3vfx nn1 = normalize(n1);
        const Vec3vfx lp0 = madd(p0.w,nn0,Vec3vfx(p0));
        const Vec3vfx lp1 = madd(p1.w,nn1,Vec3vfx(p1));
        const Vec3vfx up0 = nmadd(p0.w,nn0,Vec3vfx(p0));
        const Vec3vfx up1 = nmadd(p1.w,nn1,Vec3vfx(p1));
        
        vfloatx vu,vv,vt;
        vboolx valid0 = intersect_quad_backface_culling(valid,zero,Vec3fa(0,0,1),ray_tnear,ray_tfar,lp0,lp1,up1,up0,vu,vv,vt);

        if (any(valid0))
        {
          /* ignore self intersections */
          if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f) {
            vfloatx r = lerp(p0.w, p1.w, vu);
            valid0 &= vt > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*depth_scale;
          }
          
          if (any(valid0))
          {
            vv = madd(2.0f,vv,vfloatx(-1.0f));
            RibbonHit<NativeCurve3fa,VSIZEX> bhit(valid0,vu,vv,vt,0,N,curve3D);
            ishit |= epilog(bhit.valid,bhit);
          }
        }
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
          valid &= cylinder_culling_test(zero,Vec2vfx(p0.x,p0.y),Vec2vfx(p1.x,p1.y),max(p0.w,p1.w));
          if (none(valid)) continue;
          
          Vec3vfx dp0dt = curve2D.template derivative0<VSIZEX>(i,N);
          Vec3vfx dp1dt = curve2D.template derivative1<VSIZEX>(i,N);
          dp0dt = select(reduce_max(abs(dp0dt)) < vfloatx(eps),Vec3vfx(p1-p0),dp0dt);
          dp1dt = select(reduce_max(abs(dp1dt)) < vfloatx(eps),Vec3vfx(p1-p0),dp1dt);
          const Vec3vfx n0(dp0dt.y,-dp0dt.x,0.0f);
          const Vec3vfx n1(dp1dt.y,-dp1dt.x,0.0f);
          const Vec3vfx nn0 = normalize(n0);
          const Vec3vfx nn1 = normalize(n1);
          const Vec3vfx lp0 = madd(p0.w,nn0,Vec3vfx(p0));
          const Vec3vfx lp1 = madd(p1.w,nn1,Vec3vfx(p1));
          const Vec3vfx up0 = nmadd(p0.w,nn0,Vec3vfx(p0));
          const Vec3vfx up1 = nmadd(p1.w,nn1,Vec3vfx(p1));
          
          vfloatx vu,vv,vt;
          vboolx valid0 = intersect_quad_backface_culling(valid,zero,Vec3fa(0,0,1),ray_tnear,ray_tfar,lp0,lp1,up1,up0,vu,vv,vt);

          if (any(valid0))
          {
            /* ignore self intersections */
            if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f) {
              vfloatx r = lerp(p0.w, p1.w, vu);
              valid0 &= vt > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*depth_scale;
            }
            
            if (any(valid0))
            {
              vv = madd(2.0f,vv,vfloatx(-1.0f));
              RibbonHit<NativeCurve3fa,VSIZEX> bhit(valid0,vu,vv,vt,i,N,curve3D);
              ishit |= epilog(bhit.valid,bhit);
            }
          }
        }
      }
      return ishit;
    }
        
    template<typename NativeCurve3fa>
    struct RibbonCurve1Intersector1
    {
      template<typename GeometryT, typename Epilog>
      __forceinline bool intersect(const CurvePrecalculations1& pre, Ray& ray,
                                   const GeometryT* geom, const unsigned int primID,
                                   const Vec3fa& v0, const Vec3fa& v1, const Vec3fa& v2, const Vec3fa& v3,
                                   const Epilog& epilog)
      {
        const int N = geom->tessellationRate;
        const NativeCurve3fa curve(v0,v1,v2,v3);
        return intersect_ribbon<NativeCurve3fa>(ray.org,ray.dir,ray.tnear(),ray.tfar,
                                                pre.ray_space,pre.depth_scale,
                                                curve,N,
                                                epilog);
      }
    };
    
    template<typename NativeCurve3fa, int K>
    struct RibbonCurve1IntersectorK
    {
      template<typename GeometryT, typename Epilog>
      __forceinline bool intersect(const CurvePrecalculationsK<K>& pre, RayK<K>& ray, size_t k,
                                   const GeometryT* geom, const unsigned int primID,
                                   const Vec3fa& v0, const Vec3fa& v1, const Vec3fa& v2, const Vec3fa& v3,
                                   const Epilog& epilog)
      {
        const int N = geom->tessellationRate;
        const NativeCurve3fa curve(v0,v1,v2,v3);
        const Vec3fa ray_org(ray.org.x[k],ray.org.y[k],ray.org.z[k]);
        const Vec3fa ray_dir(ray.dir.x[k],ray.dir.y[k],ray.dir.z[k]);
        return intersect_ribbon<NativeCurve3fa>(ray_org,ray_dir,ray.tnear()[k],ray.tfar[k],
                                                pre.ray_space[k],pre.depth_scale[k],
                                                curve,N,
                                                epilog);
      }
    };
  }
}
