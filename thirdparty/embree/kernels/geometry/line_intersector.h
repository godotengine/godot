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
      struct LineIntersectorHitM
      {
        __forceinline LineIntersectorHitM() {}

        __forceinline LineIntersectorHitM(const vfloat<M>& u, const vfloat<M>& v, const vfloat<M>& t, const Vec3vf<M>& Ng)
          : vu(u), vv(v), vt(t), vNg(Ng) {}
        
        __forceinline void finalize() {}
        
        __forceinline Vec2f uv (const size_t i) const { return Vec2f(vu[i],vv[i]); }
        __forceinline float t  (const size_t i) const { return vt[i]; }
        __forceinline Vec3fa Ng(const size_t i) const { return Vec3fa(vNg.x[i],vNg.y[i],vNg.z[i]); }
        
      public:
        vfloat<M> vu;
        vfloat<M> vv;
        vfloat<M> vt;
        Vec3vf<M> vNg;
      };
    
    template<int M>
      struct FlatLinearCurveIntersector1
      {
        typedef CurvePrecalculations1 Precalculations;
        
        template<typename Epilog>
        static __forceinline bool intersect(const vbool<M>& valid_i,
                                            Ray& ray, const Precalculations& pre,
                                            const Vec4vf<M>& v0, const Vec4vf<M>& v1,
                                            const Epilog& epilog)
        {
          /* transform end points into ray space */
          vbool<M> valid = valid_i;
          vfloat<M> depth_scale = pre.depth_scale;
          LinearSpace3<Vec3vf<M>> ray_space = pre.ray_space;
          Vec4vf<M> p0(xfmVector(ray_space,v0.xyz()-Vec3vf<M>(ray.org)), v0.w);
          Vec4vf<M> p1(xfmVector(ray_space,v1.xyz()-Vec3vf<M>(ray.org)), v1.w);
          
          /* approximative intersection with cone */
          const Vec4vf<M> v = p1-p0;
          const Vec4vf<M> w = -p0;
          const vfloat<M> d0 = madd(w.x,v.x,w.y*v.y);
          const vfloat<M> d1 = madd(v.x,v.x,v.y*v.y);
          const vfloat<M> u = clamp(d0*rcp(d1),vfloat<M>(zero),vfloat<M>(one));
          const Vec4vf<M> p = madd(u,v,p0);
          const vfloat<M> t = p.z;
          const vfloat<M> d2 = madd(p.x,p.x,p.y*p.y);
          const vfloat<M> r = p.w;
          const vfloat<M> r2 = r*r;
          valid &= (d2 <= r2) & (vfloat<M>(ray.tnear()) < t) & (t <= vfloat<M>(ray.tfar));
          if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f) 
            valid &= t > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*depth_scale; // ignore self intersections
          if (unlikely(none(valid))) return false;
          
          /* ignore denormalized segments */
          const Vec3vf<M> T = v1.xyz()-v0.xyz();
          valid &= (T.x != vfloat<M>(zero)) | (T.y != vfloat<M>(zero)) | (T.z != vfloat<M>(zero));
          if (unlikely(none(valid))) return false;
          
          /* update hit information */
          LineIntersectorHitM<M> hit(u,zero,t,T);
          return epilog(valid,hit);
        }
      };
    
    template<int M, int K>
      struct FlatLinearCurveIntersectorK
      {
        typedef CurvePrecalculationsK<K> Precalculations;
        
        template<typename Epilog>
        static __forceinline bool intersect(const vbool<M>& valid_i,
                                            RayK<K>& ray, size_t k, const Precalculations& pre,
                                            const Vec4vf<M>& v0, const Vec4vf<M>& v1,
                                            const Epilog& epilog)
        {
          /* transform end points into ray space */
          vbool<M> valid = valid_i;
          vfloat<M> depth_scale = pre.depth_scale[k];
          LinearSpace3<Vec3vf<M>> ray_space = pre.ray_space[k];
          const Vec3vf<M> ray_org(ray.org.x[k],ray.org.y[k],ray.org.z[k]);
          const Vec3vf<M> ray_dir(ray.dir.x[k],ray.dir.y[k],ray.dir.z[k]);
          Vec4vf<M> p0(xfmVector(ray_space,v0.xyz()-ray_org), v0.w);
          Vec4vf<M> p1(xfmVector(ray_space,v1.xyz()-ray_org), v1.w);
          
          /* approximative intersection with cone */
          const Vec4vf<M> v = p1-p0;
          const Vec4vf<M> w = -p0;
          const vfloat<M> d0 = madd(w.x,v.x,w.y*v.y);
          const vfloat<M> d1 = madd(v.x,v.x,v.y*v.y);
          const vfloat<M> u = clamp(d0*rcp(d1),vfloat<M>(zero),vfloat<M>(one));
          const Vec4vf<M> p = madd(u,v,p0);
          const vfloat<M> t = p.z;
          const vfloat<M> d2 = madd(p.x,p.x,p.y*p.y);
          const vfloat<M> r = p.w;
          const vfloat<M> r2 = r*r;
          valid &= (d2 <= r2) & (vfloat<M>(ray.tnear()[k]) < t) & (t <= vfloat<M>(ray.tfar[k]));
          if (EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR != 0.0f) 
            valid &= t > float(EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR)*r*depth_scale; // ignore self intersections
          if (unlikely(none(valid))) return false;
          
          /* ignore denormalized segments */
          const Vec3vf<M> T = v1.xyz()-v0.xyz();
          valid &= (T.x != vfloat<M>(zero)) | (T.y != vfloat<M>(zero)) | (T.z != vfloat<M>(zero));
          if (unlikely(none(valid))) return false;
          
          /* update hit information */
          LineIntersectorHitM<M> hit(u,zero,t,T);
          return epilog(valid,hit);
        }
      };
  }
}
