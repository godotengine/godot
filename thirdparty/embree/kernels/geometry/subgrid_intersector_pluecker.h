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

#include "subgrid.h"
#include "quad_intersector_moeller.h"
#include "quad_intersector_pluecker.h"

namespace embree
{
  namespace isa
  {

    template<int M>
    struct SubGridQuadHitPlueckerM
    {
      __forceinline SubGridQuadHitPlueckerM() {}

      __forceinline SubGridQuadHitPlueckerM(const vbool<M>& valid,
                                            const vfloat<M>& U,
                                            const vfloat<M>& V,
                                            const vfloat<M>& W,
                                            const vfloat<M>& T,
                                            const vfloat<M>& absDen,
                                            const Vec3vf<M>& Ng,
                                            const vbool<M>& flags) : valid(valid)
      {
        const vfloat<M> rcpAbsDen = rcp(absDen);
        vt = T * rcpAbsDen;
        const vfloat<M> UVW = U+V+W;
        const vbool<M> invalid = abs(UVW) < min_rcp_input;
        const vfloat<M> rcpUVW = select(invalid,vfloat<M>(0.0f),rcp(UVW));
        const vfloat<M> u = U * rcpUVW;
        const vfloat<M> v = V * rcpUVW;
        const vfloat<M> u1 = vfloat<M>(1.0f) - u;
        const vfloat<M> v1 = vfloat<M>(1.0f) - v;
#if !defined(__AVX__) || defined(EMBREE_BACKFACE_CULLING)
        vu = select(flags,u1,u);
        vv = select(flags,v1,v);
        vNg = Vec3vf<M>(Ng.x,Ng.y,Ng.z);
#else
        const vfloat<M> flip = select(flags,vfloat<M>(-1.0f),vfloat<M>(1.0f));
        vv = select(flags,u1,v);
        vu = select(flags,v1,u);
        vNg = Vec3vf<M>(flip*Ng.x,flip*Ng.y,flip*Ng.z);
#endif
      }

      __forceinline void finalize()
      {
      }

      __forceinline Vec2f uv(const size_t i)
      {
        const float u = vu[i];
        const float v = vv[i];
        return Vec2f(u,v);
      }

      __forceinline float   t(const size_t i) { return vt[i]; }
      __forceinline Vec3fa Ng(const size_t i) { return Vec3fa(vNg.x[i],vNg.y[i],vNg.z[i]); }

    public:
      vbool<M> valid;
      vfloat<M> vu;
      vfloat<M> vv;
      vfloat<M> vt;
      Vec3vf<M> vNg;
    };

    template<int M>
      __forceinline void interpolateUV(SubGridQuadHitPlueckerM<M> &hit,const GridMesh::Grid &g, const SubGrid& subgrid, const vint<M> &stepX, const vint<M> &stepY) 
    {
      /* correct U,V interpolation across the entire grid */
      const vint<M> sx((int)subgrid.x());
      const vint<M> sy((int)subgrid.y());
      const vint<M> sxM(sx + stepX);
      const vint<M> syM(sy + stepY);
      const float inv_resX = rcp((float)((int)g.resX-1));
      const float inv_resY = rcp((float)((int)g.resY-1));          
      hit.vu = (hit.vu + vfloat<M>(sxM)) * inv_resX;
      hit.vv = (hit.vv + vfloat<M>(syM)) * inv_resY;
    }

    template<int M>
    __forceinline static bool intersectPluecker(Ray& ray,
                                                const Vec3vf<M>& tri_v0,
                                                const Vec3vf<M>& tri_v1,
                                                const Vec3vf<M>& tri_v2,
                                                const vbool<M>& flags,
                                                SubGridQuadHitPlueckerM<M> &hit)
    {
        /* calculate vertices relative to ray origin */
        const Vec3vf<M> O = Vec3vf<M>(ray.org);
        const Vec3vf<M> D = Vec3vf<M>(ray.dir);
        const Vec3vf<M> v0 = tri_v0-O;
        const Vec3vf<M> v1 = tri_v1-O;
        const Vec3vf<M> v2 = tri_v2-O;

        /* calculate triangle edges */
        const Vec3vf<M> e0 = v2-v0;
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v1-v2;

        /* perform edge tests */
        const vfloat<M> U = dot(cross(e0,v2+v0),D);
        const vfloat<M> V = dot(cross(e1,v0+v1),D);
        const vfloat<M> W = dot(cross(e2,v1+v2),D);
#if defined(EMBREE_BACKFACE_CULLING)
        const vfloat<M> maxUVW = max(U,V,W);
        vbool<M> valid = maxUVW <= 0.0f;
#else
        const vfloat<M> minUVW = min(U,V,W);
        const vfloat<M> maxUVW = max(U,V,W);
        vbool<M> valid = (maxUVW <= 0.0f) | (minUVW >= 0.0f);
#endif
        if (unlikely(none(valid))) return false;

        /* calculate geometry normal and denominator */
        const Vec3vf<M> Ng = stable_triangle_normal(e0,e1,e2);
        const vfloat<M> den = twice(dot(Ng,D));
        const vfloat<M> absDen = abs(den);
        const vfloat<M> sgnDen = signmsk(den);

        /* perform depth test */
        const vfloat<M> T = twice(dot(v0,Ng));
        valid &= absDen*vfloat<M>(ray.tnear()) < (T^sgnDen);
        valid &= (T^sgnDen) <= absDen*vfloat<M>(ray.tfar);
        if (unlikely(none(valid))) return false;

        /* avoid division by 0 */
        valid &= den != vfloat<M>(zero);
        if (unlikely(none(valid))) return false;

        /* update hit information */
        new (&hit) SubGridQuadHitPlueckerM<M>(valid,U,V,W,T,den,Ng,flags);
        return true;
      }

    template<int M, bool filter>
      struct SubGridQuadMIntersector1Pluecker;

    template<int M, bool filter>
      struct SubGridQuadMIntersector1Pluecker
      {
        __forceinline SubGridQuadMIntersector1Pluecker() {}

        __forceinline SubGridQuadMIntersector1Pluecker(const Ray& ray, const void* ptr) {}

        __forceinline void intersect(RayHit& ray, IntersectContext* context,
                                     const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3,
                                     const GridMesh::Grid &g, const SubGrid& subgrid) const
        {
          SubGridQuadHitPlueckerM<M> hit;
          Intersect1EpilogMU<M,filter> epilog(ray,context,subgrid.geomID(),subgrid.primID());

          /* intersect first triangle */
          if (intersectPluecker(ray,v0,v1,v3,vbool<M>(false),hit)) 
          {
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            epilog(hit.valid,hit);
          }

          /* intersect second triangle */
          if (intersectPluecker(ray,v2,v3,v1,vbool<M>(true),hit)) 
          {
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            epilog(hit.valid,hit);
          }
        }
      
        __forceinline bool occluded(Ray& ray, IntersectContext* context,
                                    const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3,
                                    const GridMesh::Grid &g, const SubGrid& subgrid) const
        {
          SubGridQuadHitPlueckerM<M> hit;
          Occluded1EpilogMU<M,filter> epilog(ray,context,subgrid.geomID(),subgrid.primID());
          
          /* intersect first triangle */
          if (intersectPluecker(ray,v0,v1,v3,vbool<M>(false),hit)) 
          {
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            if (epilog(hit.valid,hit))
              return true;
          }

          /* intersect second triangle */
          if (intersectPluecker(ray,v2,v3,v1,vbool<M>(true),hit)) 
          {
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            if (epilog(hit.valid,hit))
              return true;
          }

          return false;
        }
      };

#if defined (__AVX__)

    /*! Intersects 4 quads with 1 ray using AVX */
    template<bool filter>
      struct SubGridQuadMIntersector1Pluecker<4,filter>
    {
      __forceinline SubGridQuadMIntersector1Pluecker() {}

      __forceinline SubGridQuadMIntersector1Pluecker(const Ray& ray, const void* ptr) {}
      
      template<typename Epilog>
        __forceinline bool intersect(Ray& ray, const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, const GridMesh::Grid &g, const SubGrid& subgrid, const Epilog& epilog) const
      {
        const Vec3vf8 vtx0(vfloat8(v0.x,v2.x),vfloat8(v0.y,v2.y),vfloat8(v0.z,v2.z));
#if !defined(EMBREE_BACKFACE_CULLING)
        const Vec3vf8 vtx1(vfloat8(v1.x),vfloat8(v1.y),vfloat8(v1.z));
        const Vec3vf8 vtx2(vfloat8(v3.x),vfloat8(v3.y),vfloat8(v3.z));        
#else
        const Vec3vf8 vtx1(vfloat8(v1.x,v3.x),vfloat8(v1.y,v3.y),vfloat8(v1.z,v3.z));
        const Vec3vf8 vtx2(vfloat8(v3.x,v1.x),vfloat8(v3.y,v1.y),vfloat8(v3.z,v1.z));
#endif
        SubGridQuadHitPlueckerM<8> hit;
        const vbool8 flags(0,0,0,0,1,1,1,1);
        if (unlikely(intersectPluecker(ray,vtx0,vtx1,vtx2,flags,hit)))
        {
          /* correct U,V interpolation across the entire grid */
          interpolateUV<8>(hit,g,subgrid,vint<8>(0,1,1,0,0,1,1,0),vint<8>(0,0,1,1,0,0,1,1));            
          if (unlikely(epilog(hit.valid,hit)))
            return true;
        }
        return false;
      }
      
      __forceinline bool intersect(RayHit& ray, IntersectContext* context,
                                   const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, 
                                   const GridMesh::Grid &g, const SubGrid& subgrid) const
      {
          return intersect(ray,v0,v1,v2,v3,g,subgrid,Intersect1EpilogMU<8,filter>(ray,context,subgrid.geomID(),subgrid.primID()));
      }
      
      __forceinline bool occluded(Ray& ray, IntersectContext* context,
                                  const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, 
                                  const GridMesh::Grid &g, const SubGrid& subgrid) const
      {
          return intersect(ray,v0,v1,v2,v3,g,subgrid,Occluded1EpilogMU<8,filter>(ray,context,subgrid.geomID(),subgrid.primID()));
      }
    };

#endif


    /* ----------------------------- */
    /* -- ray packet intersectors -- */
    /* ----------------------------- */

    template<int K>
      struct SubGridQuadHitPlueckerK
      {
         __forceinline SubGridQuadHitPlueckerK(const vfloat<K>& U,
                                               const vfloat<K>& V,
                                               const vfloat<K>& W,
                                               const vfloat<K>& T,
                                               const vfloat<K>& absDen,
                                               const Vec3vf<K>& Ng,
                                               const vbool<K>& flags,
                                               const GridMesh::Grid &g, 
                                               const SubGrid& subgrid,
                                               const unsigned int i)
         : U(U), V(V), W(W), T(T), absDen(absDen), flags(flags), tri_Ng(Ng), g(g), subgrid(subgrid), i(i) {}

        __forceinline std::tuple<vfloat<K>,vfloat<K>,vfloat<K>,Vec3vf<K>> operator() () const
        {
          const vfloat<K> rcpAbsDen = rcp(absDen);
          const vfloat<K> t = T * rcpAbsDen;
          const vfloat<K> UVW = U+V+W;
          const vbool<K> invalid = abs(UVW) < min_rcp_input;
          const vfloat<K> rcpUVW = select(invalid,vfloat<K>(0.0f),rcp(UVW));
          const vfloat<K> u0 = U * rcpUVW;
          const vfloat<K> v0 = V * rcpUVW;
          const vfloat<K> u1 = vfloat<K>(1.0f) - u0;
          const vfloat<K> v1 = vfloat<K>(1.0f) - v0;
          const vfloat<K> uu = select(flags,u1,u0);
          const vfloat<K> vv = select(flags,v1,v0);
          const unsigned int sx = subgrid.x() + (unsigned int)(i % 2);
          const unsigned int sy = subgrid.y() + (unsigned int)(i >>1);
          const float inv_resX = rcp((float)(int)(g.resX-1));
          const float inv_resY = rcp((float)(int)(g.resY-1));
          const vfloat<K> u = (uu + (float)(int)sx) * inv_resX;
          const vfloat<K> v = (vv + (float)(int)sy) * inv_resY;
          const Vec3vf<K> Ng(tri_Ng.x,tri_Ng.y,tri_Ng.z);
          return std::make_tuple(u,v,t,Ng);
        }

      private:
        const vfloat<K> U;
        const vfloat<K> V;
        const vfloat<K> W;
        const vfloat<K> T;
        const vfloat<K> absDen;
        const vbool<K> flags;
        const Vec3vf<K> tri_Ng;

        const GridMesh::Grid &g;
        const SubGrid& subgrid;
        const size_t i;
      };


    template<int M, int K, bool filter>
      struct SubGridQuadMIntersectorKPlueckerBase
      {
        __forceinline SubGridQuadMIntersectorKPlueckerBase(const vbool<K>& valid, const RayK<K>& ray) {}
            
        template<typename Epilog>
        __forceinline vbool<K> intersectK(const vbool<K>& valid0,
                                          RayK<K>& ray,
                                          const Vec3vf<K>& tri_v0,
                                          const Vec3vf<K>& tri_v1,
                                          const Vec3vf<K>& tri_v2,
                                          const Vec3vf<K>& tri_Ng,
                                          const vbool<K>& flags,
                                          const GridMesh::Grid &g, 
                                          const SubGrid &subgrid,
                                          const unsigned int i,
                                          const Epilog& epilog) const
        { 
          /* calculate denominator */
        /* calculate vertices relative to ray origin */
          vbool<K> valid = valid0;
          const Vec3vf<K> O = ray.org;
          const Vec3vf<K> D = ray.dir;
          const Vec3vf<K> v0 = tri_v0-O;
          const Vec3vf<K> v1 = tri_v1-O;
          const Vec3vf<K> v2 = tri_v2-O;
          
          /* calculate triangle edges */
          const Vec3vf<K> e0 = v2-v0;
          const Vec3vf<K> e1 = v0-v1;
          const Vec3vf<K> e2 = v1-v2;
           
          /* perform edge tests */
          const vfloat<K> U = dot(Vec3vf<K>(cross(e0,v2+v0)),D);
          const vfloat<K> V = dot(Vec3vf<K>(cross(e1,v0+v1)),D);
          const vfloat<K> W = dot(Vec3vf<K>(cross(e2,v1+v2)),D);
#if defined(EMBREE_BACKFACE_CULLING)
          const vfloat<K> maxUVW = max(U,V,W);
          valid &= maxUVW <= 0.0f;
#else
          const vfloat<K> minUVW = min(U,V,W);
          const vfloat<K> maxUVW = max(U,V,W);
          valid &= (maxUVW <= 0.0f) | (minUVW >= 0.0f);
#endif
          if (unlikely(none(valid))) return false;
          
           /* calculate geometry normal and denominator */
          const Vec3vf<K> Ng = stable_triangle_normal(e0,e1,e2);
          const vfloat<K> den = twice(dot(Vec3vf<K>(Ng),D));
          const vfloat<K> absDen = abs(den);
          const vfloat<K> sgnDen = signmsk(den);

          /* perform depth test */
          const vfloat<K> T = twice(dot(v0,Vec3vf<K>(Ng)));
          valid &= absDen*ray.tnear() < (T^sgnDen);
          valid &= (T^sgnDen) <= absDen*ray.tfar;
          if (unlikely(none(valid))) return false;
          
          /* avoid division by 0 */
          valid &= den != vfloat<K>(zero);
          if (unlikely(none(valid))) return false;
                  
          /* calculate hit information */
          SubGridQuadHitPlueckerK<K> hit(U,V,W,T,den,tri_Ng,flags,g,subgrid,i);
          return epilog(valid,hit);
        }
      
        template<typename Epilog>
        __forceinline vbool<K> intersectK(const vbool<K>& valid0, 
                                          RayK<K>& ray,
                                          const Vec3vf<K>& v0,
                                          const Vec3vf<K>& v1,
                                          const Vec3vf<K>& v2,
                                          const vbool<K>& flags,
                                          const GridMesh::Grid &g, 
                                          const SubGrid &subgrid,
                                          const unsigned int i,
                                          const Epilog& epilog) const
        {
          const Vec3vf<K> e1 = v0-v1;
          const Vec3vf<K> e2 = v2-v0;
          const Vec3vf<K> Ng = cross(e2,e1);
          return intersectK(valid0,ray,v0,v1,v2,Ng,flags,g,subgrid,i,epilog);
        }

        template<typename Epilog>
        __forceinline bool intersectK(const vbool<K>& valid0, 
                                      RayK<K>& ray,
                                      const Vec3vf<K>& v0,
                                      const Vec3vf<K>& v1,
                                      const Vec3vf<K>& v2,
                                      const Vec3vf<K>& v3,
                                      const GridMesh::Grid &g, 
                                      const SubGrid &subgrid,
                                      const unsigned int i,
                                      const Epilog& epilog) const
        {
          intersectK(valid0,ray,v0,v1,v3,vbool<K>(false),g,subgrid,i,epilog);
          if (none(valid0)) return true;
          intersectK(valid0,ray,v2,v3,v1,vbool<K>(true ),g,subgrid,i,epilog);
          return none(valid0);
        }

        static  __forceinline bool intersect1(RayK<K>& ray,
                                              size_t k,
                                              const Vec3vf<M>& tri_v0,
                                              const Vec3vf<M>& tri_v1,
                                              const Vec3vf<M>& tri_v2,
                                              const Vec3vf<M>& tri_Ng,
                                              const vbool<M>& flags,
                                              SubGridQuadHitPlueckerM<M> &hit)
        {
          /* calculate vertices relative to ray origin */
          const Vec3vf<M> O = broadcast<vfloat<M>>(ray.org,k);
          const Vec3vf<M> D = broadcast<vfloat<M>>(ray.dir,k);
          const Vec3vf<M> v0 = tri_v0-O;
          const Vec3vf<M> v1 = tri_v1-O;
          const Vec3vf<M> v2 = tri_v2-O;
          
          /* calculate triangle edges */
          const Vec3vf<M> e0 = v2-v0;
          const Vec3vf<M> e1 = v0-v1;
          const Vec3vf<M> e2 = v1-v2;
          
          /* perform edge tests */
          const vfloat<M> U = dot(cross(e0,v2+v0),D);
          const vfloat<M> V = dot(cross(e1,v0+v1),D);
          const vfloat<M> W = dot(cross(e2,v1+v2),D);
          const vfloat<M> minUVW MAYBE_UNUSED = min(U,V,W);
          const vfloat<M> maxUVW = max(U,V,W);
#if defined(EMBREE_BACKFACE_CULLING)
          vbool<M> valid = maxUVW <= 0.0f;
#else
          vbool<M> valid = (minUVW >= 0.0f) | (maxUVW <= 0.0f);
#endif
          if (unlikely(none(valid))) return false;
          
          /* calculate geometry normal and denominator */
          const Vec3vf<M> Ng = stable_triangle_normal(e0,e1,e2);
          const vfloat<M> den = twice(dot(Ng,D));
          const vfloat<M> absDen = abs(den);
          const vfloat<M> sgnDen = signmsk(den);

          /* perform depth test */
          const vfloat<M> T = twice(dot(v0,Ng));
          valid &= absDen*vfloat<M>(ray.tnear()[k]) < (T^sgnDen);
          valid &= (T^sgnDen) <= absDen*vfloat<M>(ray.tfar[k]);
          if (unlikely(none(valid))) return false;
          
          /* avoid division by 0 */
          valid &= den != vfloat<M>(zero);
          if (unlikely(none(valid))) return false;
        
          /* calculate hit information */
          new (&hit) SubGridQuadHitPlueckerM<M>(valid,U,V,W,T,den,tri_Ng,flags);
          return true;
        }

        static __forceinline bool intersect1(RayK<K>& ray,
                                             size_t k,
                                             const Vec3vf<M>& v0,
                                             const Vec3vf<M>& v1,
                                             const Vec3vf<M>& v2,
                                             const vbool<M>& flags,
                                             SubGridQuadHitPlueckerM<M> &hit)
        {
          const Vec3vf<M> e1 = v0-v1;
          const Vec3vf<M> e2 = v2-v0;
          const Vec3vf<M> Ng = cross(e2,e1);
          return intersect1(ray,k,v0,v1,v2,Ng,flags,hit);
        }

      };

    template<int M, int K, bool filter>
      struct SubGridQuadMIntersectorKPluecker : public SubGridQuadMIntersectorKPlueckerBase<M,K,filter>
    {
      __forceinline SubGridQuadMIntersectorKPluecker(const vbool<K>& valid, const RayK<K>& ray)
        : SubGridQuadMIntersectorKPlueckerBase<M,K,filter>(valid,ray) {}

      __forceinline void intersect1(RayHitK<K>& ray, size_t k, IntersectContext* context,
                                    const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3, const GridMesh::Grid &g, const SubGrid &subgrid) const
      {
        Intersect1KEpilogMU<M,K,filter> epilog(ray,k,context,subgrid.geomID(),subgrid.primID());

        SubGridQuadHitPlueckerM<4> hit;
        if (SubGridQuadMIntersectorKPlueckerBase<4,K,filter>::intersect1(ray,k,v0,v1,v3,vboolf4(false),hit))
        {
          interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
          epilog(hit.valid,hit);
        }

        if (SubGridQuadMIntersectorKPlueckerBase<4,K,filter>::intersect1(ray,k,v2,v3,v1,vboolf4(true),hit))
        {
          interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
          epilog(hit.valid,hit);
        }

      }
      
      __forceinline bool occluded1(RayK<K>& ray, size_t k, IntersectContext* context,
                                   const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3, const GridMesh::Grid &g, const SubGrid &subgrid) const
      {
        Occluded1KEpilogMU<M,K,filter> epilog(ray,k,context,subgrid.geomID(),subgrid.primID());

        SubGridQuadHitPlueckerM<4> hit;
        if (SubGridQuadMIntersectorKPlueckerBase<4,K,filter>::intersect1(ray,k,v0,v1,v3,vboolf4(false),hit))
        {
          interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
          if (epilog(hit.valid,hit)) return true;
        }

        if (SubGridQuadMIntersectorKPlueckerBase<4,K,filter>::intersect1(ray,k,v2,v3,v1,vboolf4(true),hit))
        {
          interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
          if (epilog(hit.valid,hit)) return true;
        }
        return false;
      }
    };

  }
}
