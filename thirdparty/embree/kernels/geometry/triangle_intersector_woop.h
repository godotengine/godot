// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "triangle.h"
#include "intersector_epilog.h"

/*! This intersector implements a modified version of the Woop's ray-triangle intersection test */

namespace embree
{
  namespace isa
  {
    template<int M>
    struct WoopHitM
    {
      __forceinline WoopHitM() {}

      __forceinline WoopHitM(const vbool<M>& valid, 
                             const vfloat<M>& U, 
                             const vfloat<M>& V, 
                             const vfloat<M>& T, 
                             const vfloat<M>& inv_det,                              
                             const Vec3vf<M>& Ng)
        : U(U), V(V), T(T), inv_det(inv_det), valid(valid), vNg(Ng) {}
      
      __forceinline void finalize() 
      {
        vt = T;
        vu = U*inv_det;
        vv = V*inv_det;
      }

      __forceinline Vec2f uv (const size_t i) const { return Vec2f(vu[i],vv[i]); }
      __forceinline float t  (const size_t i) const { return vt[i]; }
      __forceinline Vec3fa Ng(const size_t i) const { return Vec3fa(vNg.x[i],vNg.y[i],vNg.z[i]); }
      
    private:
      const vfloat<M> U;
      const vfloat<M> V;
      const vfloat<M> T;
      const vfloat<M> inv_det;
      
    public:
      const vbool<M> valid;
      vfloat<M> vu;
      vfloat<M> vv;
      vfloat<M> vt;
      Vec3vf<M> vNg;
    };

    template<int M>
    struct WoopPrecalculations1
    {
      unsigned int kx,ky,kz;
      Vec3vf<M> org;
      Vec3fa S;
      __forceinline WoopPrecalculations1() {}

      __forceinline WoopPrecalculations1(const Ray& ray, const void* ptr)
      {
        kz = maxDim(abs(ray.dir));
        kx = (kz+1) % 3;
        ky = (kx+1) % 3;
        const float inv_dir_kz = rcp(ray.dir[kz]);
        if (ray.dir[kz] < 0.0f) std::swap(kx,ky);
        S.x = ray.dir[kx] * inv_dir_kz;
        S.y = ray.dir[ky] * inv_dir_kz;
        S.z = inv_dir_kz;
        org = Vec3vf<M>(ray.org[kx],ray.org[ky],ray.org[kz]);
      }
    };

    
    template<int M>
    struct WoopIntersector1
    {

        typedef WoopPrecalculations1<M> Precalculations;

      __forceinline WoopIntersector1() {}

      __forceinline WoopIntersector1(const Ray& ray, const void* ptr) {}

      static __forceinline bool intersect(const vbool<M>& valid0,
                                          Ray& ray,
                                          const Precalculations& pre,
                                          const Vec3vf<M>& tri_v0,
                                          const Vec3vf<M>& tri_v1,
                                          const Vec3vf<M>& tri_v2,
                                          WoopHitM<M>& hit)
      {       
        vbool<M> valid = valid0;

        /* vertices relative to ray origin */
        const Vec3vf<M> org = Vec3vf<M>(pre.org.x,pre.org.y,pre.org.z);
        const Vec3vf<M> A = Vec3vf<M>(tri_v0[pre.kx],tri_v0[pre.ky],tri_v0[pre.kz]) - org;
        const Vec3vf<M> B = Vec3vf<M>(tri_v1[pre.kx],tri_v1[pre.ky],tri_v1[pre.kz]) - org;
        const Vec3vf<M> C = Vec3vf<M>(tri_v2[pre.kx],tri_v2[pre.ky],tri_v2[pre.kz]) - org;

        /* shear and scale vertices */
        const vfloat<M> Ax = nmadd(A.z,pre.S.x,A.x);
        const vfloat<M> Ay = nmadd(A.z,pre.S.y,A.y);
        const vfloat<M> Bx = nmadd(B.z,pre.S.x,B.x);
        const vfloat<M> By = nmadd(B.z,pre.S.y,B.y);
        const vfloat<M> Cx = nmadd(C.z,pre.S.x,C.x);
        const vfloat<M> Cy = nmadd(C.z,pre.S.y,C.y);

        /* scaled barycentric */
        const vfloat<M> U0 = Cx*By;
        const vfloat<M> U1 = Cy*Bx;
        const vfloat<M> V0 = Ax*Cy;
        const vfloat<M> V1 = Ay*Cx;
        const vfloat<M> W0 = Bx*Ay;
        const vfloat<M> W1 = By*Ax;
#if !defined(__AVX512F__)
        valid &= (U0 >= U1) & (V0 >= V1) & (W0 >= W1) |
          (U0 <= U1) & (V0 <= V1) & (W0 <= W1);
#else
        valid &= ge(ge(U0 >= U1,V0,V1),W0,W1) | le(le(U0 <= U1,V0,V1),W0,W1);
#endif

        if (likely(none(valid))) return false;
        const vfloat<M> U = U0-U1;
        const vfloat<M> V = V0-V1;
        const vfloat<M> W = W0-W1;

        const vfloat<M> det = U+V+W;

        valid &= det != 0.0f;
        const vfloat<M> inv_det = rcp(det);

        const vfloat<M> Az = pre.S.z * A.z;
        const vfloat<M> Bz = pre.S.z * B.z;
        const vfloat<M> Cz = pre.S.z * C.z;
        const vfloat<M> T  = madd(U,Az,madd(V,Bz,W*Cz)); 
        const vfloat<M> t  = T * inv_det;
        /* perform depth test */
        valid &= (vfloat<M>(ray.tnear()) < t) & (t <= vfloat<M>(ray.tfar));
        if (likely(none(valid))) return false;
        
        const Vec3vf<M> tri_Ng = cross(tri_v2-tri_v0,tri_v0-tri_v1);

        /* update hit information */
        new (&hit) WoopHitM<M>(valid,U,V,t,inv_det,tri_Ng);
        return true;
      }
      
      static __forceinline bool intersect(Ray& ray,
                                   const Precalculations& pre,
                                   const Vec3vf<M>& v0,
                                   const Vec3vf<M>& v1,
                                   const Vec3vf<M>& v2,
                                   WoopHitM<M>& hit)
      {
        vbool<M> valid = true;
        return intersect(valid,ray,pre,v0,v1,v2,hit);
      }


      template<typename Epilog>
      static __forceinline bool intersect(Ray& ray,
                                     const Precalculations& pre,
                                     const Vec3vf<M>& v0,
                                     const Vec3vf<M>& v1,
                                     const Vec3vf<M>& v2,
                                     const Epilog& epilog)
      {
        WoopHitM<M> hit;
        if (likely(intersect(ray,pre,v0,v1,v2,hit))) return epilog(hit.valid,hit);
        return false;
      }

      template<typename Epilog>
      static __forceinline bool intersect(const vbool<M>& valid,
                                   Ray& ray,
                                   const Precalculations& pre,
                                   const Vec3vf<M>& v0,
                                   const Vec3vf<M>& v1,
                                   const Vec3vf<M>& v2,
                                   const Epilog& epilog)
      {
        WoopHitM<M> hit;
        if (likely(intersect(valid,ray,pre,v0,v1,v2,hit))) return epilog(hit.valid,hit);
        return false;
      }
    };
    
#if 0
    template<int K>
    struct WoopHitK
    {
      __forceinline WoopHitK(const vfloat<K>& U, const vfloat<K>& V, const vfloat<K>& T, const vfloat<K>& absDen, const Vec3vf<K>& Ng)
        : U(U), V(V), T(T), absDen(absDen), Ng(Ng) {}
      
      __forceinline std::tuple<vfloat<K>,vfloat<K>,vfloat<K>,Vec3vf<K>> operator() () const
      {
        const vfloat<K> rcpAbsDen = rcp(absDen);
        const vfloat<K> t = T * rcpAbsDen;
        const vfloat<K> u = U * rcpAbsDen;
        const vfloat<K> v = V * rcpAbsDen;
        return std::make_tuple(u,v,t,Ng);
      }
      
    private:
      const vfloat<K> U;
      const vfloat<K> V;
      const vfloat<K> T;
      const vfloat<K> absDen;
      const Vec3vf<K> Ng;
    };
    
    template<int M, int K>
    struct WoopIntersectorK
    {
      __forceinline WoopIntersectorK(const vbool<K>& valid, const RayK<K>& ray) {}
      
      /*! Intersects K rays with one of M triangles. */
      template<typename Epilog>
      __forceinline vbool<K> intersectK(const vbool<K>& valid0,
                                        //RayK<K>& ray,
                                        const Vec3vf<K>& ray_org,
                                        const Vec3vf<K>& ray_dir,
                                        const vfloat<K>& ray_tnear,
                                        const vfloat<K>& ray_tfar,
                                        const Vec3vf<K>& tri_v0,
                                        const Vec3vf<K>& tri_e1,
                                        const Vec3vf<K>& tri_e2,
                                        const Vec3vf<K>& tri_Ng,
                                        const Epilog& epilog) const
      { 
        /* calculate denominator */
        vbool<K> valid = valid0;
        const Vec3vf<K> C = tri_v0 - ray_org;
        const Vec3vf<K> R = cross(C,ray_dir);
        const vfloat<K> den = dot(tri_Ng,ray_dir);
        const vfloat<K> absDen = abs(den);
        const vfloat<K> sgnDen = signmsk(den);
        
        /* test against edge p2 p0 */
        const vfloat<K> U = dot(tri_e2,R) ^ sgnDen;
        valid &= U >= 0.0f;
        if (likely(none(valid))) return false;
        
        /* test against edge p0 p1 */
        const vfloat<K> V = dot(tri_e1,R) ^ sgnDen;
        valid &= V >= 0.0f;
        if (likely(none(valid))) return false;
        
        /* test against edge p1 p2 */
        const vfloat<K> W = absDen-U-V;
        valid &= W >= 0.0f;
        if (likely(none(valid))) return false;
        
        /* perform depth test */
        const vfloat<K> T = dot(tri_Ng,C) ^ sgnDen;
        valid &= (absDen*ray_tnear < T) & (T <= absDen*ray_tfar);
        if (unlikely(none(valid))) return false;
        
        /* perform backface culling */
#if defined(EMBREE_BACKFACE_CULLING)
        valid &= den < vfloat<K>(zero);
        if (unlikely(none(valid))) return false;
#else
        valid &= den != vfloat<K>(zero);
        if (unlikely(none(valid))) return false;
#endif
        
        /* calculate hit information */
        WoopHitK<K> hit(U,V,T,absDen,tri_Ng);
        return epilog(valid,hit);
      }
      
      /*! Intersects K rays with one of M triangles. */
      template<typename Epilog>
      __forceinline vbool<K> intersectK(const vbool<K>& valid0, 
                                        RayK<K>& ray,
                                        const Vec3vf<K>& tri_v0,
                                        const Vec3vf<K>& tri_v1,
                                        const Vec3vf<K>& tri_v2,
                                        const Epilog& epilog) const
      {
        const Vec3vf<K> e1 = tri_v0-tri_v1;
        const Vec3vf<K> e2 = tri_v2-tri_v0;
        const Vec3vf<K> Ng = cross(e2,e1);
        return intersectK(valid0,ray.org,ray.dir,ray.tnear(),ray.tfar,tri_v0,e1,e2,Ng,epilog);
      }

      /*! Intersects K rays with one of M triangles. */
      template<typename Epilog>
      __forceinline vbool<K> intersectEdgeK(const vbool<K>& valid0, 
                                            RayK<K>& ray,
                                            const Vec3vf<K>& tri_v0, 
                                            const Vec3vf<K>& tri_e1, 
                                            const Vec3vf<K>& tri_e2, 
                                            const Epilog& epilog) const
      {
        const Vec3vf<K> tri_Ng = cross(tri_e2,tri_e1);
        return intersectK(valid0,ray.org,ray.dir,ray.tnear(),ray.tfar,tri_v0,tri_e1,tri_e2,tri_Ng,epilog);
      }
      
      /*! Intersect k'th ray from ray packet of size K with M triangles. */
      __forceinline bool intersectEdge(RayK<K>& ray,
                                       size_t k,
                                       const Vec3vf<M>& tri_v0,
                                       const Vec3vf<M>& tri_e1,
                                       const Vec3vf<M>& tri_e2,
                                       WoopHitM<M>& hit) const
      {
        /* calculate denominator */
        typedef Vec3vf<M> Vec3vfM;
        const Vec3vf<M> tri_Ng = cross(tri_e2,tri_e1);

        const Vec3vfM O = broadcast<vfloat<M>>(ray.org,k);
        const Vec3vfM D = broadcast<vfloat<M>>(ray.dir,k);
        const Vec3vfM C = Vec3vfM(tri_v0) - O;
        const Vec3vfM R = cross(C,D);
        const vfloat<M> den = dot(Vec3vfM(tri_Ng),D);
        const vfloat<M> absDen = abs(den);
        const vfloat<M> sgnDen = signmsk(den);
        
        /* perform edge tests */
        const vfloat<M> U = dot(Vec3vf<M>(tri_e2),R) ^ sgnDen;
        const vfloat<M> V = dot(Vec3vf<M>(tri_e1),R) ^ sgnDen;
        
        /* perform backface culling */
#if defined(EMBREE_BACKFACE_CULLING)
        vbool<M> valid = (den < vfloat<M>(zero)) & (U >= 0.0f) & (V >= 0.0f) & (U+V<=absDen);
#else
        vbool<M> valid = (den != vfloat<M>(zero)) & (U >= 0.0f) & (V >= 0.0f) & (U+V<=absDen);
#endif
        if (likely(none(valid))) return false;
        
        /* perform depth test */
        const vfloat<M> T = dot(Vec3vf<M>(tri_Ng),C) ^ sgnDen;
        valid &= (absDen*vfloat<M>(ray.tnear()[k]) < T) & (T <= absDen*vfloat<M>(ray.tfar[k]));
        if (likely(none(valid))) return false;
        
        /* calculate hit information */
        new (&hit) WoopHitM<M>(valid,U,V,T,absDen,tri_Ng);
        return true;
      }

      __forceinline bool intersectEdge(RayK<K>& ray,
                                       size_t k,
                                       const BBox<vfloat<M>>& time_range,
                                       const Vec3vf<M>& tri_v0, 
                                       const Vec3vf<M>& tri_e1, 
                                       const Vec3vf<M>& tri_e2, 
                                       WoopHitM<M>& hit) const
      {
        if (likely(intersect(ray,k,tri_v0,tri_e1,tri_e2,hit))) 
        {
          hit.valid &= time_range.lower <= vfloat<M>(ray.time[k]);
          hit.valid &= vfloat<M>(ray.time[k]) < time_range.upper;
          return any(hit.valid);
        }
        return false;
      }

      template<typename Epilog>
      __forceinline bool intersectEdge(RayK<K>& ray,
                                       size_t k,
                                       const Vec3vf<M>& tri_v0, 
                                       const Vec3vf<M>& tri_e1, 
                                       const Vec3vf<M>& tri_e2, 
                                       const Epilog& epilog) const
      {
        WoopHitM<M> hit;
        if (likely(intersectEdge(ray,k,tri_v0,tri_e1,tri_e2,hit))) return epilog(hit.valid,hit);
        return false;
      }

      template<typename Epilog>
      __forceinline bool intersectEdge(RayK<K>& ray,
                                       size_t k,                           
                                       const BBox<vfloat<M>>& time_range,
                                       const Vec3vf<M>& tri_v0, 
                                       const Vec3vf<M>& tri_e1, 
                                       const Vec3vf<M>& tri_e2, 
                                       const Epilog& epilog) const
      {
        WoopHitM<M> hit;
        if (likely(intersectEdge(ray,k,time_range,tri_v0,tri_e1,tri_e2,hit))) return epilog(hit.valid,hit);
        return false;
      }
      
      template<typename Epilog>
      __forceinline bool intersect(RayK<K>& ray,
                                   size_t k,
                                   const Vec3vf<M>& v0, 
                                   const Vec3vf<M>& v1, 
                                   const Vec3vf<M>& v2, 
                                   const Epilog& epilog) const      
      {
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v2-v0;
        return intersectEdge(ray,k,v0,e1,e2,epilog);
      }

      template<typename Epilog>
      __forceinline bool intersect(RayK<K>& ray,
                                   size_t k,
                                   const BBox<vfloat<M>>& time_range,
                                   const Vec3vf<M>& v0,
                                   const Vec3vf<M>& v1,
                                   const Vec3vf<M>& v2,
                                   const Epilog& epilog) const
      {
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v2-v0;
        return intersectEdge(ray,k,time_range,v0,e1,e2,epilog);
      }
    };
#endif
  }
}
