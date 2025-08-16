// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "triangle.h"
#include "intersector_epilog.h"

/*! This intersector implements a modified version of the Moeller
 *  Trumbore intersector from the paper "Fast, Minimum Storage
 *  Ray-Triangle Intersection". In contrast to the paper we
 *  precalculate some factors and factor the calculations differently
 *  to allow precalculating the cross product e1 x e2. The resulting
 *  algorithm is similar to the fastest one of the paper "Optimizing
 *  Ray-Triangle Intersection via Automated Search". */

namespace embree
{
  namespace isa
  {
    template<int M, typename UVMapper>
    struct MoellerTrumboreHitM
    {
      __forceinline MoellerTrumboreHitM(const UVMapper& mapUV) : mapUV(mapUV) {}

      __forceinline MoellerTrumboreHitM(const vbool<M>& valid, const vfloat<M>& U, const vfloat<M>& V, const vfloat<M>& T, const vfloat<M>& absDen, const Vec3vf<M>& Ng, const UVMapper& mapUV)
        : U(U), V(V), T(T), absDen(absDen), mapUV(mapUV), valid(valid), vNg(Ng) {}
      
      __forceinline void finalize() 
      {
        const vfloat<M> rcpAbsDen = rcp(absDen);
        vt = T * rcpAbsDen;
        vu = U * rcpAbsDen;
        vv = V * rcpAbsDen;
        mapUV(vu,vv,vNg);
      }

      __forceinline Vec2vf<M> uv() const { return Vec2vf<M>(vu,vv); }
      __forceinline vfloat<M> t () const { return vt; }
      __forceinline Vec3vf<M> Ng() const { return vNg; }
     
      __forceinline Vec2f uv (const size_t i) const { return Vec2f(vu[i],vv[i]); }
      __forceinline float t  (const size_t i) const { return vt[i]; }
      __forceinline Vec3fa Ng(const size_t i) const { return Vec3fa(vNg.x[i],vNg.y[i],vNg.z[i]); }
      
    public:
      vfloat<M> U;
      vfloat<M> V;
      vfloat<M> T;
      vfloat<M> absDen;
      UVMapper mapUV;
      
    public:
      vbool<M> valid;
      vfloat<M> vu;
      vfloat<M> vv;
      vfloat<M> vt;
      Vec3vf<M> vNg;
    };
    
    template<int M, bool early_out = true>
    struct MoellerTrumboreIntersector1
    {
      __forceinline MoellerTrumboreIntersector1() {}

      __forceinline MoellerTrumboreIntersector1(const Ray& ray, const void* ptr) {}

      template<typename UVMapper>
      __forceinline bool intersect(const vbool<M>& valid0,
                                   Ray& ray,
                                   const Vec3vf<M>& tri_v0,
                                   const Vec3vf<M>& tri_e1,
                                   const Vec3vf<M>& tri_e2,
                                   const Vec3vf<M>& tri_Ng,
                                   const UVMapper& mapUV,
                                   MoellerTrumboreHitM<M,UVMapper>& hit) const
      {
        /* calculate denominator */
        vbool<M> valid = valid0;

        const Vec3vf<M> O = Vec3vf<M>((Vec3fa)ray.org);
        const Vec3vf<M> D = Vec3vf<M>((Vec3fa)ray.dir);
        const Vec3vf<M> C = Vec3vf<M>(tri_v0) - O;
        const Vec3vf<M> R = cross(C,D);
        const vfloat<M> den = dot(Vec3vf<M>(tri_Ng),D);
        const vfloat<M> absDen = abs(den);
        const vfloat<M> sgnDen = signmsk(den);

        /* perform edge tests */
        const vfloat<M> U = asFloat(asInt(dot(R,Vec3vf<M>(tri_e2))) ^ asInt(sgnDen));
        const vfloat<M> V = asFloat(asInt(dot(R,Vec3vf<M>(tri_e1))) ^ asInt(sgnDen));
        
        /* perform backface culling */        
#if defined(EMBREE_BACKFACE_CULLING)
        valid &= (den < vfloat<M>(zero)) & (U >= 0.0f) & (V >= 0.0f) & (U+V<=absDen);
#else
        valid &= (den != vfloat<M>(zero)) & (U >= 0.0f) & (V >= 0.0f) & (U+V<=absDen);
#endif
        if (likely(early_out && none(valid))) return false;

        /* perform depth test */
        const vfloat<M> T = asFloat(asInt(dot(Vec3vf<M>(tri_Ng),C)) ^ asInt(sgnDen));

        valid &= (absDen*vfloat<M>(ray.tnear()) < T) & (T <= absDen*vfloat<M>(ray.tfar));
        if (likely(early_out && none(valid))) return false;
           
        /* update hit information */
        new (&hit) MoellerTrumboreHitM<M,UVMapper>(valid,U,V,T,absDen,tri_Ng,mapUV);

        return early_out || any(valid);
      }

      template<typename UVMapper>
      __forceinline bool intersectEdge(const vbool<M>& valid,
                                       Ray& ray,
                                       const Vec3vf<M>& tri_v0,
                                       const Vec3vf<M>& tri_e1,
                                       const Vec3vf<M>& tri_e2,
                                       const UVMapper& mapUV,
                                       MoellerTrumboreHitM<M,UVMapper>& hit) const
      {
        const Vec3<vfloat<M>> tri_Ng = cross(tri_e2,tri_e1);
        return intersect(valid,ray,tri_v0,tri_e1,tri_e2,tri_Ng,mapUV,hit);
      }

      template<typename UVMapper>
      __forceinline bool intersectEdge(Ray& ray,
                                       const Vec3vf<M>& tri_v0,
                                       const Vec3vf<M>& tri_e1,
                                       const Vec3vf<M>& tri_e2,
                                       const UVMapper& mapUV,
                                       MoellerTrumboreHitM<M,UVMapper>& hit) const
      {
        vbool<M> valid = true;
        const Vec3<vfloat<M>> tri_Ng = cross(tri_e2,tri_e1);
        return intersect(valid,ray,tri_v0,tri_e1,tri_e2,tri_Ng,mapUV,hit);
      }

      template<typename UVMapper>
      __forceinline bool intersect(Ray& ray,
                                   const Vec3vf<M>& v0,
                                   const Vec3vf<M>& v1,
                                   const Vec3vf<M>& v2,
                                   const UVMapper& mapUV,
                                   MoellerTrumboreHitM<M,UVMapper>& hit) const
      {
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v2-v0;
        return intersectEdge(ray,v0,e1,e2,mapUV,hit);
      }

      template<typename UVMapper>
      __forceinline bool intersect(const vbool<M>& valid,
                                   Ray& ray,
                                   const Vec3vf<M>& v0,
                                   const Vec3vf<M>& v1,
                                   const Vec3vf<M>& v2,
                                   const UVMapper& mapUV,
                                   MoellerTrumboreHitM<M,UVMapper>& hit) const
      {
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v2-v0;
        return intersectEdge(valid,ray,v0,e1,e2,mapUV,hit);
      }

      template<typename UVMapper, typename Epilog>
      __forceinline bool intersectEdge(Ray& ray,
                                       const Vec3vf<M>& v0,
                                       const Vec3vf<M>& e1,
                                       const Vec3vf<M>& e2,
                                       const UVMapper& mapUV,
                                       const Epilog& epilog) const
      {
        MoellerTrumboreHitM<M,UVMapper> hit(mapUV);
        if (likely(intersectEdge(ray,v0,e1,e2,mapUV,hit))) return epilog(hit.valid,hit);
        return false;
      }

      template<typename UVMapper, typename Epilog>
        __forceinline bool intersect(Ray& ray,
                                     const Vec3vf<M>& v0,
                                     const Vec3vf<M>& v1,
                                     const Vec3vf<M>& v2,
                                     const UVMapper& mapUV,
                                     const Epilog& epilog) const
      {
        MoellerTrumboreHitM<M,UVMapper> hit(mapUV);
        if (likely(intersect(ray,v0,v1,v2,mapUV,hit))) return epilog(hit.valid,hit);
        return false;
      }

      template<typename Epilog>
        __forceinline bool intersect(Ray& ray,
                                     const Vec3vf<M>& v0,
                                     const Vec3vf<M>& v1,
                                     const Vec3vf<M>& v2,
                                     const Epilog& epilog) const
      {
        auto mapUV = UVIdentity<M>();
        MoellerTrumboreHitM<M,UVIdentity<M>> hit(mapUV);
        if (likely(intersect(ray,v0,v1,v2,mapUV,hit))) return epilog(hit.valid,hit);
        return false;
      }

      template<typename UVMapper, typename Epilog>
      __forceinline bool intersect(const vbool<M>& valid,
                                   Ray& ray,
                                   const Vec3vf<M>& v0,
                                   const Vec3vf<M>& v1,
                                   const Vec3vf<M>& v2,
                                   const UVMapper& mapUV,
                                   const Epilog& epilog) const
      {
        MoellerTrumboreHitM<M,UVMapper> hit(mapUV);
        if (likely(intersect(valid,ray,v0,v1,v2,mapUV,hit))) return epilog(hit.valid,hit);
        return false;
      }
    };
    
    template<int K, typename UVMapper>
    struct MoellerTrumboreHitK
    {
      __forceinline MoellerTrumboreHitK(const UVMapper& mapUV) : mapUV(mapUV) {}
      __forceinline MoellerTrumboreHitK(const vfloat<K>& U, const vfloat<K>& V, const vfloat<K>& T, const vfloat<K>& absDen, const Vec3vf<K>& Ng, const UVMapper& mapUV)
        : U(U), V(V), T(T), absDen(absDen), Ng(Ng), mapUV(mapUV) {}
      
      __forceinline std::tuple<vfloat<K>,vfloat<K>,vfloat<K>,Vec3vf<K>> operator() () const
      {
        const vfloat<K> rcpAbsDen = rcp(absDen);
        const vfloat<K> t = T * rcpAbsDen;
        vfloat<K> u = U * rcpAbsDen;
        vfloat<K> v = V * rcpAbsDen;
        Vec3vf<K> vNg = Ng;
        mapUV(u,v,vNg);
        return std::make_tuple(u,v,t,vNg);
      }
      
      vfloat<K> U;
      vfloat<K> V;
      const vfloat<K> T;
      const vfloat<K> absDen;
      const Vec3vf<K> Ng;
      const UVMapper& mapUV;
    };
    
    template<int M, int K>
    struct MoellerTrumboreIntersectorK
    {
      __forceinline MoellerTrumboreIntersectorK() {}
      __forceinline MoellerTrumboreIntersectorK(const vbool<K>& valid, const RayK<K>& ray) {}
      
      /*! Intersects K rays with one of M triangles. */
      template<typename UVMapper>
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
                                        const UVMapper& mapUV,
                                        MoellerTrumboreHitK<K,UVMapper> &hit) const
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
        new (&hit) MoellerTrumboreHitK<K,UVMapper>(U,V,T,absDen,tri_Ng,mapUV);
        return valid;
      }

      /*! Intersects K rays with one of M triangles. */
      template<typename UVMapper>
      __forceinline vbool<K> intersectK(const vbool<K>& valid0, 
                                        RayK<K>& ray,
                                        const Vec3vf<K>& tri_v0,
                                        const Vec3vf<K>& tri_v1,
                                        const Vec3vf<K>& tri_v2,
                                        const UVMapper& mapUV,
                                        MoellerTrumboreHitK<K,UVMapper> &hit) const
      {
        const Vec3vf<K> e1 = tri_v0-tri_v1;
        const Vec3vf<K> e2 = tri_v2-tri_v0;
        const Vec3vf<K> Ng = cross(e2,e1);
        return intersectK(valid0,ray.org,ray.dir,ray.tnear(),ray.tfar,tri_v0,e1,e2,Ng,mapUV,hit);
      }

      /*! Intersects K rays with one of M triangles. */
      template<typename UVMapper, typename Epilog>
      __forceinline vbool<K> intersectK(const vbool<K>& valid0, 
                                        RayK<K>& ray,
                                        const Vec3vf<K>& tri_v0,
                                        const Vec3vf<K>& tri_v1,
                                        const Vec3vf<K>& tri_v2,
                                        const UVMapper& mapUV,
                                        const Epilog& epilog) const
      {
        MoellerTrumboreHitK<K,UVIdentity<K>> hit(mapUV);		
        const Vec3vf<K> e1 = tri_v0-tri_v1;
        const Vec3vf<K> e2 = tri_v2-tri_v0;
        const Vec3vf<K> Ng = cross(e2,e1);
        const vbool<K> valid = intersectK(valid0,ray.org,ray.dir,ray.tnear(),ray.tfar,tri_v0,e1,e2,Ng,mapUV,hit);
	return epilog(valid,hit);
      }


      
      template<typename Epilog>
      __forceinline vbool<K> intersectK(const vbool<K>& valid0, 
                                        RayK<K>& ray,
                                        const Vec3vf<K>& tri_v0,
                                        const Vec3vf<K>& tri_v1,
                                        const Vec3vf<K>& tri_v2,
                                        const Epilog& epilog) const
      {
	UVIdentity<K> mapUV;	
        MoellerTrumboreHitK<K,UVIdentity<K>> hit(mapUV);			
        const Vec3vf<K> e1 = tri_v0-tri_v1;
        const Vec3vf<K> e2 = tri_v2-tri_v0;
        const Vec3vf<K> Ng = cross(e2,e1);
        const vbool<K> valid = intersectK(valid0,ray.org,ray.dir,ray.tnear(),ray.tfar,tri_v0,e1,e2,Ng,mapUV,hit);
	return epilog(valid,hit);
      }

      /*! Intersects K rays with one of M triangles. */
      template<typename UVMapper, typename Epilog>
      __forceinline vbool<K> intersectEdgeK(const vbool<K>& valid0, 
                                            RayK<K>& ray,
                                            const Vec3vf<K>& tri_v0, 
                                            const Vec3vf<K>& tri_e1, 
                                            const Vec3vf<K>& tri_e2,
                                            const UVMapper& mapUV,
                                            const Epilog& epilog) const
      {
        MoellerTrumboreHitK<K,UVIdentity<K>> hit(mapUV);			
        const Vec3vf<K> tri_Ng = cross(tri_e2,tri_e1);
        const vbool<K> valid = intersectK(valid0,ray.org,ray.dir,ray.tnear(),ray.tfar,tri_v0,tri_e1,tri_e2,tri_Ng,mapUV,hit);
	return epilog(valid,hit);
      }
      
      /*! Intersect k'th ray from ray packet of size K with M triangles. */
      template<typename UVMapper>
      __forceinline bool intersectEdge(RayK<K>& ray,
                                       size_t k,
                                       const Vec3vf<M>& tri_v0,
                                       const Vec3vf<M>& tri_e1,
                                       const Vec3vf<M>& tri_e2,
                                       const UVMapper& mapUV,
                                       MoellerTrumboreHitM<M,UVMapper>& hit) const
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
        new (&hit) MoellerTrumboreHitM<M,UVMapper>(valid,U,V,T,absDen,tri_Ng,mapUV);
        return true;
      }

       template<typename UVMapper>
      __forceinline bool intersectEdge(RayK<K>& ray,
                                       size_t k,
                                       const BBox<vfloat<M>>& time_range,
                                       const Vec3vf<M>& tri_v0, 
                                       const Vec3vf<M>& tri_e1, 
                                       const Vec3vf<M>& tri_e2,
                                       const UVMapper& mapUV,
                                       MoellerTrumboreHitM<M,UVMapper>& hit) const
      {
        if (likely(intersect(ray,k,tri_v0,tri_e1,tri_e2,mapUV,hit))) 
        {
          hit.valid &= time_range.lower <= vfloat<M>(ray.time[k]);
          hit.valid &= vfloat<M>(ray.time[k]) < time_range.upper;
          return any(hit.valid);
        }
        return false;
      }

      template<typename UVMapper>
      __forceinline bool intersect(RayK<K>& ray,
                                   size_t k,
                                   const Vec3vf<M>& v0, 
                                   const Vec3vf<M>& v1, 
                                   const Vec3vf<M>& v2,
                                   const UVMapper& mapUV,
                                   MoellerTrumboreHitM<M,UVMapper>& hit) const      
      {
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v2-v0;
        return intersectEdge(ray,k,v0,e1,e2,mapUV,hit);
      }
      
      template<typename UVMapper, typename Epilog>
      __forceinline bool intersectEdge(RayK<K>& ray,
                                       size_t k,
                                       const Vec3vf<M>& tri_v0, 
                                       const Vec3vf<M>& tri_e1, 
                                       const Vec3vf<M>& tri_e2,
                                       const UVMapper& mapUV,
                                       const Epilog& epilog) const
      {
        MoellerTrumboreHitM<M,UVMapper> hit(mapUV);
        if (likely(intersectEdge(ray,k,tri_v0,tri_e1,tri_e2,mapUV,hit))) return epilog(hit.valid,hit);
        return false;
      }

      template<typename UVMapper, typename Epilog>
      __forceinline bool intersectEdge(RayK<K>& ray,
                                       size_t k,                           
                                       const BBox<vfloat<M>>& time_range,
                                       const Vec3vf<M>& tri_v0, 
                                       const Vec3vf<M>& tri_e1, 
                                       const Vec3vf<M>& tri_e2,
                                       const UVMapper& mapUV,
                                       const Epilog& epilog) const
      {
        MoellerTrumboreHitM<M,UVMapper> hit(mapUV);
        if (likely(intersectEdge(ray,k,time_range,tri_v0,tri_e1,tri_e2,mapUV,hit))) return epilog(hit.valid,hit);
        return false;
      }
      
      template<typename UVMapper, typename Epilog>
      __forceinline bool intersect(RayK<K>& ray,
                                   size_t k,
                                   const Vec3vf<M>& v0, 
                                   const Vec3vf<M>& v1, 
                                   const Vec3vf<M>& v2,
                                   const UVMapper& mapUV,
                                   const Epilog& epilog) const      
      {
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v2-v0;
        return intersectEdge(ray,k,v0,e1,e2,mapUV,epilog);
      }

      template<typename Epilog>
      __forceinline bool intersect(RayK<K>& ray,
                                   size_t k,
                                   const Vec3vf<M>& v0, 
                                   const Vec3vf<M>& v1, 
                                   const Vec3vf<M>& v2,
                                   const Epilog& epilog) const      
      {
        return intersect(ray,k,v0,v1,v2,UVIdentity<M>(),epilog);
      }

      template<typename UVMapper, typename Epilog>
      __forceinline bool intersect(RayK<K>& ray,
                                   size_t k,
                                   const BBox<vfloat<M>>& time_range,
                                   const Vec3vf<M>& v0,
                                   const Vec3vf<M>& v1,
                                   const Vec3vf<M>& v2,
                                   const UVMapper& mapUV,
                                   const Epilog& epilog) const
      {
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v2-v0;
        return intersectEdge(ray,k,time_range,v0,e1,e2,mapUV,epilog);
      }
    };
  }
}
