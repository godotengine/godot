// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "quadv.h"
#include "triangle_intersector_moeller.h"

namespace embree
{
  namespace isa
  {
    template<int M>
    struct QuadHitM
    {
      __forceinline QuadHitM() {}

      __forceinline QuadHitM(const vbool<M>& valid,
                             const vfloat<M>& U,
                             const vfloat<M>& V,
                             const vfloat<M>& T,
                             const vfloat<M>& absDen,
                             const Vec3vf<M>& Ng,
                             const vbool<M>& flags)
        : U(U), V(V), T(T), absDen(absDen), tri_Ng(Ng), valid(valid), flags(flags) {}

      __forceinline void finalize()
      {
        const vfloat<M> rcpAbsDen = rcp(absDen);
        vt = T * rcpAbsDen;
        const vfloat<M> u = min(U * rcpAbsDen,1.0f);
        const vfloat<M> v = min(V * rcpAbsDen,1.0f);
        const vfloat<M> u1 = vfloat<M>(1.0f) - u;
        const vfloat<M> v1 = vfloat<M>(1.0f) - v;
#if !defined(__AVX__) || defined(EMBREE_BACKFACE_CULLING)
        vu = select(flags,u1,u);
        vv = select(flags,v1,v);
        vNg = Vec3vf<M>(tri_Ng.x,tri_Ng.y,tri_Ng.z);
#else
        const vfloat<M> flip = select(flags,vfloat<M>(-1.0f),vfloat<M>(1.0f));
        vv = select(flags,u1,v);
        vu = select(flags,v1,u);
        vNg = Vec3vf<M>(flip*tri_Ng.x,flip*tri_Ng.y,flip*tri_Ng.z);
#endif
      }

      __forceinline Vec2f uv(const size_t i)
      {
        const float u = vu[i];
        const float v = vv[i];
        return Vec2f(u,v);
      }

      __forceinline float   t(const size_t i) { return vt[i]; }
      __forceinline Vec3fa Ng(const size_t i) { return Vec3fa(vNg.x[i],vNg.y[i],vNg.z[i]); }

    private:
      vfloat<M> U;
      vfloat<M> V;
      vfloat<M> T;
      vfloat<M> absDen;
      Vec3vf<M> tri_Ng;

    public:
      vbool<M> valid;
      vfloat<M> vu;
      vfloat<M> vv;
      vfloat<M> vt;
      Vec3vf<M> vNg;

    public:
      const vbool<M> flags;
    };

    template<int K>
    struct QuadHitK
    {
      __forceinline QuadHitK(const vfloat<K>& U,
                             const vfloat<K>& V,
                             const vfloat<K>& T,
                             const vfloat<K>& absDen,
                             const Vec3vf<K>& Ng,
                             const vbool<K>& flags)
        : U(U), V(V), T(T), absDen(absDen), flags(flags), tri_Ng(Ng) {}

      __forceinline std::tuple<vfloat<K>,vfloat<K>,vfloat<K>,Vec3vf<K>> operator() () const
      {
        const vfloat<K> rcpAbsDen = rcp(absDen);
        const vfloat<K> t = T * rcpAbsDen;
        const vfloat<K> u0 = min(U * rcpAbsDen,1.0f);
        const vfloat<K> v0 = min(V * rcpAbsDen,1.0f);
        const vfloat<K> u1 = vfloat<K>(1.0f) - u0;
        const vfloat<K> v1 = vfloat<K>(1.0f) - v0;
        const vfloat<K> u = select(flags,u1,u0);
        const vfloat<K> v = select(flags,v1,v0);
        const Vec3vf<K> Ng(tri_Ng.x,tri_Ng.y,tri_Ng.z);
        return std::make_tuple(u,v,t,Ng);
      }

    private:
      const vfloat<K> U;
      const vfloat<K> V;
      const vfloat<K> T;
      const vfloat<K> absDen;
      const vbool<K> flags;
      const Vec3vf<K> tri_Ng;
    };

    /* ----------------------------- */
    /* -- single ray intersectors -- */
    /* ----------------------------- */


    template<int M, bool filter>
    struct QuadMIntersector1MoellerTrumbore;

    /*! Intersects M quads with 1 ray */
    template<int M, bool filter>
    struct QuadMIntersector1MoellerTrumbore
    {
      __forceinline QuadMIntersector1MoellerTrumbore() {}

      __forceinline QuadMIntersector1MoellerTrumbore(const Ray& ray, const void* ptr) {}

      __forceinline void intersect(RayHit& ray, RayQueryContext* context,
                                   const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3,
                                   const vuint<M>& geomID, const vuint<M>& primID) const
      {
        UVIdentity<M> mapUV;
        MoellerTrumboreHitM<M,UVIdentity<M>> hit(mapUV);
        MoellerTrumboreIntersector1<M> intersector(ray,nullptr);
        Intersect1EpilogM<M,filter> epilog(ray,context,geomID,primID);

        /* intersect first triangle */
        if (intersector.intersect(ray,v0,v1,v3,mapUV,hit)) 
          epilog(hit.valid,hit);

        /* intersect second triangle */
        if (intersector.intersect(ray,v2,v3,v1,mapUV,hit)) 
        {
          hit.U = hit.absDen - hit.U;
          hit.V = hit.absDen - hit.V;
          epilog(hit.valid,hit);
        }
      }
      
      __forceinline bool occluded(Ray& ray, RayQueryContext* context,
                                  const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3,
                                  const vuint<M>& geomID, const vuint<M>& primID) const
      {
        UVIdentity<M> mapUV;
        MoellerTrumboreHitM<M,UVIdentity<M>> hit(mapUV);
        MoellerTrumboreIntersector1<M> intersector(ray,nullptr);
        Occluded1EpilogM<M,filter> epilog(ray,context,geomID,primID);

        /* intersect first triangle */
        if (intersector.intersect(ray,v0,v1,v3,mapUV,hit)) 
        {
          if (epilog(hit.valid,hit))
            return true;
        }

        /* intersect second triangle */
        if (intersector.intersect(ray,v2,v3,v1,mapUV,hit)) 
        {
          hit.U = hit.absDen - hit.U;
          hit.V = hit.absDen - hit.V;
          if (epilog(hit.valid,hit))
            return true;
        }
        return false;
      }
    };

#if defined(__AVX__)

    /*! Intersects 4 quads with 1 ray using AVX */
    template<bool filter>
    struct QuadMIntersector1MoellerTrumbore<4,filter>
    {
      __forceinline QuadMIntersector1MoellerTrumbore() {}

      __forceinline QuadMIntersector1MoellerTrumbore(const Ray& ray, const void* ptr) {}
      
      template<typename Epilog>
      __forceinline bool intersect(Ray& ray, const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, const Epilog& epilog) const
      {
        const Vec3vf8 vtx0(vfloat8(v0.x,v2.x),vfloat8(v0.y,v2.y),vfloat8(v0.z,v2.z));
#if !defined(EMBREE_BACKFACE_CULLING)
        const Vec3vf8 vtx1(vfloat8(v1.x),vfloat8(v1.y),vfloat8(v1.z));
        const Vec3vf8 vtx2(vfloat8(v3.x),vfloat8(v3.y),vfloat8(v3.z));        
#else
        const Vec3vf8 vtx1(vfloat8(v1.x,v3.x),vfloat8(v1.y,v3.y),vfloat8(v1.z,v3.z));
        const Vec3vf8 vtx2(vfloat8(v3.x,v1.x),vfloat8(v3.y,v1.y),vfloat8(v3.z,v1.z));
#endif
        UVIdentity<8> mapUV;
        MoellerTrumboreHitM<8,UVIdentity<8>> hit(mapUV);
        MoellerTrumboreIntersector1<8> intersector(ray,nullptr);
        const vbool8 flags(0,0,0,0,1,1,1,1);
        if (unlikely(intersector.intersect(ray,vtx0,vtx1,vtx2,mapUV,hit)))
        {
          vfloat8 U = hit.U, V = hit.V, absDen = hit.absDen;

#if !defined(EMBREE_BACKFACE_CULLING)
          hit.U = select(flags,absDen-V,U);
          hit.V = select(flags,absDen-U,V);
          hit.vNg *= select(flags,vfloat8(-1.0f),vfloat8(1.0f)); // FIXME: use XOR
#else
          hit.U = select(flags,absDen-U,U);
          hit.V = select(flags,absDen-V,V);
#endif
          if (unlikely(epilog(hit.valid,hit)))
            return true;
        }
        return false;
      }
      
      __forceinline bool intersect(RayHit& ray, RayQueryContext* context,
                                   const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, 
                                   const vuint4& geomID, const vuint4& primID) const
      {
        return intersect(ray,v0,v1,v2,v3,Intersect1EpilogM<8,filter>(ray,context,vuint8(geomID),vuint8(primID)));
      }
      
      __forceinline bool occluded(Ray& ray, RayQueryContext* context,
                                  const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, 
                                  const vuint4& geomID, const vuint4& primID) const
      {
        return intersect(ray,v0,v1,v2,v3,Occluded1EpilogM<8,filter>(ray,context,vuint8(geomID),vuint8(primID)));
      }
    };

#endif

    /* ----------------------------- */
    /* -- ray packet intersectors -- */
    /* ----------------------------- */


    struct MoellerTrumboreIntersector1KTriangleM
    {
      /*! Intersect k'th ray from ray packet of size K with M triangles. */
      template<int M, int K, typename Epilog>
      static  __forceinline bool intersect(RayK<K>& ray,
                                           size_t k,
                                           const Vec3vf<M>& tri_v0,
                                           const Vec3vf<M>& tri_e1,
                                           const Vec3vf<M>& tri_e2,
                                           const Vec3vf<M>& tri_Ng,
                                           const vbool<M>& flags,
                                           const Epilog& epilog)
      {
        /* calculate denominator */
        const Vec3vf<M> O = broadcast<vfloat<M>>(ray.org,k);
        const Vec3vf<M> D = broadcast<vfloat<M>>(ray.dir,k);
        const Vec3vf<M> C = Vec3vf<M>(tri_v0) - O;
        const Vec3vf<M> R = cross(C,D);
        const vfloat<M> den = dot(Vec3vf<M>(tri_Ng),D);
        const vfloat<M> absDen = abs(den);
        const vfloat<M> sgnDen = signmsk(den);
        
        /* perform edge tests */
        const vfloat<M> U = dot(R,Vec3vf<M>(tri_e2)) ^ sgnDen;
        const vfloat<M> V = dot(R,Vec3vf<M>(tri_e1)) ^ sgnDen;
        
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
        QuadHitM<M> hit(valid,U,V,T,absDen,tri_Ng,flags);
        return epilog(valid,hit);
      }
      
      template<int M, int K, typename Epilog>
      static __forceinline bool intersect1(RayK<K>& ray,
                                           size_t k,
                                           const Vec3vf<M>& v0,
                                           const Vec3vf<M>& v1,
                                           const Vec3vf<M>& v2,
                                           const vbool<M>& flags,
                                           const Epilog& epilog)
      {
        const Vec3vf<M> e1 = v0-v1;
        const Vec3vf<M> e2 = v2-v0;
        const Vec3vf<M> Ng = cross(e2,e1);
        return intersect<M,K>(ray,k,v0,e1,e2,Ng,flags,epilog);
      }
    };

    template<int M, int K, bool filter>
    struct QuadMIntersectorKMoellerTrumboreBase
    {
      __forceinline QuadMIntersectorKMoellerTrumboreBase(const vbool<K>& valid, const RayK<K>& ray) {}
            
      /*! Intersects K rays with one of M triangles. */
      template<typename Epilog>
      __forceinline vbool<K> intersectK(const vbool<K>& valid0,
                                        RayK<K>& ray,
                                        const Vec3vf<K>& tri_v0,
                                        const Vec3vf<K>& tri_e1,
                                        const Vec3vf<K>& tri_e2,
                                        const Vec3vf<K>& tri_Ng,
                                        const vbool<K>& flags,
                                        const Epilog& epilog) const
      { 
        /* calculate denominator */
        vbool<K> valid = valid0;
        const Vec3vf<K> C = tri_v0 - ray.org;
        const Vec3vf<K> R = cross(C,ray.dir);
        const vfloat<K> den = dot(tri_Ng,ray.dir);
        const vfloat<K> absDen = abs(den);
        const vfloat<K> sgnDen = signmsk(den);
        
        /* test against edge p2 p0 */
        const vfloat<K> U = dot(R,tri_e2) ^ sgnDen;
        valid &= U >= 0.0f;
        if (likely(none(valid))) return false;
        
        /* test against edge p0 p1 */
        const vfloat<K> V = dot(R,tri_e1) ^ sgnDen;
        valid &= V >= 0.0f;
        if (likely(none(valid))) return false;
        
        /* test against edge p1 p2 */
        const vfloat<K> W = absDen-U-V;
        valid &= W >= 0.0f;
        if (likely(none(valid))) return false;
        
        /* perform depth test */
        const vfloat<K> T = dot(tri_Ng,C) ^ sgnDen;
        valid &= (absDen*ray.tnear() < T) & (T <= absDen*ray.tfar);
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
        QuadHitK<K> hit(U,V,T,absDen,tri_Ng,flags);
        return epilog(valid,hit);
      }
      
      /*! Intersects K rays with one of M quads. */
      template<typename Epilog>
      __forceinline vbool<K> intersectK(const vbool<K>& valid0, 
                                        RayK<K>& ray,
                                        const Vec3vf<K>& tri_v0,
                                        const Vec3vf<K>& tri_v1,
                                        const Vec3vf<K>& tri_v2,
                                        const vbool<K>& flags,
                                        const Epilog& epilog) const
      {
        const Vec3vf<K> e1 = tri_v0-tri_v1;
        const Vec3vf<K> e2 = tri_v2-tri_v0;
        const Vec3vf<K> Ng = cross(e2,e1);
        return intersectK(valid0,ray,tri_v0,e1,e2,Ng,flags,epilog);
      }

      /*! Intersects K rays with one of M quads. */
      template<typename Epilog>
      __forceinline bool intersectK(const vbool<K>& valid0, 
                                    RayK<K>& ray,
                                    const Vec3vf<K>& v0,
                                    const Vec3vf<K>& v1,
                                    const Vec3vf<K>& v2,
                                    const Vec3vf<K>& v3,
                                    const Epilog& epilog) const
      {
        intersectK(valid0,ray,v0,v1,v3,vbool<K>(false),epilog);
        if (none(valid0)) return true;
        intersectK(valid0,ray,v2,v3,v1,vbool<K>(true ),epilog);
        return none(valid0);
      }
    };

    template<int M, int K, bool filter>
    struct QuadMIntersectorKMoellerTrumbore : public QuadMIntersectorKMoellerTrumboreBase<M,K,filter>
    {
      __forceinline QuadMIntersectorKMoellerTrumbore(const vbool<K>& valid, const RayK<K>& ray)
        : QuadMIntersectorKMoellerTrumboreBase<M,K,filter>(valid,ray) {}

      __forceinline void intersect1(RayHitK<K>& ray, size_t k, RayQueryContext* context,
                                    const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3,
                                    const vuint<M>& geomID, const vuint<M>& primID) const
      {
        Intersect1KEpilogM<M,K,filter> epilog(ray,k,context,geomID,primID);
        MoellerTrumboreIntersector1KTriangleM::intersect1<M,K>(ray,k,v0,v1,v3,vbool<M>(false),epilog);
        MoellerTrumboreIntersector1KTriangleM::intersect1<M,K>(ray,k,v2,v3,v1,vbool<M>(true ),epilog);
      }
      
      __forceinline bool occluded1(RayK<K>& ray, size_t k, RayQueryContext* context,
                                   const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3,
                                   const vuint<M>& geomID, const vuint<M>& primID) const
      {
        Occluded1KEpilogM<M,K,filter> epilog(ray,k,context,geomID,primID);
        if (MoellerTrumboreIntersector1KTriangleM::intersect1<M,K>(ray,k,v0,v1,v3,vbool<M>(false),epilog)) return true;
        if (MoellerTrumboreIntersector1KTriangleM::intersect1<M,K>(ray,k,v2,v3,v1,vbool<M>(true ),epilog)) return true;
        return false;
      }
    };


#if defined(__AVX__)

    /*! Intersects 4 quads with 1 ray using AVX */
    template<int K, bool filter>
    struct QuadMIntersectorKMoellerTrumbore<4,K,filter> : public QuadMIntersectorKMoellerTrumboreBase<4,K,filter>
    {
      __forceinline QuadMIntersectorKMoellerTrumbore(const vbool<K>& valid, const RayK<K>& ray)
        : QuadMIntersectorKMoellerTrumboreBase<4,K,filter>(valid,ray) {}
      
      template<typename Epilog>
      __forceinline bool intersect1(RayK<K>& ray, size_t k,
                                    const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, const Epilog& epilog) const
      {
        const Vec3vf8 vtx0(vfloat8(v0.x,v2.x),vfloat8(v0.y,v2.y),vfloat8(v0.z,v2.z));
#if !defined(EMBREE_BACKFACE_CULLING)
        const Vec3vf8 vtx1(vfloat8(v1.x),vfloat8(v1.y),vfloat8(v1.z));
        const Vec3vf8 vtx2(vfloat8(v3.x),vfloat8(v3.y),vfloat8(v3.z));
#else
        const Vec3vf8 vtx1(vfloat8(v1.x,v3.x),vfloat8(v1.y,v3.y),vfloat8(v1.z,v3.z));
        const Vec3vf8 vtx2(vfloat8(v3.x,v1.x),vfloat8(v3.y,v1.y),vfloat8(v3.z,v1.z));
#endif
        const vbool8 flags(0,0,0,0,1,1,1,1);
        return MoellerTrumboreIntersector1KTriangleM::intersect1<8,K>(ray,k,vtx0,vtx1,vtx2,flags,epilog); 
      }
      
      __forceinline bool intersect1(RayHitK<K>& ray, size_t k, RayQueryContext* context,
                                    const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, 
                                    const vuint4& geomID, const vuint4& primID) const
      {
        return intersect1(ray,k,v0,v1,v2,v3,Intersect1KEpilogM<8,K,filter>(ray,k,context,vuint8(geomID),vuint8(primID)));
      }
      
      __forceinline bool occluded1(RayK<K>& ray, size_t k, RayQueryContext* context,
                                   const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, 
                                   const vuint4& geomID, const vuint4& primID) const
      {
        return intersect1(ray,k,v0,v1,v2,v3,Occluded1KEpilogM<8,K,filter>(ray,k,context,vuint8(geomID),vuint8(primID)));
      }
    };

#endif
  }
}
