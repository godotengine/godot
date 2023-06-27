// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "subgrid.h"
#include "quad_intersector_moeller.h"

namespace embree
{
  namespace isa
  {

    /* ----------------------------- */
    /* -- single ray intersectors -- */
    /* ----------------------------- */

    template<int M>
    __forceinline void interpolateUV(MoellerTrumboreHitM<M,UVIdentity<M>> &hit,const GridMesh::Grid &g, const SubGrid& subgrid, const vint<M> &stepX, const vint<M> &stepY) 
    {
      /* correct U,V interpolation across the entire grid */
      const vint<M> sx((int)subgrid.x());
      const vint<M> sy((int)subgrid.y());
      const vint<M> sxM(sx + stepX); 
      const vint<M> syM(sy + stepY); 
      const float inv_resX = rcp((float)((int)g.resX-1));
      const float inv_resY = rcp((float)((int)g.resY-1));          
      hit.U = (hit.U + (vfloat<M>)sxM * hit.absDen) * inv_resX;
      hit.V = (hit.V + (vfloat<M>)syM * hit.absDen) * inv_resY;
    }
    
    template<int M, bool filter>
      struct SubGridQuadMIntersector1MoellerTrumbore;

    template<int M, bool filter>
      struct SubGridQuadMIntersector1MoellerTrumbore
      {
        __forceinline SubGridQuadMIntersector1MoellerTrumbore() {}

        __forceinline SubGridQuadMIntersector1MoellerTrumbore(const Ray& ray, const void* ptr) {}

        __forceinline void intersect(RayHit& ray, IntersectContext* context,
                                     const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3,
                                     const GridMesh::Grid &g, const SubGrid& subgrid) const
        {
          UVIdentity<M> mapUV;
          MoellerTrumboreHitM<M,UVIdentity<M>> hit(mapUV);
          MoellerTrumboreIntersector1<M> intersector(ray,nullptr);
          Intersect1EpilogMU<M,filter> epilog(ray,context,subgrid.geomID(),subgrid.primID());

          /* intersect first triangle */
          if (intersector.intersect(ray,v0,v1,v3,mapUV,hit)) 
          {
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            epilog(hit.valid,hit);
          }

          /* intersect second triangle */
          if (intersector.intersect(ray,v2,v3,v1,mapUV,hit)) 
          {
            hit.U = hit.absDen - hit.U;
            hit.V = hit.absDen - hit.V;
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            epilog(hit.valid,hit);
          }
        }
      
        __forceinline bool occluded(Ray& ray, IntersectContext* context,
                                    const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3,
                                    const GridMesh::Grid &g, const SubGrid& subgrid) const
        {
          UVIdentity<M> mapUV;
          MoellerTrumboreHitM<M,UVIdentity<M>> hit(mapUV);
          MoellerTrumboreIntersector1<M> intersector(ray,nullptr);
          Occluded1EpilogMU<M,filter> epilog(ray,context,subgrid.geomID(),subgrid.primID());
          
          /* intersect first triangle */
          if (intersector.intersect(ray,v0,v1,v3,mapUV,hit)) 
          {
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            if (epilog(hit.valid,hit))
              return true;
          }

          /* intersect second triangle */
          if (intersector.intersect(ray,v2,v3,v1,mapUV,hit)) 
          {
            hit.U = hit.absDen - hit.U;
            hit.V = hit.absDen - hit.V;
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
      struct SubGridQuadMIntersector1MoellerTrumbore<4,filter>
    {
      __forceinline SubGridQuadMIntersector1MoellerTrumbore() {}

      __forceinline SubGridQuadMIntersector1MoellerTrumbore(const Ray& ray, const void* ptr) {}
      
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
        UVIdentity<8> mapUV;
        MoellerTrumboreHitM<8,UVIdentity<8>> hit(mapUV);
        MoellerTrumboreIntersector1<8> intersector(ray,nullptr);
        const vbool8 flags(0,0,0,0,1,1,1,1);
        if (unlikely(intersector.intersect(ray,vtx0,vtx1,vtx2,mapUV,hit)))
        {
	  /* correct U,V interpolation across the entire grid */
	  const vfloat8 U = select(flags,hit.absDen - hit.V,hit.U);	  
	  const vfloat8 V = select(flags,hit.absDen - hit.U,hit.V);
	  hit.U = U;
	  hit.V = V;
	  hit.vNg *= select(flags,vfloat8(-1.0f),vfloat8(1.0f)); 	  
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

    // ============================================================================================================================
    // ============================================================================================================================
    // ============================================================================================================================


    /* ----------------------------- */
    /* -- ray packet intersectors -- */
    /* ----------------------------- */

    template<int K>
    __forceinline void interpolateUV(const vbool<K>& valid, MoellerTrumboreHitK<K,UVIdentity<K>> &hit,const GridMesh::Grid &g, const SubGrid& subgrid, const unsigned int i) 
    {
      /* correct U,V interpolation across the entire grid */
      const unsigned int sx = subgrid.x() + (unsigned int)(i % 2);
      const unsigned int sy = subgrid.y() + (unsigned int)(i >>1);
      const float inv_resX = rcp((float)(int)(g.resX-1));
      const float inv_resY = rcp((float)(int)(g.resY-1));      
      hit.U = select(valid,(hit.U + vfloat<K>((float)sx) * hit.absDen) * inv_resX,hit.U);
      hit.V = select(valid,(hit.V + vfloat<K>((float)sy) * hit.absDen) * inv_resY,hit.V);
    }
        
    template<int M, int K, bool filter>
      struct SubGridQuadMIntersectorKMoellerTrumboreBase
      {
        __forceinline SubGridQuadMIntersectorKMoellerTrumboreBase(const vbool<K>& valid, const RayK<K>& ray) {}

        template<typename Epilog>
        __forceinline bool intersectK(const vbool<K>& valid, 
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
	  UVIdentity<K> mapUV;
	  MoellerTrumboreHitK<K,UVIdentity<K>> hit(mapUV);
	  MoellerTrumboreIntersectorK<M,K> intersector;

          const vbool<K> valid0 = intersector.intersectK(valid,ray,v0,v1,v3,mapUV,hit);
	  if (any(valid0))
	    {
	      interpolateUV(valid0,hit,g,subgrid,i);
	      epilog(valid0,hit);
	    }
          const vbool<K> valid1 = intersector.intersectK(valid,ray,v2,v3,v1,mapUV,hit);
	  if (any(valid1))
	    {
	      hit.U = hit.absDen - hit.U;
	      hit.V = hit.absDen - hit.V;	      
	      interpolateUV(valid1,hit,g,subgrid,i);
	      epilog(valid1,hit);
	    }
	  return any(valid0|valid1);	  
        }

       template<typename Epilog>
        __forceinline bool occludedK(const vbool<K>& valid, 
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
	  UVIdentity<K> mapUV;
	  MoellerTrumboreHitK<K,UVIdentity<K>> hit(mapUV);
	  MoellerTrumboreIntersectorK<M,K> intersector;

	  vbool<K> valid_final = valid;
          const vbool<K> valid0 = intersector.intersectK(valid,ray,v0,v1,v3,mapUV,hit);
	  if (any(valid0))
	    {
	      interpolateUV(valid0,hit,g,subgrid,i);
	      epilog(valid0,hit);
	      valid_final &= !valid0;
	    }
	  if (none(valid_final)) return true;	      	  
          const vbool<K> valid1 = intersector.intersectK(valid,ray,v2,v3,v1,mapUV,hit);
	  if (any(valid1))
	    {
	      hit.U = hit.absDen - hit.U;
	      hit.V = hit.absDen - hit.V;	      
	      interpolateUV(valid1,hit,g,subgrid,i);
	      epilog(valid1,hit);
	      valid_final &= !valid1;	      
	    }
	  return none(valid_final);
        }

        static __forceinline bool intersect1(RayK<K>& ray,
                                             size_t k,
                                             const Vec3vf<M>& v0,
                                             const Vec3vf<M>& v1,
                                             const Vec3vf<M>& v2,
                                             MoellerTrumboreHitM<M,UVIdentity<M>> &hit)
        {
          const Vec3vf<M> e1 = v0-v1;
          const Vec3vf<M> e2 = v2-v0;
	  MoellerTrumboreIntersectorK<8,K> intersector;
	  UVIdentity<M> mapUV;
	  return intersector.intersectEdge(ray,k,v0,e1,e2,mapUV,hit);
        }
	
      };

    template<int M, int K, bool filter>
      struct SubGridQuadMIntersectorKMoellerTrumbore : public SubGridQuadMIntersectorKMoellerTrumboreBase<M,K,filter>
    {
      __forceinline SubGridQuadMIntersectorKMoellerTrumbore(const vbool<K>& valid, const RayK<K>& ray)
        : SubGridQuadMIntersectorKMoellerTrumboreBase<M,K,filter>(valid,ray) {}

      __forceinline void intersect1(RayHitK<K>& ray, size_t k, IntersectContext* context,
                                    const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3, const GridMesh::Grid &g, const SubGrid &subgrid) const
      {
	UVIdentity<M> mapUV;
	MoellerTrumboreHitM<M,UVIdentity<M>> hit(mapUV);
        Intersect1KEpilogMU<M,K,filter> epilog(ray,k,context,subgrid.geomID(),subgrid.primID());
	MoellerTrumboreIntersectorK<M,K> intersector;
	/* intersect first triangle */
	if (intersector.intersect(ray,k,v0,v1,v3,mapUV,hit)) 
          {
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            epilog(hit.valid,hit);
          }

	/* intersect second triangle */
	if (intersector.intersect(ray,k,v2,v3,v1,mapUV,hit)) 
          {
	    hit.U = hit.absDen - hit.U;
	    hit.V = hit.absDen - hit.V;
            interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
            epilog(hit.valid,hit);
          }
      }
      
      __forceinline bool occluded1(RayK<K>& ray, size_t k, IntersectContext* context,
                                   const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3, const GridMesh::Grid &g, const SubGrid &subgrid) const
      {
	UVIdentity<M> mapUV;
        MoellerTrumboreHitM<M,UVIdentity<M>> hit(mapUV);
        Occluded1KEpilogMU<M,K,filter> epilog(ray,k,context,subgrid.geomID(),subgrid.primID());	
	MoellerTrumboreIntersectorK<M,K> intersector;
	/* intersect first triangle */
	if (intersector.intersect(ray,k,v0,v1,v3,mapUV,hit)) 
        {
          interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
          if (epilog(hit.valid,hit)) return true;
        }

	/* intersect second triangle */
	if (intersector.intersect(ray,k,v2,v3,v1,mapUV,hit)) 
        {
          hit.U = hit.absDen - hit.U;
          hit.V = hit.absDen - hit.V;	  
          interpolateUV<M>(hit,g,subgrid,vint<M>(0,1,1,0),vint<M>(0,0,1,1));
          if (epilog(hit.valid,hit)) return true;
        }
        return false;
      }
    };


#if defined (__AVX__)

    /*! Intersects 4 quads with 1 ray using AVX */
    template<int K, bool filter>
      struct SubGridQuadMIntersectorKMoellerTrumbore<4,K,filter> : public SubGridQuadMIntersectorKMoellerTrumboreBase<4,K,filter>
    {
      __forceinline SubGridQuadMIntersectorKMoellerTrumbore(const vbool<K>& valid, const RayK<K>& ray)
        : SubGridQuadMIntersectorKMoellerTrumboreBase<4,K,filter>(valid,ray) {}
      
      template<typename Epilog>
        __forceinline bool intersect1(RayK<K>& ray, size_t k,const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, 
                                      const GridMesh::Grid &g, const SubGrid &subgrid, const Epilog& epilog) const
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

        UVIdentity<8> mapUV;
        MoellerTrumboreHitM<8,UVIdentity<8>> hit(mapUV);
        if (SubGridQuadMIntersectorKMoellerTrumboreBase<8,K,filter>::intersect1(ray,k,vtx0,vtx1,vtx2,hit))
        {
	  const vfloat8 U = select(flags,hit.absDen - hit.V,hit.U);	  
	  const vfloat8 V = select(flags,hit.absDen - hit.U,hit.V);
	  hit.U = U;
	  hit.V = V;
	  hit.vNg *= select(flags,vfloat8(-1.0f),vfloat8(1.0f)); 	  	  
	  interpolateUV<8>(hit,g,subgrid,vint<8>(0,1,1,0,0,1,1,0),vint<8>(0,0,1,1,0,0,1,1));
          if (unlikely(epilog(hit.valid,hit)))
            return true;

        }
        return false;
      }
      
      __forceinline bool intersect1(RayHitK<K>& ray, size_t k, IntersectContext* context,
                                    const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, const GridMesh::Grid &g, const SubGrid &subgrid) const
      {
        return intersect1(ray,k,v0,v1,v2,v3,g,subgrid,Intersect1KEpilogMU<8,K,filter>(ray,k,context,subgrid.geomID(),subgrid.primID()));
      }
      
      __forceinline bool occluded1(RayK<K>& ray, size_t k, IntersectContext* context,
                                   const Vec3vf4& v0, const Vec3vf4& v1, const Vec3vf4& v2, const Vec3vf4& v3, const GridMesh::Grid &g, const SubGrid &subgrid) const
      {
        return intersect1(ray,k,v0,v1,v2,v3,g,subgrid,Occluded1KEpilogMU<8,K,filter>(ray,k,context,subgrid.geomID(),subgrid.primID()));
      }
    };

#endif



  }
}
