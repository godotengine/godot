// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "subgrid.h"
#include "subgrid_intersector_moeller.h"
#include "subgrid_intersector_pluecker.h"

namespace embree
{
  namespace isa
  {

    // =======================================================================================
    // =================================== SubGridIntersectors ===============================
    // =======================================================================================


    template<int N, bool filter>
    struct SubGridIntersector1Moeller
    {
      typedef SubGridQBVHN<N> Primitive;
      typedef SubGridQuadMIntersector1MoellerTrumbore<4,filter> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const SubGrid& subgrid)
      {
        STAT3(normal.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        Vec3vf4 v0,v1,v2,v3; subgrid.gather(v0,v1,v2,v3,context->scene);
        pre.intersect(ray,context,v0,v1,v2,v3,g,subgrid);
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const SubGrid& subgrid)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        Vec3vf4 v0,v1,v2,v3; subgrid.gather(v0,v1,v2,v3,context->scene);
        return pre.occluded(ray,context,v0,v1,v2,v3,g,subgrid);
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const SubGrid& subgrid)
      {
        STAT3(point_query.trav_prims,1,1,1);
        AccelSet* accel = (AccelSet*)context->scene->get(subgrid.geomID());
        assert(accel);
        context->geomID = subgrid.geomID();
        context->primID = subgrid.primID();
        return accel->pointQuery(query, context);
      }

      template<bool robust>
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;

        for (size_t i=0;i<num;i++)
        {
          vfloat<N> dist;
          size_t mask = isec1.intersect(&prim[i].qnode,tray,dist); 
#if defined(__AVX__)
          STAT3(normal.trav_hit_boxes[popcnt(mask)],1,1,1);
#endif
          while(mask != 0)
          {
            const size_t ID = bscf(mask); 
            assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));

            if (unlikely(dist[ID] > ray.tfar)) continue;
            intersect(pre,ray,context,prim[i].subgrid(ID));
          }
        }
      }
      template<bool robust>        
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)

      {
        BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;

        for (size_t i=0;i<num;i++)
        {
          vfloat<N> dist;
          size_t mask = isec1.intersect(&prim[i].qnode,tray,dist); 
          while(mask != 0)
          {
            const size_t ID = bscf(mask); 
            assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));

            if (occluded(pre,ray,context,prim[i].subgrid(ID)))
              return true;
          }
        }
        return false;
      }
    
      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context, const Primitive* prim, size_t num, const TravPointQuery<N> &tquery, size_t& lazy_node)
      {
        bool changed = false;
        for (size_t i=0;i<num;i++)
        {
          vfloat<N> dist;
          size_t mask;
          if (likely(context->query_type == POINT_QUERY_TYPE_SPHERE)) {
            mask = BVHNQuantizedBaseNodePointQuerySphere1<N>::pointQuery(&prim[i].qnode,tquery,dist);
          } else {
            mask = BVHNQuantizedBaseNodePointQueryAABB1<N>::pointQuery(&prim[i].qnode,tquery,dist);
          }
          while(mask != 0)
          {
            const size_t ID = bscf(mask); 
            assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));
            changed |= pointQuery(query, context, prim[i].subgrid(ID));
          }
        }
        return changed;
      }
    };

    template<int N, bool filter>
    struct SubGridIntersector1Pluecker
    {
      typedef SubGridQBVHN<N> Primitive;
      typedef SubGridQuadMIntersector1Pluecker<4,filter> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const SubGrid& subgrid)
      {
        STAT3(normal.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        Vec3vf4 v0,v1,v2,v3; subgrid.gather(v0,v1,v2,v3,context->scene);
        pre.intersect(ray,context,v0,v1,v2,v3,g,subgrid);
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const SubGrid& subgrid)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        Vec3vf4 v0,v1,v2,v3; subgrid.gather(v0,v1,v2,v3,context->scene);
        return pre.occluded(ray,context,v0,v1,v2,v3,g,subgrid);
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const SubGrid& subgrid)
      {
        STAT3(point_query.trav_prims,1,1,1);
        AccelSet* accel = (AccelSet*)context->scene->get(subgrid.geomID());
        context->geomID = subgrid.geomID();
        context->primID = subgrid.primID();
        return accel->pointQuery(query, context);
      }

      template<bool robust>
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;

        for (size_t i=0;i<num;i++)
        {
          vfloat<N> dist;
          size_t mask = isec1.intersect(&prim[i].qnode,tray,dist); 
#if defined(__AVX__)
          STAT3(normal.trav_hit_boxes[popcnt(mask)],1,1,1);
#endif
          while(mask != 0)
          {
            const size_t ID = bscf(mask); 
            assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));

            if (unlikely(dist[ID] > ray.tfar)) continue;
            intersect(pre,ray,context,prim[i].subgrid(ID));
          }
        }
      }

      template<bool robust>        
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;

        for (size_t i=0;i<num;i++)
        {
          vfloat<N> dist;
          size_t mask = isec1.intersect(&prim[i].qnode,tray,dist); 
          while(mask != 0)
          {
            const size_t ID = bscf(mask); 
            assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));

            if (occluded(pre,ray,context,prim[i].subgrid(ID)))
              return true;
          }
        }
        return false;
      }
      
      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context, const Primitive* prim, size_t num, const TravPointQuery<N> &tquery, size_t& lazy_node)
      {
        bool changed = false;
        for (size_t i=0;i<num;i++)
        {
          vfloat<N> dist;
          size_t mask;
          if (likely(context->query_type == POINT_QUERY_TYPE_SPHERE)) {
            mask = BVHNQuantizedBaseNodePointQuerySphere1<N>::pointQuery(&prim[i].qnode,tquery,dist);
          } else {
            mask = BVHNQuantizedBaseNodePointQueryAABB1<N>::pointQuery(&prim[i].qnode,tquery,dist);
          }
#if defined(__AVX__)
          STAT3(point_query.trav_hit_boxes[popcnt(mask)],1,1,1);
#endif
          while(mask != 0)
          {
            const size_t ID = bscf(mask); 
            assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));
            changed |= pointQuery(query, context, prim[i].subgrid(ID));
          }
        }
        return changed;
      }
    };

    template<int N, int K, bool filter>
    struct SubGridIntersectorKMoeller
    {
      typedef SubGridQBVHN<N> Primitive;
      typedef SubGridQuadMIntersectorKMoellerTrumbore<4,K,filter> Precalculations;

      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const SubGrid& subgrid)
      {
        Vec3fa vtx[16];
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        subgrid.gather(vtx,context->scene);
        for (unsigned int i=0; i<4; i++)
        {
          const Vec3vf<K> p0 = vtx[i*4+0];
          const Vec3vf<K> p1 = vtx[i*4+1];
          const Vec3vf<K> p2 = vtx[i*4+2];
          const Vec3vf<K> p3 = vtx[i*4+3];
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          pre.intersectK(valid_i,ray,p0,p1,p2,p3,g,subgrid,i,IntersectKEpilogM<4,K,filter>(ray,context,subgrid.geomID(),subgrid.primID(),i));
        }
      }

      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const SubGrid& subgrid)
      {
        vbool<K> valid0 = valid_i;
        Vec3fa vtx[16];
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        subgrid.gather(vtx,context->scene);
        for (unsigned int i=0; i<4; i++)
        {
          const Vec3vf<K> p0 = vtx[i*4+0];
          const Vec3vf<K> p1 = vtx[i*4+1];
          const Vec3vf<K> p2 = vtx[i*4+2];
          const Vec3vf<K> p3 = vtx[i*4+3];
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          if (pre.intersectK(valid0,ray,p0,p1,p2,p3,g,subgrid,i,OccludedKEpilogM<4,K,filter>(valid0,ray,context,subgrid.geomID(),subgrid.primID(),i)))
            break;
        }
        return !valid0;
      }
      
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const SubGrid& subgrid)
      {
        STAT3(normal.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        Vec3vf4 v0,v1,v2,v3; subgrid.gather(v0,v1,v2,v3,context->scene);
        pre.intersect1(ray,k,context,v0,v1,v2,v3,g,subgrid);
      }

      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const SubGrid& subgrid)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());
        Vec3vf4 v0,v1,v2,v3; subgrid.gather(v0,v1,v2,v3,context->scene);
        return pre.occluded1(ray,k,context,v0,v1,v2,v3,g,subgrid);
      }

        template<bool robust>
          static __forceinline void intersect(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersectorK<N,K,robust> isecK;
          for (size_t j=0;j<num;j++)
          {
            size_t m_valid = movemask(prim[j].qnode.validMask());
            vfloat<K> dist;
            while(m_valid)
            {
              const size_t i = bscf(m_valid);
              if (none(valid & isecK.intersectK(&prim[j].qnode,i,tray,dist))) continue;
              intersect(valid,pre,ray,context,prim[j].subgrid(i));
            }
          }
        }

        template<bool robust>        
        static __forceinline vbool<K> occluded(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersectorK<N,K,robust> isecK;
          vbool<K> valid0 = valid;
          for (size_t j=0;j<num;j++)
          {
            size_t m_valid = movemask(prim[j].qnode.validMask());
            vfloat<K> dist;
            while(m_valid)
            {
              const size_t i = bscf(m_valid);
              if (none(valid0 & isecK.intersectK(&prim[j].qnode,i,tray,dist))) continue;
              valid0 &= !occluded(valid0,pre,ray,context,prim[j].subgrid(i));
              if (none(valid0)) break;
            }
          }
          return !valid0;
        }
        
        template<bool robust>        
          static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;

          for (size_t i=0;i<num;i++)
          {
            vfloat<N> dist;
            size_t mask = isec1.intersect(&prim[i].qnode,tray,dist); 
            while(mask != 0)
            {
              const size_t ID = bscf(mask); 
              assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));

              if (unlikely(dist[ID] > ray.tfar[k])) continue;
              intersect(pre,ray,k,context,prim[i].subgrid(ID));
            }
          }
        }
        
        template<bool robust>
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;

          for (size_t i=0;i<num;i++)
          {
            vfloat<N> dist;
            size_t mask = isec1.intersect(&prim[i].qnode,tray,dist); 
            while(mask != 0)
            {
              const size_t ID = bscf(mask); 
              assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));

              if (occluded(pre,ray,k,context,prim[i].subgrid(ID)))
                return true;
            }
          }
          return false;
        }
    };


    template<int N, int K, bool filter>
    struct SubGridIntersectorKPluecker
    {
      typedef SubGridQBVHN<N> Primitive;
      typedef SubGridQuadMIntersectorKPluecker<4,K,filter> Precalculations;

      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const SubGrid& subgrid)
      {
        Vec3fa vtx[16];
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        subgrid.gather(vtx,context->scene);
        for (unsigned int i=0; i<4; i++)
        {
          const Vec3vf<K> p0 = vtx[i*4+0];
          const Vec3vf<K> p1 = vtx[i*4+1];
          const Vec3vf<K> p2 = vtx[i*4+2];
          const Vec3vf<K> p3 = vtx[i*4+3];
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          pre.intersectK(valid_i,ray,p0,p1,p2,p3,g,subgrid,i,IntersectKEpilogM<4,K,filter>(ray,context,subgrid.geomID(),subgrid.primID(),i));
        }
      }

      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const SubGrid& subgrid)
      {
        vbool<K> valid0 = valid_i;
        Vec3fa vtx[16];
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        subgrid.gather(vtx,context->scene);
        for (unsigned int i=0; i<4; i++)
        {
          const Vec3vf<K> p0 = vtx[i*4+0];
          const Vec3vf<K> p1 = vtx[i*4+1];
          const Vec3vf<K> p2 = vtx[i*4+2];
          const Vec3vf<K> p3 = vtx[i*4+3];
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          //if (pre.intersectK(valid0,ray,p0,p1,p2,p3,g,subgrid,i,OccludedKEpilogM<4,K,filter>(valid0,ray,context,subgrid.geomID(),subgrid.primID(),i)))
          if (pre.occludedK(valid0,ray,p0,p1,p2,p3,g,subgrid,i,OccludedKEpilogM<4,K,filter>(valid0,ray,context,subgrid.geomID(),subgrid.primID(),i)))
	    
            break;
        }
        return !valid0;
      }
      
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const SubGrid& subgrid)
      {
        STAT3(normal.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        Vec3vf4 v0,v1,v2,v3; subgrid.gather(v0,v1,v2,v3,context->scene);
        pre.intersect1(ray,k,context,v0,v1,v2,v3,g,subgrid);
      }

      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const SubGrid& subgrid)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());
        Vec3vf4 v0,v1,v2,v3; subgrid.gather(v0,v1,v2,v3,context->scene);
        return pre.occluded1(ray,k,context,v0,v1,v2,v3,g,subgrid);
      }
      
        template<bool robust>
          static __forceinline void intersect(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersectorK<N,K,robust> isecK;
          for (size_t j=0;j<num;j++)
          {
            size_t m_valid = movemask(prim[j].qnode.validMask());
            vfloat<K> dist;
            while(m_valid)
            {
              const size_t i = bscf(m_valid);
              if (none(valid & isecK.intersectK(&prim[j].qnode,i,tray,dist))) continue;
              intersect(valid,pre,ray,context,prim[j].subgrid(i));
            }
          }
        }

        template<bool robust>        
        static __forceinline vbool<K> occluded(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersectorK<N,K,robust> isecK;
          vbool<K> valid0 = valid;
          for (size_t j=0;j<num;j++)
          {
            size_t m_valid = movemask(prim[j].qnode.validMask());
            vfloat<K> dist;
            while(m_valid)
            {
              const size_t i = bscf(m_valid);
              if (none(valid0 & isecK.intersectK(&prim[j].qnode,i,tray,dist))) continue;
              valid0 &= !occluded(valid0,pre,ray,context,prim[j].subgrid(i));
              if (none(valid0)) break;
            }
          }
          return !valid0;
        }
        
        template<bool robust>        
          static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;

          for (size_t i=0;i<num;i++)
          {
            vfloat<N> dist;
            size_t mask = isec1.intersect(&prim[i].qnode,tray,dist); 
            while(mask != 0)
            {
              const size_t ID = bscf(mask); 
              assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));

              if (unlikely(dist[ID] > ray.tfar[k])) continue;
              intersect(pre,ray,k,context,prim[i].subgrid(ID));
            }
          }
        }
        
        template<bool robust>
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;

          for (size_t i=0;i<num;i++)
          {
            vfloat<N> dist;
            size_t mask = isec1.intersect(&prim[i].qnode,tray,dist); 
            while(mask != 0)
            {
              const size_t ID = bscf(mask); 
              assert(((size_t)1 << ID) & movemask(prim[i].qnode.validMask()));

              if (occluded(pre,ray,k,context,prim[i].subgrid(ID)))
                return true;
            }
          }
          return false;
        }
    };
  }
}
