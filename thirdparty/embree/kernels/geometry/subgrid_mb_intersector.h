// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "subgrid_intersector.h"

namespace embree
{
  namespace isa
  {
    template<int N, bool filter>
    struct SubGridMBIntersector1Pluecker
    {
      typedef SubGridMBQBVHN<N> Primitive;
      typedef SubGridQuadMIntersector1Pluecker<4,filter> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const SubGrid& subgrid)
      {
        STAT3(normal.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        float ftime;
        const int itime = mesh->timeSegment(ray.time(), ftime);
        Vec3vf4 v0,v1,v2,v3; subgrid.gatherMB(v0,v1,v2,v3,context->scene,itime,ftime);
        pre.intersect(ray,context,v0,v1,v2,v3,g,subgrid);
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const SubGrid& subgrid)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        float ftime;
        const int itime = mesh->timeSegment(ray.time(), ftime);

        Vec3vf4 v0,v1,v2,v3; subgrid.gatherMB(v0,v1,v2,v3,context->scene,itime,ftime);
        return pre.occluded(ray,context,v0,v1,v2,v3,g,subgrid);
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const SubGrid& subgrid)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, subgrid);
      }

      template<bool robust>
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;
        for (size_t i=0;i<num;i++)
        {
          vfloat<N> dist;
          const float time = prim[i].adjustTime(ray.time());

          assert(time <= 1.0f);
          size_t mask = isec1.intersect(&prim[i].qnode,tray,time,dist); 
#if defined(__AVX__)
          STAT3(normal.trav_hit_boxes[popcnt(mask)],1,1,1);
#endif
          while(mask != 0)
          {
            const size_t ID = bscf(mask); 
            if (unlikely(dist[ID] > ray.tfar)) continue;
            intersect(pre,ray,context,prim[i].subgrid(ID));
          }
        }
      }

      template<bool robust>        
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;
        for (size_t i=0;i<num;i++)
        {
          const float time = prim[i].adjustTime(ray.time());
          assert(time <= 1.0f);
          vfloat<N> dist;
          size_t mask = isec1.intersect(&prim[i].qnode,tray,time,dist); 
          while(mask != 0)
          {
            const size_t ID = bscf(mask); 
            if (occluded(pre,ray,context,prim[i].subgrid(ID)))
              return true;
          }
        }
        return false;
      }
      
      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context, const Primitive* prim, size_t num, const TravPointQuery<N> &tquery, size_t& lazy_node)
      {
        assert(false && "not implemented");
        return false;
      }
    };


    template<int N, int K, bool filter>
    struct SubGridMBIntersectorKPluecker
    {
      typedef SubGridMBQBVHN<N> Primitive;
      typedef SubGridQuadMIntersectorKPluecker<4,K,filter> Precalculations;

      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const SubGrid& subgrid)
      {
        size_t m_valid = movemask(valid_i);
        while(m_valid)
        {
          size_t ID = bscf(m_valid);
          intersect(pre,ray,ID,context,subgrid);
        }
      }

      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const SubGrid& subgrid)
      {
        vbool<K> valid0 = valid_i;
        size_t m_valid = movemask(valid_i);
        while(m_valid)
        {
          size_t ID = bscf(m_valid);
          if (occluded(pre,ray,ID,context,subgrid))
            clear(valid0,ID);
        }
        return !valid0;
      }
      
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const SubGrid& subgrid)
      {
        STAT3(normal.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());
 
        vfloat<K> ftime;
        const vint<K> itime = mesh->timeSegment<K>(ray.time(), ftime);
        Vec3vf4 v0,v1,v2,v3; subgrid.gatherMB(v0,v1,v2,v3,context->scene,itime[k],ftime[k]);
        pre.intersect1(ray,k,context,v0,v1,v2,v3,g,subgrid);
      }

      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const SubGrid& subgrid)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const GridMesh* mesh    = context->scene->get<GridMesh>(subgrid.geomID());
        const GridMesh::Grid &g = mesh->grid(subgrid.primID());

        vfloat<K> ftime;
        const vint<K> itime = mesh->timeSegment<K>(ray.time(), ftime);
        Vec3vf4 v0,v1,v2,v3; subgrid.gatherMB(v0,v1,v2,v3,context->scene,itime[k],ftime[k]);
        return pre.occluded1(ray,k,context,v0,v1,v2,v3,g,subgrid);
      }

        template<bool robust>
          static __forceinline void intersect(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersectorK<N,K,robust> isecK;
          for (size_t j=0;j<num;j++)
          {
            size_t m_valid = movemask(prim[j].qnode.validMask());
            const vfloat<K> time = prim[j].template adjustTime<K>(ray.time());

            vfloat<K> dist;
            while(m_valid)
            {
              const size_t i = bscf(m_valid);
              if (none(valid & isecK.intersectK(&prim[j].qnode,i,tray,time,dist))) continue;
              intersect(valid,pre,ray,context,prim[j].subgrid(i));
            }
          }
        }

        template<bool robust>        
        static __forceinline vbool<K> occluded(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersectorK<N,K,robust> isecK;

          vbool<K> valid0 = valid;
          for (size_t j=0;j<num;j++)
          {
            size_t m_valid = movemask(prim[j].qnode.validMask());
            const vfloat<K> time = prim[j].template adjustTime<K>(ray.time());
            vfloat<K> dist;
            while(m_valid)
            {
              const size_t i = bscf(m_valid);
              if (none(valid0 & isecK.intersectK(&prim[j].qnode,i,tray,time,dist))) continue;
              valid0 &= !occluded(valid0,pre,ray,context,prim[j].subgrid(i));
              if (none(valid0)) break;
            }
          }
          return !valid0;
        }
        
        template<bool robust>        
          static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;
          for (size_t i=0;i<num;i++)
          {
            vfloat<N> dist;
            const float time = prim[i].adjustTime(ray.time()[k]);
            assert(time <= 1.0f);

            size_t mask = isec1.intersect(&prim[i].qnode,tray,time,dist); 
            while(mask != 0)
            {
              const size_t ID = bscf(mask); 
              if (unlikely(dist[ID] > ray.tfar[k])) continue;
              intersect(pre,ray,k,context,prim[i].subgrid(ID));
            }
          }
        }
        
        template<bool robust>
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
        {
          BVHNQuantizedBaseNodeIntersector1<N,robust> isec1;
          
          for (size_t i=0;i<num;i++)
          {
            vfloat<N> dist;
            const float time = prim[i].adjustTime(ray.time()[k]);
            assert(time <= 1.0f);

            size_t mask = isec1.intersect(&prim[i].qnode,tray,time,dist); 
            while(mask != 0)
            {
              const size_t ID = bscf(mask); 
              if (occluded(pre,ray,k,context,prim[i].subgrid(ID)))
                return true;
            }
          }
          return false;
        }
    };
  }
}
