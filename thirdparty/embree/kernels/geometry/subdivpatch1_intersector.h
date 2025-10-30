// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "subdivpatch1.h"
#include "grid_soa.h"
#include "grid_soa_intersector1.h"
#include "grid_soa_intersector_packet.h"
#include "../common/ray.h"

namespace embree
{
  namespace isa
  {
    template<typename T>
      class SubdivPatch1Precalculations : public T
    { 
    public:
      __forceinline SubdivPatch1Precalculations (const Ray& ray, const void* ptr)
        : T(ray,ptr) {}
    };

    template<int K, typename T>
      class SubdivPatch1PrecalculationsK : public T
    { 
    public:
      __forceinline SubdivPatch1PrecalculationsK (const vbool<K>& valid, RayK<K>& ray)
        : T(valid,ray) {}
    };

    class SubdivPatch1Intersector1
    {
    public:
      typedef GridSOA Primitive;
      typedef SubdivPatch1Precalculations<GridSOAIntersector1::Precalculations> Precalculations;

      static __forceinline bool processLazyNode(Precalculations& pre, RayQueryContext* context, const Primitive* prim, size_t& lazy_node)
      {
        lazy_node = prim->root(0);
        pre.grid = (Primitive*)prim;
        return false;
      }

      /*! Intersect a ray with the primitive. */
      template<int N, bool robust>
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node) 
      {
        if (likely(ty == 0)) GridSOAIntersector1::intersect(pre,ray,context,prim,lazy_node);
        else                 processLazyNode(pre,context,prim,lazy_node);
      }

      template<int N, bool robust>
      static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, RayQueryContext* context, size_t ty0, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node) {
        intersect(This,pre,ray,context,prim,ty,tray,lazy_node);
      }
      
      /*! Test if the ray is occluded by the primitive */
      template<int N, bool robust>
      static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) return GridSOAIntersector1::occluded(pre,ray,context,prim,lazy_node);
        else                 return processLazyNode(pre,context,prim,lazy_node);
      }

      template<int N, bool robust>
      static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, RayQueryContext* context, size_t ty0, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node) {
        return occluded(This,pre,ray,context,prim,ty,tray,lazy_node);
      }
      
      template<int N>
        static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context, const Primitive* prim, size_t ty, const TravPointQuery<N> &tquery, size_t& lazy_node) 
      {
          // TODO: PointQuery implement
          assert(false && "not implemented");
          return false;
      }

      template<int N>
      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context, size_t ty0, const Primitive* prim, size_t ty, const TravPointQuery<N> &tquery, size_t& lazy_node) {
        return pointQuery(This,query,context,prim,ty,tquery,lazy_node);
      }
    };

    class SubdivPatch1MBIntersector1
    {
    public:
      typedef SubdivPatch1 Primitive;
      typedef GridSOAMBIntersector1::Precalculations Precalculations;
      
      static __forceinline bool processLazyNode(Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive* prim_i, size_t& lazy_node)
      {
        Primitive* prim = (Primitive*) prim_i;
        GridSOA* grid = nullptr;
        grid = (GridSOA*) prim->root_ref.get();
        pre.itime = getTimeSegment(ray.time(), float(grid->time_steps-1), pre.ftime);
        lazy_node = grid->root(pre.itime);
        pre.grid = grid;
        return false;
      }

      /*! Intersect a ray with the primitive. */
      template<int N, bool robust>
      static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node) 
      {
        if (likely(ty == 0)) GridSOAMBIntersector1::intersect(pre,ray,context,prim,lazy_node);
        else                 processLazyNode(pre,ray,context,prim,lazy_node);
      }

      template<int N, bool robust>
      static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, RayQueryContext* context, size_t ty0, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node) {
        intersect(This,pre,ray,context,prim,ty,tray,lazy_node);
      }
      
      /*! Test if the ray is occluded by the primitive */
      template<int N, bool robust>
      static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) return GridSOAMBIntersector1::occluded(pre,ray,context,prim,lazy_node);
        else                 return processLazyNode(pre,ray,context,prim,lazy_node);
      }

      template<int N, bool robust>
      static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, RayQueryContext* context, size_t ty0, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node) {
        return occluded(This,pre,ray,context,prim,ty,tray,lazy_node);
      }
      
      template<int N>
        static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context, const Primitive* prim, size_t ty, const TravPointQuery<N> &tquery, size_t& lazy_node) 
      {
          // TODO: PointQuery implement
          assert(false && "not implemented");
          return false;
      }

      template<int N, bool robust>
      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context, size_t ty0, const Primitive* prim, size_t ty, const TravPointQuery<N> &tquery, size_t& lazy_node) {
        return pointQuery(This,query,context,prim,ty,tquery,lazy_node);
      }
    };

    template <int K>
      struct SubdivPatch1IntersectorK
    {
      typedef GridSOA Primitive;
      typedef SubdivPatch1PrecalculationsK<K,typename GridSOAIntersectorK<K>::Precalculations> Precalculations;
      
      static __forceinline bool processLazyNode(Precalculations& pre, RayQueryContext* context, const Primitive* prim, size_t& lazy_node)
      {
        lazy_node = prim->root(0);
        pre.grid = (Primitive*)prim;
        return false;
      }
      
      template<bool robust>        
      static __forceinline void intersect(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRayK<K, robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) GridSOAIntersectorK<K>::intersect(valid,pre,ray,context,prim,lazy_node);
        else                 processLazyNode(pre,context,prim,lazy_node);
      }
      
      template<bool robust>        
      static __forceinline vbool<K> occluded(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRayK<K, robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) return GridSOAIntersectorK<K>::occluded(valid,pre,ray,context,prim,lazy_node);
        else                 return processLazyNode(pre,context,prim,lazy_node);
      }
      
      template<int N, bool robust>              
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) GridSOAIntersectorK<K>::intersect(pre,ray,k,context,prim,lazy_node);
        else                 processLazyNode(pre,context,prim,lazy_node);
      }
      
      template<int N, bool robust>              
      static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) return GridSOAIntersectorK<K>::occluded(pre,ray,k,context,prim,lazy_node);
        else                 return processLazyNode(pre,context,prim,lazy_node);
      }
    };

    typedef SubdivPatch1IntersectorK<4>  SubdivPatch1Intersector4;
    typedef SubdivPatch1IntersectorK<8>  SubdivPatch1Intersector8;
    typedef SubdivPatch1IntersectorK<16> SubdivPatch1Intersector16;

    template <int K>
      struct SubdivPatch1MBIntersectorK
    {
      typedef SubdivPatch1 Primitive;
      //typedef GridSOAMBIntersectorK<K>::Precalculations Precalculations;
      typedef SubdivPatch1PrecalculationsK<K,typename GridSOAMBIntersectorK<K>::Precalculations> Precalculations;
      
      static __forceinline bool processLazyNode(Precalculations& pre, RayQueryContext* context, const Primitive* prim_i, size_t& lazy_node)
      {
        Primitive* prim = (Primitive*) prim_i;
        GridSOA* grid = (GridSOA*) prim->root_ref.get();
        lazy_node = grid->troot;
        pre.grid = grid;
        return false;
      }

      template<bool robust>
      static __forceinline void intersect(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRayK<K, robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) GridSOAMBIntersectorK<K>::intersect(valid,pre,ray,context,prim,lazy_node);
        else                 processLazyNode(pre,context,prim,lazy_node);
      }

      template<bool robust>
      static __forceinline vbool<K> occluded(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRayK<K, robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) return GridSOAMBIntersectorK<K>::occluded(valid,pre,ray,context,prim,lazy_node);
        else                 return processLazyNode(pre,context,prim,lazy_node);
      }
      
      template<int N, bool robust>      
      static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) GridSOAMBIntersectorK<K>::intersect(pre,ray,k,context,prim,lazy_node);
        else                 processLazyNode(pre,context,prim,lazy_node);
      }
      
      template<int N, bool robust>      
      static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive* prim, size_t ty, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        if (likely(ty == 0)) return GridSOAMBIntersectorK<K>::occluded(pre,ray,k,context,prim,lazy_node);
        else                 return processLazyNode(pre,context,prim,lazy_node);
      }
    };

    typedef SubdivPatch1MBIntersectorK<4>  SubdivPatch1MBIntersector4;
    typedef SubdivPatch1MBIntersectorK<8>  SubdivPatch1MBIntersector8;
    typedef SubdivPatch1MBIntersectorK<16> SubdivPatch1MBIntersector16;
  }
}
