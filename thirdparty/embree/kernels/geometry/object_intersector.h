// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "object.h"
#include "../common/ray.h"

namespace embree
{
  namespace isa
  {
    template<bool mblur>
    struct ObjectIntersector1
    {
      typedef Object Primitive;
     
      static const bool validIntersectorK = false;

      struct Precalculations {
        __forceinline Precalculations() {}
        __forceinline Precalculations (const Ray& ray, const void *ptr) {}
      };
      
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& prim) 
      {
        AccelSet* accel = (AccelSet*) context->scene->get(prim.geomID());

        /* perform ray mask test */
#if defined(EMBREE_RAY_MASK)
        if ((ray.mask & accel->mask) == 0) 
          return;
#endif

        accel->intersect(ray,prim.geomID(),prim.primID(),context);
      }
      
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& prim)
      {
        AccelSet* accel = (AccelSet*) context->scene->get(prim.geomID());
        /* perform ray mask test */
#if defined(EMBREE_RAY_MASK)
        if ((ray.mask & accel->mask) == 0) 
          return false;
#endif

        accel->occluded(ray,prim.geomID(),prim.primID(),context);
        return ray.tfar < 0.0f;
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& prim)
      {
        AccelSet* accel = (AccelSet*)context->scene->get(prim.geomID());
        context->geomID = prim.geomID();
        context->primID = prim.primID();
        return accel->pointQuery(query, context);
      }
      
      template<int K>
      static __forceinline void intersectK(const vbool<K>& valid, /* PrecalculationsK& pre, */ RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, size_t& lazy_node)
      {
        assert(false);
      }

      template<int K>
      static __forceinline vbool<K> occludedK(const vbool<K>& valid, /* PrecalculationsK& pre, */ RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, size_t& lazy_node)
      {
        assert(false);
        return valid;
      }
    };

    template<int K, bool mblur>
    struct ObjectIntersectorK
    {
      typedef Object Primitive;
      
      struct Precalculations {
        __forceinline Precalculations (const vbool<K>& valid, const RayK<K>& ray) {}
      };
      
      static __forceinline void intersect(const vbool<K>& valid_i, const Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive& prim)
      {
        vbool<K> valid = valid_i;
        AccelSet* accel = (AccelSet*) context->scene->get(prim.geomID());
        
        /* perform ray mask test */
#if defined(EMBREE_RAY_MASK)
        valid &= (ray.mask & accel->mask) != 0;
        if (none(valid)) return;
#endif
        accel->intersect(valid,ray,prim.geomID(),prim.primID(),context);
      }

      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, const Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive& prim)
      {
        vbool<K> valid = valid_i;
        AccelSet* accel = (AccelSet*) context->scene->get(prim.geomID());
        
        /* perform ray mask test */
#if defined(EMBREE_RAY_MASK)
        valid &= (ray.mask & accel->mask) != 0;
        if (none(valid)) return false;
#endif
        accel->occluded(valid,ray,prim.geomID(),prim.primID(),context);
        return ray.tfar < 0.0f;
      }
      
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& prim) {
        intersect(vbool<K>(1<<int(k)),pre,ray,context,prim);
      }
      
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& prim) {
        occluded(vbool<K>(1<<int(k)),pre,ray,context,prim);
        return ray.tfar[k] < 0.0f; 
      }
    };

    typedef ObjectIntersectorK<4,false>  ObjectIntersector4;
    typedef ObjectIntersectorK<8,false>  ObjectIntersector8;
    typedef ObjectIntersectorK<16,false> ObjectIntersector16;

    typedef ObjectIntersectorK<4,true>  ObjectIntersector4MB;
    typedef ObjectIntersectorK<8,true>  ObjectIntersector8MB;
    typedef ObjectIntersectorK<16,true> ObjectIntersector16MB;
  }
}
