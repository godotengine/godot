// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "instance.h"
#include "../common/ray.h"
#include "../common/point_query.h"

namespace embree
{
  namespace isa
  {
    struct InstanceIntersector1
    {
      typedef InstancePrimitive Primitive;

      struct Precalculations {
        __forceinline Precalculations (const Ray& ray, const void *ptr) {}
      };
      
      static void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& prim);
      static bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& prim);
      static bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& prim);
    };

    struct InstanceIntersector1MB
    {
      typedef InstancePrimitive Primitive;

      struct Precalculations {
        __forceinline Precalculations (const Ray& ray, const void *ptr) {}
      };
      
      static void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& prim);
      static bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& prim);
      static bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& prim);
    };

    template<int K>
      struct InstanceIntersectorK
    {
      typedef InstancePrimitive Primitive;
      
      struct Precalculations {
        __forceinline Precalculations (const vbool<K>& valid, const RayK<K>& ray) {}
      };
      
      static void intersect(const vbool<K>& valid_i, const Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive& prim);
      static vbool<K> occluded(const vbool<K>& valid_i, const Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive& prim);

      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive& prim) {
        intersect(vbool<K>(1<<int(k)),pre,ray,context,prim);
      }
      
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive& prim) {
        occluded(vbool<K>(1<<int(k)),pre,ray,context,prim);
        return ray.tfar[k] < 0.0f; 
      }
    };

    template<int K>
      struct InstanceIntersectorKMB
    {
      typedef InstancePrimitive Primitive;
      
      struct Precalculations {
        __forceinline Precalculations (const vbool<K>& valid, const RayK<K>& ray) {}
      };
      
      static void intersect(const vbool<K>& valid_i, const Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive& prim);
      static vbool<K> occluded(const vbool<K>& valid_i, const Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive& prim);

      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive& prim) {
        intersect(vbool<K>(1<<int(k)),pre,ray,context,prim);
      }
      
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive& prim) {
        occluded(vbool<K>(1<<int(k)),pre,ray,context,prim);
        return ray.tfar[k] < 0.0f; 
      }
    };
  }
}
