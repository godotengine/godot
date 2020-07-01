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

#include "instance.h"
#include "../common/ray.h"

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
      
      static void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& prim);
      static bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& prim);
    };

    struct InstanceIntersector1MB
    {
      typedef InstancePrimitive Primitive;

      struct Precalculations {
        __forceinline Precalculations (const Ray& ray, const void *ptr) {}
      };
      
      static void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& prim);
      static bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& prim);
    };

    template<int K>
      struct InstanceIntersectorK
    {
      typedef InstancePrimitive Primitive;
      
      struct Precalculations {
        __forceinline Precalculations (const vbool<K>& valid, const RayK<K>& ray) {}
      };
      
      static void intersect(const vbool<K>& valid_i, const Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive& prim);
      static vbool<K> occluded(const vbool<K>& valid_i, const Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive& prim);

      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& prim) {
        intersect(vbool<K>(1<<int(k)),pre,ray,context,prim);
      }
      
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& prim) {
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
      
      static void intersect(const vbool<K>& valid_i, const Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive& prim);
      static vbool<K> occluded(const vbool<K>& valid_i, const Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive& prim);

      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& prim) {
        intersect(vbool<K>(1<<int(k)),pre,ray,context,prim);
      }
      
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& prim) {
        occluded(vbool<K>(1<<int(k)),pre,ray,context,prim);
        return ray.tfar[k] < 0.0f; 
      }
    };
  }
}
