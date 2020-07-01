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

#include "../common/scene.h"
#include "../common/ray.h"
#include "../bvh/node_intersector1.h"
#include "../bvh/node_intersector_packet.h"

namespace embree
{
  namespace isa
  {
    template<typename Intersector>
    struct ArrayIntersector1
    {
      typedef typename Intersector::Primitive Primitive;
      typedef typename Intersector::Precalculations Precalculations;

      template<int N, int Nx, bool robust>
      static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,Nx,robust> &tray, size_t& lazy_node)
      {
        for (size_t i=0; i<num; i++)
          Intersector::intersect(pre,ray,context,prim[i]);
      }

      template<int N, int Nx, bool robust>
      static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,Nx,robust> &tray, size_t& lazy_node)
      {
        for (size_t i=0; i<num; i++) {
          if (Intersector::occluded(pre,ray,context,prim[i]))
            return true;
        }
        return false;
      }

      template<int K>
      static __forceinline void intersectK(const vbool<K>& valid, /* PrecalculationsK& pre, */ RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, size_t& lazy_node)
      {
      }

      template<int K>
      static __forceinline vbool<K> occludedK(const vbool<K>& valid, /* PrecalculationsK& pre, */ RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, size_t& lazy_node)
      {
        return valid;
      }
    };

    template<int K, typename Intersector>
    struct ArrayIntersectorK_1
    {
      typedef typename Intersector::Primitive Primitive;
      typedef typename Intersector::Precalculations Precalculations;

      template<bool robust>
      static __forceinline void intersect(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
      {
        for (size_t i=0; i<num; i++) {
          Intersector::intersect(valid,pre,ray,context,prim[i]);
        }
      }

      template<bool robust>
      static __forceinline vbool<K> occluded(const vbool<K>& valid, const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
      {
        vbool<K> valid0 = valid;
        for (size_t i=0; i<num; i++) {
          valid0 &= !Intersector::occluded(valid0,pre,ray,context,prim[i]);
          if (none(valid0)) break;
        }
        return !valid0;
      }

      template<int N, int Nx, bool robust>
      static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,Nx,robust> &tray, size_t& lazy_node)
      {
        for (size_t i=0; i<num; i++) {
          Intersector::intersect(pre,ray,k,context,prim[i]);
        }
      }

      template<int N, int Nx, bool robust>
      static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,Nx,robust> &tray, size_t& lazy_node)
      {
        for (size_t i=0; i<num; i++) {
          if (Intersector::occluded(pre,ray,k,context,prim[i]))
            return true;
        }
        return false;
      }
    };

    // =============================================================================================

    template<int K, typename IntersectorK>
    struct ArrayIntersectorKStream
    {
      typedef typename IntersectorK::Primitive PrimitiveK;
      typedef typename IntersectorK::Precalculations PrecalculationsK;

      static __forceinline void intersectK(const vbool<K>& valid, const Accel::Intersectors* This, /* PrecalculationsK& pre, */ RayHitK<K>& ray, IntersectContext* context, const PrimitiveK* prim, size_t num, size_t& lazy_node)
      {
        PrecalculationsK pre(valid,ray); // FIXME: might cause trouble

        for (size_t i=0; i<num; i++) {
          IntersectorK::intersect(valid,pre,ray,context,prim[i]);
        }
      }

      static __forceinline vbool<K> occludedK(const vbool<K>& valid, const Accel::Intersectors* This, /* PrecalculationsK& pre, */ RayK<K>& ray, IntersectContext* context, const PrimitiveK* prim, size_t num, size_t& lazy_node)
      {
        PrecalculationsK pre(valid,ray); // FIXME: might cause trouble
        vbool<K> valid0 = valid;
        for (size_t i=0; i<num; i++) {
          valid0 &= !IntersectorK::occluded(valid0,pre,ray,context,prim[i]);
          if (none(valid0)) break;
        }
        return !valid0;
      }

      static __forceinline void intersect(const Accel::Intersectors* This, RayHitK<K>& ray, size_t k, IntersectContext* context, const PrimitiveK* prim, size_t num, size_t& lazy_node)
      {
        PrecalculationsK pre(ray.tnear() <= ray.tfar,ray); // FIXME: might cause trouble
        for (size_t i=0; i<num; i++) {
          IntersectorK::intersect(pre,ray,k,context,prim[i]);
        }
      }

      static __forceinline bool occluded(const Accel::Intersectors* This, RayK<K>& ray, size_t k, IntersectContext* context, const PrimitiveK* prim, size_t num, size_t& lazy_node)
      {
        PrecalculationsK pre(ray.tnear() <= ray.tfar,ray); // FIXME: might cause trouble
        for (size_t i=0; i<num; i++) {
          if (IntersectorK::occluded(pre,ray,k,context,prim[i]))
            return true;
        }
        return false;
      }

      static __forceinline size_t occluded(const Accel::Intersectors* This, size_t cur_mask, RayK<K>** __restrict__ inputPackets, IntersectContext* context, const PrimitiveK* prim, size_t num, size_t& lazy_node)
      {
        size_t m_occluded = 0;
        for (size_t i=0; i<num; i++) {
          size_t bits = cur_mask & (~m_occluded);
          for (; bits!=0; )
          {
            const size_t rayID = bscf(bits);
            RayHitK<K> &ray = *inputPackets[rayID / K];
            const size_t k = rayID % K;
            PrecalculationsK pre(ray.tnear() <= ray.tfar,ray); // FIXME: might cause trouble
            if (IntersectorK::occluded(pre,ray,k,context,prim[i]))
            {
              m_occluded |= (size_t)1 << rayID;
              ray.tfar[k] = neg_inf;
            }
          }
        }
        return m_occluded;
      }
    };
  }
}
