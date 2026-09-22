// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"
#include "../common/ray.h"
#include "../common/stack_item.h"
#include "node_intersector_frustum.h"

namespace embree
{
  namespace isa 
  {
    template<int K, bool robust>
    struct TravRayK;

    /*! BVH hybrid packet intersector. Switches between packet and single ray traversal (optional). */
    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK, bool single = true>
    class BVHNIntersectorKHybrid
    {
      /* shortcuts for frequently used types */
      typedef typename PrimitiveIntersectorK::Precalculations Precalculations;
      typedef typename PrimitiveIntersectorK::Primitive Primitive;
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::BaseNode BaseNode;
      typedef typename BVH::AABBNode AABBNode;
      
      static const size_t stackSizeSingle = 1+(N-1)*BVH::maxDepth+3; // +3 due to 16-wide store
      static const size_t stackSizeChunk = 1+(N-1)*BVH::maxDepth;

      static const size_t switchThresholdIncoherent = \
      (K==4)  ? 3 :
      (K==8)  ? ((N==4) ? 5 : 7) :
      (K==16) ? 14 : // 14 seems to work best for KNL due to better ordered chunk traversal
      0;

    private:
      static void intersect1(Accel::Intersectors* This, const BVH* bvh, NodeRef root, size_t k, Precalculations& pre,
                             RayHitK<K>& ray, const TravRayK<K, robust>& tray, RayQueryContext* context);
      static bool occluded1(Accel::Intersectors* This, const BVH* bvh, NodeRef root, size_t k, Precalculations& pre,
                            RayK<K>& ray, const TravRayK<K, robust>& tray, RayQueryContext* context);

    public:
      static void intersect(vint<K>* valid, Accel::Intersectors* This, RayHitK<K>& ray, RayQueryContext* context);
      static void occluded (vint<K>* valid, Accel::Intersectors* This, RayK<K>& ray, RayQueryContext* context);

      static void intersectCoherent(vint<K>* valid, Accel::Intersectors* This, RayHitK<K>& ray, RayQueryContext* context);
      static void occludedCoherent (vint<K>* valid, Accel::Intersectors* This, RayK<K>& ray, RayQueryContext* context);

    };

    /*! BVH packet intersector. */
    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK>
    class BVHNIntersectorKChunk : public BVHNIntersectorKHybrid<N, K, types, robust, PrimitiveIntersectorK, false> {};
  }
}
