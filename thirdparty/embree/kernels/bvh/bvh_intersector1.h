// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"
#include "../common/ray.h"
#include "../common/point_query.h"

namespace embree
{
  namespace isa
  {
    /*! BVH single ray intersector. */
    template<int N, int types, bool robust, typename PrimitiveIntersector1>
    class BVHNIntersector1
    {
      /* shortcuts for frequently used types */
      typedef typename PrimitiveIntersector1::Precalculations Precalculations;
      typedef typename PrimitiveIntersector1::Primitive Primitive;
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::AABBNode AABBNode;
      typedef typename BVH::AABBNodeMB4D AABBNodeMB4D;

      static const size_t stackSize = 1+(N-1)*BVH::maxDepth+3; // +3 due to 16-wide store

    public:
      static void intersect (const Accel::Intersectors* This, RayHit& ray, IntersectContext* context);
      static void occluded  (const Accel::Intersectors* This, Ray& ray, IntersectContext* context);
      static bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context);
    };
  }
}
