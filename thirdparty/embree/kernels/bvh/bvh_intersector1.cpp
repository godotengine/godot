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

#include "bvh_intersector1.h"
#include "node_intersector1.h"
#include "bvh_traverser1.h"

#include "../geometry/intersector_iterators.h"
#include "../geometry/triangle_intersector.h"
#include "../geometry/trianglev_intersector.h"
#include "../geometry/trianglev_mb_intersector.h"
#include "../geometry/trianglei_intersector.h"
#include "../geometry/quadv_intersector.h"
#include "../geometry/quadi_intersector.h"
#include "../geometry/curveNv_intersector.h"
#include "../geometry/curveNi_intersector.h"
#include "../geometry/curveNi_mb_intersector.h"
#include "../geometry/linei_intersector.h"
#include "../geometry/subdivpatch1_intersector.h"
#include "../geometry/object_intersector.h"
#include "../geometry/instance_intersector.h"
#include "../geometry/subgrid_intersector.h"
#include "../geometry/subgrid_mb_intersector.h"
#include "../geometry/curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    template<int N, int types, bool robust, typename PrimitiveIntersector1>
    void BVHNIntersector1<N, types, robust, PrimitiveIntersector1>::intersect(const Accel::Intersectors* __restrict__ This,
                                                                              RayHit& __restrict__ ray,
                                                                              IntersectContext* __restrict__ context)
    {
      const BVH* __restrict__ bvh = (const BVH*)This->ptr;
      
      /* we may traverse an empty BVH in case all geometry was invalid */
      if (bvh->root == BVH::emptyNode)
        return;
      
      /* perform per ray precalculations required by the primitive intersector */
      Precalculations pre(ray, bvh);

      /* stack state */
      StackItemT<NodeRef> stack[stackSize];    // stack of nodes
      StackItemT<NodeRef>* stackPtr = stack+1; // current stack pointer
      StackItemT<NodeRef>* stackEnd = stack+stackSize;
      stack[0].ptr  = bvh->root;
      stack[0].dist = neg_inf;
      
      if (bvh->root == BVH::emptyNode)
        return;
      
      /* filter out invalid rays */
#if defined(EMBREE_IGNORE_INVALID_RAYS)
      if (!ray.valid()) return;
#endif
      /* verify correct input */
      assert(ray.valid());
      assert(ray.tnear() >= 0.0f);
      assert(!(types & BVH_MB) || (ray.time() >= 0.0f && ray.time() <= 1.0f));

      /* load the ray into SIMD registers */
      TravRay<N,Nx,robust> tray(ray.org, ray.dir, max(ray.tnear(), 0.0f), max(ray.tfar, 0.0f));

      /* initialize the node traverser */
      BVHNNodeTraverser1Hit<N, Nx, types> nodeTraverser;

      /* pop loop */
      while (true) pop:
      {
        /* pop next node */
        if (unlikely(stackPtr == stack)) break;
        stackPtr--;
        NodeRef cur = NodeRef(stackPtr->ptr);

        /* if popped node is too far, pop next one */
#if defined(__AVX512ER__)
        /* much faster on KNL */
        if (unlikely(any(vfloat<Nx>(*(float*)&stackPtr->dist) > tray.tfar)))
          continue;
#else
        if (unlikely(*(float*)&stackPtr->dist > ray.tfar))
          continue;
#endif

        /* downtraversal loop */
        while (true)
        {
          /* intersect node */
          size_t mask; vfloat<Nx> tNear;
          STAT3(normal.trav_nodes,1,1,1);
          bool nodeIntersected = BVHNNodeIntersector1<N, Nx, types, robust>::intersect(cur, tray, ray.time(), tNear, mask);
          if (unlikely(!nodeIntersected)) { STAT3(normal.trav_nodes,-1,-1,-1); break; }

          /* if no child is hit, pop next node */
          if (unlikely(mask == 0))
            goto pop;

          /* select next child and push other children */
          nodeTraverser.traverseClosestHit(cur, mask, tNear, stackPtr, stackEnd);
        }

        /* this is a leaf node */
        assert(cur != BVH::emptyNode);
        STAT3(normal.trav_leaves,1,1,1);
        size_t num; Primitive* prim = (Primitive*)cur.leaf(num);
        size_t lazy_node = 0;
        PrimitiveIntersector1::intersect(This, pre, ray, context, prim, num, tray, lazy_node);
        tray.tfar = ray.tfar;

        /* push lazy node onto stack */
        if (unlikely(lazy_node)) {
          stackPtr->ptr = lazy_node;
          stackPtr->dist = neg_inf;
          stackPtr++;
        }
      }
    }

    template<int N, int types, bool robust, typename PrimitiveIntersector1>
    void BVHNIntersector1<N, types, robust, PrimitiveIntersector1>::occluded(const Accel::Intersectors* __restrict__ This,
                                                                             Ray& __restrict__ ray,
                                                                             IntersectContext* __restrict__ context)
    {
      const BVH* __restrict__ bvh = (const BVH*)This->ptr;
      
      /* we may traverse an empty BVH in case all geometry was invalid */
      if (bvh->root == BVH::emptyNode)
        return;
       
      /* early out for already occluded rays */
      if (unlikely(ray.tfar < 0.0f))
        return;

      /* perform per ray precalculations required by the primitive intersector */
      Precalculations pre(ray, bvh);

      /* stack state */
      NodeRef stack[stackSize];    // stack of nodes that still need to get traversed
      NodeRef* stackPtr = stack+1; // current stack pointer
      NodeRef* stackEnd = stack+stackSize;
      stack[0] = bvh->root;

      /* filter out invalid rays */
#if defined(EMBREE_IGNORE_INVALID_RAYS)
      if (!ray.valid()) return;
#endif

      /* verify correct input */
      assert(ray.valid());
      assert(ray.tnear() >= 0.0f);
      assert(!(types & BVH_MB) || (ray.time() >= 0.0f && ray.time() <= 1.0f));

      /* load the ray into SIMD registers */
      TravRay<N,Nx,robust> tray(ray.org, ray.dir, max(ray.tnear(), 0.0f), max(ray.tfar, 0.0f));

      /* initialize the node traverser */
      BVHNNodeTraverser1Hit<N, Nx, types> nodeTraverser;

      /* pop loop */
      while (true) pop:
      {
        /* pop next node */
        if (unlikely(stackPtr == stack)) break;
        stackPtr--;
        NodeRef cur = (NodeRef)*stackPtr;

        /* downtraversal loop */
        while (true)
        {
          /* intersect node */
          size_t mask; vfloat<Nx> tNear;
          STAT3(shadow.trav_nodes,1,1,1);
          bool nodeIntersected = BVHNNodeIntersector1<N, Nx, types, robust>::intersect(cur, tray, ray.time(), tNear, mask);
          if (unlikely(!nodeIntersected)) { STAT3(shadow.trav_nodes,-1,-1,-1); break; }

          /* if no child is hit, pop next node */
          if (unlikely(mask == 0))
            goto pop;

          /* select next child and push other children */
          nodeTraverser.traverseAnyHit(cur, mask, tNear, stackPtr, stackEnd);
        }

        /* this is a leaf node */
        assert(cur != BVH::emptyNode);
        STAT3(shadow.trav_leaves,1,1,1);
        size_t num; Primitive* prim = (Primitive*)cur.leaf(num);
        size_t lazy_node = 0;
        if (PrimitiveIntersector1::occluded(This, pre, ray, context, prim, num, tray, lazy_node)) {
          ray.tfar = neg_inf;
          break;
        }

        /* push lazy node onto stack */
        if (unlikely(lazy_node)) {
          *stackPtr = (NodeRef)lazy_node;
          stackPtr++;
        }
      }
    }
  }
}
