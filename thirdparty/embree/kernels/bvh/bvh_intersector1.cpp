// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
#include "../geometry/instance_array_intersector.h"
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
                                                                              RayQueryContext* __restrict__ context)
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
      TravRay<N,robust> tray(ray.org, ray.dir, max(ray.tnear(), 0.0f), max(ray.tfar, 0.0f));

      /* initialize the node traverser */
      BVHNNodeTraverser1Hit<N, types> nodeTraverser;

      /* pop loop */
      while (true) pop:
      {
        /* pop next node */
        if (unlikely(stackPtr == stack)) break;
        stackPtr--;
        NodeRef cur = NodeRef(stackPtr->ptr);

        /* if popped node is too far, pop next one */
        if (unlikely(*(float*)&stackPtr->dist > ray.tfar))
          continue;

        /* downtraversal loop */
        while (true)
        {
          /* intersect node */
          size_t mask; vfloat<N> tNear;
          STAT3(normal.trav_nodes,1,1,1);
          bool nodeIntersected = BVHNNodeIntersector1<N, types, robust>::intersect(cur, tray, ray.time(), tNear, mask);
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
                                                                             RayQueryContext* __restrict__ context)
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
      TravRay<N,robust> tray(ray.org, ray.dir, max(ray.tnear(), 0.0f), max(ray.tfar, 0.0f));

      /* initialize the node traverser */
      BVHNNodeTraverser1Hit<N, types> nodeTraverser;

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
          size_t mask; vfloat<N> tNear;
          STAT3(shadow.trav_nodes,1,1,1);
          bool nodeIntersected = BVHNNodeIntersector1<N, types, robust>::intersect(cur, tray, ray.time(), tNear, mask);
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

    template<int N, int types, bool robust, typename PrimitiveIntersector1>
    struct PointQueryDispatch
    {
      typedef typename PrimitiveIntersector1::Precalculations Precalculations;
      typedef typename PrimitiveIntersector1::Primitive Primitive;
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::AABBNode AABBNode;
      typedef typename BVH::AABBNodeMB4D AABBNodeMB4D;

      static const size_t stackSize = 1+(N-1)*BVH::maxDepth+3; // +3 due to 16-wide store

      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context)
      {
        const BVH* __restrict__ bvh = (const BVH*)This->ptr;
        
        /* we may traverse an empty BVH in case all geometry was invalid */
        if (bvh->root == BVH::emptyNode)
          return false;
        
        /* stack state */
        StackItemT<NodeRef> stack[stackSize];    // stack of nodes
        StackItemT<NodeRef>* stackPtr = stack+1; // current stack pointer
        StackItemT<NodeRef>* stackEnd = stack+stackSize;
        stack[0].ptr  = bvh->root;
        stack[0].dist = neg_inf;
        
        /* verify correct input */
        assert(!(types & BVH_MB) || (query->time >= 0.0f && query->time <= 1.0f));

        /* load the point query into SIMD registers */
        TravPointQuery<N> tquery(query->p, context->query_radius);

        /* initialize the node traverser */
        BVHNNodeTraverser1Hit<N,types> nodeTraverser;

        bool changed = false;
        float cull_radius = context->query_type == POINT_QUERY_TYPE_SPHERE
                          ? query->radius * query->radius
                          : dot(context->query_radius, context->query_radius);

        /* pop loop */
        while (true) pop:
        {
          /* pop next node */
          if (unlikely(stackPtr == stack)) break;
          stackPtr--;
          NodeRef cur = NodeRef(stackPtr->ptr);

          /* if popped node is too far, pop next one */
          if (unlikely(*(float*)&stackPtr->dist > cull_radius))
            continue;

          /* downtraversal loop */
          while (true)
          {
            /* intersect node */
            size_t mask; vfloat<N> tNear;
            STAT3(point_query.trav_nodes,1,1,1);
            bool nodeIntersected;
            if (likely(context->query_type == POINT_QUERY_TYPE_SPHERE)) {
              nodeIntersected = BVHNNodePointQuerySphere1<N, types>::pointQuery(cur, tquery, query->time, tNear, mask);
            } else {
              nodeIntersected = BVHNNodePointQueryAABB1  <N, types>::pointQuery(cur, tquery, query->time, tNear, mask);
            }
            if (unlikely(!nodeIntersected)) { STAT3(point_query.trav_nodes,-1,-1,-1); break; }

            /* if no child is hit, pop next node */
            if (unlikely(mask == 0))
              goto pop;

            /* select next child and push other children */
            nodeTraverser.traverseClosestHit(cur, mask, tNear, stackPtr, stackEnd);
          }

          /* this is a leaf node */
          assert(cur != BVH::emptyNode);
          STAT3(point_query.trav_leaves,1,1,1);
          size_t num; Primitive* prim = (Primitive*)cur.leaf(num);
          size_t lazy_node = 0;
          if (PrimitiveIntersector1::pointQuery(This, query, context, prim, num, tquery, lazy_node))
          {
            changed = true;
            tquery.rad = context->query_radius;
            cull_radius = context->query_type == POINT_QUERY_TYPE_SPHERE
                        ? query->radius * query->radius
                        : dot(context->query_radius, context->query_radius);
          }

          /* push lazy node onto stack */
          if (unlikely(lazy_node)) {
            stackPtr->ptr = lazy_node;
            stackPtr->dist = neg_inf;
            stackPtr++;
          }
        }
        return changed;
      }
    };

    /* disable point queries for not yet supported geometry types */
    template<int N, int types, bool robust>
    struct PointQueryDispatch<N, types, robust, VirtualCurveIntersector1> {
      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context) { return false; }
    };
    
    template<int N, int types, bool robust>
    struct PointQueryDispatch<N, types, robust, SubdivPatch1Intersector1> {
      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context) { return false; }
    };
    
    template<int N, int types, bool robust>
    struct PointQueryDispatch<N, types, robust, SubdivPatch1MBIntersector1> {
      static __forceinline bool pointQuery(const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context) { return false; }
    };

    template<int N, int types, bool robust, typename PrimitiveIntersector1>
    bool BVHNIntersector1<N, types, robust, PrimitiveIntersector1>::pointQuery(
      const Accel::Intersectors* This, PointQuery* query, PointQueryContext* context)
    {
      return PointQueryDispatch<N, types, robust, PrimitiveIntersector1>::pointQuery(This, query, context);
    }
  }
}
