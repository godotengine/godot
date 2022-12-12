// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_intersector_hybrid.h"
#include "bvh_traverser1.h"
#include "node_intersector1.h"
#include "node_intersector_packet.h"

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

#define SWITCH_DURING_DOWN_TRAVERSAL 1
#define FORCE_SINGLE_MODE 0

#define ENABLE_FAST_COHERENT_CODEPATHS 1

namespace embree
{
  namespace isa
  {
    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK, bool single>
    void BVHNIntersectorKHybrid<N, K, types, robust, PrimitiveIntersectorK, single>::intersect1(Accel::Intersectors* This,
                                                                                                const BVH* bvh,
                                                                                                NodeRef root,
                                                                                                size_t k,
                                                                                                Precalculations& pre,
                                                                                                RayHitK<K>& ray,
                                                                                                const TravRayK<K, robust>& tray,
                                                                                                IntersectContext* context)
    {
      /* stack state */
      StackItemT<NodeRef> stack[stackSizeSingle];  // stack of nodes
      StackItemT<NodeRef>* stackPtr = stack + 1;   // current stack pointer
      StackItemT<NodeRef>* stackEnd = stack + stackSizeSingle;
      stack[0].ptr = root;
      stack[0].dist = neg_inf;

      /* load the ray into SIMD registers */
      TravRay<N,robust> tray1;
      tray1.template init<K>(k, tray.org, tray.dir, tray.rdir, tray.nearXYZ, tray.tnear[k], tray.tfar[k]);

      /* pop loop */
      while (true) pop:
      {
        /* pop next node */
        if (unlikely(stackPtr == stack)) break;
        stackPtr--;
        NodeRef cur = NodeRef(stackPtr->ptr);

        /* if popped node is too far, pop next one */
        if (unlikely(*(float*)&stackPtr->dist > ray.tfar[k]))
          continue;

        /* downtraversal loop */
        while (true)
        {
          /* intersect node */
          size_t mask; vfloat<N> tNear;
          STAT3(normal.trav_nodes, 1, 1, 1);
          bool nodeIntersected = BVHNNodeIntersector1<N, types, robust>::intersect(cur, tray1, ray.time()[k], tNear, mask);
          if (unlikely(!nodeIntersected)) { STAT3(normal.trav_nodes,-1,-1,-1); break; }

          /* if no child is hit, pop next node */
          if (unlikely(mask == 0))
            goto pop;

          /* select next child and push other children */
          BVHNNodeTraverser1Hit<N, types>::traverseClosestHit(cur, mask, tNear, stackPtr, stackEnd);
        }

        /* this is a leaf node */
        assert(cur != BVH::emptyNode);
        STAT3(normal.trav_leaves, 1, 1, 1);
        size_t num; Primitive* prim = (Primitive*)cur.leaf(num);

        size_t lazy_node = 0;
        PrimitiveIntersectorK::intersect(This, pre, ray, k, context, prim, num, tray1, lazy_node);

        tray1.tfar = ray.tfar[k];

        if (unlikely(lazy_node)) {
          stackPtr->ptr = lazy_node;
          stackPtr->dist = neg_inf;
          stackPtr++;
        }
      }
    }

    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK, bool single>
    void BVHNIntersectorKHybrid<N, K, types, robust, PrimitiveIntersectorK, single>::intersect(vint<K>* __restrict__ valid_i,
                                                                                               Accel::Intersectors* __restrict__ This,
                                                                                               RayHitK<K>& __restrict__ ray,
                                                                                               IntersectContext* __restrict__ context)
    {
      BVH* __restrict__ bvh = (BVH*)This->ptr;
      
      /* we may traverse an empty BVH in case all geometry was invalid */
      if (bvh->root == BVH::emptyNode)
        return;
            
#if ENABLE_FAST_COHERENT_CODEPATHS == 1
      assert(context);
      if (unlikely(types == BVH_AN1 && context->user && context->isCoherent()))
      {
        intersectCoherent(valid_i, This, ray, context);
        return;
      }
#endif

      /* filter out invalid rays */
      vbool<K> valid = *valid_i == -1;
#if defined(EMBREE_IGNORE_INVALID_RAYS)
      valid &= ray.valid();
#endif

      /* return if there are no valid rays */
      size_t valid_bits = movemask(valid);

#if defined(__AVX__)
      STAT3(normal.trav_hit_boxes[popcnt(movemask(valid))], 1, 1, 1);
#endif

      if (unlikely(valid_bits == 0)) return;

      /* verify correct input */
      assert(all(valid, ray.valid()));
      assert(all(valid, ray.tnear() >= 0.0f));
      assert(!(types & BVH_MB) || all(valid, (ray.time() >= 0.0f) & (ray.time() <= 1.0f)));
      Precalculations pre(valid, ray);

      /* load ray */
      TravRayK<K, robust> tray(ray.org, ray.dir, single ? N : 0);
      const vfloat<K> org_ray_tnear = max(ray.tnear(), 0.0f);
      const vfloat<K> org_ray_tfar  = max(ray.tfar , 0.0f);

      if (single)
      {
        tray.tnear = select(valid, org_ray_tnear, vfloat<K>(pos_inf));
        tray.tfar  = select(valid, org_ray_tfar , vfloat<K>(neg_inf));
        
        for (; valid_bits!=0; ) {
          const size_t i = bscf(valid_bits);
          intersect1(This, bvh, bvh->root, i, pre, ray, tray, context);
        }
        return;
      }

       /* determine switch threshold based on flags */
      const size_t switchThreshold = (context->user && context->isCoherent()) ? 2 : switchThresholdIncoherent;

      vint<K> octant = ray.octant();
      octant = select(valid, octant, vint<K>(0xffffffff));

      /* test whether we have ray with opposing direction signs in the packet */
      bool split = false;
      {
        size_t bits = valid_bits;
        vbool<K> vsplit( false );
        do
        {
          const size_t valid_index = bsf(bits);
          vbool<K> octant_valid = octant[valid_index] == octant;
          bits &= ~(size_t)movemask(octant_valid);
          vsplit |= vint<K>(octant[valid_index]) == (octant^vint<K>(0x7));
        } while (bits);
        if (any(vsplit)) split = true;
      }

      do
      {
        const size_t valid_index = bsf(valid_bits);
        const vint<K> diff_octant = vint<K>(octant[valid_index])^octant;
        const vint<K> count_diff_octant = \
          ((diff_octant >> 2) & 1) +
          ((diff_octant >> 1) & 1) +
          ((diff_octant >> 0) & 1);

        vbool<K> octant_valid = (count_diff_octant <= 1) & (octant != vint<K>(0xffffffff));
        if (!single || !split) octant_valid = valid; // deactivate octant sorting in pure chunk mode, otherwise instance traversal performance goes down 


        octant = select(octant_valid,vint<K>(0xffffffff),octant);
        valid_bits &= ~(size_t)movemask(octant_valid);

        tray.tnear = select(octant_valid, org_ray_tnear, vfloat<K>(pos_inf));
        tray.tfar  = select(octant_valid, org_ray_tfar , vfloat<K>(neg_inf));

        /* allocate stack and push root node */
        vfloat<K> stack_near[stackSizeChunk];
        NodeRef stack_node[stackSizeChunk];
        stack_node[0] = BVH::invalidNode;
        stack_near[0] = inf;
        stack_node[1] = bvh->root;
        stack_near[1] = tray.tnear;
        NodeRef* stackEnd MAYBE_UNUSED = stack_node+stackSizeChunk;
        NodeRef* __restrict__ sptr_node = stack_node + 2;
        vfloat<K>* __restrict__ sptr_near = stack_near + 2;

        while (1) pop:
        {
          /* pop next node from stack */
          assert(sptr_node > stack_node);
          sptr_node--;
          sptr_near--;
          NodeRef cur = *sptr_node;
          if (unlikely(cur == BVH::invalidNode)) {
            assert(sptr_node == stack_node);
            break;
          }

          /* cull node if behind closest hit point */
          vfloat<K> curDist = *sptr_near;
          const vbool<K> active = curDist < tray.tfar;
          if (unlikely(none(active)))
            continue;

          /* switch to single ray traversal */
#if (!defined(__WIN32__) || defined(__X86_64__)) && ((defined(__aarch64__)) || defined(__SSE4_2__))
#if FORCE_SINGLE_MODE == 0
          if (single)
#endif
          {
            size_t bits = movemask(active);
#if FORCE_SINGLE_MODE == 0
            if (unlikely(popcnt(bits) <= switchThreshold))
#endif
            {
              for (; bits!=0; ) {
                const size_t i = bscf(bits);
                intersect1(This, bvh, cur, i, pre, ray, tray, context);
              }
              tray.tfar = min(tray.tfar, ray.tfar);
              continue;
            }
          }
#endif
          while (likely(!cur.isLeaf()))
          {
            /* process nodes */
            const vbool<K> valid_node = tray.tfar > curDist;
            STAT3(normal.trav_nodes, 1, popcnt(valid_node), K);
            const NodeRef nodeRef = cur;
            const BaseNode* __restrict__ const node = nodeRef.baseNode();

            /* set cur to invalid */
            cur = BVH::emptyNode;
            curDist = pos_inf;

            size_t num_child_hits = 0;

            for (unsigned i = 0; i < N; i++)
            {
              const NodeRef child = node->children[i];
              if (unlikely(child == BVH::emptyNode)) break;
              vfloat<K> lnearP;
              vbool<K> lhit = valid_node;
              BVHNNodeIntersectorK<N, K, types, robust>::intersect(nodeRef, i, tray, ray.time(), lnearP, lhit);

              /* if we hit the child we choose to continue with that child if it
                 is closer than the current next child, or we push it onto the stack */
              if (likely(any(lhit)))
              {                                
                assert(sptr_node < stackEnd);
                assert(child != BVH::emptyNode);
                const vfloat<K> childDist = select(lhit, lnearP, inf);
                /* push cur node onto stack and continue with hit child */
                if (any(childDist < curDist))
                {
                  if (likely(cur != BVH::emptyNode)) {
                    num_child_hits++;
                    *sptr_node = cur; sptr_node++;
                    *sptr_near = curDist; sptr_near++;
                  }
                  curDist = childDist;
                  cur = child;
                }

                /* push hit child onto stack */
                else {
                  num_child_hits++;                  
                  *sptr_node = child; sptr_node++;
                  *sptr_near = childDist; sptr_near++;
                }
              }
            }

#if defined(__AVX__)
            //STAT3(normal.trav_hit_boxes[num_child_hits], 1, 1, 1);
#endif

            if (unlikely(cur == BVH::emptyNode))
              goto pop;
            
            /* improved distance sorting for 3 or more hits */
            if (unlikely(num_child_hits >= 2))
            {
              if (any(sptr_near[-2] < sptr_near[-1]))
              {
                std::swap(sptr_near[-2],sptr_near[-1]);
                std::swap(sptr_node[-2],sptr_node[-1]);
              }
              if (unlikely(num_child_hits >= 3))
              {
                if (any(sptr_near[-3] < sptr_near[-1]))
                {
                  std::swap(sptr_near[-3],sptr_near[-1]);
                  std::swap(sptr_node[-3],sptr_node[-1]);
                }
                if (any(sptr_near[-3] < sptr_near[-2]))
                {
                  std::swap(sptr_near[-3],sptr_near[-2]);
                  std::swap(sptr_node[-3],sptr_node[-2]);
                }
              }
            }

#if SWITCH_DURING_DOWN_TRAVERSAL == 1
            if (single)
            {
              // seems to be the best place for testing utilization
              if (unlikely(popcnt(tray.tfar > curDist) <= switchThreshold))
              {
                *sptr_node++ = cur;
                *sptr_near++ = curDist;
                goto pop;
              }
            }
#endif
          }

          /* return if stack is empty */
          if (unlikely(cur == BVH::invalidNode)) {
            assert(sptr_node == stack_node);
            break;
          }

          /* intersect leaf */
          assert(cur != BVH::emptyNode);
          const vbool<K> valid_leaf = tray.tfar > curDist;
          STAT3(normal.trav_leaves, 1, popcnt(valid_leaf), K);
          if (unlikely(none(valid_leaf))) continue;
          size_t items; const Primitive* prim = (Primitive*)cur.leaf(items);

          size_t lazy_node = 0;
          PrimitiveIntersectorK::intersect(valid_leaf, This, pre, ray, context, prim, items, tray, lazy_node);
          tray.tfar = select(valid_leaf, ray.tfar, tray.tfar);

          if (unlikely(lazy_node)) {
            *sptr_node = lazy_node; sptr_node++;
            *sptr_near = neg_inf;   sptr_near++;
          }
        }
      } while(valid_bits);
    }


    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK, bool single>
    void BVHNIntersectorKHybrid<N, K, types, robust, PrimitiveIntersectorK, single>::intersectCoherent(vint<K>* __restrict__ valid_i,
                                                                                                       Accel::Intersectors* __restrict__ This,
                                                                                                       RayHitK<K>& __restrict__ ray,
                                                                                                       IntersectContext* context)
    {
      BVH* __restrict__ bvh = (BVH*)This->ptr;
      
      /* filter out invalid rays */
      vbool<K> valid = *valid_i == -1;
#if defined(EMBREE_IGNORE_INVALID_RAYS)
      valid &= ray.valid();
#endif

      /* return if there are no valid rays */
      size_t valid_bits = movemask(valid);
      if (unlikely(valid_bits == 0)) return;

      /* verify correct input */
      assert(all(valid, ray.valid()));
      assert(all(valid, ray.tnear() >= 0.0f));
      assert(!(types & BVH_MB) || all(valid, (ray.time() >= 0.0f) & (ray.time() <= 1.0f)));
      Precalculations pre(valid, ray);

      /* load ray */
      TravRayK<K, robust> tray(ray.org, ray.dir, single ? N : 0);
      const vfloat<K> org_ray_tnear = max(ray.tnear(), 0.0f);
      const vfloat<K> org_ray_tfar  = max(ray.tfar , 0.0f);

      vint<K> octant = ray.octant();
      octant = select(valid, octant, vint<K>(0xffffffff));

      do
      {
        const size_t valid_index = bsf(valid_bits);
        const vbool<K> octant_valid = octant[valid_index] == octant;
        valid_bits &= ~(size_t)movemask(octant_valid);

        tray.tnear = select(octant_valid, org_ray_tnear, vfloat<K>(pos_inf));
        tray.tfar  = select(octant_valid, org_ray_tfar , vfloat<K>(neg_inf));

        Frustum<robust> frustum;
        frustum.template init<K>(octant_valid, tray.org, tray.rdir, tray.tnear, tray.tfar, N);

        StackItemT<NodeRef> stack[stackSizeSingle];  // stack of nodes
        StackItemT<NodeRef>* stackPtr = stack + 1;   // current stack pointer
        stack[0].ptr  = bvh->root;
        stack[0].dist = neg_inf;

        while (1) pop:
        {
          /* pop next node from stack */
          if (unlikely(stackPtr == stack)) break;

          stackPtr--;
          NodeRef cur = NodeRef(stackPtr->ptr);

          /* cull node if behind closest hit point */
          vfloat<K> curDist = *(float*)&stackPtr->dist;
          const vbool<K> active = curDist < tray.tfar;
          if (unlikely(none(active))) continue;

          while (likely(!cur.isLeaf()))
          {
            /* process nodes */
            //STAT3(normal.trav_nodes, 1, popcnt(valid_node), K);
            const NodeRef nodeRef = cur;
            const AABBNode* __restrict__ const node = nodeRef.getAABBNode();

            vfloat<N> fmin;
            size_t m_frustum_node = intersectNodeFrustum<N>(node, frustum, fmin);

            if (unlikely(!m_frustum_node)) goto pop;
            cur = BVH::emptyNode;
            curDist = pos_inf;
            
#if defined(__AVX__)
            //STAT3(normal.trav_hit_boxes[popcnt(m_frustum_node)], 1, 1, 1);
#endif
            size_t num_child_hits = 0;
            do {
              const size_t i = bscf(m_frustum_node);
              vfloat<K> lnearP;
              vbool<K> lhit = false; // motion blur is not supported, so the initial value will be ignored
              STAT3(normal.trav_nodes, 1, 1, 1);
              BVHNNodeIntersectorK<N, K, types, robust>::intersect(nodeRef, i, tray, ray.time(), lnearP, lhit);

              if (likely(any(lhit)))
              {                                
                const vfloat<K> childDist = fmin[i];
                const NodeRef child = node->child(i);
                BVHN<N>::prefetch(child);
                if (any(childDist < curDist))
                {
                  if (likely(cur != BVH::emptyNode)) {
                    num_child_hits++;
                    stackPtr->ptr = cur;
                    *(float*)&stackPtr->dist = toScalar(curDist);
                    stackPtr++;
                  }
                  curDist = childDist;
                  cur = child;
                }
                /* push hit child onto stack */
                else {
                  num_child_hits++;
                  stackPtr->ptr = child;
                  *(float*)&stackPtr->dist = toScalar(childDist);
                  stackPtr++;
                }                
              }
            } while(m_frustum_node);

            if (unlikely(cur == BVH::emptyNode)) goto pop;

            /* improved distance sorting for 3 or more hits */
            if (unlikely(num_child_hits >= 2))
            {
              if (stackPtr[-2].dist < stackPtr[-1].dist)
                std::swap(stackPtr[-2],stackPtr[-1]);
              if (unlikely(num_child_hits >= 3))
              {
                if (stackPtr[-3].dist < stackPtr[-1].dist)
                  std::swap(stackPtr[-3],stackPtr[-1]);
                if (stackPtr[-3].dist < stackPtr[-2].dist)
                  std::swap(stackPtr[-3],stackPtr[-2]);
              }
            }
          }

          /* intersect leaf */
          assert(cur != BVH::invalidNode);
          assert(cur != BVH::emptyNode);
          const vbool<K> valid_leaf = tray.tfar > curDist;
          STAT3(normal.trav_leaves, 1, popcnt(valid_leaf), K);
          if (unlikely(none(valid_leaf))) continue;
          size_t items; const Primitive* prim = (Primitive*)cur.leaf(items);

          size_t lazy_node = 0;
          PrimitiveIntersectorK::intersect(valid_leaf, This, pre, ray, context, prim, items, tray, lazy_node);

          /* reduce max distance interval on successful intersection */
          if (likely(any((ray.tfar < tray.tfar) & valid_leaf)))
          {
            tray.tfar = select(valid_leaf, ray.tfar, tray.tfar);
            frustum.template updateMaxDist<K>(tray.tfar);
          }

          if (unlikely(lazy_node)) {
            stackPtr->ptr = lazy_node;
            stackPtr->dist = neg_inf;
            stackPtr++;
          }
        }
        
      } while(valid_bits);
    }

    // ===================================================================================================================================================================
    // ===================================================================================================================================================================
    // ===================================================================================================================================================================

    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK, bool single>
    bool BVHNIntersectorKHybrid<N, K, types, robust, PrimitiveIntersectorK, single>::occluded1(Accel::Intersectors* This,
                                                                                               const BVH* bvh,
                                                                                               NodeRef root,
                                                                                               size_t k,
                                                                                               Precalculations& pre,
                                                                                               RayK<K>& ray,
                                                                                               const TravRayK<K, robust>& tray,
                                                                                               IntersectContext* context)
      {
        /* stack state */
        NodeRef stack[stackSizeSingle];  // stack of nodes that still need to get traversed
        NodeRef* stackPtr = stack+1;     // current stack pointer
	NodeRef* stackEnd = stack+stackSizeSingle;
        stack[0] = root;

        /* load the ray into SIMD registers */
        TravRay<N,robust> tray1;
        tray1.template init<K>(k, tray.org, tray.dir, tray.rdir, tray.nearXYZ, tray.tnear[k], tray.tfar[k]);

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
            STAT3(shadow.trav_nodes, 1, 1, 1);
            bool nodeIntersected = BVHNNodeIntersector1<N, types, robust>::intersect(cur, tray1, ray.time()[k], tNear, mask);
            if (unlikely(!nodeIntersected)) { STAT3(shadow.trav_nodes,-1,-1,-1); break; }

            /* if no child is hit, pop next node */
            if (unlikely(mask == 0))
              goto pop;

            /* select next child and push other children */
            BVHNNodeTraverser1Hit<N, types>::traverseAnyHit(cur, mask, tNear, stackPtr, stackEnd);
          }

          /* this is a leaf node */
          assert(cur != BVH::emptyNode);
          STAT3(shadow.trav_leaves, 1, 1, 1);
          size_t num; Primitive* prim = (Primitive*)cur.leaf(num);

          size_t lazy_node = 0;
          if (PrimitiveIntersectorK::occluded(This, pre, ray, k, context, prim, num, tray1, lazy_node)) {
	    ray.tfar[k] = neg_inf;
	    return true;
	  }

          if (unlikely(lazy_node)) {
            *stackPtr = lazy_node;
            stackPtr++;
          }
	}
	return false;
      }

    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK, bool single>
    void BVHNIntersectorKHybrid<N, K, types, robust, PrimitiveIntersectorK, single>::occluded(vint<K>* __restrict__ valid_i,
                                                                                              Accel::Intersectors* __restrict__ This,
                                                                                              RayK<K>& __restrict__ ray,
                                                                                              IntersectContext* context)
    {
      BVH* __restrict__ bvh = (BVH*)This->ptr;
      
      /* we may traverse an empty BVH in case all geometry was invalid */
      if (bvh->root == BVH::emptyNode)
        return;
      
#if ENABLE_FAST_COHERENT_CODEPATHS == 1
      assert(context);
      if (unlikely(types == BVH_AN1 && context->user && context->isCoherent()))
      {
        occludedCoherent(valid_i, This, ray, context);
        return;
      }
#endif

      /* filter out already occluded and invalid rays */
      vbool<K> valid = (*valid_i == -1) & (ray.tfar >= 0.0f);
#if defined(EMBREE_IGNORE_INVALID_RAYS)
      valid &= ray.valid();
#endif

      /* return if there are no valid rays */
      const size_t valid_bits = movemask(valid);
      if (unlikely(valid_bits == 0)) return;

      /* verify correct input */
      assert(all(valid, ray.valid()));
      assert(all(valid, ray.tnear() >= 0.0f));
      assert(!(types & BVH_MB) || all(valid, (ray.time() >= 0.0f) & (ray.time() <= 1.0f)));
      Precalculations pre(valid, ray);

      /* load ray */
      TravRayK<K, robust> tray(ray.org, ray.dir, single ? N : 0);
      const vfloat<K> org_ray_tnear = max(ray.tnear(), 0.0f);
      const vfloat<K> org_ray_tfar  = max(ray.tfar , 0.0f);

      tray.tnear = select(valid, org_ray_tnear, vfloat<K>(pos_inf));
      tray.tfar  = select(valid, org_ray_tfar , vfloat<K>(neg_inf));

      vbool<K> terminated = !valid;
      const vfloat<K> inf = vfloat<K>(pos_inf);

      /* determine switch threshold based on flags */
      const size_t switchThreshold = (context->user && context->isCoherent()) ? 2 : switchThresholdIncoherent;

      /* allocate stack and push root node */
      vfloat<K> stack_near[stackSizeChunk];
      NodeRef stack_node[stackSizeChunk];
      stack_node[0] = BVH::invalidNode;
      stack_near[0] = inf;
      stack_node[1] = bvh->root;
      stack_near[1] = tray.tnear;
      NodeRef* stackEnd MAYBE_UNUSED = stack_node+stackSizeChunk;
      NodeRef* __restrict__ sptr_node = stack_node + 2;
      vfloat<K>* __restrict__ sptr_near = stack_near + 2;

      while (1) pop:
      {
        /* pop next node from stack */
        assert(sptr_node > stack_node);
        sptr_node--;
        sptr_near--;
        NodeRef cur = *sptr_node;
        if (unlikely(cur == BVH::invalidNode)) {
          assert(sptr_node == stack_node);
          break;
        }

        /* cull node if behind closest hit point */
        vfloat<K> curDist = *sptr_near;
        const vbool<K> active = curDist < tray.tfar;
        if (unlikely(none(active)))
          continue;

        /* switch to single ray traversal */
#if (!defined(__WIN32__) || defined(__X86_64__)) && ((defined(__aarch64__)) || defined(__SSE4_2__))
#if FORCE_SINGLE_MODE == 0
        if (single)
#endif
        {
          size_t bits = movemask(active);
#if FORCE_SINGLE_MODE == 0
          if (unlikely(popcnt(bits) <= switchThreshold)) 
#endif
          {
            for (; bits!=0; ) {
              const size_t i = bscf(bits);
              if (occluded1(This, bvh, cur, i, pre, ray, tray, context))
                set(terminated, i);
            }
            if (all(terminated)) break;
            tray.tfar = select(terminated, vfloat<K>(neg_inf), tray.tfar);
            continue;
          }
        }
#endif

        while (likely(!cur.isLeaf()))
        {
          /* process nodes */
          const vbool<K> valid_node = tray.tfar > curDist;
          STAT3(shadow.trav_nodes, 1, popcnt(valid_node), K);
          const NodeRef nodeRef = cur;
          const BaseNode* __restrict__ const node = nodeRef.baseNode();

          /* set cur to invalid */
          cur = BVH::emptyNode;
          curDist = pos_inf;

          for (unsigned i = 0; i < N; i++)
          {
            const NodeRef child = node->children[i];
            if (unlikely(child == BVH::emptyNode)) break;
            vfloat<K> lnearP;
            vbool<K> lhit = valid_node;
            BVHNNodeIntersectorK<N, K, types, robust>::intersect(nodeRef, i, tray, ray.time(), lnearP, lhit);

            /* if we hit the child we push the previously hit node onto the stack, and continue with the currently hit child */
            if (likely(any(lhit)))
            {
              assert(sptr_node < stackEnd);
              assert(child != BVH::emptyNode);
              const vfloat<K> childDist = select(lhit, lnearP, inf);

              /* push 'cur' node onto stack and continue with hit child */
              if (likely(cur != BVH::emptyNode)) {
                *sptr_node = cur; sptr_node++;
                *sptr_near = curDist; sptr_near++;
              }
              curDist = childDist;
              cur = child;
            }
          }
          if (unlikely(cur == BVH::emptyNode))
            goto pop;

#if SWITCH_DURING_DOWN_TRAVERSAL == 1
          if (single)
          {
            // seems to be the best place for testing utilization
            if (unlikely(popcnt(tray.tfar > curDist) <= switchThreshold))
            {
              *sptr_node++ = cur;
              *sptr_near++ = curDist;
              goto pop;
            }
          }
#endif
	}

        /* return if stack is empty */
        if (unlikely(cur == BVH::invalidNode)) {
          assert(sptr_node == stack_node);
          break;
        }


        /* intersect leaf */
        assert(cur != BVH::emptyNode);
        const vbool<K> valid_leaf = tray.tfar > curDist;
        STAT3(shadow.trav_leaves, 1, popcnt(valid_leaf), K);
        if (unlikely(none(valid_leaf))) continue;
        size_t items; const Primitive* prim = (Primitive*) cur.leaf(items);

        size_t lazy_node = 0;
        terminated |= PrimitiveIntersectorK::occluded(!terminated, This, pre, ray, context, prim, items, tray, lazy_node);
        if (all(terminated)) break;
        tray.tfar = select(terminated, vfloat<K>(neg_inf), tray.tfar); // ignore node intersections for terminated rays

        if (unlikely(lazy_node)) {
          *sptr_node = lazy_node; sptr_node++;
          *sptr_near = neg_inf;   sptr_near++;
        }
      }

      vfloat<K>::store(valid & terminated, &ray.tfar, neg_inf);
    }


    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK, bool single>
    void BVHNIntersectorKHybrid<N, K, types, robust, PrimitiveIntersectorK, single>::occludedCoherent(vint<K>* __restrict__ valid_i,
                                                                                                      Accel::Intersectors* __restrict__ This,
                                                                                                      RayK<K>& __restrict__ ray,
                                                                                                      IntersectContext* context)
    {
      BVH* __restrict__ bvh = (BVH*)This->ptr;
      
      /* filter out invalid rays */
      vbool<K> valid = *valid_i == -1;
#if defined(EMBREE_IGNORE_INVALID_RAYS)
      valid &= ray.valid();
#endif

      /* return if there are no valid rays */
      size_t valid_bits = movemask(valid);
      if (unlikely(valid_bits == 0)) return;

      /* verify correct input */
      assert(all(valid, ray.valid()));
      assert(all(valid, ray.tnear() >= 0.0f));
      assert(!(types & BVH_MB) || all(valid, (ray.time() >= 0.0f) & (ray.time() <= 1.0f)));
      Precalculations pre(valid,ray);

      /* load ray */
      TravRayK<K, robust> tray(ray.org, ray.dir, single ? N : 0);
      const vfloat<K> org_ray_tnear = max(ray.tnear(), 0.0f);
      const vfloat<K> org_ray_tfar  = max(ray.tfar , 0.0f);

      vbool<K> terminated = !valid;

      vint<K> octant = ray.octant();
      octant = select(valid, octant, vint<K>(0xffffffff));

      do
      {
        const size_t valid_index = bsf(valid_bits);
        vbool<K> octant_valid = octant[valid_index] == octant;
        valid_bits &= ~(size_t)movemask(octant_valid);

        tray.tnear = select(octant_valid, org_ray_tnear, vfloat<K>(pos_inf));
        tray.tfar  = select(octant_valid, org_ray_tfar,  vfloat<K>(neg_inf));

        Frustum<robust> frustum;
        frustum.template init<K>(octant_valid, tray.org, tray.rdir, tray.tnear, tray.tfar, N);

        StackItemMaskT<NodeRef> stack[stackSizeSingle];  // stack of nodes
        StackItemMaskT<NodeRef>* stackPtr = stack + 1;   // current stack pointer
        stack[0].ptr  = bvh->root;
        stack[0].mask = movemask(octant_valid);

        while (1) pop:
        {
          /* pop next node from stack */
          if (unlikely(stackPtr == stack)) break;

          stackPtr--;
          NodeRef cur = NodeRef(stackPtr->ptr);

          /* cull node of active rays have already been terminated */
          size_t m_active = (size_t)stackPtr->mask & (~(size_t)movemask(terminated));

          if (unlikely(m_active == 0)) continue;

          while (likely(!cur.isLeaf()))
          {
            /* process nodes */
            //STAT3(normal.trav_nodes, 1, popcnt(valid_node), K);
            const NodeRef nodeRef = cur;
            const AABBNode* __restrict__ const node = nodeRef.getAABBNode();

            vfloat<N> fmin;
            size_t m_frustum_node = intersectNodeFrustum<N>(node, frustum, fmin);

            if (unlikely(!m_frustum_node)) goto pop;
            cur = BVH::emptyNode;
            m_active = 0;

#if defined(__AVX__)
            //STAT3(normal.trav_hit_boxes[popcnt(m_frustum_node)], 1, 1, 1);
#endif
            size_t num_child_hits = 0;
            do {
              const size_t i = bscf(m_frustum_node);
              vfloat<K> lnearP;
              vbool<K> lhit = false; // motion blur is not supported, so the initial value will be ignored
              STAT3(normal.trav_nodes, 1, 1, 1);
              BVHNNodeIntersectorK<N, K, types, robust>::intersect(nodeRef, i, tray, ray.time(), lnearP, lhit);

              if (likely(any(lhit)))
              {                                
                const NodeRef child = node->child(i);
                assert(child != BVH::emptyNode);
                BVHN<N>::prefetch(child);
                if (likely(cur != BVH::emptyNode)) {
                  num_child_hits++;
                  stackPtr->ptr  = cur;
                  stackPtr->mask = m_active;
                  stackPtr++;
                }
                cur = child;
                m_active = movemask(lhit);
              }
            } while(m_frustum_node);

            if (unlikely(cur == BVH::emptyNode)) goto pop;
          }

          /* intersect leaf */
          assert(cur != BVH::invalidNode);
          assert(cur != BVH::emptyNode);
#if defined(__AVX__)
          STAT3(normal.trav_leaves, 1, popcnt(m_active), K);
#endif
          if (unlikely(!m_active)) continue;
          size_t items; const Primitive* prim = (Primitive*)cur.leaf(items);

          size_t lazy_node = 0;
          terminated |= PrimitiveIntersectorK::occluded(!terminated, This, pre, ray, context, prim, items, tray, lazy_node);
          octant_valid &= !terminated;
          if (unlikely(none(octant_valid))) break;
          tray.tfar = select(terminated, vfloat<K>(neg_inf), tray.tfar); // ignore node intersections for terminated rays

          if (unlikely(lazy_node)) {
            stackPtr->ptr  = lazy_node;
            stackPtr->mask = movemask(octant_valid);
            stackPtr++;
          }
        }
      } while(valid_bits);

      vfloat<K>::store(valid & terminated, &ray.tfar, neg_inf);
    }
  }
}
