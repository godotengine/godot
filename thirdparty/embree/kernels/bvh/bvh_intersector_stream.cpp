// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_intersector_stream.h"

#include "../geometry/intersector_iterators.h"
#include "../geometry/triangle_intersector.h"
#include "../geometry/trianglev_intersector.h"
#include "../geometry/trianglev_mb_intersector.h"
#include "../geometry/trianglei_intersector.h"
#include "../geometry/quadv_intersector.h"
#include "../geometry/quadi_intersector.h"
#include "../geometry/linei_intersector.h"
#include "../geometry/subdivpatch1_intersector.h"
#include "../geometry/object_intersector.h"
#include "../geometry/instance_intersector.h"

#include "../common/scene.h"
#include <bitset>

namespace embree
{
  namespace isa
  {
    __aligned(64) static const int shiftTable[32] = { 
      (int)1 << 0, (int)1 << 1, (int)1 << 2, (int)1 << 3, (int)1 << 4, (int)1 << 5, (int)1 << 6, (int)1 << 7,  
      (int)1 << 8, (int)1 << 9, (int)1 << 10, (int)1 << 11, (int)1 << 12, (int)1 << 13, (int)1 << 14, (int)1 << 15,  
      (int)1 << 16, (int)1 << 17, (int)1 << 18, (int)1 << 19, (int)1 << 20, (int)1 << 21, (int)1 << 22, (int)1 << 23,  
      (int)1 << 24, (int)1 << 25, (int)1 << 26, (int)1 << 27, (int)1 << 28, (int)1 << 29, (int)1 << 30, (int)1 << 31
    };

    template<int N, int types, bool robust, typename PrimitiveIntersector>
    __forceinline void BVHNIntersectorStream<N, types, robust, PrimitiveIntersector>::intersect(Accel::Intersectors* __restrict__ This,
                                                                                                    RayHitN** inputPackets,
                                                                                                    size_t numOctantRays,
                                                                                                    IntersectContext* context)
    {
      /* we may traverse an empty BVH in case all geometry was invalid */
      BVH* __restrict__ bvh = (BVH*) This->ptr;
      if (bvh->root == BVH::emptyNode)
        return;
      
      // Only the coherent code path is implemented
      assert(context->isCoherent());
      intersectCoherent(This, (RayHitK<VSIZEL>**)inputPackets, numOctantRays, context);
    }

    template<int N, int types, bool robust, typename PrimitiveIntersector>
    template<int K>
    __forceinline void BVHNIntersectorStream<N, types, robust, PrimitiveIntersector>::intersectCoherent(Accel::Intersectors* __restrict__ This,
                                                                                                            RayHitK<K>** inputPackets,
                                                                                                            size_t numOctantRays,
                                                                                                            IntersectContext* context)
    {
      assert(context->isCoherent());

      BVH* __restrict__ bvh = (BVH*) This->ptr;
      __aligned(64) StackItemMaskCoherent stack[stackSizeSingle];  // stack of nodes
      assert(numOctantRays <= MAX_INTERNAL_STREAM_SIZE);

      __aligned(64) TravRayKStream<K, robust> packets[MAX_INTERNAL_STREAM_SIZE/K];
      __aligned(64) Frustum<robust> frustum;

      bool commonOctant = true;
      const size_t m_active = initPacketsAndFrustum((RayK<K>**)inputPackets, numOctantRays, packets, frustum, commonOctant);
      if (unlikely(m_active == 0)) return;

      /* case of non-common origin */
      if (unlikely(!commonOctant))
      {
        const size_t numPackets = (numOctantRays+K-1)/K; 
        for (size_t i = 0; i < numPackets; i++)
          This->intersect(inputPackets[i]->tnear() <= inputPackets[i]->tfar, *inputPackets[i], context);
        return;
      }

      stack[0].mask   = m_active;
      stack[0].parent = 0;
      stack[0].child  = bvh->root;

      ///////////////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////

      StackItemMaskCoherent* stackPtr = stack + 1;

      while (1) pop:
      {
        if (unlikely(stackPtr == stack)) break;

        STAT3(normal.trav_stack_pop,1,1,1);
        stackPtr--;
        /*! pop next node */
        NodeRef cur = NodeRef(stackPtr->child);
        size_t m_trav_active = stackPtr->mask;
        assert(m_trav_active);
        NodeRef parent = stackPtr->parent;

        while (1)
        {
          if (unlikely(cur.isLeaf())) break;
          const AABBNode* __restrict__ const node = cur.getAABBNode();
          parent = cur;

          __aligned(64) size_t maskK[N];
          for (size_t i = 0; i < N; i++)
            maskK[i] = m_trav_active;
          vfloat<N> dist;
          const size_t m_node_hit = traverseCoherentStream(m_trav_active, packets, node, frustum, maskK, dist);
          if (unlikely(m_node_hit == 0)) goto pop;

          BVHNNodeTraverserStreamHitCoherent<N, types>::traverseClosestHit(cur, m_trav_active, vbool<N>((int)m_node_hit), dist, (size_t*)maskK, stackPtr);
          assert(m_trav_active);
        }

        /* non-root and leaf => full culling test for all rays */
        if (unlikely(parent != 0 && cur.isLeaf()))
        {
          const AABBNode* __restrict__ const node = parent.getAABBNode();
          size_t boxID = 0xff;
          for (size_t i = 0; i < N; i++)
            if (node->child(i) == cur) { boxID = i; break; }
          assert(boxID < N);
          assert(cur == node->child(boxID));
          m_trav_active = intersectAABBNodePacket(m_trav_active, packets, node, boxID, frustum.nf);
        }

        /*! this is a leaf node */
        assert(cur != BVH::emptyNode);
        STAT3(normal.trav_leaves, 1, 1, 1);
        size_t num; PrimitiveK<K>* prim = (PrimitiveK<K>*)cur.leaf(num);

        size_t bits = m_trav_active;

        /*! intersect stream of rays with all primitives */
        size_t lazy_node = 0;
#if defined(__SSE4_2__)
        STAT_USER(1,(popcnt(bits)+K-1)/K*4);
#endif
        while(bits)
        {
          size_t i = bsf(bits) / K;
          const size_t m_isec = ((((size_t)1 << K)-1) << (i*K));
          assert(m_isec & bits);
          bits &= ~m_isec;

          TravRayKStream<K, robust>& p = packets[i];
          vbool<K> m_valid = p.tnear <= p.tfar;
          PrimitiveIntersectorK<K>::intersectK(m_valid, This, *inputPackets[i], context, prim, num, lazy_node);
          p.tfar = min(p.tfar, inputPackets[i]->tfar);
        };

      } // traversal + intersection
    }

    template<int N, int types, bool robust, typename PrimitiveIntersector>
    __forceinline void BVHNIntersectorStream<N, types, robust, PrimitiveIntersector>::occluded(Accel::Intersectors* __restrict__ This,
                                                                                                   RayN** inputPackets,
                                                                                                   size_t numOctantRays,
                                                                                                   IntersectContext* context)
    {
      /* we may traverse an empty BVH in case all geometry was invalid */
      BVH* __restrict__ bvh = (BVH*) This->ptr;
      if (bvh->root == BVH::emptyNode)
        return;
      
      if (unlikely(context->isCoherent()))
        occludedCoherent(This, (RayK<VSIZEL>**)inputPackets, numOctantRays, context);
      else
        occludedIncoherent(This, (RayK<VSIZEX>**)inputPackets, numOctantRays, context);
    }

    template<int N, int types, bool robust, typename PrimitiveIntersector>
    template<int K>
    __noinline void BVHNIntersectorStream<N, types, robust, PrimitiveIntersector>::occludedCoherent(Accel::Intersectors* __restrict__ This,
                                                                                                        RayK<K>** inputPackets,
                                                                                                        size_t numOctantRays,
                                                                                                        IntersectContext* context)
    {
      assert(context->isCoherent());

      BVH* __restrict__ bvh = (BVH*)This->ptr;
      __aligned(64) StackItemMaskCoherent stack[stackSizeSingle];  // stack of nodes
      assert(numOctantRays <= MAX_INTERNAL_STREAM_SIZE);

      /* inactive rays should have been filtered out before */
      __aligned(64) TravRayKStream<K, robust> packets[MAX_INTERNAL_STREAM_SIZE/K];
      __aligned(64) Frustum<robust> frustum;

      bool commonOctant = true;
      size_t m_active = initPacketsAndFrustum(inputPackets, numOctantRays, packets, frustum, commonOctant);

      /* valid rays */
      if (unlikely(m_active == 0)) return;

      /* case of non-common origin */
      if (unlikely(!commonOctant))
      {
        const size_t numPackets = (numOctantRays+K-1)/K; 
        for (size_t i = 0; i < numPackets; i++)
          This->occluded(inputPackets[i]->tnear() <= inputPackets[i]->tfar, *inputPackets[i], context);
        return;
      }

      stack[0].mask   = m_active;
      stack[0].parent = 0;
      stack[0].child  = bvh->root;

      ///////////////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////

      StackItemMaskCoherent* stackPtr = stack + 1;

      while (1) pop:
      {
        if (unlikely(stackPtr == stack)) break;

        STAT3(normal.trav_stack_pop,1,1,1);
        stackPtr--;
        /*! pop next node */
        NodeRef cur = NodeRef(stackPtr->child);
        size_t m_trav_active = stackPtr->mask & m_active;
        if (unlikely(!m_trav_active)) continue;
        assert(m_trav_active);
        NodeRef parent = stackPtr->parent;

        while (1)
        {
          if (unlikely(cur.isLeaf())) break;
          const AABBNode* __restrict__ const node = cur.getAABBNode();
          parent = cur;

          __aligned(64) size_t maskK[N];
          for (size_t i = 0; i < N; i++)
            maskK[i] = m_trav_active;

          vfloat<N> dist;
          const size_t m_node_hit = traverseCoherentStream(m_trav_active, packets, node, frustum, maskK, dist);
          if (unlikely(m_node_hit == 0)) goto pop;

          BVHNNodeTraverserStreamHitCoherent<N, types>::traverseAnyHit(cur, m_trav_active, vbool<N>((int)m_node_hit), (size_t*)maskK, stackPtr);
          assert(m_trav_active);
        }

        /* non-root and leaf => full culling test for all rays */
        if (unlikely(parent != 0 && cur.isLeaf()))
        {
          const AABBNode* __restrict__ const node = parent.getAABBNode();
          size_t boxID = 0xff;
          for (size_t i = 0; i < N; i++)
            if (node->child(i) == cur) { boxID = i; break; }
          assert(boxID < N);
          assert(cur == node->child(boxID));
          m_trav_active = intersectAABBNodePacket(m_trav_active, packets, node, boxID, frustum.nf);
        }

        /*! this is a leaf node */
        assert(cur != BVH::emptyNode);
        STAT3(normal.trav_leaves, 1, 1, 1);
        size_t num; PrimitiveK<K>* prim = (PrimitiveK<K>*)cur.leaf(num);

        size_t bits = m_trav_active & m_active;
        /*! intersect stream of rays with all primitives */
        size_t lazy_node = 0;
#if defined(__SSE4_2__)
        STAT_USER(1,(popcnt(bits)+K-1)/K*4);
#endif
        while (bits)
        {
          size_t i = bsf(bits) / K;
          const size_t m_isec = ((((size_t)1 << K)-1) << (i*K));
          assert(m_isec & bits);
          bits &= ~m_isec;
          TravRayKStream<K, robust>& p = packets[i];
          vbool<K> m_valid = p.tnear <= p.tfar;
          vbool<K> m_hit = PrimitiveIntersectorK<K>::occludedK(m_valid, This, *inputPackets[i], context, prim, num, lazy_node);
          inputPackets[i]->tfar = select(m_hit & m_valid, vfloat<K>(neg_inf), inputPackets[i]->tfar);
          m_active &= ~((size_t)movemask(m_hit) << (i*K));
        }

      } // traversal + intersection
    }


    template<int N, int types, bool robust, typename PrimitiveIntersector>
    template<int K>
    __forceinline void BVHNIntersectorStream<N, types, robust, PrimitiveIntersector>::occludedIncoherent(Accel::Intersectors* __restrict__ This,
                                                                                                             RayK<K>** inputPackets,
                                                                                                             size_t numOctantRays,
                                                                                                             IntersectContext* context)
    {
      assert(!context->isCoherent());
      assert(types & BVH_FLAG_ALIGNED_NODE);

      __aligned(64) TravRayKStream<K,robust> packet[MAX_INTERNAL_STREAM_SIZE/K];

      assert(numOctantRays <= 32);
      const size_t numPackets = (numOctantRays+K-1)/K;
      size_t m_active = 0;
      for (size_t i = 0; i < numPackets; i++)
      {
        const vfloat<K> tnear = inputPackets[i]->tnear();
        const vfloat<K> tfar  = inputPackets[i]->tfar;
        vbool<K> m_valid = (tnear <= tfar) & (tnear >= 0.0f);
        m_active |= (size_t)movemask(m_valid) << (K*i);
        const Vec3vf<K>& org = inputPackets[i]->org;
        const Vec3vf<K>& dir = inputPackets[i]->dir;
        vfloat<K> packet_min_dist = max(tnear, 0.0f);
        vfloat<K> packet_max_dist = select(m_valid, tfar, neg_inf);
        new (&packet[i]) TravRayKStream<K,robust>(org, dir, packet_min_dist, packet_max_dist);
      }

      BVH* __restrict__ bvh = (BVH*)This->ptr;

      StackItemMaskT<NodeRef> stack[stackSizeSingle]; // stack of nodes
      StackItemMaskT<NodeRef>* stackPtr = stack + 1;  // current stack pointer
      stack[0].ptr = bvh->root;
      stack[0].mask = m_active;

      size_t terminated = ~m_active;

      /* near/far offsets based on first ray */
      const NearFarPrecalculations nf(Vec3fa(packet[0].rdir.x[0], packet[0].rdir.y[0], packet[0].rdir.z[0]), N);

      while (1) pop:
      {
        if (unlikely(stackPtr == stack)) break;
        STAT3(shadow.trav_stack_pop,1,1,1);
        stackPtr--;
        NodeRef cur = NodeRef(stackPtr->ptr);
        size_t cur_mask = stackPtr->mask & (~terminated);
        if (unlikely(cur_mask == 0)) continue;

        while (true)
        {
          /*! stop if we found a leaf node */
          if (unlikely(cur.isLeaf())) break;
          const AABBNode* __restrict__ const node = cur.getAABBNode();

          const vint<N> vmask = traverseIncoherentStream(cur_mask, packet, node, nf, shiftTable);

          size_t mask = movemask(vmask != vint<N>(zero));
          if (unlikely(mask == 0)) goto pop;

          __aligned(64) unsigned int child_mask[N];
          vint<N>::storeu(child_mask, vmask); // this explicit store here causes much better code generation
          
          /*! one child is hit, continue with that child */
          size_t r = bscf(mask);
          assert(r < N);
          cur = node->child(r);         
          BVHN<N>::prefetch(cur,types);
          cur_mask = child_mask[r];

          /* simple in order sequence */
          assert(cur != BVH::emptyNode);
          if (likely(mask == 0)) continue;
          stackPtr->ptr  = cur;
          stackPtr->mask = cur_mask;
          stackPtr++;

          for (; ;)
          {
            r = bscf(mask);
            assert(r < N);

            cur = node->child(r);          
            BVHN<N>::prefetch(cur,types);
            cur_mask = child_mask[r];            
            assert(cur != BVH::emptyNode);
            if (likely(mask == 0)) break;
            stackPtr->ptr  = cur;
            stackPtr->mask = cur_mask;
            stackPtr++;
          }
        }
        
        /*! this is a leaf node */
        assert(cur != BVH::emptyNode);
        STAT3(shadow.trav_leaves,1,1,1);
        size_t num; PrimitiveK<K>* prim = (PrimitiveK<K>*)cur.leaf(num);

        size_t bits = cur_mask;
        size_t lazy_node = 0;

        for (; bits != 0;)
        {
          const size_t rayID = bscf(bits);

          RayK<K> &ray = *inputPackets[rayID / K];
          const size_t k = rayID % K;
          if (PrimitiveIntersectorK<K>::occluded(This, ray, k, context, prim, num, lazy_node))
          {
            ray.tfar[k] = neg_inf;
            terminated |= (size_t)1 << rayID;
          }

          /* lazy node */
          if (unlikely(lazy_node))
          {
            stackPtr->ptr = lazy_node;
            stackPtr->mask = cur_mask;
            stackPtr++;
          }
        }

        if (unlikely(terminated == (size_t)-1)) break;
      }
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// ArrayIntersectorKStream Definitions
    ////////////////////////////////////////////////////////////////////////////////

    template<bool filter>
    struct Triangle4IntersectorStreamMoeller {
      template<int K> using Type = ArrayIntersectorKStream<K,TriangleMIntersectorKMoeller<4 COMMA K COMMA true>>;
    };

    template<bool filter>
    struct Triangle4vIntersectorStreamPluecker {
      template<int K> using Type = ArrayIntersectorKStream<K,TriangleMvIntersectorKPluecker<4 COMMA K COMMA true>>;
    };

    template<bool filter>
    struct Triangle4iIntersectorStreamMoeller {
      template<int K> using Type = ArrayIntersectorKStream<K,TriangleMiIntersectorKMoeller<4 COMMA K COMMA true>>;
    };

    template<bool filter>
    struct Triangle4iIntersectorStreamPluecker {
      template<int K> using Type = ArrayIntersectorKStream<K,TriangleMiIntersectorKPluecker<4 COMMA K COMMA true>>;
    };

    template<bool filter>
    struct Quad4vIntersectorStreamMoeller {
      template<int K> using Type = ArrayIntersectorKStream<K,QuadMvIntersectorKMoeller<4 COMMA K COMMA true>>;
    };

    template<bool filter>
    struct Quad4iIntersectorStreamMoeller {
      template<int K> using Type = ArrayIntersectorKStream<K,QuadMiIntersectorKMoeller<4 COMMA K COMMA true>>;
    };

    template<bool filter>
    struct Quad4vIntersectorStreamPluecker {
      template<int K> using Type = ArrayIntersectorKStream<K,QuadMvIntersectorKPluecker<4 COMMA K COMMA true>>;
    };

    template<bool filter>
    struct Quad4iIntersectorStreamPluecker {
      template<int K> using Type = ArrayIntersectorKStream<K,QuadMiIntersectorKPluecker<4 COMMA K COMMA true>>;
    };

    struct ObjectIntersectorStream {
      template<int K> using Type = ArrayIntersectorKStream<K,ObjectIntersectorK<K COMMA false>>;
    };

    struct InstanceIntersectorStream {
      template<int K> using Type = ArrayIntersectorKStream<K,InstanceIntersectorK<K>>;
    };

    // =====================================================================================================
    // =====================================================================================================
    // =====================================================================================================

    template<int N>
    void BVHNIntersectorStreamPacketFallback<N>::intersect(Accel::Intersectors* __restrict__ This,
                                                               RayHitN** inputRays,
                                                               size_t numTotalRays,
                                                               IntersectContext* context)
    {
      if (unlikely(context->isCoherent()))
        intersectK(This, (RayHitK<VSIZEL>**)inputRays, numTotalRays, context);
      else
        intersectK(This, (RayHitK<VSIZEX>**)inputRays, numTotalRays, context);
    }

    template<int N>
    void BVHNIntersectorStreamPacketFallback<N>::occluded(Accel::Intersectors* __restrict__ This,
                                                              RayN** inputRays,
                                                              size_t numTotalRays,
                                                              IntersectContext* context)
    {
      if (unlikely(context->isCoherent()))
        occludedK(This, (RayK<VSIZEL>**)inputRays, numTotalRays, context);
      else
        occludedK(This, (RayK<VSIZEX>**)inputRays, numTotalRays, context);
    }

    template<int N>
    template<int K>
    __noinline void BVHNIntersectorStreamPacketFallback<N>::intersectK(Accel::Intersectors* __restrict__ This,
                                                                              RayHitK<K>** inputRays,
                                                                              size_t numTotalRays,
                                                                              IntersectContext* context)
    {
      /* fallback to packets */
      for (size_t i = 0; i < numTotalRays; i += K)
      {
        const vint<K> vi = vint<K>(int(i)) + vint<K>(step);
        vbool<K> valid = vi < vint<K>(int(numTotalRays));
        RayHitK<K>& ray = *(inputRays[i / K]);
        valid &= ray.tnear() <= ray.tfar;
        This->intersect(valid, ray, context);
      }
    }

    template<int N>
    template<int K>
    __noinline void BVHNIntersectorStreamPacketFallback<N>::occludedK(Accel::Intersectors* __restrict__ This,
                                                                             RayK<K>** inputRays,
                                                                             size_t numTotalRays,
                                                                             IntersectContext* context)
    {
      /* fallback to packets */
      for (size_t i = 0; i < numTotalRays; i += K)
      {
        const vint<K> vi = vint<K>(int(i)) + vint<K>(step);
        vbool<K> valid = vi < vint<K>(int(numTotalRays));
        RayK<K>& ray = *(inputRays[i / K]);
        valid &= ray.tnear() <= ray.tfar;
        This->occluded(valid, ray, context);
      }
    }
  }
}
