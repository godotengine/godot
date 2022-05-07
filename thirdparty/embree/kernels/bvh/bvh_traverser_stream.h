// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"
#include "../common/ray.h"
#include "../common/stack_item.h"

namespace embree
{
  namespace isa
  {
    template<int N, int types>
    class BVHNNodeTraverserStreamHitCoherent
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::BaseNode BaseNode;

    public:
      template<class T>
      static __forceinline void traverseClosestHit(NodeRef& cur,
                                                   size_t& m_trav_active,
                                                   const vbool<N>& vmask,
                                                   const vfloat<N>& tNear,
                                                   const T* const tMask,
                                                   StackItemMaskCoherent*& stackPtr)
      {
        const NodeRef parent = cur;
        size_t mask = movemask(vmask);
        assert(mask != 0);
        const BaseNode* node = cur.baseNode();

        /*! one child is hit, continue with that child */
        const size_t r0 = bscf(mask);
        assert(r0 < 8);
        cur = node->child(r0);
        BVHN<N>::prefetch(cur,types);
        m_trav_active = tMask[r0];
        assert(cur != BVH::emptyNode);
        if (unlikely(mask == 0)) return;

        const unsigned int* const tNear_i = (unsigned int*)&tNear;

        /*! two children are hit, push far child, and continue with closer child */
        NodeRef c0 = cur;
        unsigned int d0 = tNear_i[r0];
        const size_t r1 = bscf(mask);
        assert(r1 < 8);
        NodeRef c1 = node->child(r1);
        BVHN<N>::prefetch(c1,types);
        unsigned int d1 = tNear_i[r1];

        assert(c0 != BVH::emptyNode);
        assert(c1 != BVH::emptyNode);
        if (likely(mask == 0)) {
          if (d0 < d1) {
            assert(tNear[r1] >= 0.0f);
            stackPtr->mask    = tMask[r1];
            stackPtr->parent  = parent;
            stackPtr->child   = c1;
            stackPtr++;
            cur = c0;
            m_trav_active = tMask[r0];
            return;
          }
          else {
            assert(tNear[r0] >= 0.0f);
            stackPtr->mask    = tMask[r0];
            stackPtr->parent  = parent;
            stackPtr->child   = c0;
            stackPtr++;
            cur = c1;
            m_trav_active = tMask[r1];
            return;
          }
        }

        /*! slow path for more than two hits */
        size_t hits = movemask(vmask);
        const vint<N> dist_i = select(vmask, (asInt(tNear) & 0xfffffff8) | vint<N>(step), 0);
        const vint<N> dist_i_sorted = usort_descending(dist_i);
        const vint<N> sorted_index = dist_i_sorted & 7;

        size_t i = 0;
        for (;;)
        {
          const unsigned int index = sorted_index[i];
          assert(index < 8);
          cur = node->child(index);
          m_trav_active = tMask[index];
          assert(m_trav_active);
          BVHN<N>::prefetch(cur,types);
          bscf(hits);
          if (unlikely(hits==0)) break;
          i++;
          assert(cur != BVH::emptyNode);
          assert(tNear[index] >= 0.0f);
          stackPtr->mask    = m_trav_active;
          stackPtr->parent  = parent;
          stackPtr->child   = cur;
          stackPtr++;
        }
      }

      template<class T>
      static __forceinline void traverseAnyHit(NodeRef& cur,
                                               size_t& m_trav_active,
                                               const vbool<N>& vmask,
                                               const T* const tMask,
                                               StackItemMaskCoherent*& stackPtr)
      {
        const NodeRef parent = cur;
        size_t mask = movemask(vmask);
        assert(mask != 0);
        const BaseNode* node = cur.baseNode();

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r);
        BVHN<N>::prefetch(cur,types);
        m_trav_active = tMask[r];

        /* simple in order sequence */
        assert(cur != BVH::emptyNode);
        if (likely(mask == 0)) return;
        stackPtr->mask    = m_trav_active;
        stackPtr->parent  = parent;
        stackPtr->child   = cur;
        stackPtr++;

        for (; ;)
        {
          r = bscf(mask);
          cur = node->child(r);
          BVHN<N>::prefetch(cur,types);
          m_trav_active = tMask[r];
          assert(cur != BVH::emptyNode);
          if (likely(mask == 0)) return;
          stackPtr->mask    = m_trav_active;
          stackPtr->parent  = parent;
          stackPtr->child   = cur;
          stackPtr++;
        }
      }
    };
  }
}
