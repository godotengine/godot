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

#include "bvh.h"
#include "../common/ray.h"
#include "../common/stack_item.h"

namespace embree
{
  namespace isa
  {
    template<int N, int Nx, int types>
    class BVHNNodeTraverserStreamHitCoherent
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::BaseNode BaseNode;

    public:
      template<class T>
      static __forceinline void traverseClosestHit(NodeRef& cur,
                                                   size_t& m_trav_active,
                                                   const vbool<Nx>& vmask,
                                                   const vfloat<Nx>& tNear,
                                                   const T* const tMask,
                                                   StackItemMaskCoherent*& stackPtr)
      {
        const NodeRef parent = cur;
        size_t mask = movemask(vmask);
        assert(mask != 0);
        const BaseNode* node = cur.baseNode(types);

        /*! one child is hit, continue with that child */
        const size_t r0 = bscf(mask);
        assert(r0 < 8);
        cur = node->child(r0);
        cur.prefetch(types);
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
        c1.prefetch(types);
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
        const vint<Nx> dist_i = select(vmask, (asInt(tNear) & 0xfffffff8) | vint<Nx>(step), 0);
  #if defined(__AVX512F__) && !defined(__AVX512VL__) // KNL
        const vint<N> tmp = extractN<N,0>(dist_i);
        const vint<Nx> dist_i_sorted = usort_descending(tmp);
  #else
        const vint<Nx> dist_i_sorted = usort_descending(dist_i);
  #endif
        const vint<Nx> sorted_index = dist_i_sorted & 7;

        size_t i = 0;
        for (;;)
        {
          const unsigned int index = sorted_index[i];
          assert(index < 8);
          cur = node->child(index);
          m_trav_active = tMask[index];
          assert(m_trav_active);
          cur.prefetch(types);
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
                                               const vbool<Nx>& vmask,
                                               const T* const tMask,
                                               StackItemMaskCoherent*& stackPtr)
      {
        const NodeRef parent = cur;
        size_t mask = movemask(vmask);
        assert(mask != 0);
        const BaseNode* node = cur.baseNode(types);

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r);
        cur.prefetch(types);
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
          cur.prefetch(types);
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
