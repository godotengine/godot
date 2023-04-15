// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"
#include "node_intersector1.h"
#include "../common/stack_item.h"

#define NEW_SORTING_CODE 1

namespace embree
{
  namespace isa
  {
    /*! BVH regular node traversal for single rays. */
    template<int N, int types>
    class BVHNNodeTraverser1Hit;

#if defined(__AVX512VL__) // SKX

    template<int N>
    __forceinline void isort_update(vint<N> &dist, const vint<N> &d)
    {
      const vint<N> dist_shift = align_shift_right<N-1>(dist,dist);
      const vboolf<N> m_geq = d >= dist;
      const vboolf<N> m_geq_shift = m_geq << 1;
      dist = select(m_geq,d,dist);
      dist = select(m_geq_shift,dist_shift,dist);
    }

    template<int N>
    __forceinline void isort_quick_update(vint<N> &dist, const vint<N> &d) {
      dist = align_shift_right<N-1>(dist,permute(d,vint<N>(zero)));
    }

    __forceinline size_t permuteExtract(const vint8& index, const vllong4& n0, const vllong4& n1) {
      return toScalar(permutex2var((__m256i)index,n0,n1));
    }

    __forceinline float permuteExtract(const vint8& index, const vfloat8& n) {
      return toScalar(permute(n,index));
    }

#endif

    /* Specialization for BVH4. */
    template<int types>
    class BVHNNodeTraverser1Hit<4, types>
    {
      typedef BVH4 BVH;
      typedef BVH4::NodeRef NodeRef;
      typedef BVH4::BaseNode BaseNode;


    public:
      /* Traverses a node with at least one hit child. Optimized for finding the closest hit (intersection). */
      static __forceinline void traverseClosestHit(NodeRef& cur,
                                                   size_t mask,
                                                   const vfloat4& tNear,
                                                   StackItemT<NodeRef>*& stackPtr,
                                                   StackItemT<NodeRef>* stackEnd)
      {
        assert(mask != 0);
        const BaseNode* node = cur.baseNode();

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r);
        BVH::prefetch(cur,types);
        if (likely(mask == 0)) {
          assert(cur != BVH::emptyNode);
          return;
        }

        /*! two children are hit, push far child, and continue with closer child */
        NodeRef c0 = cur;
        const unsigned int d0 = ((unsigned int*)&tNear)[r];
        r = bscf(mask);
        NodeRef c1 = node->child(r);
        BVH::prefetch(c1,types);
        const unsigned int d1 = ((unsigned int*)&tNear)[r];
        assert(c0 != BVH::emptyNode);
        assert(c1 != BVH::emptyNode);
        if (likely(mask == 0)) {
          assert(stackPtr < stackEnd);
          if (d0 < d1) { stackPtr->ptr = c1; stackPtr->dist = d1; stackPtr++; cur = c0; return; }
          else         { stackPtr->ptr = c0; stackPtr->dist = d0; stackPtr++; cur = c1; return; }
        }

#if NEW_SORTING_CODE == 1
        vint4 s0((size_t)c0,(size_t)d0);
        vint4 s1((size_t)c1,(size_t)d1);
        r = bscf(mask);
        NodeRef c2 = node->child(r); BVH::prefetch(c2,types); unsigned int d2 = ((unsigned int*)&tNear)[r]; 
        vint4 s2((size_t)c2,(size_t)d2);
        /* 3 hits */
        if (likely(mask == 0)) {
          StackItemT<NodeRef>::sort3(s0,s1,s2);
          *(vint4*)&stackPtr[0] = s0; *(vint4*)&stackPtr[1] = s1;
          cur = toSizeT(s2);
          stackPtr+=2;
          return;
        }
        r = bscf(mask);
        NodeRef c3 = node->child(r); BVH::prefetch(c3,types); unsigned int d3 = ((unsigned int*)&tNear)[r]; 
        vint4 s3((size_t)c3,(size_t)d3);
        /* 4 hits */
        StackItemT<NodeRef>::sort4(s0,s1,s2,s3);
        *(vint4*)&stackPtr[0] = s0; *(vint4*)&stackPtr[1] = s1; *(vint4*)&stackPtr[2] = s2;
        cur = toSizeT(s3);
        stackPtr+=3;
#else
        /*! Here starts the slow path for 3 or 4 hit children. We push
         *  all nodes onto the stack to sort them there. */
        assert(stackPtr < stackEnd);
        stackPtr->ptr = c0; stackPtr->dist = d0; stackPtr++;
        assert(stackPtr < stackEnd);
        stackPtr->ptr = c1; stackPtr->dist = d1; stackPtr++;

        /*! three children are hit, push all onto stack and sort 3 stack items, continue with closest child */
        assert(stackPtr < stackEnd);
        r = bscf(mask);
        NodeRef c = node->child(r); BVH::prefetch(c,types); unsigned int d = ((unsigned int*)&tNear)[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
        assert(c != BVH::emptyNode);
        if (likely(mask == 0)) {
          sort(stackPtr[-1],stackPtr[-2],stackPtr[-3]);
          cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
          return;
        }

        /*! four children are hit, push all onto stack and sort 4 stack items, continue with closest child */
        assert(stackPtr < stackEnd);
        r = bscf(mask);
        c = node->child(r); BVH::prefetch(c,types); d = *(unsigned int*)&tNear[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
        assert(c != BVH::emptyNode);
        sort(stackPtr[-1],stackPtr[-2],stackPtr[-3],stackPtr[-4]);
        cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
#endif
      }

      /* Traverses a node with at least one hit child. Optimized for finding any hit (occlusion). */
      static __forceinline void traverseAnyHit(NodeRef& cur,
                                               size_t mask,
                                               const vfloat4& tNear,
                                               NodeRef*& stackPtr,
                                               NodeRef* stackEnd)
      {
        const BaseNode* node = cur.baseNode();

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r); 
        BVH::prefetch(cur,types);

        /* simpler in sequence traversal order */
        assert(cur != BVH::emptyNode);
        if (likely(mask == 0)) return;
        assert(stackPtr < stackEnd);
        *stackPtr = cur; stackPtr++;

        for (; ;)
        {
          r = bscf(mask);
          cur = node->child(r); BVH::prefetch(cur,types);
          assert(cur != BVH::emptyNode);
          if (likely(mask == 0)) return;
          assert(stackPtr < stackEnd);
          *stackPtr = cur; stackPtr++;
        }
      }
    };

    /* Specialization for BVH8. */
    template<int types>
    class BVHNNodeTraverser1Hit<8, types>
    {
      typedef BVH8 BVH;
      typedef BVH8::NodeRef NodeRef;
      typedef BVH8::BaseNode BaseNode;
      
#if defined(__AVX512VL__)
      template<class NodeRef, class BaseNode>
        static __forceinline void traverseClosestHitAVX512VL8(NodeRef& cur,
                                                              size_t mask,
                                                              const vfloat8& tNear,
                                                              StackItemT<NodeRef>*& stackPtr,
                                                              StackItemT<NodeRef>* stackEnd)
      {
        assert(mask != 0);
        const BaseNode* node = cur.baseNode();
        const vllong4 n0 = vllong4::loadu((vllong4*)&node->children[0]);
        const vllong4 n1 = vllong4::loadu((vllong4*)&node->children[4]);
        vint8 distance_i = (asInt(tNear) & 0xfffffff8) | vint8(step);
        distance_i = vint8::compact((int)mask,distance_i,distance_i);
        cur = permuteExtract(distance_i,n0,n1);
        BVH::prefetch(cur,types);

        mask &= mask-1;
        if (likely(mask == 0)) return;

        /* 2 hits: order A0 B0 */
        const vint8 d0(distance_i);
        const vint8 d1(shuffle<1>(distance_i));
        cur = permuteExtract(d1,n0,n1);
        BVH::prefetch(cur,types);

        const vint8 dist_A0 = min(d0, d1);
        const vint8 dist_B0 = max(d0, d1);
        assert(dist_A0[0] < dist_B0[0]);

        mask &= mask-1;
        if (likely(mask == 0)) {
          cur                        = permuteExtract(dist_A0,n0,n1);
          stackPtr[0].ptr            = permuteExtract(dist_B0,n0,n1);
          *(float*)&stackPtr[0].dist = permuteExtract(dist_B0,tNear);
          stackPtr++;
          return;
        }

        /* 3 hits: order A1 B1 C1 */

        const vint8 d2(shuffle<2>(distance_i));
        cur = permuteExtract(d2,n0,n1);
        BVH::prefetch(cur,types);

        const vint8 dist_A1     = min(dist_A0,d2);
        const vint8 dist_tmp_B1 = max(dist_A0,d2);
        const vint8 dist_B1     = min(dist_B0,dist_tmp_B1);
        const vint8 dist_C1     = max(dist_B0,dist_tmp_B1);
        assert(dist_A1[0] < dist_B1[0]);
        assert(dist_B1[0] < dist_C1[0]);

        mask &= mask-1;
        if (likely(mask == 0)) {
          cur                        = permuteExtract(dist_A1,n0,n1);
          stackPtr[0].ptr            = permuteExtract(dist_C1,n0,n1);
          *(float*)&stackPtr[0].dist = permuteExtract(dist_C1,tNear);
          stackPtr[1].ptr            = permuteExtract(dist_B1,n0,n1);
          *(float*)&stackPtr[1].dist = permuteExtract(dist_B1,tNear);
          stackPtr+=2;
          return;
        }

        /* 4 hits: order A2 B2 C2 D2 */

        const vint8 d3(shuffle<3>(distance_i));
        cur = permuteExtract(d3,n0,n1);
        BVH::prefetch(cur,types);

        const vint8 dist_A2     = min(dist_A1,d3);
        const vint8 dist_tmp_B2 = max(dist_A1,d3);
        const vint8 dist_B2     = min(dist_B1,dist_tmp_B2);
        const vint8 dist_tmp_C2 = max(dist_B1,dist_tmp_B2);
        const vint8 dist_C2     = min(dist_C1,dist_tmp_C2);
        const vint8 dist_D2     = max(dist_C1,dist_tmp_C2);
        assert(dist_A2[0] < dist_B2[0]);
        assert(dist_B2[0] < dist_C2[0]);
        assert(dist_C2[0] < dist_D2[0]);
        
        mask &= mask-1;
        if (likely(mask == 0)) {
          cur                        = permuteExtract(dist_A2,n0,n1);
          stackPtr[0].ptr            = permuteExtract(dist_D2,n0,n1);
          *(float*)&stackPtr[0].dist = permuteExtract(dist_D2,tNear);
          stackPtr[1].ptr            = permuteExtract(dist_C2,n0,n1);
          *(float*)&stackPtr[1].dist = permuteExtract(dist_C2,tNear);
          stackPtr[2].ptr            = permuteExtract(dist_B2,n0,n1);
          *(float*)&stackPtr[2].dist = permuteExtract(dist_B2,tNear);
          stackPtr+=3;
          return;
        }

        /* >=5 hits: reverse to descending order for writing to stack */

        distance_i = align_shift_right<3>(distance_i,distance_i);
        const size_t hits = 4 + popcnt(mask);
        vint8 dist(INT_MIN); // this will work with -0.0f (0x80000000) as distance, isort_update uses >= to insert
	
        isort_quick_update<8>(dist,dist_A2);
        isort_quick_update<8>(dist,dist_B2);
        isort_quick_update<8>(dist,dist_C2);
        isort_quick_update<8>(dist,dist_D2);

        do {

          distance_i = align_shift_right<1>(distance_i,distance_i);
          cur = permuteExtract(distance_i,n0,n1);
          BVH::prefetch(cur,types);
          const vint8 new_dist(permute(distance_i,vint8(zero)));
          mask &= mask-1;
          isort_update<8>(dist,new_dist);

        } while(mask);

        for (size_t i=0; i<7; i++)
          assert(dist[i+0]>=dist[i+1]);

        for (size_t i=0;i<hits-1;i++)
        {
          stackPtr->ptr            = permuteExtract(dist,n0,n1);
          *(float*)&stackPtr->dist = permuteExtract(dist,tNear);
          dist = align_shift_right<1>(dist,dist);
          stackPtr++;
        }
        cur = permuteExtract(dist,n0,n1);
      }
#endif

    public:
      static __forceinline void traverseClosestHit(NodeRef& cur,
                                                   size_t mask,
                                                   const vfloat8& tNear,
                                                   StackItemT<NodeRef>*& stackPtr,
                                                   StackItemT<NodeRef>* stackEnd)
      {
        assert(mask != 0);
#if defined(__AVX512VL__)
        traverseClosestHitAVX512VL8<NodeRef,BaseNode>(cur,mask,tNear,stackPtr,stackEnd);
#else

        const BaseNode* node = cur.baseNode();

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r);
        BVH::prefetch(cur,types);
        if (likely(mask == 0)) {
          assert(cur != BVH::emptyNode);
          return;
        }

        /*! two children are hit, push far child, and continue with closer child */
        NodeRef c0 = cur;
        const unsigned int d0 = ((unsigned int*)&tNear)[r];
        r = bscf(mask);
        NodeRef c1 = node->child(r);
        BVH::prefetch(c1,types);
        const unsigned int d1 = ((unsigned int*)&tNear)[r];

        assert(c0 != BVH::emptyNode);
        assert(c1 != BVH::emptyNode);
        if (likely(mask == 0)) {
          assert(stackPtr < stackEnd);
          if (d0 < d1) { stackPtr->ptr = c1; stackPtr->dist = d1; stackPtr++; cur = c0; return; }
          else         { stackPtr->ptr = c0; stackPtr->dist = d0; stackPtr++; cur = c1; return; }
        }
#if NEW_SORTING_CODE == 1
        vint4 s0((size_t)c0,(size_t)d0);
        vint4 s1((size_t)c1,(size_t)d1);

        r = bscf(mask);
        NodeRef c2 = node->child(r); BVH::prefetch(c2,types); unsigned int d2 = ((unsigned int*)&tNear)[r]; 
        vint4 s2((size_t)c2,(size_t)d2);
        /* 3 hits */
        if (likely(mask == 0)) {
          StackItemT<NodeRef>::sort3(s0,s1,s2);
          *(vint4*)&stackPtr[0] = s0; *(vint4*)&stackPtr[1] = s1;
          cur = toSizeT(s2);
          stackPtr+=2;
          return;
        }
        r = bscf(mask);
        NodeRef c3 = node->child(r); BVH::prefetch(c3,types); unsigned int d3 = ((unsigned int*)&tNear)[r]; 
        vint4 s3((size_t)c3,(size_t)d3);
        /* 4 hits */
        if (likely(mask == 0)) {
          StackItemT<NodeRef>::sort4(s0,s1,s2,s3);
          *(vint4*)&stackPtr[0] = s0; *(vint4*)&stackPtr[1] = s1; *(vint4*)&stackPtr[2] = s2;
          cur = toSizeT(s3);
          stackPtr+=3;
          return;
        }
        *(vint4*)&stackPtr[0] = s0; *(vint4*)&stackPtr[1] = s1; *(vint4*)&stackPtr[2] = s2; *(vint4*)&stackPtr[3] = s3;
        /*! fallback case if more than 4 children are hit */
        StackItemT<NodeRef>* stackFirst = stackPtr;
        stackPtr+=4;      
        while (1)
        {
          assert(stackPtr < stackEnd);
          r = bscf(mask);
          NodeRef c = node->child(r); BVH::prefetch(c,types); unsigned int d = *(unsigned int*)&tNear[r]; 
          const vint4 s((size_t)c,(size_t)d);
          *(vint4*)stackPtr++ = s;
          assert(c != BVH::emptyNode);
          if (unlikely(mask == 0)) break;
        }
        sort(stackFirst,stackPtr);
        cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
#else
        /*! Here starts the slow path for 3 or 4 hit children. We push
         *  all nodes onto the stack to sort them there. */
        assert(stackPtr < stackEnd);
        stackPtr->ptr = c0; stackPtr->dist = d0; stackPtr++;
        assert(stackPtr < stackEnd);
        stackPtr->ptr = c1; stackPtr->dist = d1; stackPtr++;

        /*! three children are hit, push all onto stack and sort 3 stack items, continue with closest child */
        assert(stackPtr < stackEnd);
        r = bscf(mask);
        NodeRef c = node->child(r); BVH::prefetch(c,types); unsigned int d = ((unsigned int*)&tNear)[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
        assert(c != BVH::emptyNode);
        if (likely(mask == 0)) {
          sort(stackPtr[-1],stackPtr[-2],stackPtr[-3]);
          cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
          return;
        }

        /*! four children are hit, push all onto stack and sort 4 stack items, continue with closest child */
        assert(stackPtr < stackEnd);
        r = bscf(mask);
        c = node->child(r); BVH::prefetch(c,types); d = *(unsigned int*)&tNear[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
        assert(c != BVH::emptyNode);
        if (likely(mask == 0)) {
          sort(stackPtr[-1],stackPtr[-2],stackPtr[-3],stackPtr[-4]);
          cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
          return;
        }
        /*! fallback case if more than 4 children are hit */
        StackItemT<NodeRef>* stackFirst = stackPtr-4;
        while (1)
        {
          assert(stackPtr < stackEnd);
          r = bscf(mask);
          c = node->child(r); BVH::prefetch(c,types); d = *(unsigned int*)&tNear[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
          assert(c != BVH::emptyNode);
          if (unlikely(mask == 0)) break;
        }
        sort(stackFirst,stackPtr);
        cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
#endif
#endif
      }

      static __forceinline void traverseAnyHit(NodeRef& cur,
                                               size_t mask,
                                               const vfloat8& tNear,
                                               NodeRef*& stackPtr,
                                               NodeRef* stackEnd)
      {
        const BaseNode* node = cur.baseNode();

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r);
        BVH::prefetch(cur,types);

        /* simpler in sequence traversal order */
        assert(cur != BVH::emptyNode);
        if (likely(mask == 0)) return;
        assert(stackPtr < stackEnd);
        *stackPtr = cur; stackPtr++;

        for (; ;)
        {
          r = bscf(mask);
          cur = node->child(r); BVH::prefetch(cur,types);
          assert(cur != BVH::emptyNode);
          if (likely(mask == 0)) return;
          assert(stackPtr < stackEnd);
          *stackPtr = cur; stackPtr++;
        }
      }
    };
  }
}
