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
#include "node_intersector1.h"
#include "../common/stack_item.h"

#define NEW_SORTING_CODE 1

namespace embree
{
  namespace isa
  {
    /*! BVH regular node traversal for single rays. */
    template<int N, int Nx, int types>
    class BVHNNodeTraverser1Hit;

    /*! Helper functions for fast sorting using AVX512 instructions. */
#if defined(__AVX512ER__)

    /* KNL code path */
    __forceinline void isort_update(vfloat16 &dist, vllong8 &ptr, const vfloat16 &d, const vllong8 &p)
    {
      const vfloat16 dist_shift = align_shift_right<15>(dist,dist);
      const vllong8  ptr_shift  = align_shift_right<7>(ptr,ptr);
      const vbool16 m_geq = d >= dist;
      const vbool16 m_geq_shift = m_geq << 1;
      dist = select(m_geq,d,dist);
      ptr  = select(vboold8(m_geq),p,ptr);
      dist = select(m_geq_shift,dist_shift,dist);
      ptr  = select(vboold8(m_geq_shift),ptr_shift,ptr);
    }

    __forceinline void isort_quick_update(vfloat16 &dist, vllong8 &ptr, const vfloat16 &d, const vllong8 &p)
    {
      //dist = align_shift_right<15>(dist,d);
      //ptr  = align_shift_right<7>(ptr,p);
      dist = align_shift_right<15>(dist,permute(d,vint16(zero)));
      ptr  = align_shift_right<7>(ptr,permute(p,vllong8(zero)));
    }

    template<int N, int Nx, int types, class NodeRef, class BaseNode>
    __forceinline void traverseClosestHitAVX512(NodeRef& cur,
                                                size_t mask,
                                                const vfloat<Nx>& tNear,
                                                StackItemT<NodeRef>*& stackPtr,
                                                StackItemT<NodeRef>* stackEnd)
    {
      assert(mask != 0);
      const BaseNode* node = cur.baseNode(types);

      vllong8 children( vllong<N>::loadu((void*)node->children) );
      children = vllong8::compact((int)mask,children);
      vfloat16 distance = tNear;
      distance = vfloat16::compact((int)mask,distance,tNear);

      cur = toScalar(children);
      cur.prefetch(types);


      mask &= mask-1;
      if (likely(mask == 0)) return;

      /* 2 hits: order A0 B0 */
      const vllong8 c0(children);
      const vfloat16 d0(distance);
      children = align_shift_right<1>(children,children);
      distance = align_shift_right<1>(distance,distance);
      const vllong8 c1(children);
      const vfloat16 d1(distance);

      cur = toScalar(children);
      cur.prefetch(types);

      /* a '<' keeps the order for equal distances, scenes like powerplant largely benefit from it */
      const vboolf16 m_dist  = d0 < d1;
      const vfloat16 dist_A0 = select(m_dist, d0, d1);
      const vfloat16 dist_B0 = select(m_dist, d1, d0);
      const vllong8 ptr_A0   = select(vboold8(m_dist), c0, c1);
      const vllong8 ptr_B0   = select(vboold8(m_dist), c1, c0);

      mask &= mask-1;
      if (likely(mask == 0)) {
        cur = toScalar(ptr_A0);
        stackPtr[0].ptr            = toScalar(ptr_B0);
        *(float*)&stackPtr[0].dist = toScalar(dist_B0);
        stackPtr++;
        return;
      }

      /* 3 hits: order A1 B1 C1 */

      children = align_shift_right<1>(children,children);
      distance = align_shift_right<1>(distance,distance);

      const vllong8 c2(children);
      const vfloat16 d2(distance);

      cur = toScalar(children);
      cur.prefetch(types);

      const vboolf16 m_dist1     = dist_A0 <= d2;
      const vfloat16 dist_tmp_B1 = select(m_dist1, d2, dist_A0);
      const vllong8  ptr_A1      = select(vboold8(m_dist1), ptr_A0, c2);
      const vllong8  ptr_tmp_B1  = select(vboold8(m_dist1), c2, ptr_A0);

      const vboolf16 m_dist2     = dist_B0 <= dist_tmp_B1;
      const vfloat16 dist_B1     = select(m_dist2, dist_B0 , dist_tmp_B1);
      const vfloat16 dist_C1     = select(m_dist2, dist_tmp_B1, dist_B0);
      const vllong8  ptr_B1      = select(vboold8(m_dist2), ptr_B0, ptr_tmp_B1);
      const vllong8  ptr_C1      = select(vboold8(m_dist2), ptr_tmp_B1, ptr_B0);

      mask &= mask-1;
      if (likely(mask == 0)) {
        cur = toScalar(ptr_A1);
        stackPtr[0].ptr  = toScalar(ptr_C1);
        *(float*)&stackPtr[0].dist = toScalar(dist_C1);
        stackPtr[1].ptr  = toScalar(ptr_B1);
        *(float*)&stackPtr[1].dist = toScalar(dist_B1);
        stackPtr+=2;
        return;
      }

      /* 4 hits: order A2 B2 C2 D2 */

      const vfloat16 dist_A1  = select(m_dist1, dist_A0, d2);

      children = align_shift_right<1>(children,children);
      distance = align_shift_right<1>(distance,distance);

      const vllong8 c3(children);
      const vfloat16 d3(distance);

      cur = toScalar(children);
      cur.prefetch(types);

      const vboolf16 m_dist3     = dist_A1 <= d3;
      const vfloat16 dist_tmp_B2 = select(m_dist3, d3, dist_A1);
      const vllong8  ptr_A2      = select(vboold8(m_dist3), ptr_A1, c3);
      const vllong8  ptr_tmp_B2  = select(vboold8(m_dist3), c3, ptr_A1);

      const vboolf16 m_dist4     = dist_B1 <= dist_tmp_B2;
      const vfloat16 dist_B2     = select(m_dist4, dist_B1 , dist_tmp_B2);
      const vfloat16 dist_tmp_C2 = select(m_dist4, dist_tmp_B2, dist_B1);
      const vllong8  ptr_B2      = select(vboold8(m_dist4), ptr_B1, ptr_tmp_B2);
      const vllong8  ptr_tmp_C2  = select(vboold8(m_dist4), ptr_tmp_B2, ptr_B1);

      const vboolf16 m_dist5     = dist_C1 <= dist_tmp_C2;
      const vfloat16 dist_C2     = select(m_dist5, dist_C1 , dist_tmp_C2);
      const vfloat16 dist_D2     = select(m_dist5, dist_tmp_C2, dist_C1);
      const vllong8  ptr_C2      = select(vboold8(m_dist5), ptr_C1, ptr_tmp_C2);
      const vllong8  ptr_D2      = select(vboold8(m_dist5), ptr_tmp_C2, ptr_C1);

      mask &= mask-1;
      if (likely(mask == 0)) {
        cur = toScalar(ptr_A2);
        stackPtr[0].ptr  = toScalar(ptr_D2);
        *(float*)&stackPtr[0].dist = toScalar(dist_D2);
        stackPtr[1].ptr  = toScalar(ptr_C2);
        *(float*)&stackPtr[1].dist = toScalar(dist_C2);
        stackPtr[2].ptr  = toScalar(ptr_B2);
        *(float*)&stackPtr[2].dist = toScalar(dist_B2);
        stackPtr+=3;
        return;
      }

      /* >=5 hits: reverse to descending order for writing to stack */

      const size_t hits = 4 + popcnt(mask);
      const vfloat16 dist_A2  = select(m_dist3, dist_A1, d3);
      vfloat16 dist(neg_inf);
      vllong8 ptr(zero);


      isort_quick_update(dist,ptr,dist_A2,ptr_A2);
      isort_quick_update(dist,ptr,dist_B2,ptr_B2);
      isort_quick_update(dist,ptr,dist_C2,ptr_C2);
      isort_quick_update(dist,ptr,dist_D2,ptr_D2);

      do {

        children = align_shift_right<1>(children,children);
        distance = align_shift_right<1>(distance,distance);

        cur = toScalar(children);
        cur.prefetch(types);

        const vfloat16 new_dist(permute(distance,vint16(zero)));
        const vllong8 new_ptr(permute(children,vllong8(zero)));

        mask &= mask-1;
        isort_update(dist,ptr,new_dist,new_ptr);

      } while(mask);

      const vboold8 m_stack_ptr(0x55);  // 10101010 (lsb -> msb)
      const vboolf16 m_stack_dist(0x4444); // 0010001000100010 (lsb -> msb)

      /* extract current noderef */
      cur = toScalar(permute(ptr,vllong8(hits-1)));
      /* rearrange pointers to beginning of 16 bytes block */
      vllong8 stackElementA0;
      stackElementA0 = vllong8::expand(m_stack_ptr,ptr,stackElementA0);
      /* put distances in between */
      vuint16 stackElementA1((__m512i)stackElementA0);
      stackElementA1 = vuint16::expand(m_stack_dist,asUInt(dist),stackElementA1);
      /* write out first 4 x 16 bytes block to stack */
      vuint16::storeu(stackPtr,stackElementA1);
      /* get upper half of dist and ptr */
      dist = align_shift_right<4>(dist,dist);
      ptr  = align_shift_right<4>(ptr,ptr);
      /* assemble and write out second block */
      vllong8 stackElementB0;
      stackElementB0 = vllong8::expand(m_stack_ptr,ptr,stackElementB0);
      vuint16 stackElementB1((__m512i)stackElementB0);
      stackElementB1 = vuint16::expand(m_stack_dist,asUInt(dist),stackElementB1);
      vuint16::storeu(stackPtr + 4,stackElementB1);
      /* increase stack pointer */
      stackPtr += hits-1;
    }
#endif

#if defined(__AVX512VL__) // SKX

    template<int N>
    __forceinline void isort_update(vfloat<N> &dist, vint<N> &ptr, const vfloat<N> &d, const vint<N> &p)
    {
      const vfloat<N> dist_shift = align_shift_right<N-1>(dist,dist);
      const vint<N>  ptr_shift  = align_shift_right<N-1>(ptr,ptr);
      const vboolf<N> m_geq = d >= dist;
      const vboolf<N> m_geq_shift = m_geq << 1;
      dist = select(m_geq,d,dist);
      ptr  = select(m_geq,p,ptr);
      dist = select(m_geq_shift,dist_shift,dist);
      ptr  = select(m_geq_shift,ptr_shift,ptr);
    }

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
    __forceinline void isort_quick_update(vfloat<N> &dist, vint<N> &ptr, const vfloat<N> &d, const vint<N> &p)
    {
      dist = align_shift_right<N-1>(dist,permute(d,vint<N>(zero)));
      ptr  = align_shift_right<N-1>(ptr,permute(p,vint<N>(zero)));
    }

    template<int N>
    __forceinline void isort_quick_update(vint<N> &dist, const vint<N> &d)
    {
      dist = align_shift_right<N-1>(dist,permute(d,vint<N>(zero)));
    }


    __forceinline size_t permuteExtract(const vint8& index, const vllong4& n0, const vllong4& n1)
    {
      return toScalar(permutex2var((__m256i)index,n0,n1));
    }

    __forceinline size_t permuteExtract(const vint8& index, const vllong4& n0)
    {
      return toScalar(permute(n0,(__m256i)index));
    }

    __forceinline size_t permuteExtract(const vint4& index, const vllong4& n0)
    {
      return permuteExtract(_mm256_castsi128_si256(index),n0);
    }

    template<int N>
    __forceinline float permuteExtract(const vint<N>& index, const vfloat<N>& n)
    {
      return toScalar(permute(n,index));
    }

#endif

    /* Specialization for BVH4. */
    template<int Nx, int types>
    class BVHNNodeTraverser1Hit<4, Nx, types>
    {
      typedef BVH4 BVH;
      typedef BVH4::NodeRef NodeRef;
      typedef BVH4::BaseNode BaseNode;


    public:
      /* Traverses a node with at least one hit child. Optimized for finding the closest hit (intersection). */
      static __forceinline void traverseClosestHit(NodeRef& cur,
                                                   size_t mask,
                                                   const vfloat<Nx>& tNear,
                                                   StackItemT<NodeRef>*& stackPtr,
                                                   StackItemT<NodeRef>* stackEnd)
      {
        assert(mask != 0);
#if defined(__AVX512ER__)
        traverseClosestHitAVX512<4,Nx,types,NodeRef,BaseNode>(cur,mask,tNear,stackPtr,stackEnd);
#else
        const BaseNode* node = cur.baseNode(types);

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r);
        cur.prefetch(types);
        if (likely(mask == 0)) {
          assert(cur != BVH::emptyNode);
          return;
        }

        /*! two children are hit, push far child, and continue with closer child */
        NodeRef c0 = cur;
        const unsigned int d0 = ((unsigned int*)&tNear)[r];
        r = bscf(mask);
        NodeRef c1 = node->child(r);
        c1.prefetch(types);
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
        NodeRef c2 = node->child(r); c2.prefetch(types); unsigned int d2 = ((unsigned int*)&tNear)[r]; 
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
        NodeRef c3 = node->child(r); c3.prefetch(types); unsigned int d3 = ((unsigned int*)&tNear)[r]; 
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
        NodeRef c = node->child(r); c.prefetch(types); unsigned int d = ((unsigned int*)&tNear)[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
        assert(c != BVH::emptyNode);
        if (likely(mask == 0)) {
          sort(stackPtr[-1],stackPtr[-2],stackPtr[-3]);
          cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
          return;
        }

        /*! four children are hit, push all onto stack and sort 4 stack items, continue with closest child */
        assert(stackPtr < stackEnd);
        r = bscf(mask);
        c = node->child(r); c.prefetch(types); d = *(unsigned int*)&tNear[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
        assert(c != BVH::emptyNode);
        sort(stackPtr[-1],stackPtr[-2],stackPtr[-3],stackPtr[-4]);
        cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
#endif
#endif
      }

      /* Traverses a node with at least one hit child. Optimized for finding any hit (occlusion). */
      static __forceinline void traverseAnyHit(NodeRef& cur,
                                               size_t mask,
                                               const vfloat<Nx>& tNear,
                                               NodeRef*& stackPtr,
                                               NodeRef* stackEnd)
      {
        const BaseNode* node = cur.baseNode(types);

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r); 
        cur.prefetch(types);

        /* simpler in sequence traversal order */
        assert(cur != BVH::emptyNode);
        if (likely(mask == 0)) return;
        assert(stackPtr < stackEnd);
        *stackPtr = cur; stackPtr++;

        for (; ;)
        {
          r = bscf(mask);
          cur = node->child(r); cur.prefetch(types);
          assert(cur != BVH::emptyNode);
          if (likely(mask == 0)) return;
          assert(stackPtr < stackEnd);
          *stackPtr = cur; stackPtr++;
        }
      }
    };

    /* Specialization for BVH8. */
    template<int Nx, int types>
    class BVHNNodeTraverser1Hit<8, Nx, types>
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
        const BaseNode* node = (types == BVH_FLAG_ALIGNED_NODE) ? cur.alignedNode() : cur.baseNode(types);
        const vllong4 n0 = vllong4::loadu((vllong4*)&node->children[0]);
        const vllong4 n1 = vllong4::loadu((vllong4*)&node->children[4]);
        vint8 distance_i = (asInt(tNear) & 0xfffffff8) | vint8(step);
        distance_i = vint8::compact((int)mask,distance_i,distance_i);
        cur = permuteExtract(distance_i,n0,n1);
        cur.prefetch(types);

        mask &= mask-1;
        if (likely(mask == 0)) return;

        /* 2 hits: order A0 B0 */
        const vint8 d0(distance_i);
        const vint8 d1(shuffle<1>(distance_i));
        cur = permuteExtract(d1,n0,n1);
        cur.prefetch(types);

        const vint8 dist_A0 = min(d0, d1);
        const vint8 dist_B0 = max(d0, d1);

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
        cur.prefetch(types);

        const vint8 dist_A1     = min(dist_A0,d2);
        const vint8 dist_tmp_B1 = max(dist_A0,d2);
        const vint8 dist_B1     = min(dist_B0,dist_tmp_B1);
        const vint8 dist_C1     = max(dist_B0,dist_tmp_B1);        

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
        cur.prefetch(types);

        const vint8 dist_A2     = min(dist_A1,d3);
        const vint8 dist_tmp_B2 = max(dist_A1,d3);
        const vint8 dist_B2     = min(dist_B1,dist_tmp_B2);
        const vint8 dist_tmp_C2 = max(dist_B1,dist_tmp_B2);
        const vint8 dist_C2     = min(dist_C1,dist_tmp_C2);
        const vint8 dist_D2     = max(dist_C1,dist_tmp_C2);

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
        vint8 dist(-1);

        isort_quick_update(dist,dist_A2);
        isort_quick_update(dist,dist_B2);
        isort_quick_update(dist,dist_C2);
        isort_quick_update(dist,dist_D2);

        do {

          distance_i = align_shift_right<1>(distance_i,distance_i);
          cur = permuteExtract(distance_i,n0,n1);
          cur.prefetch(types);
          const vint8 new_dist(permute(distance_i,vint8(zero)));
          mask &= mask-1;
          isort_update(dist,new_dist);

        } while(mask);

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
                                                   const vfloat<Nx>& tNear,
                                                   StackItemT<NodeRef>*& stackPtr,
                                                   StackItemT<NodeRef>* stackEnd)
      {
        assert(mask != 0);
#if defined(__AVX512ER__)
        traverseClosestHitAVX512<8,Nx,types,NodeRef,BaseNode>(cur,mask,tNear,stackPtr,stackEnd);
#elif defined(__AVX512VL__)
        traverseClosestHitAVX512VL8<NodeRef,BaseNode>(cur,mask,tNear,stackPtr,stackEnd);
#else

        const BaseNode* node = cur.baseNode(types);

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r);
        cur.prefetch(types);
        if (likely(mask == 0)) {
          assert(cur != BVH::emptyNode);
          return;
        }

        /*! two children are hit, push far child, and continue with closer child */
        NodeRef c0 = cur;
        const unsigned int d0 = ((unsigned int*)&tNear)[r];
        r = bscf(mask);
        NodeRef c1 = node->child(r);
        c1.prefetch(types);
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
        NodeRef c2 = node->child(r); c2.prefetch(types); unsigned int d2 = ((unsigned int*)&tNear)[r]; 
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
        NodeRef c3 = node->child(r); c3.prefetch(types); unsigned int d3 = ((unsigned int*)&tNear)[r]; 
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
          NodeRef c = node->child(r); c.prefetch(types); unsigned int d = *(unsigned int*)&tNear[r]; 
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
        NodeRef c = node->child(r); c.prefetch(types); unsigned int d = ((unsigned int*)&tNear)[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
        assert(c != BVH::emptyNode);
        if (likely(mask == 0)) {
          sort(stackPtr[-1],stackPtr[-2],stackPtr[-3]);
          cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
          return;
        }

        /*! four children are hit, push all onto stack and sort 4 stack items, continue with closest child */
        assert(stackPtr < stackEnd);
        r = bscf(mask);
        c = node->child(r); c.prefetch(types); d = *(unsigned int*)&tNear[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
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
          c = node->child(r); c.prefetch(types); d = *(unsigned int*)&tNear[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
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
                                               const vfloat<Nx>& tNear,
                                               NodeRef*& stackPtr,
                                               NodeRef* stackEnd)
      {
        const BaseNode* node = cur.baseNode(types);

        /*! one child is hit, continue with that child */
        size_t r = bscf(mask);
        cur = node->child(r);
        cur.prefetch(types);

        /* simpler in sequence traversal order */
        assert(cur != BVH::emptyNode);
        if (likely(mask == 0)) return;
        assert(stackPtr < stackEnd);
        *stackPtr = cur; stackPtr++;

        for (; ;)
        {
          r = bscf(mask);
          cur = node->child(r); cur.prefetch(types);
          assert(cur != BVH::emptyNode);
          if (likely(mask == 0)) return;
          assert(stackPtr < stackEnd);
          *stackPtr = cur; stackPtr++;
        }
      }
    };
  }
}
