// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curveNi.h"

namespace embree
{
  template<int M>
    struct CurveNv : public CurveNi<M>
  {
    using CurveNi<M>::N;
      
    struct Type : public PrimitiveType {
      const char* name() const;
      size_t sizeActive(const char* This) const;
      size_t sizeTotal(const char* This) const;
      size_t getBytes(const char* This) const;
    };
    static Type type;

  public:

    /* Returns maximum number of stored primitives */
    static __forceinline size_t max_size() { return M; }

    /* Returns required number of primitive blocks for N primitives */
    static __forceinline size_t blocks(size_t N) { return (N+M-1)/M; }

    static __forceinline size_t bytes(size_t N)
    {
      const size_t f = N/M, r = N%M;
      static_assert(sizeof(CurveNv) == 22+25*M+4*16*M, "internal data layout issue");
      return f*sizeof(CurveNv) + (r!=0)*(22 + 25*r + 4*16*r);
    }

  public:

    /*! Default constructor. */
    __forceinline CurveNv () {}

    /*! fill curve from curve list */
    __forceinline void fill(const PrimRef* prims, size_t& begin, size_t _end, Scene* scene)
    {
      size_t end = min(begin+M,_end);
      size_t N = end-begin;

      /* encode all primitives */
      for (size_t i=0; i<N; i++)
      {
        const PrimRef& prim = prims[begin+i];
        const unsigned int geomID = prim.geomID();
        const unsigned int primID = prim.primID();
        CurveGeometry* mesh = (CurveGeometry*) scene->get(geomID);
        const unsigned vtxID = mesh->curve(primID);
        Vec3fa::storeu(&this->vertices(i,N)[0],mesh->vertex(vtxID+0));
        Vec3fa::storeu(&this->vertices(i,N)[1],mesh->vertex(vtxID+1));
        Vec3fa::storeu(&this->vertices(i,N)[2],mesh->vertex(vtxID+2));
        Vec3fa::storeu(&this->vertices(i,N)[3],mesh->vertex(vtxID+3));
      }
    }

    template<typename BVH, typename Allocator>
      __forceinline static typename BVH::NodeRef createLeaf (BVH* bvh, const PrimRef* prims, const range<size_t>& set, const Allocator& alloc)
    {
      if (set.size() == 0)
        return BVH::emptyNode;
      
      /* fall back to CurveNi for oriented curves */
      unsigned int geomID = prims[set.begin()].geomID();
      if (bvh->scene->get(geomID)->getCurveType() == Geometry::GTY_SUBTYPE_ORIENTED_CURVE) {
        return CurveNi<M>::createLeaf(bvh,prims,set,alloc);
      }
      if (bvh->scene->get(geomID)->getCurveBasis() == Geometry::GTY_BASIS_HERMITE) {
        return CurveNi<M>::createLeaf(bvh,prims,set,alloc);
      }
      
      size_t start = set.begin();
      size_t items = CurveNv::blocks(set.size());
      size_t numbytes = CurveNv::bytes(set.size());
      CurveNv* accel = (CurveNv*) alloc.malloc1(numbytes,BVH::byteAlignment);
      for (size_t i=0; i<items; i++) {
        accel[i].CurveNv<M>::fill(prims,start,set.end(),bvh->scene);
        accel[i].CurveNi<M>::fill(prims,start,set.end(),bvh->scene);
      }
      return bvh->encodeLeaf((char*)accel,items);
    };
    
  public:
    unsigned char data[4*16*M];
    __forceinline       Vec3fa* vertices(size_t i, size_t N)       { return (Vec3fa*)CurveNi<M>::end(N)+4*i; }
    __forceinline const Vec3fa* vertices(size_t i, size_t N) const { return (Vec3fa*)CurveNi<M>::end(N)+4*i; }
  };

  template<int M>
    typename CurveNv<M>::Type CurveNv<M>::type;

  typedef CurveNv<4> Curve4v;
  typedef CurveNv<8> Curve8v;
}
