// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh_node_base.h"

namespace embree
{
  template<typename NodeRef, int N>
    struct OBBNodeMB_t : public BaseNode_t<NodeRef, N>
  {
    using BaseNode_t<NodeRef,N>::children;
    
    struct Create
    {
      __forceinline NodeRef operator() (const FastAllocator::CachedAllocator& alloc) const
      {
        OBBNodeMB_t* node = (OBBNodeMB_t*) alloc.malloc0(sizeof(OBBNodeMB_t),NodeRef::byteNodeAlignment); node->clear();
        return NodeRef::encodeNode(node);
      }
    };
    
    struct Set
    {
      __forceinline void operator() (NodeRef node, size_t i, NodeRef child, const LinearSpace3fa& space, const LBBox3fa& lbounds, const BBox1f dt) const {
        node.ungetAABBNodeMB()->setRef(i,child);
        node.ungetAABBNodeMB()->setBounds(i,space,lbounds.global(dt));
      }
    };
    
    /*! Clears the node. */
    __forceinline void clear()
    {
      space0 = one;
      //b0.lower = b0.upper = Vec3fa(nan);
      b1.lower = b1.upper = Vec3fa(nan);
      BaseNode_t<NodeRef,N>::clear();
    }
    
    /*! Sets space and bounding boxes. */
    __forceinline void setBounds(size_t i, const AffineSpace3fa& space, const LBBox3fa& lbounds) {
      setBounds(i,space,lbounds.bounds0,lbounds.bounds1);
    }
    
    /*! Sets space and bounding boxes. */
    __forceinline void setBounds(size_t i, const AffineSpace3fa& s0, const BBox3fa& a, const BBox3fa& c)
    {
      assert(i < N);
      
      AffineSpace3fa space = s0;
      space.p -= a.lower;
      Vec3fa scale = 1.0f/max(Vec3fa(1E-19f),a.upper-a.lower);
      space = AffineSpace3fa::scale(scale)*space;
      BBox3fa a1((a.lower-a.lower)*scale,(a.upper-a.lower)*scale);
      BBox3fa c1((c.lower-a.lower)*scale,(c.upper-a.lower)*scale);
      
      space0.l.vx.x[i] = space.l.vx.x; space0.l.vx.y[i] = space.l.vx.y; space0.l.vx.z[i] = space.l.vx.z;
      space0.l.vy.x[i] = space.l.vy.x; space0.l.vy.y[i] = space.l.vy.y; space0.l.vy.z[i] = space.l.vy.z;
      space0.l.vz.x[i] = space.l.vz.x; space0.l.vz.y[i] = space.l.vz.y; space0.l.vz.z[i] = space.l.vz.z;
      space0.p   .x[i] = space.p   .x; space0.p   .y[i] = space.p   .y; space0.p   .z[i] = space.p   .z;
      
      /*b0.lower.x[i] = a1.lower.x; b0.lower.y[i] = a1.lower.y; b0.lower.z[i] = a1.lower.z;
        b0.upper.x[i] = a1.upper.x; b0.upper.y[i] = a1.upper.y; b0.upper.z[i] = a1.upper.z;*/
      
      b1.lower.x[i] = c1.lower.x; b1.lower.y[i] = c1.lower.y; b1.lower.z[i] = c1.lower.z;
      b1.upper.x[i] = c1.upper.x; b1.upper.y[i] = c1.upper.y; b1.upper.z[i] = c1.upper.z;
    }
    
    /*! Sets ID of child. */
    __forceinline void setRef(size_t i, const NodeRef& ref) {
      assert(i < N);
      children[i] = ref;
    }
    
    /*! Returns the extent of the bounds of the ith child */
    __forceinline Vec3fa extent0(size_t i) const {
      assert(i < N);
      const Vec3fa vx(space0.l.vx.x[i],space0.l.vx.y[i],space0.l.vx.z[i]);
      const Vec3fa vy(space0.l.vy.x[i],space0.l.vy.y[i],space0.l.vy.z[i]);
      const Vec3fa vz(space0.l.vz.x[i],space0.l.vz.y[i],space0.l.vz.z[i]);
      return rsqrt(vx*vx + vy*vy + vz*vz);
    }
    
  public:
    AffineSpace3vf<N> space0;
    //BBox3vf<N> b0; // these are the unit bounds
    BBox3vf<N> b1;
  };
}
