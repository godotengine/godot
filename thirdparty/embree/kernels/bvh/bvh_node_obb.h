// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh_node_base.h"

namespace embree
{
  /*! Node with unaligned bounds */
  template<typename NodeRef, int N>
    struct OBBNode_t : public BaseNode_t<NodeRef, N>
  {
    using BaseNode_t<NodeRef,N>::children;
    
    struct Create
    {
      __forceinline NodeRef operator() (const FastAllocator::CachedAllocator& alloc) const
      {
        OBBNode_t* node = (OBBNode_t*) alloc.malloc0(sizeof(OBBNode_t),NodeRef::byteNodeAlignment); node->clear();
        return NodeRef::encodeNode(node);
      }
    };
    
    struct Set
    {
      __forceinline void operator() (NodeRef node, size_t i, NodeRef child, const OBBox3fa& bounds) const {
        node.ungetAABBNode()->setRef(i,child);
        node.ungetAABBNode()->setBounds(i,bounds);
      }
    };
    
    /*! Clears the node. */
    __forceinline void clear()
    {
      naabb.l.vx = Vec3fa(nan);
      naabb.l.vy = Vec3fa(nan);
      naabb.l.vz = Vec3fa(nan);
      naabb.p    = Vec3fa(nan);
      BaseNode_t<NodeRef,N>::clear();
    }
    
    /*! Sets bounding box. */
    __forceinline void setBounds(size_t i, const OBBox3fa& b)
    {
      assert(i < N);
      
      AffineSpace3fa space = b.space;
      space.p -= b.bounds.lower;
      space = AffineSpace3fa::scale(1.0f/max(Vec3fa(1E-19f),b.bounds.upper-b.bounds.lower))*space;
      
      naabb.l.vx.x[i] = space.l.vx.x;
      naabb.l.vx.y[i] = space.l.vx.y;
      naabb.l.vx.z[i] = space.l.vx.z;
      
      naabb.l.vy.x[i] = space.l.vy.x;
      naabb.l.vy.y[i] = space.l.vy.y;
      naabb.l.vy.z[i] = space.l.vy.z;
      
      naabb.l.vz.x[i] = space.l.vz.x;
      naabb.l.vz.y[i] = space.l.vz.y;
      naabb.l.vz.z[i] = space.l.vz.z;
      
      naabb.p.x[i] = space.p.x;
      naabb.p.y[i] = space.p.y;
      naabb.p.z[i] = space.p.z;
    }
    
    /*! Sets ID of child. */
    __forceinline void setRef(size_t i, const NodeRef& ref) {
      assert(i < N);
      children[i] = ref;
    }
    
    /*! Returns the extent of the bounds of the ith child */
    __forceinline Vec3fa extent(size_t i) const {
      assert(i<N);
      const Vec3fa vx(naabb.l.vx.x[i],naabb.l.vx.y[i],naabb.l.vx.z[i]);
      const Vec3fa vy(naabb.l.vy.x[i],naabb.l.vy.y[i],naabb.l.vy.z[i]);
      const Vec3fa vz(naabb.l.vz.x[i],naabb.l.vz.y[i],naabb.l.vz.z[i]);
      return rsqrt(vx*vx + vy*vy + vz*vz);
    }
    
    /*! Returns reference to specified child */
    __forceinline       NodeRef& child(size_t i)       { assert(i<N); return children[i]; }
    __forceinline const NodeRef& child(size_t i) const { assert(i<N); return children[i]; }
    
    /*! output operator */
    friend embree_ostream operator<<(embree_ostream o, const OBBNode_t& n)
    {
      o << "UnAABBNode { " << n.naabb << " } " << embree_endl;
      return o;
    }
    
  public:
    AffineSpace3vf<N> naabb;   //!< non-axis aligned bounding boxes (bounds are [0,1] in specified space)
  };
}
