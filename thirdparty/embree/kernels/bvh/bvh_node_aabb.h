// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh_node_base.h"

namespace embree
{
  /*! BVHN AABBNode */
  template<typename NodeRef, int N>
    struct AABBNode_t : public BaseNode_t<NodeRef, N>
  {
    using BaseNode_t<NodeRef,N>::children;
    
    struct Create
    {
      __forceinline NodeRef operator() (const FastAllocator::CachedAllocator& alloc, size_t numChildren = 0) const
      {
        AABBNode_t* node = (AABBNode_t*) alloc.malloc0(sizeof(AABBNode_t),NodeRef::byteNodeAlignment); node->clear();
        return NodeRef::encodeNode(node);
      }
    };
    
    struct Set
    {
      __forceinline void operator() (NodeRef node, size_t i, NodeRef child, const BBox3fa& bounds) const {
        node.getAABBNode()->setRef(i,child);
        node.getAABBNode()->setBounds(i,bounds);
      }
    };
    
    struct Create2
    {
      template<typename BuildRecord>
      __forceinline NodeRef operator() (BuildRecord* children, const size_t num, const FastAllocator::CachedAllocator& alloc) const
      {
        AABBNode_t* node = (AABBNode_t*) alloc.malloc0(sizeof(AABBNode_t), NodeRef::byteNodeAlignment); node->clear();
        for (size_t i=0; i<num; i++) node->setBounds(i,children[i].bounds());
        return NodeRef::encodeNode(node);
      }
    };
    
    struct Set2
    {
      template<typename BuildRecord>
      __forceinline NodeRef operator() (const BuildRecord& precord, const BuildRecord* crecords, NodeRef ref, NodeRef* children, const size_t num) const
      {
#if defined(DEBUG)
        // check that empty children are only at the end of the child list
        bool emptyChild = false;
        for (size_t i=0; i<num; i++) {
          emptyChild |= (children[i] == NodeRef::emptyNode);
          assert(emptyChild == (children[i] == NodeRef::emptyNode));
        }
#endif
        AABBNode_t* node = ref.getAABBNode();
        for (size_t i=0; i<num; i++) node->setRef(i,children[i]);
        return ref;
      }
    };
    
    struct Set3
    {
      Set3 (FastAllocator* allocator, PrimRef* prims)
      : allocator(allocator), prims(prims) {}
      
      template<typename BuildRecord>
      __forceinline NodeRef operator() (const BuildRecord& precord, const BuildRecord* crecords, NodeRef ref, NodeRef* children, const size_t num) const
      {
#if defined(DEBUG)
        // check that empty children are only at the end of the child list
        bool emptyChild = false;
        for (size_t i=0; i<num; i++) {
          emptyChild |= (children[i] == NodeRef::emptyNode);
          assert(emptyChild == (children[i] == NodeRef::emptyNode));
        }
#endif
        AABBNode_t* node = ref.getAABBNode();
        for (size_t i=0; i<num; i++) node->setRef(i,children[i]);
        
        if (unlikely(precord.alloc_barrier))
        {
          PrimRef* begin = &prims[precord.prims.begin()];
          PrimRef* end   = &prims[precord.prims.end()]; // FIXME: extended end for spatial split builder!!!!!
          size_t bytes = (size_t)end - (size_t)begin;
          allocator->addBlock(begin,bytes);
        }
        
        return ref;
      }
      
      FastAllocator* const allocator;
      PrimRef* const prims;
    };
    
    /*! Clears the node. */
    __forceinline void clear() {
      lower_x = lower_y = lower_z = pos_inf;
      upper_x = upper_y = upper_z = neg_inf;
      BaseNode_t<NodeRef,N>::clear();
    }
    
    /*! Sets bounding box and ID of child. */
    __forceinline void setRef(size_t i, const NodeRef& ref) {
      assert(i < N);
      children[i] = ref;
    }
    
    /*! Sets bounding box of child. */
    __forceinline void setBounds(size_t i, const BBox3fa& bounds)
    {
      assert(i < N);
      lower_x[i] = bounds.lower.x; lower_y[i] = bounds.lower.y; lower_z[i] = bounds.lower.z;
      upper_x[i] = bounds.upper.x; upper_y[i] = bounds.upper.y; upper_z[i] = bounds.upper.z;
    }
    
    /*! Sets bounding box and ID of child. */
    __forceinline void set(size_t i, const NodeRef& ref, const BBox3fa& bounds) {
      setBounds(i,bounds);
      children[i] = ref;
    }
    
    /*! Returns bounds of node. */
    __forceinline BBox3fa bounds() const {
      const Vec3fa lower(reduce_min(lower_x),reduce_min(lower_y),reduce_min(lower_z));
      const Vec3fa upper(reduce_max(upper_x),reduce_max(upper_y),reduce_max(upper_z));
      return BBox3fa(lower,upper);
    }
    
    /*! Returns bounds of specified child. */
    __forceinline BBox3fa bounds(size_t i) const
    {
      assert(i < N);
      const Vec3fa lower(lower_x[i],lower_y[i],lower_z[i]);
      const Vec3fa upper(upper_x[i],upper_y[i],upper_z[i]);
      return BBox3fa(lower,upper);
    }
    
    /*! Returns extent of bounds of specified child. */
    __forceinline Vec3fa extend(size_t i) const {
      return bounds(i).size();
    }
    
    /*! Returns bounds of all children (implemented later as specializations) */
    __forceinline void bounds(BBox<vfloat4>& bounds0, BBox<vfloat4>& bounds1, BBox<vfloat4>& bounds2, BBox<vfloat4>& bounds3) const;
    
    /*! swap two children of the node */
    __forceinline void swap(size_t i, size_t j)
    {
      assert(i<N && j<N);
      std::swap(children[i],children[j]);
      std::swap(lower_x[i],lower_x[j]);
      std::swap(lower_y[i],lower_y[j]);
      std::swap(lower_z[i],lower_z[j]);
      std::swap(upper_x[i],upper_x[j]);
      std::swap(upper_y[i],upper_y[j]);
      std::swap(upper_z[i],upper_z[j]);
    }

    /*! swap the children of two nodes */
    __forceinline static void swap(AABBNode_t* a, size_t i, AABBNode_t* b, size_t j)
    {
      assert(i<N && j<N);
      std::swap(a->children[i],b->children[j]);
      std::swap(a->lower_x[i],b->lower_x[j]);
      std::swap(a->lower_y[i],b->lower_y[j]);
      std::swap(a->lower_z[i],b->lower_z[j]);
      std::swap(a->upper_x[i],b->upper_x[j]);
      std::swap(a->upper_y[i],b->upper_y[j]);
      std::swap(a->upper_z[i],b->upper_z[j]);
    }

    /*! compacts a node (moves empty children to the end) */
    __forceinline static void compact(AABBNode_t* a)
    {
      /* find right most filled node */
      ssize_t j=N;
      for (j=j-1; j>=0; j--)
        if (a->child(j) != NodeRef::emptyNode)
          break;

      /* replace empty nodes with filled nodes */
      for (ssize_t i=0; i<j; i++) {
        if (a->child(i) == NodeRef::emptyNode) {
          a->swap(i,j);
          for (j=j-1; j>i; j--)
            if (a->child(j) != NodeRef::emptyNode)
              break;
        }
      }
    }
    
    /*! Returns reference to specified child */
    __forceinline       NodeRef& child(size_t i)       { assert(i<N); return children[i]; }
    __forceinline const NodeRef& child(size_t i) const { assert(i<N); return children[i]; }
    
    /*! output operator */
    friend embree_ostream operator<<(embree_ostream o, const AABBNode_t& n)
    {
      o << "AABBNode { " << embree_endl;
      o << "  lower_x " << n.lower_x << embree_endl;
      o << "  upper_x " << n.upper_x << embree_endl;
      o << "  lower_y " << n.lower_y << embree_endl;
      o << "  upper_y " << n.upper_y << embree_endl;
      o << "  lower_z " << n.lower_z << embree_endl;
      o << "  upper_z " << n.upper_z << embree_endl;
      o << "  children = ";
      for (size_t i=0; i<N; i++) o << n.children[i] << " ";
      o << embree_endl;
      o << "}" << embree_endl;
      return o;
    }
    
  public:
    vfloat<N> lower_x;           //!< X dimension of lower bounds of all N children.
    vfloat<N> upper_x;           //!< X dimension of upper bounds of all N children.
    vfloat<N> lower_y;           //!< Y dimension of lower bounds of all N children.
    vfloat<N> upper_y;           //!< Y dimension of upper bounds of all N children.
    vfloat<N> lower_z;           //!< Z dimension of lower bounds of all N children.
    vfloat<N> upper_z;           //!< Z dimension of upper bounds of all N children.
  };

  template<>
    __forceinline void AABBNode_t<NodeRefPtr<4>,4>::bounds(BBox<vfloat4>& bounds0, BBox<vfloat4>& bounds1, BBox<vfloat4>& bounds2, BBox<vfloat4>& bounds3) const {
    transpose(lower_x,lower_y,lower_z,vfloat4(zero),bounds0.lower,bounds1.lower,bounds2.lower,bounds3.lower);
    transpose(upper_x,upper_y,upper_z,vfloat4(zero),bounds0.upper,bounds1.upper,bounds2.upper,bounds3.upper);
  }
}
