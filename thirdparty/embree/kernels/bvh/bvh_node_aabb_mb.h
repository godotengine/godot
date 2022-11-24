// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh_node_base.h"

namespace embree
{
  /*! Motion Blur AABBNode */
  template<typename NodeRef, int N>
    struct AABBNodeMB_t : public BaseNode_t<NodeRef, N>
  {
    using BaseNode_t<NodeRef,N>::children;
    typedef BVHNodeRecord<NodeRef>     NodeRecord;
    typedef BVHNodeRecordMB<NodeRef>   NodeRecordMB;
    typedef BVHNodeRecordMB4D<NodeRef> NodeRecordMB4D;
    
    struct Create
    {
      template<typename BuildRecord>
      __forceinline NodeRef operator() (BuildRecord* children, const size_t num, const FastAllocator::CachedAllocator& alloc) const
      {
        AABBNodeMB_t* node = (AABBNodeMB_t*) alloc.malloc0(sizeof(AABBNodeMB_t),NodeRef::byteNodeAlignment); node->clear();
        return NodeRef::encodeNode(node);
      }
    };
    
    struct Set
    { 
      template<typename BuildRecord>
      __forceinline NodeRecordMB operator() (const BuildRecord& precord, const BuildRecord* crecords, NodeRef ref, NodeRecordMB* children, const size_t num) const
      {
#if defined(DEBUG)
        // check that empty children are only at the end of the child list
        bool emptyChild = false;
        for (size_t i=0; i<num; i++) {
          emptyChild |= (children[i].ref == NodeRef::emptyNode);
          assert(emptyChild == (children[i].ref == NodeRef::emptyNode));
        }
#endif
        AABBNodeMB_t* node = ref.getAABBNodeMB();
        
        LBBox3fa bounds = empty;
        for (size_t i=0; i<num; i++) {
          node->setRef(i,children[i].ref);
          node->setBounds(i,children[i].lbounds);
          bounds.extend(children[i].lbounds);
        }
        return NodeRecordMB(ref,bounds);
      }
    };
    
    struct SetTimeRange
    {
      __forceinline SetTimeRange(BBox1f tbounds) : tbounds(tbounds) {}
      
      template<typename BuildRecord>
      __forceinline NodeRecordMB operator() (const BuildRecord& precord, const BuildRecord* crecords, NodeRef ref, NodeRecordMB* children, const size_t num) const
      {
        AABBNodeMB_t* node = ref.getAABBNodeMB();
        
        LBBox3fa bounds = empty;
        for (size_t i=0; i<num; i++) {
          node->setRef(i, children[i].ref);
          node->setBounds(i, children[i].lbounds, tbounds);
          bounds.extend(children[i].lbounds);
        }
        return NodeRecordMB(ref,bounds);
      }
      
      BBox1f tbounds;
    };
    
    /*! Clears the node. */
    __forceinline void clear()  {
      lower_x = lower_y = lower_z = vfloat<N>(pos_inf);
      upper_x = upper_y = upper_z = vfloat<N>(neg_inf);
      lower_dx = lower_dy = lower_dz = vfloat<N>(0.0f);
      upper_dx = upper_dy = upper_dz = vfloat<N>(0.0f);
      BaseNode_t<NodeRef,N>::clear();
    }
    
    /*! Sets ID of child. */
    __forceinline void setRef(size_t i, NodeRef ref) {
      children[i] = ref;
    }
    
    /*! Sets bounding box of child. */
    __forceinline void setBounds(size_t i, const BBox3fa& bounds0_i, const BBox3fa& bounds1_i)
    {
      /*! for empty bounds we have to avoid inf-inf=nan */
      BBox3fa bounds0(min(bounds0_i.lower,Vec3fa(+FLT_MAX)),max(bounds0_i.upper,Vec3fa(-FLT_MAX)));
      BBox3fa bounds1(min(bounds1_i.lower,Vec3fa(+FLT_MAX)),max(bounds1_i.upper,Vec3fa(-FLT_MAX)));
      bounds0 = bounds0.enlarge_by(4.0f*float(ulp));
      bounds1 = bounds1.enlarge_by(4.0f*float(ulp));
      Vec3fa dlower = bounds1.lower-bounds0.lower;
      Vec3fa dupper = bounds1.upper-bounds0.upper;
      
      lower_x[i] = bounds0.lower.x; lower_y[i] = bounds0.lower.y; lower_z[i] = bounds0.lower.z;
      upper_x[i] = bounds0.upper.x; upper_y[i] = bounds0.upper.y; upper_z[i] = bounds0.upper.z;
      
      lower_dx[i] = dlower.x; lower_dy[i] = dlower.y; lower_dz[i] = dlower.z;
      upper_dx[i] = dupper.x; upper_dy[i] = dupper.y; upper_dz[i] = dupper.z;
    }
    
    /*! Sets bounding box of child. */
    __forceinline void setBounds(size_t i, const LBBox3fa& bounds) {
      setBounds(i, bounds.bounds0, bounds.bounds1);
    }
    
    /*! Sets bounding box of child. */
    __forceinline void setBounds(size_t i, const LBBox3fa& bounds, const BBox1f& tbounds) {
      setBounds(i, bounds.global(tbounds));
    }
    
    /*! Sets bounding box and ID of child. */
    __forceinline void set(size_t i, NodeRef ref, const BBox3fa& bounds) {
      lower_x[i] = bounds.lower.x; lower_y[i] = bounds.lower.y; lower_z[i] = bounds.lower.z;
      upper_x[i] = bounds.upper.x; upper_y[i] = bounds.upper.y; upper_z[i] = bounds.upper.z;
      children[i] = ref;
    }
    
    /*! Sets bounding box and ID of child. */
    __forceinline void set(size_t i, const NodeRecordMB4D& child)
    {
      setRef(i, child.ref);
      setBounds(i, child.lbounds, child.dt);
    }
    
    /*! Return bounding box for time 0 */
    __forceinline BBox3fa bounds0(size_t i) const {
      return BBox3fa(Vec3fa(lower_x[i],lower_y[i],lower_z[i]),
                     Vec3fa(upper_x[i],upper_y[i],upper_z[i]));
    }
    
    /*! Return bounding box for time 1 */
    __forceinline BBox3fa bounds1(size_t i) const {
      return BBox3fa(Vec3fa(lower_x[i]+lower_dx[i],lower_y[i]+lower_dy[i],lower_z[i]+lower_dz[i]),
                     Vec3fa(upper_x[i]+upper_dx[i],upper_y[i]+upper_dy[i],upper_z[i]+upper_dz[i]));
    }
    
    /*! Returns bounds of node. */
    __forceinline BBox3fa bounds() const {
      return BBox3fa(Vec3fa(reduce_min(min(lower_x,lower_x+lower_dx)),
                            reduce_min(min(lower_y,lower_y+lower_dy)),
                            reduce_min(min(lower_z,lower_z+lower_dz))),
                     Vec3fa(reduce_max(max(upper_x,upper_x+upper_dx)),
                            reduce_max(max(upper_y,upper_y+upper_dy)),
                            reduce_max(max(upper_z,upper_z+upper_dz))));
    }
    
    /*! Return bounding box of child i */
    __forceinline BBox3fa bounds(size_t i) const {
      return merge(bounds0(i),bounds1(i));
    }
    
    /*! Return linear bounding box of child i */
    __forceinline LBBox3fa lbounds(size_t i) const {
      return LBBox3fa(bounds0(i),bounds1(i));
    }
    
    /*! Return bounding box of child i at specified time */
    __forceinline BBox3fa bounds(size_t i, float time) const {
      return lerp(bounds0(i),bounds1(i),time);
    }
    
    /*! Returns the expected surface area when randomly sampling the time. */
    __forceinline float expectedHalfArea(size_t i) const {
      return lbounds(i).expectedHalfArea();
    }
    
    /*! Returns the expected surface area when randomly sampling the time. */
    __forceinline float expectedHalfArea(size_t i, const BBox1f& t0t1) const {
      return lbounds(i).expectedHalfArea(t0t1); 
    }
    
    /*! swap two children of the node */
    __forceinline void swap(size_t i, size_t j)
    {
      assert(i<N && j<N);
      std::swap(children[i],children[j]);
      
      std::swap(lower_x[i],lower_x[j]);
      std::swap(upper_x[i],upper_x[j]);
      std::swap(lower_y[i],lower_y[j]);
      std::swap(upper_y[i],upper_y[j]);
      std::swap(lower_z[i],lower_z[j]);
      std::swap(upper_z[i],upper_z[j]);
      
      std::swap(lower_dx[i],lower_dx[j]);
      std::swap(upper_dx[i],upper_dx[j]);
      std::swap(lower_dy[i],lower_dy[j]);
      std::swap(upper_dy[i],upper_dy[j]);
      std::swap(lower_dz[i],lower_dz[j]);
      std::swap(upper_dz[i],upper_dz[j]);
    }

    /*! compacts a node (moves empty children to the end) */
    __forceinline static void compact(AABBNodeMB_t* a)
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
    
    /*! stream output operator */
    friend embree_ostream operator<<(embree_ostream cout, const AABBNodeMB_t& n) 
    {
      cout << "AABBNodeMB {" << embree_endl;
      for (size_t i=0; i<N; i++) 
      {
        const BBox3fa b0 = n.bounds0(i);
        const BBox3fa b1 = n.bounds1(i);
        cout << "  child" << i << " { " << embree_endl;
        cout << "    bounds0 = " << b0 << ", " << embree_endl;
        cout << "    bounds1 = " << b1 << ", " << embree_endl;
        cout << "  }";
      }
      cout << "}";
      return cout;
    }
    
  public:
    vfloat<N> lower_x;        //!< X dimension of lower bounds of all N children.
    vfloat<N> upper_x;        //!< X dimension of upper bounds of all N children.
    vfloat<N> lower_y;        //!< Y dimension of lower bounds of all N children.
    vfloat<N> upper_y;        //!< Y dimension of upper bounds of all N children.
    vfloat<N> lower_z;        //!< Z dimension of lower bounds of all N children.
    vfloat<N> upper_z;        //!< Z dimension of upper bounds of all N children.
    
    vfloat<N> lower_dx;        //!< X dimension of lower bounds of all N children.
    vfloat<N> upper_dx;        //!< X dimension of upper bounds of all N children.
    vfloat<N> lower_dy;        //!< Y dimension of lower bounds of all N children.
    vfloat<N> upper_dy;        //!< Y dimension of upper bounds of all N children.
    vfloat<N> lower_dz;        //!< Z dimension of lower bounds of all N children.
    vfloat<N> upper_dz;        //!< Z dimension of upper bounds of all N children.
  };
}
