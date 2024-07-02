// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <iostream>

#include "leaf.h"

#if defined(__INTEL_LLVM_COMPILER) && defined(WIN32)
inline float embree_frexp(float value, int* exp)
{
   // using the Intel(R) oneAPI DPC++/C++ Compiler with -no-intel-libs results
   // in an unresolved external symbol "__imp_frexp" error and therefore we
   // provide a the manual implemetation referenced here
   // https://en.cppreference.com/w/c/numeric/math/frexp in this case
   static_assert(FLT_RADIX == 2, "custom implementation of frexp only works for base 2 floating point representations");
   *exp = (value == 0) ? 0 : (int)(1 + logb(value));
   return scalbn(value, -(*exp));
}
#endif

namespace embree
{
  /* The NodeRef structure references a node of the BVH. It stores the
     * pointer to that node as well as the node's type. If a leaf node
     * is referenced the current primitive to intersect is also
     * stored. */

  struct NodeRef
  {
    NodeRef ()
    : node(nullptr), type(NODE_TYPE_INVALID), cur_prim(0) {}
    
    NodeRef (void* node, NodeType type, uint8_t cur_prim)
    : node((char*)node), type(type), cur_prim(cur_prim)
    {
      assert(cur_prim < 16);
    }
    
    /* decode from 64 bit encoding used in MemRay and Instances */
    NodeRef (uint64_t nodePtr, uint64_t offset = 0)
    {
      node = (char*) (nodePtr & ~(uint64_t)0xF) + offset;
      //type = NODE_TYPE_INTERNAL; // we can only reference internal nodes inside ray and instances
      type = (NodeType) (nodePtr & 0xF);
      cur_prim = 0;
    }
    
    /* 64 bit encoding used in MemRay and Instances */
    operator uint64_t() const
    {
      //assert(type == NODE_TYPE_INTERNAL);
      assert(((uint64_t)node & 0xF) == 0);
      assert(cur_prim == 0);
      return (uint64_t)node + (uint64_t) type;
    }
    
    /* returns the internal node that is referenced */
    template<typename InternalNode>
    InternalNode* innerNode() const {
      assert(type == NODE_TYPE_INTERNAL);
      return (InternalNode*)node;
    }

    /* returns the instance leaf node that is referenced */
    InstanceLeaf* leafNodeInstance() const {
      assert(type == NODE_TYPE_INSTANCE);
      return (InstanceLeaf*)node;
    }
    
    /* returns the quad leaf node that is referenced */
    QuadLeaf* leafNodeQuad() const {
      assert(type == NODE_TYPE_QUAD);
      return (QuadLeaf*)node;
    }

    /* returns the procedural leaf node that is referenced */
    ProceduralLeaf* leafNodeProcedural() const {
      assert(type == NODE_TYPE_PROCEDURAL);
      return (ProceduralLeaf*)node;
    }
    
    friend bool operator ==(const NodeRef& a, const NodeRef& b) {
      return (a.node == b.node) && (a.type == b.type) && (a.cur_prim == b.cur_prim);
    }
    
    friend bool operator !=(const NodeRef& a, const NodeRef& b) {
      return !(a == b);
    }
    
#if !defined(__RTRT_GSIM)
    friend inline std::ostream& operator<<(std::ostream& _cout, const NodeRef& node) {
      return _cout << "NodeRef { " << (void*)node.node << ", " << node.type << ", " << (int)node.cur_prim << " }";
    }
#endif
    
  public:
    char* node;           // pointer to the referenced node
    NodeType type;        // type of the node referenced
    uint8_t cur_prim : 4; // current primitive referenced in the leaf
  };

   /*

      The internal nodes of the BVH store references to 6 children and
      quantized bounds for each of these children.

      All children are stored consecutively in memory at a location
      refered to by the childOffset. To calculate the relative
      location of the i'th child the size (as encoded in blockIncr) of
      all the children with index smaller than i has to get added to
      that childOffset. The calculated offset specifies the signed
      number of 64 bytes blocks relative to the node address to reach
      the child.

      If the nodeType is INTERNAL we are in mixed mode and the type of
      each child is encoded inside the startPrim member. Otherwise we
      are in fat leaf mode and each child has the same type 'nodeType'
      and startPrim identifies the primitive where the leaf
      starts. The leaf spans all primitives from this start primitive
      to the end primitive which is marked as 'last'.

      The bounding boxes of the children are quantized into a regular
      3D grid. The world space position of the origin of that grid is
      stored at full precision in the lower member, while the step
      size is encoded in the exp_x, exp_y, and exp_z members as power
      of 2. Thus grid coordinates together with their exponent
      (xi,exp_x), (yi,exp_y), (zi,exp_z) correspond to the mantissa
      and exponent of a floating point number representation without
      leading zero. Thus the world space position of the bounding
      planes can get calculated as follows:

        x = lower.x + pow(2,exp_x) * 0.xi
        y = lower.y + pow(2,exp_y) * 0.yi
        z = lower.z + pow(2,exp_z) * 0.zi

      As the stored grid coordinates for child bounds are only
      unsigned 8-bit values, ray/box intersections can get performed
      with reduced precision.

      The node also stores a mask used for ray filtering. Only rays
      with (node.nodeMask & ray.rayMask) != 0 are traversed, all
      others are culled.

    */
  
  struct InternalNode6Data
  {
    static constexpr uint32_t NUM_CHILDREN = 6;

    Vec3f lower;          // world space origin of quantization grid
    int32_t childOffset; // offset to all children in 64B multiples 

    NodeType nodeType;    // the type of the node    
    uint8_t pad;          // unused byte

    int8_t exp_x;          // 2^exp_x is the size of the grid in x dimension
    int8_t exp_y;          // 2^exp_y is the size of the grid in y dimension
    int8_t exp_z;          // 2^exp_z is the size of the grid in z dimension
    uint8_t nodeMask;      // mask used for ray filtering

    struct ChildData
    {
      uint8_t blockIncr : 2; // size of child in 64 byte blocks
      uint8_t startPrim : 4; // start primitive in fat leaf mode or child type in mixed mode
      uint8_t pad : 2; // unused bits
    } childData[NUM_CHILDREN];

    uint8_t lower_x[NUM_CHILDREN];  // the quantized lower bounds in x-dimension
    uint8_t upper_x[NUM_CHILDREN];  // the quantized upper bounds in x-dimension
    uint8_t lower_y[NUM_CHILDREN];  // the quantized lower bounds in y-dimension
    uint8_t upper_y[NUM_CHILDREN];  // the quantized upper bounds in y-dimension
    uint8_t lower_z[NUM_CHILDREN];  // the quantized lower bounds in z-dimension
    uint8_t upper_z[NUM_CHILDREN];  // the quantized upper bounds in z-dimension
  };

  static_assert(sizeof(InternalNode6Data) == 64, "InternalNode6Data must be 64 bytes large");

  template<typename InternalNodeData>
    struct InternalNodeCommon : public InternalNodeData
  {
    using InternalNodeData::NUM_CHILDREN;
    
    InternalNodeCommon() {
    }
    
    InternalNodeCommon(NodeType type)
    {
      this->nodeType = type;
      this->childOffset = 0;
      this->nodeMask = 0xFF;
      
      for (uint32_t i = 0; i < InternalNodeData::NUM_CHILDREN; i++)
        this->childData[i] = { 0, 0, 0 };
      
      this->lower = Vec3f(0.0f);
      this->exp_x = 0;
      this->exp_y = 0;
      this->exp_z = 0;
      
      /* set all child bounds to invalid */
      for (uint32_t i = 0; i < InternalNodeData::NUM_CHILDREN; i++) {
        this->lower_x[i] = this->lower_y[i] = this->lower_z[i] = 0x80;
        this->upper_x[i] = this->upper_y[i] = this->upper_z[i] = 0x00;
      }
    }
    
    /* this function slightly enlarges bounds in order to make traversal watertight */
    static const BBox3f conservativeBox(const BBox3f box, float ulps = 1.0f) {
      const float err = ulps*std::numeric_limits<float>::epsilon() * std::max(reduce_max(abs(box.lower)), reduce_max(abs(box.upper)));
      return enlarge(box, Vec3f(err));
    }

    /* this function quantizes the provided bounds */
    const BBox3f quantize_bounds(BBox3f fbounds, Vec3f base) const
    {
      const Vec3f lower = fbounds.lower-base;
      const Vec3f upper = fbounds.upper-base;
      float qlower_x = ldexpf(lower.x, -this->exp_x + 8); 
      float qlower_y = ldexpf(lower.y, -this->exp_y + 8); 
      float qlower_z = ldexpf(lower.z, -this->exp_z + 8); 
      float qupper_x = ldexpf(upper.x, -this->exp_x + 8); 
      float qupper_y = ldexpf(upper.y, -this->exp_y + 8); 
      float qupper_z = ldexpf(upper.z, -this->exp_z + 8); 
      assert(qlower_x >= 0.0f && qlower_x <= 255.0f);
      assert(qlower_y >= 0.0f && qlower_y <= 255.0f);
      assert(qlower_z >= 0.0f && qlower_z <= 255.0f);
      assert(qupper_x >= 0.0f && qupper_x <= 255.0f);
      assert(qupper_y >= 0.0f && qupper_y <= 255.0f);
      assert(qupper_z >= 0.0f && qupper_z <= 255.0f); 
      qlower_x = min(max(floorf(qlower_x),0.0f),255.0f);
      qlower_y = min(max(floorf(qlower_y),0.0f),255.0f);
      qlower_z = min(max(floorf(qlower_z),0.0f),255.0f);
      qupper_x = min(max(ceilf(qupper_x),0.0f),255.0f);
      qupper_y = min(max(ceilf(qupper_y),0.0f),255.0f);
      qupper_z = min(max(ceilf(qupper_z),0.0f),255.0f);
      BBox3f qbounds(Vec3f(qlower_x, qlower_y, qlower_z), Vec3f(qupper_x, qupper_y, qupper_z));

      /* verify that quantized bounds are conservative */
      BBox3f dbounds = dequantize_bounds(qbounds, base);
      dbounds.lower.x -= 2.0f*float(ulp) * (fabs(base.x) + ldexpf(255.0f,this->exp_x-8));
      dbounds.lower.y -= 2.0f*float(ulp) * (fabs(base.y) + ldexpf(255.0f,this->exp_y-8));
      dbounds.lower.z -= 2.0f*float(ulp) * (fabs(base.z) + ldexpf(255.0f,this->exp_z-8));
      dbounds.upper.x += 2.0f*float(ulp) * (fabs(base.x) + ldexpf(255.0f,this->exp_x-8));
      dbounds.upper.y += 2.0f*float(ulp) * (fabs(base.y) + ldexpf(255.0f,this->exp_y-8));
      dbounds.upper.z += 2.0f*float(ulp) * (fabs(base.z) + ldexpf(255.0f,this->exp_z-8));
      assert(subset(fbounds, dbounds));

      return qbounds;
    }
    
    /* this function de-quantizes the provided bounds */
    const BBox3f dequantize_bounds(const BBox3f& qbounds, Vec3f base) const
    {
      const float dlower_x = base.x + ldexpf(qbounds.lower.x, this->exp_x - 8);
      const float dlower_y = base.y + ldexpf(qbounds.lower.y, this->exp_y - 8);
      const float dlower_z = base.z + ldexpf(qbounds.lower.z, this->exp_z - 8);
      const float dupper_x = base.x + ldexpf(qbounds.upper.x, this->exp_x - 8);
      const float dupper_y = base.y + ldexpf(qbounds.upper.y, this->exp_y - 8);
      const float dupper_z = base.z + ldexpf(qbounds.upper.z, this->exp_z - 8);
      return BBox3f(Vec3f(dlower_x, dlower_y, dlower_z), Vec3f(dupper_x, dupper_y, dupper_z));
    }
    
    /* Determines if a child is valid. We have only to look at the
     * topmost bit of lower_x and upper_x to determine if child is
     * valid */
    bool valid(int i) const {
      return !(this->lower_x[i] & 0x80) || (this->upper_x[i] & 0x80);
    }
    
    /* Determines if the node is in fat leaf mode. */
    bool isFatLeaf() const {
      return this->nodeType != NODE_TYPE_MIXED;
    }
    
    /* Sets the offset to the child memory. */
    void setChildOffset(void* childDataPtr)
    {
      int64_t childDataOffset = childDataPtr ? (char*)childDataPtr - (char*)this : 0;
      assert(childDataOffset % 64 == 0);
      assert((int64_t)(int32_t)(childDataOffset / 64) == (childDataOffset / 64));
      this->childOffset = (int32_t)(childDataOffset / 64);
    }
    
    /* Sets the type, size, and current primitive of a child */
    void setChildType(uint32_t child, NodeType childType, uint32_t block_delta, uint32_t cur_prim)
    {
      // there is no need to store block_delta for last child
      if (child == NUM_CHILDREN-1) block_delta = 0;
      
      assert(block_delta < 4);
      assert(cur_prim < 16);
      
      if (isFatLeaf())
      {
        assert(this->nodeType == childType);
        this->childData[child].startPrim = cur_prim;
        this->childData[child].blockIncr = block_delta;
      }
      else
      {
        assert(cur_prim == 0);
        this->childData[child].startPrim = childType;
        this->childData[child].blockIncr = block_delta;
      }
    }
    
    void invalidateChild(uint32_t childID)
    {
      /* set child bounds to invalid */
      this->lower_x[childID] = this->lower_y[childID] = this->lower_z[childID] = 0x80;
      this->upper_x[childID] = this->upper_y[childID] = this->upper_z[childID] = 0x00;
    }

    /* Sets child bounds */
    void setChildBounds(uint32_t childID, const BBox3f& fbounds)
    {
      assert(fbounds.lower.x <= fbounds.upper.x);
      assert(fbounds.lower.y <= fbounds.upper.y);
      assert(fbounds.lower.z <= fbounds.upper.z);
      const BBox3f qbounds = quantize_bounds(conservativeBox(fbounds), this->lower);
      this->lower_x[childID] = (uint8_t)qbounds.lower.x;
      this->lower_y[childID] = (uint8_t)qbounds.lower.y;
      this->lower_z[childID] = (uint8_t)qbounds.lower.z;
      this->upper_x[childID] = (uint8_t)qbounds.upper.x;
      this->upper_y[childID] = (uint8_t)qbounds.upper.y;
      this->upper_z[childID] = (uint8_t)qbounds.upper.z;
      assert(valid(childID));
    }
    
    /* Sets an entire child, including bounds, type, size, and referenced primitive. */
    void setChild(uint32_t childID, const BBox3f& fbounds, NodeType type, uint32_t block_delta, uint32_t cur_prim = 0)
    {
      setChildType(childID, type, block_delta, cur_prim);
      setChildBounds(childID, fbounds);
    }
    
    /* Calculates the byte offset to the child. The offset is
     * relative to the address this node. */
    int64_t getChildOffset(uint32_t childID) const
    {
      int64_t ofs = this->childOffset;
      for (uint32_t j = 0; j < childID; j++)
        ofs += this->childData[j].blockIncr;
      return 64 * ofs;
    }
    
    /* Returns the type of the child. In fat leaf mode the type is
     * shared between all children, otherwise a per-child type is
     * encoded inside the startPrim member for each child. */
    NodeType getChildType(uint32_t childID) const
    {
      if (isFatLeaf())
        return this->nodeType;
      
      else
        return (NodeType)(this->childData[childID].startPrim);
    }
    
    /* Returns the start primitive of a child. In case of children
     * in fat-leaf mode, all children are leaves, and the start
     * primitive specifies the primitive in a leaf block where the
     * leaf start. */
    uint32_t getChildStartPrim(uint32_t childID) const
    {
      if (isFatLeaf())
        return  this->childData[childID].startPrim;
      
      else
        return 0;
    }
    
    /* Returns a node reference for the given child. This reference
     * includes the node pointer, type, and start primitive. */
    NodeRef child(void* This, int childID) const {
      return NodeRef((char*)This + getChildOffset(childID), getChildType(childID), getChildStartPrim(childID));
    }
    
    NodeRef child(int i) const {
      return child((void*)this, i);
    }
  };

  template<typename QInternalNode>
    struct InternalNode : public InternalNodeCommon<QInternalNode>
  {
    using InternalNodeCommon<QInternalNode>::valid;
    using InternalNodeCommon<QInternalNode>::getChildType;
    using InternalNodeCommon<QInternalNode>::getChildOffset;
    using InternalNodeCommon<QInternalNode>::getChildStartPrim;
    using InternalNodeCommon<QInternalNode>::conservativeBox;
    using InternalNodeCommon<QInternalNode>::dequantize_bounds;
    using InternalNodeCommon<QInternalNode>::NUM_CHILDREN;
    
    InternalNode() {
    }
    
    InternalNode (NodeType type)
      : InternalNodeCommon<QInternalNode>(type) {}

    /* Constructs an internal node. The quantization grid gets
     * initialized from the provided parent bounds. */
    InternalNode (BBox3f box, NodeType type = NODE_TYPE_MIXED)
      : InternalNode(type)
    {
      setNodeBounds(box);
    }

    void setNodeBounds(BBox3f box)
    {
      /* initialize quantization grid */
      box = conservativeBox(box);
      const float _ulp = std::numeric_limits<float>::epsilon();
      const float up = 1.0f + float(_ulp);
      Vec3f len = box.size() * up;
      this->lower = box.lower;
#if defined(__INTEL_LLVM_COMPILER) && defined(WIN32)
      int _exp_x; float mant_x = embree_frexp(len.x, &_exp_x); _exp_x += (mant_x > 255.0f / 256.0f);
      int _exp_y; float mant_y = embree_frexp(len.y, &_exp_y); _exp_y += (mant_y > 255.0f / 256.0f);
      int _exp_z; float mant_z = embree_frexp(len.z, &_exp_z); _exp_z += (mant_z > 255.0f / 256.0f);
#else
      int _exp_x; float mant_x = frexp(len.x, &_exp_x); _exp_x += (mant_x > 255.0f / 256.0f);
      int _exp_y; float mant_y = frexp(len.y, &_exp_y); _exp_y += (mant_y > 255.0f / 256.0f);
      int _exp_z; float mant_z = frexp(len.z, &_exp_z); _exp_z += (mant_z > 255.0f / 256.0f);
#endif
      _exp_x = max(-128,_exp_x); // enlarge too tight bounds
      _exp_y = max(-128,_exp_y);
      _exp_z = max(-128,_exp_z);
      this->exp_x = _exp_x; assert(_exp_x >= -128 && _exp_x <= 127);
      this->exp_y = _exp_y; assert(_exp_y >= -128 && _exp_y <= 127);
      this->exp_z = _exp_z; assert(_exp_z >= -128 && _exp_z <= 127);
    }
    
    /* dequantizes the bounds of the specified child */
    const BBox3f bounds(uint32_t childID) const
    {
      return dequantize_bounds(BBox3f(Vec3f(this->lower_x[childID], this->lower_y[childID], this->lower_z[childID]),
                                      Vec3f(this->upper_x[childID], this->upper_y[childID], this->upper_z[childID])),
                               this->lower);
    }

    const BBox3f bounds() const
    {
      BBox3f b = empty;
      for (size_t i=0; i<NUM_CHILDREN; i++) {
        if (!valid(i)) continue;
        b.extend(bounds(i));
      }
      return b;
    }

    void copy_to( InternalNode* dst ) const
    {
      *dst = *this;
      dst->setChildOffset((char*)this + getChildOffset(0));
    }
    
#if !defined(__RTRT_GSIM)
    
    /* output of internal node */
    void print(std::ostream& cout, uint32_t depth, bool close) const
    {
      cout << tab(depth) << "InternalNode" << NUM_CHILDREN << " {" << std::endl;
      cout << tab(depth) << "  addr = " << this << std::endl;
      cout << tab(depth) << "  childOffset = " << 64 * int64_t(this->childOffset) << std::endl;
      cout << tab(depth) << "  nodeType = " << NodeType(this->nodeType) << std::endl;
      cout << tab(depth) << "  nodeMask = " << std::bitset<8>(this->nodeMask) << std::endl;
      
      for (uint32_t i = 0; i < NUM_CHILDREN; i++)
      {
        cout << tab(depth) << "  child" << i << " = { ";
        if (valid(i))
        {
          cout << "type = " << getChildType(i);
          cout << ", offset = " << getChildOffset(i);
          cout << ", prim = " << getChildStartPrim(i);
          cout << ", bounds = " << bounds(i);
        }
        else {
          cout << "INVALID";
        }
        cout << "  }" << std::endl;
      }
      
      if (close)
        cout << tab(depth) << "}";
    }
    
    /* output operator for internal node */
    friend inline std::ostream& operator<<(std::ostream& cout, const InternalNode& node) {
      node.print(cout, 0, true); return cout;
    }
#endif
  };

  inline size_t GetInternalNodeSize(uint32_t numChildren)
  {
    if (numChildren <= 6)
      return sizeof(InternalNode6Data);
    else
      assert(false);
    return 0;
  }
    
  typedef InternalNode<InternalNode6Data> InternalNode6;
}
