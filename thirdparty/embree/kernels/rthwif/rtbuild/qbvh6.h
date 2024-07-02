// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "qnode.h"
#include "statistics.h"
#include "rtbuild.h"

namespace embree
{
  /*
    
    The QBVH6 structure defines the bounding volume hierarchy (BVH)
    that is used by the hardware. It is a BVH with 6-wide branching
    factor, and quantized bounding boxes. At the leaf level quads
    (QuadLeaf type), procedural geometries (ProceduralLeaf
    type), and instances (InstanceLeaf type) can get referenced.

   */

  inline constexpr size_t roundOffsetTo128(size_t offset) {
    return 2 * ((offset + 127) / 128);
  }

  struct QBVH6
  {
    typedef NodeRef Node;
    typedef InternalNode<InternalNode6Data> InternalNode6;

    static constexpr uint64_t rootNodeOffset = 128;
    
    static_assert(sizeof(InternalNode6) == 64, "InternalNode6 must be 64 bytes large");

    /* structure used to initialize the memory allocator inside the BVH */
    struct SizeEstimate
    {
      SizeEstimate ()
      : nodeBytes(0), leafBytes(0), proceduralBytes(0) {}

      SizeEstimate (size_t nodeBytes, size_t leafBytes, size_t proceduralBytes)
      : nodeBytes(nodeBytes), leafBytes(leafBytes), proceduralBytes(proceduralBytes) {}

      size_t bytes() const {
        return sizeof(QBVH6) + nodeBytes + leafBytes + proceduralBytes;
      }

      friend bool operator<= (SizeEstimate a, SizeEstimate b)
      {
        if (a.nodeBytes > b.nodeBytes) return false;
        if (a.leafBytes > b.leafBytes) return false;
        if (a.proceduralBytes > b.proceduralBytes) return false;
        return true;
      }

      friend SizeEstimate operator+ (const SizeEstimate& a, const SizeEstimate& b)
      {
        return SizeEstimate(a.nodeBytes + b.nodeBytes,
                            a.leafBytes + b.leafBytes,
                            a.proceduralBytes + b.proceduralBytes);
      }

      /* output operator */
      friend inline std::ostream& operator<<(std::ostream& cout, const SizeEstimate& estimate)
      {
        cout << "SizeEstimate {" << std::endl;
        cout << "  nodeBytes = " << estimate.nodeBytes << ", " << std::endl;
        cout << "  leafBytes = " << estimate.leafBytes << ", " << std::endl;
        cout << "  proceduralBytes = " << estimate.proceduralBytes << ", " << std::endl;
        return cout << "}";
      }

    public:
      size_t nodeBytes;  // bytes required to store internal nodes
      size_t leafBytes;  // bytes required to store leaf nodes
      size_t proceduralBytes;  // bytes required to store procedural leaf nodes
    };

    /* Initializes a QBVH6 node with its provided size. The memory for
     * the QBVH6 structure is overallocated and the allocation size is
     * provided to the constructor, such that the allocator of the BVH
     * can get initialized properly. */

  QBVH6(SizeEstimate size)
      : nodeDataStart((uint32_t)roundOffsetTo128(sizeof(QBVH6))), nodeDataCur(nodeDataStart),
        leafDataStart(nodeDataCur + (uint32_t)(size.nodeBytes / 64)), leafDataCur(leafDataStart),
        proceduralDataStart(leafDataCur + (uint32_t)(size.leafBytes / 64)), proceduralDataCur(proceduralDataStart),
        backPointerDataStart(proceduralDataCur + (uint32_t)(size.proceduralBytes/64)), backPointerDataEnd(backPointerDataStart)
    {
      assert(size.nodeBytes % 64 == 0);
      assert(size.leafBytes % 64 == 0);
      assert(size.proceduralBytes % 64 == 0);
      assert(size.bytes() <= (64LL << 32));

      bounds = embree::empty;
    }

    /* Returns the root node of the BVH */
    Node root() const {
      return Node(rootNodeOffset,(uint64_t)this);
    }

    /* sets root not offset to point to this specified node */
    void setRootNodeOffset(Node node) {
      assert(node.cur_prim == 0);
      uint64_t MAYBE_UNUSED rootNodeOffset1 = (uint64_t)node - (uint64_t)this;
      assert(rootNodeOffset == rootNodeOffset1);
    }

    /* check if BVH is empty */
    bool empty() const {
      return root().type == NODE_TYPE_INVALID;
    }

    /* pretty printing */
    template<typename QInternalNode>
    static void printInternalNodeStatistics(std::ostream& cout, QBVH6::Node node, uint32_t depth, uint32_t numChildren = 6);
    static void print(std::ostream& cout, QBVH6::Node node, uint32_t depth, uint32_t numChildren=6);
    void print(std::ostream& cout = std::cout) const;

    /* output operator */
    friend inline std::ostream& operator<<(std::ostream& cout, const QBVH6& qbvh) {
      qbvh.print(cout); return cout;
    }
    
    /* calculates BVH statistics */
    BVHStatistics computeStatistics() const;

    /*
       This section implements a simple allocator for BVH data. The
       BVH data is separated into two section, a section where nodes
       and leaves in mixed mode are allocated, and a section where
       only leaves are allocate in fat-leaf mode.

     */
  public:

    /* allocate data in the node memory section */
    char* allocNode(size_t bytes)
    {
      assert(bytes % 64 == 0);
      uint32_t blocks = (uint32_t)bytes / 64;
      assert(nodeDataCur + blocks <= leafDataStart);
      char* ptr = (char*)this + 64 * (size_t)nodeDataCur;
      nodeDataCur += blocks;
      return ptr;
    }

    /* allocate memory in the leaf memory section */
    char* allocLeaf(size_t bytes)
    {
      assert(bytes % 64 == 0);
      uint32_t blocks = (uint32_t)bytes / 64;      
      assert(leafDataCur + blocks <= proceduralDataStart);
      char* ptr = (char*)this + 64 * (size_t)leafDataCur;
      leafDataCur += blocks;
      return ptr;
    }

    /* allocate memory in procedural leaf memory section */
    char* allocProceduralLeaf(size_t bytes)
    {
      assert(bytes % 64 == 0);
      uint32_t blocks = (uint32_t)bytes / 64;
      assert(proceduralDataCur + blocks <= backPointerDataStart);
      char* ptr = (char*)this + 64 * (size_t)proceduralDataCur;
      proceduralDataCur += blocks;
      return ptr;
    }

    /* returns pointer to node address */
    char* nodePtr(size_t ofs) {
      return (char*)this + 64 * size_t(nodeDataStart) + ofs;
    }
    /* returns pointer to address for next leaf allocation */
    char* leafPtr() {
      return (char*)this + 64 * (size_t)leafDataCur;
    }

    /* returns the total number of bytes of the BVH */
    size_t getTotalBytes() const {
      return 64 * (size_t)backPointerDataEnd;
    }

    /* returns number of bytes available for node allocations */
    size_t getFreeNodeBytes() const {
      return 64 * (size_t)(leafDataStart - nodeDataCur);
    }

    /* returns number of bytes available for leaf allocations */
    size_t getFreeLeafBytes() const {
      return 64 * (size_t)(proceduralDataStart - leafDataCur);
    }

    /* returns number of bytes available for procedural leaf allocations */
    size_t getFreeProceduralLeafBytes() const {
      return 64 * (size_t)(backPointerDataStart - proceduralDataCur);
    }

    /* returns the bytes used by allocations */
    size_t getUsedBytes() const {
      return getTotalBytes() - getFreeNodeBytes() - getFreeLeafBytes() - getFreeProceduralLeafBytes();
    }

    bool hasBackPointers() const {
      return backPointerDataStart < backPointerDataEnd;
    }

  public:
    ze_raytracing_accel_format_internal_t rtas_format = ZE_RTAS_DEVICE_FORMAT_EXP_VERSION_1;
    uint32_t reserved1;
    BBox3f bounds;                  // bounding box of the BVH

    uint32_t nodeDataStart;         // first 64 byte block of node data
    uint32_t nodeDataCur;           // next free 64 byte block for node allocations
    uint32_t leafDataStart;         // first 64 byte block of leaf data
    uint32_t leafDataCur;           // next free 64 byte block for leaf allocations
    uint32_t proceduralDataStart;   // first 64 byte block for procedural leaf data
    uint32_t proceduralDataCur;     // next free 64 byte block for procedural leaf allocations
    uint32_t backPointerDataStart;  // first 64 byte block for back pointers
    uint32_t backPointerDataEnd;    // end of back pointer array
    uint32_t numTimeSegments = 1;
    uint32_t numPrims = 0;              // number of primitives in this BVH
    uint32_t reserved[12];
    uint64_t dispatchGlobalsPtr;
  };

  static_assert(sizeof(QBVH6) == 128, "QBVH6 must be 128 bytes large");
}

