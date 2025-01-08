// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

/* include all node types */
#include "bvh_node_aabb.h"
#include "bvh_node_aabb_mb.h"
#include "bvh_node_aabb_mb4d.h"
#include "bvh_node_obb.h"
#include "bvh_node_obb_mb.h"
#include "bvh_node_qaabb.h"

namespace embree
{
  /*! flags used to enable specific node types in intersectors */
  enum BVHNodeFlags
  {
    BVH_FLAG_ALIGNED_NODE = 0x00001,
    BVH_FLAG_ALIGNED_NODE_MB = 0x00010,
    BVH_FLAG_UNALIGNED_NODE = 0x00100,
    BVH_FLAG_UNALIGNED_NODE_MB = 0x01000,
    BVH_FLAG_QUANTIZED_NODE = 0x100000,
    BVH_FLAG_ALIGNED_NODE_MB4D = 0x1000000,
    
    /* short versions */
    BVH_AN1 = BVH_FLAG_ALIGNED_NODE,
    BVH_AN2 = BVH_FLAG_ALIGNED_NODE_MB,
    BVH_AN2_AN4D = BVH_FLAG_ALIGNED_NODE_MB | BVH_FLAG_ALIGNED_NODE_MB4D,
    BVH_UN1 = BVH_FLAG_UNALIGNED_NODE,
    BVH_UN2 = BVH_FLAG_UNALIGNED_NODE_MB,
    BVH_MB = BVH_FLAG_ALIGNED_NODE_MB | BVH_FLAG_UNALIGNED_NODE_MB | BVH_FLAG_ALIGNED_NODE_MB4D,
    BVH_AN1_UN1 = BVH_FLAG_ALIGNED_NODE | BVH_FLAG_UNALIGNED_NODE,
    BVH_AN2_UN2 = BVH_FLAG_ALIGNED_NODE_MB | BVH_FLAG_UNALIGNED_NODE_MB,
    BVH_AN2_AN4D_UN2 = BVH_FLAG_ALIGNED_NODE_MB | BVH_FLAG_ALIGNED_NODE_MB4D | BVH_FLAG_UNALIGNED_NODE_MB,
    BVH_QN1 = BVH_FLAG_QUANTIZED_NODE
  };
  
  /*! Multi BVH with N children. Each node stores the bounding box of
   * it's N children as well as N child references. */
  template<int N>
    class BVHN : public AccelData
  {
    ALIGNED_CLASS_(16);
  public:
    
    /*! forward declaration of node ref type */
    typedef NodeRefPtr<N> NodeRef;
    typedef BaseNode_t<NodeRef,N> BaseNode;
    typedef AABBNode_t<NodeRef,N> AABBNode;
    typedef AABBNodeMB_t<NodeRef,N> AABBNodeMB;
    typedef AABBNodeMB4D_t<NodeRef,N> AABBNodeMB4D;
    typedef OBBNode_t<NodeRef,N> OBBNode;
    typedef OBBNodeMB_t<NodeRef,N> OBBNodeMB;
    typedef QuantizedBaseNode_t<N> QuantizedBaseNode;
    typedef QuantizedBaseNodeMB_t<N> QuantizedBaseNodeMB;
    typedef QuantizedNode_t<NodeRef,N> QuantizedNode;
    
    /*! Number of bytes the nodes and primitives are minimally aligned to.*/
    static const size_t byteAlignment = 16;
    static const size_t byteNodeAlignment = 4*N;
    
    /*! Empty node */
    static const size_t emptyNode = NodeRef::emptyNode;
    
    /*! Invalid node, used as marker in traversal */
    static const size_t invalidNode = NodeRef::invalidNode;
    static const size_t popRay      = NodeRef::popRay;
    
    /*! Maximum depth of the BVH. */
    static const size_t maxBuildDepth = 32;
    static const size_t maxBuildDepthLeaf = maxBuildDepth+8;
    static const size_t maxDepth = 2*maxBuildDepthLeaf; // 2x because of two level builder
    
    /*! Maximum number of primitive blocks in a leaf. */
    static const size_t maxLeafBlocks = NodeRef::maxLeafBlocks;
    
  public:
    
    /*! Builder interface to create allocator */
    struct CreateAlloc : public FastAllocator::Create {
      __forceinline CreateAlloc (BVHN* bvh) : FastAllocator::Create(&bvh->alloc) {}
    };
    
    typedef BVHNodeRecord<NodeRef>     NodeRecord;
    typedef BVHNodeRecordMB<NodeRef>   NodeRecordMB;
    typedef BVHNodeRecordMB4D<NodeRef> NodeRecordMB4D;

  public:
    
    /*! BVHN default constructor. */
    BVHN (const PrimitiveType& primTy, Scene* scene);
    
    /*! BVHN destruction */
    ~BVHN ();
    
    /*! clears the acceleration structure */
    void clear();
    
    /*! sets BVH members after build */
    void set (NodeRef root, const LBBox3fa& bounds, size_t numPrimitives);
    
    /*! Clears the barrier bits of a subtree. */
    void clearBarrier(NodeRef& node);
    
    /*! lays out num large nodes of the BVH */
    void layoutLargeNodes(size_t num);
    NodeRef layoutLargeNodesRecursion(NodeRef& node, const FastAllocator::CachedAllocator& allocator);
    
    /*! called by all builders before build starts */
    double preBuild(const std::string& builderName);
    
    /*! called by all builders after build ended */
    void postBuild(double t0);
    
    /*! allocator class */
    struct Allocator {
      BVHN* bvh;
      Allocator (BVHN* bvh) : bvh(bvh) {}
      __forceinline void* operator() (size_t bytes) const { 
        return bvh->alloc._threadLocal()->malloc(&bvh->alloc,bytes); 
      }
    };
    
    /*! post build cleanup */
    void cleanup() {
      alloc.cleanup();
    }
    
  public:
    
    /*! Encodes a node */
    static __forceinline NodeRef encodeNode(AABBNode* node) { return NodeRef::encodeNode(node); }
    static __forceinline NodeRef encodeNode(AABBNodeMB* node) { return NodeRef::encodeNode(node); }
    static __forceinline NodeRef encodeNode(AABBNodeMB4D* node) { return NodeRef::encodeNode(node); }
    static __forceinline NodeRef encodeNode(OBBNode* node) { return NodeRef::encodeNode(node); }
    static __forceinline NodeRef encodeNode(OBBNodeMB* node) { return NodeRef::encodeNode(node); }
    static __forceinline NodeRef encodeLeaf(void* tri, size_t num) { return NodeRef::encodeLeaf(tri,num); }
    static __forceinline NodeRef encodeTypedLeaf(void* ptr, size_t ty) { return NodeRef::encodeTypedLeaf(ptr,ty); }
    
  public:
    
    /*! Prefetches the node this reference points to */
    __forceinline static void prefetch(const NodeRef ref, int types=0)
    {
#if defined(__AVX512PF__) // MIC
      if (types != BVH_FLAG_QUANTIZED_NODE) {
        prefetchL2(((char*)ref.ptr)+0*64);
        prefetchL2(((char*)ref.ptr)+1*64);
        if ((N >= 8) || (types > BVH_FLAG_ALIGNED_NODE)) {
          prefetchL2(((char*)ref.ptr)+2*64);
          prefetchL2(((char*)ref.ptr)+3*64);
        }
        if ((N >= 8) && (types > BVH_FLAG_ALIGNED_NODE)) {
          /* KNL still needs L2 prefetches for large nodes */
          prefetchL2(((char*)ref.ptr)+4*64);
          prefetchL2(((char*)ref.ptr)+5*64);
          prefetchL2(((char*)ref.ptr)+6*64);
          prefetchL2(((char*)ref.ptr)+7*64);
        }
      }
      else
      {
        /* todo: reduce if 32bit offsets are enabled */
        prefetchL2(((char*)ref.ptr)+0*64);
        prefetchL2(((char*)ref.ptr)+1*64);
        prefetchL2(((char*)ref.ptr)+2*64);
      }
#else
      if (types != BVH_FLAG_QUANTIZED_NODE) {
        prefetchL1(((char*)ref.ptr)+0*64);
        prefetchL1(((char*)ref.ptr)+1*64);
        if ((N >= 8) || (types > BVH_FLAG_ALIGNED_NODE)) {
          prefetchL1(((char*)ref.ptr)+2*64);
          prefetchL1(((char*)ref.ptr)+3*64);
        }
        if ((N >= 8) && (types > BVH_FLAG_ALIGNED_NODE)) {
          /* deactivate for large nodes on Xeon, as it introduces regressions */
          //prefetchL1(((char*)ref.ptr)+4*64);
          //prefetchL1(((char*)ref.ptr)+5*64);
          //prefetchL1(((char*)ref.ptr)+6*64);
          //prefetchL1(((char*)ref.ptr)+7*64);
        }
      }
      else
      {
        /* todo: reduce if 32bit offsets are enabled */
        prefetchL1(((char*)ref.ptr)+0*64);
        prefetchL1(((char*)ref.ptr)+1*64);
        prefetchL1(((char*)ref.ptr)+2*64);
      }
#endif
    }
    
    __forceinline static void prefetchW(const NodeRef ref, int types=0)
    {
      embree::prefetchEX(((char*)ref.ptr)+0*64);
      embree::prefetchEX(((char*)ref.ptr)+1*64);
      if ((N >= 8) || (types > BVH_FLAG_ALIGNED_NODE)) {
        embree::prefetchEX(((char*)ref.ptr)+2*64);
        embree::prefetchEX(((char*)ref.ptr)+3*64);
      }
      if ((N >= 8) && (types > BVH_FLAG_ALIGNED_NODE)) {
        embree::prefetchEX(((char*)ref.ptr)+4*64);
        embree::prefetchEX(((char*)ref.ptr)+5*64);
        embree::prefetchEX(((char*)ref.ptr)+6*64);
        embree::prefetchEX(((char*)ref.ptr)+7*64);
      }
    }
    
    /*! bvh type information */
  public:
    const PrimitiveType* primTy;       //!< primitive type stored in the BVH
    
    /*! bvh data */
  public:
    Device* device;                    //!< device pointer
    Scene* scene;                      //!< scene pointer
    NodeRef root;                      //!< root node
    FastAllocator alloc;               //!< allocator used to allocate nodes
    
    /*! statistics data */
  public:
    size_t numPrimitives;              //!< number of primitives the BVH is build over
    size_t numVertices;                //!< number of vertices the BVH references
    
    /*! data arrays for special builders */
  public:
    std::vector<BVHN*> objects;
    vector_t<char,aligned_allocator<char,32>> subdiv_patches;
  };
  
  typedef BVHN<4> BVH4;
  typedef BVHN<8> BVH8;
}
