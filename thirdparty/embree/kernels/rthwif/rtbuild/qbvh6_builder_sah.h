// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "qbvh6.h"
#include "statistics.h"
#include "quadifier.h"
#include "rtbuild.h"
#include <atomic>

#if defined(ZE_RAYTRACING)
#include "builders/priminfo.h"
#include "builders/primrefgen_presplit.h"
#include "builders/heuristic_binning_array_aligned.h"
#include "algorithms/parallel_for_for_prefix_sum.h"
#else
#include "../../builders/priminfo.h"
#include "../../builders/primrefgen_presplit.h"
#include "../../builders/heuristic_binning_array_aligned.h"
#include "../../../common/algorithms/parallel_for_for_prefix_sum.h"
#endif

namespace embree
{
  namespace isa
  {
    struct QBVH6BuilderSAH
    {
      static const size_t BVH_WIDTH = QBVH6::InternalNode6::NUM_CHILDREN;
      static const size_t MIN_LARGE_LEAF_LEVELS = 8; //!< create balanced tree of we are that many levels before the maximum tree depth
      
      /* the type of primitive that is referenced */
      enum Type { TRIANGLE=0, QUAD=1, PROCEDURAL=2, INSTANCE=3, UNKNOWN=4, NUM_TYPES=5 };

      /* check when we use spatial splits */
      static bool useSpatialSplits(ze_rtas_builder_build_quality_hint_exp_t build_quality, ze_rtas_builder_build_op_exp_flags_t build_flags) {
        return build_quality == ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH && !(build_flags & ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION);
      }

      /* BVH allocator */
      struct Allocator
      {
        Allocator() {}

        void init(char* data_in, size_t bytes_in) {
          ptr = data_in;
          end = bytes_in;
          cur.store(0);
        }

        size_t bytesAllocated() const {
          return cur.load();
        }

        __forceinline void* malloc(size_t bytes, size_t align = 16)
        {
          assert(align <= 128); //ZE_RAYTRACING_ACCELERATION_STRUCTURE_ALIGNMENT_EXT
          if (unlikely(cur.load() >= end)) return nullptr;
          const size_t extra = (align - cur) & (align-1);
          const size_t bytes_align = bytes + extra;
          const size_t cur_old = cur.fetch_add(bytes_align);
          const size_t cur_new = cur_old + bytes_align;
          if (unlikely(cur_new >= end)) return nullptr;
          return &ptr[cur_old + extra];
        }

      private:
        char* ptr;                             // data buffer pointer
        size_t end;                            // size of data buffer in bytes
        __aligned(64) std::atomic<size_t> cur; // current pointer to allocate next data block from
      };

      /* triangle data for leaf creation */
      struct Triangle
      {
        Triangle ()
          : gmask(0) {}
        
        Triangle (uint32_t i0, uint32_t i1, uint32_t i2,
                  Vec3f p0, Vec3f p1, Vec3f p2, 
                  GeometryFlags gflags,
                  uint8_t gmask)
          : i0(i0), i1(i1), i2(i2), p0(p0), p1(p1), p2(p2), gflags(gflags), gmask(gmask) {}

        __forceinline bool valid() const {
          return gmask != 0;
        }
        
        uint32_t i0,i1,i2;
        Vec3f p0,p1,p2;
        GeometryFlags gflags;
        uint8_t gmask;
      };
      
      /* quad data for leaf creation */
      struct Quad
      {
        Quad (Vec3f p0, Vec3f p1, Vec3f p2, Vec3f p3, GeometryFlags gflags, uint8_t gmask)
          : p0(p0), p1(p1), p2(p2), p3(p3), gflags(gflags), gmask(gmask) {}
        
        Vec3f p0,p1,p2,p3;
        GeometryFlags gflags;
        uint8_t gmask;
      };
      
      /* procedural data for leaf creation */
      struct Procedural
      {
        Procedural (uint8_t gmask)
          : gmask(gmask) {}
        
        PrimLeafDesc desc(uint32_t geomID) const {
          return PrimLeafDesc(0,geomID,GeometryFlags::NONE,gmask,PrimLeafDesc::TYPE_OPACITY_CULLING_ENABLED);
        }
       
        uint8_t gmask;
      };
      
      /* instance data for leaf creation */
      struct Instance
      {
        Instance (AffineSpace3f local2world, void* accel, uint8_t imask, uint32_t instanceUserID)
          : local2world(local2world), accel(accel), imask(imask), instanceUserID(instanceUserID) {}
        
        AffineSpace3f local2world;
        void* accel;
        uint8_t imask;
        uint32_t instanceUserID;
      };

      struct Stats
      {
        size_t numTriangles = 0;
        size_t numQuads = 0;
        size_t numProcedurals = 0;
        size_t numInstances = 0;
        
        /* assume some reasonable quadification rate */
        void estimate_quadification()
        {
          numQuads += (numTriangles+1)/2 + numTriangles/8;
          numTriangles = 0;
        }
        
        void estimate_presplits( double factor )
        {
          numTriangles = max(numTriangles, size_t(numTriangles*factor));
          numQuads     = max(numQuads    , size_t(numQuads*factor));
          numInstances = max(numInstances, size_t(numInstances*factor));
        }
        
        size_t size() {
          return numTriangles+numQuads+numProcedurals+numInstances;
        }
        
        size_t expected_bvh_bytes()
        {
          const size_t blocks = (size()+5)/6;
          const size_t expected_bytes   = 128 + 64*size_t(1+1.5*blocks) + numTriangles*64 + numQuads*64 + numProcedurals*8 + numInstances*128;
          const size_t bytes = 2*4096 + size_t(1.1*expected_bytes); // FIXME: FastAllocator wastes memory and always allocates 4kB per thread
          return (bytes+127)&-128;
        }
        
        size_t worst_case_bvh_bytes()
        {
          const size_t numPrimitives = size();
          const size_t blocks = (numPrimitives+5)/6;
          const size_t worst_case_bytes = 128 + 64*(1+blocks + numPrimitives) + numTriangles*64 + numQuads*64 + numProcedurals*64 + numInstances*128;
          const size_t bytes = 2*4096 + size_t(1.1*worst_case_bytes); // FIXME: FastAllocator wastes memory and always allocates 4kB per thread
          return (bytes+127)&-128;
        }
        
        size_t scratch_space_bytes() {
          return size()*sizeof(PrimRef)+64;  // 64 to align to 64 bytes
        }
      };
      
      /*! settings for SAH builder */
      struct Settings
      {
      public:
        size_t maxDepth = 27;        //!< maximum depth of BVH to build
        size_t sahBlockSize = 6;     //!< blocksize for SAH heuristic
        size_t leafSize[NUM_TYPES] = { 9,9,6,6,6 }; //!< target size of a leaf
        size_t typeSplitSize = 128;  //!< number of primitives when performing type splitting
      };
      
      /*! recursive state of builder */
      struct BuildRecord
      {
      public:
        __forceinline BuildRecord () {}
        
        __forceinline BuildRecord (size_t depth, const PrimInfoRange& prims, Type type)
          : depth(depth), prims(prims), type(type) {}
        
        __forceinline BBox3fa bounds() const { return prims.geomBounds; }
        
        __forceinline friend bool operator< (const BuildRecord& a, const BuildRecord& b) { return a.prims.size() < b.prims.size(); }
        __forceinline friend bool operator> (const BuildRecord& a, const BuildRecord& b) { return a.prims.size() > b.prims.size();  }
        
        __forceinline size_t begin() const { return prims.begin(); }
        __forceinline size_t end  () const { return prims.end(); }
        __forceinline size_t size () const { return prims.size(); }
        __forceinline bool   equalType() const { return type != UNKNOWN; }
        
        friend inline std::ostream& operator<<(std::ostream& cout, const BuildRecord& r) {
          return cout << "BuildRecord { depth = " << r.depth << ", pinfo = " << r.prims << ", type = " << r.type << " }";
        }
        
      public:
        size_t depth;        //!< Depth of the root of this subtree.
        PrimInfoRange prims; //!< The list of primitives.
        Type type;           //!< shared type when type of primitives are equal otherwise UNKNOWN
      };
      
      struct PrimRange
      {
        PrimRange () : block_delta(0), cur_prim(0) {}
        
        PrimRange (uint8_t block_delta, uint8_t start_prim = 0)
          : block_delta(block_delta), cur_prim(start_prim)
        {
          assert(block_delta < 4);
          assert(start_prim < 16);
        }
        
        friend std::ostream& operator<<(std::ostream& cout,const PrimRange& range) {
          return cout << "PrimRange { " << (int)range.block_delta << ", " << (int)range.cur_prim << " }";
        }
        
      public:
        uint8_t block_delta;
        uint8_t cur_prim;
      };
      
      struct ReductionTy
      {
        ReductionTy() : node(nullptr) {}
        ReductionTy (void* node, NodeType type, uint8_t nodeMask, PrimRange primRange)
          : node((char*)node), type(type), nodeMask(nodeMask), primRange(primRange) {}

        inline bool valid() { return node != nullptr; }

      public:
        char* node;
        NodeType type;
        uint8_t nodeMask;
        PrimRange primRange;
      };
      
      class ProceduralLeafBuilder
      {
      public:
        
        ProceduralLeafBuilder (char* data, size_t numBlocks)
          : data(data), numBlocks(numBlocks), prevBlockID(0), currBlockID(0), currProcedural(nullptr) {}
        
        ProceduralLeaf* getCurProcedural()
        {
          if (!currProcedural)
          {
            assert(numBlocks);
            currProcedural = new (data) ProceduralLeaf();
            data += sizeof(ProceduralLeaf); numBlocks--;
          }
          return currProcedural;
        }
        
        PrimRange addProcedural(uint32_t geomID, uint32_t primID, const Procedural* procedural, bool last)
        {
          assert(currProcedural);
          
          if (!currProcedural->add(procedural->desc(geomID),primID,last))
          {
            assert(numBlocks);
            currProcedural = (ProceduralLeaf*) data;
            data += sizeof(ProceduralLeaf); numBlocks--;
            
            new (currProcedural) ProceduralLeaf(procedural->desc(geomID),primID,last);
            currBlockID+=1;
          }
          
          uint32_t blockDelta = currBlockID - prevBlockID;
          uint32_t currPrim = (uint32_t)currProcedural->size() - 1;
          prevBlockID = currBlockID;
          
          return PrimRange(blockDelta,currPrim);
        }
        
      protected:
        char*                data;
        size_t               numBlocks;
        uint32_t             prevBlockID;
        uint32_t             currBlockID;
        ProceduralLeaf*      currProcedural;
      };
      
      template<typename getSizeFunc,
               typename getTypeFunc,
               typename createPrimRefArrayFunc,
               typename getTriangleFunc,
               typename getTriangleIndicesFunc,
               typename getQuadFunc,
               typename getProceduralFunc,
               typename getInstanceFunc>
      class BuilderT
      {
      public:
        static const size_t BINS = 32;
        typedef HeuristicArrayBinningSAH<PrimRef,BINS> CentroidBinner;
        
        BuilderT (Device* device,
                  const getSizeFunc& getSize,
                  const getTypeFunc& getType,
                  const createPrimRefArrayFunc& createPrimRefArray,
                  const getTriangleFunc& getTriangle,
                  const getTriangleIndicesFunc& getTriangleIndices,
                  const getQuadFunc& getQuad,
                  const getProceduralFunc& getProcedural,
                  const getInstanceFunc& getInstance,
                  void* scratch_ptr, size_t scratch_bytes,
                  ze_rtas_format_exp_t rtas_format,
                  ze_rtas_builder_build_quality_hint_exp_t build_quality,
                  ze_rtas_builder_build_op_exp_flags_t build_flags,
                  bool verbose)
          : getSize(getSize),
            getType(getType),
            createPrimRefArray(createPrimRefArray),
            getTriangle(getTriangle),
            getTriangleIndices(getTriangleIndices),
            getQuad(getQuad),
            getProcedural(getProcedural),
            getInstance(getInstance),
            prims(scratch_ptr,scratch_bytes),
            rtas_format((ze_raytracing_accel_format_internal_t)rtas_format),
            build_quality(build_quality),
            build_flags(build_flags),
            verbose(verbose) {} 
        
        ReductionTy setInternalNode(char* curAddr, size_t curBytes, NodeType nodeTy, char* childAddr,
                                    BuildRecord children[BVH_WIDTH], ReductionTy values[BVH_WIDTH], size_t numChildren)
        {
          assert(curBytes >= sizeof(QBVH6::InternalNode6));
          assert(numChildren <= QBVH6::InternalNode6::NUM_CHILDREN);
          
          BBox3f bounds = empty;
          for (size_t i=0; i<numChildren; i++)
            bounds.extend(children[i].bounds());
          
          QBVH6::InternalNode6* qnode = new (curAddr) QBVH6::InternalNode6(bounds,nodeTy);
          qnode->setChildOffset(childAddr);
          
          uint8_t nodeMask = 0;
          for (uint32_t i = 0; i < numChildren; i++)
          {
            qnode->setChild(i,children[i].bounds(),values[i].type,values[i].primRange.block_delta);
            nodeMask |= values[i].nodeMask;
          }
          qnode->nodeMask = nodeMask;
          
          return ReductionTy(curAddr, NODE_TYPE_INTERNAL, nodeMask, PrimRange(curBytes/64));
        }
        
        ReductionTy setNode(char* curAddr, size_t curBytes, NodeType nodeTy, char* childAddr,
                            BuildRecord children[BVH_WIDTH], ReductionTy values[BVH_WIDTH], size_t numChildren)
        {
          return setInternalNode(curAddr,curBytes,nodeTy,childAddr,children,values,numChildren);
        }
        
        QuadLeaf getTriangleInternal(unsigned int geomID, unsigned int primID)
        {
          QBVH6BuilderSAH::Triangle tri = getTriangle(geomID,primID);
          const Vec3f p0 = tri.p0;
          const Vec3f p1 = tri.p1;
          const Vec3f p2 = tri.p2;
          Vec3f p3 = p2;
          
          uint8_t lb0 = 0,lb1 = 0,lb2 = 0;
          uint16_t second = quadification[geomID][primID];
          
          /* handle paired triangle */
          if (second)
          {
            QBVH6BuilderSAH::Triangle tri1 = getTriangle(geomID,primID+second);
            assert(tri.gflags == tri1.gflags);
            assert(tri.gmask  == tri1.gmask );
            
            bool pair MAYBE_UNUSED = pair_triangles(Vec3<uint32_t>(tri.i0,tri.i1,tri.i2),Vec3<uint32_t>(tri1.i0,tri1.i1,tri1.i2),lb0,lb1,lb2);
            assert(pair);
            
            if (lb0 == 3) p3 = tri1.p0;
            if (lb1 == 3) p3 = tri1.p1;
            if (lb2 == 3) p3 = tri1.p2;
          }
          
          return QuadLeaf( p0,p1,p2,p3, lb0,lb1,lb2, 0, geomID, primID, primID+second, tri.gflags, tri.gmask, false );
        };
        
        QuadLeaf createQuadLeaf(Type ty, const PrimRef& prim)
        {
          const unsigned int geomID = prim.geomID();
          const unsigned int primID = prim.primID();
          
          if (ty == TRIANGLE)
            return getTriangleInternal(geomID, primID);
          else
          {
            assert(ty == QUAD);
            const Quad quad = getQuad(geomID,primID);
            return QuadLeaf(quad.p0,quad.p1,quad.p3,quad.p2, 3,2,1, 0, geomID, primID, primID, quad.gflags, quad.gmask, false );
          }
        }

        const ReductionTy createQuads(Type ty, const BuildRecord& curRecord, char* curAddr_)
        {
          QuadLeaf* curAddr = (QuadLeaf*) curAddr_;
          uint8_t nodeMask = 0;
          for (size_t i = curRecord.begin(); i < curRecord.end(); i++, curAddr++)
          {
            *curAddr = createQuadLeaf(ty,prims[i]);
            curAddr->last = (i+1) == curRecord.end();
            nodeMask |= curAddr->leafDesc.geomMask;
          }
          return ReductionTy(curAddr, NODE_TYPE_QUAD, nodeMask, PrimRange(curRecord.size()*sizeof(QuadLeaf)/64));
        }
        
        const ReductionTy createFatQuadLeaf(Type ty, const BuildRecord& curRecord, char* curAddr, size_t curBytes,
                                            BuildRecord children[BVH_WIDTH], size_t numChildren)
        {
          /*! allocate data for all children */
          char* childData = (char*) allocator.malloc(curRecord.prims.size()*sizeof(QuadLeaf), 64);

          if (!childData)
            return ReductionTy();

          /* create each child */
          ReductionTy values[BVH_WIDTH];
          for (size_t i=0, j=0; i<numChildren; i++) {
            values[i] = createQuads(ty,children[i],childData+j*sizeof(QuadLeaf));
            j += children[i].size();
          }

          return setNode(curAddr,curBytes,NODE_TYPE_QUAD,childData,children,values,numChildren);
        }

        const ReductionTy createProcedurals(const BuildRecord& curRecord, char* curAddr, size_t curBytes)
        {
          const uint32_t numPrims MAYBE_UNUSED = curRecord.size();
          assert(numPrims <= QBVH6::InternalNode6::NUM_CHILDREN);
          
          PrimRange ranges[QBVH6::InternalNode6::NUM_CHILDREN+1];
          QBVH6::InternalNode6* qnode = new (curAddr) QBVH6::InternalNode6(curRecord.bounds(),NODE_TYPE_PROCEDURAL);
          
          /* allocate data for all procedural leaves */
          size_t numGeometries = 1;
          auto prim0 = prims[curRecord.begin()];
          auto desc0 = getProcedural(prim0.geomID(),prim0.primID()).desc(prim0.geomID());
          for (size_t i=curRecord.begin()+1; i<curRecord.end(); i++) {
            auto desc1 = getProcedural(prims[i].geomID(),prims[i].primID()).desc(prims[i].geomID());
            numGeometries += desc0 != desc1;
            desc0 = desc1;
          }

          char* childData = (char*) allocator.malloc(numGeometries*sizeof(ProceduralLeaf), 64);

          if (!childData)
            return ReductionTy();

          ProceduralLeafBuilder procedural_leaf_builder(childData, numGeometries);
          ProceduralLeaf* first_procedural = procedural_leaf_builder.getCurProcedural();
          
          uint8_t nodeMask = 0;
          for (size_t i = curRecord.begin(), j=0; i < curRecord.end(); i++, j++)
          {
            const uint32_t geomID = prims[i].geomID();
            const uint32_t primID = prims[i].primID();
            auto procedural = getProcedural(geomID,primID);
            ranges[j] = procedural_leaf_builder.addProcedural(geomID,primID,&procedural,true);
            nodeMask |= procedural.gmask;
          }
          // FIXME: explicitely set ranges[numPrims]!
          
          qnode->setChildOffset(first_procedural + ranges[0].block_delta);
          qnode->nodeMask = nodeMask;
          ranges[0].block_delta = 0;
          
          for (size_t i = curRecord.begin(), j=0; i < curRecord.end(); i++, j++)
            qnode->setChild(j,prims[i].bounds(),NODE_TYPE_PROCEDURAL,ranges[j+1].block_delta,ranges[j].cur_prim);
          
          return ReductionTy(curAddr, NODE_TYPE_INTERNAL, nodeMask, PrimRange(curBytes/64));
        }

        template<typename InstanceLeaf>
        const ReductionTy createInstances(const BuildRecord& curRecord, char* curAddr, size_t curBytes)
        {
          uint32_t numPrimitives = curRecord.size();
          assert(numPrimitives <= QBVH6::InternalNode6::NUM_CHILDREN);
          
          /* allocate data for all children */
          InstanceLeaf* childData = (InstanceLeaf*) allocator.malloc(numPrimitives*sizeof(InstanceLeaf), 64);

          if (!childData)
            return ReductionTy();

          QBVH6::InternalNode6* qnode = new (curAddr) QBVH6::InternalNode6(curRecord.bounds(),NODE_TYPE_INSTANCE);
          qnode->setChildOffset(childData);
          
          uint8_t nodeMask = 0;
          for (size_t i=curRecord.begin(), c=0; i<curRecord.end(); i++, c++)
          {
            const uint32_t geomID = prims[i].geomID();
            const int64_t  rootOfs = (int32_t) prims[i].primID();
            const Instance instance = getInstance(geomID,0); //primID);
            
            uint64_t root = static_cast<QBVH6*>(instance.accel)->root();
            root += 64*rootOfs; // goto sub-BVH
            new (&childData[c]) InstanceLeaf(instance.local2world,root,geomID,instance.instanceUserID,instance.imask);
            
            qnode->setChild(c,prims[i].bounds(),NODE_TYPE_INSTANCE,sizeof(InstanceLeaf)/64,0);
            nodeMask |= instance.imask;
          }
          qnode->nodeMask = nodeMask;
          
          return ReductionTy(curAddr, NODE_TYPE_INTERNAL, nodeMask, PrimRange(curBytes/64));
        }
        
        /* finds the index of the child with largest surface area */
        int findChildWithLargestArea(BuildRecord children[BVH_WIDTH], size_t numChildren, size_t leafThreshold)
        {
          /*! find best child to split */
          float bestArea = neg_inf;
          int bestChild = -1;
          for (uint32_t i=0; i<(uint32_t)numChildren; i++)
          {
            /* ignore leaves as they cannot get split */
            if (children[i].prims.size() <= leafThreshold) continue;
            
            /* find child with largest surface area */
            const float area = halfArea(children[i].prims.geomBounds);
            if (area > bestArea)
            {
              bestArea = area;
              bestChild = i;
            }
          }
          return bestChild;
        }
        
        /* finds the index of the child with most primitives */
        int findChildWithMostPrimitives(BuildRecord children[BVH_WIDTH], size_t numChildren, size_t leafThreshold)
        {
          /* find best child with largest size */
          size_t  bestSize = 0;
          int bestChild = -1;
          for (uint32_t i=0; i<(uint32_t)numChildren; i++)
          {
            /* ignore leaves as they cannot get split */
            if (children[i].prims.size() <= leafThreshold) continue;
            
            /* remember child with largest size */
            if (children[i].prims.size() > bestSize)
            {
              bestSize = children[i].size();
              bestChild = i;
            }
          }
          return bestChild;
        }
        
        /* finds the index of the child with most primitives */
        int findChildWithNonEqualTypes(BuildRecord children[BVH_WIDTH], size_t numChildren)
        {
          for (uint32_t i=0; i<(uint32_t)numChildren; i++)
            if (!children[i].equalType())
              return i;
          
          return -1;
        }
        
        void SAHSplit(size_t depth, size_t sahBlockSize, int bestChild, BuildRecord children[BVH_WIDTH], size_t& numChildren)
        {
          PrimInfoRange linfo, rinfo;
          BuildRecord brecord = children[bestChild];
          
          /* first perform centroid binning */
          CentroidBinner centroid_binner(prims.data());
          CentroidBinner::Split bestSplit = centroid_binner.find_block_size(brecord.prims,sahBlockSize);
          
          /* now split the primitive list */
          if (bestSplit.valid())
            centroid_binner.split(bestSplit,brecord.prims,linfo,rinfo);
          
          /* the above techniques may fail, and we fall back to some brute force split in the middle */
          else
            centroid_binner.splitFallback(brecord.prims,linfo,rinfo);
          
          children[bestChild  ] = BuildRecord(depth+1, linfo, brecord.type);
          children[numChildren] = BuildRecord(depth+1, rinfo, brecord.type);
          numChildren++;
        }
        
        void TypeSplit(size_t depth, int bestChild, BuildRecord children[BVH_WIDTH], size_t& numChildren)
        {
          BuildRecord brecord = children[bestChild];
          
          PrimInfoRange linfo, rinfo;
          auto type = getType(prims[brecord.prims.begin()].geomID());
          performTypeSplit(getType,type,prims.data(),brecord.prims.get_range(),linfo,rinfo);
          
          for (size_t i=linfo.begin(); i<linfo.end(); i++)
            assert(getType(prims[i].geomID()) == getType(prims[linfo.begin()].geomID()));
          
          bool equalTy = true;
          Type rtype = getType(prims[rinfo.begin()].geomID());
          for (size_t i=rinfo.begin()+1; i<rinfo.end(); i++)
            equalTy &= rtype == getType(prims[i].geomID());
          
          children[bestChild  ] = BuildRecord(depth+1, linfo, type);
          children[numChildren] = BuildRecord(depth+1, rinfo, equalTy ? rtype : UNKNOWN);
          numChildren++;
        }
        
        void FallbackSplit(size_t depth, int bestChild, BuildRecord children[BVH_WIDTH], size_t& numChildren)
        {
          BuildRecord brecord = children[bestChild];
          
          PrimInfoRange linfo, rinfo;
          performFallbackSplit(prims.data(),brecord.prims,linfo,rinfo);
          
          children[bestChild  ] = BuildRecord(depth+1, linfo, brecord.type);
          children[numChildren] = BuildRecord(depth+1, rinfo, brecord.type);
          
          numChildren++;
        }
        
        /* creates a fat leaf, which is an internal node that only points to real leaves */
        const ReductionTy createFatLeaf(const BuildRecord& curRecord, char* curAddr, size_t curBytes)
        {
          /* this should never occur but is a fatal error */
          if (curRecord.depth > cfg.maxDepth)
            throw std::runtime_error("BVH too deep");
          
          /* there should be at least one primitive and not too many */
          assert(curRecord.size() > 0);
          assert(curRecord.size() <= cfg.leafSize[curRecord.type]);
          
          /* all primitives have to have the same type */
          Type ty = getType(prims[curRecord.begin()].geomID());
          for (size_t i=curRecord.begin(); i<curRecord.end(); i++)
            assert(getType(prims[i].geomID()) == ty);
          
          /*! initialize child list with first child */
          BuildRecord children[BVH_WIDTH];
          size_t numChildren = 0;
          
          /* fast path when we can put one primitive per child */
          if (curRecord.size() <= BVH_WIDTH)
          {
            for (size_t j=curRecord.begin(); j<curRecord.end(); j++)
            {
              CentGeomBBox3fa b(empty); b.extend_primref(prims[j]);
              children[numChildren++] = BuildRecord(curRecord.depth+1, PrimInfoRange(j,j+1,b), curRecord.type);
            }
          }
          
          else
          {
            /*! initialize child list with first child */
            children[0] = curRecord;
            numChildren = 1;
            
            /*! split until node is full */
            while (numChildren < BVH_WIDTH)
            {
              const int bestChild = findChildWithMostPrimitives(children,numChildren,1);
              if (bestChild == -1) break;
              SAHSplit(curRecord.depth,1,bestChild,children,numChildren);
            }
            
            /* fallback in case largest leaf if still too large */
            const int bestChild = findChildWithMostPrimitives(children,numChildren,1);
            if (bestChild != -1 && children[bestChild].size() > 3)
            {
              children[0] = curRecord;
              numChildren = 1;
              
              /*! perform fallback splits until node is full */
              while (numChildren < BVH_WIDTH)
              {
                const int bestChild = findChildWithMostPrimitives(children,numChildren,1);
                if (bestChild == -1) break;
                FallbackSplit(curRecord.depth,bestChild,children,numChildren);
              }
            }
          }
          
          /* sort build records for faster shadow ray traversal */
          std::sort(children,children+numChildren, [](const BuildRecord& a,const BuildRecord& b) {
                                                     return area(a.prims.geomBounds) > area(b.prims.geomBounds);
                                                   });
          
          /* create leaf of proper type */
          if (ty == TRIANGLE || ty == QUAD)
            return createFatQuadLeaf(ty, curRecord, curAddr, curBytes, children, numChildren);
          else if (ty == PROCEDURAL)
            return createProcedurals(curRecord,curAddr,curBytes);
          else if (ty == INSTANCE) {
            if (rtas_format == ZE_RTAS_DEVICE_FORMAT_EXP_VERSION_1)
              return createInstances<InstanceLeaf>(curRecord,curAddr,curBytes);
          }
          
          assert(false);
          return ReductionTy();
        }

        const ReductionTy createLargeLeaf(const BuildRecord& curRecord, char* curAddr, size_t curBytes)
        {
          /* this should never occur but is a fatal error */
          if (curRecord.depth > cfg.maxDepth)
            throw std::runtime_error("BVH too deep");
                      
          /* all primitives have to have the same type */
          Type ty MAYBE_UNUSED = getType(prims[curRecord.begin()].geomID());
          for (size_t i=curRecord.begin(); i<curRecord.end(); i++)
            assert(getType(prims[i].geomID()) == ty);
          
          /* create leaf for few primitives */
          if (curRecord.prims.size() <= cfg.leafSize[curRecord.type])
            return createFatLeaf(curRecord,curAddr,curBytes);
          
          /*! initialize child list with first child */
          ReductionTy values[BVH_WIDTH];
          BuildRecord children[BVH_WIDTH];
          size_t numChildren = 1;
          children[0] = curRecord;
          
          /* fill all children by always splitting the largest one */
          while (numChildren < BVH_WIDTH)
          {
            const int bestChild = findChildWithMostPrimitives(children,numChildren,cfg.leafSize[curRecord.type]);
            if (bestChild == -1) break;
            FallbackSplit(curRecord.depth,bestChild,children,numChildren);
          }
          
          /*! allocate data for all children */
          size_t childrenBytes = numChildren*sizeof(QBVH6::InternalNode6);
          char* childBase = (char*) allocator.malloc(childrenBytes, 64);

          if (!childBase)
            return ReductionTy();

          /* recurse into each child  and perform reduction */
          char* childPtr = childBase;
          for (size_t i=0; i<numChildren; i++) {
            values[i] = createLargeLeaf(children[i],childPtr,sizeof(QBVH6::InternalNode6));
            if (!values[i].valid()) return ReductionTy();
            childPtr += sizeof(QBVH6::InternalNode6);
          }

          return setNode(curAddr,curBytes,NODE_TYPE_INTERNAL,childBase,children,values,numChildren);
        }
        
        const ReductionTy createInternalNode(BuildRecord& curRecord, char* curAddr, size_t curBytes)
        {
          /* create leaf when threshold reached or we are too deep */
          bool createLeaf = curRecord.prims.size() <= cfg.leafSize[curRecord.type] ||
            curRecord.depth+MIN_LARGE_LEAF_LEVELS >= cfg.maxDepth;
          
          bool performTypeSplit = !curRecord.equalType() && (createLeaf || curRecord.size() <= cfg.typeSplitSize);
          
          /* check if types are really not equal when we attempt to split by type */
          if (performTypeSplit)
          {
            /* check if types are already equal */
            bool equalTy = true;
            Type type = getType(prims[curRecord.begin()].geomID());
            for (size_t i=curRecord.begin()+1; i<curRecord.end(); i++)
              equalTy &= getType(prims[i].geomID()) == type;
            
            curRecord.type = equalTy ? type : UNKNOWN;
            performTypeSplit &= !curRecord.equalType();
          }
          
          /* create leaf node */
          if (!performTypeSplit && createLeaf)
            return createLargeLeaf(curRecord,curAddr,curBytes);
          
          /*! initialize child list with first child */
          ReductionTy values[BVH_WIDTH];
          BuildRecord children[BVH_WIDTH];
          children[0] = curRecord;
          size_t numChildren = 1;
          
          /*! perform type splitting */
          if (performTypeSplit)
          {
            /*! split until node is full */
            while (numChildren < BVH_WIDTH)
            {
              const int bestChild = findChildWithNonEqualTypes(children,numChildren);
              if (bestChild == -1) break;
              TypeSplit(curRecord.depth,bestChild,children,numChildren);
            }
          }
          
          /*! perform SAH splits until node is full */
          while (numChildren < BVH_WIDTH)
          {
            const int bestChild = findChildWithLargestArea(children,numChildren,cfg.leafSize[curRecord.type]);
            if (bestChild == -1) break;
            SAHSplit(curRecord.depth,cfg.sahBlockSize,bestChild,children,numChildren);
          }
          
          /* sort build records for faster shadow ray traversal */
          std::sort(children,children+numChildren,std::less<BuildRecord>());
          
          /*! allocate data for all children */
          size_t childrenBytes = numChildren*sizeof(QBVH6::InternalNode6);
          char* childBase = (char*) allocator.malloc(childrenBytes, 64);

          if (!childBase)
            return ReductionTy();

          /* spawn tasks */
          if (curRecord.size() > 1024) // cfg.singleThreadThreshold
          {
            std::atomic<bool> success = true;
            parallel_for(size_t(0), numChildren, [&] (const range<size_t>& r) {
              if (!success) return;
              for (size_t i=r.begin(); i<r.end(); i++) {
                values[i] = createInternalNode(children[i],childBase+i*sizeof(QBVH6::InternalNode6),sizeof(QBVH6::InternalNode6));
                if (!values[i].valid()) {
                  success = false;
                  return;
                }
              }
            });

            if (!success)
              return ReductionTy();

            /* create node */
            return setNode(curAddr,curBytes,NODE_TYPE_INTERNAL,childBase,children,values,numChildren);
          }

          /* recurse into each child */
          else
          {
            /* recurse into each child */
            for (size_t i=0; i<numChildren; i++) {
              values[i] = createInternalNode(children[i],childBase+i*sizeof(QBVH6::InternalNode6),sizeof(QBVH6::InternalNode6));
              if (!values[i].valid()) return ReductionTy();
            }

            /* create node */
            return setNode(curAddr,curBytes,NODE_TYPE_INTERNAL,childBase,children,values,numChildren);
          }
        }

        const ReductionTy createEmptyNode(char* addr)
        {
          const size_t curBytes = sizeof(QBVH6::InternalNode6);
          new (addr) QBVH6::InternalNode6(NODE_TYPE_INTERNAL);
          return ReductionTy(addr, NODE_TYPE_INTERNAL, 0x00, PrimRange(curBytes/64));
        }
        
        PrimInfo createTrianglePairPrimRefArray(PrimRef* prims, const range<size_t>& r, size_t k, unsigned int geomID)
        {
          PrimInfo pinfo(empty);
          for (size_t j=r.begin(); j<r.end(); j++)
          {
            uint16_t pair = quadification[geomID][j];
            if (pair == QUADIFIER_PAIRED) continue;
            
            BBox3fa bounds = empty;
            Triangle tri0 = getTriangle(geomID,j);
            bounds.extend(tri0.p0);
            bounds.extend(tri0.p1);
            bounds.extend(tri0.p2);
            if (!tri0.valid()) continue;
            
            if (pair != QUADIFIER_TRIANGLE)
            {
              Triangle tri1 = getTriangle(geomID,j+pair);
              bounds.extend(tri1.p0);
              bounds.extend(tri1.p1);
              bounds.extend(tri1.p2);
              if (!tri1.valid()) continue;
            }

            const PrimRef prim(bounds,geomID,unsigned(j));
            pinfo.add_center2(prim);
            prims[k++] = prim;
          }
          return pinfo;
        }

        void splitTrianglePair(const PrimRef& prim, const size_t dim, const float pos, PrimRef& left_o, PrimRef& right_o) const
        {
          const uint32_t geomID = prim.geomID();
          const uint32_t primID = prim.primID();
          const uint16_t pair = quadification[geomID][primID];
          assert(pair != QUADIFIER_PAIRED);

          const Triangle tri0 = getTriangle(geomID,primID);
          const Vec3fa v[4] = { tri0.p0, tri0.p1, tri0.p2, tri0.p0 };

          BBox3fa left,right;
          splitPolygon<3>(prim.bounds(),dim,pos,v,left,right);

          if (pair != QUADIFIER_TRIANGLE)
          {
            const Triangle tri1 = getTriangle(geomID,primID+pair);
            const Vec3fa v[4] = { tri1.p0, tri1.p1, tri1.p2, tri1.p0 };

            BBox3fa left1, right1;
            splitPolygon<3>(prim.bounds(),dim,pos,v,left1,right1);

            left.extend(left1);
            right.extend(right1);
          }

          left_o  = PrimRef(left , geomID, primID);
          right_o = PrimRef(right, geomID, primID);
        }

        void splitQuad(const PrimRef& prim, const size_t dim, const float pos, PrimRef& left_o, PrimRef& right_o) const
        {
          const uint32_t geomID = prim.geomID();
          const uint32_t primID = prim.primID();
          const Quad quad = getQuad(geomID,primID);
          const Vec3fa v[5] = { quad.p0, quad.p1, quad.p2, quad.p3, quad.p0 };
          splitPolygon<4>(prim,dim,pos,v,left_o,right_o);
        }

        void splitTriangleOrQuad(const PrimRef& prim, const size_t dim, const float pos, PrimRef& left_o, PrimRef& right_o) const
        {
          switch (getType(prim.geomID())) {
          case TRIANGLE: splitTrianglePair(prim,dim,pos,left_o,right_o); break;
          case QUAD    : splitQuad        (prim,dim,pos,left_o,right_o); break;
          default: assert(false); break;
          }
        }

        void openInstance(const PrimRef& prim,
                          const unsigned int splitprims,
                          PrimRef subPrims[MAX_PRESPLITS_PER_PRIMITIVE],
                          unsigned int& numSubPrims)
        {
          struct Item
          {
            QBVH6::InternalNode6* node;
            float priority;

            Item () {}
            
            Item (QBVH6::InternalNode6* node)
              : node(node), priority(halfArea(node->bounds()))
            {
              /* fat leaves cannot get opened */
              if (node->isFatLeaf())
                priority = 0.0f;
            }

            inline bool operator< ( const Item& other) const {
              return priority < other.priority;
            }
          };
          
          const uint32_t targetSubPrims = splitprims;
          const uint32_t geomID = prim.geomID();
          const uint32_t primID MAYBE_UNUSED = prim.primID();
          assert(primID == 0); // has to be zero as we encode root offset here

          const Instance instance = getInstance(geomID,0);
          QBVH6::InternalNode6* root = static_cast<QBVH6*>(instance.accel)->root().innerNode<QBVH6::InternalNode6>();
          
          darray_t<Item,MAX_PRESPLITS_PER_PRIMITIVE> heap;
          heap.push_back(root);

          while (heap.size() + (QBVH6::InternalNode6::NUM_CHILDREN-1) <= MAX_PRESPLITS_PER_PRIMITIVE)
          {
            /* terminate when budget exceeded */
            if (heap.size() >= targetSubPrims)
              break;

            /* get top heap element */
            std::pop_heap(heap.begin(), heap.end());
            auto top = heap.back();

            /* if that happens there are only leaf nodes left that cannot get opened */
            if (top.priority == 0.0f) break;
            heap.pop_back();

            /* add all children to the heap */
            for (uint32_t i=0; i<QBVH6::InternalNode6::NUM_CHILDREN; i++)
            {
              if (!top.node->valid(i)) continue;
              heap.push_back(top.node->child(i).template innerNode<QBVH6::InternalNode6>());
              std::push_heap(heap.begin(), heap.end());
            }
          }

          /* create primrefs */
          for (size_t i=0; i<heap.size(); i++)
          {
            QBVH6::InternalNode6* node = heap[i].node;
            BBox3fa bounds = xfmBounds(instance.local2world,node->bounds());
            int64_t ofs = ((int64_t)node-(int64_t)root)/64;
            assert(ofs >= INT_MIN && ofs <= INT_MAX);
            subPrims[numSubPrims++] = PrimRef(bounds,geomID,(int32_t)ofs);
          }
        }

        float primitiveAreaTrianglePair(const PrimRef& prim)
        {
          const uint32_t geomID = prim.geomID();
          const uint32_t primID = prim.primID();
          
          const uint16_t pair = quadification[geomID][primID];
          assert(pair != QUADIFIER_PAIRED);

          const Triangle tri0 = getTriangle(geomID,primID);
          float A = areaProjectedTriangle(tri0.p0,tri0.p1,tri0.p2);
          if (pair == QUADIFIER_TRIANGLE)
            return A;

          const Triangle tri1 = getTriangle(geomID,primID+pair);
          A += areaProjectedTriangle(tri1.p0,tri1.p1,tri1.p2);
          return A;
        }

        float primitiveAreaQuad(const PrimRef& prim)
        {
          const uint32_t geomID = prim.geomID();
          const uint32_t primID = prim.primID();
          const Quad quad = getQuad(geomID,primID);
          const float A0 = areaProjectedTriangle(quad.p0,quad.p1,quad.p3);
          const float A1 = areaProjectedTriangle(quad.p2,quad.p3,quad.p1);
          return A0+A1;
        }

        float primitiveAreaInstance(const PrimRef& prim) {
          return halfArea(prim.bounds());
        }

        float primitiveArea(const PrimRef& prim)
        {
          switch (getType(prim.geomID())) {
          case TRIANGLE: return primitiveAreaTrianglePair(prim);
          case QUAD    : return primitiveAreaQuad(prim);
          case INSTANCE: return primitiveAreaInstance(prim);
          default      : return 0.0f;
          }
        }

        ReductionTy build(uint32_t numGeometries, PrimInfo& pinfo_o, char* root)
        {
          double t1 = verbose ? getSeconds() : 0.0;

          /* quadify all triangles */
          ParallelForForPrefixSumState<PrimInfo> pstate;
          pstate.init(numGeometries,getSize,size_t(1024));
          PrimInfo pinfo = parallel_for_for_prefix_sum0_( pstate, size_t(1), getSize, PrimInfo(empty), [&](size_t geomID, const range<size_t>& r, size_t k) -> PrimInfo {
            if (getType(geomID) == QBVH6BuilderSAH::TRIANGLE)
              return PrimInfo(pair_triangles(geomID,(QuadifierType*) quadification[geomID].data(), r.begin(), r.end(), getTriangleIndices));
            else
              return PrimInfo(r.size());
          }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });

          double t2 = verbose ? getSeconds() : 0.0;
          if (verbose) std::cout << "quadification: " << std::setw(10) << (t2-t1)*1000.0 << "ms, " << std::endl; //<< std::setw(10) << 1E-6*double(numTriangles)/(t2-t1) << " Mtris/s" << std::endl;

          size_t numPrimitives = pinfo.size();
          
          /* first try */
          //pstate.init(numGeometries,getSize,size_t(1024));
          pinfo = parallel_for_for_prefix_sum1_( pstate, size_t(1), getSize, PrimInfo(empty), [&](size_t geomID, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo {
            if (getType(geomID) == QBVH6BuilderSAH::TRIANGLE)
              return createTrianglePairPrimRefArray(prims.data(),r,base.size(),(unsigned)geomID);
            else
              return createPrimRefArray(prims,BBox1f(0,1),r,base.size(),(unsigned)geomID);
          }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });

          double t3 = verbose ? getSeconds() : 0.0;
          if (verbose) std::cout << "primrefgen   : " << std::setw(10) << (t3-t2)*1000.0 << "ms, " << std::setw(10) << 1E-6*double(numPrimitives)/(t3-t2) << " Mprims/s" << std::endl;
          
          /* if we need to filter out geometry, run again */
          if (pinfo.size() != numPrimitives)
          {
            numPrimitives = pinfo.size();
            
            pinfo = parallel_for_for_prefix_sum1_( pstate, size_t(1), getSize, PrimInfo(empty), [&](size_t geomID, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo {
              if (getType(geomID) == QBVH6BuilderSAH::TRIANGLE) {
                return createTrianglePairPrimRefArray(prims.data(),r,base.size(),(unsigned)geomID);
              }
              else                                                                               
                return createPrimRefArray(prims,BBox1f(0,1),r,base.size(),(unsigned)geomID);
            }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
          }
          assert(pinfo.size() == numPrimitives);
          
          double t4 = verbose ? getSeconds() : 0.0;
          if (verbose) std::cout << "primrefgen2  : " << std::setw(10) << (t4-t3)*1000.0 << "ms, " << std::setw(10) << 1E-6*double(numPrimitives)/(t4-t3) << " Mprims/s" << std::endl;
          
          /* perform pre-splitting */
          if (useSpatialSplits(build_quality,build_flags) &&  numPrimitives)
          {
            auto splitter = [this] (const PrimRef& prim, const size_t dim, const float pos, PrimRef& left_o, PrimRef& right_o) {
              splitTriangleOrQuad(prim,dim,pos,left_o,right_o);
            };

            auto splitter1 = [&] (const PrimRef& prim,
                                  const unsigned int splitprims,
                                  const SplittingGrid& grid,
                                  PrimRef subPrims[MAX_PRESPLITS_PER_PRIMITIVE],
                                  unsigned int& numSubPrims)
            {
              if (getType(prim.geomID()) == QBVH6BuilderSAH::INSTANCE) {
                openInstance(prim,splitprims,subPrims,numSubPrims);
              } else {
                splitPrimitive(splitter,prim,splitprims,grid,subPrims,numSubPrims);
              }
            };

            auto primitiveArea1 = [this] (const PrimRef& prim) -> float {
              return primitiveArea(prim);
            };
            
            pinfo = createPrimRefArray_presplit(numPrimitives, prims, pinfo, splitter1, primitiveArea1);
          }

          /* exit early if scene is empty */
          if (pinfo.size() == 0) {
            pinfo_o = pinfo;
            return createEmptyNode(root);
          }
          
          /* build hierarchy */
          BuildRecord record(1,pinfo,UNKNOWN);
          ReductionTy r = createInternalNode(record,root,sizeof(QBVH6::InternalNode6));
          
          double t5 = verbose ? getSeconds() : 0.0;
          if (verbose) std::cout << "bvh_build    : " << std::setw(10) << (t5-t4)*1000.0 << "ms, " << std::setw(10) << 1E-6*double(numPrimitives)/(t5-t4) << " Mprims/s" << std::endl;

          pinfo_o = pinfo;
          return r;
        }

        bool build(size_t numGeometries, char* accel, size_t bytes, BBox3f* boundsOut, size_t* accelBufferBytesOut, void* dispatchGlobalsPtr)
        {
          double t0 = verbose ? getSeconds() : 0.0;

          Stats stats;
          size_t numPrimitives = 0;
          quadification.resize(numGeometries);
          for (size_t geomID=0; geomID<numGeometries; geomID++)
          {
            const uint32_t N = getSize(geomID);
            numPrimitives += N;
            if (N == 0) continue;

            switch (getType(geomID)) {
            case QBVH6BuilderSAH::TRIANGLE  :
              stats.numTriangles += numPrimitives;
              quadification[geomID].resize(N);
              break;
            case QBVH6BuilderSAH::QUAD      : stats.numQuads += N; break;
            case QBVH6BuilderSAH::PROCEDURAL: stats.numProcedurals += N; break;
            case QBVH6BuilderSAH::INSTANCE  : stats.numInstances += N; break;
            default: assert(false); break;
            }
          }

          stats.estimate_presplits(1.2);
          size_t worstCaseBytes = stats.worst_case_bvh_bytes();
          if (accelBufferBytesOut) *accelBufferBytesOut = std::min(std::max(bytes+64,size_t(1.2*bytes)), worstCaseBytes);

          prims.resize(numPrimitives);
          
          double t1 = verbose ? getSeconds() : 0.0;
          if (verbose) std::cout << "scene_size   : " << std::setw(10) << (t1-t0)*1000.0 << "ms" << std::endl;

          PrimInfo pinfo;
          BBox3f bounds = empty;

          if (verbose) std::cout << "trying BVH build with " << bytes << " bytes" << std::endl;
            
          /* allocate BVH memory */
          allocator.init(accel,bytes);
          allocator.malloc(128); // header

          uint32_t numRoots = 1;
          QBVH6::InternalNode6* roots = (QBVH6::InternalNode6*) allocator.malloc(numRoots*sizeof(QBVH6::InternalNode6),64);
          assert(roots);

          /* build BVH static BVH */
          QBVH6::InternalNode6* root = roots+0;
          ReductionTy r = build(numGeometries,pinfo,(char*)root);

          /* check if build failed */
          if (!r.valid()) {
            return false;
          }

          bounds.extend(pinfo.geomBounds);

          if (boundsOut) *boundsOut = bounds;
          if (accelBufferBytesOut)
            *accelBufferBytesOut = allocator.bytesAllocated();

          /* fill QBVH6 header */
          QBVH6* qbvh = new (accel) QBVH6(QBVH6::SizeEstimate());
          qbvh->rtas_format = rtas_format;
          qbvh->numPrims = 0; //numPrimitives;
          uint64_t rootNodeOffset = QBVH6::Node((char*)(r.node - (char*)qbvh), r.type, r.primRange.cur_prim);
          assert(rootNodeOffset == QBVH6::rootNodeOffset);
          _unused(rootNodeOffset);
          qbvh->bounds = bounds;
          qbvh->numTimeSegments = 1; 
          qbvh->dispatchGlobalsPtr = (uint64_t) dispatchGlobalsPtr;

#if 0
          BVHStatistics stats = qbvh->computeStatistics();
          stats.print(std::cout);
          stats.print_raw(std::cout);
          qbvh->print();

          /*std::cout << "#define bvh_bytes " << bytes << std::endl;
          std::cout << "const unsigned char bvh_data[bvh_bytes] = {";
          for (size_t i=0; i<bytes; i++) {
            if (i % 32 == 0) std::cout << std::endl << "  ";
            std::cout << "0x" << std::hex << std::setw(2) << std::setfill('0') << (unsigned)((unsigned char*)accel)[i] << ", ";
          }
          std::cout << std::endl << "};" << std::endl;*/
#endif
          return true;
        }
        
      private:
        const getSizeFunc getSize;
        const getTypeFunc getType;
        const createPrimRefArrayFunc createPrimRefArray;
        const getTriangleFunc getTriangle;
        const getTriangleIndicesFunc getTriangleIndices;
        const getQuadFunc getQuad;
        const getProceduralFunc getProcedural;
        const getInstanceFunc getInstance;
        Settings cfg;
        evector<PrimRef> prims;
        Allocator allocator;
        std::vector<std::vector<uint16_t>> quadification;
        ze_raytracing_accel_format_internal_t rtas_format;
        ze_rtas_builder_build_quality_hint_exp_t build_quality;
        ze_rtas_builder_build_op_exp_flags_t build_flags;
        bool verbose;
        
      };

      template<typename getSizeFunc,
               typename getTypeFunc>
       
      static void estimateSize(size_t numGeometries,
                               const getSizeFunc& getSize,
                               const getTypeFunc& getType,
                               ze_rtas_format_exp_t rtas_format,
                               ze_rtas_builder_build_quality_hint_exp_t build_quality,
                               ze_rtas_builder_build_op_exp_flags_t build_flags,
                               size_t& expectedBytes,
                               size_t& worstCaseBytes,
                               size_t& scratchBytes)
      {
        Stats stats;
        for (size_t geomID=0; geomID<numGeometries; geomID++)
        {
          uint32_t numPrimitives = getSize(geomID);
          if (numPrimitives == 0) continue;
          
          switch (getType(geomID)) {
          default: assert(false); break;
          case QBVH6BuilderSAH::TRIANGLE  : stats.numTriangles += numPrimitives; break;
          case QBVH6BuilderSAH::QUAD      : stats.numQuads += numPrimitives; break;
          case QBVH6BuilderSAH::PROCEDURAL: stats.numProcedurals += numPrimitives; break;
          case QBVH6BuilderSAH::INSTANCE  : stats.numInstances += numPrimitives; break;
          };
        }
        
        if (useSpatialSplits(build_quality,build_flags))
          stats.estimate_presplits(1.2);
        
        worstCaseBytes = stats.worst_case_bvh_bytes();
        scratchBytes = stats.scratch_space_bytes();
        stats.estimate_quadification();
        expectedBytes = stats.expected_bvh_bytes();
      }      

       template<typename getSizeFunc,
               typename getTypeFunc,
               typename createPrimRefArrayFunc,
               typename getTriangleFunc,
               typename getTriangleIndicesFunc,
               typename getQuadFunc,
               typename getProceduralFunc,
               typename getInstanceFunc>
       
      static bool build(size_t numGeometries,
                          Device* device,
                          const getSizeFunc& getSize,
                          const getTypeFunc& getType,
                          const createPrimRefArrayFunc& createPrimRefArray,
                          const getTriangleFunc& getTriangle,
                          const getTriangleIndicesFunc& getTriangleIndices,
                          const getQuadFunc& getQuad,
                          const getProceduralFunc& getProcedural,
                          const getInstanceFunc& getInstance,
                          char* accel_ptr, size_t accel_bytes,
                          void* scratch_ptr, size_t scratch_bytes,
                          BBox3f* boundsOut,
                          size_t* accelBufferBytesOut,
                          ze_rtas_format_exp_t rtas_format,
                          ze_rtas_builder_build_quality_hint_exp_t build_quality,
                          ze_rtas_builder_build_op_exp_flags_t build_flags,
                          bool verbose,
                          void* dispatchGlobalsPtr)
      {
        /* align scratch buffer to 64 bytes */
        bool scratchAligned = std::align(64,0,scratch_ptr,scratch_bytes);
        if (!scratchAligned)
          throw std::runtime_error("scratch buffer cannot get aligned");
    
        BuilderT<getSizeFunc, getTypeFunc, createPrimRefArrayFunc, getTriangleFunc, getTriangleIndicesFunc, getQuadFunc, getProceduralFunc, getInstanceFunc> builder
          (device, getSize, getType, createPrimRefArray, getTriangle, getTriangleIndices, getQuad, getProcedural, getInstance, scratch_ptr, scratch_bytes, rtas_format, build_quality, build_flags, verbose);
        
        return builder.build(numGeometries, accel_ptr, accel_bytes, boundsOut, accelBufferBytesOut, dispatchGlobalsPtr);
      }
    };
  }
}
