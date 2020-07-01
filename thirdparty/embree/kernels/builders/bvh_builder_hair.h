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

#include "../bvh/bvh.h"
#include "../geometry/primitive.h"
#include "../builders/bvh_builder_sah.h"
#include "../builders/heuristic_binning_array_aligned.h"
#include "../builders/heuristic_binning_array_unaligned.h"
#include "../builders/heuristic_strand_array.h"

#define NUM_HAIR_OBJECT_BINS 32

namespace embree
{
  namespace isa
  {
    struct BVHBuilderHair
    {
      /*! settings for builder */
      struct Settings
      {
        /*! default settings */
        Settings ()
        : branchingFactor(2), maxDepth(32), logBlockSize(0), minLeafSize(1), maxLeafSize(8), finished_range_threshold(inf) {}

      public:
        size_t branchingFactor;  //!< branching factor of BVH to build
        size_t maxDepth;         //!< maximum depth of BVH to build
        size_t logBlockSize;     //!< log2 of blocksize for SAH heuristic
        size_t minLeafSize;      //!< minimum size of a leaf
        size_t maxLeafSize;      //!< maximum size of a leaf
        size_t finished_range_threshold;  //!< finished range threshold
      };

      template<typename NodeRef,
        typename CreateAllocFunc,
        typename CreateAlignedNodeFunc,
        typename SetAlignedNodeFunc,
        typename CreateUnalignedNodeFunc,
        typename SetUnalignedNodeFunc,
        typename CreateLeafFunc,
        typename ProgressMonitor,
        typename ReportFinishedRangeFunc>

        class BuilderT
        {
          ALIGNED_CLASS_(16);
          friend struct BVHBuilderHair;

          typedef FastAllocator::CachedAllocator Allocator;
          typedef HeuristicArrayBinningSAH<PrimRef,NUM_HAIR_OBJECT_BINS> HeuristicBinningSAH;
          typedef UnalignedHeuristicArrayBinningSAH<PrimRef,NUM_HAIR_OBJECT_BINS> UnalignedHeuristicBinningSAH;
          typedef HeuristicStrandSplit HeuristicStrandSplitSAH;

          static const size_t MAX_BRANCHING_FACTOR =  8;         //!< maximum supported BVH branching factor
          static const size_t MIN_LARGE_LEAF_LEVELS = 8;         //!< create balanced tree if we are that many levels before the maximum tree depth
          static const size_t SINGLE_THREADED_THRESHOLD = 4096;  //!< threshold to switch to single threaded build

          static const size_t travCostAligned = 1;
          static const size_t travCostUnaligned = 5;
          static const size_t intCost = 6;

          BuilderT (Scene* scene,
                    PrimRef* prims,
                    const CreateAllocFunc& createAlloc,
                    const CreateAlignedNodeFunc& createAlignedNode,
                    const SetAlignedNodeFunc& setAlignedNode,
                    const CreateUnalignedNodeFunc& createUnalignedNode,
                    const SetUnalignedNodeFunc& setUnalignedNode,
                    const CreateLeafFunc& createLeaf,
                    const ProgressMonitor& progressMonitor,
                    const ReportFinishedRangeFunc& reportFinishedRange,
                    const Settings settings)

            : cfg(settings),
            prims(prims),
            createAlloc(createAlloc),
            createAlignedNode(createAlignedNode),
            setAlignedNode(setAlignedNode),
            createUnalignedNode(createUnalignedNode),
            setUnalignedNode(setUnalignedNode),
            createLeaf(createLeaf),
            progressMonitor(progressMonitor),
            reportFinishedRange(reportFinishedRange),
            alignedHeuristic(prims), unalignedHeuristic(scene,prims), strandHeuristic(scene,prims) {}

          /*! checks if all primitives are from the same geometry */
          __forceinline bool sameGeometry(const PrimInfoRange& range)
          {
            if (range.size() == 0) return true;
            unsigned int firstGeomID = prims[range.begin()].geomID();
            for (size_t i=range.begin()+1; i<range.end(); i++) {
              if (prims[i].geomID() != firstGeomID){
                return false;
              }
            }
            return true;
          }

          /*! creates a large leaf that could be larger than supported by the BVH */
          NodeRef createLargeLeaf(size_t depth, const PrimInfoRange& pinfo, Allocator alloc)
          {
            /* this should never occur but is a fatal error */
            if (depth > cfg.maxDepth)
              throw_RTCError(RTC_ERROR_UNKNOWN,"depth limit reached");

            /* create leaf for few primitives */
            if (pinfo.size() <= cfg.maxLeafSize && sameGeometry(pinfo))
              return createLeaf(prims,pinfo,alloc);

            /* fill all children by always splitting the largest one */
            PrimInfoRange children[MAX_BRANCHING_FACTOR];
            unsigned numChildren = 1;
            children[0] = pinfo;

            do {

              /* find best child with largest bounding box area */
              int bestChild = -1;
              size_t bestSize = 0;
              for (unsigned i=0; i<numChildren; i++)
              {
                /* ignore leaves as they cannot get split */
                if (children[i].size() <= cfg.maxLeafSize && sameGeometry(children[i]))
                  continue;

                /* remember child with largest size */
                if (children[i].size() > bestSize) {
                  bestSize = children[i].size();
                  bestChild = i;
                }
              }
              if (bestChild == -1) break;

              /*! split best child into left and right child */
              __aligned(64) PrimInfoRange left, right;
              if (!sameGeometry(children[bestChild])) {
                alignedHeuristic.splitByGeometry(children[bestChild],left,right);
              } else {
                alignedHeuristic.splitFallback(children[bestChild],left,right);
              }

              /* add new children left and right */
              children[bestChild] = children[numChildren-1];
              children[numChildren-1] = left;
              children[numChildren+0] = right;
              numChildren++;

            } while (numChildren < cfg.branchingFactor);

            /* create node */
            auto node = createAlignedNode(alloc);

            for (size_t i=0; i<numChildren; i++) {
              const NodeRef child = createLargeLeaf(depth+1,children[i],alloc);
              setAlignedNode(node,i,child,children[i].geomBounds);
            }

            return node;
          }

          /*! performs split */
          __noinline void split(const PrimInfoRange& pinfo, PrimInfoRange& linfo, PrimInfoRange& rinfo, bool& aligned) // FIXME: not inlined as ICC otherwise uses much stack
          {
            /* variable to track the SAH of the best splitting approach */
            float bestSAH = inf;
            const size_t blocks = (pinfo.size()+(1ull<<cfg.logBlockSize)-1ull) >> cfg.logBlockSize;
            const float leafSAH = intCost*float(blocks)*halfArea(pinfo.geomBounds);

            /* try standard binning in aligned space */
            float alignedObjectSAH = inf;
            HeuristicBinningSAH::Split alignedObjectSplit;
            if (aligned) {
              alignedObjectSplit = alignedHeuristic.find(pinfo,cfg.logBlockSize);
              alignedObjectSAH = travCostAligned*halfArea(pinfo.geomBounds) + intCost*alignedObjectSplit.splitSAH();
              bestSAH = min(alignedObjectSAH,bestSAH);
            }

            /* try standard binning in unaligned space */
            UnalignedHeuristicBinningSAH::Split unalignedObjectSplit;
            LinearSpace3fa uspace;
            float unalignedObjectSAH = inf;
            if (bestSAH > 0.7f*leafSAH) {
              uspace = unalignedHeuristic.computeAlignedSpace(pinfo);
              const PrimInfoRange sinfo = unalignedHeuristic.computePrimInfo(pinfo,uspace);
              unalignedObjectSplit = unalignedHeuristic.find(sinfo,cfg.logBlockSize,uspace);
              unalignedObjectSAH = travCostUnaligned*halfArea(pinfo.geomBounds) + intCost*unalignedObjectSplit.splitSAH();
              bestSAH = min(unalignedObjectSAH,bestSAH);
            }

            /* try splitting into two strands */
            HeuristicStrandSplitSAH::Split strandSplit;
            float strandSAH = inf;
            if (bestSAH > 0.7f*leafSAH && pinfo.size() <= 256) {
              strandSplit = strandHeuristic.find(pinfo,cfg.logBlockSize);
              strandSAH = travCostUnaligned*halfArea(pinfo.geomBounds) + intCost*strandSplit.splitSAH();
              bestSAH = min(strandSAH,bestSAH);
            }

            /* fallback if SAH heuristics failed */
            if (unlikely(!std::isfinite(bestSAH)))
            {
              alignedHeuristic.deterministic_order(pinfo);
              alignedHeuristic.splitFallback(pinfo,linfo,rinfo);
            }

            /* perform aligned split if this is best */
            else if (bestSAH == alignedObjectSAH) {
              alignedHeuristic.split(alignedObjectSplit,pinfo,linfo,rinfo);
            }

            /* perform unaligned split if this is best */
            else if (bestSAH == unalignedObjectSAH) {
              unalignedHeuristic.split(unalignedObjectSplit,uspace,pinfo,linfo,rinfo);
              aligned = false;
            }

            /* perform strand split if this is best */
            else if (bestSAH == strandSAH) {
              strandHeuristic.split(strandSplit,pinfo,linfo,rinfo);
              aligned = false;
            }

            /* can never happen */
            else
              assert(false);
          }

          /*! recursive build */
          NodeRef recurse(size_t depth, const PrimInfoRange& pinfo, Allocator alloc, bool toplevel, bool alloc_barrier)
          {
            /* get thread local allocator */
            if (!alloc)
              alloc = createAlloc();

            /* call memory monitor function to signal progress */
            if (toplevel && pinfo.size() <= SINGLE_THREADED_THRESHOLD)
              progressMonitor(pinfo.size());

            PrimInfoRange children[MAX_BRANCHING_FACTOR];

            /* create leaf node */
            if (depth+MIN_LARGE_LEAF_LEVELS >= cfg.maxDepth || pinfo.size() <= cfg.minLeafSize) {
              alignedHeuristic.deterministic_order(pinfo);
              return createLargeLeaf(depth,pinfo,alloc);
            }

            /* fill all children by always splitting the one with the largest surface area */
            size_t numChildren = 1;
            children[0] = pinfo;
            bool aligned = true;

            do {

              /* find best child with largest bounding box area */
              ssize_t bestChild = -1;
              float bestArea = neg_inf;
              for (size_t i=0; i<numChildren; i++)
              {
                /* ignore leaves as they cannot get split */
                if (children[i].size() <= cfg.minLeafSize)
                  continue;

                /* remember child with largest area */
                if (area(children[i].geomBounds) > bestArea) {
                  bestArea = area(children[i].geomBounds);
                  bestChild = i;
                }
              }
              if (bestChild == -1) break;

              /*! split best child into left and right child */
              PrimInfoRange left, right;
              split(children[bestChild],left,right,aligned);

              /* add new children left and right */
              children[bestChild] = children[numChildren-1];
              children[numChildren-1] = left;
              children[numChildren+0] = right;
              numChildren++;

            } while (numChildren < cfg.branchingFactor);

            NodeRef node;

            /* create aligned node */
            if (aligned)
            {
              node = createAlignedNode(alloc);

              /* spawn tasks or ... */
              if (pinfo.size() > SINGLE_THREADED_THRESHOLD)
              {
                parallel_for(size_t(0), numChildren, [&] (const range<size_t>& r) {
                    for (size_t i=r.begin(); i<r.end(); i++) {
                      const bool child_alloc_barrier = pinfo.size() > cfg.finished_range_threshold && children[i].size() <= cfg.finished_range_threshold;
                      setAlignedNode(node,i,recurse(depth+1,children[i],nullptr,true,child_alloc_barrier),children[i].geomBounds);
                      _mm_mfence(); // to allow non-temporal stores during build
                    }
                  });
              }
              /* ... continue sequentially */
              else {
                for (size_t i=0; i<numChildren; i++) {
                  const bool child_alloc_barrier = pinfo.size() > cfg.finished_range_threshold && children[i].size() <= cfg.finished_range_threshold;
                  setAlignedNode(node,i,recurse(depth+1,children[i],alloc,false,child_alloc_barrier),children[i].geomBounds);
                }
              }
            }

            /* create unaligned node */
            else
            {
              node = createUnalignedNode(alloc);

              /* spawn tasks or ... */
              if (pinfo.size() > SINGLE_THREADED_THRESHOLD)
              {
                parallel_for(size_t(0), numChildren, [&] (const range<size_t>& r) {
                    for (size_t i=r.begin(); i<r.end(); i++) {
                      const LinearSpace3fa space = unalignedHeuristic.computeAlignedSpace(children[i]);
                      const PrimInfoRange sinfo = unalignedHeuristic.computePrimInfo(children[i],space);
                      const OBBox3fa obounds(space,sinfo.geomBounds);
                      const bool child_alloc_barrier = pinfo.size() > cfg.finished_range_threshold && children[i].size() <= cfg.finished_range_threshold;
                      setUnalignedNode(node,i,recurse(depth+1,children[i],nullptr,true,child_alloc_barrier),obounds);
                      _mm_mfence(); // to allow non-temporal stores during build
                    }
                  });
              }
              /* ... continue sequentially */
              else
              {
                for (size_t i=0; i<numChildren; i++) {
                  const LinearSpace3fa space = unalignedHeuristic.computeAlignedSpace(children[i]);
                  const PrimInfoRange sinfo = unalignedHeuristic.computePrimInfo(children[i],space);
                  const OBBox3fa obounds(space,sinfo.geomBounds);
                  const bool child_alloc_barrier = pinfo.size() > cfg.finished_range_threshold && children[i].size() <= cfg.finished_range_threshold;
                  setUnalignedNode(node,i,recurse(depth+1,children[i],alloc,false,child_alloc_barrier),obounds);
                }
              }
            }

            /* reports a finished range of primrefs */
            if (unlikely(alloc_barrier))
              reportFinishedRange(pinfo);

            return node;
          }

        private:
          Settings cfg;
          PrimRef* prims;
          const CreateAllocFunc& createAlloc;
          const CreateAlignedNodeFunc& createAlignedNode;
          const SetAlignedNodeFunc& setAlignedNode;
          const CreateUnalignedNodeFunc& createUnalignedNode;
          const SetUnalignedNodeFunc& setUnalignedNode;
          const CreateLeafFunc& createLeaf;
          const ProgressMonitor& progressMonitor;
          const ReportFinishedRangeFunc& reportFinishedRange;

        private:
          HeuristicBinningSAH alignedHeuristic;
          UnalignedHeuristicBinningSAH unalignedHeuristic;
          HeuristicStrandSplitSAH strandHeuristic;
        };

      template<typename NodeRef,
        typename CreateAllocFunc,
        typename CreateAlignedNodeFunc,
        typename SetAlignedNodeFunc,
        typename CreateUnalignedNodeFunc,
        typename SetUnalignedNodeFunc,
        typename CreateLeafFunc,
        typename ProgressMonitor,
        typename ReportFinishedRangeFunc>

        static NodeRef build (const CreateAllocFunc& createAlloc,
                              const CreateAlignedNodeFunc& createAlignedNode,
                              const SetAlignedNodeFunc& setAlignedNode,
                              const CreateUnalignedNodeFunc& createUnalignedNode,
                              const SetUnalignedNodeFunc& setUnalignedNode,
                              const CreateLeafFunc& createLeaf,
                              const ProgressMonitor& progressMonitor,
                              const ReportFinishedRangeFunc& reportFinishedRange,
                              Scene* scene,
                              PrimRef* prims,
                              const PrimInfo& pinfo,
                              const Settings settings)
        {
          typedef BuilderT<NodeRef,
            CreateAllocFunc,
            CreateAlignedNodeFunc,SetAlignedNodeFunc,
            CreateUnalignedNodeFunc,SetUnalignedNodeFunc,
            CreateLeafFunc,ProgressMonitor,
            ReportFinishedRangeFunc> Builder;

          Builder builder(scene,prims,createAlloc,
                          createAlignedNode,setAlignedNode,
                          createUnalignedNode,setUnalignedNode,
                          createLeaf,progressMonitor,reportFinishedRange,settings);

          NodeRef root = builder.recurse(1,pinfo,nullptr,true,false);
          _mm_mfence(); // to allow non-temporal stores during build
          return root;
        }
    };
  }
}
