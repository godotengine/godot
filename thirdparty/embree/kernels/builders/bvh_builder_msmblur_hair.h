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
#include "../builders/bvh_builder_msmblur.h"
#include "../builders/heuristic_binning_array_aligned.h"
#include "../builders/heuristic_binning_array_unaligned.h"
#include "../builders/heuristic_timesplit_array.h"

namespace embree
{
  namespace isa
  {
    struct BVHBuilderHairMSMBlur
    {
      /*! settings for msmblur builder */
      struct Settings
      {
        /*! default settings */
        Settings ()
        : branchingFactor(2), maxDepth(32), logBlockSize(0), minLeafSize(1), maxLeafSize(8) {}

      public:
        size_t branchingFactor;  //!< branching factor of BVH to build
        size_t maxDepth;         //!< maximum depth of BVH to build
        size_t logBlockSize;     //!< log2 of blocksize for SAH heuristic
        size_t minLeafSize;      //!< minimum size of a leaf
        size_t maxLeafSize;      //!< maximum size of a leaf
      };

      struct BuildRecord
      {
      public:
	__forceinline BuildRecord () {}

        __forceinline BuildRecord (size_t depth)
          : depth(depth) {}

        __forceinline BuildRecord (const SetMB& prims, size_t depth)
          : depth(depth), prims(prims) {}

        __forceinline size_t size() const {
          return prims.size();
        }

      public:
	size_t depth;       //!< depth of the root of this subtree
	SetMB prims;        //!< the list of primitives
      };

      template<typename NodeRef,
        typename RecalculatePrimRef,
        typename CreateAllocFunc,
        typename CreateAlignedNodeMBFunc,
        typename SetAlignedNodeMBFunc,
        typename CreateUnalignedNodeMBFunc,
        typename SetUnalignedNodeMBFunc,
        typename CreateLeafFunc,
        typename ProgressMonitor>

        class BuilderT
        {
          ALIGNED_CLASS_(16);

          static const size_t MAX_BRANCHING_FACTOR =  8;         //!< maximum supported BVH branching factor
          static const size_t MIN_LARGE_LEAF_LEVELS = 8;         //!< create balanced tree if we are that many levels before the maximum tree depth
          static const size_t SINGLE_THREADED_THRESHOLD = 4096;  //!< threshold to switch to single threaded build

          typedef BVHNodeRecordMB<NodeRef> NodeRecordMB;
          typedef BVHNodeRecordMB4D<NodeRef> NodeRecordMB4D;

          typedef FastAllocator::CachedAllocator Allocator;
          typedef LocalChildListT<BuildRecord,MAX_BRANCHING_FACTOR> LocalChildList;

          typedef HeuristicMBlurTemporalSplit<PrimRefMB,RecalculatePrimRef,MBLUR_NUM_TEMPORAL_BINS> HeuristicTemporal;
          typedef HeuristicArrayBinningMB<PrimRefMB,MBLUR_NUM_OBJECT_BINS> HeuristicBinning;
          typedef UnalignedHeuristicArrayBinningMB<PrimRefMB,MBLUR_NUM_OBJECT_BINS> UnalignedHeuristicBinning;

        public:

          BuilderT (Scene* scene,
                    const RecalculatePrimRef& recalculatePrimRef,
                    const CreateAllocFunc& createAlloc,
                    const CreateAlignedNodeMBFunc& createAlignedNodeMB,
                    const SetAlignedNodeMBFunc& setAlignedNodeMB,
                    const CreateUnalignedNodeMBFunc& createUnalignedNodeMB,
                    const SetUnalignedNodeMBFunc& setUnalignedNodeMB,
                    const CreateLeafFunc& createLeaf,
                    const ProgressMonitor& progressMonitor,
                    const Settings settings)

            : cfg(settings),
            scene(scene),
            recalculatePrimRef(recalculatePrimRef),
            createAlloc(createAlloc),
            createAlignedNodeMB(createAlignedNodeMB), setAlignedNodeMB(setAlignedNodeMB),
            createUnalignedNodeMB(createUnalignedNodeMB), setUnalignedNodeMB(setUnalignedNodeMB),
            createLeaf(createLeaf),
            progressMonitor(progressMonitor),
            unalignedHeuristic(scene),
            temporalSplitHeuristic(scene->device,recalculatePrimRef) {}

        private:

          /*! checks if all primitives are from the same geometry */
          __forceinline bool sameGeometry(const SetMB& set)
          {
            mvector<PrimRefMB>& prims = *set.prims;
            unsigned int firstGeomID = prims[set.begin()].geomID();
            for (size_t i=set.begin()+1; i<set.end(); i++) {
              if (prims[i].geomID() != firstGeomID){
                return false;
              }
            }
            return true;
          }
          
          /*! performs some split if SAH approaches fail */
          void splitFallback(const SetMB& set, SetMB& lset, SetMB& rset)
          {
            mvector<PrimRefMB>& prims = *set.prims;

            const size_t begin = set.begin();
            const size_t end   = set.end();
            const size_t center = (begin + end)/2;

            PrimInfoMB linfo = empty;
            for (size_t i=begin; i<center; i++)
              linfo.add_primref(prims[i]);

            PrimInfoMB rinfo = empty;
            for (size_t i=center; i<end; i++)
              rinfo.add_primref(prims[i]);

            new (&lset) SetMB(linfo,set.prims,range<size_t>(begin,center),set.time_range);
            new (&rset) SetMB(rinfo,set.prims,range<size_t>(center,end  ),set.time_range);
          }

          void splitByGeometry(const SetMB& set, SetMB& lset, SetMB& rset)
          {
            assert(set.size() > 1);
            const size_t begin = set.begin();
            const size_t end   = set.end();
            PrimInfoMB linfo(empty);
            PrimInfoMB rinfo(empty);
            unsigned int geomID = (*set.prims)[begin].geomID();
            size_t center = serial_partitioning(set.prims->data(),begin,end,linfo,rinfo,
                                                [&] ( const PrimRefMB& prim ) { return prim.geomID() == geomID; },
                                                [ ] ( PrimInfoMB& a, const PrimRefMB& ref ) { a.add_primref(ref); });

            new (&lset) SetMB(linfo,set.prims,range<size_t>(begin,center),set.time_range);
            new (&rset) SetMB(rinfo,set.prims,range<size_t>(center,end  ),set.time_range);
          }

          /*! creates a large leaf that could be larger than supported by the BVH */
          NodeRecordMB4D createLargeLeaf(BuildRecord& current, Allocator alloc)
          {
            /* this should never occur but is a fatal error */
            if (current.depth > cfg.maxDepth)
              throw_RTCError(RTC_ERROR_UNKNOWN,"depth limit reached");

            /* special case when directly creating leaf without any splits that could shrink time_range */
            bool force_split = false;
            if (current.depth == 1 && current.size() > 0)
            {
              BBox1f c = empty;
              BBox1f p = current.prims.time_range;
              for (size_t i=current.prims.begin(); i<current.prims.end(); i++) {
                mvector<PrimRefMB>& prims = *current.prims.prims;
                c.extend(prims[i].time_range);
              }
              
              force_split = c.lower > p.lower || c.upper < p.upper;
            }

            /* create leaf for few primitives */
            if (current.size() <= cfg.maxLeafSize && sameGeometry(current.prims) && !force_split)
              return createLeaf(current.prims,alloc);

            /* fill all children by always splitting the largest one */
            LocalChildList children(current);

            do {

              /* find best child with largest bounding box area */
              int bestChild = -1;
              size_t bestSize = 0;
              for (unsigned i=0; i<children.size(); i++)
              {
                /* ignore leaves as they cannot get split */
                if (children[i].size() <= cfg.maxLeafSize && sameGeometry(children[i].prims) && !force_split)
                  continue;

                force_split = false;

                /* remember child with largest size */
                if (children[i].size() > bestSize) {
                  bestSize = children[i].size();
                  bestChild = i;
                }
              }
              if (bestChild == -1) break;

              /*! split best child into left and right child */
              BuildRecord left(current.depth+1);
              BuildRecord right(current.depth+1);
              if (!sameGeometry(children[bestChild].prims)) {
                splitByGeometry(children[bestChild].prims,left.prims,right.prims);
              } else {
                splitFallback(children[bestChild].prims,left.prims,right.prims);
              }
              children.split(bestChild,left,right,std::unique_ptr<mvector<PrimRefMB>>());

            } while (children.size() < cfg.branchingFactor);


            /* detect time_ranges that have shrunken */
            bool timesplit = false;
            for (size_t i=0; i<children.size(); i++) {
              const BBox1f c = children[i].prims.time_range;
              const BBox1f p = current.prims.time_range;
              timesplit |= c.lower > p.lower || c.upper < p.upper;
            }
            
            /* create node */
            NodeRef node = createAlignedNodeMB(alloc,timesplit);

            LBBox3fa bounds = empty;
            for (size_t i=0; i<children.size(); i++) {
              const auto child = createLargeLeaf(children[i],alloc);
              setAlignedNodeMB(node,i,child);
              bounds.extend(child.lbounds);
            }

            if (timesplit)
              bounds = current.prims.linearBounds(recalculatePrimRef);
              
            return NodeRecordMB4D(node,bounds,current.prims.time_range);
          }

          /*! performs split */
          std::unique_ptr<mvector<PrimRefMB>> split(const BuildRecord& current, BuildRecord& lrecord, BuildRecord& rrecord, bool& aligned, bool& timesplit)
          {
            /* variable to track the SAH of the best splitting approach */
            float bestSAH = inf;
            const float leafSAH = current.prims.leafSAH(cfg.logBlockSize);

            /* perform standard binning in aligned space */
            HeuristicBinning::Split alignedObjectSplit = alignedHeuristic.find(current.prims,cfg.logBlockSize);
            float alignedObjectSAH = alignedObjectSplit.splitSAH();
            bestSAH = min(alignedObjectSAH,bestSAH);

            /* perform standard binning in unaligned space */
            UnalignedHeuristicBinning::Split unalignedObjectSplit;
            LinearSpace3fa uspace;
            float unalignedObjectSAH = inf;
            if (alignedObjectSAH > 0.7f*leafSAH) {
              uspace = unalignedHeuristic.computeAlignedSpaceMB(scene,current.prims);
              const SetMB sset = current.prims.primInfo(recalculatePrimRef,uspace);
              unalignedObjectSplit = unalignedHeuristic.find(sset,cfg.logBlockSize,uspace);
              unalignedObjectSAH = 1.3f*unalignedObjectSplit.splitSAH(); // makes unaligned splits more expensive
              bestSAH = min(unalignedObjectSAH,bestSAH);
            }

            /* do temporal splits only if previous approaches failed to produce good SAH and the the time range is large enough */
            float temporal_split_sah = inf;
            typename HeuristicTemporal::Split temporal_split;
            if (bestSAH > 0.5f*leafSAH) {
              if (current.prims.time_range.size() > 1.01f/float(current.prims.max_num_time_segments)) {
                temporal_split = temporalSplitHeuristic.find(current.prims,cfg.logBlockSize);
                temporal_split_sah = temporal_split.splitSAH();
                bestSAH = min(temporal_split_sah,bestSAH);
              }
            }

            /* perform fallback split if SAH heuristics failed */
            if (unlikely(!std::isfinite(bestSAH))) {
              current.prims.deterministic_order();
              splitFallback(current.prims,lrecord.prims,rrecord.prims);
            }
            /* perform aligned split if this is best */
            else if (likely(bestSAH == alignedObjectSAH)) {
              alignedHeuristic.split(alignedObjectSplit,current.prims,lrecord.prims,rrecord.prims);
            }
            /* perform unaligned split if this is best */
            else if (likely(bestSAH == unalignedObjectSAH)) {
              unalignedHeuristic.split(unalignedObjectSplit,uspace,current.prims,lrecord.prims,rrecord.prims);
              aligned = false;
            }
            /* perform temporal split if this is best */
            else if (likely(bestSAH == temporal_split_sah)) {
              timesplit = true;
              return temporalSplitHeuristic.split(temporal_split,current.prims,lrecord.prims,rrecord.prims);
            }
            else
              assert(false);

            return std::unique_ptr<mvector<PrimRefMB>>();
          }

          /*! recursive build */
          NodeRecordMB4D recurse(BuildRecord& current, Allocator alloc, bool toplevel)
          {
            /* get thread local allocator */
            if (!alloc)
              alloc = createAlloc();

            /* call memory monitor function to signal progress */
            if (toplevel && current.size() <= SINGLE_THREADED_THRESHOLD)
              progressMonitor(current.size());

            /* create leaf node */
            if (current.depth+MIN_LARGE_LEAF_LEVELS >= cfg.maxDepth || current.size() <= cfg.minLeafSize) {
              current.prims.deterministic_order();
              return createLargeLeaf(current,alloc);
            }

            /* fill all children by always splitting the one with the largest surface area */
            LocalChildList children(current);
            bool aligned = true;
            bool timesplit = false;

            do {

              /* find best child with largest bounding box area */
              ssize_t bestChild = -1;
              float bestArea = neg_inf;
              for (size_t i=0; i<children.size(); i++)
              {
                /* ignore leaves as they cannot get split */
                if (children[i].size() <= cfg.minLeafSize)
                  continue;

                /* remember child with largest area */
                const float A = children[i].prims.halfArea();
                if (A > bestArea) {
                  bestArea = children[i].prims.halfArea();
                  bestChild = i;
                }
              }
              if (bestChild == -1) break;

              /*! split best child into left and right child */
              BuildRecord left(current.depth+1);
              BuildRecord right(current.depth+1);
              std::unique_ptr<mvector<PrimRefMB>> new_vector = split(children[bestChild],left,right,aligned,timesplit);
              children.split(bestChild,left,right,std::move(new_vector));

            } while (children.size() < cfg.branchingFactor);

            /* detect time_ranges that have shrunken */
            for (size_t i=0; i<children.size(); i++) {
              const BBox1f c = children[i].prims.time_range;
              const BBox1f p = current.prims.time_range;
              timesplit |= c.lower > p.lower || c.upper < p.upper;
            }

            /* create time split node */
            if (timesplit)
            {
              const NodeRef node = createAlignedNodeMB(alloc,true);

              /* spawn tasks or ... */
              if (current.size() > SINGLE_THREADED_THRESHOLD)
              {
                parallel_for(size_t(0), children.size(), [&] (const range<size_t>& r) {
                    for (size_t i=r.begin(); i<r.end(); i++) {
                      const auto child = recurse(children[i],nullptr,true);
                      setAlignedNodeMB(node,i,child);
                      _mm_mfence(); // to allow non-temporal stores during build
                    }
                  });
              }
              /* ... continue sequential */
              else {
                for (size_t i=0; i<children.size(); i++) {
                  const auto child = recurse(children[i],alloc,false);
                  setAlignedNodeMB(node,i,child);
                }
              }

              const LBBox3fa bounds = current.prims.linearBounds(recalculatePrimRef);
              return NodeRecordMB4D(node,bounds,current.prims.time_range);
            }

            /* create aligned node */
            else if (aligned)
            {
              const NodeRef node = createAlignedNodeMB(alloc,false);

              /* spawn tasks or ... */
              if (current.size() > SINGLE_THREADED_THRESHOLD)
              {
                LBBox3fa cbounds[MAX_BRANCHING_FACTOR];
                parallel_for(size_t(0), children.size(), [&] (const range<size_t>& r) {
                    for (size_t i=r.begin(); i<r.end(); i++) {
                      const auto child = recurse(children[i],nullptr,true);
                      setAlignedNodeMB(node,i,child);
                      cbounds[i] = child.lbounds;
                      _mm_mfence(); // to allow non-temporal stores during build
                    }
                  });

                LBBox3fa bounds = empty;
                for (size_t i=0; i<children.size(); i++)
                  bounds.extend(cbounds[i]);

                return NodeRecordMB4D(node,bounds,current.prims.time_range);
              }
              /* ... continue sequentially */
              else
              {
                LBBox3fa bounds = empty;
                for (size_t i=0; i<children.size(); i++) {
                  const auto child = recurse(children[i],alloc,false);
                  setAlignedNodeMB(node,i,child);
                  bounds.extend(child.lbounds);
                }
                return NodeRecordMB4D(node,bounds,current.prims.time_range);
              }
            }

            /* create unaligned node */
            else
            {
              const NodeRef node = createUnalignedNodeMB(alloc);

              /* spawn tasks or ... */
              if (current.size() > SINGLE_THREADED_THRESHOLD)
              {
                parallel_for(size_t(0), children.size(), [&] (const range<size_t>& r) {
                    for (size_t i=r.begin(); i<r.end(); i++) {
                      const LinearSpace3fa space = unalignedHeuristic.computeAlignedSpaceMB(scene,children[i].prims);
                      const LBBox3fa lbounds = children[i].prims.linearBounds(recalculatePrimRef,space);
                      const auto child = recurse(children[i],nullptr,true);
                      setUnalignedNodeMB(node,i,child.ref,space,lbounds,children[i].prims.time_range);
                      _mm_mfence(); // to allow non-temporal stores during build
                    }
                  });
              }
              /* ... continue sequentially */
              else
              {
                for (size_t i=0; i<children.size(); i++) {
                  const LinearSpace3fa space = unalignedHeuristic.computeAlignedSpaceMB(scene,children[i].prims);
                  const LBBox3fa lbounds = children[i].prims.linearBounds(recalculatePrimRef,space);
                  const auto child = recurse(children[i],alloc,false);
                  setUnalignedNodeMB(node,i,child.ref,space,lbounds,children[i].prims.time_range);
                }
              }

              const LBBox3fa bounds = current.prims.linearBounds(recalculatePrimRef);
              return NodeRecordMB4D(node,bounds,current.prims.time_range);
            }
          }

        public:

          /*! entry point into builder */
          NodeRecordMB4D operator() (mvector<PrimRefMB>& prims, const PrimInfoMB& pinfo)
          {
            BuildRecord record(SetMB(pinfo,&prims),1);
            auto root = recurse(record,nullptr,true);
            _mm_mfence(); // to allow non-temporal stores during build
            return root;
          }

        private:
          Settings cfg;
          Scene* scene;
          const RecalculatePrimRef& recalculatePrimRef;
          const CreateAllocFunc& createAlloc;
          const CreateAlignedNodeMBFunc& createAlignedNodeMB;
          const SetAlignedNodeMBFunc& setAlignedNodeMB;
          const CreateUnalignedNodeMBFunc& createUnalignedNodeMB;
          const SetUnalignedNodeMBFunc& setUnalignedNodeMB;
          const CreateLeafFunc& createLeaf;
          const ProgressMonitor& progressMonitor;

        private:
          HeuristicBinning alignedHeuristic;
          UnalignedHeuristicBinning unalignedHeuristic;
          HeuristicTemporal temporalSplitHeuristic;
        };

      template<typename NodeRef,
        typename RecalculatePrimRef,
        typename CreateAllocFunc,
        typename CreateAlignedNodeMBFunc,
        typename SetAlignedNodeMBFunc,
        typename CreateUnalignedNodeMBFunc,
        typename SetUnalignedNodeMBFunc,
        typename CreateLeafFunc,
        typename ProgressMonitor>

        static BVHNodeRecordMB4D<NodeRef> build (Scene* scene, mvector<PrimRefMB>& prims, const PrimInfoMB& pinfo,
                                               const RecalculatePrimRef& recalculatePrimRef,
                                               const CreateAllocFunc& createAlloc,
                                               const CreateAlignedNodeMBFunc& createAlignedNodeMB,
                                               const SetAlignedNodeMBFunc& setAlignedNodeMB,
                                               const CreateUnalignedNodeMBFunc& createUnalignedNodeMB,
                                               const SetUnalignedNodeMBFunc& setUnalignedNodeMB,
                                               const CreateLeafFunc& createLeaf,
                                               const ProgressMonitor& progressMonitor,
                                               const Settings settings)
        {
          typedef BuilderT<NodeRef,RecalculatePrimRef,CreateAllocFunc,
            CreateAlignedNodeMBFunc,SetAlignedNodeMBFunc,
            CreateUnalignedNodeMBFunc,SetUnalignedNodeMBFunc,
            CreateLeafFunc,ProgressMonitor> Builder;

          Builder builder(scene,recalculatePrimRef,createAlloc,
                          createAlignedNodeMB,setAlignedNodeMB,
                          createUnalignedNodeMB,setUnalignedNodeMB,
                          createLeaf,progressMonitor,settings);

          return builder(prims,pinfo);
        }
    };
  }
}
