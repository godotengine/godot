// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define MBLUR_NUM_TEMPORAL_BINS 2
#define MBLUR_NUM_OBJECT_BINS   32

#include "../bvh/bvh.h"
#include "../common/primref_mb.h"
#include "heuristic_binning_array_aligned.h"
#include "heuristic_timesplit_array.h"

namespace embree
{
  namespace isa
  {
    template<typename T>
      struct SharedVector
      {
        __forceinline SharedVector() {}

        __forceinline SharedVector(T* ptr, size_t refCount = 1)
          : prims(ptr), refCount(refCount) {}

        __forceinline void incRef() {
          refCount++;
        }

        __forceinline void decRef()
        {
          if (--refCount == 0)
            delete prims;
        }

        T* prims;
        size_t refCount;
      };

    template<typename BuildRecord, int MAX_BRANCHING_FACTOR>
      struct LocalChildListT
      {
        typedef SharedVector<mvector<PrimRefMB>> SharedPrimRefVector;

        __forceinline LocalChildListT (const BuildRecord& record)
          : numChildren(1), numSharedPrimVecs(1)
        {
          /* the local root will be freed in the ancestor where it was created (thus refCount is 2) */
          children[0] = record;
          primvecs[0] = new (&sharedPrimVecs[0]) SharedPrimRefVector(record.prims.prims, 2);
        }

        __forceinline ~LocalChildListT()
        {
          for (size_t i = 0; i < numChildren; i++)
            primvecs[i]->decRef();
        }

        __forceinline BuildRecord& operator[] ( const size_t i ) {
          return children[i];
        }

        __forceinline size_t size() const {
          return numChildren;
        }

        __forceinline void split(ssize_t bestChild, const BuildRecord& lrecord, const BuildRecord& rrecord, std::unique_ptr<mvector<PrimRefMB>> new_vector)
        {
          SharedPrimRefVector* bsharedPrimVec = primvecs[bestChild];
          if (lrecord.prims.prims == bsharedPrimVec->prims) {
            primvecs[bestChild] = bsharedPrimVec;
            bsharedPrimVec->incRef();
          }
          else {
            primvecs[bestChild] = new (&sharedPrimVecs[numSharedPrimVecs++]) SharedPrimRefVector(lrecord.prims.prims);
          }

          if (rrecord.prims.prims == bsharedPrimVec->prims) {
            primvecs[numChildren] = bsharedPrimVec;
            bsharedPrimVec->incRef();
          }
          else {
            primvecs[numChildren] = new (&sharedPrimVecs[numSharedPrimVecs++]) SharedPrimRefVector(rrecord.prims.prims);
          }
          bsharedPrimVec->decRef();
          new_vector.release();

          children[bestChild] = lrecord;
          children[numChildren] = rrecord;
          numChildren++;
        }

      public:
        array_t<BuildRecord,MAX_BRANCHING_FACTOR> children;
        array_t<SharedPrimRefVector*,MAX_BRANCHING_FACTOR> primvecs;
        size_t numChildren;

        array_t<SharedPrimRefVector,2*MAX_BRANCHING_FACTOR> sharedPrimVecs;
        size_t numSharedPrimVecs;
      };

    template<typename Mesh>
      struct RecalculatePrimRef
      {
        Scene* scene;

        __forceinline RecalculatePrimRef (Scene* scene)
          : scene(scene) {}

        __forceinline PrimRefMB operator() (const PrimRefMB& prim, const BBox1f time_range) const
        {
          const unsigned geomID = prim.geomID();
          const unsigned primID = prim.primID();
          const Mesh* mesh = scene->get<Mesh>(geomID);
          const LBBox3fa lbounds = mesh->linearBounds(primID, time_range);
          const range<int> tbounds = mesh->timeSegmentRange(time_range);
          return PrimRefMB (lbounds, tbounds.size(), mesh->time_range, mesh->numTimeSegments(), geomID, primID);
        }

        // __noinline is workaround for ICC16 bug under MacOSX
        __noinline PrimRefMB operator() (const PrimRefMB& prim, const BBox1f time_range, const LinearSpace3fa& space) const
        {
          const unsigned geomID = prim.geomID();
          const unsigned primID = prim.primID();
          const Mesh* mesh = scene->get<Mesh>(geomID);
          const LBBox3fa lbounds = mesh->linearBounds(space, primID, time_range);
          const range<int> tbounds = mesh->timeSegmentRange(time_range);
          return PrimRefMB (lbounds, tbounds.size(), mesh->time_range, mesh->numTimeSegments(), geomID, primID);
        }

        __forceinline LBBox3fa linearBounds(const PrimRefMB& prim, const BBox1f time_range) const {
          return scene->get<Mesh>(prim.geomID())->linearBounds(prim.primID(), time_range);
        }

        // __noinline is workaround for ICC16 bug under MacOSX
        __noinline LBBox3fa linearBounds(const PrimRefMB& prim, const BBox1f time_range, const LinearSpace3fa& space) const {
          return scene->get<Mesh>(prim.geomID())->linearBounds(space, prim.primID(), time_range);
        }
      };

    struct VirtualRecalculatePrimRef
    {
      Scene* scene;
      
      __forceinline VirtualRecalculatePrimRef (Scene* scene)
        : scene(scene) {}
      
      __forceinline PrimRefMB operator() (const PrimRefMB& prim, const BBox1f time_range) const
      {
        const unsigned geomID = prim.geomID();
        const unsigned primID = prim.primID();
        const Geometry* mesh = scene->get(geomID);
        const LBBox3fa lbounds = mesh->vlinearBounds(primID, time_range);
        const range<int> tbounds = mesh->timeSegmentRange(time_range);
        return PrimRefMB (lbounds, tbounds.size(), mesh->time_range, mesh->numTimeSegments(), geomID, primID);
      }
      
      __forceinline PrimRefMB operator() (const PrimRefMB& prim, const BBox1f time_range, const LinearSpace3fa& space) const
      {
        const unsigned geomID = prim.geomID();
        const unsigned primID = prim.primID();
        const Geometry* mesh = scene->get(geomID);
        const LBBox3fa lbounds = mesh->vlinearBounds(space, primID, time_range);
        const range<int> tbounds = mesh->timeSegmentRange(time_range);
        return PrimRefMB (lbounds, tbounds.size(), mesh->time_range, mesh->numTimeSegments(), geomID, primID);
      }
      
      __forceinline LBBox3fa linearBounds(const PrimRefMB& prim, const BBox1f time_range) const {
        return scene->get(prim.geomID())->vlinearBounds(prim.primID(), time_range);
      }
      
      __forceinline LBBox3fa linearBounds(const PrimRefMB& prim, const BBox1f time_range, const LinearSpace3fa& space) const {
        return scene->get(prim.geomID())->vlinearBounds(space, prim.primID(), time_range);
      }
    };

    struct BVHBuilderMSMBlur
    {
      /*! settings for msmblur builder */
      struct Settings
      {
        /*! default settings */
        Settings ()
        : branchingFactor(2), maxDepth(32), logBlockSize(0), minLeafSize(1), maxLeafSize(8),
          travCost(1.0f), intCost(1.0f), singleLeafTimeSegment(false),
          singleThreadThreshold(1024) {}


        Settings (size_t sahBlockSize, size_t minLeafSize, size_t maxLeafSize, float travCost, float intCost, size_t singleThreadThreshold)
        : branchingFactor(2), maxDepth(32), logBlockSize(bsr(sahBlockSize)), minLeafSize(minLeafSize), maxLeafSize(maxLeafSize),
          travCost(travCost), intCost(intCost), singleThreadThreshold(singleThreadThreshold)
        {
          minLeafSize = min(minLeafSize,maxLeafSize);
        }

      public:
        size_t branchingFactor;  //!< branching factor of BVH to build
        size_t maxDepth;         //!< maximum depth of BVH to build
        size_t logBlockSize;     //!< log2 of blocksize for SAH heuristic
        size_t minLeafSize;      //!< minimum size of a leaf
        size_t maxLeafSize;      //!< maximum size of a leaf
        float travCost;          //!< estimated cost of one traversal step
        float intCost;           //!< estimated cost of one primitive intersection
        bool singleLeafTimeSegment; //!< split time to single time range
        size_t singleThreadThreshold; //!< threshold when we switch to single threaded build
      };

      struct BuildRecord
      {
      public:
	__forceinline BuildRecord () {}

        __forceinline BuildRecord (size_t depth)
          : depth(depth) {}

        __forceinline BuildRecord (const SetMB& prims, size_t depth)
          : depth(depth), prims(prims) {}

        __forceinline friend bool operator< (const BuildRecord& a, const BuildRecord& b) {
          return a.prims.size() < b.prims.size();
        }

        __forceinline size_t size() const {
          return prims.size();
        }

      public:
	size_t depth;                     //!< Depth of the root of this subtree.
	SetMB prims;                      //!< The list of primitives.
      };

      struct BuildRecordSplit : public BuildRecord
      {
        __forceinline BuildRecordSplit () {}

        __forceinline BuildRecordSplit (size_t depth) 
          : BuildRecord(depth) {}

        __forceinline BuildRecordSplit (const BuildRecord& record, const BinSplit<MBLUR_NUM_OBJECT_BINS>& split)
          : BuildRecord(record), split(split) {}
        
        BinSplit<MBLUR_NUM_OBJECT_BINS> split;
      };

      template<
        typename NodeRef,
        typename RecalculatePrimRef,
        typename Allocator,
        typename CreateAllocFunc,
        typename CreateNodeFunc,
        typename SetNodeFunc,
        typename CreateLeafFunc,
        typename ProgressMonitor>

        class BuilderT
        {
          ALIGNED_CLASS_(16);
          static const size_t MAX_BRANCHING_FACTOR = 16;       //!< maximum supported BVH branching factor	  
          static const size_t MIN_LARGE_LEAF_LEVELS = 8;        //!< create balanced tree if we are that many levels before the maximum tree depth

          typedef BVHNodeRecordMB4D<NodeRef> NodeRecordMB4D;
          typedef BinSplit<MBLUR_NUM_OBJECT_BINS> Split;
          typedef mvector<PrimRefMB>* PrimRefVector;
          typedef SharedVector<mvector<PrimRefMB>> SharedPrimRefVector;
          typedef LocalChildListT<BuildRecord,MAX_BRANCHING_FACTOR> LocalChildList;
          typedef LocalChildListT<BuildRecordSplit,MAX_BRANCHING_FACTOR> LocalChildListSplit;

        public:

          BuilderT (MemoryMonitorInterface* device,
                    const RecalculatePrimRef recalculatePrimRef,
                    const CreateAllocFunc createAlloc,
                    const CreateNodeFunc createNode,
                    const SetNodeFunc setNode,
                    const CreateLeafFunc createLeaf,
                    const ProgressMonitor progressMonitor,
                    const Settings& settings)
            : cfg(settings),
            heuristicObjectSplit(),
            heuristicTemporalSplit(device, recalculatePrimRef),
            recalculatePrimRef(recalculatePrimRef), createAlloc(createAlloc), createNode(createNode), setNode(setNode), createLeaf(createLeaf),
            progressMonitor(progressMonitor)
          {
            if (cfg.branchingFactor > MAX_BRANCHING_FACTOR)
              throw_RTCError(RTC_ERROR_UNKNOWN,"bvh_builder: branching factor too large");
          }

          /*! finds the best split */
          const Split find(const SetMB& set)
          {
            /* first try standard object split */
            const Split object_split = heuristicObjectSplit.find(set,cfg.logBlockSize);
            const float object_split_sah = object_split.splitSAH();

            /* test temporal splits only when object split was bad */
            const float leaf_sah = set.leafSAH(cfg.logBlockSize);
            if (object_split_sah < 0.50f*leaf_sah)
              return object_split;

            /* do temporal splits only if the time range is big enough */
            if (set.time_range.size() > 1.01f/float(set.max_num_time_segments))
            {
              const Split temporal_split = heuristicTemporalSplit.find(set,cfg.logBlockSize);
              const float temporal_split_sah = temporal_split.splitSAH();

              /* take temporal split if it improved SAH */
              if (temporal_split_sah < object_split_sah)
                return temporal_split;
            }

            return object_split;
          }

          /*! array partitioning */
          __forceinline std::unique_ptr<mvector<PrimRefMB>> split(const Split& split, const SetMB& set, SetMB& lset, SetMB& rset)
          {
            /* perform object split */
            if (likely(split.data == Split::SPLIT_OBJECT)) {
              heuristicObjectSplit.split(split,set,lset,rset);
            }
            /* perform temporal split */
            else if (likely(split.data == Split::SPLIT_TEMPORAL)) {
              return heuristicTemporalSplit.split(split,set,lset,rset);
            }
            /* perform fallback split */
            else if (unlikely(split.data == Split::SPLIT_FALLBACK)) {
              set.deterministic_order();
              splitFallback(set,lset,rset);
            }
            /* split by geometry */
            else if (unlikely(split.data == Split::SPLIT_GEOMID)) {
              set.deterministic_order();
              splitByGeometry(set,lset,rset);
            }
            else
              assert(false);

            return std::unique_ptr<mvector<PrimRefMB>>();
          }

          /*! finds the best fallback split */
          __noinline Split findFallback(const SetMB& set)
          {
            /* split if primitives are not from same geometry */
            if (!sameGeometry(set))
              return Split(0.0f,Split::SPLIT_GEOMID);
            
            /* if a leaf can only hold a single time-segment, we might have to do additional temporal splits */
            if (cfg.singleLeafTimeSegment)
            {
              /* test if one primitive has more than one time segment in time range, if so split time */
              for (size_t i=set.begin(); i<set.end(); i++)
              {
                const PrimRefMB& prim = (*set.prims)[i];
                const range<int> itime_range = prim.timeSegmentRange(set.time_range);
                const int localTimeSegments = itime_range.size();
                assert(localTimeSegments > 0);
                if (localTimeSegments > 1) {
                  const int icenter = (itime_range.begin() + itime_range.end())/2;
                  const float splitTime = prim.timeStep(icenter);
                  return Split(0.0f,(unsigned)Split::SPLIT_TEMPORAL,0,splitTime);
                }
              }
            }        

            /* otherwise return fallback split */
            return Split(0.0f,Split::SPLIT_FALLBACK);
          }

          /*! performs fallback split */
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

          /*! checks if all primitives are from the same geometry */
          __forceinline bool sameGeometry(const SetMB& set)
          {
            if (set.size() == 0) return true;
            mvector<PrimRefMB>& prims = *set.prims;
            const size_t begin = set.begin();
            const size_t end   = set.end();
            unsigned int firstGeomID = prims[begin].geomID();
            for (size_t i=begin+1; i<end; i++) {
              if (prims[i].geomID() != firstGeomID){
                return false;
              }
            }
            return true;
          }

          /* split by geometry ID */
          void splitByGeometry(const SetMB& set, SetMB& lset, SetMB& rset)
          {
            assert(set.size() > 1);

            mvector<PrimRefMB>& prims = *set.prims;
            const size_t begin = set.begin();
            const size_t end   = set.end();
            
            PrimInfoMB left(empty);
            PrimInfoMB right(empty);
            unsigned int geomID = prims[begin].geomID();
            size_t center = serial_partitioning(prims.data(),begin,end,left,right,
                                                [&] ( const PrimRefMB& prim ) { return prim.geomID() == geomID; },
                                                [ ] ( PrimInfoMB& dst, const PrimRefMB& prim ) { dst.add_primref(prim); });
            
            new (&lset) SetMB(left, set.prims,range<size_t>(begin,center),set.time_range);
            new (&rset) SetMB(right,set.prims,range<size_t>(center,end  ),set.time_range);
          }

          const NodeRecordMB4D createLargeLeaf(const BuildRecord& in, Allocator alloc)
          {
            /* this should never occur but is a fatal error */
            if (in.depth > cfg.maxDepth)
              throw_RTCError(RTC_ERROR_UNKNOWN,"depth limit reached");

            /* replace already found split by fallback split */
            const BuildRecordSplit current(BuildRecord(in.prims,in.depth),findFallback(in.prims));

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
            if (current.size() <= cfg.maxLeafSize && current.split.data < Split::SPLIT_ENFORCE && !force_split)
              return createLeaf(current,alloc);
	  
            /* fill all children by always splitting the largest one */
            bool hasTimeSplits = false;
            NodeRecordMB4D values[MAX_BRANCHING_FACTOR];
            LocalChildListSplit children(current);

            do {
              /* find best child with largest bounding box area */
              size_t bestChild = -1;
              size_t bestSize = 0;
              for (size_t i=0; i<children.size(); i++)
              {
                /* ignore leaves as they cannot get split */
                if (children[i].size() <= cfg.maxLeafSize && children[i].split.data < Split::SPLIT_ENFORCE && !force_split)
                  continue;

                force_split = false;
                
                /* remember child with largest size */
                if (children[i].size() > bestSize) {
                  bestSize = children[i].size();
                  bestChild = i;
                }
              }
              if (bestChild == -1) break;

              /* perform best found split */
              BuildRecordSplit& brecord = children[bestChild];
              BuildRecordSplit lrecord(current.depth+1);
              BuildRecordSplit rrecord(current.depth+1);
              std::unique_ptr<mvector<PrimRefMB>> new_vector = split(brecord.split,brecord.prims,lrecord.prims,rrecord.prims);
              hasTimeSplits |= new_vector != nullptr;

              /* find new splits */
              lrecord.split = findFallback(lrecord.prims);
              rrecord.split = findFallback(rrecord.prims);
              children.split(bestChild,lrecord,rrecord,std::move(new_vector));

            } while (children.size() < cfg.branchingFactor);

            /* detect time_ranges that have shrunken */
            for (size_t i=0; i<children.size(); i++) {
              const BBox1f c = children[i].prims.time_range;
              const BBox1f p = in.prims.time_range;
              hasTimeSplits |= c.lower > p.lower || c.upper < p.upper;
            }

            /* create node */
            auto node = createNode(children.children.data(),children.numChildren,alloc,hasTimeSplits);

            /* recurse into each child and perform reduction */
            LBBox3fa gbounds = empty;
            for (size_t i=0; i<children.size(); i++) {
              values[i] = createLargeLeaf(children[i],alloc);
              gbounds.extend(values[i].lbounds);
            }

            setNode(current,children.children.data(),node,values,children.numChildren);

            /* calculate geometry bounds of this node */
            if (hasTimeSplits)
              return NodeRecordMB4D(node,current.prims.linearBounds(recalculatePrimRef),current.prims.time_range);
            else
              return NodeRecordMB4D(node,gbounds,current.prims.time_range);
          }

          const NodeRecordMB4D recurse(const BuildRecord& current, Allocator alloc, bool toplevel)
          {
            /* get thread local allocator */
            if (!alloc)
              alloc = createAlloc();

            /* call memory monitor function to signal progress */
            if (toplevel && current.size() <= cfg.singleThreadThreshold)
              progressMonitor(current.size());

            /*! find best split */
            const Split csplit = find(current.prims);

            /*! compute leaf and split cost */
            const float leafSAH  = cfg.intCost*current.prims.leafSAH(cfg.logBlockSize);
            const float splitSAH = cfg.travCost*current.prims.halfArea()+cfg.intCost*csplit.splitSAH();
            assert((current.size() == 0) || ((leafSAH >= 0) && (splitSAH >= 0)));

            /*! create a leaf node when threshold reached or SAH tells us to stop */
            if (current.size() <= cfg.minLeafSize || current.depth+MIN_LARGE_LEAF_LEVELS >= cfg.maxDepth || (current.size() <= cfg.maxLeafSize && leafSAH <= splitSAH)) {
              current.prims.deterministic_order();
              return createLargeLeaf(current,alloc);
            }

            /*! perform initial split */
            SetMB lprims,rprims;
            std::unique_ptr<mvector<PrimRefMB>> new_vector = split(csplit,current.prims,lprims,rprims);
            bool hasTimeSplits = new_vector != nullptr;
            NodeRecordMB4D values[MAX_BRANCHING_FACTOR];
            LocalChildList children(current);
            {
              BuildRecord lrecord(lprims,current.depth+1);
              BuildRecord rrecord(rprims,current.depth+1);
              children.split(0,lrecord,rrecord,std::move(new_vector));
            }

            /*! split until node is full or SAH tells us to stop */
            while (children.size() < cfg.branchingFactor) 
            {
              /*! find best child to split */
              float bestArea = neg_inf;
              ssize_t bestChild = -1;
              for (size_t i=0; i<children.size(); i++)
              {
                if (children[i].size() <= cfg.minLeafSize) continue;
                if (expectedApproxHalfArea(children[i].prims.geomBounds) > bestArea) {
                  bestChild = i; bestArea = expectedApproxHalfArea(children[i].prims.geomBounds);
                }
              }
              if (bestChild == -1) break;

              /* perform split */
              BuildRecord& brecord = children[bestChild];
              BuildRecord lrecord(current.depth+1);
              BuildRecord rrecord(current.depth+1);
              Split csplit = find(brecord.prims);
              std::unique_ptr<mvector<PrimRefMB>> new_vector = split(csplit,brecord.prims,lrecord.prims,rrecord.prims);
              hasTimeSplits |= new_vector != nullptr;
              children.split(bestChild,lrecord,rrecord,std::move(new_vector));
            }

            /* detect time_ranges that have shrunken */
            for (size_t i=0; i<children.size(); i++) {
              const BBox1f c = children[i].prims.time_range;
              const BBox1f p = current.prims.time_range;
              hasTimeSplits |= c.lower > p.lower || c.upper < p.upper;
            }

            /* sort buildrecords for simpler shadow ray traversal */
            //std::sort(&children[0],&children[children.size()],std::greater<BuildRecord>()); // FIXME: reduces traversal performance of bvh8.triangle4 (need to verified) !!

            /*! create an inner node */
            auto node = createNode(children.children.data(), children.numChildren, alloc, hasTimeSplits);
            LBBox3fa gbounds = empty;

            /* spawn tasks */
            if (unlikely(current.size() > cfg.singleThreadThreshold))
            {
              /*! parallel_for is faster than spawing sub-tasks */
              parallel_for(size_t(0), children.size(), [&] (const range<size_t>& r) {
                  for (size_t i=r.begin(); i<r.end(); i++) {
                    values[i] = recurse(children[i],nullptr,true);
                    _mm_mfence(); // to allow non-temporal stores during build
                  }
                });

              /*! merge bounding boxes */
              for (size_t i=0; i<children.size(); i++)
                gbounds.extend(values[i].lbounds);
            }
            /* recurse into each child */
            else
            {
              //for (size_t i=0; i<children.size(); i++)
              for (ssize_t i=children.size()-1; i>=0; i--) {
                values[i] = recurse(children[i],alloc,false);
                gbounds.extend(values[i].lbounds);
              }
            }

            setNode(current,children.children.data(),node,values,children.numChildren);

            /* calculate geometry bounds of this node */
            if (unlikely(hasTimeSplits))
              return NodeRecordMB4D(node,current.prims.linearBounds(recalculatePrimRef),current.prims.time_range);
            else
              return NodeRecordMB4D(node,gbounds,current.prims.time_range);
          }

          /*! builder entry function */
          __forceinline const NodeRecordMB4D operator() (mvector<PrimRefMB>& prims, const PrimInfoMB& pinfo)
          {
            const SetMB set(pinfo,&prims);
            auto ret = recurse(BuildRecord(set,1),nullptr,true);
            _mm_mfence(); // to allow non-temporal stores during build
            return ret;
          }

        private:
          Settings cfg;
          HeuristicArrayBinningMB<PrimRefMB,MBLUR_NUM_OBJECT_BINS> heuristicObjectSplit;
          HeuristicMBlurTemporalSplit<PrimRefMB,RecalculatePrimRef,MBLUR_NUM_TEMPORAL_BINS> heuristicTemporalSplit;
          const RecalculatePrimRef recalculatePrimRef;
          const CreateAllocFunc createAlloc;
          const CreateNodeFunc createNode;
          const SetNodeFunc setNode;
          const CreateLeafFunc createLeaf;
          const ProgressMonitor progressMonitor;
        };

      template<typename NodeRef,
        typename RecalculatePrimRef,
        typename CreateAllocFunc,
        typename CreateNodeFunc,
        typename SetNodeFunc,
        typename CreateLeafFunc,
        typename ProgressMonitorFunc>

        static const BVHNodeRecordMB4D<NodeRef> build(mvector<PrimRefMB>& prims,
                                                      const PrimInfoMB& pinfo,
                                                      MemoryMonitorInterface* device,
                                                      const RecalculatePrimRef recalculatePrimRef,
                                                      const CreateAllocFunc createAlloc,
                                                      const CreateNodeFunc createNode,
                                                      const SetNodeFunc setNode,
                                                      const CreateLeafFunc createLeaf,
                                                      const ProgressMonitorFunc progressMonitor,
                                                      const Settings& settings)
      {
          typedef BuilderT<
            NodeRef,
            RecalculatePrimRef,
            decltype(createAlloc()),
            CreateAllocFunc,
            CreateNodeFunc,
            SetNodeFunc,
            CreateLeafFunc,
            ProgressMonitorFunc> Builder;

          Builder builder(device,
                          recalculatePrimRef,
                          createAlloc,
                          createNode,
                          setNode,
                          createLeaf,
                          progressMonitor,
                          settings);


          return builder(prims,pinfo);
        }
    };
  }
}
