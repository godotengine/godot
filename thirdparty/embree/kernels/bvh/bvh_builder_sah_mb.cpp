// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh.h"
#include "bvh_builder.h"
#include "../builders/bvh_builder_msmblur.h"

#include "../builders/primrefgen.h"
#include "../builders/splitter.h"

#include "../geometry/linei.h"
#include "../geometry/triangle.h"
#include "../geometry/trianglev.h"
#include "../geometry/trianglev_mb.h"
#include "../geometry/trianglei.h"
#include "../geometry/quadv.h"
#include "../geometry/quadi.h"
#include "../geometry/object.h"
#include "../geometry/instance.h"
#include "../geometry/subgrid.h"

#include "../common/state.h"

// FIXME: remove after removing BVHNBuilderMBlurRootTimeSplitsSAH
#include "../../common/algorithms/parallel_for_for.h"
#include "../../common/algorithms/parallel_for_for_prefix_sum.h"


namespace embree
{
  namespace isa
  {

#if 0
    template<int N, typename Primitive>
    struct CreateMBlurLeaf
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecordMB NodeRecordMB;

      __forceinline CreateMBlurLeaf (BVH* bvh, PrimRef* prims, size_t time) : bvh(bvh), prims(prims), time(time) {}

      __forceinline NodeRecordMB operator() (const PrimRef* prims, const range<size_t>& set, const FastAllocator::CachedAllocator& alloc) const
      {
        size_t items = Primitive::blocks(set.size());
        size_t start = set.begin();
        for (size_t i=start; i<end; i++) assert((*current.prims.prims)[start].geomID() == (*current.prims.prims)[i].geomID()); // assert that all geomIDs are identical
        Primitive* accel = (Primitive*) alloc.malloc1(items*sizeof(Primitive),BVH::byteAlignment);
        NodeRef node = bvh->encodeLeaf((char*)accel,items);

        LBBox3fa allBounds = empty;
        for (size_t i=0; i<items; i++)
          allBounds.extend(accel[i].fillMB(prims, start, set.end(), bvh->scene, time));

        return NodeRecordMB(node,allBounds);
      }

      BVH* bvh;
      PrimRef* prims;
      size_t time;
    };
#endif

    template<int N, typename Mesh, typename Primitive>
    struct CreateMSMBlurLeaf
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecordMB4D NodeRecordMB4D;

      __forceinline CreateMSMBlurLeaf (BVH* bvh) : bvh(bvh) {}

      __forceinline const NodeRecordMB4D operator() (const BVHBuilderMSMBlur::BuildRecord& current, const FastAllocator::CachedAllocator& alloc) const
      {
        size_t items = Primitive::blocks(current.prims.size());
        size_t start = current.prims.begin();
        size_t end   = current.prims.end();
        for (size_t i=start; i<end; i++) assert((*current.prims.prims)[start].geomID() == (*current.prims.prims)[i].geomID()); // assert that all geomIDs are identical
        Primitive* accel = (Primitive*) alloc.malloc1(items*sizeof(Primitive),BVH::byteNodeAlignment);
        NodeRef node = bvh->encodeLeaf((char*)accel,items);
        LBBox3fa allBounds = empty;
        for (size_t i=0; i<items; i++)
          allBounds.extend(accel[i].fillMB(current.prims.prims->data(), start, current.prims.end(), bvh->scene, current.prims.time_range));
        return NodeRecordMB4D(node,allBounds,current.prims.time_range);
      }

      BVH* bvh;
    };

    /* Motion blur BVH with 4D nodes and internal time splits */
    template<int N, typename Mesh, typename Primitive>
    struct BVHNBuilderMBlurSAH : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVHN<N>::NodeRef NodeRef;
      typedef typename BVHN<N>::NodeRecordMB NodeRecordMB;
      typedef typename BVHN<N>::AABBNodeMB AABBNodeMB;

      BVH* bvh;
      Scene* scene;
      const size_t sahBlockSize;
      const float intCost;
      const size_t minLeafSize;
      const size_t maxLeafSize;
      const Geometry::GTypeMask gtype_;

      BVHNBuilderMBlurSAH (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const Geometry::GTypeMask gtype)
        : bvh(bvh), scene(scene), sahBlockSize(sahBlockSize), intCost(intCost), minLeafSize(minLeafSize), maxLeafSize(min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks)), gtype_(gtype) {}

      void build()
      {
	/* skip build for empty scene */
        const size_t numPrimitives = scene->getNumPrimitives(gtype_,true);
        if (numPrimitives == 0) { bvh->clear(); return; }

        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH" + toString(N) + "BuilderMBlurSAH");

#if PROFILE
        profile(2,PROFILE_RUNS,numPrimitives,[&] (ProfileTimer& timer) {
#endif

            //const size_t numTimeSteps = scene->getNumTimeSteps<typename Mesh::type_t,true>();
            //const size_t numTimeSegments = numTimeSteps-1; assert(numTimeSteps > 1);

            /*if (numTimeSegments == 1)
              buildSingleSegment(numPrimitives);
              else*/
              buildMultiSegment(numPrimitives);

#if PROFILE
          });
#endif

	/* clear temporary data for static geometry */
	bvh->cleanup();
        bvh->postBuild(t0);
      }

#if 0 // No longer compatible when time_ranges are present for geometries. Would have to create temporal nodes sometimes, and put only a single geometry into leaf.
      void buildSingleSegment(size_t numPrimitives)
      {
        /* create primref array */
        mvector<PrimRef> prims(scene->device,numPrimitives);
	const PrimInfo pinfo = createPrimRefArrayMBlur(scene,gtype_,numPrimitives,prims,bvh->scene->progressInterface,0);
        /* early out if no valid primitives */
        if (pinfo.size() == 0) { bvh->clear(); return; }
        /* estimate acceleration structure size */
        const size_t node_bytes = pinfo.size()*sizeof(AABBNodeMB)/(4*N);
        const size_t leaf_bytes = size_t(1.2*Primitive::blocks(pinfo.size())*sizeof(Primitive));
        bvh->alloc.init_estimate(node_bytes+leaf_bytes);

        /* settings for BVH build */
        GeneralBVHBuilder::Settings settings;
        settings.branchingFactor = N;
        settings.maxDepth = BVH::maxBuildDepthLeaf;
        settings.logBlockSize = bsr(sahBlockSize);
        settings.minLeafSize = min(minLeafSize,maxLeafSize);
        settings.maxLeafSize = maxLeafSize;
        settings.travCost = travCost;
        settings.intCost = intCost;
        settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,pinfo.size(),node_bytes+leaf_bytes);

        /* build hierarchy */
        auto root = BVHBuilderBinnedSAH::build<NodeRecordMB>
          (typename BVH::CreateAlloc(bvh),typename BVH::AABBNodeMB::Create(),typename BVH::AABBNodeMB::Set(),
           CreateMBlurLeaf<N,Primitive>(bvh,prims.data(),0),bvh->scene->progressInterface,
           prims.data(),pinfo,settings);

        bvh->set(root.ref,root.lbounds,pinfo.size());
      }
#endif

      void buildMultiSegment(size_t numPrimitives)
      {
        /* create primref array */
        mvector<PrimRefMB> prims(scene->device,numPrimitives);
	PrimInfoMB pinfo = createPrimRefArrayMSMBlur(scene,gtype_,numPrimitives,prims,bvh->scene->progressInterface);

        /* early out if no valid primitives */
        if (pinfo.size() == 0) { bvh->clear(); return; }

        /* estimate acceleration structure size */
        const size_t node_bytes = pinfo.num_time_segments*sizeof(AABBNodeMB)/(4*N);
        const size_t leaf_bytes = size_t(1.2*Primitive::blocks(pinfo.num_time_segments)*sizeof(Primitive));
        bvh->alloc.init_estimate(node_bytes+leaf_bytes);

        /* settings for BVH build */
        BVHBuilderMSMBlur::Settings settings;
        settings.branchingFactor = N;
        settings.maxDepth = BVH::maxDepth;
        settings.logBlockSize = bsr(sahBlockSize);
        settings.minLeafSize = min(minLeafSize,maxLeafSize);
        settings.maxLeafSize = maxLeafSize;
        settings.travCost = travCost;
        settings.intCost = intCost;
        settings.singleLeafTimeSegment = Primitive::singleTimeSegment;
        settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,pinfo.size(),node_bytes+leaf_bytes);
        
        /* build hierarchy */
        auto root =
          BVHBuilderMSMBlur::build<NodeRef>(prims,pinfo,scene->device,
                                            RecalculatePrimRef<Mesh>(scene),
                                            typename BVH::CreateAlloc(bvh),
                                            typename BVH::AABBNodeMB4D::Create(),
                                            typename BVH::AABBNodeMB4D::Set(),
                                            CreateMSMBlurLeaf<N,Mesh,Primitive>(bvh),
                                            bvh->scene->progressInterface,
                                            settings);

        bvh->set(root.ref,root.lbounds,pinfo.num_time_segments);
      }

      void clear() {
      }
    };

    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/

    struct GridRecalculatePrimRef
    {
      Scene* scene;
      const SubGridBuildData * const sgrids;

      __forceinline GridRecalculatePrimRef (Scene* scene, const SubGridBuildData * const sgrids)
        : scene(scene), sgrids(sgrids) {}

        __forceinline PrimRefMB operator() (const PrimRefMB& prim, const BBox1f time_range) const
        {
          const unsigned int geomID  = prim.geomID();
          const GridMesh* mesh = scene->get<GridMesh>(geomID);
          const unsigned int buildID = prim.primID();
          const SubGridBuildData &subgrid = sgrids[buildID];                      
          const unsigned int primID = subgrid.primID;
          const size_t x = subgrid.x();
          const size_t y = subgrid.y();
          const LBBox3fa lbounds = mesh->linearBounds(mesh->grid(primID),x,y,time_range);
          const unsigned num_time_segments = mesh->numTimeSegments();
          const range<int> tbounds = mesh->timeSegmentRange(time_range);
          return PrimRefMB (lbounds, tbounds.size(), mesh->time_range, num_time_segments, geomID, buildID);
        }

        __forceinline LBBox3fa linearBounds(const PrimRefMB& prim, const BBox1f time_range) const {
          const unsigned int geomID  = prim.geomID();
          const GridMesh* mesh = scene->get<GridMesh>(geomID);
          const unsigned int buildID = prim.primID();
          const SubGridBuildData &subgrid = sgrids[buildID];                      
          const unsigned int primID = subgrid.primID;
          const size_t x = subgrid.x();
          const size_t y = subgrid.y();
          return mesh->linearBounds(mesh->grid(primID),x,y,time_range);
        }

    };

    template<int N>
    struct CreateMSMBlurLeafGrid
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecordMB4D NodeRecordMB4D;

      __forceinline CreateMSMBlurLeafGrid (Scene* scene, BVH* bvh, const SubGridBuildData * const sgrids) : scene(scene), bvh(bvh), sgrids(sgrids) {}

      __forceinline const NodeRecordMB4D operator() (const BVHBuilderMSMBlur::BuildRecord& current, const FastAllocator::CachedAllocator& alloc) const
      {
        const size_t items = current.prims.size(); 
        const size_t start = current.prims.begin();

        const PrimRefMB* prims = current.prims.prims->data();
        /* collect all subsets with unique geomIDs */
        assert(items <= N);
        unsigned int geomIDs[N];
        unsigned int num_geomIDs = 1;
        geomIDs[0] = prims[start].geomID();

        for (size_t i=1;i<items;i++)
        {
          bool found = false;
          const unsigned int new_geomID = prims[start+i].geomID();
          for (size_t j=0;j<num_geomIDs;j++)
            if (new_geomID == geomIDs[j])
            { found = true; break; }
          if (!found) 
            geomIDs[num_geomIDs++] = new_geomID;
        }

        /* allocate all leaf memory in one single block */
        SubGridMBQBVHN<N>* accel = (SubGridMBQBVHN<N>*) alloc.malloc1(num_geomIDs*sizeof(SubGridMBQBVHN<N>),BVH::byteAlignment);
        typename BVH::NodeRef node = bvh->encodeLeaf((char*)accel,num_geomIDs);

        LBBox3fa allBounds = empty;

        for (size_t g=0;g<num_geomIDs;g++)
        {
          const GridMesh* __restrict__ const mesh = scene->get<GridMesh>(geomIDs[g]);
          unsigned int x[N];
          unsigned int y[N];
          unsigned int primID[N];
          BBox3fa bounds0[N];
          BBox3fa bounds1[N];
          unsigned int pos = 0;
          for (size_t i=0;i<items;i++)
          {
            if (unlikely(prims[start+i].geomID() != geomIDs[g])) continue;

            const SubGridBuildData  &sgrid_bd = sgrids[prims[start+i].primID()];                      
            x[pos] = sgrid_bd.sx;
            y[pos] = sgrid_bd.sy;
            primID[pos] = sgrid_bd.primID;
            const size_t x = sgrid_bd.x();
            const size_t y = sgrid_bd.y();
            LBBox3fa newBounds = mesh->linearBounds(mesh->grid(sgrid_bd.primID),x,y,current.prims.time_range);
            allBounds.extend(newBounds);
            bounds0[pos] = newBounds.bounds0;
            bounds1[pos] = newBounds.bounds1;
            pos++;
          }
          assert(pos <= N);
          new (&accel[g]) SubGridMBQBVHN<N>(x,y,primID,bounds0,bounds1,geomIDs[g],current.prims.time_range.lower,1.0f/current.prims.time_range.size(),pos);
        }
        return NodeRecordMB4D(node,allBounds,current.prims.time_range);       
      }

      Scene *scene;
      BVH* bvh;
      const SubGridBuildData * const sgrids;
    };

#if 0
    template<int N>
    struct CreateLeafGridMB
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecordMB NodeRecordMB;

      __forceinline CreateLeafGridMB (Scene* scene, BVH* bvh, const SubGridBuildData * const sgrids) 
		  : scene(scene), bvh(bvh), sgrids(sgrids) {}

      __forceinline NodeRecordMB operator() (const PrimRef* prims, const range<size_t>& set, const FastAllocator::CachedAllocator& alloc) const
      {
        const size_t items = set.size(); 
        const size_t start = set.begin();

        /* collect all subsets with unique geomIDs */
        assert(items <= N);
        unsigned int geomIDs[N];
        unsigned int num_geomIDs = 1;
        geomIDs[0] = prims[start].geomID();

        for (size_t i=1;i<items;i++)
        {
          bool found = false;
          const unsigned int new_geomID = prims[start+i].geomID();
          for (size_t j=0;j<num_geomIDs;j++)
            if (new_geomID == geomIDs[j])
            { found = true; break; }
          if (!found) 
            geomIDs[num_geomIDs++] = new_geomID;
        }

        /* allocate all leaf memory in one single block */
        SubGridMBQBVHN<N>* accel = (SubGridMBQBVHN<N>*) alloc.malloc1(num_geomIDs*sizeof(SubGridMBQBVHN<N>),BVH::byteAlignment);
        typename BVH::NodeRef node = bvh->encodeLeaf((char*)accel,num_geomIDs);

        LBBox3fa allBounds = empty;

        for (size_t g=0;g<num_geomIDs;g++)
        {
          const GridMesh* __restrict__ const mesh = scene->get<GridMesh>(geomIDs[g]);

          unsigned int x[N];
          unsigned int y[N];
          unsigned int primID[N];
          BBox3fa bounds0[N];
          BBox3fa bounds1[N];
          unsigned int pos = 0;
          for (size_t i=0;i<items;i++)
          {
            if (unlikely(prims[start+i].geomID() != geomIDs[g])) continue;

            const SubGridBuildData  &sgrid_bd = sgrids[prims[start+i].primID()];                      
            x[pos] = sgrid_bd.sx;
            y[pos] = sgrid_bd.sy;
            primID[pos] = sgrid_bd.primID;
            const size_t x = sgrid_bd.x();
            const size_t y = sgrid_bd.y();
            bool MAYBE_UNUSED valid0 = mesh->buildBounds(mesh->grid(sgrid_bd.primID),x,y,0,bounds0[pos]);
            bool MAYBE_UNUSED valid1 = mesh->buildBounds(mesh->grid(sgrid_bd.primID),x,y,1,bounds1[pos]);
            assert(valid0);
            assert(valid1);
            allBounds.extend(LBBox3fa(bounds0[pos],bounds1[pos]));
            pos++;
          }
          new (&accel[g]) SubGridMBQBVHN<N>(x,y,primID,bounds0,bounds1,geomIDs[g],0.0f,1.0f,pos);
        }
        return NodeRecordMB(node,allBounds);
      }

      Scene *scene;
      BVH* bvh;
      const SubGridBuildData * const sgrids;
    };
#endif


    /* Motion blur BVH with 4D nodes and internal time splits */
    template<int N>
    struct BVHNBuilderMBlurSAHGrid : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVHN<N>::NodeRef NodeRef;
      typedef typename BVHN<N>::NodeRecordMB NodeRecordMB;
      typedef typename BVHN<N>::AABBNodeMB AABBNodeMB;

      BVH* bvh;
      Scene* scene;
      const size_t sahBlockSize;
      const float intCost;
      const size_t minLeafSize;
      const size_t maxLeafSize;
      mvector<SubGridBuildData> sgrids;


      BVHNBuilderMBlurSAHGrid (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize)
        : bvh(bvh), scene(scene), sahBlockSize(sahBlockSize), intCost(intCost), minLeafSize(minLeafSize), maxLeafSize(min(maxLeafSize,BVH::maxLeafBlocks)), sgrids(scene->device,0) {}


      PrimInfo createPrimRefArrayMBlurGrid(Scene* scene, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor, size_t itime)
      {
        /* first run to get #primitives */
        ParallelForForPrefixSumState<PrimInfo> pstate;
        Scene::Iterator<GridMesh,true> iter(scene);

        pstate.init(iter,size_t(1024));

        /* iterate over all meshes in the scene */
        PrimInfo pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfo(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, size_t geomID) -> PrimInfo {
            
            PrimInfo pinfo(empty);
            for (size_t j=r.begin(); j<r.end(); j++)
            {
              if (!mesh->valid(j,range<size_t>(0,1))) continue;
              BBox3fa bounds = empty;
              const PrimRef prim(bounds,unsigned(geomID),unsigned(j));
              pinfo.add_center2(prim,mesh->getNumSubGrids(j));
            }
            return pinfo;
          }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
        
        size_t numPrimitives = pinfo.size();
        if (numPrimitives == 0) return pinfo;

        /* resize arrays */
        sgrids.resize(numPrimitives); 
        prims.resize(numPrimitives); 

        /* second run to fill primrefs and SubGridBuildData arrays */
        pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfo(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfo& base) -> PrimInfo {
            
            k = base.size();
            size_t p_index = k;
            PrimInfo pinfo(empty);
            for (size_t j=r.begin(); j<r.end(); j++)
            {
              const GridMesh::Grid &g = mesh->grid(j);
              if (!mesh->valid(j,range<size_t>(0,1))) continue;
              
              for (unsigned int y=0; y<g.resY-1u; y+=2)
                for (unsigned int x=0; x<g.resX-1u; x+=2)
                {
                  BBox3fa bounds = empty;
                  if (!mesh->buildBounds(g,x,y,itime,bounds)) continue; // get bounds of subgrid
                  const PrimRef prim(bounds,unsigned(geomID),unsigned(p_index));
                  pinfo.add_center2(prim);
                  sgrids[p_index] = SubGridBuildData(x | g.get3x3FlagsX(x), y | g.get3x3FlagsY(y), unsigned(j));
                                                      prims[p_index++] = prim;                
                }
            }
            return pinfo;
          }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
        
        assert(pinfo.size() == numPrimitives);
        return pinfo;
      }

      PrimInfoMB createPrimRefArrayMSMBlurGrid(Scene* scene, mvector<PrimRefMB>& prims, BuildProgressMonitor& progressMonitor, BBox1f t0t1 = BBox1f(0.0f,1.0f))
      {
        /* first run to get #primitives */
        ParallelForForPrefixSumState<PrimInfoMB> pstate;
        Scene::Iterator<GridMesh,true> iter(scene);

        pstate.init(iter,size_t(1024));
        /* iterate over all meshes in the scene */
        PrimInfoMB pinfoMB = parallel_for_for_prefix_sum0( pstate, iter, PrimInfoMB(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, size_t /*geomID*/) -> PrimInfoMB {
            
            PrimInfoMB pinfoMB(empty);
            for (size_t j=r.begin(); j<r.end(); j++)
            {
              if (!mesh->valid(j, mesh->timeSegmentRange(t0t1))) continue;
              LBBox3fa bounds(empty);
              PrimInfoMB gridMB(0,mesh->getNumSubGrids(j));
              pinfoMB.merge(gridMB);
            }
            return pinfoMB;
          }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
        
        size_t numPrimitives = pinfoMB.size();
        if (numPrimitives == 0) return pinfoMB;

        /* resize arrays */
        sgrids.resize(numPrimitives); 
        prims.resize(numPrimitives); 
        /* second run to fill primrefs and SubGridBuildData arrays */
        pinfoMB = parallel_for_for_prefix_sum1( pstate, iter, PrimInfoMB(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfoMB& base) -> PrimInfoMB {
            
            k = base.size();
            size_t p_index = k;
            PrimInfoMB pinfoMB(empty);
            for (size_t j=r.begin(); j<r.end(); j++)
            {
              if (!mesh->valid(j, mesh->timeSegmentRange(t0t1))) continue;
              const GridMesh::Grid &g = mesh->grid(j);
              
              for (unsigned int y=0; y<g.resY-1u; y+=2)
                for (unsigned int x=0; x<g.resX-1u; x+=2)
                {
                  const PrimRefMB prim(mesh->linearBounds(g,x,y,t0t1),mesh->numTimeSegments(),mesh->time_range,mesh->numTimeSegments(),unsigned(geomID),unsigned(p_index));
                  pinfoMB.add_primref(prim);
                  sgrids[p_index] = SubGridBuildData(x | g.get3x3FlagsX(x), y | g.get3x3FlagsY(y), unsigned(j));
                  prims[p_index++] = prim;                
                }
            }
            return pinfoMB;
          }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
        
        assert(pinfoMB.size() == numPrimitives);
        pinfoMB.time_range = t0t1;
        return pinfoMB;
      }

      void build()
      {
	/* skip build for empty scene */
        const size_t numPrimitives = scene->getNumPrimitives(GridMesh::geom_type,true);
        if (numPrimitives == 0) { bvh->clear(); return; }

        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH" + toString(N) + "BuilderMBlurSAHGrid");

        //const size_t numTimeSteps = scene->getNumTimeSteps<GridMesh,true>();
        //const size_t numTimeSegments = numTimeSteps-1; assert(numTimeSteps > 1);
        //if (numTimeSegments == 1)
        //  buildSingleSegment(numPrimitives);
        //else
        buildMultiSegment(numPrimitives);

	/* clear temporary data for static geometry */
	bvh->cleanup();
        bvh->postBuild(t0);
      }

#if 0
      void buildSingleSegment(size_t numPrimitives)
      {
        /* create primref array */
        mvector<PrimRef> prims(scene->device,numPrimitives);
        const PrimInfo pinfo = createPrimRefArrayMBlurGrid(scene,prims,bvh->scene->progressInterface,0);
        /* early out if no valid primitives */
        if (pinfo.size() == 0) { bvh->clear(); return; }

        /* estimate acceleration structure size */
        const size_t node_bytes = pinfo.size()*sizeof(AABBNodeMB)/(4*N);
        //TODO: check leaf_bytes
        const size_t leaf_bytes = size_t(1.2*(float)numPrimitives/N * sizeof(SubGridQBVHN<N>));
        bvh->alloc.init_estimate(node_bytes+leaf_bytes);

        /* settings for BVH build */
        GeneralBVHBuilder::Settings settings;
        settings.branchingFactor = N;
        settings.maxDepth = BVH::maxBuildDepthLeaf;
        settings.logBlockSize = bsr(sahBlockSize);
        settings.minLeafSize = min(minLeafSize,maxLeafSize);
        settings.maxLeafSize = maxLeafSize;
        settings.travCost = travCost;
        settings.intCost = intCost;
        settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,pinfo.size(),node_bytes+leaf_bytes);

        /* build hierarchy */
        auto root = BVHBuilderBinnedSAH::build<NodeRecordMB>
          (typename BVH::CreateAlloc(bvh),
           typename BVH::AABBNodeMB::Create(),
           typename BVH::AABBNodeMB::Set(),
           CreateLeafGridMB<N>(scene,bvh,sgrids.data()),
           bvh->scene->progressInterface,
           prims.data(),pinfo,settings);

        bvh->set(root.ref,root.lbounds,pinfo.size());
      }
#endif
      
      void buildMultiSegment(size_t numPrimitives)
      {
        /* create primref array */
        mvector<PrimRefMB> prims(scene->device,numPrimitives);
        PrimInfoMB pinfo = createPrimRefArrayMSMBlurGrid(scene,prims,bvh->scene->progressInterface);

        /* early out if no valid primitives */
        if (pinfo.size() == 0) { bvh->clear(); return; }



        GridRecalculatePrimRef recalculatePrimRef(scene,sgrids.data());

        /* estimate acceleration structure size */
        const size_t node_bytes = pinfo.num_time_segments*sizeof(AABBNodeMB)/(4*N);
        //FIXME: check leaf_bytes
        //const size_t leaf_bytes = size_t(1.2*Primitive::blocks(pinfo.num_time_segments)*sizeof(SubGridQBVHN<N>));
        const size_t leaf_bytes = size_t(1.2*(float)numPrimitives/N * sizeof(SubGridQBVHN<N>));

        bvh->alloc.init_estimate(node_bytes+leaf_bytes);

        /* settings for BVH build */
        BVHBuilderMSMBlur::Settings settings;
        settings.branchingFactor = N;
        settings.maxDepth = BVH::maxDepth;
        settings.logBlockSize = bsr(sahBlockSize);
        settings.minLeafSize = min(minLeafSize,maxLeafSize);
        settings.maxLeafSize = maxLeafSize;
        settings.travCost = travCost;
        settings.intCost = intCost;
        settings.singleLeafTimeSegment = false; 
        settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,pinfo.size(),node_bytes+leaf_bytes);
        
        /* build hierarchy */
        auto root =
          BVHBuilderMSMBlur::build<NodeRef>(prims,pinfo,scene->device,
                                            recalculatePrimRef,
                                            typename BVH::CreateAlloc(bvh),
                                            typename BVH::AABBNodeMB4D::Create(),
                                            typename BVH::AABBNodeMB4D::Set(),
                                            CreateMSMBlurLeafGrid<N>(scene,bvh,sgrids.data()),
                                            bvh->scene->progressInterface,
                                            settings);
        bvh->set(root.ref,root.lbounds,pinfo.num_time_segments);
      }

      void clear() {
      }
    };

    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/

#if defined(EMBREE_GEOMETRY_TRIANGLE)
    Builder* BVH4Triangle4iMBSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderMBlurSAH<4,TriangleMesh,Triangle4i>((BVH4*)bvh,scene,4,1.0f,4,inf,Geometry::MTY_TRIANGLE_MESH); }
    Builder* BVH4Triangle4vMBSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderMBlurSAH<4,TriangleMesh,Triangle4vMB>((BVH4*)bvh,scene,4,1.0f,4,inf,Geometry::MTY_TRIANGLE_MESH); }
#if defined(__AVX__)
    Builder* BVH8Triangle4iMBSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderMBlurSAH<8,TriangleMesh,Triangle4i>((BVH8*)bvh,scene,4,1.0f,4,inf,Geometry::MTY_TRIANGLE_MESH); }
    Builder* BVH8Triangle4vMBSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderMBlurSAH<8,TriangleMesh,Triangle4vMB>((BVH8*)bvh,scene,4,1.0f,4,inf,Geometry::MTY_TRIANGLE_MESH); }
#endif
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
    Builder* BVH4Quad4iMBSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderMBlurSAH<4,QuadMesh,Quad4i>((BVH4*)bvh,scene,4,1.0f,4,inf,Geometry::MTY_QUAD_MESH); }
#if defined(__AVX__)
    Builder* BVH8Quad4iMBSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderMBlurSAH<8,QuadMesh,Quad4i>((BVH8*)bvh,scene,4,1.0f,4,inf,Geometry::MTY_QUAD_MESH); }
#endif
#endif

#if defined(EMBREE_GEOMETRY_USER)
    Builder* BVH4VirtualMBSceneBuilderSAH    (void* bvh, Scene* scene, size_t mode) {
      int minLeafSize = scene->device->object_accel_mb_min_leaf_size;
      int maxLeafSize = scene->device->object_accel_mb_max_leaf_size;
      return new BVHNBuilderMBlurSAH<4,UserGeometry,Object>((BVH4*)bvh,scene,4,1.0f,minLeafSize,maxLeafSize,Geometry::MTY_USER_GEOMETRY);
    }
#if defined(__AVX__)
    Builder* BVH8VirtualMBSceneBuilderSAH    (void* bvh, Scene* scene, size_t mode) {
      int minLeafSize = scene->device->object_accel_mb_min_leaf_size;
      int maxLeafSize = scene->device->object_accel_mb_max_leaf_size;
      return new BVHNBuilderMBlurSAH<8,UserGeometry,Object>((BVH8*)bvh,scene,8,1.0f,minLeafSize,maxLeafSize,Geometry::MTY_USER_GEOMETRY);
    }
#endif
#endif

#if defined(EMBREE_GEOMETRY_INSTANCE)
    Builder* BVH4InstanceMBSceneBuilderSAH (void* bvh, Scene* scene, Geometry::GTypeMask gtype) { return new BVHNBuilderMBlurSAH<4,Instance,InstancePrimitive>((BVH4*)bvh,scene,4,1.0f,1,1,gtype); }
#if defined(__AVX__)
    Builder* BVH8InstanceMBSceneBuilderSAH (void* bvh, Scene* scene, Geometry::GTypeMask gtype) { return new BVHNBuilderMBlurSAH<8,Instance,InstancePrimitive>((BVH8*)bvh,scene,8,1.0f,1,1,gtype); }
#endif
#endif

#if defined(EMBREE_GEOMETRY_GRID)
    Builder* BVH4GridMBSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderMBlurSAHGrid<4>((BVH4*)bvh,scene,4,1.0f,4,4); }
#if defined(__AVX__)
    Builder* BVH8GridMBSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderMBlurSAHGrid<8>((BVH8*)bvh,scene,8,1.0f,8,8); }
#endif
#endif
  }
}
