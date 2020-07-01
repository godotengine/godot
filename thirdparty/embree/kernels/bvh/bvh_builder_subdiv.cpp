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

#include "bvh.h"

#if defined(EMBREE_GEOMETRY_SUBDIVISION)

#include "bvh_refit.h"
#include "bvh_builder.h"

#include "../builders/primrefgen.h"
#include "../builders/bvh_builder_sah.h"
#include "../builders/bvh_builder_msmblur.h"

#include "../../common/algorithms/parallel_for_for.h"
#include "../../common/algorithms/parallel_for_for_prefix_sum.h"

#include "../subdiv/bezier_curve.h"
#include "../subdiv/bspline_curve.h"

#include "../geometry/subdivpatch1.h"
#include "../geometry/grid_soa.h"

namespace embree
{
  namespace isa
  {
    typedef FastAllocator::CachedAllocator Allocator;

    template<int N>
    struct BVHNSubdivPatch1BuilderSAH : public Builder
    {
      ALIGNED_STRUCT_(64);

      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;

      BVH* bvh;
      Scene* scene;
      mvector<PrimRef> prims;
            
      BVHNSubdivPatch1BuilderSAH (BVH* bvh, Scene* scene)
        : bvh(bvh), scene(scene), prims(scene->device,0) {}

#define SUBGRID 9

      static unsigned getNumEagerLeaves(unsigned pwidth, unsigned pheight) {
        const unsigned swidth = pwidth-1;
        const unsigned sheight = pheight-1;
        const unsigned sblock = SUBGRID-1;
        const unsigned w = (swidth+sblock-1)/sblock;
        const unsigned h = (sheight+sblock-1)/sblock;
        return w*h;
      }

      __forceinline static unsigned createEager(SubdivPatch1Base& patch, Scene* scene, SubdivMesh* mesh, unsigned primID, Allocator& alloc, PrimRef* prims)
      {
        unsigned NN = 0;
        const unsigned x0 = 0, x1 = patch.grid_u_res-1;
        const unsigned y0 = 0, y1 = patch.grid_v_res-1;
        
        for (unsigned y=y0; y<y1; y+=SUBGRID-1)
        {
          for (unsigned x=x0; x<x1; x+=SUBGRID-1) 
          {
            const unsigned lx0 = x, lx1 = min(lx0+SUBGRID-1,x1);
            const unsigned ly0 = y, ly1 = min(ly0+SUBGRID-1,y1);
            BBox3fa bounds;
            GridSOA* leaf = GridSOA::create(&patch,1,lx0,lx1,ly0,ly1,scene,alloc,&bounds);
            *prims = PrimRef(bounds,BVH4::encodeTypedLeaf(leaf,1)); prims++;
            NN++;
          }
        }
        return NN;
      }

      void build() 
      {
        /* skip build for empty scene */
        const size_t numPrimitives = scene->getNumPrimitives<SubdivMesh,false>();
        if (numPrimitives == 0) {
          prims.resize(numPrimitives);
          bvh->set(BVH::emptyNode,empty,0);
          return;
        }
 
        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH" + toString(N) + "SubdivPatch1BuilderSAH");

        //bvh->alloc.reset();
        bvh->alloc.init_estimate(numPrimitives*sizeof(PrimRef));

        auto progress = [&] (size_t dn) { bvh->scene->progressMonitor(double(dn)); };
        auto virtualprogress = BuildProgressMonitorFromClosure(progress);

        ParallelForForPrefixSumState<PrimInfo> pstate;
        
        /* initialize allocator and parallel_for_for_prefix_sum */
        Scene::Iterator<SubdivMesh> iter(scene);
        pstate.init(iter,size_t(1024));

        PrimInfo pinfo1 = parallel_for_for_prefix_sum0( pstate, iter, PrimInfo(empty), [&](SubdivMesh* mesh, const range<size_t>& r, size_t k) -> PrimInfo
        { 
          size_t p = 0;
          size_t g = 0;
          for (size_t f=r.begin(); f!=r.end(); ++f) {          
            if (!mesh->valid(f)) continue;
            patch_eval_subdivision(mesh->getHalfEdge(0,f),[&](const Vec2f uv[4], const int subdiv[4], const float edge_level[4], int subPatch)
            {
              float level[4]; SubdivPatch1Base::computeEdgeLevels(edge_level,subdiv,level);
              Vec2i grid = SubdivPatch1Base::computeGridSize(level);
              size_t num = getNumEagerLeaves(grid.x,grid.y);
              g+=num;
              p++;
            });
          }
          return PrimInfo(p,g,empty);
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo(a.begin+b.begin,a.end+b.end,empty); });
        size_t numSubPatches = pinfo1.begin;
        if (numSubPatches == 0) {
          bvh->set(BVH::emptyNode,empty,0);
          return;
        }

        prims.resize(pinfo1.end);
        if (pinfo1.end == 0) {
          bvh->set(BVH::emptyNode,empty,0);
          return;
        }

        PrimInfo pinfo3 = parallel_for_for_prefix_sum1( pstate, iter, PrimInfo(empty), [&](SubdivMesh* mesh, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo
        {
          Allocator alloc = bvh->alloc.getCachedAllocator();
          
          PrimInfo s(empty);
          for (size_t f=r.begin(); f!=r.end(); ++f) {
            if (!mesh->valid(f)) continue;
            
            patch_eval_subdivision(mesh->getHalfEdge(0,f),[&](const Vec2f uv[4], const int subdiv[4], const float edge_level[4], int subPatch)
            {
              SubdivPatch1Base patch(mesh->geomID,unsigned(f),subPatch,mesh,0,uv,edge_level,subdiv,VSIZEX);
              size_t num = createEager(patch,scene,mesh,unsigned(f),alloc,&prims[base.end+s.end]);
              assert(num == getNumEagerLeaves(patch.grid_u_res,patch.grid_v_res));
              for (size_t i=0; i<num; i++)
                s.add_center2(prims[base.end+s.end]);
              s.begin++;
            });
          }
          return s;
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a, b); });

        PrimInfo pinfo(0,pinfo3.end,pinfo3);
        
        auto createLeaf = [&] (const PrimRef* prims, const range<size_t>& range, Allocator alloc) -> NodeRef {
          assert(range.size() == 1);
          size_t leaf = (size_t) prims[range.begin()].ID();
          return NodeRef(leaf);
        };

        /* settings for BVH build */
        GeneralBVHBuilder::Settings settings;
        settings.logBlockSize = bsr(N);
        settings.minLeafSize = 1;
        settings.maxLeafSize = 1;
        settings.travCost = 1.0f;
        settings.intCost = 1.0f;
        settings.singleThreadThreshold = DEFAULT_SINGLE_THREAD_THRESHOLD;

        NodeRef root = BVHNBuilderVirtual<N>::build(&bvh->alloc,createLeaf,virtualprogress,prims.data(),pinfo,settings);
        bvh->set(root,LBBox3fa(pinfo.geomBounds),pinfo.size());
        bvh->layoutLargeNodes(size_t(pinfo.size()*0.005f));
        
	/* clear temporary data for static geometry */
	if (scene->isStaticAccel()) {
          prims.clear();
          bvh->shrink();
        }
        bvh->cleanup();
        bvh->postBuild(t0);
      }

      void clear() {
        prims.clear();
      }
    };

    // =======================================================================================================
    // =======================================================================================================
    // =======================================================================================================

    struct SubdivRecalculatePrimRef
    {
      mvector<BBox3fa>& bounds;
      SubdivPatch1* patches;

      __forceinline SubdivRecalculatePrimRef (mvector<BBox3fa>& bounds, SubdivPatch1* patches)
        : bounds(bounds), patches(patches) {}

      __forceinline PrimRefMB operator() (const size_t patchIndexMB, const BBox1f prim_time_range, const unsigned prim_num_time_segments, const BBox1f time_range) const
      {
        const LBBox3fa lbounds = LBBox3fa([&] (size_t itime) { return bounds[patchIndexMB+itime]; }, time_range, prim_time_range, (float)prim_num_time_segments);
        const range<int> tbounds = getTimeSegmentRange(time_range, prim_time_range, (float)prim_num_time_segments);
        return PrimRefMB (empty, lbounds, tbounds.size(), prim_time_range, prim_num_time_segments, patchIndexMB);
      }

      __forceinline PrimRefMB operator() (const PrimRefMB& prim, const BBox1f time_range) const {
        return operator()(prim.ID(),prim.time_range,prim.totalTimeSegments(),time_range);
      }

      __forceinline LBBox3fa linearBounds(const PrimRefMB& prim, const BBox1f time_range) const {
        return LBBox3fa([&] (size_t itime) { return bounds[prim.ID()+itime]; }, time_range, prim.time_range, (float)prim.totalTimeSegments());
      }
    };

    template<int N>
    struct BVHNSubdivPatch1MBlurBuilderSAH : public Builder
    {
      ALIGNED_STRUCT_(64);

      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecordMB4D NodeRecordMB4D;
      typedef typename BVHN<N>::AlignedNodeMB AlignedNodeMB;
      typedef typename BVHN<N>::AlignedNodeMB4D AlignedNodeMB4D;
      typedef typename BVHN<N>::Allocator BVH_Allocator;

      typedef SetMB Set;

      BVH* bvh;
      Scene* scene;
      mvector<PrimRefMB> primsMB;
      mvector<BBox3fa> bounds;
      
      BVHNSubdivPatch1MBlurBuilderSAH (BVH* bvh, Scene* scene)
        : bvh(bvh), scene(scene), primsMB(scene->device,0), bounds(scene->device,0) {}

      void countSubPatches(size_t& numSubPatches, size_t& numSubPatchesMB, ParallelForForPrefixSumState<PrimInfoMB>& pstate)
      {
        Scene::Iterator<SubdivMesh,true> iter(scene);
        pstate.init(iter,size_t(1024));

        PrimInfoMB pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfoMB(empty), [&](SubdivMesh* mesh, const range<size_t>& r, size_t k) -> PrimInfoMB
        { 
          size_t s = 0;
          size_t sMB = 0;
          for (size_t f=r.begin(); f!=r.end(); ++f) 
          {          
            if (!mesh->valid(f)) continue;
            size_t count = patch_eval_subdivision_count(mesh->getHalfEdge(0,f));
            s += count;
            sMB += count * mesh->numTimeSteps;
          }
          return PrimInfoMB(s,sMB);
        }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB(a.begin()+b.begin(),a.end()+b.end()); });

        numSubPatches = pinfo.begin();
        numSubPatchesMB = pinfo.end();
      }

      void rebuild(size_t numPrimitives, ParallelForForPrefixSumState<PrimInfoMB>& pstate)
      {
        SubdivPatch1* const subdiv_patches = (SubdivPatch1*) bvh->subdiv_patches.data();
        SubdivRecalculatePrimRef recalculatePrimRef(bounds,subdiv_patches);
        bvh->alloc.reset();

        Scene::Iterator<SubdivMesh,true> iter(scene);
        PrimInfoMB pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfoMB(empty), [&](SubdivMesh* mesh, const range<size_t>& r, size_t k, const PrimInfoMB& base) -> PrimInfoMB
        {
          size_t s = 0;
          size_t sMB = 0;
          PrimInfoMB pinfo(empty);
          for (size_t f=r.begin(); f!=r.end(); ++f) 
          {
            if (!mesh->valid(f)) continue;
            
            BVH_Allocator alloc(bvh);
            patch_eval_subdivision(mesh->getHalfEdge(0,f),[&](const Vec2f uv[4], const int subdiv[4], const float edge_level[4], int subPatch)
            {
              const size_t patchIndex = base.begin()+s;
              const size_t patchIndexMB = base.end()+sMB;
              assert(patchIndex < numPrimitives);

              for (size_t t=0; t<mesh->numTimeSteps; t++)
              {
                SubdivPatch1Base& patch = subdiv_patches[patchIndexMB+t];
                new (&patch) SubdivPatch1(mesh->geomID,unsigned(f),subPatch,mesh,t,uv,edge_level,subdiv,VSIZEX);
              }
              SubdivPatch1Base& patch0 = subdiv_patches[patchIndexMB];
              patch0.root_ref.set((int64_t) GridSOA::create(&patch0,(unsigned)mesh->numTimeSteps,scene,alloc,&bounds[patchIndexMB]));
              primsMB[patchIndex] = recalculatePrimRef(patchIndexMB,mesh->time_range,mesh->numTimeSegments(),BBox1f(0.0f,1.0f));
              s++;
              sMB += mesh->numTimeSteps;
              pinfo.add_primref(primsMB[patchIndex]);
            });
          }
          pinfo.object_range._begin = s;
          pinfo.object_range._end = sMB;
          return pinfo;
        }, [](const PrimInfoMB& a, const PrimInfoMB& b) -> PrimInfoMB { return PrimInfoMB::merge2(a,b); });
        pinfo.object_range._end = pinfo.begin();
        pinfo.object_range._begin = 0;

        auto createLeafFunc = [&] (const BVHBuilderMSMBlur::BuildRecord& current, const Allocator& alloc) -> NodeRecordMB4D {
          mvector<PrimRefMB>& prims = *current.prims.prims;
          size_t items MAYBE_UNUSED = current.prims.size();
          assert(items == 1);
          const size_t patchIndexMB = prims[current.prims.begin()].ID();
          SubdivPatch1Base& patch = subdiv_patches[patchIndexMB+0];
          NodeRef node = bvh->encodeLeaf((char*)&patch,1);
          size_t patchNumTimeSteps = scene->get<SubdivMesh>(patch.geomID())->numTimeSteps;
          const LBBox3fa lbounds = LBBox3fa([&] (size_t itime) { return bounds[patchIndexMB+itime]; }, current.prims.time_range, (float)(patchNumTimeSteps-1));
          return NodeRecordMB4D(node,lbounds,current.prims.time_range);
        };

        /* configuration for BVH build */
        BVHBuilderMSMBlur::Settings settings;
        settings.branchingFactor = N;
        settings.maxDepth = BVH::maxDepth;
        settings.logBlockSize = bsr(N);
        settings.minLeafSize = 1;
        settings.maxLeafSize = 1;
        settings.travCost = 1.0f;
        settings.intCost = 1.0f;
        settings.singleLeafTimeSegment = false;

        /* build hierarchy */
        auto root =
          BVHBuilderMSMBlur::build<NodeRef>(primsMB,pinfo,scene->device,
                                             recalculatePrimRef,
                                             typename BVH::CreateAlloc(bvh),
                                             typename BVH::AlignedNodeMB4D::Create(),
                                             typename BVH::AlignedNodeMB4D::Set(),
                                             createLeafFunc,
                                             bvh->scene->progressInterface,
                                             settings);
        
        bvh->set(root.ref,root.lbounds,pinfo.num_time_segments);
      }

      void build() 
      {
        /* initialize all half edge structures */
        size_t numPatches = scene->getNumPrimitives<SubdivMesh,true>();

        /* skip build for empty scene */
        if (numPatches == 0) {
          primsMB.resize(0);
          bounds.resize(0);
          bvh->set(BVH::emptyNode,empty,0);
          return;
        }

        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH" + toString(N) + "SubdivPatch1MBlurBuilderSAH");

        ParallelForForPrefixSumState<PrimInfoMB> pstate;

        /* calculate number of primitives (some patches need initial subdivision) */
        size_t numSubPatches, numSubPatchesMB;
        countSubPatches(numSubPatches, numSubPatchesMB, pstate);
        primsMB.resize(numSubPatches);
        bounds.resize(numSubPatchesMB);
        
        /* exit if there are no primitives to process */
        if (numSubPatches == 0) {
          bvh->set(BVH::emptyNode,empty,0);
          bvh->postBuild(t0);
          return;
        }
        
        /* Allocate memory for gregory and b-spline patches */
        bvh->subdiv_patches.resize(sizeof(SubdivPatch1) * numSubPatchesMB);

        /* rebuild BVH */
        rebuild(numSubPatches, pstate);
        
	/* clear temporary data for static geometry */
	if (scene->isStaticAccel()) {
          primsMB.clear();
          bvh->shrink();
        }
        bvh->cleanup();
        bvh->postBuild(t0);        
      }
      
      void clear() {
        primsMB.clear();
      }
    };
    
    /* entry functions for the scene builder */
    Builder* BVH4SubdivPatch1BuilderSAH(void* bvh, Scene* scene, size_t mode) { return new BVHNSubdivPatch1BuilderSAH<4>((BVH4*)bvh,scene); }
    Builder* BVH4SubdivPatch1MBBuilderSAH(void* bvh, Scene* scene, size_t mode) { return new BVHNSubdivPatch1MBlurBuilderSAH<4>((BVH4*)bvh,scene); }
  }
}
#endif
