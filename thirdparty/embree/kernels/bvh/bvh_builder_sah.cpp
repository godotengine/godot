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
#include "bvh_builder.h"
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
#include "../../common/algorithms/parallel_for_for.h"
#include "../../common/algorithms/parallel_for_for_prefix_sum.h"

#define PROFILE 0
#define PROFILE_RUNS 20

namespace embree
{
  namespace isa
  {
    template<int N, typename Primitive>
    struct CreateLeaf
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;

      __forceinline CreateLeaf (BVH* bvh) : bvh(bvh) {}

      __forceinline NodeRef operator() (const PrimRef* prims, const range<size_t>& set, const FastAllocator::CachedAllocator& alloc) const
      {
        size_t n = set.size();
        size_t items = Primitive::blocks(n);
        size_t start = set.begin();
        Primitive* accel = (Primitive*) alloc.malloc1(items*sizeof(Primitive),BVH::byteAlignment);
        typename BVH::NodeRef node = BVH::encodeLeaf((char*)accel,items);
        for (size_t i=0; i<items; i++) {
          accel[i].fill(prims,start,set.end(),bvh->scene);
        }
        return node;
      }

      BVH* bvh;
    };


    template<int N, typename Primitive>
    struct CreateLeafQuantized
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;

      __forceinline CreateLeafQuantized (BVH* bvh) : bvh(bvh) {}

      __forceinline NodeRef operator() (const PrimRef* prims, const range<size_t>& set, const FastAllocator::CachedAllocator& alloc) const
      {
        size_t n = set.size();
        size_t items = Primitive::blocks(n);
        size_t start = set.begin();
        Primitive* accel = (Primitive*) alloc.malloc1(items*sizeof(Primitive),BVH::byteAlignment);
        typename BVH::NodeRef node = BVH::encodeLeaf((char*)accel,items);
        for (size_t i=0; i<items; i++) {
          accel[i].fill(prims,start,set.end(),bvh->scene);
        }
        return node;
      }

      BVH* bvh;
    };

    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/

    template<int N, typename Mesh, typename Primitive>
    struct BVHNBuilderSAH : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVHN<N>::NodeRef NodeRef;

      BVH* bvh;
      Scene* scene;
      Mesh* mesh;
      mvector<PrimRef> prims;
      GeneralBVHBuilder::Settings settings;
      bool primrefarrayalloc;

      BVHNBuilderSAH (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize,
                      const size_t mode, bool primrefarrayalloc = false)
        : bvh(bvh), scene(scene), mesh(nullptr), prims(scene->device,0),
          settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD), primrefarrayalloc(primrefarrayalloc) {}

      BVHNBuilderSAH (BVH* bvh, Mesh* mesh, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(nullptr), mesh(mesh), prims(bvh->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD), primrefarrayalloc(false) {}

      // FIXME: shrink bvh->alloc in destructor here and in other builders too

      void build()
      {
        /* we reset the allocator when the mesh size changed */
        if (mesh && mesh->numPrimitivesChanged) {
          bvh->alloc.clear();
        }

        /* if we use the primrefarray for allocations we have to take it back from the BVH */
        if (settings.primrefarrayalloc != size_t(inf))
          bvh->alloc.unshare(prims);

	/* skip build for empty scene */
        const size_t numPrimitives = mesh ? mesh->size() : scene->getNumPrimitives<Mesh,false>();
        if (numPrimitives == 0) {
          bvh->clear();
          prims.clear();
          return;
        }

        double t0 = bvh->preBuild(mesh ? "" : TOSTRING(isa) "::BVH" + toString(N) + "BuilderSAH");

#if PROFILE
        profile(2,PROFILE_RUNS,numPrimitives,[&] (ProfileTimer& timer) {
#endif

            /* create primref array */
            if (primrefarrayalloc) {
              settings.primrefarrayalloc = numPrimitives/1000;
              if (settings.primrefarrayalloc < 1000)
                settings.primrefarrayalloc = inf;
            }

            /* enable os_malloc for two level build */
            if (mesh)
              bvh->alloc.setOSallocation(true);

            /* initialize allocator */
            const size_t node_bytes = numPrimitives*sizeof(typename BVH::AlignedNodeMB)/(4*N);
            const size_t leaf_bytes = size_t(1.2*Primitive::blocks(numPrimitives)*sizeof(Primitive));
            bvh->alloc.init_estimate(node_bytes+leaf_bytes);
            settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,numPrimitives,node_bytes+leaf_bytes);
            prims.resize(numPrimitives); 

            PrimInfo pinfo = mesh ?
              createPrimRefArray(mesh,prims,bvh->scene->progressInterface) :
              createPrimRefArray(scene,Mesh::geom_type,false,prims,bvh->scene->progressInterface);

            /* pinfo might has zero size due to invalid geometry */
            if (unlikely(pinfo.size() == 0))
            {
              bvh->clear();
              prims.clear();
              return;
            }

            /* call BVH builder */
            NodeRef root = BVHNBuilderVirtual<N>::build(&bvh->alloc,CreateLeaf<N,Primitive>(bvh),bvh->scene->progressInterface,prims.data(),pinfo,settings);
            bvh->set(root,LBBox3fa(pinfo.geomBounds),pinfo.size());
            bvh->layoutLargeNodes(size_t(pinfo.size()*0.005f));

#if PROFILE
          });
#endif

        /* if we allocated using the primrefarray we have to keep it alive */
        if (settings.primrefarrayalloc != size_t(inf))
          bvh->alloc.share(prims);

        /* for static geometries we can do some cleanups */
        else if (scene && scene->isStaticAccel()) {
          bvh->shrink();
          prims.clear();
        }
	bvh->cleanup();
        bvh->postBuild(t0);
      }

      void clear() {
        prims.clear();
      }
    };

    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/

    template<int N, typename Mesh, typename Primitive>
    struct BVHNBuilderSAHQuantized : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVHN<N>::NodeRef NodeRef;

      BVH* bvh;
      Scene* scene;
      Mesh* mesh;
      mvector<PrimRef> prims;
      GeneralBVHBuilder::Settings settings;

      BVHNBuilderSAHQuantized (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(scene), mesh(nullptr), prims(scene->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD) {}

      BVHNBuilderSAHQuantized (BVH* bvh, Mesh* mesh, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(nullptr), mesh(mesh), prims(bvh->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD) {}

      // FIXME: shrink bvh->alloc in destructor here and in other builders too

      void build()
      {
        /* we reset the allocator when the mesh size changed */
        if (mesh && mesh->numPrimitivesChanged) {
          bvh->alloc.clear();
        }

	/* skip build for empty scene */
        const size_t numPrimitives = mesh ? mesh->size() : scene->getNumPrimitives<Mesh,false>();
        if (numPrimitives == 0) {
          prims.clear();
          bvh->clear();
          return;
        }

        double t0 = bvh->preBuild(mesh ? "" : TOSTRING(isa) "::QBVH" + toString(N) + "BuilderSAH");

#if PROFILE
        profile(2,PROFILE_RUNS,numPrimitives,[&] (ProfileTimer& timer) {
#endif
            /* create primref array */
            prims.resize(numPrimitives);
            PrimInfo pinfo = mesh ?
              createPrimRefArray(mesh,prims,bvh->scene->progressInterface) :
              createPrimRefArray(scene,Mesh::geom_type,false,prims,bvh->scene->progressInterface);

            /* enable os_malloc for two level build */
            if (mesh)
              bvh->alloc.setOSallocation(true);

            /* call BVH builder */
            const size_t node_bytes = numPrimitives*sizeof(typename BVH::QuantizedNode)/(4*N);
            const size_t leaf_bytes = size_t(1.2*Primitive::blocks(numPrimitives)*sizeof(Primitive));
            bvh->alloc.init_estimate(node_bytes+leaf_bytes);
            settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,numPrimitives,node_bytes+leaf_bytes);
            NodeRef root = BVHNBuilderQuantizedVirtual<N>::build(&bvh->alloc,CreateLeafQuantized<N,Primitive>(bvh),bvh->scene->progressInterface,prims.data(),pinfo,settings);
            bvh->set(root,LBBox3fa(pinfo.geomBounds),pinfo.size());
            //bvh->layoutLargeNodes(pinfo.size()*0.005f); // FIXME: COPY LAYOUT FOR LARGE NODES !!!
#if PROFILE
          });
#endif

	/* clear temporary data for static geometry */
	if (scene && scene->isStaticAccel()) {
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

    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/


    template<int N, typename Primitive>
    struct CreateLeafGrid
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;

      __forceinline CreateLeafGrid (BVH* bvh, const SubGridBuildData * const sgrids) : bvh(bvh),sgrids(sgrids) {}

      __forceinline NodeRef operator() (const PrimRef* prims, const range<size_t>& set, const FastAllocator::CachedAllocator& alloc) const
      {
        const size_t items = set.size(); //Primitive::blocks(n);
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
        SubGridQBVHN<N>* accel = (SubGridQBVHN<N>*) alloc.malloc1(num_geomIDs*sizeof(SubGridQBVHN<N>),BVH::byteAlignment);
        typename BVH::NodeRef node = BVH::encodeLeaf((char*)accel,num_geomIDs);

        for (size_t g=0;g<num_geomIDs;g++)
        {
          unsigned int x[N];
          unsigned int y[N];
          unsigned int primID[N];
          BBox3fa bounds[N];
          unsigned int pos = 0;
          for (size_t i=0;i<items;i++)
          {
            if (unlikely(prims[start+i].geomID() != geomIDs[g])) continue;

            const SubGridBuildData  &sgrid_bd = sgrids[prims[start+i].primID()];                      
            x[pos] = sgrid_bd.sx;
            y[pos] = sgrid_bd.sy;
            primID[pos] = sgrid_bd.primID;
            bounds[pos] = prims[start+i].bounds();
            pos++;
          }
          new (&accel[g]) SubGridQBVHN<N>(x,y,primID,bounds,geomIDs[g],pos);
        }

        return node;
      }

      BVH* bvh;
      const SubGridBuildData * const sgrids;
    };


    template<int N>
    struct BVHNBuilderSAHGrid : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVHN<N>::NodeRef NodeRef;
      
      BVH* bvh;
      Scene* scene;
      GridMesh* mesh;
      mvector<PrimRef> prims;
      mvector<SubGridBuildData> sgrids;
      GeneralBVHBuilder::Settings settings;

      BVHNBuilderSAHGrid (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(scene), mesh(nullptr), prims(scene->device,0), sgrids(scene->device,0), settings(sahBlockSize, minLeafSize, maxLeafSize, travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD) {}

      BVHNBuilderSAHGrid (BVH* bvh, GridMesh* mesh, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(nullptr), mesh(mesh), prims(bvh->device,0), sgrids(scene->device,0), settings(sahBlockSize, minLeafSize, maxLeafSize, travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD) {}

      void build()
      {
        /* we reset the allocator when the mesh size changed */
        if (mesh && mesh->numPrimitivesChanged) {
          bvh->alloc.clear();
        }

        /* if we use the primrefarray for allocations we have to take it back from the BVH */
        if (settings.primrefarrayalloc != size_t(inf))
          bvh->alloc.unshare(prims);
        
        PrimInfo pinfo(empty);
        size_t numPrimitives = 0;

        if (!mesh)
        {
          /* first run to get #primitives */

          ParallelForForPrefixSumState<PrimInfo> pstate;
          Scene::Iterator<GridMesh,false> iter(scene);

          pstate.init(iter,size_t(1024));

          /* iterate over all meshes in the scene */
          pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfo(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k) -> PrimInfo
                                                {
                                                  PrimInfo pinfo(empty);
                                                  for (size_t j=r.begin(); j<r.end(); j++)
                                                  {
                                                    if (!mesh->valid(j)) continue;
                                                    BBox3fa bounds = empty;
                                                    const PrimRef prim(bounds,mesh->geomID,unsigned(j));                                                          if (!mesh->valid(j)) continue;

                                                    pinfo.add_center2(prim,mesh->getNumSubGrids(j));
                                                  }
                                                  return pinfo;
                                                }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
          numPrimitives = pinfo.size();

          /* resize arrays */
          sgrids.resize(numPrimitives); 
          prims.resize(numPrimitives); 

          /* second run to fill primrefs and SubGridBuildData arrays */
          pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfo(empty), [&](GridMesh* mesh, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo
                                                {
                                                  k = base.size();
                                                  size_t p_index = k;
                                                  PrimInfo pinfo(empty);
                                                  for (size_t j=r.begin(); j<r.end(); j++)
                                                  {
                                                    if (!mesh->valid(j)) continue;
                                                    const GridMesh::Grid &g = mesh->grid(j);
                                                    for (unsigned int y=0; y<g.resY-1u; y+=2)
                                                      for (unsigned int x=0; x<g.resX-1u; x+=2)
                                                      {
                                                        BBox3fa bounds = empty;
                                                        if (!mesh->buildBounds(g,x,y,bounds)) continue; // get bounds of subgrid
                                                        const PrimRef prim(bounds,mesh->geomID,unsigned(p_index));
                                                        pinfo.add_center2(prim);
                                                        sgrids[p_index] = SubGridBuildData(x | g.get3x3FlagsX(x), y | g.get3x3FlagsY(y), unsigned(j));
                                                        prims[p_index++] = prim;                
                                                      }
                                                  }
                                                  return pinfo;
                                                }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
          assert(pinfo.size() == numPrimitives);
        }
        else
        {
          ParallelPrefixSumState<PrimInfo> pstate;
          /* iterate over all grids in a single mesh */
          pinfo = parallel_prefix_sum( pstate, size_t(0), mesh->size(), size_t(1024), PrimInfo(empty), [&](const range<size_t>& r, const PrimInfo& base) -> PrimInfo
                                       {
                                         PrimInfo pinfo(empty);
                                         for (size_t j=r.begin(); j<r.end(); j++)
                                         {
                                           if (!mesh->valid(j)) continue;
                                           BBox3fa bounds = empty;
                                           const PrimRef prim(bounds,mesh->geomID,unsigned(j));
                                           pinfo.add_center2(prim,mesh->getNumSubGrids(j));
                                         }
                                         return pinfo;
                                       }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
          numPrimitives = pinfo.size();
          /* resize arrays */
          sgrids.resize(numPrimitives); 
          prims.resize(numPrimitives); 

          /* second run to fill primrefs and SubGridBuildData arrays */
          pinfo = parallel_prefix_sum( pstate, size_t(0), mesh->size(), size_t(1024), PrimInfo(empty), [&](const range<size_t>& r, const PrimInfo& base) -> PrimInfo
                                       {

                                         size_t p_index = base.size();
                                         PrimInfo pinfo(empty);
                                         for (size_t j=r.begin(); j<r.end(); j++)
                                         {
                                           if (!mesh->valid(j)) continue;
                                           const GridMesh::Grid &g = mesh->grid(j);
                                           for (unsigned int y=0; y<g.resY-1u; y+=2)
                                             for (unsigned int x=0; x<g.resX-1u; x+=2)
                                             {
                                               BBox3fa bounds = empty;
                                               if (!mesh->buildBounds(g,x,y,bounds)) continue; // get bounds of subgrid
                                               const PrimRef prim(bounds,mesh->geomID,unsigned(p_index));
                                               pinfo.add_center2(prim);
                                               sgrids[p_index] = SubGridBuildData(x | g.get3x3FlagsX(x), y | g.get3x3FlagsY(y), unsigned(j));
                                               prims[p_index++] = prim;                
                                             }
                                         }
                                         return pinfo;
                                       }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });

        }


        /* no primitives */
        if (numPrimitives == 0) {
          bvh->clear();
          prims.clear();
          sgrids.clear();
          return;
        }

        double t0 = bvh->preBuild(mesh ? "" : TOSTRING(isa) "::BVH" + toString(N) + "BuilderSAH");

        /* create primref array */
        settings.primrefarrayalloc = numPrimitives/1000;
        if (settings.primrefarrayalloc < 1000)
          settings.primrefarrayalloc = inf;

        /* enable os_malloc for two level build */
        if (mesh)
          bvh->alloc.setOSallocation(true);

        /* initialize allocator */
        const size_t node_bytes = numPrimitives*sizeof(typename BVH::AlignedNodeMB)/(4*N);
        const size_t leaf_bytes = size_t(1.2*(float)numPrimitives/N * sizeof(SubGridQBVHN<N>));

        bvh->alloc.init_estimate(node_bytes+leaf_bytes);
        settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,numPrimitives,node_bytes+leaf_bytes);

        /* pinfo might has zero size due to invalid geometry */
        if (unlikely(pinfo.size() == 0))
        {
          bvh->clear();
          sgrids.clear();
          prims.clear();
          return;
        }

        /* call BVH builder */
        NodeRef root = BVHNBuilderVirtual<N>::build(&bvh->alloc,CreateLeafGrid<N,SubGridQBVHN<N>>(bvh,sgrids.data()),bvh->scene->progressInterface,prims.data(),pinfo,settings);
        bvh->set(root,LBBox3fa(pinfo.geomBounds),pinfo.size());
        bvh->layoutLargeNodes(size_t(pinfo.size()*0.005f));

        /* clear temporary array */
        sgrids.clear();

        /* if we allocated using the primrefarray we have to keep it alive */
        if (settings.primrefarrayalloc != size_t(inf))
          bvh->alloc.share(prims);

        /* for static geometries we can do some cleanups */
        else if (scene && scene->isStaticAccel()) {
          bvh->shrink();
          prims.clear();
        }
	bvh->cleanup();
        bvh->postBuild(t0);
      }

      void clear() {
        prims.clear();
      }
    };

    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/

#if defined(EMBREE_GEOMETRY_TRIANGLE)
    Builder* BVH4Triangle4MeshBuilderSAH  (void* bvh, TriangleMesh* mesh, size_t mode) { return new BVHNBuilderSAH<4,TriangleMesh,Triangle4>((BVH4*)bvh,mesh,4,1.0f,4,inf,mode); }
    Builder* BVH4Triangle4vMeshBuilderSAH (void* bvh, TriangleMesh* mesh, size_t mode) { return new BVHNBuilderSAH<4,TriangleMesh,Triangle4v>((BVH4*)bvh,mesh,4,1.0f,4,inf,mode); }
    Builder* BVH4Triangle4iMeshBuilderSAH (void* bvh, TriangleMesh* mesh, size_t mode) { return new BVHNBuilderSAH<4,TriangleMesh,Triangle4i>((BVH4*)bvh,mesh,4,1.0f,4,inf,mode); }

    Builder* BVH4Triangle4SceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,TriangleMesh,Triangle4>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH4Triangle4vSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,TriangleMesh,Triangle4v>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH4Triangle4iSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,TriangleMesh,Triangle4i>((BVH4*)bvh,scene,4,1.0f,4,inf,mode,true); }


    Builder* BVH4QuantizedTriangle4iSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<4,TriangleMesh,Triangle4i>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }
#if defined(__AVX__)
    Builder* BVH8Triangle4MeshBuilderSAH  (void* bvh, TriangleMesh* mesh, size_t mode) { return new BVHNBuilderSAH<8,TriangleMesh,Triangle4>((BVH8*)bvh,mesh,4,1.0f,4,inf,mode); }
    Builder* BVH8Triangle4vMeshBuilderSAH (void* bvh, TriangleMesh* mesh, size_t mode) { return new BVHNBuilderSAH<8,TriangleMesh,Triangle4v>((BVH8*)bvh,mesh,4,1.0f,4,inf,mode); }
    Builder* BVH8Triangle4iMeshBuilderSAH (void* bvh, TriangleMesh* mesh, size_t mode) { return new BVHNBuilderSAH<8,TriangleMesh,Triangle4i>((BVH8*)bvh,mesh,4,1.0f,4,inf,mode); }

    Builder* BVH8Triangle4SceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,TriangleMesh,Triangle4>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH8Triangle4vSceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,TriangleMesh,Triangle4v>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH8Triangle4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,TriangleMesh,Triangle4i>((BVH8*)bvh,scene,4,1.0f,4,inf,mode,true); }
    Builder* BVH8QuantizedTriangle4iSceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<8,TriangleMesh,Triangle4i>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH8QuantizedTriangle4SceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<8,TriangleMesh,Triangle4>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }

#endif
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
    Builder* BVH4Quad4vMeshBuilderSAH     (void* bvh, QuadMesh* mesh, size_t mode)     { return new BVHNBuilderSAH<4,QuadMesh,Quad4v>((BVH4*)bvh,mesh,4,1.0f,4,inf,mode); }
    Builder* BVH4Quad4iMeshBuilderSAH     (void* bvh, QuadMesh* mesh, size_t mode)     { return new BVHNBuilderSAH<4,QuadMesh,Quad4i>((BVH4*)bvh,mesh,4,1.0f,4,inf,mode); }
    Builder* BVH4Quad4vSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,QuadMesh,Quad4v>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH4Quad4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,QuadMesh,Quad4i>((BVH4*)bvh,scene,4,1.0f,4,inf,mode,true); }
    Builder* BVH4QuantizedQuad4vSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<4,QuadMesh,Quad4v>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH4QuantizedQuad4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<4,QuadMesh,Quad4i>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }

#if defined(__AVX__)
    Builder* BVH8Quad4vSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,QuadMesh,Quad4v>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH8Quad4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,QuadMesh,Quad4i>((BVH8*)bvh,scene,4,1.0f,4,inf,mode,true); }
    Builder* BVH8QuantizedQuad4vSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<8,QuadMesh,Quad4v>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH8QuantizedQuad4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<8,QuadMesh,Quad4i>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH8Quad4vMeshBuilderSAH     (void* bvh, QuadMesh* mesh, size_t mode)     { return new BVHNBuilderSAH<8,QuadMesh,Quad4v>((BVH8*)bvh,mesh,4,1.0f,4,inf,mode); }

#endif
#endif

#if defined(EMBREE_GEOMETRY_USER)

    Builder* BVH4VirtualSceneBuilderSAH    (void* bvh, Scene* scene, size_t mode) {
      int minLeafSize = scene->device->object_accel_min_leaf_size;
      int maxLeafSize = scene->device->object_accel_max_leaf_size;
      return new BVHNBuilderSAH<4,UserGeometry,Object>((BVH4*)bvh,scene,4,1.0f,minLeafSize,maxLeafSize,mode);
    }

    Builder* BVH4VirtualMeshBuilderSAH    (void* bvh, UserGeometry* mesh, size_t mode) {
      return new BVHNBuilderSAH<4,UserGeometry,Object>((BVH4*)bvh,mesh,4,1.0f,1,inf,mode);
    }
#if defined(__AVX__)

    Builder* BVH8VirtualSceneBuilderSAH    (void* bvh, Scene* scene, size_t mode) {
      int minLeafSize = scene->device->object_accel_min_leaf_size;
      int maxLeafSize = scene->device->object_accel_max_leaf_size;
      return new BVHNBuilderSAH<8,UserGeometry,Object>((BVH8*)bvh,scene,8,1.0f,minLeafSize,maxLeafSize,mode);
    }

    Builder* BVH8VirtualMeshBuilderSAH    (void* bvh, UserGeometry* mesh, size_t mode) {
      return new BVHNBuilderSAH<8,UserGeometry,Object>((BVH8*)bvh,mesh,8,1.0f,1,inf,mode);
    }
#endif
#endif

#if defined(EMBREE_GEOMETRY_INSTANCE)
    Builder* BVH4InstanceSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,Instance,InstancePrimitive>((BVH4*)bvh,scene,4,1.0f,1,1,mode); }
#if defined(__AVX__)
    Builder* BVH8InstanceSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,Instance,InstancePrimitive>((BVH8*)bvh,scene,8,1.0f,1,1,mode); }
#endif
#endif

#if defined(EMBREE_GEOMETRY_GRID)
    Builder* BVH4GridMeshBuilderSAH  (void* bvh, GridMesh* mesh, size_t mode) { return new BVHNBuilderSAHGrid<4>((BVH4*)bvh,mesh,4,1.0f,4,4,mode); }
    Builder* BVH4GridSceneBuilderSAH (void* bvh, Scene* scene, size_t mode)   { return new BVHNBuilderSAHGrid<4>((BVH4*)bvh,scene,4,1.0f,4,4,mode); } // FIXME: check whether cost factors are correct

#if defined(__AVX__)
    Builder* BVH8GridMeshBuilderSAH  (void* bvh, GridMesh* mesh, size_t mode) { return new BVHNBuilderSAHGrid<8>((BVH8*)bvh,mesh,8,1.0f,8,8,mode); }
    Builder* BVH8GridSceneBuilderSAH (void* bvh, Scene* scene, size_t mode)   { return new BVHNBuilderSAHGrid<8>((BVH8*)bvh,scene,8,1.0f,8,8,mode); } // FIXME: check whether cost factors are correct
#endif
#endif

  }
}
