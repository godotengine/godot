// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
#include "../geometry/instance_array.h"
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

    template<int N, typename Primitive>
    struct BVHNBuilderSAH : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVHN<N>::NodeRef NodeRef;

      BVH* bvh;
      Scene* scene;
      Geometry* mesh;
      mvector<PrimRef> prims;
      GeneralBVHBuilder::Settings settings;
      Geometry::GTypeMask gtype_;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max ();
      bool primrefarrayalloc;
      unsigned int numPreviousPrimitives = 0;

      BVHNBuilderSAH (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize,
                      const Geometry::GTypeMask gtype, bool primrefarrayalloc = false)
        : bvh(bvh), scene(scene), mesh(nullptr), prims(scene->device,0),
          settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD), gtype_(gtype), primrefarrayalloc(primrefarrayalloc) {}

      BVHNBuilderSAH (BVH* bvh, Geometry* mesh, unsigned int geomID, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const Geometry::GTypeMask gtype)
        : bvh(bvh), scene(nullptr), mesh(mesh), prims(bvh->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD), gtype_(gtype), geomID_(geomID), primrefarrayalloc(false) {}

      // FIXME: shrink bvh->alloc in destructor here and in other builders too

      void build()
      {
        /* we reset the allocator when the mesh size changed */
        if (mesh && mesh->numPrimitives != numPreviousPrimitives) {
          bvh->alloc.clear();
        }

        /* if we use the primrefarray for allocations we have to take it back from the BVH */
        if (settings.primrefarrayalloc != size_t(inf))
          bvh->alloc.unshare(prims);

	/* skip build for empty scene */
        const size_t numPrimitives = mesh ? mesh->size() : scene->getNumPrimitives(gtype_,false);
        numPreviousPrimitives = numPrimitives;
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
            const size_t node_bytes = numPrimitives*sizeof(typename BVH::AABBNodeMB)/(4*N);
            const size_t leaf_bytes = size_t(1.2*Primitive::blocks(numPrimitives)*sizeof(Primitive));
            bvh->alloc.init_estimate(node_bytes+leaf_bytes);
            settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,numPrimitives,node_bytes+leaf_bytes);
            prims.resize(numPrimitives);

            PrimInfo pinfo = mesh ?
              createPrimRefArray(mesh,geomID_,numPrimitives,prims,bvh->scene->progressInterface) :
              createPrimRefArray(scene,gtype_,false,numPrimitives,prims,bvh->scene->progressInterface);

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

    template<int N, typename Primitive>
    struct BVHNBuilderSAHQuantized : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVHN<N>::NodeRef NodeRef;

      BVH* bvh;
      Scene* scene;
      Geometry* mesh;
      mvector<PrimRef> prims;
      GeneralBVHBuilder::Settings settings;
      Geometry::GTypeMask gtype_;
      unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
      unsigned int numPreviousPrimitives = 0;

      BVHNBuilderSAHQuantized (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const Geometry::GTypeMask gtype)
        : bvh(bvh), scene(scene), mesh(nullptr), prims(scene->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD), gtype_(gtype) {}

      BVHNBuilderSAHQuantized (BVH* bvh, Geometry* mesh, unsigned int geomID, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const Geometry::GTypeMask gtype)
        : bvh(bvh), scene(nullptr), mesh(mesh), prims(bvh->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD), gtype_(gtype), geomID_(geomID) {}

      // FIXME: shrink bvh->alloc in destructor here and in other builders too

      void build()
      {
        /* we reset the allocator when the mesh size changed */
        if (mesh && mesh->numPrimitives != numPreviousPrimitives) {
          bvh->alloc.clear();
        }

	/* skip build for empty scene */
        const size_t numPrimitives = mesh ? mesh->size() : scene->getNumPrimitives(gtype_,false);
        numPreviousPrimitives = numPrimitives;
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
              createPrimRefArray(mesh,geomID_,numPrimitives,prims,bvh->scene->progressInterface) :
	      createPrimRefArray(scene,gtype_,false,numPrimitives,prims,bvh->scene->progressInterface);

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

            const SubGridBuildData& sgrid_bd = sgrids[prims[start+i].primID()];
            x[pos] = sgrid_bd.sx;
            y[pos] = sgrid_bd.sy;
            primID[pos] = sgrid_bd.primID;
            bounds[pos] = prims[start+i].bounds();
            pos++;
          }
          assert(pos <= N);
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
      const unsigned int geomID_ = std::numeric_limits<unsigned int>::max();
      unsigned int numPreviousPrimitives = 0;

      BVHNBuilderSAHGrid (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(scene), mesh(nullptr), prims(scene->device,0), sgrids(scene->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD) {}

      BVHNBuilderSAHGrid (BVH* bvh, GridMesh* mesh, unsigned int geomID, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(nullptr), mesh(mesh), prims(bvh->device,0), sgrids(scene->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD), geomID_(geomID) {}

      void build()
      {
        /* we reset the allocator when the mesh size changed */
        if (mesh && mesh->numPrimitives != numPreviousPrimitives) {
          bvh->alloc.clear();
        }
        
        /* if we use the primrefarray for allocations we have to take it back from the BVH */
        if (settings.primrefarrayalloc != size_t(inf))
          bvh->alloc.unshare(prims);

        const size_t numGridPrimitives = mesh ? mesh->size() : scene->getNumPrimitives(GridMesh::geom_type,false);
        numPreviousPrimitives = numGridPrimitives;


        PrimInfo pinfo = mesh ? createPrimRefArrayGrids(mesh,prims,sgrids) : createPrimRefArrayGrids(scene,prims,sgrids);
        const size_t numPrimitives = pinfo.size();
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
        const size_t node_bytes = numPrimitives*sizeof(typename BVH::AABBNodeMB)/(4*N);
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
    Builder* BVH4Triangle4MeshBuilderSAH  (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new BVHNBuilderSAH<4,Triangle4>((BVH4*)bvh,mesh,geomID,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH4Triangle4vMeshBuilderSAH (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new BVHNBuilderSAH<4,Triangle4v>((BVH4*)bvh,mesh,geomID,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH4Triangle4iMeshBuilderSAH (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new BVHNBuilderSAH<4,Triangle4i>((BVH4*)bvh,mesh,geomID,4,1.0f,4,inf,TriangleMesh::geom_type); }

    Builder* BVH4Triangle4SceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,Triangle4>((BVH4*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH4Triangle4vSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,Triangle4v>((BVH4*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH4Triangle4iSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,Triangle4i>((BVH4*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type,true); }

    Builder* BVH4QuantizedTriangle4iSceneBuilderSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<4,Triangle4i>((BVH4*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type); }
#if defined(__AVX__)
    Builder* BVH8Triangle4MeshBuilderSAH  (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new BVHNBuilderSAH<8,Triangle4>((BVH8*)bvh,mesh,geomID,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH8Triangle4vMeshBuilderSAH (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new BVHNBuilderSAH<8,Triangle4v>((BVH8*)bvh,mesh,geomID,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH8Triangle4iMeshBuilderSAH (void* bvh, TriangleMesh* mesh, unsigned int geomID, size_t mode) { return new BVHNBuilderSAH<8,Triangle4i>((BVH8*)bvh,mesh,geomID,4,1.0f,4,inf,TriangleMesh::geom_type); }

    Builder* BVH8Triangle4SceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,Triangle4>((BVH8*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH8Triangle4vSceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,Triangle4v>((BVH8*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH8Triangle4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,Triangle4i>((BVH8*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type,true); }
    Builder* BVH8QuantizedTriangle4iSceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<8,Triangle4i>((BVH8*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type); }
    Builder* BVH8QuantizedTriangle4SceneBuilderSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<8,Triangle4>((BVH8*)bvh,scene,4,1.0f,4,inf,TriangleMesh::geom_type); }

    

#endif
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
    Builder* BVH4Quad4vMeshBuilderSAH     (void* bvh, QuadMesh* mesh, unsigned int geomID, size_t mode)     { return new BVHNBuilderSAH<4,Quad4v>((BVH4*)bvh,mesh,geomID,4,1.0f,4,inf,QuadMesh::geom_type); }
    Builder* BVH4Quad4iMeshBuilderSAH     (void* bvh, QuadMesh* mesh, unsigned int geomID, size_t mode)     { return new BVHNBuilderSAH<4,Quad4i>((BVH4*)bvh,mesh,geomID,4,1.0f,4,inf,QuadMesh::geom_type); }
    Builder* BVH4Quad4vSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,Quad4v>((BVH4*)bvh,scene,4,1.0f,4,inf,QuadMesh::geom_type); }
    Builder* BVH4Quad4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<4,Quad4i>((BVH4*)bvh,scene,4,1.0f,4,inf,QuadMesh::geom_type,true); }
    Builder* BVH4QuantizedQuad4vSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<4,Quad4v>((BVH4*)bvh,scene,4,1.0f,4,inf,QuadMesh::geom_type); }
    Builder* BVH4QuantizedQuad4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<4,Quad4i>((BVH4*)bvh,scene,4,1.0f,4,inf,QuadMesh::geom_type); }

#if defined(__AVX__)
    Builder* BVH8Quad4vSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,Quad4v>((BVH8*)bvh,scene,4,1.0f,4,inf,QuadMesh::geom_type); }
    Builder* BVH8Quad4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAH<8,Quad4i>((BVH8*)bvh,scene,4,1.0f,4,inf,QuadMesh::geom_type,true); }
    Builder* BVH8QuantizedQuad4vSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<8,Quad4v>((BVH8*)bvh,scene,4,1.0f,4,inf,QuadMesh::geom_type); }
    Builder* BVH8QuantizedQuad4iSceneBuilderSAH     (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderSAHQuantized<8,Quad4i>((BVH8*)bvh,scene,4,1.0f,4,inf,QuadMesh::geom_type); }
    Builder* BVH8Quad4vMeshBuilderSAH     (void* bvh, QuadMesh* mesh, unsigned int geomID, size_t mode)     { return new BVHNBuilderSAH<8,Quad4v>((BVH8*)bvh,mesh,geomID,4,1.0f,4,inf,QuadMesh::geom_type); }

#endif
#endif

#if defined(EMBREE_GEOMETRY_USER)

    Builder* BVH4VirtualSceneBuilderSAH    (void* bvh, Scene* scene, size_t mode) {
      int minLeafSize = scene->device->object_accel_min_leaf_size;
      int maxLeafSize = scene->device->object_accel_max_leaf_size;
      return new BVHNBuilderSAH<4,Object>((BVH4*)bvh,scene,4,1.0f,minLeafSize,maxLeafSize,UserGeometry::geom_type);
    }

    Builder* BVH4VirtualMeshBuilderSAH    (void* bvh, UserGeometry* mesh, unsigned int geomID, size_t mode) {
      return new BVHNBuilderSAH<4,Object>((BVH4*)bvh,mesh,geomID,4,1.0f,1,inf,UserGeometry::geom_type);
    }
#if defined(__AVX__)

    Builder* BVH8VirtualSceneBuilderSAH    (void* bvh, Scene* scene, size_t mode) {
      int minLeafSize = scene->device->object_accel_min_leaf_size;
      int maxLeafSize = scene->device->object_accel_max_leaf_size;
      return new BVHNBuilderSAH<8,Object>((BVH8*)bvh,scene,8,1.0f,minLeafSize,maxLeafSize,UserGeometry::geom_type);
    }

    Builder* BVH8VirtualMeshBuilderSAH    (void* bvh, UserGeometry* mesh, unsigned int geomID, size_t mode) {
      return new BVHNBuilderSAH<8,Object>((BVH8*)bvh,mesh,geomID,8,1.0f,1,inf,UserGeometry::geom_type);
    }
#endif
#endif

#if defined(EMBREE_GEOMETRY_INSTANCE)
    Builder* BVH4InstanceSceneBuilderSAH (void* bvh, Scene* scene, Geometry::GTypeMask gtype) {
      return new BVHNBuilderSAH<4,InstancePrimitive>((BVH4*)bvh,scene,4,1.0f,1,1,gtype);
    }
    Builder* BVH4InstanceMeshBuilderSAH (void* bvh, Instance* mesh, Geometry::GTypeMask gtype, unsigned int geomID, size_t mode) {
      return new BVHNBuilderSAH<4,InstancePrimitive>((BVH4*)bvh,mesh,geomID,4,1.0f,1,inf,gtype);
    }
#if defined(__AVX__)
    Builder* BVH8InstanceSceneBuilderSAH (void* bvh, Scene* scene, Geometry::GTypeMask gtype) {
      return new BVHNBuilderSAH<8,InstancePrimitive>((BVH8*)bvh,scene,8,1.0f,1,1,gtype);
    }
    Builder* BVH8InstanceMeshBuilderSAH (void* bvh, Instance* mesh, Geometry::GTypeMask gtype, unsigned int geomID, size_t mode) {
      return new BVHNBuilderSAH<8,InstancePrimitive>((BVH8*)bvh,mesh,geomID,8,1.0f,1,1,gtype);
    }
#endif
#endif

#if defined(EMBREE_GEOMETRY_INSTANCE_ARRAY)
    Builder* BVH4InstanceArraySceneBuilderSAH (void* bvh, Scene* scene, Geometry::GTypeMask gtype) {
      return new BVHNBuilderSAH<4,InstanceArrayPrimitive>((BVH4*)bvh,scene,4,1.0f,1,1,gtype);
    }
    Builder* BVH4InstanceArrayMeshBuilderSAH (void* bvh, InstanceArray* mesh, Geometry::GTypeMask gtype, unsigned int geomID, size_t mode) {
      return new BVHNBuilderSAH<4,InstanceArrayPrimitive>((BVH4*)bvh,mesh,geomID,4,1.0f,1,1,gtype);
    }
#if defined(__AVX__)
    Builder* BVH8InstanceArraySceneBuilderSAH (void* bvh, Scene* scene, Geometry::GTypeMask gtype) {
      return new BVHNBuilderSAH<8,InstanceArrayPrimitive>((BVH8*)bvh,scene,8,1.0f,1,1,gtype);
    }
    Builder* BVH8InstanceArrayMeshBuilderSAH (void* bvh, InstanceArray* mesh, Geometry::GTypeMask gtype, unsigned int geomID, size_t mode) {
      return new BVHNBuilderSAH<8,InstanceArrayPrimitive>((BVH8*)bvh,mesh,geomID,8,1.0f,1,1,gtype);
    }
#endif
#endif

#if defined(EMBREE_GEOMETRY_GRID)
    Builder* BVH4GridMeshBuilderSAH  (void* bvh, GridMesh* mesh, unsigned int geomID, size_t mode) { return new BVHNBuilderSAHGrid<4>((BVH4*)bvh,mesh,geomID,4,1.0f,4,4,mode); }
    Builder* BVH4GridSceneBuilderSAH (void* bvh, Scene* scene, size_t mode)   { return new BVHNBuilderSAHGrid<4>((BVH4*)bvh,scene,4,1.0f,4,4,mode); } // FIXME: check whether cost factors are correct

#if defined(__AVX__)
    Builder* BVH8GridMeshBuilderSAH  (void* bvh, GridMesh* mesh, unsigned int geomID, size_t mode) { return new BVHNBuilderSAHGrid<8>((BVH8*)bvh,mesh,geomID,8,1.0f,8,8,mode); }
    Builder* BVH8GridSceneBuilderSAH (void* bvh, Scene* scene, size_t mode)   { return new BVHNBuilderSAHGrid<8>((BVH8*)bvh,scene,8,1.0f,8,8,mode); } // FIXME: check whether cost factors are correct
#endif
#endif
  }
}
