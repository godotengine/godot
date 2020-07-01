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

namespace embree
{
  namespace isa
  {
    template<int N, typename Primitive>
    struct CreateLeafSpatial
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;

      __forceinline CreateLeafSpatial (BVH* bvh) : bvh(bvh) {}

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

    template<int N, typename Mesh, typename Primitive, typename Splitter>
    struct BVHNBuilderFastSpatialSAH : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      BVH* bvh;
      Scene* scene;
      Mesh* mesh;
      mvector<PrimRef> prims0;
      GeneralBVHBuilder::Settings settings;
      const float splitFactor;

      BVHNBuilderFastSpatialSAH (BVH* bvh, Scene* scene, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(scene), mesh(nullptr), prims0(scene->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD),
          splitFactor(scene->device->max_spatial_split_replications) {}

      BVHNBuilderFastSpatialSAH (BVH* bvh, Mesh* mesh, const size_t sahBlockSize, const float intCost, const size_t minLeafSize, const size_t maxLeafSize, const size_t mode)
        : bvh(bvh), scene(nullptr), mesh(mesh), prims0(bvh->device,0), settings(sahBlockSize, minLeafSize, min(maxLeafSize,Primitive::max_size()*BVH::maxLeafBlocks), travCost, intCost, DEFAULT_SINGLE_THREAD_THRESHOLD),
          splitFactor(scene->device->max_spatial_split_replications) {}

      // FIXME: shrink bvh->alloc in destructor here and in other builders too

      void build()
      {
        /* we reset the allocator when the mesh size changed */
        if (mesh && mesh->numPrimitivesChanged) {
          bvh->alloc.clear();
        }

	/* skip build for empty scene */
        const size_t numOriginalPrimitives = mesh ? mesh->size() : scene->getNumPrimitives<Mesh,false>();
        if (numOriginalPrimitives == 0) {
          prims0.clear();
          bvh->clear();
          return;
        }

        const unsigned int maxGeomID = mesh ? mesh->geomID : scene->getMaxGeomID<Mesh,false>();
        double t0 = bvh->preBuild(mesh ? "" : TOSTRING(isa) "::BVH" + toString(N) + "BuilderFastSpatialSAH");

        /* create primref array */
        const size_t numSplitPrimitives = max(numOriginalPrimitives,size_t(splitFactor*numOriginalPrimitives));
        prims0.resize(numSplitPrimitives);
        PrimInfo pinfo = mesh ?
          createPrimRefArray(mesh,prims0,bvh->scene->progressInterface) :
          createPrimRefArray(scene,Mesh::geom_type,false,prims0,bvh->scene->progressInterface);

        Splitter splitter(scene);

        /* enable os_malloc for two level build */
        if (mesh)
          bvh->alloc.setOSallocation(true);

        const size_t node_bytes = pinfo.size()*sizeof(typename BVH::AlignedNode)/(4*N);
        const size_t leaf_bytes = size_t(1.2*Primitive::blocks(pinfo.size())*sizeof(Primitive));
        bvh->alloc.init_estimate(node_bytes+leaf_bytes);
        settings.singleThreadThreshold = bvh->alloc.fixSingleThreadThreshold(N,DEFAULT_SINGLE_THREAD_THRESHOLD,pinfo.size(),node_bytes+leaf_bytes);

        settings.branchingFactor = N;
        settings.maxDepth = BVH::maxBuildDepthLeaf;

        /* call BVH builder */
        NodeRef root(0);

        const bool usePreSplits = scene->device->useSpatialPreSplits;
        if (likely( !usePreSplits && (maxGeomID < ((unsigned int)1 << (32-RESERVED_NUM_SPATIAL_SPLITS_GEOMID_BITS)))))
        {
          root = BVHBuilderBinnedFastSpatialSAH::build<NodeRef>(
            typename BVH::CreateAlloc(bvh),
            typename BVH::AlignedNode::Create2(),
            typename BVH::AlignedNode::Set2(),
            CreateLeafSpatial<N,Primitive>(bvh),
            splitter,
            bvh->scene->progressInterface,
            prims0.data(),
            numSplitPrimitives,
            pinfo,settings);
        }
        else
        {
          /* fallback for max geomID > 2^27 or activated pre-splits */

#define ENABLE_PRESPLITS 0
#if ENABLE_PRESPLITS == 1
          /* pre splits */
          const unsigned int LATTICE_BITS_PER_DIM = 10;
          const unsigned int LATTICE_SIZE_PER_DIM = (1 << LATTICE_BITS_PER_DIM);
          const Vec3fa base  = pinfo.geomBounds.lower;
          const Vec3fa diag  = pinfo.geomBounds.upper - pinfo.geomBounds.lower;
          const Vec3fa scale = select(gt_mask(diag,Vec3fa(1E-19f)), (Vec3fa)((float)LATTICE_SIZE_PER_DIM) * (Vec3fa)(1.0f-(float)ulp) / diag, (Vec3fa)(0.0f));
          const float inv_lattice_size_per_dim = 1.0f / (float)LATTICE_SIZE_PER_DIM;
          const Vec3fa min_diag_threshold = diag * (Vec3fa)(inv_lattice_size_per_dim * 0.5f);

          struct PreSplitProbEntry{ 
            unsigned int index; float prob; 
            __forceinline bool operator<(PreSplitProbEntry const &b) const { return prob > b.prob; }
          };

          avector<PreSplitProbEntry> presplit_prio;
          presplit_prio.resize(pinfo.size());

          /* =========================================== */                   
          /* == compute split probability per primref == */
          /* =========================================== */

          for (size_t i=0;i<pinfo.size();i++)
          {
            const Vec3fa lower = prims0[i].lower;
            const Vec3fa upper = prims0[i].upper;
            const Vec3ia lower_binID = (Vec3ia)((lower-base)*scale);
            const Vec3ia upper_binID = (Vec3ia)((upper-base)*scale);
            const unsigned int lower_code = bitInterleave(lower_binID.x,lower_binID.y,lower_binID.z);
            const unsigned int upper_code = bitInterleave(upper_binID.x,upper_binID.y,upper_binID.z);

            const unsigned int diff = lzcnt(lower_code^upper_code);
            const float priority_diff = diff < 32 ? 1.0f / (float)diff : 0.0f;
            const unsigned int dim = (31 - diff) % 3;

            const unsigned int split_binID = (lower_binID[dim] + upper_binID[dim] + 1)/2;
            const float split_ratio = (float)split_binID * inv_lattice_size_per_dim;
            const float pos = base[dim] + split_ratio * diag[dim];

            const float pos_prob = ((pos - lower[dim]) > min_diag_threshold[dim] && (upper[dim] - pos) > min_diag_threshold[dim]) ? 1.0f : 0.0f;
            const float prob = priority_diff * pos_prob;
            presplit_prio[i].index = (unsigned int)i;
            presplit_prio[i].prob  = prob;
          }

          /* ============================================================ */                   
          /* == sort split probabilities to ensure being deterministic == */
          /* ============================================================ */

          std::sort(presplit_prio.data(),presplit_prio.data()+pinfo.size());

          std::atomic<size_t> ext_elements;
          ext_elements.store(pinfo.size());

          /* =================================== */                   
          /* == split selected primrefs once  == */
          /* =================================== */

          const size_t extraSize = min(numSplitPrimitives - pinfo.size(),pinfo.size());
          for (size_t i=0;i<extraSize;i++)
          {
            const unsigned int ID  = presplit_prio[i].index;
            const float prob       = presplit_prio[i].prob;

            if (prob > 0.0f)
            {
              const Vec3fa lower = prims0[ID].lower;
              const Vec3fa upper = prims0[ID].upper;
              const Vec3ia lower_binID = (Vec3ia)((lower-base)*scale);
              const Vec3ia upper_binID = (Vec3ia)((upper-base)*scale);
              const unsigned int lower_code = bitInterleave(lower_binID.x,lower_binID.y,lower_binID.z);
              const unsigned int upper_code = bitInterleave(upper_binID.x,upper_binID.y,upper_binID.z);
              const unsigned int diff = lzcnt(lower_code^upper_code);
              const unsigned int dim = (31 - diff) % 3;

              const unsigned int split_binID = (lower_binID[dim] + upper_binID[dim] + 1)/2;
              const float split_ratio = (float)split_binID * inv_lattice_size_per_dim;
              const float pos = base[dim] + split_ratio * diag[dim];
              BBox3fa left,right;

              if ( (pos - lower[dim]) > min_diag_threshold[dim] && (upper[dim] - pos) > min_diag_threshold[dim])
              {
                BBox3fa rest = prims0[ID].bounds();
                const auto spatial_splitter = splitter(prims0[ID]);
                BBox3fa left,right;
                spatial_splitter(rest,dim,pos,left,right);
                const size_t rightID = ext_elements.fetch_add(1);

                prims0[ID     ] = PrimRef(  left,lower.u,upper.u);
                prims0[rightID] = PrimRef( right,lower.u,upper.u);
              }
            }            
          }

          pinfo.end = ext_elements.load();
          assert(numSplitPrimitives >= pinfo.end);

          /* ================================ */                   
          /* == recompute centroid bounds  == */
          /* ================================ */

          BBox3fa centroid_bounds(empty);

          for (size_t i=pinfo.begin;i<pinfo.end;i++)
            centroid_bounds.extend(prims0[i].bounds().center2());

          pinfo.centBounds = centroid_bounds;

          /* ==================== */
#endif
          root = BVHNBuilderVirtual<N>::build(&bvh->alloc,CreateLeafSpatial<N,Primitive>(bvh),bvh->scene->progressInterface,prims0.data(),pinfo,settings);
        }

        bvh->set(root,LBBox3fa(pinfo.geomBounds),pinfo.size());
        bvh->layoutLargeNodes(size_t(pinfo.size()*0.005f));

	/* clear temporary data for static geometry */
	if (scene && scene->isStaticAccel()) {
          prims0.clear();
          bvh->shrink();
        }
	bvh->cleanup();
        bvh->postBuild(t0);
      }

      void clear() {
        prims0.clear();
      }
    };

    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/


#if defined(EMBREE_GEOMETRY_TRIANGLE)

    Builder* BVH4Triangle4SceneBuilderFastSpatialSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderFastSpatialSAH<4,TriangleMesh,Triangle4,TriangleSplitterFactory>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH4Triangle4vSceneBuilderFastSpatialSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderFastSpatialSAH<4,TriangleMesh,Triangle4v,TriangleSplitterFactory>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH4Triangle4iSceneBuilderFastSpatialSAH (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderFastSpatialSAH<4,TriangleMesh,Triangle4i,TriangleSplitterFactory>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }

#if defined(__AVX__)
    Builder* BVH8Triangle4SceneBuilderFastSpatialSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderFastSpatialSAH<8,TriangleMesh,Triangle4,TriangleSplitterFactory>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
    Builder* BVH8Triangle4vSceneBuilderFastSpatialSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderFastSpatialSAH<8,TriangleMesh,Triangle4v,TriangleSplitterFactory>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
#endif
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
    Builder* BVH4Quad4vSceneBuilderFastSpatialSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderFastSpatialSAH<4,QuadMesh,Quad4v,QuadSplitterFactory>((BVH4*)bvh,scene,4,1.0f,4,inf,mode); }

#if defined(__AVX__)
    Builder* BVH8Quad4vSceneBuilderFastSpatialSAH  (void* bvh, Scene* scene, size_t mode) { return new BVHNBuilderFastSpatialSAH<8,QuadMesh,Quad4v,QuadSplitterFactory>((BVH8*)bvh,scene,4,1.0f,4,inf,mode); }
#endif

#endif
  }
}
