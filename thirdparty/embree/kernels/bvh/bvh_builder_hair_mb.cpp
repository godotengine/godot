// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../builders/bvh_builder_msmblur_hair.h"
#include "../builders/primrefgen.h"

#include "../geometry/pointi.h"
#include "../geometry/linei.h"
#include "../geometry/curveNi_mb.h"

#if defined(EMBREE_GEOMETRY_CURVE) || defined(EMBREE_GEOMETRY_POINT)

namespace embree
{
  namespace isa
  {
    /* FIXME: add fast path for single-segment motion blur */
    template<int N, typename CurvePrimitive, typename LinePrimitive, typename PointPrimitive>
    struct BVHNHairMBlurBuilderSAH : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::NodeRecordMB4D NodeRecordMB4D;

      BVH* bvh;
      Scene* scene;

      BVHNHairMBlurBuilderSAH (BVH* bvh, Scene* scene)
        : bvh(bvh), scene(scene) {}
      
      void build() 
      {
        /* fast path for empty BVH */
        const size_t numPrimitives = scene->getNumPrimitives(Geometry::MTY_CURVES,true);
        if (numPrimitives == 0) {
          bvh->set(BVH::emptyNode,empty,0);
          return;
        }

        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH" + toString(N) + "HairMBlurBuilderSAH");

        //profile(1,5,numPrimitives,[&] (ProfileTimer& timer) {

        /* create primref array */
        mvector<PrimRefMB> prims0(scene->device,numPrimitives);
        const PrimInfoMB pinfo = createPrimRefArrayMSMBlur(scene,Geometry::MTY_CURVES,numPrimitives,prims0,bvh->scene->progressInterface);

        /* estimate acceleration structure size */
        const size_t node_bytes = pinfo.num_time_segments*sizeof(typename BVH::AABBNodeMB)/(4*N);
        const size_t leaf_bytes = CurvePrimitive::bytes(pinfo.num_time_segments);
        bvh->alloc.init_estimate(node_bytes+leaf_bytes);
    
        /* settings for BVH build */
        BVHBuilderHairMSMBlur::Settings settings;
        settings.branchingFactor = N;
        settings.maxDepth = BVH::maxBuildDepthLeaf;
        settings.logBlockSize = bsf(CurvePrimitive::max_size());
        settings.minLeafSize = CurvePrimitive::max_size();
        settings.maxLeafSize = CurvePrimitive::max_size();

        /* creates a leaf node */
        auto createLeaf = [&] (const SetMB& prims, const FastAllocator::CachedAllocator& alloc) -> NodeRecordMB4D {

          if (prims.size() == 0)
            return NodeRecordMB4D(BVH::emptyNode,empty,empty);
          
          const unsigned int geomID0 = (*prims.prims)[prims.begin()].geomID();
          if (scene->get(geomID0)->getTypeMask() & Geometry::MTY_POINTS)
            return PointPrimitive::createLeafMB(bvh,prims,alloc);
          else if (scene->get(geomID0)->getCurveBasis() == Geometry::GTY_BASIS_LINEAR)
            return LinePrimitive::createLeafMB(bvh,prims,alloc);
          else
            return CurvePrimitive::createLeafMB(bvh,prims,alloc);
        };

        /* build the hierarchy */
        auto root = BVHBuilderHairMSMBlur::build<NodeRef>
          (scene, prims0, pinfo,
           VirtualRecalculatePrimRef(scene),
           typename BVH::CreateAlloc(bvh),
           typename BVH::AABBNodeMB4D::Create(),
           typename BVH::AABBNodeMB4D::Set(),
           typename BVH::OBBNodeMB::Create(),
           typename BVH::OBBNodeMB::Set(),
           createLeaf,
           bvh->scene->progressInterface,
           settings);
        
        bvh->set(root.ref,root.lbounds,pinfo.num_time_segments);
        
        //});
        
        /* clear temporary data for static geometry */
        bvh->cleanup();
        bvh->postBuild(t0);
      }

      void clear() {
      }
    };
    
    /*! entry functions for the builder */
    Builder* BVH4OBBCurve4iMBBuilder_OBB (void* bvh, Scene* scene, size_t mode) { return new BVHNHairMBlurBuilderSAH<4,Curve4iMB,Line4i,Point4i>((BVH4*)bvh,scene); }

#if defined(__AVX__)
    Builder* BVH4OBBCurve8iMBBuilder_OBB (void* bvh, Scene* scene, size_t mode) { return new BVHNHairMBlurBuilderSAH<4,Curve8iMB,Line8i,Point8i>((BVH4*)bvh,scene); }
    Builder* BVH8OBBCurve8iMBBuilder_OBB (void* bvh, Scene* scene, size_t mode) { return new BVHNHairMBlurBuilderSAH<8,Curve8iMB,Line8i,Point8i>((BVH8*)bvh,scene); }
#endif

  }
}
#endif
