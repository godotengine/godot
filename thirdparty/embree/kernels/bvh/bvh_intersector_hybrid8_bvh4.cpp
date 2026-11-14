// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_intersector_hybrid.cpp"

namespace embree
{
  namespace isa
  {
    ////////////////////////////////////////////////////////////////////////////////
    /// BVH4Intersector8 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4Intersector8HybridMoeller,         BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMIntersectorKMoeller  <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4Intersector8HybridMoellerNoFilter, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMIntersectorKMoeller  <4 COMMA 8 COMMA false> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4iIntersector8HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMiIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4vIntersector8HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMvIntersectorKPluecker<4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4iIntersector8HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMiIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4vMBIntersector8HybridMoeller,  BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMvMBIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4iMBIntersector8HybridMoeller,  BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMiMBIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4vMBIntersector8HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMvMBIntersectorKPluecker<4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4iMBIntersector8HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMiMBIntersectorKPluecker<4 COMMA 8 COMMA true> > >));
    
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4vIntersector8HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKMoeller<4 COMMA 8 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4vIntersector8HybridMoellerNoFilter,BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKMoeller<4 COMMA 8 COMMA false> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4iIntersector8HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMiIntersectorKMoeller<4 COMMA 8 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4vIntersector8HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKPluecker<4 COMMA 8 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4iIntersector8HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA QuadMiIntersectorKPluecker<4 COMMA 8 COMMA true > > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4iMBIntersector8HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMiMBIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4iMBIntersector8HybridPluecker,BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA true COMMA ArrayIntersectorK_1<8 COMMA QuadMiMBIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR8(BVH4OBBVirtualCurveIntersector8Hybrid, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersectorK<8> >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR8(BVH4OBBVirtualCurveIntersector8HybridMB,BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersectorK<8> >));

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR8(BVH4OBBVirtualCurveIntersectorRobust8Hybrid, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1_UN1 COMMA true COMMA VirtualCurveIntersectorK<8> >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR8(BVH4OBBVirtualCurveIntersectorRobust8HybridMB,BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D_UN2 COMMA true COMMA VirtualCurveIntersectorK<8> >));
    
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR8(BVH4SubdivPatch1Intersector8, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true COMMA SubdivPatch1Intersector8>));
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR8(BVH4SubdivPatch1MBIntersector8, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA SubdivPatch1MBIntersector8>));

    IF_ENABLED_USER(DEFINE_INTERSECTOR8(BVH4VirtualIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA ObjectIntersector8> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR8(BVH4VirtualMBIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA ObjectIntersector8MB> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR8(BVH4InstanceIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceIntersectorK<8>> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR8(BVH4InstanceMBIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceIntersectorKMB<8>> >));

    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR8(BVH4InstanceArrayIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceArrayIntersectorK<8>> >));
    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR8(BVH4InstanceArrayMBIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceArrayIntersectorKMB<8>> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH4GridIntersector8HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA SubGridIntersectorKMoeller <4 COMMA 8 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH4GridMBIntersector8HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA true COMMA SubGridMBIntersectorKPluecker <4 COMMA 8 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH4GridIntersector8HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true COMMA SubGridIntersectorKPluecker <4 COMMA 8 COMMA true> >));

  }


}
