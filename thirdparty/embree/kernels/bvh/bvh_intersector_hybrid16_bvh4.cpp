// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_intersector_hybrid.cpp"

namespace embree
{
  namespace isa
  {
    ////////////////////////////////////////////////////////////////////////////////
    /// BVH4Intersector16 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4Intersector16HybridMoeller,         BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMIntersectorKMoeller  <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4Intersector16HybridMoellerNoFilter, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMIntersectorKMoeller  <4 COMMA 16 COMMA false> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4iIntersector16HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMiIntersectorKMoeller <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4vIntersector16HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMvIntersectorKPluecker<4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4iIntersector16HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMiIntersectorKPluecker<4 COMMA 16 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4vMBIntersector16HybridMoeller,  BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMvMBIntersectorKMoeller <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4iMBIntersector16HybridMoeller,  BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMiMBIntersectorKMoeller <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4vMBIntersector16HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMvMBIntersectorKPluecker<4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4iMBIntersector16HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMiMBIntersectorKPluecker<4 COMMA 16 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4vIntersector16HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKMoeller <4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4vIntersector16HybridMoellerNoFilter,BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKMoeller <4 COMMA 16 COMMA false> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4iIntersector16HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMiIntersectorKMoeller <4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4vIntersector16HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKPluecker<4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4iIntersector16HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMiIntersectorKPluecker<4 COMMA 16 COMMA true > > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4iMBIntersector16HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMiMBIntersectorKMoeller <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4iMBIntersector16HybridPluecker,BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMiMBIntersectorKPluecker<4 COMMA 16 COMMA true> > >));

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR16(BVH4OBBVirtualCurveIntersector16Hybrid, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersectorK<16> >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR16(BVH4OBBVirtualCurveIntersector16HybridMB,BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersectorK<16> >));
 
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR16(BVH4OBBVirtualCurveIntersectorRobust16Hybrid, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1_UN1 COMMA true COMMA VirtualCurveIntersectorK<16> >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR16(BVH4OBBVirtualCurveIntersectorRobust16HybridMB,BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D_UN2 COMMA true COMMA VirtualCurveIntersectorK<16> >));
 
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR16(BVH4SubdivPatch1Intersector16, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true COMMA SubdivPatch1Intersector16>));
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR16(BVH4SubdivPatch1MBIntersector16, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA SubdivPatch1MBIntersector16>));

    IF_ENABLED_USER(DEFINE_INTERSECTOR16(BVH4VirtualIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA ObjectIntersector16> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR16(BVH4VirtualMBIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA ObjectIntersector16MB> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR16(BVH4InstanceIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceIntersectorK<16>> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR16(BVH4InstanceMBIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceIntersectorKMB<16>> >));

    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR16(BVH4InstanceArrayIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceArrayIntersectorK<16>> >));
    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR16(BVH4InstanceArrayMBIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceArrayIntersectorKMB<16>> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR16(BVH4GridIntersector16HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA SubGridIntersectorKMoeller <4 COMMA 16 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR16(BVH4GridMBIntersector16HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA true COMMA SubGridMBIntersectorKPluecker <4 COMMA 16 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR16(BVH4GridIntersector16HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true COMMA SubGridIntersectorKPluecker <4 COMMA 16 COMMA true> >));

  }
}

