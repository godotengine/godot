// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_intersector_hybrid.cpp"

namespace embree
{
  namespace isa
  {
    ////////////////////////////////////////////////////////////////////////////////
    /// BVH8Intersector16 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4Intersector16HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMIntersectorKMoeller  <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4Intersector16HybridMoellerNoFilter,BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMIntersectorKMoeller  <4 COMMA 16 COMMA false> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4iIntersector16HybridMoeller,       BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMiIntersectorKMoeller <4 COMMA 16 COMMA true > > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4vIntersector16HybridPluecker,      BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMvIntersectorKPluecker<4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4iIntersector16HybridPluecker,      BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMiIntersectorKPluecker<4 COMMA 16 COMMA true > > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4vMBIntersector16HybridMoeller, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMvMBIntersectorKMoeller <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4iMBIntersector16HybridMoeller, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMiMBIntersectorKMoeller <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4vMBIntersector16HybridPluecker,BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMvMBIntersectorKPluecker<4 COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH8Triangle4iMBIntersector16HybridPluecker,BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMiMBIntersectorKPluecker<4 COMMA 16 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH8Quad4vIntersector16HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKMoeller <4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH8Quad4vIntersector16HybridMoellerNoFilter,BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKMoeller <4 COMMA 16 COMMA false> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH8Quad4iIntersector16HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMiIntersectorKMoeller <4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH8Quad4vIntersector16HybridPluecker,       BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKPluecker<4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH8Quad4iIntersector16HybridPluecker,       BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMiIntersectorKPluecker<4 COMMA 16 COMMA true > > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH8Quad4iMBIntersector16HybridMoeller, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMiMBIntersectorKMoeller <4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH8Quad4iMBIntersector16HybridPluecker,BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMiMBIntersectorKPluecker<4 COMMA 16 COMMA true > > >));

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR16(BVH8OBBVirtualCurveIntersector16Hybrid, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersectorK<16> >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR16(BVH8OBBVirtualCurveIntersector16HybridMB, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersectorK<16> >));

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR16(BVH8OBBVirtualCurveIntersectorRobust16Hybrid, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1_UN1 COMMA true COMMA VirtualCurveIntersectorK<16> >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR16(BVH8OBBVirtualCurveIntersectorRobust16HybridMB, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN2_AN4D_UN2 COMMA true COMMA VirtualCurveIntersectorK<16> >));

    IF_ENABLED_USER(DEFINE_INTERSECTOR16(BVH8VirtualIntersector16Chunk, BVHNIntersectorKChunk<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA ObjectIntersector16> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR16(BVH8VirtualMBIntersector16Chunk, BVHNIntersectorKChunk<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA ObjectIntersector16MB> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR16(BVH8InstanceIntersector16Chunk, BVHNIntersectorKChunk<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceIntersectorK<16>> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR16(BVH8InstanceMBIntersector16Chunk, BVHNIntersectorKChunk<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceIntersectorKMB<16>> >));

    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR16(BVH8InstanceArrayIntersector16Chunk, BVHNIntersectorKChunk<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceArrayIntersectorK<16>> >));
    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR16(BVH8InstanceArrayMBIntersector16Chunk, BVHNIntersectorKChunk<8 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceArrayIntersectorKMB<16>> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR16(BVH8GridIntersector16HybridMoeller, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA false COMMA SubGridIntersectorKMoeller <8 COMMA 16 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR16(BVH8GridIntersector16HybridPluecker, BVHNIntersectorKHybrid<8 COMMA 16 COMMA BVH_AN1 COMMA true COMMA SubGridIntersectorKPluecker <8 COMMA 16 COMMA true> >));

  }
}

