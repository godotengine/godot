// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_intersector_hybrid.cpp"

namespace embree
{
  namespace isa
  {
    ////////////////////////////////////////////////////////////////////////////////
    /// BVH8Intersector8 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4Intersector8HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMIntersectorKMoeller  <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4Intersector8HybridMoellerNoFilter,BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMIntersectorKMoeller  <4 COMMA 8 COMMA false> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4iIntersector8HybridMoeller,       BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMiIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4vIntersector8HybridPluecker,      BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMvIntersectorKPluecker<4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4iIntersector8HybridPluecker,      BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMiIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4vMBIntersector8HybridMoeller,  BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMvMBIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4iMBIntersector8HybridMoeller,  BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMiMBIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4vMBIntersector8HybridPluecker, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMvMBIntersectorKPluecker<4 COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4iMBIntersector8HybridPluecker, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMiMBIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4vIntersector8HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4vIntersector8HybridMoellerNoFilter,BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKMoeller <4 COMMA 8 COMMA false> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4iIntersector8HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMiIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4vIntersector8HybridPluecker,       BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKPluecker<4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4iIntersector8HybridPluecker,       BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA QuadMiIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4iMBIntersector8HybridMoeller, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMiMBIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4iMBIntersector8HybridPluecker,BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA true COMMA ArrayIntersectorK_1<8 COMMA QuadMiMBIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR8(BVH8OBBVirtualCurveIntersector8Hybrid, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersectorK<8> >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR8(BVH8OBBVirtualCurveIntersector8HybridMB, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersectorK<8> >));

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR8(BVH8OBBVirtualCurveIntersectorRobust8Hybrid, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1_UN1 COMMA true COMMA VirtualCurveIntersectorK<8> >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR8(BVH8OBBVirtualCurveIntersectorRobust8HybridMB, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D_UN2 COMMA true COMMA VirtualCurveIntersectorK<8> >));

    IF_ENABLED_USER(DEFINE_INTERSECTOR8(BVH8VirtualIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA ObjectIntersector8> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR8(BVH8VirtualMBIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA ObjectIntersector8MB> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR8(BVH8InstanceIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceIntersectorK<8>> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR8(BVH8InstanceMBIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceIntersectorKMB<8>> >));

    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR8(BVH8InstanceArrayIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceArrayIntersectorK<8>> >));
    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR8(BVH8InstanceArrayMBIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceArrayIntersectorKMB<8>> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH8GridIntersector8HybridMoeller, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA SubGridIntersectorKMoeller <8 COMMA 8 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH8GridIntersector8HybridPluecker, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true COMMA SubGridIntersectorKPluecker <8 COMMA 8 COMMA true> >));

  }
}

