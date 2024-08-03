// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_intersector1.cpp"

namespace embree
{
  namespace isa
  {
    ////////////////////////////////////////////////////////////////////////////////
    /// BVH8Intersector1 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR1(BVH8OBBVirtualCurveIntersector1,BVHNIntersector1<8 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersector1 >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR1(BVH8OBBVirtualCurveIntersector1MB,BVHNIntersector1<8 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersector1 >));

    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR1(BVH8OBBVirtualCurveIntersectorRobust1,BVHNIntersector1<8 COMMA BVH_AN1_UN1 COMMA true COMMA VirtualCurveIntersector1 >));
    IF_ENABLED_CURVES_OR_POINTS(DEFINE_INTERSECTOR1(BVH8OBBVirtualCurveIntersectorRobust1MB,BVHNIntersector1<8 COMMA BVH_AN2_AN4D_UN2 COMMA true COMMA VirtualCurveIntersector1 >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4Intersector1Moeller,  BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<TriangleMIntersector1Moeller  <4 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4iIntersector1Moeller, BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<TriangleMiIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4vIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersector1<TriangleMvIntersector1Pluecker<4 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4iIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersector1<TriangleMiIntersector1Pluecker<4 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4vIntersector1Woop,  BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<TriangleMvIntersector1Woop  <4 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4vMBIntersector1Moeller, BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<TriangleMvMBIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4iMBIntersector1Moeller, BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<TriangleMiMBIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4vMBIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersector1<TriangleMvMBIntersector1Pluecker<4 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH8Triangle4iMBIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersector1<TriangleMiMBIntersector1Pluecker<4 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH8Quad4vIntersector1Moeller, BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<QuadMvIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH8Quad4iIntersector1Moeller, BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<QuadMiIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH8Quad4vIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersector1<QuadMvIntersector1Pluecker<4 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH8Quad4iIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersector1<QuadMiIntersector1Pluecker<4 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH8Quad4iMBIntersector1Moeller, BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<QuadMiMBIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH8Quad4iMBIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersector1<QuadMiMBIntersector1Pluecker<4 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(QBVH8Triangle4iIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_QN1 COMMA false COMMA ArrayIntersector1<TriangleMiIntersector1Pluecker<4 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(QBVH8Triangle4Intersector1Moeller,BVHNIntersector1<8 COMMA BVH_QN1 COMMA false COMMA ArrayIntersector1<TriangleMIntersector1Moeller  <4 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(QBVH8Quad4iIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_QN1 COMMA false COMMA ArrayIntersector1<QuadMiIntersector1Pluecker<4 COMMA true> > >));

    IF_ENABLED_USER(DEFINE_INTERSECTOR1(BVH8VirtualIntersector1,BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<ObjectIntersector1<false>> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR1(BVH8VirtualMBIntersector1,BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<ObjectIntersector1<true>> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR1(BVH8InstanceIntersector1,BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<InstanceIntersector1> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR1(BVH8InstanceMBIntersector1,BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<InstanceIntersector1MB> >));

    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR1(BVH8InstanceArrayIntersector1,BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<InstanceArrayIntersector1> >));
    IF_ENABLED_INSTANCE_ARRAY(DEFINE_INTERSECTOR1(BVH8InstanceArrayMBIntersector1,BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<InstanceArrayIntersector1MB> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR1(BVH8GridIntersector1Moeller,BVHNIntersector1<8 COMMA BVH_AN1 COMMA false COMMA SubGridIntersector1Moeller<8 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR1(BVH8GridMBIntersector1Moeller,BVHNIntersector1<8 COMMA BVH_AN2_AN4D COMMA true COMMA SubGridMBIntersector1Pluecker<8 COMMA true> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR1(BVH8GridIntersector1Pluecker,BVHNIntersector1<8 COMMA BVH_AN1 COMMA true COMMA SubGridIntersector1Pluecker<8 COMMA true> >));

  }
}
