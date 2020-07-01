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

#include "bvh_intersector1.cpp"

namespace embree
{
  namespace isa
  {
    int getISA() {
      return VerifyMultiTargetLinking::getISA();
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// BVH4Intersector1 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_CURVES(DEFINE_INTERSECTOR1(BVH4OBBVirtualCurveIntersector1,BVHNIntersector1<4 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersector1 >));
    IF_ENABLED_CURVES(DEFINE_INTERSECTOR1(BVH4OBBVirtualCurveIntersector1MB,BVHNIntersector1<4 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersector1 >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH4Triangle4Intersector1Moeller,  BVHNIntersector1<4 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<TriangleMIntersector1Moeller  <SIMD_MODE(4) COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH4Triangle4iIntersector1Moeller, BVHNIntersector1<4 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<TriangleMiIntersector1Moeller <SIMD_MODE(4) COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH4Triangle4vIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersector1<TriangleMvIntersector1Pluecker<SIMD_MODE(4) COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH4Triangle4iIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersector1<TriangleMiIntersector1Pluecker<SIMD_MODE(4) COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH4Triangle4vMBIntersector1Moeller, BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<TriangleMvMBIntersector1Moeller <SIMD_MODE(4) COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH4Triangle4iMBIntersector1Moeller, BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<TriangleMiMBIntersector1Moeller <SIMD_MODE(4) COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH4Triangle4vMBIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersector1<TriangleMvMBIntersector1Pluecker<SIMD_MODE(4) COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(BVH4Triangle4iMBIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersector1<TriangleMiMBIntersector1Pluecker<SIMD_MODE(4) COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH4Quad4vIntersector1Moeller, BVHNIntersector1<4 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<QuadMvIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH4Quad4iIntersector1Moeller, BVHNIntersector1<4 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<QuadMiIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH4Quad4vIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersector1<QuadMvIntersector1Pluecker<4 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH4Quad4iIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersector1<QuadMiIntersector1Pluecker<4 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH4Quad4iMBIntersector1Moeller, BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<QuadMiMBIntersector1Moeller <4 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(BVH4Quad4iMBIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersector1<QuadMiMBIntersector1Pluecker<4 COMMA true> > >));

    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR1(BVH4SubdivPatch1Intersector1,BVHNIntersector1<4 COMMA BVH_AN1 COMMA true COMMA SubdivPatch1Intersector1>));
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR1(BVH4SubdivPatch1MBIntersector1,BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA true COMMA SubdivPatch1MBIntersector1>));
    
    IF_ENABLED_USER(DEFINE_INTERSECTOR1(BVH4VirtualIntersector1,BVHNIntersector1<4 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<ObjectIntersector1<false>> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR1(BVH4VirtualMBIntersector1,BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<ObjectIntersector1<true>> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR1(BVH4InstanceIntersector1,BVHNIntersector1<4 COMMA BVH_AN1 COMMA false COMMA ArrayIntersector1<InstanceIntersector1> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR1(BVH4InstanceMBIntersector1,BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersector1<InstanceIntersector1MB> >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR1(QBVH4Triangle4iIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_QN1 COMMA false COMMA ArrayIntersector1<TriangleMiIntersector1Pluecker<SIMD_MODE(4) COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR1(QBVH4Quad4iIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_QN1 COMMA false COMMA ArrayIntersector1<QuadMiIntersector1Pluecker<4 COMMA true> > >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR1(BVH4GridIntersector1Moeller,BVHNIntersector1<4 COMMA BVH_AN1 COMMA false COMMA SubGridIntersector1Moeller<4 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR1(BVH4GridMBIntersector1Moeller,BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA true COMMA SubGridMBIntersector1Pluecker<4 COMMA true> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR1(BVH4GridIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN1 COMMA true COMMA SubGridIntersector1Pluecker<4 COMMA true> >));
    //IF_ENABLED_GRIDS(DEFINE_INTERSECTOR1(BVH4GridMBIntersector1Pluecker,BVHNIntersector1<4 COMMA BVH_AN2_AN4D COMMA false COMMA SubGridMBIntersector1Pluecker<4 COMMA true> >));

  }
}
