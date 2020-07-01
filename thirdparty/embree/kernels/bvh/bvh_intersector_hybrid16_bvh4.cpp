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

#include "bvh_intersector_hybrid.cpp"

namespace embree
{
  namespace isa
  {
    ////////////////////////////////////////////////////////////////////////////////
    /// BVH4Intersector16 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4Intersector16HybridMoeller,         BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMIntersectorKMoeller  <SIMD_MODE(4) COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4Intersector16HybridMoellerNoFilter, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMIntersectorKMoeller  <SIMD_MODE(4) COMMA 16 COMMA false> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4iIntersector16HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMiIntersectorKMoeller <SIMD_MODE(4) COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4vIntersector16HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMvIntersectorKPluecker<SIMD_MODE(4) COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4iIntersector16HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMiIntersectorKPluecker<SIMD_MODE(4) COMMA 16 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4vMBIntersector16HybridMoeller,  BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMvMBIntersectorKMoeller <SIMD_MODE(4) COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4iMBIntersector16HybridMoeller,  BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA TriangleMiMBIntersectorKMoeller <SIMD_MODE(4) COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4vMBIntersector16HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMvMBIntersectorKPluecker<SIMD_MODE(4) COMMA 16 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR16(BVH4Triangle4iMBIntersector16HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA TriangleMiMBIntersectorKPluecker<SIMD_MODE(4) COMMA 16 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4vIntersector16HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKMoeller <4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4vIntersector16HybridMoellerNoFilter,BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKMoeller <4 COMMA 16 COMMA false> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4iIntersector16HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMiIntersectorKMoeller <4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4vIntersector16HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMvIntersectorKPluecker<4 COMMA 16 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4iIntersector16HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMiIntersectorKPluecker<4 COMMA 16 COMMA true > > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4iMBIntersector16HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA QuadMiMBIntersectorKMoeller <4 COMMA 16 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR16(BVH4Quad4iMBIntersector16HybridPluecker,BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<16 COMMA QuadMiMBIntersectorKPluecker<4 COMMA 16 COMMA true> > >));

    IF_ENABLED_CURVES(DEFINE_INTERSECTOR16(BVH4OBBVirtualCurveIntersector16Hybrid, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersectorK<16> >));
    IF_ENABLED_CURVES(DEFINE_INTERSECTOR16(BVH4OBBVirtualCurveIntersector16HybridMB,BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersectorK<16> >));
 
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR16(BVH4SubdivPatch1Intersector16, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true COMMA SubdivPatch1Intersector16>));
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR16(BVH4SubdivPatch1MBIntersector16, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA SubdivPatch1MBIntersector16>));

    IF_ENABLED_USER(DEFINE_INTERSECTOR16(BVH4VirtualIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA ObjectIntersector16> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR16(BVH4VirtualMBIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA ObjectIntersector16MB> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR16(BVH4InstanceIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceIntersectorK<16>> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR16(BVH4InstanceMBIntersector16Chunk, BVHNIntersectorKChunk<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<16 COMMA InstanceIntersectorKMB<16>> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR16(BVH4GridIntersector16HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA false COMMA SubGridIntersectorKMoeller <4 COMMA 16 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR16(BVH4GridMBIntersector16HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN2_AN4D COMMA true COMMA SubGridMBIntersectorKPluecker <4 COMMA 16 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR16(BVH4GridIntersector16HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 16 COMMA BVH_AN1 COMMA true COMMA SubGridIntersectorKPluecker <4 COMMA 16 COMMA true> >));

  }
}

