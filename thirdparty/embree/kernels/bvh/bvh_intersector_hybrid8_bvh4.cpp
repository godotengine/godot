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
    /// BVH4Intersector8 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4Intersector8HybridMoeller,         BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMIntersectorKMoeller  <SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4Intersector8HybridMoellerNoFilter, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMIntersectorKMoeller  <SIMD_MODE(4) COMMA 8 COMMA false> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4iIntersector8HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMiIntersectorKMoeller <SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4vIntersector8HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMvIntersectorKPluecker<SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4iIntersector8HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMiIntersectorKPluecker<SIMD_MODE(4) COMMA 8 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4vMBIntersector8HybridMoeller,  BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMvMBIntersectorKMoeller <SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4iMBIntersector8HybridMoeller,  BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMiMBIntersectorKMoeller <SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4vMBIntersector8HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMvMBIntersectorKPluecker<SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH4Triangle4iMBIntersector8HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMiMBIntersectorKPluecker<SIMD_MODE(4) COMMA 8 COMMA true> > >));
    
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4vIntersector8HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKMoeller<4 COMMA 8 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4vIntersector8HybridMoellerNoFilter,BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKMoeller<4 COMMA 8 COMMA false> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4iIntersector8HybridMoeller,        BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMiIntersectorKMoeller<4 COMMA 8 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4vIntersector8HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKPluecker<4 COMMA 8 COMMA true > > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4iIntersector8HybridPluecker,       BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA QuadMiIntersectorKPluecker<4 COMMA 8 COMMA true > > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4iMBIntersector8HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMiMBIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH4Quad4iMBIntersector8HybridPluecker,BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA true COMMA ArrayIntersectorK_1<8 COMMA QuadMiMBIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_CURVES(DEFINE_INTERSECTOR8(BVH4OBBVirtualCurveIntersector8Hybrid, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersectorK<8> >));
    IF_ENABLED_CURVES(DEFINE_INTERSECTOR8(BVH4OBBVirtualCurveIntersector8HybridMB,BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersectorK<8> >));
    
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR8(BVH4SubdivPatch1Intersector8, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true COMMA SubdivPatch1Intersector8>));
    IF_ENABLED_SUBDIV(DEFINE_INTERSECTOR8(BVH4SubdivPatch1MBIntersector8, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA SubdivPatch1MBIntersector8>));

    IF_ENABLED_USER(DEFINE_INTERSECTOR8(BVH4VirtualIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA ObjectIntersector8> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR8(BVH4VirtualMBIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA ObjectIntersector8MB> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR8(BVH4InstanceIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceIntersectorK<8>> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR8(BVH4InstanceMBIntersector8Chunk, BVHNIntersectorKChunk<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceIntersectorKMB<8>> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH4GridIntersector8HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA false COMMA SubGridIntersectorKMoeller <4 COMMA 8 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH4GridMBIntersector8HybridMoeller, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN2_AN4D COMMA true COMMA SubGridMBIntersectorKPluecker <4 COMMA 8 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH4GridIntersector8HybridPluecker, BVHNIntersectorKHybrid<4 COMMA 8 COMMA BVH_AN1 COMMA true COMMA SubGridIntersectorKPluecker <4 COMMA 8 COMMA true> >));

  }


}
