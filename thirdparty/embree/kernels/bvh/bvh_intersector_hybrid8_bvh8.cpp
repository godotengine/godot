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
    /// BVH8Intersector8 Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4Intersector8HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMIntersectorKMoeller  <SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4Intersector8HybridMoellerNoFilter,BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMIntersectorKMoeller  <SIMD_MODE(4) COMMA 8 COMMA false> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4iIntersector8HybridMoeller,       BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMiIntersectorKMoeller <SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4vIntersector8HybridPluecker,      BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMvIntersectorKPluecker<SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4iIntersector8HybridPluecker,      BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMiIntersectorKPluecker<SIMD_MODE(4) COMMA 8 COMMA true> > >));

    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4vMBIntersector8HybridMoeller,  BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMvMBIntersectorKMoeller <SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4iMBIntersector8HybridMoeller,  BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA TriangleMiMBIntersectorKMoeller <SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4vMBIntersector8HybridPluecker, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMvMBIntersectorKPluecker<SIMD_MODE(4) COMMA 8 COMMA true> > >));
    IF_ENABLED_TRIS(DEFINE_INTERSECTOR8(BVH8Triangle4iMBIntersector8HybridPluecker, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA true  COMMA ArrayIntersectorK_1<8 COMMA TriangleMiMBIntersectorKPluecker<SIMD_MODE(4) COMMA 8 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4vIntersector8HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4vIntersector8HybridMoellerNoFilter,BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKMoeller <4 COMMA 8 COMMA false> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4iIntersector8HybridMoeller,        BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMiIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4vIntersector8HybridPluecker,       BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA QuadMvIntersectorKPluecker<4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4iIntersector8HybridPluecker,       BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true  COMMA ArrayIntersectorK_1<8 COMMA QuadMiIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4iMBIntersector8HybridMoeller, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA QuadMiMBIntersectorKMoeller <4 COMMA 8 COMMA true> > >));
    IF_ENABLED_QUADS(DEFINE_INTERSECTOR8(BVH8Quad4iMBIntersector8HybridPluecker,BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA true COMMA ArrayIntersectorK_1<8 COMMA QuadMiMBIntersectorKPluecker<4 COMMA 8 COMMA true> > >));

    IF_ENABLED_CURVES(DEFINE_INTERSECTOR8(BVH8OBBVirtualCurveIntersector8Hybrid, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1_UN1 COMMA false COMMA VirtualCurveIntersectorK<8> >));
    IF_ENABLED_CURVES(DEFINE_INTERSECTOR8(BVH8OBBVirtualCurveIntersector8HybridMB, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN2_AN4D_UN2 COMMA false COMMA VirtualCurveIntersectorK<8> >));

    IF_ENABLED_USER(DEFINE_INTERSECTOR8(BVH8VirtualIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA ObjectIntersector8> >));
    IF_ENABLED_USER(DEFINE_INTERSECTOR8(BVH8VirtualMBIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA ObjectIntersector8MB> >));

    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR8(BVH8InstanceIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceIntersectorK<8>> >));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTOR8(BVH8InstanceMBIntersector8Chunk, BVHNIntersectorKChunk<8 COMMA 8 COMMA BVH_AN2_AN4D COMMA false COMMA ArrayIntersectorK_1<8 COMMA InstanceIntersectorKMB<8>> >));

    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH8GridIntersector8HybridMoeller, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA false COMMA SubGridIntersectorKMoeller <8 COMMA 8 COMMA true> >));
    IF_ENABLED_GRIDS(DEFINE_INTERSECTOR8(BVH8GridIntersector8HybridPluecker, BVHNIntersectorKHybrid<8 COMMA 8 COMMA BVH_AN1 COMMA true COMMA SubGridIntersectorKPluecker <8 COMMA 8 COMMA true> >));

  }
}

