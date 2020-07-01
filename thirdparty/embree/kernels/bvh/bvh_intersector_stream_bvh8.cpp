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

#include "bvh_intersector_stream.cpp"

namespace embree
{
  namespace isa
  {
    ////////////////////////////////////////////////////////////////////////////////
    /// General BVHIntersectorStreamPacketFallback Intersector
    ////////////////////////////////////////////////////////////////////////////////

    DEFINE_INTERSECTORN(BVH8IntersectorStreamPacketFallback,BVHNIntersectorStreamPacketFallback<SIMD_MODE(8)>);

    ////////////////////////////////////////////////////////////////////////////////
    /// BVH8IntersectorStream Definitions
    ////////////////////////////////////////////////////////////////////////////////

    IF_ENABLED_TRIS(DEFINE_INTERSECTORN(BVH8Triangle4IntersectorStreamMoeller,         BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA false COMMA Triangle4IntersectorStreamMoeller<true>>));
    IF_ENABLED_TRIS(DEFINE_INTERSECTORN(BVH8Triangle4IntersectorStreamMoellerNoFilter, BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA false COMMA Triangle4IntersectorStreamMoeller<false>>));
    IF_ENABLED_TRIS(DEFINE_INTERSECTORN(BVH8Triangle4iIntersectorStreamMoeller,        BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA false COMMA Triangle4iIntersectorStreamMoeller<true>>));
    IF_ENABLED_TRIS(DEFINE_INTERSECTORN(BVH8Triangle4vIntersectorStreamPluecker,       BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA true  COMMA Triangle4vIntersectorStreamPluecker<true>>));
    IF_ENABLED_TRIS(DEFINE_INTERSECTORN(BVH8Triangle4iIntersectorStreamPluecker,       BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA true  COMMA Triangle4iIntersectorStreamPluecker<true>>));

    IF_ENABLED_QUADS(DEFINE_INTERSECTORN(BVH8Quad4vIntersectorStreamMoeller,         BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA false COMMA Quad4vIntersectorStreamMoeller<true>>));
    IF_ENABLED_QUADS(DEFINE_INTERSECTORN(BVH8Quad4vIntersectorStreamMoellerNoFilter, BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA false COMMA Quad4vIntersectorStreamMoeller<false>>));
    IF_ENABLED_QUADS(DEFINE_INTERSECTORN(BVH8Quad4iIntersectorStreamMoeller,         BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA false COMMA Quad4iIntersectorStreamMoeller<true>>));
    IF_ENABLED_QUADS(DEFINE_INTERSECTORN(BVH8Quad4vIntersectorStreamPluecker,        BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA true  COMMA Quad4vIntersectorStreamPluecker<true>>));
    IF_ENABLED_QUADS(DEFINE_INTERSECTORN(BVH8Quad4iIntersectorStreamPluecker,        BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA true  COMMA Quad4iIntersectorStreamPluecker<true>>));

    IF_ENABLED_USER(DEFINE_INTERSECTORN(BVH8VirtualIntersectorStream,BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA false COMMA ObjectIntersectorStream>));
    IF_ENABLED_INSTANCE(DEFINE_INTERSECTORN(BVH8InstanceIntersectorStream,BVHNIntersectorStream<SIMD_MODE(8) COMMA BVH_AN1 COMMA false COMMA InstanceIntersectorStream>));
  }
}
