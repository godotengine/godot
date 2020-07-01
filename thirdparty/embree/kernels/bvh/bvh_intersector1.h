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

#pragma once

#include "bvh.h"
#include "../common/ray.h"

namespace embree
{
  namespace isa
  {
    /*! BVH single ray intersector. */
    template<int N, int types, bool robust, typename PrimitiveIntersector1>
    class BVHNIntersector1
    {
      /* shortcuts for frequently used types */
      typedef typename PrimitiveIntersector1::Precalculations Precalculations;
      typedef typename PrimitiveIntersector1::Primitive Primitive;
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::AlignedNode AlignedNode;
      typedef typename BVH::AlignedNodeMB4D AlignedNodeMB4D;

      static const size_t stackSize = 1+(N-1)*BVH::maxDepth+3; // +3 due to 16-wide store

      /* right now AVX512KNL SIMD extension only for standard node types */
      static const size_t Nx = (types == BVH_AN1 || types == BVH_QN1) ? vextend<N>::size : N;

    public:
      static void intersect(const Accel::Intersectors* This, RayHit& ray, IntersectContext* context);
      static void occluded (const Accel::Intersectors* This, Ray& ray, IntersectContext* context);
    };
  }
}
