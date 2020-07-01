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
#include "../common/stack_item.h"
#include "node_intersector_frustum.h"

namespace embree
{
  namespace isa 
  {
    template<int K, bool robust>
    struct TravRayK;

    /*! BVH hybrid packet intersector. Switches between packet and single ray traversal (optional). */
    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK, bool single = true>
    class BVHNIntersectorKHybrid
    {
      /* right now AVX512KNL SIMD extension only for standard node types */
      static const size_t Nx = types == BVH_AN1 ? vextend<N>::size : N;

      /* shortcuts for frequently used types */
      typedef typename PrimitiveIntersectorK::Precalculations Precalculations;
      typedef typename PrimitiveIntersectorK::Primitive Primitive;
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::BaseNode BaseNode;
      typedef typename BVH::AlignedNode AlignedNode;
      
      static const size_t stackSizeSingle = 1+(N-1)*BVH::maxDepth+3; // +3 due to 16-wide store
      static const size_t stackSizeChunk = 1+(N-1)*BVH::maxDepth;

      static const size_t switchThresholdIncoherent = \
      (K==4)  ? 3 :
      (K==8)  ? ((N==4) ? 5 : 7) :
      (K==16) ? 14 : // 14 seems to work best for KNL due to better ordered chunk traversal
      0;

    private:
      static void intersect1(Accel::Intersectors* This, const BVH* bvh, NodeRef root, size_t k, Precalculations& pre,
                             RayHitK<K>& ray, const TravRayK<K, robust>& tray, IntersectContext* context);
      static bool occluded1(Accel::Intersectors* This, const BVH* bvh, NodeRef root, size_t k, Precalculations& pre,
                            RayK<K>& ray, const TravRayK<K, robust>& tray, IntersectContext* context);

    public:
      static void intersect(vint<K>* valid, Accel::Intersectors* This, RayHitK<K>& ray, IntersectContext* context);
      static void occluded (vint<K>* valid, Accel::Intersectors* This, RayK<K>& ray, IntersectContext* context);

      static void intersectCoherent(vint<K>* valid, Accel::Intersectors* This, RayHitK<K>& ray, IntersectContext* context);
      static void occludedCoherent (vint<K>* valid, Accel::Intersectors* This, RayK<K>& ray, IntersectContext* context);

    };

    /*! BVH packet intersector. */
    template<int N, int K, int types, bool robust, typename PrimitiveIntersectorK>
    class BVHNIntersectorKChunk : public BVHNIntersectorKHybrid<N, K, types, robust, PrimitiveIntersectorK, false> {};
  }
}
