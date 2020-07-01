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

#include "../common/default.h"
#include "../common/ray.h"
#include "../common/scene.h"

namespace embree
{
  namespace isa
  {
    class RayStreamFilter
    {
    public:
      static void intersectAOS(Scene* scene, RTCRayHit* rays, size_t N, size_t stride, IntersectContext* context);
      static void intersectAOP(Scene* scene, RTCRayHit** rays, size_t N, IntersectContext* context);
      static void intersectSOA(Scene* scene, char* rays, size_t N, size_t numPackets, size_t stride, IntersectContext* context);
      static void intersectSOP(Scene* scene, const RTCRayHitNp* rays, size_t N, IntersectContext* context);

      static void occludedAOS(Scene* scene, RTCRay* rays, size_t N, size_t stride, IntersectContext* context);
      static void occludedAOP(Scene* scene, RTCRay** rays, size_t N, IntersectContext* context);
      static void occludedSOA(Scene* scene, char* rays, size_t N, size_t numPackets, size_t stride, IntersectContext* context);
      static void occludedSOP(Scene* scene, const RTCRayNp* rays, size_t N, IntersectContext* context);

    private:
      template<int K, bool intersect>
      static void filterAOS(Scene* scene, void* rays, size_t N, size_t stride, IntersectContext* context);

      template<int K, bool intersect>
      static void filterAOP(Scene* scene, void** rays, size_t N, IntersectContext* context);

      template<int K, bool intersect>
      static void filterSOA(Scene* scene, char* rays, size_t N, size_t numPackets, size_t stride, IntersectContext* context);

      template<int K, bool intersect>
      static void filterSOP(Scene* scene, const void* rays, size_t N, IntersectContext* context);
    };
  }
};
