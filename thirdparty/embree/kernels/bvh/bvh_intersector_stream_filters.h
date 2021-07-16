// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
