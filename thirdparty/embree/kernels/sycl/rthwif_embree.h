// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-W#pragma-messages"

#include <sycl/sycl.hpp>

#pragma clang diagnostic pop

#include "../config.h"

#include "../../include/embree4/rtcore.h"
#include "../../include/embree4/rtcore_ray.h"

namespace embree
{
  inline uint8_t mask32_to_mask8( uint32_t mask ) {
#if defined(EMBREE_RAY_MASK)
    return (mask & 0xFFFFFF80) ? (0x80 | mask) : mask; // bit 7 indicates that some bit >= 7 is set
#else
    return 1;
#endif
  }

  struct EmbreeHWAccel
  {
    uint32_t numTimeSegments;
    void* AccelTable[1];
  };
}

SYCL_EXTERNAL void rtcIntersectRTHW(sycl::global_ptr<RTCSceneTy> hscene, sycl::private_ptr<RTCRayQueryContext> context, sycl::private_ptr<RTCRayHit> rayhit, sycl::private_ptr<RTCIntersectArguments> args);
SYCL_EXTERNAL void rtcOccludedRTHW(sycl::global_ptr<RTCSceneTy> hscene, sycl::private_ptr<RTCRayQueryContext> context, sycl::private_ptr<RTCRay> ray, sycl::private_ptr<RTCOccludedArguments> args);
