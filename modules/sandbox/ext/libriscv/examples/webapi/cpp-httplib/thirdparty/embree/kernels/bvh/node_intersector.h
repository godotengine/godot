// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"

namespace embree
{
  namespace isa
  {
    struct NearFarPrecalculations
    {
      size_t nearX, nearY, nearZ;
      size_t farX, farY, farZ;

      __forceinline NearFarPrecalculations() {}

      __forceinline NearFarPrecalculations(const Vec3fa& dir, size_t N)
      {
        const size_t size = sizeof(float)*N;
        nearX = (dir.x < 0.0f) ? 1*size : 0*size;
        nearY = (dir.y < 0.0f) ? 3*size : 2*size;
        nearZ = (dir.z < 0.0f) ? 5*size : 4*size;
        farX  = nearX ^ size;
        farY  = nearY ^ size;
        farZ  = nearZ ^ size;
      }
    };
  }
}
