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
