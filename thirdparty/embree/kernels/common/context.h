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

#include "default.h"
#include "rtcore.h"

namespace embree
{
  class Scene;

  struct IntersectContext
  {
  public:
    __forceinline IntersectContext(Scene* scene, RTCIntersectContext* user_context)
      : scene(scene), user(user_context), instID(user_context->instID[0]) {}

    __forceinline bool hasContextFilter() const {
      return user->filter != nullptr;
    }

    __forceinline bool isCoherent() const {
      return embree::isCoherent(user->flags);
    }

    __forceinline bool isIncoherent() const {
      return embree::isIncoherent(user->flags);
    }
    
  public:
    Scene* scene;
    RTCIntersectContext* user;
    unsigned int instID;
  };
}
