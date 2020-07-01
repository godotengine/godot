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

#include "scene_user_geometry.h"
#include "scene.h"

namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  UserGeometry::UserGeometry (Device* device, unsigned int items, unsigned int numTimeSteps) 
    : AccelSet(device,Geometry::GTY_USER_GEOMETRY,items,numTimeSteps) {}

  void UserGeometry::enabling () {
    if (numTimeSteps == 1) scene->world.numUserGeometries += numPrimitives;
    else                   scene->worldMB.numUserGeometries += numPrimitives;
  }
  
  void UserGeometry::disabling() { 
    if (numTimeSteps == 1) scene->world.numUserGeometries -= numPrimitives;
    else                   scene->worldMB.numUserGeometries -= numPrimitives;
  }
  
  void UserGeometry::setMask (unsigned mask) 
  {
    this->mask = mask; 
    Geometry::update();
  }

  void UserGeometry::setBoundsFunction (RTCBoundsFunction bounds, void* userPtr) {
    this->boundsFunc = bounds;
  }

  void UserGeometry::setIntersectFunctionN (RTCIntersectFunctionN intersect) {
    intersectorN.intersect = intersect;
  }

  void UserGeometry::setOccludedFunctionN (RTCOccludedFunctionN occluded) {
    intersectorN.occluded = occluded;
  }
  
#endif

  namespace isa
  {
    UserGeometry* createUserGeometry(Device* device) {
      return new UserGeometryISA(device);
    }
  }
}
