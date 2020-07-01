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

#include "accelset.h"
#include "scene.h"

namespace embree
{
  AccelSet::AccelSet (Device* device, Geometry::GType gtype, size_t numItems, size_t numTimeSteps) 
    : Geometry(device,gtype,(unsigned int)numItems,(unsigned int)numTimeSteps), boundsFunc(nullptr) {}

  AccelSet::IntersectorN::IntersectorN (ErrorFunc error) 
    : intersect((IntersectFuncN)error), occluded((OccludedFuncN)error), name(nullptr) {}
  
  AccelSet::IntersectorN::IntersectorN (IntersectFuncN intersect, OccludedFuncN occluded, const char* name)
    : intersect(intersect), occluded(occluded), name(name) {}
}
