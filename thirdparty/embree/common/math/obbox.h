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

#include "bbox.h"
#include "linearspace3.h"

namespace embree
{
  /*! Oriented bounding box */
  template<typename T>
    struct OBBox 
  {
  public:
    
    __forceinline OBBox () {}
    
    __forceinline OBBox (EmptyTy) 
      : space(one), bounds(empty) {}
    
    __forceinline OBBox (const BBox<T>& bounds) 
      : space(one), bounds(bounds) {}
      
    __forceinline OBBox (const LinearSpace3<T>& space, const BBox<T>& bounds) 
      : space(space), bounds(bounds) {}
    
    friend std::ostream& operator<<(std::ostream& cout, const OBBox& p) {
      return std::cout << "{ space = " << p.space << ", bounds = " << p.bounds << "}";
    }
    
  public:
    LinearSpace3<T> space; //!< orthonormal transformation
    BBox<T> bounds;        //!< bounds in transformed space
  };

  typedef OBBox<Vec3f> OBBox3f;
  typedef OBBox<Vec3fa> OBBox3fa;
}
