// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
    
    friend embree_ostream operator<<(embree_ostream cout, const OBBox& p) {
      return cout << "{ space = " << p.space << ", bounds = " << p.bounds << "}";
    }
    
  public:
    LinearSpace3<T> space; //!< orthonormal transformation
    BBox<T> bounds;        //!< bounds in transformed space
  };

  typedef OBBox<Vec3f> OBBox3f;
  typedef OBBox<Vec3fa> OBBox3fa;
}
