// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
