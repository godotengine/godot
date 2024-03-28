// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scene.h"

#include "../../common/algorithms/parallel_any_of.h"

namespace embree
{

void Scene::checkIfModifiedAndSet ()
{
  if (isModified ()) return;

  auto geometryIsModified = [this](size_t geomID)->bool {
    return isGeometryModified(geomID);
  };

  if (parallel_any_of (size_t(0), geometries.size (), geometryIsModified)) {
    setModified ();
  }
}

}