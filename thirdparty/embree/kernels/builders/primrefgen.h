// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/scene.h"
#include "../common/primref.h"
#include "../common/primref_mb.h"
#include "priminfo.h"
#include "bvh_builder_morton.h"

namespace embree
{ 
  namespace isa
  {
    PrimInfo createPrimRefArray(Geometry* geometry, unsigned int geomID, size_t numPrimitives, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor);
   
    PrimInfo createPrimRefArray(Scene* scene, Geometry::GTypeMask types, bool mblur, size_t numPrimitives, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor);
   
    PrimInfo createPrimRefArrayMBlur(Scene* scene, Geometry::GTypeMask types, size_t numPrimitives, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor, size_t itime = 0);

    PrimInfoMB createPrimRefArrayMSMBlur(Scene* scene, Geometry::GTypeMask types, size_t numPrimitives, mvector<PrimRefMB>& prims, BuildProgressMonitor& progressMonitor, BBox1f t0t1 = BBox1f(0.0f,1.0f));

    template<typename Mesh>
      size_t createMortonCodeArray(Mesh* mesh, mvector<BVHBuilderMorton::BuildPrim>& morton, BuildProgressMonitor& progressMonitor);

    /* special variants for grids */
    PrimInfo createPrimRefArrayGrids(Scene* scene, mvector<PrimRef>& prims, mvector<SubGridBuildData>& sgrids);

    PrimInfo createPrimRefArrayGrids(GridMesh* mesh, mvector<PrimRef>& prims, mvector<SubGridBuildData>& sgrids);
    
  }
}

