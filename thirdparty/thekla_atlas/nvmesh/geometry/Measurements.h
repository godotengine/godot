// This code is in the public domain -- castano@gmail.com

#pragma once
#ifndef NV_MESH_MESHMEASUREMENTS_H
#define NV_MESH_MESHMEASUREMENTS_H

#include "nvmesh/nvmesh.h"

namespace nv
{
    namespace HalfEdge { class Mesh; }

	float computeSurfaceArea(const HalfEdge::Mesh * mesh);
	float computeParametricArea(const HalfEdge::Mesh * mesh);

} // nv namespace

#endif // NV_MESH_MESHMEASUREMENTS_H
