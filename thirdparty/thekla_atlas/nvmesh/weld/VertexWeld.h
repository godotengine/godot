// Copyright NVIDIA Corporation 2006 -- Ignacio Castano <icastano@nvidia.com>

#ifndef NV_MESH_VERTEXWELD_H
#define NV_MESH_VERTEXWELD_H

#include <nvmesh/nvmesh.h>

namespace nv
{
	class TriMesh;
	class QuadMesh;

	NVMESH_API void WeldVertices(TriMesh * mesh);
	NVMESH_API void WeldVertices(QuadTriMesh * mesh);

} // nv namespace


#endif // NV_MESH_VERTEXWELD_H
