// This code is in the public domain -- castanyo@yahoo.es

#ifndef NV_MESH_SNAP_H
#define NV_MESH_SNAP_H

#include <nvmesh/nvmesh.h>
#include <nvmath/nvmath.h>

namespace nv
{
	class TriMesh;

	NVMESH_API uint SnapVertices(TriMesh * mesh, float posThreshold=NV_EPSILON, float texThreshold=1.0f/1024, float norThreshold=NV_NORMAL_EPSILON);

} // nv namespace


#endif // NV_MESH_SNAP_H
