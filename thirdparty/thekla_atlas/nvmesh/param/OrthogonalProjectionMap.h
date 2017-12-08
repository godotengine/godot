// This code is in the public domain -- castano@gmail.com

#pragma once
#ifndef NV_MESH_ORTHOGONALPROJECTIONMAP_H
#define NV_MESH_ORTHOGONALPROJECTIONMAP_H

namespace nv
{
    namespace HalfEdge { class Mesh; }

    bool computeOrthogonalProjectionMap(HalfEdge::Mesh * mesh);

} // nv namespace

#endif // NV_MESH_ORTHOGONALPROJECTIONMAP_H
