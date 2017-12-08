// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MESH_MESHBOUNDS_H
#define NV_MESH_MESHBOUNDS_H

#include <nvmath/Sphere.h>
#include <nvmath/Box.h>

#include <nvmesh/nvmesh.h>

namespace nv
{
    class BaseMesh;
    namespace HalfEdge { class Mesh; }

    // Bounding volumes computation.
    namespace MeshBounds
    {
        Box box(const BaseMesh * mesh);
        Box box(const HalfEdge::Mesh * mesh);

        Sphere sphere(const HalfEdge::Mesh * mesh);
    }

} // nv namespace

#endif // NV_MESH_MESHBOUNDS_H
