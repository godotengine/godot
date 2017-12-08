// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "nvmesh.h" // pch

#include "Bounds.h"

#include "nvmesh/BaseMesh.h"
#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Vertex.h"

#include "nvmath/Box.inl"

using namespace nv;

Box MeshBounds::box(const BaseMesh * mesh)
{
    nvCheck(mesh != NULL);

    Box bounds;
    bounds.clearBounds();

    const uint vertexCount = mesh->vertexCount();
    for (uint v = 0; v < vertexCount; v++)
    {
        const BaseMesh::Vertex & vertex = mesh->vertexAt(v);
        bounds.addPointToBounds( vertex.pos );
    }

    return bounds;
}

Box MeshBounds::box(const HalfEdge::Mesh * mesh)
{
    nvCheck(mesh != NULL);

    Box bounds;
    bounds.clearBounds();

    const uint vertexCount = mesh->vertexCount();
    for (uint v = 0; v < vertexCount; v++)
    {
        const HalfEdge::Vertex * vertex = mesh->vertexAt(v);
        nvDebugCheck(vertex != NULL);
        bounds.addPointToBounds( vertex->pos );
    }

    return bounds;
}

/*Sphere MeshBounds::sphere(const HalfEdge::Mesh * mesh)
{
    // @@ TODO
    return Sphere();
}*/
