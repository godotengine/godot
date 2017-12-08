// This code is in the public domain -- castano@gmail.com

#include "nvmesh.h" // pch

#include "Measurements.h"
#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Face.h"

using namespace nv;

float nv::computeSurfaceArea(const HalfEdge::Mesh * mesh)
{
    float area = 0;

    for (HalfEdge::Mesh::ConstFaceIterator it(mesh->faces()); !it.isDone(); it.advance())
    {
        const HalfEdge::Face * face = it.current();
        area += face->area();
    }
    nvDebugCheck(area >= 0);

    return area;
}

float nv::computeParametricArea(const HalfEdge::Mesh * mesh)
{
    float area = 0;

    for (HalfEdge::Mesh::ConstFaceIterator it(mesh->faces()); !it.isDone(); it.advance())
    {
        const HalfEdge::Face * face = it.current();
        area += face->parametricArea();
    }

    return area;
}
