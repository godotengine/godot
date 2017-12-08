// This code is in the public domain -- castano@gmail.com

#include "nvmesh.h" // pch

#include "OrthogonalProjectionMap.h"

#include "nvcore/Array.inl"

#include "nvmath/Fitting.h"
#include "nvmath/Vector.inl"
#include "nvmath/Box.inl"
#include "nvmath/Plane.inl"

#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Vertex.h"
#include "nvmesh/halfedge/Face.h"
#include "nvmesh/geometry/Bounds.h"


using namespace nv;

bool nv::computeOrthogonalProjectionMap(HalfEdge::Mesh * mesh)
{
    Vector3 axis[2];

#if 1

    uint vertexCount = mesh->vertexCount();
    Array<Vector3> points(vertexCount);
    points.resize(vertexCount);

    for (uint i = 0; i < vertexCount; i++)
    {
        points[i] = mesh->vertexAt(i)->pos;
    }

#if 0
    axis[0] = Fit::computePrincipalComponent_EigenSolver(vertexCount, points.buffer());
    axis[0] = normalize(axis[0]);

    Plane plane = Fit::bestPlane(vertexCount, points.buffer());

    Vector3 n = plane.vector();

    axis[1] = cross(axis[0], n);
    axis[1] = normalize(axis[1]);
#else
    // Avoid redundant computations.
    float matrix[6];
    Fit::computeCovariance(vertexCount, points.buffer(), matrix);

    if (matrix[0] == 0 && matrix[3] == 0 && matrix[5] == 0) {
        return false;
    }

    float eigenValues[3];
    Vector3 eigenVectors[3];
    if (!nv::Fit::eigenSolveSymmetric3(matrix, eigenValues, eigenVectors)) {
        return false;
    }

    axis[0] = normalize(eigenVectors[0]);
    axis[1] = normalize(eigenVectors[1]);
#endif


#else

    // IC: I thought this was generally more robust, but turns out it's not even guaranteed to return a valid projection. Imagine a narrow quad perpendicular to one plane, but rotated so that the shortest axis of 
    // the bounding box is in the direction of that plane.

    // Use the shortest box axis
    Box box = MeshBounds::box(mesh);
    Vector3 dir = box.extents();

    if (fabs(dir.x) <= fabs(dir.y) && fabs(dir.x) <= fabs(dir.z)) {
        axis[0] = Vector3(0, 1, 0); 
        axis[1] = Vector3(0, 0, 1);
    }
    else if (fabs(dir.y) <= fabs(dir.z)) {
        axis[0] = Vector3(1, 0, 0); 
        axis[1] = Vector3(0, 0, 1);
    }
    else {
        axis[0] = Vector3(1, 0, 0); 
        axis[1] = Vector3(0, 1, 0);
    }
#endif

    // Project vertices to plane.
    for (HalfEdge::Mesh::VertexIterator it(mesh->vertices()); !it.isDone(); it.advance())
    {
        HalfEdge::Vertex * vertex = it.current();
        vertex->tex.x = dot(axis[0], vertex->pos);
        vertex->tex.y = dot(axis[1], vertex->pos);
    }

    return true;
}
