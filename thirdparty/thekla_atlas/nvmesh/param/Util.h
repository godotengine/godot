// This code is in the public domain -- castano@gmail.com

#include "nvmesh/nvmesh.h"

namespace nv {

    namespace HalfEdge { class Mesh; class Vertex; }

    bool isQuadMesh(const HalfEdge::Mesh * mesh);
    bool isTriangularMesh(const HalfEdge::Mesh * mesh);

    uint countMeshTriangles(const HalfEdge::Mesh * mesh);
    const HalfEdge::Vertex * findBoundaryVertex(const HalfEdge::Mesh * mesh);

    HalfEdge::Mesh * unifyVertices(const HalfEdge::Mesh * inputMesh);
    HalfEdge::Mesh * triangulate(const HalfEdge::Mesh * inputMesh);

} // nv namespace
