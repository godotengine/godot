// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "TriMesh.h"

using namespace nv;


/// Triangle mesh.
Vector3 TriMesh::faceNormal(uint f) const
{
    const Face & face = this->faceAt(f);
    const Vector3 & p0 = this->vertexAt(face.v[0]).pos;
    const Vector3 & p1 = this->vertexAt(face.v[1]).pos;
    const Vector3 & p2 = this->vertexAt(face.v[2]).pos;
    return normalizeSafe(cross(p1 - p0, p2 - p0), Vector3(0.0f), 0.0f);
}

/// Get face vertex.
const TriMesh::Vertex & TriMesh::faceVertex(uint f, uint v) const
{
    nvDebugCheck(v < 3);
    const Face & face = this->faceAt(f);
    return this->vertexAt(face.v[v]);
}

