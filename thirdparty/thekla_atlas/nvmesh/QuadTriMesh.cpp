// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "QuadTriMesh.h"
#include "Stream.h"

using namespace nv;


bool QuadTriMesh::isQuadFace(uint i) const 
{ 
    return m_faceArray[i].isQuadFace();
}

const QuadTriMesh::Vertex & QuadTriMesh::faceVertex(uint f, uint v) const 
{
    if (isQuadFace(f)) nvDebugCheck(v < 4);
    else nvDebugCheck(v < 3);

    const Face & face = this->faceAt(f);
    return this->vertexAt(face.v[v]);
}


namespace nv
{
    static Stream & operator<< (Stream & s, QuadTriMesh::Face & face)
    {
        return s << face.id << face.v[0] << face.v[1] << face.v[2] << face.v[3];
    }

    Stream & operator<< (Stream & s, QuadTriMesh & mesh)
    {
        return s << mesh.m_faceArray << (BaseMesh &) mesh;
    }
}

