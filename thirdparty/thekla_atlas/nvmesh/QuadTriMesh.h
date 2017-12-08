// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MESH_QUADTRIMESH_H
#define NV_MESH_QUADTRIMESH_H

#include "nvcore/Array.h"
#include "nvmath/Vector.h"
#include "nvmesh/nvmesh.h"
#include "nvmesh/BaseMesh.h"

namespace nv
{
    class Stream;

    /// Mixed quad/triangle mesh.
    class QuadTriMesh : public BaseMesh
    {
    public:
        struct Face;
        typedef BaseMesh::Vertex Vertex;

        QuadTriMesh() {};
        QuadTriMesh(uint faceCount, uint vertexCount) : BaseMesh(vertexCount), m_faceArray(faceCount) {}

        // Face methods.
        uint faceCount() const { return m_faceArray.count(); }

        const Face & faceAt(uint i) const { return m_faceArray[i]; }
        Face & faceAt(uint i) { return m_faceArray[i]; }

        const Array<Face> & faces() const { return m_faceArray; }
        Array<Face> & faces() { return m_faceArray; }

        bool isQuadFace(uint i) const;

        const Vertex & faceVertex(uint f, uint v) const;

        friend Stream & operator<< (Stream & s, QuadTriMesh & obj);

    private:

        Array<Face> m_faceArray;

    };


    /// QuadTriMesh face.
    struct QuadTriMesh::Face
    {
        uint id;
        uint v[4];

        bool isQuadFace() const { return v[3] != NIL; }
    };

} // nv namespace


#endif // NV_MESH_QUADTRIMESH_H
