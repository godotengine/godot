// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MESH_BASEMESH_H
#define NV_MESH_BASEMESH_H

#include "nvmesh.h"
#include "nvmath/Vector.h"
#include "nvcore/Array.h"
#include "nvcore/Hash.h"

namespace nv
{

    /// Base mesh without connectivity.
    class BaseMesh
    {
    public:
        struct Vertex;

        BaseMesh() {}

        BaseMesh(uint vertexNum) :
            m_vertexArray(vertexNum) {}

        // Vertex methods.
        uint vertexCount() const { return m_vertexArray.count(); }
        const Vertex & vertexAt(uint i) const { return m_vertexArray[i]; }
        Vertex & vertexAt(uint i) { return m_vertexArray[i]; }
        const Array<Vertex> & vertices() const { return m_vertexArray; }
        Array<Vertex> & vertices() { return m_vertexArray; }

        friend Stream & operator<< (Stream & s, BaseMesh & obj);

    protected:

        Array<Vertex> m_vertexArray;
    };


    /// BaseMesh vertex.
    struct BaseMesh::Vertex
    {
        Vertex() : id(NIL), pos(0.0f), nor(0.0f), tex(0.0f) {}

        uint id;		// @@ Vertex should be an index into the vertex data.
        Vector3 pos;
        Vector3 nor;
        Vector2 tex;
    };

    inline bool operator==(const BaseMesh::Vertex & a, const BaseMesh::Vertex & b)
    {
        return a.pos == b.pos && a.nor == b.nor && a.tex == b.tex;
    }

    inline bool operator!=(const BaseMesh::Vertex & a, const BaseMesh::Vertex & b)
    {
        return a.pos != b.pos && a.nor != b.nor && a.tex != b.tex;
    }

    template <> struct Hash<BaseMesh::Vertex>
    {
        uint operator()(const BaseMesh::Vertex & v) const
        {
            return Hash<Vector3>()(v.pos);
        }
    };

} // nv namespace

#endif // NV_MESH_BASEMESH_H
