// This code is in the public domain -- castano@gmail.com

#include "nvmesh.h" // pch

#include "Vertex.h"

#include "nvmath/Vector.inl"

using namespace nv;
using namespace HalfEdge;


// Set first edge of all colocals.
void Vertex::setEdge(Edge * e)
{
    for (VertexIterator it(colocals()); !it.isDone(); it.advance()) { 
        it.current()->edge = e;
    }
}

// Update position of all colocals.
void Vertex::setPos(const Vector3 & p)
{
    for (VertexIterator it(colocals()); !it.isDone(); it.advance()) {
        it.current()->pos = p;
    }
}


uint HalfEdge::Vertex::colocalCount() const
{
    uint count = 0;
    for (ConstVertexIterator it(colocals()); !it.isDone(); it.advance()) { ++count; }
    return count;
}

uint HalfEdge::Vertex::valence() const
{
    uint count = 0;
    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance()) { ++count; }
    return count;
}

const HalfEdge::Vertex * HalfEdge::Vertex::firstColocal() const
{
    uint firstId = id;
    const Vertex * vertex = this;

    for (ConstVertexIterator it(colocals()); !it.isDone(); it.advance())
    {
        if (it.current()->id < firstId) {
            firstId = vertex->id;
            vertex = it.current();
        }
    }

    return vertex;
}

HalfEdge::Vertex * HalfEdge::Vertex::firstColocal()
{
    Vertex * vertex = this;
    uint firstId = id;

    for (VertexIterator it(colocals()); !it.isDone(); it.advance())
    {
        if (it.current()->id < firstId) {
            firstId = vertex->id;
            vertex = it.current();
        }
    }

    return vertex;
}

bool HalfEdge::Vertex::isFirstColocal() const
{
    return firstColocal() == this;
}

bool HalfEdge::Vertex::isColocal(const Vertex * v) const {
    if (this == v) return true;
    if (pos != v->pos) return false;

    for (ConstVertexIterator it(colocals()); !it.isDone(); it.advance())
    {
        if (v == it.current()) {
            return true;
        }
    }

    return false;
}

