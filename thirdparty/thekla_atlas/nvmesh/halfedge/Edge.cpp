// This code is in the public domain -- castanyo@yahoo.es

#include "nvmesh.h" // pch

#include "Edge.h"
#include "Vertex.h"

#include "nvmath/Vector.inl"

using namespace nv;
using namespace HalfEdge;

Vector3 Edge::midPoint() const
{
    return (to()->pos + from()->pos) * 0.5f;
}

float Edge::length() const
{
    return ::length(to()->pos - from()->pos); 
}

// Return angle between this edge and the previous one.
float Edge::angle() const {
    Vector3 p = vertex->pos;
    Vector3 a = prev->vertex->pos;
    Vector3 b = next->vertex->pos;

    Vector3 v0 = a - p;
    Vector3 v1 = b - p;

    return acosf(dot(v0, v1) / (nv::length(v0) * nv::length(v1)));
}

bool Edge::isValid() const
{
    // null face is OK.
    if (next == NULL || prev == NULL || pair == NULL || vertex == NULL) return false;
    if (next->prev != this) return false;
    if (prev->next != this) return false;
    if (pair->pair != this) return false;
    return true;
}

/*
Edge * Edge::nextBoundary() {
    nvDebugCheck(this->m_pair == NULL);

}

Edge * Edge::prevBoundary() {
    nvDebugCheck(this->m_pair == NULL);

}
*/


