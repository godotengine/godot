// This code is in the public domain -- castanyo@yahoo.es

#include "nvmesh.h" // pch

#include "Face.h"
#include "Vertex.h"

#include "nvmath/Fitting.h"
#include "nvmath/Plane.h"
#include "nvmath/Vector.inl"

#include "nvcore/Array.h"


using namespace nv;
using namespace HalfEdge;

/// Get face area.
float Face::area() const
{
    float area = 0;
    const Vector3 & v0 = edge->from()->pos;

    for (ConstEdgeIterator it(edges(edge->next)); it.current() != edge->prev; it.advance())
    {
        const Edge * e = it.current();

        const Vector3 & v1 = e->vertex->pos;
        const Vector3 & v2 = e->next->vertex->pos; 

        area += length(cross(v1-v0, v2-v0));
    }

    return area * 0.5f;
}

float Face::parametricArea() const
{
    float area = 0;
    const Vector2 & v0 = edge->from()->tex;

    for (ConstEdgeIterator it(edges(edge->next)); it.current() != edge->prev; it.advance())
    {
        const Edge * e = it.current();

        const Vector2 & v1 = e->vertex->tex;
        const Vector2 & v2 = e->next->vertex->tex;

        area += triangleArea(v0, v1, v2);
    }

    return area * 0.5f;
}


/// Get boundary length.
float Face::boundaryLength() const
{
    float bl = 0;

    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance())
    {
        const Edge * edge = it.current();
        bl += edge->length();
    }

    return bl;
}


/// Get face normal.
Vector3 Face::normal() const
{
    Vector3 n(0);

    const Vertex * vertex0 = NULL;

    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance())
    {
        const Edge * edge = it.current();
        nvCheck(edge != NULL);

        if (vertex0 == NULL)
        {
            vertex0 = edge->vertex;
        }
        else if (edge->next->vertex != vertex0)
        {
            const HalfEdge::Vertex * vertex1 = edge->from();
            const HalfEdge::Vertex * vertex2 = edge->to();

            const Vector3 & p0 = vertex0->pos;
            const Vector3 & p1 = vertex1->pos;
            const Vector3 & p2 = vertex2->pos;

            Vector3 v10 = p1 - p0;
            Vector3 v20 = p2 - p0;

            n += cross(v10, v20);
        }
    }

    return normalizeSafe(n, Vector3(0, 0, 1), 0.0f);


    // Get face points eliminating duplicates.
    /*Array<Vector3> points(4);

    points.append(m_edge->prev()->from()->pos);

    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance())
    {
        const Edge * edge = it.current();
        nvDebugCheck(edge != NULL);

        const Vector3 & p = edge->from()->pos;
        if (points.back() != p)
        {
            points.append(edge->from()->pos);
        }
    }

    points.popBack();

    if (points.count() < 3)
    {
        // Invalid normal.
        return Vector3(0.0f);
    }
    else
    {
        // Compute regular normal.
        Vector3 normal = normalizeSafe(cross(points[1] - points[0], points[2] - points[0]), Vector3(0.0f), 0.0f);

#pragma NV_MESSAGE("TODO: make sure these three points are not colinear")

        if (points.count() > 3)
        {
            // Compute best fitting plane to the points.
            Plane plane = Fit::bestPlane(points.count(), points.buffer());

            // Adjust normal orientation.
            if (dot(normal, plane.vector()) > 0) {
                normal = plane.vector();
            }
            else {
                normal = -plane.vector();
            }
        }

        nvDebugCheck(isNormalized(normal));
        return normal;
    }*/
}

Vector3 Face::centroid() const
{
    Vector3 sum(0.0f);
    uint count = 0;

    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance())
    {
        const Edge * edge = it.current();
        sum += edge->from()->pos;
        count++;
    }

    return sum / float(count);
}


bool Face::isValid() const
{
    uint count = 0;

    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance())
    {
        const Edge * edge = it.current();
        if (edge->face != this) return false;
        if (!edge->isValid()) return false;
        if (!edge->pair->isValid()) return false;
        count++;
    }

    if (count < 3) return false;

    return true;
}


// Determine if this face contains the given edge.
bool Face::contains(const Edge * e) const
{
    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance())
    {
        if(it.current() == e) return true;
    }
    return false;
}

// Returns index in this face of the given edge.
uint Face::edgeIndex(const Edge * e) const
{
    int i = 0;
    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance(), i++)
    {
        if(it.current() == e) return i;
    }
    return NIL;
}


Edge * Face::edgeAt(uint idx)
{ 
    int i = 0;
    for(EdgeIterator it(edges()); !it.isDone(); it.advance(), i++) {
        if (i == idx) return it.current();
    }
    return NULL;
}
const Edge * Face::edgeAt(uint idx) const 
{
    int i = 0;
    for(ConstEdgeIterator it(edges()); !it.isDone(); it.advance(), i++) {
        if (i == idx) return it.current();
    }
    return NULL;
}


// Count the number of edges in this face.
uint Face::edgeCount() const
{
    uint count = 0;
    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance()) { ++count; }
    return count;
}

// Determine if this is a boundary face.
bool Face::isBoundary() const
{
    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance())
    {
        const Edge * edge = it.current();
        nvDebugCheck(edge->pair != NULL);

        if (edge->pair->face == NULL) {
            return true;
        }
    }
    return false;
}

// Count the number of boundary edges in the face.
uint Face::boundaryCount() const
{
    uint count = 0;
    for (ConstEdgeIterator it(edges()); !it.isDone(); it.advance())
    {
        const Edge * edge = it.current();
        nvDebugCheck(edge->pair != NULL);

        if (edge->pair->face == NULL) {
            count++;
        }
    }
    return count;
}
