// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MESH_HALFEDGE_EDGE_H
#define NV_MESH_HALFEDGE_EDGE_H

#include "nvmath/Vector.h"

namespace nv
{
    namespace HalfEdge { class Vertex; class Face; class Edge; }

    /// Half edge edge. 
    class HalfEdge::Edge
    {
        NV_FORBID_COPY(Edge);
    public:

        uint id;

        Edge * next;
        Edge * prev;	// This is not strictly half-edge, but makes algorithms easier and faster.
        Edge * pair;
        Vertex * vertex;
        Face * face;


        // Default constructor.
        Edge(uint id) : id(id), next(NULL), prev(NULL), pair(NULL), vertex(NULL), face(NULL)
        {
        }


        // Vertex queries.
        const Vertex * from() const { return vertex; }
        Vertex * from() { return vertex; }

        const Vertex * to() const { return pair->vertex; }  // This used to be 'next->vertex', but that changed often when the connectivity of the mesh changes.
        Vertex * to() { return pair->vertex; }


        // Edge queries.
        void setNext(Edge * e) { next = e; if (e != NULL) e->prev = this; }
        void setPrev(Edge * e) { prev = e; if (e != NULL) e->next = this; }

        // @@ Add these helpers:
        //Edge * nextBoundary();
        //Edge * prevBoundary();


        // @@ It would be more simple to only check m_pair == NULL
        // Face queries.
        bool isBoundary() const { return !(face && pair->face); }

        // @@ This is not exactly accurate, we should compare the texture coordinates...
        bool isSeam() const { return vertex != pair->next->vertex || next->vertex != pair->vertex; }

        bool isValid() const;

        // Geometric queries.
        Vector3 midPoint() const;
        float length() const;
        float angle() const;

    };

} // nv namespace


#endif // NV_MESH_HALFEDGE_EDGE_H
