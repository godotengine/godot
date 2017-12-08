// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MESH_HALFEDGE_FACE_H
#define NV_MESH_HALFEDGE_FACE_H

#include <nvmesh/halfedge/Edge.h>

namespace nv
{
    namespace HalfEdge { class Vertex; class Face; class Edge; }

    /// Face of a half-edge mesh.
    class HalfEdge::Face
    {
        NV_FORBID_COPY(Face);
    public:

        uint id;
        uint16 group;
        uint16 material;
        Edge * edge;


        Face(uint id) : id(id), group(~0), material(~0), edge(NULL) {}

        float area() const;
        float parametricArea() const;
        float boundaryLength() const;
        Vector3 normal() const;
        Vector3 centroid() const;

        bool isValid() const;

        bool contains(const Edge * e) const;
        uint edgeIndex(const Edge * e) const;
        
        Edge * edgeAt(uint idx);
        const Edge * edgeAt(uint idx) const;

        uint edgeCount() const;
        bool isBoundary() const;
        uint boundaryCount() const;


        // The iterator that visits the edges of this face in clockwise order.
        class EdgeIterator //: public Iterator<Edge *>
        {
        public:
            EdgeIterator(Edge * e) : m_end(NULL), m_current(e) { }

            virtual void advance()
            {
                if (m_end == NULL) m_end = m_current;
                m_current = m_current->next;
            }

            virtual bool isDone() const { return m_end == m_current; }
            virtual Edge * current() const { return m_current; }
            Vertex * vertex() const { return m_current->vertex; }

        private:
            Edge * m_end;
            Edge * m_current;
        };

        EdgeIterator edges() { return EdgeIterator(edge); }
        EdgeIterator edges(Edge * e)
        { 
            nvDebugCheck(contains(e));
            return EdgeIterator(e); 
        }

        // The iterator that visits the edges of this face in clockwise order.
        class ConstEdgeIterator //: public Iterator<const Edge *>
        {
        public:
            ConstEdgeIterator(const Edge * e) : m_end(NULL), m_current(e) { }
            ConstEdgeIterator(const EdgeIterator & it) : m_end(NULL), m_current(it.current()) { }

            virtual void advance()
            {
                if (m_end == NULL) m_end = m_current;
                m_current = m_current->next;
            }

            virtual bool isDone() const { return m_end == m_current; }
            virtual const Edge * current() const { return m_current; }
            const Vertex * vertex() const { return m_current->vertex; }

        private:
            const Edge * m_end;
            const Edge * m_current;
        };

        ConstEdgeIterator edges() const { return ConstEdgeIterator(edge); }
        ConstEdgeIterator edges(const Edge * e) const
        { 
            nvDebugCheck(contains(e));
            return ConstEdgeIterator(e); 
        }
    };

} // nv namespace

#endif // NV_MESH_HALFEDGE_FACE_H
