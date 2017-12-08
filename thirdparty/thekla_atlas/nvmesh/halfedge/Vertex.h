// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MESH_HALFEDGE_VERTEX_H
#define NV_MESH_HALFEDGE_VERTEX_H

#include "nvmesh/halfedge/Edge.h"

namespace nv
{
    namespace HalfEdge { class Vertex; class Face; class Edge; }

    // Half edge vertex.
    class HalfEdge::Vertex
    {
        NV_FORBID_COPY(Vertex);
    public:

        uint id;

        Edge * edge;
        Vertex * next;
        Vertex * prev;

        Vector3 pos;
        Vector3 nor;
        Vector2 tex;
        Vector4 col;


        Vertex(uint id) : id(id), edge(NULL), pos(0.0f), nor(0.0f), tex(0.0f), col(0.0f) {
            next = this;
            prev = this;
        }


        void setEdge(Edge * e);
        void setPos(const Vector3 & p);

        uint colocalCount() const;
        uint valence() const;
        bool isFirstColocal() const;
        const Vertex * firstColocal() const;
        Vertex * firstColocal();

        bool isColocal(const Vertex * v) const;

        
        void linkColocal(Vertex * v) {
            next->prev = v;
            v->next = next; 
            next = v;
            v->prev = this;
        }
        void unlinkColocal() {
            next->prev = prev;
            prev->next = next;
            next = this;
            prev = this;
        }


        // @@ Note: This only works if linkBoundary has been called.
        bool isBoundary() const {
            return (edge && !edge->face);
        }


        //	for(EdgeIterator it(iterator()); !it.isDone(); it.advance()) { ... }
        //
        //	EdgeIterator it(iterator());
        //	while(!it.isDone()) {
        //		...
        //		id.advance(); 
        //	}

        // Iterator that visits the edges around this vertex in counterclockwise order.
        class EdgeIterator //: public Iterator<Edge *>
        {
        public:
            EdgeIterator(Edge * e) : m_end(NULL), m_current(e) { }

            virtual void advance()
            {
                if (m_end == NULL) m_end = m_current;
                m_current = m_current->pair->next;
                //m_current = m_current->prev->pair;
            }

            virtual bool isDone() const { return m_end == m_current; }
            virtual Edge * current() const { return m_current; }
            Vertex * vertex() const { return m_current->vertex; }

        private:
            Edge * m_end;
            Edge * m_current;
        };

        EdgeIterator edges() { return EdgeIterator(edge); }
        EdgeIterator edges(Edge * e) { return EdgeIterator(e); }

        // Iterator that visits the edges around this vertex in counterclockwise order.
        class ConstEdgeIterator //: public Iterator<Edge *>
        {
        public:
            ConstEdgeIterator(const Edge * e) : m_end(NULL), m_current(e) { }
            ConstEdgeIterator(EdgeIterator it) : m_end(NULL), m_current(it.current()) { }

            virtual void advance()
            {
                if (m_end == NULL) m_end = m_current;
                m_current = m_current->pair->next;
                //m_current = m_current->prev->pair;
            }

            virtual bool isDone() const { return m_end == m_current; }
            virtual const Edge * current() const { return m_current; }
            const Vertex * vertex() const { return m_current->to(); }

        private:
            const Edge * m_end;
            const Edge * m_current;
        };

        ConstEdgeIterator edges() const { return ConstEdgeIterator(edge); }
        ConstEdgeIterator edges(const Edge * e) const { return ConstEdgeIterator(e); }


        // Iterator that visits the edges around this vertex in counterclockwise order.
        class ReverseEdgeIterator //: public Iterator<Edge *>
        {
        public:
            ReverseEdgeIterator(Edge * e) : m_end(NULL), m_current(e) { }

            virtual void advance()
            {
                if (m_end == NULL) m_end = m_current;
                m_current = m_current->prev->pair;
            }

            virtual bool isDone() const { return m_end == m_current; }
            virtual Edge * current() const { return m_current; }
            Vertex * vertex() const { return m_current->vertex; }

        private:
            Edge * m_end;
            Edge * m_current;
        };

        // Iterator that visits the edges around this vertex in counterclockwise order.
        class ReverseConstEdgeIterator //: public Iterator<Edge *>
        {
        public:
            ReverseConstEdgeIterator(const Edge * e) : m_end(NULL), m_current(e) { }

            virtual void advance()
            {
                if (m_end == NULL) m_end = m_current;
                m_current = m_current->prev->pair;
            }

            virtual bool isDone() const { return m_end == m_current; }
            virtual const Edge * current() const { return m_current; }
            const Vertex * vertex() const { return m_current->to(); }

        private:
            const Edge * m_end;
            const Edge * m_current;
        };



        // Iterator that visits all the colocal vertices.
        class VertexIterator //: public Iterator<Edge *>
        {
        public:
            VertexIterator(Vertex * v) : m_end(NULL), m_current(v) { }

            virtual void advance()
            {
                if (m_end == NULL) m_end = m_current;
                m_current = m_current->next;
            }

            virtual bool isDone() const { return m_end == m_current; }
            virtual Vertex * current() const { return m_current; }

        private:
            Vertex * m_end;
            Vertex * m_current;
        };

        VertexIterator colocals() { return VertexIterator(this); }

        // Iterator that visits all the colocal vertices.
        class ConstVertexIterator //: public Iterator<Edge *>
        {
        public:
            ConstVertexIterator(const Vertex * v) : m_end(NULL), m_current(v) { }

            virtual void advance()
            {
                if (m_end == NULL) m_end = m_current;
                m_current = m_current->next;
            }

            virtual bool isDone() const { return m_end == m_current; }
            virtual const Vertex * current() const { return m_current; }

        private:
            const Vertex * m_end;
            const Vertex * m_current;
        };

        ConstVertexIterator colocals() const { return ConstVertexIterator(this); }

    };

} // nv namespace

#endif // NV_MESH_HALFEDGE_VERTEX_H
