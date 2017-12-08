// Copyright NVIDIA Corporation 2007 -- Denis Kovacs <den.kovacs@gmail.com>

#pragma once
#ifndef NV_MESH_CLIPPEDTRIANGLE_H
#define NV_MESH_CLIPPEDTRIANGLE_H

#include <nvmath/Vector.h>

namespace nv
{

    class ClippedTriangle
    {
    public:
        ClippedTriangle(Vector2::Arg a, Vector2::Arg b, Vector2::Arg c) 
        {
            m_numVertices = 3;
            m_activeVertexBuffer = 0;

            m_verticesA[0]=a;
            m_verticesA[1]=b;
            m_verticesA[2]=c;

            m_vertexBuffers[0] = m_verticesA;
            m_vertexBuffers[1] = m_verticesB;
        }

        uint vertexCount()
        {
            return m_numVertices;
        }

        const Vector2 * vertices()
        {
            return m_vertexBuffers[m_activeVertexBuffer];
        }

        inline void clipHorizontalPlane(float offset, float clipdirection) 
        {
            Vector2 * v  = m_vertexBuffers[m_activeVertexBuffer];
            m_activeVertexBuffer ^= 1;
            Vector2 * v2 = m_vertexBuffers[m_activeVertexBuffer];

            v[m_numVertices] = v[0];

            float dy2,   dy1 = offset - v[0].y;
            int   dy2in, dy1in = clipdirection*dy1 >= 0;
            uint  p=0;

            for (uint k=0; k<m_numVertices; k++)
            {
                dy2   = offset - v[k+1].y;
                dy2in = clipdirection*dy2 >= 0;

                if (dy1in) v2[p++] = v[k];

                if ( dy1in + dy2in == 1 ) // not both in/out
                {
                    float dx = v[k+1].x - v[k].x;
                    float dy = v[k+1].y - v[k].y;
                    v2[p++] = Vector2(v[k].x + dy1*(dx/dy), offset);
                }

                dy1 = dy2; dy1in = dy2in;
            }
            m_numVertices = p;

            //for (uint k=0; k<m_numVertices; k++) printf("(%f, %f)\n", v2[k].x, v2[k].y); printf("\n");
        }

        inline void clipVerticalPlane(float offset, float clipdirection ) 
        {
            Vector2 * v  = m_vertexBuffers[m_activeVertexBuffer];
            m_activeVertexBuffer ^= 1;
            Vector2 * v2 = m_vertexBuffers[m_activeVertexBuffer];

            v[m_numVertices] = v[0];

            float dx2,   dx1   = offset - v[0].x;
            int   dx2in, dx1in = clipdirection*dx1 >= 0;
            uint  p=0;

            for (uint k=0; k<m_numVertices; k++)
            {
                dx2 = offset - v[k+1].x;
                dx2in = clipdirection*dx2 >= 0;

                if (dx1in) v2[p++] = v[k];

                if ( dx1in + dx2in == 1 ) // not both in/out
                {
                    float dx = v[k+1].x - v[k].x;
                    float dy = v[k+1].y - v[k].y;
                    v2[p++] = Vector2(offset, v[k].y + dx1*(dy/dx));
                }

                dx1 = dx2; dx1in = dx2in;
            }
            m_numVertices = p;

            //for (uint k=0; k<m_numVertices; k++) printf("(%f, %f)\n", v2[k].x, v2[k].y); printf("\n");
        }

        void computeAreaCentroid()
        {
            Vector2 * v  = m_vertexBuffers[m_activeVertexBuffer];
            v[m_numVertices] = v[0];

            m_area = 0;
            float centroidx=0, centroidy=0;
            for (uint k=0; k<m_numVertices; k++)
            {
                // http://local.wasp.uwa.edu.au/~pbourke/geometry/polyarea/
                float f = v[k].x*v[k+1].y - v[k+1].x*v[k].y;
                m_area += f;
                centroidx += f * (v[k].x + v[k+1].x);
                centroidy += f * (v[k].y + v[k+1].y);
            }
            m_area = 0.5f * fabs(m_area);
            if (m_area==0) {
                m_centroid = Vector2(0.0f);
            } else {
                m_centroid = Vector2(centroidx/(6*m_area), centroidy/(6*m_area));
            }
        }

        void clipAABox(float x0, float y0, float x1, float y1)
        {
            clipVerticalPlane  ( x0, -1);
            clipHorizontalPlane( y0, -1);
            clipVerticalPlane  ( x1,  1);
            clipHorizontalPlane( y1,  1);

            computeAreaCentroid();
        }

        Vector2 centroid()
        {
            return m_centroid;
        }

        float area()
        {
            return m_area;
        }

    private:
        Vector2 m_verticesA[7+1];
        Vector2 m_verticesB[7+1];
        Vector2 * m_vertexBuffers[2];
        uint    m_numVertices;
        uint    m_activeVertexBuffer;
        float   m_area;
        Vector2 m_centroid;
    };

} // nv namespace

#endif // NV_MESH_CLIPPEDTRIANGLE_H
