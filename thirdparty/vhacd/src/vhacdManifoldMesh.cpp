/* Copyright (c) 2011 Khaled Mamou (kmamou at gmail dot com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "vhacdManifoldMesh.h"
namespace VHACD {
TMMVertex::TMMVertex(void)
{
    Initialize();
}
void TMMVertex::Initialize()
{
    m_name = 0;
    m_id = 0;
    m_duplicate = 0;
    m_onHull = false;
    m_tag = false;
}

TMMVertex::~TMMVertex(void)
{
}
TMMEdge::TMMEdge(void)
{
    Initialize();
}
void TMMEdge::Initialize()
{
    m_id = 0;
    m_triangles[0] = m_triangles[1] = m_newFace = 0;
    m_vertices[0] = m_vertices[1] = 0;
}
TMMEdge::~TMMEdge(void)
{
}
void TMMTriangle::Initialize()
{
    m_id = 0;
    for (int32_t i = 0; i < 3; i++) {
        m_edges[i] = 0;
        m_vertices[0] = 0;
    }
    m_visible = false;
}
TMMTriangle::TMMTriangle(void)
{
    Initialize();
}
TMMTriangle::~TMMTriangle(void)
{
}
TMMesh::TMMesh()
{
}
TMMesh::~TMMesh(void)
{
}
void TMMesh::GetIFS(Vec3<double>* const points, Vec3<int32_t>* const triangles)
{
    size_t nV = m_vertices.GetSize();
    size_t nT = m_triangles.GetSize();

    for (size_t v = 0; v < nV; v++) {
        points[v] = m_vertices.GetData().m_pos;
        m_vertices.GetData().m_id = v;
        m_vertices.Next();
    }
    for (size_t f = 0; f < nT; f++) {
        TMMTriangle& currentTriangle = m_triangles.GetData();
        triangles[f].X() = static_cast<int32_t>(currentTriangle.m_vertices[0]->GetData().m_id);
        triangles[f].Y() = static_cast<int32_t>(currentTriangle.m_vertices[1]->GetData().m_id);
        triangles[f].Z() = static_cast<int32_t>(currentTriangle.m_vertices[2]->GetData().m_id);
        m_triangles.Next();
    }
}
void TMMesh::Clear()
{
    m_vertices.Clear();
    m_edges.Clear();
    m_triangles.Clear();
}
void TMMesh::Copy(TMMesh& mesh)
{
    Clear();
    // updating the id's
    size_t nV = mesh.m_vertices.GetSize();
    size_t nE = mesh.m_edges.GetSize();
    size_t nT = mesh.m_triangles.GetSize();
    for (size_t v = 0; v < nV; v++) {
        mesh.m_vertices.GetData().m_id = v;
        mesh.m_vertices.Next();
    }
    for (size_t e = 0; e < nE; e++) {
        mesh.m_edges.GetData().m_id = e;
        mesh.m_edges.Next();
    }
    for (size_t f = 0; f < nT; f++) {
        mesh.m_triangles.GetData().m_id = f;
        mesh.m_triangles.Next();
    }
    // copying data
    m_vertices = mesh.m_vertices;
    m_edges = mesh.m_edges;
    m_triangles = mesh.m_triangles;

    // generate mapping
    CircularListElement<TMMVertex>** vertexMap = new CircularListElement<TMMVertex>*[nV];
    CircularListElement<TMMEdge>** edgeMap = new CircularListElement<TMMEdge>*[nE];
    CircularListElement<TMMTriangle>** triangleMap = new CircularListElement<TMMTriangle>*[nT];
    for (size_t v = 0; v < nV; v++) {
        vertexMap[v] = m_vertices.GetHead();
        m_vertices.Next();
    }
    for (size_t e = 0; e < nE; e++) {
        edgeMap[e] = m_edges.GetHead();
        m_edges.Next();
    }
    for (size_t f = 0; f < nT; f++) {
        triangleMap[f] = m_triangles.GetHead();
        m_triangles.Next();
    }

    // updating pointers
    for (size_t v = 0; v < nV; v++) {
        if (vertexMap[v]->GetData().m_duplicate) {
            vertexMap[v]->GetData().m_duplicate = edgeMap[vertexMap[v]->GetData().m_duplicate->GetData().m_id];
        }
    }
    for (size_t e = 0; e < nE; e++) {
        if (edgeMap[e]->GetData().m_newFace) {
            edgeMap[e]->GetData().m_newFace = triangleMap[edgeMap[e]->GetData().m_newFace->GetData().m_id];
        }
        if (nT > 0) {
            for (int32_t f = 0; f < 2; f++) {
                if (edgeMap[e]->GetData().m_triangles[f]) {
                    edgeMap[e]->GetData().m_triangles[f] = triangleMap[edgeMap[e]->GetData().m_triangles[f]->GetData().m_id];
                }
            }
        }
        for (int32_t v = 0; v < 2; v++) {
            if (edgeMap[e]->GetData().m_vertices[v]) {
                edgeMap[e]->GetData().m_vertices[v] = vertexMap[edgeMap[e]->GetData().m_vertices[v]->GetData().m_id];
            }
        }
    }
    for (size_t f = 0; f < nT; f++) {
        if (nE > 0) {
            for (int32_t e = 0; e < 3; e++) {
                if (triangleMap[f]->GetData().m_edges[e]) {
                    triangleMap[f]->GetData().m_edges[e] = edgeMap[triangleMap[f]->GetData().m_edges[e]->GetData().m_id];
                }
            }
        }
        for (int32_t v = 0; v < 3; v++) {
            if (triangleMap[f]->GetData().m_vertices[v]) {
                triangleMap[f]->GetData().m_vertices[v] = vertexMap[triangleMap[f]->GetData().m_vertices[v]->GetData().m_id];
            }
        }
    }
    delete[] vertexMap;
    delete[] edgeMap;
    delete[] triangleMap;
}
bool TMMesh::CheckConsistancy()
{
    size_t nE = m_edges.GetSize();
    size_t nT = m_triangles.GetSize();
    for (size_t e = 0; e < nE; e++) {
        for (int32_t f = 0; f < 2; f++) {
            if (!m_edges.GetHead()->GetData().m_triangles[f]) {
                return false;
            }
        }
        m_edges.Next();
    }
    for (size_t f = 0; f < nT; f++) {
        for (int32_t e = 0; e < 3; e++) {
            int32_t found = 0;
            for (int32_t k = 0; k < 2; k++) {
                if (m_triangles.GetHead()->GetData().m_edges[e]->GetData().m_triangles[k] == m_triangles.GetHead()) {
                    found++;
                }
            }
            if (found != 1) {
                return false;
            }
        }
        m_triangles.Next();
    }
    return true;
}
}