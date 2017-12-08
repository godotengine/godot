// This code is in the public domain -- castano@gmail.com

#include "nvmesh.h" // pch

#include "Util.h"

#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Face.h"
#include "nvmesh/halfedge/Vertex.h"

#include "nvmath/Vector.inl"

#include "nvcore/Array.inl"


using namespace nv;

// Determine if the given mesh is a	quad mesh.
bool nv::isQuadMesh(const HalfEdge::Mesh * mesh)
{
    nvDebugCheck(mesh != NULL);

    const uint faceCount = mesh->faceCount();
    for(uint i = 0; i < faceCount; i++) {
        const HalfEdge::Face * face = mesh->faceAt(i);
        if (face->edgeCount() != 4) {
            return false;
        }
    }

    return true;
}

bool nv::isTriangularMesh(const HalfEdge::Mesh * mesh)
{
    for (HalfEdge::Mesh::ConstFaceIterator it(mesh->faces()); !it.isDone(); it.advance())
    {
        const HalfEdge::Face * face = it.current();
        if (face->edgeCount() != 3) return false;
    }
    return true;
}


uint nv::countMeshTriangles(const HalfEdge::Mesh * mesh)
{
    const uint faceCount = mesh->faceCount();

    uint triangleCount = 0;

    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = mesh->faceAt(f);
        
        uint edgeCount = face->edgeCount();
        nvDebugCheck(edgeCount > 2);

        triangleCount += edgeCount - 2;
    }

    return triangleCount;
}

const HalfEdge::Vertex * nv::findBoundaryVertex(const HalfEdge::Mesh * mesh)
{
    const uint vertexCount = mesh->vertexCount();

    for (uint v = 0; v < vertexCount; v++)
    {
        const HalfEdge::Vertex * vertex = mesh->vertexAt(v);
        if (vertex->isBoundary()) return vertex;
    }

    return NULL;
}


HalfEdge::Mesh * nv::unifyVertices(const HalfEdge::Mesh * inputMesh)
{
    HalfEdge::Mesh * mesh = new HalfEdge::Mesh;
    
    // Only add the first colocal.
    const uint vertexCount = inputMesh->vertexCount();
    for (uint v = 0; v < vertexCount; v++) {
        const HalfEdge::Vertex * vertex = inputMesh->vertexAt(v);
        
        if (vertex->isFirstColocal()) {
            mesh->addVertex(vertex->pos);
        }
    }

    nv::Array<uint> indexArray;

    // Add new faces pointing to first colocals.
    uint faceCount = inputMesh->faceCount();
    for (uint f = 0; f < faceCount; f++) {
        const HalfEdge::Face * face = inputMesh->faceAt(f);

        indexArray.clear();

        for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
            const HalfEdge::Edge * edge = it.current();
            const HalfEdge::Vertex * vertex = edge->vertex->firstColocal();

            indexArray.append(vertex->id);
        }

        mesh->addFace(indexArray);
    }

    mesh->linkBoundary();

    return mesh;
}

#include "nvmath/Basis.h"

static bool pointInTriangle(const Vector2 & p, const Vector2 & a, const Vector2 & b, const Vector2 & c)
{
    return triangleArea(a, b, p) >= 0.00001f && 
        triangleArea(b, c, p) >= 0.00001f && 
        triangleArea(c, a, p) >= 0.00001f; 
}


// This is doing a simple ear-clipping algorithm that skips invalid triangles. Ideally, we should
// also sort the ears by angle, start with the ones that have the smallest angle and proceed in order.
HalfEdge::Mesh * nv::triangulate(const HalfEdge::Mesh * inputMesh)
{
    HalfEdge::Mesh * mesh = new HalfEdge::Mesh;
    
    // Add all vertices.
    const uint vertexCount = inputMesh->vertexCount();
    for (uint v = 0; v < vertexCount; v++) {
        const HalfEdge::Vertex * vertex = inputMesh->vertexAt(v);
        mesh->addVertex(vertex->pos);
    }

    Array<int> polygonVertices;
    Array<float> polygonAngles;
    Array<Vector2> polygonPoints;

    const uint faceCount = inputMesh->faceCount();
    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = inputMesh->faceAt(f);
        nvDebugCheck(face != NULL);

        const uint edgeCount = face->edgeCount();
        nvDebugCheck(edgeCount >= 3);

        polygonVertices.clear();
        polygonVertices.reserve(edgeCount);

        if (edgeCount == 3) {
            // Simple case for triangles.
            for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
            {
                const HalfEdge::Edge * edge = it.current();
                const HalfEdge::Vertex * vertex = edge->vertex;
                polygonVertices.append(vertex->id);
            }

            int v0 = polygonVertices[0];
            int v1 = polygonVertices[1];
            int v2 = polygonVertices[2];

            mesh->addFace(v0, v1, v2);
        }
        else {
            // Build 2D polygon projecting vertices onto normal plane.
            // Faces are not necesarily planar, this is for example the case, when the face comes from filling a hole. In such cases
            // it's much better to use the best fit plane.
            const Vector3 fn = face->normal();

            Basis basis;
            basis.buildFrameForDirection(fn);

            polygonPoints.clear();
            polygonPoints.reserve(edgeCount);
            polygonAngles.clear();
            polygonAngles.reserve(edgeCount);

            for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
            {
                const HalfEdge::Edge * edge = it.current();
                const HalfEdge::Vertex * vertex = edge->vertex;
                polygonVertices.append(vertex->id);
                
                Vector2 p;
                p.x = dot(basis.tangent, vertex->pos);
                p.y = dot(basis.bitangent, vertex->pos);

                polygonPoints.append(p);
            }
            polygonAngles.resize(edgeCount);

            while (polygonVertices.size() > 2) {
                uint size = polygonVertices.size();

                // Update polygon angles. @@ Update only those that have changed.
                float minAngle = 2 * PI;
                uint bestEar = 0; // Use first one if none of them is valid.
                bool bestIsValid = false;
                for (uint i = 0; i < size; i++) {
                    uint i0 = i; 
                    uint i1 = (i+1) % size; // Use Sean's polygon interation trick.
                    uint i2 = (i+2) % size;

                    Vector2 p0 = polygonPoints[i0];
                    Vector2 p1 = polygonPoints[i1];
                    Vector2 p2 = polygonPoints[i2];

                    float d = clamp(dot(p0-p1, p2-p1) / (length(p0-p1) * length(p2-p1)), -1.0f, 1.0f);
                    float angle = acosf(d);
                    
                    float area = triangleArea(p0, p1, p2);
                    if (area < 0.0f) angle = 2.0f * PI - angle;

                    polygonAngles[i1] = angle;

                    if (angle < minAngle || !bestIsValid) {

                        // Make sure this is a valid ear, if not, skip this point.
                        bool valid = true;
                        for (uint j = 0; j < size; j++) {
                            if (j == i0 || j == i1 || j == i2) continue;
                            Vector2 p = polygonPoints[j];

                            if (pointInTriangle(p, p0, p1, p2)) {
                                valid = false;
                                break;
                            }
                        }

                        if (valid || !bestIsValid) {
                            minAngle = angle;
                            bestEar = i1;
                            bestIsValid = valid;
                        }
                    }
                }

                nvDebugCheck(minAngle <= 2 * PI);

                // Clip best ear:

                uint i0 = (bestEar+size-1) % size;
                uint i1 = (bestEar+0) % size;
                uint i2 = (bestEar+1) % size;

                int v0 = polygonVertices[i0];
                int v1 = polygonVertices[i1];
                int v2 = polygonVertices[i2];
                
                mesh->addFace(v0, v1, v2);

                polygonVertices.removeAt(i1);
                polygonPoints.removeAt(i1);
                polygonAngles.removeAt(i1);
            }
        }

#if 0

        uint i = 0;
        while (polygonVertices.size() > 2 && i < polygonVertices.size()) {
            uint size = polygonVertices.size();
            uint i0 = (i+0) % size;
            uint i1 = (i+1) % size;
            uint i2 = (i+2) % size;

            const HalfEdge::Vertex * v0 = polygonVertices[i0];
            const HalfEdge::Vertex * v1 = polygonVertices[i1];
            const HalfEdge::Vertex * v2 = polygonVertices[i2];

            const Vector3 p0 = v0->pos;
            const Vector3 p1 = v1->pos;
            const Vector3 p2 = v2->pos;

            const Vector3 e0 = p2 - p1;
            const Vector3 e1 = p0 - p1;

            // If this ear forms a valid triangle, setup relations, remove v1 and repeat.
            Vector3 n = cross(e0, e1);
            float len = dot(fn, n); // = sin(angle)
            
            float angle = asin(len);


            if (len > 0.0f) {
                mesh->addFace(v0->id(), v1->id(), v2->id());
                polygonVertices.removeAt(i1);
                polygonAngles.removeAt(i1);
                if (i2 > i1) i2--;
                // @@ Update angles at i0 and i2
            }
            else {
                i++;
            }
        }

        // @@ Create a few degenerate triangles to avoid introducing holes.
        i = 0;
        const uint size = polygonVertices.size();
        while (i < size - 2) {
            uint i0 = (i+0) % size;
            uint i1 = (i+1) % size;
            uint i2 = (i+2) % size;

            const HalfEdge::Vertex * v0 = polygonVertices[i0];
            const HalfEdge::Vertex * v1 = polygonVertices[i1];
            const HalfEdge::Vertex * v2 = polygonVertices[i2];

            mesh->addFace(v0->id(), v1->id(), v2->id());
            i++;
        }
#endif
    }

    mesh->linkBoundary();

    return mesh;
}


