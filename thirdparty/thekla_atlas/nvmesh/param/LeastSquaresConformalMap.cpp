// Copyright NVIDIA Corporation 2008 -- Ignacio Castano <icastano@nvidia.com>

#include "nvmesh.h" // pch

#include "LeastSquaresConformalMap.h"
#include "ParameterizationQuality.h"
#include "Util.h"

#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Vertex.h"
#include "nvmesh/halfedge/Face.h"

#include "nvmath/Sparse.h"
#include "nvmath/Solver.h"
#include "nvmath/Vector.inl"

#include "nvcore/Array.inl"


using namespace nv;
using namespace HalfEdge;

namespace
{

    // Test all pairs of vertices in the boundary and check distance.
    static void findDiameterVertices(HalfEdge::Mesh * mesh, HalfEdge::Vertex ** a, HalfEdge::Vertex ** b)
    {
        nvDebugCheck(mesh != NULL);
        nvDebugCheck(a != NULL);
        nvDebugCheck(b != NULL);

        const uint vertexCount = mesh->vertexCount();

        float maxLength = 0.0f;

        for (uint v0 = 1; v0 < vertexCount; v0++)
        {
            HalfEdge::Vertex * vertex0 = mesh->vertexAt(v0);
            nvDebugCheck(vertex0 != NULL);

            if (!vertex0->isBoundary()) continue;

            for (uint v1 = 0; v1 < v0; v1++)
            {
                HalfEdge::Vertex * vertex1 = mesh->vertexAt(v1);
                nvDebugCheck(vertex1 != NULL);

                if (!vertex1->isBoundary()) continue;

                float len = length(vertex0->pos - vertex1->pos);

                if (len > maxLength)
                {
                    maxLength = len;

                    *a = vertex0;
                    *b = vertex1;
                }
            }
        }

        nvDebugCheck(*a != NULL && *b != NULL);
    }

    // Fast sweep in 3 directions
    static bool findApproximateDiameterVertices(HalfEdge::Mesh * mesh, HalfEdge::Vertex ** a, HalfEdge::Vertex ** b)
    {
        nvDebugCheck(mesh != NULL);
        nvDebugCheck(a != NULL);
        nvDebugCheck(b != NULL);

        const uint vertexCount = mesh->vertexCount();

        HalfEdge::Vertex * minVertex[3];
        HalfEdge::Vertex * maxVertex[3];

        minVertex[0] = minVertex[1] = minVertex[2] = NULL;
        maxVertex[0] = maxVertex[1] = maxVertex[2] = NULL;

        for (uint v = 1; v < vertexCount; v++)
        {
            HalfEdge::Vertex * vertex = mesh->vertexAt(v);
            nvDebugCheck(vertex != NULL);

            if (vertex->isBoundary())
            {
                minVertex[0] = minVertex[1] = minVertex[2] = vertex;
                maxVertex[0] = maxVertex[1] = maxVertex[2] = vertex;
                break;
            }
        }

        if (minVertex[0] == NULL)
        {
            // Input mesh has not boundaries.
            return false;
        }

        for (uint v = 1; v < vertexCount; v++)
        {
            HalfEdge::Vertex * vertex = mesh->vertexAt(v);
            nvDebugCheck(vertex != NULL);

            if (!vertex->isBoundary())
            {
                // Skip interior vertices.
                continue;
            }

            if (vertex->pos.x < minVertex[0]->pos.x) minVertex[0] = vertex;
            else if (vertex->pos.x > maxVertex[0]->pos.x) maxVertex[0] = vertex;

            if (vertex->pos.y < minVertex[1]->pos.y) minVertex[1] = vertex;
            else if (vertex->pos.y > maxVertex[1]->pos.y) maxVertex[1] = vertex;

            if (vertex->pos.z < minVertex[2]->pos.z) minVertex[2] = vertex;
            else if (vertex->pos.z > maxVertex[2]->pos.z) maxVertex[2] = vertex;
        }

        float lengths[3];
        for (int i = 0; i < 3; i++)
        {
            lengths[i] = length(minVertex[i]->pos - maxVertex[i]->pos);
        }

        if (lengths[0] > lengths[1] && lengths[0] > lengths[2])
        {
            *a = minVertex[0];
            *b = maxVertex[0];
        }
        else if (lengths[1] > lengths[2])
        {
            *a = minVertex[1];
            *b = maxVertex[1];
        }
        else
        {
            *a = minVertex[2];
            *b = maxVertex[2];
        }

        return true;
    }

    // Conformal relations from Bruno Levy:

    // Computes the coordinates of the vertices of a triangle
    // in a local 2D orthonormal basis of the triangle's plane.
    static void project_triangle(Vector3::Arg p0, Vector3::Arg p1, Vector3::Arg p2, Vector2 * z0, Vector2 * z1, Vector2 * z2)
    {
        Vector3 X = normalize(p1 - p0, 0.0f);
        Vector3 Z = normalize(cross(X, (p2 - p0)), 0.0f);
        Vector3 Y = normalize(cross(Z, X), 0.0f);

        float x0 = 0.0f;
        float y0 = 0.0f;
        float x1 = length(p1 - p0);
        float y1 = 0.0f;
        float x2 = dot((p2 - p0), X);
        float y2 = dot((p2 - p0), Y);

        *z0 = Vector2(x0, y0);
        *z1 = Vector2(x1, y1);
        *z2 = Vector2(x2, y2);
    }

    // LSCM equation, geometric form :
    // (Z1 - Z0)(U2 - U0) = (Z2 - Z0)(U1 - U0)
    // Where Uk = uk + i.vk is the complex number 
    //                       corresponding to (u,v) coords
    //       Zk = xk + i.yk is the complex number 
    //                       corresponding to local (x,y) coords
    // cool: no divide with this expression,
    //  makes it more numerically stable in
    //  the presence of degenerate triangles.

    static void setup_conformal_map_relations(SparseMatrix & A, int row, const HalfEdge::Vertex * v0, const HalfEdge::Vertex * v1, const HalfEdge::Vertex * v2)
    {
        int id0 = v0->id;
        int id1 = v1->id;
        int id2 = v2->id;

        Vector3 p0 = v0->pos;
        Vector3 p1 = v1->pos;
        Vector3 p2 = v2->pos;

        Vector2 z0, z1, z2;
        project_triangle(p0, p1, p2, &z0, &z1, &z2);

        Vector2 z01 = z1 - z0;
        Vector2 z02 = z2 - z0;

        float a = z01.x;
        float b = z01.y;
        float c = z02.x;
        float d = z02.y;
        nvCheck(b == 0.0f);

        // Note  : 2*id + 0 --> u
        //         2*id + 1 --> v
        int u0_id = 2 * id0 + 0;
        int v0_id = 2 * id0 + 1;
        int u1_id = 2 * id1 + 0;
        int v1_id = 2 * id1 + 1;
        int u2_id = 2 * id2 + 0;
        int v2_id = 2 * id2 + 1;

        // Note : b = 0

        // Real part
        A.setCoefficient(u0_id, 2 * row + 0, -a+c);
        A.setCoefficient(v0_id, 2 * row + 0,  b-d);
        A.setCoefficient(u1_id, 2 * row + 0,   -c);
        A.setCoefficient(v1_id, 2 * row + 0,    d);
        A.setCoefficient(u2_id, 2 * row + 0,    a);

        // Imaginary part
        A.setCoefficient(u0_id, 2 * row + 1, -b+d);
        A.setCoefficient(v0_id, 2 * row + 1, -a+c);
        A.setCoefficient(u1_id, 2 * row + 1,   -d);
        A.setCoefficient(v1_id, 2 * row + 1,   -c);
        A.setCoefficient(v2_id, 2 * row + 1,    a);
    }


    // Conformal relations from Brecht Van Lommel (based on ABF):

    static float vec_angle_cos(Vector3::Arg v1, Vector3::Arg v2, Vector3::Arg v3)
    {
        Vector3 d1 = v1 - v2;
        Vector3 d2 = v3 - v2;
        return clamp(dot(d1, d2) / (length(d1) * length(d2)), -1.0f, 1.0f);
    }

    static float vec_angle(Vector3::Arg v1, Vector3::Arg v2, Vector3::Arg v3)
    {
        float dot = vec_angle_cos(v1, v2, v3);
        return acosf(dot);
    }

    static void triangle_angles(Vector3::Arg v1, Vector3::Arg v2, Vector3::Arg v3, float *a1, float *a2, float *a3)
    {
        *a1 = vec_angle(v3, v1, v2);
        *a2 = vec_angle(v1, v2, v3);
        *a3 = PI - *a2 - *a1;
    }

    static void triangle_cosines(Vector3::Arg v1, Vector3::Arg v2, Vector3::Arg v3, float *a1, float *a2, float *a3)
    {
        *a1 = vec_angle_cos(v3, v1, v2);
        *a2 = vec_angle_cos(v1, v2, v3);
        *a3 = vec_angle_cos(v2, v3, v1);
    }

    static void setup_abf_relations(SparseMatrix & A, int row, const HalfEdge::Vertex * v0, const HalfEdge::Vertex * v1, const HalfEdge::Vertex * v2)
    {
        int id0 = v0->id;
        int id1 = v1->id;
        int id2 = v2->id;

        Vector3 p0 = v0->pos;
        Vector3 p1 = v1->pos;
        Vector3 p2 = v2->pos;

#if 1
        // @@ IC: Wouldn't it be more accurate to return cos and compute 1-cos^2?
        // It does indeed seem to be a little bit more robust.
        // @@ Need to revisit this more carefully!

        float a0, a1, a2;
        triangle_angles(p0, p1, p2, &a0, &a1, &a2);

        float s0 = sinf(a0);
        float s1 = sinf(a1);
        float s2 = sinf(a2);

        /*// Hack for degenerate triangles.
        if (equal(s0, 0) && equal(s1, 0) && equal(s2, 0)) {
            if (equal(a0, 0)) a0 += 0.001f;
            if (equal(a1, 0)) a1 += 0.001f;
            if (equal(a2, 0)) a2 += 0.001f;

            if (equal(a0, PI)) a0 = PI - a1 - a2;
            if (equal(a1, PI)) a1 = PI - a0 - a2;
            if (equal(a2, PI)) a2 = PI - a0 - a1;

            s0 = sinf(a0);
            s1 = sinf(a1);
            s2 = sinf(a2);
        }*/

        if (s1 > s0 && s1 > s2)
        {
            swap(s1, s2);
            swap(s0, s1);

            swap(a1, a2);
            swap(a0, a1);

            swap(id1, id2);
            swap(id0, id1);
        }
        else if (s0 > s1 && s0 > s2)
        {
            swap(s0, s2);
            swap(s0, s1);

            swap(a0, a2);
            swap(a0, a1);

            swap(id0, id2);
            swap(id0, id1);
        }

        float c0 = cosf(a0);
#else
        float c0, c1, c2;
        triangle_cosines(p0, p1, p2, &c0, &c1, &c2);

        float s0 = 1 - c0*c0;
        float s1 = 1 - c1*c1;
        float s2 = 1 - c2*c2;

        nvDebugCheck(s0 != 0 || s1 != 0 || s2 != 0);

        if (s1 > s0 && s1 > s2)
        {
            swap(s1, s2);
            swap(s0, s1);

            swap(c1, c2);
            swap(c0, c1);

            swap(id1, id2);
            swap(id0, id1);
        }
        else if (s0 > s1 && s0 > s2)
        {
            swap(s0, s2);
            swap(s0, s1);

            swap(c0, c2);
            swap(c0, c1);

            swap(id0, id2);
            swap(id0, id1);
        }
#endif

        float ratio = (s2 == 0.0f) ? 1.0f: s1/s2;
        float cosine = c0 * ratio;
        float sine = s0 * ratio;

        // Note  : 2*id + 0 --> u
        //         2*id + 1 --> v
        int u0_id = 2 * id0 + 0;
        int v0_id = 2 * id0 + 1;
        int u1_id = 2 * id1 + 0;
        int v1_id = 2 * id1 + 1;
        int u2_id = 2 * id2 + 0;
        int v2_id = 2 * id2 + 1;

        // Real part
        A.setCoefficient(u0_id, 2 * row + 0, cosine - 1.0f);
        A.setCoefficient(v0_id, 2 * row + 0, -sine);
        A.setCoefficient(u1_id, 2 * row + 0, -cosine);
        A.setCoefficient(v1_id, 2 * row + 0, sine);
        A.setCoefficient(u2_id, 2 * row + 0, 1);

        // Imaginary part
        A.setCoefficient(u0_id, 2 * row + 1, sine);
        A.setCoefficient(v0_id, 2 * row + 1, cosine - 1.0f);
        A.setCoefficient(u1_id, 2 * row + 1, -sine);
        A.setCoefficient(v1_id, 2 * row + 1, -cosine);
        A.setCoefficient(v2_id, 2 * row + 1, 1);
    }

} // namespace


bool nv::computeLeastSquaresConformalMap(HalfEdge::Mesh * mesh)
{
    nvDebugCheck(mesh != NULL);

    // For this to work properly, mesh should not have colocals that have the same 
    // attributes, unless you want the vertices to actually have different texcoords.

    const uint vertexCount = mesh->vertexCount();
    const uint D = 2 * vertexCount;
    const uint N = 2 * countMeshTriangles(mesh);

    // N is the number of equations (one per triangle)
    // D is the number of variables (one per vertex; there are 2 pinned vertices).
	if (N < D - 4) {
		return false;
	}

    SparseMatrix A(D, N);
    FullVector b(N);
    FullVector x(D);

    // Fill b:
    b.fill(0.0f);

    // Fill x:
    HalfEdge::Vertex * v0;
    HalfEdge::Vertex * v1;
    if (!findApproximateDiameterVertices(mesh, &v0, &v1))
    {
        // Mesh has no boundaries.
        return false;
    }
    if (v0->tex == v1->tex)
    {
        // LSCM expects an existing parameterization.
        return false;
    }

    for (uint v = 0; v < vertexCount; v++)
    {
        HalfEdge::Vertex * vertex = mesh->vertexAt(v);
        nvDebugCheck(vertex != NULL);

        // Initial solution.
        x[2 * v + 0] = vertex->tex.x;
        x[2 * v + 1] = vertex->tex.y;
    }

    // Fill A:
    const uint faceCount = mesh->faceCount();
    for (uint f = 0, t = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = mesh->faceAt(f);
        nvDebugCheck(face != NULL);
        nvDebugCheck(face->edgeCount() == 3);

        const HalfEdge::Vertex * vertex0 = NULL;

        for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            const HalfEdge::Edge * edge = it.current();
            nvCheck(edge != NULL);

            if (vertex0 == NULL)
            {
                vertex0 = edge->vertex;
            }
            else if (edge->next->vertex != vertex0)
            {
                const HalfEdge::Vertex * vertex1 = edge->from();
                const HalfEdge::Vertex * vertex2 = edge->to();

                setup_abf_relations(A, t, vertex0, vertex1, vertex2);
                //setup_conformal_map_relations(A, t, vertex0, vertex1, vertex2);

                t++;
            }
        }
    }

    const uint lockedParameters[] =
    {
        2 * v0->id + 0,
        2 * v0->id + 1,
        2 * v1->id + 0,
        2 * v1->id + 1
    };

    // Solve
    LeastSquaresSolver(A, b, x, lockedParameters, 4, 0.000001f);

    // Map x back to texcoords:
    for (uint v = 0; v < vertexCount; v++)
    {
        HalfEdge::Vertex * vertex = mesh->vertexAt(v);
        nvDebugCheck(vertex != NULL);

        vertex->tex = Vector2(x[2 * v + 0], x[2 * v + 1]);
    }

    return true;
}
