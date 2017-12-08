// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MESH_MESHBUILDER_H
#define NV_MESH_MESHBUILDER_H

#include "nvmesh.h"
#include "nvcore/Array.h"
#include "nvmath/Vector.h"

namespace nv
{
    class String;
    class TriMesh;
    class QuadTriMesh;
    namespace HalfEdge { class Mesh; }


    /// Mesh builder is a helper class for importers.
    /// Ideally it should handle any vertex data, but for now it only accepts positions, 
    /// normals and texcoords.
    class MeshBuilder
    {
        NV_FORBID_COPY(MeshBuilder);
        NV_FORBID_HEAPALLOC();
    public:
        MeshBuilder();
        ~MeshBuilder();

        // Builder methods.
        uint addPosition(const Vector3 & v);
        uint addNormal(const Vector3 & v);
        uint addTexCoord(const Vector2 & v, uint set = 0);
        uint addColor(const Vector4 & v, uint set = 0);

        void beginGroup(uint id);
        void endGroup();

        uint addMaterial(const char * name);
        void beginMaterial(uint id);
        void endMaterial();

        void beginPolygon(uint id = 0);
        uint addVertex(uint p, uint n = NIL, uint t0 = NIL, uint t1 = NIL, uint c0 = NIL, uint c1 = NIL, uint c2 = NIL);
        uint addVertex(const Vector3 & p);
        //uint addVertex(const Vector3 & p, const Vector3 & n, const Vector2 & t0 = Vector2(0), const Vector2 & t1 = Vector2(0), const Vector4 & c0 = Vector4(0), const Vector4 & c1 = Vector4(0));
        bool endPolygon();

        uint weldPositions();
        uint weldNormals();
        uint weldTexCoords(uint set = 0);
        uint weldColors(uint set = 0);
        void weldVertices();

        void optimize(); // eliminate duplicate components and duplicate vertices.
        void removeUnusedMaterials(Array<uint> & newMaterialId);
        void sortFacesByGroup();
        void sortFacesByMaterial();

        void done();
        void reset();

        // Hints.
        void hintTriangleCount(uint count);
        void hintVertexCount(uint count);
        void hintPositionCount(uint count);
        void hintNormalCount(uint count);
        void hintTexCoordCount(uint count, uint set = 0);
        void hintColorCount(uint count, uint set = 0);

        // Helpers.
        void addTriangle(uint v0, uint v1, uint v2);
        void addQuad(uint v0, uint v1, uint v2, uint v3);

        // Get result.
        TriMesh * buildTriMesh() const;
        QuadTriMesh * buildQuadTriMesh() const;

        enum Error {
            Error_None,
            Error_NonManifoldEdge,
            Error_NonManifoldVertex,
        };

        HalfEdge::Mesh * buildHalfEdgeMesh(bool weldPositions, Error * error = NULL, Array<uint> * badFaces = NULL) const;

        bool buildPositions(Array<Vector3> & positionArray);
        bool buildNormals(Array<Vector3> & normalArray);
        bool buildTexCoords(Array<Vector2> & texCoordArray, uint set = 0);
        bool buildColors(Array<Vector4> & colorArray, uint set = 0);
		void buildVertexToPositionMap(Array<int> & map);


        // Expose attribute indices of the unified vertex array.
        uint vertexCount() const;
        
        uint positionCount() const;
        uint normalCount() const;
        uint texCoordCount(uint set = 0) const;
        uint colorCount(uint set = 0) const;

        uint materialCount() const;
        const char * material(uint i) const;

        uint positionIndex(uint vertex) const;
        uint normalIndex(uint vertex) const;
        uint texCoordIndex(uint vertex, uint set = 0) const;
        uint colorIndex(uint vertex, uint set = 0) const;

    private:

        struct PrivateData;
        PrivateData * d;

    };

} // nv namespace

#endif // NV_MESH_MESHBUILDER_H
