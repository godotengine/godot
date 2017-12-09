// Copyright NVIDIA Corporation 2006 -- Ignacio Castano <icastano@nvidia.com>

#pragma once
#ifndef NV_MESH_ATLAS_H
#define NV_MESH_ATLAS_H

#include "nvcore/Array.h"
#include "nvcore/Ptr.h"
#include "nvmath/Vector.h"
#include "nvmesh/nvmesh.h"
#include "nvmesh/halfedge/Mesh.h"


namespace nv
{
    namespace HalfEdge { class Mesh; }

    class Chart;
    class MeshCharts;
    class VertexMap;

    struct SegmentationSettings
    {
        SegmentationSettings();

        float maxChartArea;
        float maxBoundaryLength;

        float proxyFitMetricWeight;
        float roundnessMetricWeight;
        float straightnessMetricWeight;
        float normalSeamMetricWeight;
        float textureSeamMetricWeight;
    };


    /// An atlas is a set of charts.
    class Atlas
    {
    public:

        Atlas();
        ~Atlas();

        uint meshCount() const { return m_meshChartsArray.count(); }
        const MeshCharts * meshAt(uint i) const { return m_meshChartsArray[i]; }
        MeshCharts * meshAt(uint i) { return m_meshChartsArray[i]; }

        uint chartCount() const;
        const Chart * chartAt(uint i) const;
        Chart * chartAt(uint i);

        // Add mesh charts and takes ownership.
        void addMeshCharts(MeshCharts * meshCharts);

        void extractCharts(const HalfEdge::Mesh * mesh);
        void computeCharts(const HalfEdge::Mesh * mesh, const SegmentationSettings & settings, const Array<uint> & unchartedMaterialArray);


        // Compute a trivial seamless texture similar to ZBrush.
        //bool computeSeamlessTextureAtlas(bool groupFaces = true, bool scaleTiles = false, uint w = 1024, uint h = 1024);

        void parameterizeCharts();

        // Pack charts in the smallest possible rectangle.
        float packCharts(int quality, float texelArea, bool blockAlign, bool conservative);
        void setFailed() { failed = true; }
        bool hasFailed() const { return failed; }

    private:

        bool failed;
        Array<MeshCharts *> m_meshChartsArray;

    };


    // Set of charts corresponding to a single mesh.
    class MeshCharts
    {
    public:
        MeshCharts(const HalfEdge::Mesh * mesh);
        ~MeshCharts();

        uint chartCount() const { return m_chartArray.count(); }
        uint vertexCount () const { return m_totalVertexCount; }

        const Chart * chartAt(uint i) const { return m_chartArray[i]; }
        Chart * chartAt(uint i) { return m_chartArray[i]; }

        void computeVertexMap(const Array<uint> & unchartedMaterialArray);

        // Extract the charts of the input mesh.
        void extractCharts();

        // Compute charts using a simple segmentation algorithm.
        void computeCharts(const SegmentationSettings & settings, const Array<uint> & unchartedMaterialArray);

        void parameterizeCharts();

        uint faceChartAt(uint i) const { return m_faceChart[i]; }
        uint faceIndexWithinChartAt(uint i) const { return m_faceIndex[i]; }

        uint vertexCountBeforeChartAt(uint i) const { return m_chartVertexCountPrefixSum[i]; }

    private:

        const HalfEdge::Mesh * m_mesh;

        Array<Chart *> m_chartArray;
        
        Array<uint> m_chartVertexCountPrefixSum;
        uint m_totalVertexCount;

        Array<uint> m_faceChart; // the chart of every face of the input mesh.
        Array<uint> m_faceIndex; // the index within the chart for every face of the input mesh.
    };


    /// A chart is a connected set of faces with a certain topology (usually a disk).
    class Chart
    {
    public:

        Chart();

        void build(const HalfEdge::Mesh * originalMesh, const Array<uint> & faceArray);
        void buildVertexMap(const HalfEdge::Mesh * originalMesh, const Array<uint> & unchartedMaterialArray);

        bool closeHoles();

        bool isDisk() const { return m_isDisk; }
        bool isVertexMapped() const { return m_isVertexMapped; }

        uint vertexCount() const { return m_chartMesh->vertexCount(); }
        uint colocalVertexCount() const { return m_unifiedMesh->vertexCount(); }

        uint faceCount() const { return m_faceArray.count(); }
        uint faceAt(uint i) const { return m_faceArray[i]; }

        const HalfEdge::Mesh * chartMesh() const { return m_chartMesh.ptr(); }
        HalfEdge::Mesh * chartMesh() { return m_chartMesh.ptr(); }
        const HalfEdge::Mesh * unifiedMesh() const { return m_unifiedMesh.ptr(); }
        HalfEdge::Mesh * unifiedMesh() { return m_unifiedMesh.ptr(); }

        //uint vertexIndex(uint i) const { return m_vertexIndexArray[i]; }

        uint mapChartVertexToOriginalVertex(uint i) const { return m_chartToOriginalMap[i]; }
        uint mapChartVertexToUnifiedVertex(uint i) const { return m_chartToUnifiedMap[i]; }

        const Array<uint> & faceArray() const { return m_faceArray; }

        void transferParameterization();

        float computeSurfaceArea() const;
        float computeParametricArea() const;
        Vector2 computeParametricBounds() const;


        float scale = 1.0f;
        uint vertexMapWidth;
        uint vertexMapHeight;

    private:

        bool closeLoop(uint start, const Array<HalfEdge::Edge *> & loop);

        // Chart mesh.
        AutoPtr<HalfEdge::Mesh> m_chartMesh;
        AutoPtr<HalfEdge::Mesh> m_unifiedMesh;

        bool m_isDisk;
        bool m_isVertexMapped;

        // List of faces of the original mesh that belong to this chart.
        Array<uint> m_faceArray;

        // Map vertices of the chart mesh to vertices of the original mesh.
        Array<uint> m_chartToOriginalMap;

        Array<uint> m_chartToUnifiedMap;
    };

} // nv namespace

#endif // NV_MESH_ATLAS_H
