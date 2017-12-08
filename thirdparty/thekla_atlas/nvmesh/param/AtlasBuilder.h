// This code is in the public domain -- castano@gmail.com

#pragma once
#ifndef NV_MESH_ATLASBUILDER_H
#define NV_MESH_ATLASBUILDER_H

#include "Atlas.h"

#include "nvmath/Vector.h"
#include "nvmath/Random.h"
#include "nvmesh/nvmesh.h"

#include "nvcore/Array.h"
#include "nvcore/BitArray.h"



namespace nv
{
    namespace HalfEdge { class Mesh; }

    struct ChartBuildData;

    struct AtlasBuilder
    {
        AtlasBuilder(const HalfEdge::Mesh * m);
        ~AtlasBuilder();

        void markUnchartedFaces(const Array<uint> & unchartedFaces);

        void computeShortestPaths();

        void placeSeeds(float threshold, uint maxSeedCount);
        void createRandomChart(float threshold);

        void addFaceToChart(ChartBuildData * chart, uint f, bool recomputeProxy=false);

        bool growCharts(float threshold, uint faceCount);
        bool growChart(ChartBuildData * chart, float threshold, uint faceCount);

        void resetCharts();

        void updateCandidates(ChartBuildData * chart, uint face);

        void updateProxies();
        void updateProxy(ChartBuildData * chart);

        bool relocateSeeds();
        bool relocateSeed(ChartBuildData * chart);

        void updatePriorities(ChartBuildData * chart);

        float evaluatePriority(ChartBuildData * chart, uint face);
        float evaluateProxyFitMetric(ChartBuildData * chart, uint face);
        float evaluateDistanceToBoundary(ChartBuildData * chart, uint face);
        float evaluateDistanceToSeed(ChartBuildData * chart, uint face);
        float evaluateRoundnessMetric(ChartBuildData * chart, uint face, float newBoundaryLength, float newChartArea);
        float evaluateStraightnessMetric(ChartBuildData * chart, uint face);

        float evaluateNormalSeamMetric(ChartBuildData * chart, uint f);
        float evaluateTextureSeamMetric(ChartBuildData * chart, uint f);
        float evaluateSeamMetric(ChartBuildData * chart, uint f);

        float evaluateChartArea(ChartBuildData * chart, uint f);
        float evaluateBoundaryLength(ChartBuildData * chart, uint f);
        Vector3 evaluateChartNormalSum(ChartBuildData * chart, uint f);
        Vector3 evaluateChartCentroidSum(ChartBuildData * chart, uint f);

        Vector3 computeChartCentroid(const ChartBuildData * chart);


        void fillHoles(float threshold);
        void mergeCharts();

        // @@ Cleanup.
        struct Candidate {
            uint face;
            ChartBuildData * chart;
            float metric;
        };

        const Candidate & getBestCandidate() const;
        void removeCandidate(uint f);
        void updateCandidate(ChartBuildData * chart, uint f, float metric);

        void mergeChart(ChartBuildData * owner, ChartBuildData * chart, float sharedBoundaryLength);


        uint chartCount() const { return chartArray.count(); }
        const Array<uint> & chartFaces(uint i) const;

        const HalfEdge::Mesh * mesh;
        uint facesLeft;
        Array<int> faceChartArray;
        Array<ChartBuildData *> chartArray;
        Array<float> shortestPaths;

        Array<float> edgeLengths;
        Array<float> faceAreas;

        Array<Candidate> candidateArray; //
        Array<uint> faceCandidateArray; // Map face index to candidate index.

        MTRand rand;

        SegmentationSettings settings;
    };

} // nv namespace

#endif // NV_MESH_ATLASBUILDER_H
