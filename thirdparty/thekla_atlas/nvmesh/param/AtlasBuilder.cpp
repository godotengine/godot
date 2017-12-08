// This code is in the public domain -- castano@gmail.com

#include "nvmesh.h" // pch

#include "AtlasBuilder.h"
#include "Util.h"

#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Face.h"
#include "nvmesh/halfedge/Vertex.h"

#include "nvmath/Matrix.inl"
#include "nvmath/Vector.inl"

//#include "nvcore/IntroSort.h"
#include "nvcore/Array.inl"

#include <algorithm> // std::sort

#include <float.h> // FLT_MAX
#include <limits.h> // UINT_MAX

using namespace nv;

namespace
{

    // Dummy implementation of a priority queue using sort at insertion.
    // - Insertion is o(n)
    // - Smallest element goes at the end, so that popping it is o(1).
    // - Resorting is n*log(n)
    // @@ Number of elements in the queue is usually small, and we'd have to rebalance often. I'm not sure it's worth implementing a heap.
    // @@ Searcing at removal would remove the need for sorting when priorities change.
    struct PriorityQueue
    {
        PriorityQueue(uint size = UINT_MAX) : maxSize(size) {}

        void push(float priority, uint face) {
            uint i = 0;
            const uint count = pairs.count();
            for (; i < count; i++) {
                if (pairs[i].priority > priority) break;
            }

            Pair p = { priority, face };
            pairs.insertAt(i, p);

            if (pairs.count() > maxSize) {
                pairs.removeAt(0);
            }
        }

        // push face out of order, to be sorted later.
        void push(uint face) {
            Pair p = { 0.0f, face };
            pairs.append(p);
        }

        uint pop() {
            uint f = pairs.back().face;
            pairs.pop_back();
            return f;
        }

        void sort() {
            //nv::sort(pairs); // @@ My intro sort appears to be much slower than it should!
            std::sort(pairs.buffer(), pairs.buffer() + pairs.count());
        }

        void clear() {
            pairs.clear();
        }

        uint count() const { return pairs.count(); }

        float firstPriority() const { return pairs.back().priority; }


        const uint maxSize;
        
        struct Pair {
            bool operator <(const Pair & p) const { return priority > p.priority; } // !! Sort in inverse priority order!
            float priority;
            uint face;
        };
        

        Array<Pair> pairs;
    };

    static bool isNormalSeam(const HalfEdge::Edge * edge) {
        return (edge->vertex->nor != edge->pair->next->vertex->nor || edge->next->vertex->nor != edge->pair->vertex->nor);
    }

    static bool isTextureSeam(const HalfEdge::Edge * edge) {
        return (edge->vertex->tex != edge->pair->next->vertex->tex || edge->next->vertex->tex != edge->pair->vertex->tex);
    }

} // namespace


struct nv::ChartBuildData
{
    ChartBuildData(int id) : id(id) {
        planeNormal = Vector3(0);
        centroid = Vector3(0);
        coneAxis = Vector3(0);
        coneAngle = 0;
        area = 0;
        boundaryLength = 0;
        normalSum = Vector3(0);
        centroidSum = Vector3(0);
    }

    int id;

    // Proxy info:
    Vector3 planeNormal;
    Vector3 centroid;
    Vector3 coneAxis;
    float coneAngle;
    
    float area;
    float boundaryLength;
    Vector3 normalSum;
    Vector3 centroidSum;
    
    Array<uint> seeds;  // @@ These could be a pointers to the HalfEdge faces directly.
	Array<uint> faces;
    PriorityQueue candidates;
};



AtlasBuilder::AtlasBuilder(const HalfEdge::Mesh * m) : mesh(m), facesLeft(m->faceCount())
{
    const uint faceCount = m->faceCount();
    faceChartArray.resize(faceCount, -1);
    faceCandidateArray.resize(faceCount, -1);

    // @@ Floyd for the whole mesh is too slow. We could compute floyd progressively per patch as the patch grows. We need a better solution to compute most central faces.
    //computeShortestPaths();

    // Precompute edge lengths and face areas.
    uint edgeCount = m->edgeCount();
    edgeLengths.resize(edgeCount);

    for (uint i = 0; i < edgeCount; i++) {
        uint id = m->edgeAt(i)->id;
        nvDebugCheck(id / 2 == i);

        edgeLengths[i] = m->edgeAt(i)->length();
    }

    faceAreas.resize(faceCount);
    for (uint i = 0; i < faceCount; i++) {
        faceAreas[i] = m->faceAt(i)->area();
    }
}

AtlasBuilder::~AtlasBuilder()
{
    const uint chartCount = chartArray.count();
    for (uint i = 0; i < chartCount; i++)
    {
        delete chartArray[i];
    }
}


void AtlasBuilder::markUnchartedFaces(const Array<uint> & unchartedFaces)
{
    const uint unchartedFaceCount = unchartedFaces.count();
    for (uint i = 0; i < unchartedFaceCount; i++){ 
        uint f = unchartedFaces[i];
        faceChartArray[f] = -2;
        //faceCandidateArray[f] = -2; // @@ ?

        removeCandidate(f);
    }

    nvDebugCheck(facesLeft >= unchartedFaceCount);
    facesLeft -= unchartedFaceCount;
}


void AtlasBuilder::computeShortestPaths()
{
    const uint faceCount = mesh->faceCount();
    shortestPaths.resize(faceCount*faceCount, FLT_MAX);

    // Fill edges:
    for (uint i = 0; i < faceCount; i++)
    {
        shortestPaths[i*faceCount + i] = 0.0f;

        const HalfEdge::Face * face_i = mesh->faceAt(i);
        Vector3 centroid_i = face_i->centroid();

        for (HalfEdge::Face::ConstEdgeIterator it(face_i->edges()); !it.isDone(); it.advance())
        {
            const HalfEdge::Edge * edge = it.current();

            if (!edge->isBoundary())
            {
                const HalfEdge::Face * face_j = edge->pair->face;

                uint j = face_j->id;
                Vector3 centroid_j = face_j->centroid();

                shortestPaths[i*faceCount + j] = shortestPaths[j*faceCount + i] = length(centroid_i - centroid_j);
            }
        }
    }

    // Use Floyd-Warshall algorithm to compute all paths:
    for (uint k = 0; k < faceCount; k++)
    {
        for (uint i = 0; i < faceCount; i++)
        {
            for (uint j = 0; j < faceCount; j++)
            {
                shortestPaths[i*faceCount + j] = min(shortestPaths[i*faceCount + j], shortestPaths[i*faceCount + k]+shortestPaths[k*faceCount + j]);
            }
        }
    }
}


void AtlasBuilder::placeSeeds(float threshold, uint maxSeedCount)
{
    // Instead of using a predefiened number of seeds:
    // - Add seeds one by one, growing chart until a certain treshold.
    // - Undo charts and restart growing process.

    // @@ How can we give preference to faces far from sharp features as in the LSCM paper?
    //   - those points can be found using a simple flood filling algorithm.
    //   - how do we weight the probabilities?

    for (uint i = 0; i < maxSeedCount; i++)
    {
        if (facesLeft == 0) {
            // No faces left, stop creating seeds.
            break;
        }

        createRandomChart(threshold);
    }
}


void AtlasBuilder::createRandomChart(float threshold)
{
    ChartBuildData * chart = new ChartBuildData(chartArray.count());
    chartArray.append(chart);

    // Pick random face that is not used by any chart yet.
    uint randomFaceIdx = rand.getRange(facesLeft - 1);
    uint i = 0;
    for (uint f = 0; f != randomFaceIdx; f++, i++)
    {
        while (faceChartArray[i] != -1) i++;
    }
    while (faceChartArray[i] != -1) i++;

    chart->seeds.append(i);

    addFaceToChart(chart, i, true);

    // Grow the chart as much as possible within the given threshold.
    growChart(chart, threshold * 0.5f, facesLeft);
    //growCharts(threshold - threshold * 0.75f / chartCount(), facesLeft);
}

void AtlasBuilder::addFaceToChart(ChartBuildData * chart, uint f, bool recomputeProxy)
{
    // Add face to chart.
    chart->faces.append(f);

    nvDebugCheck(faceChartArray[f] == -1);
    faceChartArray[f] = chart->id;

    facesLeft--;

    // Update area and boundary length.
    chart->area = evaluateChartArea(chart, f);
    chart->boundaryLength = evaluateBoundaryLength(chart, f);
    chart->normalSum = evaluateChartNormalSum(chart, f);
    chart->centroidSum = evaluateChartCentroidSum(chart, f);

    if (recomputeProxy) {
        // Update proxy and candidate's priorities.
        updateProxy(chart);
    }

    // Update candidates.
    removeCandidate(f);
    updateCandidates(chart, f);
    updatePriorities(chart);
}

// @@ Get N best candidates in one pass.
const AtlasBuilder::Candidate & AtlasBuilder::getBestCandidate() const
{
    uint best = 0;
    float bestCandidateMetric = FLT_MAX;

    const uint candidateCount = candidateArray.count();
    nvCheck(candidateCount > 0);

    for (uint i = 0; i < candidateCount; i++)
    {
        const Candidate & candidate = candidateArray[i];
    
        if (candidate.metric < bestCandidateMetric) {
            bestCandidateMetric = candidate.metric;
            best = i;
        }
    }

    return candidateArray[best];
}


// Returns true if any of the charts can grow more.
bool AtlasBuilder::growCharts(float threshold, uint faceCount)
{
#if 1 // Using one global list.

    faceCount = min(faceCount, facesLeft);

    for (uint i = 0; i < faceCount; i++)
    {
        const Candidate & candidate = getBestCandidate();
        
        if (candidate.metric > threshold) {
            return false; // Can't grow more.
        }

        addFaceToChart(candidate.chart, candidate.face);
    }

    return facesLeft != 0; // Can continue growing.

#else // Using one list per chart.
    bool canGrowMore = false;

    const uint chartCount = chartArray.count();
    for (uint i = 0; i < chartCount; i++)
    {
        if (growChart(chartArray[i], threshold, faceCount))
        {
            canGrowMore = true;
        }
    }

    return canGrowMore;
#endif
}

bool AtlasBuilder::growChart(ChartBuildData * chart, float threshold, uint faceCount)
{
    // Try to add faceCount faces within threshold to chart.
    for (uint i = 0; i < faceCount; )
    {
        if (chart->candidates.count() == 0 || chart->candidates.firstPriority() > threshold)
        {
            return false;
        }

        uint f = chart->candidates.pop();
        if (faceChartArray[f] == -1)
        {
            addFaceToChart(chart, f);
            i++;
        }
    }

    if (chart->candidates.count() == 0 || chart->candidates.firstPriority() > threshold)
    {
        return false;
    }

    return true;
}


void AtlasBuilder::resetCharts()
{
    const uint faceCount = mesh->faceCount();
    for (uint i = 0; i < faceCount; i++)
    {
        faceChartArray[i] = -1;
        faceCandidateArray[i] = -1;
    }

    facesLeft = faceCount;

    candidateArray.clear();

    const uint chartCount = chartArray.count();
    for (uint i = 0; i < chartCount; i++)
    {
        ChartBuildData * chart = chartArray[i];

        const uint seed = chart->seeds.back();

        chart->area = 0.0f;
        chart->boundaryLength = 0.0f;
        chart->normalSum = Vector3(0);
        chart->centroidSum = Vector3(0);

        chart->faces.clear();
        chart->candidates.clear();

        addFaceToChart(chart, seed);
    }
}


void AtlasBuilder::updateCandidates(ChartBuildData * chart, uint f)
{
    const HalfEdge::Face * face = mesh->faceAt(f);

    // Traverse neighboring faces, add the ones that do not belong to any chart yet.
    for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
    {
        const HalfEdge::Edge * edge = it.current()->pair;

        if (!edge->isBoundary())
        {
            uint f = edge->face->id;

            if (faceChartArray[f] == -1)
            {
                chart->candidates.push(f);
            }
        }
    }
}


void AtlasBuilder::updateProxies()
{
    const uint chartCount = chartArray.count();
    for (uint i = 0; i < chartCount; i++)
    {
        updateProxy(chartArray[i]);
    }
}


namespace {

    float absoluteSum(Vector4::Arg v)
    {
        return fabs(v.x) + fabs(v.y) + fabs(v.z) + fabs(v.w);
    }

    //#pragma message(NV_FILE_LINE "FIXME: Using the c=cos(teta) substitution, the equation system becomes linear and we can avoid the newton solver.")

    struct ConeFitting
    {
        ConeFitting(const HalfEdge::Mesh * m, float g, float tf, float tx) : mesh(m), gamma(g), tolf(tf), tolx(tx), F(0), D(0), H(0) {
        }

        void addTerm(Vector3 N, float A)
        {
            const float c = cosf(X.w);
            const float s = sinf(X.w);
            const float tmp = dot(X.xyz(), N) - c;

            F += tmp * tmp;

            D.x += 2 * X.x * tmp;
            D.y += 2 * X.y * tmp;
            D.z += 2 * X.z * tmp;
            D.w += 2 * s * tmp;

            H(0,0) = 2 * X.x * N.x + 2 * tmp;
            H(0,1) = 2 * X.x * N.y;
            H(0,2) = 2 * X.x * N.z;
            H(0,3) = 2 * X.x * s;

            H(1,0) = 2 * X.y * N.x;
            H(1,1) = 2 * X.y * N.y + 2 * tmp;
            H(1,2) = 2 * X.y * N.z;
            H(1,3) = 2 * X.y * s;

            H(2,0) = 2 * X.z * N.x;
            H(2,1) = 2 * X.z * N.y;
            H(2,2) = 2 * X.z * N.z + 2 * tmp;
            H(2,3) = 2 * X.z * s;

            H(3,0) = 2 * s * N.x;
            H(3,1) = 2 * s * N.y;
            H(3,2) = 2 * s * N.z;
            H(3,3) = 2 * s * s + 2 * c * tmp;
        }

        Vector4 solve(ChartBuildData * chart, Vector4 start)
        {
            const uint faceCount = chart->faces.count();

            X = start;
            
            Vector4 dX;

            do {
                for (uint i = 0; i < faceCount; i++)
                {
                    const HalfEdge::Face * face = mesh->faceAt(chart->faces[i]);

                    addTerm(face->normal(), face->area());
                }

                Vector4 dX;
                //solveKramer(H, D, &dX);
                solveLU(H, D, &dX);

                // @@ Do a full newton step and reduce by half if F doesn't decrease.
                X -= gamma * dX;

                // Constrain normal to be normalized.
                X = Vector4(normalize(X.xyz()), X.w);
                
            } while(absoluteSum(D) > tolf || absoluteSum(dX) > tolx);

            return X;
        }

        HalfEdge::Mesh const * const mesh;
        const float gamma;
        const float tolf;
        const float tolx;

        Vector4 X;

        float F;
        Vector4 D;
        Matrix H;
    };

    // Unnormalized face normal assuming it's a triangle.
    static Vector3 triangleNormal(const HalfEdge::Face * face)
    {
        Vector3 p0 = face->edge->vertex->pos;
        Vector3 p1 = face->edge->next->vertex->pos;
        Vector3 p2 = face->edge->next->next->vertex->pos;

        Vector3 e0 = p2 - p0;
        Vector3 e1 = p1 - p0;

        return normalizeSafe(cross(e0, e1), Vector3(0), 0.0f);
    }

    static Vector3 triangleNormalAreaScaled(const HalfEdge::Face * face)
    {
        Vector3 p0 = face->edge->vertex->pos;
        Vector3 p1 = face->edge->next->vertex->pos;
        Vector3 p2 = face->edge->next->next->vertex->pos;

        Vector3 e0 = p2 - p0;
        Vector3 e1 = p1 - p0;

        return cross(e0, e1);
    }

    // Average of the edge midpoints weighted by the edge length.
    // I want a point inside the triangle, but closer to the cirumcenter.
    static Vector3 triangleCenter(const HalfEdge::Face * face)
    {
        Vector3 p0 = face->edge->vertex->pos;
        Vector3 p1 = face->edge->next->vertex->pos;
        Vector3 p2 = face->edge->next->next->vertex->pos;

        float l0 = length(p1 - p0);
        float l1 = length(p2 - p1);
        float l2 = length(p0 - p2);

        Vector3 m0 = (p0 + p1) * l0 / (l0 + l1 + l2);
        Vector3 m1 = (p1 + p2) * l1 / (l0 + l1 + l2);
        Vector3 m2 = (p2 + p0) * l2 / (l0 + l1 + l2);

        return m0 + m1 + m2;
    }

} // namespace

void AtlasBuilder::updateProxy(ChartBuildData * chart)
{
    //#pragma message(NV_FILE_LINE "TODO: Use best fit plane instead of average normal.")

    chart->planeNormal = normalizeSafe(chart->normalSum, Vector3(0), 0.0f);
    chart->centroid = chart->centroidSum / float(chart->faces.count());

    //#pragma message(NV_FILE_LINE "TODO: Experiment with conic fitting.")

    // F = (Nc*Nt - cos Oc)^2 = (x*Nt_x + y*Nt_y + z*Nt_z - cos w)^2
    // dF/dx = 2 * x * (x*Nt_x + y*Nt_y + z*Nt_z - cos w)
    // dF/dy = 2 * y * (x*Nt_x + y*Nt_y + z*Nt_z - cos w)
    // dF/dz = 2 * z * (x*Nt_x + y*Nt_y + z*Nt_z - cos w)
    // dF/dw = 2 * sin w * (x*Nt_x + y*Nt_y + z*Nt_z - cos w)

    // JacobianMatrix({
    // 2 * x * (x*Nt_x + y*Nt_y + z*Nt_z - Cos(w)),
    // 2 * y * (x*Nt_x + y*Nt_y + z*Nt_z - Cos(w)),
    // 2 * z * (x*Nt_x + y*Nt_y + z*Nt_z - Cos(w)),
    // 2 * Sin(w) * (x*Nt_x + y*Nt_y + z*Nt_z - Cos(w))}, {x,y,z,w})

    // H[0,0] = 2 * x * Nt_x + 2 * (x*Nt_x + y*Nt_y + z*Nt_z - cos(w));
    // H[0,1] = 2 * x * Nt_y;
    // H[0,2] = 2 * x * Nt_z;
    // H[0,3] = 2 * x * sin(w);

    // H[1,0] = 2 * y * Nt_x;
    // H[1,1] = 2 * y * Nt_y + 2 * (x*Nt_x + y*Nt_y + z*Nt_z - cos(w));
    // H[1,2] = 2 * y * Nt_z;
    // H[1,3] = 2 * y * sin(w);

    // H[2,0] = 2 * z * Nt_x;
    // H[2,1] = 2 * z * Nt_y;
    // H[2,2] = 2 * z * Nt_z + 2 * (x*Nt_x + y*Nt_y + z*Nt_z - cos(w));
    // H[2,3] = 2 * z * sin(w);

    // H[3,0] = 2 * sin(w) * Nt_x;
    // H[3,1] = 2 * sin(w) * Nt_y;
    // H[3,2] = 2 * sin(w) * Nt_z;
    // H[3,3] = 2 * sin(w) * sin(w) + 2 * cos(w) * (x*Nt_x + y*Nt_y + z*Nt_z - cos(w));

    // @@ Cone fitting might be quite slow.

    /*ConeFitting coneFitting(mesh, 0.1f, 0.001f, 0.001f);

    Vector4 start = Vector4(chart->coneAxis, chart->coneAngle);
    Vector4 solution = coneFitting.solve(chart, start);

    chart->coneAxis = solution.xyz();
    chart->coneAngle = solution.w;*/
}



bool AtlasBuilder::relocateSeeds()
{
    bool anySeedChanged = false;

    const uint chartCount = chartArray.count();
    for (uint i = 0; i < chartCount; i++)
    {
        if (relocateSeed(chartArray[i]))
        {
            anySeedChanged = true;
        }
    }

    return anySeedChanged;
}


bool AtlasBuilder::relocateSeed(ChartBuildData * chart)
{
    Vector3 centroid = computeChartCentroid(chart);

    const uint N = 10;  // @@ Hardcoded to 10?
    PriorityQueue bestTriangles(N); 

    // Find the first N triangles that fit the proxy best.
    const uint faceCount = chart->faces.count();
    for (uint i = 0; i < faceCount; i++)
    {
        float priority = evaluateProxyFitMetric(chart, chart->faces[i]);
        bestTriangles.push(priority, chart->faces[i]);
    }

    // Of those, choose the most central triangle.
    uint mostCentral;
    float maxDistance = -1;

    const uint bestCount = bestTriangles.count();
    for (uint i = 0; i < bestCount; i++)
    {
        const HalfEdge::Face * face = mesh->faceAt(bestTriangles.pairs[i].face);
        Vector3 faceCentroid = triangleCenter(face);

        float distance = length(centroid - faceCentroid);

        /*#pragma message(NV_FILE_LINE "TODO: Implement evaluateDistanceToBoundary.")
        float distance = evaluateDistanceToBoundary(chart, bestTriangles.pairs[i].face);*/
        
        if (distance > maxDistance)
        {
            maxDistance = distance;
            mostCentral = bestTriangles.pairs[i].face;
        }
    }
    nvDebugCheck(maxDistance >= 0);

    // In order to prevent k-means cyles we record all the previously chosen seeds.
    uint index;
    if (chart->seeds.find(mostCentral, &index))
    {
        // Move new seed to the end of the seed array.
        uint last = chart->seeds.count() - 1;
        swap(chart->seeds[index], chart->seeds[last]);
        return false;
    }
    else
    {
        // Append new seed.
        chart->seeds.append(mostCentral);
        return true;
    }
}

void AtlasBuilder::removeCandidate(uint f)
{
    int c = faceCandidateArray[f];
    if (c != -1) {
        faceCandidateArray[f] = -1;

        if (c == candidateArray.count() - 1) {
            candidateArray.popBack();
        }
        else {
            candidateArray.replaceWithLast(c);
            faceCandidateArray[candidateArray[c].face] = c;
        }
    }
}

void AtlasBuilder::updateCandidate(ChartBuildData * chart, uint f, float metric)
{
    if (faceCandidateArray[f] == -1) {
        const uint index = candidateArray.count();
        faceCandidateArray[f] = index;
        candidateArray.resize(index + 1);
        candidateArray[index].face = f;
        candidateArray[index].chart = chart;
        candidateArray[index].metric = metric;
    }
    else {
        int c = faceCandidateArray[f];
        nvDebugCheck(c != -1);

        Candidate & candidate = candidateArray[c];
        nvDebugCheck(candidate.face == f);

        if (metric < candidate.metric || chart == candidate.chart) {
            candidate.metric = metric;
            candidate.chart = chart;
        }
    }

}


void AtlasBuilder::updatePriorities(ChartBuildData * chart)
{
    // Re-evaluate candidate priorities.
    uint candidateCount = chart->candidates.count();
    for (uint i = 0; i < candidateCount; i++)
    {
        chart->candidates.pairs[i].priority = evaluatePriority(chart, chart->candidates.pairs[i].face);

        if (faceChartArray[chart->candidates.pairs[i].face] == -1)
        {
            updateCandidate(chart, chart->candidates.pairs[i].face, chart->candidates.pairs[i].priority);
        }
    }

    // Sort candidates.
    chart->candidates.sort();
}


// Evaluate combined metric.
float AtlasBuilder::evaluatePriority(ChartBuildData * chart, uint face)
{
    // Estimate boundary length and area:
    float newBoundaryLength = evaluateBoundaryLength(chart, face);
    float newChartArea = evaluateChartArea(chart, face);

    float F = evaluateProxyFitMetric(chart, face);
    float C = evaluateRoundnessMetric(chart, face, newBoundaryLength, newChartArea);
    float P = evaluateStraightnessMetric(chart, face);

    // Penalize faces that cross seams, reward faces that close seams or reach boundaries.
    float N = evaluateNormalSeamMetric(chart, face);
    float T = evaluateTextureSeamMetric(chart, face);

    //float R = evaluateCompletenessMetric(chart, face);

    //float D = evaluateDihedralAngleMetric(chart, face);
    // @@ Add a metric based on local dihedral angle.

    // @@ Tweaking the normal and texture seam metrics.
    // - Cause more impedance. Never cross 90 degree edges.
    // - 

    float cost = float(
        settings.proxyFitMetricWeight * F + 
        settings.roundnessMetricWeight * C + 
        settings.straightnessMetricWeight * P +
        settings.normalSeamMetricWeight * N +
        settings.textureSeamMetricWeight * T);

    /*cost = settings.proxyFitMetricWeight * powf(F, settings.proxyFitMetricExponent);
    cost = max(cost, settings.roundnessMetricWeight * powf(C, settings.roundnessMetricExponent));
    cost = max(cost, settings.straightnessMetricWeight * pow(P, settings.straightnessMetricExponent));
    cost = max(cost, settings.normalSeamMetricWeight * N);
    cost = max(cost, settings.textureSeamMetricWeight * T);*/

    // Enforce limits strictly:
    if (newChartArea > settings.maxChartArea) cost = FLT_MAX;
    if (newBoundaryLength > settings.maxBoundaryLength) cost = FLT_MAX;

    // Make sure normal seams are fully respected:
    if (settings.normalSeamMetricWeight >= 1000 && N != 0) cost = FLT_MAX;

    nvCheck(isFinite(cost));
    return cost;
}


// Returns a value in [0-1].
float AtlasBuilder::evaluateProxyFitMetric(ChartBuildData * chart, uint f)
{
    const HalfEdge::Face * face = mesh->faceAt(f);
    Vector3 faceNormal = triangleNormal(face);
    //return square(dot(chart->coneAxis, faceNormal) - cosf(chart->coneAngle));

    // Use plane fitting metric for now:
    //return square(1 - dot(faceNormal, chart->planeNormal)); // @@ normal deviations should be weighted by face area
    return 1 - dot(faceNormal, chart->planeNormal); // @@ normal deviations should be weighted by face area

    // Find distance to chart.
    /*Vector3 faceCentroid = face->centroid();

    float dist = 0;
    int count = 0;

    for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
    {
        const HalfEdge::Edge * edge = it.current();

        if (!edge->isBoundary()) {
            const HalfEdge::Face * neighborFace = edge->pair()->face();
            if (faceChartArray[neighborFace->id()] == chart->id) {
                dist += length(neighborFace->centroid() - faceCentroid);
                count++;
            }
        }
    }

    dist /= (count * count);

    return (1 - dot(faceNormal, chart->planeNormal)) * dist;*/

    //return (1 - dot(faceNormal, chart->planeNormal));
}

float AtlasBuilder::evaluateDistanceToBoundary(ChartBuildData * chart, uint face)
{
//#pragma message(NV_FILE_LINE "TODO: Evaluate distance to boundary metric.")

    // @@ This is needed for the seed relocation code.
    // @@ This could provide a better roundness metric.
    
    return 0.0f;
}

float AtlasBuilder::evaluateDistanceToSeed(ChartBuildData * chart, uint f)
{
    //const uint seed = chart->seeds.back();
    //const uint faceCount = mesh->faceCount();
    //return shortestPaths[seed * faceCount + f];

    const HalfEdge::Face * seed = mesh->faceAt(chart->seeds.back());
    const HalfEdge::Face * face = mesh->faceAt(f);
    return length(triangleCenter(seed) - triangleCenter(face));
}


float AtlasBuilder::evaluateRoundnessMetric(ChartBuildData * chart, uint face, float newBoundaryLength, float newChartArea)
{
    // @@ D-charts use distance to seed.
    // C(c,t) = pi * D(S_c,t)^2 / A_c
    //return PI * square(evaluateDistanceToSeed(chart, face)) / chart->area;
    //return PI * square(evaluateDistanceToSeed(chart, face)) / chart->area;
    //return 2 * PI * evaluateDistanceToSeed(chart, face) / chart->boundaryLength;

    // Garland's Hierarchical Face Clustering paper uses ratio between boundary and area, which is easier to compute and might work as well:
    // roundness = D^2/4*pi*A -> circle = 1, non circle greater than 1

    //return square(newBoundaryLength) / (newChartArea * 4 * PI);
    float roundness = square(chart->boundaryLength) / chart->area;
    float newRoundness = square(newBoundaryLength) / newChartArea;
    if (newRoundness > roundness) {
        return square(newBoundaryLength) / (newChartArea * 4 * PI);
    }
    else {
        // Offer no impedance to faces that improve roundness.
        return 0;
    }

    //return square(newBoundaryLength) / (4 * PI * newChartArea);
    //return clamp(1 - (4 * PI * newChartArea) / square(newBoundaryLength), 0.0f, 1.0f);

    // Use the ratio between the new roundness vs. the previous roundness.
    // - If we use the absolute metric, when the initial face is very long, then it's hard to make any progress.
    //return (square(newBoundaryLength) * chart->area) / (square(chart->boundaryLength) * newChartArea);
    //return (4 * PI * newChartArea) / square(newBoundaryLength) - (4 * PI * chart->area) / square(chart->boundaryLength);

    //if (square(newBoundaryLength) * chart->area) / (square(chart->boundaryLength) * newChartArea);

}

float AtlasBuilder::evaluateStraightnessMetric(ChartBuildData * chart, uint f)
{
    float l_out = 0.0f;
    float l_in = 0.0f;

    const HalfEdge::Face * face = mesh->faceAt(f);
    for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
    {
        const HalfEdge::Edge * edge = it.current();

        //float l = edge->length();
        float l = edgeLengths[edge->id/2];

        if (edge->isBoundary())
        {
            l_out += l;
        }
        else
        {
            uint neighborFaceId = edge->pair->face->id;
            if (faceChartArray[neighborFaceId] != chart->id) {
                l_out += l;
            }
            else {
                l_in += l;
            }
        }
    }
    nvDebugCheck(l_in != 0.0f); // Candidate face must be adjacent to chart. @@ This is not true if the input mesh has zero-length edges.

    //return l_out / l_in;
    float ratio = (l_out - l_in) / (l_out + l_in);
    //if (ratio < 0) ratio *= 10; // Encourage closing gaps.
    return min(ratio, 0.0f); // Only use the straightness metric to close gaps.
    //return ratio;
}


float AtlasBuilder::evaluateNormalSeamMetric(ChartBuildData * chart, uint f)
{
    float seamFactor = 0.0f;
    float totalLength = 0.0f;
    
    const HalfEdge::Face * face = mesh->faceAt(f);
    for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
    {
        const HalfEdge::Edge * edge = it.current();

        if (edge->isBoundary()) {
            continue;
        }

        const uint neighborFaceId = edge->pair->face->id;
        if (faceChartArray[neighborFaceId] != chart->id) {
            continue;
        }

        //float l = edge->length();
        float l = edgeLengths[edge->id/2];

        totalLength += l;

        if (!edge->isSeam()) {
            continue;
        }

        // Make sure it's a normal seam.
        if (isNormalSeam(edge))
        {
            float d0 = clamp(dot(edge->vertex->nor, edge->pair->next->vertex->nor), 0.0f, 1.0f);
            float d1 = clamp(dot(edge->next->vertex->nor, edge->pair->vertex->nor), 0.0f, 1.0f);
            //float a0 = clamp(acosf(d0) / (PI/2), 0.0f, 1.0f);
            //float a1 = clamp(acosf(d1) / (PI/2), 0.0f, 1.0f);
            //l *= (a0 + a1) * 0.5f;

            l *= 1 - (d0 + d1) * 0.5f;

            seamFactor += l;
        }
    }

    if (seamFactor == 0) return 0.0f;
    return seamFactor / totalLength;
}


float AtlasBuilder::evaluateTextureSeamMetric(ChartBuildData * chart, uint f)
{
    float seamLength = 0.0f;
    //float newSeamLength = 0.0f;
    //float oldSeamLength = 0.0f;
    float totalLength = 0.0f;
    
    const HalfEdge::Face * face = mesh->faceAt(f);
    for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
    {
        const HalfEdge::Edge * edge = it.current();

        /*float l = edge->length();
        totalLength += l;

        if (edge->isBoundary() || !edge->isSeam()) {
            continue;
        }

        // Make sure it's a texture seam.
        if (isTextureSeam(edge))
        {
            uint neighborFaceId = edge->pair()->face()->id();
            if (faceChartArray[neighborFaceId] != chart->id) {
                newSeamLength += l;
            }
            else {
                oldSeamLength += l;
            }
        }*/

        if (edge->isBoundary()) {
            continue;
        }

        const uint neighborFaceId = edge->pair->face->id;
        if (faceChartArray[neighborFaceId] != chart->id) {
            continue;
        }

        //float l = edge->length();
        float l = edgeLengths[edge->id/2];
        totalLength += l;

        if (!edge->isSeam()) {
            continue;
        }

        // Make sure it's a texture seam.
        if (isTextureSeam(edge))
        {
            seamLength += l;
        }
    }

    if (seamLength == 0.0f) {
        return 0.0f; // Avoid division by zero.
    }
    
    return seamLength / totalLength;
}


float AtlasBuilder::evaluateSeamMetric(ChartBuildData * chart, uint f)
{
    float newSeamLength = 0.0f;
    float oldSeamLength = 0.0f;
    float totalLength = 0.0f;
    
    const HalfEdge::Face * face = mesh->faceAt(f);
    for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
    {
        const HalfEdge::Edge * edge = it.current();

        //float l = edge->length();
        float l = edgeLengths[edge->id/2];

        if (edge->isBoundary())
        {
            newSeamLength += l;
        }
        else
        {
            if (edge->isSeam())
            {
                uint neighborFaceId = edge->pair->face->id;
                if (faceChartArray[neighborFaceId] != chart->id) {
                    newSeamLength += l;
                }
                else {
                    oldSeamLength += l;
                }
            }
        }

        totalLength += l;
    }

    return (newSeamLength - oldSeamLength) / totalLength;
}


float AtlasBuilder::evaluateChartArea(ChartBuildData * chart, uint f)
{
    const HalfEdge::Face * face = mesh->faceAt(f);
    //return chart->area + face->area();
    return chart->area + faceAreas[face->id];
}


float AtlasBuilder::evaluateBoundaryLength(ChartBuildData * chart, uint f)
{
    float boundaryLength = chart->boundaryLength;

    // Add new edges, subtract edges shared with the chart.
    const HalfEdge::Face * face = mesh->faceAt(f);
    for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
    {
        const HalfEdge::Edge * edge = it.current();
        //float edgeLength = edge->length();
        float edgeLength = edgeLengths[edge->id/2];

        if (edge->isBoundary())
        {
            boundaryLength += edgeLength;
        }
        else
        {
            uint neighborFaceId = edge->pair->face->id;
            if (faceChartArray[neighborFaceId] != chart->id) {
                boundaryLength += edgeLength;
            }
            else {
                boundaryLength -= edgeLength;
            }
        }
    }
    //nvDebugCheck(boundaryLength >= 0);

    return max(0.0f, boundaryLength);  // @@ Hack!
}

Vector3 AtlasBuilder::evaluateChartNormalSum(ChartBuildData * chart, uint f)
{
    const HalfEdge::Face * face = mesh->faceAt(f);
    return chart->normalSum + triangleNormalAreaScaled(face);
}

Vector3 AtlasBuilder::evaluateChartCentroidSum(ChartBuildData * chart, uint f)
{
    const HalfEdge::Face * face = mesh->faceAt(f);
    return chart->centroidSum + face->centroid();
}


Vector3 AtlasBuilder::computeChartCentroid(const ChartBuildData * chart)
{
    Vector3 centroid(0);

    const uint faceCount = chart->faces.count();
    for (uint i = 0; i < faceCount; i++)
    {
        const HalfEdge::Face * face = mesh->faceAt(chart->faces[i]);
        centroid += triangleCenter(face);
    }

    return centroid / float(faceCount);
}


void AtlasBuilder::fillHoles(float threshold)
{
    while (facesLeft > 0)
    {
        createRandomChart(threshold);
    }
}


void AtlasBuilder::mergeChart(ChartBuildData * owner, ChartBuildData * chart, float sharedBoundaryLength)
{
    const uint faceCount = chart->faces.count();
    for (uint i = 0; i < faceCount; i++)
    {
        uint f = chart->faces[i];
        
        nvDebugCheck(faceChartArray[f] == chart->id);
        faceChartArray[f] = owner->id;

        owner->faces.append(f);
    }

    // Update adjacencies?

    owner->area += chart->area;
    owner->boundaryLength += chart->boundaryLength - sharedBoundaryLength;

    owner->normalSum += chart->normalSum;
    owner->centroidSum += chart->centroidSum;

    updateProxy(owner);
}

void AtlasBuilder::mergeCharts()
{
    Array<float> sharedBoundaryLengths;

    const uint chartCount = chartArray.count();
    for (int c = chartCount-1; c >= 0; c--)
    {
        sharedBoundaryLengths.clear();
        sharedBoundaryLengths.resize(chartCount, 0.0f);

        ChartBuildData * chart = chartArray[c];

        float externalBoundary = 0.0f;

        const uint faceCount = chart->faces.count();
        for (uint i = 0; i < faceCount; i++)
        {
            uint f = chart->faces[i];
            const HalfEdge::Face * face = mesh->faceAt(f);

            for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
            {
                const HalfEdge::Edge * edge = it.current();

                //float l = edge->length();
                float l = edgeLengths[edge->id/2];

                if (edge->isBoundary()) {
                    externalBoundary += l;
                }
                else {
                    uint neighborFace = edge->pair->face->id;
                    uint neighborChart = faceChartArray[neighborFace];

                    if (neighborChart != c) {
                        if ((edge->isSeam() && (isNormalSeam(edge) || isTextureSeam(edge))) || neighborChart == -2) {
                            externalBoundary += l;
                        }
                        else {
                            sharedBoundaryLengths[neighborChart] += l;
                        }
                    }
                }
            }
        }

        for (int cc = chartCount-1; cc >= 0; cc--)
        {
            if (cc == c) 
                continue;

            ChartBuildData * chart2 = chartArray[cc];
            if (chart2 == NULL) 
                continue;

            if (sharedBoundaryLengths[cc] > 0.8 * max(0.0f, chart->boundaryLength - externalBoundary)) {

                // Try to avoid degenerate configurations.
                if (chart2->boundaryLength > sharedBoundaryLengths[cc])
                {
                    if (dot(chart2->planeNormal, chart->planeNormal) > -0.25) {
                        mergeChart(chart2, chart, sharedBoundaryLengths[cc]);
                        delete chart;
                        chartArray[c] = NULL;
                        break;
                    }
                }
            }

            if (sharedBoundaryLengths[cc] > 0.20 * max(0.0f, chart->boundaryLength - externalBoundary)) {

                // Compare proxies.
                if (dot(chart2->planeNormal, chart->planeNormal) > 0) {
                    mergeChart(chart2, chart, sharedBoundaryLengths[cc]);
                    delete chart;
                    chartArray[c] = NULL;
                    break;
                }
            }
        }
    }

    // Remove deleted charts.
    for (int c = 0; c < I32(chartArray.count()); /*do not increment if removed*/)
    {
        if (chartArray[c] == NULL) {
            chartArray.removeAt(c);

            // Update faceChartArray.
            const uint faceCount = faceChartArray.count();
            for (uint i = 0; i < faceCount; i++) {
                nvDebugCheck (faceChartArray[i] != -1);
                nvDebugCheck (faceChartArray[i] != c);
                nvDebugCheck (faceChartArray[i] <= I32(chartArray.count()));

                if (faceChartArray[i] > c) {
                    faceChartArray[i]--;
                }
            }
        }
        else {
            chartArray[c]->id = c;
            c++;
        }
    }
}



const Array<uint> & AtlasBuilder::chartFaces(uint i) const
{
    return chartArray[i]->faces;
}
