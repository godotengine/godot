// Copyright NVIDIA Corporation 2006 -- Ignacio Castano <icastano@nvidia.com>

#include "nvmesh.h" // pch

#include "Atlas.h"
#include "Util.h"
#include "AtlasBuilder.h"
#include "AtlasPacker.h"
#include "SingleFaceMap.h"
#include "OrthogonalProjectionMap.h"
#include "LeastSquaresConformalMap.h"
#include "ParameterizationQuality.h"

//#include "nvmesh/export/MeshExportOBJ.h"

#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Face.h"
#include "nvmesh/halfedge/Vertex.h"

#include "nvmesh/MeshBuilder.h"
#include "nvmesh/MeshTopology.h"
#include "nvmesh/param/Util.h"
#include "nvmesh/geometry/Measurements.h"

#include "nvmath/Vector.inl"
#include "nvmath/Fitting.h"
#include "nvmath/Box.inl"
#include "nvmath/ProximityGrid.h"
#include "nvmath/Morton.h"

#include "nvcore/StrLib.h"
#include "nvcore/Array.inl"
#include "nvcore/HashMap.inl"

using namespace nv;


/// Ctor.
Atlas::Atlas()
{
    failed=false;
}

// Dtor.
Atlas::~Atlas()
{
    deleteAll(m_meshChartsArray);
}

uint Atlas::chartCount() const
{
    uint count = 0;
    foreach(c, m_meshChartsArray) {
        count += m_meshChartsArray[c]->chartCount();
    }
    return count;
}

const Chart * Atlas::chartAt(uint i) const
{
    foreach(c, m_meshChartsArray) {
        uint count = m_meshChartsArray[c]->chartCount();

        if (i < count) {
            return m_meshChartsArray[c]->chartAt(i);
        }

        i -= count;
    }

    return NULL;
}

Chart * Atlas::chartAt(uint i) 
{
    foreach(c, m_meshChartsArray) {
        uint count = m_meshChartsArray[c]->chartCount();

        if (i < count) {
            return m_meshChartsArray[c]->chartAt(i);
        }

        i -= count;
    }

    return NULL;
}

// Extract the charts and add to this atlas.
void Atlas::addMeshCharts(MeshCharts * meshCharts)
{
    m_meshChartsArray.append(meshCharts);
}

void Atlas::extractCharts(const HalfEdge::Mesh * mesh)
{
    MeshCharts * meshCharts = new MeshCharts(mesh);
    meshCharts->extractCharts();
    addMeshCharts(meshCharts);
}

void Atlas::computeCharts(const HalfEdge::Mesh * mesh, const SegmentationSettings & settings, const Array<uint> & unchartedMaterialArray)
{
    failed=false;
    MeshCharts * meshCharts = new MeshCharts(mesh);
    meshCharts->computeCharts(settings, unchartedMaterialArray);
    addMeshCharts(meshCharts);
}




#if 0

/// Compute a seamless texture atlas.
bool Atlas::computeSeamlessTextureAtlas(bool groupFaces/*= true*/, bool scaleTiles/*= false*/, uint w/*= 1024*/, uint h/* = 1024*/)
{
    // Implement seamless texture atlas similar to what ZBrush does. See also:
    // "Meshed Atlases for Real-Time Procedural Solid Texturing"
    // http://graphics.cs.uiuc.edu/~jch/papers/rtpst.pdf

    // Other methods that we should experiment with:
    // 
    // Seamless Texture Atlases:
    // http://www.cs.jhu.edu/~bpurnomo/STA/index.html
    // 
    // Rectangular Multi-Chart Geometry Images:
    // http://graphics.cs.uiuc.edu/~jch/papers/rmcgi.pdf
    // 
    // Discrete differential geometry also provide a way of constructing  
    // seamless quadrangulations as shown in:
    // http://www.geometry.caltech.edu/pubs/TACD06.pdf
    // 

#pragma message(NV_FILE_LINE "TODO: Implement seamless texture atlas.")

    if (groupFaces)
    {
        // @@ TODO.
    }
    else
    {
        // @@ Create one atlas per face.
    }

    if (scaleTiles)
    {
        // @@ TODO
    }

    /*
    if (!isQuadMesh(m_mesh)) {
        // Only handle quads for now.
        return false;
    }

    // Each face is a chart.
    const uint faceCount = m_mesh->faceCount();
    m_chartArray.resize(faceCount);

    for(uint f = 0; f < faceCount; f++) {
        m_chartArray[f].faceArray.clear();
        m_chartArray[f].faceArray.append(f);
    }

    // Map each face to a separate square.

    // Determine face layout according to width and height.
    float aspect = float(m_width) / float(m_height);

    uint i = 2;
    uint total = (m_width / (i+1)) * (m_height / (i+1));
    while(total > faceCount) {
        i *= 2;
        total = (m_width / (i+1)) * (m_height / (i+1));
    }

    uint tileSize = i / 2;

    int x = 0;
    int y = 0;

    m_result = new HalfEdge::Mesh();

    // Once you have that it's just matter of traversing the faces.
    for(uint f = 0; f < faceCount; f++) {
        // Compute texture coordinates.
        Vector2 tex[4];
        tex[0] = Vector2(float(x), float(y));
        tex[1] = Vector2(float(x+tileSize), float(y));
        tex[2] = Vector2(float(x+tileSize), float(y+tileSize));
        tex[3] = Vector2(float(x), float(y+tileSize));

        Array<uint> indexArray(4);

        const HalfEdge::Face * face = m_mesh->faceAt(f);

        int i = 0;
        for(HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance(), i++) {
            const HalfEdge::Edge * edge = it.current();
            const HalfEdge::Vertex * vertex = edge->from();

            HalfEdge::Vertex * newVertex = m_result->addVertex(vertex->id(), vertex->pos());

            newVertex->setTex(Vector3(tex[i], 0));
            newVertex->setNor(vertex->nor());

            indexArray.append(m_result->vertexCount() + 1);
        }

        m_result->addFace(indexArray);

        // Move to the next tile.
        x += tileSize + 1;
        if (x + tileSize > m_width) {
            x = 0;
            y += tileSize + 1;
        }
    }
    */

    return false;
}

#endif


void Atlas::parameterizeCharts()
{
    foreach(i, m_meshChartsArray) {
        m_meshChartsArray[i]->parameterizeCharts();
    }
}


float Atlas::packCharts(int quality, float texelsPerUnit, bool blockAlign, bool conservative)
{
    AtlasPacker packer(this);
    packer.packCharts(quality, texelsPerUnit, blockAlign, conservative);
    if (hasFailed())
        return 0;
    return packer.computeAtlasUtilization();
}




/// Ctor.
MeshCharts::MeshCharts(const HalfEdge::Mesh * mesh) : m_mesh(mesh)
{
}

// Dtor.
MeshCharts::~MeshCharts()
{
    deleteAll(m_chartArray);
}


void MeshCharts::extractCharts()
{
    const uint faceCount = m_mesh->faceCount();

    int first = 0;
    Array<uint> queue(faceCount);

    BitArray bitFlags(faceCount);
    bitFlags.clearAll();

    for (uint f = 0; f < faceCount; f++)
    {
        if (bitFlags.bitAt(f) == false)
        {
            // Start new patch. Reset queue.
            first = 0;
            queue.clear();
            queue.append(f);
            bitFlags.setBitAt(f);

            while (first != queue.count())
            {
                const HalfEdge::Face * face = m_mesh->faceAt(queue[first]);

                // Visit face neighbors of queue[first]
                for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
                {
                    const HalfEdge::Edge * edge = it.current();
                    nvDebugCheck(edge->pair != NULL);

                    if (!edge->isBoundary() && /*!edge->isSeam()*/ 
                        //!(edge->from()->tex() != edge->pair()->to()->tex() || edge->to()->tex() != edge->pair()->from()->tex()))
                        !(edge->from() != edge->pair->to() || edge->to() != edge->pair->from())) // Preserve existing seams (not just texture seams).
                    {
                        const HalfEdge::Face * neighborFace = edge->pair->face;
                        nvDebugCheck(neighborFace != NULL);

                        if (bitFlags.bitAt(neighborFace->id) == false)
                        {
                            queue.append(neighborFace->id);
                            bitFlags.setBitAt(neighborFace->id);
                        }
                    }
                }

                first++;
            }

            Chart * chart = new Chart();
            chart->build(m_mesh, queue);

            m_chartArray.append(chart);
        }
    }
}


/*
LSCM:
- identify sharp features using local dihedral angles.
- identify seed faces farthest from sharp features.
- grow charts from these seeds.

MCGIM:
- phase 1: chart growth
  - grow all charts simultaneously using dijkstra search on the dual graph of the mesh.
  - graph edges are weighted based on planarity metric.
  - metric uses distance to global chart normal.
  - terminate when all faces have been assigned.
- phase 2: seed computation:
  - place new seed of the chart at the most interior face.
  - most interior is evaluated using distance metric only.

- method repeates the two phases, until the location of the seeds does not change.
  - cycles are detected by recording all the previous seeds and chartification terminates.

D-Charts:

- Uniaxial conic metric:
  - N_c = axis of the generalized cone that best fits the chart. (cone can a be cylinder or a plane).
  - omega_c = angle between the face normals and the axis.
  - Fitting error between chart C and tringle t: F(c,t) = (N_c*n_t - cos(omega_c))^2

- Compactness metrics:
  - Roundness:
    - C(c,t) = pi * D(S_c,t)^2 / A_c
    - S_c = chart seed.
    - D(S_c,t) = length of the shortest path inside the chart betwen S_c and t.
    - A_c = chart area.
  - Straightness:
    - P(c,t) = l_out(c,t) / l_in(c,t)
    - l_out(c,t) = lenght of the edges not shared between C and t.
    - l_in(c,t) = lenght of the edges shared between C and t.

- Combined metric:
  - Cost(c,t) = F(c,t)^alpha + C(c,t)^beta + P(c,t)^gamma
  - alpha = 1, beta = 0.7, gamma = 0.5




Our basic approach:
- Just one iteration of k-means?
- Avoid dijkstra by greedily growing charts until a threshold is met. Increase threshold and repeat until no faces left.
- If distortion metric is too high, split chart, add two seeds.
- If chart size is low, try removing chart.


Postprocess:
- If topology is not disk:
  - Fill holes, if new faces fit proxy.
  - Find best cut, otherwise.
- After parameterization:
  - If boundary self-intersects: 
    - cut chart along the closest two diametral boundary vertices, repeat parametrization.
    - what if the overlap is on an appendix? How do we find that out and cut appropiately?
      - emphasize roundness metrics to prevent those cases.
  - If interior self-overlaps: preserve boundary parameterization and use mean-value map.

*/


SegmentationSettings::SegmentationSettings()
{
    // Charts have no area or boundary limits right now.
    maxChartArea = NV_FLOAT_MAX;
    maxBoundaryLength = NV_FLOAT_MAX;

    proxyFitMetricWeight = 1.0f;
    roundnessMetricWeight = 0.1f;
    straightnessMetricWeight = 0.25f;
    normalSeamMetricWeight = 1.0f;
    textureSeamMetricWeight = 0.1f;
}



void MeshCharts::computeCharts(const SegmentationSettings & settings, const Array<uint> & unchartedMaterialArray)
{
    Chart * vertexMap = NULL;
    
    if (unchartedMaterialArray.count() != 0) {
        vertexMap = new Chart();
        vertexMap->buildVertexMap(m_mesh, unchartedMaterialArray);

        if (vertexMap->faceCount() == 0) {
            delete vertexMap;
            vertexMap = NULL;
        }
    }
    

    AtlasBuilder builder(m_mesh);

    if (vertexMap != NULL) {
        // Mark faces that do not need to be charted.
        builder.markUnchartedFaces(vertexMap->faceArray());

        m_chartArray.append(vertexMap);
    }

    if (builder.facesLeft != 0) {

        // Tweak these values:
        const float maxThreshold = 2;
        const uint growFaceCount = 32;
        const uint maxIterations = 4;
        
        builder.settings = settings;

        //builder.settings.proxyFitMetricWeight *= 0.75; // relax proxy fit weight during initial seed placement.
        //builder.settings.roundnessMetricWeight = 0;
        //builder.settings.straightnessMetricWeight = 0;

        // This seems a reasonable estimate.
        uint maxSeedCount = max(6U, builder.facesLeft);

        // Create initial charts greedely.
        nvDebug("### Placing seeds\n");
        builder.placeSeeds(maxThreshold, maxSeedCount);
        nvDebug("###   Placed %d seeds (max = %d)\n", builder.chartCount(), maxSeedCount);

        builder.updateProxies();

        builder.mergeCharts();

    #if 1
        nvDebug("### Relocating seeds\n");
        builder.relocateSeeds();

        nvDebug("### Reset charts\n");
        builder.resetCharts();

        if (vertexMap != NULL) {
            builder.markUnchartedFaces(vertexMap->faceArray());
        }

        builder.settings = settings;

        nvDebug("### Growing charts\n");

        // Restart process growing charts in parallel.
        uint iteration = 0;
        while (true)
        {
            if (!builder.growCharts(maxThreshold, growFaceCount))
            {
                nvDebug("### Can't grow anymore\n");

                // If charts cannot grow more: fill holes, merge charts, relocate seeds and start new iteration.

                nvDebug("### Filling holes\n");
                builder.fillHoles(maxThreshold);
                nvDebug("###   Using %d charts now\n", builder.chartCount());

                builder.updateProxies();

                nvDebug("### Merging charts\n");
                builder.mergeCharts();
                nvDebug("###   Using %d charts now\n", builder.chartCount());

                nvDebug("### Reseeding\n");
                if (!builder.relocateSeeds())
                {
                    nvDebug("### Cannot relocate seeds anymore\n");

                    // Done!
                    break;
                }

                if (iteration == maxIterations)
                {
                    nvDebug("### Reached iteration limit\n");
                    break;
                }
                iteration++;

                nvDebug("### Reset charts\n");
                builder.resetCharts();

                if (vertexMap != NULL) {
                    builder.markUnchartedFaces(vertexMap->faceArray());
                }

                nvDebug("### Growing charts\n");
            }
        };
    #endif

        // Make sure no holes are left!
        nvDebugCheck(builder.facesLeft == 0);

        const uint chartCount = builder.chartArray.count();
        for (uint i = 0; i < chartCount; i++)
        {
            Chart * chart = new Chart();
            m_chartArray.append(chart);

            chart->build(m_mesh, builder.chartFaces(i));
        }
    }


    const uint chartCount = m_chartArray.count();

    // Build face indices.
    m_faceChart.resize(m_mesh->faceCount());
    m_faceIndex.resize(m_mesh->faceCount());

    for (uint i = 0; i < chartCount; i++)
    {
        const Chart * chart = m_chartArray[i];

        const uint faceCount = chart->faceCount();
        for (uint f = 0; f < faceCount; f++)
        {
            uint idx = chart->faceAt(f);
            m_faceChart[idx] = i;
            m_faceIndex[idx] = f;
        }
    }

    // Build an exclusive prefix sum of the chart vertex counts.
    m_chartVertexCountPrefixSum.resize(chartCount);
    
    if (chartCount > 0)
    {
        m_chartVertexCountPrefixSum[0] = 0;
        
        for (uint i = 1; i < chartCount; i++)
        {
            const Chart * chart = m_chartArray[i-1];
            m_chartVertexCountPrefixSum[i] = m_chartVertexCountPrefixSum[i-1] + chart->vertexCount();
        }

        m_totalVertexCount = m_chartVertexCountPrefixSum[chartCount - 1] + m_chartArray[chartCount-1]->vertexCount();
    }
    else
    {
        m_totalVertexCount = 0;
    }
}


void MeshCharts::parameterizeCharts()
{
    ParameterizationQuality globalParameterizationQuality;

    // Parameterize the charts.
    uint diskCount = 0;
    const uint chartCount = m_chartArray.count();
    for (uint i = 0; i < chartCount; i++)\
    {
        Chart * chart = m_chartArray[i];

        bool isValid = false;

        if (chart->isVertexMapped()) {
            continue;
        }

        if (chart->isDisk())
        {
            diskCount++;

            ParameterizationQuality chartParameterizationQuality;

            if (chart->faceCount() == 1) {
                computeSingleFaceMap(chart->unifiedMesh());

                chartParameterizationQuality = ParameterizationQuality(chart->unifiedMesh());
            }
            else {
                computeOrthogonalProjectionMap(chart->unifiedMesh());
                ParameterizationQuality orthogonalQuality(chart->unifiedMesh());

                computeLeastSquaresConformalMap(chart->unifiedMesh());
                ParameterizationQuality lscmQuality(chart->unifiedMesh());
                
                // If the orthogonal projection produces better results, just use that.
                // @@ It may be dangerous to do this, because isValid() does not detect self-overlaps.
                // @@ Another problem is that with very thin patches with nearly zero parametric area, the results of our metric are not accurate.
                /*if (orthogonalQuality.isValid() && orthogonalQuality.rmsStretchMetric() < lscmQuality.rmsStretchMetric()) {
                    computeOrthogonalProjectionMap(chart->unifiedMesh());
                    chartParameterizationQuality = orthogonalQuality;
                }
                else*/ {
                    chartParameterizationQuality = lscmQuality;
                }

                // If conformal map failed, 

                // @@ Experiment with other parameterization methods.
                //computeCircularBoundaryMap(chart->unifiedMesh());
                //computeConformalMap(chart->unifiedMesh());
                //computeNaturalConformalMap(chart->unifiedMesh());
                //computeGuidanceGradientMap(chart->unifiedMesh());
            }

            //ParameterizationQuality chartParameterizationQuality(chart->unifiedMesh());

            isValid = chartParameterizationQuality.isValid();

            if (!isValid)
            {
                nvDebug("*** Invalid parameterization.\n");
#if 0
                // Dump mesh to inspect problem:
                static int pieceCount = 0;
            
                StringBuilder fileName;
                fileName.format("invalid_chart_%d.obj", pieceCount++);
                exportMesh(chart->unifiedMesh(), fileName.str()); 
#endif
            }

            // @@ Check that parameterization quality is above a certain threshold.

            // @@ Detect boundary self-intersections.

            globalParameterizationQuality += chartParameterizationQuality;
        }

        if (!isValid)
        {
            //nvDebugBreak();
            // @@ Run the builder again, but only on this chart.
            //AtlasBuilder builder(chart->chartMesh());
        }

        // Transfer parameterization from unified mesh to chart mesh.
        chart->transferParameterization();

    }

    nvDebug("  Parameterized %d/%d charts.\n", diskCount, chartCount);
    nvDebug("  RMS stretch metric: %f\n", globalParameterizationQuality.rmsStretchMetric());
    nvDebug("  MAX stretch metric: %f\n", globalParameterizationQuality.maxStretchMetric());
    nvDebug("  RMS conformal metric: %f\n", globalParameterizationQuality.rmsConformalMetric());
    nvDebug("  RMS authalic metric: %f\n", globalParameterizationQuality.maxAuthalicMetric());
}



Chart::Chart() : m_chartMesh(NULL), m_unifiedMesh(NULL), m_isDisk(false), m_isVertexMapped(false)
{
}

void Chart::build(const HalfEdge::Mesh * originalMesh, const Array<uint> & faceArray)
{
    // Copy face indices.
    m_faceArray = faceArray;

    const uint meshVertexCount = originalMesh->vertexCount();

    m_chartMesh = new HalfEdge::Mesh();
    m_unifiedMesh = new HalfEdge::Mesh();

    Array<uint> chartMeshIndices;
    chartMeshIndices.resize(meshVertexCount, ~0);

    Array<uint> unifiedMeshIndices;
    unifiedMeshIndices.resize(meshVertexCount, ~0);

    // Add vertices.
    const uint faceCount = faceArray.count();
    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = originalMesh->faceAt(faceArray[f]);
        nvDebugCheck(face != NULL);

        for(HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            const HalfEdge::Vertex * vertex = it.current()->vertex;
            const HalfEdge::Vertex * unifiedVertex = vertex->firstColocal();

            if (unifiedMeshIndices[unifiedVertex->id] == ~0)
            {
                unifiedMeshIndices[unifiedVertex->id] = m_unifiedMesh->vertexCount();

                nvDebugCheck(vertex->pos == unifiedVertex->pos);
                m_unifiedMesh->addVertex(vertex->pos);
            }

            if (chartMeshIndices[vertex->id] == ~0)
            {
                chartMeshIndices[vertex->id] = m_chartMesh->vertexCount();
                m_chartToOriginalMap.append(vertex->id);
                m_chartToUnifiedMap.append(unifiedMeshIndices[unifiedVertex->id]);

                HalfEdge::Vertex * v = m_chartMesh->addVertex(vertex->pos);
                v->nor = vertex->nor;
                v->tex = vertex->tex;
            }
        }
    }

    // This is ignoring the canonical map:
    // - Is it really necessary to link colocals?

    m_chartMesh->linkColocals();    
    //m_unifiedMesh->linkColocals();  // Not strictly necessary, no colocals in the unified mesh. # Wrong.

    // This check is not valid anymore, if the original mesh vertices were linked with a canonical map, then it might have
    // some colocal vertices that were unlinked. So, the unified mesh might have some duplicate vertices, because firstColocal()
    // is not guaranteed to return the same vertex for two colocal vertices.
    //nvCheck(m_chartMesh->colocalVertexCount() == m_unifiedMesh->vertexCount());

    // Is that OK? What happens in meshes were that happens? Does anything break? Apparently not...
    


    Array<uint> faceIndices(7);

    // Add faces.
    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = originalMesh->faceAt(faceArray[f]);
        nvDebugCheck(face != NULL);

        faceIndices.clear();

        for(HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            const HalfEdge::Vertex * vertex = it.current()->vertex;
            nvDebugCheck(vertex != NULL);

            faceIndices.append(chartMeshIndices[vertex->id]);
        }

        m_chartMesh->addFace(faceIndices);

        faceIndices.clear();

        for(HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            const HalfEdge::Vertex * vertex = it.current()->vertex;
            nvDebugCheck(vertex != NULL);

            vertex = vertex->firstColocal();

            faceIndices.append(unifiedMeshIndices[vertex->id]);
        }

        m_unifiedMesh->addFace(faceIndices);
    }

    m_chartMesh->linkBoundary();
    m_unifiedMesh->linkBoundary();

    //exportMesh(m_unifiedMesh.ptr(), "debug_input.obj");

    if (m_unifiedMesh->splitBoundaryEdges()) {
        m_unifiedMesh = unifyVertices(m_unifiedMesh.ptr());
    }

    //exportMesh(m_unifiedMesh.ptr(), "debug_split.obj");

    // Closing the holes is not always the best solution and does not fix all the problems.
    // We need to do some analysis of the holes and the genus to:
    // - Find cuts that reduce genus.
    // - Find cuts to connect holes.
    // - Use minimal spanning trees or seamster.
    if (!closeHoles()) {
        /*static int pieceCount = 0;
        StringBuilder fileName;
        fileName.format("debug_hole_%d.obj", pieceCount++);
        exportMesh(m_unifiedMesh.ptr(), fileName.str());*/
    }

    m_unifiedMesh = triangulate(m_unifiedMesh.ptr());
    
    //exportMesh(m_unifiedMesh.ptr(), "debug_triangulated.obj");


    // Analyze chart topology.
    MeshTopology topology(m_unifiedMesh.ptr());
    m_isDisk = topology.isDisk();

    // This is sometimes failing, when triangulate fails to add a triangle, it generates a hole in the mesh.
    //nvDebugCheck(m_isDisk);

    /*if (!m_isDisk) {
        static int pieceCount = 0;
        StringBuilder fileName;
        fileName.format("debug_hole_%d.obj", pieceCount++);
        exportMesh(m_unifiedMesh.ptr(), fileName.str());
    }*/


#if 0
    if (!m_isDisk) {
        nvDebugBreak();

        static int pieceCount = 0;
        
        StringBuilder fileName;
        fileName.format("debug_nodisk_%d.obj", pieceCount++);
        exportMesh(m_chartMesh.ptr(), fileName.str()); 
    }
#endif

}


void Chart::buildVertexMap(const HalfEdge::Mesh * originalMesh, const Array<uint> & unchartedMaterialArray)
{
    nvCheck(m_chartMesh == NULL && m_unifiedMesh == NULL);

    m_isVertexMapped = true;

    // Build face indices.
    m_faceArray.clear();

    const uint meshFaceCount = originalMesh->faceCount();
    for (uint f = 0; f < meshFaceCount; f++) {
        const HalfEdge::Face * face = originalMesh->faceAt(f);

        if (unchartedMaterialArray.contains(face->material)) {
            m_faceArray.append(f);
        }
    }

    const uint faceCount = m_faceArray.count();

    if (faceCount == 0) {
        return;
    }


    // @@ The chartMesh construction is basically the same as with regular charts, don't duplicate!

    const uint meshVertexCount = originalMesh->vertexCount();

    m_chartMesh = new HalfEdge::Mesh();

    Array<uint> chartMeshIndices;
    chartMeshIndices.resize(meshVertexCount, ~0);

    // Vertex map mesh only has disconnected vertices.
    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = originalMesh->faceAt(m_faceArray[f]);
        nvDebugCheck(face != NULL);

        for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            const HalfEdge::Vertex * vertex = it.current()->vertex;

            if (chartMeshIndices[vertex->id] == ~0)
            {
                chartMeshIndices[vertex->id] = m_chartMesh->vertexCount();
                m_chartToOriginalMap.append(vertex->id);

                HalfEdge::Vertex * v = m_chartMesh->addVertex(vertex->pos);
                v->nor = vertex->nor;
                v->tex = vertex->tex; // @@ Not necessary.
            }
        }
    }

    // @@ Link colocals using the original mesh canonical map? Build canonical map on the fly? Do we need to link colocals at all for this?
    //m_chartMesh->linkColocals();

    Array<uint> faceIndices(7);

    // Add faces.
    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = originalMesh->faceAt(m_faceArray[f]);
        nvDebugCheck(face != NULL);

        faceIndices.clear();

        for(HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            const HalfEdge::Vertex * vertex = it.current()->vertex;
            nvDebugCheck(vertex != NULL);
            nvDebugCheck(chartMeshIndices[vertex->id] != ~0);

            faceIndices.append(chartMeshIndices[vertex->id]);
        }

        HalfEdge::Face * new_face = m_chartMesh->addFace(faceIndices);
        nvDebugCheck(new_face != NULL);
    }

    m_chartMesh->linkBoundary();


    const uint chartVertexCount = m_chartMesh->vertexCount();

    Box bounds;
    bounds.clearBounds();

    for (uint i = 0; i < chartVertexCount; i++) {
        HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(i);
        bounds.addPointToBounds(vertex->pos);
    }

    ProximityGrid grid;
    grid.init(bounds, chartVertexCount);

    for (uint i = 0; i < chartVertexCount; i++) {
        HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(i);
        grid.add(vertex->pos, i);
    }


#if 0
    // Arrange vertices in a rectangle.
    vertexMapWidth = ftoi_ceil(sqrtf(float(chartVertexCount)));
    vertexMapHeight = (chartVertexCount + vertexMapWidth - 1) / vertexMapWidth;
    nvDebugCheck(vertexMapWidth >= vertexMapHeight);

    int x = 0, y = 0;
    for (uint i = 0; i < chartVertexCount; i++) {
        HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(i);

        vertex->tex.x = float(x);
        vertex->tex.y = float(y);

        x++;
        if (x == vertexMapWidth) {
            x = 0;
            y++;
            nvCheck(y < vertexMapHeight);
        }
    }

#elif 0
    // Arrange vertices in a rectangle, traversing grid in 3D morton order and laying them down in 2D morton order.
    vertexMapWidth = ftoi_ceil(sqrtf(float(chartVertexCount)));
    vertexMapHeight = (chartVertexCount + vertexMapWidth - 1) / vertexMapWidth;
    nvDebugCheck(vertexMapWidth >= vertexMapHeight);

    int n = 0;
    uint32 texelCode = 0;

    uint cellsVisited = 0;

    const uint32 cellCodeCount = grid.mortonCount();
    for (uint32 cellCode = 0; cellCode < cellCodeCount; cellCode++) {
        int cell = grid.mortonIndex(cellCode);
        if (cell < 0) continue;

        cellsVisited++;

        const Array<uint> & indexArray = grid.cellArray[cell].indexArray;

        foreach(i, indexArray) {
            uint idx = indexArray[i];
            HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(idx);

            //vertex->tex.x = float(n % rectangleWidth) + 0.5f;
            //vertex->tex.y = float(n / rectangleWidth) + 0.5f;

            // Lay down the points in z order too.
            uint x, y;
            do {
                x = decodeMorton2X(texelCode);
                y = decodeMorton2Y(texelCode);
                texelCode++;
            } while (x >= U32(vertexMapWidth) || y >= U32(vertexMapHeight));
            
            vertex->tex.x = float(x);
            vertex->tex.y = float(y);

            n++;
        }
    }

    nvDebugCheck(cellsVisited == grid.cellArray.count());
    nvDebugCheck(n == chartVertexCount);

#else

    uint texelCount = 0;

    const float positionThreshold = 0.01f;
    const float normalThreshold = 0.01f;

    uint verticesVisited = 0;
    uint cellsVisited = 0;

    Array<int> vertexIndexArray;
    vertexIndexArray.resize(chartVertexCount, -1); // Init all indices to -1.

    // Traverse vertices in morton order. @@ It may be more interesting to sort them based on orientation.
    const uint cellCodeCount = grid.mortonCount();
    for (uint cellCode = 0; cellCode < cellCodeCount; cellCode++) {
        int cell = grid.mortonIndex(cellCode);
        if (cell < 0) continue;

        cellsVisited++;

        const Array<uint> & indexArray = grid.cellArray[cell].indexArray;

        foreach(i, indexArray) {
            uint idx = indexArray[i];
            HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(idx);

            nvDebugCheck(vertexIndexArray[idx] == -1);

            Array<uint> neighbors;
            grid.gather(vertex->pos, positionThreshold, /*ref*/neighbors);

            // Compare against all nearby vertices, cluster greedily.
            foreach(j, neighbors) {
                uint otherIdx = neighbors[j];

                if (vertexIndexArray[otherIdx] != -1) {
                    HalfEdge::Vertex * otherVertex = m_chartMesh->vertexAt(otherIdx);

                    if (distance(vertex->pos, otherVertex->pos) < positionThreshold &&
                        distance(vertex->nor, otherVertex->nor) < normalThreshold) 
                    {
                        vertexIndexArray[idx] = vertexIndexArray[otherIdx];
                        break;
                    }
                }
            }

            // If index not assigned, assign new one.
            if (vertexIndexArray[idx] == -1) {
                vertexIndexArray[idx] = texelCount++;
            }

            verticesVisited++;
        }
    }

    nvDebugCheck(cellsVisited == grid.cellArray.count());
    nvDebugCheck(verticesVisited == chartVertexCount);

    vertexMapWidth = ftoi_ceil(sqrtf(float(texelCount)));
    vertexMapWidth = (vertexMapWidth + 3) & ~3;                             // Width aligned to 4.
    vertexMapHeight = vertexMapWidth == 0 ? 0 : (texelCount + vertexMapWidth - 1) / vertexMapWidth;
    //vertexMapHeight = (vertexMapHeight + 3) & ~3;                           // Height aligned to 4.
    nvDebugCheck(vertexMapWidth >= vertexMapHeight);

    nvDebug("Reduced vertex count from %d to %d.\n", chartVertexCount, texelCount);

#if 0
    // This lays down the clustered vertices linearly.
    for (uint i = 0; i < chartVertexCount; i++) {
        HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(i);

        int idx = vertexIndexArray[i];

        vertex->tex.x = float(idx % vertexMapWidth);
        vertex->tex.y = float(idx / vertexMapWidth);
    }
#else
    // Lay down the clustered vertices in morton order.

    Array<uint> texelCodes;
    texelCodes.resize(texelCount);

    // For each texel, assign one morton code.
    uint texelCode = 0;
    for (uint i = 0; i < texelCount; i++) {
        uint x, y;
        do {
            x = decodeMorton2X(texelCode);
            y = decodeMorton2Y(texelCode);
            texelCode++;
        } while (x >= U32(vertexMapWidth) || y >= U32(vertexMapHeight));

        texelCodes[i] = texelCode - 1;
    }

    for (uint i = 0; i < chartVertexCount; i++) {
        HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(i);

        int idx = vertexIndexArray[i];
        if (idx != -1) {
            uint texelCode = texelCodes[idx];
            uint x = decodeMorton2X(texelCode);
            uint y = decodeMorton2Y(texelCode);

            vertex->tex.x = float(x);
            vertex->tex.y = float(y);
        }
    }

#endif
   
#endif

}



static void getBoundaryEdges(HalfEdge::Mesh * mesh, Array<HalfEdge::Edge *> & boundaryEdges)
{
    nvDebugCheck(mesh != NULL);

    const uint edgeCount = mesh->edgeCount();

    BitArray bitFlags(edgeCount);
    bitFlags.clearAll();

    boundaryEdges.clear();

    // Search for boundary edges. Mark all the edges that belong to the same boundary.
    for (uint e = 0; e < edgeCount; e++)
    {
        HalfEdge::Edge * startEdge = mesh->edgeAt(e);

        if (startEdge != NULL && startEdge->isBoundary() && bitFlags.bitAt(e) == false)
        {
            nvDebugCheck(startEdge->face != NULL);
            nvDebugCheck(startEdge->pair->face == NULL);

            startEdge = startEdge->pair;

            const HalfEdge::Edge * edge = startEdge;
            do {
                nvDebugCheck(edge->face == NULL);
                nvDebugCheck(bitFlags.bitAt(edge->id/2) == false);

                bitFlags.setBitAt(edge->id / 2);
                edge = edge->next;
            } while(startEdge != edge);

            boundaryEdges.append(startEdge);
        }
    }
}


bool Chart::closeLoop(uint start, const Array<HalfEdge::Edge *> & loop)
{
    const uint vertexCount = loop.count() - start;

    nvDebugCheck(vertexCount >= 3);
    if (vertexCount < 3) return false;

    nvDebugCheck(loop[start]->vertex->isColocal(loop[start+vertexCount-1]->to()));

    // If the hole is planar, then we add a single face that will be properly triangulated later.
    // If the hole is not planar, we add a triangle fan with a vertex at the hole centroid.
    // This is still a bit of a hack. There surely are better hole filling algorithms out there.

    Array<Vector3> points;
    points.resize(vertexCount);
    for (uint i = 0; i < vertexCount; i++) {
        points[i] = loop[start+i]->vertex->pos;
    }

    bool isPlanar = Fit::isPlanar(vertexCount, points.buffer());

    if (isPlanar) {
        // Add face and connect edges.
        HalfEdge::Face * face = m_unifiedMesh->addFace();
        for (uint i = 0; i < vertexCount; i++) {
            HalfEdge::Edge * edge = loop[start + i];
            
            edge->face = face;
            edge->setNext(loop[start + (i + 1) % vertexCount]);
        }
        face->edge = loop[start];

        nvDebugCheck(face->isValid());
    }
    else {
        // If the polygon is not planar, we just cross our fingers, and hope this will work:

        // Compute boundary centroid:
        Vector3 centroidPos(0);

        for (uint i = 0; i < vertexCount; i++) {
            centroidPos += points[i];
        }

        centroidPos *= (1.0f / vertexCount);

        HalfEdge::Vertex * centroid = m_unifiedMesh->addVertex(centroidPos);

        // Add one pair of edges for each boundary vertex.
        for (uint j = vertexCount-1, i = 0; i < vertexCount; j = i++) {
            HalfEdge::Face * face = m_unifiedMesh->addFace(centroid->id, loop[start+j]->vertex->id, loop[start+i]->vertex->id);
            nvDebugCheck(face != NULL);
        }
    }

    return true;
}


bool Chart::closeHoles()
{
    nvDebugCheck(!m_isVertexMapped);

    Array<HalfEdge::Edge *> boundaryEdges;
    getBoundaryEdges(m_unifiedMesh.ptr(), boundaryEdges);

    uint boundaryCount = boundaryEdges.count();
    if (boundaryCount <= 1)
    {
        // Nothing to close.
        return true;
    }

    // Compute lengths and areas.
    Array<float> boundaryLengths;
    //Array<Vector3> boundaryCentroids;

    for (uint i = 0; i < boundaryCount; i++)
    {
        const HalfEdge::Edge * startEdge = boundaryEdges[i];
        nvCheck(startEdge->face == NULL);

        //float boundaryEdgeCount = 0;
        float boundaryLength = 0.0f;
        //Vector3 boundaryCentroid(zero);

        const HalfEdge::Edge * edge = startEdge;
        do {
            Vector3 t0 = edge->from()->pos;
            Vector3 t1 = edge->to()->pos;

            //boundaryEdgeCount++;
            boundaryLength += length(t1 - t0);
            //boundaryCentroid += edge->vertex()->pos;

            edge = edge->next;
        } while(edge != startEdge);

        boundaryLengths.append(boundaryLength);
        //boundaryCentroids.append(boundaryCentroid / boundaryEdgeCount);
    }


    // Find disk boundary.
    uint diskBoundary = 0;
    float maxLength = boundaryLengths[0];

    for (uint i = 1; i < boundaryCount; i++)
    {
        if (boundaryLengths[i] > maxLength)
        {
            maxLength = boundaryLengths[i];
            diskBoundary = i;
        }
    }


    // Sew holes.
    /*for (uint i = 0; i < boundaryCount; i++)
    {
        if (diskBoundary == i)
        {
            // Skip disk boundary.
            continue;
        }

        HalfEdge::Edge * startEdge = boundaryEdges[i];
        nvCheck(startEdge->face() == NULL);

        boundaryEdges[i] = m_unifiedMesh->sewBoundary(startEdge);
    }

    exportMesh(m_unifiedMesh.ptr(), "debug_sewn.obj");*/

    //bool hasNewHoles = false;

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // @@ Close loop is wrong, after closing a loop, we do not only have to add the face, but make sure that every edge in he loop is pointing to the right place.

    // Close holes.
    for (uint i = 0; i < boundaryCount; i++)
    {
        if (diskBoundary == i)
        {
            // Skip disk boundary.
            continue;
        }

        HalfEdge::Edge * startEdge = boundaryEdges[i];
        nvDebugCheck(startEdge != NULL);
        nvDebugCheck(startEdge->face == NULL);

#if 1
        Array<HalfEdge::Vertex *> vertexLoop;
        Array<HalfEdge::Edge *> edgeLoop;

        HalfEdge::Edge * edge = startEdge;
        do {
            HalfEdge::Vertex * vertex = edge->next->vertex; // edge->to()

            uint i;
            for (i = 0; i < vertexLoop.count(); i++) {
                if (vertex->isColocal(vertexLoop[i])) {
                    break;
                }
            }
            
            bool isCrossing = (i != vertexLoop.count());

            if (isCrossing) {

                HalfEdge::Edge * prev = edgeLoop[i];    // Previous edge before the loop.
                HalfEdge::Edge * next = edge->next;   // Next edge after the loop.

                nvDebugCheck(prev->to()->isColocal(next->from()));

                // Close loop.
                edgeLoop.append(edge);
                closeLoop(i+1, edgeLoop);

                // Link boundary loop.
                prev->setNext(next);
                vertex->setEdge(next);

                // Start over again.
                vertexLoop.clear();
                edgeLoop.clear();
                
                edge = startEdge;
                vertex = edge->to();
            }

            vertexLoop.append(vertex);
            edgeLoop.append(edge);

            edge = edge->next;
        } while(edge != startEdge);

        closeLoop(0, edgeLoop);
#endif

        /*

        // Add face and connect boundary edges.
        HalfEdge::Face * face = m_unifiedMesh->addFace();
        face->setEdge(startEdge);

        HalfEdge::Edge * edge = startEdge;
        do {
            edge->setFace(face);

            edge = edge->next();
        } while(edge != startEdge);

        */


        /*
        uint edgeCount = 0;
        HalfEdge::Edge * edge = startEdge;
        do {
            edgeCount++;
            edge = edge->next();
        } while(edge != startEdge);



        // Count edges in this boundary.
        uint edgeCount = 0;
        HalfEdge::Edge * edge = startEdge;
        do {
            edgeCount++;
            edge = edge->next();
        } while(edge != startEdge);

        // Trivial hole, fill with one triangle. This actually works for all convex boundaries with non colinear vertices.
        if (edgeCount == 3) {
            // Add face and connect boundary edges.
            HalfEdge::Face * face = m_unifiedMesh->addFace();
            face->setEdge(startEdge);

            edge = startEdge;
            do {
                edge->setFace(face);

                edge = edge->next();
            } while(edge != startEdge);

            // @@ Implement the above using addFace, it should now work with existing edges, as long as their face pointers is zero.

        }
        else {
            // Ideally we should:
            // - compute best fit plane of boundary vertices.
            // - project boundary polygon onto plane.
            // - triangulate boundary polygon.
            // - add faces of the resulting triangulation.

            // I don't have a good triangulator available. A more simple solution that works in more (but not all) cases:
            // - compute boundary centroid.
            // - add vertex centroid.
            // - connect centroid vertex with boundary vertices.
            // - connect radial edges with boundary edges.

            // This should work for non-convex boundaries with colinear vertices as long as the kernel of the polygon is not empty.

            // Compute boundary centroid:
            Vector3 centroid_pos(0);
            Vector2 centroid_tex(0);

            HalfEdge::Edge * edge = startEdge;
            do {
                centroid_pos += edge->vertex()->pos;
                centroid_tex += edge->vertex()->tex;
                edge = edge->next();
            } while(edge != startEdge);

            centroid_pos *= (1.0f / edgeCount);
            centroid_tex *= (1.0f / edgeCount);

            HalfEdge::Vertex * centroid = m_unifiedMesh->addVertex(centroid_pos);
            centroid->tex = centroid_tex;

            // Add one pair of edges for each boundary vertex.
            edge = startEdge;
            do {
                HalfEdge::Edge * next = edge->next();

                nvCheck(edge->face() == NULL);
                HalfEdge::Face * face = m_unifiedMesh->addFace(centroid->id(), edge->from()->id(), edge->to()->id());
                
                if (face != NULL) {
                    nvCheck(edge->face() == face);
                }
                else {
                    hasNewHoles = true;
                }

                edge = next;
            } while(edge != startEdge);
        }
        */
    }

    /*nvDebugCheck(!hasNewHoles);

    if (hasNewHoles) {
        // Link boundary again, in case closeHoles created new holes!
        m_unifiedMesh->linkBoundary();
    }*/

    // Because some algorithms do not expect sparse edge buffers.
    //m_unifiedMesh->compactEdges();

    // In case we messed up:
    //m_unifiedMesh->linkBoundary();

    getBoundaryEdges(m_unifiedMesh.ptr(), boundaryEdges);

    boundaryCount = boundaryEdges.count();
    nvDebugCheck(boundaryCount == 1);

    //exportMesh(m_unifiedMesh.ptr(), "debug_hole_filled.obj");

    return boundaryCount == 1;
}


// Transfer parameterization from unified mesh to chart mesh.
void Chart::transferParameterization() {
    nvDebugCheck(!m_isVertexMapped);

    uint vertexCount = m_chartMesh->vertexCount();
    for (uint v = 0; v < vertexCount; v++) {
        HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(v);
        HalfEdge::Vertex * unifiedVertex = m_unifiedMesh->vertexAt(mapChartVertexToUnifiedVertex(v));
        vertex->tex = unifiedVertex->tex;
    }
}

float Chart::computeSurfaceArea() const {
    return nv::computeSurfaceArea(m_chartMesh.ptr()) * scale;
}

float Chart::computeParametricArea() const {
    // This only makes sense in parameterized meshes.
    nvDebugCheck(m_isDisk);            
    nvDebugCheck(!m_isVertexMapped);

    return nv::computeParametricArea(m_chartMesh.ptr());
}

Vector2 Chart::computeParametricBounds() const {
    // This only makes sense in parameterized meshes.
    nvDebugCheck(m_isDisk);
    nvDebugCheck(!m_isVertexMapped);

    Box bounds;
    bounds.clearBounds();

    uint vertexCount = m_chartMesh->vertexCount();
    for (uint v = 0; v < vertexCount; v++) {
        HalfEdge::Vertex * vertex = m_chartMesh->vertexAt(v);
        bounds.addPointToBounds(Vector3(vertex->tex, 0));
    }

    return bounds.extents().xy();
}
