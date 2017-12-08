// This code is in the public domain -- castanyo@yahoo.es

#include "nvmesh.h" // pch

#include "Mesh.h"
#include "Edge.h"
#include "Vertex.h"
#include "Face.h"

#include "nvmesh/TriMesh.h"
#include "nvmesh/QuadTriMesh.h"
#include "nvmesh/MeshBuilder.h"

#include "nvmath/Vector.inl"
#include "nvcore/Array.inl"
#include "nvcore/HashMap.inl"


using namespace nv;
using namespace HalfEdge;

Mesh::Mesh() : m_colocalVertexCount(0)
{
    errorCount = 0;
}

Mesh::Mesh(const Mesh * mesh)
{
    errorCount = 0;

    // Copy mesh vertices.
    const uint vertexCount = mesh->vertexCount();
    m_vertexArray.resize(vertexCount);

    for (uint v = 0; v < vertexCount; v++)
    {
        const Vertex * vertex = mesh->vertexAt(v);
        nvDebugCheck(vertex->id == v);

        m_vertexArray[v] = new Vertex(v);
        m_vertexArray[v]->pos = vertex->pos;
        m_vertexArray[v]->nor = vertex->nor;
        m_vertexArray[v]->tex = vertex->tex;
    }

    m_colocalVertexCount = vertexCount;


    // Copy mesh faces.
    const uint faceCount = mesh->faceCount();

    Array<uint> indexArray(3);

    for (uint f = 0; f < faceCount; f++)
    {
        const Face * face = mesh->faceAt(f);

        for(Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
            const Vertex * vertex = it.current()->from();
            indexArray.append(vertex->id);
        }

        addFace(indexArray);
        indexArray.clear();
    }
}

Mesh::~Mesh()
{
    clear();
}


void Mesh::clear()
{
    deleteAll(m_vertexArray); 
    m_vertexArray.clear();

    foreach(i, m_edgeMap)
    {
        delete m_edgeMap[i].value;
    }
    //deleteAll(m_edgeArray);	// edgeArray only contains 1/2 of the edges!
    m_edgeArray.clear();
    m_edgeMap.clear();

    deleteAll(m_faceArray);
    m_faceArray.clear();
}


Vertex * Mesh::addVertex(const Vector3 & pos)
{
    nvDebugCheck(isFinite(pos));

    Vertex * v = new Vertex(m_vertexArray.count());
    v->pos = pos;
    m_vertexArray.append(v);

    return v;

//    return addVertex(m_vertexArray.count(), pos);
}

/*Vertex * Mesh::addVertex(uint id, const Vector3 & pos)
{
    nvDebugCheck(isFinite(pos));

    Vertex * v = new Vertex(id);
    v->pos = pos;
    m_vertexArray.append(v);

    return v;
}*/

/*void Mesh::addVertices(const Mesh * mesh)
{
nvCheck(mesh != NULL);

// Add mesh vertices
for (uint v = 0; v < vertexCount; v++)
{
const Vertex * vertex = mesh->vertexAt(v);
nvDebugCheck(vertex != NULL);

Vertex * v = addVertex(vertex->pos());
nvDebugCheck(v != NULL);

v->setNor(vertex->nor());
v->setTex(vertex->tex());
}
}*/


/// Link colocal vertices based on geometric location only.
void Mesh::linkColocals()
{
    nvDebug("--- Linking colocals:\n");

    const uint vertexCount = this->vertexCount();
    HashMap<Vector3, Vertex *> vertexMap(vertexCount);

    for (uint v = 0; v < vertexCount; v++)
    {
        Vertex * vertex = vertexAt(v);

        Vertex * colocal;
        if (vertexMap.get(vertex->pos, &colocal))
        {
            colocal->linkColocal(vertex);
        }
        else
        {
            vertexMap.add(vertex->pos, vertex);
        }
    }

    m_colocalVertexCount = vertexMap.count();

    nvDebug("---   %d vertex positions.\n", m_colocalVertexCount);

    // @@ Remove duplicated vertices? or just leave them as colocals?
}

void Mesh::linkColocalsWithCanonicalMap(const Array<uint> & canonicalMap)
{
    nvDebug("--- Linking colocals:\n");

    uint vertexMapSize = 0;
    foreach(i, canonicalMap) {
        vertexMapSize = max(vertexMapSize, canonicalMap[i] + 1);
    }
    
    Array<Vertex *> vertexMap;
    vertexMap.resize(vertexMapSize, NULL);

    m_colocalVertexCount = 0;

    const uint vertexCount = this->vertexCount();
    for (uint v = 0; v < vertexCount; v++)
    {
        Vertex * vertex = vertexAt(v);

        Vertex * colocal = vertexMap[canonicalMap[v]];
        if (colocal != NULL)
        {
            nvDebugCheck(vertex->pos == colocal->pos);
            colocal->linkColocal(vertex);
        }
        else
        {
            vertexMap[canonicalMap[v]] = vertex;
            m_colocalVertexCount++;
        }
    }

    nvDebug("---   %d vertex positions.\n", m_colocalVertexCount);
}


Face * Mesh::addFace()
{
    Face * f = new Face(m_faceArray.count());
    m_faceArray.append(f);
    return f;
}

Face * Mesh::addFace(uint v0, uint v1, uint v2)
{
    Array<uint> indexArray(3);
    indexArray << v0 << v1 << v2;
    return addFace(indexArray, 0, 3);
}

Face * Mesh::addFace(uint v0, uint v1, uint v2, uint v3)
{
    Array<uint> indexArray(4);
    indexArray << v0 << v1 << v2 << v3;
    return addFace(indexArray, 0, 4);
}

Face * Mesh::addFace(const Array<uint> & indexArray)
{
    return addFace(indexArray, 0, indexArray.count());
}


Face * Mesh::addFace(const Array<uint> & indexArray, uint first, uint num)
{
    nvDebugCheck(first < indexArray.count());
    nvDebugCheck(num <= indexArray.count()-first);
    nvDebugCheck(num > 2);

    if (!canAddFace(indexArray, first, num)) {
        errorCount++;
        return NULL;
    }

    Face * f = new Face(m_faceArray.count());

    Edge * firstEdge = NULL;
    Edge * last = NULL;
    Edge * current = NULL;

    for(uint i = 0; i < num-1; i++)
    {
        current = addEdge(indexArray[first+i], indexArray[first+i+1]);
        nvCheck(current != NULL && current->face == NULL);

        current->face = f;

        if (last != NULL) last->setNext(current);
        else firstEdge = current;

        last = current;
    }

    current = addEdge(indexArray[first+num-1], indexArray[first]);
    nvCheck(current != NULL && current->face == NULL);

    current->face = f;

    last->setNext(current);
    current->setNext(firstEdge);

    f->edge = firstEdge;
    m_faceArray.append(f);

    return f;
}

/*void Mesh::addFaces(const Mesh * mesh)
{
nvCheck(mesh != NULL);

Array indexArray;
// Add faces

}*/


// Return true if the face can be added to the manifold mesh.
bool Mesh::canAddFace(const Array<uint> & indexArray, uint first, uint num) const
{
    for (uint j = num - 1, i = 0; i < num; j = i++) {
        if (!canAddEdge(indexArray[first+j], indexArray[first+i])) {
            errorIndex0 = indexArray[first+j];
            errorIndex1 = indexArray[first+i];
            return false;
        }
    }

    // We also have to make sure the face does not have any duplicate edge!
    for (uint i = 0; i < num; i++) {

        int i0 = indexArray[first + i + 0];
        int i1 = indexArray[first + (i + 1)%num];

        for (uint j = i + 1; j < num; j++) {
            int j0 = indexArray[first + j + 0];
            int j1 = indexArray[first + (j + 1)%num];

            if (i0 == j0 && i1 == j1) {
                return false;
            }
        }
    }

    return true;
}

// Return true if the edge doesn't exist or doesn't have any adjacent face. 
bool Mesh::canAddEdge(uint i, uint j) const
{
    if (i == j) {
        // Skip degenerate edges.
        return false;
    }

    // Same check, but taking into account colocal vertices.
    const Vertex * v0 = vertexAt(i);
    const Vertex * v1 = vertexAt(j);

    for(Vertex::ConstVertexIterator it(v0->colocals()); !it.isDone(); it.advance())
    {
        if (it.current() == v1)
        {
            // Skip degenerate edges.
            return false;
        }
    }

    // Make sure edge has not been added yet.
    Edge * edge = findEdge(i, j);

    return edge == NULL || edge->face == NULL; // We ignore edges that don't have an adjacent face yet, since this face could become the edge's face.
}

Edge * Mesh::addEdge(uint i, uint j)
{
    nvCheck(i != j);

    Edge * edge = findEdge(i, j);

    if (edge != NULL) {
        // Edge may already exist, but its face must not be set.
        nvDebugCheck(edge->face == NULL);

        // Nothing else to do!

    }
    else {
        // Add new edge.

        // Lookup pair.
        Edge * pair = findEdge(j, i);

        if (pair != NULL)
        {
            // Create edge with same id.
            edge = new Edge(pair->id + 1);

            // Link edge pairs.
            edge->pair = pair;
            pair->pair = edge;

            // @@ I'm not sure this is necessary!
            pair->vertex->setEdge(pair);
        }
        else
        {
            // Create edge.
            edge = new Edge(2*m_edgeArray.count());

            // Add only unpaired edges.
            m_edgeArray.append(edge);
        }

        edge->vertex = m_vertexArray[i];
        m_edgeMap.add(Key(i,j), edge);
    }

    // Face and Next are set by addFace.

    return edge;
}


/// Find edge, test all colocals.
Edge * Mesh::findEdge(uint i, uint j) const
{
    Edge * edge = NULL;

    const Vertex * v0 = vertexAt(i);
    const Vertex * v1 = vertexAt(j);

    // Test all colocal pairs.
    for(Vertex::ConstVertexIterator it0(v0->colocals()); !it0.isDone(); it0.advance())
    {
        for(Vertex::ConstVertexIterator it1(v1->colocals()); !it1.isDone(); it1.advance())
        {
            Key key(it0.current()->id, it1.current()->id);

            if (edge == NULL) {
                m_edgeMap.get(key, &edge);
#if !defined(_DEBUG)
                if (edge != NULL) return edge;
#endif
            }
            else {
                // Make sure that only one edge is found.
                nvDebugCheck(!m_edgeMap.get(key));
            }
        }
    }

    return edge;
}

/// Link boundary edges once the mesh has been created.
void Mesh::linkBoundary()
{
    nvDebug("--- Linking boundaries:\n");

    int num = 0;

    // Create boundary edges.
    uint edgeCount = this->edgeCount();
    for(uint e = 0; e < edgeCount; e++)
    {
        Edge * edge = edgeAt(e);
        if (edge != NULL && edge->pair == NULL) {
            Edge * pair = new Edge(edge->id + 1);

            uint i = edge->from()->id;
            uint j = edge->next->from()->id;

            Key key(j,i);
            nvCheck(!m_edgeMap.get(key));

            pair->vertex = m_vertexArray[j];
            m_edgeMap.add(key, pair);

            edge->pair = pair;
            pair->pair = edge;

            num++;
        }
    }

    // Link boundary edges.
    for (uint e = 0; e < edgeCount; e++) {
        Edge * edge = edgeAt(e);
        if (edge != NULL && edge->pair->face == NULL) {
            linkBoundaryEdge(edge->pair);
        }
    }

    nvDebug("---   %d boundary edges.\n", num);
}

/// Link this boundary edge.
void Mesh::linkBoundaryEdge(Edge * edge)
{
    nvCheck(edge->face == NULL);

    // Make sure next pointer has not been set. @@ We want to be able to relink boundary edges after mesh changes.
    //nvCheck(edge->next() == NULL);

    Edge * next = edge;
    while(next->pair->face != NULL) {
        // Get pair prev
        Edge * e = next->pair->next;
        while (e->next != next->pair) {
            e = e->next;
        }
        next = e;
    }
    edge->setNext(next->pair);

    // Adjust vertex edge, so that it's the boundary edge. (required for isBoundary())
    if (edge->vertex->edge != edge)
    {
        // Multiple boundaries in the same edge.
        //nvCheck( edge->vertex()->edge() == NULL || edge->vertex()->edge()->face() != NULL );
        edge->vertex->edge = edge;
    }
}


/// Convert to tri mesh.
TriMesh * Mesh::toTriMesh() const
{
    uint triangleCount = 0;

    // Count triangle faces.
    const uint faceCount = this->faceCount();
    for(uint f = 0; f < faceCount; f++)
    {
        const Face * face = faceAt(f);
        triangleCount += face->edgeCount() - 2;
    }

    TriMesh * triMesh = new TriMesh(triangleCount, vertexCount());

    // Add vertices.
    Array<TriMesh::Vertex> & vertices = triMesh->vertices();

    const uint vertexCount = this->vertexCount();
    for(uint v = 0; v < vertexCount; v++)
    {
        const Vertex * vertex = vertexAt(v);

        TriMesh::Vertex triVertex;
        triVertex.id = vertices.count();
        triVertex.pos = vertex->pos;
        triVertex.nor = vertex->nor;
        triVertex.tex = vertex->tex;

        vertices.append(triVertex);
    }

    // Add triangles.
    Array<TriMesh::Face> & triangles = triMesh->faces();

    for(uint f = 0; f < faceCount; f++)
    {
        const Face * face = faceAt(f);

        // @@ Triangulate arbitrary polygons correctly.
        const uint v0 = face->edge->vertex->id;
        uint v1 = face->edge->next->vertex->id;

        for(Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            uint v2 = it.current()->vertex->id;

            // Skip the first two vertices.
            if (v2 == v0 || v2 == v1) continue;

            TriMesh::Face triangle;
            triangle.id = triangles.count();
            triangle.v[0] = v0;
            triangle.v[1] = v1;
            triangle.v[2] = v2;

            v1 = v2;

            triangles.append(triangle);
        }
    }

    return triMesh;
}

QuadTriMesh * Mesh::toQuadTriMesh() const
{
    MeshBuilder builder;

    const uint vertexCount = this->vertexCount();
    builder.hintVertexCount(vertexCount);

    for(uint v = 0; v < vertexCount; v++)
    {
        const Vertex * vertex = vertexAt(v);

        builder.addPosition(vertex->pos);
        builder.addNormal(vertex->nor);
        builder.addTexCoord(vertex->tex);
    }

    const uint faceCount = this->faceCount();
    builder.hintTriangleCount(faceCount);

    for(uint f = 0; f < faceCount; f++)
    {
        const Face * face = faceAt(f);

        builder.beginPolygon();

        for(Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            uint v = it.current()->vertex->id;
            builder.addVertex(v, v, v);
        }

        builder.endPolygon();
    }

    builder.done();

    return builder.buildQuadTriMesh();
}


// Triangulate in place.
void Mesh::triangulate() {

    bool all_triangles = true;

    const uint faceCount = m_faceArray.count();
    for (uint f = 0; f < faceCount; f++) {
        Face * face = m_faceArray[f];
        if (face->edgeCount() != 3) {
            all_triangles = false;
            break;
        }
    }

    if (all_triangles) {
        return;
    }


    // Do not touch vertices, but rebuild edges and faces.
    Array<Edge *> edgeArray;
    Array<Face *> faceArray;

    swap(edgeArray, m_edgeArray);
    swap(faceArray, m_faceArray);
    m_edgeMap.clear();

    for (uint f = 0; f < faceCount; f++) {
        Face * face = faceArray[f];

        // Trivial fan-like triangulation.
        const uint v0 = face->edge->vertex->id;
        uint v2, v1 = -1;

        for (Face::EdgeIterator it(face->edges()); !it.isDone(); it.advance()) {
            Edge * edge = it.current();
            v2 = edge->to()->id;
            if (v2 == v0) break;
            if (v1 != -1) addFace(v0, v1, v2);
            v1 = v2;
        }
    }

    nvDebugCheck(m_faceArray.count() > faceCount); // triangle count > face count

    linkBoundary();

    deleteAll(edgeArray);
    deleteAll(faceArray);
}


/*
Fixing T-junctions.

- Find T-junctions. Find  vertices that are on an edge. 
    - This test is approximate.
    - Insert edges on a spatial index to speedup queries.
    - Consider only open edges, that is edges that have no pairs.
    - Consider only vertices on boundaries.
- Close T-junction.
    - Split edge.

*/
bool Mesh::splitBoundaryEdges() {
    
    Array<Vertex *> boundaryVertices;

    foreach(i, m_vertexArray) {
        Vertex * v = m_vertexArray[i];
        if (v->isBoundary()) {
            boundaryVertices.append(v);
        }
    }

    nvDebug("Fixing T-junctions:\n");

    int splitCount = 0;

    foreach(v, boundaryVertices) {
        Vertex * vertex = boundaryVertices[v];

        Vector3 x0 = vertex->pos;

        // Find edges that this vertex overlaps with.
        foreach(e, m_edgeArray) {
        //for (uint e = 0; e < m_edgeArray.count(); e++) {
            Edge * edge = m_edgeArray[e];
            if (edge != NULL && edge->isBoundary()) {

                if (edge->from() == vertex || edge->to() == vertex) {
                    continue;
                }

                Vector3 x1 = edge->from()->pos;
                Vector3 x2 = edge->to()->pos;

                Vector3 v01 = x0 - x1;
                Vector3 v21 = x2 - x1;

                float l = length(v21);
                float d = length(cross(v01, v21)) / l;

                if (isZero(d)) {
                    float t = dot(v01, v21) / (l * l);

                    // @@ Snap x0 to x1 or x2, if too close? No, do vertex snapping elsewhere.
                    /*if (equal(t, 0.0f, 0.01f)) {
                        //vertex->setPos(x1);
                    }
                    else if (equal(t, 1.0f, 0.01f)) {
                        //vertex->setPos(x2);
                    }
                    else*/
                    if (t > 0.0f + NV_EPSILON && t < 1.0f - NV_EPSILON) {
                        nvDebugCheck(equal(lerp(x1, x2, t), x0));

                        Vertex * splitVertex = splitBoundaryEdge(edge, t, x0);
                        vertex->linkColocal(splitVertex);   // @@ Should we do this here?
                        splitCount++;
                    }
                }
            }
        }
    }

    nvDebug(" - %d edges split.\n", splitCount);

    nvDebugCheck(isValid());

    return splitCount != 0;
}


// For this to be effective, we have to fix the boundary junctions first.
Edge * Mesh::sewBoundary(Edge * startEdge) {
    nvDebugCheck(startEdge->face == NULL);

    // @@ We may want to be more conservative linking colocals in order to preserve the input topology. One way of doing that is by linking colocals only 
    // if the vertices next to them are linked as well. That is, by sewing boundaries after detecting them. If any pair of consecutive edges have their first
    // and last vertex in the same position, then it can be linked.

    Edge * lastBoundarySeen = startEdge;

    nvDebug("Sewing Boundary:\n");

    int count = 0;
    int sewnCount = 0;

    Edge * edge = startEdge;
    do {
        nvDebugCheck(edge->face == NULL);

        Edge * edge_a = edge;
        Edge * edge_b = edge->prev;

        Edge * pair_a = edge_a->pair;
        Edge * pair_b = edge_b->pair;

        Vertex * v0a = edge_a->to();
        Vertex * v0b = edge_b->from();
        Vertex * v1a = edge_a->from();
        Vertex * v1b = edge_b->to();

        nvDebugCheck(v1a->isColocal(v1b));

        /*
        v0b +      _+ v0a
             \     /
            b \   / a
               \|/
            v1b + v1a
        */

        // @@ This should not happen while sewing, but it may be produced somewhere else.
        nvDebugCheck(edge_a != edge_b);

        if (v0a->pos == v0b->pos) {

            // Link vertices.
            v0a->linkColocal(v0b);
            
            // Remove edges to be collapsed.
            disconnect(edge_a);
            disconnect(edge_b);
            disconnect(pair_a);
            disconnect(pair_b);

            // Link new boundary edges.
            Edge * prevBoundary = edge_b->prev;
            Edge * nextBoundary = edge_a->next;
            if (nextBoundary != NULL) {
                nvDebugCheck(nextBoundary->face == NULL);
                nvDebugCheck(prevBoundary->face == NULL);
                nextBoundary->setPrev(prevBoundary);
            
                // Make sure boundary vertex points to boundary edge.
                v0a->setEdge(nextBoundary); // This updates all colocals.
            }
            lastBoundarySeen = prevBoundary;

            // Creat new edge.
            Edge * newEdge_a = addEdge(v0a->id, v1a->id);   // pair_a->from()->id, pair_a->to()->id
            Edge * newEdge_b = addEdge(v1b->id, v0b->id);

            newEdge_a->pair = newEdge_b;
            newEdge_b->pair = newEdge_a;

            newEdge_a->face = pair_a->face;
            newEdge_b->face = pair_b->face;

            newEdge_a->setNext(pair_a->next);
            newEdge_a->setPrev(pair_a->prev);

            newEdge_b->setNext(pair_b->next);
            newEdge_b->setPrev(pair_b->prev);

            delete edge_a;
            delete edge_b;
            delete pair_a;
            delete pair_b;

            edge = nextBoundary;    // If nextBoundary is NULL we have closed the loop.
            sewnCount++;
        }
        else {
            edge = edge->next;
        }
        
        count++;
    } while(edge != NULL && edge != lastBoundarySeen);

    nvDebug(" - Sewn %d out of %d.\n", sewnCount, count);

    if (lastBoundarySeen != NULL) {
        nvDebugCheck(lastBoundarySeen->face == NULL);
    }

    return lastBoundarySeen;
}


// @@ We must always disconnect edge pairs simultaneously.
void Mesh::disconnect(Edge * edge) {
    nvDebugCheck(edge != NULL);

    // Remove from edge list.
    if ((edge->id & 1) == 0) {
        nvDebugCheck(m_edgeArray[edge->id / 2] == edge);
        m_edgeArray[edge->id / 2] = NULL;
    }

    // Remove edge from map. @@ Store map key inside edge?
    nvDebugCheck(edge->from() != NULL && edge->to() != NULL);
    bool removed = m_edgeMap.remove(Key(edge->from()->id, edge->to()->id));
    nvDebugCheck(removed == true);

    // Disconnect from vertex.
    if (edge->vertex != NULL) {
        if (edge->vertex->edge == edge) {
            if (edge->prev && edge->prev->pair) {
                edge->vertex->edge = edge->prev->pair;
            }
            else if (edge->pair && edge->pair->next) {
                edge->vertex->edge = edge->pair->next;
            }
            else {
                edge->vertex->edge = NULL;
                // @@ Remove disconnected vertex?
            }
        }
        //edge->setVertex(NULL);
    }

    // Disconnect from face.
    if (edge->face != NULL) {
        if (edge->face->edge == edge) {
            if (edge->next != NULL && edge->next != edge) {
                edge->face->edge = edge->next;
            }
            else if (edge->prev != NULL && edge->prev != edge) {
                edge->face->edge = edge->prev;
            }
            else {
                edge->face->edge = NULL;
                // @@ Remove disconnected face?
            }
        }
        //edge->setFace(NULL);
    }

    // @@ Hack, we don't disconnect from pair, because pair needs us to remove itself from the map.
    // Disconect from pair.
    /*if (edge->pair != NULL) {
        if (edge->pair->pair == edge) {
            edge->pair->setPair(NULL);
        }
        //edge->setPair(NULL);
    }*/

    // Disconnect from previous.
    if (edge->prev) {
        if (edge->prev->next == edge) {
            edge->prev->setNext(NULL);
        }
        //edge->setPrev(NULL);
    }

    // Disconnect from next.
    if (edge->next) {
        if (edge->next->prev == edge) {
            edge->next->setPrev(NULL);
        }
        //edge->setNext(NULL);
    }
}


void Mesh::remove(Edge * edge) {
    nvDebugCheck(edge != NULL);

    disconnect(edge);

    delete edge;
}

void Mesh::remove(Vertex * vertex) {
    nvDebugCheck(vertex != NULL);

    // Remove from vertex list.
    m_vertexArray[vertex->id] = NULL;

    // Disconnect from colocals.
    vertex->unlinkColocal();

    // Disconnect from edges.
    if (vertex->edge != NULL) {
        // @@ Removing a connected vertex is asking for trouble...
        if (vertex->edge->vertex == vertex) {
            // @@ Connect edge to a colocal?
            vertex->edge->vertex = NULL;
        }

        vertex->setEdge(NULL);
    }

    delete vertex;
}

void Mesh::remove(Face * face) {
    nvDebugCheck(face != NULL);

    // Remove from face list.
    m_faceArray[face->id] = NULL;

    // Disconnect from edges.
    if (face->edge != NULL) {
        nvDebugCheck(face->edge->face == face);

        face->edge->face = NULL;

        face->edge = NULL;
    }

    delete face;
}


void Mesh::compactEdges() {
    const uint edgeCount = m_edgeArray.count();

    uint c = 0;
    for (uint i = 0; i < edgeCount; i++) {
        if (m_edgeArray[i] != NULL) {
            if (i != c) {
                m_edgeArray[c] = m_edgeArray[i];
                m_edgeArray[c]->id = 2 * c;
                if (m_edgeArray[c]->pair != NULL) {
                    m_edgeArray[c]->pair->id = 2 * c + 1;
                }
            }
            c++;
        }
    }

    m_edgeArray.resize(c);
}


void Mesh::compactVertices() {
    const uint vertexCount = m_vertexArray.count();

    uint c = 0;
    for (uint i = 0; i < vertexCount; i++) {
        if (m_vertexArray[i] != NULL) {
            if (i != c) {
                m_vertexArray[c] = m_vertexArray[i];
                m_vertexArray[c]->id = c;
            }
            c++;
        }
    }

    m_vertexArray.resize(c);

    // @@ Generate xref array for external attributes.
}


void Mesh::compactFaces() {
    const uint faceCount = m_faceArray.count();

    uint c = 0;
    for (uint i = 0; i < faceCount; i++) {
        if (m_faceArray[i] != NULL) {
            if (i != c) {
                m_faceArray[c] = m_faceArray[i];
                m_faceArray[c]->id = c;
            }
            c++;
        }
    }

    m_faceArray.resize(c);
}


Vertex * Mesh::splitBoundaryEdge(Edge * edge, float t, const Vector3 & pos) {

    /*
      We want to go from this configuration:
           
            +   +
            |   ^
       edge |<->|  pair
            v   |
            +   +
      
      To this one:

            +   +
            |   ^
         e0 |<->| p0
            v   |
     vertex +   + 
            |   ^
         e1 |<->| p1
            v   |
            +   +

    */


    Edge * pair = edge->pair;

    // Make sure boundaries are linked.
    nvDebugCheck(pair != NULL); 

    // Make sure edge is a boundary edge.
    nvDebugCheck(pair->face == NULL);

    // Add new vertex.
    Vertex * vertex = addVertex(pos);
    vertex->nor = lerp(edge->from()->nor, edge->to()->nor, t);
    vertex->tex = lerp(edge->from()->tex, edge->to()->tex, t);
    vertex->col = lerp(edge->from()->col, edge->to()->col, t);

    disconnect(edge);
    disconnect(pair);

    // Add edges.
    Edge * e0 = addEdge(edge->from()->id, vertex->id);
    Edge * p0 = addEdge(vertex->id, pair->to()->id);

    Edge * e1 = addEdge(vertex->id, edge->to()->id);
    Edge * p1 = addEdge(pair->from()->id, vertex->id);

    // Link edges.
    e0->setNext(e1);
    p1->setNext(p0);

    e0->setPrev(edge->prev);
    e1->setNext(edge->next);

    p1->setPrev(pair->prev);
    p0->setNext(pair->next);

    nvDebugCheck(e0->next == e1);
    nvDebugCheck(e1->prev == e0);

    nvDebugCheck(p1->next == p0);
    nvDebugCheck(p0->prev == p1);

    nvDebugCheck(p0->pair == e0);
    nvDebugCheck(e0->pair == p0);

    nvDebugCheck(p1->pair == e1);
    nvDebugCheck(e1->pair == p1);

    // Link faces.
    e0->face = edge->face;
    e1->face = edge->face;

    // Link vertices.
    edge->from()->setEdge(e0);
    vertex->setEdge(e1);

    delete edge;
    delete pair;

    return vertex;
}

#if 0
// Without introducing new vertices.
void Mesh::splitBoundaryEdge(Edge * edge, Vertex * vertex) {

    /*
      We want to go from this configuration:

            |   | pn
            +   +
            |   ^
            |   |
       edge |<->| pair
            |   |
            v   |
            +   +
            |   | pp
      
      To this one:
          \       /
           \     /
            +   +
            |   ^
         e0 |<->| p0
            v   |
     vertex +   + 
            |   ^
         e1 |<->| p1
            v   |
            +   +
           /     \
          /       \
    */


    Edge * pair = edge->pair;
    Edge * pn = pair->next();
    Edge * pp = pair->prev();

    // Make sure boundaries are linked.
    nvDebugCheck(pair != NULL);

    // Make sure edge is a boundary edge.
    nvDebugCheck(pair->face() == NULL);

    nvDebugCheck(edge->isValid());
    nvDebugCheck(pair->isValid());

    disconnect(edge);
    disconnect(pair);

    // Add edges.
    Edge * e0 = addEdge(edge->from()->id(), vertex->id());
    Edge * e1 = addEdge(vertex->id(), edge->to()->id());

    // Link faces.
    e0->setFace(edge->face());
    e1->setFace(edge->face());

    // Link pairs.
    Edge * p0 = findEdge(vertex->id(), pair->to()->id());
    if (p0 == NULL) {
        p0 = addEdge(vertex->id(), pair->to()->id());
        pn->setPrev(p0);
    }
    else {
        nvDebugCheck(p0->face() != NULL);
        if (e0->prev() != NULL) {
            pn->setPrev(e0->prev());
        }
        else {
            nvDebugCheck(pn == e0);
        }
    }
    
    Edge * p1 = findEdge(pair->from()->id(), vertex->id());
    if (p1 == NULL) {
        p1 = addEdge(pair->from()->id(), vertex->id());
        pp->setNext(p1);
    }
    else {
        nvDebugCheck(p1->face() != NULL);
        if (e1->next() != NULL) {
            pp->setPrev(e1->next());
        }
        else {
            nvDebugCheck(pp == e1);
        }
    }

    // Link edges.
    e0->setNext(e1); // e1->setPrev(e0)

    if (p0->face() == p1->face()) { // can be null
        p1->setNext(p0); // p0->setPrev(p1)
    }
    else {
        //if (p1->face() == NULL) p1->setNext(
    }
    

    e0->setPrev(edge->prev());
    e1->setNext(edge->next());

    nvDebugCheck(e0->pair == p0);
    nvDebugCheck(e1->pair == p1);
    nvDebugCheck(p0->pair == e0);
    nvDebugCheck(p1->pair == e1);

    nvDebugCheck(e0->isValid());
    nvDebugCheck(e1->isValid());
    nvDebugCheck(pp->isValid());
    nvDebugCheck(pn->isValid());

    nvDebugCheck(e0->pair->isValid());
    nvDebugCheck(e1->pair->isValid());
    nvDebugCheck(pp->pair->isValid());
    nvDebugCheck(pn->pair->isValid());

    nvDebugCheck(edge->face->isValid());

    if (pn->pair->face != NULL) {
        nvDebugCheck(pn->pair->face->isValid());
    }

    if (pp->pair->face() != NULL) {
        nvDebugCheck(pn->pair->face->isValid());
    }

    if (p0->face != NULL) {
        nvDebugCheck(p0->face->isValid());
    }

    if (p1->face() != NULL) {
        nvDebugCheck(p1->face()->isValid());
    }

    nvDebugCheck(isValid()); // Only for extreme debugging.

    // Link vertices.
    edge->from()->setEdge(e0);
    vertex->setEdge(p0);

    delete edge;
    delete pair;
}
#endif

bool Mesh::isValid() const
{
    // Make sure all edges are valid.
    const uint edgeCount = m_edgeArray.count();
    for (uint e = 0; e < edgeCount; e++) {
        Edge * edge = m_edgeArray[e];
        if (edge != NULL) {
            if (edge->id != 2*e) {
                return false;
            }
            if (!edge->isValid()) {
                return false;
            }

            if (edge->pair->id != 2*e+1) {
                return false;
            }
            if (!edge->pair->isValid()) {
                return false;
            }
        }
    }

    // @@ Make sure all faces are valid.

    // @@ Make sure all vertices are valid.

    return true;
}
