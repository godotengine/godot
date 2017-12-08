// This code is in the public domain -- castanyo@yahoo.es

#include "nvmesh.h" // pch

#include "MeshBuilder.h"
#include "TriMesh.h"
#include "QuadTriMesh.h"
#include "halfedge/Mesh.h"
#include "halfedge/Vertex.h"
#include "halfedge/Face.h"

#include "weld/Weld.h"

#include "nvmath/Box.h"
#include "nvmath/Vector.inl"

#include "nvcore/StrLib.h"
#include "nvcore/RadixSort.h"
#include "nvcore/Ptr.h"
#include "nvcore/Array.inl"
#include "nvcore/HashMap.inl"


using namespace nv;

/*
By default the mesh builder creates 3 streams (position, normal, texcoord), I'm planning to add support for extra streams as follows:

enum StreamType { StreamType_Float, StreamType_Vector2, StreamType_Vector3, StreamType_Vector4 };

uint addStream(const char *, uint idx, StreamType);

uint addAttribute(float)
uint addAttribute(Vector2)
uint addAttribute(Vector3)
uint addAttribute(Vector4)

struct Vertex
{
    uint pos;
    uint nor;
    uint tex;
    uint * attribs;	// NULL or NIL terminated array?
};

All streams must be added before hand, so that you know the size of the attribs array.

The vertex hash function could be kept as is, but the == operator should be extended to test 
the extra atributes when available.

That might require a custom hash implementation, or an extension of the current one. How to
handle the variable number of attributes in the attribs array?

bool operator()(const Vertex & a, const Vertex & b) const
{ 
    if (a.pos != b.pos || a.nor != b.nor || a.tex != b.tex) return false;
    if (a.attribs == NULL && b.attribs == NULL) return true;
    return 0 == memcmp(a.attribs, b.attribs, ???);
}

We could use a NIL terminated array, or provide custom user data to the equals functor.

vertexMap.setUserData((void *)vertexAttribCount);

bool operator()(const Vertex & a, const Vertex & b, void * userData) const { ... }

*/



namespace 
{
    struct Material
    {
        Material() : faceCount(0) {}
        Material(const String & str) : name(str), faceCount(0) {}

        String name;
        uint faceCount;
    };

    struct Vertex
    {
        //Vertex() {}
        //Vertex(uint p, uint n, uint t0, uint t1, uint c) : pos(p), nor(n), tex0(t0), tex1(t1), col(c) {}

        friend bool operator==(const Vertex & a, const Vertex & b)
        {
            return a.pos == b.pos && a.nor == b.nor && a.tex[0] == b.tex[0] && a.tex[1] == b.tex[1] && a.col[0] == b.col[0] && a.col[1] == b.col[1] && a.col[2] == b.col[2];
        }

        uint pos;
        uint nor;
        uint tex[2];
        uint col[3];
    };

    struct Face
    {
        uint id;
        uint firstIndex;
        uint indexCount;
        uint material;
        uint group;
    };

} // namespace


namespace nv
{
    // This is a much better hash than the default and greatly improves performance!
    template <> struct Hash<Vertex>
    {
        uint operator()(const Vertex & v) const { return v.pos + v.nor + v.tex[0]/* + v.col*/; }
    };
}

struct MeshBuilder::PrivateData
{
    PrivateData() : currentGroup(NIL), currentMaterial(NIL), maxFaceIndexCount(0) {}

    uint pushVertex(uint p, uint n, uint t0, uint t1, uint c0, uint c1, uint c2);
    uint pushVertex(const Vertex & v);

    Array<Vector3> posArray;
    Array<Vector3> norArray;
    Array<Vector2> texArray[2];
    Array<Vector4> colArray[3];

    Array<Vertex> vertexArray;
    HashMap<Vertex, uint> vertexMap;

    HashMap<String, uint> materialMap;
    Array<Material> materialArray;

    uint currentGroup;
    uint currentMaterial;

    Array<uint> indexArray;
    Array<Face> faceArray;

    uint maxFaceIndexCount;
};


uint MeshBuilder::PrivateData::pushVertex(uint p, uint n, uint t0, uint t1, uint c0, uint c1, uint c2)
{
    Vertex v;
    v.pos = p;
    v.nor = n;
    v.tex[0] = t0;
    v.tex[1] = t1;
    v.col[0] = c0;
    v.col[1] = c1;
    v.col[2] = c2;
    return pushVertex(v);
}

uint MeshBuilder::PrivateData::pushVertex(const Vertex & v)
{
    // Lookup vertex v in map.
    uint idx;
    if (vertexMap.get(v, &idx))
    {
        return idx;
    }

    idx = vertexArray.count();
    vertexArray.pushBack(v);
    vertexMap.add(v, idx);

    return idx;
}


MeshBuilder::MeshBuilder() : d(new PrivateData())
{
}

MeshBuilder::~MeshBuilder()
{
    nvDebugCheck(d != NULL);
    delete d;
}


// Builder methods.
uint MeshBuilder::addPosition(const Vector3 & v)
{
    d->posArray.pushBack(validate(v));
    return d->posArray.count() - 1;
}

uint MeshBuilder::addNormal(const Vector3 & v)
{
    d->norArray.pushBack(validate(v));
    return d->norArray.count() - 1;
}

uint MeshBuilder::addTexCoord(const Vector2 & v, uint set/*=0*/)
{
    d->texArray[set].pushBack(validate(v));
    return d->texArray[set].count() - 1;
}

uint MeshBuilder::addColor(const Vector4 & v, uint set/*=0*/)
{
    d->colArray[set].pushBack(validate(v));
    return d->colArray[set].count() - 1;
}

void MeshBuilder::beginGroup(uint id)
{
    d->currentGroup = id;
}

void MeshBuilder::endGroup()
{
    d->currentGroup = NIL;
}

// Add named material, check for uniquenes.
uint MeshBuilder::addMaterial(const char * name)
{
    uint index;
    if (d->materialMap.get(name, &index)) {
        nvDebugCheck(d->materialArray[index].name == name);
    }
    else {
        index = d->materialArray.count();
        d->materialMap.add(name, index);
        
        Material material(name);
        d->materialArray.append(material);
    }
    return index;
}

void MeshBuilder::beginMaterial(uint id)
{
    d->currentMaterial = id;
}

void MeshBuilder::endMaterial()
{
    d->currentMaterial = NIL;
}

void MeshBuilder::beginPolygon(uint id/*=0*/)
{
    Face face;
    face.id = id;
    face.firstIndex = d->indexArray.count();
    face.indexCount = 0;
    face.material = d->currentMaterial;
    face.group = d->currentGroup;

    d->faceArray.pushBack(face);
}

uint MeshBuilder::addVertex(uint p, uint n/*= NIL*/, uint t0/*= NIL*/, uint t1/*= NIL*/, uint c0/*= NIL*/, uint c1/*= NIL*/, uint c2/*= NIL*/)
{
    // @@ In theory there's no need to add vertices before faces, but I'm adding this to debug problems in our maya exporter:
    nvDebugCheck(p < d->posArray.count());
    nvDebugCheck(n == NIL || n < d->norArray.count());
    nvDebugCheck(t0 == NIL || t0 < d->texArray[0].count());
    nvDebugCheck(t1 == NIL || t1 < d->texArray[1].count());
    //nvDebugCheck(c0 == NIL || c0 < d->colArray[0].count());
    if (c0 > d->colArray[0].count()) c0 = NIL;    // @@ This seems to be happening in loc_swamp_catwalk.mb! No idea why.
    nvDebugCheck(c1 == NIL || c1 < d->colArray[1].count());
    nvDebugCheck(c2 == NIL || c2 < d->colArray[2].count());

    uint idx = d->pushVertex(p, n, t0, t1, c0, c1, c2);
    d->indexArray.pushBack(idx);
    d->faceArray.back().indexCount++;
    return idx;
}

uint MeshBuilder::addVertex(const Vector3 & pos)
{
    uint p = addPosition(pos);
    return addVertex(p);
}

#if 0
uint MeshBuilder::addVertex(const Vector3 & pos, const Vector3 & nor, const Vector2 & tex0, const Vector2 & tex1, const Vector4 & col0, const Vector4 & col1)
{
    uint p = addPosition(pos);
    uint n = addNormal(nor);
    uint t0 = addTexCoord(tex0, 0);
    uint t1 = addTexCoord(tex1, 1);
    uint c0 = addColor(col0);
    uint c1 = addColor(col1);
    return addVertex(p, n, t0, t1, c0, c1);
}
#endif

// Return true if the face is valid and was added to the mesh.
bool MeshBuilder::endPolygon()
{
    const Face & face = d->faceArray.back();
    const uint count = face.indexCount;

    // Validate polygon here.
    bool invalid = count <= 2;

    if (!invalid) {
        // Skip zero area polygons. Or polygons with degenerate edges (which will result in zero-area triangles).
        const uint first = face.firstIndex;
        for (uint j = count - 1, i = 0; i < count; j = i, i++) {
            uint v0 = d->indexArray[first + i];
            uint v1 = d->indexArray[first + j];

            uint p0 = d->vertexArray[v0].pos;
            uint p1 = d->vertexArray[v1].pos;

            if (p0 == p1) {
                invalid = true;
                break;
            }

            if (equal(d->posArray[p0], d->posArray[p1], FLT_EPSILON)) {
                invalid = true;
                break;
            }
        }

        uint v0 = d->indexArray[first];
        uint p0 = d->vertexArray[v0].pos;
        Vector3 x0 = d->posArray[p0];

        float area = 0.0f;
        for (uint j = 1, i = 2; i < count; j = i, i++) {
            uint v1 = d->indexArray[first + i];
            uint v2 = d->indexArray[first + j];

            uint p1 = d->vertexArray[v1].pos;
            uint p2 = d->vertexArray[v2].pos;

            Vector3 x1 = d->posArray[p1];
            Vector3 x2 = d->posArray[p2];

            area += length(cross(x1-x0, x2-x0));
        }

        if (0.5 * area < 1e-6) {    // Reduce this threshold if artists have legitimate complains.
            invalid = true;
        }

        // @@ This is not complete. We may still get zero area triangles after triangulation.
        // However, our plugin triangulates before building the mesh, so hopefully that's not a problem.

    }

    if (invalid)
    {
        d->indexArray.resize(d->indexArray.size() - count);
        d->faceArray.popBack();
        return false;
    }
    else
    {
        if (d->currentMaterial != NIL) {
            d->materialArray[d->currentMaterial].faceCount++;
        }

        d->maxFaceIndexCount = max(d->maxFaceIndexCount, count);
        return true;
    }
}


uint MeshBuilder::weldPositions()
{
    Array<uint> xrefs;
    Weld<Vector3> weldVector3;

    if (d->posArray.count()) {
        // Weld vertex attributes.
        weldVector3(d->posArray, xrefs);

        // Remap vertex indices.
        const uint vertexCount = d->vertexArray.count();
        for (uint v = 0; v < vertexCount; v++)
        {
            Vertex & vertex = d->vertexArray[v];
            if (vertex.pos != NIL) vertex.pos = xrefs[vertex.pos];
        }
    }

    return d->posArray.count();
}

uint MeshBuilder::weldNormals()
{
    Array<uint> xrefs;
    Weld<Vector3> weldVector3;

    if (d->norArray.count()) {
        // Weld vertex attributes.
        weldVector3(d->norArray, xrefs);

        // Remap vertex indices.
        const uint vertexCount = d->vertexArray.count();
        for (uint v = 0; v < vertexCount; v++)
        {
            Vertex & vertex = d->vertexArray[v];
            if (vertex.nor != NIL) vertex.nor = xrefs[vertex.nor];
        }
    }

    return d->norArray.count();
}

uint MeshBuilder::weldTexCoords(uint set/*=0*/)
{
    Array<uint> xrefs;
    Weld<Vector2> weldVector2;

    if (d->texArray[set].count()) {
        // Weld vertex attributes.
        weldVector2(d->texArray[set], xrefs);

        // Remap vertex indices.
        const uint vertexCount = d->vertexArray.count();
        for (uint v = 0; v < vertexCount; v++)
        {
            Vertex & vertex = d->vertexArray[v];
            if (vertex.tex[set] != NIL) vertex.tex[set] = xrefs[vertex.tex[set]];
        }
    }

    return d->texArray[set].count();
}

uint  MeshBuilder::weldColors(uint set/*=0*/)
{
    Array<uint> xrefs;
    Weld<Vector4> weldVector4;

    if (d->colArray[set].count()) {
        // Weld vertex attributes.
        weldVector4(d->colArray[set], xrefs);

        // Remap vertex indices.
        const uint vertexCount = d->vertexArray.count();
        for (uint v = 0; v < vertexCount; v++)
        {
            Vertex & vertex = d->vertexArray[v];
            if (vertex.col[set] != NIL) vertex.col[set] = xrefs[vertex.col[set]];
        }
    }

    return d->colArray[set].count();
}

void MeshBuilder::weldVertices() {

    if (d->vertexArray.count() == 0) {
        // Nothing to do.
        return;
    }

    Array<uint> xrefs;
    Weld<Vertex> weldVertex;

    // Weld vertices.
    weldVertex(d->vertexArray, xrefs);

    // Remap face indices.
    const uint indexCount = d->indexArray.count();
    for (uint i = 0; i < indexCount; i++)
    {
        d->indexArray[i] = xrefs[d->indexArray[i]];
    }

    // Remap vertex map.
    foreach(i, d->vertexMap)
    {
        d->vertexMap[i].value = xrefs[d->vertexMap[i].value];
    }
}


void MeshBuilder::optimize()
{
    if (d->vertexArray.count() == 0)
    {
        return;
    }

    weldPositions();
    weldNormals();
    weldTexCoords(0);
    weldTexCoords(1);
    weldColors();

    weldVertices();
}






void MeshBuilder::removeUnusedMaterials(Array<uint> & newMaterialId)
{
    uint materialCount = d->materialArray.count();

    // Reset face counts.
    for (uint i = 0; i < materialCount; i++) {
        d->materialArray[i].faceCount = 0;
    }

    // Count faces.
    foreach(i, d->faceArray) {
        Face & face = d->faceArray[i];

        if (face.material != NIL) {
            nvDebugCheck(face.material < materialCount);

            d->materialArray[face.material].faceCount++;
        }
    }

    // Remove unused materials.
    newMaterialId.resize(materialCount);

    for (uint i = 0, m = 0; i < materialCount; i++)
    {
        if (d->materialArray[m].faceCount > 0)
        {
            newMaterialId[i] = m++;
        }
        else
        {
            newMaterialId[i] = NIL;
            d->materialArray.removeAt(m);
        }
    }

    materialCount = d->materialArray.count();

    // Update face material ids.
    foreach(i, d->faceArray) {
        Face & face = d->faceArray[i];

        if (face.material != NIL) {
            uint id = newMaterialId[face.material];
            nvDebugCheck(id != NIL && id < materialCount);

            face.material = id;
        }
    }
}

void MeshBuilder::sortFacesByGroup()
{
    const uint faceCount = d->faceArray.count();

    Array<uint> faceGroupArray;
    faceGroupArray.resize(faceCount);
    
    for (uint i = 0; i < faceCount; i++) {
        faceGroupArray[i] = d->faceArray[i].group;
    }

    RadixSort radix;
    radix.sort(faceGroupArray);

    Array<Face> newFaceArray;
    newFaceArray.resize(faceCount);

    for (uint i = 0; i < faceCount; i++) {
        newFaceArray[i] = d->faceArray[radix.rank(i)];
    }

    swap(newFaceArray, d->faceArray);
}

void MeshBuilder::sortFacesByMaterial()
{
    const uint faceCount = d->faceArray.count();

    Array<uint> faceMaterialArray;
    faceMaterialArray.resize(faceCount);
    
    for (uint i = 0; i < faceCount; i++) {
        faceMaterialArray[i] = d->faceArray[i].material;
    }

    RadixSort radix;
    radix.sort(faceMaterialArray);

    Array<Face> newFaceArray;
    newFaceArray.resize(faceCount);

    for (uint i = 0; i < faceCount; i++) {
        newFaceArray[i] = d->faceArray[radix.rank(i)];
    }

    swap(newFaceArray, d->faceArray);
}


void MeshBuilder::reset()
{
    nvDebugCheck(d != NULL);
    delete d;
    d = new PrivateData();
}

void MeshBuilder::done()
{
    if (d->currentGroup != NIL) {
        endGroup();
    }

    if (d->currentMaterial != NIL) {
        endMaterial();
    }
}

// Hints.
void MeshBuilder::hintTriangleCount(uint count)
{
    d->indexArray.reserve(d->indexArray.count() + count * 4);
}

void MeshBuilder::hintVertexCount(uint count)
{
    d->vertexArray.reserve(d->vertexArray.count() + count);
    d->vertexMap.resize(d->vertexMap.count() + count);
}

void MeshBuilder::hintPositionCount(uint count)
{
    d->posArray.reserve(d->posArray.count() + count);
}

void MeshBuilder::hintNormalCount(uint count)
{
    d->norArray.reserve(d->norArray.count() + count);
}

void MeshBuilder::hintTexCoordCount(uint count, uint set/*=0*/)
{
    d->texArray[set].reserve(d->texArray[set].count() + count);
}

void MeshBuilder::hintColorCount(uint count, uint set/*=0*/)
{
    d->colArray[set].reserve(d->colArray[set].count() + count);
}


// Helpers.
void MeshBuilder::addTriangle(uint v0, uint v1, uint v2)
{
    beginPolygon();
    addVertex(v0);
    addVertex(v1);
    addVertex(v2);
    endPolygon();
}

void MeshBuilder::addQuad(uint v0, uint v1, uint v2, uint v3)
{
    beginPolygon();
    addVertex(v0);
    addVertex(v1);
    addVertex(v2);
    addVertex(v3);
    endPolygon();
}


// Get tri mesh.
TriMesh * MeshBuilder::buildTriMesh() const
{
    const uint faceCount = d->faceArray.count();
    uint triangleCount = 0;
    for (uint f = 0; f < faceCount; f++) {
        triangleCount += d->faceArray[f].indexCount - 2;
    }
    
    const uint vertexCount = d->vertexArray.count();
    TriMesh * mesh = new TriMesh(triangleCount, vertexCount);

    // Build faces.
    Array<TriMesh::Face> & faces = mesh->faces();

    for(uint f = 0; f < faceCount; f++)
    {
        int firstIndex = d->faceArray[f].firstIndex;
        int indexCount = d->faceArray[f].indexCount;

        int v0 = d->indexArray[firstIndex + 0];
        int v1 = d->indexArray[firstIndex + 1];

        for(int t = 0; t < indexCount - 2; t++) {
            int v2 = d->indexArray[firstIndex + t + 2];

            TriMesh::Face face;
            face.id = faces.count();
            face.v[0] = v0;
            face.v[1] = v1;
            face.v[2] = v2;
            faces.append(face);

            v1 = v2;
        }
    }

    // Build vertices.
    Array<BaseMesh::Vertex> & vertices = mesh->vertices();

    for(uint i = 0; i < vertexCount; i++)
    {
        BaseMesh::Vertex vertex;
        vertex.id = i;
        if (d->vertexArray[i].pos != NIL) vertex.pos = d->posArray[d->vertexArray[i].pos];
        if (d->vertexArray[i].nor != NIL) vertex.nor = d->norArray[d->vertexArray[i].nor];
        if (d->vertexArray[i].tex[0] != NIL) vertex.tex = d->texArray[0][d->vertexArray[i].tex[0]];

        vertices.append(vertex);
    }

    return mesh;
}

// Get quad/tri mesh.
QuadTriMesh * MeshBuilder::buildQuadTriMesh() const
{
    const uint faceCount = d->faceArray.count();
    const uint vertexCount = d->vertexArray.count();
    QuadTriMesh * mesh = new QuadTriMesh(faceCount, vertexCount);

    // Build faces.
    Array<QuadTriMesh::Face> & faces = mesh->faces();

    for (uint f = 0; f < faceCount; f++) 
    {
        int firstIndex = d->faceArray[f].firstIndex;
        int indexCount = d->faceArray[f].indexCount;

        QuadTriMesh::Face face;
        face.id = f;

        face.v[0] = d->indexArray[firstIndex + 0];
        face.v[1] = d->indexArray[firstIndex + 1];
        face.v[2] = d->indexArray[firstIndex + 2];

        // Only adds triangles and quads. Ignores polygons.
        if (indexCount == 3) {
            face.v[3] = NIL;
            faces.append(face);
        }
        else if (indexCount == 4) {
            face.v[3] = d->indexArray[firstIndex + 3];
            faces.append(face);
        }
    }

    // Build vertices.
    Array<BaseMesh::Vertex> & vertices = mesh->vertices();

    for(uint i = 0; i < vertexCount; i++)
    {
        BaseMesh::Vertex vertex;
        vertex.id = i;
        if (d->vertexArray[i].pos != NIL) vertex.pos = d->posArray[d->vertexArray[i].pos];
        if (d->vertexArray[i].nor != NIL) vertex.nor = d->norArray[d->vertexArray[i].nor];
        if (d->vertexArray[i].tex[0] != NIL) vertex.tex = d->texArray[0][d->vertexArray[i].tex[0]];

        vertices.append(vertex);
    }

    return mesh;
}

// Get half edge mesh.
HalfEdge::Mesh * MeshBuilder::buildHalfEdgeMesh(bool weldPositions, Error * error/*=NULL*/, Array<uint> * badFaces/*=NULL*/) const
{
    if (error != NULL) *error = Error_None;

    const uint vertexCount = d->vertexArray.count();
    AutoPtr<HalfEdge::Mesh> mesh(new HalfEdge::Mesh());

    for(uint v = 0; v < vertexCount; v++)
    {
        HalfEdge::Vertex * vertex = mesh->addVertex(d->posArray[d->vertexArray[v].pos]);
        if (d->vertexArray[v].nor != NIL) vertex->nor = d->norArray[d->vertexArray[v].nor];
        if (d->vertexArray[v].tex[0] != NIL) vertex->tex = Vector2(d->texArray[0][d->vertexArray[v].tex[0]]);
        if (d->vertexArray[v].col[0] != NIL) vertex->col = d->colArray[0][d->vertexArray[v].col[0]];
    }

    if (weldPositions) {
        mesh->linkColocals();
    }
    else {
        // Build canonical map from position indices.
        Array<uint> canonicalMap(vertexCount);
        
        foreach (i, d->vertexArray) {
            canonicalMap.append(d->vertexArray[i].pos);
        }

        mesh->linkColocalsWithCanonicalMap(canonicalMap);
    }

    const uint faceCount = d->faceArray.count();
    for (uint f = 0; f < faceCount; f++)
    {
        const uint firstIndex = d->faceArray[f].firstIndex;
        const uint indexCount = d->faceArray[f].indexCount;

        HalfEdge::Face * face = mesh->addFace(d->indexArray, firstIndex, indexCount);
        
        // @@ This is too late, removing the face here will leave the mesh improperly connected.
        /*if (face->area() <= FLT_EPSILON) {
            mesh->remove(face);
            face = NULL;
        }*/

        if (face == NULL) {
            // Non manifold mesh.
            if (error != NULL) *error = Error_NonManifoldEdge;
            if (badFaces != NULL) {
                badFaces->append(d->faceArray[f].id);
            }
            //return NULL; // IC: Ignore error and continue building the mesh.
        }

        if (face != NULL) {
            face->group = d->faceArray[f].group;
            face->material = d->faceArray[f].material;
        }
    }

    mesh->linkBoundary();

    // We cannot fix functions here, because this would introduce new vertices and these vertices won't have the corresponding builder data.

    // Maybe the builder should perform the search for T-junctions and update the vertex data directly.

    // For now, we don't fix T-junctions at export time, but only during parameterization.

    //mesh->fixBoundaryJunctions();

    //mesh->sewBoundary();

    return mesh.release();
}


bool MeshBuilder::buildPositions(Array<Vector3> & positionArray)
{
    const uint vertexCount = d->vertexArray.count();
    positionArray.resize(vertexCount);

    for (uint v = 0; v < vertexCount; v++)
    {
        nvDebugCheck(d->vertexArray[v].pos != NIL);
        positionArray[v] = d->posArray[d->vertexArray[v].pos];
    }

    return true;
}

bool MeshBuilder::buildNormals(Array<Vector3> & normalArray)
{
    bool anyNormal = false;

    const uint vertexCount = d->vertexArray.count();
    normalArray.resize(vertexCount);

    for (uint v = 0; v < vertexCount; v++)
    {
        if (d->vertexArray[v].nor == NIL) {
            normalArray[v] = Vector3(0, 0, 1);
        }
        else {
            anyNormal = true;
            normalArray[v] = d->norArray[d->vertexArray[v].nor];
        }
    }

    return anyNormal;
}

bool MeshBuilder::buildTexCoords(Array<Vector2> & texCoordArray, uint set/*=0*/)
{
    bool anyTexCoord = false;

    const uint vertexCount = d->vertexArray.count();
    texCoordArray.resize(vertexCount);

    for (uint v = 0; v < vertexCount; v++)
    {
        if (d->vertexArray[v].tex[set] == NIL) {
            texCoordArray[v] = Vector2(0, 0);
        }
        else {
            anyTexCoord = true;
            texCoordArray[v] = d->texArray[set][d->vertexArray[v].tex[set]];
        }
    }

    return anyTexCoord;
}

bool MeshBuilder::buildColors(Array<Vector4> & colorArray, uint set/*=0*/)
{
    bool anyColor = false;

    const uint vertexCount = d->vertexArray.count();
    colorArray.resize(vertexCount);

    for (uint v = 0; v < vertexCount; v++)
    {
        if (d->vertexArray[v].col[set] == NIL) {
            colorArray[v] = Vector4(0, 0, 0, 1);
        }
        else {
            anyColor = true;
            colorArray[v] = d->colArray[set][d->vertexArray[v].col[set]];
        }
    }

    return anyColor;
}

void MeshBuilder::buildVertexToPositionMap(Array<int> &map)
{
	const uint vertexCount = d->vertexArray.count();
	map.resize(vertexCount);

	foreach (i, d->vertexArray) {
		map[i] = d->vertexArray[i].pos;
	}
}



uint MeshBuilder::vertexCount() const
{
    return d->vertexArray.count();
}


uint MeshBuilder::positionCount() const
{
    return d->posArray.count();
}

uint MeshBuilder::normalCount() const
{
    return d->norArray.count();
}

uint MeshBuilder::texCoordCount(uint set/*=0*/) const
{
    return d->texArray[set].count();
}

uint MeshBuilder::colorCount(uint set/*=0*/) const
{
    return d->colArray[set].count();
}


uint MeshBuilder::materialCount() const
{
    return d->materialArray.count();
}

const char * MeshBuilder::material(uint i) const
{
    return d->materialArray[i].name;
}


uint MeshBuilder::positionIndex(uint vertex) const
{
    return d->vertexArray[vertex].pos;
}
uint MeshBuilder::normalIndex(uint vertex) const
{
    return d->vertexArray[vertex].nor;
}
uint MeshBuilder::texCoordIndex(uint vertex, uint set/*=0*/) const
{
    return d->vertexArray[vertex].tex[set];
}
uint MeshBuilder::colorIndex(uint vertex, uint set/*=0*/) const
{
    return d->vertexArray[vertex].col[set];
}
