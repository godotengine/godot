/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file  FBXMeshGeometry.cpp
 *  @brief Assimp::FBX::MeshGeometry implementation
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

#include <functional>

#include "FBXMeshGeometry.h"
#include "FBXDocument.h"
#include "FBXImporter.h"
#include "FBXImportSettings.h"
#include "FBXDocumentUtil.h"


namespace Assimp {
namespace FBX {

using namespace Util;

// ------------------------------------------------------------------------------------------------
Geometry::Geometry(uint64_t id, const Element& element, const std::string& name, const Document& doc)
    : Object(id, element, name)
    , skin()
{
    const std::vector<const Connection*>& conns = doc.GetConnectionsByDestinationSequenced(ID(),"Deformer");
    for(const Connection* con : conns) {
        const Skin* const sk = ProcessSimpleConnection<Skin>(*con, false, "Skin -> Geometry", element);
        if(sk) {
            skin = sk;
        }
        const BlendShape* const bsp = ProcessSimpleConnection<BlendShape>(*con, false, "BlendShape -> Geometry", element);
        if (bsp) {
            blendShapes.push_back(bsp);
        }
    }
}

// ------------------------------------------------------------------------------------------------
Geometry::~Geometry()
{
    // empty
}

// ------------------------------------------------------------------------------------------------
const std::vector<const BlendShape*>& Geometry::GetBlendShapes() const {
    return blendShapes;
}

// ------------------------------------------------------------------------------------------------
const Skin* Geometry::DeformerSkin() const {
    return skin;
}

// ------------------------------------------------------------------------------------------------
MeshGeometry::MeshGeometry(uint64_t id, const Element& element, const std::string& name, const Document& doc)
: Geometry(id, element,name, doc)
{
    const Scope* sc = element.Compound();
    if (!sc) {
        DOMError("failed to read Geometry object (class: Mesh), no data scope found");
    }

    // must have Mesh elements:
    const Element& Vertices = GetRequiredElement(*sc,"Vertices",&element);
    const Element& PolygonVertexIndex = GetRequiredElement(*sc,"PolygonVertexIndex",&element);

    // optional Mesh elements:
    const ElementCollection& Layer = sc->GetCollection("Layer");

    std::vector<aiVector3D> tempVerts;
    ParseVectorDataArray(tempVerts,Vertices);

    if(tempVerts.empty()) {
        FBXImporter::LogWarn("encountered mesh with no vertices");
    }

    std::vector<int> tempFaces;
    ParseVectorDataArray(tempFaces,PolygonVertexIndex);

    if(tempFaces.empty()) {
        FBXImporter::LogWarn("encountered mesh with no faces");
    }

    m_vertices.reserve(tempFaces.size());
    m_faces.reserve(tempFaces.size() / 3);

    m_mapping_offsets.resize(tempVerts.size());
    m_mapping_counts.resize(tempVerts.size(),0);
    m_mappings.resize(tempFaces.size());

    const size_t vertex_count = tempVerts.size();

    // generate output vertices, computing an adjacency table to
    // preserve the mapping from fbx indices to *this* indexing.
    unsigned int count = 0;
    for(int index : tempFaces) {
        const int absi = index < 0 ? (-index - 1) : index;
        if(static_cast<size_t>(absi) >= vertex_count) {
            DOMError("polygon vertex index out of range",&PolygonVertexIndex);
        }

        m_vertices.push_back(tempVerts[absi]);
        ++count;

        ++m_mapping_counts[absi];

        if (index < 0) {
            m_faces.push_back(count);
            count = 0;
        }
    }

    unsigned int cursor = 0;
    for (size_t i = 0, e = tempVerts.size(); i < e; ++i) {
        m_mapping_offsets[i] = cursor;
        cursor += m_mapping_counts[i];

        m_mapping_counts[i] = 0;
    }

    cursor = 0;
    for(int index : tempFaces) {
        const int absi = index < 0 ? (-index - 1) : index;
        m_mappings[m_mapping_offsets[absi] + m_mapping_counts[absi]++] = cursor++;
    }

    // if settings.readAllLayers is true:
    //  * read all layers, try to load as many vertex channels as possible
    // if settings.readAllLayers is false:
    //  * read only the layer with index 0, but warn about any further layers
    for (ElementMap::const_iterator it = Layer.first; it != Layer.second; ++it) {
        const TokenList& tokens = (*it).second->Tokens();

        const char* err;
        const int index = ParseTokenAsInt(*tokens[0], err);
        if(err) {
            DOMError(err,&element);
        }

        if(doc.Settings().readAllLayers || index == 0) {
            const Scope& layer = GetRequiredScope(*(*it).second);
            ReadLayer(layer);
        }
        else {
            FBXImporter::LogWarn("ignoring additional geometry layers");
        }
    }
}

// ------------------------------------------------------------------------------------------------
MeshGeometry::~MeshGeometry() {
    // empty
}

// ------------------------------------------------------------------------------------------------
const std::vector<aiVector3D>& MeshGeometry::GetVertices() const {
    return m_vertices;
}

// ------------------------------------------------------------------------------------------------
const std::vector<aiVector3D>& MeshGeometry::GetNormals() const {
    return m_normals;
}

// ------------------------------------------------------------------------------------------------
const std::vector<aiVector3D>& MeshGeometry::GetTangents() const {
    return m_tangents;
}

// ------------------------------------------------------------------------------------------------
const std::vector<aiVector3D>& MeshGeometry::GetBinormals() const {
    return m_binormals;
}

// ------------------------------------------------------------------------------------------------
const std::vector<unsigned int>& MeshGeometry::GetFaceIndexCounts() const {
    return m_faces;
}

// ------------------------------------------------------------------------------------------------
const std::vector<aiVector2D>& MeshGeometry::GetTextureCoords( unsigned int index ) const {
    static const std::vector<aiVector2D> empty;
    return index >= AI_MAX_NUMBER_OF_TEXTURECOORDS ? empty : m_uvs[ index ];
}

std::string MeshGeometry::GetTextureCoordChannelName( unsigned int index ) const {
    return index >= AI_MAX_NUMBER_OF_TEXTURECOORDS ? "" : m_uvNames[ index ];
}

const std::vector<aiColor4D>& MeshGeometry::GetVertexColors( unsigned int index ) const {
    static const std::vector<aiColor4D> empty;
    return index >= AI_MAX_NUMBER_OF_COLOR_SETS ? empty : m_colors[ index ];
}

const MatIndexArray& MeshGeometry::GetMaterialIndices() const {
    return m_materials;
}
// ------------------------------------------------------------------------------------------------
const unsigned int* MeshGeometry::ToOutputVertexIndex( unsigned int in_index, unsigned int& count ) const {
    if ( in_index >= m_mapping_counts.size() ) {
        return NULL;
    }

    ai_assert( m_mapping_counts.size() == m_mapping_offsets.size() );
    count = m_mapping_counts[ in_index ];

    ai_assert( m_mapping_offsets[ in_index ] + count <= m_mappings.size() );

    return &m_mappings[ m_mapping_offsets[ in_index ] ];
}

// ------------------------------------------------------------------------------------------------
unsigned int MeshGeometry::FaceForVertexIndex( unsigned int in_index ) const {
    ai_assert( in_index < m_vertices.size() );

    // in the current conversion pattern this will only be needed if
    // weights are present, so no need to always pre-compute this table
    if ( m_facesVertexStartIndices.empty() ) {
        m_facesVertexStartIndices.resize( m_faces.size() + 1, 0 );

        std::partial_sum( m_faces.begin(), m_faces.end(), m_facesVertexStartIndices.begin() + 1 );
        m_facesVertexStartIndices.pop_back();
    }

    ai_assert( m_facesVertexStartIndices.size() == m_faces.size() );
    const std::vector<unsigned int>::iterator it = std::upper_bound(
        m_facesVertexStartIndices.begin(),
        m_facesVertexStartIndices.end(),
        in_index
        );

    return static_cast< unsigned int >( std::distance( m_facesVertexStartIndices.begin(), it - 1 ) );
}

// ------------------------------------------------------------------------------------------------
void MeshGeometry::ReadLayer(const Scope& layer)
{
    const ElementCollection& LayerElement = layer.GetCollection("LayerElement");
    for (ElementMap::const_iterator eit = LayerElement.first; eit != LayerElement.second; ++eit) {
        const Scope& elayer = GetRequiredScope(*(*eit).second);

        ReadLayerElement(elayer);
    }
}


// ------------------------------------------------------------------------------------------------
void MeshGeometry::ReadLayerElement(const Scope& layerElement)
{
    const Element& Type = GetRequiredElement(layerElement,"Type");
    const Element& TypedIndex = GetRequiredElement(layerElement,"TypedIndex");

    const std::string& type = ParseTokenAsString(GetRequiredToken(Type,0));
    const int typedIndex = ParseTokenAsInt(GetRequiredToken(TypedIndex,0));

    const Scope& top = GetRequiredScope(element);
    const ElementCollection candidates = top.GetCollection(type);

    for (ElementMap::const_iterator it = candidates.first; it != candidates.second; ++it) {
        const int index = ParseTokenAsInt(GetRequiredToken(*(*it).second,0));
        if(index == typedIndex) {
            ReadVertexData(type,typedIndex,GetRequiredScope(*(*it).second));
            return;
        }
    }

    FBXImporter::LogError(Formatter::format("failed to resolve vertex layer element: ")
        << type << ", index: " << typedIndex);
}

// ------------------------------------------------------------------------------------------------
void MeshGeometry::ReadVertexData(const std::string& type, int index, const Scope& source)
{
    const std::string& MappingInformationType = ParseTokenAsString(GetRequiredToken(
        GetRequiredElement(source,"MappingInformationType"),0)
    );

    const std::string& ReferenceInformationType = ParseTokenAsString(GetRequiredToken(
        GetRequiredElement(source,"ReferenceInformationType"),0)
    );

    if (type == "LayerElementUV") {
        if(index >= AI_MAX_NUMBER_OF_TEXTURECOORDS) {
            FBXImporter::LogError(Formatter::format("ignoring UV layer, maximum number of UV channels exceeded: ")
                << index << " (limit is " << AI_MAX_NUMBER_OF_TEXTURECOORDS << ")" );
            return;
        }

        const Element* Name = source["Name"];
        m_uvNames[index] = "";
        if(Name) {
            m_uvNames[index] = ParseTokenAsString(GetRequiredToken(*Name,0));
        }

        ReadVertexDataUV(m_uvs[index],source,
            MappingInformationType,
            ReferenceInformationType
        );
    }
    else if (type == "LayerElementMaterial") {
        if (m_materials.size() > 0) {
            FBXImporter::LogError("ignoring additional material layer");
            return;
        }

        std::vector<int> temp_materials;

        ReadVertexDataMaterials(temp_materials,source,
            MappingInformationType,
            ReferenceInformationType
        );

        // sometimes, there will be only negative entries. Drop the material
        // layer in such a case (I guess it means a default material should
        // be used). This is what the converter would do anyway, and it
        // avoids losing the material if there are more material layers
        // coming of which at least one contains actual data (did observe
        // that with one test file).
        const size_t count_neg = std::count_if(temp_materials.begin(),temp_materials.end(),[](int n) { return n < 0; });
        if(count_neg == temp_materials.size()) {
            FBXImporter::LogWarn("ignoring dummy material layer (all entries -1)");
            return;
        }

        std::swap(temp_materials, m_materials);
    }
    else if (type == "LayerElementNormal") {
        if (m_normals.size() > 0) {
            FBXImporter::LogError("ignoring additional normal layer");
            return;
        }

        ReadVertexDataNormals(m_normals,source,
            MappingInformationType,
            ReferenceInformationType
        );
    }
    else if (type == "LayerElementTangent") {
        if (m_tangents.size() > 0) {
            FBXImporter::LogError("ignoring additional tangent layer");
            return;
        }

        ReadVertexDataTangents(m_tangents,source,
            MappingInformationType,
            ReferenceInformationType
        );
    }
    else if (type == "LayerElementBinormal") {
        if (m_binormals.size() > 0) {
            FBXImporter::LogError("ignoring additional binormal layer");
            return;
        }

        ReadVertexDataBinormals(m_binormals,source,
            MappingInformationType,
            ReferenceInformationType
        );
    }
    else if (type == "LayerElementColor") {
        if(index >= AI_MAX_NUMBER_OF_COLOR_SETS) {
            FBXImporter::LogError(Formatter::format("ignoring vertex color layer, maximum number of color sets exceeded: ")
                << index << " (limit is " << AI_MAX_NUMBER_OF_COLOR_SETS << ")" );
            return;
        }

        ReadVertexDataColors(m_colors[index],source,
            MappingInformationType,
            ReferenceInformationType
        );
    }
}

// ------------------------------------------------------------------------------------------------
// Lengthy utility function to read and resolve a FBX vertex data array - that is, the
// output is in polygon vertex order. This logic is used for reading normals, UVs, colors,
// tangents ..
template <typename T>
void ResolveVertexDataArray(std::vector<T>& data_out, const Scope& source,
    const std::string& MappingInformationType,
    const std::string& ReferenceInformationType,
    const char* dataElementName,
    const char* indexDataElementName,
    size_t vertex_count,
    const std::vector<unsigned int>& mapping_counts,
    const std::vector<unsigned int>& mapping_offsets,
    const std::vector<unsigned int>& mappings)
{
    bool isDirect = ReferenceInformationType == "Direct";
    bool isIndexToDirect = ReferenceInformationType == "IndexToDirect";

    // fall-back to direct data if there is no index data element
    if ( isIndexToDirect && !HasElement( source, indexDataElementName ) ) {
        isDirect = true;
        isIndexToDirect = false;
    }

    // handle permutations of Mapping and Reference type - it would be nice to
    // deal with this more elegantly and with less redundancy, but right
    // now it seems unavoidable.
    if (MappingInformationType == "ByVertice" && isDirect) {
        if (!HasElement(source, dataElementName)) {
            return;
        }
        std::vector<T> tempData;
        ParseVectorDataArray(tempData, GetRequiredElement(source, dataElementName));

        if (tempData.size() != mapping_offsets.size()) {
            FBXImporter::LogError(Formatter::format("length of input data unexpected for ByVertice mapping: ")
                                  << tempData.size() << ", expected " << mapping_offsets.size());
            return;
        }

        data_out.resize(vertex_count);
        for (size_t i = 0, e = tempData.size(); i < e; ++i) {
            const unsigned int istart = mapping_offsets[i], iend = istart + mapping_counts[i];
            for (unsigned int j = istart; j < iend; ++j) {
                data_out[mappings[j]] = tempData[i];
            }
        }
    }
    else if (MappingInformationType == "ByVertice" && isIndexToDirect) {
		std::vector<T> tempData;
		ParseVectorDataArray(tempData, GetRequiredElement(source, dataElementName));

        std::vector<int> uvIndices;
        ParseVectorDataArray(uvIndices,GetRequiredElement(source,indexDataElementName));

        if (uvIndices.size() != vertex_count) {
            FBXImporter::LogError(Formatter::format("length of input data unexpected for ByVertice mapping: ")
                                  << uvIndices.size() << ", expected " << vertex_count);
            return;
        }

        data_out.resize(vertex_count);

        for (size_t i = 0, e = uvIndices.size(); i < e; ++i) {

            const unsigned int istart = mapping_offsets[i], iend = istart + mapping_counts[i];
            for (unsigned int j = istart; j < iend; ++j) {
				if (static_cast<size_t>(uvIndices[i]) >= tempData.size()) {
                    DOMError("index out of range",&GetRequiredElement(source,indexDataElementName));
                }
				data_out[mappings[j]] = tempData[uvIndices[i]];
            }
        }
    }
    else if (MappingInformationType == "ByPolygonVertex" && isDirect) {
		std::vector<T> tempData;
		ParseVectorDataArray(tempData, GetRequiredElement(source, dataElementName));

		if (tempData.size() != vertex_count) {
            FBXImporter::LogError(Formatter::format("length of input data unexpected for ByPolygon mapping: ")
				<< tempData.size() << ", expected " << vertex_count
            );
            return;
        }

		data_out.swap(tempData);
    }
    else if (MappingInformationType == "ByPolygonVertex" && isIndexToDirect) {
		std::vector<T> tempData;
		ParseVectorDataArray(tempData, GetRequiredElement(source, dataElementName));

        std::vector<int> uvIndices;
        ParseVectorDataArray(uvIndices,GetRequiredElement(source,indexDataElementName));

        if (uvIndices.size() != vertex_count) {
            FBXImporter::LogError(Formatter::format("length of input data unexpected for ByPolygonVertex mapping: ")
                                  << uvIndices.size() << ", expected " << vertex_count);
            return;
        }

        data_out.resize(vertex_count);

        const T empty;
        unsigned int next = 0;
        for(int i : uvIndices) {
            if ( -1 == i ) {
                data_out[ next++ ] = empty;
                continue;
            }
            if (static_cast<size_t>(i) >= tempData.size()) {
                DOMError("index out of range",&GetRequiredElement(source,indexDataElementName));
            }

			data_out[next++] = tempData[i];
        }
    }
    else {
        FBXImporter::LogError(Formatter::format("ignoring vertex data channel, access type not implemented: ")
            << MappingInformationType << "," << ReferenceInformationType);
    }
}

// ------------------------------------------------------------------------------------------------
void MeshGeometry::ReadVertexDataNormals(std::vector<aiVector3D>& normals_out, const Scope& source,
    const std::string& MappingInformationType,
    const std::string& ReferenceInformationType)
{
    ResolveVertexDataArray(normals_out,source,MappingInformationType,ReferenceInformationType,
        "Normals",
        "NormalsIndex",
        m_vertices.size(),
        m_mapping_counts,
        m_mapping_offsets,
        m_mappings);
}

// ------------------------------------------------------------------------------------------------
void MeshGeometry::ReadVertexDataUV(std::vector<aiVector2D>& uv_out, const Scope& source,
    const std::string& MappingInformationType,
    const std::string& ReferenceInformationType)
{
    ResolveVertexDataArray(uv_out,source,MappingInformationType,ReferenceInformationType,
        "UV",
        "UVIndex",
        m_vertices.size(),
        m_mapping_counts,
        m_mapping_offsets,
        m_mappings);
}

// ------------------------------------------------------------------------------------------------
void MeshGeometry::ReadVertexDataColors(std::vector<aiColor4D>& colors_out, const Scope& source,
    const std::string& MappingInformationType,
    const std::string& ReferenceInformationType)
{
    ResolveVertexDataArray(colors_out,source,MappingInformationType,ReferenceInformationType,
        "Colors",
        "ColorIndex",
        m_vertices.size(),
        m_mapping_counts,
        m_mapping_offsets,
        m_mappings);
}

// ------------------------------------------------------------------------------------------------
static const char *TangentIndexToken = "TangentIndex";
static const char *TangentsIndexToken = "TangentsIndex";

void MeshGeometry::ReadVertexDataTangents(std::vector<aiVector3D>& tangents_out, const Scope& source,
    const std::string& MappingInformationType,
    const std::string& ReferenceInformationType)
{
    const char * str = source.Elements().count( "Tangents" ) > 0 ? "Tangents" : "Tangent";
    const char * strIdx = source.Elements().count( "Tangents" ) > 0 ? TangentsIndexToken : TangentIndexToken;
    ResolveVertexDataArray(tangents_out,source,MappingInformationType,ReferenceInformationType,
        str,
        strIdx,
        m_vertices.size(),
        m_mapping_counts,
        m_mapping_offsets,
        m_mappings);
}

// ------------------------------------------------------------------------------------------------
static const std::string BinormalIndexToken = "BinormalIndex";
static const std::string BinormalsIndexToken = "BinormalsIndex";

void MeshGeometry::ReadVertexDataBinormals(std::vector<aiVector3D>& binormals_out, const Scope& source,
    const std::string& MappingInformationType,
    const std::string& ReferenceInformationType)
{
    const char * str = source.Elements().count( "Binormals" ) > 0 ? "Binormals" : "Binormal";
    const char * strIdx = source.Elements().count( "Binormals" ) > 0 ? BinormalsIndexToken.c_str() : BinormalIndexToken.c_str();
    ResolveVertexDataArray(binormals_out,source,MappingInformationType,ReferenceInformationType,
        str,
        strIdx,
        m_vertices.size(),
        m_mapping_counts,
        m_mapping_offsets,
        m_mappings);
}


// ------------------------------------------------------------------------------------------------
void MeshGeometry::ReadVertexDataMaterials(std::vector<int>& materials_out, const Scope& source,
    const std::string& MappingInformationType,
    const std::string& ReferenceInformationType)
{
    const size_t face_count = m_faces.size();
    if( 0 == face_count )
    {
        return;
    }
    
    // materials are handled separately. First of all, they are assigned per-face
    // and not per polyvert. Secondly, ReferenceInformationType=IndexToDirect
    // has a slightly different meaning for materials.
    ParseVectorDataArray(materials_out,GetRequiredElement(source,"Materials"));

    if (MappingInformationType == "AllSame") {
        // easy - same material for all faces
        if (materials_out.empty()) {
            FBXImporter::LogError(Formatter::format("expected material index, ignoring"));
            return;
        } else if (materials_out.size() > 1) {
            FBXImporter::LogWarn(Formatter::format("expected only a single material index, ignoring all except the first one"));
            materials_out.clear();
        }

        materials_out.resize(m_vertices.size());
        std::fill(materials_out.begin(), materials_out.end(), materials_out.at(0));
    } else if (MappingInformationType == "ByPolygon" && ReferenceInformationType == "IndexToDirect") {
        materials_out.resize(face_count);

        if(materials_out.size() != face_count) {
            FBXImporter::LogError(Formatter::format("length of input data unexpected for ByPolygon mapping: ")
                << materials_out.size() << ", expected " << face_count
            );
            return;
        }
    } else {
        FBXImporter::LogError(Formatter::format("ignoring material assignments, access type not implemented: ")
            << MappingInformationType << "," << ReferenceInformationType);
    }
}
// ------------------------------------------------------------------------------------------------
ShapeGeometry::ShapeGeometry(uint64_t id, const Element& element, const std::string& name, const Document& doc)
: Geometry(id, element, name, doc) {
    const Scope *sc = element.Compound();
    if (nullptr == sc) {
        DOMError("failed to read Geometry object (class: Shape), no data scope found");
    }
    const Element& Indexes = GetRequiredElement(*sc, "Indexes", &element);
    const Element& Normals = GetRequiredElement(*sc, "Normals", &element);
    const Element& Vertices = GetRequiredElement(*sc, "Vertices", &element);
    ParseVectorDataArray(m_indices, Indexes);
    ParseVectorDataArray(m_vertices, Vertices);
    ParseVectorDataArray(m_normals, Normals);
}

// ------------------------------------------------------------------------------------------------
ShapeGeometry::~ShapeGeometry() {
    // empty
}
// ------------------------------------------------------------------------------------------------
const std::vector<aiVector3D>& ShapeGeometry::GetVertices() const {
    return m_vertices;
}
// ------------------------------------------------------------------------------------------------
const std::vector<aiVector3D>& ShapeGeometry::GetNormals() const {
    return m_normals;
}
// ------------------------------------------------------------------------------------------------
const std::vector<unsigned int>& ShapeGeometry::GetIndices() const {
    return m_indices;
}
// ------------------------------------------------------------------------------------------------
LineGeometry::LineGeometry(uint64_t id, const Element& element, const std::string& name, const Document& doc)
    : Geometry(id, element, name, doc)
{
    const Scope* sc = element.Compound();
    if (!sc) {
        DOMError("failed to read Geometry object (class: Line), no data scope found");
    }
    const Element& Points = GetRequiredElement(*sc, "Points", &element);
    const Element& PointsIndex = GetRequiredElement(*sc, "PointsIndex", &element);
    ParseVectorDataArray(m_vertices, Points);
    ParseVectorDataArray(m_indices, PointsIndex);
}

// ------------------------------------------------------------------------------------------------
LineGeometry::~LineGeometry() {
    // empty
}
// ------------------------------------------------------------------------------------------------
const std::vector<aiVector3D>& LineGeometry::GetVertices() const {
    return m_vertices;
}
// ------------------------------------------------------------------------------------------------
const std::vector<int>& LineGeometry::GetIndices() const {
    return m_indices;
}
} // !FBX
} // !Assimp
#endif

