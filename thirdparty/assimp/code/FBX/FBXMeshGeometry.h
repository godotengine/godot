/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


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

/** @file  FBXImporter.h
*  @brief Declaration of the FBX main importer class
*/
#ifndef INCLUDED_AI_FBX_MESHGEOMETRY_H
#define INCLUDED_AI_FBX_MESHGEOMETRY_H

#include "FBXParser.h"
#include "FBXDocument.h"

namespace Assimp {
namespace FBX {

/** 
 *  DOM base class for all kinds of FBX geometry 
 */
class Geometry : public Object
{
public:
    Geometry( uint64_t id, const Element& element, const std::string& name, const Document& doc );
    virtual ~Geometry();

    /** Get the Skin attached to this geometry or NULL */
    const Skin* DeformerSkin() const;

    /** Get the BlendShape attached to this geometry or NULL */
    const std::vector<const BlendShape*>& GetBlendShapes() const;

private:
    const Skin* skin;
    std::vector<const BlendShape*> blendShapes;

};

typedef std::vector<int> MatIndexArray;


/** 
 *  DOM class for FBX geometry of type "Mesh"
 */
class MeshGeometry : public Geometry
{
public:
    /** The class constructor */
    MeshGeometry( uint64_t id, const Element& element, const std::string& name, const Document& doc );
    
    /** The class destructor */
    virtual ~MeshGeometry();

    /** Get a list of all vertex points, non-unique*/
    const std::vector<aiVector3D>& GetVertices() const;

    /** Get a list of all vertex normals or an empty array if
    *  no normals are specified. */
    const std::vector<aiVector3D>& GetNormals() const;

    /** Get a list of all vertex tangents or an empty array
    *  if no tangents are specified */
    const std::vector<aiVector3D>& GetTangents() const;

    /** Get a list of all vertex bi-normals or an empty array
    *  if no bi-normals are specified */
    const std::vector<aiVector3D>& GetBinormals() const;

    /** Return list of faces - each entry denotes a face and specifies
    *  how many vertices it has. Vertices are taken from the
    *  vertex data arrays in sequential order. */
    const std::vector<unsigned int>& GetFaceIndexCounts() const;

    /** Get a UV coordinate slot, returns an empty array if
    *  the requested slot does not exist. */
    const std::vector<aiVector2D>& GetTextureCoords( unsigned int index ) const;

    /** Get a UV coordinate slot, returns an empty array if
    *  the requested slot does not exist. */
    std::string GetTextureCoordChannelName( unsigned int index ) const;

    /** Get a vertex color coordinate slot, returns an empty array if
    *  the requested slot does not exist. */
    const std::vector<aiColor4D>& GetVertexColors( unsigned int index ) const;

    /** Get per-face-vertex material assignments */
    const MatIndexArray& GetMaterialIndices() const;

    /** Convert from a fbx file vertex index (for example from a #Cluster weight) or NULL
    * if the vertex index is not valid. */
    const unsigned int* ToOutputVertexIndex( unsigned int in_index, unsigned int& count ) const;

    /** Determine the face to which a particular output vertex index belongs.
    *  This mapping is always unique. */
    unsigned int FaceForVertexIndex( unsigned int in_index ) const;
private:
    void ReadLayer( const Scope& layer );
    void ReadLayerElement( const Scope& layerElement );
    void ReadVertexData( const std::string& type, int index, const Scope& source );

    void ReadVertexDataUV( std::vector<aiVector2D>& uv_out, const Scope& source,
        const std::string& MappingInformationType,
        const std::string& ReferenceInformationType );

    void ReadVertexDataNormals( std::vector<aiVector3D>& normals_out, const Scope& source,
        const std::string& MappingInformationType,
        const std::string& ReferenceInformationType );

    void ReadVertexDataColors( std::vector<aiColor4D>& colors_out, const Scope& source,
        const std::string& MappingInformationType,
        const std::string& ReferenceInformationType );

    void ReadVertexDataTangents( std::vector<aiVector3D>& tangents_out, const Scope& source,
        const std::string& MappingInformationType,
        const std::string& ReferenceInformationType );

    void ReadVertexDataBinormals( std::vector<aiVector3D>& binormals_out, const Scope& source,
        const std::string& MappingInformationType,
        const std::string& ReferenceInformationType );

    void ReadVertexDataMaterials( MatIndexArray& materials_out, const Scope& source,
        const std::string& MappingInformationType,
        const std::string& ReferenceInformationType );

private:
    // cached data arrays
    MatIndexArray m_materials;
    std::vector<aiVector3D> m_vertices;
    std::vector<unsigned int> m_faces;
    mutable std::vector<unsigned int> m_facesVertexStartIndices;
    std::vector<aiVector3D> m_tangents;
    std::vector<aiVector3D> m_binormals;
    std::vector<aiVector3D> m_normals;

    std::string m_uvNames[ AI_MAX_NUMBER_OF_TEXTURECOORDS ];
    std::vector<aiVector2D> m_uvs[ AI_MAX_NUMBER_OF_TEXTURECOORDS ];
    std::vector<aiColor4D> m_colors[ AI_MAX_NUMBER_OF_COLOR_SETS ];

    std::vector<unsigned int> m_mapping_counts;
    std::vector<unsigned int> m_mapping_offsets;
    std::vector<unsigned int> m_mappings;
};

/**
*  DOM class for FBX geometry of type "Shape"
*/
class ShapeGeometry : public Geometry
{
public:
    /** The class constructor */
    ShapeGeometry(uint64_t id, const Element& element, const std::string& name, const Document& doc);

    /** The class destructor */
    virtual ~ShapeGeometry();

    /** Get a list of all vertex points, non-unique*/
    const std::vector<aiVector3D>& GetVertices() const;

    /** Get a list of all vertex normals or an empty array if
    *  no normals are specified. */
    const std::vector<aiVector3D>& GetNormals() const;

    /** Return list of vertex indices. */
    const std::vector<unsigned int>& GetIndices() const;

private:
    std::vector<aiVector3D> m_vertices;
    std::vector<aiVector3D> m_normals;
    std::vector<unsigned int> m_indices;
};
/**
*  DOM class for FBX geometry of type "Line"
*/
class LineGeometry : public Geometry
{
public:
    /** The class constructor */
    LineGeometry(uint64_t id, const Element& element, const std::string& name, const Document& doc);

    /** The class destructor */
    virtual ~LineGeometry();

    /** Get a list of all vertex points, non-unique*/
    const std::vector<aiVector3D>& GetVertices() const;

    /** Return list of vertex indices. */
    const std::vector<int>& GetIndices() const;

private:
    std::vector<aiVector3D> m_vertices;
    std::vector<int> m_indices;
};

}
}

#endif // INCLUDED_AI_FBX_MESHGEOMETRY_H

