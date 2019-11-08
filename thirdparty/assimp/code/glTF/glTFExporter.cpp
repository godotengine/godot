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
#ifndef ASSIMP_BUILD_NO_EXPORT
#ifndef ASSIMP_BUILD_NO_GLTF_EXPORTER

#include "glTF/glTFExporter.h"
#include "glTF/glTFAssetWriter.h"
#include "PostProcessing/SplitLargeMeshes.h"

#include <assimp/Exceptional.h>
#include <assimp/StringComparison.h>
#include <assimp/ByteSwapper.h>
#include <assimp/SceneCombiner.h>
#include <assimp/version.h>
#include <assimp/IOSystem.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/material.h>
#include <assimp/scene.h>

// Header files, standard library.
#include <memory>
#include <inttypes.h>

#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
	// Header files, Open3DGC.
#	include <Open3DGC/o3dgcSC3DMCEncoder.h>
#endif

using namespace rapidjson;

using namespace Assimp;
using namespace glTF;

namespace Assimp {

    // ------------------------------------------------------------------------------------------------
    // Worker function for exporting a scene to GLTF. Prototyped and registered in Exporter.cpp
    void ExportSceneGLTF(const char* pFile, IOSystem* pIOSystem, const aiScene* pScene, const ExportProperties* pProperties)
    {
        // invoke the exporter
        glTFExporter exporter(pFile, pIOSystem, pScene, pProperties, false);
    }

    // ------------------------------------------------------------------------------------------------
    // Worker function for exporting a scene to GLB. Prototyped and registered in Exporter.cpp
    void ExportSceneGLB(const char* pFile, IOSystem* pIOSystem, const aiScene* pScene, const ExportProperties* pProperties)
    {
        // invoke the exporter
        glTFExporter exporter(pFile, pIOSystem, pScene, pProperties, true);
    }

} // end of namespace Assimp

glTFExporter::glTFExporter(const char* filename, IOSystem* pIOSystem, const aiScene* pScene,
                           const ExportProperties* pProperties, bool isBinary)
    : mFilename(filename)
    , mIOSystem(pIOSystem)
    , mProperties(pProperties)
{
    aiScene* sceneCopy_tmp;
    SceneCombiner::CopyScene(&sceneCopy_tmp, pScene);
    aiScene *sceneCopy(sceneCopy_tmp);

    SplitLargeMeshesProcess_Triangle tri_splitter;
    tri_splitter.SetLimit(0xffff);
    tri_splitter.Execute(sceneCopy);

    SplitLargeMeshesProcess_Vertex vert_splitter;
    vert_splitter.SetLimit(0xffff);
    vert_splitter.Execute(sceneCopy);

    mScene = sceneCopy;

    mAsset.reset( new glTF::Asset( pIOSystem ) );

    if (isBinary) {
        mAsset->SetAsBinary();
    }

    ExportMetadata();

    //for (unsigned int i = 0; i < pScene->mNumCameras; ++i) {}

    //for (unsigned int i = 0; i < pScene->mNumLights; ++i) {}

    ExportMaterials();

    if (mScene->mRootNode) {
        ExportNodeHierarchy(mScene->mRootNode);
    }

    ExportMeshes();

    //for (unsigned int i = 0; i < pScene->mNumTextures; ++i) {}

    ExportScene();

    ExportAnimations();

    glTF::AssetWriter writer(*mAsset);

    if (isBinary) {
        writer.WriteGLBFile(filename);
    } else {
        writer.WriteFile(filename);
    }
}

/*
 * Copy a 4x4 matrix from struct aiMatrix to typedef mat4.
 * Also converts from row-major to column-major storage.
 */
static void CopyValue(const aiMatrix4x4& v, glTF::mat4& o)
{
    o[ 0] = v.a1; o[ 1] = v.b1; o[ 2] = v.c1; o[ 3] = v.d1;
    o[ 4] = v.a2; o[ 5] = v.b2; o[ 6] = v.c2; o[ 7] = v.d2;
    o[ 8] = v.a3; o[ 9] = v.b3; o[10] = v.c3; o[11] = v.d3;
    o[12] = v.a4; o[13] = v.b4; o[14] = v.c4; o[15] = v.d4;
}

static void CopyValue(const aiMatrix4x4& v, aiMatrix4x4& o)
{
    o.a1 = v.a1; o.a2 = v.a2; o.a3 = v.a3; o.a4 = v.a4;
    o.b1 = v.b1; o.b2 = v.b2; o.b3 = v.b3; o.b4 = v.b4;
    o.c1 = v.c1; o.c2 = v.c2; o.c3 = v.c3; o.c4 = v.c4;
    o.d1 = v.d1; o.d2 = v.d2; o.d3 = v.d3; o.d4 = v.d4;
}

static void IdentityMatrix4(glTF::mat4& o)
{
    o[ 0] = 1; o[ 1] = 0; o[ 2] = 0; o[ 3] = 0;
    o[ 4] = 0; o[ 5] = 1; o[ 6] = 0; o[ 7] = 0;
    o[ 8] = 0; o[ 9] = 0; o[10] = 1; o[11] = 0;
    o[12] = 0; o[13] = 0; o[14] = 0; o[15] = 1;
}

inline Ref<Accessor> ExportData(Asset& a, std::string& meshName, Ref<Buffer>& buffer,
    unsigned int count, void* data, AttribType::Value typeIn, AttribType::Value typeOut, ComponentType compType, bool isIndices = false)
{
    if (!count || !data) return Ref<Accessor>();

    unsigned int numCompsIn = AttribType::GetNumComponents(typeIn);
    unsigned int numCompsOut = AttribType::GetNumComponents(typeOut);
    unsigned int bytesPerComp = ComponentTypeSize(compType);

    size_t offset = buffer->byteLength;
    // make sure offset is correctly byte-aligned, as required by spec
    size_t padding = offset % bytesPerComp;
    offset += padding;
    size_t length = count * numCompsOut * bytesPerComp;
    buffer->Grow(length + padding);

    // bufferView
    Ref<BufferView> bv = a.bufferViews.Create(a.FindUniqueID(meshName, "view"));
    bv->buffer = buffer;
    bv->byteOffset = unsigned(offset);
    bv->byteLength = length; //! The target that the WebGL buffer should be bound to.
    bv->target = isIndices ? BufferViewTarget_ELEMENT_ARRAY_BUFFER : BufferViewTarget_ARRAY_BUFFER;

    // accessor
    Ref<Accessor> acc = a.accessors.Create(a.FindUniqueID(meshName, "accessor"));
    acc->bufferView = bv;
    acc->byteOffset = 0;
    acc->byteStride = 0;
    acc->componentType = compType;
    acc->count = count;
    acc->type = typeOut;

    // calculate min and max values
    {
        // Allocate and initialize with large values.
        float float_MAX = 10000000000000.0f;
        for (unsigned int i = 0 ; i < numCompsOut ; i++) {
            acc->min.push_back( float_MAX);
            acc->max.push_back(-float_MAX);
        }

        // Search and set extreme values.
        float valueTmp;
        for (unsigned int i = 0 ; i < count       ; i++) {
            for (unsigned int j = 0 ; j < numCompsOut ; j++) {
                if (numCompsOut == 1) {
                  valueTmp = static_cast<unsigned short*>(data)[i];
                } else {
                  valueTmp = static_cast<aiVector3D*>(data)[i][j];
                }

                if (valueTmp < acc->min[j]) {
                    acc->min[j] = valueTmp;
                }
                if (valueTmp > acc->max[j]) {
                    acc->max[j] = valueTmp;
                }
            }
        }
    }

    // copy the data
    acc->WriteData(count, data, numCompsIn*bytesPerComp);

    return acc;
}

namespace {
    void GetMatScalar(const aiMaterial* mat, float& val, const char* propName, int type, int idx) {
        ai_assert( nullptr != mat );
        if ( nullptr != mat ) {
            mat->Get(propName, type, idx, val);
        }
    }
}

void glTFExporter::GetTexSampler(const aiMaterial* mat, glTF::TexProperty& prop)
{
    std::string samplerId = mAsset->FindUniqueID("", "sampler");
    prop.texture->sampler = mAsset->samplers.Create(samplerId);

    aiTextureMapMode mapU, mapV;
    aiGetMaterialInteger(mat,AI_MATKEY_MAPPINGMODE_U_DIFFUSE(0),(int*)&mapU);
    aiGetMaterialInteger(mat,AI_MATKEY_MAPPINGMODE_V_DIFFUSE(0),(int*)&mapV);

    switch (mapU) {
        case aiTextureMapMode_Wrap:
            prop.texture->sampler->wrapS = SamplerWrap_Repeat;
            break;
        case aiTextureMapMode_Clamp:
            prop.texture->sampler->wrapS = SamplerWrap_Clamp_To_Edge;
            break;
        case aiTextureMapMode_Mirror:
            prop.texture->sampler->wrapS = SamplerWrap_Mirrored_Repeat;
            break;
        case aiTextureMapMode_Decal:
        default:
            prop.texture->sampler->wrapS = SamplerWrap_Repeat;
            break;
    };

    switch (mapV) {
        case aiTextureMapMode_Wrap:
            prop.texture->sampler->wrapT = SamplerWrap_Repeat;
            break;
        case aiTextureMapMode_Clamp:
            prop.texture->sampler->wrapT = SamplerWrap_Clamp_To_Edge;
            break;
        case aiTextureMapMode_Mirror:
            prop.texture->sampler->wrapT = SamplerWrap_Mirrored_Repeat;
            break;
        case aiTextureMapMode_Decal:
        default:
            prop.texture->sampler->wrapT = SamplerWrap_Repeat;
            break;
    };

    // Hard coded Texture filtering options because I do not know where to find them in the aiMaterial.
    prop.texture->sampler->magFilter = SamplerMagFilter_Linear;
    prop.texture->sampler->minFilter = SamplerMinFilter_Linear;
}

void glTFExporter::GetMatColorOrTex(const aiMaterial* mat, glTF::TexProperty& prop, const char* propName, int type, int idx, aiTextureType tt)
{
    aiString tex;
    aiColor4D col;
    if (mat->GetTextureCount(tt) > 0) {
        if (mat->Get(AI_MATKEY_TEXTURE(tt, 0), tex) == AI_SUCCESS) {
            std::string path = tex.C_Str();

            if (path.size() > 0) {
                if (path[0] != '*') {
                    std::map<std::string, unsigned int>::iterator it = mTexturesByPath.find(path);
                    if (it != mTexturesByPath.end()) {
                        prop.texture = mAsset->textures.Get(it->second);
                    }
                }

                if (!prop.texture) {
                    std::string texId = mAsset->FindUniqueID("", "texture");
                    prop.texture = mAsset->textures.Create(texId);
                    mTexturesByPath[path] = prop.texture.GetIndex();

                    std::string imgId = mAsset->FindUniqueID("", "image");
                    prop.texture->source = mAsset->images.Create(imgId);

                    if (path[0] == '*') { // embedded
                        aiTexture* tex = mScene->mTextures[atoi(&path[1])];

                        uint8_t* data = reinterpret_cast<uint8_t*>(tex->pcData);
                        prop.texture->source->SetData(data, tex->mWidth, *mAsset);

                        if (tex->achFormatHint[0]) {
                            std::string mimeType = "image/";
                            mimeType += (memcmp(tex->achFormatHint, "jpg", 3) == 0) ? "jpeg" : tex->achFormatHint;
                            prop.texture->source->mimeType = mimeType;
                        }
                    }
                    else {
                        prop.texture->source->uri = path;
                    }

                    GetTexSampler(mat, prop);
                }
            }
        }
    }

    if (mat->Get(propName, type, idx, col) == AI_SUCCESS) {
        prop.color[0] = col.r; prop.color[1] = col.g; prop.color[2] = col.b; prop.color[3] = col.a;
    }
}


void glTFExporter::ExportMaterials()
{
    aiString aiName;
    for (unsigned int i = 0; i < mScene->mNumMaterials; ++i) {
        const aiMaterial* mat = mScene->mMaterials[i];


        std::string name;
        if (mat->Get(AI_MATKEY_NAME, aiName) == AI_SUCCESS) {
            name = aiName.C_Str();
        }
        name = mAsset->FindUniqueID(name, "material");

        Ref<Material> m = mAsset->materials.Create(name);

        GetMatColorOrTex(mat, m->ambient, AI_MATKEY_COLOR_AMBIENT, aiTextureType_AMBIENT);
        GetMatColorOrTex(mat, m->diffuse, AI_MATKEY_COLOR_DIFFUSE, aiTextureType_DIFFUSE);
        GetMatColorOrTex(mat, m->specular, AI_MATKEY_COLOR_SPECULAR, aiTextureType_SPECULAR);
        GetMatColorOrTex(mat, m->emission, AI_MATKEY_COLOR_EMISSIVE, aiTextureType_EMISSIVE);

        m->transparent = mat->Get(AI_MATKEY_OPACITY, m->transparency) == aiReturn_SUCCESS && m->transparency != 1.0;

        GetMatScalar(mat, m->shininess, AI_MATKEY_SHININESS);
    }
}

/*
 * Search through node hierarchy and find the node containing the given meshID.
 * Returns true on success, and false otherwise.
 */
bool FindMeshNode(Ref<Node>& nodeIn, Ref<Node>& meshNode, std::string meshID)
{
    for (unsigned int i = 0; i < nodeIn->meshes.size(); ++i) {
        if (meshID.compare(nodeIn->meshes[i]->id) == 0) {
          meshNode = nodeIn;
          return true;
        }
    }

    for (unsigned int i = 0; i < nodeIn->children.size(); ++i) {
        if(FindMeshNode(nodeIn->children[i], meshNode, meshID)) {
          return true;
        }
    }

    return false;
}

/*
 * Find the root joint of the skeleton.
 * Starts will any joint node and traces up the tree,
 * until a parent is found that does not have a jointName.
 * Returns the first parent Ref<Node> found that does not have a jointName.
 */
Ref<Node> FindSkeletonRootJoint(Ref<Skin>& skinRef)
{
    Ref<Node> startNodeRef;
    Ref<Node> parentNodeRef;

    // Arbitrarily use the first joint to start the search.
    startNodeRef = skinRef->jointNames[0];
    parentNodeRef = skinRef->jointNames[0];

    do {
        startNodeRef = parentNodeRef;
        parentNodeRef = startNodeRef->parent;
    } while (!parentNodeRef->jointName.empty());

    return parentNodeRef;
}

void ExportSkin(Asset& mAsset, const aiMesh* aimesh, Ref<Mesh>& meshRef, Ref<Buffer>& bufferRef, Ref<Skin>& skinRef, std::vector<aiMatrix4x4>& inverseBindMatricesData)
{
    if (aimesh->mNumBones < 1) {
        return;
    }

    // Store the vertex joint and weight data.
    const size_t NumVerts( aimesh->mNumVertices );
    vec4* vertexJointData = new vec4[ NumVerts ];
    vec4* vertexWeightData = new vec4[ NumVerts ];
    int* jointsPerVertex = new int[ NumVerts ];
    for (size_t i = 0; i < NumVerts; ++i) {
        jointsPerVertex[i] = 0;
        for (size_t j = 0; j < 4; ++j) {
            vertexJointData[i][j] = 0;
            vertexWeightData[i][j] = 0;
        }
    }

    for (unsigned int idx_bone = 0; idx_bone < aimesh->mNumBones; ++idx_bone) {
        const aiBone* aib = aimesh->mBones[idx_bone];

        // aib->mName   =====>  skinRef->jointNames
        // Find the node with id = mName.
        Ref<Node> nodeRef = mAsset.nodes.Get(aib->mName.C_Str());
        nodeRef->jointName = nodeRef->id;

        unsigned int jointNamesIndex = 0;
        bool addJointToJointNames = true;
        for ( unsigned int idx_joint = 0; idx_joint < skinRef->jointNames.size(); ++idx_joint) {
            if (skinRef->jointNames[idx_joint]->jointName.compare(nodeRef->jointName) == 0) {
                addJointToJointNames = false;
                jointNamesIndex = idx_joint;
            }
        }

        if (addJointToJointNames) {
            skinRef->jointNames.push_back(nodeRef);

            // aib->mOffsetMatrix   =====>  skinRef->inverseBindMatrices
            aiMatrix4x4 tmpMatrix4;
            CopyValue(aib->mOffsetMatrix, tmpMatrix4);
            inverseBindMatricesData.push_back(tmpMatrix4);
            jointNamesIndex = static_cast<unsigned int>(inverseBindMatricesData.size() - 1);
        }

        // aib->mWeights   =====>  vertexWeightData
        for (unsigned int idx_weights = 0; idx_weights < aib->mNumWeights; ++idx_weights) {
            unsigned int vertexId = aib->mWeights[idx_weights].mVertexId;
            float vertWeight      = aib->mWeights[idx_weights].mWeight;

            // A vertex can only have at most four joint weights. Ignore all others.
            if (jointsPerVertex[vertexId] > 3) {
                continue;
            }

            vertexJointData[vertexId][jointsPerVertex[vertexId]] = static_cast<float>(jointNamesIndex);
            vertexWeightData[vertexId][jointsPerVertex[vertexId]] = vertWeight;

            jointsPerVertex[vertexId] += 1;
        }

    } // End: for-loop mNumMeshes

    Mesh::Primitive& p = meshRef->primitives.back();
    Ref<Accessor> vertexJointAccessor = ExportData(mAsset, skinRef->id, bufferRef, aimesh->mNumVertices, vertexJointData, AttribType::VEC4, AttribType::VEC4, ComponentType_FLOAT);
    if ( vertexJointAccessor ) {
        p.attributes.joint.push_back( vertexJointAccessor );
    }

    Ref<Accessor> vertexWeightAccessor = ExportData(mAsset, skinRef->id, bufferRef, aimesh->mNumVertices, vertexWeightData, AttribType::VEC4, AttribType::VEC4, ComponentType_FLOAT);
    if ( vertexWeightAccessor ) {
        p.attributes.weight.push_back( vertexWeightAccessor );
    }
    delete[] jointsPerVertex;
    delete[] vertexWeightData;
    delete[] vertexJointData;
}

void glTFExporter::ExportMeshes()
{
    // Not for
    //     using IndicesType = decltype(aiFace::mNumIndices);
    // But yes for
    //     using IndicesType = unsigned short;
    // because "ComponentType_UNSIGNED_SHORT" used for indices. And it's a maximal type according to glTF specification.
    typedef unsigned short IndicesType;

    // Variables needed for compression. BEGIN.
    // Indices, not pointers - because pointer to buffer is changing while writing to it.
    size_t idx_srcdata_begin = 0; // Index of buffer before writing mesh data. Also, index of begin of coordinates array in buffer.
    size_t idx_srcdata_normal = SIZE_MAX;// Index of begin of normals array in buffer. SIZE_MAX - mean that mesh has no normals.
    std::vector<size_t> idx_srcdata_tc;// Array of indices. Every index point to begin of texture coordinates array in buffer.
    size_t idx_srcdata_ind;// Index of begin of coordinates indices array in buffer.
    bool comp_allow;// Point that data of current mesh can be compressed.
    // Variables needed for compression. END.

    std::string fname = std::string(mFilename);
    std::string bufferIdPrefix = fname.substr(0, fname.rfind(".gltf"));
    std::string bufferId = mAsset->FindUniqueID("", bufferIdPrefix.c_str());

    Ref<Buffer> b = mAsset->GetBodyBuffer();
    if (!b) {
       b = mAsset->buffers.Create(bufferId);
    }

    //----------------------------------------
    // Initialize variables for the skin
    bool createSkin = false;
    for (unsigned int idx_mesh = 0; idx_mesh < mScene->mNumMeshes; ++idx_mesh) {
        const aiMesh* aim = mScene->mMeshes[idx_mesh];
        if(aim->HasBones()) {
            createSkin = true;
            break;
        }
    }

    Ref<Skin> skinRef;
    std::string skinName = mAsset->FindUniqueID("skin", "skin");
    std::vector<aiMatrix4x4> inverseBindMatricesData;
    if(createSkin) {
        skinRef = mAsset->skins.Create(skinName);
        skinRef->name = skinName;
    }
    //----------------------------------------

	for (unsigned int idx_mesh = 0; idx_mesh < mScene->mNumMeshes; ++idx_mesh) {
		const aiMesh* aim = mScene->mMeshes[idx_mesh];

		// Check if compressing requested and mesh can be encoded.
#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
		comp_allow = mProperties->GetPropertyBool("extensions.Open3DGC.use", false);
#else
		comp_allow = false;
#endif

		if(comp_allow && (aim->mPrimitiveTypes == aiPrimitiveType_TRIANGLE) && (aim->mNumVertices > 0) && (aim->mNumFaces > 0))
		{
			idx_srcdata_tc.clear();
			idx_srcdata_tc.reserve(AI_MAX_NUMBER_OF_TEXTURECOORDS);
		}
		else
		{
			std::string msg;

			if(aim->mPrimitiveTypes != aiPrimitiveType_TRIANGLE)
				msg = "all primitives of the mesh must be a triangles.";
			else
				msg = "mesh must has vertices and faces.";

            ASSIMP_LOG_WARN_F("GLTF: can not use Open3DGC-compression: ", msg);
            comp_allow = false;
		}

        std::string meshId = mAsset->FindUniqueID(aim->mName.C_Str(), "mesh");
        Ref<Mesh> m = mAsset->meshes.Create(meshId);
        m->primitives.resize(1);
        Mesh::Primitive& p = m->primitives.back();

        p.material = mAsset->materials.Get(aim->mMaterialIndex);

		/******************* Vertices ********************/
		// If compression is used then you need parameters of uncompressed region: begin and size. At this step "begin" is stored.
		if(comp_allow) idx_srcdata_begin = b->byteLength;

        Ref<Accessor> v = ExportData(*mAsset, meshId, b, aim->mNumVertices, aim->mVertices, AttribType::VEC3, AttribType::VEC3, ComponentType_FLOAT);
		if (v) p.attributes.position.push_back(v);

		/******************** Normals ********************/
		if(comp_allow && (aim->mNormals != 0)) idx_srcdata_normal = b->byteLength;// Store index of normals array.

		Ref<Accessor> n = ExportData(*mAsset, meshId, b, aim->mNumVertices, aim->mNormals, AttribType::VEC3, AttribType::VEC3, ComponentType_FLOAT);
		if (n) p.attributes.normal.push_back(n);

		/************** Texture coordinates **************/
        for (int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++i) {
            // Flip UV y coords
            if (aim -> mNumUVComponents[i] > 1) {
                for (unsigned int j = 0; j < aim->mNumVertices; ++j) {
                    aim->mTextureCoords[i][j].y = 1 - aim->mTextureCoords[i][j].y;
                }
            }

            if (aim->mNumUVComponents[i] > 0) {
                AttribType::Value type = (aim->mNumUVComponents[i] == 2) ? AttribType::VEC2 : AttribType::VEC3;

				if(comp_allow) idx_srcdata_tc.push_back(b->byteLength);// Store index of texture coordinates array.

				Ref<Accessor> tc = ExportData(*mAsset, meshId, b, aim->mNumVertices, aim->mTextureCoords[i], AttribType::VEC3, type, ComponentType_FLOAT, false);
				if (tc) p.attributes.texcoord.push_back(tc);
			}
		}

		/*************** Vertices indices ****************/
		idx_srcdata_ind = b->byteLength;// Store index of indices array.

		if (aim->mNumFaces > 0) {
			std::vector<IndicesType> indices;
			unsigned int nIndicesPerFace = aim->mFaces[0].mNumIndices;
            indices.resize(aim->mNumFaces * nIndicesPerFace);
            for (size_t i = 0; i < aim->mNumFaces; ++i) {
                for (size_t j = 0; j < nIndicesPerFace; ++j) {
                    indices[i*nIndicesPerFace + j] = uint16_t(aim->mFaces[i].mIndices[j]);
                }
            }

			p.indices = ExportData(*mAsset, meshId, b, unsigned(indices.size()), &indices[0], AttribType::SCALAR, AttribType::SCALAR, ComponentType_UNSIGNED_SHORT, true);
		}

        switch (aim->mPrimitiveTypes) {
            case aiPrimitiveType_POLYGON:
                p.mode = PrimitiveMode_TRIANGLES; break; // TODO implement this
            case aiPrimitiveType_LINE:
                p.mode = PrimitiveMode_LINES; break;
            case aiPrimitiveType_POINT:
                p.mode = PrimitiveMode_POINTS; break;
            default: // aiPrimitiveType_TRIANGLE
                p.mode = PrimitiveMode_TRIANGLES;
        }

    /*************** Skins ****************/
    if(aim->HasBones()) {
        ExportSkin(*mAsset, aim, m, b, skinRef, inverseBindMatricesData);
    }

		/****************** Compression ******************/
		///TODO: animation: weights, joints.
		if(comp_allow)
		{
#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
			// Only one type of compression supported at now - Open3DGC.
			//
			o3dgc::BinaryStream bs;
			o3dgc::SC3DMCEncoder<IndicesType> encoder;
			o3dgc::IndexedFaceSet<IndicesType> comp_o3dgc_ifs;
			o3dgc::SC3DMCEncodeParams comp_o3dgc_params;

			//
			// Fill data for encoder.
			//
			// Quantization
			unsigned quant_coord = mProperties->GetPropertyInteger("extensions.Open3DGC.quantization.POSITION", 12);
			unsigned quant_normal = mProperties->GetPropertyInteger("extensions.Open3DGC.quantization.NORMAL", 10);
			unsigned quant_texcoord = mProperties->GetPropertyInteger("extensions.Open3DGC.quantization.TEXCOORD", 10);

			// Prediction
			o3dgc::O3DGCSC3DMCPredictionMode prediction_position = o3dgc::O3DGC_SC3DMC_PARALLELOGRAM_PREDICTION;
			o3dgc::O3DGCSC3DMCPredictionMode prediction_normal =  o3dgc::O3DGC_SC3DMC_SURF_NORMALS_PREDICTION;
			o3dgc::O3DGCSC3DMCPredictionMode prediction_texcoord = o3dgc::O3DGC_SC3DMC_PARALLELOGRAM_PREDICTION;

			// IndexedFacesSet: "Crease angle", "solid", "convex" are set to default.
			comp_o3dgc_ifs.SetCCW(true);
			comp_o3dgc_ifs.SetIsTriangularMesh(true);
			comp_o3dgc_ifs.SetNumFloatAttributes(0);
			// Coordinates
			comp_o3dgc_params.SetCoordQuantBits(quant_coord);
			comp_o3dgc_params.SetCoordPredMode(prediction_position);
			comp_o3dgc_ifs.SetNCoord(aim->mNumVertices);
			comp_o3dgc_ifs.SetCoord((o3dgc::Real* const)&b->GetPointer()[idx_srcdata_begin]);
			// Normals
			if(idx_srcdata_normal != SIZE_MAX)
			{
				comp_o3dgc_params.SetNormalQuantBits(quant_normal);
				comp_o3dgc_params.SetNormalPredMode(prediction_normal);
				comp_o3dgc_ifs.SetNNormal(aim->mNumVertices);
				comp_o3dgc_ifs.SetNormal((o3dgc::Real* const)&b->GetPointer()[idx_srcdata_normal]);
			}

			// Texture coordinates
			for(size_t num_tc = 0; num_tc < idx_srcdata_tc.size(); num_tc++)
			{
				size_t num = comp_o3dgc_ifs.GetNumFloatAttributes();

				comp_o3dgc_params.SetFloatAttributeQuantBits(static_cast<unsigned long>(num), quant_texcoord);
				comp_o3dgc_params.SetFloatAttributePredMode(static_cast<unsigned long>(num), prediction_texcoord);
				comp_o3dgc_ifs.SetNFloatAttribute(static_cast<unsigned long>(num), aim->mNumVertices);// number of elements.
				comp_o3dgc_ifs.SetFloatAttributeDim(static_cast<unsigned long>(num), aim->mNumUVComponents[num_tc]);// components per element: aiVector3D => x * float
				comp_o3dgc_ifs.SetFloatAttributeType(static_cast<unsigned long>(num), o3dgc::O3DGC_IFS_FLOAT_ATTRIBUTE_TYPE_TEXCOORD);
				comp_o3dgc_ifs.SetFloatAttribute(static_cast<unsigned long>(num), (o3dgc::Real* const)&b->GetPointer()[idx_srcdata_tc[num_tc]]);
				comp_o3dgc_ifs.SetNumFloatAttributes(static_cast<unsigned long>(num + 1));
			}

			// Coordinates indices
			comp_o3dgc_ifs.SetNCoordIndex(aim->mNumFaces);
			comp_o3dgc_ifs.SetCoordIndex((IndicesType* const)&b->GetPointer()[idx_srcdata_ind]);
			// Prepare to encoding
			comp_o3dgc_params.SetNumFloatAttributes(comp_o3dgc_ifs.GetNumFloatAttributes());
			if(mProperties->GetPropertyBool("extensions.Open3DGC.binary", true))
				comp_o3dgc_params.SetStreamType(o3dgc::O3DGC_STREAM_TYPE_BINARY);
			else
				comp_o3dgc_params.SetStreamType(o3dgc::O3DGC_STREAM_TYPE_ASCII);

			comp_o3dgc_ifs.ComputeMinMax(o3dgc::O3DGC_SC3DMC_MAX_ALL_DIMS);
			//
			// Encoding
			//
			encoder.Encode(comp_o3dgc_params, comp_o3dgc_ifs, bs);
			// Replace data in buffer.
			b->ReplaceData(idx_srcdata_begin, b->byteLength - idx_srcdata_begin, bs.GetBuffer(), bs.GetSize());
			//
			// Add information about extension to mesh.
			//
			// Create extension structure.
			Mesh::SCompression_Open3DGC* ext = new Mesh::SCompression_Open3DGC;

			// Fill it.
			ext->Buffer = b->id;
			ext->Offset = idx_srcdata_begin;
			ext->Count = b->byteLength - idx_srcdata_begin;
			ext->Binary = mProperties->GetPropertyBool("extensions.Open3DGC.binary");
			ext->IndicesCount = comp_o3dgc_ifs.GetNCoordIndex() * 3;
			ext->VerticesCount = comp_o3dgc_ifs.GetNCoord();
			// And assign to mesh.
			m->Extension.push_back(ext);
#endif
		}// if(comp_allow)
	}// for (unsigned int i = 0; i < mScene->mNumMeshes; ++i)

    //----------------------------------------
    // Finish the skin
    // Create the Accessor for skinRef->inverseBindMatrices
    if (createSkin) {
        mat4* invBindMatrixData = new mat4[inverseBindMatricesData.size()];
        for ( unsigned int idx_joint = 0; idx_joint < inverseBindMatricesData.size(); ++idx_joint) {
            CopyValue(inverseBindMatricesData[idx_joint], invBindMatrixData[idx_joint]);
        }

        Ref<Accessor> invBindMatrixAccessor = ExportData(*mAsset, skinName, b, static_cast<unsigned int>(inverseBindMatricesData.size()), invBindMatrixData, AttribType::MAT4, AttribType::MAT4, ComponentType_FLOAT);
        if (invBindMatrixAccessor) skinRef->inverseBindMatrices = invBindMatrixAccessor;

        // Identity Matrix   =====>  skinRef->bindShapeMatrix
        // Temporary. Hard-coded identity matrix here
        skinRef->bindShapeMatrix.isPresent = true;
        IdentityMatrix4(skinRef->bindShapeMatrix.value);

        // Find node that contains this mesh and add "skeletons" and "skin" attributes to that node.
        Ref<Node> rootNode = mAsset->nodes.Get(unsigned(0));
        Ref<Node> meshNode;
        std::string meshID = mAsset->meshes.Get(unsigned(0))->id;
        FindMeshNode(rootNode, meshNode, meshID);

        Ref<Node> rootJoint = FindSkeletonRootJoint(skinRef);
        meshNode->skeletons.push_back(rootJoint);
        meshNode->skin = skinRef;
    }
}

/*
 * Export the root node of the node hierarchy.
 * Calls ExportNode for all children.
 */
unsigned int glTFExporter::ExportNodeHierarchy(const aiNode* n)
{
    Ref<Node> node = mAsset->nodes.Create(mAsset->FindUniqueID(n->mName.C_Str(), "node"));

    if (!n->mTransformation.IsIdentity()) {
        node->matrix.isPresent = true;
        CopyValue(n->mTransformation, node->matrix.value);
    }

    for (unsigned int i = 0; i < n->mNumMeshes; ++i) {
        node->meshes.push_back(mAsset->meshes.Get(n->mMeshes[i]));
    }

    for (unsigned int i = 0; i < n->mNumChildren; ++i) {
        unsigned int idx = ExportNode(n->mChildren[i], node);
        node->children.push_back(mAsset->nodes.Get(idx));
    }

    return node.GetIndex();
}

/*
 * Export node and recursively calls ExportNode for all children.
 * Since these nodes are not the root node, we also export the parent Ref<Node>
 */
unsigned int glTFExporter::ExportNode(const aiNode* n, Ref<Node>& parent)
{
    Ref<Node> node = mAsset->nodes.Create(mAsset->FindUniqueID(n->mName.C_Str(), "node"));

    node->parent = parent;

    if (!n->mTransformation.IsIdentity()) {
        node->matrix.isPresent = true;
        CopyValue(n->mTransformation, node->matrix.value);
    }

    for (unsigned int i = 0; i < n->mNumMeshes; ++i) {
        node->meshes.push_back(mAsset->meshes.Get(n->mMeshes[i]));
    }

    for (unsigned int i = 0; i < n->mNumChildren; ++i) {
        unsigned int idx = ExportNode(n->mChildren[i], node);
        node->children.push_back(mAsset->nodes.Get(idx));
    }

    return node.GetIndex();
}


void glTFExporter::ExportScene()
{
    const char* sceneName = "defaultScene";
    Ref<Scene> scene = mAsset->scenes.Create(sceneName);

    // root node will be the first one exported (idx 0)
    if (mAsset->nodes.Size() > 0) {
        scene->nodes.push_back(mAsset->nodes.Get(0u));
    }

    // set as the default scene
    mAsset->scene = scene;
}

void glTFExporter::ExportMetadata()
{
    glTF::AssetMetadata& asset = mAsset->asset;
    asset.version = "1.0";

    char buffer[256];
    ai_snprintf(buffer, 256, "Open Asset Import Library (assimp v%d.%d.%d)",
        aiGetVersionMajor(), aiGetVersionMinor(), aiGetVersionRevision());

    asset.generator = buffer;
}

inline void ExtractAnimationData(Asset& mAsset, std::string& animId, Ref<Animation>& animRef, Ref<Buffer>& buffer, const aiNodeAnim* nodeChannel, float ticksPerSecond)
{
    // Loop over the data and check to see if it exactly matches an existing buffer.
    //    If yes, then reference the existing corresponding accessor.
    //    Otherwise, add to the buffer and create a new accessor.

    size_t counts[3] = {
        nodeChannel->mNumPositionKeys,
        nodeChannel->mNumScalingKeys,
        nodeChannel->mNumRotationKeys,
    };
    size_t numKeyframes = 1;
    for (int i = 0; i < 3; ++i) {
        if (counts[i] > numKeyframes) {
            numKeyframes = counts[i];
        }
    }

    //-------------------------------------------------------
    // Extract TIME parameter data.
    // Check if the timeStamps are the same for mPositionKeys, mRotationKeys, and mScalingKeys.
    if(nodeChannel->mNumPositionKeys > 0) {
        typedef float TimeType;
        std::vector<TimeType> timeData;
        timeData.resize(numKeyframes);
        for (size_t i = 0; i < numKeyframes; ++i) {
            size_t frameIndex = i * nodeChannel->mNumPositionKeys / numKeyframes;
            // mTime is measured in ticks, but GLTF time is measured in seconds, so convert.
            // Check if we have to cast type here. e.g. uint16_t()
            timeData[i] = static_cast<float>(nodeChannel->mPositionKeys[frameIndex].mTime / ticksPerSecond);
        }

        Ref<Accessor> timeAccessor = ExportData(mAsset, animId, buffer, static_cast<unsigned int>(numKeyframes), &timeData[0], AttribType::SCALAR, AttribType::SCALAR, ComponentType_FLOAT);
        if (timeAccessor) animRef->Parameters.TIME = timeAccessor;
    }

    //-------------------------------------------------------
    // Extract translation parameter data
    if(nodeChannel->mNumPositionKeys > 0) {
        C_STRUCT aiVector3D* translationData = new aiVector3D[numKeyframes];
        for (size_t i = 0; i < numKeyframes; ++i) {
            size_t frameIndex = i * nodeChannel->mNumPositionKeys / numKeyframes;
            translationData[i] = nodeChannel->mPositionKeys[frameIndex].mValue;
        }

        Ref<Accessor> tranAccessor = ExportData(mAsset, animId, buffer, static_cast<unsigned int>(numKeyframes), translationData, AttribType::VEC3, AttribType::VEC3, ComponentType_FLOAT);
        if ( tranAccessor ) {
            animRef->Parameters.translation = tranAccessor;
        }
        delete[] translationData;
    }

    //-------------------------------------------------------
    // Extract scale parameter data
    if(nodeChannel->mNumScalingKeys > 0) {
        C_STRUCT aiVector3D* scaleData = new aiVector3D[numKeyframes];
        for (size_t i = 0; i < numKeyframes; ++i) {
            size_t frameIndex = i * nodeChannel->mNumScalingKeys / numKeyframes;
            scaleData[i] = nodeChannel->mScalingKeys[frameIndex].mValue;
        }

        Ref<Accessor> scaleAccessor = ExportData(mAsset, animId, buffer, static_cast<unsigned int>(numKeyframes), scaleData, AttribType::VEC3, AttribType::VEC3, ComponentType_FLOAT);
        if ( scaleAccessor ) {
            animRef->Parameters.scale = scaleAccessor;
        }
        delete[] scaleData;
    }

    //-------------------------------------------------------
    // Extract rotation parameter data
    if(nodeChannel->mNumRotationKeys > 0) {
        vec4* rotationData = new vec4[numKeyframes];
        for (size_t i = 0; i < numKeyframes; ++i) {
            size_t frameIndex = i * nodeChannel->mNumRotationKeys / numKeyframes;
            rotationData[i][0] = nodeChannel->mRotationKeys[frameIndex].mValue.x;
            rotationData[i][1] = nodeChannel->mRotationKeys[frameIndex].mValue.y;
            rotationData[i][2] = nodeChannel->mRotationKeys[frameIndex].mValue.z;
            rotationData[i][3] = nodeChannel->mRotationKeys[frameIndex].mValue.w;
        }

        Ref<Accessor> rotAccessor = ExportData(mAsset, animId, buffer, static_cast<unsigned int>(numKeyframes), rotationData, AttribType::VEC4, AttribType::VEC4, ComponentType_FLOAT);
        if ( rotAccessor ) {
            animRef->Parameters.rotation = rotAccessor;
        }
        delete[] rotationData;
    }
}

void glTFExporter::ExportAnimations()
{
    Ref<Buffer> bufferRef = mAsset->buffers.Get(unsigned (0));

    for (unsigned int i = 0; i < mScene->mNumAnimations; ++i) {
        const aiAnimation* anim = mScene->mAnimations[i];

        std::string nameAnim = "anim";
        if (anim->mName.length > 0) {
            nameAnim = anim->mName.C_Str();
        }

        for (unsigned int channelIndex = 0; channelIndex < anim->mNumChannels; ++channelIndex) {
            const aiNodeAnim* nodeChannel = anim->mChannels[channelIndex];

            // It appears that assimp stores this type of animation as multiple animations.
            // where each aiNodeAnim in mChannels animates a specific node.
            std::string name = nameAnim + "_" + to_string(channelIndex);
            name = mAsset->FindUniqueID(name, "animation");
            Ref<Animation> animRef = mAsset->animations.Create(name);

            /******************* Parameters ********************/
            ExtractAnimationData(*mAsset, name, animRef, bufferRef, nodeChannel, static_cast<float>(anim->mTicksPerSecond));

            for (unsigned int j = 0; j < 3; ++j) {
                std::string channelType;
                int channelSize;
                switch (j) {
                    case 0:
                        channelType = "rotation";
                        channelSize = nodeChannel->mNumRotationKeys;
                        break;
                    case 1:
                        channelType = "scale";
                        channelSize = nodeChannel->mNumScalingKeys;
                        break;
                    case 2:
                        channelType = "translation";
                        channelSize = nodeChannel->mNumPositionKeys;
                        break;
                }

                if (channelSize < 1) { continue; }

                Animation::AnimChannel tmpAnimChannel;
                Animation::AnimSampler tmpAnimSampler;

                tmpAnimChannel.sampler = name + "_" + channelType;
                tmpAnimChannel.target.path = channelType;
                tmpAnimSampler.output = channelType;
                tmpAnimSampler.id = name + "_" + channelType;

                tmpAnimChannel.target.id = mAsset->nodes.Get(nodeChannel->mNodeName.C_Str());

                tmpAnimSampler.input = "TIME";
                tmpAnimSampler.interpolation = "LINEAR";

                animRef->Channels.push_back(tmpAnimChannel);
                animRef->Samplers.push_back(tmpAnimSampler);
            }

        }

        // Assimp documentation staes this is not used (not implemented)
        // for (unsigned int channelIndex = 0; channelIndex < anim->mNumMeshChannels; ++channelIndex) {
        //     const aiMeshAnim* meshChannel = anim->mMeshChannels[channelIndex];
        // }

    } // End: for-loop mNumAnimations
}


#endif // ASSIMP_BUILD_NO_GLTF_EXPORTER
#endif // ASSIMP_BUILD_NO_EXPORT
