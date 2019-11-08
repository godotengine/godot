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

#ifndef ASSIMP_BUILD_NO_GLTF_IMPORTER

#include "glTF2/glTF2Importer.h"
#include "glTF2/glTF2Asset.h"
#include "glTF2/glTF2AssetWriter.h"
#include "PostProcessing/MakeVerboseFormat.h"

#include <assimp/StringComparison.h>
#include <assimp/StringUtils.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/ai_assert.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/importerdesc.h>
#include <assimp/CreateAnimMesh.h>

#include <memory>
#include <unordered_map>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>

using namespace Assimp;
using namespace glTF2;
using namespace glTFCommon;

namespace {
    // generate bi-tangents from normals and tangents according to spec
    struct Tangent {
        aiVector3D xyz;
        ai_real w;
    };
} // namespace

//
// glTF2Importer
//

static const aiImporterDesc desc = {
    "glTF2 Importer",
    "",
    "",
    "",
    aiImporterFlags_SupportTextFlavour | aiImporterFlags_SupportBinaryFlavour | aiImporterFlags_LimitedSupport | aiImporterFlags_Experimental,
    0,
    0,
    0,
    0,
    "gltf glb"
};

glTF2Importer::glTF2Importer()
: BaseImporter()
, meshOffsets()
, embeddedTexIdxs()
, mScene( NULL ) {
    // empty
}

glTF2Importer::~glTF2Importer() {
    // empty
}

const aiImporterDesc* glTF2Importer::GetInfo() const
{
    return &desc;
}

bool glTF2Importer::CanRead(const std::string& pFile, IOSystem* pIOHandler, bool /* checkSig */) const
{
    const std::string &extension = GetExtension(pFile);

    if (extension != "gltf" && extension != "glb")
        return false;

    if (pIOHandler) {
        glTF2::Asset asset(pIOHandler);
        asset.Load(pFile, extension == "glb");
        std::string version = asset.asset.version;
        return !version.empty() && version[0] == '2';
    }

    return false;
}

static aiTextureMapMode ConvertWrappingMode(SamplerWrap gltfWrapMode)
{
    switch (gltfWrapMode) {
        case SamplerWrap::Mirrored_Repeat:
            return aiTextureMapMode_Mirror;

        case SamplerWrap::Clamp_To_Edge:
            return aiTextureMapMode_Clamp;

        case SamplerWrap::UNSET:
        case SamplerWrap::Repeat:
        default:
            return aiTextureMapMode_Wrap;
    }
}

/*static void CopyValue(const glTF2::vec3& v, aiColor3D& out)
{
    out.r = v[0]; out.g = v[1]; out.b = v[2];
}


static void CopyValue(const glTF2::vec4& v, aiColor4D& out)
{
    out.r = v[0]; out.g = v[1]; out.b = v[2]; out.a = v[3];
}*/

/*static void CopyValue(const glTF2::vec4& v, aiColor3D& out)
{
    out.r = v[0]; out.g = v[1]; out.b = v[2];
}*/

/*static void CopyValue(const glTF2::vec3& v, aiColor4D& out)
{
    out.r = v[0]; out.g = v[1]; out.b = v[2]; out.a = 1.0;
}

static void CopyValue(const glTF2::vec3& v, aiVector3D& out)
{
    out.x = v[0]; out.y = v[1]; out.z = v[2];
}

static void CopyValue(const glTF2::vec4& v, aiQuaternion& out)
{
    out.x = v[0]; out.y = v[1]; out.z = v[2]; out.w = v[3];
}*/

/*static void CopyValue(const glTF2::mat4& v, aiMatrix4x4& o)
{
    o.a1 = v[ 0]; o.b1 = v[ 1]; o.c1 = v[ 2]; o.d1 = v[ 3];
    o.a2 = v[ 4]; o.b2 = v[ 5]; o.c2 = v[ 6]; o.d2 = v[ 7];
    o.a3 = v[ 8]; o.b3 = v[ 9]; o.c3 = v[10]; o.d3 = v[11];
    o.a4 = v[12]; o.b4 = v[13]; o.c4 = v[14]; o.d4 = v[15];
}*/

inline void SetMaterialColorProperty(Asset& /*r*/, vec4& prop, aiMaterial* mat, const char* pKey, unsigned int type, unsigned int idx)
{
    aiColor4D col;
    CopyValue(prop, col);
    mat->AddProperty(&col, 1, pKey, type, idx);
}

inline void SetMaterialColorProperty(Asset& /*r*/, vec3& prop, aiMaterial* mat, const char* pKey, unsigned int type, unsigned int idx)
{
    aiColor4D col;
    glTFCommon::CopyValue(prop, col);
    mat->AddProperty(&col, 1, pKey, type, idx);
}

inline void SetMaterialTextureProperty(std::vector<int>& embeddedTexIdxs, Asset& /*r*/, glTF2::TextureInfo prop, aiMaterial* mat, aiTextureType texType, unsigned int texSlot = 0)
{
    if (prop.texture && prop.texture->source) {
        aiString uri(prop.texture->source->uri);

        int texIdx = embeddedTexIdxs[prop.texture->source.GetIndex()];
        if (texIdx != -1) { // embedded
            // setup texture reference string (copied from ColladaLoader::FindFilenameForEffectTexture)
            uri.data[0] = '*';
            uri.length = 1 + ASSIMP_itoa10(uri.data + 1, MAXLEN - 1, texIdx);
        }

        mat->AddProperty(&uri, AI_MATKEY_TEXTURE(texType, texSlot));
        mat->AddProperty(&prop.texCoord, 1, _AI_MATKEY_GLTF_TEXTURE_TEXCOORD_BASE, texType, texSlot);

        if (prop.texture->sampler) {
            Ref<Sampler> sampler = prop.texture->sampler;

            aiString name(sampler->name);
            aiString id(sampler->id);

            mat->AddProperty(&name, AI_MATKEY_GLTF_MAPPINGNAME(texType, texSlot));
            mat->AddProperty(&id, AI_MATKEY_GLTF_MAPPINGID(texType, texSlot));

            aiTextureMapMode wrapS = ConvertWrappingMode(sampler->wrapS);
            aiTextureMapMode wrapT = ConvertWrappingMode(sampler->wrapT);
            mat->AddProperty(&wrapS, 1, AI_MATKEY_MAPPINGMODE_U(texType, texSlot));
            mat->AddProperty(&wrapT, 1, AI_MATKEY_MAPPINGMODE_V(texType, texSlot));

            if (sampler->magFilter != SamplerMagFilter::UNSET) {
                mat->AddProperty(&sampler->magFilter, 1, AI_MATKEY_GLTF_MAPPINGFILTER_MAG(texType, texSlot));
            }

            if (sampler->minFilter != SamplerMinFilter::UNSET) {
                mat->AddProperty(&sampler->minFilter, 1, AI_MATKEY_GLTF_MAPPINGFILTER_MIN(texType, texSlot));
            }
        }
    }
}

inline void SetMaterialTextureProperty(std::vector<int>& embeddedTexIdxs, Asset& r, glTF2::NormalTextureInfo& prop, aiMaterial* mat, aiTextureType texType, unsigned int texSlot = 0)
{
    SetMaterialTextureProperty( embeddedTexIdxs, r, (glTF2::TextureInfo) prop, mat, texType, texSlot );

    if (prop.texture && prop.texture->source) {
         mat->AddProperty(&prop.scale, 1, AI_MATKEY_GLTF_TEXTURE_SCALE(texType, texSlot));
    }
}

inline void SetMaterialTextureProperty(std::vector<int>& embeddedTexIdxs, Asset& r, glTF2::OcclusionTextureInfo& prop, aiMaterial* mat, aiTextureType texType, unsigned int texSlot = 0)
{
    SetMaterialTextureProperty( embeddedTexIdxs, r, (glTF2::TextureInfo) prop, mat, texType, texSlot );

    if (prop.texture && prop.texture->source) {
        mat->AddProperty(&prop.strength, 1, AI_MATKEY_GLTF_TEXTURE_STRENGTH(texType, texSlot));
    }
}

static aiMaterial* ImportMaterial(std::vector<int>& embeddedTexIdxs, Asset& r, Material& mat)
{
    aiMaterial* aimat = new aiMaterial();

   if (!mat.name.empty()) {
        aiString str(mat.name);

        aimat->AddProperty(&str, AI_MATKEY_NAME);
    }

    SetMaterialColorProperty(r, mat.pbrMetallicRoughness.baseColorFactor, aimat, AI_MATKEY_COLOR_DIFFUSE);
    SetMaterialColorProperty(r, mat.pbrMetallicRoughness.baseColorFactor, aimat, AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR);

    SetMaterialTextureProperty(embeddedTexIdxs, r, mat.pbrMetallicRoughness.baseColorTexture, aimat, aiTextureType_DIFFUSE);
    SetMaterialTextureProperty(embeddedTexIdxs, r, mat.pbrMetallicRoughness.baseColorTexture, aimat, AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_TEXTURE);

    SetMaterialTextureProperty(embeddedTexIdxs, r, mat.pbrMetallicRoughness.metallicRoughnessTexture, aimat, AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE);

    aimat->AddProperty(&mat.pbrMetallicRoughness.metallicFactor, 1, AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR);
    aimat->AddProperty(&mat.pbrMetallicRoughness.roughnessFactor, 1, AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR);

    float roughnessAsShininess = 1 - mat.pbrMetallicRoughness.roughnessFactor;
    roughnessAsShininess *= roughnessAsShininess * 1000;
    aimat->AddProperty(&roughnessAsShininess, 1, AI_MATKEY_SHININESS);

    SetMaterialTextureProperty(embeddedTexIdxs, r, mat.normalTexture, aimat, aiTextureType_NORMALS);
    SetMaterialTextureProperty(embeddedTexIdxs, r, mat.occlusionTexture, aimat, aiTextureType_LIGHTMAP);
    SetMaterialTextureProperty(embeddedTexIdxs, r, mat.emissiveTexture, aimat, aiTextureType_EMISSIVE);
    SetMaterialColorProperty(r, mat.emissiveFactor, aimat, AI_MATKEY_COLOR_EMISSIVE);

    aimat->AddProperty(&mat.doubleSided, 1, AI_MATKEY_TWOSIDED);

    aiString alphaMode(mat.alphaMode);
    aimat->AddProperty(&alphaMode, AI_MATKEY_GLTF_ALPHAMODE);
    aimat->AddProperty(&mat.alphaCutoff, 1, AI_MATKEY_GLTF_ALPHACUTOFF);

    //pbrSpecularGlossiness
    if (mat.pbrSpecularGlossiness.isPresent) {
        PbrSpecularGlossiness &pbrSG = mat.pbrSpecularGlossiness.value;

        aimat->AddProperty(&mat.pbrSpecularGlossiness.isPresent, 1, AI_MATKEY_GLTF_PBRSPECULARGLOSSINESS);
        SetMaterialColorProperty(r, pbrSG.diffuseFactor, aimat, AI_MATKEY_COLOR_DIFFUSE);
        SetMaterialColorProperty(r, pbrSG.specularFactor, aimat, AI_MATKEY_COLOR_SPECULAR);

        float glossinessAsShininess = pbrSG.glossinessFactor * 1000.0f;
        aimat->AddProperty(&glossinessAsShininess, 1, AI_MATKEY_SHININESS);
        aimat->AddProperty(&pbrSG.glossinessFactor, 1, AI_MATKEY_GLTF_PBRSPECULARGLOSSINESS_GLOSSINESS_FACTOR);

        SetMaterialTextureProperty(embeddedTexIdxs, r, pbrSG.diffuseTexture, aimat, aiTextureType_DIFFUSE);

        SetMaterialTextureProperty(embeddedTexIdxs, r, pbrSG.specularGlossinessTexture, aimat, aiTextureType_SPECULAR);
    }
    if (mat.unlit) {
        aimat->AddProperty(&mat.unlit, 1, AI_MATKEY_GLTF_UNLIT);
    }

    return aimat;
}

void glTF2Importer::ImportMaterials(glTF2::Asset& r)
{
    const unsigned int numImportedMaterials = unsigned(r.materials.Size());
    Material defaultMaterial;

    mScene->mNumMaterials = numImportedMaterials + 1;
    mScene->mMaterials = new aiMaterial*[mScene->mNumMaterials];
    mScene->mMaterials[numImportedMaterials] = ImportMaterial(embeddedTexIdxs, r, defaultMaterial);

    for (unsigned int i = 0; i < numImportedMaterials; ++i) {
       mScene->mMaterials[i] = ImportMaterial(embeddedTexIdxs, r, r.materials[i]);
    }
}


static inline void SetFace(aiFace& face, int a)
{
    face.mNumIndices = 1;
    face.mIndices = new unsigned int[1];
    face.mIndices[0] = a;
}

static inline void SetFace(aiFace& face, int a, int b)
{
    face.mNumIndices = 2;
    face.mIndices = new unsigned int[2];
    face.mIndices[0] = a;
    face.mIndices[1] = b;
}

static inline void SetFace(aiFace& face, int a, int b, int c)
{
    face.mNumIndices = 3;
    face.mIndices = new unsigned int[3];
    face.mIndices[0] = a;
    face.mIndices[1] = b;
    face.mIndices[2] = c;
}

#ifdef ASSIMP_BUILD_DEBUG
static inline bool CheckValidFacesIndices(aiFace* faces, unsigned nFaces, unsigned nVerts)
{
    for (unsigned i = 0; i < nFaces; ++i) {
        for (unsigned j = 0; j < faces[i].mNumIndices; ++j) {
            unsigned idx = faces[i].mIndices[j];
            if (idx >= nVerts)
                return false;
        }
    }
    return true;
}
#endif // ASSIMP_BUILD_DEBUG

void glTF2Importer::ImportMeshes(glTF2::Asset& r)
{
    std::vector<aiMesh*> meshes;

    unsigned int k = 0;

    for (unsigned int m = 0; m < r.meshes.Size(); ++m) {
        Mesh& mesh = r.meshes[m];

        meshOffsets.push_back(k);
        k += unsigned(mesh.primitives.size());

        for (unsigned int p = 0; p < mesh.primitives.size(); ++p) {
            Mesh::Primitive& prim = mesh.primitives[p];

            aiMesh* aim = new aiMesh();
            meshes.push_back(aim);

            aim->mName = mesh.name.empty() ? mesh.id : mesh.name;

            if (mesh.primitives.size() > 1) {
                ai_uint32& len = aim->mName.length;
                aim->mName.data[len] = '-';
                len += 1 + ASSIMP_itoa10(aim->mName.data + len + 1, unsigned(MAXLEN - len - 1), p);
            }

            switch (prim.mode) {
                case PrimitiveMode_POINTS:
                    aim->mPrimitiveTypes |= aiPrimitiveType_POINT;
                    break;

                case PrimitiveMode_LINES:
                case PrimitiveMode_LINE_LOOP:
                case PrimitiveMode_LINE_STRIP:
                    aim->mPrimitiveTypes |= aiPrimitiveType_LINE;
                    break;

                case PrimitiveMode_TRIANGLES:
                case PrimitiveMode_TRIANGLE_STRIP:
                case PrimitiveMode_TRIANGLE_FAN:
                    aim->mPrimitiveTypes |= aiPrimitiveType_TRIANGLE;
                    break;

            }

            Mesh::Primitive::Attributes& attr = prim.attributes;

            if (attr.position.size() > 0 && attr.position[0]) {
                aim->mNumVertices = static_cast<unsigned int>(attr.position[0]->count);
                attr.position[0]->ExtractData(aim->mVertices);
            }

            if (attr.normal.size() > 0 && attr.normal[0]) {
                attr.normal[0]->ExtractData(aim->mNormals);

                // only extract tangents if normals are present
                if (attr.tangent.size() > 0 && attr.tangent[0]) {
                    // generate bitangents from normals and tangents according to spec
                    Tangent *tangents = nullptr;

                    attr.tangent[0]->ExtractData(tangents);

                    aim->mTangents = new aiVector3D[aim->mNumVertices];
                    aim->mBitangents = new aiVector3D[aim->mNumVertices];

                    for (unsigned int i = 0; i < aim->mNumVertices; ++i) {
                        aim->mTangents[i] = tangents[i].xyz;
                        aim->mBitangents[i] = (aim->mNormals[i] ^ tangents[i].xyz) * tangents[i].w;
                    }

                    delete [] tangents;
                }
            }

            for (size_t c = 0; c < attr.color.size() && c < AI_MAX_NUMBER_OF_COLOR_SETS; ++c) {
                if (attr.color[c]->count != aim->mNumVertices) {
                    DefaultLogger::get()->warn("Color stream size in mesh \"" + mesh.name +
                        "\" does not match the vertex count");
                    continue;
                }
                attr.color[c]->ExtractData(aim->mColors[c]);
            }
            for (size_t tc = 0; tc < attr.texcoord.size() && tc < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++tc) {
                if (attr.texcoord[tc]->count != aim->mNumVertices) {
                    DefaultLogger::get()->warn("Texcoord stream size in mesh \"" + mesh.name +
                                               "\" does not match the vertex count");
                    continue;
                }

                attr.texcoord[tc]->ExtractData(aim->mTextureCoords[tc]);
                aim->mNumUVComponents[tc] = attr.texcoord[tc]->GetNumComponents();

                aiVector3D* values = aim->mTextureCoords[tc];
                for (unsigned int i = 0; i < aim->mNumVertices; ++i) {
                    values[i].y = 1 - values[i].y; // Flip Y coords
                }
            }

            std::vector<Mesh::Primitive::Target>& targets = prim.targets;
            if (targets.size() > 0) {
                aim->mNumAnimMeshes = (unsigned int)targets.size();
                aim->mAnimMeshes = new aiAnimMesh*[aim->mNumAnimMeshes];
                for (size_t i = 0; i < targets.size(); i++) {
                    aim->mAnimMeshes[i] = aiCreateAnimMesh(aim);
                    aiAnimMesh& aiAnimMesh = *(aim->mAnimMeshes[i]);
                    Mesh::Primitive::Target& target = targets[i];

                    if (target.position.size() > 0) {
                        aiVector3D *positionDiff = nullptr;
                        target.position[0]->ExtractData(positionDiff);
                        for(unsigned int vertexId = 0; vertexId < aim->mNumVertices; vertexId++) {
                            aiAnimMesh.mVertices[vertexId] += positionDiff[vertexId];
                        }
                        delete [] positionDiff;
                    }
                    if (target.normal.size() > 0) {
                        aiVector3D *normalDiff = nullptr;
                        target.normal[0]->ExtractData(normalDiff);
                        for(unsigned int vertexId = 0; vertexId < aim->mNumVertices; vertexId++) {
                            aiAnimMesh.mNormals[vertexId] += normalDiff[vertexId];
                        }
                        delete [] normalDiff;
                    }
                    if (target.tangent.size() > 0) {
                        Tangent *tangent = nullptr;
                        attr.tangent[0]->ExtractData(tangent);

                        aiVector3D *tangentDiff = nullptr;
                        target.tangent[0]->ExtractData(tangentDiff);

                        for (unsigned int vertexId = 0; vertexId < aim->mNumVertices; ++vertexId) {
                            tangent[vertexId].xyz += tangentDiff[vertexId];
                            aiAnimMesh.mTangents[vertexId] = tangent[vertexId].xyz;
                            aiAnimMesh.mBitangents[vertexId] = (aiAnimMesh.mNormals[vertexId] ^ tangent[vertexId].xyz) * tangent[vertexId].w;
                        }
                        delete [] tangent;
                        delete [] tangentDiff;
                    }
                    if (mesh.weights.size() > i) {
                        aiAnimMesh.mWeight = mesh.weights[i];
                    }
                }
            }


            aiFace* faces = 0;
            size_t nFaces = 0;

            if (prim.indices) {
                size_t count = prim.indices->count;

                Accessor::Indexer data = prim.indices->GetIndexer();
                ai_assert(data.IsValid());

                switch (prim.mode) {
                    case PrimitiveMode_POINTS: {
                        nFaces = count;
                        faces = new aiFace[nFaces];
                        for (unsigned int i = 0; i < count; ++i) {
                            SetFace(faces[i], data.GetUInt(i));
                        }
                        break;
                    }

                    case PrimitiveMode_LINES: {
                        nFaces = count / 2;
                        if (nFaces * 2 != count) {
                            ASSIMP_LOG_WARN("The number of vertices was not compatible with the LINES mode. Some vertices were dropped.");
                            count = nFaces * 2;
                        }
                        faces = new aiFace[nFaces];
                        for (unsigned int i = 0; i < count; i += 2) {
                            SetFace(faces[i / 2], data.GetUInt(i), data.GetUInt(i + 1));
                        }
                        break;
                    }

                    case PrimitiveMode_LINE_LOOP:
                    case PrimitiveMode_LINE_STRIP: {
                        nFaces = count - ((prim.mode == PrimitiveMode_LINE_STRIP) ? 1 : 0);
                        faces = new aiFace[nFaces];
                        SetFace(faces[0], data.GetUInt(0), data.GetUInt(1));
                        for (unsigned int i = 2; i < count; ++i) {
                            SetFace(faces[i - 1], faces[i - 2].mIndices[1], data.GetUInt(i));
                        }
                        if (prim.mode == PrimitiveMode_LINE_LOOP) { // close the loop
                            SetFace(faces[count - 1], faces[count - 2].mIndices[1], faces[0].mIndices[0]);
                        }
                        break;
                    }

                    case PrimitiveMode_TRIANGLES: {
                        nFaces = count / 3;
                        if (nFaces * 3 != count) {
                            ASSIMP_LOG_WARN("The number of vertices was not compatible with the TRIANGLES mode. Some vertices were dropped.");
                            count = nFaces * 3;
                        }
                        faces = new aiFace[nFaces];
                        for (unsigned int i = 0; i < count; i += 3) {
                            SetFace(faces[i / 3], data.GetUInt(i), data.GetUInt(i + 1), data.GetUInt(i + 2));
                        }
                        break;
                    }
                    case PrimitiveMode_TRIANGLE_STRIP: {
                        nFaces = count - 2;
                        faces = new aiFace[nFaces];
                        for (unsigned int i = 0; i < nFaces; ++i) {
                            //The ordering is to ensure that the triangles are all drawn with the same orientation
                            if ((i + 1) % 2 == 0)
                            {
                                //For even n, vertices n + 1, n, and n + 2 define triangle n
                                SetFace(faces[i], data.GetUInt(i + 1), data.GetUInt(i), data.GetUInt(i + 2));
                            }
                            else
                            {
                                //For odd n, vertices n, n+1, and n+2 define triangle n
                                SetFace(faces[i], data.GetUInt(i), data.GetUInt(i + 1), data.GetUInt(i + 2));
                            }
                        }
                        break;
                    }
                    case PrimitiveMode_TRIANGLE_FAN:
                        nFaces = count - 2;
                        faces = new aiFace[nFaces];
                        SetFace(faces[0], data.GetUInt(0), data.GetUInt(1), data.GetUInt(2));
                        for (unsigned int i = 1; i < nFaces; ++i) {
                            SetFace(faces[i], faces[0].mIndices[0], faces[i - 1].mIndices[2], data.GetUInt(i + 2));
                        }
                        break;
                }
            }
            else { // no indices provided so directly generate from counts

                // use the already determined count as it includes checks
                unsigned int count = aim->mNumVertices;

                switch (prim.mode) {
                case PrimitiveMode_POINTS: {
                    nFaces = count;
                    faces = new aiFace[nFaces];
                    for (unsigned int i = 0; i < count; ++i) {
                        SetFace(faces[i], i);
                    }
                    break;
                }

                case PrimitiveMode_LINES: {
                    nFaces = count / 2;
                    if (nFaces * 2 != count) {
                        ASSIMP_LOG_WARN("The number of vertices was not compatible with the LINES mode. Some vertices were dropped.");
                        count = nFaces * 2;
                    }
                    faces = new aiFace[nFaces];
                    for (unsigned int i = 0; i < count; i += 2) {
                        SetFace(faces[i / 2], i, i + 1);
                    }
                    break;
                }

                case PrimitiveMode_LINE_LOOP:
                case PrimitiveMode_LINE_STRIP: {
                    nFaces = count - ((prim.mode == PrimitiveMode_LINE_STRIP) ? 1 : 0);
                    faces = new aiFace[nFaces];
                    SetFace(faces[0], 0, 1);
                    for (unsigned int i = 2; i < count; ++i) {
                        SetFace(faces[i - 1], faces[i - 2].mIndices[1], i);
                    }
                    if (prim.mode == PrimitiveMode_LINE_LOOP) { // close the loop
                        SetFace(faces[count - 1], faces[count - 2].mIndices[1], faces[0].mIndices[0]);
                    }
                    break;
                }

                case PrimitiveMode_TRIANGLES: {
                    nFaces = count / 3;
                    if (nFaces * 3 != count) {
                        ASSIMP_LOG_WARN("The number of vertices was not compatible with the TRIANGLES mode. Some vertices were dropped.");
                        count = nFaces * 3;
                    }
                    faces = new aiFace[nFaces];
                    for (unsigned int i = 0; i < count; i += 3) {
                        SetFace(faces[i / 3], i, i + 1, i + 2);
                    }
                    break;
                }
                case PrimitiveMode_TRIANGLE_STRIP: {
                    nFaces = count - 2;
                    faces = new aiFace[nFaces];
                    for (unsigned int i = 0; i < nFaces; ++i) {
                        //The ordering is to ensure that the triangles are all drawn with the same orientation
                        if ((i+1) % 2 == 0)
                        {
                            //For even n, vertices n + 1, n, and n + 2 define triangle n
                            SetFace(faces[i], i+1, i, i+2);
                        }
                        else
                        {
                            //For odd n, vertices n, n+1, and n+2 define triangle n
                            SetFace(faces[i], i, i+1, i+2);
                        }
                    }
                    break;
                }
                case PrimitiveMode_TRIANGLE_FAN:
                    nFaces = count - 2;
                    faces = new aiFace[nFaces];
                    SetFace(faces[0], 0, 1, 2);
                    for (unsigned int i = 1; i < nFaces; ++i) {
                        SetFace(faces[i], faces[0].mIndices[0], faces[i - 1].mIndices[2], i + 2);
                    }
                    break;
                }
            }

            if (faces) {
                aim->mFaces = faces;
                aim->mNumFaces = static_cast<unsigned int>(nFaces);
                ai_assert(CheckValidFacesIndices(faces, static_cast<unsigned>(nFaces), aim->mNumVertices));
            }

            if (prim.material) {
                aim->mMaterialIndex = prim.material.GetIndex();
            }
            else {
                aim->mMaterialIndex = mScene->mNumMaterials - 1;
            }

        }
    }

    meshOffsets.push_back(k);

    CopyVector(meshes, mScene->mMeshes, mScene->mNumMeshes);
}

void glTF2Importer::ImportCameras(glTF2::Asset& r)
{
    if (!r.cameras.Size()) return;

    mScene->mNumCameras = r.cameras.Size();
    mScene->mCameras = new aiCamera*[r.cameras.Size()];

    for (size_t i = 0; i < r.cameras.Size(); ++i) {
        Camera& cam = r.cameras[i];

        aiCamera* aicam = mScene->mCameras[i] = new aiCamera();

        // cameras point in -Z by default, rest is specified in node transform
        aicam->mLookAt = aiVector3D(0.f,0.f,-1.f);

        if (cam.type == Camera::Perspective) {

            aicam->mAspect        = cam.cameraProperties.perspective.aspectRatio;
            aicam->mHorizontalFOV = cam.cameraProperties.perspective.yfov * ((aicam->mAspect == 0.f) ? 1.f : aicam->mAspect);
            aicam->mClipPlaneFar  = cam.cameraProperties.perspective.zfar;
            aicam->mClipPlaneNear = cam.cameraProperties.perspective.znear;
        } else {
            aicam->mClipPlaneFar = cam.cameraProperties.ortographic.zfar;
            aicam->mClipPlaneNear = cam.cameraProperties.ortographic.znear;
            aicam->mHorizontalFOV = 0.0;
            aicam->mAspect = 1.0f;
            if (0.f != cam.cameraProperties.ortographic.ymag ) {
                aicam->mAspect = cam.cameraProperties.ortographic.xmag / cam.cameraProperties.ortographic.ymag;
            }
        }
    }
}

void glTF2Importer::ImportLights(glTF2::Asset& r)
{
    if (!r.lights.Size())
        return;

    mScene->mNumLights = r.lights.Size();
    mScene->mLights = new aiLight*[r.lights.Size()];

    for (size_t i = 0; i < r.lights.Size(); ++i) {
        Light& light = r.lights[i];

        aiLight* ail = mScene->mLights[i] = new aiLight();

        switch (light.type)
        {
        case Light::Directional:
            ail->mType = aiLightSource_DIRECTIONAL; break;
        case Light::Point:
            ail->mType = aiLightSource_POINT; break;
        case Light::Spot:
            ail->mType = aiLightSource_SPOT; break;
        }

        if (ail->mType != aiLightSource_POINT)
        {
            ail->mDirection = aiVector3D(0.0f, 0.0f, -1.0f);
            ail->mUp = aiVector3D(0.0f, 1.0f, 0.0f);
        }

        vec3 colorWithIntensity = { light.color[0] * light.intensity, light.color[1] * light.intensity, light.color[2] * light.intensity };
        CopyValue(colorWithIntensity, ail->mColorAmbient);
        CopyValue(colorWithIntensity, ail->mColorDiffuse);
        CopyValue(colorWithIntensity, ail->mColorSpecular);

        if (ail->mType == aiLightSource_DIRECTIONAL)
        {
            ail->mAttenuationConstant = 1.0;
            ail->mAttenuationLinear = 0.0;
            ail->mAttenuationQuadratic = 0.0;
        }
        else
        {
            //in PBR attenuation is calculated using inverse square law which can be expressed
            //using assimps equation: 1/(att0 + att1 * d + att2 * d*d) with the following parameters
            //this is correct equation for the case when range (see
            //https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual)
            //is not present. When range is not present it is assumed that it is infinite and so numerator is 1.
            //When range is present then numerator might be any value in range [0,1] and then assimps equation
            //will not suffice. In this case range is added into metadata in ImportNode function
            //and its up to implementation to read it when it wants to
            ail->mAttenuationConstant = 0.0;
            ail->mAttenuationLinear = 0.0;
            ail->mAttenuationQuadratic = 1.0;
        }

        if (ail->mType == aiLightSource_SPOT)
        {
            ail->mAngleInnerCone = light.innerConeAngle;
            ail->mAngleOuterCone = light.outerConeAngle;
        }
    }
}

static void GetNodeTransform(aiMatrix4x4& matrix, const glTF2::Node& node) {
    if (node.matrix.isPresent) {
        CopyValue(node.matrix.value, matrix);
    }
    else {
        if (node.translation.isPresent) {
            aiVector3D trans;
            CopyValue(node.translation.value, trans);
            aiMatrix4x4 t;
            aiMatrix4x4::Translation(trans, t);
            matrix = matrix * t;
        }

        if (node.rotation.isPresent) {
            aiQuaternion rot;
            CopyValue(node.rotation.value, rot);
            matrix = matrix * aiMatrix4x4(rot.GetMatrix());
        }

        if (node.scale.isPresent) {
            aiVector3D scal(1.f);
            CopyValue(node.scale.value, scal);
            aiMatrix4x4 s;
            aiMatrix4x4::Scaling(scal, s);
            matrix = matrix * s;
        }
    }
}

static void BuildVertexWeightMapping(Mesh::Primitive& primitive, std::vector<std::vector<aiVertexWeight>>& map)
{
    Mesh::Primitive::Attributes& attr = primitive.attributes;
    if (attr.weight.empty() || attr.joint.empty()) {
        return;
    }
    if (attr.weight[0]->count != attr.joint[0]->count) {
        return;
    }

    size_t num_vertices = attr.weight[0]->count;

    struct Weights { float values[4]; };
    Weights* weights = nullptr;
    attr.weight[0]->ExtractData(weights);

    struct Indices8 { uint8_t values[4]; };
    struct Indices16 { uint16_t values[4]; };
    Indices8* indices8 = nullptr;
    Indices16* indices16 = nullptr;
    if (attr.joint[0]->GetElementSize() == 4) {
        attr.joint[0]->ExtractData(indices8);
    }else {
        attr.joint[0]->ExtractData(indices16);
    }
    // 
    if (nullptr == indices8 && nullptr == indices16) {
        // Something went completely wrong!
        ai_assert(false);
        return;
    }

    for (size_t i = 0; i < num_vertices; ++i) {
        for (int j = 0; j < 4; ++j) {
            const unsigned int bone = (indices8!=nullptr) ? indices8[i].values[j] : indices16[i].values[j];
            const float weight = weights[i].values[j];
            if (weight > 0 && bone < map.size()) {
                map[bone].reserve(8);
                map[bone].emplace_back(static_cast<unsigned int>(i), weight);
            }
        }
    }

    delete[] weights;
    delete[] indices8;
    delete[] indices16;
}

static std::string GetNodeName(const Node& node)
{
    return node.name.empty() ? node.id : node.name;
}

aiNode* ImportNode(aiScene* pScene, glTF2::Asset& r, std::vector<unsigned int>& meshOffsets, glTF2::Ref<glTF2::Node>& ptr)
{
    Node& node = *ptr;

    aiNode* ainode = new aiNode(GetNodeName(node));

    if (!node.children.empty()) {
        ainode->mNumChildren = unsigned(node.children.size());
        ainode->mChildren = new aiNode*[ainode->mNumChildren];

        for (unsigned int i = 0; i < ainode->mNumChildren; ++i) {
            aiNode* child = ImportNode(pScene, r, meshOffsets, node.children[i]);
            child->mParent = ainode;
            ainode->mChildren[i] = child;
        }
    }

    GetNodeTransform(ainode->mTransformation, node);

    if (!node.meshes.empty()) {
        // GLTF files contain at most 1 mesh per node.
        assert(node.meshes.size() == 1);
        int mesh_idx = node.meshes[0].GetIndex();
        int count = meshOffsets[mesh_idx + 1] - meshOffsets[mesh_idx];

        ainode->mNumMeshes = count;
        ainode->mMeshes = new unsigned int[count];

        if (node.skin) {
            for (int primitiveNo = 0; primitiveNo < count; ++primitiveNo) {
                aiMesh* mesh = pScene->mMeshes[meshOffsets[mesh_idx]+primitiveNo];
                mesh->mNumBones = static_cast<unsigned int>(node.skin->jointNames.size());
                mesh->mBones = new aiBone*[mesh->mNumBones];

                // GLTF and Assimp choose to store bone weights differently.
                // GLTF has each vertex specify which bones influence the vertex.
                // Assimp has each bone specify which vertices it has influence over.
                // To convert this data, we first read over the vertex data and pull
                // out the bone-to-vertex mapping.  Then, when creating the aiBones,
                // we copy the bone-to-vertex mapping into the bone.  This is unfortunate
                // both because it's somewhat slow and because, for many applications,
                // we then need to reconvert the data back into the vertex-to-bone
                // mapping which makes things doubly-slow.
                std::vector<std::vector<aiVertexWeight>> weighting(mesh->mNumBones);
                BuildVertexWeightMapping(node.meshes[0]->primitives[primitiveNo], weighting);

                mat4* pbindMatrices = nullptr;
                node.skin->inverseBindMatrices->ExtractData(pbindMatrices);

                for (uint32_t i = 0; i < mesh->mNumBones; ++i) {
                    aiBone* bone = new aiBone();

                    Ref<Node> joint = node.skin->jointNames[i];
                    if (!joint->name.empty()) {
                      bone->mName = joint->name;
                    } else {
                      // Assimp expects each bone to have a unique name.
                      static const std::string kDefaultName = "bone_";
                      char postfix[10] = {0};
                      ASSIMP_itoa10(postfix, i);
                      bone->mName = (kDefaultName + postfix);
                    }
                    GetNodeTransform(bone->mOffsetMatrix, *joint);

                    CopyValue(pbindMatrices[i], bone->mOffsetMatrix);

                    std::vector<aiVertexWeight>& weights = weighting[i];

                    bone->mNumWeights = static_cast<uint32_t>(weights.size());
                    if (bone->mNumWeights > 0) {
                      bone->mWeights = new aiVertexWeight[bone->mNumWeights];
                      memcpy(bone->mWeights, weights.data(), bone->mNumWeights * sizeof(aiVertexWeight));
                    } else {
                      // Assimp expects all bones to have at least 1 weight.
                      bone->mWeights = new aiVertexWeight[1];
                      bone->mNumWeights = 1;
                      bone->mWeights->mVertexId = 0;
                      bone->mWeights->mWeight = 0.f;
                    }
                    mesh->mBones[i] = bone;
                }

                if (pbindMatrices) {
                    delete[] pbindMatrices;
                }
            }
        }

        int k = 0;
        for (unsigned int j = meshOffsets[mesh_idx]; j < meshOffsets[mesh_idx + 1]; ++j, ++k) {
            ainode->mMeshes[k] = j;
        }
    }

    if (node.camera) {
        pScene->mCameras[node.camera.GetIndex()]->mName = ainode->mName;
    }

    if (node.light) {
        pScene->mLights[node.light.GetIndex()]->mName = ainode->mName;

        //range is optional - see https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual
        //it is added to meta data of parent node, because there is no other place to put it
        if (node.light->range.isPresent)
        {
            ainode->mMetaData = aiMetadata::Alloc(1);
            ainode->mMetaData->Set(0, "PBR_LightRange", node.light->range.value);
        }
    }

    return ainode;
}

void glTF2Importer::ImportNodes(glTF2::Asset& r)
{
    if (!r.scene) return;

    std::vector< Ref<Node> > rootNodes = r.scene->nodes;

    // The root nodes
    unsigned int numRootNodes = unsigned(rootNodes.size());
    if (numRootNodes == 1) { // a single root node: use it
        mScene->mRootNode = ImportNode(mScene, r, meshOffsets, rootNodes[0]);
    }
    else if (numRootNodes > 1) { // more than one root node: create a fake root
        aiNode* root = new aiNode("ROOT");
        root->mChildren = new aiNode*[numRootNodes];
        for (unsigned int i = 0; i < numRootNodes; ++i) {
            aiNode* node = ImportNode(mScene, r, meshOffsets, rootNodes[i]);
            node->mParent = root;
            root->mChildren[root->mNumChildren++] = node;
        }
        mScene->mRootNode = root;
    }

    //if (!mScene->mRootNode) {
    //  mScene->mRootNode = new aiNode("EMPTY");
    //}
}

struct AnimationSamplers {
    AnimationSamplers()
    : translation(nullptr)
    , rotation(nullptr)
    , scale(nullptr)
    , weight(nullptr) {
        // empty
    }

    Animation::Sampler* translation;
    Animation::Sampler* rotation;
    Animation::Sampler* scale;
    Animation::Sampler* weight;
};

aiNodeAnim* CreateNodeAnim(glTF2::Asset& r, Node& node, AnimationSamplers& samplers)
{
    aiNodeAnim* anim = new aiNodeAnim();
    anim->mNodeName = GetNodeName(node);

    static const float kMillisecondsFromSeconds = 1000.f;

    if (samplers.translation) {
        float* times = nullptr;
        samplers.translation->input->ExtractData(times);
        aiVector3D* values = nullptr;
        samplers.translation->output->ExtractData(values);
        anim->mNumPositionKeys = static_cast<uint32_t>(samplers.translation->input->count);
        anim->mPositionKeys = new aiVectorKey[anim->mNumPositionKeys];
        for (unsigned int i = 0; i < anim->mNumPositionKeys; ++i) {
            anim->mPositionKeys[i].mTime = times[i] * kMillisecondsFromSeconds;
            anim->mPositionKeys[i].mValue = values[i];
        }
        delete[] times;
        delete[] values;
    } else if (node.translation.isPresent) {
        anim->mNumPositionKeys = 1;
        anim->mPositionKeys = new aiVectorKey[anim->mNumPositionKeys];
        anim->mPositionKeys->mTime = 0.f;
        anim->mPositionKeys->mValue.x = node.translation.value[0];
        anim->mPositionKeys->mValue.y = node.translation.value[1];
        anim->mPositionKeys->mValue.z = node.translation.value[2];
    }

    if (samplers.rotation) {
        float* times = nullptr;
        samplers.rotation->input->ExtractData(times);
        aiQuaternion* values = nullptr;
        samplers.rotation->output->ExtractData(values);
        anim->mNumRotationKeys = static_cast<uint32_t>(samplers.rotation->input->count);
        anim->mRotationKeys = new aiQuatKey[anim->mNumRotationKeys];
        for (unsigned int i = 0; i < anim->mNumRotationKeys; ++i) {
            anim->mRotationKeys[i].mTime = times[i] * kMillisecondsFromSeconds;
            anim->mRotationKeys[i].mValue.x = values[i].w;
            anim->mRotationKeys[i].mValue.y = values[i].x;
            anim->mRotationKeys[i].mValue.z = values[i].y;
            anim->mRotationKeys[i].mValue.w = values[i].z;
        }
        delete[] times;
        delete[] values;
    } else if (node.rotation.isPresent) {
        anim->mNumRotationKeys = 1;
        anim->mRotationKeys = new aiQuatKey[anim->mNumRotationKeys];
        anim->mRotationKeys->mTime = 0.f;
        anim->mRotationKeys->mValue.x = node.rotation.value[0];
        anim->mRotationKeys->mValue.y = node.rotation.value[1];
        anim->mRotationKeys->mValue.z = node.rotation.value[2];
        anim->mRotationKeys->mValue.w = node.rotation.value[3];
    }

    if (samplers.scale) {
        float* times = nullptr;
        samplers.scale->input->ExtractData(times);
        aiVector3D* values = nullptr;
        samplers.scale->output->ExtractData(values);
        anim->mNumScalingKeys = static_cast<uint32_t>(samplers.scale->input->count);
        anim->mScalingKeys = new aiVectorKey[anim->mNumScalingKeys];
        for (unsigned int i = 0; i < anim->mNumScalingKeys; ++i) {
            anim->mScalingKeys[i].mTime = times[i] * kMillisecondsFromSeconds;
            anim->mScalingKeys[i].mValue = values[i];
        }
        delete[] times;
        delete[] values;
    } else if (node.scale.isPresent) {
        anim->mNumScalingKeys = 1;
        anim->mScalingKeys = new aiVectorKey[anim->mNumScalingKeys];
        anim->mScalingKeys->mTime = 0.f;
        anim->mScalingKeys->mValue.x = node.scale.value[0];
        anim->mScalingKeys->mValue.y = node.scale.value[1];
        anim->mScalingKeys->mValue.z = node.scale.value[2];
    }

    return anim;
}

aiMeshMorphAnim* CreateMeshMorphAnim(glTF2::Asset& r, Node& node, AnimationSamplers& samplers)
{
    aiMeshMorphAnim* anim = new aiMeshMorphAnim();
    anim->mName = GetNodeName(node);

    static const float kMillisecondsFromSeconds = 1000.f;

    if (nullptr != samplers.weight) {
        float* times = nullptr;
        samplers.weight->input->ExtractData(times);
        float* values = nullptr;
        samplers.weight->output->ExtractData(values);
        anim->mNumKeys = static_cast<uint32_t>(samplers.weight->input->count);

        const unsigned int numMorphs = samplers.weight->output->count / anim->mNumKeys;

        anim->mKeys = new aiMeshMorphKey[anim->mNumKeys];
        unsigned int k = 0u;
        for (unsigned int i = 0u; i < anim->mNumKeys; ++i) {
            anim->mKeys[i].mTime = times[i] * kMillisecondsFromSeconds;
            anim->mKeys[i].mNumValuesAndWeights = numMorphs;
            anim->mKeys[i].mValues = new unsigned int[numMorphs];
            anim->mKeys[i].mWeights = new double[numMorphs];

            for (unsigned int j = 0u; j < numMorphs; ++j, ++k) {
                anim->mKeys[i].mValues[j] = j;
                anim->mKeys[i].mWeights[j] = ( 0.f > values[k] ) ? 0.f : values[k];
            }
        }

        delete[] times;
        delete[] values;
    }

    return anim;
}

std::unordered_map<unsigned int, AnimationSamplers> GatherSamplers(Animation& anim)
{
    std::unordered_map<unsigned int, AnimationSamplers> samplers;
    for (unsigned int c = 0; c < anim.channels.size(); ++c) {
        Animation::Channel& channel = anim.channels[c];
        if (channel.sampler >= static_cast<int>(anim.samplers.size())) {
            continue;
        }

        const unsigned int node_index = channel.target.node.GetIndex();

        AnimationSamplers& sampler = samplers[node_index];
        if (channel.target.path == AnimationPath_TRANSLATION) {
            sampler.translation = &anim.samplers[channel.sampler];
        } else if (channel.target.path == AnimationPath_ROTATION) {
            sampler.rotation = &anim.samplers[channel.sampler];
        } else if (channel.target.path == AnimationPath_SCALE) {
            sampler.scale = &anim.samplers[channel.sampler];
        } else if (channel.target.path == AnimationPath_WEIGHTS) {
            sampler.weight = &anim.samplers[channel.sampler];
        }
    }

    return samplers;
}

void glTF2Importer::ImportAnimations(glTF2::Asset& r)
{
    if (!r.scene) return;

    mScene->mNumAnimations = r.animations.Size();
    if (mScene->mNumAnimations == 0) {
        return;
    }

    mScene->mAnimations = new aiAnimation*[mScene->mNumAnimations];
    for (unsigned int i = 0; i < r.animations.Size(); ++i) {
        Animation& anim = r.animations[i];

        aiAnimation* ai_anim = new aiAnimation();
        ai_anim->mName = anim.name;
        ai_anim->mDuration = 0;
        ai_anim->mTicksPerSecond = 0;

        std::unordered_map<unsigned int, AnimationSamplers> samplers = GatherSamplers(anim);

        uint32_t numChannels = 0u;
        uint32_t numMorphMeshChannels = 0u;

        for (auto& iter : samplers) {
            if ((nullptr != iter.second.rotation) || (nullptr != iter.second.scale) || (nullptr != iter.second.translation)) {
                ++numChannels;
            }
            if (nullptr != iter.second.weight) {
                ++numMorphMeshChannels;
            }
        }

        ai_anim->mNumChannels = numChannels;
        if (ai_anim->mNumChannels > 0) {
            ai_anim->mChannels = new aiNodeAnim*[ai_anim->mNumChannels];
            int j = 0;
            for (auto& iter : samplers) {
                if ((nullptr != iter.second.rotation) || (nullptr != iter.second.scale) || (nullptr != iter.second.translation)) {
                    ai_anim->mChannels[j] = CreateNodeAnim(r, r.nodes[iter.first], iter.second);
                    ++j;
                }
            }
        }

        ai_anim->mNumMorphMeshChannels = numMorphMeshChannels;
        if (ai_anim->mNumMorphMeshChannels > 0) {
            ai_anim->mMorphMeshChannels = new aiMeshMorphAnim*[ai_anim->mNumMorphMeshChannels];
            int j = 0;
            for (auto& iter : samplers) {
                if (nullptr != iter.second.weight) {
                  ai_anim->mMorphMeshChannels[j] = CreateMeshMorphAnim(r, r.nodes[iter.first], iter.second);
                  ++j;
                }
            }
        }

        // Use the latest keyframe for the duration of the animation
        double maxDuration = 0;
        unsigned int maxNumberOfKeys = 0;
        for (unsigned int j = 0; j < ai_anim->mNumChannels; ++j) {
            auto chan = ai_anim->mChannels[j];
            if (chan->mNumPositionKeys) {
                auto lastPosKey = chan->mPositionKeys[chan->mNumPositionKeys - 1];
                if (lastPosKey.mTime > maxDuration) {
                    maxDuration = lastPosKey.mTime;
                }
                maxNumberOfKeys = std::max(maxNumberOfKeys, chan->mNumPositionKeys);
            }
            if (chan->mNumRotationKeys) {
                auto lastRotKey = chan->mRotationKeys[chan->mNumRotationKeys - 1];
                if (lastRotKey.mTime > maxDuration) {
                    maxDuration = lastRotKey.mTime;
                }
                maxNumberOfKeys = std::max(maxNumberOfKeys, chan->mNumRotationKeys);
            }
            if (chan->mNumScalingKeys) {
                auto lastScaleKey = chan->mScalingKeys[chan->mNumScalingKeys - 1];
                if (lastScaleKey.mTime > maxDuration) {
                    maxDuration = lastScaleKey.mTime;
                }
                maxNumberOfKeys = std::max(maxNumberOfKeys, chan->mNumScalingKeys);
            }
        }

        for (unsigned int j = 0; j < ai_anim->mNumMorphMeshChannels; ++j) {
            const auto* const chan = ai_anim->mMorphMeshChannels[j];

            if (0u != chan->mNumKeys) {
                const auto& lastKey = chan->mKeys[chan->mNumKeys - 1u];
                if (lastKey.mTime > maxDuration) {
                    maxDuration = lastKey.mTime;
                }
                maxNumberOfKeys = std::max(maxNumberOfKeys, chan->mNumKeys);
            }
        }

        ai_anim->mDuration = maxDuration;
        ai_anim->mTicksPerSecond = 1000.0;

        mScene->mAnimations[i] = ai_anim;
    }
}

void glTF2Importer::ImportEmbeddedTextures(glTF2::Asset& r)
{
    embeddedTexIdxs.resize(r.images.Size(), -1);

    int numEmbeddedTexs = 0;
    for (size_t i = 0; i < r.images.Size(); ++i) {
        if (r.images[i].HasData())
            numEmbeddedTexs += 1;
    }

    if (numEmbeddedTexs == 0)
        return;

    mScene->mTextures = new aiTexture*[numEmbeddedTexs];

    // Add the embedded textures
    for (size_t i = 0; i < r.images.Size(); ++i) {
        Image &img = r.images[i];
        if (!img.HasData()) continue;

        int idx = mScene->mNumTextures++;
        embeddedTexIdxs[i] = idx;

        aiTexture* tex = mScene->mTextures[idx] = new aiTexture();

        size_t length = img.GetDataLength();
        void* data = img.StealData();

        tex->mWidth = static_cast<unsigned int>(length);
        tex->mHeight = 0;
        tex->pcData = reinterpret_cast<aiTexel*>(data);

        if (!img.mimeType.empty()) {
            const char* ext = strchr(img.mimeType.c_str(), '/') + 1;
            if (ext) {
                if (strcmp(ext, "jpeg") == 0) ext = "jpg";

                size_t len = strlen(ext);
                if (len <= 3) {
                    strcpy(tex->achFormatHint, ext);
                }
            }
        }
    }
}

void glTF2Importer::InternReadFile(const std::string& pFile, aiScene* pScene, IOSystem* pIOHandler)
{
    // clean all member arrays
    meshOffsets.clear();
    embeddedTexIdxs.clear();

    this->mScene = pScene;

    // read the asset file
    glTF2::Asset asset(pIOHandler);
    asset.Load(pFile, GetExtension(pFile) == "glb");

    //
    // Copy the data out
    //

    ImportEmbeddedTextures(asset);
    ImportMaterials(asset);

    ImportMeshes(asset);

    ImportCameras(asset);
    ImportLights(asset);

    ImportNodes(asset);

    ImportAnimations(asset);

    if (pScene->mNumMeshes == 0) {
        pScene->mFlags |= AI_SCENE_FLAGS_INCOMPLETE;
    }
}

#endif // ASSIMP_BUILD_NO_GLTF_IMPORTER

