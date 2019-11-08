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

#include "glTF/glTFImporter.h"
#include "glTF/glTFAsset.h"
#include "glTF/glTFAssetWriter.h"
#include "PostProcessing/MakeVerboseFormat.h"

#include <assimp/StringComparison.h>
#include <assimp/StringUtils.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/ai_assert.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/importerdesc.h>

#include <memory>

using namespace Assimp;
using namespace glTF;

//
// glTFImporter
//

static const aiImporterDesc desc = {
    "glTF Importer",
    "",
    "",
    "",
    aiImporterFlags_SupportTextFlavour | aiImporterFlags_SupportBinaryFlavour | aiImporterFlags_SupportCompressedFlavour
        | aiImporterFlags_LimitedSupport | aiImporterFlags_Experimental,
    0,
    0,
    0,
    0,
    "gltf glb"
};

glTFImporter::glTFImporter()
: BaseImporter()
, meshOffsets()
, embeddedTexIdxs()
, mScene( nullptr ) {
    // empty
}

glTFImporter::~glTFImporter() {
    // empty
}

const aiImporterDesc* glTFImporter::GetInfo() const {
    return &desc;
}

bool glTFImporter::CanRead(const std::string& pFile, IOSystem* pIOHandler, bool /* checkSig */) const {
    const std::string &extension = GetExtension(pFile);

    if (extension != "gltf" && extension != "glb") {
        return false;
    }

    if (pIOHandler) {
        glTF::Asset asset(pIOHandler);
        try {
            asset.Load(pFile, extension == "glb");
            std::string version = asset.asset.version;
            return !version.empty() && version[0] == '1';
        } catch (...) {
            return false;
        }
    }

    return false;
}

inline
void SetMaterialColorProperty(std::vector<int>& embeddedTexIdxs, Asset& /*r*/, glTF::TexProperty prop, aiMaterial* mat,
        aiTextureType texType, const char* pKey, unsigned int type, unsigned int idx) {
    if (prop.texture) {
        if (prop.texture->source) {
            aiString uri(prop.texture->source->uri);

            int texIdx = embeddedTexIdxs[prop.texture->source.GetIndex()];
            if (texIdx != -1) { // embedded
                // setup texture reference string (copied from ColladaLoader::FindFilenameForEffectTexture)
                uri.data[0] = '*';
                uri.length = 1 + ASSIMP_itoa10(uri.data + 1, MAXLEN - 1, texIdx);
            }

            mat->AddProperty(&uri, _AI_MATKEY_TEXTURE_BASE, texType, 0);
        }
    } else {
        aiColor4D col;
        CopyValue(prop.color, col);
        mat->AddProperty(&col, 1, pKey, type, idx);
    }
}

void glTFImporter::ImportMaterials(glTF::Asset& r) {
    mScene->mNumMaterials = unsigned(r.materials.Size());
    mScene->mMaterials = new aiMaterial*[mScene->mNumMaterials];

    for (unsigned int i = 0; i < mScene->mNumMaterials; ++i) {
        aiMaterial* aimat = mScene->mMaterials[i] = new aiMaterial();

        Material& mat = r.materials[i];

        /*if (!mat.name.empty())*/ {
            aiString str(mat.id /*mat.name*/);
            aimat->AddProperty(&str, AI_MATKEY_NAME);
        }

        SetMaterialColorProperty(embeddedTexIdxs, r, mat.ambient,  aimat, aiTextureType_AMBIENT,  AI_MATKEY_COLOR_AMBIENT );
        SetMaterialColorProperty(embeddedTexIdxs, r, mat.diffuse,  aimat, aiTextureType_DIFFUSE,  AI_MATKEY_COLOR_DIFFUSE );
        SetMaterialColorProperty(embeddedTexIdxs, r, mat.specular, aimat, aiTextureType_SPECULAR, AI_MATKEY_COLOR_SPECULAR);
        SetMaterialColorProperty(embeddedTexIdxs, r, mat.emission, aimat, aiTextureType_EMISSIVE, AI_MATKEY_COLOR_EMISSIVE);

        aimat->AddProperty(&mat.doubleSided, 1, AI_MATKEY_TWOSIDED);

        if (mat.transparent && (mat.transparency != 1.0f)) {
            aimat->AddProperty(&mat.transparency, 1, AI_MATKEY_OPACITY);
        }

        if (mat.shininess > 0.f) {
            aimat->AddProperty(&mat.shininess, 1, AI_MATKEY_SHININESS);
        }
    }

    if (mScene->mNumMaterials == 0) {
        mScene->mNumMaterials = 1;
        // Delete the array of length zero created above.
        delete[] mScene->mMaterials;
        mScene->mMaterials = new aiMaterial*[1];
        mScene->mMaterials[0] = new aiMaterial();
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

void glTFImporter::ImportMeshes(glTF::Asset& r)
{
    std::vector<aiMesh*> meshes;

    unsigned int k = 0;

    for (unsigned int m = 0; m < r.meshes.Size(); ++m) {
        Mesh& mesh = r.meshes[m];

		// Check if mesh extensions is used
		if(mesh.Extension.size() > 0)
		{
			for(Mesh::SExtension* cur_ext : mesh.Extension)
			{
#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
				if(cur_ext->Type == Mesh::SExtension::EType::Compression_Open3DGC)
				{
					// Limitations for meshes when using Open3DGC-compression.
					// It's a current limitation of sp... Specification have not this part still - about mesh compression. Why only one primitive?
					// Because glTF is very flexibly. But in fact it ugly flexible. Every primitive can has own set of accessors and accessors can
					// point to a-a-a-a-any part of buffer (through bufferview of course) and even to another buffer. We know that "Open3DGC-compression"
					// is applicable only to part of buffer. As we can't guaranty continuity of the data for decoder, we will limit quantity of primitives.
					// Yes indices, coordinates etc. still can br stored in different buffers, but with current specification it's a exporter problem.
					// Also primitive can has only one of "POSITION", "NORMAL" and less then "AI_MAX_NUMBER_OF_TEXTURECOORDS" of "TEXCOORD". All accessor
					// of primitive must point to one continuous region of the buffer.
					if(mesh.primitives.size() > 2) throw DeadlyImportError("GLTF: When using Open3DGC compression then only one primitive per mesh are allowed.");

					Mesh::SCompression_Open3DGC* o3dgc_ext = (Mesh::SCompression_Open3DGC*)cur_ext;
					Ref<Buffer> buf = r.buffers.Get(o3dgc_ext->Buffer);

					buf->EncodedRegion_SetCurrent(mesh.id);
				}
				else
#endif
				{
					throw DeadlyImportError("GLTF: Can not import mesh: unknown mesh extension (code: \"" + to_string(cur_ext->Type) +
											"\"), only Open3DGC is supported.");
				}
			}
		}// if(mesh.Extension.size() > 0)

		meshOffsets.push_back(k);
        k += unsigned(mesh.primitives.size());

        for (unsigned int p = 0; p < mesh.primitives.size(); ++p) {
            Mesh::Primitive& prim = mesh.primitives[p];

            aiMesh* aim = new aiMesh();
            meshes.push_back(aim);

            aim->mName = mesh.id;
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
                aim->mNumVertices = attr.position[0]->count;
                attr.position[0]->ExtractData(aim->mVertices);
			}

			if (attr.normal.size() > 0 && attr.normal[0]) attr.normal[0]->ExtractData(aim->mNormals);

            for (size_t tc = 0; tc < attr.texcoord.size() && tc < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++tc) {
                attr.texcoord[tc]->ExtractData(aim->mTextureCoords[tc]);
                aim->mNumUVComponents[tc] = attr.texcoord[tc]->GetNumComponents();

                aiVector3D* values = aim->mTextureCoords[tc];
                for (unsigned int i = 0; i < aim->mNumVertices; ++i) {
                    values[i].y = 1 - values[i].y; // Flip Y coords
                }
            }


            aiFace* faces = 0;
            unsigned int nFaces = 0;

            if (prim.indices) {
                unsigned int count = prim.indices->count;

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
                        SetFace(faces[0], data.GetUInt(0), data.GetUInt(1), data.GetUInt(2));
                        for (unsigned int i = 3; i < count; ++i) {
                            SetFace(faces[i - 2], faces[i - 1].mIndices[1], faces[i - 1].mIndices[2], data.GetUInt(i));
                        }
                        break;
                    }
                    case PrimitiveMode_TRIANGLE_FAN:
                        nFaces = count - 2;
                        faces = new aiFace[nFaces];
                        SetFace(faces[0], data.GetUInt(0), data.GetUInt(1), data.GetUInt(2));
                        for (unsigned int i = 3; i < count; ++i) {
                            SetFace(faces[i - 2], faces[0].mIndices[0], faces[i - 1].mIndices[2], data.GetUInt(i));
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
                    SetFace(faces[0], 0, 1, 2);
                    for (unsigned int i = 3; i < count; ++i) {
                        SetFace(faces[i - 2], faces[i - 1].mIndices[1], faces[i - 1].mIndices[2], i);
                    }
                    break;
                }
                case PrimitiveMode_TRIANGLE_FAN:
                    nFaces = count - 2;
                    faces = new aiFace[nFaces];
                    SetFace(faces[0], 0, 1, 2);
                    for (unsigned int i = 3; i < count; ++i) {
                        SetFace(faces[i - 2], faces[0].mIndices[0], faces[i - 1].mIndices[2], i);
                    }
                    break;
                }
            }

            if (faces) {
                aim->mFaces = faces;
                aim->mNumFaces = nFaces;
                ai_assert(CheckValidFacesIndices(faces, nFaces, aim->mNumVertices));
            }

            if (prim.material) {
                aim->mMaterialIndex = prim.material.GetIndex();
            }
        }
    }

    meshOffsets.push_back(k);

    CopyVector(meshes, mScene->mMeshes, mScene->mNumMeshes);
}

void glTFImporter::ImportCameras(glTF::Asset& r) {
    if (!r.cameras.Size()) {
        return;
    }

    mScene->mNumCameras = r.cameras.Size();
    mScene->mCameras = new aiCamera*[r.cameras.Size()];
    for (size_t i = 0; i < r.cameras.Size(); ++i) {
        Camera& cam = r.cameras[i];

        aiCamera* aicam = mScene->mCameras[i] = new aiCamera();

        if (cam.type == Camera::Perspective) {
            aicam->mAspect        = cam.perspective.aspectRatio;
            aicam->mHorizontalFOV = cam.perspective.yfov * ((aicam->mAspect == 0.f) ? 1.f : aicam->mAspect);
            aicam->mClipPlaneFar  = cam.perspective.zfar;
            aicam->mClipPlaneNear = cam.perspective.znear;
        } else {
            aicam->mClipPlaneFar = cam.ortographic.zfar;
            aicam->mClipPlaneNear = cam.ortographic.znear;
            aicam->mHorizontalFOV = 0.0;
            aicam->mAspect = 1.0f;
            if (0.f != cam.ortographic.ymag) {
                aicam->mAspect = cam.ortographic.xmag / cam.ortographic.ymag;
            }
        }
    }
}

void glTFImporter::ImportLights(glTF::Asset& r)
{
    if (!r.lights.Size()) return;

    mScene->mNumLights = r.lights.Size();
    mScene->mLights = new aiLight*[r.lights.Size()];

    for (size_t i = 0; i < r.lights.Size(); ++i) {
        Light& l = r.lights[i];

        aiLight* ail = mScene->mLights[i] = new aiLight();

        switch (l.type) {
            case Light::Type_directional:
                ail->mType = aiLightSource_DIRECTIONAL; break;

            case Light::Type_spot:
                ail->mType = aiLightSource_SPOT; break;

            case Light::Type_ambient:
                ail->mType = aiLightSource_AMBIENT; break;

            default: // Light::Type_point
                ail->mType = aiLightSource_POINT; break;
        }

        CopyValue(l.color, ail->mColorAmbient);
        CopyValue(l.color, ail->mColorDiffuse);
        CopyValue(l.color, ail->mColorSpecular);

        ail->mAngleOuterCone = l.falloffAngle;
        ail->mAngleInnerCone = l.falloffExponent; // TODO fix this, it does not look right at all

        ail->mAttenuationConstant  = l.constantAttenuation;
        ail->mAttenuationLinear    = l.linearAttenuation;
        ail->mAttenuationQuadratic = l.quadraticAttenuation;
    }
}


aiNode* ImportNode(aiScene* pScene, glTF::Asset& r, std::vector<unsigned int>& meshOffsets, glTF::Ref<glTF::Node>& ptr)
{
    Node& node = *ptr;

    aiNode* ainode = new aiNode(node.id);

    if (!node.children.empty()) {
        ainode->mNumChildren = unsigned(node.children.size());
        ainode->mChildren = new aiNode*[ainode->mNumChildren];

        for (unsigned int i = 0; i < ainode->mNumChildren; ++i) {
            aiNode* child = ImportNode(pScene, r, meshOffsets, node.children[i]);
            child->mParent = ainode;
            ainode->mChildren[i] = child;
        }
    }

    aiMatrix4x4& matrix = ainode->mTransformation;
    if (node.matrix.isPresent) {
        CopyValue(node.matrix.value, matrix);
    }
    else {
        if (node.translation.isPresent) {
            aiVector3D trans;
            CopyValue(node.translation.value, trans);
            aiMatrix4x4 t;
            aiMatrix4x4::Translation(trans, t);
            matrix = t * matrix;
        }

        if (node.scale.isPresent) {
            aiVector3D scal(1.f);
            CopyValue(node.scale.value, scal);
            aiMatrix4x4 s;
            aiMatrix4x4::Scaling(scal, s);
            matrix = s * matrix;
        }


        if (node.rotation.isPresent) {
            aiQuaternion rot;
            CopyValue(node.rotation.value, rot);
            matrix = aiMatrix4x4(rot.GetMatrix()) * matrix;
        }
    }

    if (!node.meshes.empty()) {
        int count = 0;
        for (size_t i = 0; i < node.meshes.size(); ++i) {
            int idx = node.meshes[i].GetIndex();
            count += meshOffsets[idx + 1] - meshOffsets[idx];
        }

        ainode->mNumMeshes = count;
        ainode->mMeshes = new unsigned int[count];

        int k = 0;
        for (size_t i = 0; i < node.meshes.size(); ++i) {
            int idx = node.meshes[i].GetIndex();
            for (unsigned int j = meshOffsets[idx]; j < meshOffsets[idx + 1]; ++j, ++k) {
                ainode->mMeshes[k] = j;
            }
        }
    }

    if (node.camera) {
        pScene->mCameras[node.camera.GetIndex()]->mName = ainode->mName;
    }

    if (node.light) {
        pScene->mLights[node.light.GetIndex()]->mName = ainode->mName;
    }

    return ainode;
}

void glTFImporter::ImportNodes(glTF::Asset& r)
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

void glTFImporter::ImportEmbeddedTextures(glTF::Asset& r)
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

void glTFImporter::InternReadFile(const std::string& pFile, aiScene* pScene, IOSystem* pIOHandler)
{
    // clean all member arrays
    meshOffsets.clear();
    embeddedTexIdxs.clear();

    this->mScene = pScene;

    // read the asset file
    glTF::Asset asset(pIOHandler);
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


    if (pScene->mNumMeshes == 0) {
        pScene->mFlags |= AI_SCENE_FLAGS_INCOMPLETE;
    }
}

#endif // ASSIMP_BUILD_NO_GLTF_IMPORTER

