/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2016, assimp team

All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

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
---------------------------------------------------------------------------
*/

#ifndef ASSIMP_BUILD_NO_MMD_IMPORTER

#include "MMD/MMDImporter.h"
#include "MMD/MMDPmdParser.h"
#include "MMD/MMDPmxParser.h"
#include "MMD/MMDVmdParser.h"
#include "PostProcessing/ConvertToLHProcess.h"

#include <assimp/DefaultIOSystem.h>
#include <assimp/Importer.hpp>
#include <assimp/ai_assert.h>
#include <assimp/scene.h>

#include <fstream>
#include <iomanip>
#include <memory>

static const aiImporterDesc desc = {"MMD Importer",
                                    "",
                                    "",
                                    "surfaces supported?",
                                    aiImporterFlags_SupportTextFlavour,
                                    0,
                                    0,
                                    0,
                                    0,
                                    "pmx"};

namespace Assimp {

using namespace std;

// ------------------------------------------------------------------------------------------------
//  Default constructor
MMDImporter::MMDImporter()
: m_Buffer()
, m_strAbsPath("") {
    DefaultIOSystem io;
    m_strAbsPath = io.getOsSeparator();
}

// ------------------------------------------------------------------------------------------------
//  Destructor.
MMDImporter::~MMDImporter() {
    // empty
}

// ------------------------------------------------------------------------------------------------
//  Returns true, if file is an pmx file.
bool MMDImporter::CanRead(const std::string &pFile, IOSystem *pIOHandler,
                          bool checkSig) const {
  if (!checkSig) // Check File Extension
  {
    return SimpleExtensionCheck(pFile, "pmx");
  } else // Check file Header
  {
    static const char *pTokens[] = {"PMX "};
    return BaseImporter::SearchFileHeaderForToken(pIOHandler, pFile, pTokens, 1);
  }
}

// ------------------------------------------------------------------------------------------------
const aiImporterDesc *MMDImporter::GetInfo() const { return &desc; }

// ------------------------------------------------------------------------------------------------
//  MMD import implementation
void MMDImporter::InternReadFile(const std::string &file, aiScene *pScene,
                                 IOSystem * /*pIOHandler*/) {
  // Read file by istream
  std::filebuf fb;
  if (!fb.open(file, std::ios::in | std::ios::binary)) {
    throw DeadlyImportError("Failed to open file " + file + ".");
  }

  std::istream fileStream(&fb);

  // Get the file-size and validate it, throwing an exception when fails
  fileStream.seekg(0, fileStream.end);
  size_t fileSize = static_cast<size_t>(fileStream.tellg());
  fileStream.seekg(0, fileStream.beg);

  if (fileSize < sizeof(pmx::PmxModel)) {
    throw DeadlyImportError(file + " is too small.");
  }

  pmx::PmxModel model;
  model.Read(&fileStream);

  CreateDataFromImport(&model, pScene);
}

// ------------------------------------------------------------------------------------------------
void MMDImporter::CreateDataFromImport(const pmx::PmxModel *pModel,
                                       aiScene *pScene) {
  if (pModel == NULL) {
    return;
  }

  aiNode *pNode = new aiNode;
  if (!pModel->model_name.empty()) {
    pNode->mName.Set(pModel->model_name);
  }

  pScene->mRootNode = pNode;

  pNode = new aiNode;
  pScene->mRootNode->addChildren(1, &pNode);
  pNode->mName.Set(string(pModel->model_name) + string("_mesh"));

  // split mesh by materials
  pNode->mNumMeshes = pModel->material_count;
  pNode->mMeshes = new unsigned int[pNode->mNumMeshes];
  for (unsigned int index = 0; index < pNode->mNumMeshes; index++) {
    pNode->mMeshes[index] = index;
  }

  pScene->mNumMeshes = pModel->material_count;
  pScene->mMeshes = new aiMesh *[pScene->mNumMeshes];
  for (unsigned int i = 0, indexStart = 0; i < pScene->mNumMeshes; i++) {
    const int indexCount = pModel->materials[i].index_count;

    pScene->mMeshes[i] = CreateMesh(pModel, indexStart, indexCount);
    pScene->mMeshes[i]->mName = pModel->materials[i].material_name;
    pScene->mMeshes[i]->mMaterialIndex = i;
    indexStart += indexCount;
  }

  // create node hierarchy for bone position
  std::unique_ptr<aiNode *[]> ppNode(new aiNode *[pModel->bone_count]);
  for (auto i = 0; i < pModel->bone_count; i++) {
    ppNode[i] = new aiNode(pModel->bones[i].bone_name);
  }

  for (auto i = 0; i < pModel->bone_count; i++) {
    const pmx::PmxBone &bone = pModel->bones[i];

    if (bone.parent_index < 0) {
      pScene->mRootNode->addChildren(1, ppNode.get() + i);
    } else {
      ppNode[bone.parent_index]->addChildren(1, ppNode.get() + i);

      aiVector3D v3 = aiVector3D(
          bone.position[0] - pModel->bones[bone.parent_index].position[0],
          bone.position[1] - pModel->bones[bone.parent_index].position[1],
          bone.position[2] - pModel->bones[bone.parent_index].position[2]);
      aiMatrix4x4::Translation(v3, ppNode[i]->mTransformation);
    }
  }

  // create materials
  pScene->mNumMaterials = pModel->material_count;
  pScene->mMaterials = new aiMaterial *[pScene->mNumMaterials];
  for (unsigned int i = 0; i < pScene->mNumMaterials; i++) {
    pScene->mMaterials[i] = CreateMaterial(&pModel->materials[i], pModel);
  }

  // Convert everything to OpenGL space
  MakeLeftHandedProcess convertProcess;
  convertProcess.Execute(pScene);

  FlipUVsProcess uvFlipper;
  uvFlipper.Execute(pScene);

  FlipWindingOrderProcess windingFlipper;
  windingFlipper.Execute(pScene);
}

// ------------------------------------------------------------------------------------------------
aiMesh *MMDImporter::CreateMesh(const pmx::PmxModel *pModel,
                                const int indexStart, const int indexCount) {
  aiMesh *pMesh = new aiMesh;

  pMesh->mNumVertices = indexCount;

  pMesh->mNumFaces = indexCount / 3;
  pMesh->mFaces = new aiFace[pMesh->mNumFaces];

  const int numIndices = 3; // triangular face
  for (unsigned int index = 0; index < pMesh->mNumFaces; index++) {
    pMesh->mFaces[index].mNumIndices = numIndices;
    unsigned int *indices = new unsigned int[numIndices];
    indices[0] = numIndices * index;
    indices[1] = numIndices * index + 1;
    indices[2] = numIndices * index + 2;
    pMesh->mFaces[index].mIndices = indices;
  }

  pMesh->mVertices = new aiVector3D[pMesh->mNumVertices];
  pMesh->mNormals = new aiVector3D[pMesh->mNumVertices];
  pMesh->mTextureCoords[0] = new aiVector3D[pMesh->mNumVertices];
  pMesh->mNumUVComponents[0] = 2;

  // additional UVs
  for (int i = 1; i <= pModel->setting.uv; i++) {
    pMesh->mTextureCoords[i] = new aiVector3D[pMesh->mNumVertices];
    pMesh->mNumUVComponents[i] = 4;
  }

  map<int, vector<aiVertexWeight>> bone_vertex_map;

  // fill in contents and create bones
  for (int index = 0; index < indexCount; index++) {
    const pmx::PmxVertex *v =
        &pModel->vertices[pModel->indices[indexStart + index]];
    const float *position = v->position;
    pMesh->mVertices[index].Set(position[0], position[1], position[2]);
    const float *normal = v->normal;

    pMesh->mNormals[index].Set(normal[0], normal[1], normal[2]);
    pMesh->mTextureCoords[0][index].x = v->uv[0];
    pMesh->mTextureCoords[0][index].y = v->uv[1];

    for (int i = 1; i <= pModel->setting.uv; i++) {
      // TODO: wrong here? use quaternion transform?
      pMesh->mTextureCoords[i][index].x = v->uva[i][0];
      pMesh->mTextureCoords[i][index].y = v->uva[i][1];
    }

    // handle bone map
    const auto vsBDEF1_ptr =
        dynamic_cast<pmx::PmxVertexSkinningBDEF1 *>(v->skinning.get());
    const auto vsBDEF2_ptr =
        dynamic_cast<pmx::PmxVertexSkinningBDEF2 *>(v->skinning.get());
    const auto vsBDEF4_ptr =
        dynamic_cast<pmx::PmxVertexSkinningBDEF4 *>(v->skinning.get());
    const auto vsSDEF_ptr =
        dynamic_cast<pmx::PmxVertexSkinningSDEF *>(v->skinning.get());
    switch (v->skinning_type) {
    case pmx::PmxVertexSkinningType::BDEF1:
      bone_vertex_map[vsBDEF1_ptr->bone_index].push_back(
          aiVertexWeight(index, 1.0));
      break;
    case pmx::PmxVertexSkinningType::BDEF2:
      bone_vertex_map[vsBDEF2_ptr->bone_index1].push_back(
          aiVertexWeight(index, vsBDEF2_ptr->bone_weight));
      bone_vertex_map[vsBDEF2_ptr->bone_index2].push_back(
          aiVertexWeight(index, 1.0f - vsBDEF2_ptr->bone_weight));
      break;
    case pmx::PmxVertexSkinningType::BDEF4:
      bone_vertex_map[vsBDEF4_ptr->bone_index1].push_back(
          aiVertexWeight(index, vsBDEF4_ptr->bone_weight1));
      bone_vertex_map[vsBDEF4_ptr->bone_index2].push_back(
          aiVertexWeight(index, vsBDEF4_ptr->bone_weight2));
      bone_vertex_map[vsBDEF4_ptr->bone_index3].push_back(
          aiVertexWeight(index, vsBDEF4_ptr->bone_weight3));
      bone_vertex_map[vsBDEF4_ptr->bone_index4].push_back(
          aiVertexWeight(index, vsBDEF4_ptr->bone_weight4));
      break;
    case pmx::PmxVertexSkinningType::SDEF: // TODO: how to use sdef_c, sdef_r0,
                                           // sdef_r1?
      bone_vertex_map[vsSDEF_ptr->bone_index1].push_back(
          aiVertexWeight(index, vsSDEF_ptr->bone_weight));
      bone_vertex_map[vsSDEF_ptr->bone_index2].push_back(
          aiVertexWeight(index, 1.0f - vsSDEF_ptr->bone_weight));
      break;
    case pmx::PmxVertexSkinningType::QDEF:
      const auto vsQDEF_ptr =
          dynamic_cast<pmx::PmxVertexSkinningQDEF *>(v->skinning.get());
      bone_vertex_map[vsQDEF_ptr->bone_index1].push_back(
          aiVertexWeight(index, vsQDEF_ptr->bone_weight1));
      bone_vertex_map[vsQDEF_ptr->bone_index2].push_back(
          aiVertexWeight(index, vsQDEF_ptr->bone_weight2));
      bone_vertex_map[vsQDEF_ptr->bone_index3].push_back(
          aiVertexWeight(index, vsQDEF_ptr->bone_weight3));
      bone_vertex_map[vsQDEF_ptr->bone_index4].push_back(
          aiVertexWeight(index, vsQDEF_ptr->bone_weight4));
      break;
    }
  }

  // make all bones for each mesh
  // assign bone weights to skinned bones (otherwise just initialize)
  auto bone_ptr_ptr = new aiBone *[pModel->bone_count];
  pMesh->mNumBones = pModel->bone_count;
  pMesh->mBones = bone_ptr_ptr;
  for (auto ii = 0; ii < pModel->bone_count; ++ii) {
    auto pBone = new aiBone;
    const auto &pmxBone = pModel->bones[ii];
    pBone->mName = pmxBone.bone_name;
    aiVector3D pos(pmxBone.position[0], pmxBone.position[1], pmxBone.position[2]);
    aiMatrix4x4::Translation(-pos, pBone->mOffsetMatrix);
    auto it = bone_vertex_map.find(ii);
    if (it != bone_vertex_map.end()) {
      pBone->mNumWeights = static_cast<unsigned int>(it->second.size());
      pBone->mWeights = new aiVertexWeight[pBone->mNumWeights];
      for (unsigned int j = 0; j < pBone->mNumWeights; j++) {
          pBone->mWeights[j] = it->second[j];
      }
    }
    bone_ptr_ptr[ii] = pBone;
  }

  return pMesh;
}

// ------------------------------------------------------------------------------------------------
aiMaterial *MMDImporter::CreateMaterial(const pmx::PmxMaterial *pMat,
                                        const pmx::PmxModel *pModel) {
  aiMaterial *mat = new aiMaterial();
  aiString name(pMat->material_english_name);
  mat->AddProperty(&name, AI_MATKEY_NAME);

  aiColor3D diffuse(pMat->diffuse[0], pMat->diffuse[1], pMat->diffuse[2]);
  mat->AddProperty(&diffuse, 1, AI_MATKEY_COLOR_DIFFUSE);
  aiColor3D specular(pMat->specular[0], pMat->specular[1], pMat->specular[2]);
  mat->AddProperty(&specular, 1, AI_MATKEY_COLOR_SPECULAR);
  aiColor3D ambient(pMat->ambient[0], pMat->ambient[1], pMat->ambient[2]);
  mat->AddProperty(&ambient, 1, AI_MATKEY_COLOR_AMBIENT);

  float opacity = pMat->diffuse[3];
  mat->AddProperty(&opacity, 1, AI_MATKEY_OPACITY);
  float shininess = pMat->specularlity;
  mat->AddProperty(&shininess, 1, AI_MATKEY_SHININESS_STRENGTH);

  if(pMat->diffuse_texture_index >= 0) {
      aiString texture_path(pModel->textures[pMat->diffuse_texture_index]);
      mat->AddProperty(&texture_path, AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0));
  }

  int mapping_uvwsrc = 0;
  mat->AddProperty(&mapping_uvwsrc, 1,
                   AI_MATKEY_UVWSRC(aiTextureType_DIFFUSE, 0));

  return mat;
}

// ------------------------------------------------------------------------------------------------

} // Namespace Assimp

#endif // !! ASSIMP_BUILD_NO_MMD_IMPORTER
