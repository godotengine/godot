/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



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

/** @file Implementation of the post processing step to generate face
* normals for all imported faces.
*/


#include "GenFaceNormalsProcess.h"
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Exceptional.h>
#include <assimp/qnan.h>


using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
GenFaceNormalsProcess::GenFaceNormalsProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
GenFaceNormalsProcess::~GenFaceNormalsProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool GenFaceNormalsProcess::IsActive( unsigned int pFlags) const {
    force_ = (pFlags & aiProcess_ForceGenNormals) != 0;
    return  (pFlags & aiProcess_GenNormals) != 0;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void GenFaceNormalsProcess::Execute( aiScene* pScene) {
    ASSIMP_LOG_DEBUG("GenFaceNormalsProcess begin");

    if (pScene->mFlags & AI_SCENE_FLAGS_NON_VERBOSE_FORMAT) {
        throw DeadlyImportError("Post-processing order mismatch: expecting pseudo-indexed (\"verbose\") vertices here");
    }

    bool bHas = false;
    for( unsigned int a = 0; a < pScene->mNumMeshes; a++)   {
        if(this->GenMeshFaceNormals( pScene->mMeshes[a])) {
            bHas = true;
        }
    }
    if (bHas)   {
        ASSIMP_LOG_INFO("GenFaceNormalsProcess finished. "
            "Face normals have been calculated");
    } else {
        ASSIMP_LOG_DEBUG("GenFaceNormalsProcess finished. "
            "Normals are already there");
    }
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
bool GenFaceNormalsProcess::GenMeshFaceNormals (aiMesh* pMesh)
{
    if (NULL != pMesh->mNormals) {
        if (force_) delete[] pMesh->mNormals;
        else return false;
    }

    // If the mesh consists of lines and/or points but not of
    // triangles or higher-order polygons the normal vectors
    // are undefined.
    if (!(pMesh->mPrimitiveTypes & (aiPrimitiveType_TRIANGLE | aiPrimitiveType_POLYGON)))   {
        ASSIMP_LOG_INFO("Normal vectors are undefined for line and point meshes");
        return false;
    }

    // allocate an array to hold the output normals
    pMesh->mNormals = new aiVector3D[pMesh->mNumVertices];
    const float qnan = get_qnan();

    // iterate through all faces and compute per-face normals but store them per-vertex.
    for( unsigned int a = 0; a < pMesh->mNumFaces; a++) {
        const aiFace& face = pMesh->mFaces[a];
        if (face.mNumIndices < 3)   {
            // either a point or a line -> no well-defined normal vector
            for (unsigned int i = 0;i < face.mNumIndices;++i) {
                pMesh->mNormals[face.mIndices[i]] = aiVector3D(qnan);
            }
            continue;
        }

        const aiVector3D* pV1 = &pMesh->mVertices[face.mIndices[0]];
        const aiVector3D* pV2 = &pMesh->mVertices[face.mIndices[1]];
        const aiVector3D* pV3 = &pMesh->mVertices[face.mIndices[face.mNumIndices-1]];
        const aiVector3D vNor = ((*pV2 - *pV1) ^ (*pV3 - *pV1)).NormalizeSafe();

        for (unsigned int i = 0;i < face.mNumIndices;++i) {
            pMesh->mNormals[face.mIndices[i]] = vNor;
        }
    }
    return true;
}
