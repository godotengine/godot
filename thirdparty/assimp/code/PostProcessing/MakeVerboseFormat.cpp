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
/** @file Implementation of the post processing step "MakeVerboseFormat"
*/


#include "MakeVerboseFormat.h"
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
MakeVerboseFormatProcess::MakeVerboseFormatProcess()
{
    // nothing to do here
}
// ------------------------------------------------------------------------------------------------
MakeVerboseFormatProcess::~MakeVerboseFormatProcess()
{
    // nothing to do here
}
// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void MakeVerboseFormatProcess::Execute( aiScene* pScene)
{
    ai_assert(NULL != pScene);
    ASSIMP_LOG_DEBUG("MakeVerboseFormatProcess begin");

    bool bHas = false;
    for( unsigned int a = 0; a < pScene->mNumMeshes; a++)
    {
        if( MakeVerboseFormat( pScene->mMeshes[a]))
            bHas = true;
    }
    if (bHas) {
        ASSIMP_LOG_INFO("MakeVerboseFormatProcess finished. There was much work to do ...");
    } else {
        ASSIMP_LOG_DEBUG("MakeVerboseFormatProcess. There was nothing to do.");
    }

    pScene->mFlags &= ~AI_SCENE_FLAGS_NON_VERBOSE_FORMAT;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
bool MakeVerboseFormatProcess::MakeVerboseFormat(aiMesh* pcMesh)
{
    ai_assert(NULL != pcMesh);

    unsigned int iOldNumVertices = pcMesh->mNumVertices;
    const unsigned int iNumVerts = pcMesh->mNumFaces*3;

    aiVector3D* pvPositions = new aiVector3D[ iNumVerts ];

    aiVector3D* pvNormals = NULL;
    if (pcMesh->HasNormals())
    {
        pvNormals = new aiVector3D[iNumVerts];
    }
    aiVector3D* pvTangents = NULL, *pvBitangents = NULL;
    if (pcMesh->HasTangentsAndBitangents())
    {
        pvTangents = new aiVector3D[iNumVerts];
        pvBitangents = new aiVector3D[iNumVerts];
    }

    aiVector3D* apvTextureCoords[AI_MAX_NUMBER_OF_TEXTURECOORDS] = {0};
    aiColor4D* apvColorSets[AI_MAX_NUMBER_OF_COLOR_SETS] = {0};

    unsigned int p = 0;
    while (pcMesh->HasTextureCoords(p))
        apvTextureCoords[p++] = new aiVector3D[iNumVerts];

    p = 0;
    while (pcMesh->HasVertexColors(p))
        apvColorSets[p++] = new aiColor4D[iNumVerts];

    // allocate enough memory to hold output bones and vertex weights ...
    std::vector<aiVertexWeight>* newWeights = new std::vector<aiVertexWeight>[pcMesh->mNumBones];
    for (unsigned int i = 0;i < pcMesh->mNumBones;++i) {
        newWeights[i].reserve(pcMesh->mBones[i]->mNumWeights*3);
    }

    // iterate through all faces and build a clean list
    unsigned int iIndex = 0;
    for (unsigned int a = 0; a< pcMesh->mNumFaces;++a)
    {
        aiFace* pcFace = &pcMesh->mFaces[a];
        for (unsigned int q = 0; q < pcFace->mNumIndices;++q,++iIndex)
        {
            // need to build a clean list of bones, too
            for (unsigned int i = 0;i < pcMesh->mNumBones;++i)
            {
                for (unsigned int a = 0;  a < pcMesh->mBones[i]->mNumWeights;a++)
                {
                    const aiVertexWeight& w = pcMesh->mBones[i]->mWeights[a];
                    if(pcFace->mIndices[q] == w.mVertexId)
                    {
                        aiVertexWeight wNew;
                        wNew.mVertexId = iIndex;
                        wNew.mWeight = w.mWeight;
                        newWeights[i].push_back(wNew);
                    }
                }
            }

            pvPositions[iIndex] = pcMesh->mVertices[pcFace->mIndices[q]];

            if (pcMesh->HasNormals())
            {
                pvNormals[iIndex] = pcMesh->mNormals[pcFace->mIndices[q]];
            }
            if (pcMesh->HasTangentsAndBitangents())
            {
                pvTangents[iIndex] = pcMesh->mTangents[pcFace->mIndices[q]];
                pvBitangents[iIndex] = pcMesh->mBitangents[pcFace->mIndices[q]];
            }

            unsigned int p = 0;
            while (pcMesh->HasTextureCoords(p))
            {
                apvTextureCoords[p][iIndex] = pcMesh->mTextureCoords[p][pcFace->mIndices[q]];
                ++p;
            }
            p = 0;
            while (pcMesh->HasVertexColors(p))
            {
                apvColorSets[p][iIndex] = pcMesh->mColors[p][pcFace->mIndices[q]];
                ++p;
            }
            pcFace->mIndices[q] = iIndex;
        }
    }



    // build output vertex weights
    for (unsigned int i = 0;i < pcMesh->mNumBones;++i)
    {
        delete [] pcMesh->mBones[i]->mWeights;
        if (!newWeights[i].empty()) {
            pcMesh->mBones[i]->mWeights = new aiVertexWeight[newWeights[i].size()];
            aiVertexWeight *weightToCopy = &( newWeights[i][0] );
            memcpy(pcMesh->mBones[i]->mWeights, weightToCopy,
                sizeof(aiVertexWeight) * newWeights[i].size());
        } else {
            pcMesh->mBones[i]->mWeights = NULL;
        }
    }
    delete[] newWeights;

    // delete the old members
    delete[] pcMesh->mVertices;
    pcMesh->mVertices = pvPositions;

    p = 0;
    while (pcMesh->HasTextureCoords(p))
    {
        delete[] pcMesh->mTextureCoords[p];
        pcMesh->mTextureCoords[p] = apvTextureCoords[p];
        ++p;
    }
    p = 0;
    while (pcMesh->HasVertexColors(p))
    {
        delete[] pcMesh->mColors[p];
        pcMesh->mColors[p] = apvColorSets[p];
        ++p;
    }
    pcMesh->mNumVertices = iNumVerts;

    if (pcMesh->HasNormals())
    {
        delete[] pcMesh->mNormals;
        pcMesh->mNormals = pvNormals;
    }
    if (pcMesh->HasTangentsAndBitangents())
    {
        delete[] pcMesh->mTangents;
        pcMesh->mTangents = pvTangents;
        delete[] pcMesh->mBitangents;
        pcMesh->mBitangents = pvBitangents;
    }
    return (pcMesh->mNumVertices != iOldNumVertices);
}


// ------------------------------------------------------------------------------------------------
bool IsMeshInVerboseFormat(const aiMesh* mesh) {
    // avoid slow vector<bool> specialization
    std::vector<unsigned int> seen(mesh->mNumVertices,0);
    for(unsigned int i = 0; i < mesh->mNumFaces; ++i) {
        const aiFace& f = mesh->mFaces[i];
        for(unsigned int j = 0; j < f.mNumIndices; ++j) {
            if(++seen[f.mIndices[j]] == 2) {
                // found a duplicate index
                return false;
            }
        }
    }

    return true;
}

// ------------------------------------------------------------------------------------------------
bool MakeVerboseFormatProcess::IsVerboseFormat(const aiScene* pScene) {
    for(unsigned int i = 0; i < pScene->mNumMeshes; ++i) {
        if(!IsMeshInVerboseFormat(pScene->mMeshes[i])) {
            return false;
        }
    }

    return true;
}
