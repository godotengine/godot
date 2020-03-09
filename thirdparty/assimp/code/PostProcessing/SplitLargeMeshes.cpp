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

/** 
 *  @file Implementation of the SplitLargeMeshes postprocessing step
 */

// internal headers of the post-processing framework
#include "SplitLargeMeshes.h"
#include "ProcessHelper.h"

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
SplitLargeMeshesProcess_Triangle::SplitLargeMeshesProcess_Triangle() {
    LIMIT = AI_SLM_DEFAULT_MAX_TRIANGLES;
}

// ------------------------------------------------------------------------------------------------
SplitLargeMeshesProcess_Triangle::~SplitLargeMeshesProcess_Triangle() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool SplitLargeMeshesProcess_Triangle::IsActive( unsigned int pFlags) const {
    return (pFlags & aiProcess_SplitLargeMeshes) != 0;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void SplitLargeMeshesProcess_Triangle::Execute( aiScene* pScene) {
    if (0xffffffff == this->LIMIT || nullptr == pScene ) {
        return;
    }

    ASSIMP_LOG_DEBUG("SplitLargeMeshesProcess_Triangle begin");
    std::vector<std::pair<aiMesh*, unsigned int> > avList;

    for( unsigned int a = 0; a < pScene->mNumMeshes; ++a) {
        this->SplitMesh(a, pScene->mMeshes[a],avList);
    }

    if (avList.size() != pScene->mNumMeshes) {
        // it seems something has been split. rebuild the mesh list
        delete[] pScene->mMeshes;
        pScene->mNumMeshes = (unsigned int)avList.size();
        pScene->mMeshes = new aiMesh*[avList.size()];

        for (unsigned int i = 0; i < avList.size();++i) {
            pScene->mMeshes[i] = avList[i].first;
        }

        // now we need to update all nodes
        this->UpdateNode(pScene->mRootNode,avList);
        ASSIMP_LOG_INFO("SplitLargeMeshesProcess_Triangle finished. Meshes have been split");
    } else {
        ASSIMP_LOG_DEBUG("SplitLargeMeshesProcess_Triangle finished. There was nothing to do");
    }
}

// ------------------------------------------------------------------------------------------------
// Setup properties
void SplitLargeMeshesProcess_Triangle::SetupProperties( const Importer* pImp) {
    // get the current value of the split property
    this->LIMIT = pImp->GetPropertyInteger(AI_CONFIG_PP_SLM_TRIANGLE_LIMIT,AI_SLM_DEFAULT_MAX_TRIANGLES);
}

// ------------------------------------------------------------------------------------------------
// Update a node after some meshes have been split
void SplitLargeMeshesProcess_Triangle::UpdateNode(aiNode* pcNode,
        const std::vector<std::pair<aiMesh*, unsigned int> >& avList) {
    // for every index in out list build a new entry
    std::vector<unsigned int> aiEntries;
    aiEntries.reserve(pcNode->mNumMeshes + 1);
    for (unsigned int i = 0; i < pcNode->mNumMeshes;++i) {
        for (unsigned int a = 0; a < avList.size();++a) {
            if (avList[a].second == pcNode->mMeshes[i]) {
                aiEntries.push_back(a);
            }
        }
    }

    // now build the new list
    delete[] pcNode->mMeshes;
    pcNode->mNumMeshes = (unsigned int)aiEntries.size();
    pcNode->mMeshes = new unsigned int[pcNode->mNumMeshes];

    for (unsigned int b = 0; b < pcNode->mNumMeshes;++b) {
        pcNode->mMeshes[b] = aiEntries[b];
    }

    // recusively update all other nodes
    for (unsigned int i = 0; i < pcNode->mNumChildren;++i) {
        UpdateNode ( pcNode->mChildren[i], avList );
    }
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void SplitLargeMeshesProcess_Triangle::SplitMesh(
        unsigned int a,
        aiMesh* pMesh,
        std::vector<std::pair<aiMesh*, unsigned int> >& avList) {
    if (pMesh->mNumFaces > SplitLargeMeshesProcess_Triangle::LIMIT) {
        ASSIMP_LOG_INFO("Mesh exceeds the triangle limit. It will be split ...");

        // we need to split this mesh into sub meshes
        // determine the size of a submesh
        const unsigned int iSubMeshes = (pMesh->mNumFaces / LIMIT) + 1;

        const unsigned int iOutFaceNum = pMesh->mNumFaces / iSubMeshes;
        const unsigned int iOutVertexNum = iOutFaceNum * 3;

        // now generate all submeshes
        for (unsigned int i = 0; i < iSubMeshes;++i) {
            aiMesh* pcMesh          = new aiMesh;
            pcMesh->mNumFaces       = iOutFaceNum;
            pcMesh->mMaterialIndex  = pMesh->mMaterialIndex;

            // the name carries the adjacency information between the meshes
            pcMesh->mName = pMesh->mName;

            if (i == iSubMeshes-1) {
                pcMesh->mNumFaces = iOutFaceNum + (
                    pMesh->mNumFaces - iOutFaceNum * iSubMeshes);
            }
            // copy the list of faces
            pcMesh->mFaces = new aiFace[pcMesh->mNumFaces];

            const unsigned int iBase = iOutFaceNum * i;

            // get the total number of indices
            unsigned int iCnt = 0;
            for (unsigned int p = iBase; p < pcMesh->mNumFaces + iBase;++p) {
                iCnt += pMesh->mFaces[p].mNumIndices;
            }
            pcMesh->mNumVertices = iCnt;

            // allocate storage
            if (pMesh->mVertices != nullptr) {
                pcMesh->mVertices = new aiVector3D[iCnt];
            }

            if (pMesh->HasNormals()) {
                pcMesh->mNormals = new aiVector3D[iCnt];
            }

            if (pMesh->HasTangentsAndBitangents()) {
                pcMesh->mTangents = new aiVector3D[iCnt];
                pcMesh->mBitangents = new aiVector3D[iCnt];
            }

            // texture coordinates
            for (unsigned int c = 0;  c < AI_MAX_NUMBER_OF_TEXTURECOORDS;++c) {
                pcMesh->mNumUVComponents[c] = pMesh->mNumUVComponents[c];
                if (pMesh->HasTextureCoords( c)) {
                    pcMesh->mTextureCoords[c] = new aiVector3D[iCnt];
                }
            }

            // vertex colors
            for (unsigned int c = 0;  c < AI_MAX_NUMBER_OF_COLOR_SETS;++c) {
                if (pMesh->HasVertexColors( c)) {
                    pcMesh->mColors[c] = new aiColor4D[iCnt];
                }
            }

            if (pMesh->HasBones()) {
                // assume the number of bones won't change in most cases
                pcMesh->mBones = new aiBone*[pMesh->mNumBones];

                // iterate through all bones of the mesh and find those which
                // need to be copied to the split mesh
                std::vector<aiVertexWeight> avTempWeights;
                for (unsigned int p = 0; p < pcMesh->mNumBones;++p) {
                    aiBone* const bone = pcMesh->mBones[p];
                    avTempWeights.clear();
                    avTempWeights.reserve(bone->mNumWeights / iSubMeshes);

                    for (unsigned int q = 0; q < bone->mNumWeights;++q) {
                        aiVertexWeight& weight = bone->mWeights[q];
                        if(weight.mVertexId >= iBase && weight.mVertexId < iBase + iOutVertexNum) {
                            avTempWeights.push_back(weight);
                            weight = avTempWeights.back();
                            weight.mVertexId -= iBase;
                        }
                    }

                    if (!avTempWeights.empty()) {
                        // we'll need this bone. Copy it ...
                        aiBone* pc = new aiBone();
                        pcMesh->mBones[pcMesh->mNumBones++] = pc;
                        pc->mName = aiString(bone->mName);
                        pc->mNumWeights = (unsigned int)avTempWeights.size();
                        pc->mOffsetMatrix = bone->mOffsetMatrix;

                        // no need to reallocate the array for the last submesh.
                        // Here we can reuse the (large) source array, although
                        // we'll waste some memory
                        if (iSubMeshes-1 == i) {
                            pc->mWeights = bone->mWeights;
                            bone->mWeights = nullptr;
                        } else {
                            pc->mWeights = new aiVertexWeight[pc->mNumWeights];
                        }

                        // copy the weights
                        ::memcpy(pc->mWeights,&avTempWeights[0],sizeof(aiVertexWeight)*pc->mNumWeights);
                    }
                }
            }

            // (we will also need to copy the array of indices)
            unsigned int iCurrent = 0;
            for (unsigned int p = 0; p < pcMesh->mNumFaces;++p) {
                pcMesh->mFaces[p].mNumIndices = 3;
                // allocate a new array
                const unsigned int iTemp = p + iBase;
                const unsigned int iNumIndices = pMesh->mFaces[iTemp].mNumIndices;

                // setup face type and number of indices
                pcMesh->mFaces[p].mNumIndices = iNumIndices;
                unsigned int* pi = pMesh->mFaces[iTemp].mIndices;
                unsigned int* piOut = pcMesh->mFaces[p].mIndices = new unsigned int[iNumIndices];

                // need to update the output primitive types
                switch (iNumIndices) {
                case 1:
                    pcMesh->mPrimitiveTypes |= aiPrimitiveType_POINT;
                    break;
                case 2:
                    pcMesh->mPrimitiveTypes |= aiPrimitiveType_LINE;
                    break;
                case 3:
                    pcMesh->mPrimitiveTypes |= aiPrimitiveType_TRIANGLE;
                    break;
                default:
                    pcMesh->mPrimitiveTypes |= aiPrimitiveType_POLYGON;
                }

                // and copy the contents of the old array, offset by current base
                for (unsigned int v = 0; v < iNumIndices;++v) {
                    unsigned int iIndex = pi[v];
                    unsigned int iIndexOut = iCurrent++;
                    piOut[v] = iIndexOut;

                    // copy positions
                    if (pMesh->mVertices != nullptr) {
                        pcMesh->mVertices[iIndexOut] = pMesh->mVertices[iIndex];
                    }

                    // copy normals
                    if (pMesh->HasNormals()) {
                        pcMesh->mNormals[iIndexOut] = pMesh->mNormals[iIndex];
                    }

                    // copy tangents/bitangents
                    if (pMesh->HasTangentsAndBitangents()) {
                        pcMesh->mTangents[iIndexOut] = pMesh->mTangents[iIndex];
                        pcMesh->mBitangents[iIndexOut] = pMesh->mBitangents[iIndex];
                    }

                    // texture coordinates
                    for (unsigned int c = 0;  c < AI_MAX_NUMBER_OF_TEXTURECOORDS;++c) {
                        if (pMesh->HasTextureCoords( c ) ) {
                            pcMesh->mTextureCoords[c][iIndexOut] = pMesh->mTextureCoords[c][iIndex];
                        }
                    }
                    // vertex colors
                    for (unsigned int c = 0;  c < AI_MAX_NUMBER_OF_COLOR_SETS;++c) {
                        if (pMesh->HasVertexColors( c)) {
                            pcMesh->mColors[c][iIndexOut] = pMesh->mColors[c][iIndex];
                        }
                    }
                }
            }

            // add the newly created mesh to the list
            avList.push_back(std::pair<aiMesh*, unsigned int>(pcMesh,a));
        }

        // now delete the old mesh data
        delete pMesh;
    } else {
        avList.push_back(std::pair<aiMesh*, unsigned int>(pMesh,a));
    }
}

// ------------------------------------------------------------------------------------------------
SplitLargeMeshesProcess_Vertex::SplitLargeMeshesProcess_Vertex() {
    LIMIT = AI_SLM_DEFAULT_MAX_VERTICES;
}

// ------------------------------------------------------------------------------------------------
SplitLargeMeshesProcess_Vertex::~SplitLargeMeshesProcess_Vertex() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool SplitLargeMeshesProcess_Vertex::IsActive( unsigned int pFlags) const {
    return (pFlags & aiProcess_SplitLargeMeshes) != 0;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void SplitLargeMeshesProcess_Vertex::Execute( aiScene* pScene) {
    if (0xffffffff == this->LIMIT || nullptr == pScene ) {
        return;
    }

    ASSIMP_LOG_DEBUG("SplitLargeMeshesProcess_Vertex begin");

    std::vector<std::pair<aiMesh*, unsigned int> > avList;

    //Check for point cloud first, 
    //Do not process point cloud, splitMesh works only with faces data
    for (unsigned int a = 0; a < pScene->mNumMeshes; a++) {
        if ( pScene->mMeshes[a]->mPrimitiveTypes == aiPrimitiveType_POINT ) {
            return;
        }
    }

    for( unsigned int a = 0; a < pScene->mNumMeshes; ++a ) {
        this->SplitMesh(a, pScene->mMeshes[a], avList);
    }

    if (avList.size() != pScene->mNumMeshes) {
        // it seems something has been split. rebuild the mesh list
        delete[] pScene->mMeshes;
        pScene->mNumMeshes = (unsigned int)avList.size();
        pScene->mMeshes = new aiMesh*[avList.size()];

        for (unsigned int i = 0; i < avList.size();++i) {
            pScene->mMeshes[i] = avList[i].first;
        }

        // now we need to update all nodes
        SplitLargeMeshesProcess_Triangle::UpdateNode(pScene->mRootNode,avList);
        ASSIMP_LOG_INFO("SplitLargeMeshesProcess_Vertex finished. Meshes have been split");
    } else {
        ASSIMP_LOG_DEBUG("SplitLargeMeshesProcess_Vertex finished. There was nothing to do");
    }
}

// ------------------------------------------------------------------------------------------------
// Setup properties
void SplitLargeMeshesProcess_Vertex::SetupProperties( const Importer* pImp) {
    this->LIMIT = pImp->GetPropertyInteger(AI_CONFIG_PP_SLM_VERTEX_LIMIT,AI_SLM_DEFAULT_MAX_VERTICES);
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void SplitLargeMeshesProcess_Vertex::SplitMesh(
        unsigned int a,
        aiMesh* pMesh,
        std::vector<std::pair<aiMesh*, unsigned int> >& avList) {
    if (pMesh->mNumVertices > SplitLargeMeshesProcess_Vertex::LIMIT) {
        typedef std::vector< std::pair<unsigned int,float> > VertexWeightTable;

        // build a per-vertex weight list if necessary
        VertexWeightTable* avPerVertexWeights = ComputeVertexBoneWeightTable(pMesh);

        // we need to split this mesh into sub meshes
        // determine the estimated size of a submesh
        // (this could be too large. Max waste is a single digit percentage)
        const unsigned int iSubMeshes = (pMesh->mNumVertices / SplitLargeMeshesProcess_Vertex::LIMIT) + 1;

        // create a std::vector<unsigned int> to indicate which vertices
        // have already been copied
        std::vector<unsigned int> avWasCopied;
        avWasCopied.resize(pMesh->mNumVertices,0xFFFFFFFF);

        // try to find a good estimate for the number of output faces
        // per mesh. Add 12.5% as buffer
        unsigned int iEstimatedSize = pMesh->mNumFaces / iSubMeshes;
        iEstimatedSize += iEstimatedSize >> 3;

        // now generate all submeshes
        unsigned int iBase( 0 );
        while (true) {
            const unsigned int iOutVertexNum = SplitLargeMeshesProcess_Vertex::LIMIT;
            aiMesh* pcMesh          = new aiMesh;
            pcMesh->mNumVertices    = 0;
            pcMesh->mMaterialIndex  = pMesh->mMaterialIndex;

            // the name carries the adjacency information between the meshes
            pcMesh->mName = pMesh->mName;

            typedef std::vector<aiVertexWeight> BoneWeightList;
            if (pMesh->HasBones()) {
                pcMesh->mBones = new aiBone*[pMesh->mNumBones];
                ::memset(pcMesh->mBones,0,sizeof(void*)*pMesh->mNumBones);
            }

            // clear the temporary helper array
            if (iBase) {
                // we can't use memset here we unsigned int needn' be 32 bits
                for (auto &elem : avWasCopied) {
                    elem = 0xffffffff;
                }
            }

            // output vectors
            std::vector<aiFace> vFaces;

            // reserve enough storage for most cases
            if (pMesh->HasPositions()) {
                pcMesh->mVertices = new aiVector3D[iOutVertexNum];
            }
            if (pMesh->HasNormals()) {
                pcMesh->mNormals = new aiVector3D[iOutVertexNum];
            }
            if (pMesh->HasTangentsAndBitangents()) {
                pcMesh->mTangents = new aiVector3D[iOutVertexNum];
                pcMesh->mBitangents = new aiVector3D[iOutVertexNum];
            }
            for (unsigned int c = 0; pMesh->HasVertexColors(c);++c) {
                pcMesh->mColors[c] = new aiColor4D[iOutVertexNum];
            }
            for (unsigned int c = 0; pMesh->HasTextureCoords(c);++c) {
                pcMesh->mNumUVComponents[c] = pMesh->mNumUVComponents[c];
                pcMesh->mTextureCoords[c] = new aiVector3D[iOutVertexNum];
            }
            vFaces.reserve(iEstimatedSize);

            // (we will also need to copy the array of indices)
            while (iBase < pMesh->mNumFaces) {
                // allocate a new array
                const unsigned int iNumIndices = pMesh->mFaces[iBase].mNumIndices;

                // doesn't catch degenerates but is quite fast
                unsigned int iNeed = 0;
                for (unsigned int v = 0; v < iNumIndices;++v) {
                    unsigned int iIndex = pMesh->mFaces[iBase].mIndices[v];

                    // check whether we do already have this vertex
                    if (0xFFFFFFFF == avWasCopied[iIndex]) {
                        iNeed++;
                    }
                }
                if (pcMesh->mNumVertices + iNeed > iOutVertexNum) {
                    // don't use this face
                    break;
                }

                vFaces.push_back(aiFace());
                aiFace& rFace = vFaces.back();

                // setup face type and number of indices
                rFace.mNumIndices = iNumIndices;
                rFace.mIndices = new unsigned int[iNumIndices];

                // need to update the output primitive types
                switch (rFace.mNumIndices) {
                case 1:
                    pcMesh->mPrimitiveTypes |= aiPrimitiveType_POINT;
                    break;
                case 2:
                    pcMesh->mPrimitiveTypes |= aiPrimitiveType_LINE;
                    break;
                case 3:
                    pcMesh->mPrimitiveTypes |= aiPrimitiveType_TRIANGLE;
                    break;
                default:
                    pcMesh->mPrimitiveTypes |= aiPrimitiveType_POLYGON;
                }

                // and copy the contents of the old array, offset by current base
                for (unsigned int v = 0; v < iNumIndices;++v) {
                    unsigned int iIndex = pMesh->mFaces[iBase].mIndices[v];

                    // check whether we do already have this vertex
                    if (0xFFFFFFFF != avWasCopied[iIndex]) {
                        rFace.mIndices[v] = avWasCopied[iIndex];
                        continue;
                    }

                    // copy positions
                    pcMesh->mVertices[pcMesh->mNumVertices] = (pMesh->mVertices[iIndex]);

                    // copy normals
                    if (pMesh->HasNormals()) {
                        pcMesh->mNormals[pcMesh->mNumVertices] = (pMesh->mNormals[iIndex]);
                    }

                    // copy tangents/bitangents
                    if (pMesh->HasTangentsAndBitangents()) {
                        pcMesh->mTangents[pcMesh->mNumVertices] = (pMesh->mTangents[iIndex]);
                        pcMesh->mBitangents[pcMesh->mNumVertices] = (pMesh->mBitangents[iIndex]);
                    }

                    // texture coordinates
                    for (unsigned int c = 0;  c < AI_MAX_NUMBER_OF_TEXTURECOORDS;++c) {
                        if (pMesh->HasTextureCoords( c)) {
                            pcMesh->mTextureCoords[c][pcMesh->mNumVertices] = pMesh->mTextureCoords[c][iIndex];
                        }
                    }
                    // vertex colors
                    for (unsigned int c = 0;  c < AI_MAX_NUMBER_OF_COLOR_SETS;++c) {
                        if (pMesh->HasVertexColors( c)) {
                            pcMesh->mColors[c][pcMesh->mNumVertices] = pMesh->mColors[c][iIndex];
                        }
                    }
                    // check whether we have bone weights assigned to this vertex
                    rFace.mIndices[v] = pcMesh->mNumVertices;
                    if (avPerVertexWeights) {
                        VertexWeightTable& table = avPerVertexWeights[ pcMesh->mNumVertices ];
                        if( !table.empty() ) {
                            for (VertexWeightTable::const_iterator iter =  table.begin();
                                    iter != table.end();++iter) {
                                // allocate the bone weight array if necessary
                                BoneWeightList* pcWeightList = (BoneWeightList*)pcMesh->mBones[(*iter).first];
                                if (nullptr == pcWeightList) {
                                    pcMesh->mBones[(*iter).first] = (aiBone*)(pcWeightList = new BoneWeightList());
                                }
                                pcWeightList->push_back(aiVertexWeight(pcMesh->mNumVertices,(*iter).second));
                            }
                        }
                    }

                    avWasCopied[iIndex] = pcMesh->mNumVertices;
                    pcMesh->mNumVertices++;
                }
                ++iBase;
                if(pcMesh->mNumVertices == iOutVertexNum) {
                    // break here. The face is only added if it was complete
                    break;
                }
            }

            // check which bones we'll need to create for this submesh
            if (pMesh->HasBones()) {
                aiBone** ppCurrent = pcMesh->mBones;
                for (unsigned int k = 0; k < pMesh->mNumBones;++k) {
                    // check whether the bone is existing
                    BoneWeightList* pcWeightList;
                    if ((pcWeightList = (BoneWeightList*)pcMesh->mBones[k])) {
                        aiBone* pcOldBone = pMesh->mBones[k];
                        aiBone* pcOut( nullptr );
                        *ppCurrent++ = pcOut = new aiBone();
                        pcOut->mName = aiString(pcOldBone->mName);
                        pcOut->mOffsetMatrix = pcOldBone->mOffsetMatrix;
                        pcOut->mNumWeights = (unsigned int)pcWeightList->size();
                        pcOut->mWeights = new aiVertexWeight[pcOut->mNumWeights];

                        // copy the vertex weights
                        ::memcpy(pcOut->mWeights,&pcWeightList->operator[](0),
                            pcOut->mNumWeights * sizeof(aiVertexWeight));

                        // delete the temporary bone weight list
                        delete pcWeightList;
                        pcMesh->mNumBones++;
                    }
                }
            }

            // copy the face list to the mesh
            pcMesh->mFaces = new aiFace[vFaces.size()];
            pcMesh->mNumFaces = (unsigned int)vFaces.size();

            for (unsigned int p = 0; p < pcMesh->mNumFaces;++p) {
                pcMesh->mFaces[p] = vFaces[p];
            }

            // add the newly created mesh to the list
            avList.push_back(std::pair<aiMesh*, unsigned int>(pcMesh,a));

            if (iBase == pMesh->mNumFaces) {
                // have all faces ... finish the outer loop, too
                break;
            }
        }

        // delete the per-vertex weight list again
        delete[] avPerVertexWeights;

        // now delete the old mesh data
        delete pMesh;
        return;
    }
    avList.push_back(std::pair<aiMesh*, unsigned int>(pMesh,a));
}
