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

/// @file DeboneProcess.cpp
/** Implementation of the DeboneProcess post processing step */



// internal headers of the post-processing framework
#include "ProcessHelper.h"
#include "DeboneProcess.h"
#include <stdio.h>


using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
DeboneProcess::DeboneProcess()
{
    mNumBones = 0;
    mNumBonesCanDoWithout = 0;

    mThreshold = AI_DEBONE_THRESHOLD;
    mAllOrNone = false;
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
DeboneProcess::~DeboneProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool DeboneProcess::IsActive( unsigned int pFlags) const
{
    return (pFlags & aiProcess_Debone) != 0;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void DeboneProcess::SetupProperties(const Importer* pImp)
{
    // get the current value of the property
    mAllOrNone = pImp->GetPropertyInteger(AI_CONFIG_PP_DB_ALL_OR_NONE,0)?true:false;
    mThreshold = pImp->GetPropertyFloat(AI_CONFIG_PP_DB_THRESHOLD,AI_DEBONE_THRESHOLD);
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void DeboneProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("DeboneProcess begin");

    if(!pScene->mNumMeshes) {
        return;
    }

    std::vector<bool> splitList(pScene->mNumMeshes);
    for( unsigned int a = 0; a < pScene->mNumMeshes; a++) {
        splitList[a] = ConsiderMesh( pScene->mMeshes[a] );
    }

    int numSplits = 0;

    if(!!mNumBonesCanDoWithout && (!mAllOrNone||mNumBonesCanDoWithout==mNumBones))  {
        for(unsigned int a = 0; a < pScene->mNumMeshes; a++)    {
            if(splitList[a]) {
                numSplits++;
            }
        }
    }

    if(numSplits)   {
        // we need to do something. Let's go.
        //mSubMeshIndices.clear();                  // really needed?
        mSubMeshIndices.resize(pScene->mNumMeshes); // because we're doing it here anyway

        // build a new array of meshes for the scene
        std::vector<aiMesh*> meshes;

        for(unsigned int a=0;a<pScene->mNumMeshes;a++)
        {
            aiMesh* srcMesh = pScene->mMeshes[a];

            std::vector<std::pair<aiMesh*,const aiBone*> > newMeshes;

            if(splitList[a]) {
                SplitMesh(srcMesh,newMeshes);
            }

            // mesh was split
            if(!newMeshes.empty())  {
                unsigned int out = 0, in = srcMesh->mNumBones;

                // store new meshes and indices of the new meshes
                for(unsigned int b=0;b<newMeshes.size();b++)    {
                    const aiString *find = newMeshes[b].second?&newMeshes[b].second->mName:0;

                    aiNode *theNode = find?pScene->mRootNode->FindNode(*find):0;
                    std::pair<unsigned int,aiNode*> push_pair(static_cast<unsigned int>(meshes.size()),theNode);

                    mSubMeshIndices[a].push_back(push_pair);
                    meshes.push_back(newMeshes[b].first);

                    out+=newMeshes[b].first->mNumBones;
                }

                if(!DefaultLogger::isNullLogger()) {
                    ASSIMP_LOG_INFO_F("Removed %u bones. Input bones:", in - out, ". Output bones: ", out);
                }

                // and destroy the source mesh. It should be completely contained inside the new submeshes
                delete srcMesh;
            }
            else    {
                // Mesh is kept unchanged - store it's new place in the mesh array
                mSubMeshIndices[a].push_back(std::pair<unsigned int,aiNode*>(static_cast<unsigned int>(meshes.size()),(aiNode*)0));
                meshes.push_back(srcMesh);
            }
        }

        // rebuild the scene's mesh array
        pScene->mNumMeshes = static_cast<unsigned int>(meshes.size());
        delete [] pScene->mMeshes;
        pScene->mMeshes = new aiMesh*[pScene->mNumMeshes];
        std::copy( meshes.begin(), meshes.end(), pScene->mMeshes);

        // recurse through all nodes and translate the node's mesh indices to fit the new mesh array
        UpdateNode( pScene->mRootNode);
    }

    ASSIMP_LOG_DEBUG("DeboneProcess end");
}

// ------------------------------------------------------------------------------------------------
// Counts bones total/removable in a given mesh.
bool DeboneProcess::ConsiderMesh(const aiMesh* pMesh)
{
    if(!pMesh->HasBones()) {
        return false;
    }

    bool split = false;

    //interstitial faces not permitted
    bool isInterstitialRequired = false;

    std::vector<bool> isBoneNecessary(pMesh->mNumBones,false);
    std::vector<unsigned int> vertexBones(pMesh->mNumVertices,UINT_MAX);

    const unsigned int cUnowned = UINT_MAX;
    const unsigned int cCoowned = UINT_MAX-1;

    for(unsigned int i=0;i<pMesh->mNumBones;i++)    {
        for(unsigned int j=0;j<pMesh->mBones[i]->mNumWeights;j++)   {
            float w = pMesh->mBones[i]->mWeights[j].mWeight;

            if(w==0.0f) {
                continue;
            }

            unsigned int vid = pMesh->mBones[i]->mWeights[j].mVertexId;
            if(w>=mThreshold)   {

                if(vertexBones[vid]!=cUnowned)  {
                    if(vertexBones[vid]==i) //double entry
                    {
                        ASSIMP_LOG_WARN("Encountered double entry in bone weights");
                    }
                    else //TODO: track attraction in order to break tie
                    {
                        vertexBones[vid] = cCoowned;
                    }
                }
                else vertexBones[vid] = i;
            }

            if(!isBoneNecessary[i]) {
                isBoneNecessary[i] = w<mThreshold;
            }
        }

        if(!isBoneNecessary[i])  {
            isInterstitialRequired = true;
        }
    }

    if(isInterstitialRequired) {
        for(unsigned int i=0;i<pMesh->mNumFaces;i++) {
            unsigned int v = vertexBones[pMesh->mFaces[i].mIndices[0]];

            for(unsigned int j=1;j<pMesh->mFaces[i].mNumIndices;j++) {
                unsigned int w = vertexBones[pMesh->mFaces[i].mIndices[j]];

                if(v!=w)    {
                    if(v<pMesh->mNumBones) isBoneNecessary[v] = true;
                    if(w<pMesh->mNumBones) isBoneNecessary[w] = true;
                }
            }
        }
    }

    for(unsigned int i=0;i<pMesh->mNumBones;i++)    {
        if(!isBoneNecessary[i]) {
            mNumBonesCanDoWithout++;
            split = true;
        }

        mNumBones++;
    }
    return split;
}

// ------------------------------------------------------------------------------------------------
// Splits the given mesh by bone count.
void DeboneProcess::SplitMesh( const aiMesh* pMesh, std::vector< std::pair< aiMesh*,const aiBone* > >& poNewMeshes) const
{
    // same deal here as ConsiderMesh basically

    std::vector<bool> isBoneNecessary(pMesh->mNumBones,false);
    std::vector<unsigned int> vertexBones(pMesh->mNumVertices,UINT_MAX);

    const unsigned int cUnowned = UINT_MAX;
    const unsigned int cCoowned = UINT_MAX-1;

    for(unsigned int i=0;i<pMesh->mNumBones;i++)    {
        for(unsigned int j=0;j<pMesh->mBones[i]->mNumWeights;j++)   {
            float w = pMesh->mBones[i]->mWeights[j].mWeight;

            if(w==0.0f) {
                continue;
            }

            unsigned int vid = pMesh->mBones[i]->mWeights[j].mVertexId;

            if(w>=mThreshold) {
                if(vertexBones[vid]!=cUnowned)  {
                    if(vertexBones[vid]==i) //double entry
                    {
                        ASSIMP_LOG_WARN("Encountered double entry in bone weights");
                    }
                    else //TODO: track attraction in order to break tie
                    {
                        vertexBones[vid] = cCoowned;
                    }
                }
                else vertexBones[vid] = i;
            }

            if(!isBoneNecessary[i]) {
                isBoneNecessary[i] = w<mThreshold;
            }
        }
    }

    unsigned int nFacesUnowned = 0;

    std::vector<unsigned int> faceBones(pMesh->mNumFaces,UINT_MAX);
    std::vector<unsigned int> facesPerBone(pMesh->mNumBones,0);

    for(unsigned int i=0;i<pMesh->mNumFaces;i++) {
        unsigned int nInterstitial = 1;

        unsigned int v = vertexBones[pMesh->mFaces[i].mIndices[0]];

        for(unsigned int j=1;j<pMesh->mFaces[i].mNumIndices;j++) {
            unsigned int w = vertexBones[pMesh->mFaces[i].mIndices[j]];

            if(v!=w)    {
                if(v<pMesh->mNumBones) isBoneNecessary[v] = true;
                if(w<pMesh->mNumBones) isBoneNecessary[w] = true;
            }
            else nInterstitial++;
        }

        if(v<pMesh->mNumBones &&nInterstitial==pMesh->mFaces[i].mNumIndices)    {
            faceBones[i] = v; //primitive belongs to bone #v
            facesPerBone[v]++;
        }
        else nFacesUnowned++;
    }

    // invalidate any "cojoined" faces
    for(unsigned int i=0;i<pMesh->mNumFaces;i++) {
        if(faceBones[i]<pMesh->mNumBones&&isBoneNecessary[faceBones[i]])
        {
            ai_assert(facesPerBone[faceBones[i]]>0);
            facesPerBone[faceBones[i]]--;

            nFacesUnowned++;
            faceBones[i] = cUnowned;
        }
    }

    if(nFacesUnowned) {
        std::vector<unsigned int> subFaces;

        for(unsigned int i=0;i<pMesh->mNumFaces;i++)    {
            if(faceBones[i]==cUnowned) {
                subFaces.push_back(i);
            }
        }

        aiMesh *baseMesh = MakeSubmesh(pMesh,subFaces,0);
        std::pair<aiMesh*,const aiBone*> push_pair(baseMesh,(const aiBone*)0);

        poNewMeshes.push_back(push_pair);
    }

    for(unsigned int i=0;i<pMesh->mNumBones;i++) {

        if(!isBoneNecessary[i]&&facesPerBone[i]>0)  {
            std::vector<unsigned int> subFaces;

            for(unsigned int j=0;j<pMesh->mNumFaces;j++)    {
                if(faceBones[j]==i) {
                    subFaces.push_back(j);
                }
            }

            unsigned int f = AI_SUBMESH_FLAGS_SANS_BONES;
            aiMesh *subMesh =MakeSubmesh(pMesh,subFaces,f);

            //Lifted from PretransformVertices.cpp
            ApplyTransform(subMesh,pMesh->mBones[i]->mOffsetMatrix);
            std::pair<aiMesh*,const aiBone*> push_pair(subMesh,pMesh->mBones[i]);

            poNewMeshes.push_back(push_pair);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Recursively updates the node's mesh list to account for the changed mesh list
void DeboneProcess::UpdateNode(aiNode* pNode) const
{
    // rebuild the node's mesh index list

    std::vector<unsigned int> newMeshList;

    // this will require two passes

    unsigned int m = static_cast<unsigned int>(pNode->mNumMeshes), n = static_cast<unsigned int>(mSubMeshIndices.size());

    // first pass, look for meshes which have not moved

    for(unsigned int a=0;a<m;a++)   {

        unsigned int srcIndex = pNode->mMeshes[a];
        const std::vector< std::pair< unsigned int,aiNode* > > &subMeshes = mSubMeshIndices[srcIndex];
        unsigned int nSubmeshes = static_cast<unsigned int>(subMeshes.size());

        for(unsigned int b=0;b<nSubmeshes;b++) {
            if(!subMeshes[b].second) {
                newMeshList.push_back(subMeshes[b].first);
            }
        }
    }

    // second pass, collect deboned meshes

    for(unsigned int a=0;a<n;a++)
    {
        const std::vector< std::pair< unsigned int,aiNode* > > &subMeshes = mSubMeshIndices[a];
        unsigned int nSubmeshes = static_cast<unsigned int>(subMeshes.size());

        for(unsigned int b=0;b<nSubmeshes;b++) {
            if(subMeshes[b].second == pNode)    {
                newMeshList.push_back(subMeshes[b].first);
            }
        }
    }

    if( pNode->mNumMeshes > 0 ) {
        delete [] pNode->mMeshes; pNode->mMeshes = NULL;
    }

    pNode->mNumMeshes = static_cast<unsigned int>(newMeshList.size());

    if(pNode->mNumMeshes)   {
        pNode->mMeshes = new unsigned int[pNode->mNumMeshes];
        std::copy( newMeshList.begin(), newMeshList.end(), pNode->mMeshes);
    }

    // do that also recursively for all children
    for( unsigned int a = 0; a < pNode->mNumChildren; ++a ) {
        UpdateNode( pNode->mChildren[a]);
    }
}

// ------------------------------------------------------------------------------------------------
// Apply the node transformation to a mesh
void DeboneProcess::ApplyTransform(aiMesh* mesh, const aiMatrix4x4& mat)const
{
    // Check whether we need to transform the coordinates at all
    if (!mat.IsIdentity()) {

        if (mesh->HasPositions()) {
            for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                mesh->mVertices[i] = mat * mesh->mVertices[i];
            }
        }
        if (mesh->HasNormals() || mesh->HasTangentsAndBitangents()) {
            aiMatrix4x4 mWorldIT = mat;
            mWorldIT.Inverse().Transpose();

            // TODO: implement Inverse() for aiMatrix3x3
            aiMatrix3x3 m = aiMatrix3x3(mWorldIT);

            if (mesh->HasNormals()) {
                for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                    mesh->mNormals[i] = (m * mesh->mNormals[i]).Normalize();
                }
            }
            if (mesh->HasTangentsAndBitangents()) {
                for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                    mesh->mTangents[i]   = (m * mesh->mTangents[i]).Normalize();
                    mesh->mBitangents[i] = (m * mesh->mBitangents[i]).Normalize();
                }
            }
        }
    }
}
