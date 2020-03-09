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

/** Implementation of the LimitBoneWeightsProcess post processing step */


#include "LimitBoneWeightsProcess.h"
#include <assimp/StringUtils.h>
#include <assimp/postprocess.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/scene.h>
#include <stdio.h>

using namespace Assimp;


// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
LimitBoneWeightsProcess::LimitBoneWeightsProcess()
{
    mMaxWeights = AI_LMW_MAX_WEIGHTS;
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
LimitBoneWeightsProcess::~LimitBoneWeightsProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool LimitBoneWeightsProcess::IsActive( unsigned int pFlags) const
{
    return (pFlags & aiProcess_LimitBoneWeights) != 0;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void LimitBoneWeightsProcess::Execute( aiScene* pScene) {
    ASSIMP_LOG_DEBUG("LimitBoneWeightsProcess begin");
    for (unsigned int a = 0; a < pScene->mNumMeshes; ++a ) {
        ProcessMesh(pScene->mMeshes[a]);
    }

    ASSIMP_LOG_DEBUG("LimitBoneWeightsProcess end");
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void LimitBoneWeightsProcess::SetupProperties(const Importer* pImp)
{
    // get the current value of the property
    this->mMaxWeights = pImp->GetPropertyInteger(AI_CONFIG_PP_LBW_MAX_WEIGHTS,AI_LMW_MAX_WEIGHTS);
}

// ------------------------------------------------------------------------------------------------
// Unites identical vertices in the given mesh
void LimitBoneWeightsProcess::ProcessMesh( aiMesh* pMesh)
{
    if( !pMesh->HasBones())
        return;

    // collect all bone weights per vertex
    typedef std::vector< std::vector< Weight > > WeightsPerVertex;
    WeightsPerVertex vertexWeights( pMesh->mNumVertices);

    // collect all weights per vertex
    for( unsigned int a = 0; a < pMesh->mNumBones; a++)
    {
        const aiBone* bone = pMesh->mBones[a];
        for( unsigned int b = 0; b < bone->mNumWeights; b++)
        {
            const aiVertexWeight& w = bone->mWeights[b];
            vertexWeights[w.mVertexId].push_back( Weight( a, w.mWeight));
        }
    }

    unsigned int removed = 0, old_bones = pMesh->mNumBones;

    // now cut the weight count if it exceeds the maximum
    bool bChanged = false;
    for( WeightsPerVertex::iterator vit = vertexWeights.begin(); vit != vertexWeights.end(); ++vit)
    {
        if( vit->size() <= mMaxWeights)
            continue;

        bChanged = true;

        // more than the defined maximum -> first sort by weight in descending order. That's
        // why we defined the < operator in such a weird way.
        std::sort( vit->begin(), vit->end());

        // now kill everything beyond the maximum count
        unsigned int m = static_cast<unsigned int>(vit->size());
        vit->erase( vit->begin() + mMaxWeights, vit->end());
        removed += static_cast<unsigned int>(m-vit->size());

        // and renormalize the weights
        float sum = 0.0f;
        for( std::vector<Weight>::const_iterator it = vit->begin(); it != vit->end(); ++it ) {
            sum += it->mWeight;
        }
        if( 0.0f != sum ) {
            const float invSum = 1.0f / sum;
            for( std::vector<Weight>::iterator it = vit->begin(); it != vit->end(); ++it ) {
                it->mWeight *= invSum;
            }
        }
    }

    if (bChanged)   {
        // rebuild the vertex weight array for all bones
        typedef std::vector< std::vector< aiVertexWeight > > WeightsPerBone;
        WeightsPerBone boneWeights( pMesh->mNumBones);
        for( unsigned int a = 0; a < vertexWeights.size(); a++)
        {
            const std::vector<Weight>& vw = vertexWeights[a];
            for( std::vector<Weight>::const_iterator it = vw.begin(); it != vw.end(); ++it)
                boneWeights[it->mBone].push_back( aiVertexWeight( a, it->mWeight));
        }

        // and finally copy the vertex weight list over to the mesh's bones
        std::vector<bool> abNoNeed(pMesh->mNumBones,false);
        bChanged = false;

        for( unsigned int a = 0; a < pMesh->mNumBones; a++)
        {
            const std::vector<aiVertexWeight>& bw = boneWeights[a];
            aiBone* bone = pMesh->mBones[a];

            if ( bw.empty() )
            {
                abNoNeed[a] = bChanged = true;
                continue;
            }

            // copy the weight list. should always be less weights than before, so we don't need a new allocation
            ai_assert( bw.size() <= bone->mNumWeights);
            bone->mNumWeights = static_cast<unsigned int>( bw.size() );
            ::memcpy( bone->mWeights, &bw[0], bw.size() * sizeof( aiVertexWeight));
        }

        if (bChanged)   {
            // the number of new bones is smaller than before, so we can reuse the old array
            aiBone** ppcCur = pMesh->mBones;aiBone** ppcSrc = ppcCur;

            for (std::vector<bool>::const_iterator iter  = abNoNeed.begin();iter != abNoNeed.end()  ;++iter)    {
                if (*iter)  {
                    delete *ppcSrc;
                    --pMesh->mNumBones;
                }
                else *ppcCur++ = *ppcSrc;
                ++ppcSrc;
            }
        }

        if (!DefaultLogger::isNullLogger()) {
            ASSIMP_LOG_INFO_F("Removed ", removed, " weights. Input bones: ", old_bones, ". Output bones: ", pMesh->mNumBones );
        }
    }
}
