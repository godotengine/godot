/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2012, assimp team

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

/** @file Generation of normal vectors basing on smoothing groups */

#pragma once
#ifndef AI_SMOOTHINGGROUPS_INL_INCLUDED
#define AI_SMOOTHINGGROUPS_INL_INCLUDED

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/SGSpatialSort.h>

#include <algorithm>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
template <class T>
void ComputeNormalsWithSmoothingsGroups(MeshWithSmoothingGroups<T>& sMesh)
{
    // First generate face normals
    sMesh.mNormals.resize(sMesh.mPositions.size(),aiVector3D());
    for( unsigned int a = 0; a < sMesh.mFaces.size(); a++)
    {
        T& face = sMesh.mFaces[a];

        aiVector3D* pV1 = &sMesh.mPositions[face.mIndices[0]];
        aiVector3D* pV2 = &sMesh.mPositions[face.mIndices[1]];
        aiVector3D* pV3 = &sMesh.mPositions[face.mIndices[2]];

        aiVector3D pDelta1 = *pV2 - *pV1;
        aiVector3D pDelta2 = *pV3 - *pV1;
        aiVector3D vNor = pDelta1 ^ pDelta2;

        for (unsigned int c = 0; c < 3;++c)
            sMesh.mNormals[face.mIndices[c]] = vNor;
    }

    // calculate the position bounds so we have a reliable epsilon to check position differences against 
    aiVector3D minVec( 1e10f, 1e10f, 1e10f), maxVec( -1e10f, -1e10f, -1e10f);
    for( unsigned int a = 0; a < sMesh.mPositions.size(); a++)
    {
        minVec.x = std::min( minVec.x, sMesh.mPositions[a].x);
        minVec.y = std::min( minVec.y, sMesh.mPositions[a].y);
        minVec.z = std::min( minVec.z, sMesh.mPositions[a].z);
        maxVec.x = std::max( maxVec.x, sMesh.mPositions[a].x);
        maxVec.y = std::max( maxVec.y, sMesh.mPositions[a].y);
        maxVec.z = std::max( maxVec.z, sMesh.mPositions[a].z);
    }
    const float posEpsilon = (maxVec - minVec).Length() * 1e-5f;
    std::vector<aiVector3D> avNormals;
    avNormals.resize(sMesh.mNormals.size());
    
    // now generate the spatial sort tree
    SGSpatialSort sSort;
    for( typename std::vector<T>::iterator i =  sMesh.mFaces.begin();
        i != sMesh.mFaces.end();++i)
    {
        for (unsigned int c = 0; c < 3;++c)
            sSort.Add(sMesh.mPositions[(*i).mIndices[c]],(*i).mIndices[c],(*i).iSmoothGroup);
    }
    sSort.Prepare();

    std::vector<bool> vertexDone(sMesh.mPositions.size(),false);
    for( typename std::vector<T>::iterator i =  sMesh.mFaces.begin();
        i != sMesh.mFaces.end();++i)
    {
        std::vector<unsigned int> poResult;
        for (unsigned int c = 0; c < 3;++c)
        {
            unsigned int idx = (*i).mIndices[c];
            if (vertexDone[idx])continue;

            sSort.FindPositions(sMesh.mPositions[idx],(*i).iSmoothGroup,
                posEpsilon,poResult);

            aiVector3D vNormals;
            for (std::vector<unsigned int>::const_iterator
                a =  poResult.begin();
                a != poResult.end();++a)
            {
                vNormals += sMesh.mNormals[(*a)];
            }
            vNormals.NormalizeSafe();

            // write back into all affected normals
            for (std::vector<unsigned int>::const_iterator
                a =  poResult.begin();
                a != poResult.end();++a)
            {
                idx = *a;
                avNormals [idx] = vNormals;
                vertexDone[idx] = true;
            }
        }
    }
    sMesh.mNormals = avNormals;
}

#endif // !! AI_SMOOTHINGGROUPS_INL_INCLUDED
