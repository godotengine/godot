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

/** @file Implementation of the post processing step to invert
 * all normals in meshes with infacing normals.
 */

// internal headers
#include "FixNormalsStep.h"
#include <assimp/StringUtils.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <stdio.h>


using namespace Assimp;


// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
FixInfacingNormalsProcess::FixInfacingNormalsProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
FixInfacingNormalsProcess::~FixInfacingNormalsProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool FixInfacingNormalsProcess::IsActive( unsigned int pFlags) const
{
    return (pFlags & aiProcess_FixInfacingNormals) != 0;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void FixInfacingNormalsProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("FixInfacingNormalsProcess begin");

    bool bHas( false );
    for (unsigned int a = 0; a < pScene->mNumMeshes; ++a) {
        if (ProcessMesh(pScene->mMeshes[a], a)) {
            bHas = true;
        }
    }

    if (bHas) {
        ASSIMP_LOG_DEBUG("FixInfacingNormalsProcess finished. Found issues.");
    } else {
        ASSIMP_LOG_DEBUG("FixInfacingNormalsProcess finished. No changes to the scene.");
    }
}

// ------------------------------------------------------------------------------------------------
// Apply the step to the mesh
bool FixInfacingNormalsProcess::ProcessMesh( aiMesh* pcMesh, unsigned int index)
{
    ai_assert(nullptr != pcMesh);

    // Nothing to do if there are no model normals
    if (!pcMesh->HasNormals()) {
        return false;
    }

    // Compute the bounding box of both the model vertices + normals and
    // the unmodified model vertices. Then check whether the first BB
    // is smaller than the second. In this case we can assume that the
    // normals need to be flipped, although there are a few special cases ..
    // convex, concave, planar models ...

    aiVector3D vMin0 (1e10f,1e10f,1e10f);
    aiVector3D vMin1 (1e10f,1e10f,1e10f);
    aiVector3D vMax0 (-1e10f,-1e10f,-1e10f);
    aiVector3D vMax1 (-1e10f,-1e10f,-1e10f);

    for (unsigned int i = 0; i < pcMesh->mNumVertices;++i)
    {
        vMin1.x = std::min(vMin1.x,pcMesh->mVertices[i].x);
        vMin1.y = std::min(vMin1.y,pcMesh->mVertices[i].y);
        vMin1.z = std::min(vMin1.z,pcMesh->mVertices[i].z);

        vMax1.x = std::max(vMax1.x,pcMesh->mVertices[i].x);
        vMax1.y = std::max(vMax1.y,pcMesh->mVertices[i].y);
        vMax1.z = std::max(vMax1.z,pcMesh->mVertices[i].z);

        const aiVector3D vWithNormal = pcMesh->mVertices[i] + pcMesh->mNormals[i];

        vMin0.x = std::min(vMin0.x,vWithNormal.x);
        vMin0.y = std::min(vMin0.y,vWithNormal.y);
        vMin0.z = std::min(vMin0.z,vWithNormal.z);

        vMax0.x = std::max(vMax0.x,vWithNormal.x);
        vMax0.y = std::max(vMax0.y,vWithNormal.y);
        vMax0.z = std::max(vMax0.z,vWithNormal.z);
    }

    const float fDelta0_x = (vMax0.x - vMin0.x);
    const float fDelta0_y = (vMax0.y - vMin0.y);
    const float fDelta0_z = (vMax0.z - vMin0.z);

    const float fDelta1_x = (vMax1.x - vMin1.x);
    const float fDelta1_y = (vMax1.y - vMin1.y);
    const float fDelta1_z = (vMax1.z - vMin1.z);

    // Check whether the boxes are overlapping
    if ((fDelta0_x > 0.0f) != (fDelta1_x > 0.0f))return false;
    if ((fDelta0_y > 0.0f) != (fDelta1_y > 0.0f))return false;
    if ((fDelta0_z > 0.0f) != (fDelta1_z > 0.0f))return false;

    // Check whether this is a planar surface
    const float fDelta1_yz = fDelta1_y * fDelta1_z;

    if (fDelta1_x < 0.05f * std::sqrt( fDelta1_yz ))return false;
    if (fDelta1_y < 0.05f * std::sqrt( fDelta1_z * fDelta1_x ))return false;
    if (fDelta1_z < 0.05f * std::sqrt( fDelta1_y * fDelta1_x ))return false;

    // now compare the volumes of the bounding boxes
    if (std::fabs(fDelta0_x * fDelta0_y * fDelta0_z) < std::fabs(fDelta1_x * fDelta1_yz)) {
        if (!DefaultLogger::isNullLogger()) {
            ASSIMP_LOG_INFO_F("Mesh ", index, ": Normals are facing inwards (or the mesh is planar)", index);
        }

        // Invert normals
        for (unsigned int i = 0; i < pcMesh->mNumVertices;++i)
            pcMesh->mNormals[i] *= -1.0f;

        // ... and flip faces
        for (unsigned int i = 0; i < pcMesh->mNumFaces;++i)
        {
            aiFace& face = pcMesh->mFaces[i];
            for( unsigned int b = 0; b < face.mNumIndices / 2; b++)
                std::swap( face.mIndices[b], face.mIndices[ face.mNumIndices - 1 - b]);
        }
        return true;
    }
    return false;
}
