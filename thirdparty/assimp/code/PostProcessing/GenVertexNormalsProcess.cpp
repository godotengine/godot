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



// internal headers
#include "GenVertexNormalsProcess.h"
#include "ProcessHelper.h"
#include <assimp/Exceptional.h>
#include <assimp/qnan.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
GenVertexNormalsProcess::GenVertexNormalsProcess()
: configMaxAngle( AI_DEG_TO_RAD( 175.f ) ) {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
GenVertexNormalsProcess::~GenVertexNormalsProcess() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool GenVertexNormalsProcess::IsActive( unsigned int pFlags) const
{
    force_ = (pFlags & aiProcess_ForceGenNormals) != 0;
    return (pFlags & aiProcess_GenSmoothNormals) != 0;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void GenVertexNormalsProcess::SetupProperties(const Importer* pImp)
{
    // Get the current value of the AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE property
    configMaxAngle = pImp->GetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE,(ai_real)175.0);
    configMaxAngle = AI_DEG_TO_RAD(std::max(std::min(configMaxAngle,(ai_real)175.0),(ai_real)0.0));
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void GenVertexNormalsProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("GenVertexNormalsProcess begin");

    if (pScene->mFlags & AI_SCENE_FLAGS_NON_VERBOSE_FORMAT) {
        throw DeadlyImportError("Post-processing order mismatch: expecting pseudo-indexed (\"verbose\") vertices here");
    }

    bool bHas = false;
    for( unsigned int a = 0; a < pScene->mNumMeshes; ++a) {
        if(GenMeshVertexNormals( pScene->mMeshes[a],a))
            bHas = true;
    }

    if (bHas)   {
        ASSIMP_LOG_INFO("GenVertexNormalsProcess finished. "
            "Vertex normals have been calculated");
    } else {
        ASSIMP_LOG_DEBUG("GenVertexNormalsProcess finished. "
            "Normals are already there");
    }
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
bool GenVertexNormalsProcess::GenMeshVertexNormals (aiMesh* pMesh, unsigned int meshIndex)
{
    if (NULL != pMesh->mNormals) {
        if (force_) delete[] pMesh->mNormals;
        else return false;
    }

    // If the mesh consists of lines and/or points but not of
    // triangles or higher-order polygons the normal vectors
    // are undefined.
    if (!(pMesh->mPrimitiveTypes & (aiPrimitiveType_TRIANGLE | aiPrimitiveType_POLYGON)))
    {
        ASSIMP_LOG_INFO("Normal vectors are undefined for line and point meshes");
        return false;
    }

    // Allocate the array to hold the output normals
    const float qnan = std::numeric_limits<ai_real>::quiet_NaN();
    pMesh->mNormals = new aiVector3D[pMesh->mNumVertices];

    // Compute per-face normals but store them per-vertex
    for( unsigned int a = 0; a < pMesh->mNumFaces; a++)
    {
        const aiFace& face = pMesh->mFaces[a];
        if (face.mNumIndices < 3)
        {
            // either a point or a line -> no normal vector
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

    // Set up a SpatialSort to quickly find all vertices close to a given position
    // check whether we can reuse the SpatialSort of a previous step.
    SpatialSort* vertexFinder = NULL;
    SpatialSort  _vertexFinder;
    ai_real posEpsilon = ai_real( 1e-5 );
    if (shared) {
        std::vector<std::pair<SpatialSort,ai_real> >* avf;
        shared->GetProperty(AI_SPP_SPATIAL_SORT,avf);
        if (avf)
        {
            std::pair<SpatialSort,ai_real>& blubb = avf->operator [] (meshIndex);
            vertexFinder = &blubb.first;
            posEpsilon = blubb.second;
        }
    }
    if (!vertexFinder)  {
        _vertexFinder.Fill(pMesh->mVertices, pMesh->mNumVertices, sizeof( aiVector3D));
        vertexFinder = &_vertexFinder;
        posEpsilon = ComputePositionEpsilon(pMesh);
    }
    std::vector<unsigned int> verticesFound;
    aiVector3D* pcNew = new aiVector3D[pMesh->mNumVertices];

    if (configMaxAngle >= AI_DEG_TO_RAD( 175.f ))   {
        // There is no angle limit. Thus all vertices with positions close
        // to each other will receive the same vertex normal. This allows us
        // to optimize the whole algorithm a little bit ...
        std::vector<bool> abHad(pMesh->mNumVertices,false);
        for (unsigned int i = 0; i < pMesh->mNumVertices;++i)   {
            if (abHad[i]) {
                continue;
            }

            // Get all vertices that share this one ...
            vertexFinder->FindPositions( pMesh->mVertices[i], posEpsilon, verticesFound);

            aiVector3D pcNor;
            for (unsigned int a = 0; a < verticesFound.size(); ++a) {
                const aiVector3D& v = pMesh->mNormals[verticesFound[a]];
                if (is_not_qnan(v.x))pcNor += v;
            }
            pcNor.NormalizeSafe();

            // Write the smoothed normal back to all affected normals
            for (unsigned int a = 0; a < verticesFound.size(); ++a)
            {
                unsigned int vidx = verticesFound[a];
                pcNew[vidx] = pcNor;
                abHad[vidx] = true;
            }
        }
    }
    // Slower code path if a smooth angle is set. There are many ways to achieve
    // the effect, this one is the most straightforward one.
    else    {
        const ai_real fLimit = std::cos(configMaxAngle);
        for (unsigned int i = 0; i < pMesh->mNumVertices;++i)   {
            // Get all vertices that share this one ...
            vertexFinder->FindPositions( pMesh->mVertices[i] , posEpsilon, verticesFound);

            aiVector3D vr = pMesh->mNormals[i];

            aiVector3D pcNor;
            for (unsigned int a = 0; a < verticesFound.size(); ++a) {
                aiVector3D v = pMesh->mNormals[verticesFound[a]];

                // Check whether the angle between the two normals is not too large.
                // Skip the angle check on our own normal to avoid false negatives
                // (v*v is not guaranteed to be 1.0 for all unit vectors v)
                if (is_not_qnan(v.x) && (verticesFound[a] == i || (v * vr >= fLimit)))
                    pcNor += v;
            }
            pcNew[i] = pcNor.NormalizeSafe();
        }
    }

    delete[] pMesh->mNormals;
    pMesh->mNormals = pcNew;

    return true;
}
