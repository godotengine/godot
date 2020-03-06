/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team



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

/** @file  MakeLeftHandedProcess.cpp
 *  @brief Implementation of the post processing step to convert all
 *  imported data to a left-handed coordinate system.
 *
 *  Face order & UV flip are also implemented here, for the sake of a
 *  better location.
 */


#include "ConvertToLHProcess.h"
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/DefaultLogger.hpp>

using namespace Assimp;

#ifndef ASSIMP_BUILD_NO_MAKELEFTHANDED_PROCESS

namespace {

template <typename aiMeshType>
void flipUVs(aiMeshType* pMesh) {
    if (pMesh == nullptr) { return; }
    // mirror texture y coordinate
    for (unsigned int tcIdx = 0; tcIdx < AI_MAX_NUMBER_OF_TEXTURECOORDS; tcIdx++) {
        if (!pMesh->HasTextureCoords(tcIdx)) {
            break;
        }

        for (unsigned int vIdx = 0; vIdx < pMesh->mNumVertices; vIdx++) {
            pMesh->mTextureCoords[tcIdx][vIdx].y = 1.0f - pMesh->mTextureCoords[tcIdx][vIdx].y;
        }
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
MakeLeftHandedProcess::MakeLeftHandedProcess()
: BaseProcess() {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
MakeLeftHandedProcess::~MakeLeftHandedProcess() {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool MakeLeftHandedProcess::IsActive( unsigned int pFlags) const
{
    return 0 != (pFlags & aiProcess_MakeLeftHanded);
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void MakeLeftHandedProcess::Execute( aiScene* pScene)
{
    // Check for an existent root node to proceed
    ai_assert(pScene->mRootNode != NULL);
    ASSIMP_LOG_DEBUG("MakeLeftHandedProcess begin");

    // recursively convert all the nodes
    ProcessNode( pScene->mRootNode, aiMatrix4x4());

    // process the meshes accordingly
    for ( unsigned int a = 0; a < pScene->mNumMeshes; ++a ) {
        ProcessMesh( pScene->mMeshes[ a ] );
    }

    // process the materials accordingly
    for ( unsigned int a = 0; a < pScene->mNumMaterials; ++a ) {
        ProcessMaterial( pScene->mMaterials[ a ] );
    }

    // transform all animation channels as well
    for( unsigned int a = 0; a < pScene->mNumAnimations; a++)
    {
        aiAnimation* anim = pScene->mAnimations[a];
        for( unsigned int b = 0; b < anim->mNumChannels; b++)
        {
            aiNodeAnim* nodeAnim = anim->mChannels[b];
            ProcessAnimation( nodeAnim);
        }
    }
    ASSIMP_LOG_DEBUG("MakeLeftHandedProcess finished");
}

// ------------------------------------------------------------------------------------------------
// Recursively converts a node, all of its children and all of its meshes
void MakeLeftHandedProcess::ProcessNode( aiNode* pNode, const aiMatrix4x4& pParentGlobalRotation)
{
    // mirror all base vectors at the local Z axis
    pNode->mTransformation.c1 = -pNode->mTransformation.c1;
    pNode->mTransformation.c2 = -pNode->mTransformation.c2;
    pNode->mTransformation.c3 = -pNode->mTransformation.c3;
    pNode->mTransformation.c4 = -pNode->mTransformation.c4;

    // now invert the Z axis again to keep the matrix determinant positive.
    // The local meshes will be inverted accordingly so that the result should look just fine again.
    pNode->mTransformation.a3 = -pNode->mTransformation.a3;
    pNode->mTransformation.b3 = -pNode->mTransformation.b3;
    pNode->mTransformation.c3 = -pNode->mTransformation.c3;
    pNode->mTransformation.d3 = -pNode->mTransformation.d3; // useless, but anyways...

    // continue for all children
    for( size_t a = 0; a < pNode->mNumChildren; ++a ) {
        ProcessNode( pNode->mChildren[ a ], pParentGlobalRotation * pNode->mTransformation );
    }
}

// ------------------------------------------------------------------------------------------------
// Converts a single mesh to left handed coordinates.
void MakeLeftHandedProcess::ProcessMesh( aiMesh* pMesh) {
    if ( nullptr == pMesh ) {
        ASSIMP_LOG_ERROR( "Nullptr to mesh found." );
        return;
    }
    // mirror positions, normals and stuff along the Z axis
    for( size_t a = 0; a < pMesh->mNumVertices; ++a)
    {
        pMesh->mVertices[a].z *= -1.0f;
        if (pMesh->HasNormals()) {
            pMesh->mNormals[a].z *= -1.0f;
        }
        if( pMesh->HasTangentsAndBitangents())
        {
            pMesh->mTangents[a].z *= -1.0f;
            pMesh->mBitangents[a].z *= -1.0f;
        }
    }

    // mirror anim meshes positions, normals and stuff along the Z axis
    for (size_t m = 0; m < pMesh->mNumAnimMeshes; ++m)
    {
        for (size_t a = 0; a < pMesh->mAnimMeshes[m]->mNumVertices; ++a)
        {
            pMesh->mAnimMeshes[m]->mVertices[a].z *= -1.0f;
            if (pMesh->mAnimMeshes[m]->HasNormals()) {
                pMesh->mAnimMeshes[m]->mNormals[a].z *= -1.0f;
            }
            if (pMesh->mAnimMeshes[m]->HasTangentsAndBitangents())
            {
                pMesh->mAnimMeshes[m]->mTangents[a].z *= -1.0f;
                pMesh->mAnimMeshes[m]->mBitangents[a].z *= -1.0f;
            }
        }
    }

    // mirror offset matrices of all bones
    for( size_t a = 0; a < pMesh->mNumBones; ++a)
    {
        aiBone* bone = pMesh->mBones[a];
        bone->mOffsetMatrix.a3 = -bone->mOffsetMatrix.a3;
        bone->mOffsetMatrix.b3 = -bone->mOffsetMatrix.b3;
        bone->mOffsetMatrix.d3 = -bone->mOffsetMatrix.d3;
        bone->mOffsetMatrix.c1 = -bone->mOffsetMatrix.c1;
        bone->mOffsetMatrix.c2 = -bone->mOffsetMatrix.c2;
        bone->mOffsetMatrix.c4 = -bone->mOffsetMatrix.c4;
    }

    // mirror bitangents as well as they're derived from the texture coords
    if( pMesh->HasTangentsAndBitangents())
    {
        for( unsigned int a = 0; a < pMesh->mNumVertices; a++)
            pMesh->mBitangents[a] *= -1.0f;
    }
}

// ------------------------------------------------------------------------------------------------
// Converts a single material to left handed coordinates.
void MakeLeftHandedProcess::ProcessMaterial( aiMaterial* _mat) {
    if ( nullptr == _mat ) {
        ASSIMP_LOG_ERROR( "Nullptr to aiMaterial found." );
        return;
    }

    aiMaterial* mat = (aiMaterial*)_mat;
    for (unsigned int a = 0; a < mat->mNumProperties;++a)   {
        aiMaterialProperty* prop = mat->mProperties[a];

        // Mapping axis for UV mappings?
        if (!::strcmp( prop->mKey.data, "$tex.mapaxis"))    {
            ai_assert( prop->mDataLength >= sizeof(aiVector3D)); /* something is wrong with the validation if we end up here */
            aiVector3D* pff = (aiVector3D*)prop->mData;
            pff->z *= -1.f;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Converts the given animation to LH coordinates.
void MakeLeftHandedProcess::ProcessAnimation( aiNodeAnim* pAnim)
{
    // position keys
    for( unsigned int a = 0; a < pAnim->mNumPositionKeys; a++)
        pAnim->mPositionKeys[a].mValue.z *= -1.0f;

    // rotation keys
    for( unsigned int a = 0; a < pAnim->mNumRotationKeys; a++)
    {
        /* That's the safe version, but the float errors add up. So we try the short version instead
        aiMatrix3x3 rotmat = pAnim->mRotationKeys[a].mValue.GetMatrix();
        rotmat.a3 = -rotmat.a3; rotmat.b3 = -rotmat.b3;
        rotmat.c1 = -rotmat.c1; rotmat.c2 = -rotmat.c2;
        aiQuaternion rotquat( rotmat);
        pAnim->mRotationKeys[a].mValue = rotquat;
        */
        pAnim->mRotationKeys[a].mValue.x *= -1.0f;
        pAnim->mRotationKeys[a].mValue.y *= -1.0f;
    }
}

#endif // !!  ASSIMP_BUILD_NO_MAKELEFTHANDED_PROCESS
#ifndef  ASSIMP_BUILD_NO_FLIPUVS_PROCESS
// # FlipUVsProcess

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
FlipUVsProcess::FlipUVsProcess()
{}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
FlipUVsProcess::~FlipUVsProcess()
{}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool FlipUVsProcess::IsActive( unsigned int pFlags) const
{
    return 0 != (pFlags & aiProcess_FlipUVs);
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void FlipUVsProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("FlipUVsProcess begin");
    for (unsigned int i = 0; i < pScene->mNumMeshes;++i)
        ProcessMesh(pScene->mMeshes[i]);

    for (unsigned int i = 0; i < pScene->mNumMaterials;++i)
        ProcessMaterial(pScene->mMaterials[i]);
    ASSIMP_LOG_DEBUG("FlipUVsProcess finished");
}

// ------------------------------------------------------------------------------------------------
// Converts a single material
void FlipUVsProcess::ProcessMaterial (aiMaterial* _mat)
{
    aiMaterial* mat = (aiMaterial*)_mat;
    for (unsigned int a = 0; a < mat->mNumProperties;++a)   {
        aiMaterialProperty* prop = mat->mProperties[a];
        if( !prop ) {
            ASSIMP_LOG_DEBUG( "Property is null" );
            continue;
        }

        // UV transformation key?
        if (!::strcmp( prop->mKey.data, "$tex.uvtrafo"))    {
            ai_assert( prop->mDataLength >= sizeof(aiUVTransform));  /* something is wrong with the validation if we end up here */
            aiUVTransform* uv = (aiUVTransform*)prop->mData;

            // just flip it, that's everything
            uv->mTranslation.y *= -1.f;
            uv->mRotation *= -1.f;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Converts a single mesh
void FlipUVsProcess::ProcessMesh( aiMesh* pMesh)
{
    flipUVs(pMesh);
    for (unsigned int idx = 0; idx < pMesh->mNumAnimMeshes; idx++) {
        flipUVs(pMesh->mAnimMeshes[idx]);
    }
}

#endif // !ASSIMP_BUILD_NO_FLIPUVS_PROCESS
#ifndef  ASSIMP_BUILD_NO_FLIPWINDING_PROCESS
// # FlipWindingOrderProcess

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
FlipWindingOrderProcess::FlipWindingOrderProcess()
{}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
FlipWindingOrderProcess::~FlipWindingOrderProcess()
{}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool FlipWindingOrderProcess::IsActive( unsigned int pFlags) const
{
    return 0 != (pFlags & aiProcess_FlipWindingOrder);
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void FlipWindingOrderProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("FlipWindingOrderProcess begin");
    for (unsigned int i = 0; i < pScene->mNumMeshes;++i)
        ProcessMesh(pScene->mMeshes[i]);
    ASSIMP_LOG_DEBUG("FlipWindingOrderProcess finished");
}

// ------------------------------------------------------------------------------------------------
// Converts a single mesh
void FlipWindingOrderProcess::ProcessMesh( aiMesh* pMesh)
{
    // invert the order of all faces in this mesh
    for( unsigned int a = 0; a < pMesh->mNumFaces; a++)
    {
        aiFace& face = pMesh->mFaces[a];
        for (unsigned int b = 0; b < face.mNumIndices / 2; b++) {
            std::swap(face.mIndices[b], face.mIndices[face.mNumIndices - 1 - b]);
        }
    }

    // invert the order of all components in this mesh anim meshes
    for (unsigned int m = 0; m < pMesh->mNumAnimMeshes; m++) {
        aiAnimMesh* animMesh = pMesh->mAnimMeshes[m];
        unsigned int numVertices = animMesh->mNumVertices;
        if (animMesh->HasPositions()) {
            for (unsigned int a = 0; a < numVertices; a++)
            {
                std::swap(animMesh->mVertices[a], animMesh->mVertices[numVertices - 1 - a]);
            }
        }
        if (animMesh->HasNormals()) {
            for (unsigned int a = 0; a < numVertices; a++)
            {
                std::swap(animMesh->mNormals[a], animMesh->mNormals[numVertices - 1 - a]);
            }
        }
        for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; i++) {
            if (animMesh->HasTextureCoords(i)) {
                for (unsigned int a = 0; a < numVertices; a++)
                {
                    std::swap(animMesh->mTextureCoords[i][a], animMesh->mTextureCoords[i][numVertices - 1 - a]);
                }
            }
        }
        if (animMesh->HasTangentsAndBitangents()) {
            for (unsigned int a = 0; a < numVertices; a++)
            {
                std::swap(animMesh->mTangents[a], animMesh->mTangents[numVertices - 1 - a]);
                std::swap(animMesh->mBitangents[a], animMesh->mBitangents[numVertices - 1 - a]);
            }
        }
        for (unsigned int v = 0; v < AI_MAX_NUMBER_OF_COLOR_SETS; v++) {
            if (animMesh->HasVertexColors(v)) {
                for (unsigned int a = 0; a < numVertices; a++)
                {
                    std::swap(animMesh->mColors[v][a], animMesh->mColors[v][numVertices - 1 - a]);
                }
            }
        }
    }
}

#endif // !! ASSIMP_BUILD_NO_FLIPWINDING_PROCESS
