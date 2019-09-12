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

/** @file Defines a post processing step to search an importer's output
    for data that is obviously invalid  */



#ifndef ASSIMP_BUILD_NO_FINDINVALIDDATA_PROCESS

// internal headers
#include "FindInvalidDataProcess.h"
#include "ProcessHelper.h"

#include <assimp/Macros.h>
#include <assimp/Exceptional.h>
#include <assimp/qnan.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
FindInvalidDataProcess::FindInvalidDataProcess()
: configEpsilon(0.0)
, mIgnoreTexCoods( false ){
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
FindInvalidDataProcess::~FindInvalidDataProcess() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool FindInvalidDataProcess::IsActive( unsigned int pFlags) const {
    return 0 != (pFlags & aiProcess_FindInvalidData);
}

// ------------------------------------------------------------------------------------------------
// Setup import configuration
void FindInvalidDataProcess::SetupProperties(const Importer* pImp) {
    // Get the current value of AI_CONFIG_PP_FID_ANIM_ACCURACY
    configEpsilon = (0 != pImp->GetPropertyFloat(AI_CONFIG_PP_FID_ANIM_ACCURACY,0.f));
    mIgnoreTexCoods = pImp->GetPropertyBool(AI_CONFIG_PP_FID_IGNORE_TEXTURECOORDS, false);
}

// ------------------------------------------------------------------------------------------------
// Update mesh references in the node graph
void UpdateMeshReferences(aiNode* node, const std::vector<unsigned int>& meshMapping) {
    if (node->mNumMeshes)   {
        unsigned int out = 0;
        for (unsigned int a = 0; a < node->mNumMeshes;++a)  {

            unsigned int ref = node->mMeshes[a];
            if (UINT_MAX != (ref = meshMapping[ref]))   {
                node->mMeshes[out++] = ref;
            }
        }
        // just let the members that are unused, that's much cheaper
        // than a full array realloc'n'copy party ...
        if(!(node->mNumMeshes = out))   {

            delete[] node->mMeshes;
            node->mMeshes = NULL;
        }
    }
    // recursively update all children
    for (unsigned int i = 0; i < node->mNumChildren;++i) {
        UpdateMeshReferences(node->mChildren[i],meshMapping);
    }
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void FindInvalidDataProcess::Execute( aiScene* pScene) {
    ASSIMP_LOG_DEBUG("FindInvalidDataProcess begin");

    bool out = false;
    std::vector<unsigned int> meshMapping(pScene->mNumMeshes);
    unsigned int real = 0;

    // Process meshes
    for( unsigned int a = 0; a < pScene->mNumMeshes; a++)   {

        int result;
        if ((result = ProcessMesh( pScene->mMeshes[a])))    {
            out = true;

            if (2 == result)    {
                // remove this mesh
                delete pScene->mMeshes[a];
                AI_DEBUG_INVALIDATE_PTR(pScene->mMeshes[a]);

                meshMapping[a] = UINT_MAX;
                continue;
            }
        }
        pScene->mMeshes[real] = pScene->mMeshes[a];
        meshMapping[a] = real++;
    }

    // Process animations
    for (unsigned int a = 0; a < pScene->mNumAnimations;++a) {
        ProcessAnimation( pScene->mAnimations[a]);
    }


    if (out)    {
        if ( real != pScene->mNumMeshes)    {
            if (!real) {
                throw DeadlyImportError("No meshes remaining");
            }

            // we need to remove some meshes.
            // therefore we'll also need to remove all references
            // to them from the scenegraph
            UpdateMeshReferences(pScene->mRootNode,meshMapping);
            pScene->mNumMeshes = real;
        }

        ASSIMP_LOG_INFO("FindInvalidDataProcess finished. Found issues ...");
    } else {
        ASSIMP_LOG_DEBUG("FindInvalidDataProcess finished. Everything seems to be OK.");
    }
}

// ------------------------------------------------------------------------------------------------
template <typename T>
inline
const char* ValidateArrayContents(const T* /*arr*/, unsigned int /*size*/,
        const std::vector<bool>& /*dirtyMask*/, bool /*mayBeIdentical = false*/, bool /*mayBeZero = true*/)
{
    return nullptr;
}

// ------------------------------------------------------------------------------------------------
template <>
inline
const char* ValidateArrayContents<aiVector3D>(const aiVector3D* arr, unsigned int size,
        const std::vector<bool>& dirtyMask, bool mayBeIdentical , bool mayBeZero ) {
    bool b = false;
    unsigned int cnt = 0;
    for (unsigned int i = 0; i < size;++i)  {

        if (dirtyMask.size() && dirtyMask[i]) {
            continue;
        }
        ++cnt;

        const aiVector3D& v = arr[i];
        if (is_special_float(v.x) || is_special_float(v.y) || is_special_float(v.z))    {
            return "INF/NAN was found in a vector component";
        }
        if (!mayBeZero && !v.x && !v.y && !v.z )    {
            return "Found zero-length vector";
        }
        if (i && v != arr[i-1])b = true;
    }
    if (cnt > 1 && !b && !mayBeIdentical) {
        return "All vectors are identical";
    }
    return nullptr;
}

// ------------------------------------------------------------------------------------------------
template <typename T>
inline 
bool ProcessArray(T*& in, unsigned int num,const char* name,
        const std::vector<bool>& dirtyMask, bool mayBeIdentical = false, bool mayBeZero = true) {
    const char* err = ValidateArrayContents(in,num,dirtyMask,mayBeIdentical,mayBeZero);
    if (err)    {
        ASSIMP_LOG_ERROR_F( "FindInvalidDataProcess fails on mesh ", name, ": ", err);
        delete[] in;
        in = NULL;
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
template <typename T>
AI_FORCE_INLINE bool EpsilonCompare(const T& n, const T& s, ai_real epsilon);

// ------------------------------------------------------------------------------------------------
AI_FORCE_INLINE bool EpsilonCompare(ai_real n, ai_real s, ai_real epsilon) {
    return std::fabs(n-s)>epsilon;
}

// ------------------------------------------------------------------------------------------------
template <>
bool EpsilonCompare<aiVectorKey>(const aiVectorKey& n, const aiVectorKey& s, ai_real epsilon) {
    return
        EpsilonCompare(n.mValue.x,s.mValue.x,epsilon) &&
        EpsilonCompare(n.mValue.y,s.mValue.y,epsilon) &&
        EpsilonCompare(n.mValue.z,s.mValue.z,epsilon);
}

// ------------------------------------------------------------------------------------------------
template <>
bool EpsilonCompare<aiQuatKey>(const aiQuatKey& n, const aiQuatKey& s, ai_real epsilon)   {
    return
        EpsilonCompare(n.mValue.x,s.mValue.x,epsilon) &&
        EpsilonCompare(n.mValue.y,s.mValue.y,epsilon) &&
        EpsilonCompare(n.mValue.z,s.mValue.z,epsilon) &&
        EpsilonCompare(n.mValue.w,s.mValue.w,epsilon);
}

// ------------------------------------------------------------------------------------------------
template <typename T>
inline
bool AllIdentical(T* in, unsigned int num, ai_real epsilon) {
    if (num <= 1) {
        return true;
    }

    if (fabs(epsilon) > 0.f) {
        for (unsigned int i = 0; i < num-1;++i) {
            if (!EpsilonCompare(in[i],in[i+1],epsilon)) {
                return false;
            }
        }
    } else {
        for (unsigned int i = 0; i < num-1;++i) {
            if (in[i] != in[i+1]) {
                return false;
            }
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// Search an animation for invalid content
void FindInvalidDataProcess::ProcessAnimation (aiAnimation* anim) {
    // Process all animation channels
    for ( unsigned int a = 0; a < anim->mNumChannels; ++a ) {
        ProcessAnimationChannel( anim->mChannels[a]);
    }
}

// ------------------------------------------------------------------------------------------------
void FindInvalidDataProcess::ProcessAnimationChannel (aiNodeAnim* anim) {
    ai_assert( nullptr != anim );
    if (anim->mNumPositionKeys == 0 && anim->mNumRotationKeys == 0 && anim->mNumScalingKeys == 0) {
        ai_assert_entry();
        return;
    }

    // Check whether all values in a tracks are identical - in this case
    // we can remove al keys except one.
    // POSITIONS
    int i = 0;
    if (anim->mNumPositionKeys > 1 && AllIdentical(anim->mPositionKeys,anim->mNumPositionKeys,configEpsilon)) {
        aiVectorKey v = anim->mPositionKeys[0];

        // Reallocate ... we need just ONE element, it makes no sense to reuse the array
        delete[] anim->mPositionKeys;
        anim->mPositionKeys = new aiVectorKey[anim->mNumPositionKeys = 1];
        anim->mPositionKeys[0] = v;
        i = 1;
    }

    // ROTATIONS
    if (anim->mNumRotationKeys > 1 && AllIdentical(anim->mRotationKeys,anim->mNumRotationKeys,configEpsilon)) {
        aiQuatKey v = anim->mRotationKeys[0];

        // Reallocate ... we need just ONE element, it makes no sense to reuse the array
        delete[] anim->mRotationKeys;
        anim->mRotationKeys = new aiQuatKey[anim->mNumRotationKeys = 1];
        anim->mRotationKeys[0] = v;
        i = 1;
    }

    // SCALINGS
    if (anim->mNumScalingKeys > 1 && AllIdentical(anim->mScalingKeys,anim->mNumScalingKeys,configEpsilon)) {
        aiVectorKey v = anim->mScalingKeys[0];

        // Reallocate ... we need just ONE element, it makes no sense to reuse the array
        delete[] anim->mScalingKeys;
        anim->mScalingKeys = new aiVectorKey[anim->mNumScalingKeys = 1];
        anim->mScalingKeys[0] = v;
        i = 1;
    }
    if ( 1 == i ) {
        ASSIMP_LOG_WARN("Simplified dummy tracks with just one key");
    }
}

// ------------------------------------------------------------------------------------------------
// Search a mesh for invalid contents
int FindInvalidDataProcess::ProcessMesh(aiMesh* pMesh)
{
    bool ret = false;
    std::vector<bool> dirtyMask(pMesh->mNumVertices, pMesh->mNumFaces != 0);

    // Ignore elements that are not referenced by vertices.
    // (they are, for example, caused by the FindDegenerates step)
    for (unsigned int m = 0; m < pMesh->mNumFaces; ++m) {
        const aiFace& f = pMesh->mFaces[m];

        for (unsigned int i = 0; i < f.mNumIndices; ++i) {
            dirtyMask[f.mIndices[i]] = false;
        }
    }

    // Process vertex positions
    if (pMesh->mVertices && ProcessArray(pMesh->mVertices, pMesh->mNumVertices, "positions", dirtyMask)) {
        ASSIMP_LOG_ERROR("Deleting mesh: Unable to continue without vertex positions");

        return 2;
    }

    // process texture coordinates
    if (!mIgnoreTexCoods) {
        for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS && pMesh->mTextureCoords[i]; ++i) {
            if (ProcessArray(pMesh->mTextureCoords[i], pMesh->mNumVertices, "uvcoords", dirtyMask)) {
                pMesh->mNumUVComponents[i] = 0;

                // delete all subsequent texture coordinate sets.
                for (unsigned int a = i + 1; a < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++a) {
                    delete[] pMesh->mTextureCoords[a];
                    pMesh->mTextureCoords[a] = NULL;
                    pMesh->mNumUVComponents[a] = 0;
                }

                ret = true;
            }
        }
    }

    // -- we don't validate vertex colors, it's difficult to say whether
    // they are invalid or not.

    // Normals and tangents are undefined for point and line faces.
    if (pMesh->mNormals || pMesh->mTangents)    {

        if (aiPrimitiveType_POINT & pMesh->mPrimitiveTypes ||
            aiPrimitiveType_LINE  & pMesh->mPrimitiveTypes)
        {
            if (aiPrimitiveType_TRIANGLE & pMesh->mPrimitiveTypes ||
                aiPrimitiveType_POLYGON  & pMesh->mPrimitiveTypes)
            {
                // We need to update the lookup-table
                for (unsigned int m = 0; m < pMesh->mNumFaces;++m) {
                    const aiFace& f = pMesh->mFaces[ m ];

                    if (f.mNumIndices < 3)  {
                        dirtyMask[f.mIndices[0]] = true;
                        if (f.mNumIndices == 2) {
                            dirtyMask[f.mIndices[1]] = true;
                        }
                    }
                }
            }
            // Normals, tangents and bitangents are undefined for
            // the whole mesh (and should not even be there)
            else {
                return ret;
            }
        }

        // Process mesh normals
        if (pMesh->mNormals && ProcessArray(pMesh->mNormals,pMesh->mNumVertices,
            "normals",dirtyMask,true,false))
            ret = true;

        // Process mesh tangents
        if (pMesh->mTangents && ProcessArray(pMesh->mTangents,pMesh->mNumVertices,"tangents",dirtyMask))    {
            delete[] pMesh->mBitangents; pMesh->mBitangents = NULL;
            ret = true;
        }

        // Process mesh bitangents
        if (pMesh->mBitangents && ProcessArray(pMesh->mBitangents,pMesh->mNumVertices,"bitangents",dirtyMask))  {
            delete[] pMesh->mTangents; pMesh->mTangents = NULL;
            ret = true;
        }
    }
    return ret ? 1 : 0;
}

#endif // !! ASSIMP_BUILD_NO_FINDINVALIDDATA_PROCESS
