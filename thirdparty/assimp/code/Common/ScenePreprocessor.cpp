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

#include "ScenePreprocessor.h"
#include <assimp/ai_assert.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>


using namespace Assimp;

// ---------------------------------------------------------------------------------------------
void ScenePreprocessor::ProcessScene ()
{
    ai_assert(scene != NULL);

    // Process all meshes
    for (unsigned int i = 0; i < scene->mNumMeshes;++i)
        ProcessMesh(scene->mMeshes[i]);

    // - nothing to do for nodes for the moment
    // - nothing to do for textures for the moment
    // - nothing to do for lights for the moment
    // - nothing to do for cameras for the moment

    // Process all animations
    for (unsigned int i = 0; i < scene->mNumAnimations;++i)
        ProcessAnimation(scene->mAnimations[i]);

    // Generate a default material if none was specified
    if (!scene->mNumMaterials && scene->mNumMeshes) {
        scene->mMaterials      = new aiMaterial*[2];
        aiMaterial* helper;

        aiString name;

        scene->mMaterials[scene->mNumMaterials] = helper = new aiMaterial();
        aiColor3D clr(0.6f,0.6f,0.6f);
        helper->AddProperty(&clr,1,AI_MATKEY_COLOR_DIFFUSE);

        // setup the default name to make this material identifiable
        name.Set(AI_DEFAULT_MATERIAL_NAME);
        helper->AddProperty(&name,AI_MATKEY_NAME);

        ASSIMP_LOG_DEBUG("ScenePreprocessor: Adding default material \'" AI_DEFAULT_MATERIAL_NAME  "\'");

        for (unsigned int i = 0; i < scene->mNumMeshes;++i) {
            scene->mMeshes[i]->mMaterialIndex = scene->mNumMaterials;
        }

        scene->mNumMaterials++;
    }
}

// ---------------------------------------------------------------------------------------------
void ScenePreprocessor::ProcessMesh (aiMesh* mesh)
{
    // If aiMesh::mNumUVComponents is *not* set assign the default value of 2
    for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++i)   {
        if (!mesh->mTextureCoords[i]) {
            mesh->mNumUVComponents[i] = 0;
        } else {
            if (!mesh->mNumUVComponents[i])
                mesh->mNumUVComponents[i] = 2;

            aiVector3D* p = mesh->mTextureCoords[i], *end = p+mesh->mNumVertices;

            // Ensure unused components are zeroed. This will make 1D texture channels work
            // as if they were 2D channels .. just in case an application doesn't handle
            // this case
            if (2 == mesh->mNumUVComponents[i]) {
                for (; p != end; ++p)
                    p->z = 0.f;
            }
            else if (1 == mesh->mNumUVComponents[i]) {
                for (; p != end; ++p)
                    p->z = p->y = 0.f;
            }
            else if (3 == mesh->mNumUVComponents[i]) {
                // Really 3D coordinates? Check whether the third coordinate is != 0 for at least one element
                for (; p != end; ++p) {
                    if (p->z != 0)
                        break;
                }
                if (p == end) {
                    ASSIMP_LOG_WARN("ScenePreprocessor: UVs are declared to be 3D but they're obviously not. Reverting to 2D.");
                    mesh->mNumUVComponents[i] = 2;
                }
            }
        }
    }

    // If the information which primitive types are there in the
    // mesh is currently not available, compute it.
    if (!mesh->mPrimitiveTypes) {
        for (unsigned int a = 0; a < mesh->mNumFaces; ++a)  {
            aiFace& face = mesh->mFaces[a];
            switch (face.mNumIndices)
            {
            case 3u:
                mesh->mPrimitiveTypes |= aiPrimitiveType_TRIANGLE;
                break;

            case 2u:
                mesh->mPrimitiveTypes |= aiPrimitiveType_LINE;
                break;

            case 1u:
                mesh->mPrimitiveTypes |= aiPrimitiveType_POINT;
                break;

            default:
                mesh->mPrimitiveTypes |= aiPrimitiveType_POLYGON;
                break;
            }
        }
    }

    // If tangents and normals are given but no bitangents compute them
    if (mesh->mTangents && mesh->mNormals && !mesh->mBitangents)    {

        mesh->mBitangents = new aiVector3D[mesh->mNumVertices];
        for (unsigned int i = 0; i < mesh->mNumVertices;++i)    {
            mesh->mBitangents[i] = mesh->mNormals[i] ^ mesh->mTangents[i];
        }
    }
}

// ---------------------------------------------------------------------------------------------
void ScenePreprocessor::ProcessAnimation (aiAnimation* anim)
{
    double first = 10e10, last = -10e10;
    for (unsigned int i = 0; i < anim->mNumChannels;++i)    {
        aiNodeAnim* channel = anim->mChannels[i];

        /*  If the exact duration of the animation is not given
         *  compute it now.
         */
        if (anim->mDuration == -1.) {

            // Position keys
            for (unsigned int j = 0; j < channel->mNumPositionKeys;++j) {
                aiVectorKey& key = channel->mPositionKeys[j];
                first = std::min (first, key.mTime);
                last  = std::max (last,  key.mTime);
            }

            // Scaling keys
            for (unsigned int j = 0; j < channel->mNumScalingKeys;++j )  {
                aiVectorKey& key = channel->mScalingKeys[j];
                first = std::min (first, key.mTime);
                last  = std::max (last,  key.mTime);
            }

            // Rotation keys
            for (unsigned int j = 0; j < channel->mNumRotationKeys;++j ) {
                aiQuatKey& key = channel->mRotationKeys[ j ];
                first = std::min (first, key.mTime);
                last  = std::max (last,  key.mTime);
            }
        }

        /*  Check whether the animation channel has no rotation
         *  or position tracks. In this case we generate a dummy
         *  track from the information we have in the transformation
         *  matrix of the corresponding node.
         */
        if (!channel->mNumRotationKeys || !channel->mNumPositionKeys || !channel->mNumScalingKeys)  {
            // Find the node that belongs to this animation
            aiNode* node = scene->mRootNode->FindNode(channel->mNodeName);
            if (node) // ValidateDS will complain later if 'node' is NULL
            {
                // Decompose the transformation matrix of the node
                aiVector3D scaling, position;
                aiQuaternion rotation;

                node->mTransformation.Decompose(scaling, rotation,position);

                // No rotation keys? Generate a dummy track
                if (!channel->mNumRotationKeys) {
                    ai_assert(!channel->mRotationKeys);
                    channel->mNumRotationKeys = 1;
                    channel->mRotationKeys = new aiQuatKey[1];
                    aiQuatKey& q = channel->mRotationKeys[0];

                    q.mTime  = 0.;
                    q.mValue = rotation;

                    ASSIMP_LOG_DEBUG("ScenePreprocessor: Dummy rotation track has been generated");
                } else {
                    ai_assert(channel->mRotationKeys);
                }

                // No scaling keys? Generate a dummy track
                if (!channel->mNumScalingKeys)  {
                    ai_assert(!channel->mScalingKeys);
                    channel->mNumScalingKeys = 1;
                    channel->mScalingKeys = new aiVectorKey[1];
                    aiVectorKey& q = channel->mScalingKeys[0];

                    q.mTime  = 0.;
                    q.mValue = scaling;

                    ASSIMP_LOG_DEBUG("ScenePreprocessor: Dummy scaling track has been generated");
                } else {
                    ai_assert(channel->mScalingKeys);
                }

                // No position keys? Generate a dummy track
                if (!channel->mNumPositionKeys) {
                    ai_assert(!channel->mPositionKeys);
                    channel->mNumPositionKeys = 1;
                    channel->mPositionKeys = new aiVectorKey[1];
                    aiVectorKey& q = channel->mPositionKeys[0];

                    q.mTime  = 0.;
                    q.mValue = position;

                    ASSIMP_LOG_DEBUG("ScenePreprocessor: Dummy position track has been generated");
                } else {
                    ai_assert(channel->mPositionKeys);
                }
            }
        }
    }

    if (anim->mDuration == -1.)     {
        ASSIMP_LOG_DEBUG("ScenePreprocessor: Setting animation duration");
        anim->mDuration = last - std::min( first, 0. );
    }
}
