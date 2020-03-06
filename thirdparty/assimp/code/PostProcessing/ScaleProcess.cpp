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
#include "ScaleProcess.h"

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/BaseImporter.h>

namespace Assimp {

ScaleProcess::ScaleProcess()
: BaseProcess()
, mScale( AI_CONFIG_GLOBAL_SCALE_FACTOR_DEFAULT ) {
}

ScaleProcess::~ScaleProcess() {
    // empty
}

void ScaleProcess::setScale( ai_real scale ) {
    mScale = scale;
}

ai_real ScaleProcess::getScale() const {
    return mScale;
}

bool ScaleProcess::IsActive( unsigned int pFlags ) const {
    return ( pFlags & aiProcess_GlobalScale ) != 0;
}

void ScaleProcess::SetupProperties( const Importer* pImp ) {
    // User scaling
    mScale = pImp->GetPropertyFloat( AI_CONFIG_GLOBAL_SCALE_FACTOR_KEY, 1.0f );

    // File scaling * Application Scaling
    float importerScale = pImp->GetPropertyFloat( AI_CONFIG_APP_SCALE_KEY, 1.0f );

    // apply scale to the scale 
    // helps prevent bugs with backward compatibility for anyone using normal scaling.
    mScale *= importerScale;
}

void ScaleProcess::Execute( aiScene* pScene ) {
    if(mScale == 1.0f)  {
        return; // nothing to scale
    }
    
    ai_assert( mScale != 0 );
    ai_assert( nullptr != pScene );
    ai_assert( nullptr != pScene->mRootNode );

    if ( nullptr == pScene ) {
        return;
    }

    if ( nullptr == pScene->mRootNode ) {
        return;
    }
    
    // Process animations and update position transform to new unit system
    for( unsigned int animationID = 0; animationID < pScene->mNumAnimations; animationID++ )
    {
        aiAnimation* animation = pScene->mAnimations[animationID];

        for( unsigned int animationChannel = 0; animationChannel < animation->mNumChannels; animationChannel++)
        {
            aiNodeAnim* anim = animation->mChannels[animationChannel];
            
            for( unsigned int posKey = 0; posKey < anim->mNumPositionKeys; posKey++)
            {
                aiVectorKey& vectorKey = anim->mPositionKeys[posKey];
                vectorKey.mValue *= mScale;
            }
        }
    }

    for( unsigned int meshID = 0; meshID < pScene->mNumMeshes; meshID++)
    {
        aiMesh *mesh = pScene->mMeshes[meshID]; 
        
        // Reconstruct mesh vertexes to the new unit system
        for( unsigned int vertexID = 0; vertexID < mesh->mNumVertices; vertexID++)
        {
            aiVector3D& vertex = mesh->mVertices[vertexID];
            vertex *= mScale;
        }


        // bone placement / scaling
        for( unsigned int boneID = 0; boneID < mesh->mNumBones; boneID++)
        {
            // Reconstruct matrix by transform rather than by scale 
            // This prevent scale values being changed which can
            // be meaningful in some cases 
            // like when you want the modeller to see 1:1 compatibility.
            aiBone* bone = mesh->mBones[boneID];

            aiVector3D pos, scale;
            aiQuaternion rotation;

            bone->mOffsetMatrix.Decompose( scale, rotation, pos);
            
            aiMatrix4x4 translation;
            aiMatrix4x4::Translation( pos * mScale, translation );
            
            aiMatrix4x4 scaling;
            aiMatrix4x4::Scaling( aiVector3D(scale), scaling );

            aiMatrix4x4 RotMatrix = aiMatrix4x4 (rotation.GetMatrix());

            bone->mOffsetMatrix = translation * RotMatrix * scaling;
        }


        // animation mesh processing
        // convert by position rather than scale.
        for( unsigned int animMeshID = 0; animMeshID < mesh->mNumAnimMeshes; animMeshID++)
        {
            aiAnimMesh * animMesh = mesh->mAnimMeshes[animMeshID];
            
            for( unsigned int vertexID = 0; vertexID < animMesh->mNumVertices; vertexID++)
            {
                aiVector3D& vertex = animMesh->mVertices[vertexID];
                vertex *= mScale;
            }
        }
    }

    traverseNodes( pScene->mRootNode );
}

void ScaleProcess::traverseNodes( aiNode *node, unsigned int nested_node_id ) {    
    applyScaling( node );

    for( size_t i = 0; i < node->mNumChildren; i++)
    {
        // recurse into the tree until we are done!
        traverseNodes( node->mChildren[i], nested_node_id+1 ); 
    }
}

void ScaleProcess::applyScaling( aiNode *currentNode ) {
    if ( nullptr != currentNode ) {
        // Reconstruct matrix by transform rather than by scale 
        // This prevent scale values being changed which can
        // be meaningful in some cases 
        // like when you want the modeller to 
        // see 1:1 compatibility.
        
        aiVector3D pos, scale;
        aiQuaternion rotation;
        currentNode->mTransformation.Decompose( scale, rotation, pos);
        
        aiMatrix4x4 translation;
        aiMatrix4x4::Translation( pos * mScale, translation );
        
        aiMatrix4x4 scaling;

        // note: we do not use mScale here, this is on purpose.
        aiMatrix4x4::Scaling( scale, scaling );

        aiMatrix4x4 RotMatrix = aiMatrix4x4 (rotation.GetMatrix());

        currentNode->mTransformation = translation * RotMatrix * scaling;
    }
}

} // Namespace Assimp
