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

/** @file  SkeletonMeshBuilder.cpp
 *  @brief Implementation of a little class to construct a dummy mesh for a skeleton
 */

#include <assimp/scene.h>
#include <assimp/SkeletonMeshBuilder.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// The constructor processes the given scene and adds a mesh there.
SkeletonMeshBuilder::SkeletonMeshBuilder( aiScene* pScene, aiNode* root, bool bKnobsOnly)
{
    // nothing to do if there's mesh data already present at the scene
    if( pScene->mNumMeshes > 0 || pScene->mRootNode == NULL)
        return;

    if (!root)
        root = pScene->mRootNode;

    mKnobsOnly = bKnobsOnly;

    // build some faces around each node
    CreateGeometry( root );

    // create a mesh to hold all the generated faces
    pScene->mNumMeshes = 1;
    pScene->mMeshes = new aiMesh*[1];
    pScene->mMeshes[0] = CreateMesh();
    // and install it at the root node
    root->mNumMeshes = 1;
    root->mMeshes = new unsigned int[1];
    root->mMeshes[0] = 0;

    // create a dummy material for the mesh
    if(pScene->mNumMaterials==0){
		pScene->mNumMaterials = 1;
		pScene->mMaterials = new aiMaterial*[1];
		pScene->mMaterials[0] = CreateMaterial();
    }
}

// ------------------------------------------------------------------------------------------------
// Recursively builds a simple mesh representation for the given node
void SkeletonMeshBuilder::CreateGeometry( const aiNode* pNode)
{
    // add a joint entry for the node.
    const unsigned int vertexStartIndex = static_cast<unsigned int>(mVertices.size());

    // now build the geometry.
    if( pNode->mNumChildren > 0 && !mKnobsOnly)
    {
        // If the node has children, we build little pointers to each of them
        for( unsigned int a = 0; a < pNode->mNumChildren; a++)
        {
            // find a suitable coordinate system
            const aiMatrix4x4& childTransform = pNode->mChildren[a]->mTransformation;
            aiVector3D childpos( childTransform.a4, childTransform.b4, childTransform.c4);
            ai_real distanceToChild = childpos.Length();
            if( distanceToChild < 0.0001)
                continue;
            aiVector3D up = aiVector3D( childpos).Normalize();

            aiVector3D orth( 1.0, 0.0, 0.0);
            if( std::fabs( orth * up) > 0.99)
                orth.Set( 0.0, 1.0, 0.0);

            aiVector3D front = (up ^ orth).Normalize();
            aiVector3D side = (front ^ up).Normalize();

            unsigned int localVertexStart = static_cast<unsigned int>(mVertices.size());
            mVertices.push_back( -front * distanceToChild * (ai_real)0.1);
            mVertices.push_back( childpos);
            mVertices.push_back( -side * distanceToChild * (ai_real)0.1);
            mVertices.push_back( -side * distanceToChild * (ai_real)0.1);
            mVertices.push_back( childpos);
            mVertices.push_back( front * distanceToChild * (ai_real)0.1);
            mVertices.push_back( front * distanceToChild * (ai_real)0.1);
            mVertices.push_back( childpos);
            mVertices.push_back( side * distanceToChild * (ai_real)0.1);
            mVertices.push_back( side * distanceToChild * (ai_real)0.1);
            mVertices.push_back( childpos);
            mVertices.push_back( -front * distanceToChild * (ai_real)0.1);

            mFaces.push_back( Face( localVertexStart + 0, localVertexStart + 1, localVertexStart + 2));
            mFaces.push_back( Face( localVertexStart + 3, localVertexStart + 4, localVertexStart + 5));
            mFaces.push_back( Face( localVertexStart + 6, localVertexStart + 7, localVertexStart + 8));
            mFaces.push_back( Face( localVertexStart + 9, localVertexStart + 10, localVertexStart + 11));
        }
    }
    else
    {
        // if the node has no children, it's an end node. Put a little knob there instead
        aiVector3D ownpos( pNode->mTransformation.a4, pNode->mTransformation.b4, pNode->mTransformation.c4);
        ai_real sizeEstimate = ownpos.Length() * ai_real( 0.18 );

        mVertices.push_back( aiVector3D( -sizeEstimate, 0.0, 0.0));
        mVertices.push_back( aiVector3D( 0.0, sizeEstimate, 0.0));
        mVertices.push_back( aiVector3D( 0.0, 0.0, -sizeEstimate));
        mVertices.push_back( aiVector3D( 0.0, sizeEstimate, 0.0));
        mVertices.push_back( aiVector3D( sizeEstimate, 0.0, 0.0));
        mVertices.push_back( aiVector3D( 0.0, 0.0, -sizeEstimate));
        mVertices.push_back( aiVector3D( sizeEstimate, 0.0, 0.0));
        mVertices.push_back( aiVector3D( 0.0, -sizeEstimate, 0.0));
        mVertices.push_back( aiVector3D( 0.0, 0.0, -sizeEstimate));
        mVertices.push_back( aiVector3D( 0.0, -sizeEstimate, 0.0));
        mVertices.push_back( aiVector3D( -sizeEstimate, 0.0, 0.0));
        mVertices.push_back( aiVector3D( 0.0, 0.0, -sizeEstimate));

        mVertices.push_back( aiVector3D( -sizeEstimate, 0.0, 0.0));
        mVertices.push_back( aiVector3D( 0.0, 0.0, sizeEstimate));
        mVertices.push_back( aiVector3D( 0.0, sizeEstimate, 0.0));
        mVertices.push_back( aiVector3D( 0.0, sizeEstimate, 0.0));
        mVertices.push_back( aiVector3D( 0.0, 0.0, sizeEstimate));
        mVertices.push_back( aiVector3D( sizeEstimate, 0.0, 0.0));
        mVertices.push_back( aiVector3D( sizeEstimate, 0.0, 0.0));
        mVertices.push_back( aiVector3D( 0.0, 0.0, sizeEstimate));
        mVertices.push_back( aiVector3D( 0.0, -sizeEstimate, 0.0));
        mVertices.push_back( aiVector3D( 0.0, -sizeEstimate, 0.0));
        mVertices.push_back( aiVector3D( 0.0, 0.0, sizeEstimate));
        mVertices.push_back( aiVector3D( -sizeEstimate, 0.0, 0.0));

        mFaces.push_back( Face( vertexStartIndex + 0, vertexStartIndex + 1, vertexStartIndex + 2));
        mFaces.push_back( Face( vertexStartIndex + 3, vertexStartIndex + 4, vertexStartIndex + 5));
        mFaces.push_back( Face( vertexStartIndex + 6, vertexStartIndex + 7, vertexStartIndex + 8));
        mFaces.push_back( Face( vertexStartIndex + 9, vertexStartIndex + 10, vertexStartIndex + 11));
        mFaces.push_back( Face( vertexStartIndex + 12, vertexStartIndex + 13, vertexStartIndex + 14));
        mFaces.push_back( Face( vertexStartIndex + 15, vertexStartIndex + 16, vertexStartIndex + 17));
        mFaces.push_back( Face( vertexStartIndex + 18, vertexStartIndex + 19, vertexStartIndex + 20));
        mFaces.push_back( Face( vertexStartIndex + 21, vertexStartIndex + 22, vertexStartIndex + 23));
    }

    unsigned int numVertices = static_cast<unsigned int>(mVertices.size() - vertexStartIndex);
    if( numVertices > 0)
    {
        // create a bone affecting all the newly created vertices
        aiBone* bone = new aiBone;
        mBones.push_back( bone);
        bone->mName = pNode->mName;

        // calculate the bone offset matrix by concatenating the inverse transformations of all parents
        bone->mOffsetMatrix = aiMatrix4x4( pNode->mTransformation).Inverse();
        for( aiNode* parent = pNode->mParent; parent != NULL; parent = parent->mParent)
            bone->mOffsetMatrix = aiMatrix4x4( parent->mTransformation).Inverse() * bone->mOffsetMatrix;

        // add all the vertices to the bone's influences
        bone->mNumWeights = numVertices;
        bone->mWeights = new aiVertexWeight[numVertices];
        for( unsigned int a = 0; a < numVertices; a++)
            bone->mWeights[a] = aiVertexWeight( vertexStartIndex + a, 1.0);

        // HACK: (thom) transform all vertices to the bone's local space. Should be done before adding
        // them to the array, but I'm tired now and I'm annoyed.
        aiMatrix4x4 boneToMeshTransform = aiMatrix4x4( bone->mOffsetMatrix).Inverse();
        for( unsigned int a = vertexStartIndex; a < mVertices.size(); a++)
            mVertices[a] = boneToMeshTransform * mVertices[a];
    }

    // and finally recurse into the children list
    for( unsigned int a = 0; a < pNode->mNumChildren; a++)
        CreateGeometry( pNode->mChildren[a]);
}

// ------------------------------------------------------------------------------------------------
// Creates the mesh from the internally accumulated stuff and returns it.
aiMesh* SkeletonMeshBuilder::CreateMesh()
{
    aiMesh* mesh = new aiMesh();

    // add points
    mesh->mNumVertices = static_cast<unsigned int>(mVertices.size());
    mesh->mVertices = new aiVector3D[mesh->mNumVertices];
    std::copy( mVertices.begin(), mVertices.end(), mesh->mVertices);

    mesh->mNormals = new aiVector3D[mesh->mNumVertices];

    // add faces
    mesh->mNumFaces = static_cast<unsigned int>(mFaces.size());
    mesh->mFaces = new aiFace[mesh->mNumFaces];
    for( unsigned int a = 0; a < mesh->mNumFaces; a++)
    {
        const Face& inface = mFaces[a];
        aiFace& outface = mesh->mFaces[a];
        outface.mNumIndices = 3;
        outface.mIndices = new unsigned int[3];
        outface.mIndices[0] = inface.mIndices[0];
        outface.mIndices[1] = inface.mIndices[1];
        outface.mIndices[2] = inface.mIndices[2];

        // Compute per-face normals ... we don't want the bones to be smoothed ... they're built to visualize
        // the skeleton, so it's good if there's a visual difference to the rest of the geometry
        aiVector3D nor = ((mVertices[inface.mIndices[2]] - mVertices[inface.mIndices[0]]) ^
            (mVertices[inface.mIndices[1]] - mVertices[inface.mIndices[0]]));

        if (nor.Length() < 1e-5) /* ensure that FindInvalidData won't remove us ...*/
            nor = aiVector3D(1.0,0.0,0.0);

        for (unsigned int n = 0; n < 3; ++n)
            mesh->mNormals[inface.mIndices[n]] = nor;
    }

    // add the bones
    mesh->mNumBones = static_cast<unsigned int>(mBones.size());
    mesh->mBones = new aiBone*[mesh->mNumBones];
    std::copy( mBones.begin(), mBones.end(), mesh->mBones);

    // default
    mesh->mMaterialIndex = 0;

    return mesh;
}

// ------------------------------------------------------------------------------------------------
// Creates a dummy material and returns it.
aiMaterial* SkeletonMeshBuilder::CreateMaterial()
{
    aiMaterial* matHelper = new aiMaterial;

    // Name
    aiString matName( std::string( "SkeletonMaterial"));
    matHelper->AddProperty( &matName, AI_MATKEY_NAME);

    // Prevent backface culling
    const int no_cull = 1;
    matHelper->AddProperty(&no_cull,1,AI_MATKEY_TWOSIDED);

    return matHelper;
}
