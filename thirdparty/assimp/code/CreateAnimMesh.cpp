/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (C) 2016 The Qt Company Ltd.
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

#include <assimp/CreateAnimMesh.h>

namespace Assimp    {

aiAnimMesh *aiCreateAnimMesh(const aiMesh *mesh)
{
    aiAnimMesh *animesh = new aiAnimMesh;
    animesh->mVertices = NULL;
    animesh->mNormals = NULL;
    animesh->mTangents = NULL;
    animesh->mBitangents = NULL;
    animesh->mNumVertices = mesh->mNumVertices;
    if (mesh->mVertices) {
        animesh->mVertices = new aiVector3D[animesh->mNumVertices];
        std::memcpy(animesh->mVertices, mesh->mVertices, mesh->mNumVertices * sizeof(aiVector3D));
    }
    if (mesh->mNormals) {
        animesh->mNormals = new aiVector3D[animesh->mNumVertices];
        std::memcpy(animesh->mNormals, mesh->mNormals, mesh->mNumVertices * sizeof(aiVector3D));
    }
    if (mesh->mTangents) {
        animesh->mTangents = new aiVector3D[animesh->mNumVertices];
        std::memcpy(animesh->mTangents, mesh->mTangents, mesh->mNumVertices * sizeof(aiVector3D));
    }
    if (mesh->mBitangents) {
        animesh->mBitangents = new aiVector3D[animesh->mNumVertices];
        std::memcpy(animesh->mBitangents, mesh->mBitangents, mesh->mNumVertices * sizeof(aiVector3D));
    }

    for (int i = 0; i < AI_MAX_NUMBER_OF_COLOR_SETS; ++i) {
        if (mesh->mColors[i]) {
            animesh->mColors[i] = new aiColor4D[animesh->mNumVertices];
            std::memcpy(animesh->mColors[i], mesh->mColors[i], mesh->mNumVertices * sizeof(aiColor4D));
        } else {
            animesh->mColors[i] = NULL;
        }
    }

    for (int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++i) {
        if (mesh->mTextureCoords[i]) {
            animesh->mTextureCoords[i] = new aiVector3D[animesh->mNumVertices];
            std::memcpy(animesh->mTextureCoords[i], mesh->mTextureCoords[i], mesh->mNumVertices * sizeof(aiVector3D));
        } else {
            animesh->mTextureCoords[i] = NULL;
        }
    }
    return animesh;
}

} // end of namespace Assimp
