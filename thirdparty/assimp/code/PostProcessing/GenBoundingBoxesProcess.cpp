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

#ifndef ASSIMP_BUILD_NO_GENBOUNDINGBOXES_PROCESS

#include "PostProcessing/GenBoundingBoxesProcess.h"

#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace Assimp {

GenBoundingBoxesProcess::GenBoundingBoxesProcess()
: BaseProcess() {

}

GenBoundingBoxesProcess::~GenBoundingBoxesProcess() {
    // empty
}

bool GenBoundingBoxesProcess::IsActive(unsigned int pFlags) const {
    return 0 != ( pFlags & aiProcess_GenBoundingBoxes );
}

void checkMesh(aiMesh* mesh, aiVector3D& min, aiVector3D& max) {
    ai_assert(nullptr != mesh);

    if (0 == mesh->mNumVertices) {
        return;
    }

    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        const aiVector3D &pos = mesh->mVertices[i];
        if (pos.x < min.x) {
            min.x = pos.x;
        }
        if (pos.y < min.y) {
            min.y = pos.y;
        }
        if (pos.z < min.z) {
            min.z = pos.z;
        }

        if (pos.x > max.x) {
            max.x = pos.x;
        }
        if (pos.y > max.y) {
            max.y = pos.y;
        }
        if (pos.z > max.z) {
            max.z = pos.z;
        }
    }
}

void GenBoundingBoxesProcess::Execute(aiScene* pScene) {
    if (nullptr == pScene) {
        return;
    }

    for (unsigned int i = 0; i < pScene->mNumMeshes; ++i) {
        aiMesh* mesh = pScene->mMeshes[i];
        if (nullptr == mesh) {
            continue;
        }

        aiVector3D min(999999, 999999, 999999), max(-999999, -999999, -999999);
        checkMesh(mesh, min, max);
        mesh->mAABB.mMin = min;
        mesh->mAABB.mMax = max;
    }
}

} // Namespace Assimp

#endif // ASSIMP_BUILD_NO_GENBOUNDINGBOXES_PROCESS
