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
#ifndef ASSIMP_BUILD_NO_GLOBALSCALE_PROCESS

#include "ScaleProcess.h"

#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace Assimp {

ScaleProcess::ScaleProcess()
: BaseProcess()
, mScale( AI_CONFIG_GLOBAL_SCALE_FACTOR_DEFAULT ) {
    // empty
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
    mScale = pImp->GetPropertyFloat( AI_CONFIG_GLOBAL_SCALE_FACTOR_KEY, 0 );
}

void ScaleProcess::Execute( aiScene* pScene ) {
    if ( nullptr == pScene ) {
        return;
    }

    if ( nullptr == pScene->mRootNode ) {
        return;
    }

    traverseNodes( pScene->mRootNode );
}

void ScaleProcess::traverseNodes( aiNode *node ) {
    applyScaling( node );
}

void ScaleProcess::applyScaling( aiNode *currentNode ) {
    if ( nullptr != currentNode ) {
        currentNode->mTransformation.a1 = currentNode->mTransformation.a1 * mScale;
        currentNode->mTransformation.b2 = currentNode->mTransformation.b2 * mScale;
        currentNode->mTransformation.c3 = currentNode->mTransformation.c3 * mScale;
    }
}

} // Namespace Assimp

#endif // !! ASSIMP_BUILD_NO_GLOBALSCALE_PROCESS
