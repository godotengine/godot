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

/** @file AssimpCExport.cpp
Assimp C export interface. See Exporter.cpp for some notes.
*/

#ifndef ASSIMP_BUILD_NO_EXPORT

#include "CInterfaceIOWrapper.h"
#include <assimp/SceneCombiner.h>
#include "Common/ScenePrivate.h"
#include <assimp/Exporter.hpp>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
ASSIMP_API size_t aiGetExportFormatCount(void)
{
    return Exporter().GetExportFormatCount();
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API const aiExportFormatDesc* aiGetExportFormatDescription( size_t index)
{
    // Note: this is valid as the index always pertains to a built-in exporter,
    // for which the returned structure is guaranteed to be of static storage duration.
    Exporter exporter;
    const aiExportFormatDesc* orig( exporter.GetExportFormatDescription( index ) );
    if (NULL == orig) {
        return NULL;
    }

    aiExportFormatDesc *desc = new aiExportFormatDesc;
    desc->description = new char[ strlen( orig->description ) + 1 ]();
    ::strncpy( (char*) desc->description, orig->description, strlen( orig->description ) );
    desc->fileExtension = new char[ strlen( orig->fileExtension ) + 1 ]();
    ::strncpy( ( char* ) desc->fileExtension, orig->fileExtension, strlen( orig->fileExtension ) );
    desc->id = new char[ strlen( orig->id ) + 1 ]();
    ::strncpy( ( char* ) desc->id, orig->id, strlen( orig->id ) );

    return desc;
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiReleaseExportFormatDescription( const aiExportFormatDesc *desc ) {
    if (NULL == desc) {
        return;
    }

    delete [] desc->description;
    delete [] desc->fileExtension;
    delete [] desc->id;
    delete desc;
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiCopyScene(const aiScene* pIn, aiScene** pOut)
{
    if (!pOut || !pIn) {
        return;
    }

    SceneCombiner::CopyScene(pOut,pIn,true);
    ScenePriv(*pOut)->mIsCopy = true;
}


// ------------------------------------------------------------------------------------------------
ASSIMP_API void aiFreeScene(const C_STRUCT aiScene* pIn)
{
    // note: aiReleaseImport() is also able to delete scene copies, but in addition
    // it also handles scenes with import metadata.
    delete pIn;
}


// ------------------------------------------------------------------------------------------------
ASSIMP_API aiReturn aiExportScene( const aiScene* pScene, const char* pFormatId, const char* pFileName, unsigned int pPreprocessing )
{
    return ::aiExportSceneEx(pScene,pFormatId,pFileName,NULL,pPreprocessing);
}


// ------------------------------------------------------------------------------------------------
ASSIMP_API aiReturn aiExportSceneEx( const aiScene* pScene, const char* pFormatId, const char* pFileName, aiFileIO* pIO, unsigned int pPreprocessing )
{
    Exporter exp;

    if (pIO) {
        exp.SetIOHandler(new CIOSystemWrapper(pIO));
    }
    return exp.Export(pScene,pFormatId,pFileName,pPreprocessing);
}


// ------------------------------------------------------------------------------------------------
ASSIMP_API const C_STRUCT aiExportDataBlob* aiExportSceneToBlob( const aiScene* pScene, const char* pFormatId, unsigned int pPreprocessing  )
{
    Exporter exp;
    if (!exp.ExportToBlob(pScene,pFormatId,pPreprocessing)) {
        return NULL;
    }
    const aiExportDataBlob* blob = exp.GetOrphanedBlob();
    ai_assert(blob);

    return blob;
}

// ------------------------------------------------------------------------------------------------
ASSIMP_API C_STRUCT void aiReleaseExportBlob( const aiExportDataBlob* pData )
{
    delete pData;
}

#endif // !ASSIMP_BUILD_NO_EXPORT
