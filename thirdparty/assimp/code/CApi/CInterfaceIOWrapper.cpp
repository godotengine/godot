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

/** @file aiFileIO -> IOSystem wrapper*/

#include "CInterfaceIOWrapper.h"

namespace Assimp    {

CIOStreamWrapper::~CIOStreamWrapper(void)
{
    /* Various places depend on this destructor to close the file */
    if (mFile) {
        mIO->mFileSystem->CloseProc(mIO->mFileSystem, mFile);
        mFile = nullptr;
    }
}

// ...................................................................
size_t CIOStreamWrapper::Read(void* pvBuffer,
    size_t pSize,
    size_t pCount
){
    // need to typecast here as C has no void*
    return mFile->ReadProc(mFile,(char*)pvBuffer,pSize,pCount);
}

// ...................................................................
size_t CIOStreamWrapper::Write(const void* pvBuffer,
    size_t pSize,
    size_t pCount
){
    // need to typecast here as C has no void*
    return mFile->WriteProc(mFile,(const char*)pvBuffer,pSize,pCount);
}

// ...................................................................
aiReturn CIOStreamWrapper::Seek(size_t pOffset,
    aiOrigin pOrigin
){
    return mFile->SeekProc(mFile,pOffset,pOrigin);
}

// ...................................................................
size_t CIOStreamWrapper::Tell(void) const {
    return mFile->TellProc(mFile);
}

// ...................................................................
size_t CIOStreamWrapper::FileSize() const {
    return mFile->FileSizeProc(mFile);
}

// ...................................................................
void CIOStreamWrapper::Flush () {
    return mFile->FlushProc(mFile);
}

// ------------------------------------------------------------------------------------------------
// Custom IOStream implementation for the C-API
bool CIOSystemWrapper::Exists( const char* pFile) const {
    aiFile* p = mFileSystem->OpenProc(mFileSystem,pFile,"rb");
    if (p){
        mFileSystem->CloseProc(mFileSystem,p);
        return true;
    }
    return false;
}

// ...................................................................
char CIOSystemWrapper::getOsSeparator() const {
#ifndef _WIN32
    return '/';
#else
    return '\\';
#endif
}

// ...................................................................
IOStream* CIOSystemWrapper::Open(const char* pFile,const char* pMode) {
    aiFile* p = mFileSystem->OpenProc(mFileSystem,pFile,pMode);
    if (!p) {
        return NULL;
    }
    return new CIOStreamWrapper(p, this);
}

// ...................................................................
void CIOSystemWrapper::Close( IOStream* pFile) {
    if (!pFile) {
        return;
    }
    delete pFile;
}

}
