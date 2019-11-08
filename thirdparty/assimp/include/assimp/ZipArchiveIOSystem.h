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

/** @file ZipArchiveIOSystem.h
 *  @brief Implementation of IOSystem to read a ZIP file from another IOSystem
*/

#pragma once
#ifndef AI_ZIPARCHIVEIOSYSTEM_H_INC
#define AI_ZIPARCHIVEIOSYSTEM_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>

namespace Assimp {

class ZipArchiveIOSystem : public IOSystem {
public:
    //! Open a Zip using the proffered IOSystem
    ZipArchiveIOSystem(IOSystem* pIOHandler, const char *pFilename, const char* pMode = "r");
    ZipArchiveIOSystem(IOSystem* pIOHandler, const std::string& rFilename, const char* pMode = "r");
    virtual ~ZipArchiveIOSystem();
    bool Exists(const char* pFilename) const override;
    char getOsSeparator() const override;
    IOStream* Open(const char* pFilename, const char* pMode = "rb") override;
    void Close(IOStream* pFile) override;

    // Specific to ZIP
    //! The file was opened and is a ZIP
    bool isOpen() const;

    //! Get the list of all files with their simplified paths
    //! Intended for use within Assimp library boundaries
    void getFileList(std::vector<std::string>& rFileList) const;

    //! Get the list of all files with extension (must be lowercase)
    //! Intended for use within Assimp library boundaries
    void getFileListExtension(std::vector<std::string>& rFileList, const std::string& extension) const;

    static bool isZipArchive(IOSystem* pIOHandler, const char *pFilename);
    static bool isZipArchive(IOSystem* pIOHandler, const std::string& rFilename);

private:
    class Implement;
    Implement *pImpl = nullptr;
};

} // Namespace Assimp

#endif // AI_ZIPARCHIVEIOSYSTEM_H_INC
