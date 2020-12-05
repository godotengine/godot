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

/** @file Default implementation of IOSystem using the standard C file functions */
#pragma once
#ifndef AI_DEFAULTIOSYSTEM_H_INC
#define AI_DEFAULTIOSYSTEM_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/IOSystem.hpp>

namespace Assimp    {

// ---------------------------------------------------------------------------
/** Default implementation of IOSystem using the standard C file functions */
class ASSIMP_API DefaultIOSystem : public IOSystem {
public:
    // -------------------------------------------------------------------
    /** Tests for the existence of a file at the given path. */
    bool Exists( const char* pFile) const;

    // -------------------------------------------------------------------
    /** Returns the directory separator. */
    char getOsSeparator() const;

    // -------------------------------------------------------------------
    /** Open a new file with a given path. */
    IOStream* Open( const char* pFile, const char* pMode = "rb");

    // -------------------------------------------------------------------
    /** Closes the given file and releases all resources associated with it. */
    void Close( IOStream* pFile);

    // -------------------------------------------------------------------
    /** Compare two paths */
    bool ComparePaths (const char* one, const char* second) const;

    /** @brief get the file name of a full filepath
     * example: /tmp/archive.tar.gz -> archive.tar.gz
     */
    static std::string fileName( const std::string &path );

    /** @brief get the complete base name of a full filepath
     * example: /tmp/archive.tar.gz -> archive.tar
     */
    static std::string completeBaseName( const std::string &path);

    /** @brief get the path of a full filepath
     * example: /tmp/archive.tar.gz -> /tmp/
     */
    static std::string absolutePath( const std::string &path);
};

} //!ns Assimp

#endif //AI_DEFAULTIOSYSTEM_H_INC
