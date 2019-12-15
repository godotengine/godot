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

/** @file MemoryIOWrapper.h
 *  Handy IOStream/IOSystem implemetation to read directly from a memory buffer */
#pragma once
#ifndef AI_MEMORYIOSTREAM_H_INC
#define AI_MEMORYIOSTREAM_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>
#include <assimp/ai_assert.h>

#include <stdint.h>

namespace Assimp    {
    
#define AI_MEMORYIO_MAGIC_FILENAME "$$$___magic___$$$"
#define AI_MEMORYIO_MAGIC_FILENAME_LENGTH 17

// ----------------------------------------------------------------------------------
/** Implementation of IOStream to read directly from a memory buffer */
// ----------------------------------------------------------------------------------
class MemoryIOStream : public IOStream {
public:
    MemoryIOStream (const uint8_t* buff, size_t len, bool own = false)
    : buffer (buff)
    , length(len)
    , pos((size_t)0)
    , own(own) {
        // empty
    }

    ~MemoryIOStream ()  {
        if(own) {
            delete[] buffer;
        }
    }

    // -------------------------------------------------------------------
    // Read from stream
    size_t Read(void* pvBuffer, size_t pSize, size_t pCount)    {
        ai_assert(nullptr != pvBuffer);
        ai_assert(0 != pSize);
        
        const size_t cnt = std::min( pCount, (length-pos) / pSize);
        const size_t ofs = pSize * cnt;

        ::memcpy(pvBuffer,buffer+pos,ofs);
        pos += ofs;

        return cnt;
    }

    // -------------------------------------------------------------------
    // Write to stream
    size_t Write(const void* /*pvBuffer*/, size_t /*pSize*/,size_t /*pCount*/)  {
        ai_assert(false); // won't be needed
        return 0;
    }

    // -------------------------------------------------------------------
    // Seek specific position
    aiReturn Seek(size_t pOffset, aiOrigin pOrigin) {
        if (aiOrigin_SET == pOrigin) {
            if (pOffset > length) {
                return AI_FAILURE;
            }
            pos = pOffset;
        } else if (aiOrigin_END == pOrigin) {
            if (pOffset > length) {
                return AI_FAILURE;
            }
            pos = length-pOffset;
        } else {
            if (pOffset+pos > length) {
                return AI_FAILURE;
            }
            pos += pOffset;
        }
        return AI_SUCCESS;
    }

    // -------------------------------------------------------------------
    // Get current seek position
    size_t Tell() const {
        return pos;
    }

    // -------------------------------------------------------------------
    // Get size of file
    size_t FileSize() const {
        return length;
    }

    // -------------------------------------------------------------------
    // Flush file contents
    void Flush() {
        ai_assert(false); // won't be needed
    }

private:
    const uint8_t* buffer;
    size_t length,pos;
    bool own;
};

// ---------------------------------------------------------------------------
/** Dummy IO system to read from a memory buffer */
class MemoryIOSystem : public IOSystem {
public:
    /** Constructor. */
    MemoryIOSystem(const uint8_t* buff, size_t len, IOSystem* io)
    : buffer(buff)
    , length(len)
    , existing_io(io)
    , created_streams() {
        // empty
    }

    /** Destructor. */
    ~MemoryIOSystem() {
    }

    // -------------------------------------------------------------------
    /** Tests for the existence of a file at the given path. */
    bool Exists(const char* pFile) const override {
        if (0 == strncmp( pFile, AI_MEMORYIO_MAGIC_FILENAME, AI_MEMORYIO_MAGIC_FILENAME_LENGTH ) ) {
            return true;
        }
        return existing_io ? existing_io->Exists(pFile) : false;
    }

    // -------------------------------------------------------------------
    /** Returns the directory separator. */
    char getOsSeparator() const override {
        return existing_io ? existing_io->getOsSeparator()
                           : '/';  // why not? it doesn't care
    }

    // -------------------------------------------------------------------
    /** Open a new file with a given path. */
    IOStream* Open(const char* pFile, const char* pMode = "rb") override {
        if ( 0 == strncmp( pFile, AI_MEMORYIO_MAGIC_FILENAME, AI_MEMORYIO_MAGIC_FILENAME_LENGTH ) ) {
            created_streams.emplace_back(new MemoryIOStream(buffer, length));
            return created_streams.back();
        }
        return existing_io ? existing_io->Open(pFile, pMode) : NULL;
    }

    // -------------------------------------------------------------------
    /** Closes the given file and releases all resources associated with it. */
    void Close( IOStream* pFile) override {
        auto it = std::find(created_streams.begin(), created_streams.end(), pFile);
        if (it != created_streams.end()) {
            delete pFile;
            created_streams.erase(it);
        } else if (existing_io) {
            existing_io->Close(pFile);
        }
    }

    // -------------------------------------------------------------------
    /** Compare two paths */
    bool ComparePaths(const char* one, const char* second) const override {
        return existing_io ? existing_io->ComparePaths(one, second) : false;
    }

    bool PushDirectory( const std::string &path ) override { 
        return existing_io ? existing_io->PushDirectory(path) : false;
    }

    const std::string &CurrentDirectory() const override {
        static std::string empty;
        return existing_io ? existing_io->CurrentDirectory() : empty;
    }

    size_t StackSize() const override {
        return existing_io ? existing_io->StackSize() : 0;
    }

    bool PopDirectory() override {
        return existing_io ? existing_io->PopDirectory() : false;
    }

    bool CreateDirectory( const std::string &path ) override {
        return existing_io ? existing_io->CreateDirectory(path) : false;
    }

    bool ChangeDirectory( const std::string &path ) override {
        return existing_io ? existing_io->ChangeDirectory(path) : false;
    }

    bool DeleteFile( const std::string &file ) override {
        return existing_io ? existing_io->DeleteFile(file) : false;
    }

private:
    const uint8_t* buffer;
    size_t length;
    IOSystem* existing_io;
    std::vector<IOStream*> created_streams;
};

} // end namespace Assimp

#endif
