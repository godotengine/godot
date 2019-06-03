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
/** @file IOStream.hpp
 *  @brief File I/O wrappers for C++.
 */

#pragma once
#ifndef AI_IOSTREAM_H_INC
#define AI_IOSTREAM_H_INC

#include "types.h"

#ifndef __cplusplus
#   error This header requires C++ to be used. aiFileIO.h is the \
    corresponding C interface.
#endif

namespace Assimp    {

// ----------------------------------------------------------------------------------
/** @brief CPP-API: Class to handle file I/O for C++
 *
 *  Derive an own implementation from this interface to provide custom IO handling
 *  to the Importer. If you implement this interface, be sure to also provide an
 *  implementation for IOSystem that creates instances of your custom IO class.
*/
class ASSIMP_API IOStream
#ifndef SWIG
    : public Intern::AllocateFromAssimpHeap
#endif
{
protected:
    /** Constructor protected, use IOSystem::Open() to create an instance. */
    IOStream() AI_NO_EXCEPT;

public:
    // -------------------------------------------------------------------
    /** @brief Destructor. Deleting the object closes the underlying file,
     * alternatively you may use IOSystem::Close() to release the file.
     */
    virtual ~IOStream();

    // -------------------------------------------------------------------
    /** @brief Read from the file
     *
     * See fread() for more details
     * This fails for write-only files */
    virtual size_t Read(void* pvBuffer,
        size_t pSize,
        size_t pCount) = 0;

    // -------------------------------------------------------------------
    /** @brief Write to the file
    *
    * See fwrite() for more details
    * This fails for read-only files */
    virtual size_t Write(const void* pvBuffer,
        size_t pSize,
        size_t pCount) = 0;

    // -------------------------------------------------------------------
    /** @brief Set the read/write cursor of the file
     *
     * Note that the offset is _negative_ for aiOrigin_END.
     * See fseek() for more details */
    virtual aiReturn Seek(size_t pOffset,
        aiOrigin pOrigin) = 0;

    // -------------------------------------------------------------------
    /** @brief Get the current position of the read/write cursor
     *
     * See ftell() for more details */
    virtual size_t Tell() const = 0;

    // -------------------------------------------------------------------
    /** @brief Returns filesize
     *  Returns the filesize. */
    virtual size_t FileSize() const = 0;

    // -------------------------------------------------------------------
    /** @brief Flush the contents of the file buffer (for writers)
     *  See fflush() for more details.
     */
    virtual void Flush() = 0;
}; //! class IOStream

// ----------------------------------------------------------------------------------
inline
IOStream::IOStream() AI_NO_EXCEPT {
    // empty
}

// ----------------------------------------------------------------------------------
inline
IOStream::~IOStream() {
    // empty
}
// ----------------------------------------------------------------------------------

} //!namespace Assimp

#endif //!!AI_IOSTREAM_H_INC
