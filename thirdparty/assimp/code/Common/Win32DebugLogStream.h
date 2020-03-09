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

/** @file  Win32DebugLogStream.h
*  @brief Implementation of Win32DebugLogStream
*/
#ifndef AI_WIN32DEBUGLOGSTREAM_H_INC
#define AI_WIN32DEBUGLOGSTREAM_H_INC

#ifdef _WIN32

#include <assimp/LogStream.hpp>
#include "windows.h"

namespace Assimp    {

// ---------------------------------------------------------------------------
/** @class  Win32DebugLogStream
 *  @brief  Logs into the debug stream from win32.
 */
class Win32DebugLogStream : public LogStream {
public:
    /** @brief  Default constructor */
    Win32DebugLogStream();

    /** @brief  Destructor  */
    ~Win32DebugLogStream();

    /** @brief  Writer  */
    void write(const char* messgae);
};

// ---------------------------------------------------------------------------
inline 
Win32DebugLogStream::Win32DebugLogStream(){ 
    // empty
}

// ---------------------------------------------------------------------------
inline 
Win32DebugLogStream::~Win32DebugLogStream(){
    // empty
}

// ---------------------------------------------------------------------------
inline 
void Win32DebugLogStream::write(const char* message) {
    ::OutputDebugStringA( message);
}

// ---------------------------------------------------------------------------
}   // Namespace Assimp

#endif // ! _WIN32
#endif // guard
