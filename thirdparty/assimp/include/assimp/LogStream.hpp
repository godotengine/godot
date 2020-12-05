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

/** @file LogStream.hpp
 *  @brief Abstract base class 'LogStream', representing an output log stream.
 */
#ifndef INCLUDED_AI_LOGSTREAM_H
#define INCLUDED_AI_LOGSTREAM_H

#include "types.h"

namespace Assimp    {

class IOSystem;

// ------------------------------------------------------------------------------------
/** @brief CPP-API: Abstract interface for log stream implementations.
 *
 *  Several default implementations are provided, see #aiDefaultLogStream for more
 *  details. Writing your own implementation of LogStream is just necessary if these
 *  are not enough for your purpose. */
class ASSIMP_API LogStream
#ifndef SWIG
    : public Intern::AllocateFromAssimpHeap
#endif
{
protected:
    /** @brief  Default constructor */
    LogStream() AI_NO_EXCEPT;

public:
    /** @brief  Virtual destructor  */
    virtual ~LogStream();

    // -------------------------------------------------------------------
    /** @brief  Overwrite this for your own output methods
     *
     *  Log messages *may* consist of multiple lines and you shouldn't
     *  expect a consistent formatting. If you want custom formatting
     *  (e.g. generate HTML), supply a custom instance of Logger to
     *  #DefaultLogger:set(). Usually you can *expect* that a log message
     *  is exactly one line and terminated with a single \n character.
     *  @param message Message to be written */
    virtual void write(const char* message) = 0;

    // -------------------------------------------------------------------
    /** @brief Creates a default log stream
     *  @param streams Type of the default stream
     *  @param name For aiDefaultLogStream_FILE: name of the output file
     *  @param io For aiDefaultLogStream_FILE: IOSystem to be used to open the output
     *   file. Pass NULL for the default implementation.
     *  @return New LogStream instance.  */
    static LogStream* createDefaultStream(aiDefaultLogStream stream,
        const char* name = "AssimpLog.txt",
        IOSystem* io = nullptr );

}; // !class LogStream

inline
LogStream::LogStream() AI_NO_EXCEPT {
    // empty
}

inline
LogStream::~LogStream() {
    // empty
}

// ------------------------------------------------------------------------------------
} // Namespace Assimp

#endif
