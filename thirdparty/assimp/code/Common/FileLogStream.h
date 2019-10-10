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
/** @file FileLofStream.h
*/
#ifndef ASSIMP_FILELOGSTREAM_H_INC
#define ASSIMP_FILELOGSTREAM_H_INC

#include <assimp/LogStream.hpp>
#include <assimp/IOStream.hpp>
#include <assimp/DefaultIOSystem.h>

namespace Assimp    {

// ----------------------------------------------------------------------------------
/** @class  FileLogStream
 *  @brief  Logstream to write into a file.
 */
class FileLogStream :
    public LogStream
{
public:
    FileLogStream( const char* file, IOSystem* io = NULL );
    ~FileLogStream();
    void write( const char* message );

private:
    IOStream *m_pStream;
};

// ----------------------------------------------------------------------------------
//  Constructor
inline FileLogStream::FileLogStream( const char* file, IOSystem* io ) :
    m_pStream(NULL)
{
    if ( !file || 0 == *file )
        return;

    // If no IOSystem is specified: take a default one
    if (!io)
    {
        DefaultIOSystem FileSystem;
        m_pStream = FileSystem.Open( file, "wt");
    }
    else m_pStream = io->Open( file, "wt" );
}

// ----------------------------------------------------------------------------------
//  Destructor
inline FileLogStream::~FileLogStream()
{
    // The virtual d'tor should destroy the underlying file
    delete m_pStream;
}

// ----------------------------------------------------------------------------------
//  Write method
inline void FileLogStream::write( const char* message )
{
    if (m_pStream != NULL)
    {
        m_pStream->Write(message, sizeof(char), ::strlen(message));
        m_pStream->Flush();
    }
}

// ----------------------------------------------------------------------------------
} // !Namespace Assimp

#endif // !! ASSIMP_FILELOGSTREAM_H_INC
