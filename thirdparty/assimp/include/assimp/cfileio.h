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

/** @file cfileio.h
 *  @brief Defines generic C routines to access memory-mapped files
 */
#pragma once
#ifndef AI_FILEIO_H_INC
#define AI_FILEIO_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct aiFileIO;
struct aiFile;

// aiFile callbacks
typedef size_t          (*aiFileWriteProc) (C_STRUCT aiFile*,   const char*, size_t, size_t);
typedef size_t          (*aiFileReadProc)  (C_STRUCT aiFile*,   char*, size_t,size_t);
typedef size_t          (*aiFileTellProc)  (C_STRUCT aiFile*);
typedef void            (*aiFileFlushProc) (C_STRUCT aiFile*);
typedef C_ENUM aiReturn (*aiFileSeek)      (C_STRUCT aiFile*, size_t, C_ENUM aiOrigin);

// aiFileIO callbacks
typedef C_STRUCT aiFile* (*aiFileOpenProc)  (C_STRUCT aiFileIO*, const char*, const char*);
typedef void             (*aiFileCloseProc) (C_STRUCT aiFileIO*, C_STRUCT aiFile*);

// Represents user-defined data
typedef char* aiUserData;

// ----------------------------------------------------------------------------------
/** @brief C-API: File system callbacks
 *
 *  Provided are functions to open and close files. Supply a custom structure to
 *  the import function. If you don't, a default implementation is used. Use custom
 *  file systems to enable reading from other sources, such as ZIPs
 *  or memory locations. */
struct aiFileIO
{
    /** Function used to open a new file
     */
    aiFileOpenProc OpenProc;

    /** Function used to close an existing file
     */
    aiFileCloseProc CloseProc;

    /** User-defined, opaque data */
    aiUserData UserData;
};

// ----------------------------------------------------------------------------------
/** @brief C-API: File callbacks
 *
 *  Actually, it's a data structure to wrap a set of fXXXX (e.g fopen)
 *  replacement functions.
 *
 *  The default implementation of the functions utilizes the fXXX functions from
 *  the CRT. However, you can supply a custom implementation to Assimp by
 *  delivering a custom aiFileIO. Use this to enable reading from other sources,
 *  such as ZIP archives or memory locations. */
struct aiFile
{
    /** Callback to read from a file */
    aiFileReadProc ReadProc;

    /** Callback to write to a file */
    aiFileWriteProc WriteProc;

    /** Callback to retrieve the current position of
     *  the file cursor (ftell())
     */
    aiFileTellProc TellProc;

    /** Callback to retrieve the size of the file,
     *  in bytes
     */
    aiFileTellProc FileSizeProc;

    /** Callback to set the current position
     * of the file cursor (fseek())
     */
    aiFileSeek SeekProc;

    /** Callback to flush the file contents
     */
    aiFileFlushProc FlushProc;

    /** User-defined, opaque data
     */
    aiUserData UserData;
};

#ifdef __cplusplus
}
#endif
#endif // AI_FILEIO_H_INC
