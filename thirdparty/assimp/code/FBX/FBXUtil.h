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

/** @file  FBXUtil.h
 *  @brief FBX utility functions for internal use
 */
#ifndef INCLUDED_AI_FBX_UTIL_H
#define INCLUDED_AI_FBX_UTIL_H

#include "FBXCompileConfig.h"
#include "FBXTokenizer.h"
#include <stdint.h>

namespace Assimp {
namespace FBX {


namespace Util {


/** helper for std::for_each to delete all heap-allocated items in a container */
template<typename T>
struct delete_fun
{
    void operator()(const volatile T* del) {
        delete del;
    }
};

/** Get a string representation for a #TokenType. */
const char* TokenTypeString(TokenType t);



/** Format log/error messages using a given offset in the source binary file
 *
 *  @param prefix Message prefix to be preprended to the location info.
 *  @param text Message text
 *  @param line Line index, 1-based
 *  @param column Column index, 1-based
 *  @return A string of the following format: {prefix} (offset 0x{offset}) {text}*/
std::string AddOffset(const std::string& prefix, const std::string& text, size_t offset);


/** Format log/error messages using a given line location in the source file.
 *
 *  @param prefix Message prefix to be preprended to the location info.
 *  @param text Message text
 *  @param line Line index, 1-based
 *  @param column Column index, 1-based
 *  @return A string of the following format: {prefix} (line {line}, col {column}) {text}*/
std::string AddLineAndColumn(const std::string& prefix, const std::string& text, unsigned int line, unsigned int column);


/** Format log/error messages using a given cursor token.
 *
 *  @param prefix Message prefix to be preprended to the location info.
 *  @param text Message text
 *  @param tok Token where parsing/processing stopped
 *  @return A string of the following format: {prefix} ({token-type}, line {line}, col {column}) {text}*/
std::string AddTokenText(const std::string& prefix, const std::string& text, const Token* tok);

/** Decode a single Base64-encoded character.
*
*  @param ch Character to decode (from base64 to binary).
*  @return decoded byte value*/
uint8_t DecodeBase64(char ch);

/** Compute decoded size of a Base64-encoded string
*
*  @param in Characters to decode.
*  @param inLength Number of characters to decode.
*  @return size of the decoded data (number of bytes)*/
size_t ComputeDecodedSizeBase64(const char* in, size_t inLength);

/** Decode a Base64-encoded string
*
*  @param in Characters to decode.
*  @param inLength Number of characters to decode.
*  @param out Pointer where we will store the decoded data.
*  @param maxOutLength Size of output buffer.
*  @return size of the decoded data (number of bytes)*/
size_t DecodeBase64(const char* in, size_t inLength, uint8_t* out, size_t maxOutLength);

char EncodeBase64(char byte);

/** Encode bytes in base64-encoding
*
*  @param data Binary data to encode.
*  @param inLength Number of bytes to encode.
*  @return base64-encoded string*/
std::string EncodeBase64(const char* data, size_t length);

}
}
}

#endif // ! INCLUDED_AI_FBX_UTIL_H
