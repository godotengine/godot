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

/** @file  FBXUtil.cpp
 *  @brief Implementation of internal FBX utility functions
 */

#include "FBXUtil.h"
#include "FBXTokenizer.h"

#include <assimp/TinyFormatter.h>
#include <string>
#include <cstring>

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

namespace Assimp {
namespace FBX {
namespace Util {

// ------------------------------------------------------------------------------------------------
const char* TokenTypeString(TokenType t)
{
    switch(t) {
        case TokenType_OPEN_BRACKET:
            return "TOK_OPEN_BRACKET";

        case TokenType_CLOSE_BRACKET:
            return "TOK_CLOSE_BRACKET";

        case TokenType_DATA:
            return "TOK_DATA";

        case TokenType_COMMA:
            return "TOK_COMMA";

        case TokenType_KEY:
            return "TOK_KEY";

        case TokenType_BINARY_DATA:
            return "TOK_BINARY_DATA";
    }

    ai_assert(false);
    return "";
}


// ------------------------------------------------------------------------------------------------
std::string AddOffset(const std::string& prefix, const std::string& text, unsigned int offset)
{
    return static_cast<std::string>( (Formatter::format() << prefix << " (offset 0x" << std::hex << offset << ") " << text) );
}

// ------------------------------------------------------------------------------------------------
std::string AddLineAndColumn(const std::string& prefix, const std::string& text, unsigned int line, unsigned int column)
{
    return static_cast<std::string>( (Formatter::format() << prefix << " (line " << line << " <<  col " << column << ") " << text) );
}

// ------------------------------------------------------------------------------------------------
std::string AddTokenText(const std::string& prefix, const std::string& text, const Token* tok)
{
    if(tok->IsBinary()) {
        return static_cast<std::string>( (Formatter::format() << prefix <<
            " (" << TokenTypeString(tok->Type()) <<
            ", offset 0x" << std::hex << tok->Offset() << ") " <<
            text) );
    }

    return static_cast<std::string>( (Formatter::format() << prefix <<
        " (" << TokenTypeString(tok->Type()) <<
        ", line " << tok->Line() <<
        ", col " << tok->Column() << ") " <<
        text) );
}

static const uint8_t base64DecodeTable[128] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 62,  0,  0,  0, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  0,  0,  0, 64,  0,  0,
    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  0,  0,  0,  0,  0,
    0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,  0,  0,  0,  0,  0
};

uint8_t DecodeBase64(char ch)
{
    return base64DecodeTable[size_t(ch)];
}

size_t DecodeBase64(const char* in, size_t inLength, uint8_t*& out)
{
    if (inLength < 4) {
        out = 0;
        return 0;
    }

    const size_t outLength = (inLength * 3) / 4;
    out = new uint8_t[outLength];
    memset(out, 0, outLength);

    size_t i = 0;
    size_t j = 0;
    for (i = 0; i < inLength - 4; i += 4)
    {
        uint8_t b0 = Util::DecodeBase64(in[i]);
        uint8_t b1 = Util::DecodeBase64(in[i + 1]);
        uint8_t b2 = Util::DecodeBase64(in[i + 2]);
        uint8_t b3 = Util::DecodeBase64(in[i + 3]);

        out[j++] = (uint8_t)((b0 << 2) | (b1 >> 4));
        out[j++] = (uint8_t)((b1 << 4) | (b2 >> 2));
        out[j++] = (uint8_t)((b2 << 6) | b3);
    }
    return outLength;
}

} // !Util
} // !FBX
} // !Assimp

#endif
