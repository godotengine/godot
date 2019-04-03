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

} // !Util
} // !FBX
} // !Assimp

#endif
