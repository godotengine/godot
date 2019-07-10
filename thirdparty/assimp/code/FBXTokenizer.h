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

/** @file  FBXTokenizer.h
 *  @brief FBX lexer
 */
#ifndef INCLUDED_AI_FBX_TOKENIZER_H
#define INCLUDED_AI_FBX_TOKENIZER_H

#include "FBXCompileConfig.h"
#include <assimp/ai_assert.h>
#include <vector>
#include <string>

namespace Assimp {
namespace FBX {

/** Rough classification for text FBX tokens used for constructing the
 *  basic scope hierarchy. */
enum TokenType
{
    // {
    TokenType_OPEN_BRACKET = 0,

    // }
    TokenType_CLOSE_BRACKET,

    // '"blablubb"', '2', '*14' - very general token class,
    // further processing happens at a later stage.
    TokenType_DATA,

    //
    TokenType_BINARY_DATA,

    // ,
    TokenType_COMMA,

    // blubb:
    TokenType_KEY
};


/** Represents a single token in a FBX file. Tokens are
 *  classified by the #TokenType enumerated types.
 *
 *  Offers iterator protocol. Tokens are immutable. */
class Token
{
private:
    static const unsigned int BINARY_MARKER = static_cast<unsigned int>(-1);

public:
    /** construct a textual token */
    Token(const char* sbegin, const char* send, TokenType type, unsigned int line, unsigned int column);

    /** construct a binary token */
    Token(const char* sbegin, const char* send, TokenType type, unsigned int offset);

    ~Token();

public:
    std::string StringContents() const {
        return std::string(begin(),end());
    }

    bool IsBinary() const {
        return column == BINARY_MARKER;
    }

    const char* begin() const {
        return sbegin;
    }

    const char* end() const {
        return send;
    }

    TokenType Type() const {
        return type;
    }

    unsigned int Offset() const {
        ai_assert(IsBinary());
        return offset;
    }

    unsigned int Line() const {
        ai_assert(!IsBinary());
        return line;
    }

    unsigned int Column() const {
        ai_assert(!IsBinary());
        return column;
    }

private:

#ifdef DEBUG
    // full string copy for the sole purpose that it nicely appears
    // in msvc's debugger window.
    const std::string contents;
#endif


    const char* const sbegin;
    const char* const send;
    const TokenType type;

    union {
        const unsigned int line;
        unsigned int offset;
    };
    const unsigned int column;
};

// XXX should use C++11's unique_ptr - but assimp's need to keep working with 03
typedef const Token* TokenPtr;
typedef std::vector< TokenPtr > TokenList;

#define new_Token new Token


/** Main FBX tokenizer function. Transform input buffer into a list of preprocessed tokens.
 *
 *  Skips over comments and generates line and column numbers.
 *
 * @param output_tokens Receives a list of all tokens in the input data.
 * @param input_buffer Textual input buffer to be processed, 0-terminated.
 * @throw DeadlyImportError if something goes wrong */
void Tokenize(TokenList& output_tokens, const char* input);


/** Tokenizer function for binary FBX files.
 *
 *  Emits a token list suitable for direct parsing.
 *
 * @param output_tokens Receives a list of all tokens in the input data.
 * @param input_buffer Binary input buffer to be processed.
 * @param length Length of input buffer, in bytes. There is no 0-terminal.
 * @throw DeadlyImportError if something goes wrong */
void TokenizeBinary(TokenList& output_tokens, const char* input, unsigned int length);


} // ! FBX
} // ! Assimp

#endif // ! INCLUDED_AI_FBX_PARSER_H
