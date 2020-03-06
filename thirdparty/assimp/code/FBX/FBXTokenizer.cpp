/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


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

/** @file  FBXTokenizer.cpp
 *  @brief Implementation of the FBX broadphase lexer
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

// tab width for logging columns
#define ASSIMP_FBX_TAB_WIDTH 4

#include <assimp/ParsingUtils.h>

#include "FBXTokenizer.h"
#include "FBXUtil.h"
#include <assimp/Exceptional.h>

namespace Assimp {
namespace FBX {

// ------------------------------------------------------------------------------------------------
Token::Token(const char* sbegin, const char* send, TokenType type, unsigned int line, unsigned int column)
    :
#ifdef DEBUG
    contents(sbegin, static_cast<size_t>(send-sbegin)),
#endif
    sbegin(sbegin)
    , send(send)
    , type(type)
    , line(line)
    , column(column)
{
    ai_assert(sbegin);
    ai_assert(send);

    // tokens must be of non-zero length
    ai_assert(static_cast<size_t>(send-sbegin) > 0);
}

// ------------------------------------------------------------------------------------------------
Token::~Token()
{
}

namespace {

// ------------------------------------------------------------------------------------------------
// signal tokenization error, this is always unrecoverable. Throws DeadlyImportError.
AI_WONT_RETURN void TokenizeError(const std::string& message, unsigned int line, unsigned int column) AI_WONT_RETURN_SUFFIX;
AI_WONT_RETURN void TokenizeError(const std::string& message, unsigned int line, unsigned int column)
{
    throw DeadlyImportError(Util::AddLineAndColumn("FBX-Tokenize",message,line,column));
}


// process a potential data token up to 'cur', adding it to 'output_tokens'.
// ------------------------------------------------------------------------------------------------
void ProcessDataToken( TokenList& output_tokens, const char*& start, const char*& end,
                      unsigned int line,
                      unsigned int column,
                      TokenType type = TokenType_DATA,
                      bool must_have_token = false)
{
    if (start && end) {
        // sanity check:
        // tokens should have no whitespace outside quoted text and [start,end] should
        // properly delimit the valid range.
        bool in_double_quotes = false;
        for (const char* c = start; c != end + 1; ++c) {
            if (*c == '\"') {
                in_double_quotes = !in_double_quotes;
            }

            if (!in_double_quotes && IsSpaceOrNewLine(*c)) {
                TokenizeError("unexpected whitespace in token", line, column);
            }
        }

        if (in_double_quotes) {
            TokenizeError("non-terminated double quotes", line, column);
        }

        output_tokens.push_back(new_Token(start,end + 1,type,line,column));
    }
    else if (must_have_token) {
        TokenizeError("unexpected character, expected data token", line, column);
    }

    start = end = NULL;
}

}

// ------------------------------------------------------------------------------------------------
void Tokenize(TokenList& output_tokens, const char* input)
{
    ai_assert(input);

    // line and column numbers numbers are one-based
    unsigned int line = 1;
    unsigned int column = 1;

    bool comment = false;
    bool in_double_quotes = false;
    bool pending_data_token = false;

    const char* token_begin = NULL, *token_end = NULL;
    for (const char* cur = input;*cur;column += (*cur == '\t' ? ASSIMP_FBX_TAB_WIDTH : 1), ++cur) {
        const char c = *cur;

        if (IsLineEnd(c)) {
            comment = false;

            column = 0;
            ++line;
        }

        if(comment) {
            continue;
        }

        if(in_double_quotes) {
            if (c == '\"') {
                in_double_quotes = false;
                token_end = cur;

                ProcessDataToken(output_tokens,token_begin,token_end,line,column);
                pending_data_token = false;
            }
            continue;
        }

        switch(c)
        {
        case '\"':
            if (token_begin) {
                TokenizeError("unexpected double-quote", line, column);
            }
            token_begin = cur;
            in_double_quotes = true;
            continue;

        case ';':
            ProcessDataToken(output_tokens,token_begin,token_end,line,column);
            comment = true;
            continue;

        case '{':
            ProcessDataToken(output_tokens,token_begin,token_end, line, column);
            output_tokens.push_back(new_Token(cur,cur+1,TokenType_OPEN_BRACKET,line,column));
            continue;

        case '}':
            ProcessDataToken(output_tokens,token_begin,token_end,line,column);
            output_tokens.push_back(new_Token(cur,cur+1,TokenType_CLOSE_BRACKET,line,column));
            continue;

        case ',':
            if (pending_data_token) {
                ProcessDataToken(output_tokens,token_begin,token_end,line,column,TokenType_DATA,true);
            }
            output_tokens.push_back(new_Token(cur,cur+1,TokenType_COMMA,line,column));
            continue;

        case ':':
            if (pending_data_token) {
                ProcessDataToken(output_tokens,token_begin,token_end,line,column,TokenType_KEY,true);
            }
            else {
                TokenizeError("unexpected colon", line, column);
            }
            continue;
        }

        if (IsSpaceOrNewLine(c)) {

            if (token_begin) {
                // peek ahead and check if the next token is a colon in which
                // case this counts as KEY token.
                TokenType type = TokenType_DATA;
                for (const char* peek = cur;  *peek && IsSpaceOrNewLine(*peek); ++peek) {
                    if (*peek == ':') {
                        type = TokenType_KEY;
                        cur = peek;
                        break;
                    }
                }

                ProcessDataToken(output_tokens,token_begin,token_end,line,column,type);
            }

            pending_data_token = false;
        }
        else {
            token_end = cur;
            if (!token_begin) {
                token_begin = cur;
            }

            pending_data_token = true;
        }
    }
}

} // !FBX
} // !Assimp

#endif
