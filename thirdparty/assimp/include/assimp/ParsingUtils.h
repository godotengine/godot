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


/** @file ParsingUtils.h
 *  @brief Defines helper functions for text parsing
 */
#ifndef AI_PARSING_UTILS_H_INC
#define AI_PARSING_UTILS_H_INC

#include "StringComparison.h"
#include "StringUtils.h"
#include <assimp/defs.h>

namespace Assimp {

// NOTE: the functions below are mostly intended as replacement for
// std::upper, std::lower, std::isupper, std::islower, std::isspace.
// we don't bother of locales. We don't want them. We want reliable
// (i.e. identical) results across all locales.

// The functions below accept any character type, but know only
// about ASCII. However, UTF-32 is the only safe ASCII superset to
// use since it doesn't have multi-byte sequences.

static const unsigned int BufferSize = 4096;

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
char_t ToLower( char_t in ) {
    return (in >= (char_t)'A' && in <= (char_t)'Z') ? (char_t)(in+0x20) : in;
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
char_t ToUpper( char_t in) {
    return (in >= (char_t)'a' && in <= (char_t)'z') ? (char_t)(in-0x20) : in;
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool IsUpper( char_t in) {
    return (in >= (char_t)'A' && in <= (char_t)'Z');
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool IsLower( char_t in) {
    return (in >= (char_t)'a' && in <= (char_t)'z');
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool IsSpace( char_t in) {
    return (in == (char_t)' ' || in == (char_t)'\t');
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool IsLineEnd( char_t in) {
    return (in==(char_t)'\r'||in==(char_t)'\n'||in==(char_t)'\0'||in==(char_t)'\f');
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool IsSpaceOrNewLine( char_t in) {
    return IsSpace<char_t>(in) || IsLineEnd<char_t>(in);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool SkipSpaces( const char_t* in, const char_t** out) {
    while( *in == ( char_t )' ' || *in == ( char_t )'\t' ) {
        ++in;
    }
    *out = in;
    return !IsLineEnd<char_t>(*in);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool SkipSpaces( const char_t** inout) {
    return SkipSpaces<char_t>(*inout,inout);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool SkipLine( const char_t* in, const char_t** out) {
    while( *in != ( char_t )'\r' && *in != ( char_t )'\n' && *in != ( char_t )'\0' ) {
        ++in;
    }

    // files are opened in binary mode. Ergo there are both NL and CR
    while( *in == ( char_t )'\r' || *in == ( char_t )'\n' ) {
        ++in;
    }
    *out = in;
    return *in != (char_t)'\0';
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool SkipLine( const char_t** inout) {
    return SkipLine<char_t>(*inout,inout);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool SkipSpacesAndLineEnd( const char_t* in, const char_t** out) {
    while( *in == ( char_t )' ' || *in == ( char_t )'\t' || *in == ( char_t )'\r' || *in == ( char_t )'\n' ) {
        ++in;
    }
    *out = in;
    return *in != '\0';
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool SkipSpacesAndLineEnd( const char_t** inout) {
    return SkipSpacesAndLineEnd<char_t>(*inout,inout);
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool GetNextLine( const char_t*& buffer, char_t out[ BufferSize ] ) {
    if( ( char_t )'\0' == *buffer ) {
        return false;
    }

    char* _out = out;
    char* const end = _out + BufferSize;
    while( !IsLineEnd( *buffer ) && _out < end ) {
        *_out++ = *buffer++;
    }
    *_out = (char_t)'\0';

    while( IsLineEnd( *buffer ) && '\0' != *buffer ) {
        ++buffer;
    }

    return true;
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE bool IsNumeric( char_t in)
{
    return ( in >= '0' && in <= '9' ) || '-' == in || '+' == in;
}

// ---------------------------------------------------------------------------------
template <class char_t>
AI_FORCE_INLINE
bool TokenMatch(char_t*& in, const char* token, unsigned int len)
{
    if (!::strncmp(token,in,len) && IsSpaceOrNewLine(in[len])) {
        if (in[len] != '\0') {
            in += len+1;
        } else {
            // If EOF after the token make sure we don't go past end of buffer
            in += len;
        }
        return true;
    }

    return false;
}
// ---------------------------------------------------------------------------------
/** @brief Case-ignoring version of TokenMatch
 *  @param in Input
 *  @param token Token to check for
 *  @param len Number of characters to check
 */
AI_FORCE_INLINE
bool TokenMatchI(const char*& in, const char* token, unsigned int len) {
    if (!ASSIMP_strincmp(token,in,len) && IsSpaceOrNewLine(in[len])) {
        in += len+1;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------------
AI_FORCE_INLINE
void SkipToken(const char*& in) {
    SkipSpaces(&in);
    while ( !IsSpaceOrNewLine( *in ) ) {
        ++in;
    }
}

// ---------------------------------------------------------------------------------
AI_FORCE_INLINE
std::string GetNextToken(const char*& in) {
    SkipSpacesAndLineEnd(&in);
    const char* cur = in;
    while ( !IsSpaceOrNewLine( *in ) ) {
        ++in;
    }
    return std::string(cur,(size_t)(in-cur));
}

// ---------------------------------------------------------------------------------

} // ! namespace Assimp

#endif // ! AI_PARSING_UTILS_H_INC
