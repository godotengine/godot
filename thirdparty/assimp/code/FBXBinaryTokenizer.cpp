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
/** @file  FBXBinaryTokenizer.cpp
 *  @brief Implementation of a fake lexer for binary fbx files -
 *    we emit tokens so the parser needs almost no special handling
 *    for binary files.
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

#include "FBXTokenizer.h"
#include "FBXUtil.h"
#include <assimp/defs.h>
#include <stdint.h>
#include <assimp/Exceptional.h>
#include <assimp/ByteSwapper.h>

namespace Assimp {
namespace FBX {

//enum Flag
//{
//   e_unknown_0 = 1 << 0,
//   e_unknown_1 = 1 << 1,
//   e_unknown_2 = 1 << 2,
//   e_unknown_3 = 1 << 3,
//   e_unknown_4 = 1 << 4,
//   e_unknown_5 = 1 << 5,
//   e_unknown_6 = 1 << 6,
//   e_unknown_7 = 1 << 7,
//   e_unknown_8 = 1 << 8,
//   e_unknown_9 = 1 << 9,
//   e_unknown_10 = 1 << 10,
//   e_unknown_11 = 1 << 11,
//   e_unknown_12 = 1 << 12,
//   e_unknown_13 = 1 << 13,
//   e_unknown_14 = 1 << 14,
//   e_unknown_15 = 1 << 15,
//   e_unknown_16 = 1 << 16,
//   e_unknown_17 = 1 << 17,
//   e_unknown_18 = 1 << 18,
//   e_unknown_19 = 1 << 19,
//   e_unknown_20 = 1 << 20,
//   e_unknown_21 = 1 << 21,
//   e_unknown_22 = 1 << 22,
//   e_unknown_23 = 1 << 23,
//   e_flag_field_size_64_bit = 1 << 24, // Not sure what is
//   e_unknown_25 = 1 << 25,
//   e_unknown_26 = 1 << 26,
//   e_unknown_27 = 1 << 27,
//   e_unknown_28 = 1 << 28,
//   e_unknown_29 = 1 << 29,
//   e_unknown_30 = 1 << 30,
//   e_unknown_31 = 1 << 31
//};
//
//bool check_flag(uint32_t flags, Flag to_check)
//{
//	return (flags & to_check) != 0;
//}
// ------------------------------------------------------------------------------------------------
Token::Token(const char* sbegin, const char* send, TokenType type, unsigned int offset)
    :
    #ifdef DEBUG
    contents(sbegin, static_cast<size_t>(send-sbegin)),
    #endif
    sbegin(sbegin)
    , send(send)
    , type(type)
    , line(offset)
    , column(BINARY_MARKER)
{
    ai_assert(sbegin);
    ai_assert(send);

    // binary tokens may have zero length because they are sometimes dummies
    // inserted by TokenizeBinary()
    ai_assert(send >= sbegin);
}


namespace {

// ------------------------------------------------------------------------------------------------
// signal tokenization error, this is always unrecoverable. Throws DeadlyImportError.
AI_WONT_RETURN void TokenizeError(const std::string& message, unsigned int offset) AI_WONT_RETURN_SUFFIX;
AI_WONT_RETURN void TokenizeError(const std::string& message, unsigned int offset)
{
    throw DeadlyImportError(Util::AddOffset("FBX-Tokenize",message,offset));
}


// ------------------------------------------------------------------------------------------------
uint32_t Offset(const char* begin, const char* cursor) {
    ai_assert(begin <= cursor);

    return static_cast<unsigned int>(cursor - begin);
}

// ------------------------------------------------------------------------------------------------
void TokenizeError(const std::string& message, const char* begin, const char* cursor) {
    TokenizeError(message, Offset(begin, cursor));
}

// ------------------------------------------------------------------------------------------------
uint32_t ReadWord(const char* input, const char*& cursor, const char* end) {
    const size_t k_to_read = sizeof( uint32_t );
    if(Offset(cursor, end) < k_to_read ) {
        TokenizeError("cannot ReadWord, out of bounds",input, cursor);
    }

    uint32_t word;
    ::memcpy(&word, cursor, 4);
    AI_SWAP4(word);

    cursor += k_to_read;

    return word;
}

// ------------------------------------------------------------------------------------------------
uint64_t ReadDoubleWord(const char* input, const char*& cursor, const char* end) {
    const size_t k_to_read = sizeof(uint64_t);
    if(Offset(cursor, end) < k_to_read) {
        TokenizeError("cannot ReadDoubleWord, out of bounds",input, cursor);
    }

    uint64_t dword /*= *reinterpret_cast<const uint64_t*>(cursor)*/;
    ::memcpy( &dword, cursor, sizeof( uint64_t ) );
    AI_SWAP8(dword);

    cursor += k_to_read;

    return dword;
}

// ------------------------------------------------------------------------------------------------
uint8_t ReadByte(const char* input, const char*& cursor, const char* end) {
    if(Offset(cursor, end) < sizeof( uint8_t ) ) {
        TokenizeError("cannot ReadByte, out of bounds",input, cursor);
    }

    uint8_t word;/* = *reinterpret_cast< const uint8_t* >( cursor )*/
    ::memcpy( &word, cursor, sizeof( uint8_t ) );
    ++cursor;

    return word;
}

// ------------------------------------------------------------------------------------------------
unsigned int ReadString(const char*& sbegin_out, const char*& send_out, const char* input,
        const char*& cursor, const char* end, bool long_length = false, bool allow_null = false) {
    const uint32_t len_len = long_length ? 4 : 1;
    if(Offset(cursor, end) < len_len) {
        TokenizeError("cannot ReadString, out of bounds reading length",input, cursor);
    }

    const uint32_t length = long_length ? ReadWord(input, cursor, end) : ReadByte(input, cursor, end);

    if (Offset(cursor, end) < length) {
        TokenizeError("cannot ReadString, length is out of bounds",input, cursor);
    }

    sbegin_out = cursor;
    cursor += length;

    send_out = cursor;

    if(!allow_null) {
        for (unsigned int i = 0; i < length; ++i) {
            if(sbegin_out[i] == '\0') {
                TokenizeError("failed ReadString, unexpected NUL character in string",input, cursor);
            }
        }
    }

    return length;
}

// ------------------------------------------------------------------------------------------------
void ReadData(const char*& sbegin_out, const char*& send_out, const char* input, const char*& cursor, const char* end) {
    if(Offset(cursor, end) < 1) {
        TokenizeError("cannot ReadData, out of bounds reading length",input, cursor);
    }

    const char type = *cursor;
    sbegin_out = cursor++;

    switch(type)
    {
        // 16 bit int
    case 'Y':
        cursor += 2;
        break;

        // 1 bit bool flag (yes/no)
    case 'C':
        cursor += 1;
        break;

        // 32 bit int
    case 'I':
        // <- fall through

        // float
    case 'F':
        cursor += 4;
        break;

        // double
    case 'D':
        cursor += 8;
        break;

        // 64 bit int
    case 'L':
        cursor += 8;
        break;

        // note: do not write cursor += ReadWord(...cursor) as this would be UB

        // raw binary data
    case 'R':
    {
        const uint32_t length = ReadWord(input, cursor, end);
        cursor += length;
        break;
    }

    case 'b':
        // TODO: what is the 'b' type code? Right now we just skip over it /
        // take the full range we could get
        cursor = end;
        break;

        // array of *
    case 'f':
    case 'd':
    case 'l':
    case 'i':
    case 'c':   {
        const uint32_t length = ReadWord(input, cursor, end);
        const uint32_t encoding = ReadWord(input, cursor, end);

        const uint32_t comp_len = ReadWord(input, cursor, end);

        // compute length based on type and check against the stored value
        if(encoding == 0) {
            uint32_t stride = 0;
            switch(type)
            {
            case 'f':
            case 'i':
                stride = 4;
                break;

            case 'd':
            case 'l':
                stride = 8;
                break;

            case 'c':
                stride = 1;
                break;

            default:
                ai_assert(false);
            };
            ai_assert(stride > 0);
            if(length * stride != comp_len) {
                TokenizeError("cannot ReadData, calculated data stride differs from what the file claims",input, cursor);
            }
        }
        // zip/deflate algorithm (encoding==1)? take given length. anything else? die
        else if (encoding != 1) {
            TokenizeError("cannot ReadData, unknown encoding",input, cursor);
        }
        cursor += comp_len;
        break;
    }

        // string
    case 'S': {
        const char* sb, *se;
        // 0 characters can legally happen in such strings
        ReadString(sb, se, input, cursor, end, true, true);
        break;
    }
    default:
        TokenizeError("cannot ReadData, unexpected type code: " + std::string(&type, 1),input, cursor);
    }

    if(cursor > end) {
        TokenizeError("cannot ReadData, the remaining size is too small for the data type: " + std::string(&type, 1),input, cursor);
    }

    // the type code is contained in the returned range
    send_out = cursor;
}


// ------------------------------------------------------------------------------------------------
bool ReadScope(TokenList& output_tokens, const char* input, const char*& cursor, const char* end, bool const is64bits)
{
    // the first word contains the offset at which this block ends
	const uint64_t end_offset = is64bits ? ReadDoubleWord(input, cursor, end) : ReadWord(input, cursor, end);

    // we may get 0 if reading reached the end of the file -
    // fbx files have a mysterious extra footer which I don't know
    // how to extract any information from, but at least it always
    // starts with a 0.
    if(!end_offset) {
        return false;
    }

    if(end_offset > Offset(input, end)) {
        TokenizeError("block offset is out of range",input, cursor);
    }
    else if(end_offset < Offset(input, cursor)) {
        TokenizeError("block offset is negative out of range",input, cursor);
    }

    // the second data word contains the number of properties in the scope
	const uint64_t prop_count = is64bits ? ReadDoubleWord(input, cursor, end) : ReadWord(input, cursor, end);

    // the third data word contains the length of the property list
	const uint64_t prop_length = is64bits ? ReadDoubleWord(input, cursor, end) : ReadWord(input, cursor, end);

    // now comes the name of the scope/key
    const char* sbeg, *send;
    ReadString(sbeg, send, input, cursor, end);

    output_tokens.push_back(new_Token(sbeg, send, TokenType_KEY, Offset(input, cursor) ));

    // now come the individual properties
    const char* begin_cursor = cursor;
    for (unsigned int i = 0; i < prop_count; ++i) {
        ReadData(sbeg, send, input, cursor, begin_cursor + prop_length);

        output_tokens.push_back(new_Token(sbeg, send, TokenType_DATA, Offset(input, cursor) ));

        if(i != prop_count-1) {
            output_tokens.push_back(new_Token(cursor, cursor + 1, TokenType_COMMA, Offset(input, cursor) ));
        }
    }

    if (Offset(begin_cursor, cursor) != prop_length) {
        TokenizeError("property length not reached, something is wrong",input, cursor);
    }

    // at the end of each nested block, there is a NUL record to indicate
    // that the sub-scope exists (i.e. to distinguish between P: and P : {})
    // this NUL record is 13 bytes long on 32 bit version and 25 bytes long on 64 bit.
	const size_t sentinel_block_length = is64bits ? (sizeof(uint64_t)* 3 + 1) : (sizeof(uint32_t)* 3 + 1);

    if (Offset(input, cursor) < end_offset) {
        if (end_offset - Offset(input, cursor) < sentinel_block_length) {
            TokenizeError("insufficient padding bytes at block end",input, cursor);
        }

        output_tokens.push_back(new_Token(cursor, cursor + 1, TokenType_OPEN_BRACKET, Offset(input, cursor) ));

        // XXX this is vulnerable to stack overflowing ..
        while(Offset(input, cursor) < end_offset - sentinel_block_length) {
			ReadScope(output_tokens, input, cursor, input + end_offset - sentinel_block_length, is64bits);
        }
        output_tokens.push_back(new_Token(cursor, cursor + 1, TokenType_CLOSE_BRACKET, Offset(input, cursor) ));

        for (unsigned int i = 0; i < sentinel_block_length; ++i) {
            if(cursor[i] != '\0') {
                TokenizeError("failed to read nested block sentinel, expected all bytes to be 0",input, cursor);
            }
        }
        cursor += sentinel_block_length;
    }

    if (Offset(input, cursor) != end_offset) {
        TokenizeError("scope length not reached, something is wrong",input, cursor);
    }

    return true;
}

} // anonymous namespace

// ------------------------------------------------------------------------------------------------
// TODO: Test FBX Binary files newer than the 7500 version to check if the 64 bits address behaviour is consistent
void TokenizeBinary(TokenList& output_tokens, const char* input, unsigned int length)
{
    ai_assert(input);

    if(length < 0x1b) {
        TokenizeError("file is too short",0);
    }

    //uint32_t offset = 0x15;
/*    const char* cursor = input + 0x15;

    const uint32_t flags = ReadWord(input, cursor, input + length);

    const uint8_t padding_0 = ReadByte(input, cursor, input + length); // unused
    const uint8_t padding_1 = ReadByte(input, cursor, input + length); // unused*/

    if (strncmp(input,"Kaydara FBX Binary",18)) {
        TokenizeError("magic bytes not found",0);
    }

    const char* cursor = input + 18;
	/*Result ignored*/ ReadByte(input, cursor, input + length);
	/*Result ignored*/ ReadByte(input, cursor, input + length);
	/*Result ignored*/ ReadByte(input, cursor, input + length);
	/*Result ignored*/ ReadByte(input, cursor, input + length);
	/*Result ignored*/ ReadByte(input, cursor, input + length);
	const uint32_t version = ReadWord(input, cursor, input + length);
	const bool is64bits = version >= 7500;
    const char *end = input + length;
    while (cursor < end ) {
		if (!ReadScope(output_tokens, input, cursor, input + length, is64bits)) {
            break;
        }
    }
}

} // !FBX
} // !Assimp

#endif
