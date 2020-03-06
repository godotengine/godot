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

/** @file  FBXParser.cpp
 *  @brief Implementation of the FBX parser and the rudimentary DOM that we use
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

#ifdef ASSIMP_BUILD_NO_OWN_ZLIB
#   include <zlib.h>
#else
#   include "../contrib/zlib/zlib.h"
#endif

#include "FBXTokenizer.h"
#include "FBXParser.h"
#include "FBXUtil.h"

#include <assimp/ParsingUtils.h>
#include <assimp/fast_atof.h>
#include <assimp/ByteSwapper.h>

#include <iostream>

using namespace Assimp;
using namespace Assimp::FBX;

namespace {

    // ------------------------------------------------------------------------------------------------
    // signal parse error, this is always unrecoverable. Throws DeadlyImportError.
    AI_WONT_RETURN void ParseError(const std::string& message, const Token& token) AI_WONT_RETURN_SUFFIX;
    AI_WONT_RETURN void ParseError(const std::string& message, const Token& token)
    {
        throw DeadlyImportError(Util::AddTokenText("FBX-Parser",message,&token));
    }

    // ------------------------------------------------------------------------------------------------
    AI_WONT_RETURN void ParseError(const std::string& message, const Element* element = NULL) AI_WONT_RETURN_SUFFIX;
    AI_WONT_RETURN void ParseError(const std::string& message, const Element* element)
    {
        if(element) {
            ParseError(message,element->KeyToken());
        }
        throw DeadlyImportError("FBX-Parser " + message);
    }


    // ------------------------------------------------------------------------------------------------
    void ParseError(const std::string& message, TokenPtr token)
    {
        if(token) {
            ParseError(message, *token);
        }
        ParseError(message);
    }

    // Initially, we did reinterpret_cast, breaking strict aliasing rules.
    // This actually caused trouble on Android, so let's be safe this time.
    // https://github.com/assimp/assimp/issues/24
    template <typename T>
    T SafeParse(const char* data, const char* end) {
        // Actual size validation happens during Tokenization so
        // this is valid as an assertion.
        (void)(end);
        ai_assert(static_cast<size_t>(end - data) >= sizeof(T));
        T result = static_cast<T>(0);
        ::memcpy(&result, data, sizeof(T));
        return result;
    }
}

namespace Assimp {
namespace FBX {

// ------------------------------------------------------------------------------------------------
Element::Element(const Token& key_token, Parser& parser)
: key_token(key_token)
{
    TokenPtr n = nullptr;
    do {
        n = parser.AdvanceToNextToken();
        if(!n) {
            ParseError("unexpected end of file, expected closing bracket",parser.LastToken());
        }

        if (n->Type() == TokenType_DATA) {
            tokens.push_back(n);
			TokenPtr prev = n;
            n = parser.AdvanceToNextToken();
            if(!n) {
                ParseError("unexpected end of file, expected bracket, comma or key",parser.LastToken());
            }

			const TokenType ty = n->Type();

			// some exporters are missing a comma on the next line
			if (ty == TokenType_DATA && prev->Type() == TokenType_DATA && (n->Line() == prev->Line() + 1)) {
				tokens.push_back(n);
				continue;
			}

            if (ty != TokenType_OPEN_BRACKET && ty != TokenType_CLOSE_BRACKET && ty != TokenType_COMMA && ty != TokenType_KEY) {
                ParseError("unexpected token; expected bracket, comma or key",n);
            }
        }

        if (n->Type() == TokenType_OPEN_BRACKET) {
            compound.reset(new Scope(parser));

            // current token should be a TOK_CLOSE_BRACKET
            n = parser.CurrentToken();
            ai_assert(n);

            if (n->Type() != TokenType_CLOSE_BRACKET) {
                ParseError("expected closing bracket",n);
            }

            parser.AdvanceToNextToken();
            return;
        }
    }
    while(n->Type() != TokenType_KEY && n->Type() != TokenType_CLOSE_BRACKET);
}

// ------------------------------------------------------------------------------------------------
Element::~Element()
{
     // no need to delete tokens, they are owned by the parser
}

// ------------------------------------------------------------------------------------------------
Scope::Scope(Parser& parser,bool topLevel)
{
    if(!topLevel) {
        TokenPtr t = parser.CurrentToken();
        if (t->Type() != TokenType_OPEN_BRACKET) {
            ParseError("expected open bracket",t);
        }
    }

    TokenPtr n = parser.AdvanceToNextToken();
    if(n == NULL) {
        ParseError("unexpected end of file");
    }

    // note: empty scopes are allowed
    while(n->Type() != TokenType_CLOSE_BRACKET) {
        if (n->Type() != TokenType_KEY) {
            ParseError("unexpected token, expected TOK_KEY",n);
        }

        const std::string& str = n->StringContents();
        elements.insert(ElementMap::value_type(str,new_Element(*n,parser)));

        // Element() should stop at the next Key token (or right after a Close token)
        n = parser.CurrentToken();
        if(n == NULL) {
            if (topLevel) {
                return;
            }
            ParseError("unexpected end of file",parser.LastToken());
        }
    }
}

// ------------------------------------------------------------------------------------------------
Scope::~Scope()
{
    for(ElementMap::value_type& v : elements) {
        delete v.second;
    }
}

// ------------------------------------------------------------------------------------------------
Parser::Parser (const TokenList& tokens, bool is_binary)
: tokens(tokens)
, last()
, current()
, cursor(tokens.begin())
, is_binary(is_binary)
{
    root.reset(new Scope(*this,true));
}

// ------------------------------------------------------------------------------------------------
Parser::~Parser()
{
    // empty
}

// ------------------------------------------------------------------------------------------------
TokenPtr Parser::AdvanceToNextToken()
{
    last = current;
    if (cursor == tokens.end()) {
        current = NULL;
    } else {
        current = *cursor++;
    }
    return current;
}

// ------------------------------------------------------------------------------------------------
TokenPtr Parser::CurrentToken() const
{
    return current;
}

// ------------------------------------------------------------------------------------------------
TokenPtr Parser::LastToken() const
{
    return last;
}

// ------------------------------------------------------------------------------------------------
uint64_t ParseTokenAsID(const Token& t, const char*& err_out)
{
    err_out = NULL;

    if (t.Type() != TokenType_DATA) {
        err_out = "expected TOK_DATA token";
        return 0L;
    }

    if(t.IsBinary())
    {
        const char* data = t.begin();
        if (data[0] != 'L') {
            err_out = "failed to parse ID, unexpected data type, expected L(ong) (binary)";
            return 0L;
        }

        BE_NCONST uint64_t id = SafeParse<uint64_t>(data+1, t.end());
        AI_SWAP8(id);
        return id;
    }

    // XXX: should use size_t here
    unsigned int length = static_cast<unsigned int>(t.end() - t.begin());
    ai_assert(length > 0);

    const char* out = nullptr;
    const uint64_t id = strtoul10_64(t.begin(),&out,&length);
    if (out > t.end()) {
        err_out = "failed to parse ID (text)";
        return 0L;
    }

    return id;
}

// ------------------------------------------------------------------------------------------------
size_t ParseTokenAsDim(const Token& t, const char*& err_out)
{
    // same as ID parsing, except there is a trailing asterisk
    err_out = NULL;

    if (t.Type() != TokenType_DATA) {
        err_out = "expected TOK_DATA token";
        return 0;
    }

    if(t.IsBinary())
    {
        const char* data = t.begin();
        if (data[0] != 'L') {
            err_out = "failed to parse ID, unexpected data type, expected L(ong) (binary)";
            return 0;
        }

        BE_NCONST uint64_t id = SafeParse<uint64_t>(data+1, t.end());
        AI_SWAP8(id);
        return static_cast<size_t>(id);
    }

    if(*t.begin() != '*') {
        err_out = "expected asterisk before array dimension";
        return 0;
    }

    // XXX: should use size_t here
    unsigned int length = static_cast<unsigned int>(t.end() - t.begin());
    if(length == 0) {
        err_out = "expected valid integer number after asterisk";
        return 0;
    }

    const char* out = nullptr;
    const size_t id = static_cast<size_t>(strtoul10_64(t.begin() + 1,&out,&length));
    if (out > t.end()) {
        err_out = "failed to parse ID";
        return 0;
    }

    return id;
}


// ------------------------------------------------------------------------------------------------
float ParseTokenAsFloat(const Token& t, const char*& err_out)
{
    err_out = NULL;

    if (t.Type() != TokenType_DATA) {
        err_out = "expected TOK_DATA token";
        return 0.0f;
    }

    if(t.IsBinary())
    {
        const char* data = t.begin();
        if (data[0] != 'F' && data[0] != 'D') {
            err_out = "failed to parse F(loat) or D(ouble), unexpected data type (binary)";
            return 0.0f;
        }

        if (data[0] == 'F') {
            return SafeParse<float>(data+1, t.end());
        }
        else {
            return static_cast<float>( SafeParse<double>(data+1, t.end()) );
        }
    }

    // need to copy the input string to a temporary buffer
    // first - next in the fbx token stream comes ',',
    // which fast_atof could interpret as decimal point.
#define MAX_FLOAT_LENGTH 31
    const size_t length = static_cast<size_t>(t.end()-t.begin());
    if (length > MAX_FLOAT_LENGTH) {
        return 0.f;
    }

    char temp[MAX_FLOAT_LENGTH + 1];
    std::copy(t.begin(), t.end(), temp);
    temp[std::min(static_cast<size_t>(MAX_FLOAT_LENGTH),length)] = '\0';

    return fast_atof(temp);
}


// ------------------------------------------------------------------------------------------------
int ParseTokenAsInt(const Token& t, const char*& err_out)
{
    err_out = NULL;

    if (t.Type() != TokenType_DATA) {
        err_out = "expected TOK_DATA token";
        return 0;
    }

    if(t.IsBinary())
    {
        const char* data = t.begin();
        if (data[0] != 'I') {
            err_out = "failed to parse I(nt), unexpected data type (binary)";
            return 0;
        }

        BE_NCONST int32_t ival = SafeParse<int32_t>(data+1, t.end());
        AI_SWAP4(ival);
        return static_cast<int>(ival);
    }

    ai_assert(static_cast<size_t>(t.end() - t.begin()) > 0);

    const char* out;
    const int intval = strtol10(t.begin(),&out);
    if (out != t.end()) {
        err_out = "failed to parse ID";
        return 0;
    }

    return intval;
}


// ------------------------------------------------------------------------------------------------
int64_t ParseTokenAsInt64(const Token& t, const char*& err_out)
{
    err_out = NULL;

    if (t.Type() != TokenType_DATA) {
        err_out = "expected TOK_DATA token";
        return 0L;
    }

    if (t.IsBinary())
    {
        const char* data = t.begin();
        if (data[0] != 'L') {
            err_out = "failed to parse Int64, unexpected data type";
            return 0L;
        }

        BE_NCONST int64_t id = SafeParse<int64_t>(data + 1, t.end());
        AI_SWAP8(id);
        return id;
    }

    // XXX: should use size_t here
    unsigned int length = static_cast<unsigned int>(t.end() - t.begin());
    ai_assert(length > 0);

    const char* out = nullptr;
    const int64_t id = strtol10_64(t.begin(), &out, &length);
    if (out > t.end()) {
        err_out = "failed to parse Int64 (text)";
        return 0L;
    }

    return id;
}

// ------------------------------------------------------------------------------------------------
std::string ParseTokenAsString(const Token& t, const char*& err_out)
{
    err_out = NULL;

    if (t.Type() != TokenType_DATA) {
        err_out = "expected TOK_DATA token";
        return "";
    }

    if(t.IsBinary())
    {
        const char* data = t.begin();
        if (data[0] != 'S') {
            err_out = "failed to parse S(tring), unexpected data type (binary)";
            return "";
        }

        // read string length
        BE_NCONST int32_t len = SafeParse<int32_t>(data+1, t.end());
        AI_SWAP4(len);

        ai_assert(t.end() - data == 5 + len);
        return std::string(data + 5, len);
    }

    const size_t length = static_cast<size_t>(t.end() - t.begin());
    if(length < 2) {
        err_out = "token is too short to hold a string";
        return "";
    }

    const char* s = t.begin(), *e = t.end() - 1;
    if (*s != '\"' || *e != '\"') {
        err_out = "expected double quoted string";
        return "";
    }

    return std::string(s+1,length-2);
}


namespace {

// ------------------------------------------------------------------------------------------------
// read the type code and element count of a binary data array and stop there
void ReadBinaryDataArrayHead(const char*& data, const char* end, char& type, uint32_t& count,
    const Element& el)
{
    if (static_cast<size_t>(end-data) < 5) {
        ParseError("binary data array is too short, need five (5) bytes for type signature and element count",&el);
    }

    // data type
    type = *data;

    // read number of elements
    BE_NCONST uint32_t len = SafeParse<uint32_t>(data+1, end);
    AI_SWAP4(len);

    count = len;
    data += 5;
}


// ------------------------------------------------------------------------------------------------
// read binary data array, assume cursor points to the 'compression mode' field (i.e. behind the header)
void ReadBinaryDataArray(char type, uint32_t count, const char*& data, const char* end,
    std::vector<char>& buff,
    const Element& /*el*/)
{
    BE_NCONST uint32_t encmode = SafeParse<uint32_t>(data, end);
    AI_SWAP4(encmode);
    data += 4;

    // next comes the compressed length
    BE_NCONST uint32_t comp_len = SafeParse<uint32_t>(data, end);
    AI_SWAP4(comp_len);
    data += 4;

    ai_assert(data + comp_len == end);

    // determine the length of the uncompressed data by looking at the type signature
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

        default:
            ai_assert(false);
    };

    const uint32_t full_length = stride * count;
    buff.resize(full_length);

    if(encmode == 0) {
        ai_assert(full_length == comp_len);

        // plain data, no compression
        std::copy(data, end, buff.begin());
    }
    else if(encmode == 1) {
        // zlib/deflate, next comes ZIP head (0x78 0x01)
        // see http://www.ietf.org/rfc/rfc1950.txt

        z_stream zstream;
        zstream.opaque = Z_NULL;
        zstream.zalloc = Z_NULL;
        zstream.zfree  = Z_NULL;
        zstream.data_type = Z_BINARY;

        // http://hewgill.com/journal/entries/349-how-to-decompress-gzip-stream-with-zlib
        if(Z_OK != inflateInit(&zstream)) {
            ParseError("failure initializing zlib");
        }

        zstream.next_in   = reinterpret_cast<Bytef*>( const_cast<char*>(data) );
        zstream.avail_in  = comp_len;

        zstream.avail_out = static_cast<uInt>(buff.size());
        zstream.next_out = reinterpret_cast<Bytef*>(&*buff.begin());
        const int ret = inflate(&zstream, Z_FINISH);

        if (ret != Z_STREAM_END && ret != Z_OK) {
            ParseError("failure decompressing compressed data section");
        }

        // terminate zlib
        inflateEnd(&zstream);
    }
#ifdef ASSIMP_BUILD_DEBUG
    else {
        // runtime check for this happens at tokenization stage
        ai_assert(false);
    }
#endif

    data += comp_len;
    ai_assert(data == end);
}

} // !anon


// ------------------------------------------------------------------------------------------------
// read an array of float3 tuples
void ParseVectorDataArray(std::vector<aiVector3D>& out, const Element& el)
{
    out.resize( 0 );

    const TokenList& tok = el.Tokens();
    if(tok.empty()) {
        ParseError("unexpected empty element",&el);
    }

    if(tok[0]->IsBinary()) {
        const char* data = tok[0]->begin(), *end = tok[0]->end();

        char type;
        uint32_t count;
        ReadBinaryDataArrayHead(data, end, type, count, el);

        if(count % 3 != 0) {
            ParseError("number of floats is not a multiple of three (3) (binary)",&el);
        }

        if(!count) {
            return;
        }

        if (type != 'd' && type != 'f') {
            ParseError("expected float or double array (binary)",&el);
        }

        std::vector<char> buff;
        ReadBinaryDataArray(type, count, data, end, buff, el);

        ai_assert(data == end);
        ai_assert(buff.size() == count * (type == 'd' ? 8 : 4));

        const uint32_t count3 = count / 3;
        out.reserve(count3);

        if (type == 'd') {
            const double* d = reinterpret_cast<const double*>(&buff[0]);
            for (unsigned int i = 0; i < count3; ++i, d += 3) {
                out.push_back(aiVector3D(static_cast<ai_real>(d[0]),
                    static_cast<ai_real>(d[1]),
                    static_cast<ai_real>(d[2])));
            }
            // for debugging
            /*for ( size_t i = 0; i < out.size(); i++ ) {
                aiVector3D vec3( out[ i ] );
                std::stringstream stream;
                stream << " vec3.x = " << vec3.x << " vec3.y = " << vec3.y << " vec3.z = " << vec3.z << std::endl;
                DefaultLogger::get()->info( stream.str() );
            }*/
        }
        else if (type == 'f') {
            const float* f = reinterpret_cast<const float*>(&buff[0]);
            for (unsigned int i = 0; i < count3; ++i, f += 3) {
                out.push_back(aiVector3D(f[0],f[1],f[2]));
            }
        }

        return;
    }

    const size_t dim = ParseTokenAsDim(*tok[0]);

    // may throw bad_alloc if the input is rubbish, but this need
    // not to be prevented - importing would fail but we wouldn't
    // crash since assimp handles this case properly.
    out.reserve(dim);

    const Scope& scope = GetRequiredScope(el);
    const Element& a = GetRequiredElement(scope,"a",&el);

    if (a.Tokens().size() % 3 != 0) {
        ParseError("number of floats is not a multiple of three (3)",&el);
    }
    for (TokenList::const_iterator it = a.Tokens().begin(), end = a.Tokens().end(); it != end; ) {
        aiVector3D v;
        v.x = ParseTokenAsFloat(**it++);
        v.y = ParseTokenAsFloat(**it++);
        v.z = ParseTokenAsFloat(**it++);

        out.push_back(v);
    }
}


// ------------------------------------------------------------------------------------------------
// read an array of color4 tuples
void ParseVectorDataArray(std::vector<aiColor4D>& out, const Element& el)
{
    out.resize( 0 );
    const TokenList& tok = el.Tokens();
    if(tok.empty()) {
        ParseError("unexpected empty element",&el);
    }

    if(tok[0]->IsBinary()) {
        const char* data = tok[0]->begin(), *end = tok[0]->end();

        char type;
        uint32_t count;
        ReadBinaryDataArrayHead(data, end, type, count, el);

        if(count % 4 != 0) {
            ParseError("number of floats is not a multiple of four (4) (binary)",&el);
        }

        if(!count) {
            return;
        }

        if (type != 'd' && type != 'f') {
            ParseError("expected float or double array (binary)",&el);
        }

        std::vector<char> buff;
        ReadBinaryDataArray(type, count, data, end, buff, el);

        ai_assert(data == end);
        ai_assert(buff.size() == count * (type == 'd' ? 8 : 4));

        const uint32_t count4 = count / 4;
        out.reserve(count4);

        if (type == 'd') {
            const double* d = reinterpret_cast<const double*>(&buff[0]);
            for (unsigned int i = 0; i < count4; ++i, d += 4) {
                out.push_back(aiColor4D(static_cast<float>(d[0]),
                    static_cast<float>(d[1]),
                    static_cast<float>(d[2]),
                    static_cast<float>(d[3])));
            }
        }
        else if (type == 'f') {
            const float* f = reinterpret_cast<const float*>(&buff[0]);
            for (unsigned int i = 0; i < count4; ++i, f += 4) {
                out.push_back(aiColor4D(f[0],f[1],f[2],f[3]));
            }
        }
        return;
    }

    const size_t dim = ParseTokenAsDim(*tok[0]);

    //  see notes in ParseVectorDataArray() above
    out.reserve(dim);

    const Scope& scope = GetRequiredScope(el);
    const Element& a = GetRequiredElement(scope,"a",&el);

    if (a.Tokens().size() % 4 != 0) {
        ParseError("number of floats is not a multiple of four (4)",&el);
    }
    for (TokenList::const_iterator it = a.Tokens().begin(), end = a.Tokens().end(); it != end; ) {
        aiColor4D v;
        v.r = ParseTokenAsFloat(**it++);
        v.g = ParseTokenAsFloat(**it++);
        v.b = ParseTokenAsFloat(**it++);
        v.a = ParseTokenAsFloat(**it++);

        out.push_back(v);
    }
}


// ------------------------------------------------------------------------------------------------
// read an array of float2 tuples
void ParseVectorDataArray(std::vector<aiVector2D>& out, const Element& el)
{
    out.resize( 0 );
    const TokenList& tok = el.Tokens();
    if(tok.empty()) {
        ParseError("unexpected empty element",&el);
    }

    if(tok[0]->IsBinary()) {
        const char* data = tok[0]->begin(), *end = tok[0]->end();

        char type;
        uint32_t count;
        ReadBinaryDataArrayHead(data, end, type, count, el);

        if(count % 2 != 0) {
            ParseError("number of floats is not a multiple of two (2) (binary)",&el);
        }

        if(!count) {
            return;
        }

        if (type != 'd' && type != 'f') {
            ParseError("expected float or double array (binary)",&el);
        }

        std::vector<char> buff;
        ReadBinaryDataArray(type, count, data, end, buff, el);

        ai_assert(data == end);
        ai_assert(buff.size() == count * (type == 'd' ? 8 : 4));

        const uint32_t count2 = count / 2;
        out.reserve(count2);

        if (type == 'd') {
            const double* d = reinterpret_cast<const double*>(&buff[0]);
            for (unsigned int i = 0; i < count2; ++i, d += 2) {
                out.push_back(aiVector2D(static_cast<float>(d[0]),
                    static_cast<float>(d[1])));
            }
        }
        else if (type == 'f') {
            const float* f = reinterpret_cast<const float*>(&buff[0]);
            for (unsigned int i = 0; i < count2; ++i, f += 2) {
                out.push_back(aiVector2D(f[0],f[1]));
            }
        }

        return;
    }

    const size_t dim = ParseTokenAsDim(*tok[0]);

    // see notes in ParseVectorDataArray() above
    out.reserve(dim);

    const Scope& scope = GetRequiredScope(el);
    const Element& a = GetRequiredElement(scope,"a",&el);

    if (a.Tokens().size() % 2 != 0) {
        ParseError("number of floats is not a multiple of two (2)",&el);
    }
    for (TokenList::const_iterator it = a.Tokens().begin(), end = a.Tokens().end(); it != end; ) {
        aiVector2D v;
        v.x = ParseTokenAsFloat(**it++);
        v.y = ParseTokenAsFloat(**it++);

        out.push_back(v);
    }
}


// ------------------------------------------------------------------------------------------------
// read an array of ints
void ParseVectorDataArray(std::vector<int>& out, const Element& el)
{
    out.resize( 0 );
    const TokenList& tok = el.Tokens();
    if(tok.empty()) {
        ParseError("unexpected empty element",&el);
    }

    if(tok[0]->IsBinary()) {
        const char* data = tok[0]->begin(), *end = tok[0]->end();

        char type;
        uint32_t count;
        ReadBinaryDataArrayHead(data, end, type, count, el);

        if(!count) {
            return;
        }

        if (type != 'i') {
            ParseError("expected int array (binary)",&el);
        }

        std::vector<char> buff;
        ReadBinaryDataArray(type, count, data, end, buff, el);

        ai_assert(data == end);
        ai_assert(buff.size() == count * 4);

        out.reserve(count);

        const int32_t* ip = reinterpret_cast<const int32_t*>(&buff[0]);
        for (unsigned int i = 0; i < count; ++i, ++ip) {
            BE_NCONST int32_t val = *ip;
            AI_SWAP4(val);
            out.push_back(val);
        }

        return;
    }

    const size_t dim = ParseTokenAsDim(*tok[0]);

    // see notes in ParseVectorDataArray()
    out.reserve(dim);

    const Scope& scope = GetRequiredScope(el);
    const Element& a = GetRequiredElement(scope,"a",&el);

    for (TokenList::const_iterator it = a.Tokens().begin(), end = a.Tokens().end(); it != end; ) {
        const int ival = ParseTokenAsInt(**it++);
        out.push_back(ival);
    }
}


// ------------------------------------------------------------------------------------------------
// read an array of floats
void ParseVectorDataArray(std::vector<float>& out, const Element& el)
{
    out.resize( 0 );
    const TokenList& tok = el.Tokens();
    if(tok.empty()) {
        ParseError("unexpected empty element",&el);
    }

    if(tok[0]->IsBinary()) {
        const char* data = tok[0]->begin(), *end = tok[0]->end();

        char type;
        uint32_t count;
        ReadBinaryDataArrayHead(data, end, type, count, el);

        if(!count) {
            return;
        }

        if (type != 'd' && type != 'f') {
            ParseError("expected float or double array (binary)",&el);
        }

        std::vector<char> buff;
        ReadBinaryDataArray(type, count, data, end, buff, el);

        ai_assert(data == end);
        ai_assert(buff.size() == count * (type == 'd' ? 8 : 4));

        if (type == 'd') {
            const double* d = reinterpret_cast<const double*>(&buff[0]);
            for (unsigned int i = 0; i < count; ++i, ++d) {
                out.push_back(static_cast<float>(*d));
            }
        }
        else if (type == 'f') {
            const float* f = reinterpret_cast<const float*>(&buff[0]);
            for (unsigned int i = 0; i < count; ++i, ++f) {
                out.push_back(*f);
            }
        }

        return;
    }

    const size_t dim = ParseTokenAsDim(*tok[0]);

    // see notes in ParseVectorDataArray()
    out.reserve(dim);

    const Scope& scope = GetRequiredScope(el);
    const Element& a = GetRequiredElement(scope,"a",&el);

    for (TokenList::const_iterator it = a.Tokens().begin(), end = a.Tokens().end(); it != end; ) {
        const float ival = ParseTokenAsFloat(**it++);
        out.push_back(ival);
    }
}

// ------------------------------------------------------------------------------------------------
// read an array of uints
void ParseVectorDataArray(std::vector<unsigned int>& out, const Element& el)
{
    out.resize( 0 );
    const TokenList& tok = el.Tokens();
    if(tok.empty()) {
        ParseError("unexpected empty element",&el);
    }

    if(tok[0]->IsBinary()) {
        const char* data = tok[0]->begin(), *end = tok[0]->end();

        char type;
        uint32_t count;
        ReadBinaryDataArrayHead(data, end, type, count, el);

        if(!count) {
            return;
        }

        if (type != 'i') {
            ParseError("expected (u)int array (binary)",&el);
        }

        std::vector<char> buff;
        ReadBinaryDataArray(type, count, data, end, buff, el);

        ai_assert(data == end);
        ai_assert(buff.size() == count * 4);

        out.reserve(count);

        const int32_t* ip = reinterpret_cast<const int32_t*>(&buff[0]);
        for (unsigned int i = 0; i < count; ++i, ++ip) {
            BE_NCONST int32_t val = *ip;
            if(val < 0) {
                ParseError("encountered negative integer index (binary)");
            }

            AI_SWAP4(val);
            out.push_back(val);
        }

        return;
    }

    const size_t dim = ParseTokenAsDim(*tok[0]);

    // see notes in ParseVectorDataArray()
    out.reserve(dim);

    const Scope& scope = GetRequiredScope(el);
    const Element& a = GetRequiredElement(scope,"a",&el);

    for (TokenList::const_iterator it = a.Tokens().begin(), end = a.Tokens().end(); it != end; ) {
        const int ival = ParseTokenAsInt(**it++);
        if(ival < 0) {
            ParseError("encountered negative integer index");
        }
        out.push_back(static_cast<unsigned int>(ival));
    }
}


// ------------------------------------------------------------------------------------------------
// read an array of uint64_ts
void ParseVectorDataArray(std::vector<uint64_t>& out, const Element& el)
{
    out.resize( 0 );
    const TokenList& tok = el.Tokens();
    if(tok.empty()) {
        ParseError("unexpected empty element",&el);
    }

    if(tok[0]->IsBinary()) {
        const char* data = tok[0]->begin(), *end = tok[0]->end();

        char type;
        uint32_t count;
        ReadBinaryDataArrayHead(data, end, type, count, el);

        if(!count) {
            return;
        }

        if (type != 'l') {
            ParseError("expected long array (binary)",&el);
        }

        std::vector<char> buff;
        ReadBinaryDataArray(type, count, data, end, buff, el);

        ai_assert(data == end);
        ai_assert(buff.size() == count * 8);

        out.reserve(count);

        const uint64_t* ip = reinterpret_cast<const uint64_t*>(&buff[0]);
        for (unsigned int i = 0; i < count; ++i, ++ip) {
            BE_NCONST uint64_t val = *ip;
            AI_SWAP8(val);
            out.push_back(val);
        }

        return;
    }

    const size_t dim = ParseTokenAsDim(*tok[0]);

    // see notes in ParseVectorDataArray()
    out.reserve(dim);

    const Scope& scope = GetRequiredScope(el);
    const Element& a = GetRequiredElement(scope,"a",&el);

    for (TokenList::const_iterator it = a.Tokens().begin(), end = a.Tokens().end(); it != end; ) {
        const uint64_t ival = ParseTokenAsID(**it++);

        out.push_back(ival);
    }
}

// ------------------------------------------------------------------------------------------------
// read an array of int64_ts
void ParseVectorDataArray(std::vector<int64_t>& out, const Element& el)
{
    out.resize( 0 );
    const TokenList& tok = el.Tokens();
    if (tok.empty()) {
        ParseError("unexpected empty element", &el);
    }

    if (tok[0]->IsBinary()) {
        const char* data = tok[0]->begin(), *end = tok[0]->end();

        char type;
        uint32_t count;
        ReadBinaryDataArrayHead(data, end, type, count, el);

        if (!count) {
            return;
        }

        if (type != 'l') {
            ParseError("expected long array (binary)", &el);
        }

        std::vector<char> buff;
        ReadBinaryDataArray(type, count, data, end, buff, el);

        ai_assert(data == end);
        ai_assert(buff.size() == count * 8);

        out.reserve(count);

        const int64_t* ip = reinterpret_cast<const int64_t*>(&buff[0]);
        for (unsigned int i = 0; i < count; ++i, ++ip) {
            BE_NCONST int64_t val = *ip;
            AI_SWAP8(val);
            out.push_back(val);
        }

        return;
    }

    const size_t dim = ParseTokenAsDim(*tok[0]);

    // see notes in ParseVectorDataArray()
    out.reserve(dim);

    const Scope& scope = GetRequiredScope(el);
    const Element& a = GetRequiredElement(scope, "a", &el);

    for (TokenList::const_iterator it = a.Tokens().begin(), end = a.Tokens().end(); it != end;) {
        const int64_t ival = ParseTokenAsInt64(**it++);

        out.push_back(ival);
    }
}

// ------------------------------------------------------------------------------------------------
aiMatrix4x4 ReadMatrix(const Element& element)
{
    std::vector<float> values;
    ParseVectorDataArray(values,element);

    if(values.size() != 16) {
        ParseError("expected 16 matrix elements");
    }

    aiMatrix4x4 result;


    result.a1 = values[0];
    result.a2 = values[1];
    result.a3 = values[2];
    result.a4 = values[3];

    result.b1 = values[4];
    result.b2 = values[5];
    result.b3 = values[6];
    result.b4 = values[7];

    result.c1 = values[8];
    result.c2 = values[9];
    result.c3 = values[10];
    result.c4 = values[11];

    result.d1 = values[12];
    result.d2 = values[13];
    result.d3 = values[14];
    result.d4 = values[15];

    result.Transpose();
    return result;
}


// ------------------------------------------------------------------------------------------------
// wrapper around ParseTokenAsString() with ParseError handling
std::string ParseTokenAsString(const Token& t)
{
    const char* err;
    const std::string& i = ParseTokenAsString(t,err);
    if(err) {
        ParseError(err,t);
    }
    return i;
}

bool HasElement( const Scope& sc, const std::string& index ) {
    const Element* el = sc[ index ];
    if ( nullptr == el ) {
        return false;
    }

    return true;
}

// ------------------------------------------------------------------------------------------------
// extract a required element from a scope, abort if the element cannot be found
const Element& GetRequiredElement(const Scope& sc, const std::string& index, const Element* element /*= NULL*/)
{
    const Element* el = sc[index];
    if(!el) {
        ParseError("did not find required element \"" + index + "\"",element);
    }
    return *el;
}


// ------------------------------------------------------------------------------------------------
// extract required compound scope
const Scope& GetRequiredScope(const Element& el)
{
    const Scope* const s = el.Compound();
    if(!s) {
        ParseError("expected compound scope",&el);
    }

    return *s;
}


// ------------------------------------------------------------------------------------------------
// get token at a particular index
const Token& GetRequiredToken(const Element& el, unsigned int index)
{
    const TokenList& t = el.Tokens();
    if(index >= t.size()) {
        ParseError(Formatter::format( "missing token at index " ) << index,&el);
    }

    return *t[index];
}


// ------------------------------------------------------------------------------------------------
// wrapper around ParseTokenAsID() with ParseError handling
uint64_t ParseTokenAsID(const Token& t)
{
    const char* err;
    const uint64_t i = ParseTokenAsID(t,err);
    if(err) {
        ParseError(err,t);
    }
    return i;
}


// ------------------------------------------------------------------------------------------------
// wrapper around ParseTokenAsDim() with ParseError handling
size_t ParseTokenAsDim(const Token& t)
{
    const char* err;
    const size_t i = ParseTokenAsDim(t,err);
    if(err) {
        ParseError(err,t);
    }
    return i;
}


// ------------------------------------------------------------------------------------------------
// wrapper around ParseTokenAsFloat() with ParseError handling
float ParseTokenAsFloat(const Token& t)
{
    const char* err;
    const float i = ParseTokenAsFloat(t,err);
    if(err) {
        ParseError(err,t);
    }
    return i;
}

// ------------------------------------------------------------------------------------------------
// wrapper around ParseTokenAsInt() with ParseError handling
int ParseTokenAsInt(const Token& t)
{
    const char* err;
    const int i = ParseTokenAsInt(t,err);
    if(err) {
        ParseError(err,t);
    }
    return i;
}

// ------------------------------------------------------------------------------------------------
// wrapper around ParseTokenAsInt64() with ParseError handling
int64_t ParseTokenAsInt64(const Token& t)
{
    const char* err;
    const int64_t i = ParseTokenAsInt64(t, err);
    if (err) {
        ParseError(err, t);
    }
    return i;
}

} // !FBX
} // !Assimp

#endif
