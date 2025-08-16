//
// Copyright (C) 2013 LunarG, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

//
// This holds context specific to the GLSL scanner, which
// sits between the preprocessor scanner and parser.
//

#pragma once

#include "ParseHelper.h"

namespace glslang {

class TPpContext;
class TPpToken;
class TParserToken;

class TScanContext {
public:
    explicit TScanContext(TParseContextBase& pc) :
        parseContext(pc),
        afterType(false), afterStruct(false),
        field(false), afterBuffer(false) { }
    virtual ~TScanContext() { }

    static void fillInKeywordMap();
    static void deleteKeywordMap();

    int tokenize(TPpContext*, TParserToken&);

protected:
    TScanContext(TScanContext&);
    TScanContext& operator=(TScanContext&);

    int tokenizeIdentifier();
    int identifierOrType();
    int reservedWord();
    int identifierOrReserved(bool reserved);
    int es30ReservedFromGLSL(int version);
    int nonreservedKeyword(int esVersion, int nonEsVersion);
    int precisionKeyword();
    int matNxM();
    int dMat();
    int firstGenerationImage(bool inEs310);
    int secondGenerationImage();

    TParseContextBase& parseContext;
    bool afterType;           // true if we've recognized a type, so can only be looking for an identifier
    bool afterStruct;         // true if we've recognized the STRUCT keyword, so can only be looking for an identifier
    bool field;               // true if we're on a field, right after a '.'
    bool afterBuffer;         // true if we've recognized the BUFFER keyword
    TSourceLoc loc;
    TParserToken* parserToken;
    TPpToken* ppToken;

    const char* tokenText;
    int keyword;
};

} // end namespace glslang
