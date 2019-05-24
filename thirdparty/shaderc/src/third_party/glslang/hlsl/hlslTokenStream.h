//
// Copyright (C) 2016 Google, Inc.
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
//    Neither the name of Google, Inc., nor the names of its
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

#ifndef HLSLTOKENSTREAM_H_
#define HLSLTOKENSTREAM_H_

#include "hlslScanContext.h"

namespace glslang {

    class HlslTokenStream {
    public:
        explicit HlslTokenStream(HlslScanContext& scanner)
            : scanner(scanner), preTokenStackSize(0), tokenBufferPos(0) { }
        virtual ~HlslTokenStream() { }

    public:
        void advanceToken();
        void recedeToken();
        bool acceptTokenClass(EHlslTokenClass);
        EHlslTokenClass peek() const;
        bool peekTokenClass(EHlslTokenClass) const;
        glslang::TBuiltInVariable mapSemantic(const char* upperCase) { return scanner.mapSemantic(upperCase); }

        void pushTokenStream(const TVector<HlslToken>* tokens);
        void popTokenStream();

    protected:
        HlslToken token;                  // the token we are currently looking at, but have not yet accepted

    private:
        HlslTokenStream();
        HlslTokenStream& operator=(const HlslTokenStream&);

        HlslScanContext& scanner;         // lexical scanner, to get next token from source file
        TVector<const TVector<HlslToken>*> tokenStreamStack; // for getting the next token from an existing vector of tokens
        TVector<int> tokenPosition;
        TVector<HlslToken> currentTokenStack;

        // This is the number of tokens we can recedeToken() over.
        static const int tokenBufferSize = 2;

        // Previously scanned tokens, returned for future advances,
        // so logically in front of the token stream.
        // Is logically a stack; needs last in last out semantics.
        // Currently implemented as a stack of size 2.
        HlslToken preTokenStack[tokenBufferSize];
        int preTokenStackSize;
        void pushPreToken(const HlslToken&);
        HlslToken popPreToken();

        // Previously scanned tokens, not yet returned for future advances,
        // but available for that.
        // Is logically a fifo for normal advances, and a stack for recession.
        // Currently implemented with an intrinsic size of 2.
        HlslToken tokenBuffer[tokenBufferSize];
        int tokenBufferPos;
        void pushTokenBuffer(const HlslToken&);
        HlslToken popTokenBuffer();
    };

} // end namespace glslang

#endif // HLSLTOKENSTREAM_H_
