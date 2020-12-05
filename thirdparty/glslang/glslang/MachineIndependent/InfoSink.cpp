//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
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

#include "../Include/InfoSink.h"

#include <cstring>

namespace glslang {

void TInfoSinkBase::append(const char* s)
{
    if (outputStream & EString) {
        if (s == nullptr)
            sink.append("(null)");
        else {
            checkMem(strlen(s));
            sink.append(s);
        }
    }

//#ifdef _WIN32
//    if (outputStream & EDebugger)
//        OutputDebugString(s);
//#endif

    if (outputStream & EStdOut)
        fprintf(stdout, "%s", s);
}

void TInfoSinkBase::append(int count, char c)
{
    if (outputStream & EString) {
        checkMem(count);
        sink.append(count, c);
    }

//#ifdef _WIN32
//    if (outputStream & EDebugger) {
//        char str[2];
//        str[0] = c;
//        str[1] = '\0';
//        OutputDebugString(str);
//    }
//#endif

    if (outputStream & EStdOut)
        fprintf(stdout, "%c", c);
}

void TInfoSinkBase::append(const TPersistString& t)
{
    if (outputStream & EString) {
        checkMem(t.size());
        sink.append(t);
    }

//#ifdef _WIN32
//    if (outputStream & EDebugger)
//        OutputDebugString(t.c_str());
//#endif

    if (outputStream & EStdOut)
        fprintf(stdout, "%s", t.c_str());
}

void TInfoSinkBase::append(const TString& t)
{
    if (outputStream & EString) {
        checkMem(t.size());
        sink.append(t.c_str());
    }

//#ifdef _WIN32
//    if (outputStream & EDebugger)
//        OutputDebugString(t.c_str());
//#endif

    if (outputStream & EStdOut)
        fprintf(stdout, "%s", t.c_str());
}

} // end namespace glslang
