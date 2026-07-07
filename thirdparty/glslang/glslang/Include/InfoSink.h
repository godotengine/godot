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

#ifndef _INFOSINK_INCLUDED_
#define _INFOSINK_INCLUDED_

#include "../Include/Common.h"
//#include <filesystem>
#include <cmath>

namespace glslang {

//
// TPrefixType is used to centralize how info log messages start.
// See below.
//
enum TPrefixType {
    EPrefixNone,
    EPrefixWarning,
    EPrefixError,
    EPrefixInternalError,
    EPrefixUnimplemented,
    EPrefixNote
};

enum TOutputStream {
    ENull = 0,
    EDebugger = 0x01,
    EStdOut = 0x02,
    EString = 0x04,
};
//
// Encapsulate info logs for all objects that have them.
//
// The methods are a general set of tools for getting a variety of
// messages and types inserted into the log.
//
class TInfoSinkBase {
public:
    TInfoSinkBase() : outputStream(4), shaderFileName(nullptr) {}
    void erase() { sink.erase(); }
    TInfoSinkBase& operator<<(const TPersistString& t) { append(t); return *this; }
    TInfoSinkBase& operator<<(char c)                  { append(1, c); return *this; }
    TInfoSinkBase& operator<<(const char* s)           { append(s); return *this; }
    TInfoSinkBase& operator<<(int n)                   { append(String(n)); return *this; }
    TInfoSinkBase& operator<<(unsigned int n)          { append(String(n)); return *this; }
    TInfoSinkBase& operator<<(float n)                 { const int size = 40; char buf[size];
                                                         snprintf(buf, size, (fabs(n) > 1e-8 && fabs(n) < 1e8) || n == 0.0f ? "%f" : "%g", n);
                                                         append(buf);
                                                         return *this; }
    TInfoSinkBase& operator+(const TPersistString& t)  { append(t); return *this; }
    TInfoSinkBase& operator+(const TString& t)         { append(t); return *this; }
    TInfoSinkBase& operator<<(const TString& t)        { append(t); return *this; }
    TInfoSinkBase& operator+(const char* s)            { append(s); return *this; }
    const char* c_str() const { return sink.c_str(); }
    void prefix(TPrefixType message) {
        switch(message) {
        case EPrefixNone:                                      break;
        case EPrefixWarning:       append("WARNING: ");        break;
        case EPrefixError:         append("ERROR: ");          break;
        case EPrefixInternalError: append("INTERNAL ERROR: "); break;
        case EPrefixUnimplemented: append("UNIMPLEMENTED: ");  break;
        case EPrefixNote:          append("NOTE: ");           break;
        default:                   append("UNKNOWN ERROR: ");   break;
        }
    }
    void location(const TSourceLoc& loc, bool absolute = false, bool displayColumn = false) {
        const int maxSize = 24;
        char locText[maxSize];
        if (displayColumn) {
            snprintf(locText, maxSize, ":%d:%d", loc.line, loc.column);
        } else {
            snprintf(locText, maxSize, ":%d", loc.line);
        }

        if(loc.getFilename() == nullptr && shaderFileName != nullptr && absolute) {
            //append(std::filesystem::absolute(shaderFileName).string());
        } else {
            std::string location = loc.getStringNameOrNum(false);
            //if (absolute) {
            //    append(std::filesystem::absolute(location).string());
            //} else {
                append(location);
            //}
        }

        append(locText);
        append(": ");
    }
    void message(TPrefixType message, const char* s) {
        prefix(message);
        append(s);
        append("\n");
    }
    void message(TPrefixType message, const char* s, const TSourceLoc& loc, bool absolute = false,
                 bool displayColumn = false)
    {
        prefix(message);
        location(loc, absolute, displayColumn);
        append(s);
        append("\n");
    }

    void setOutputStream(int output = 4)
    {
        outputStream = output;
    }

    void setShaderFileName(const char* file = nullptr)
    {
        shaderFileName = file;
    }

protected:
    void append(const char* s);

    void append(int count, char c);
    void append(const TPersistString& t);
    void append(const TString& t);

    void checkMem(size_t growth) { if (sink.capacity() < sink.size() + growth + 2)
                                       sink.reserve(sink.capacity() +  sink.capacity() / 2); }
    void appendToStream(const char* s);
    TPersistString sink;
    int outputStream;
    const char* shaderFileName;
};

} // end namespace glslang

class TInfoSink {
public:
    glslang::TInfoSinkBase info;
    glslang::TInfoSinkBase debug;
};

#endif // _INFOSINK_INCLUDED_
