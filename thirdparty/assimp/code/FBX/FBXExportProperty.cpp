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
#ifndef ASSIMP_BUILD_NO_EXPORT
#ifndef ASSIMP_BUILD_NO_FBX_EXPORTER

#include "FBXExportProperty.h"

#include <assimp/StreamWriter.h> // StreamWriterLE
#include <assimp/Exceptional.h> // DeadlyExportError

#include <string>
#include <vector>
#include <ostream>
#include <locale>
#include <sstream> // ostringstream

namespace Assimp {
namespace FBX {

// constructors for single element properties

FBXExportProperty::FBXExportProperty(bool v)
: type('C')
, data(1, uint8_t(v)) {}

FBXExportProperty::FBXExportProperty(int16_t v)
: type('Y')
, data(2) {
    uint8_t* d = data.data();
    (reinterpret_cast<int16_t*>(d))[0] = v;
}

FBXExportProperty::FBXExportProperty(int32_t v)
: type('I')
, data(4) {
    uint8_t* d = data.data();
    (reinterpret_cast<int32_t*>(d))[0] = v;
}

FBXExportProperty::FBXExportProperty(float v)
: type('F')
, data(4) {
    uint8_t* d = data.data();
    (reinterpret_cast<float*>(d))[0] = v;
}

FBXExportProperty::FBXExportProperty(double v)
: type('D')
, data(8) {
    uint8_t* d = data.data();
    (reinterpret_cast<double*>(d))[0] = v;
}

FBXExportProperty::FBXExportProperty(int64_t v)
: type('L')
, data(8) {
    uint8_t* d = data.data();
    (reinterpret_cast<int64_t*>(d))[0] = v;
}

// constructors for array-type properties

FBXExportProperty::FBXExportProperty(const char* c, bool raw)
: FBXExportProperty(std::string(c), raw) {
    // empty
}

// strings can either be saved as "raw" (R) data, or "string" (S) data
FBXExportProperty::FBXExportProperty(const std::string& s, bool raw)
: type(raw ? 'R' : 'S')
, data(s.size()) {
    for (size_t i = 0; i < s.size(); ++i) {
        data[i] = uint8_t(s[i]);
    }
}

FBXExportProperty::FBXExportProperty(const std::vector<uint8_t>& r)
: type('R')
, data(r) {
    // empty
}

FBXExportProperty::FBXExportProperty(const std::vector<int32_t>& va)
: type('i')
, data(4 * va.size() ) {
    int32_t* d = reinterpret_cast<int32_t*>(data.data());
    for (size_t i = 0; i < va.size(); ++i) {
        d[i] = va[i];
    }
}

FBXExportProperty::FBXExportProperty(const std::vector<int64_t>& va)
: type('l')
, data(8 * va.size()) {
    int64_t* d = reinterpret_cast<int64_t*>(data.data());
    for (size_t i = 0; i < va.size(); ++i) {
        d[i] = va[i];
    }
}

FBXExportProperty::FBXExportProperty(const std::vector<float>& va)
: type('f')
, data(4 * va.size()) {
    float* d = reinterpret_cast<float*>(data.data());
    for (size_t i = 0; i < va.size(); ++i) {
        d[i] = va[i];
    }
}

FBXExportProperty::FBXExportProperty(const std::vector<double>& va)
: type('d')
, data(8 * va.size()) {
    double* d = reinterpret_cast<double*>(data.data());
    for (size_t i = 0; i < va.size(); ++i) {
        d[i] = va[i];
    }
}

FBXExportProperty::FBXExportProperty(const aiMatrix4x4& vm)
: type('d')
, data(8 * 16) {
    double* d = reinterpret_cast<double*>(data.data());
    for (unsigned int c = 0; c < 4; ++c) {
        for (unsigned int r = 0; r < 4; ++r) {
            d[4 * c + r] = vm[r][c];
        }
    }
}

// public member functions

size_t FBXExportProperty::size() {
    switch (type) {
        case 'C':
        case 'Y':
        case 'I':
        case 'F':
        case 'D':
        case 'L':
            return data.size() + 1;
        case 'S':
        case 'R':
            return data.size() + 5;
        case 'i':
        case 'd':
            return data.size() + 13;
        default:
            throw DeadlyExportError("Requested size on property of unknown type");
    }
}

void FBXExportProperty::DumpBinary(Assimp::StreamWriterLE& s) {
    s.PutU1(type);
    uint8_t* d = data.data();
    size_t N;
    switch (type) {
        case 'C': s.PutU1(*(reinterpret_cast<uint8_t*>(d))); return;
        case 'Y': s.PutI2(*(reinterpret_cast<int16_t*>(d))); return;
        case 'I': s.PutI4(*(reinterpret_cast<int32_t*>(d))); return;
        case 'F': s.PutF4(*(reinterpret_cast<float*>(d))); return;
        case 'D': s.PutF8(*(reinterpret_cast<double*>(d))); return;
        case 'L': s.PutI8(*(reinterpret_cast<int64_t*>(d))); return;
        case 'S':
        case 'R':
            s.PutU4(uint32_t(data.size()));
            for (size_t i = 0; i < data.size(); ++i) { s.PutU1(data[i]); }
            return;
        case 'i':
            N = data.size() / 4;
            s.PutU4(uint32_t(N)); // number of elements
            s.PutU4(0); // no encoding (1 would be zip-compressed)
            // TODO: compress if large?
            s.PutU4(uint32_t(data.size())); // data size
            for (size_t i = 0; i < N; ++i) {
                s.PutI4((reinterpret_cast<int32_t*>(d))[i]);
            }
            return;
        case 'l':
            N = data.size() / 8;
            s.PutU4(uint32_t(N)); // number of elements
            s.PutU4(0); // no encoding (1 would be zip-compressed)
            // TODO: compress if large?
            s.PutU4(uint32_t(data.size())); // data size
            for (size_t i = 0; i < N; ++i) {
                s.PutI8((reinterpret_cast<int64_t*>(d))[i]);
            }
            return;
        case 'f':
            N = data.size() / 4;
            s.PutU4(uint32_t(N)); // number of elements
            s.PutU4(0); // no encoding (1 would be zip-compressed)
            // TODO: compress if large?
            s.PutU4(uint32_t(data.size())); // data size
            for (size_t i = 0; i < N; ++i) {
                s.PutF4((reinterpret_cast<float*>(d))[i]);
            }
            return;
        case 'd':
            N = data.size() / 8;
            s.PutU4(uint32_t(N)); // number of elements
            s.PutU4(0); // no encoding (1 would be zip-compressed)
            // TODO: compress if large?
            s.PutU4(uint32_t(data.size())); // data size
            for (size_t i = 0; i < N; ++i) {
                s.PutF8((reinterpret_cast<double*>(d))[i]);
            }
            return;
        default:
            std::ostringstream err;
            err << "Tried to dump property with invalid type '";
            err << type << "'!";
            throw DeadlyExportError(err.str());
    }
}

void FBXExportProperty::DumpAscii(Assimp::StreamWriterLE& outstream, int indent) {
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss.precision(15); // this seems to match official FBX SDK exports
    DumpAscii(ss, indent);
    outstream.PutString(ss.str());
}

void FBXExportProperty::DumpAscii(std::ostream& s, int indent) {
    // no writing type... or anything. just shove it into the stream.
    uint8_t* d = data.data();
    size_t N;
    size_t swap = data.size();
    size_t count = 0;
    switch (type) {
    case 'C':
        if (*(reinterpret_cast<uint8_t*>(d))) { s << 'T'; }
        else { s << 'F'; }
        return;
    case 'Y': s << *(reinterpret_cast<int16_t*>(d)); return;
    case 'I': s << *(reinterpret_cast<int32_t*>(d)); return;
    case 'F': s << *(reinterpret_cast<float*>(d)); return;
    case 'D': s << *(reinterpret_cast<double*>(d)); return;
    case 'L': s << *(reinterpret_cast<int64_t*>(d)); return;
    case 'S':
        // first search to see if it has "\x00\x01" in it -
        // which separates fields which are reversed in the ascii version.
        // yeah.
        // FBX, yeah.
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] == '\0') {
                swap = i;
                break;
            }
        }
    case 'R':
        s << '"';
        // we might as well check this now,
        // probably it will never happen
        for (size_t i = 0; i < data.size(); ++i) {
            char c = data[i];
            if (c == '"') {
                throw runtime_error("can't handle quotes in property string");
            }
        }
        // first write the SWAPPED member (if any)
        for (size_t i = swap + 2; i < data.size(); ++i) {
            char c = data[i];
            s << c;
        }
        // then a separator
        if (swap != data.size()) {
            s << "::";
        }
        // then the initial member
        for (size_t i = 0; i < swap; ++i) {
            char c = data[i];
            s << c;
        }
        s << '"';
        return;
    case 'i':
        N = data.size() / 4; // number of elements
        s << '*' << N << " {\n";
        for (int i = 0; i < indent + 1; ++i) { s << '\t'; }
        s << "a: ";
        for (size_t i = 0; i < N; ++i) {
            if (i > 0) { s << ','; }
            if (count++ > 120) { s << '\n'; count = 0; }
            s << (reinterpret_cast<int32_t*>(d))[i];
        }
        s << '\n';
        for (int i = 0; i < indent; ++i) { s << '\t'; }
        s << "} ";
        return;
    case 'l':
        N = data.size() / 8;
        s << '*' << N << " {\n";
        for (int i = 0; i < indent + 1; ++i) { s << '\t'; }
        s << "a: ";
        for (size_t i = 0; i < N; ++i) {
            if (i > 0) { s << ','; }
            if (count++ > 120) { s << '\n'; count = 0; }
            s << (reinterpret_cast<int64_t*>(d))[i];
        }
        s << '\n';
        for (int i = 0; i < indent; ++i) { s << '\t'; }
        s << "} ";
        return;
    case 'f':
        N = data.size() / 4;
        s << '*' << N << " {\n";
        for (int i = 0; i < indent + 1; ++i) { s << '\t'; }
        s << "a: ";
        for (size_t i = 0; i < N; ++i) {
            if (i > 0) { s << ','; }
            if (count++ > 120) { s << '\n'; count = 0; }
            s << (reinterpret_cast<float*>(d))[i];
        }
        s << '\n';
        for (int i = 0; i < indent; ++i) { s << '\t'; }
        s << "} ";
        return;
    case 'd':
        N = data.size() / 8;
        s << '*' << N << " {\n";
        for (int i = 0; i < indent + 1; ++i) { s << '\t'; }
        s << "a: ";
        // set precision to something that can handle doubles
        s.precision(15);
        for (size_t i = 0; i < N; ++i) {
            if (i > 0) { s << ','; }
            if (count++ > 120) { s << '\n'; count = 0; }
            s << (reinterpret_cast<double*>(d))[i];
        }
        s << '\n';
        for (int i = 0; i < indent; ++i) { s << '\t'; }
        s << "} ";
        return;
    default:
        std::ostringstream err;
        err << "Tried to dump property with invalid type '";
        err << type << "'!";
        throw runtime_error(err.str());
    }
}

} // Namespace FBX
} // Namespace Assimp

#endif // ASSIMP_BUILD_NO_FBX_EXPORTER
#endif // ASSIMP_BUILD_NO_EXPORT
