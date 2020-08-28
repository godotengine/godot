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

/** @file FBXExportProperty.h
* Declares the FBX::Property helper class for fbx export.
*/
#ifndef AI_FBXEXPORTPROPERTY_H_INC
#define AI_FBXEXPORTPROPERTY_H_INC

#ifndef ASSIMP_BUILD_NO_FBX_EXPORTER

#include <assimp/types.h> // aiMatrix4x4
#include <assimp/StreamWriter.h> // StreamWriterLE

#include <string>
#include <vector>
#include <ostream>
#include <type_traits> // is_void

namespace Assimp {
namespace FBX {

/** @brief FBX::Property
 *
 *  Holds a value of any of FBX's recognized types,
 *  each represented by a particular one-character code.
 *  C : 1-byte uint8, usually 0x00 or 0x01 to represent boolean false and true
 *  Y : 2-byte int16
 *  I : 4-byte int32
 *  F : 4-byte float
 *  D : 8-byte double
 *  L : 8-byte int64
 *  i : array of int32
 *  f : array of float
 *  d : array of double
 *  l : array of int64
 *  b : array of 1-byte booleans (0x00 or 0x01)
 *  S : string (array of 1-byte char)
 *  R : raw data (array of bytes)
 */
class FBXExportProperty {
public:
    // constructors for basic types.
    // all explicit to avoid accidental typecasting
    explicit FBXExportProperty(bool v);
    // TODO: determine if there is actually a byte type,
    // or if this always means <bool>. 'C' seems to imply <char>,
    // so possibly the above was intended to represent both.
    explicit FBXExportProperty(int16_t v);
    explicit FBXExportProperty(int32_t v);
    explicit FBXExportProperty(float v);
    explicit FBXExportProperty(double v);
    explicit FBXExportProperty(int64_t v);
    // strings can either be stored as 'R' (raw) or 'S' (string) type
    explicit FBXExportProperty(const char* c, bool raw = false);
    explicit FBXExportProperty(const std::string& s, bool raw = false);
    explicit FBXExportProperty(const std::vector<uint8_t>& r);
    explicit FBXExportProperty(const std::vector<int32_t>& va);
    explicit FBXExportProperty(const std::vector<int64_t>& va);
    explicit FBXExportProperty(const std::vector<double>& va);
    explicit FBXExportProperty(const std::vector<float>& va);
    explicit FBXExportProperty(const aiMatrix4x4& vm);

    // this will catch any type not defined above,
    // so that we don't accidentally convert something we don't want.
    // for example (const char*) --> (bool)... seriously wtf C++
    template <class T>
    explicit FBXExportProperty(T v) : type('X') {
        static_assert(std::is_void<T>::value, "TRIED TO CREATE FBX PROPERTY WITH UNSUPPORTED TYPE, CHECK YOUR PROPERTY INSTANTIATION");
    } // note: no line wrap so it appears verbatim on the compiler error

    // the size of this property node in a binary file, in bytes
    size_t size();

    // write this property node as binary data to the given stream
    void DumpBinary(Assimp::StreamWriterLE& s);
    void DumpAscii(Assimp::StreamWriterLE& s, int indent = 0);
    void DumpAscii(std::ostream& s, int indent = 0);
    // note: make sure the ostream is in classic "C" locale

private:
    char type;
    std::vector<uint8_t> data;
};

} // Namespace FBX
} // Namespace Assimp

#endif // ASSIMP_BUILD_NO_FBX_EXPORTER

#endif // AI_FBXEXPORTPROPERTY_H_INC
