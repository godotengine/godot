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
#ifndef AI_GLFTCOMMON_H_INC
#define AI_GLFTCOMMON_H_INC

#ifndef ASSIMP_BUILD_NO_GLTF_IMPORTER

#include <assimp/Exceptional.h>

#include <map>
#include <string>
#include <list>
#include <vector>
#include <algorithm>
#include <stdexcept>

#define RAPIDJSON_HAS_STDSTRING 1
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#ifdef ASSIMP_API
#   include <memory>
#   include <assimp/DefaultIOSystem.h>
#   include <assimp/ByteSwapper.h>
#else
#   include <memory>
#   define AI_SWAP4(p)
#   define ai_assert
#endif


#if _MSC_VER > 1500 || (defined __GNUC___)
#       define ASSIMP_GLTF_USE_UNORDERED_MULTIMAP
#   else
#       define gltf_unordered_map map
#endif

#ifdef ASSIMP_GLTF_USE_UNORDERED_MULTIMAP
#   include <unordered_map>
#   if _MSC_VER > 1600
#       define gltf_unordered_map unordered_map
#   else
#       define gltf_unordered_map tr1::unordered_map
#   endif
#endif

namespace glTFCommon {

#ifdef ASSIMP_API
    using Assimp::IOStream;
    using Assimp::IOSystem;
    using std::shared_ptr;
#else
    using std::shared_ptr;

    typedef std::runtime_error DeadlyImportError;
    typedef std::runtime_error DeadlyExportError;

    enum aiOrigin {
        aiOrigin_SET = 0,
        aiOrigin_CUR = 1,
        aiOrigin_END = 2
    };

    class IOSystem;

    class IOStream {
    public:
        IOStream(FILE* file) : f(file) {}
        ~IOStream() { fclose(f); f = 0; }

        size_t Read(void* b, size_t sz, size_t n) { return fread(b, sz, n, f); }
        size_t Write(const void* b, size_t sz, size_t n) { return fwrite(b, sz, n, f); }
        int    Seek(size_t off, aiOrigin orig) { return fseek(f, off, int(orig)); }
        size_t Tell() const { return ftell(f); }

        size_t FileSize() {
            long p = Tell(), len = (Seek(0, aiOrigin_END), Tell());
            return size_t((Seek(p, aiOrigin_SET), len));
        }

    private:
        FILE* f;
    };
#endif

    // Vec/matrix types, as raw float arrays
    typedef float(vec3)[3];
    typedef float(vec4)[4];
    typedef float(mat4)[16];

    inline
    void CopyValue(const glTFCommon::vec3& v, aiColor4D& out) {
        out.r = v[0];
        out.g = v[1];
        out.b = v[2];
        out.a = 1.0;
    }

    inline
    void CopyValue(const glTFCommon::vec4& v, aiColor4D& out) {
        out.r = v[0];
        out.g = v[1];
        out.b = v[2];
        out.a = v[3];
    }

    inline
    void CopyValue(const glTFCommon::vec4& v, aiColor3D& out) {
        out.r = v[0];
        out.g = v[1];
        out.b = v[2];
    }

    inline
    void CopyValue(const glTFCommon::vec3& v, aiColor3D& out) {
        out.r = v[0];
        out.g = v[1];
        out.b = v[2];
    }

    inline
    void CopyValue(const glTFCommon::vec3& v, aiVector3D& out) {
        out.x = v[0];
        out.y = v[1];
        out.z = v[2];
    }

    inline
    void CopyValue(const glTFCommon::vec4& v, aiQuaternion& out) {
        out.x = v[0];
        out.y = v[1];
        out.z = v[2];
        out.w = v[3];
    }

    inline
    void CopyValue(const glTFCommon::mat4& v, aiMatrix4x4& o) {
        o.a1 = v[0]; o.b1 = v[1]; o.c1 = v[2]; o.d1 = v[3];
        o.a2 = v[4]; o.b2 = v[5]; o.c2 = v[6]; o.d2 = v[7];
        o.a3 = v[8]; o.b3 = v[9]; o.c3 = v[10]; o.d3 = v[11];
        o.a4 = v[12]; o.b4 = v[13]; o.c4 = v[14]; o.d4 = v[15];
    }

    namespace Util {

        void EncodeBase64(const uint8_t* in, size_t inLength, std::string& out);

        size_t DecodeBase64(const char* in, size_t inLength, uint8_t*& out);

        inline
            size_t DecodeBase64(const char* in, uint8_t*& out) {
            return DecodeBase64(in, strlen(in), out);
        }

        struct DataURI {
            const char* mediaType;
            const char* charset;
            bool base64;
            const char* data;
            size_t dataLength;
        };

        //! Check if a uri is a data URI
        bool ParseDataURI(const char* const_uri, size_t uriLen, DataURI& out);

        template<bool B>
        struct DATA {
            static const uint8_t tableDecodeBase64[128];
        };

        template<bool B>
        const uint8_t DATA<B>::tableDecodeBase64[128] = {
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 62,  0,  0,  0, 63,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  0,  0,  0, 64,  0,  0,
                0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  0,  0,  0,  0,  0,
                0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,  0,  0,  0,  0,  0
        };

        inline
            char EncodeCharBase64(uint8_t b) {
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[size_t(b)];
        }

        inline
            uint8_t DecodeCharBase64(char c) {
            return DATA<true>::tableDecodeBase64[size_t(c)]; // TODO faster with lookup table or ifs?
            /*if (c >= 'A' && c <= 'Z') return c - 'A';
            if (c >= 'a' && c <= 'z') return c - 'a' + 26;
            if (c >= '0' && c <= '9') return c - '0' + 52;
            if (c == '+') return 62;
            if (c == '/') return 63;
            return 64; // '-' */
        }

        size_t DecodeBase64(const char* in, size_t inLength, uint8_t*& out);

        void EncodeBase64(const uint8_t* in, size_t inLength, std::string& out);
    }

}

#endif  // ASSIMP_BUILD_NO_GLTF_IMPORTER

#endif // AI_GLFTCOMMON_H_INC
