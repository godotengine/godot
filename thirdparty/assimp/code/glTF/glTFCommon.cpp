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
#include "glTF/glTFCommon.h"

namespace glTFCommon {

using namespace glTFCommon::Util;

namespace Util {

size_t DecodeBase64(const char* in, size_t inLength, uint8_t*& out) {
    ai_assert(inLength % 4 == 0);

    if (inLength < 4) {
        out = 0;
        return 0;
    }

    int nEquals = int(in[inLength - 1] == '=') +
        int(in[inLength - 2] == '=');

    size_t outLength = (inLength * 3) / 4 - nEquals;
    out = new uint8_t[outLength];
    memset(out, 0, outLength);

    size_t i, j = 0;

    for (i = 0; i + 4 < inLength; i += 4) {
        uint8_t b0 = DecodeCharBase64(in[i]);
        uint8_t b1 = DecodeCharBase64(in[i + 1]);
        uint8_t b2 = DecodeCharBase64(in[i + 2]);
        uint8_t b3 = DecodeCharBase64(in[i + 3]);

        out[j++] = (uint8_t)((b0 << 2) | (b1 >> 4));
        out[j++] = (uint8_t)((b1 << 4) | (b2 >> 2));
        out[j++] = (uint8_t)((b2 << 6) | b3);
    }

    {
        uint8_t b0 = DecodeCharBase64(in[i]);
        uint8_t b1 = DecodeCharBase64(in[i + 1]);
        uint8_t b2 = DecodeCharBase64(in[i + 2]);
        uint8_t b3 = DecodeCharBase64(in[i + 3]);

        out[j++] = (uint8_t)((b0 << 2) | (b1 >> 4));
        if (b2 < 64) out[j++] = (uint8_t)((b1 << 4) | (b2 >> 2));
        if (b3 < 64) out[j++] = (uint8_t)((b2 << 6) | b3);
    }

    return outLength;
}

void EncodeBase64(const uint8_t* in, size_t inLength, std::string& out) {
    size_t outLength = ((inLength + 2) / 3) * 4;

    size_t j = out.size();
    out.resize(j + outLength);

    for (size_t i = 0; i < inLength; i += 3) {
        uint8_t b = (in[i] & 0xFC) >> 2;
        out[j++] = EncodeCharBase64(b);

        b = (in[i] & 0x03) << 4;
        if (i + 1 < inLength) {
            b |= (in[i + 1] & 0xF0) >> 4;
            out[j++] = EncodeCharBase64(b);

            b = (in[i + 1] & 0x0F) << 2;
            if (i + 2 < inLength) {
                b |= (in[i + 2] & 0xC0) >> 6;
                out[j++] = EncodeCharBase64(b);

                b = in[i + 2] & 0x3F;
                out[j++] = EncodeCharBase64(b);
            }
            else {
                out[j++] = EncodeCharBase64(b);
                out[j++] = '=';
            }
        }
        else {
            out[j++] = EncodeCharBase64(b);
            out[j++] = '=';
            out[j++] = '=';
        }
    }
}

bool ParseDataURI(const char* const_uri, size_t uriLen, DataURI& out) {
    if (nullptr == const_uri) {
        return false;
    }

    if (const_uri[0] != 0x10) { // we already parsed this uri?
        if (strncmp(const_uri, "data:", 5) != 0) // not a data uri?
            return false;
    }

    // set defaults
    out.mediaType = "text/plain";
    out.charset = "US-ASCII";
    out.base64 = false;

    char* uri = const_cast<char*>(const_uri);
    if (uri[0] != 0x10) {
        uri[0] = 0x10;
        uri[1] = uri[2] = uri[3] = uri[4] = 0;

        size_t i = 5, j;
        if (uri[i] != ';' && uri[i] != ',') { // has media type?
            uri[1] = char(i);
            for (; uri[i] != ';' && uri[i] != ',' && i < uriLen; ++i) {
                // nothing to do!
            }
        }
        while (uri[i] == ';' && i < uriLen) {
            uri[i++] = '\0';
            for (j = i; uri[i] != ';' && uri[i] != ',' && i < uriLen; ++i) {
                // nothing to do!
            }

            if (strncmp(uri + j, "charset=", 8) == 0) {
                uri[2] = char(j + 8);
            }
            else if (strncmp(uri + j, "base64", 6) == 0) {
                uri[3] = char(j);
            }
        }
        if (i < uriLen) {
            uri[i++] = '\0';
            uri[4] = char(i);
        }
        else {
            uri[1] = uri[2] = uri[3] = 0;
            uri[4] = 5;
        }
    }

    if (uri[1] != 0) {
        out.mediaType = uri + uri[1];
    }
    if (uri[2] != 0) {
        out.charset = uri + uri[2];
    }
    if (uri[3] != 0) {
        out.base64 = true;
    }
    out.data = uri + uri[4];
    out.dataLength = (uri + uriLen) - out.data;

    return true;
}

}
}
