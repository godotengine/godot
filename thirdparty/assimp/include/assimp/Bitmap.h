/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

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
---------------------------------------------------------------------------
*/

/** @file Bitmap.h
 *  @brief Defines bitmap format helper for textures
 *
 * Used for file formats which embed their textures into the model file.
 */
#pragma once
#ifndef AI_BITMAP_H_INC
#define AI_BITMAP_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include "defs.h"
#include <stdint.h>
#include <cstddef>

struct aiTexture;

namespace Assimp {

class IOStream;

class ASSIMP_API Bitmap {
protected:

    struct Header {
        uint16_t type;
        uint32_t size;
        uint16_t reserved1;
        uint16_t reserved2;
        uint32_t offset;

        // We define the struct size because sizeof(Header) might return a wrong result because of structure padding.
        // Moreover, we must use this ugly and error prone syntax because Visual Studio neither support constexpr or sizeof(name_of_field).
        static const std::size_t header_size =
            sizeof(uint16_t) + // type
            sizeof(uint32_t) + // size
            sizeof(uint16_t) + // reserved1
            sizeof(uint16_t) + // reserved2
            sizeof(uint32_t);  // offset
    };

    struct DIB {
        uint32_t size;
        int32_t width;
        int32_t height;
        uint16_t planes;
        uint16_t bits_per_pixel;
        uint32_t compression;
        uint32_t image_size;
        int32_t x_resolution;
        int32_t y_resolution;
        uint32_t nb_colors;
        uint32_t nb_important_colors;

        // We define the struct size because sizeof(DIB) might return a wrong result because of structure padding.
        // Moreover, we must use this ugly and error prone syntax because Visual Studio neither support constexpr or sizeof(name_of_field).
        static const std::size_t dib_size =
            sizeof(uint32_t) + // size
            sizeof(int32_t) +  // width
            sizeof(int32_t) +  // height
            sizeof(uint16_t) + // planes
            sizeof(uint16_t) + // bits_per_pixel
            sizeof(uint32_t) + // compression
            sizeof(uint32_t) + // image_size
            sizeof(int32_t) +  // x_resolution
            sizeof(int32_t) +  // y_resolution
            sizeof(uint32_t) + // nb_colors
            sizeof(uint32_t);  // nb_important_colors
    };

    static const std::size_t mBytesPerPixel = 4;

public:
    static void Save(aiTexture* texture, IOStream* file);

protected:
    static void WriteHeader(Header& header, IOStream* file);
    static void WriteDIB(DIB& dib, IOStream* file);
    static void WriteData(aiTexture* texture, IOStream* file);
};

}

#endif // AI_BITMAP_H_INC
