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

/** @file Bitmap.cpp
 *  @brief Defines bitmap format helper for textures
 *
 * Used for file formats which embed their textures into the model file.
 */


#include <assimp/Bitmap.h>
#include <assimp/texture.h>
#include <assimp/IOStream.hpp>
#include <assimp/ByteSwapper.h>

namespace Assimp {

    void Bitmap::Save(aiTexture* texture, IOStream* file) {
        if(file != NULL) {
            Header header;
            DIB dib;

            dib.size = DIB::dib_size;
            dib.width = texture->mWidth;
            dib.height = texture->mHeight;
            dib.planes = 1;
            dib.bits_per_pixel = 8 * mBytesPerPixel;
            dib.compression = 0;
            dib.image_size = (((dib.width * mBytesPerPixel) + 3) & 0x0000FFFC) * dib.height;
            dib.x_resolution = 0;
            dib.y_resolution = 0;
            dib.nb_colors = 0;
            dib.nb_important_colors = 0;

            header.type = 0x4D42; // 'BM'
            header.offset = Header::header_size + DIB::dib_size;
            header.size = header.offset + dib.image_size;
            header.reserved1 = 0;
            header.reserved2 = 0;

            WriteHeader(header, file);
            WriteDIB(dib, file);
            WriteData(texture, file);
        }
    }

    template<typename T>
    inline 
    std::size_t Copy(uint8_t* data, const T &field) {
#ifdef AI_BUILD_BIG_ENDIAN
        T field_swapped=AI_BE(field);
        std::memcpy(data, &field_swapped, sizeof(field)); return sizeof(field);
#else
        std::memcpy(data, &AI_BE(field), sizeof(field)); return sizeof(field);
#endif
    }

    void Bitmap::WriteHeader(Header& header, IOStream* file) {
        uint8_t data[Header::header_size];

        std::size_t offset = 0;

        offset += Copy(&data[offset], header.type);
        offset += Copy(&data[offset], header.size);
        offset += Copy(&data[offset], header.reserved1);
        offset += Copy(&data[offset], header.reserved2);
                  Copy(&data[offset], header.offset);

        file->Write(data, Header::header_size, 1);
    }

    void Bitmap::WriteDIB(DIB& dib, IOStream* file) {
        uint8_t data[DIB::dib_size];

        std::size_t offset = 0;

        offset += Copy(&data[offset], dib.size);
        offset += Copy(&data[offset], dib.width);
        offset += Copy(&data[offset], dib.height);
        offset += Copy(&data[offset], dib.planes);
        offset += Copy(&data[offset], dib.bits_per_pixel);
        offset += Copy(&data[offset], dib.compression);
        offset += Copy(&data[offset], dib.image_size);
        offset += Copy(&data[offset], dib.x_resolution);
        offset += Copy(&data[offset], dib.y_resolution);
        offset += Copy(&data[offset], dib.nb_colors);
                  Copy(&data[offset], dib.nb_important_colors);

        file->Write(data, DIB::dib_size, 1);
    }

    void Bitmap::WriteData(aiTexture* texture, IOStream* file) {
        static const std::size_t padding_offset = 4;
        static const uint8_t padding_data[padding_offset] = {0x0, 0x0, 0x0, 0x0};

        unsigned int padding = (padding_offset - ((mBytesPerPixel * texture->mWidth) % padding_offset)) % padding_offset;
        uint8_t pixel[mBytesPerPixel];

        for(std::size_t i = 0; i < texture->mHeight; ++i) {
            for(std::size_t j = 0; j < texture->mWidth; ++j) {
                const aiTexel& texel = texture->pcData[(texture->mHeight - i - 1) * texture->mWidth + j]; // Bitmap files are stored in bottom-up format

                pixel[0] = texel.r;
                pixel[1] = texel.g;
                pixel[2] = texel.b;
                pixel[3] = texel.a;

                file->Write(pixel, mBytesPerPixel, 1);
            }

            file->Write(padding_data, padding, 1);
        }
    }

}
