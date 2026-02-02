/*
Copyright (c) 2022 - Present, Syoyo Fujita.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Syoyo Fujita nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace tinyusdz {

enum ColorSpace {
  COLORSPACE_NONE, // No explicit colorspace
  COLORSPACE_SRGB,
  COLORSPACE_LINEAR
};

//
// Simple tile coordinate for UDIM texture
//
struct TileCoord {
  int32_t x{0};
  int32_t y{0};
};

struct Texture {

  uint32_t _image_id; // ID to `Image`.

  int32_t _width;
  int32_t _height;
  int32_t _stride; // width stride
  int32_t _channels;

  ColorSpace colorspace{COLORSPACE_SRGB}; // Default = sRGB

};

struct UDIMTexture {
  std::vector<std::pair<TileCoord, Texture>> _textures;
};

// For CPU texture mapping
struct TextureSampler {

  bool Sample(const Texture &tex, float u, float v, float w);

};

} // namespace tinyusdz

