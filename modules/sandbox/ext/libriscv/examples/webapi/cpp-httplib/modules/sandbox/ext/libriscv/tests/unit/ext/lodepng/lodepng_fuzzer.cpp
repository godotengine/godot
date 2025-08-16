/*
LodePNG Fuzzer

Copyright (c) 2005-2019 Lode Vandevenne

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

// clang++ -fsanitize=fuzzer lodepng.cpp lodepng_fuzzer.cpp -O3 && ./a.out

#include "lodepng.h"

#include <cstdint>

namespace {
// Amount of valid colortype/bidthdepth combinations in the PNG file format.
const size_t num_combinations = 15;

LodePNGColorType colortypes[num_combinations] = {
  LCT_GREY, LCT_GREY, LCT_GREY, LCT_GREY, LCT_GREY, // 1, 2, 4, 8 or 16 bits
  LCT_RGB, LCT_RGB, // 8 or 16 bits
  LCT_PALETTE, LCT_PALETTE, LCT_PALETTE, LCT_PALETTE, // 1, 2, 4 or 8 bits
  LCT_GREY_ALPHA, LCT_GREY_ALPHA, // 8 or 16 bits
  LCT_RGBA, LCT_RGBA, // 8 or 16 bits
};

unsigned bitdepths[num_combinations] = {
  1, 2, 4, 8, 16, // gray
  8, 16, // rgb
  1, 2, 4, 8, // palette
  8, 16, // gray+alpha
  8, 16, // rgb+alpha
};

unsigned testDecode(lodepng::State& state, const uint8_t* data, size_t size) {
  unsigned w, h;
  std::vector<unsigned char> image;
  return lodepng::decode(image, w, h, state, (const unsigned char*)data, size);
}
} // end anonymous namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if(size == 0) return 0;
  
  // Setting last byte of input as random_color_type
  // Fuzzer will still be able to mutate the data accordingly as
  // last byte of png file can be changed and file will still remain valid.
  size_t random_color_type = data[size-1] % num_combinations;

  lodepng::State state;

  // Make the decoder ignore three types of checksums the PNG/zlib format have
  // built-in, because they are less likely to be correct in the random input
  // data, and if invalid make the decoder return an error before much gets ran.
  state.decoder.zlibsettings.ignore_adler32 = 1;
  state.decoder.zlibsettings.ignore_nlen = 1;
  state.decoder.ignore_crc = 1;
  // Also make decoder attempt to support partial files with missing ending to
  // go further with parsing.
  state.decoder.ignore_end = 1;

  // First test without color conversion (keep color type of the PNG)
  state.decoder.color_convert = 0;

  unsigned error = testDecode(state, data, size);

  // If valid PNG found, try decoding with color conversion to the most common
  // default color type, and to the randomly chosen type.
  if(error == 0) {
    state.decoder.color_convert = 1;
    testDecode(state, data, size);

    state.info_raw.colortype = colortypes[random_color_type];
    state.info_raw.bitdepth = bitdepths[random_color_type];
    testDecode(state, data, size);
  }

  return 0;
}
