/*
LodePNG Examples

Copyright (c) 2005-2012 Lode Vandevenne

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

//g++ lodepng.cpp example_4bit_palette.cpp -ansi -pedantic -Wall -Wextra -O3



/*
LodePNG 4-bit palette example.
This example encodes a 511x511 PNG with a 4-bit palette.
Both image and palette contain sine waves, resulting in a sort of plasma.
The 511 (rather than power of two 512) size is of course chosen on purpose to
confirm that scanlines not filling up an entire byte size are working.

NOTE: a PNG image with a translucent palette is perfectly valid. However there
exist some programs that cannot correctly read those, including, surprisingly,
Gimp 2.8 image editor (until you set mode to RGB).
*/

#include <cmath>
#include <iostream>

#include "lodepng.h"

int main(int argc, char *argv[]) {
  //check if user gave a filename
  if(argc < 2) {
    std::cout << "please provide a filename to save to" << std::endl;
    return 0;
  }

  //create encoder and set settings and info (optional)
  lodepng::State state;

  //generate palette
  for(int i = 0; i < 16; i++) {
    unsigned char r = 127 * (1 + std::sin(5 * i * 6.28318531 / 16));
    unsigned char g = 127 * (1 + std::sin(2 * i * 6.28318531 / 16));
    unsigned char b = 127 * (1 + std::sin(3 * i * 6.28318531 / 16));
    unsigned char a = 63 * (1 + std::sin(8 * i * 6.28318531 / 16)) + 128; /*alpha channel of the palette (tRNS chunk)*/

    //palette must be added both to input and output color mode, because in this
    //sample both the raw image and the expected PNG image use that palette.
    lodepng_palette_add(&state.info_png.color, r, g, b, a);
    lodepng_palette_add(&state.info_raw, r, g, b, a);
  }

  //both the raw image and the encoded image must get colorType 3 (palette)
  state.info_png.color.colortype = LCT_PALETTE; //if you comment this line, and create the above palette in info_raw instead, then you get the same image in a RGBA PNG.
  state.info_png.color.bitdepth = 4;
  state.info_raw.colortype = LCT_PALETTE;
  state.info_raw.bitdepth = 4;
  state.encoder.auto_convert = 0; //we specify ourselves exactly what output PNG color mode we want

  //generate some image
  const unsigned w = 511;
  const unsigned h = 511;
  std::vector<unsigned char> image;
  image.resize((w * h * 4 + 7) / 8, 0);
  for(unsigned y = 0; y < h; y++)
  for(unsigned x = 0; x < w; x++) {
    size_t byte_index = (y * w + x) / 2;
    bool byte_half = (y * w + x) % 2 == 1;

    int color = (int)(4 * ((1 + std::sin(2.0 * 6.28318531 * x / (double)w))
                         + (1 + std::sin(2.0 * 6.28318531 * y / (double)h))) );

    image[byte_index] |= (unsigned char)(color << (byte_half ? 0 : 4));
  }

  //encode and save
  std::vector<unsigned char> buffer;
  unsigned error = lodepng::encode(buffer, image.empty() ? 0 : &image[0], w, h, state);
  if(error) {
    std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
    return 0;
  }
  lodepng::save_file(buffer, argv[1]);
}
