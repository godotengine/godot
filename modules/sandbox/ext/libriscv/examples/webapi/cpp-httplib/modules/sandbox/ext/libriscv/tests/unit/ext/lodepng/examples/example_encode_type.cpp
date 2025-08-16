/*
LodePNG Examples

Copyright (c) 2005-2015 Lode Vandevenne

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

//g++ -I ./ lodepng.cpp examples/example_encode_type.cpp -ansi -pedantic -Wall -Wextra -O3



/*
This example shows how to enforce a certain color type of the PNG image when
encoding a PNG (because by default, LodePNG automatically chooses an optimal
color type, no matter what your raw data's color type is)
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

  //generate some image
  const unsigned w = 256;
  const unsigned h = 256;
  std::vector<unsigned char> image(w * h * 4);
  for(unsigned y = 0; y < h; y++)
  for(unsigned x = 0; x < w; x++) {
    int index = y * w * 4 + x * 4;
    image[index + 0] = 0;
    image[index + 1] = 0;
    image[index + 2] = 0;
    image[index + 3] = 255;
  }

  // we're going to encode with a state rather than a convenient function, because enforcing a color type requires setting options
  lodepng::State state;
  // input color type
  state.info_raw.colortype = LCT_RGBA;
  state.info_raw.bitdepth = 8;
  // output color type
  state.info_png.color.colortype = LCT_RGBA;
  state.info_png.color.bitdepth = 8;
  state.encoder.auto_convert = 0; // without this, it would ignore the output color type specified above and choose an optimal one instead

  //encode and save
  std::vector<unsigned char> buffer;
  unsigned error = lodepng::encode(buffer, &image[0], w, h, state);
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
  else lodepng::save_file(buffer, argv[1]);
}
