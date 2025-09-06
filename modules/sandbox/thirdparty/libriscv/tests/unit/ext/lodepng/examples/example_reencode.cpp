/*
LodePNG Examples

Copyright (c) 2005-2010 Lode Vandevenne

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

/*
LodePNG decode-encode: decodes the image, then encodes it again, with all the
same information, chunks, color types, etc... as the original image had.
This sample shows how LodePNG can be used for a conforming PNG editor.
*/

//g++ lodepng.cpp example_reencode.cpp -ansi -pedantic -Wall -Wextra -lSDL -O3 -o reencode

#include "lodepng.h"

#include <iostream>

int main(int argc, char *argv[]) {
  std::vector<unsigned char> image;
  unsigned w, h;
  std::vector<unsigned char> buffer;
  lodepng::State state;
  unsigned error;

  //check if user gave a filename
  if(argc < 3) {
    std::cout << "please provide in and out filename" << std::endl;
    return 0;
  }

  state.decoder.color_convert = 0;
  state.decoder.remember_unknown_chunks = 1; //make it reproduce even unknown chunks in the saved image

  lodepng::load_file(buffer, argv[1]);
  error = lodepng::decode(image, w, h, state, buffer);
  if(error) {
    std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    return 0;
  }

  buffer.clear();

  state.encoder.text_compression = 1;

  error = lodepng::encode(buffer, image, w, h, state);
  if(error) {
    std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    return 0;
  }

  lodepng::save_file(buffer, argv[2]);
}
