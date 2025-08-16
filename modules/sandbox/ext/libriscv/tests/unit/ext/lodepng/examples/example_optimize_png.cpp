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

/*
This example saves the PNG with the best compression LodePNG can do, and with
unnecessary chunks removed. It tries out several combinations of settings and
keeps the smallest one.

NOTE: This is not as good as a true PNG optimizer like optipng or pngcrush.
*/

//g++ lodepng.cpp example_optimize_png.cpp -ansi -pedantic -Wall -Wextra -O3

#include "lodepng.h"

#include <iostream>

int main(int argc, char *argv[]) {
  std::vector<unsigned char> image;
  unsigned w, h;
  std::vector<unsigned char> buffer;
  unsigned error;

  //check if user gave a filename
  if(argc < 3) {
    std::cout << "please provide in and out filename" << std::endl;
    return 0;
  }

  lodepng::load_file(buffer, argv[1]);
  error = lodepng::decode(image, w, h, buffer);

  if(error) {
    std::cout << "decoding error " << error << ": " << lodepng_error_text(error) << std::endl;
    return 0;
  }

  size_t origsize = buffer.size();
  std::cout << "Original size: " << origsize << " (" << (origsize / 1024) << "K)" << std::endl;
  buffer.clear();

  //Now encode as hard as possible with several filter types and window sizes

  lodepng::State state;
  state.encoder.filter_palette_zero = 0; //We try several filter types, including zero, allow trying them all on palette images too.
  state.encoder.add_id = false; //Don't add LodePNG version chunk to save more bytes
  state.encoder.text_compression = 1; //Not needed because we don't add text chunks, but this demonstrates another optimization setting
  state.encoder.zlibsettings.nicematch = 258; //Set this to the max possible, otherwise it can hurt compression
  state.encoder.zlibsettings.lazymatching = 1; //Definitely use lazy matching for better compression
  state.encoder.zlibsettings.windowsize = 32768; //Use maximum possible window size for best compression

  size_t bestsize = 0;
  bool inited = false;

  int beststrategy = 0;
  LodePNGFilterStrategy strategies[4] = { LFS_ZERO, LFS_MINSUM, LFS_ENTROPY, LFS_BRUTE_FORCE };
  std::string strategynames[4] = { "LFS_ZERO", "LFS_MINSUM", "LFS_ENTROPY", "LFS_BRUTE_FORCE" };

  // min match 3 allows all deflate lengths. min match 6 is similar to "Z_FILTERED" of zlib.
  int minmatches[2] = { 3, 6 };
  int bestminmatch = 0;

  int autoconverts[2] = { 0, 1 };
  std::string autoconvertnames[2] = { "0", "1" };
  int bestautoconvert = 0;

  int bestblocktype = 0;

  // Try out all combinations of everything
  for(int i = 0; i < 4; i++)   //filter strategy
  for(int j = 0; j < 2; j++)   //min match
  for(int k = 0; k < 2; k++)   //block type (for small images only)
  for(int l = 0; l < 2; l++) { //color convert strategy
    if(bestsize > 3000 && (k > 0 || l > 0)) continue; /* these only make sense on small images */
    std::vector<unsigned char> temp;
    state.encoder.filter_strategy = strategies[i];
    state.encoder.zlibsettings.minmatch = minmatches[j];
    state.encoder.zlibsettings.btype = k == 0 ? 2 : 1;
    state.encoder.auto_convert = autoconverts[l];
    error = lodepng::encode(temp, image, w, h, state);

    if(error)
    {
      std::cout << "encoding error " << error << ": " << lodepng_error_text(error) << std::endl;
      return 0;
    }

    if(!inited || temp.size() < bestsize)
    {
      bestsize = temp.size();
      beststrategy = i;
      bestminmatch = state.encoder.zlibsettings.minmatch;
      bestautoconvert = l;
      bestblocktype = state.encoder.zlibsettings.btype;
      temp.swap(buffer);
      inited = true;
    }
  }

  std::cout << "Chosen filter strategy: " << strategynames[beststrategy] << std::endl;
  std::cout << "Chosen min match: " << bestminmatch << std::endl;
  std::cout << "Chosen block type: " << bestblocktype << std::endl;
  std::cout << "Chosen auto convert: " << autoconvertnames[bestautoconvert] << std::endl;

  lodepng::save_file(buffer, argv[2]);
  std::cout << "New size: " << buffer.size() << " (" << (buffer.size() / 1024) << "K)" << std::endl;
}
