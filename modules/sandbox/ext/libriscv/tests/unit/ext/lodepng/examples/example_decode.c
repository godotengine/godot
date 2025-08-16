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

#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>

/*
3 ways to decode a PNG from a file to RGBA pixel data (and 2 in-memory ways).
*/

/*
Example 1
Decode from disk to raw pixels with a single function call
*/
void decodeOneStep(const char* filename) {
  unsigned error;
  unsigned char* image = 0;
  unsigned width, height;

  error = lodepng_decode32_file(&image, &width, &height, filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  /*use image here*/

  free(image);
}

/*
Example 2
Load PNG file from disk to memory first, then decode to raw pixels in memory.
*/
void decodeTwoSteps(const char* filename) {
  unsigned error;
  unsigned char* image = 0;
  unsigned width, height;
  unsigned char* png = 0;
  size_t pngsize;

  error = lodepng_load_file(&png, &pngsize, filename);
  if(!error) error = lodepng_decode32(&image, &width, &height, png, pngsize);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  free(png);

  /*use image here*/

  free(image);
}

/*
Example 3
Load PNG file from disk using a State, normally needed for more advanced usage.
*/
void decodeWithState(const char* filename) {
  unsigned error;
  unsigned char* image = 0;
  unsigned width, height;
  unsigned char* png = 0;
  size_t pngsize;
  LodePNGState state;

  lodepng_state_init(&state);
  /*optionally customize the state*/

  error = lodepng_load_file(&png, &pngsize, filename);
  if(!error) error = lodepng_decode(&image, &width, &height, &state, png, pngsize);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  free(png);

  /*use image here*/
  /*state contains extra information about the PNG such as text chunks, ...*/

  lodepng_state_cleanup(&state);
  free(image);
}

int main(int argc, char *argv[]) {
  const char* filename = argc > 1 ? argv[1] : "test.png";

  decodeOneStep(filename);

  return 0;
}

