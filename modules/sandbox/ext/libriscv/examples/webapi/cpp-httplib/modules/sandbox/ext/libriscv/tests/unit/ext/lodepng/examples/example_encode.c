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
3 ways to encode a PNG from RGBA pixel data to a file (and 2 in-memory ways).
NOTE: these samples overwrite the file or test.png without warning!
*/

/*
Example 1
Encode from raw pixels to disk with a single function call
The image argument has width * height RGBA pixels or width * height * 4 bytes
*/
void encodeOneStep(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
  /*Encode the image*/
  unsigned error = lodepng_encode32_file(filename, image, width, height);

  /*if there's an error, display it*/
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

/*
Example 2
Encode from raw pixels to an in-memory PNG file first, then write it to disk
The image argument has width * height RGBA pixels or width * height * 4 bytes
*/
void encodeTwoSteps(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
  unsigned char* png;
  size_t pngsize;

  unsigned error = lodepng_encode32(&png, &pngsize, image, width, height);
  if(!error) lodepng_save_file(png, pngsize, filename);

  /*if there's an error, display it*/
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  free(png);
}

/*
Example 3
Save a PNG file to disk using a State, normally needed for more advanced usage.
The image argument has width * height RGBA pixels or width * height * 4 bytes
*/
void encodeWithState(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
  unsigned error;
  unsigned char* png;
  size_t pngsize;
  LodePNGState state;

  lodepng_state_init(&state);
  /*optionally customize the state*/

  error = lodepng_encode(&png, &pngsize, image, width, height, &state);
  if(!error) lodepng_save_file(png, pngsize, filename);

  /*if there's an error, display it*/
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  lodepng_state_cleanup(&state);
  free(png);
}

int main(int argc, char *argv[]) {
  const char* filename = argc > 1 ? argv[1] : "test.png";

  /*generate some image*/
  unsigned width = 512, height = 512;
  unsigned char* image = malloc(width * height * 4);
  unsigned x, y;
  for(y = 0; y < height; y++)
  for(x = 0; x < width; x++) {
    image[4 * width * y + 4 * x + 0] = 255 * !(x & y);
    image[4 * width * y + 4 * x + 1] = x ^ y;
    image[4 * width * y + 4 * x + 2] = x | y;
    image[4 * width * y + 4 * x + 3] = 255;
  }

  /*run an example*/
  encodeOneStep(filename, image, width, height);

  free(image);
  return 0;
}
