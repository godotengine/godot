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
Load a BMP image and convert it to a PNG image. This example also shows how
to use other data with the same memory structure as BMP, such as the image
format native to win32, GDI (HBITMAP, BITMAPINFO, ...) often encountered if
you're programming for Windows in Visual Studio.

This example only supports uncompressed 24-bit RGB or 32-bit RGBA bitmaps.
For other types of BMP's, use a full fledged BMP decoder, or convert the
bitmap to 24-bit or 32-bit format.

NOTE: it overwrites the output file without warning if it exists!
*/

//g++ lodepng.cpp example_bmp2png.cpp -ansi -pedantic -Wall -Wextra -O3

#include "lodepng.h"

#include <iostream>

//returns 0 if all went ok, non-0 if error
//output image is always given in RGBA (with alpha channel), even if it's a BMP without alpha channel
unsigned decodeBMP(std::vector<unsigned char>& image, unsigned& w, unsigned& h, const std::vector<unsigned char>& bmp) {
  static const unsigned MINHEADER = 54; //minimum BMP header size

  if(bmp.size() < MINHEADER) return -1;
  if(bmp[0] != 'B' || bmp[1] != 'M') return 1; //It's not a BMP file if it doesn't start with marker 'BM'
  unsigned pixeloffset = bmp[10] + 256 * bmp[11]; //where the pixel data starts
  //read width and height from BMP header
  w = bmp[18] + bmp[19] * 256;
  h = bmp[22] + bmp[23] * 256;
  //read number of channels from BMP header
  if(bmp[28] != 24 && bmp[28] != 32) return 2; //only 24-bit and 32-bit BMPs are supported.
  unsigned numChannels = bmp[28] / 8;

  //The amount of scanline bytes is width of image times channels, with extra bytes added if needed
  //to make it a multiple of 4 bytes.
  unsigned scanlineBytes = w * numChannels;
  if(scanlineBytes % 4 != 0) scanlineBytes = (scanlineBytes / 4) * 4 + 4;

  unsigned dataSize = scanlineBytes * h;
  if(bmp.size() < dataSize + pixeloffset) return 3; //BMP file too small to contain all pixels

  image.resize(w * h * 4);

  /*
  There are 3 differences between BMP and the raw image buffer for LodePNG:
  -it's upside down
  -it's in BGR instead of RGB format (or BRGA instead of RGBA)
  -each scanline has padding bytes to make it a multiple of 4 if needed
  The 2D for loop below does all these 3 conversions at once.
  */
  for(unsigned y = 0; y < h; y++)
  for(unsigned x = 0; x < w; x++) {
    //pixel start byte position in the BMP
    unsigned bmpos = pixeloffset + (h - y - 1) * scanlineBytes + numChannels * x;
    //pixel start byte position in the new raw image
    unsigned newpos = 4 * y * w + 4 * x;
    if(numChannels == 3) {
      image[newpos + 0] = bmp[bmpos + 2]; //R
      image[newpos + 1] = bmp[bmpos + 1]; //G
      image[newpos + 2] = bmp[bmpos + 0]; //B
      image[newpos + 3] = 255;            //A
    } else {
      image[newpos + 0] = bmp[bmpos + 2]; //R
      image[newpos + 1] = bmp[bmpos + 1]; //G
      image[newpos + 2] = bmp[bmpos + 0]; //B
      image[newpos + 3] = bmp[bmpos + 3]; //A
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  if(argc < 3) {
    std::cout << "Please provide input BMP and output PNG file names" << std::endl;
    return 0;
  }

  std::vector<unsigned char> bmp;
  lodepng::load_file(bmp, argv[1]);
  std::vector<unsigned char> image;
  unsigned w, h;
  unsigned error = decodeBMP(image, w, h, bmp);

  if(error) {
    std::cout << "BMP decoding error " << error << std::endl;
    return 0;
  }

  std::vector<unsigned char> png;
  error = lodepng::encode(png, image, w, h);

  if(error) {
    std::cout << "PNG encoding error " << error << ": " << lodepng_error_text(error) << std::endl;
    return 0;
  }

  lodepng::save_file(png, argv[2]);

}
