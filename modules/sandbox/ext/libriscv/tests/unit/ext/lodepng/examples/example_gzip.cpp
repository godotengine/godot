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
#include <iostream>
#include <stdlib.h>

/*
Encodes given file as a gzip file.

See also the gzip specification, RFC 1952: http://www.gzip.org/zlib/rfc-gzip.html
*/

//g++ lodepng.cpp example_gzip.cpp -ansi -pedantic -Wall -Wextra -O3

//saves image to filename given as argument. Warning, this overwrites the file without warning!
int main(int argc, char *argv[]) {
  if(argc < 2) {
    std::cout << "Please provide input filename (output is input with .gz)" << std::endl;
    return 0;
  }

  //NOTE: this sample will overwrite the output file without warning!
  std::string infilename = argv[1];
  std::string outfilename = infilename + ".gz";

  std::vector<unsigned char> in;
  lodepng::load_file(in, infilename);

  size_t outsize = 10;
  unsigned char* out = (unsigned char*)malloc(outsize);
  out[0] = 31;  //ID1
  out[1] = 139; //ID2
  out[2] = 8; //CM
  out[3] = 0; //FLG
  //MTIME
  out[4] = 0;
  out[5] = 0;
  out[6] = 0;
  out[7] = 0;

  out[8] = 2; //2 = slow, 4 = fast compression
  out[9] = 255; //OS unknown

  lodepng_deflate(&out, &outsize, &in[0], in.size(), &lodepng_default_compress_settings);

  unsigned crc = lodepng_crc32(&in[0], in.size());

  size_t footer = outsize;

  outsize += 8;
  out = (unsigned char*)realloc(out, outsize);

  //CRC
  out[footer + 0] = crc % 256;
  out[footer + 1] = (crc >> 8) % 256;
  out[footer + 2] = (crc >> 16) % 256;
  out[footer + 3] = (crc >> 24) % 256;

  //ISIZE
  out[footer + 4] = in.size() % 256;
  out[footer + 5] = (in.size() >> 8) % 256;
  out[footer + 6] = (in.size() >> 16) % 256;
  out[footer + 7] = (in.size() >> 24) % 256;

  lodepng_save_file(out, outsize, outfilename.c_str());

  free(out);
}
