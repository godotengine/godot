/*
LodePNG Benchmark

Copyright (c) 2005-2023 Lode Vandevenne

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

//g++ lodepng.cpp lodepng_benchmark.cpp -Wall -Wextra -pedantic -ansi -lSDL -O3
//g++ lodepng.cpp lodepng_benchmark.cpp -Wall -Wextra -pedantic -ansi -lSDL -O3 && ./a.out

#include "lodepng.h"

#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>

#include <SDL2/SDL.h> //SDL is used for timing.

bool apply_mods = false;

#define NUM_DECODE 5 //decode multiple times to measure better. Must be at least 1.

size_t total_pixels = 0;
size_t total_png_orig_size = 0;
size_t total_raw_orig_size = 0; // This is the uncompressed data in the raw color format in the original input PNGs

double total_dec_time = 0;
size_t total_png_in_size = 0;
size_t total_raw_in_size = 0; // This is the uncompressed data in the raw color format in the input PNGs given to the decoder (not same as orig when using the encoded ones)
size_t total_raw_dec_size = 0; // This is the uncompressed data in the raw color format of raw image buffers output by the decoder

double total_enc_time = 0;
size_t total_raw_enc_size = 0; // This is the uncompressed data in the raw color format of the raw images given to the encoder
size_t total_png_out_size = 0;
size_t total_raw_out_size = 0; // This is the uncompressed data in the raw color format of the encoded PNGs

bool verbose = false;
bool do_decode = false;
bool do_encode = false;
bool decode_encoded = false; // do the decoding benchmark on the encoded images rather than the original inputs

std::string dumpdir;

////////////////////////////////////////////////////////////////////////////////

double getTime() {
  return SDL_GetTicks() / 1000.0;
}

template<typename T, typename U>
void assertEquals(const T& expected, const U& actual, const std::string& message = "") {
  if(expected != (T)actual) {
    std::cout << "Error: Not equal! Expected " << expected << " got " << actual << "." << std::endl;
    std::cout << "Message: " << message << std::endl;
    std::exit(1);
  }
}

void assertTrue(bool value, const std::string& message = "") {
  if(!value) {
    std::cout << "Error: expected true." << std::endl;
    std::cout << "Message: " << message << std::endl;
    std::exit(1);
  }
}

//Test image data
struct Image {
  std::vector<unsigned char> data;
  unsigned width;
  unsigned height;
  LodePNGColorType colorType;
  unsigned bitDepth;
  std::string name;
};

//Utility for debug messages
template<typename T>
std::string valtostr(const T& val) {
  std::ostringstream sstream;
  sstream << val;
  return sstream.str();
}

template<typename T>
void printValue(const std::string& name, const T& value, const std::string& unit = "") {
  std::cout << name << ": " << value << unit << std::endl;
}

template<typename T, typename U>
void printValue(const std::string& name, const T& value, const std::string& s2, const U& value2, const std::string& unit = "") {
  std::cout << name << ": " << value << s2 << value2 << unit << std::endl;
}

//Test LodePNG encoding and decoding the encoded result, using the C interface
std::vector<unsigned char> testEncode(Image& image) {
  unsigned char* encoded = 0;
  size_t encoded_size = 0;
  lodepng::State state;
  state.info_raw.colortype = image.colorType;
  state.info_raw.bitdepth = image.bitDepth;

  // Try custom compression settings
  if(apply_mods) {
    //state.encoder.filter_strategy = LFS_ZERO;
    //state.encoder.filter_strategy = LFS_ENTROPY;
    //state.encoder.filter_strategy = LFS_FOUR;
    //state.encoder.zlibsettings.btype = 0;
    //state.encoder.zlibsettings.btype = 1;
    //state.encoder.auto_convert = 0;
    //state.encoder.zlibsettings.use_lz77 = 0;
    state.encoder.zlibsettings.windowsize = 1;
    //state.encoder.zlibsettings.windowsize = 32768;
  }

  double t_enc0 = getTime();

  unsigned error_enc = lodepng_encode(&encoded, &encoded_size, &image.data[0],
                                      image.width, image.height, &state);

  double t_enc1 = getTime();

  assertEquals(0, error_enc, "encoder error");

  total_raw_enc_size += lodepng_get_raw_size(image.width, image.height, &state.info_raw);
  total_png_out_size += encoded_size;
  total_enc_time += (t_enc1 - t_enc0);

  if(verbose) {
    printValue("encoding time", t_enc1 - t_enc0, "s");
    std::cout << "compression: " << ((double)(encoded_size) / (double)(image.data.size())) * 100 << "%"
              << ", ratio: " << ((double)(image.data.size()) / (double)(encoded_size))
              << ", size: " << encoded_size
              << ", bpp: " << (8.0 * encoded_size / image.width / image.height) << std::endl;
  }

  if(!dumpdir.empty()) {
    std::string dumpname = dumpdir;
    if(dumpname[dumpname.size() - 1] != '/') dumpname += "/";
    dumpname += image.name;
    if(lodepng_save_file(encoded, encoded_size, dumpname.c_str())) {
      std::cout << "WARNING: failed to dump " << dumpname << ". The dir must already exist." << std::endl;
    } else if(verbose) {
      std::cout << "saved to: " << dumpname << std::endl;
    }
  }

  // output image stats
  {
    lodepng::State inspect;
    unsigned w, h;
    lodepng_inspect(&w, &h, &inspect, encoded, encoded_size);
    total_raw_out_size += lodepng_get_raw_size(w, h, &inspect.info_png.color);
  }

  std::vector<unsigned char> result(encoded, encoded + encoded_size);
  free(encoded);
  return result;
}

void testDecode(const std::vector<unsigned char>& png) {
  lodepng::State state;
  unsigned char* decoded = 0;
  unsigned w, h;

  // Try custom decompression settings
  if(apply_mods) {
    state.decoder.color_convert = 0;
    //state.decoder.ignore_crc = 1;
  }

  double t_dec0 = getTime();
  for(int i = 0; i < NUM_DECODE; i++) {
    unsigned error_dec = lodepng_decode(&decoded, &w, &h, &state, png.data(), png.size());
    assertEquals(0, error_dec, "decoder error");
  }
  double t_dec1 = getTime();

  total_dec_time += (t_dec1 - t_dec0);

  total_raw_dec_size += lodepng_get_raw_size(w, h, &state.info_raw);

  if(verbose) {
    printValue("decoding time", t_dec1 - t_dec0, "/", NUM_DECODE, " s");
  }
  free(decoded);

  // input image stats
  {
    lodepng::State inspect;
    unsigned w, h;
    lodepng_inspect(&w, &h, &inspect, png.data(), png.size());
    total_raw_in_size += lodepng_get_raw_size(w, h, &inspect.info_png.color);
    total_png_in_size += png.size();
  }
}

std::string getFilePart(const std::string& path) {
  if(path.empty()) return "";
  int slash = path.size() - 1;
  while(slash >= 0 && path[(size_t)slash] != '/') slash--;
  return path.substr((size_t)(slash + 1));
}

void testFile(const std::string& filename) {
  if(verbose) std::cout << "file " << filename << std::endl;

  std::vector<unsigned char> png;
  if(lodepng::load_file(png, filename)) {
    std::cout << "\nfailed to load file " << filename << std::endl << std::endl;
    return;
  }

  // input image stats
  {
    lodepng::State inspect;
    unsigned w, h;
    lodepng_inspect(&w, &h, &inspect, png.data(), png.size());
    total_pixels += (w * h);
    total_png_orig_size += png.size();
    size_t raw_size = lodepng_get_raw_size(w, h, &inspect.info_png.color);
    total_raw_orig_size += raw_size;
    if(verbose) std::cout << "orig compressed size: " << png.size() << ", pixels: " << (w * h) << ", raw size: " << raw_size << std::endl;
  }

  if(do_encode) {
    Image image;
    image.name = getFilePart(filename);
    image.colorType = LCT_RGBA;
    image.bitDepth = 8;
    assertEquals(0, lodepng::decode(image.data, image.width, image.height, filename, image.colorType, image.bitDepth));

    std::vector<unsigned char> temp = testEncode(image);
    if(decode_encoded) png = temp;
  }

  if(do_decode) {
    testDecode(png);
  }

  if(verbose) std::cout << std::endl;
}

void showHelp(int argc, char *argv[]) {
  (void)argc;
  std::cout << "Usage: " << argv[0] << " png_filenames... [OPTIONS...] [--dumpdir directory]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -h: show this help" << std::endl;
  std::cout << "  -v: verbose" << std::endl;
  std::cout << "  -d: decode only" << std::endl;
  std::cout << "  -e: encode only" << std::endl;
  std::cout << "  -o: decode on original images rather than encoded ones (always true if -d without -e)" << std::endl;
  std::cout << "  -m: apply modifications to encoder and decoder settings, the modification itself must be implemented or changed in the benchmark source code (search for apply_mods in the code, for encode and for decode)" << std::endl;
}

int main(int argc, char *argv[]) {
  verbose = false;
  do_decode = true;
  do_encode = true;
  decode_encoded = true;

  std::vector<std::string> files;

  for(int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if(arg == "-v") verbose = true;
    else if(arg == "-h" || arg == "--help") { showHelp(argc, argv); return 0; }
    else if(arg == "-d") do_decode ? (do_encode = false) : (do_decode = true);
    else if(arg == "-e") do_encode ? (do_decode = false) : (do_encode = true);
    else if(arg == "-o") decode_encoded = false;
    else if(arg == "-m") apply_mods = true;
    else if(arg == "--dumpdir" && i + 1 < argc) {
      dumpdir = argv[++i];
    }
    else files.push_back(arg);
  }

  if(!do_encode) decode_encoded = false;

  if(files.empty()) {
    std::cout << "must give .png filenames to benchamrk" << std::endl;
    showHelp(argc, argv);
    return 1;
  }



  for(size_t i = 0; i < files.size(); i++) {
    testFile(files[i]);
  }

  // test images stats
  if(verbose) {
    std::cout << "Final Summary: " << std::endl;
    std::cout << "input images: " << files.size() << std::endl;
    std::cout << "total_pixels: " << total_pixels << std::endl;
    // file size of original PNGs that were given as command line arguments to the tool
    std::cout << "total_png_orig_size: " << total_png_orig_size << " (" << (8.0 * total_png_orig_size / total_pixels) << " bpp)" << std::endl;
    // size of the data inside the original PNGs, dependent on color encoding (bit depth used in the PNG)
    std::cout << "total_raw_orig_size: " << total_raw_orig_size << " (" << (8.0 * total_raw_orig_size / total_pixels) << " bpp)" << std::endl;
    // size of the pixel data given to the benchmark encoder, dependent on color encoding (bit depth used in the image representation in the benchmark tool, probably 32 bits)
    if(do_encode) std::cout << "total_raw_enc_size: " << total_raw_enc_size << " (" << (8.0 * total_raw_enc_size / total_pixels) << " bpp)" << std::endl;
    // file size of PNGs created by the benchmark encoder
    if(do_encode) std::cout << "total_png_out_size: " << total_png_out_size << " (" << (8.0 * total_png_out_size / total_pixels) << " bpp)" << std::endl;
    // size of the data inside the PNGs created by the benchmark encoder, dependent on color encoding (bit depth used in the PNG), may differ from total_raw_orig_size since the encoder may choose to use a different color encoding
    if(do_encode) std::cout << "total_raw_out_size: " << total_raw_out_size << " (" << (8.0 * total_raw_out_size / total_pixels) << " bpp)" << std::endl;
    // size of file size of the PNGs given to the benchmark decoder, this could either be the original ones or the ones encoded by the benchmark encoder depending on user options
    if(do_decode) std::cout << "total_png_in_size: " << total_png_in_size << " (" << (8.0 * total_png_in_size / total_pixels) << " bpp)" << std::endl;
    // size of the data inside the PNGs mentioned at total_png_in_size, dependent on color encoding (bit depth used in the image representation in the benchmark tool, probably 32 bits)
    if(do_decode) std::cout << "total_raw_in_size: " << total_raw_in_size << " (" << (8.0 * total_raw_in_size / total_pixels) << " bpp)" << std::endl;
    // size of the pixel data requested from the benchmark decoder, dependent on color encoding requested (bit depth used in the image representation in the benchmark tool, probably 32 bits)
    if(do_decode) std::cout << "total_raw_dec_size: " << total_raw_dec_size << " (" << (8.0 * total_raw_dec_size / total_pixels) << " bpp)" << std::endl;
  }

  // final encoding stats
  if(do_encode) {
    std::cout << "encoding time: " << total_enc_time << "s on " << total_pixels << " pixels and " << total_raw_out_size << " raw bytes ("
              << ((total_raw_enc_size/1024.0/1024.0)/(total_enc_time)) << " MB/s, " << ((total_pixels/1024.0/1024.0)/(total_enc_time)) << " MP/s)" << std::endl;
    std::cout << "encoded size: " << total_png_out_size << " (" << (100.0 * total_png_out_size / total_raw_out_size) << "%), bpp: "
              << (8.0 * total_png_out_size / total_pixels) << std::endl;
  }

  // final decoding stats
  if(do_decode) {
    if(verbose) std::cout << "decoding iterations: " << NUM_DECODE << std::endl;
    std::cout << "decoding time: " << total_dec_time/NUM_DECODE << "s on " << total_pixels << " pixels and " << total_png_in_size
              << " compressed bytes (" << ((total_raw_in_size/1024.0/1024.0)/(total_dec_time/NUM_DECODE)) << " MB/s, "
              << ((total_pixels/1024.0/1024.0)/(total_dec_time/NUM_DECODE)) << " MP/s)" << std::endl;
  }
}
