/*
LodePNG Unit Test

Copyright (c) 2005-2025 Lode Vandevenne

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

//g++ lodepng.cpp lodepng_util.cpp lodepng_unittest.cpp -Wall -Wextra -Wsign-conversion -pedantic -ansi -O3

/*
Testing instructions:

*) Ensure no tests commented out below or early return in doMain

*) Compile with g++ with all warnings and run the unit test
g++ lodepng.cpp lodepng_util.cpp lodepng_unittest.cpp -Werror -Wall -Wextra -Wsign-conversion -Wshadow -pedantic -ansi -O3 && ./a.out

*) Compile with clang, which may sometimes give different warnings
clang++ lodepng.cpp -c -Werror -Wall -Wextra -Wsign-conversion -Wshorten-64-to-32 -Wshadow -pedantic -ansi -O3

*) Compile with pure ISO C90 and all warnings:
mv lodepng.cpp lodepng.c ; gcc -I ./ lodepng.c examples/example_decode.c -ansi -pedantic -Werror -Wall -Wextra -O3 ; mv lodepng.c lodepng.cpp

mv lodepng.cpp lodepng.c ; clang -I ./ lodepng.c examples/example_decode.c -ansi -pedantic -Werror -Wall -Wextra -O3 ; mv lodepng.c lodepng.cpp

*) Compile with C with -pedantic but not -ansi flag so it warns about // style comments in C++-only ifdefs
mv lodepng.cpp lodepng.c ; gcc -I ./ lodepng.c examples/example_decode.c -pedantic -Werror -Wall -Wextra -O3 ; mv lodepng.c lodepng.cpp

*) test other compilers

*) try lodepng_benchmark.cpp
g++ lodepng.cpp lodepng_benchmark.cpp -Werror -Wall -Wextra -pedantic -ansi -lSDL2 -O3 && ./a.out testdata/corpus/''*

*) try the fuzzer
clang++ -fsanitize=fuzzer -DLODEPNG_MAX_ALLOC=100000000 lodepng.cpp lodepng_fuzzer.cpp -O3 -o fuzzer && ./fuzzer

clang++ -fsanitize=fuzzer,address,undefined -DLODEPNG_MAX_ALLOC=100000000 lodepng.cpp lodepng_fuzzer.cpp -O3 -o fuzzer && ./fuzzer

*) Check if all C++ examples compile without warnings:
g++ -I ./ lodepng.cpp examples/''*.cpp -Werror -W -Wall -ansi -pedantic -O3 -c

*) Check if all C examples compile without warnings:
mv lodepng.cpp lodepng.c ; gcc -I ./ lodepng.c examples/''*.c -Werror -W -Wall -ansi -pedantic -O3 -c ; mv lodepng.c lodepng.cpp

*) Check pngdetail.cpp:
g++ lodepng.cpp lodepng_util.cpp pngdetail.cpp -Werror -W -Wall -ansi -pedantic -O3 -o pngdetail
./pngdetail testdata/PngSuite/basi0g01.png

*) Test compiling with some code sections with #defines disabled, for unused static function warnings etc...
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_CRC
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_ZLIB
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_PNG
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_DECODER
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_ENCODER
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_DISK
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_ANCILLARY_CHUNKS
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_ERROR_TEXT
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_CPP
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_ZLIB -DLODEPNG_NO_COMPILE_DECODER
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_ZLIB -DLODEPNG_NO_COMPILE_ENCODER
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_PNG -DLODEPNG_NO_COMPILE_DECODER
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_PNG -DLODEPNG_NO_COMPILE_ENCODER
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_DECODER -DLODEPNG_NO_COMPILE_ANCILLARY_CHUNKS -DLODEPNG_NO_COMPILE_ERROR_TEXT -DLODEPNG_NO_COMPILE_DISK
g++ lodepng.cpp -W -Wall -ansi -pedantic -O3 -c -DLODEPNG_NO_COMPILE_ENCODER -DLODEPNG_NO_COMPILE_ANCILLARY_CHUNKS -DLODEPNG_NO_COMPILE_ERROR_TEXT -DLODEPNG_NO_COMPILE_DISK
rm *.o

*) analyze with clang:
clang++ lodepng.cpp --analyze

More verbose:
clang++ --analyze -Xanalyzer -analyzer-output=text lodepng.cpp

Or html, look under lodepng.plist dir afterwards and find the numbered locations in the pages:
clang++ --analyze -Xanalyzer -analyzer-output=html lodepng.cpp

*) check for memory leaks and vulnerabilities with valgrind
(DISABLE_SLOW disables a few tests that are very slow with valgrind)
g++ -DDISABLE_SLOW lodepng.cpp lodepng_util.cpp lodepng_unittest.cpp -Wall -Wextra -pedantic -ansi -O3 -DLODEPNG_MAX_ALLOC=100000000 && valgrind --leak-check=full --track-origins=yes ./a.out

*) Try with clang++ and address sanitizer (to get line numbers, make sure 'llvm' is also installed to get 'llvm-symbolizer'
clang++ -O3 -fsanitize=address,undefined lodepng.cpp lodepng_util.cpp lodepng_unittest.cpp -Werror -Wall -Wextra -Wshadow -pedantic -ansi && ASAN_OPTIONS=allocator_may_return_null=1 ./a.out

clang++ -g3 -fsanitize=address,undefined lodepng.cpp lodepng_util.cpp lodepng_unittest.cpp -Werror -Wall -Wextra -Wshadow -pedantic -ansi && ASAN_OPTIONS=allocator_may_return_null=1 ./a.out

*) remove "#include <iostream>" from lodepng.cpp if it's still in there (some are legit)
cat lodepng.cpp lodepng_util.cpp | grep iostream
cat lodepng.cpp lodepng_util.cpp | grep stdio
cat lodepng.cpp lodepng_util.cpp | grep "#include"

*) try the Makefile
make clean && make -j
rm *.o *.obj

*) check that no plain free, malloc, realloc, strlen, memcpy, memset, ... used, but the lodepng_* versions instead

*) check version dates in copyright message and LODEPNG_VERSION_STRING

*) check year in copyright message at top of all files

*) check examples/sdl.cpp with the png test suite images (the "x" ones are expected to show error)
g++ -I ./ lodepng.cpp examples/example_sdl.cpp -Werror -Wall -Wextra -pedantic -ansi -O3 -lSDL2 -o showpng && ./showpng testdata/PngSuite/''*.png

*) strip trailing spaces and ensure consistent newlines

*) test warnings in other compilers

*) check diff of lodepng.cpp and lodepng.h before submitting
git difftool -y

*/

#include "lodepng.h"
#include "lodepng_util.h"

#include <cmath>
#include <map>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////

void fail() {
  throw 1; //that's how to let a unittest fail
}

//Utility for debug messages
template<typename T>
std::string valtostr(const T& val) {
  std::ostringstream sstream;
  sstream << val;
  return sstream.str();
}

//Print char as a numeric value rather than a character
template<>
std::string valtostr(const unsigned char& val) {
  std::ostringstream sstream;
  sstream << (int)val;
  return sstream.str();
}

//Print char pointer as pointer, not as string
template<typename T>
std::string valtostr(const T* val) {
  std::ostringstream sstream;
  sstream << (const void*)val;
  return sstream.str();
}

template<typename T>
std::string valtostr(const std::vector<T>& val) {
  std::ostringstream sstream;
  sstream << "[vector with size " << val.size() << "]";
  return sstream.str();
}

// TODO: remove, use only ASSERT_EQUALS (it prints line number). Requires adding extra message ability to ASSERT_EQUALS
template<typename T, typename U>
void assertEquals(const T& expected, const U& actual, const std::string& message = "") {
  if(expected != (T)actual) {
    std::cout << "Error: Not equal! Expected " << valtostr(expected)
              << " got " << valtostr((T)actual) << ". "
              << "Message: " << message << std::endl;
    fail();
  }
}

// TODO: remove, use only ASSERT_TRUE (it prints line number). Requires adding extra message ability to ASSERT_TRUE
void assertTrue(bool value, const std::string& message = "") {
  if(!value) {
    std::cout << "Error: expected true. " << "Message: " << message << std::endl;
    fail();
  }
}

//assert that no error
void assertNoPNGError(unsigned error, const std::string& message = "") {
  if(error) {
    std::string msg = (message == "") ? lodepng_error_text(error)
                                      : message + std::string(": ") + lodepng_error_text(error);
    assertEquals(0, error, msg);
  }
}

void assertNoError(unsigned error) {
  if(error) {
    assertEquals(0, error, "Expected no error");
  }
}

#define STR_EXPAND(s) #s
#define STR(s) STR_EXPAND(s)
#define ASSERT_TRUE(v) {\
  if(!(v)) {\
    std::cout << std::string("line ") + STR(__LINE__) + ": " + STR(v) + " ASSERT_TRUE failed: ";\
    std::cout << "Expected true but got " << valtostr(v) << ". " << std::endl;\
    fail();\
  }\
}
#define ASSERT_EQUALS(e, v) {\
  if((e) != (v)) {\
    std::cout << std::string("line ") + STR(__LINE__) + ": " + STR(v) + " ASSERT_EQUALS failed: ";\
    std::cout << "Expected " << valtostr(e) << " but got " << valtostr(v) << ". " << std::endl;\
    fail();\
  }\
}
#define ASSERT_NOT_EQUALS(e, v) {\
  if((e) == (v)) {\
    std::cout << std::string("line ") + STR(__LINE__) + ": " + STR(v) + " ASSERT_NOT_EQUALS failed: ";\
    std::cout << "Expected not " << valtostr(e) << " but got " << valtostr(v) << ". " << std::endl;\
    fail();\
  }\
}

template<typename T, typename U, typename V>
bool isNear(T e, U v, V maxdist) {
  T dist = e > (T)v ? e - (T)v : (T)v - e;
  return dist <= (T)maxdist;
}

template<typename T, typename U>
T diff(T e, U v) {
  return v > e ? v - e : e - v;
}

#define ASSERT_NEAR(e, v, maxdist) {\
  if(!isNear(e, v, maxdist)) {\
    std::cout << std::string("line ") + STR(__LINE__) + ": " + STR(v) + " ASSERT_NEAR failed: ";\
    std::cout << "dist too great! Expected near " << valtostr(e) << " but got " << valtostr(v) << ", with max dist " << valtostr(maxdist)\
              << " but got dist " << valtostr(diff(e, v)) << ". " << std::endl;\
    fail();\
  }\
}

#define ASSERT_STRING_EQUALS(e, v) ASSERT_EQUALS(std::string(e), std::string(v))
#define ASSERT_NO_PNG_ERROR_MSG(error, message) assertNoPNGError(error, std::string("line ") + STR(__LINE__) + (std::string(message).empty() ? std::string("") : (": " + std::string(message))))
#define ASSERT_NO_PNG_ERROR(error) ASSERT_NO_PNG_ERROR_MSG(error, std::string(""))

static const std::string BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";



//T and U can be std::string or std::vector<unsigned char>
template<typename T, typename U>
void toBase64(T& out, const U& in) {
  for(size_t i = 0; i < in.size(); i += 3) {
    int v = 65536 * in[i];
    if(i + 1 < in.size()) v += 256 * in[i + 1];
    if(i + 2 < in.size()) v += in[i + 2];
    out.push_back(BASE64[(v >> 18) & 0x3f]);
    out.push_back(BASE64[(v >> 12) & 0x3f]);
    if(i + 1 < in.size()) out.push_back(BASE64[(v >> 6) & 0x3f]);
    else out.push_back('=');
    if(i + 2 < in.size()) out.push_back(BASE64[(v >> 0) & 0x3f]);
    else out.push_back('=');
  }
}

int fromBase64(int v) {
  if(v >= 'A' && v <= 'Z') return (v - 'A');
  if(v >= 'a' && v <= 'z') return (v - 'a' + 26);
  if(v >= '0' && v <= '9') return (v - '0' + 52);
  if(v == '+') return 62;
  if(v == '/') return 63;
  return 0; //v == '='
}

//T and U can be std::string or std::vector<unsigned char>
template<typename T, typename U>
void fromBase64(T& out, const U& in) {
  for(size_t i = 0; i + 3 < in.size(); i += 4) {
    int v = 262144 * fromBase64(in[i]) + 4096 * fromBase64(in[i + 1]) + 64 * fromBase64(in[i + 2]) + fromBase64(in[i + 3]);
    out.push_back((v >> 16) & 0xff);
    if(in[i + 2] != '=') out.push_back((v >> 8) & 0xff);
    if(in[i + 3] != '=') out.push_back((v >> 0) & 0xff);
  }
}

unsigned getRandom() {
  static unsigned s = 1000000000;
  // xorshift32, good enough for testing
  s ^= (s << 13);
  s ^= (s >> 17);
  s ^= (s << 5);
  return s;
}

////////////////////////////////////////////////////////////////////////////////


unsigned leftrotate(unsigned x, unsigned c) {
  return (x << c) | (x >> (32u - c));
}

// the 128-bit result is output in 4 32-bit integers a0..d0 (to make 16-byte digest: append a0|b0|c0|d0 in little endian)
void md5sum(const unsigned char* in, size_t size, unsigned* a0, unsigned* b0, unsigned* c0, unsigned* d0) {
  ASSERT_EQUALS(4, sizeof(unsigned));
  // per-round shift amounts
  static const unsigned s[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  };
  // precomputed table from sines
  static const unsigned k[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
  };

  *a0 = 0x67452301;
  *b0 = 0xefcdab89;
  *c0 = 0x98badcfe;
  *d0 = 0x10325476;

  // append bit, padding and size to input
  std::vector<unsigned char> data(in, in + size);
  data.resize(((size + 1 + 8 + 63) / 64) * 64, 0);
  data[size] = 128; // append 1 bit (msb)
  size_t bitsize = size * 8; // append the size (shifts > 31 are avoided)
  data[data.size() - 1] = ((bitsize >> 28u) >> 28u) & 255u;
  data[data.size() - 2] = ((bitsize >> 24u) >> 24u) & 255u;
  data[data.size() - 3] = ((bitsize >> 20u) >> 20u) & 255u;
  data[data.size() - 4] = ((bitsize >> 16u) >> 16u) & 255u;
  data[data.size() - 5] = (bitsize >> 24u) & 255u;
  data[data.size() - 6] = (bitsize >> 16u) & 255u;
  data[data.size() - 7] = (bitsize >> 8u) & 255u;
  data[data.size() - 8] = bitsize & 255u;

  // per chunk
  for(size_t i = 0; i < data.size(); i += 64) {
    unsigned a = *a0;
    unsigned b = *b0;
    unsigned c = *c0;
    unsigned d = *d0;

    for(size_t j = 0; j < 64; j++) {
      unsigned f, g;
      if(j <= 15u) {
        f = (b & c) | (~b & d);
        g = j;
      } else if(j <= 31u) {
        f = (d & b) | (~d & c);
        g = (5u * j + 1u) & 15u;
      } else if(j <= 47u) {
        f = b ^ c ^ d;
        g = (3u * j + 5u) & 15u;
      } else {
        f = c ^ (b | ~d);
        g = (7u * j) & 15u;
      }
      unsigned m = (unsigned)(data[i + g * 4 + 3] << 24u) | (unsigned)(data[i + g * 4 + 2] << 16u)
                 | (unsigned)(data[i + g * 4 + 1] << 8u) | (unsigned)data[i + g * 4];
      f += a + k[j] + m;
      a = d;
      d = c;
      c = b;
      b += leftrotate(f, s[j]);
    }
    *a0 += a;
    *b0 += b;
    *c0 += c;
    *d0 += d;
  }
}

std::string md5sum(const unsigned char* data, size_t size) {
  unsigned a0, b0, c0, d0;
  md5sum(data, size, &a0, &b0, &c0, &d0);
  char result[33];
  //sprintf(result, "%8.8x%8.8x%8.8x%8.8x", a0, b0, c0, d0);
  sprintf(result, "%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x",
          a0 & 255, (a0 >> 8) & 255, (a0 >> 16) & 255, (a0 >> 24) & 255,
          b0 & 255, (b0 >> 8) & 255, (b0 >> 16) & 255, (b0 >> 24) & 255,
          c0 & 255, (c0 >> 8) & 255, (c0 >> 16) & 255, (c0 >> 24) & 255,
          d0 & 255, (d0 >> 8) & 255, (d0 >> 16) & 255, (d0 >> 24) & 255);
  return std::string(result);
}

std::string md5sum(const std::vector<unsigned char>& in) {
  return md5sum(in.data(), in.size());
}

////////////////////////////////////////////////////////////////////////////////

//Test image data
struct Image {
  std::vector<unsigned char> data;
  unsigned width;
  unsigned height;
  LodePNGColorType colorType;
  unsigned bitDepth;
};

//Get number of color channels for a given PNG color type
unsigned getNumColorChannels(unsigned colorType) {
  switch(colorType) {
    case 0: return 1; /*gray*/
    case 2: return 3; /*RGB*/
    case 3: return 1; /*palette*/
    case 4: return 2; /*gray + alpha*/
    case 6: return 4; /*RGBA*/
  }
  return 0; /*unexisting color type*/
}

//Generate a test image with some data in it, the contents of the data is unspecified,
//except the content is not just one plain color, and not true random either to be compressible.
void generateTestImage(Image& image, unsigned width, unsigned height, LodePNGColorType colorType = LCT_RGBA, unsigned bitDepth = 8) {
  image.width = width;
  image.height = height;
  image.colorType = colorType;
  image.bitDepth = bitDepth;

  size_t bits = bitDepth * getNumColorChannels(colorType); //bits per pixel
  size_t size = (width * height * bits + 7) / 8; //total image size in bytes
  image.data.resize(size);
  unsigned char value = 128;
  for(size_t i = 0; i < size; i++) {
    image.data[i] = value++;
  }
}

//Generate a 16-bit test image with minimal size that requires at minimum the given color type (bit depth, grayscaleness, ...)
//If key is true, makes it such that exactly one color is transparent, so it can use a key. If false, adds a translucent color depending on
//whether it's an alpha color type or not.
void generateTestImageRequiringColorType16(Image& image, LodePNGColorType colorType, unsigned bitDepth, bool key) {
  image.colorType = colorType;
  image.bitDepth = bitDepth;
  unsigned w = 1;
  unsigned h = 1;

  bool gray = colorType == LCT_GREY || colorType == LCT_GREY_ALPHA;
  bool alpha = colorType == LCT_RGBA || colorType == LCT_GREY_ALPHA;

  if(colorType == LCT_PALETTE) {
    w = 1u << bitDepth;
    h = 256; // ensure it'll really choose palette, not omit it due to small image size
    image.data.resize(w * h * 8);
    for(size_t y = 0; y < h; y++) {
      for(size_t x = 0; x < w; x++) {
        size_t i = y * w * 8 + x * 8;
        image.data[i + 0] = image.data[i + 1] = y;
        image.data[i + 2] = image.data[i + 3] = 255;
        image.data[i + 4] = image.data[i + 5] = 0;
        image.data[i + 6] = image.data[i + 7] = (key && y == 0) ? 0 : 255;
      }
    }
  } else if(bitDepth == 16) {
    // one color suffices for this model. But add one more to support key.
    w = 2;
    image.data.resize(w * h * 8);
    image.data[0] = 10; image.data[1] = 20;
    image.data[2] = 10; image.data[3] = 20;
    image.data[4] = gray ? 10 : 110; image.data[5] = gray ? 20 : 120;
    image.data[6] = alpha ? 128 : 255; image.data[7] = alpha ? 20 : 255;

    image.data[8] = 40; image.data[9] = 50;
    image.data[10] = 40; image.data[11] = 50;
    image.data[12] = gray ? 40 : 140; image.data[13] = gray ? 50 : 150;
    image.data[14] = key ? 0 : 255; image.data[15] = key ? 0 : 255;
  } else if(gray) {
    w = 2;
    unsigned v = 255u / ((1u << bitDepth) - 1u); // value that forces at least this bitdepth
    image.data.resize(w * h * 8);
    image.data[0] = v; image.data[1] = v;
    image.data[2] = v; image.data[3] = v;
    image.data[4] = v; image.data[5] = v;
    image.data[6] = alpha ? v : 255; image.data[7] = alpha ? v : 255;

    image.data[8] = image.data[9] = 0;
    image.data[10] = image.data[11] = 0;
    image.data[12] = image.data[13] = 0;
    image.data[14] = image.data[15] = key ? 0 : 255;
  } else {
    // now it's RGB or RGBA with bitdepth 8
    w = 257; // must have at least more than 256 colors so it won't use palette
    image.data.resize(w * h * 8);
    for(size_t y = 0; y < h; y++) {
      for(size_t x = 0; x < w; x++) {
        size_t i = y * w * 8 + x * 8;
        image.data[i + 0] = image.data[i + 1] = i / 2;
        image.data[i + 2] = image.data[i + 3] = i / 3;
        image.data[i + 4] = image.data[i + 5] = i / 5;
        image.data[i + 6] = image.data[i + 7] = (key && y == 0) ? 0 : (alpha ? i : 255);
      }
    }
  }

  image.width = w;
  image.height = h;
}

//Generate a 8-bit test image with minimal size that requires at minimum the given color type (bit depth, grayscaleness, ...). bitDepth max 8 here.
//If key is true, makes it such that exactly one color is transparent, so it can use a key. If false, adds a translucent color depending on
//whether it's an alpha color type or not.
void generateTestImageRequiringColorType8(Image& image, LodePNGColorType colorType, unsigned bitDepth, bool key) {
  image.colorType = colorType;
  image.bitDepth = bitDepth;
  unsigned w = 1;
  unsigned h = 1;

  bool gray = colorType == LCT_GREY || colorType == LCT_GREY_ALPHA;
  bool alpha = colorType == LCT_RGBA || colorType == LCT_GREY_ALPHA;

  if(colorType == LCT_PALETTE) {
    w = 1u << bitDepth;
    h = 256; // ensure it'll really choose palette, not omit it due to small image size
    image.data.resize(w * h * 4);
    for(size_t y = 0; y < h; y++) {
      for(size_t x = 0; x < w; x++) {
        size_t i = y * w * 4 + x * 4;
        image.data[i + 0] = x;
        image.data[i + 1] = 255;
        image.data[i + 2] = 0;
        image.data[i + 3] = (key && x == 0) ? 0 : 255;
      }
    }
  } else if(gray) {
    w = 2;
    unsigned v = 255u / ((1u << bitDepth) - 1u); // value that forces at least this bitdepth
    image.data.resize(w * h * 4);
    image.data[0] = v;
    image.data[1] = v;
    image.data[2] = v;
    image.data[3] = alpha ? v : 255;

    image.data[4] = 0;
    image.data[5] = 0;
    image.data[6] = 0;
    image.data[7] = key ? 0 : 255;
  } else {
    // now it's RGB or RGBA with bitdepth 8
    w = 257; // must have at least more than 256 colors so it won't use palette
    image.data.resize(w * h * 4);
    for(size_t y = 0; y < h; y++) {
      for(size_t x = 0; x < w; x++) {
        size_t i = y * w * 4 + x * 4;
        image.data[i + 0] = i / 2;
        image.data[i + 1] = i / 3;
        image.data[i + 2] = i / 5;
        image.data[i + 3] = (key && x == 0) ? 0 : (alpha ? i : 255);
      }
    }
  }

  image.width = w;
  image.height = h;
}

//Check that the decoded PNG pixels are the same as the pixels in the image
void assertPixels(Image& image, const unsigned char* decoded, const std::string& message) {
  for(size_t i = 0; i < image.data.size(); i++) {
    int byte_expected = image.data[i];
    int byte_actual = decoded[i];

    //last byte is special due to possible random padding bits which need not to be equal
    if(i == image.data.size() - 1) {
      size_t numbits = getNumColorChannels(image.colorType) * image.bitDepth * image.width * image.height;
      size_t padding = 8u - (numbits - 8u * (numbits / 8u));
      if(padding != 8u) {
        //set all padding bits of both to 0
        for(size_t j = 0; j < padding; j++) {
          byte_expected = (byte_expected & (~(1 << j))) % 256;
          byte_actual = (byte_actual & (~(1 << j))) % 256;
        }
      }
    }

    assertEquals(byte_expected, byte_actual, message + " " + valtostr(i));
  }
}

//Test LodePNG encoding and decoding the encoded result, using the C interface
void doCodecTestC(Image& image) {
  unsigned char* encoded = 0;
  size_t encoded_size = 0;
  unsigned char* decoded = 0;
  unsigned decoded_w;
  unsigned decoded_h;

  struct OnExitScope {
    unsigned char** a;
    unsigned char** b;
    OnExitScope(unsigned char** ca, unsigned char** cb) : a(ca), b(cb) {}
    ~OnExitScope() { free(*a); free(*b); }
  } onExitScope(&encoded, &decoded);

  unsigned error_enc = lodepng_encode_memory(&encoded, &encoded_size, &image.data[0],
                                             image.width, image.height, image.colorType, image.bitDepth);

  if(error_enc != 0) std::cout << "Error: " << lodepng_error_text(error_enc) << std::endl;
  ASSERT_NO_PNG_ERROR_MSG(error_enc, "encoder error C");

  //if the image is large enough, compressing it should result in smaller size
  if(image.data.size() > 512) assertTrue(encoded_size < image.data.size(), "compressed size");

  unsigned error_dec = lodepng_decode_memory(&decoded, &decoded_w, &decoded_h,
                                             encoded, encoded_size, image.colorType, image.bitDepth);

  if(error_dec != 0) std::cout << "Error: " << lodepng_error_text(error_dec) << std::endl;
  ASSERT_NO_PNG_ERROR_MSG(error_dec, "decoder error C");

  ASSERT_EQUALS(image.width, decoded_w);
  ASSERT_EQUALS(image.height, decoded_h);
  assertPixels(image, decoded, "Pixels C");
}

//Test LodePNG encoding and decoding the encoded result, using the C++ interface
void doCodecTestCPP(Image& image) {
  std::vector<unsigned char> encoded;
  std::vector<unsigned char> decoded;
  unsigned decoded_w;
  unsigned decoded_h;

  unsigned error_enc = lodepng::encode(encoded, image.data, image.width, image.height,
                                       image.colorType, image.bitDepth);

  ASSERT_NO_PNG_ERROR_MSG(error_enc, "encoder error C++");

  //if the image is large enough, compressing it should result in smaller size
  if(image.data.size() > 512) assertTrue(encoded.size() < image.data.size(), "compressed size");

  unsigned error_dec = lodepng::decode(decoded, decoded_w, decoded_h, encoded, image.colorType, image.bitDepth);

  ASSERT_NO_PNG_ERROR_MSG(error_dec, "decoder error C++");

  ASSERT_EQUALS(image.width, decoded_w);
  ASSERT_EQUALS(image.height, decoded_h);
  ASSERT_EQUALS(image.data.size(), decoded.size());
  assertPixels(image, &decoded[0], "Pixels C++");
}


void doCodecTestWithEncState(Image& image, lodepng::State& state) {
  std::vector<unsigned char> encoded;
  std::vector<unsigned char> decoded;
  unsigned decoded_w;
  unsigned decoded_h;
  state.info_raw.colortype = image.colorType;
  state.info_raw.bitdepth = image.bitDepth;


  unsigned error_enc = lodepng::encode(encoded, image.data, image.width, image.height, state);
  ASSERT_NO_PNG_ERROR_MSG(error_enc, "encoder error uncompressed");

  unsigned error_dec = lodepng::decode(decoded, decoded_w, decoded_h, encoded, image.colorType, image.bitDepth);

  ASSERT_NO_PNG_ERROR_MSG(error_dec, "decoder error uncompressed");

  ASSERT_EQUALS(image.width, decoded_w);
  ASSERT_EQUALS(image.height, decoded_h);
  ASSERT_EQUALS(image.data.size(), decoded.size());
  assertPixels(image, &decoded[0], "Pixels uncompressed");
}


//Test LodePNG encoding and decoding the encoded result, using the C++ interface
void doCodecTestUncompressed(Image& image) {
  lodepng::State state;
  state.encoder.zlibsettings.btype = 0;
  doCodecTestWithEncState(image, state);
}

void doCodecTestNoLZ77(Image& image) {
  lodepng::State state;
  state.encoder.zlibsettings.use_lz77 = 0;
  doCodecTestWithEncState(image, state);
}

void testGetFilterTypes() {
  std::cout << "testGetFilterTypes" << std::endl;
  // Test that getFilterTypes works on the special case of 1-pixel wide interlaced image
  std::string png64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAHCAIAAAExKYBVAAAAHUlEQVR4ASXHAQoAAAjCwPX/R9tK4ZBN4EHKcPcLXCgGAQa0TV8AAAAASUVORK5CYII=";
  std::vector<unsigned char> png;
  fromBase64(png, png64);
  std::vector<unsigned char> types;
  lodepng::getFilterTypes(types, png);
  ASSERT_EQUALS(7, types.size());
  ASSERT_EQUALS(1, types[0]);
  ASSERT_EQUALS(1, types[1]);
  ASSERT_EQUALS(1, types[2]);
  ASSERT_EQUALS(0, types[3]);
  ASSERT_EQUALS(1, types[4]);
  ASSERT_EQUALS(1, types[5]);
  ASSERT_EQUALS(1, types[6]);
}

//Test LodePNG encoding and decoding the encoded result, using the C++ interface, with interlace
void doCodecTestInterlaced(Image& image) {
  std::vector<unsigned char> encoded;
  std::vector<unsigned char> decoded;
  unsigned decoded_w;
  unsigned decoded_h;

  lodepng::State state;
  state.info_png.interlace_method = 1;
  state.info_raw.colortype = image.colorType;
  state.info_raw.bitdepth = image.bitDepth;

  unsigned error_enc = lodepng::encode(encoded, image.data, image.width, image.height, state);

  ASSERT_NO_PNG_ERROR_MSG(error_enc, "encoder error interlaced");

  //if the image is large enough, compressing it should result in smaller size
  if(image.data.size() > 512) assertTrue(encoded.size() < image.data.size(), "compressed size");

  state.info_raw.colortype = image.colorType;
  state.info_raw.bitdepth = image.bitDepth;
  unsigned error_dec = lodepng::decode(decoded, decoded_w, decoded_h, state, encoded);

  ASSERT_NO_PNG_ERROR_MSG(error_dec, "decoder error interlaced");

  ASSERT_EQUALS(image.width, decoded_w);
  ASSERT_EQUALS(image.height, decoded_h);
  ASSERT_EQUALS(image.data.size(), decoded.size());
  assertPixels(image, &decoded[0], "Pixels interlaced");
}

//Test LodePNG encoding and decoding the encoded result
void doCodecTest(Image& image) {
  doCodecTestC(image);
  doCodecTestCPP(image);
  doCodecTestInterlaced(image);
  doCodecTestUncompressed(image);
  doCodecTestNoLZ77(image);
}


//Test LodePNG encoding and decoding using some image generated with the given parameters
void codecTest(unsigned width, unsigned height, LodePNGColorType colorType = LCT_RGBA, unsigned bitDepth = 8) {
  std::cout << "codec test " << width << " " << height << std::endl;
  Image image;
  generateTestImage(image, width, height, colorType, bitDepth);
  doCodecTest(image);
}

std::string removeSpaces(const std::string& s) {
  std::string result;
  for(size_t i = 0; i < s.size(); i++) if(s[i] != ' ') result += s[i];
  return result;
}

void bitStringToBytes(std::vector<unsigned char>& bytes, const std::string& bits_) {
  std::string bits = removeSpaces(bits_);
  bytes.resize((bits.size()) + 7 / 8);
  for(size_t i = 0; i < bits.size(); i++) {
    size_t j = i / 8;
    size_t k = i % 8;
    char c = bits[i];
    if(k == 0) bytes[j] = 0;
    if(c == '1') bytes[j] |= (1 << (7 - k));
  }
}

/*
test color convert on a single pixel. Testing palette and testing color keys is
not supported by this function. Pixel values given using bits in an std::string
of 0's and 1's.
*/
void colorConvertTest(const std::string& bits_in, LodePNGColorType colorType_in, unsigned bitDepth_in,
                      const std::string& bits_out, LodePNGColorType colorType_out, unsigned bitDepth_out) {
  std::cout << "color convert test " << bits_in << " - " << bits_out << std::endl;

  std::vector<unsigned char> expected, actual, image;
  bitStringToBytes(expected, bits_out);
  actual.resize(expected.size());
  bitStringToBytes(image, bits_in);
  LodePNGColorMode mode_in, mode_out;
  lodepng_color_mode_init(&mode_in);
  lodepng_color_mode_init(&mode_out);
  mode_in.colortype = colorType_in;
  mode_in.bitdepth = bitDepth_in;
  mode_out.colortype = colorType_out;
  mode_out.bitdepth = bitDepth_out;
  unsigned error = lodepng_convert(&actual[0], &image[0], &mode_out, &mode_in, 1, 1);

  ASSERT_NO_PNG_ERROR_MSG(error, "convert error");

  for(size_t i = 0; i < expected.size(); i++) {
    assertEquals((int)expected[i], (int)actual[i], "byte " + valtostr(i));
  }

  lodepng_color_mode_cleanup(&mode_in);
  lodepng_color_mode_cleanup(&mode_out);
}

void testOtherPattern1() {
  std::cout << "codec other pattern 1" << std::endl;

  Image image1;
  size_t w = 192;
  size_t h = 192;
  image1.width = w;
  image1.height = h;
  image1.colorType = LCT_RGBA;
  image1.bitDepth = 8;
  image1.data.resize(w * h * 4u);
  for(size_t y = 0; y < h; y++)
  for(size_t x = 0; x < w; x++) {
    //pattern 1
    image1.data[4u * w * y + 4u * x + 0u] = (unsigned char)(127 * (1 + std::sin((                    x * x +                     y * y) / (w * h / 8.0))));
    image1.data[4u * w * y + 4u * x + 1u] = (unsigned char)(127 * (1 + std::sin(((w - x - 1) * (w - x - 1) +                     y * y) / (w * h / 8.0))));
    image1.data[4u * w * y + 4u * x + 2u] = (unsigned char)(127 * (1 + std::sin((                    x * x + (h - y - 1) * (h - y - 1)) / (w * h / 8.0))));
    image1.data[4u * w * y + 4u * x + 3u] = (unsigned char)(127 * (1 + std::sin(((w - x - 1) * (w - x - 1) + (h - y - 1) * (h - y - 1)) / (w * h / 8.0))));
  }

  doCodecTest(image1);
}

void testOtherPattern2() {
  std::cout << "codec other pattern 2" << std::endl;

  Image image1;
  size_t w = 192;
  size_t h = 192;
  image1.width = w;
  image1.height = h;
  image1.colorType = LCT_RGBA;
  image1.bitDepth = 8;
  image1.data.resize(w * h * 4u);
  for(size_t y = 0; y < h; y++)
  for(size_t x = 0; x < w; x++) {
    image1.data[4u * w * y + 4u * x + 0u] = 255 * !(x & y);
    image1.data[4u * w * y + 4u * x + 1u] = x ^ y;
    image1.data[4u * w * y + 4u * x + 2u] = x | y;
    image1.data[4u * w * y + 4u * x + 3u] = 255;
  }

  doCodecTest(image1);
}

void testSinglePixel(int r, int g, int b, int a) {
  std::cout << "codec single pixel " << r << " " << g << " " << b << " " << a << std::endl;
  Image pixel;
  pixel.width = 1;
  pixel.height = 1;
  pixel.colorType = LCT_RGBA;
  pixel.bitDepth = 8;
  pixel.data.resize(4);
  pixel.data[0] = r;
  pixel.data[1] = g;
  pixel.data[2] = b;
  pixel.data[3] = a;

  doCodecTest(pixel);
}

void testColor(int r, int g, int b, int a) {
  std::cout << "codec test color " << r << " " << g << " " << b << " " << a << std::endl;
  Image image;
  image.width = 20;
  image.height = 20;
  image.colorType = LCT_RGBA;
  image.bitDepth = 8;
  image.data.resize(20 * 20 * 4);
  for(size_t y = 0; y < 20; y++)
  for(size_t x = 0; x < 20; x++) {
    image.data[20 * 4 * y + 4 * x + 0] = r;
    image.data[20 * 4 * y + 4 * x + 0] = g;
    image.data[20 * 4 * y + 4 * x + 0] = b;
    image.data[20 * 4 * y + 4 * x + 0] = a;
  }

  doCodecTest(image);

  Image image2 = image;
  image2.data[3] = 0; //one fully transparent pixel
  doCodecTest(image2);
  image2.data[3] = 128; //one semi transparent pixel
  doCodecTest(image2);

  Image image3 = image;
  // add 255 different colors
  for(size_t i = 0; i < 255; i++) {
    image.data[i * 4 + 0] = i;
    image.data[i * 4 + 1] = i;
    image.data[i * 4 + 2] = i;
    image.data[i * 4 + 3] = 255;
  }
  doCodecTest(image3);
  // a 256th color
  image.data[255 * 4 + 0] = 255;
  image.data[255 * 4 + 1] = 255;
  image.data[255 * 4 + 2] = 255;
  image.data[255 * 4 + 3] = 255;
  doCodecTest(image3);

  testSinglePixel(r, g, b, a);
}

// Tests combinations of various colors in different orders
void testFewColors() {
  std::cout << "codec test few colors " << std::endl;
  Image image;
  image.width = 4;
  image.height = 4;
  image.colorType = LCT_RGBA;
  image.bitDepth = 8;
  image.data.resize(image.width * image.height * 4);
  std::vector<unsigned char> colors;
  colors.push_back(0); colors.push_back(0); colors.push_back(0); colors.push_back(255); // black
  colors.push_back(255); colors.push_back(255); colors.push_back(255); colors.push_back(255); // white
  colors.push_back(128); colors.push_back(128); colors.push_back(128); colors.push_back(255); // gray
  colors.push_back(0); colors.push_back(0); colors.push_back(255); colors.push_back(255); // blue
  colors.push_back(255); colors.push_back(255); colors.push_back(255); colors.push_back(0); // transparent white
  colors.push_back(255); colors.push_back(255); colors.push_back(255); colors.push_back(1); // translucent white
  for(size_t i = 0; i < colors.size(); i += 4)
  for(size_t j = 0; j < colors.size(); j += 4)
  for(size_t k = 0; k < colors.size(); k += 4)
  for(size_t l = 0; l < colors.size(); l += 4) {
    for(unsigned y = 0; y < image.height; y++)
    for(unsigned x = 0; x < image.width; x++) {
      size_t a = (y * image.width + x) & 3;
      size_t b = (a == 0) ? i : ((a == 1) ? j : ((a == 2) ? k : l));
      for(size_t c = 0; c < 4; c++) {
        image.data[y * image.width * 4 + x * 4 + c] = colors[b + c];
      }
    }
    doCodecTest(image);
  }
  image.width = 20;
  image.height = 20;
  image.data.resize(image.width * image.height * 4);
  for(size_t i = 0; i < colors.size(); i += 4)
  for(size_t j = 0; j < colors.size(); j += 4)
  for(size_t k = 0; k < colors.size(); k += 4) {
    for(unsigned y = 0; y < image.height; y++)
    for(unsigned x = 0; x < image.width; x++) {
      size_t a = (y * image.width + x) % 3;
      size_t b = (a == 0) ? i : ((a == 1) ? j : k);
      for(size_t c = 0; c < 4; c++) {
        image.data[y * image.width * 4 + x * 4 + c] = colors[b + c];
      }
    }
    doCodecTest(image);
  }
}

void testSize(unsigned w, unsigned h) {
  std::cout << "codec test size " << w << " " << h << std::endl;
  Image image;
  image.width = w;
  image.height = h;
  image.colorType = LCT_RGBA;
  image.bitDepth = 8;
  image.data.resize(w * h * 4);
  for(size_t y = 0; y < h; y++)
  for(size_t x = 0; x < w; x++) {
    image.data[w * 4 * y + 4 * x + 0] = x & 255;
    image.data[w * 4 * y + 4 * x + 1] = y & 255;
    image.data[w * 4 * y + 4 * x + 2] = 255;
    image.data[w * 4 * y + 4 * x + 3] = 255;
  }

  doCodecTest(image);
}

void testPNGCodec() {
  codecTest(1, 1);
  codecTest(2, 2);
  codecTest(1, 1, LCT_GREY, 1);
  codecTest(7, 7, LCT_GREY, 1);
#ifndef DISABLE_SLOW
  codecTest(127, 127);
  codecTest(127, 127, LCT_GREY, 1);
  codecTest(320, 320);
  codecTest(1, 10000);
  codecTest(10000, 1);

  testOtherPattern1();
  testOtherPattern2();
#endif // DISABLE_SLOW

  testColor(255, 255, 255, 255);
  testColor(0, 0, 0, 255);
  testColor(1, 2, 3, 255);
  testColor(255, 0, 0, 255);
  testColor(0, 255, 0, 255);
  testColor(0, 0, 255, 255);
  testColor(0, 0, 0, 255);
  testColor(1, 1, 1, 255);
  testColor(1, 1, 1, 1);
  testColor(0, 0, 0, 128);
  testColor(255, 0, 0, 128);
  testColor(127, 127, 127, 255);
  testColor(128, 128, 128, 255);
  testColor(127, 127, 127, 128);
  testColor(128, 128, 128, 128);
  //transparent single pixels
  testColor(0, 0, 0, 0);
  testColor(255, 0, 0, 0);
  testColor(1, 2, 3, 0);
  testColor(255, 255, 255, 0);
  testColor(254, 254, 254, 0);

  // This is mainly to test the Adam7 interlacing
  for(unsigned h = 1; h < 12; h++)
  for(unsigned w = 1; w < 12; w++) {
    testSize(w, h);
  }
}

//Tests some specific color conversions with specific color bit combinations
void testColorConvert() {
  //test color conversions to RGBA8
  colorConvertTest("1", LCT_GREY, 1, "11111111 11111111 11111111 11111111", LCT_RGBA, 8);
  colorConvertTest("10", LCT_GREY, 2, "10101010 10101010 10101010 11111111", LCT_RGBA, 8);
  colorConvertTest("1001", LCT_GREY, 4, "10011001 10011001 10011001 11111111", LCT_RGBA, 8);
  colorConvertTest("10010101", LCT_GREY, 8, "10010101 10010101 10010101 11111111", LCT_RGBA, 8);
  colorConvertTest("10010101 11111110", LCT_GREY_ALPHA, 8, "10010101 10010101 10010101 11111110", LCT_RGBA, 8);
  colorConvertTest("10010101 00000001 11111110 00000001", LCT_GREY_ALPHA, 16, "10010101 10010101 10010101 11111110", LCT_RGBA, 8);
  colorConvertTest("01010101 00000000 00110011", LCT_RGB, 8, "01010101 00000000 00110011 11111111", LCT_RGBA, 8);
  colorConvertTest("01010101 00000000 00110011 10101010", LCT_RGBA, 8, "01010101 00000000 00110011 10101010", LCT_RGBA, 8);
  colorConvertTest("10101010 01010101 11111111 00000000 11001100 00110011", LCT_RGB, 16, "10101010 11111111 11001100 11111111", LCT_RGBA, 8);
  colorConvertTest("10101010 01010101 11111111 00000000 11001100 00110011 11100111 00011000", LCT_RGBA, 16, "10101010 11111111 11001100 11100111", LCT_RGBA, 8);

  //test color conversions to RGB8
  colorConvertTest("1", LCT_GREY, 1, "11111111 11111111 11111111", LCT_RGB, 8);
  colorConvertTest("10", LCT_GREY, 2, "10101010 10101010 10101010", LCT_RGB, 8);
  colorConvertTest("1001", LCT_GREY, 4, "10011001 10011001 10011001", LCT_RGB, 8);
  colorConvertTest("10010101", LCT_GREY, 8, "10010101 10010101 10010101", LCT_RGB, 8);
  colorConvertTest("10010101 11111110", LCT_GREY_ALPHA, 8, "10010101 10010101 10010101", LCT_RGB, 8);
  colorConvertTest("10010101 00000001 11111110 00000001", LCT_GREY_ALPHA, 16, "10010101 10010101 10010101", LCT_RGB, 8);
  colorConvertTest("01010101 00000000 00110011", LCT_RGB, 8, "01010101 00000000 00110011", LCT_RGB, 8);
  colorConvertTest("01010101 00000000 00110011 10101010", LCT_RGBA, 8, "01010101 00000000 00110011", LCT_RGB, 8);
  colorConvertTest("10101010 01010101 11111111 00000000 11001100 00110011", LCT_RGB, 16, "10101010 11111111 11001100", LCT_RGB, 8);
  colorConvertTest("10101010 01010101 11111111 00000000 11001100 00110011 11100111 00011000", LCT_RGBA, 16, "10101010 11111111 11001100", LCT_RGB, 8);

  //test color conversions to RGBA16
  colorConvertTest("1", LCT_GREY, 1, "11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111", LCT_RGBA, 16);
  colorConvertTest("10", LCT_GREY, 2, "10101010 10101010 10101010 10101010 10101010 10101010 11111111 11111111", LCT_RGBA, 16);

  //test grayscale color conversions
  colorConvertTest("1", LCT_GREY, 1, "11111111", LCT_GREY, 8);
  colorConvertTest("1", LCT_GREY, 1, "1111111111111111", LCT_GREY, 16);
  colorConvertTest("0", LCT_GREY, 1, "00000000", LCT_GREY, 8);
  colorConvertTest("0", LCT_GREY, 1, "0000000000000000", LCT_GREY, 16);
  colorConvertTest("11", LCT_GREY, 2, "11111111", LCT_GREY, 8);
  colorConvertTest("11", LCT_GREY, 2, "1111111111111111", LCT_GREY, 16);
  colorConvertTest("10", LCT_GREY, 2, "10101010", LCT_GREY, 8);
  colorConvertTest("10", LCT_GREY, 2, "1010101010101010", LCT_GREY, 16);
  colorConvertTest("1000", LCT_GREY, 4, "10001000", LCT_GREY, 8);
  colorConvertTest("1000", LCT_GREY, 4, "1000100010001000", LCT_GREY, 16);
  colorConvertTest("10110101", LCT_GREY, 8, "1011010110110101", LCT_GREY, 16);
  colorConvertTest("1011010110110101", LCT_GREY, 16, "10110101", LCT_GREY, 8);

  //others
  colorConvertTest("11111111 11111111 11111111 00000000 00000000 00000000", LCT_RGB, 8, "10", LCT_GREY, 1);
  colorConvertTest("11111111 11111111 11111111 11111111 11111111 11111111 00000000 00000000 00000000 00000000 00000000 00000000", LCT_RGB, 16, "10", LCT_GREY, 1);
}

//This tests color conversions from any color model to any color model, with any bit depth
//But it tests only with colors black and white, because that are the only colors every single model supports
void testColorConvert2() {
  std::cout << "testColorConvert2" << std::endl;
  struct Combo {
    LodePNGColorType colortype;
    unsigned bitdepth;
  };

  Combo combos[15] = { { LCT_GREY, 1}, { LCT_GREY, 2}, { LCT_GREY, 4}, { LCT_GREY, 8}, { LCT_GREY, 16}, { LCT_RGB, 8}, { LCT_RGB, 16}, { LCT_PALETTE, 1}, { LCT_PALETTE, 2}, { LCT_PALETTE, 4}, { LCT_PALETTE, 8}, { LCT_GREY_ALPHA, 8}, { LCT_GREY_ALPHA, 16}, { LCT_RGBA, 8}, { LCT_RGBA, 16},
  };

  lodepng::State state;
  LodePNGColorMode& mode_in = state.info_png.color;
  LodePNGColorMode& mode_out = state.info_raw;
  LodePNGColorMode mode_8;
  lodepng_color_mode_init(&mode_8);

  for(size_t i = 0; i < 256; i++) {
    size_t j = i == 1 ? 255 : i;
    lodepng_palette_add(&mode_in, j, j, j, 255);
    lodepng_palette_add(&mode_out, j, j, j, 255);
  }

  for(size_t i = 0; i < 15; i++) {
    mode_in.colortype = combos[i].colortype;
    mode_in.bitdepth = combos[i].bitdepth;

    for(size_t j = 0; j < 15; j++) {
      mode_out.colortype = combos[j].colortype;
      mode_out.bitdepth = combos[j].bitdepth;

      unsigned char eight[36] = {
          0,0,0,255, 255,255,255,255,
          0,0,0,255, 255,255,255,255,
          255,255,255,255, 0,0,0,255,
          255,255,255,255, 255,255,255,255,
          0,0,0,255 }; //input in RGBA8
      unsigned char in[72]; //custom input color type
      unsigned char out[72]; //custom output color type
      unsigned char eight2[36]; //back in RGBA8 after all conversions to check correctness
      unsigned error = 0;

      error |= lodepng_convert(in, eight, &mode_in, &mode_8, 3, 3);
      if(!error) error |= lodepng_convert(out, in, &mode_out, &mode_in, 3, 3); //Test input to output type
      if(!error) error |= lodepng_convert(eight2, out, &mode_8, &mode_out, 3, 3);

      if(!error) {
        for(size_t k = 0; k < 36; k++) {
          if(eight[k] != eight2[k]) {
            error = 99999;
            break;
          }
        }
      }

      if(error) {
        std::cout << "Error " << error << " i: " << i << " j: " << j
          << " colortype i: " << combos[i].colortype
          << " bitdepth i: " << combos[i].bitdepth
          << " colortype j: " << combos[j].colortype
          << " bitdepth j: " << combos[j].bitdepth
          << std::endl;
        if(error != 99999) ASSERT_NO_PNG_ERROR(error);
        else fail();
      }
    }
  }
}

//if compressible is true, the test will also assert that the compressed string is smaller
void testCompressStringZlib(const std::string& text, bool compressible) {
  if(text.size() < 500) std::cout << "compress test with text: " << text << std::endl;
  else std::cout << "compress test with text length: " << text.size() << std::endl;

  std::vector<unsigned char> in(text.size());
  for(size_t i = 0; i < text.size(); i++) in[i] = (unsigned char)text[i];
  unsigned char* out = 0;
  size_t outsize = 0;
  unsigned error = 0;

  error = lodepng_zlib_compress(&out, &outsize, in.empty() ? 0 : &in[0], in.size(), &lodepng_default_compress_settings);
  ASSERT_NO_PNG_ERROR(error);
  if(compressible) assertTrue(outsize < in.size());

  unsigned char* out2 = 0;
  size_t outsize2 = 0;

  error = lodepng_zlib_decompress(&out2, &outsize2, out, outsize, &lodepng_default_decompress_settings);
  ASSERT_NO_PNG_ERROR(error);
  ASSERT_EQUALS(outsize2, in.size());
  for(size_t i = 0; i < in.size(); i++) ASSERT_EQUALS(in[i], out2[i]);

  free(out);
  free(out2);
}

void testCompressZlib() {
  testCompressStringZlib("", false);
  testCompressStringZlib("a", false);
  testCompressStringZlib("aa", false);
  testCompressStringZlib("ababababababababababababababababababababababababababababababababababababababababababab", true);
  testCompressStringZlib("abaaaabaabbbaabbabbababbbbabababbbaabbbaaaabbbbabbbabbbaababbbbbaaabaabbabaaaabbbbbbab", true);
  testCompressStringZlib("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", true);
  testCompressStringZlib("omnomnomnomnomnomnomnomnomnomnom", true);
  testCompressStringZlib("the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.", true);
  testCompressStringZlib("abracadabra", false);
  testCompressStringZlib("hello hello hello hello hello hello hello hello hello hello hello?", true);
  testCompressStringZlib("WPgZX2D*um0H::,4/KU\"kt\"Ne\"#Qa.&#<aF9{jag]|{hv,IXez\
\\DKn5zYdV{XxBi=n|1J-TwakWvp[b8|-kOcZ@QkAxJSMeZ0l&<*w0BP/CXM(LFH'", false);
  testCompressStringZlib("asdfhlkhfafsduyfbasiuytfgbiasuidygiausygdifaubsydfsdf", false);
  testCompressStringZlib("418541499849814614617987416457317375467441841687487", true);
  testCompressStringZlib("3.141592653589793238462643383279502884197169399375105820974944592307816406286", true);
  testCompressStringZlib("lodepng_zlib_decompress(&out2, &outsize2, out, outsize, &lodepng_default_decompress_settings);", true);
}

void testDiskCompressZlib(const std::string& filename) {
  std::cout << "testDiskCompressZlib: File " << filename << std::endl;

  std::vector<unsigned char> buffer;
  lodepng::load_file(buffer, filename);
  std::string f;
  for(size_t i = 0; i < buffer.size(); i++) f += (char)buffer[i];
  testCompressStringZlib(f, false);
}

void testDiskPNG(const std::string& filename) {
  std::cout << "testDiskPNG: File " << filename << std::endl;

  Image image;
  image.colorType = LCT_RGB;
  image.bitDepth = 8;
  unsigned error = lodepng::decode(image.data, image.width, image.height, filename, image.colorType, image.bitDepth);
  ASSERT_NO_PNG_ERROR(error);

  doCodecTest(image);
}

std::vector<unsigned> strtovector(const std::string& numbers) {
  std::vector<unsigned> result;
  std::stringstream ss(numbers);
  unsigned i;
  while(ss >> i) result.push_back(i);
  return result;
}

void doTestHuffmanCodeLengths(const std::string& expectedstr, const std::string& counts, size_t bitlength) {
  std::vector<unsigned> expected = strtovector(expectedstr);
  std::vector<unsigned> count = strtovector(counts);
  std::cout << "doTestHuffmanCodeLengths: " << counts << std::endl;
  std::vector<unsigned> result(count.size());
  unsigned error = lodepng_huffman_code_lengths(&result[0], &count[0], count.size(), bitlength);
  ASSERT_NO_PNG_ERROR_MSG(error, "errorcode");
  std::stringstream ss1, ss2;
  for(size_t i = 0; i < count.size(); i++) {
    ss1 << expected[i] << " ";
    ss2 << result[i] << " ";
  }
  assertEquals(ss1.str(), ss2.str(), "value");
}

void testHuffmanCodeLengths() {
  bool atleasttwo = true; //LodePNG generates at least two, instead of at least one, symbol
  if(atleasttwo) {
    doTestHuffmanCodeLengths("1 1", "0 0", 16);
    doTestHuffmanCodeLengths("1 1 0", "0 0 0", 16);
    doTestHuffmanCodeLengths("1 1", "1 0", 16);
    doTestHuffmanCodeLengths("1 1 0 0 0 0 0 0 0", "0 0 0 0 0 0 0 0 0", 16);
    doTestHuffmanCodeLengths("1 1 0 0 0 0 0 0 0", "1 0 0 0 0 0 0 0 0", 16);
    doTestHuffmanCodeLengths("1 1 0 0 0 0 0 0 0", "0 1 0 0 0 0 0 0 0", 16);
    doTestHuffmanCodeLengths("1 0 0 0 0 0 0 0 1", "0 0 0 0 0 0 0 0 1", 16);
    doTestHuffmanCodeLengths("0 0 0 0 0 0 0 1 1", "0 0 0 0 0 0 0 1 1", 16);
  } else {
    doTestHuffmanCodeLengths("1 0", "0 0", 16);
    doTestHuffmanCodeLengths("1 0 0", "0 0 0", 16);
    doTestHuffmanCodeLengths("1 0", "1 0", 16);
    doTestHuffmanCodeLengths("1", "1", 16);
    doTestHuffmanCodeLengths("1", "0", 16);
  }
  doTestHuffmanCodeLengths("1 1", "1 1", 16);
  doTestHuffmanCodeLengths("1 1", "1 100", 16);
  doTestHuffmanCodeLengths("2 2 1", "1 2 3", 16);
  doTestHuffmanCodeLengths("2 1 2", "2 3 1", 16);
  doTestHuffmanCodeLengths("1 2 2", "3 1 2", 16);
  doTestHuffmanCodeLengths("3 3 2 1", "1 30 31 32", 16);
  doTestHuffmanCodeLengths("2 2 2 2", "1 30 31 32", 2);
  doTestHuffmanCodeLengths("5 5 4 4 4 3 3 1", "1 2 3 4 5 6 7 500", 16);
}

/*
Create a PNG image with all known chunks (except only one of tEXt or zTXt) plus
unknown chunks, and a palette.
*/
void createComplexPNG(std::vector<unsigned char>& png) {
  unsigned w = 16, h = 17;
  std::vector<unsigned char> image(w * h);
  for(size_t i = 0; i < w * h; i++) {
    image[i] = i % 256;
  }

  lodepng::State state;
  LodePNGInfo& info = state.info_png;
  info.color.colortype = LCT_PALETTE;
  info.color.bitdepth = 8;
  state.info_raw.colortype = LCT_PALETTE;
  state.info_raw.bitdepth = 8;
  state.encoder.auto_convert = false;
  state.encoder.text_compression = 1;
  state.encoder.add_id = 1;
  for(size_t i = 0; i < 256; i++) {
    lodepng_palette_add(&info.color, i, i, i, i);
    lodepng_palette_add(&state.info_raw, i, i, i, i);
  }

  info.background_defined = 1;
  info.background_r = 127;

  lodepng_add_text(&info, "key0", "string0");
  lodepng_add_text(&info, "key1", "string1");

  lodepng_add_itext(&info, "ikey0", "ilangtag0", "itranskey0", "istring0");
  lodepng_add_itext(&info, "ikey1", "ilangtag1", "itranskey1", "istring1");

  info.time_defined = 1;
  info.time.year = 2012;
  info.time.month = 1;
  info.time.day = 2;
  info.time.hour = 3;
  info.time.minute = 4;
  info.time.second = 5;

  info.phys_defined = 1;
  info.phys_x = 1;
  info.phys_y = 2;
  info.phys_unit = 1;

  lodepng_chunk_create(&info.unknown_chunks_data[0], &info.unknown_chunks_size[0], 3, "uNKa", (unsigned char*)"a00");
  lodepng_chunk_create(&info.unknown_chunks_data[0], &info.unknown_chunks_size[0], 3, "uNKa", (unsigned char*)"a01");
  lodepng_chunk_create(&info.unknown_chunks_data[1], &info.unknown_chunks_size[1], 3, "uNKb", (unsigned char*)"b00");
  lodepng_chunk_create(&info.unknown_chunks_data[2], &info.unknown_chunks_size[2], 3, "uNKc", (unsigned char*)"c00");

  unsigned error = lodepng::encode(png, &image[0], w, h, state);
  ASSERT_NO_PNG_ERROR(error);
}

std::string extractChunkNames(const std::vector<unsigned char>& png) {
  const unsigned char* chunk = &png[8];
  const unsigned char* end = &png.back() + 1;
  char name[5];
  std::string result = "";
  for(;;) {
    lodepng_chunk_type(name, chunk);
    result += (std::string(" ") + name);
    if(std::string(name) == "IEND") break;
    chunk = lodepng_chunk_next_const(chunk, end);
    assertTrue(chunk < &png.back(), "jumped out of chunks");
  }
  return result;
}

void testComplexPNG() {
  std::cout << "testComplexPNG" << std::endl;

  std::vector<unsigned char> png;
  createComplexPNG(png);
 {
    lodepng::State state;
    LodePNGInfo& info = state.info_png;
    unsigned w, h;
    std::vector<unsigned char> image;
    unsigned error = lodepng::decode(image, w, h, state, &png[0], png.size());
    ASSERT_NO_PNG_ERROR(error);

    ASSERT_EQUALS(16, w);
    ASSERT_EQUALS(17, h);
    ASSERT_EQUALS(1, info.background_defined);
    ASSERT_EQUALS(127, info.background_r);
    ASSERT_EQUALS(1, info.time_defined);
    ASSERT_EQUALS(2012, info.time.year);
    ASSERT_EQUALS(1, info.time.month);
    ASSERT_EQUALS(2, info.time.day);
    ASSERT_EQUALS(3, info.time.hour);
    ASSERT_EQUALS(4, info.time.minute);
    ASSERT_EQUALS(5, info.time.second);
    ASSERT_EQUALS(1, info.phys_defined);
    ASSERT_EQUALS(1, info.phys_x);
    ASSERT_EQUALS(2, info.phys_y);
    ASSERT_EQUALS(1, info.phys_unit);

    std::string chunknames = extractChunkNames(png);
    //std::string expectednames = " IHDR uNKa uNKa PLTE tRNS bKGD pHYs uNKb IDAT tIME tEXt tEXt tEXt iTXt iTXt uNKc IEND";
    std::string expectednames = " IHDR uNKa uNKa PLTE tRNS bKGD pHYs uNKb IDAT tIME zTXt zTXt tEXt iTXt iTXt uNKc IEND";
    ASSERT_EQUALS(expectednames, chunknames);

    ASSERT_EQUALS(3, info.text_num);
    ASSERT_STRING_EQUALS("key0", info.text_keys[0]);
    ASSERT_STRING_EQUALS("string0", info.text_strings[0]);
    ASSERT_STRING_EQUALS("key1", info.text_keys[1]);
    ASSERT_STRING_EQUALS("string1", info.text_strings[1]);
    ASSERT_STRING_EQUALS("LodePNG", info.text_keys[2]);
    ASSERT_STRING_EQUALS(LODEPNG_VERSION_STRING, info.text_strings[2]);

    ASSERT_EQUALS(2, info.itext_num);
    ASSERT_STRING_EQUALS("ikey0", info.itext_keys[0]);
    ASSERT_STRING_EQUALS("ilangtag0", info.itext_langtags[0]);
    ASSERT_STRING_EQUALS("itranskey0", info.itext_transkeys[0]);
    ASSERT_STRING_EQUALS("istring0", info.itext_strings[0]);
    ASSERT_STRING_EQUALS("ikey1", info.itext_keys[1]);
    ASSERT_STRING_EQUALS("ilangtag1", info.itext_langtags[1]);
    ASSERT_STRING_EQUALS("itranskey1", info.itext_transkeys[1]);
    ASSERT_STRING_EQUALS("istring1", info.itext_strings[1]);

    // TODO: test if unknown chunks listed too
  }


  // Test that if read_text_chunks is disabled, we do not get the texts
  {
    lodepng::State state;
    state.decoder.read_text_chunks = 0;
    unsigned w, h;
    std::vector<unsigned char> image;
    unsigned error = lodepng::decode(image, w, h, state, &png[0], png.size());
    ASSERT_NO_PNG_ERROR(error);

    ASSERT_EQUALS(0, state.info_png.text_num);
    ASSERT_EQUALS(0, state.info_png.itext_num);

    // But we should still get other values.
    ASSERT_EQUALS(2012, state.info_png.time.year);
  }
}

// Tests lodepng_inspect_chunk, and also lodepng_chunk_find to find the chunk to inspect
void testInspectChunk() {
  std::cout << "testInspectChunk" << std::endl;

  std::vector<unsigned char> png;
  createComplexPNG(png);

  const unsigned char* chunk;
  lodepng::State state;
  LodePNGInfo& info = state.info_png;
  state.decoder.read_text_chunks = 0;
  lodepng_inspect(0, 0, &state, png.data(), png.size());
  chunk = lodepng_chunk_find(png.data(), png.data() + png.size(), "tIME");
  ASSERT_NOT_EQUALS((const unsigned char*)0, chunk); // should be non-null, since it should find it
  ASSERT_EQUALS(0, info.time_defined);
  lodepng_inspect_chunk(&state, (size_t)(chunk - png.data()), png.data(), png.size());
  ASSERT_EQUALS(1, info.time_defined);
  ASSERT_EQUALS(2012, state.info_png.time.year);
  ASSERT_EQUALS(1, info.time.month);
  ASSERT_EQUALS(2, info.time.day);
  ASSERT_EQUALS(3, info.time.hour);
  ASSERT_EQUALS(4, info.time.minute);
  ASSERT_EQUALS(5, info.time.second);

  ASSERT_EQUALS(0, info.text_num);
  chunk = lodepng_chunk_find_const(png.data(), png.data() + png.size(), "zTXt");
  lodepng_inspect_chunk(&state, (size_t)(chunk - png.data()), png.data(), png.size());
  ASSERT_EQUALS(1, info.text_num);
  chunk = lodepng_chunk_find_const(chunk, png.data() + png.size(), "zTXt");
  lodepng_inspect_chunk(&state, (size_t)(chunk - png.data()), png.data(), png.size());
  ASSERT_EQUALS(2, info.text_num);
}

//test that, by default, it chooses filter type zero for all scanlines if the image has a palette
void testPaletteFilterTypesZero() {
  std::cout << "testPaletteFilterTypesZero" << std::endl;

  std::vector<unsigned char> png;
  createComplexPNG(png);

  std::vector<unsigned char> filterTypes;
  lodepng::getFilterTypes(filterTypes, png);

  ASSERT_EQUALS(17, filterTypes.size());
  for(size_t i = 0; i < 17; i++) ASSERT_EQUALS(0, filterTypes[i]);
}

//tests that there are no crashes with auto color chooser in case of palettes with translucency etc...
void testPaletteToPaletteConvert() {
  std::cout << "testPaletteToPaletteConvert" << std::endl;
  unsigned error;
  unsigned w = 16, h = 16;
  std::vector<unsigned char> image(w * h);
  for(size_t i = 0; i < w * h; i++) image[i] = i % 256;
  lodepng::State state;
  LodePNGInfo& info = state.info_png;
  info.color.colortype = state.info_raw.colortype = LCT_PALETTE;
  info.color.bitdepth = state.info_raw.bitdepth = 8;
  ASSERT_EQUALS(true, state.encoder.auto_convert);
  for(size_t i = 0; i < 256; i++) {
    lodepng_palette_add(&info.color, i, i, i, i);
  }
  std::vector<unsigned char> png;
  for(size_t i = 0; i < 256; i++) {
    lodepng_palette_add(&state.info_raw, i, i, i, i);
  }
  error = lodepng::encode(png, &image[0], w, h, state);
  ASSERT_NO_PNG_ERROR(error);
}

//for this test, you have to choose palette colors that cause LodePNG to actually use a palette,
//so don't use all grayscale colors for example
void doRGBAToPaletteTest(unsigned char* palette, size_t size, LodePNGColorType expectedType = LCT_PALETTE) {
  std::cout << "testRGBToPaletteConvert " << size << std::endl;
  unsigned error;
  unsigned w = size, h = 257 /*LodePNG encodes no palette if image is too small*/;
  std::vector<unsigned char> image(w * h * 4);
  for(size_t i = 0; i < image.size(); i++) image[i] = palette[i % (size * 4)];
  std::vector<unsigned char> png;
  error = lodepng::encode(png, &image[0], w, h);
  ASSERT_NO_PNG_ERROR(error);
  lodepng::State state;
  std::vector<unsigned char> image2;
  error = lodepng::decode(image2, w, h, state, png);
  ASSERT_NO_PNG_ERROR(error);
  ASSERT_EQUALS(image.size(), image2.size());
  for(size_t i = 0; i < image.size(); i++) ASSERT_EQUALS(image[i], image2[i]);

  ASSERT_EQUALS(expectedType, state.info_png.color.colortype);
  if(expectedType == LCT_PALETTE) {

    ASSERT_EQUALS(size, state.info_png.color.palettesize);
    for(size_t i = 0; i < size * 4; i++) ASSERT_EQUALS(state.info_png.color.palette[i], image[i]);
  }
}

void testRGBToPaletteConvert() {
  unsigned char palette1[4] = {1,2,3,4};
  doRGBAToPaletteTest(palette1, 1);
  unsigned char palette2[8] = {1,2,3,4, 5,6,7,8};
  doRGBAToPaletteTest(palette2, 2);
  unsigned char palette3[12] = {1,1,1,255, 20,20,20,255, 20,20,21,255};
  doRGBAToPaletteTest(palette3, 3);

  std::vector<unsigned char> palette;
  for(int i = 0; i < 256; i++) {
    palette.push_back(i);
    palette.push_back(5);
    palette.push_back(6);
    palette.push_back(128);
  }
  doRGBAToPaletteTest(&palette[0], 256);
  palette.push_back(5);
  palette.push_back(6);
  palette.push_back(7);
  palette.push_back(8);
  doRGBAToPaletteTest(&palette[0], 257, LCT_RGBA);
}

void testColorKeyConvert() {
  std::cout << "testColorKeyConvert" << std::endl;
  unsigned error;
  unsigned w = 32, h = 32;
  std::vector<unsigned char> image(w * h * 4);
  for(size_t i = 0; i < w * h; i++) {
    image[i * 4 + 0] = i % 256;
    image[i * 4 + 1] = i / 256;
    image[i * 4 + 2] = 0;
    image[i * 4 + 3] = i == 23 ? 0 : 255;
  }
  std::vector<unsigned char> png;
  error = lodepng::encode(png, &image[0], w, h);
  ASSERT_NO_PNG_ERROR(error);

  lodepng::State state;
  std::vector<unsigned char> image2;
  error = lodepng::decode(image2, w, h, state, png);
  ASSERT_NO_PNG_ERROR(error);
  ASSERT_EQUALS(32, w);
  ASSERT_EQUALS(32, h);
  ASSERT_EQUALS(1, state.info_png.color.key_defined);
  ASSERT_EQUALS(23, state.info_png.color.key_r);
  ASSERT_EQUALS(0, state.info_png.color.key_g);
  ASSERT_EQUALS(0, state.info_png.color.key_b);
  ASSERT_EQUALS(image.size(), image2.size());
  for(size_t i = 0; i < image.size(); i++) {
    ASSERT_EQUALS(image[i], image2[i]);
  }
}

void testNoAutoConvert() {
  std::cout << "testNoAutoConvert" << std::endl;
  unsigned error;
  unsigned w = 32, h = 32;
  std::vector<unsigned char> image(w * h * 4);
  for(size_t i = 0; i < w * h; i++) {
    image[i * 4 + 0] = (i % 2) ? 255 : 0;
    image[i * 4 + 1] = (i % 2) ? 255 : 0;
    image[i * 4 + 2] = (i % 2) ? 255 : 0;
    image[i * 4 + 3] = 0;
  }
  std::vector<unsigned char> png;
  lodepng::State state;
  state.info_png.color.colortype = LCT_RGBA;
  state.info_png.color.bitdepth = 8;
  state.encoder.auto_convert = false;
  error = lodepng::encode(png, &image[0], w, h, state);
  ASSERT_NO_PNG_ERROR(error);

  lodepng::State state2;
  std::vector<unsigned char> image2;
  error = lodepng::decode(image2, w, h, state2, png);
  ASSERT_NO_PNG_ERROR(error);
  ASSERT_EQUALS(32, w);
  ASSERT_EQUALS(32, h);
  ASSERT_EQUALS(LCT_RGBA, state2.info_png.color.colortype);
  ASSERT_EQUALS(8, state2.info_png.color.bitdepth);
  ASSERT_EQUALS(image.size(), image2.size());
  for(size_t i = 0; i < image.size(); i++) {
    ASSERT_EQUALS(image[i], image2[i]);
  }
}

unsigned char flipBit(unsigned char c, int bitpos) {
  return c ^ (1 << bitpos);
}

//Test various broken inputs. Returned errors are not checked, what is tested is
//that is doesn't crash, and, when run with valgrind, no memory warnings are
//given.
void testFuzzing() {
  std::cout << "testFuzzing" << std::endl;
  std::vector<unsigned char> png;
  createComplexPNG(png);
  std::vector<unsigned char> broken = png;
  std::vector<unsigned char> result;
  std::map<unsigned, unsigned> errors;
  unsigned w, h;
  lodepng::State state;
  state.decoder.ignore_crc = 1;
  state.decoder.zlibsettings.ignore_adler32 = 1;
  for(size_t i = 0; i < png.size(); i++) {
    result.clear();
    broken[i] = ~png[i];
    errors[lodepng::decode(result, w, h, state, broken)]++;
    broken[i] = 0;
    errors[lodepng::decode(result, w, h, state, broken)]++;
    for(int j = 0; j < 8; j++) {
      broken[i] = flipBit(png[i], j);
      errors[lodepng::decode(result, w, h, state, broken)]++;
    }
    broken[i] = 255;
    errors[lodepng::decode(result, w, h, state, broken)]++;
    broken[i] = png[i]; //fix it again for the next test
  }
  std::cout << "testFuzzing shrinking" << std::endl;
  broken = png;
  while(broken.size() > 0) {
    broken.resize(broken.size() - 1);
    errors[lodepng::decode(result, w, h, state, broken)]++;
  }

  //For fun, print the number of each error
  std::cout << "Fuzzing error code counts: ";
  for(std::map<unsigned, unsigned>::iterator it = errors.begin(); it != errors.end(); ++it) {
    std::cout << it->first << ":" << it->second << ", ";
  }
  std::cout << std::endl;
}

int custom_proof = 0; // global variable for nested function to call. Of course when this test is switched to modern C++ we can use a lamba instead.

void testCustomZlibCompress() {
  std::cout << "testCustomZlibCompress" << std::endl;
  Image image;
  generateTestImage(image, 5, 5, LCT_RGBA, 8);

  std::vector<unsigned char> encoded;
  int customcontext = 5;

  struct TestFun {
    static unsigned custom_zlib(unsigned char**, size_t*,
                          const unsigned char*, size_t,
                          const LodePNGCompressSettings* settings) {
      ASSERT_EQUALS(5, *(int*)(settings->custom_context));
      custom_proof = 1;
      return 5555; // return a custom error code, which will be converted to an error known to lodepng.
    }
  };

  lodepng::State state;
  state.encoder.zlibsettings.custom_zlib = TestFun::custom_zlib;
  state.encoder.zlibsettings.custom_context = &customcontext;

  custom_proof = 0;
  unsigned error = lodepng::encode(encoded, image.data, image.width, image.height, state);
  ASSERT_EQUALS(1, custom_proof); // check that the custom zlib was called

  ASSERT_EQUALS(111, error); // expect a known lodepng error, not the custom 5555
}

void testCustomZlibCompress2() {
  std::cout << "testCustomZlibCompress2" << std::endl;
  Image image;
  generateTestImage(image, 5, 5, LCT_RGBA, 8);

  std::vector<unsigned char> encoded;

  lodepng::State state;
  state.encoder.zlibsettings.custom_zlib = lodepng_zlib_compress;

  unsigned error = lodepng::encode(encoded, image.data, image.width, image.height,
                                   state);
  ASSERT_NO_PNG_ERROR(error);

  std::vector<unsigned char> decoded;
  unsigned w, h;
  state.decoder.zlibsettings.ignore_adler32 = 0;
  state.decoder.ignore_crc = 0;
  error = lodepng::decode(decoded, w, h, state, encoded);
  ASSERT_NO_PNG_ERROR(error);
  ASSERT_EQUALS(5, w);
  ASSERT_EQUALS(5, h);
}

void testCustomDeflate() {
  std::cout << "testCustomDeflate" << std::endl;
  Image image;
  generateTestImage(image, 5, 5, LCT_RGBA, 8);

  std::vector<unsigned char> encoded;
  int customcontext = 5;

  struct TestFun {
    static unsigned custom_deflate(unsigned char**, size_t*,
                                   const unsigned char*, size_t,
                                   const LodePNGCompressSettings* settings) {
      ASSERT_EQUALS(5, *(int*)(settings->custom_context));
      custom_proof = 1;
      return 5555; // return a custom error code, which will be converted to an error known to lodepng.
    }
  };

  lodepng::State state;
  state.encoder.zlibsettings.custom_deflate = TestFun::custom_deflate;
  state.encoder.zlibsettings.custom_context = &customcontext;

  custom_proof = 0;
  unsigned error = lodepng::encode(encoded, image.data, image.width, image.height, state);
  ASSERT_EQUALS(1, custom_proof); // check that the custom deflate was called

  ASSERT_EQUALS(111, error); // expect a known lodepng error, not the custom 5555
}

void testCustomZlibDecompress() {
  std::cout << "testCustomZlibDecompress" << std::endl;
  Image image;
  generateTestImage(image, 5, 5, LCT_RGBA, 8);

  std::vector<unsigned char> encoded;

  unsigned error_enc = lodepng::encode(encoded, image.data, image.width, image.height,
                                   image.colorType, image.bitDepth);
  ASSERT_NO_PNG_ERROR_MSG(error_enc, "encoder error not expected");


  std::vector<unsigned char> decoded;
  unsigned w, h;
  int customcontext = 5;

  struct TestFun {
    static unsigned custom_zlib(unsigned char**, size_t*,
                          const unsigned char*, size_t,
                          const LodePNGDecompressSettings* settings) {
      ASSERT_EQUALS(5, *(int*)(settings->custom_context));
      custom_proof = 1;
      return 5555; // return a custom error code, which will be converted to an error known to lodepng.
    }
  };

  lodepng::State state;
  state.decoder.zlibsettings.custom_zlib = TestFun::custom_zlib;
  state.decoder.zlibsettings.custom_context = &customcontext;
  state.decoder.zlibsettings.ignore_adler32 = 0;
  state.decoder.ignore_crc = 0;
  custom_proof = 0;
  unsigned error = lodepng::decode(decoded, w, h, state, encoded);
  ASSERT_EQUALS(1, custom_proof); // check that the custom zlib was called

  ASSERT_EQUALS(110, error);
}

void testCustomInflate() {
  std::cout << "testCustomInflate" << std::endl;
  Image image;
  generateTestImage(image, 5, 5, LCT_RGBA, 8);

  std::vector<unsigned char> encoded;

  unsigned error_enc = lodepng::encode(encoded, image.data, image.width, image.height,
                                   image.colorType, image.bitDepth);
  ASSERT_NO_PNG_ERROR_MSG(error_enc, "encoder error not expected");


  std::vector<unsigned char> decoded;
  unsigned w, h;
  int customcontext = 5;

  struct TestFun {
    static unsigned custom_inflate(unsigned char**, size_t*,
                                   const unsigned char*, size_t,
                                   const LodePNGDecompressSettings* settings) {
      ASSERT_EQUALS(5, *(int*)(settings->custom_context));
      custom_proof = 1;
      return 5555; // return a custom error code, which will be converted to an error known to lodepng.
    }
  };

  lodepng::State state;
  state.decoder.zlibsettings.custom_inflate = TestFun::custom_inflate;
  state.decoder.zlibsettings.custom_context = &customcontext;
  state.decoder.zlibsettings.ignore_adler32 = 0;
  state.decoder.ignore_crc = 0;
  custom_proof = 0;
  unsigned error = lodepng::decode(decoded, w, h, state, encoded);
  ASSERT_EQUALS(1, custom_proof); // check that the custom zlib was called

  ASSERT_EQUALS(110, error);
}


void testChunkUtil() {
  std::cout << "testChunkUtil" << std::endl;
  std::vector<unsigned char> png;
  createComplexPNG(png);

  std::vector<std::string> names[3];
  std::vector<std::vector<unsigned char> > chunks[3];

  assertNoError(lodepng::getChunks(names, chunks, png));

  std::vector<std::vector<unsigned char> > chunks2[3];
  chunks2[0].push_back(chunks[2][2]); //zTXt
  chunks2[1].push_back(chunks[2][3]); //tEXt
  chunks2[2].push_back(chunks[2][4]); //iTXt

  assertNoError(lodepng::insertChunks(png, chunks2));

  std::string chunknames = extractChunkNames(png);
  //                                        chunks2[0]                    chunks2[1]                                   chunks2[2]
  //                                             v                             v                                            v
  std::string expectednames = " IHDR uNKa uNKa zTXt PLTE tRNS bKGD pHYs uNKb tEXt IDAT tIME zTXt zTXt tEXt iTXt iTXt uNKc iTXt IEND";
  ASSERT_EQUALS(expectednames, chunknames);

  std::vector<unsigned char> image;
  unsigned w, h;
  ASSERT_NO_PNG_ERROR(lodepng::decode(image, w, h, png));
}

//Test that when decoding to 16-bit per channel, it always uses big endian consistently.
//It should always output big endian, the convention used inside of PNG, even though x86 CPU's are little endian.
void test16bitColorEndianness() {
  std::cout << "test16bitColorEndianness" << std::endl;

  //basn0g16.png from the PNG test suite
  std::string base64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAYagMeiWXwAAAF5JREFU"
                       "eJzV0jEKwDAMQ1E5W+9/xtygk8AoezLVKgSj2Y8/OICnuFcTE2OgOoJgHQiZAN2C9kDKBOgW3AZC"
                       "JkC3oD2QMgG6BbeBkAnQLWgPpExgP28H7E/0GTjPfwAW2EvYX64rn9cAAAAASUVORK5CYII=";
  std::vector<unsigned char> png;
  fromBase64(png, base64);
  unsigned w, h;
  std::vector<unsigned char> image;
  lodepng::State state;

  // Decode from 16-bit gray image to 16-bit per channel RGBA
  state.info_raw.bitdepth = 16;
  ASSERT_NO_PNG_ERROR(lodepng::decode(image, w, h, state, png));
  ASSERT_EQUALS(0x09, image[8]);
  ASSERT_EQUALS(0x00, image[9]);

  // Decode from 16-bit gray image to 16-bit gray raw image (no conversion)
  image.clear();
  state = lodepng::State();
  state.decoder.color_convert = false;
  ASSERT_NO_PNG_ERROR(lodepng::decode(image, w, h, state, png));
  ASSERT_EQUALS(0x09, image[2]);
  ASSERT_EQUALS(0x00, image[3]);

  // Decode from 16-bit per channel RGB image to 16-bit per channel RGBA
  base64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAACsiDHgAAAABGdBTUEAAYagMeiWXwAAAANzQklU"
           "DQ0N0DeNwQAAAH5JREFUeJztl8ENxEAIAwcJ6cpI+q8qKeNepAgelq2dCjz4AdQM1jRcf3WIDQ13"
           "qUNsiBBQZ1gR0cARUFIz3pug3586wo5+rOcfIaBOsCSggSOgpcB8D4D3R9DgfUyECIhDbAhp4Ajo"
           "KPD+CBq8P4IG72MiQkCdYUVEA0dAyQcwUyZpXH92ZwAAAABJRU5ErkJggg=="; //cs3n2c16.png
  png.clear();
  fromBase64(png, base64);
  image.clear();
  state = lodepng::State();
  state.info_raw.bitdepth = 16;
  ASSERT_NO_PNG_ERROR(lodepng::decode(image, w, h, state, png));
  ASSERT_EQUALS(0x1f, image[258]);
  ASSERT_EQUALS(0xf9, image[259]);

  // Decode from 16-bit per channel RGB image to 16-bit per channel RGBA raw image (no conversion)
  image.clear();
  state = lodepng::State();
  state.decoder.color_convert = false;
  ASSERT_NO_PNG_ERROR(lodepng::decode(image, w, h, state, png));

  ASSERT_EQUALS(0x1f, image[194]);
  ASSERT_EQUALS(0xf9, image[195]);

  image.clear();
  state = lodepng::State();

  // Decode from palette image to 16-bit per channel RGBA
  base64 = "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAHAgMAAAC5PL9AAAAABGdBTUEAAYagMeiWXwAAAANzQklU"
           "BAQEd/i1owAAAAxQTFRF/wB3AP93//8AAAD/G0OznAAAABpJREFUeJxj+P+H4WoMw605DDfmgEgg"
           "+/8fAHF5CrkeXW0HAAAAAElFTkSuQmCC"; //s07n3p02.png
  png.clear();
  fromBase64(png, base64);
  image.clear();
  state = lodepng::State();
  state.info_raw.bitdepth = 16;
  ASSERT_NO_PNG_ERROR(lodepng::decode(image, w, h, state, png));
  ASSERT_EQUALS(0x77, image[84]);
  ASSERT_EQUALS(0x77, image[85]);
}

void testPredefinedFilters() {
  size_t w = 32, h = 32;
  std::cout << "testPredefinedFilters" << std::endl;
  Image image;
  generateTestImage(image, w, h, LCT_RGBA, 8);

  // everything to filter type '3'
  std::vector<unsigned char> predefined(h, 3);
  lodepng::State state;
  state.encoder.filter_strategy = LFS_PREDEFINED;
  state.encoder.filter_palette_zero = 0;
  state.encoder.predefined_filters = &predefined[0];

  std::vector<unsigned char> png;
  unsigned error = lodepng::encode(png, &image.data[0], w, h, state);
  assertNoError(error);

  std::vector<unsigned char> outfilters;
  error = lodepng::getFilterTypes(outfilters, png);
  assertNoError(error);

  ASSERT_EQUALS(outfilters.size(), h);
  for(size_t i = 0; i < h; i++) ASSERT_EQUALS(3, outfilters[i]);
}

void testEncoderErrors() {
  std::cout << "testEncoderErrors" << std::endl;

  std::vector<unsigned char> png;
  unsigned w = 32, h = 32;
  Image image;
  generateTestImage(image, w, h);

  lodepng::State def;

  lodepng::State state;

  ASSERT_EQUALS(0, lodepng::encode(png, &image.data[0], w, h, state));

  // test window sizes
  state.encoder.zlibsettings.windowsize = 0;
  ASSERT_EQUALS(60, lodepng::encode(png, &image.data[0], w, h, state));
  state.encoder.zlibsettings.windowsize = 65536;
  ASSERT_EQUALS(60, lodepng::encode(png, &image.data[0], w, h, state));
  state.encoder.zlibsettings.windowsize = 1000; // not power of two
  ASSERT_EQUALS(90, lodepng::encode(png, &image.data[0], w, h, state));
  state.encoder.zlibsettings.windowsize = 256;
  ASSERT_EQUALS(0, lodepng::encode(png, &image.data[0], w, h, state));

  state = def;
  state.info_png.color.bitdepth = 3;
  ASSERT_EQUALS(37, lodepng::encode(png, &image.data[0], w, h, state));

  state = def;
  state.info_png.color.colortype = (LodePNGColorType)5;
  ASSERT_EQUALS(31, lodepng::encode(png, &image.data[0], w, h, state));

  state = def;
  state.info_png.color.colortype = LCT_PALETTE;
  ASSERT_EQUALS(68, lodepng::encode(png, &image.data[0], w, h, state));

  state = def;
  state.info_png.interlace_method = 0;
  ASSERT_EQUALS(0, lodepng::encode(png, &image.data[0], w, h, state));
  state.info_png.interlace_method = 1;
  ASSERT_EQUALS(0, lodepng::encode(png, &image.data[0], w, h, state));
  state.info_png.interlace_method = 2;
  ASSERT_EQUALS(71, lodepng::encode(png, &image.data[0], w, h, state));

  state = def;
  state.encoder.zlibsettings.btype = 0;
  ASSERT_EQUALS(0, lodepng::encode(png, &image.data[0], w, h, state));
  state.encoder.zlibsettings.btype = 1;
  ASSERT_EQUALS(0, lodepng::encode(png, &image.data[0], w, h, state));
  state.encoder.zlibsettings.btype = 2;
  ASSERT_EQUALS(0, lodepng::encode(png, &image.data[0], w, h, state));
  state.encoder.zlibsettings.btype = 3;
  ASSERT_EQUALS(61, lodepng::encode(png, &image.data[0], w, h, state));
}

void addColor(std::vector<unsigned char>& colors, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
  colors.push_back(r);
  colors.push_back(g);
  colors.push_back(b);
  colors.push_back(a);
}

void addColor16(std::vector<unsigned char>& colors, unsigned short r, unsigned short g, unsigned short b, unsigned short a) {
  colors.push_back(r & 255);
  colors.push_back((r >> 8) & 255);
  colors.push_back(g & 255);
  colors.push_back((g >> 8) & 255);
  colors.push_back(b & 255);
  colors.push_back((b >> 8) & 255);
  colors.push_back(a & 255);
  colors.push_back((a >> 8) & 255);
}

// Tests auto_convert
// colors is in RGBA, inbitdepth must be 8 or 16, the amount of bits per channel.
// colortype and bitdepth are the expected values. insize is amount of pixels. So the amount of bytes is insize * 4 * (inbitdepth / 8)
void testAutoColorModel(const std::vector<unsigned char>& colors, unsigned inbitdepth, LodePNGColorType colortype, unsigned bitdepth, bool key) {
  std::cout << "testAutoColorModel " << inbitdepth << " " << colortype << " " << bitdepth << " " << key << std::endl;
  size_t innum = colors.size() / 4 * inbitdepth / 8;
  size_t num = innum < 65536 ? 65536 : innum; // Make image bigger so the convert doesn't avoid palette due to small image.
  std::vector<unsigned char> colors2(num * 4 * (inbitdepth / 8));
  for(size_t i = 0; i < colors2.size(); i++) colors2[i] = colors[i % colors.size()];

  std::vector<unsigned char> png;
  lodepng::encode(png, colors2, num, 1, LCT_RGBA, inbitdepth);

  // now extract the color type it chose
  unsigned w, h;
  lodepng::State state;
  std::vector<unsigned char> decoded;
  lodepng::decode(decoded, w, h, state, png);
  ASSERT_EQUALS(num, w);
  ASSERT_EQUALS(1, h);
  ASSERT_EQUALS(colortype, state.info_png.color.colortype);
  ASSERT_EQUALS(bitdepth, state.info_png.color.bitdepth);
  ASSERT_EQUALS(key, state.info_png.color.key_defined);
  // also check that the PNG decoded correctly and has same colors as input
  if(inbitdepth == 8) { for(size_t i = 0; i < colors.size(); i++) ASSERT_EQUALS(colors[i], decoded[i]); }
  else { for(size_t i = 0; i < colors.size() / 2; i++) ASSERT_EQUALS(colors[i * 2], decoded[i]); }
}

void testAutoColorModels() {
  // 1-bit gray
  std::vector<unsigned char> gray1;
  for(size_t i = 0; i < 2; i++) addColor(gray1, i * 255, i * 255, i * 255, 255);
  testAutoColorModel(gray1, 8, LCT_GREY, 1, false);

  // 2-bit gray
  std::vector<unsigned char> gray2;
  for(size_t i = 0; i < 4; i++) addColor(gray2, i * 85, i * 85, i * 85, 255);
  testAutoColorModel(gray2, 8, LCT_GREY, 2, false);

  // 4-bit gray
  std::vector<unsigned char> gray4;
  for(size_t i = 0; i < 16; i++) addColor(gray4, i * 17, i * 17, i * 17, 255);
  testAutoColorModel(gray4, 8, LCT_GREY, 4, false);

  // 8-bit gray
  std::vector<unsigned char> gray8;
  for(size_t i = 0; i < 256; i++) addColor(gray8, i, i, i, 255);
  testAutoColorModel(gray8, 8, LCT_GREY, 8, false);

  // 16-bit gray
  std::vector<unsigned char> gray16;
  for(size_t i = 0; i < 257; i++) addColor16(gray16, i, i, i, 65535);
  testAutoColorModel(gray16, 16, LCT_GREY, 16, false);

  // 8-bit gray+alpha
  std::vector<unsigned char> gray8a;
  for(size_t i = 0; i < 17; i++) addColor(gray8a, i, i, i, i);
  testAutoColorModel(gray8a, 8, LCT_PALETTE, 8, false);
  // palette not possible, becomes gray alpha
  for(size_t i = 0; i < 256; i++) addColor(gray8a, i, i, i, i ^ 1);
  testAutoColorModel(gray8a, 8, LCT_GREY_ALPHA, 8, false);

  // 16-bit gray+alpha
  std::vector<unsigned char> gray16a;
  for(size_t i = 0; i < 257; i++) addColor16(gray16a, i, i, i, i);
  testAutoColorModel(gray16a, 16, LCT_GREY_ALPHA, 16, false);


  // various palette tests
  std::vector<unsigned char> palette;
  addColor(palette, 0, 0, 1, 255);
  testAutoColorModel(palette, 8, LCT_PALETTE, 1, false);
  addColor(palette, 0, 0, 2, 255);
  testAutoColorModel(palette, 8, LCT_PALETTE, 1, false);
  for(int i = 3; i <= 4; i++) addColor(palette, 0, 0, i, 255);
  testAutoColorModel(palette, 8, LCT_PALETTE, 2, false);
  for(int i = 5; i <= 7; i++) addColor(palette, 0, 0, i, 255);
  testAutoColorModel(palette, 8, LCT_PALETTE, 4, false);
  for(int i = 8; i <= 17; i++) addColor(palette, 0, 0, i, 255);
  testAutoColorModel(palette, 8, LCT_PALETTE, 8, false);
  addColor(palette, 0, 0, 18, 0); // transparent
  testAutoColorModel(palette, 8, LCT_PALETTE, 8, false);
  addColor(palette, 0, 0, 18, 1); // translucent
  testAutoColorModel(palette, 8, LCT_PALETTE, 8, false);

  // 1-bit gray + alpha not possible, becomes palette
  std::vector<unsigned char> gray1a;
  for(size_t i = 0; i < 2; i++) addColor(gray1a, i, i, i, 128);
  testAutoColorModel(gray1a, 8, LCT_PALETTE, 1, false);

  // 2-bit gray + alpha not possible, becomes palette
  std::vector<unsigned char> gray2a;
  for(size_t i = 0; i < 4; i++) addColor(gray2a, i, i, i, 128);
  testAutoColorModel(gray2a, 8, LCT_PALETTE, 2, false);

  // 4-bit gray + alpha not possible, becomes palette
  std::vector<unsigned char> gray4a;
  for(size_t i = 0; i < 16; i++) addColor(gray4a, i, i, i, 128);
  testAutoColorModel(gray4a, 8, LCT_PALETTE, 4, false);

  // 8-bit rgb
  std::vector<unsigned char> rgb = gray8;
  addColor(rgb, 255, 0, 0, 255);
  testAutoColorModel(rgb, 8, LCT_RGB, 8, false);

  // 8-bit rgb + key
  std::vector<unsigned char> rgb_key = rgb;
  addColor(rgb_key, 128, 0, 0, 0);
  testAutoColorModel(rgb_key, 8, LCT_RGB, 8, true);

  // 8-bit rgb, not key due to edge case: single key color, but opaque color has same RGB value
  std::vector<unsigned char> rgb_key2 = rgb_key;
  addColor(rgb_key2, 128, 0, 0, 255); // same color but opaque ==> no more key
  testAutoColorModel(rgb_key2, 8, LCT_RGBA, 8, false);

  // 8-bit rgb, not key due to semi translucent
  std::vector<unsigned char> rgb_key3 = rgb_key;
  addColor(rgb_key3, 128, 0, 0, 255); // semi-translucent ==> no more key
  testAutoColorModel(rgb_key3, 8, LCT_RGBA, 8, false);

  // 8-bit rgb, not key due to multiple transparent colors
  std::vector<unsigned char> rgb_key4 = rgb_key;
  addColor(rgb_key4, 128, 0, 0, 255);
  addColor(rgb_key4, 129, 0, 0, 255); // two different transparent colors ==> no more key
  testAutoColorModel(rgb_key4, 8, LCT_RGBA, 8, false);

  // 1-bit gray with key
  std::vector<unsigned char> gray1_key = gray1;
  gray1_key[7] = 0;
  testAutoColorModel(gray1_key, 8, LCT_GREY, 1, true);

  // 2-bit gray with key
  std::vector<unsigned char> gray2_key = gray2;
  gray2_key[7] = 0;
  testAutoColorModel(gray2_key, 8, LCT_GREY, 2, true);

  // 4-bit gray with key
  std::vector<unsigned char> gray4_key = gray4;
  gray4_key[7] = 0;
  testAutoColorModel(gray4_key, 8, LCT_GREY, 4, true);

  // 8-bit gray with key
  std::vector<unsigned char> gray8_key = gray8;
  gray8_key[7] = 0;
  testAutoColorModel(gray8_key, 8, LCT_GREY, 8, true);

  // 16-bit gray with key
  std::vector<unsigned char> gray16_key = gray16;
  gray16_key[14] = gray16_key[15] = 0;
  testAutoColorModel(gray16_key, 16, LCT_GREY, 16, true);

  // a single 16-bit color, can't become palette due to being 16-bit
  std::vector<unsigned char> small16;
  addColor16(small16, 1, 0, 0, 65535);
  testAutoColorModel(small16, 16, LCT_RGB, 16, false);

  std::vector<unsigned char> small16a;
  addColor16(small16a, 1, 0, 0, 1);
  testAutoColorModel(small16a, 16, LCT_RGBA, 16, false);

  // what we provide as 16-bit is actually representable as 8-bit, so 8-bit palette expected for single color
  std::vector<unsigned char> not16;
  addColor16(not16, 257, 257, 257, 0);
  testAutoColorModel(not16, 16, LCT_PALETTE, 1, false);

  // the rgb color is representable as 8-bit, but the alpha channel only as 16-bit, so ensure it uses 16-bit and not palette for this single color
  std::vector<unsigned char> alpha16;
  addColor16(alpha16, 257, 0, 0, 10000);
  testAutoColorModel(alpha16, 16, LCT_RGBA, 16, false);

  // 1-bit gray, with attempt to get color key but can't do it due to opaque color with same value
  std::vector<unsigned char> gray1k;
  addColor(gray1k, 0, 0, 0, 255);
  addColor(gray1k, 255, 255, 255, 255);
  addColor(gray1k, 255, 255, 255, 0);
  testAutoColorModel(gray1k, 8, LCT_PALETTE, 2, false);
}

void testPaletteToPaletteDecode() {
  std::cout << "testPaletteToPaletteDecode" << std::endl;
  // It's a bit big for a 2x2 image... but this tests needs one with 256 palette entries in it.
  std::string base64 = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAMAAABFaP0WAAAAA3NCSVQICAjb4U/gAAADAFBMVEUA"
                       "AAAAADMAAGYAAJkAAMwAAP8AMwAAMzMAM2YAM5kAM8wAM/8AZgAAZjMAZmYAZpkAZswAZv8AmQAA"
                       "mTMAmWYAmZkAmcwAmf8AzAAAzDMAzGYAzJkAzMwAzP8A/wAA/zMA/2YA/5kA/8wA//8zAAAzADMz"
                       "AGYzAJkzAMwzAP8zMwAzMzMzM2YzM5kzM8wzM/8zZgAzZjMzZmYzZpkzZswzZv8zmQAzmTMzmWYz"
                       "mZkzmcwzmf8zzAAzzDMzzGYzzJkzzMwzzP8z/wAz/zMz/2Yz/5kz/8wz//9mAABmADNmAGZmAJlm"
                       "AMxmAP9mMwBmMzNmM2ZmM5lmM8xmM/9mZgBmZjNmZmZmZplmZsxmZv9mmQBmmTNmmWZmmZlmmcxm"
                       "mf9mzABmzDNmzGZmzJlmzMxmzP9m/wBm/zNm/2Zm/5lm/8xm//+ZAACZADOZAGaZAJmZAMyZAP+Z"
                       "MwCZMzOZM2aZM5mZM8yZM/+ZZgCZZjOZZmaZZpmZZsyZZv+ZmQCZmTOZmWaZmZmZmcyZmf+ZzACZ"
                       "zDOZzGaZzJmZzMyZzP+Z/wCZ/zOZ/2aZ/5mZ/8yZ///MAADMADPMAGbMAJnMAMzMAP/MMwDMMzPM"
                       "M2bMM5nMM8zMM//MZgDMZjPMZmbMZpnMZszMZv/MmQDMmTPMmWbMmZnMmczMmf/MzADMzDPMzGbM"
                       "zJnMzMzMzP/M/wDM/zPM/2bM/5nM/8zM////AAD/ADP/AGb/AJn/AMz/AP//MwD/MzP/M2b/M5n/"
                       "M8z/M///ZgD/ZjP/Zmb/Zpn/Zsz/Zv//mQD/mTP/mWb/mZn/mcz/mf//zAD/zDP/zGb/zJn/zMz/"
                       "zP///wD//zP//2b//5n//8z///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABlenwdAAABAHRSTlP/////////////////////////"
                       "////////////////////////////////////////////////////////////////////////////"
                       "////////////////////////////////////////////////////////////////////////////"
                       "////////////////////////////////////////////////////////////////////////////"
                       "//////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                       "AAAAAAAAAAAAG8mZagAAAAlwSFlzAAAOTQAADpwB3vacVwAAAA5JREFUCJlj2CLHwHodAATjAa+k"
                       "lTE5AAAAAElFTkSuQmCC";
  std::vector<unsigned char> png;
  fromBase64(png, base64);

  std::vector<unsigned char> image;
  unsigned width, height;
  unsigned error = lodepng::decode(image, width, height, png, LCT_PALETTE, 8);
  ASSERT_EQUALS(0, error);
  ASSERT_EQUALS(2, width);
  ASSERT_EQUALS(2, height);
  ASSERT_EQUALS(180, image[0]);
  ASSERT_EQUALS(30, image[1]);
  ASSERT_EQUALS(5, image[2]);
  ASSERT_EQUALS(215, image[3]);
}

//2-bit palette
void testPaletteToPaletteDecode2() {
  std::cout << "testPaletteToPaletteDecode2" << std::endl;
  std::string base64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAgMAAAAOFJJnAAAADFBMVEX/AAAA/wAAAP/////7AGD2AAAAE0lEQVR4AWMQhAKG3VCALDIqAgDl2WYBCQHY9gAAAABJRU5ErkJggg==";
  std::vector<unsigned char> png;
  fromBase64(png, base64);

  std::vector<unsigned char> image;
  unsigned width, height;
  unsigned error = lodepng::decode(image, width, height, png, LCT_PALETTE, 8);
  ASSERT_EQUALS(0, error);
  ASSERT_EQUALS(32, width);
  ASSERT_EQUALS(32, height);
  ASSERT_EQUALS(0, image[0]);
  ASSERT_EQUALS(1, image[1]);

  //Now add a user-specified output palette, that differs from the input palette. That should give error 82.
  LodePNGState state;
  lodepng_state_init(&state);
  state.info_raw.colortype = LCT_PALETTE;
  state.info_raw.bitdepth = 8;
  lodepng_palette_add(&state.info_raw, 0, 0, 0, 255);
  lodepng_palette_add(&state.info_raw, 1, 1, 1, 255);
  lodepng_palette_add(&state.info_raw, 2, 2, 2, 255);
  lodepng_palette_add(&state.info_raw, 3, 3, 3, 255);
  unsigned char* image2 = 0;
  unsigned error2 = lodepng_decode(&image2, &width, &height, &state, &png[0], png.size());
  lodepng_state_cleanup(&state);
  ASSERT_EQUALS(82, error2);
  free(image2);
}

void assertColorProfileDataEqual(const lodepng::State& a, const lodepng::State& b) {
  ASSERT_EQUALS(a.info_png.gama_defined, b.info_png.gama_defined);
  if(a.info_png.gama_defined) {
    ASSERT_EQUALS(a.info_png.gama_gamma, b.info_png.gama_gamma);
  }

  ASSERT_EQUALS(a.info_png.chrm_defined, b.info_png.chrm_defined);
  if(a.info_png.chrm_defined) {
    ASSERT_EQUALS(a.info_png.chrm_white_x, b.info_png.chrm_white_x);
    ASSERT_EQUALS(a.info_png.chrm_white_y, b.info_png.chrm_white_y);
    ASSERT_EQUALS(a.info_png.chrm_red_x, b.info_png.chrm_red_x);
    ASSERT_EQUALS(a.info_png.chrm_red_y, b.info_png.chrm_red_y);
    ASSERT_EQUALS(a.info_png.chrm_green_x, b.info_png.chrm_green_x);
    ASSERT_EQUALS(a.info_png.chrm_green_y, b.info_png.chrm_green_y);
    ASSERT_EQUALS(a.info_png.chrm_blue_x, b.info_png.chrm_blue_x);
    ASSERT_EQUALS(a.info_png.chrm_blue_y, b.info_png.chrm_blue_y);
  }

  ASSERT_EQUALS(a.info_png.srgb_defined, b.info_png.srgb_defined);
  if(a.info_png.srgb_defined) {
    ASSERT_EQUALS(a.info_png.srgb_intent, b.info_png.srgb_intent);
  }

  ASSERT_EQUALS(a.info_png.cicp_defined, b.info_png.cicp_defined);
  if(a.info_png.cicp_defined) {
    ASSERT_EQUALS(a.info_png.cicp_color_primaries, b.info_png.cicp_color_primaries);
    ASSERT_EQUALS(a.info_png.cicp_transfer_function, b.info_png.cicp_transfer_function);
    ASSERT_EQUALS(a.info_png.cicp_matrix_coefficients, b.info_png.cicp_matrix_coefficients);
    ASSERT_EQUALS(a.info_png.cicp_video_full_range_flag, b.info_png.cicp_video_full_range_flag);
  }

  ASSERT_EQUALS(a.info_png.mdcv_defined, b.info_png.mdcv_defined);
  if(a.info_png.mdcv_defined) {
    ASSERT_EQUALS(a.info_png.mdcv_red_x, b.info_png.mdcv_red_x);
    ASSERT_EQUALS(a.info_png.mdcv_red_y, b.info_png.mdcv_red_y);
    ASSERT_EQUALS(a.info_png.mdcv_green_x, b.info_png.mdcv_green_x);
    ASSERT_EQUALS(a.info_png.mdcv_green_y, b.info_png.mdcv_green_y);
    ASSERT_EQUALS(a.info_png.mdcv_blue_x, b.info_png.mdcv_blue_x);
    ASSERT_EQUALS(a.info_png.mdcv_blue_y, b.info_png.mdcv_blue_y);
    ASSERT_EQUALS(a.info_png.mdcv_white_x, b.info_png.mdcv_white_x);
    ASSERT_EQUALS(a.info_png.mdcv_white_y, b.info_png.mdcv_white_y);
    ASSERT_EQUALS(a.info_png.mdcv_max_luminance, b.info_png.mdcv_max_luminance);
    ASSERT_EQUALS(a.info_png.mdcv_min_luminance, b.info_png.mdcv_min_luminance);
  }

  ASSERT_EQUALS(a.info_png.clli_defined, b.info_png.clli_defined);
  if(a.info_png.clli_defined) {
    ASSERT_EQUALS(a.info_png.clli_max_cll, b.info_png.clli_max_cll);
    ASSERT_EQUALS(a.info_png.clli_max_fall, b.info_png.clli_max_fall);
  }

  ASSERT_EQUALS(a.info_png.iccp_defined, b.info_png.iccp_defined);
  if(a.info_png.iccp_defined) {
    //ASSERT_EQUALS(std::string(a.info_png.iccp_name), std::string(b.info_png.iccp_name));
    ASSERT_EQUALS(a.info_png.iccp_profile_size, b.info_png.iccp_profile_size);
    for(size_t i = 0; i < a.info_png.iccp_profile_size; ++i) {
      ASSERT_EQUALS(a.info_png.iccp_profile[i], b.info_png.iccp_profile[i]);
    }
  }
}

// Tests the gAMA, cHRM, sRGB, iCCP chunks
void testColorProfile() {
  std::cout << "testColorProfile" << std::endl;
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i < image.size(); i++) image[i] = i & 255;
    std::vector<unsigned char> png;
    lodepng::State state;
    state.info_png.gama_defined = 1;
    state.info_png.gama_gamma = 12345;
    state.info_png.chrm_defined = 1;
    state.info_png.chrm_white_x = 10;
    state.info_png.chrm_white_y = 20;
    state.info_png.chrm_red_x = 30;
    state.info_png.chrm_red_y = 40;
    state.info_png.chrm_green_x = 100000;
    state.info_png.chrm_green_y = 200000;
    state.info_png.chrm_blue_x = 300000;
    state.info_png.chrm_blue_y = 400000;
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    ASSERT_EQUALS(32, w);
    ASSERT_EQUALS(32, h);
    ASSERT_EQUALS(image.size(), image2.size());
    for(size_t i = 0; i < image.size(); i++) ASSERT_EQUALS(image[i], image2[i]);
  }
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i < image.size(); i++) image[i] = i & 255;
    std::vector<unsigned char> png;
    lodepng::State state;
    state.info_png.srgb_defined = 1;
    state.info_png.srgb_intent = 2;
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    ASSERT_EQUALS(32, w);
    ASSERT_EQUALS(32, h);
    ASSERT_EQUALS(image.size(), image2.size());
    for(size_t i = 0; i < image.size(); i++) ASSERT_EQUALS(image[i], image2[i]);
  }
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i < image.size(); i++) image[i] = i & 255;
    std::vector<unsigned char> png;
    lodepng::State state;
    state.info_png.cicp_defined = 1;
    state.info_png.cicp_color_primaries = 4;
    state.info_png.cicp_transfer_function = 3;
    state.info_png.cicp_matrix_coefficients = 2;
    state.info_png.cicp_video_full_range_flag = 1;
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    ASSERT_EQUALS(32, w);
    ASSERT_EQUALS(32, h);
    ASSERT_EQUALS(image.size(), image2.size());
    for(size_t i = 0; i < image.size(); i++) ASSERT_EQUALS(image[i], image2[i]);
  }
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i < image.size(); i++) image[i] = i & 255;
    std::vector<unsigned char> png;
    lodepng::State state;
    state.info_png.mdcv_defined = 1;
    state.info_png.mdcv_red_x = 2;
    state.info_png.mdcv_red_y = 3;
    state.info_png.mdcv_green_x = 4;
    state.info_png.mdcv_green_y = 5;
    state.info_png.mdcv_blue_x = 6;
    state.info_png.mdcv_blue_y = 7;
    state.info_png.mdcv_white_x = 8;
    state.info_png.mdcv_white_y = 9;
    state.info_png.mdcv_max_luminance = 10;
    state.info_png.mdcv_min_luminance = 11;
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    ASSERT_EQUALS(32, w);
    ASSERT_EQUALS(32, h);
    ASSERT_EQUALS(image.size(), image2.size());
    for(size_t i = 0; i < image.size(); i++) ASSERT_EQUALS(image[i], image2[i]);
  }
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i < image.size(); i++) image[i] = i & 255;
    std::vector<unsigned char> png;
    lodepng::State state;
    state.info_png.clli_defined = 1;
    state.info_png.clli_max_cll = 2;
    state.info_png.clli_max_fall = 3;
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    ASSERT_EQUALS(32, w);
    ASSERT_EQUALS(32, h);
    ASSERT_EQUALS(image.size(), image2.size());
    for(size_t i = 0; i < image.size(); i++) ASSERT_EQUALS(image[i], image2[i]);
  }
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i < image.size(); i++) image[i] = i & 255;
    std::vector<unsigned char> png;
    lodepng::State state;
    std::string testprofile = "0123456789abcdefRGB fake iccp profile for testing";
    testprofile[0] = testprofile[1] = 0;
    lodepng_set_icc(&state.info_png, "test", (const unsigned char*)testprofile.c_str(), testprofile.size());
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    ASSERT_EQUALS(32, w);
    ASSERT_EQUALS(32, h);
    ASSERT_EQUALS(image.size(), image2.size());
    for(size_t i = 0; i < image.size(); i++) ASSERT_EQUALS(image[i], image2[i]);
  }

  // grayscale ICC profile
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i + 4 <= image.size(); i += 4) {
      image[i] = image[i + 1] = image[i + 2] = image[i + 3] = i;
    }
    std::vector<unsigned char> png;
    lodepng::State state;
    std::string testprofile = "0123456789abcdefGRAYfake iccp profile for testing";
    testprofile[0] = testprofile[1] = 0;
    lodepng_set_icc(&state.info_png, "test", (const unsigned char*)testprofile.c_str(), testprofile.size());
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    ASSERT_EQUALS(32, w);
    ASSERT_EQUALS(32, h);
    ASSERT_EQUALS(image.size(), image2.size());
    for(size_t i = 0; i < image.size(); i++) ASSERT_EQUALS(image[i], image2[i]);
  }

  // grayscale ICC profile, using an input image with grayscale colors but that
  // would normally benefit from a palette (which auto_convert would normally
  // choose). But the PNG spec does not allow combining palette with GRAY ICC
  // profile, so the encoder should not choose to use palette after all.
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    int colors[3] = {0, 3, 133};
    for(size_t i = 0; i + 4 <= image.size(); i += 4) {
      image[i] = image[i + 1] = image[i + 2] = image[i + 3] = colors[(i / 4) % 3];
    }
    std::vector<unsigned char> png;
    lodepng::State state;
    std::string testprofile = "0123456789abcdefGRAYfake iccp profile for testing";
    testprofile[0] = testprofile[1] = 0;
    lodepng_set_icc(&state.info_png, "test", (const unsigned char*)testprofile.c_str(), testprofile.size());
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    ASSERT_NOT_EQUALS(LCT_PALETTE, state2.info_png.color.colortype);
  }

  // RGB ICC profile, using an input image with grayscale colors: the encoder
  // is forced to choose an RGB color type anyway with auto_convert
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i + 4 <= image.size(); i += 4) {
      image[i] = image[i + 1] = image[i + 2] = (i / 4) & 255;
      image[i + 3] = 255;
    }
    std::vector<unsigned char> png;
    lodepng::State state;
    std::string testprofile = "0123456789abcdefRGB fake iccp profile for testing";
    testprofile[0] = testprofile[1] = 0;
    lodepng_set_icc(&state.info_png, "test", (const unsigned char*)testprofile.c_str(), testprofile.size());
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertColorProfileDataEqual(state, state2);
    // LCT_RGB or LCT_PALETTE are both ok, gray is not (it likely chooses palette in practice)
    ASSERT_NOT_EQUALS(LCT_GREY, state2.info_png.color.colortype);
    ASSERT_NOT_EQUALS(LCT_GREY_ALPHA, state2.info_png.color.colortype);
  }

  // Encoder must give error when forcing invalid combination of color/gray
  // PNG with gray/color ICC Profile
  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    int colors[3] = {0, 5, 33};
    for(size_t i = 0; i + 4 <= image.size(); i += 4) {
      image[i] = 255;
      image[i + 1] = image[i + 2] = image[i + 3] = colors[(i / 4) % 3];
    }
    std::vector<unsigned char> png;
    lodepng::State state;
    std::string testprofile = "0123456789abcdefGRAYfake iccp profile for testing";
    testprofile[0] = testprofile[1] = 0;
    lodepng_set_icc(&state.info_png, "test", (const unsigned char*)testprofile.c_str(), testprofile.size());
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NOT_EQUALS(0, error);  // must give error due to color image input with gray profile
  }
}

void assertExifDataEqual(const lodepng::State& a, const lodepng::State& b) {
  ASSERT_EQUALS(a.info_png.exif_defined, b.info_png.exif_defined);
  if(!a.info_png.exif_defined) return;

  ASSERT_EQUALS(a.info_png.exif_size, b.info_png.exif_size);
  for(size_t i = 0; i < a.info_png.exif_size; i++) {
    ASSERT_EQUALS(a.info_png.exif[i], b.info_png.exif[i]);
  }
}

void testExif() {
  std::cout << "testExif" << std::endl;

  {
    unsigned error;
    unsigned w = 32, h = 32;
    std::vector<unsigned char> image(w * h * 4);
    for(size_t i = 0; i + 4 <= image.size(); i += 4) {
      image[i] = image[i + 1] = image[i + 2] = image[i + 3] = i;
    }
    std::vector<unsigned char> png;
    lodepng::State state;
    std::string testexif = "MM  0123456789";
    lodepng_set_exif(&state.info_png, (const unsigned char*)testexif.c_str(), testexif.size());
    error = lodepng::encode(png, &image[0], w, h, state);
    ASSERT_NO_PNG_ERROR(error);

    lodepng::State state2;
    std::vector<unsigned char> image2;
    error = lodepng::decode(image2, w, h, state2, png);
    ASSERT_NO_PNG_ERROR(error);
    assertExifDataEqual(state, state2);
    ASSERT_EQUALS(32, w);
    ASSERT_EQUALS(32, h);
    ASSERT_EQUALS(image.size(), image2.size());
  }

  {
    // exif2c08.png PngSuite image
    std::string base64 = "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";
    std::vector<unsigned char> png;
    fromBase64(png, base64);
    lodepng::State state;
    std::vector<unsigned char> image;
    unsigned w, h;
    unsigned error = lodepng::decode(image, w, h, state, png);
    ASSERT_NO_PNG_ERROR(error);
    ASSERT_EQUALS(978, state.info_png.exif_size);
    ASSERT_EQUALS("072f0ad39affebf437689f935fab270c", md5sum(state.info_png.exif, state.info_png.exif_size));
  }
}

// r, g, b is input background color to encoder, given in png color model
// r2, g2, b2 is expected decoded background color, in color model it auto chose if auto_convert is on
// pixels must be given in mode_raw color format
void testBkgdChunk(unsigned r, unsigned g, unsigned b,
                   unsigned r2, unsigned g2, unsigned b2,
                   const std::vector<unsigned char>& pixels,
                   unsigned w, unsigned h,
                   const LodePNGColorMode& mode_raw,
                   const LodePNGColorMode& mode_png,
                   bool auto_convert, bool expect_encoder_error = false) {
  unsigned error;

  lodepng::State state;
  LodePNGInfo& info = state.info_png;
  lodepng_color_mode_copy(&info.color, &mode_png);
  lodepng_color_mode_copy(&state.info_raw, &mode_raw);
  state.encoder.auto_convert = auto_convert;

  info.background_defined = 1;
  info.background_r = r;
  info.background_g = g;
  info.background_b = b;

  std::vector<unsigned char> png;
  error = lodepng::encode(png, pixels, w, h, state);
  if(expect_encoder_error) {
    ASSERT_NOT_EQUALS(0, error);
    return;
  }
  ASSERT_NO_PNG_ERROR(error);

  lodepng::State state2;
  LodePNGInfo& info2 = state2.info_png;
  state2.info_raw.colortype = LCT_RGBA;
  state2.info_raw.bitdepth = 16;
  unsigned w2, h2;
  std::vector<unsigned char> image2;
  error = lodepng::decode(image2, w2, h2, state2, &png[0], png.size());
  ASSERT_NO_PNG_ERROR(error);

  ASSERT_EQUALS(w, w2);
  ASSERT_EQUALS(h, h2);
  ASSERT_EQUALS(1, info2.background_defined);
  ASSERT_EQUALS(r2, info2.background_r);
  ASSERT_EQUALS(g2, info2.background_g);
  ASSERT_EQUALS(b2, info2.background_b);

  // compare pixels in the "raw" color model
  LodePNGColorMode mode_decoded; lodepng_color_mode_init(&mode_decoded); mode_decoded.bitdepth = 16; mode_decoded.colortype = LCT_RGBA;
  std::vector<unsigned char> image3((w * h * lodepng_get_bpp(&mode_raw) + 7) / 8);
  error = lodepng_convert(image3.data(), image2.data(), &mode_raw, &mode_decoded, w, h);
  ASSERT_NO_PNG_ERROR(error);
  ASSERT_EQUALS(pixels.size(), image3.size());
  for(size_t i = 0; i < image3.size(); i++) {
    ASSERT_EQUALS((int)image3[i], (int)pixels[i]);
  }
}

// r, g, b is input background color to encoder, given in png color model
// r2, g2, b2 is expected decoded background color, in color model it auto chose if auto_convert is on
void testBkgdChunk(unsigned r, unsigned g, unsigned b,
                   unsigned r2, unsigned g2, unsigned b2,
                   LodePNGColorType type_pixels, unsigned bitdepth_pixels,
                   LodePNGColorType type_raw, unsigned bitdepth_raw,
                   LodePNGColorType type_png, unsigned bitdepth_png,
                   bool auto_convert, bool expect_encoder_error = false) {
  unsigned error;
  Image image;
  generateTestImageRequiringColorType16(image, type_pixels, bitdepth_pixels, false);

  LodePNGColorMode mode_raw; lodepng_color_mode_init(&mode_raw); mode_raw.bitdepth = bitdepth_raw; mode_raw.colortype = type_raw;
  LodePNGColorMode mode_test; lodepng_color_mode_init(&mode_test); mode_test.bitdepth = 16; mode_test.colortype = LCT_RGBA;
  LodePNGColorMode mode_png; lodepng_color_mode_init(&mode_png); mode_png.bitdepth = bitdepth_png; mode_png.colortype = type_png;
  std::vector<unsigned char> temp((image.width * image.height * lodepng_get_bpp(&mode_raw) + 7) / 8);
  error = lodepng_convert(temp.data(), image.data.data(), &mode_raw, &mode_test, image.width, image.height);
  ASSERT_NO_PNG_ERROR(error);
  image.data = temp;

  testBkgdChunk(r, g, b, r2, g2, b2,
                image.data, image.width, image.height,
                mode_raw, mode_png, auto_convert, expect_encoder_error);
}

void testBkgdChunk() {
  std::cout << "testBkgdChunk" << std::endl;
  // color param order is: generated, raw, png ( == bKGD)
  // here generated means: what color values the pixels will get, so what auto_convert will make it choose
  testBkgdChunk(255, 0, 0, 255, 0, 0, LCT_RGBA, 8, LCT_RGBA, 8, LCT_RGBA, 8, true);
  testBkgdChunk(255, 0, 0, 255, 0, 0, LCT_RGBA, 8, LCT_RGB, 8, LCT_RGB, 8, true);
  testBkgdChunk(255, 0, 0, 255, 0, 0, LCT_RGB, 8, LCT_RGB, 8, LCT_RGB, 8, true);
  testBkgdChunk(255, 255, 255, 1, 1, 1, LCT_GREY, 1, LCT_RGB, 8, LCT_RGB, 8, true);
  testBkgdChunk(255, 255, 255, 3, 3, 3, LCT_GREY, 2, LCT_RGB, 8, LCT_RGB, 8, true);
  testBkgdChunk(255, 255, 255, 15, 15, 15, LCT_GREY, 4, LCT_RGB, 8, LCT_RGB, 8, true);
  testBkgdChunk(255, 255, 255, 255, 255, 255, LCT_GREY, 8, LCT_RGB, 8, LCT_RGB, 8, true);
  testBkgdChunk(255, 255, 255, 65535, 65535, 65535, LCT_GREY, 16, LCT_RGB, 16, LCT_RGB, 8, true);
  testBkgdChunk(123, 0, 0, 123, 0, 0, LCT_GREY, 1, LCT_RGB, 8, LCT_RGB, 8, true);
  testBkgdChunk(170, 170, 170, 2, 2, 2, LCT_GREY, 1, LCT_RGB, 8, LCT_RGB, 8, true); // 170 = value 2 in 2-bit

  // without auto_convert. Note that it will still convert if different colortype is given for raw and png, it's just
  // not automatic in that case.
  testBkgdChunk(255, 0, 0, 255, 0, 0, LCT_RGBA, 8, LCT_RGBA, 8, LCT_RGBA, 8, false);
  testBkgdChunk(60000, 0, 0, 60000, 0, 0, LCT_RGBA, 8, LCT_RGBA, 8, LCT_RGBA, 16, false);
  testBkgdChunk(128, 128, 128, 128, 128, 128, LCT_GREY, 8, LCT_RGBA, 8, LCT_GREY, 8, false);
 {
    LodePNGColorMode pal;
    lodepng_color_mode_init(&pal);
    for(int i = 0; i < 200; i++) lodepng_palette_add(&pal, i, i / 2, 0, 255);
    pal.colortype = LCT_PALETTE;
    pal.bitdepth = 8;
    unsigned w = 200;
    unsigned h = 200;
    std::vector<unsigned char> img(w * h);
    for(unsigned y = 0; y < h; y++)
    for(unsigned x = 0; x < w; x++) {
      img[y * w + x] = x;
    }

    testBkgdChunk(100, 0, 0, 100, 100, 100, img, w, h, pal, pal, true, false);
    testBkgdChunk(100, 0, 0, 100, 100, 100, img, w, h, pal, pal, false, false);
    testBkgdChunk(250, 0, 0, 250, 250, 250, img, w, h, pal, pal, true, true);

    std::vector<unsigned char> fourcolor(w * h);
    for(unsigned y = 0; y < h; y++)
    for(unsigned x = 0; x < w; x++) {
      fourcolor[y * w + x] = x & 3;
    }
    // palette index 4 expected for output bKGD: auto_convert should turn the 200-sized
    // palette in one of size 5, 4 values for the fourcolor image above, and then a 5th for
    // the bkgd index. The other two 4's actually shouldn't matter, it's not defined what
    // they should be though currently lodepng sets them also to the palette index...
    testBkgdChunk(100, 0, 0, 4, 4, 4, fourcolor, w, h, pal, pal, true, false);


    std::vector<unsigned char> mini(4);
    mini[0] = 1; mini[1] = 2; mini[2] = 3; mini[3] = 4;
    // here we expect RGB color from the output image, since the image is tiny so it chooses to not add PLTE
    testBkgdChunk(100, 0, 0, 100, 50, 0, mini, 2, 2, pal, pal, true, false);

    lodepng_color_mode_cleanup(&pal);
  }
}

void testBkgdChunk2() {
  std::cout << "testBkgdChunk2" << std::endl;
  Image image;
  generateTestImageRequiringColorType8(image, LCT_GREY, 2, false);

  // without background, it should choose 2-bit gray for this PNG
  std::vector<unsigned char> png0;
  ASSERT_NO_PNG_ERROR(lodepng::encode(png0, image.data, image.width, image.height));
  lodepng::State state0;
  unsigned w0, h0;
  lodepng_inspect(&w0, &h0, &state0, png0.data(), png0.size());
  ASSERT_EQUALS(2, state0.info_png.color.bitdepth);
  ASSERT_EQUALS(LCT_GREY, state0.info_png.color.colortype);

  // red background, with auto_convert, it is forced to choose RGB
  lodepng::State state;
  LodePNGInfo& info = state.info_png;
  info.background_defined = 1;
  info.background_r = 255;
  info.background_g = 0;
  info.background_b = 0;
  std::vector<unsigned char> png1;
  ASSERT_NO_PNG_ERROR(lodepng::encode(png1, image.data, image.width, image.height, state));
  lodepng::State state1;
  unsigned w1, h1;
  lodepng_inspect(&w1, &h1, &state1, png1.data(), png1.size());
  ASSERT_EQUALS(8, state1.info_png.color.bitdepth);
  ASSERT_EQUALS(LCT_RGB, state1.info_png.color.colortype);

  // gray output required, background color also interpreted as gray
  state.info_raw.colortype = LCT_RGB;
  state.info_png.color.colortype = LCT_GREY;
  state.info_png.color.bitdepth = 1;
  state.encoder.auto_convert = 0;
  info.background_defined = 1;
  info.background_r = 1;
  info.background_g = 1;
  info.background_b = 1;
  std::vector<unsigned char> png2;
  ASSERT_NO_PNG_ERROR(lodepng::encode(png2, image.data, image.width, image.height, state));
  lodepng::State state2;
  unsigned w2, h2;
  lodepng_inspect(&w2, &h2, &state2, png2.data(), png2.size());
  ASSERT_EQUALS(1, state2.info_png.color.bitdepth);
  ASSERT_EQUALS(LCT_GREY, state2.info_png.color.colortype);
}

// r, g, b, a are the bit depths to store
void testSbitChunk(unsigned r, unsigned g, unsigned b, unsigned a,
                   const std::vector<unsigned char>& pixels,
                   unsigned w, unsigned h,
                   const LodePNGColorMode& mode_raw,
                   const LodePNGColorMode& mode_png,
                   bool auto_convert,
                   bool expect_encoder_error = false) {
  unsigned error;

  lodepng::State state;
  LodePNGInfo& info = state.info_png;
  lodepng_color_mode_copy(&info.color, &mode_png);
  lodepng_color_mode_copy(&state.info_raw, &mode_raw);
  state.encoder.auto_convert = auto_convert;
  if(mode_raw.colortype == LCT_PALETTE) {
    for(size_t i = 0; i < 256; i++) {
      // TODO: consider allowing to set only 1 of these palettes in lodepng in the case
      // where both info_raw and info_png have the palette color type
      lodepng_palette_add(&state.info_raw, i, i, i, 255);
      lodepng_palette_add(&info.color, i, i, i, 255);
    }
  }

  info.sbit_defined = 1;
  info.sbit_r = r;
  info.sbit_g = g;
  info.sbit_b = b;
  info.sbit_a = a;

  std::vector<unsigned char> png;
  error = lodepng::encode(png, pixels, w, h, state);
  if(expect_encoder_error) {
    ASSERT_NOT_EQUALS(0, error);
    return;
  }
  ASSERT_NO_PNG_ERROR(error);

  lodepng::State state2;
  LodePNGInfo& info2 = state2.info_png;
  unsigned w2, h2;
  std::vector<unsigned char> image2;
  error = lodepng::decode(image2, w2, h2, state2, &png[0], png.size());
  ASSERT_NO_PNG_ERROR(error);

  LodePNGColorType type = mode_png.colortype;

  ASSERT_EQUALS(w, w2);
  ASSERT_EQUALS(h, h2);
  ASSERT_EQUALS(1, info2.sbit_defined);
  ASSERT_EQUALS(r, info2.sbit_r);
  if(type == LCT_RGB || type == LCT_RGBA || type == LCT_PALETTE) {
    ASSERT_EQUALS(g, info2.sbit_g);
    ASSERT_EQUALS(b, info2.sbit_b);
  }
  if(type == LCT_GREY_ALPHA || type == LCT_RGBA) {
    ASSERT_EQUALS(a, info2.sbit_a);
  }

  // compare pixels in a 16-bit color model
  LodePNGColorMode mode_compare; lodepng_color_mode_init(&mode_compare); mode_compare.bitdepth = 16; mode_compare.colortype = LCT_RGBA;
  LodePNGColorMode mode_decoded; lodepng_color_mode_init(&mode_decoded); mode_decoded.bitdepth = 8; mode_decoded.colortype = LCT_RGBA;
  std::vector<unsigned char> image3(w * h * 8);
  error = lodepng_convert(image3.data(), image2.data(), &mode_compare, &mode_decoded, w, h);
  std::vector<unsigned char> image4(w * h * 8);
  error = lodepng_convert(image4.data(), pixels.data(), &mode_compare, &state.info_raw, w, h);
  ASSERT_NO_PNG_ERROR(error);
  ASSERT_EQUALS(image4.size(), image3.size());
  for(size_t i = 0; i < image3.size(); i++) {
    ASSERT_EQUALS((int)image4[i], (int)image3[i]);
  }
}


void testSbitChunk(unsigned r, unsigned g, unsigned b, unsigned a,
                   LodePNGColorType type, unsigned bitdepth,
                   bool expect_encoder_error = false) {
  LodePNGColorMode mode_raw;
  lodepng_color_mode_init(&mode_raw);
  mode_raw.bitdepth = bitdepth;
  mode_raw.colortype = type;
  LodePNGColorMode mode_png;
  lodepng_color_mode_init(&mode_png);
  mode_png.bitdepth = bitdepth;
  mode_png.colortype = type;

  std::vector<unsigned char> pixels(8, 255); // force all pixels to be white, so encoder tries to use auto_convert as much as possible

  testSbitChunk(r, g, b, a, pixels, 1, 1, mode_raw, mode_png, false, expect_encoder_error);
  testSbitChunk(r, g, b, a, pixels, 1, 1, mode_raw, mode_png, true, expect_encoder_error);
}

// type_pixels = what the pixels should require at least for auto_convert
// type_raw = actual raw pixel type to give to the encoder
// type_png = PNG type to request from the encoder (if not auto_convert)
// auto_convert: 0 = no, 1 = yes, 2 = try both
void testSbitChunk2(unsigned r, unsigned g, unsigned b, unsigned a,
                   LodePNGColorType type_pixels, unsigned bitdepth_pixels,
                   LodePNGColorType type_raw, unsigned bitdepth_raw,
                   LodePNGColorType type_png, unsigned bitdepth_png,
                   int auto_convert,
                   bool expect_encoder_error = false) {

  unsigned error;
  Image image;
  generateTestImageRequiringColorType16(image, type_pixels, bitdepth_pixels, false);

  LodePNGColorMode mode_raw; lodepng_color_mode_init(&mode_raw); mode_raw.bitdepth = bitdepth_raw; mode_raw.colortype = type_raw;
  LodePNGColorMode mode_test; lodepng_color_mode_init(&mode_test); mode_test.bitdepth = 16; mode_test.colortype = LCT_RGBA;
  LodePNGColorMode mode_png; lodepng_color_mode_init(&mode_png); mode_png.bitdepth = bitdepth_png; mode_png.colortype = type_png;
  std::vector<unsigned char> temp((image.width * image.height * lodepng_get_bpp(&mode_raw) + 7) / 8);
  error = lodepng_convert(temp.data(), image.data.data(), &mode_raw, &mode_test, image.width, image.height);
  ASSERT_NO_PNG_ERROR(error);
  image.data = temp;

  if(auto_convert == 0 || auto_convert == 2) testSbitChunk(r, g, b, a, image.data, image.width, image.height, mode_raw, mode_png, false, expect_encoder_error);
  if(auto_convert == 1 || auto_convert == 2) testSbitChunk(r, g, b, a, image.data, image.width, image.height, mode_raw, mode_png, true, expect_encoder_error);
}

// Test the sBIT chunk for all color types, and for possible combinations of pixel colors where auto_convert conversions occur (only conversions that
// still allow storing all the sBIT information within the PNG specification limitations may occur)
void testSbitChunk() {
  std::cout << "testSbitChunk" << std::endl;
  testSbitChunk(8, 8, 8, 0, LCT_RGB, 8, false);
  testSbitChunk(1, 2, 3, 0, LCT_RGB, 8, false);
  testSbitChunk(0, 2, 3, 0, LCT_RGB, 8, true);
  testSbitChunk(9, 2, 3, 0, LCT_RGB, 8, true);

  testSbitChunk(8, 8, 8, 8, LCT_RGBA, 8, false);
  testSbitChunk(1, 2, 3, 4, LCT_RGBA, 8, false);
  testSbitChunk(0, 2, 3, 4, LCT_RGBA, 8, true);
  testSbitChunk(9, 2, 3, 4, LCT_RGBA, 8, true);

  testSbitChunk(1, 2, 3, 0, LCT_RGB, 16, false);
  testSbitChunk(0, 2, 3, 0, LCT_RGB, 16, true);
  testSbitChunk(9, 2, 3, 0, LCT_RGB, 16, false);
  testSbitChunk(17, 2, 3, 0, LCT_RGB, 16, true);

  testSbitChunk(1, 2, 3, 4, LCT_RGBA, 16, false);
  testSbitChunk(0, 2, 3, 4, LCT_RGBA, 16, true);
  testSbitChunk(9, 2, 3, 4, LCT_RGBA, 16, false);
  testSbitChunk(17, 2, 3, 4, LCT_RGBA, 16, true);

  testSbitChunk(8, 2, 3, 0, LCT_PALETTE, 8, false);
  testSbitChunk(8, 2, 3, 0, LCT_PALETTE, 4, false); // 4-bit palette still treats the RGB as 8-bit
  testSbitChunk(9, 2, 3, 0, LCT_PALETTE, 8, true);

  testSbitChunk(8, 0, 0, 0, LCT_GREY, 8, false);
  testSbitChunk(8, 0, 0, 0, LCT_GREY, 4, true);
  testSbitChunk(5, 0, 0, 0, LCT_GREY, 8, false);
  testSbitChunk(1, 0, 0, 0, LCT_GREY, 1, false);
  testSbitChunk(3, 0, 0, 0, LCT_GREY, 1, true);
  testSbitChunk(0, 0, 0, 0, LCT_GREY, 1, true);

  testSbitChunk(16, 0, 0, 0, LCT_GREY, 16, false);
  testSbitChunk(17, 0, 0, 0, LCT_GREY, 16, true);
  testSbitChunk(8, 0, 0, 0, LCT_GREY, 16, false);
  testSbitChunk(5, 0, 0, 0, LCT_GREY, 16, false);

  testSbitChunk(8, 0, 0, 8, LCT_GREY_ALPHA, 8, false);
  testSbitChunk(8, 0, 0, 0, LCT_GREY_ALPHA, 8, true);
  testSbitChunk(8, 0, 0, 9, LCT_GREY_ALPHA, 8, true);
  testSbitChunk(5, 0, 0, 5, LCT_GREY_ALPHA, 8, false);
  testSbitChunk(5, 0, 0, 8, LCT_GREY_ALPHA, 8, false);

  testSbitChunk(16, 0, 0, 16, LCT_GREY_ALPHA, 16, false);
  testSbitChunk(16, 0, 0, 8, LCT_GREY_ALPHA, 16, false);
  testSbitChunk(8, 0, 0, 8, LCT_GREY_ALPHA, 16, false);
  testSbitChunk(16, 0, 0, 0, LCT_GREY_ALPHA, 16, true);
  testSbitChunk(16, 0, 0, 17, LCT_GREY_ALPHA, 16, true);
  testSbitChunk(5, 0, 0, 5, LCT_GREY_ALPHA, 16, false);
  testSbitChunk(5, 0, 0, 8, LCT_GREY_ALPHA, 16, false);

  testSbitChunk2(8, 8, 8, 0, LCT_RGB, 8, LCT_RGB, 8, LCT_RGB, 8, 2, false);
  testSbitChunk2(8, 8, 8, 0, LCT_GREY, 8, LCT_RGB, 8, LCT_RGB, 8, 2, false);
  testSbitChunk2(12, 12, 12, 0, LCT_GREY, 8, LCT_RGB, 16, LCT_RGB, 16, 2, false);
  testSbitChunk2(12, 12, 12, 8, LCT_GREY, 8, LCT_RGBA, 16, LCT_RGBA, 16, 2, false);
  testSbitChunk2(8, 8, 8, 0, LCT_GREY, 8, LCT_RGB, 16, LCT_RGB, 16, 2, false);
  testSbitChunk2(8, 7, 8, 0, LCT_GREY, 8, LCT_RGB, 16, LCT_RGB, 16, 2, false);
  testSbitChunk2(8, 8, 7, 0, LCT_GREY, 8, LCT_RGB, 16, LCT_RGB, 16, 2, false);
  testSbitChunk2(8, 7, 8, 0, LCT_GREY, 8, LCT_RGB, 8, LCT_RGB, 8, 2, false);
  testSbitChunk2(8, 8, 7, 0, LCT_GREY, 8, LCT_RGB, 8, LCT_RGB, 8, 2, false);
  testSbitChunk2(8, 8, 8, 0, LCT_GREY, 8, LCT_RGB, 8, LCT_GREY_ALPHA, 8, 1, false);
  testSbitChunk2(8, 8, 8, 8, LCT_GREY, 8, LCT_RGB, 8, LCT_GREY_ALPHA, 8, 0, false);


  // test png-suite image cs3n3p08.png, which has an sBIT chunk with RGB values set to 3 bits
  {
    std::vector<unsigned char> png, decoded;
    fromBase64(png, std::string("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAANzQklUAwMDo5KgQgAAAFRQTFRFkv8AAP+SAP//AP8AANv/AP9t/7YAAG3/tv8A/5IA2/8AAEn//yQA/wAAJP8ASf8AAP/bAP9JAP+2//8AAP8kALb//9sAAJL//20AACT//0kAbf8A33ArFwAAAEtJREFUeJyFyscBggAAALGzYldUsO2/pyMk73SGGE7QF3pDe2gLzdADHA7QDqIfdIUu0AocntAIbaAFdIdu0BIc1tAEvaABOkIf+AMiQDPhd/SuJgAAAABJRU5ErkJggg=="));
    lodepng::State state;
    unsigned w, h;

    unsigned error = lodepng::decode(decoded, w, h, state, png);
    assertNoError(error);
    ASSERT_EQUALS(1, state.info_png.sbit_defined);
    ASSERT_EQUALS(3, state.info_png.sbit_r);
    ASSERT_EQUALS(3, state.info_png.sbit_g);
    ASSERT_EQUALS(3, state.info_png.sbit_b);
  }

  // test png-suite image basn0g02.png, which is known to not have an sBIT chunk
  {
    std::vector<unsigned char> png, decoded;
    fromBase64(png, std::string("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAgAAAAAcoT2JAAAABGdBTUEAAYagMeiWXwAAAB9JREFUeJxjYAhd9R+M8TCIUMIAU4aPATMJH2OQuQcAvUl/gYsJiakAAAAASUVORK5CYII="));
    lodepng::State state;
    unsigned w, h;

    unsigned error = lodepng::decode(decoded, w, h, state, png);
    assertNoError(error);
    ASSERT_EQUALS(0, state.info_png.sbit_defined);
  }
}

// Test particular cHRM+gAMA conversion to srgb
// gamma = gamma given 100000x multiplied form of PNG, or 0 to set none at all
// wx..by = whitepoint and chromaticities, given in the 100000x multiplied form of PNG
// r, g, b: r, g, b values to encode in the PNG's data
// er, eg, eb: expected r, g, b values after decoding and converting to sRGB
void testChrmToSrgb(unsigned gamma, unsigned wx, unsigned wy, unsigned rx, unsigned ry, unsigned gx, unsigned gy, unsigned bx, unsigned by,
                    unsigned char r, unsigned char g, unsigned char b, unsigned char er, unsigned char eg, unsigned char eb,
                    int max_dist = 0) {
  std::vector<unsigned char> image(4);
  image[0] = r;
  image[1] = g;
  image[2] = b;
  image[3] = 255;
  lodepng::State state;
  if(gamma) {
    state.info_png.gama_defined = 1;
    state.info_png.gama_gamma = gamma;
  }
  state.info_png.chrm_defined = 1;
  state.info_png.chrm_white_x = wx;
  state.info_png.chrm_white_y = wy;
  state.info_png.chrm_red_x = rx;
  state.info_png.chrm_red_y = ry;
  state.info_png.chrm_green_x = gx;
  state.info_png.chrm_green_y = gy;
  state.info_png.chrm_blue_x = bx;
  state.info_png.chrm_blue_y = by;

  std::vector<unsigned char> image2(4);
  convertToSrgb(image2.data(), image.data(), 1, 1, &state);

  if(max_dist == 0) {
    ASSERT_EQUALS(er, image2[0]);
    ASSERT_EQUALS(eg, image2[1]);
    ASSERT_EQUALS(eb, image2[2]);
  } else {
    ASSERT_NEAR(er, image2[0], max_dist);
    ASSERT_NEAR(eg, image2[1], max_dist);
    ASSERT_NEAR(eb, image2[2], max_dist);
  }

  // Also test the opposite direction

  std::vector<unsigned char> image3(4);
  convertFromSrgb(image3.data(), image2.data(), 1, 1, &state);

  if(max_dist == 0) {
    ASSERT_EQUALS(r, image3[0]);
    ASSERT_EQUALS(g, image3[1]);
    ASSERT_EQUALS(b, image3[2]);
  } else {
    ASSERT_NEAR(r, image3[0], max_dist);
    ASSERT_NEAR(g, image3[1], max_dist);
    ASSERT_NEAR(b, image3[2], max_dist);
  }
}

void testChrmToSrgb() {
  std::cout << "testChrmToSrgb" << std::endl;
  // srgb gamma approximation and chromaticities defined as standard by png (multiplied by 100000)
  unsigned sg = 45455; // srgb gamma approximation
  unsigned swx = 31270;
  unsigned swy = 32900;
  unsigned srx = 64000;
  unsigned sry = 33000;
  unsigned sgx = 30000;
  unsigned sgy = 60000;
  unsigned sbx = 15000;
  unsigned sby = 6000;

  testChrmToSrgb(sg, swx, swy, srx, sry, sgx, sgy, sbx, sby, 0, 0, 0, 0, 0, 0);
  testChrmToSrgb(sg, swx, swy, srx, sry, sgx, sgy, sbx, sby, 255, 255, 255, 255, 255, 255);

  testChrmToSrgb(0, swx, swy, srx, sry, sgx, sgy, sbx, sby, 50, 50, 50, 50, 50, 50);
  testChrmToSrgb(0, swx, swy, srx, sry, sgx, sgy, sbx, sby, 128, 128, 128, 128, 128, 128);
  testChrmToSrgb(0, swx, swy, srx, sry, sgx, sgy, sbx, sby, 200, 200, 200, 200, 200, 200);

  testChrmToSrgb(0, swx, swy, srx, sry, sgx, sgy, sbx, sby, 255, 0, 0, 255, 0, 0);
  testChrmToSrgb(0, swx, swy, srx, sry, sgx, sgy, sbx, sby, 0, 255, 0, 0, 255, 0);
  testChrmToSrgb(0, swx, swy, srx, sry, sgx, sgy, sbx, sby, 0, 0, 255, 0, 0, 255);

  // swap red and green chromaticities
  testChrmToSrgb(0, swx, swy, sgx, sgy, srx, sry, sbx, sby, 255, 0, 0, 0, 255, 0);
  testChrmToSrgb(0, swx, swy, sgx, sgy, srx, sry, sbx, sby, 0, 255, 0, 255, 0, 0);
  testChrmToSrgb(0, swx, swy, sgx, sgy, srx, sry, sbx, sby, 0, 0, 255, 0, 0, 255);

  // swap red/green/blue chromaticities
  testChrmToSrgb(0, swx, swy, sgx, sgy, sbx, sby, srx, sry, 255, 0, 0, 0, 255, 0);
  testChrmToSrgb(0, swx, swy, sgx, sgy, sbx, sby, srx, sry, 0, 255, 0, 0, 0, 255);
  testChrmToSrgb(0, swx, swy, sgx, sgy, sbx, sby, srx, sry, 0, 0, 255, 255, 0, 0);

  // different whitepoint does not affect white or gray, due to the relative rendering intent (adaptation)
  testChrmToSrgb(0, 35000, 25000, srx, sry, sgx, sgy, sbx, sby, 0, 0, 0, 0, 0, 0);
  testChrmToSrgb(0, 35000, 25000, srx, sry, sgx, sgy, sbx, sby, 50, 50, 50, 50, 50, 50);
  testChrmToSrgb(0, 35000, 25000, srx, sry, sgx, sgy, sbx, sby, 128, 128, 128, 128, 128, 128);
  testChrmToSrgb(0, 35000, 25000, srx, sry, sgx, sgy, sbx, sby, 200, 200, 200, 200, 200, 200);
  testChrmToSrgb(0, 35000, 25000, srx, sry, sgx, sgy, sbx, sby, 255, 255, 255, 255, 255, 255);
}



void testXYZ() {
  std::cout << "testXYZ" << std::endl;
  unsigned w = 512, h = 512;
  std::vector<unsigned char> v(w * h * 4 * 2);
  for(size_t i = 0; i < v.size(); i++) {
    v[i] = getRandom() & 255;
  }

  // Test sRGB -> XYZ -> sRGB roundtrip

  unsigned rendering_intent = 3; // test with absolute for now

  // 8-bit
  {
    // Default state, the conversions use 8-bit sRGB
    lodepng::State state;
    std::vector<float> f(w * h * 4);
    float whitepoint[3];
    assertNoError(lodepng::convertToXYZ(f.data(), whitepoint, v.data(), w, h, &state));

    std::vector<unsigned char> v2(w * h * 4);
    assertNoError(lodepng::convertFromXYZ(v2.data(), f.data(), w, h, &state, whitepoint, rendering_intent));

    for(size_t i = 0; i < v2.size(); i++) {
      ASSERT_EQUALS(v[i], v2[i]);
    }
  }

  // 16-bit
  {
    // Default state but with 16-bit, the conversions use 16-bit sRGB
    lodepng::State state;
    state.info_raw.bitdepth = 16;
    std::vector<float> f(w * h * 4);
    float whitepoint[3];
    assertNoError(lodepng::convertToXYZ(f.data(), whitepoint, v.data(), w, h, &state));

    std::vector<unsigned char> v2(w * h * 8);
    assertNoError(lodepng::convertFromXYZ(v2.data(), f.data(), w, h, &state, whitepoint, rendering_intent));

    for(size_t i = 0; i < v2.size(); i++) {
      ASSERT_EQUALS(v[i], v2[i]);
    }
  }

  // Test custom RGB+gamma -> XYZ -> custom RGB+gamma roundtrip

  LodePNGInfo info_custom;
  lodepng_info_init(&info_custom);
  info_custom.gama_defined = 1;
  info_custom.gama_gamma =   30000; // default 45455
  info_custom.chrm_defined = 1;
  info_custom.chrm_white_x = 10000; // default 31270
  info_custom.chrm_white_y = 20000; // default 32900
  info_custom.chrm_red_x =   30000; // default 64000
  info_custom.chrm_red_y =   50000; // default 33000
  info_custom.chrm_green_x = 70000; // default 30000
  info_custom.chrm_green_y = 11000; // default 60000
  info_custom.chrm_blue_x =  13000; // default 15000
  info_custom.chrm_blue_y =  17000; // default 6000

  // 8-bit
  {
    lodepng::State state;
    lodepng_info_copy(&state.info_png, &info_custom);
    std::vector<float> f(w * h * 4);
    float whitepoint[3];
    assertNoError(lodepng::convertToXYZ(f.data(), whitepoint, v.data(), w, h, &state));

    std::vector<unsigned char> v2(w * h * 4);
    assertNoError(lodepng::convertFromXYZ(v2.data(), f.data(), w, h, &state, whitepoint, rendering_intent));

    for(size_t i = 0; i < v2.size(); i++) {
      // Allow near instead of exact due to numerical issues with low values,
      // see description at the 16-bit test below.
      unsigned maxdist = 0;
      if(v[i] <= 2) maxdist = 3;
      else if(v[i] <= 4) maxdist = 2;
      else maxdist = 0;
      ASSERT_NEAR(v[i], v2[i], maxdist);
    }
  }

  // 16-bit
  {
    lodepng::State state;
    lodepng_info_copy(&state.info_png, &info_custom);
    state.info_raw.bitdepth = 16;
    std::vector<float> f(w * h * 4);
    float whitepoint[3];
    assertNoError(lodepng::convertToXYZ(f.data(), whitepoint, v.data(), w, h, &state));

    std::vector<unsigned char> v2(w * h * 8);
    assertNoError(lodepng::convertFromXYZ(v2.data(), f.data(), w, h, &state, whitepoint, rendering_intent));

    for(size_t i = 0; i < v2.size(); i += 2) {
      unsigned a = v[i + 0] * 256u + v[i + 1];
      unsigned a2 = v2[i + 0] * 256u + v2[i + 1];
      // There are numerical issues with low values due to the precision of float,
      // so allow some distance for low values (low compared to 65535).
      // The issue seems to be: the combination of how the gamma correction affects
      // low values and the color conversion matrix operating on single precision
      // floating point. With the sRGB's gamma the problem seems not to happen, maybe
      // because that linear part near 0 behaves better than power.
      // TODO: check if it can be fixed without using double for the image and without slow double precision pow.
      unsigned maxdist = 0;
      if(a < 2048) maxdist = 768;
      else if(a < 4096) maxdist = 24;
      else if(a < 16384) maxdist = 4;
      else maxdist = 2;
      ASSERT_NEAR(a, a2, maxdist);
    }
  }

  lodepng_info_cleanup(&info_custom);
}


void testICC() {
  std::cout << "testICC" << std::endl;
  // approximate srgb (gamma function not exact)
  std::string icc_near_srgb_base64 =
      "AAABwHRlc3QCQAAAbW50clJHQiBYWVogB+MAAQABAAAAAAAAYWNzcFNHSSAAAAABAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAEAAPbWAAEAAAAA0y10ZXN0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAJY3BydAAAAPAAAAANZGVzYwAAAQAAAABfd3RwdAAAAWAAAAAUclhZ"
      "WgAAAXQAAAAUZ1hZWgAAAYgAAAAUYlhZWgAAAZwAAAAUclRSQwAAAbAAAAAOZ1RSQwAAAbAAAAAO"
      "YlRSQwAAAbAAAAAOdGV4dAAAAABDQzAgAAAAAGRlc2MAAAAAAAAABXRlc3QAZW5VUwAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAFhZWiAAAAAAAADzUQABAAAAARbMWFlaIAAAAAAAAG+gAAA49AAAA5BYWVogAAAA"
      "AAAAYpYAALeHAAAY2VhZWiAAAAAAAAAkngAAD4QAALbCY3VydgAAAAAAAAABAjMAAA==";
  std::vector<unsigned char> icc_near_srgb;
  fromBase64(icc_near_srgb, icc_near_srgb_base64);
  lodepng::State state_near_srgb;
  lodepng_set_icc(&state_near_srgb.info_png, "near_srgb", icc_near_srgb.data(), icc_near_srgb.size());

  // a made up RGB model.
  // it causes (when converting from this to srgb) green to become softer green, blue to become softer blue, red to become orange.
  // this model intersects sRGB, but some parts are outside of sRGB, some parts of sRGB are outside of this one.
  // so when converting between this and sRGB and clipping the values to 8-bit, and then converting back, the values will not be the same due to this clipping
  std::string icc_orange_base64 =
      "AAABwHRlc3QCQAAAbW50clJHQiBYWVogB+MAAQABAAAAAAAAYWNzcFNHSSAAAAABAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAMAAPbWAAEAAAAA0y10ZXN0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAJY3BydAAAAPAAAAANZGVzYwAAAQAAAABfd3RwdAAAAWAAAAAUclhZ"
      "WgAAAXQAAAAUZ1hZWgAAAYgAAAAUYlhZWgAAAZwAAAAUclRSQwAAAbAAAAAOZ1RSQwAAAbAAAAAO"
      "YlRSQwAAAbAAAAAOdGV4dAAAAABDQzAgAAAAAGRlc2MAAAAAAAAABXRlc3QAZW5VUwAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAFhZWiAAAAAAAAE7uwABAAAAARmZWFlaIAAAAAAAANAHAACTTAAACrRYWVogAAAA"
      "AAAABOMAAFd4AAAFzVhZWiAAAAAAAAAh6gAAFTsAAMKqY3VydgAAAAAAAAABAoAAAA==";
  std::vector<unsigned char> icc_orange;
  fromBase64(icc_orange, icc_orange_base64);
  lodepng::State state_orange;
  lodepng_set_icc(&state_orange.info_png, "orange", icc_orange.data(), icc_orange.size());

  // A made up RGB model which is a superset of sRGB, and has R/G/B shifted around (so it greatly alters colors)
  // Since this is a superset of sRGB, converting from sRGB to this model, and then back, should be lossless, but the opposite not necessarily.
  std::string icc_super_base64 =
      "AAABwHRlc3QCQAAAbW50clJHQiBYWVogB+MAAQABAAAAAAAAYWNzcFNHSSAAAAABAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAEAAPbWAAEAAAAA0y10ZXN0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAJY3BydAAAAPAAAAANZGVzYwAAAQAAAABfd3RwdAAAAWAAAAAUclhZ"
      "WgAAAXQAAAAUZ1hZWgAAAYgAAAAUYlhZWgAAAZwAAAAUclRSQwAAAbAAAAAOZ1RSQwAAAbAAAAAO"
      "YlRSQwAAAbAAAAAOdGV4dAAAAABDQzAgAAAAAGRlc2MAAAAAAAAABXRlc3QAZW5VUwAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAFhZWiAAAAAAAADzUQABAAAAARbMWFlaIAAAAAAAAFW+AADL3f//70ZYWVogAAAA"
      "AAAAJqD////UAADsUFhZWiAAAAAAAAB6dgAANE////eXY3VydgAAAAAAAAABAjMAAA==";
  std::vector<unsigned char> icc_super;
  fromBase64(icc_super, icc_super_base64);
  lodepng::State state_super;
  lodepng_set_icc(&state_super.info_png, "super", icc_super.data(), icc_super.size());

  // A made up RGB model which is a subset of sRGB, and has R/G/B shifted around (so it greatly alters colors)
  // Since this is a subset of sRGB, converting to sRGB from this model, and then back, should be lossless, but the opposite not necessarily.
  std::string icc_sub_base64 =
      "AAABwHRlc3QCQAAAbW50clJHQiBYWVogB+MAAQABAAAAAAAAYWNzcFNHSSAAAAABAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAEAAPbWAAEAAAAA0y10ZXN0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAJY3BydAAAAPAAAAANZGVzYwAAAQAAAABfd3RwdAAAAWAAAAAUclhZ"
      "WgAAAXQAAAAUZ1hZWgAAAYgAAAAUYlhZWgAAAZwAAAAUclRSQwAAAbAAAAAOZ1RSQwAAAbAAAAAO"
      "YlRSQwAAAbAAAAAOdGV4dAAAAABDQzAgAAAAAGRlc2MAAAAAAAAABXRlc3QAZW5VUwAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAFhZWiAAAAAAAADzUQABAAAAARbMWFlaIAAAAAAAAHEEAABy1AAAr8ZYWVogAAAA"
      "AAAAV5kAAEPkAAAMs1hZWiAAAAAAAAAuNwAASUcAABazY3VydgAAAAAAAAABAjMAAA==";

  std::vector<unsigned char> icc_sub;
  fromBase64(icc_sub, icc_sub_base64);
  lodepng::State state_sub;
  lodepng_set_icc(&state_sub.info_png, "sub", icc_sub.data(), icc_sub.size());

  // make 8-pixel image with following colors: white, gray, red, darkred, green, darkgreen, blue, darkblue
  unsigned w = 4, h = 2;
  std::vector<unsigned char> im(w * h * 4, 255);
  im[0 * 4 + 0] = 255; im[0 * 4 + 1] = 255; im[0 * 4 + 2] = 255;
  im[1 * 4 + 0] = 128; im[1 * 4 + 1] = 128; im[1 * 4 + 2] = 128;
  im[2 * 4 + 0] = 255; im[2 * 4 + 1] =   0; im[2 * 4 + 2] =   0;
  im[3 * 4 + 0] = 128; im[3 * 4 + 1] =   0; im[3 * 4 + 2] =   0;
  im[4 * 4 + 0] =   0; im[4 * 4 + 1] = 255; im[4 * 4 + 2] =   0;
  im[5 * 4 + 0] =   0; im[5 * 4 + 1] = 128; im[5 * 4 + 2] =   0;
  im[6 * 4 + 0] =   0; im[6 * 4 + 1] =   0; im[6 * 4 + 2] = 255;
  im[7 * 4 + 0] =   0; im[7 * 4 + 1] =   0; im[7 * 4 + 2] = 128;


  {
    std::vector<unsigned char> im2(w * h * 4, 255);
    assertNoError(convertToSrgb(im2.data(), im.data(), w, h, &state_orange));

    ASSERT_NEAR(255, im2[0 * 4 + 0], 1); ASSERT_NEAR(255, im2[0 * 4 + 1], 1); ASSERT_NEAR(255, im2[0 * 4 + 2], 1);
    ASSERT_NEAR(117, im2[1 * 4 + 0], 1); ASSERT_NEAR(117, im2[1 * 4 + 1], 1); ASSERT_NEAR(117, im2[1 * 4 + 2], 1);
    ASSERT_NEAR(255, im2[2 * 4 + 0], 1); ASSERT_NEAR(151, im2[2 * 4 + 1], 1); ASSERT_NEAR(  0, im2[2 * 4 + 2], 1);
    ASSERT_NEAR(145, im2[3 * 4 + 0], 1); ASSERT_NEAR( 66, im2[3 * 4 + 1], 1); ASSERT_NEAR(  0, im2[3 * 4 + 2], 1);
    ASSERT_NEAR(  0, im2[4 * 4 + 0], 1); ASSERT_NEAR(209, im2[4 * 4 + 1], 1); ASSERT_NEAR(  0, im2[4 * 4 + 2], 1);
    ASSERT_NEAR(  0, im2[5 * 4 + 0], 1); ASSERT_NEAR( 95, im2[5 * 4 + 1], 1); ASSERT_NEAR(  0, im2[5 * 4 + 2], 1);
    ASSERT_NEAR(  0, im2[6 * 4 + 0], 1); ASSERT_NEAR( 66, im2[6 * 4 + 1], 1); ASSERT_NEAR(255, im2[6 * 4 + 2], 1);
    ASSERT_NEAR(  0, im2[7 * 4 + 0], 1); ASSERT_NEAR( 25, im2[7 * 4 + 1], 1); ASSERT_NEAR(120, im2[7 * 4 + 2], 1);

    // Cannot test the inverse direction to see if same as original, because the color model here has values
    // outside of sRGB so several values were clipped.
  }

  {
    std::vector<unsigned char> im2(w * h * 4, 255);
    // convert between the two in one and then the other direction
    assertNoError(convertRGBModel(im2.data(), im.data(), w, h, &state_near_srgb, &state_sub, 3));
    std::vector<unsigned char> im3(w * h * 4, 255);
    assertNoError(convertRGBModel(im3.data(), im2.data(), w, h, &state_sub, &state_near_srgb, 3));
    // im3 should be same as im (allow some numerical errors), because we converted from a subset of sRGB to sRGB
    // and then back.
    // If state_super was used here instead (with a superset RGB color model), the test below would faill due to
    // the clipping of the values in the 8-bit chars (due to the superset being out of range for sRGB)
    for(size_t i = 0; i < im.size(); i++) {
      // due to the gamma (trc), small values are very imprecise (due to the 8-bit char step in between), so allow more distance there
      int tolerance = im[i] < 32 ? 16 : 1;
      ASSERT_NEAR(im[i], im3[i], tolerance);
    }
  }

  {
    std::vector<unsigned char> im2(w * h * 4, 255);
    assertNoError(convertFromSrgb(im2.data(), im.data(), w, h, &state_super));
    std::vector<unsigned char> im3(w * h * 4, 255);
    assertNoError(convertToSrgb(im3.data(), im2.data(), w, h, &state_super));
    for(size_t i = 0; i < im.size(); i++) {
      int tolerance = im[i] < 32 ? 16 : 1;
      ASSERT_NEAR(im[i], im3[i], tolerance);
    }
  }
}


void testICCGray() {
  std::cout << "testICCGray" << std::endl;
  // Grayscale, Gamma 2.2, sRGB whitepoint
  std::string icc22_base64 =
      "AAABSHRlc3QCQAAAbW50ckdSQVlYWVogB+MAAQABAAAAAAAAYWNzcFNHSSAAAAABAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAMAAPbWAAEAAAAA0y10ZXN0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAEY3BydAAAALQAAAANZGVzYwAAAMQAAABfd3RwdAAAASQAAAAUa1RS"
      "QwAAATgAAAAOdGV4dAAAAABDQzAgAAAAAGRlc2MAAAAAAAAABXRlc3QAZW5VUwAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAFhZWiAAAAAAAADzUQABAAAAARbMY3VydgAAAAAAAAABAjMAAA==";
  std::vector<unsigned char> icc22;
  fromBase64(icc22, icc22_base64);
  lodepng::State state22;
  state22.info_raw.colortype = LCT_GREY;
  lodepng_set_icc(&state22.info_png, "gray22", icc22.data(), icc22.size());

  // Grayscale, Gamma 2.9, custom whitepoint
  std::string icc29_base64 =
      "AAABSHRlc3QCQAAAbW50ckdSQVlYWVogB+MAAQABAAAAAAAAYWNzcFNHSSAAAAABAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAMAAPbWAAEAAAAA0y10ZXN0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAEY3BydAAAALQAAAANZGVzYwAAAMQAAABfd3RwdAAAASQAAAAUa1RS"
      "QwAAATgAAAAOdGV4dAAAAABDQzAgAAAAAGRlc2MAAAAAAAAABXRlc3QAZW5VUwAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAFhZWiAAAAAAAAE7uwABAAAAARmZY3VydgAAAAAAAAABAuYAAA==";
  std::vector<unsigned char> icc29;
  fromBase64(icc29, icc29_base64);
  lodepng::State state29;
  state29.info_raw.colortype = LCT_GREY;
  lodepng_set_icc(&state29.info_png, "gray29", icc29.data(), icc29.size());

  // Grayscale, Gamma 1.5, custom whitepoint
  std::string icc15_base64 =
      "AAABSHRlc3QCQAAAbW50ckdSQVlYWVogB+MAAQABAAAAAAAAYWNzcFNHSSAAAAABAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAMAAPbWAAEAAAAA0y10ZXN0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAEY3BydAAAALQAAAANZGVzYwAAAMQAAABfd3RwdAAAASQAAAAUa1RS"
      "QwAAATgAAAAOdGV4dAAAAABDQzAgAAAAAGRlc2MAAAAAAAAABXRlc3QAZW5VUwAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAFhZWiAAAAAAAAE7uwABAAAAARmZY3VydgAAAAAAAAABAYAAAA==";
  std::vector<unsigned char> icc15;
  fromBase64(icc15, icc15_base64);
  lodepng::State state15;
  state15.info_raw.colortype = LCT_GREY;
  lodepng_set_icc(&state15.info_png, "gray15", icc15.data(), icc15.size());


  // make 8-pixel grayscale image with different shades of gray
  unsigned w = 4, h = 2;
  std::vector<unsigned char> im(w * h, 255);
  im[0] = 0;
  im[1] = 40;
  im[2] = 80;
  im[3] = 120;
  im[4] = 160;
  im[5] = 200;
  im[6] = 240;
  im[7] = 255;

  {
    std::vector<unsigned char> im2(w * h, 255);
    assertNoError(convertToSrgb(im2.data(), im.data(), w, h, &state29));

    ASSERT_NEAR(0, im2[0], 1);
    ASSERT_NEAR(15, im2[1], 1);
    ASSERT_NEAR(52, im2[2], 1);
    ASSERT_NEAR(94, im2[3], 1);
    ASSERT_NEAR(139, im2[4], 1);
    ASSERT_NEAR(187, im2[5], 1);
    ASSERT_NEAR(236, im2[6], 1);
    ASSERT_NEAR(255, im2[7], 1);

    std::vector<unsigned char> im3(w * h, 255);
    assertNoError(convertFromSrgb(im3.data(), im2.data(), w, h, &state29));

    for(size_t i = 0; i < 8; i++) {
      ASSERT_NEAR(im[i], im3[i], 1);
    }
  }

  {
    std::vector<unsigned char> im2(w * h , 255);
    assertNoError(convertRGBModel(im2.data(), im.data(), w, h, &state22, &state15, 3));
    std::vector<unsigned char> im3(w * h, 255);
    assertNoError(convertRGBModel(im3.data(), im2.data(), w, h, &state15, &state22, 3));
    for(size_t i = 0; i < im.size(); i++) {
      int tolerance = im[i] < 16 ? 8 : 1;
      ASSERT_NEAR(im[i], im3[i], tolerance);
    }
  }
}

// input is base64-encoded png image and base64-encoded RGBA pixels (8 bit per channel)
void testBase64Image(const std::string& png64, bool expect_error, unsigned expect_w, unsigned expect_h, const std::string& expect_md5) {
  std::vector<unsigned char> png, pixels;
  fromBase64(png, png64);

  std::vector<unsigned char> decoded;
  unsigned w, h;
  unsigned error = lodepng::decode(decoded, w, h, png);
  if(expect_error) {
    ASSERT_EQUALS(true, error != 0);
    return;
  }
  assertNoError(error);
  ASSERT_EQUALS(expect_w, w);
  ASSERT_EQUALS(expect_h, h);
  ASSERT_EQUALS(expect_md5, md5sum(decoded));

  // test decoding without alpha channel
  {
    size_t numpixels = w * h;
    std::vector<unsigned char> expected_rgb(numpixels * 3);
    for(size_t i = 0; i < numpixels; i++) {
      expected_rgb[i * 3 + 0] = decoded[i * 4 + 0];
      expected_rgb[i * 3 + 1] = decoded[i * 4 + 1];
      expected_rgb[i * 3 + 2] = decoded[i * 4 + 2];
    }
    std::vector<unsigned char> rgb;
    ASSERT_NO_PNG_ERROR(lodepng::decode(rgb, w, h, png, LCT_RGB));
    ASSERT_EQUALS(expect_w, w);
    ASSERT_EQUALS(expect_h, h);
    ASSERT_EQUALS(expected_rgb, rgb);
  }

  // test decoding 16-bit RGBA
  // TODO: get an additional md5sum for 16-bit pixels instead to compare with
  {
    size_t numpixels = w * h;
    std::vector<unsigned char> rgba16;
    ASSERT_NO_PNG_ERROR(lodepng::decode(rgba16, w, h, png, LCT_RGBA, 16));
    ASSERT_EQUALS(expect_w, w);
    ASSERT_EQUALS(expect_h, h);
    std::vector<unsigned char> rgba8(numpixels * 4);
    for(size_t i = 0; i < numpixels; i++) {
      rgba8[i * 4 + 0] = rgba16[i * 8 + 0];
      rgba8[i * 4 + 1] = rgba16[i * 8 + 2];
      rgba8[i * 4 + 2] = rgba16[i * 8 + 4];
      rgba8[i * 4 + 3] = rgba16[i * 8 + 6];
    }
    ASSERT_EQUALS(decoded, rgba8);
  }

  // test decoding 16-bit RGB
  {
    size_t numpixels = w * h;
    std::vector<unsigned char> expected_rgb(numpixels * 3);
    for(size_t i = 0; i < numpixels; i++) {
      expected_rgb[i * 3 + 0] = decoded[i * 4 + 0];
      expected_rgb[i * 3 + 1] = decoded[i * 4 + 1];
      expected_rgb[i * 3 + 2] = decoded[i * 4 + 2];
    }
    std::vector<unsigned char> rgb16;
    ASSERT_NO_PNG_ERROR(lodepng::decode(rgb16, w, h, png, LCT_RGB, 16));
    ASSERT_EQUALS(expect_w, w);
    ASSERT_EQUALS(expect_h, h);
    std::vector<unsigned char> rgb8(numpixels * 3);
    for(size_t i = 0; i < numpixels; i++) {
      rgb8[i * 3 + 0] = rgb16[i * 6 + 0];
      rgb8[i * 3 + 1] = rgb16[i * 6 + 2];
      rgb8[i * 3 + 2] = rgb16[i * 6 + 4];
    }
    ASSERT_EQUALS(expected_rgb, rgb8);
  }

  // test encode/decode
  // TODO: also test state, for text chunks, ...
  {
    std::vector<unsigned char> rgba16;
    ASSERT_NO_PNG_ERROR(lodepng::decode(rgba16, w, h, png, LCT_RGBA, 16));

    std::vector<unsigned char> png_b;
    ASSERT_NO_PNG_ERROR(lodepng::encode(png_b, rgba16, w, h, LCT_RGBA, 16));

    std::vector<unsigned char> rgba16_b;
    ASSERT_NO_PNG_ERROR(lodepng::decode(rgba16_b, w, h, png_b, LCT_RGBA, 16));
    ASSERT_EQUALS(rgba16, rgba16_b);
  }
}
// input is base64-encoded png image and base64-encoded RGBA pixels (8 bit per channel)
void testPngSuiteImage(const std::string& png64, const std::string& name, bool expect_error, unsigned expect_w, unsigned expect_h, const std::string& expect_md5) {
  std::cout << "testPngSuiteImage: " << name << std::endl;

  testBase64Image(png64, expect_error, expect_w, expect_h, expect_md5);
}


// Tests base64-encoded PngSuite images' pixels against expected md5 sum of their pixels
void testPngSuite() {
  std::cout << "testPngSuite" << std::endl;
  /*
  LICENSE of the PngSuite images:

  PngSuite
  --------

  Permission to use, copy, modify and distribute these images for any
  purpose and without fee is hereby granted.


  (c) Willem van Schaik, 1996, 2011
  */

  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAQAAAAEsBnfPAAAABGdBTUEAAYagMeiWXwAAAJBJREFUeJwtjTEOwjAMRd/GgsQVGHoApC4Zergeg7En4AxWOQATY6WA2FgsZckQNXxLeLC/v99PcBaMGeesuXCj8tHe2Wlc5b9ZY9/ZKq9Mn9kn6kSeZIffW5w255m5G98IK01L1AFP5AFLAat6F67mlNKNMootY4N6cEUeFkhwLZqf9KEdL3pRqiHloYx//QCU41EdZhgi8gAAAABJRU5ErkJggg==",
      "basi0g01.png", false, 32, 32, "4336909be7bff35103266c9b215ab516");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAgAAAAFrpg0fAAAABGdBTUEAAYagMeiWXwAAAFFJREFUeJxjUGLoYADhcoa7YJyTw3DsGJSUlgYxNm5EZ7OuZ13PEPUh6gMDkMHKAGRE4RZDSCBkEUpIUscQuuo/GMMZGAIMMEEEA6YKwaCSOQCcUoBNhbbZfQAAAABJRU5ErkJggg==",
      "basi0g02.png", false, 32, 32, "b16bee35e71dce6c08c2447a62ccedea");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAAHk5vi/AAAABGdBTUEAAYagMeiWXwAAAK5JREFUeJxljlERwjAQRBccFBwUHAQchDoodRDqINRBwEHBQcFBwEGRECRUA5lJmM7Nftzs7bub28OywrZFdUX7xLrBvkNzR/fGanc8I9YNsV6I9cViczilQWwuaRqbR1qJzSftoSiVro39q0PWHlkHZPXIOiJrQNZpvsMH+TJHcBaHcjq/Mf+DoihLpbSua2OsZSCtcwyk7XsG0g4DA2m9ZyDtODKQNgQG0k4TgR8ngeup000HFgAAAABJRU5ErkJggg==",
      "basi0g04.png", false, 32, 32, "0b40ec7e4231183b51e1c23f818a955f");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAAEhFhW+AAAABGdBTUEAAYagMeiWXwAAALVJREFUeJy1kF0KwjAQhJ26yBxCxHv4Q88lPoh4sKoXEQ8hS9ymviQPXSGllM7T5JvNMiwWJBFVFRVJmKpCSCKoKlYkoaqKiyTFj5mZmQgTCYmgSgDXbCwJ52zyGtyyCTk6ZVNXfaFxQKLFnnDsv6OI3/HwO4L7gr0H8F98sT+AuwetL9YMARw8WI7v8fTgO77HzoMtypJ66gBeQxtiV5Y0UwewGchF5r/Du5h2nYT577AupsAPm7n/RegfnygAAAAASUVORK5CYII=",
      "basi0g08.png", false, 32, 32, "f6470f9f6296c5109e2bd730fe203773");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAFxhsn9AAAABGdBTUEAAYagMeiWXwAAAOJJREFUeJy1kTsOwjAQRMdJCqj4XYHD5DAcj1Okyg2okCyBRLOSC0BDERKCI7xJVmgaa/X8PFo7oESJEtkaTeLDjdjjgCMe7eTE96FGd3AL7HvZsdNEaJMVo0GNGm775bgwW6Afj/SAjAY+JsYNXIHtz2xYxTXiUoOek4AbFcCnDYEK4NMGsgXcMrGHJytkBX5HIP8FAhVANIMVIBVANMPfgUAFEM3wAVyG5cxcecY5/dup3LVFa1HXmA61LY59f6Ygp1Eg1gZGQaBRILYGdxoFYmtAGgXx9YmCfPD+RMHwuuAFVpjuiRT///4AAAAASUVORK5CYII=",
      "basi0g16.png", false, 32, 32, "a14e204bbf905586d3763f3cc5dcb2f3");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAGLH901AAAABGdBTUEAAYagMeiWXwAAAPJJREFUeJzVk0GqBCEMRKvAe3gTPVnTczO9iddoaLVm0Qz0Z1r4WWQxoRZifFaIkZKA4xIlfdagpM8aAQCO4xKl88acN+b8w/R+Z3agf4va9bQP7tLTPgJeL/T+LUpj4aFtkRgLc22LxFhUxW2VGGP0p+C2bc8JqQDz/6KUjUCR5TyobASKZDkPZitQSpmWYM7ZBhgrmgGovgClZASm7eGCsSI7QCXjLE3jQwRjRXaAyTqtpsmbc4Zaqy/AlJINkBogP13f4ZcNKEVngybP+6/v/NMGVPRtEZvkeT+Cc4f8DRidW8TWmjwj1Fp/24AxRleDN99NCjEh/D0zAAAAAElFTkSuQmCC",
      "basi2c08.png", false, 32, 32, "512c3874e30061e623739e2f9adc4eba");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAAHbjwF2AAAABGdBTUEAAYagMeiWXwAAAgpJREFUeJzVliFz4zAQhT/PGDjMYaGBgYWlPlZ4f6E/oTCFLjtYeLSwsDCBuX9wMAcLU+awV6CokSu7kWO58Qns7LxZvX3PK0VJJAnWbwDrHdg8tUl9ZUVq644QQJE7OywEUEyTbdWpx+LG67G48XpY6NBD2lbwbw+fYuUhe2ujobdvbJ6Z6GlqKnLixPseTUUukuyWUrgHSG0Stmab4A2zjZEUsMGWHjdUYaVfdmgqhcNnrW9oL/U6nCo1MZF2S3i7B9jdw8l8GVDzkWdFx7mFLHPC8hJgWkZqUCcFyDOAaci56E76uUHrOTqX1Mn9YxSDFCCfHB00NOhH6tY4DeKR1vJqJcHrtQR/XyTYXEnw8izB00KCxycJyrkEd78luJ1J8PNRgiKX4OqXBPNMgryUACRI7eUYZuVlau9dfGo4XLTYDmpTjOugTm3ySA6aqE3e20E7tcl7ODhFbfLU/sLHpwaYPnR00IX66CCoQXdqgwc4OJc6wEE/6i8dxKBucRCP2nMQm9rkiVStYL+Geqw85Ex8FYmnEc+Kgd+bIZZ52W0c7D2Lu+qiASYm34x4Au2iAbLS4CObQJhoFx/BBLqLdvELTaCfaBf/xgnEE+3iZ/0furRogMkPgOzPABMYXrSLR7oD3yvare8xgcuJdvGOExiHaBcPmMD4RLt4ywTGLdrFnQn8P6Jd/B2kFN6z3xNE9wAAAABJRU5ErkJggg==",
      "basi2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAQMAAAE+s9ghAAAABGdBTUEAAYagMeiWXwAAAAZQTFRF7v8iImb/bBrSJgAAAClJREFUeJxjYICCD1C4CgpD0bCxMcOZM9hJCININj8QQIgPQAAhKBADAAm6Qi12qcOeAAAAAElFTkSuQmCC",
      "basi3p01.png", false, 32, 32, "1ba59f527ff2cfdc68bb0c3487862e91");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAgMAAAF5E6LxAAAABGdBTUEAAYagMeiWXwAAAANzQklUAQEBfC53ggAAAAxQTFRFAP8A/wAA//8AAAD/ZT8rugAAAFFJREFUeJxjeMewmwGEXRgEwdjMjCE5GUreuAFi9Pais78u+LqAgT+KP4oByPjKAGTw4xZDSCBkEUpIUvc/dBUYIxiYQqugLAQDKvEfwaCSOQC0Wn3pH3XhAwAAAABJRU5ErkJggg==",
      "basi3p02.png", false, 32, 32, "0528e9ac365252a8c0e2d9ced8a2cc6b");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAAH2U1dRAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAC1QTFRFIgD/AP//iAD/Iv8AAJn//2YA3QD/d/8A/wAAAP+Z3f8A/wC7/7sAAET/AP9E0rBJvQAAALZJREFUeJxj6KljOP6QoU6W4eElhihLhsVTGCwdGKawMcQst5vIAMS+DEDMxADE2Qytp4pfQiSADBGILJBxAaIEyFCDqOsIPbOq3PjdTAYoLcgApV0YoPRdBhjNAKWVGKB0GgOU3o0wB9NATJMxrcC0C9NSTNsxnYFwT0do6Jkzq1aVlxsbv3s3cyamACpXUBBTAJXr4oIpgMq9exdTAI3LgCmAylVSwhRA5aalYQqgcnfvxhAAALN26mgMdNBfAAAAAElFTkSuQmCC",
      "basi3p04.png", false, 32, 32, "a339593b0d82103e30ed7b00afd68816");
  testPngSuiteImage("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",
      "basi3p08.png", false, 32, 32, "d36bdbefc126ef50bd57d51eb38f2ac4");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAQAAAGudILpAAAABGdBTUEAAYagMeiWXwAAAI1JREFUeJztj80KgzAMx3+BHvTWvUH7KPbB9yhzT7Dt5LUeHBWiEkFWhpgQGtL/RyIZOhLJ3Zli2UgOJAvzgECcs/ygoZsDyb7wA5Hoek2pMpAXeDw3VaVbMHTUADx/biG5Wbt+Lve2LD4W4FKoZnFYQQZovtmqd8+kNR2sMG8wBU6wwQlOuDb4hw2OCozsTz0JHVlVXQAAAABJRU5ErkJggg==",
      "basi4a08.png", false, 32, 32, "e2212ec5fa026a41826136e983bf92b2");
  testPngSuiteImage("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",
      "basi4a16.png", false, 32, 32, "f1423ebc08979252299ca238656ab0ba");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAAEEfUpiAAAABGdBTUEAAYagMeiWXwAAASBJREFUeJzFlUFOwzAQRZ+lQbi7smZBuAabhh6LTRLBwRLBRSpxCipl2EDVlJBB/EgeyYrifH8/jSfj5GSAR2AP7A0fOQ+74mM6MeKTieTk6nv9vz2aa4AKuJ8b1rVTz8uwZ56WBWPXLgqSk7cze5+YjMZ/Xw4YbSDoCAQvHJcFThMJ2kDQLX4n+S4DbL/GTfD8MRemIQobatGgDfIcGrzyoBlExxAbDLVooAGQnJz545nPPY2dRmCodUBdmmDQALBeLeVeJXgLelJo4GIhGOI5mqsGOoFYCEYvGrhokPwuA+SLsQne19Js5L9ZDbkbrABQdH/sUBXOgNoOVwAoG+Uz8M5tWQC1m8sA6m0gAxTPgB+qsgDqdSgDqNepDFA6A5+CSlP0aU5zQgAAAABJRU5ErkJggg==",
      "basi6a08.png", false, 32, 32, "e80a60aecf13ebd863b61167ba95960b");
  testPngSuiteImage("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",
      "basi6a16.png", false, 32, 32, "4d9d6473bb7403d7f85e3e7537c34e9d");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAQAAAABbAUdZAAAABGdBTUEAAYagMeiWXwAAAFtJREFUeJwtzLEJAzAMBdHr0gSySiALejRvkBU8gsGNCmFFB1Hx4IovqurSpIRszqklUwbnUzRXEuIRsiG/SyY9G0JzJSVei9qynm9qyjBpLp0pYW7pbzBl8L8fEIdJL9AvFMkAAAAASUVORK5CYII=",
      "basn0g01.png", false, 32, 32, "4336909be7bff35103266c9b215ab516");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAgAAAAAcoT2JAAAABGdBTUEAAYagMeiWXwAAAB9JREFUeJxjYAhd9R+M8TCIUMIAU4aPATMJH2OQuQcAvUl/gYsJiakAAAAASUVORK5CYII=",
      "basn0g02.png", false, 32, 32, "b16bee35e71dce6c08c2447a62ccedea");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAEhJREFUeJxjYGAQFFRSMjZ2cQkNTUsrL2cgQwCV29FBjgAqd+ZMcgRQuatWkSOAyt29mxwBVO6ZM+QIoHLv3iVHAJX77h0ZAgAfFO4B6v9B+gAAAABJRU5ErkJggg==",
      "basn0g04.png", false, 32, 32, "0b40ec7e4231183b51e1c23f818a955f");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAAABGdBTUEAAYagMeiWXwAAAEFJREFUeJxjZGAkABQIyLMMBQWMDwgp+PcfP2B5MBwUMMoRkGdkonlcDAYFjI/wyv7/z/iH5nExGBQwyuCVZWQEAFDl/nE14thZAAAAAElFTkSuQmCC",
      "basn0g08.png", false, 32, 32, "f6470f9f6296c5109e2bd730fe203773");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAYagMeiWXwAAAF5JREFUeJzV0jEKwDAMQ1E5W+9/xtygk8AoezLVKgSj2Y8/OICnuFcTE2OgOoJgHQiZAN2C9kDKBOgW3AZCJkC3oD2QMgG6BbeBkAnQLWgPpExgP28H7E/0GTjPfwAW2EvYX64rn9cAAAAASUVORK5CYII=",
      "basn0g16.png", false, 32, 32, "a14e204bbf905586d3763f3cc5dcb2f3");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAAYagMeiWXwAAAEhJREFUeJzt1cEJADAMAkCF7JH9t3ITO0Qr9KH4zuErtA0EO4AKFPgcoO3kfUx4QIECD0qHH8KEBxQo8KB0OCOpQIG7cHejwAGCsfleD0DPSwAAAABJRU5ErkJggg==",
      "basn2c08.png", false, 32, 32, "512c3874e30061e623739e2f9adc4eba");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAACsiDHgAAAABGdBTUEAAYagMeiWXwAAAOVJREFUeJzVlsEKgzAQRKfgQX/Lfrf9rfaWHgYDkoYmZpPMehiGReQ91qCPEEIAPi/gmu9kcnN+GD0nM1/O4vNad7cC6850KHCiM5fz7fJwXdEBYPOygV/o7PICeXSmsMA/dKbkGShD51xsAzXo7DIC9ehMAYG76MypZ6ANnfNJG7BAZx8uYIfOHChgjR4F+MfuDx0AtmfnDfREZ+8m0B+9m8Ao9Chg9x0Yi877jTYwA529WWAeerPAbPQoUH8GNNA5r9yAEjp7sYAeerGAKnoUyJ8BbXTOMxvwgM6eCPhBTwS8oTO/5kL+Xge7xOwAAAAASUVORK5CYII=",
      "basn2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAQMAAABJtOi3AAAABGdBTUEAAYagMeiWXwAAAAZQTFRF7v8iImb/bBrSJgAAABVJREFUeJxj4AcCBjTiAxCgEwOkDgC7Hz/Bk4JmWQAAAABJRU5ErkJggg==",
      "basn3p01.png", false, 32, 32, "1ba59f527ff2cfdc68bb0c3487862e91");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAgMAAAAOFJJnAAAABGdBTUEAAYagMeiWXwAAAANzQklUAQEBfC53ggAAAAxQTFRFAP8A/wAA//8AAAD/ZT8rugAAACJJREFUeJxj+B+6igGEGfAw8MnBGKugLHwMqNL/+BiDzD0AvUl/geqJjhsAAAAASUVORK5CYII=",
      "basn3p02.png", false, 32, 32, "0528e9ac365252a8c0e2d9ced8a2cc6b");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAC1QTFRFIgD/AP//iAD/Iv8AAJn//2YA3QD/d/8A/wAAAP+Z3f8A/wC7/7sAAET/AP9E0rBJvQAAAEdJREFUeJxj6OgIDT1zZtWq8nJj43fvZs5kIEMAlSsoSI4AKtfFhRwBVO7du+QIoHEZyBFA5SopkSOAyk1LI0cAlbt7NxkCAODE6tEPggV9AAAAAElFTkSuQmCC",
      "basn3p04.png", false, 32, 32, "a339593b0d82103e30ed7b00afd68816");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAwBQTFRFIkQA9f/td/93y///EQoAOncAIiL//xH/EQAAIiIA/6xVZv9m/2Zm/wH/IhIA3P//zP+ZRET/AFVVIgAAy8v/REQAVf9Vy8sAMxoA/+zc7f//5P/L/9zcRP9EZmb/MwAARCIA7e3/ZmYA/6RE//+q7e0AAMvL/v///f/+//8BM/8zVSoAAQH/iIj/AKqqAQEARAAAiIgA/+TLulsAIv8iZjIA//+Zqqr/VQAAqqoAy2MAEf8R1P+qdzoA/0RE3GsAZgAAAf8BiEIA7P/ca9wA/9y6ADMzAO0A7XMA//+ImUoAEf//dwAA/4MB/7q6/nsA//7/AMsA/5mZIv//iAAA//93AIiI/9z/GjMAAACqM///AJkAmQAAAAABMmYA/7r/RP///6r/AHcAAP7+qgAASpkA//9m/yIiAACZi/8RVf///wEB/4j/AFUAABER///+//3+pP9EZv///2b/ADMA//9V/3d3AACI/0T/ABEAd///AGZm///tAAEA//XtERH///9E/yL//+3tEREAiP//AAB3k/8iANzcMzP//gD+urr/mf//MzMAY8sAuroArP9V///c//8ze/4A7QDtVVX/qv//3Nz/VVUAAABm3NwA3ADcg/8Bd3f//v7////L/1VVd3cA/v4AywDLAAD+AQIAAQAAEiIA//8iAEREm/8z/9SqAABVmZn/mZkAugC6KlUA/8vLtP9m/5sz//+6qgCqQogAU6oA/6qqAADtALq6//8RAP4AAABEAJmZmQCZ/8yZugAAiACIANwA/5MiAADc/v/+qlMAdwB3AgEAywAAAAAz/+3/ALoA/zMz7f/t/8SIvP93AKoAZgBmACIi3AAA/8v/3P/c/4sRAADLAAEBVQBVAIgAAAAiAf//y//L7QAA/4iIRABEW7oA/7x3/5n/AGYAuv+6AHd3c+0A/gAAMwAzAAC6/3f/AEQAqv+q//7+AAARIgAixP+IAO3tmf+Z/1X/ACIA/7RmEQARChEA/xER3P+6uv//iP+IAQAB/zP/uY7TYgAAAbFJREFUeJwNwQcACAQQAMBHqIxIZCs7Mwlla1hlZ+8VitCw9yoqNGiYDatsyt6jjIadlVkysve+u5jC9xTmV/qyl6bcJR7kAQZzg568xXmuE2lIyUNM5So7OMAFIhvp+YgGvEtFNnOKeJonSEvwP9NZzhHiOfLzBXPoxKP8yD6iPMXITjP+oTdfsp14lTJMJjGtOMFQfiFe4wWK8BP7qUd31hBNqMos2tKYFbRnJdGGjTzPz2yjEA1ZSKymKCM5ylaWcJrZxCZK8jgfU4vc/MW3xE7K8RUvsZb3Wc/XxCEqk4v/qMQlFvMZcZIafMOnLKM13zGceJNqPMU4KnCQAqQgbrKHpXSgFK/Qn6REO9YxjWE8Sx2SMJD4jfl8wgzy0YgPuEeUJQcD6EoWWpCaHsQkHuY9RpGON/icK0RyrvE680jG22TlHaIbx6jLnySkF+M5QxzmD6pwkTsMoSAdidqsojipuMyHzOQ4sYgfyElpzjKGErQkqvMyC7jFv9xmBM2JuTzDRDLxN4l4jF1EZjIwmhfZzSOMpT4xiH70IQG/k5En2UKcowudycsG8jCBmtwHgRv+EIeWyOAAAAAASUVORK5CYII=",
      "basn3p08.png", false, 32, 32, "d36bdbefc126ef50bd57d51eb38f2ac4");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAQAAADZc7J/AAAABGdBTUEAAYagMeiWXwAAADVJREFUeJxj/M/AwAGFnGg0MSKcLN8ZKAMsP4a+AaNhMBoGVDFgNBBHw4AqBowG4mgYUMMAAN8qIH3E64XIAAAAAElFTkSuQmCC",
      "basn4a08.png", false, 32, 32, "e2212ec5fa026a41826136e983bf92b2");
  testPngSuiteImage("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",
      "basn4a16.png", false, 32, 32, "f1423ebc08979252299ca238656ab0ba");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABGdBTUEAAYagMeiWXwAAAG9JREFUeJzt1jEKgDAMRuEnZGhPofc/VQSPIcTdxUV4HVLoUCj8H00o2YoBMF57fpz/ujODHXUFRwPKBqj5DVigB041HiJ9gFyCVOMbsEIPXNwuAHkgiJL/4qABNqB7QAeUPBAE2QAZUDZAfwEb8ABSIBqcFg+4TAAAAABJRU5ErkJggg==",
      "basn6a08.png", false, 32, 32, "e80a60aecf13ebd863b61167ba95960b");
  testPngSuiteImage("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",
      "basn6a16.png", false, 32, 32, "4d9d6473bb7403d7f85e3e7537c34e9d");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAQAAAGudILpAAAABGdBTUEAAYagMeiWXwAAAI1JREFUeJztj80KgzAMx3+BHvTWvUH7KPbB9yhzT7Dt5LUeHBWiEkFWhpgQGtL/RyIZOhLJ3Zli2UgOJAvzgECcs/ygoZsDyb7wA5Hoek2pMpAXeDw3VaVbMHTUADx/biG5Wbt+Lve2LD4W4FKoZnFYQQZovtmqd8+kNR2sMG8wBU6wwQlOuDb4hw2OCozsTz0JHVlVXQAAAABJRU5ErkJggg==",
      "bgai4a08.png", false, 32, 32, "e2212ec5fa026a41826136e983bf92b2");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAQAAAH+5F6qAAAABGdBTUEAAYagMeiWXwAACt5JREFUeJyNl39wVNd1xz9aPUlvd9Hq7a4QQjIST4jfhKAYHByEBhY10NRmCeMacAYT0ZpmpjQOjmIS9JzpOE/UNXZIXOKp8SDGQxOwYzeW49SuIy0OKzA2DlCCDQjQggSykLR6Vyvt6q10teofT0wybTyTP75zztw/7nzPued8z7nZAOZM+M3rYNdAljkTjBumPhCHJUssg5EWGIibOoy0Cuu7h7LtGvj2lqrnb96MCM0P2SENfj4PNh6AS/eBMm005D+xKaR1/Dcs442dyst/gM/2hzR1nmUElr5xQ1nfBw/Nf2NnZyfiygLICmkQESH/Fg9s8ULoNBxNwtEUXHjLMhRzJvTfhk+fCGnrfxDSru6HQ7st49FoRDx5KSJclgF3WqGjA5Z3WYaqW8bpGdDZCX2NkFX1HHQNVD2/sQ829sPK78B/TnXwq6mQpasQ0v4Iy4CI+CMU5Zbu/vAlXa3wwogHEv8BV5PQloTKt8/WKw+0Q9s2XT2+TVfXPgOdBfDr78O92Wfrv3QYoTzQDkt6oOUPunrqKV195xo8lHO2fumPEMX7QLm/C6QL1h6BE0JXf1RhGTOfRuTNBmUElLfnwLUgHDsHRtnZ+p+PYV/fDbV7oKwOlLfnQksFrDp0tn7eVxGeTjjzDDT9C9y/ELICKd29cI9mbuyDjX1Ocu7mYeyRmJ2lqxCzdffsfpgT//8IpqA9OInCP/GDMNFsGUpIg57fwc2XdPU3DbraewtGs8EzBiVDUGBDv8eJ4+MS+KgUMo9bxsKCmF36qWUrIQ0S7TDghe4P4co2Xf1Zq64mimD6NPA/B+fuOElI/8IyVo3E7PIfW3ZRPRQ0gRLSQLbDWD6kP4LkMzCwHS6X6upX39XV1wRcjVqGURuzS75p2b5ucDdCbh8oh0GxDBjtBDsCw+tgoANufg8iT8OOxyyjogIOvgzeOljUBNMWQMFhcL8PeRooEQFiLvS9Aze/DBe+BjmrLSPssli/FzFzOxz6V2jOwP7dUL0CZu+B6VMhuBWyNh6A7rDu7timq65yzayKwpIoVJ2AqigUb4fzK+Hcn+B8DcxLxuyyV2O2EhGQ1WYZs962qNyAmLULZo1D8T7whEHZCtp5KGuGsWZQvwVFTXD9EXivGbI0E3T18yEMiNmfDyVrltZ4M+w38+IwJQ7+OCT7ncROxEH+LYwEIRGEeBB6gtAVhFgh6GpsxDUrDC5TMzu26eotW1f7fqKrg/N11T6hq5lHdHUsX1eT39PVgeu62lOrqzdf19Wrhbo6u99hqFRuAPcCuFqumZcX+E3fszDttvOkmWOQ9oH1EnSXwrV2uHgPLGqM2eVxKFZBmRUG33mYEoVPFmrmBcVvFtVCZS3Ib0GyAz5rgSs/gzOtsOxWzK6cA8WrIXj3gsJTEIyC/wn4vVszT8/xm7PTMPoxDNTDJ3egpRdq18TsubehZC8E4uBTwVW5AeannHevroZwG3g2a2bkaV0d+rWuXi7V1SO9urq1CGpr4b7b8IVGp1P1uwxkFEajMPIYLH4YlkagZbVmnlvpN799AF5YF7Pn3YZALXhPQ14j5MRBUUEJHIPMi5DJh/EykI9C+Sqo2AFLl2nma68KoyoK+bsgtwKU98C1GVy/gCwTlGtvQlrAyEoYPAZ3quHi/bB/GXx8JmYfPIhx+DhG6D4ob4FAKUxpALUGcm3IXluurrm90K/ELvuVT0b9SlutX3llhV/ZdUrIvzopZO4SIY8/Zdf8/kM7MnpGyORXhBxeJ2QyKWQyI6TrejNc8jhN0tYGb1XD+raYvSgas93vx+ySUMyuWROz05cso6XFUaSLDY68xWzInnVOXXMjx69c8viVj572K9UrhLzXFnLBvULOfFxI+5aQiRIhZYeQN27YNV3ftyOZ+UKO+YQc7RRSud4MnZvgcg0sORGzZ0ehJAoFByA7Cu4mKFwJ5T8GayWcexzj4k2M1CswbINyvRmub3f6W0/B9DLwfx3cSXANQW47+G5D0VswYzUMe+HScoz2IEbahmzrirpmVlhIXQpZNl/IezYJWZwt5NQlQga3Cpn+GyGHPxIydUjI9KCQsk3IzItCDjTbNVafHcnSTBCG1ug/CoFjcNf+pT7AwGYH1pa/3Le2gGaKBkVXIREGK+w3r2/RzEIThhtg5AKkMzB+HiaOgGs35DSAehI8wqn+zIsOAdkI6XWQmgFDX4PB3RA/Av2N0Pcw9C+Avk3Qb0J/MwSOCmNW2DJ8Kii6CsNhSMRBJGHgQb952auZog6GLoF9HMZmwsRzkF0HeXXgXQWjdU73AIzOgZFVkGgC6wnoPQw9TdBzHD67BD2D0OOFopAw5iUtQ4uDLwxTUpMEUmFIdsGQCoN7YWAUepf4zfM+zRyYAUP/BemLMPFFUPrBcwwKypzWBUcDBtdCfyd0fxE6n3CWpM40dNZASUIYS+osI5ALBSnIj4M3DJ5fTRJIb4CRf4aUBslGSCwHayr0r4Dubr/ZdlIz586F4Qchsx3y/g605Y5ugBP5nXfhxiG43ARXmuDKSajQhVG9wjIKb4M/Cr7T4P038MTB/U+Q9w+TBMbCMNoP6elgN8LIkzD8ZUhUw8AA9GyDGx/4zbeqNbO3C8a6ID/iiBZAdwQuroQPHoHTM2DxPmGsb7OM4lcgEHDaaEoU3M+CmoK8fsgNQ87dGhgPw/hvQSZBPg9jUUhvBrsaUikYOgkD06H7FFxe7Tf3X9PM5GOOYgK0HHS2h7+uFMauU5ZRcg0CJyG/FjweUG9BXhRy9oLyXVDikB2G7CuTBNgAE5thIgUTjTDxJEy8A5kwZDKQ+SbInTD2AdjrYHAbdHT4zaXLNBPgtVeFsWOHZRS8AuoHkLMIlF+C6+/B5QLXi5AVhawCyLoFWXHI2gD8FBRhQGYzZDyQaYTxh2D8Asi5MNYJo6NgN0Eq5OwIPb+Fi5MRv/aqMAAe3gQ7HoNFXVC8ErR68ERA7YDcXMjxgdIE2Ysh+3VwrZ2cKQYoMRtkM4zthDEvjDaAfQBGciDZBokEDByGzwRc/Qqc3uSk+oV1gqqo8wQvrIN3jmMcvAbLX4bZd2D6CgjUgc8H3lJwF4G6E3KrIScIOUdBkZME0i2QPge2B1INMFwDiU6wfgm9vdBV7VT24mPC2FokWPxDmLfPmZIA8+oh/UMorIf/OYZxpBfmPgAzWqCoCPzfAV+ZMwg9Z0ANQt6bkFc7SWCkGVJVkPTA0B4QB6D/fbjTBp1dTjvVrhFUtEPFLijPg0CTM6LB8cs7YHwXuNuhaA10HMdoiUHZDJi2z5lIWjfkfwO8QfA0g9ueJJBshqFaSHjBaoD+a9BzjyMgyxKC0iEoTUDpEExPQCDhLBfKew4Brw8C+TAyFzLLICcfpvggmA+3fRhnfFBcB4WV4O8DXxDym8F7l0DiTRhMgeWB/gZHMhc1Coo+hWkhJ6KiNTA1BP4tMGUN5IWcQgLIa4Up74K/FUZbYSICSiu4Wx29CDRCbyvGxcNQuAf8QSh4E3wlk79FcVhrtLb4zUDK+RUFRz7H/pkzgLgH4u7/Y//c2aQd8ID/qGVodaIhW0hQq+zI9FNCFucLOe0hIaeWCjl1u5DBeUIGHhdSu09I7SkhfbVC5j8rpPfrQnr/XUj3NiGzZgg5ekDIsQeFHN8r5PgqISd+ICRfEtL1j0K6KoVUHhUyZ5qQeRuEzHML6T4h5MgX7EjPe/C/SQETOWwWx8sAAAAASUVORK5CYII=",
      "bgai4a16.png", false, 32, 32, "f1423ebc08979252299ca238656ab0ba");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABGdBTUEAAYagMeiWXwAAAG9JREFUeJzt1jEKgDAMRuEnZGhPofc/VQSPIcTdxUV4HVLoUCj8H00o2YoBMF57fpz/ujODHXUFRwPKBqj5DVigB041HiJ9gFyCVOMbsEIPXNwuAHkgiJL/4qABNqB7QAeUPBAE2QAZUDZAfwEb8ABSIBqcFg+4TAAAAABJRU5ErkJggg==",
      "bgan6a08.png", false, 32, 32, "e80a60aecf13ebd863b61167ba95960b");
  testPngSuiteImage("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",
      "bgan6a16.png", false, 32, 32, "4d9d6473bb7403d7f85e3e7537c34e9d");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAQAAADZc7J/AAAABGdBTUEAAYagMeiWXwAAAAJiS0dEAACqjSMyAAAANUlEQVR4nGP8z8DAAYWcaDQxIpws3xkoAyw/hr4Bo2EwGgZUMWA0EEfDgCoGjAbiaBhQwwAA3yogfcTrhcgAAAAASUVORK5CYII=",
      "bgbn4a08.png", false, 32, 32, "e2212ec5fa026a41826136e983bf92b2");
  testPngSuiteImage("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",
      "bggn4a16.png", false, 32, 32, "f1423ebc08979252299ca238656ab0ba");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABGdBTUEAAYagMeiWXwAAAAZiS0dEAP8A/wD/oL2nkwAAAG9JREFUeJzt1jEKgDAMRuEnZGhPofc/VQSPIcTdxUV4HVLoUCj8H00o2YoBMF57fpz/ujODHXUFRwPKBqj5DVigB041HiJ9gFyCVOMbsEIPXNwuAHkgiJL/4qABNqB7QAeUPBAE2QAZUDZAfwEb8ABSIBqcFg+4TAAAAABJRU5ErkJggg==",
      "bgwn6a08.png", false, 32, 32, "e80a60aecf13ebd863b61167ba95960b");
  testPngSuiteImage("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",
      "bgyn6a16.png", false, 32, 32, "4d9d6473bb7403d7f85e3e7537c34e9d");
  testPngSuiteImage("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",
      "ccwn2c08.png", false, 32, 32, "5189ed8d023b977fa90833e90e4a830a");
  testPngSuiteImage("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",
      "ccwn3p08.png", false, 32, 32, "1a63f18f17006d850d1c041e49a7721f");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAgAAAAgCAIAAACgkq6HAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAlwSFlzAAAAAQAAAAQAMlIwkwAAASdJREFUeJxV0mFx7DAMBOCvmRAQBVMIhVAIhaNwFPogvFI4CA2FQGggVBDcH3acNpMZja1daVfWWwWClWRvZ3OPGwGSA6YOjw5QepxgcX+lg2ZYKc7BXNip1G9f1bP6X9WqvqtMregBTk691NTCcbUYCXX19d0qbuqyVvVTDZNwdjWFJS+/c/PEw/5QZNnTGa1HIsNHeAUlEc0gztheyguRLlWN8XRs2Vv0Hm12xRGt5j2rS/qY753w6+oPI+OefuPNA/KyHuISZZYCJf+VyKVrlINR8nyw3I95MRy2XeCMwSiwK0FGSwxGODk4yyV8lgpP0o612UlTU3EtjXL5nNozrQTLr8RbxZMix9Z9cDQfxz1B2kK0WY0dabc5EtlRf0B1/Ku63McfFzN1pnMg8LcAAAAASUVORK5CYII=",
      "cdfn2c08.png", false, 8, 32, "dbfdad37268883ddeeecc745da77130c");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAICAIAAAAX52r4AAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAlwSFlzAAAABAAAAAEAH+hVZQAAAOtJREFUeJx9kmuxwyAUhL/M1MBawEIsxEIsxEIs3FiohVjAAhaOhSMh9wePQtt0hwEG5rx2d7r4QADVxbg3eN3zxbr7iEc5BTOssJaHFtIHeleo9RDao8EBCawvIFgglN5dRIjwLLNsuHBh3RRy5AQgwSn8D2aYA+TlEEuZB+sr0EpeHJE/zl1PtshGrMPICAdz3GFNzIJoJANr8+foE6xRdAeBMJGQqhSGnE6kOzjAdPUULfjyRtECAQfXIEJmCYMY8D1T5HDU1JWi6WqdhkFkH63xZhCNIr8oPsAGkacvNu2d8YOH3ql+a9N/CM1cqmi++6QAAAAASUVORK5CYII=",
      "cdhn2c08.png", false, 32, 8, "650268c7196860898cbe701005cf2407");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAlwSFlzAAAAAQAAAAEATyXE1gAAAHtJREFUeJxFzlENwkAURNFTgoFnYS3UQisBC1gACSCBSsDCVgIroSuhlbB8NIX5mUwyubldQzCQ2KjMcBZ8zKFiP0zcaRd53Stb3n2LNWvJSVIwX/PoNvbbNlQkJ0dqRI0Qxz5Qg+VlffxQXQuyEsphFxNP3V93hxQKfAEqsC/QF17WrgAAAABJRU5ErkJggg==",
      "cdsn2c08.png", false, 8, 8, "9e580d6237f77bfa8e49cfef19236bcb");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAlwSFlzAAAD6AAAA+gBtXtSawAAAmdJREFUeJyVluGR3CAMhb/bcQNqgRZowVeCr4RLCZsSsiXcteASQglxC7RACckPEEbYeDeandkZDHpP0hPi7S/PzIPoL1uCCBHS08O8DQEczI3TkW0Q/hdAYAb3nF2xBAHiiwAOlpddtzYIxQLM8Hl+PPNLh3L0GI8LAAdfMJvPNfqunA48+M5Zgo8+jqn8S+8aWGE7ZaoiCrB0xfKQzLFb+beCSfA99v5km3U1FfpWM48+mR6cJj93wVbTtsLvYxyaqFvBbMxKzsGnyjYTE/DwWdWWYO2K5PcgpuKkibopkoM/ZXVTWNESAyzwUU8ZebuSu6lLTuNdSjojPApDqUw93NF2D8DWJX8E0CTHg5DgJ6zMSroIdwXJTeNrPWIrXHU7tRW3endQzt7hXjwncIn5HWLUxlO2sesMgWQBkvkoGUBKLN/6PQKeOda8qIv+bhVI3AZdr6spB9L1cnTG5Rhgb7SRSa2usRcGQdl0G+zVVcnk25tEyPmhVtIGYm3SQjX7y5kEgoeVFaToKBdZr4drgFQBGm670nMFPXhCZAOPCBKr9yL7A1wPYMXV3CJb+fALZlJoejAnzNdtZc0AaENN3ajzppnXwjO7q8LfCYUK4LsU7QAN10xk2a/SBD9gAbF+K/fQhmRsBICOKo0j3/nucF3vnXEyxcNeyak4sRj3/bKqfM5fDXIcapi9OjJDv2sBacfKmTndZswOh2boC3z10dZB0PKvE+FEl+/9CLXPFn9KaT+ert9jAdZ+7fDwkiuMKwvnr4TB23Q+inJs0cjmNQD2iXkVTS7R5fNmDFCtNsDx+f6C/QMQQNfOLmy7EgAAAABJRU5ErkJggg==",
      "cdun2c08.png", false, 32, 32, "fbe519db5608cf411e933ccbd1f92f87");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAC1QTFRFIgD/AP//iAD/Iv8AAJn//2YA3QD/d/8A/wAAAP+Z3f8A/wC7/7sAAET/AP9E0rBJvQAAAB5oSVNUAEAAcAAwAGAAYAAgACAAUAAQAIAAQAAQADAAUABwSJlZQQAAAEdJREFUeJxj6OgIDT1zZtWq8nJj43fvZs5kIEMAlSsoSI4AKtfFhRwBVO7du+QIoHEZyBFA5SopkSOAyk1LI0cAlbt7NxkCAODE6tEPggV9AAAAAElFTkSuQmCC",
      "ch1n3p04.png", false, 32, 32, "a339593b0d82103e30ed7b00afd68816");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAwBQTFRFIkQA9f/td/93y///EQoAOncAIiL//xH/EQAAIiIA/6xVZv9m/2Zm/wH/IhIA3P//zP+ZRET/AFVVIgAAy8v/REQAVf9Vy8sAMxoA/+zc7f//5P/L/9zcRP9EZmb/MwAARCIA7e3/ZmYA/6RE//+q7e0AAMvL/v///f/+//8BM/8zVSoAAQH/iIj/AKqqAQEARAAAiIgA/+TLulsAIv8iZjIA//+Zqqr/VQAAqqoAy2MAEf8R1P+qdzoA/0RE3GsAZgAAAf8BiEIA7P/ca9wA/9y6ADMzAO0A7XMA//+ImUoAEf//dwAA/4MB/7q6/nsA//7/AMsA/5mZIv//iAAA//93AIiI/9z/GjMAAACqM///AJkAmQAAAAABMmYA/7r/RP///6r/AHcAAP7+qgAASpkA//9m/yIiAACZi/8RVf///wEB/4j/AFUAABER///+//3+pP9EZv///2b/ADMA//9V/3d3AACI/0T/ABEAd///AGZm///tAAEA//XtERH///9E/yL//+3tEREAiP//AAB3k/8iANzcMzP//gD+urr/mf//MzMAY8sAuroArP9V///c//8ze/4A7QDtVVX/qv//3Nz/VVUAAABm3NwA3ADcg/8Bd3f//v7////L/1VVd3cA/v4AywDLAAD+AQIAAQAAEiIA//8iAEREm/8z/9SqAABVmZn/mZkAugC6KlUA/8vLtP9m/5sz//+6qgCqQogAU6oA/6qqAADtALq6//8RAP4AAABEAJmZmQCZ/8yZugAAiACIANwA/5MiAADc/v/+qlMAdwB3AgEAywAAAAAz/+3/ALoA/zMz7f/t/8SIvP93AKoAZgBmACIi3AAA/8v/3P/c/4sRAADLAAEBVQBVAIgAAAAiAf//y//L7QAA/4iIRABEW7oA/7x3/5n/AGYAuv+6AHd3c+0A/gAAMwAzAAC6/3f/AEQAqv+q//7+AAARIgAixP+IAO3tmf+Z/1X/ACIA/7RmEQARChEA/xER3P+6uv//iP+IAQAB/zP/uY7TYgAAAgBoSVNUAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAARNzPXjAAABsUlEQVR4nA3BBwAIBBAAwEeojEhkKzszCWVrWGVn7xWK0LD3Kio0aJgNq2zK3qOMhp2VWTKy9767mML3FOZX+rKXptwlHuQBBnODnrzFea4TaUjJQ0zlKjs4wAUiG+n5iAa8S0U2c4p4midIS/A/01nOEeI58vMFc+jEo/zIPqI8xchOM/6hN1+ynXiVMkwmMa04wVB+IV7jBYrwE/upR3fWEE2oyiza0pgVtGcl0YaNPM/PbKMQDVlIrKYoIznKVpZwmtnEJkryOB9Ti9z8xbfETsrxFS+xlvdZz9fEISqTi/+oxCUW8xlxkhp8w6csozXfMZx4k2o8xTgqcJACpCBusoeldKAUr9CfpEQ71jGNYTxLHZIwkPiN+XzCDPLRiA+4R5QlBwPoShZakJoexCQe5j1GkY43+JwrRHKu8TrzSMbbZOUdohvHqMufJKQX4zlDHOYPqnCROwyhIB2J2qyiOKm4zIfM5DixiB/ISWnOMoYStCSq8zILuMW/3GYEzYm5PMNEMvE3iXiMXURmMjCaF9nNI4ylPjGIfvQhAb+TkSfZQpyjC53JywbyMIGa3AeBG/4Qh5bI4AAAAABJRU5ErkJggg==",
      "ch2n3p08.png", false, 32, 32, "d36bdbefc126ef50bd57d51eb38f2ac4");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAAd0SU1FB9ABAQwiON2c/4AAAADISURBVHicXdHBDcIwDAVQHypACPAIHaEjsAojVOLIBTEBGzACbFBGYAPYoEY9lQKfxElR4hwq58V17ZRgFiVxa4ENSJ7xmoip8bSAbQL3f80I/LXg358J0Y09LBS4ZuxPSwrnB6DQdI7AKMjvBeSS1x6m7UYLO+hQuoCvvnt4cOddAzmHLwdwjyokKOwq6Xns1YOg1/4e2unn6ED3Q7wgEglj1HEWnotO21UjhCkxMbcujYEVchDk8GYDF+QwsIHkZ2gopYF0/QAe2cJF+P+JawAAAABJRU5ErkJggg==",
      "cm0n0g04.png", false, 32, 32, "cf2f1ab4f34d0c70f15f636ef584d53a");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAAd0SU1FB7IBAQAAAB4KVgsAAADISURBVHicXdHBDcIwDAVQHypACPAIHaEjsAojVOLIBTEBGzACbFBGYAPYoEY9lQKfxElR4hwq58V17ZRgFiVxa4ENSJ7xmoip8bSAbQL3f80I/LXg358J0Y09LBS4ZuxPSwrnB6DQdI7AKMjvBeSS1x6m7UYLO+hQuoCvvnt4cOddAzmHLwdwjyokKOwq6Xns1YOg1/4e2unn6ED3Q7wgEglj1HEWnotO21UjhCkxMbcujYEVchDk8GYDF+QwsIHkZ2gopYF0/QAe2cJF+P+JawAAAABJRU5ErkJggg==",
      "cm7n0g04.png", false, 32, 32, "cf2f1ab4f34d0c70f15f636ef584d53a");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAAd0SU1FB88MHxc7O3UwH+AAAADISURBVHicXdHBDcIwDAVQHypACPAIHaEjsAojVOLIBTEBGzACbFBGYAPYoEY9lQKfxElR4hwq58V17ZRgFiVxa4ENSJ7xmoip8bSAbQL3f80I/LXg358J0Y09LBS4ZuxPSwrnB6DQdI7AKMjvBeSS1x6m7UYLO+hQuoCvvnt4cOddAzmHLwdwjyokKOwq6Xns1YOg1/4e2unn6ED3Q7wgEglj1HEWnotO21UjhCkxMbcujYEVchDk8GYDF+QwsIHkZ2gopYF0/QAe2cJF+P+JawAAAABJRU5ErkJggg==",
      "cm9n0g04.png", false, 32, 32, "cf2f1ab4f34d0c70f15f636ef584d53a");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAACsiDHgAAAABGdBTUEAAYagMeiWXwAAAANzQklUDQ0N0DeNwQAAAH5JREFUeJztl8ENxEAIAwcJ6cpI+q8qKeNepAgelq2dCjz4AdQM1jRcf3WIDQ13qUNsiBBQZ1gR0cARUFIz3pug3586wo5+rOcfIaBOsCSggSOgpcB8D4D3R9DgfUyECIhDbAhp4AjoKPD+CBq8P4IG72MiQkCdYUVEA0dAyQcwUyZpXH92ZwAAAABJRU5ErkJggg==",
      "cs3n2c16.png", false, 32, 32, "023541189afc3aa4ae617158b59fe635");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAANzQklUAwMDo5KgQgAAAFRQTFRFkv8AAP+SAP//AP8AANv/AP9t/7YAAG3/tv8A/5IA2/8AAEn//yQA/wAAJP8ASf8AAP/bAP9JAP+2//8AAP8kALb//9sAAJL//20AACT//0kAbf8A33ArFwAAAEtJREFUeJyFyscBggAAALGzYldUsO2/pyMk73SGGE7QF3pDe2gLzdADHA7QDqIfdIUu0AocntAIbaAFdIdu0BIc1tAEvaABOkIf+AMiQDPhd/SuJgAAAABJRU5ErkJggg==",
      "cs3n3p08.png", false, 32, 32, "6b15613bf70a37c37de24edfb8f6d1df");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAAYagMeiWXwAAAANzQklUBQUFGCbeQwAAAGJJREFUeJztlbERgEAMw5Q7L0EH+w/Fd4zxbEAqUUUDROfCcW1cwmELLltw2gI9wQgaastFyOPeJ7ctWLZATzCCjsLuAfIgBPlXBHkQ/kgwgm8KeRCCPAhB/hVh2QI9wQgaXuXOFG8QELloAAAAAElFTkSuQmCC",
      "cs5n2c08.png", false, 32, 32, "3211fb3ede03caf163fe33b4cf6d78f6");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAANzQklUBQUFGCbeQwAAAGBQTFRF/xkAQv8Axf8AAP97AP+9AP//AP8AAMX/AKX//94Apf8AAGP//5wAACH//1oAAP86/zoAY/8A5v8A/wAAAP9aIf8AAP+cAP/eAOb///8AAIT//70AAEL//3sAhP8AAP8ZRy+F9QAAAEtJREFUeJyFwQUBwAAAgDDu7u79Wz4CG5NA9YJW8AhqwSUoBIdgFISCUvAKBkEgWASp4BN0glkQCVZBLNgEiWAXZIJccAoqwS1oxA/GcT4B7dbxuwAAAABJRU5ErkJggg==",
      "cs5n3p08.png", false, 32, 32, "3211fb3ede03caf163fe33b4cf6d78f6");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAAYagMeiWXwAAAExJREFUeJzt1UENADAMQlGa4GPzr2pT0olo/mkgoO9EqRYba9HADhBgmGq4CL7sffkECDBNie6B4EGw4F8R4AOgBA+CBQ+CdQIEGOYB69wUb0ah5KoAAAAASUVORK5CYII=",
      "cs8n2c08.png", false, 32, 32, "023541189afc3aa4ae617158b59fe635");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAGBQTFRFIP8A/x8AAP8fAP/foP8AAP8/AP//AP8AgP8AAP9fAP9/YP8A/wAA4P8AAP+fAOD/QP8A//8AAMD//98AAKD//78AAID/wP8AAP+//58AAGD//38AAED//18AACD//z8As4GzYwAAAEtJREFUeJyFwQUBwAAAgDDu7u79Wz4CG7UgEHyCR3AJDsEimASDoBFsgliQCypBL1CZIBQkgkJQClrBLogEqaATjIJZsApOwS14xQ8p4j4B+PNT2QAAAABJRU5ErkJggg==",
      "cs8n3p08.png", false, 32, 32, "023541189afc3aa4ae617158b59fe635");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAMhJREFUeJxd0cENwjAMBVAfKkAI8AgdoSOwCiNU4sgFMQEbMAJsUEZgA9igRj2VAp/ESVHiHCrnxXXtlGAWJXFrgQ1InvGaiKnxtIBtAvd/zQj8teDfnwnRjT0sFLhm7E9LCucHoNB0jsAoyO8F5JLXHqbtRgs76FC6gK++e3hw510DOYcvB3CPKiQo7CrpeezVg6DX/h7a6efoQPdDvCASCWPUcRaei07bVSOEKTExty6NgRVyEOTwZgMX5DCwgeRnaCilgXT9AB7ZwkX4/4lrAAAAAElFTkSuQmCC",
      "ct0n0g04.png", false, 32, 32, "cf2f1ab4f34d0c70f15f636ef584d53a");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAA50RVh0VGl0bGUAUG5nU3VpdGVPVc9MAAAAMXRFWHRBdXRob3IAV2lsbGVtIEEuSi4gdmFuIFNjaGFpawood2lsbGVtQHNjaGFpay5jb20pjsxHHwAAADh0RVh0Q29weXJpZ2h0AENvcHlyaWdodCBXaWxsZW0gdmFuIFNjaGFpaywgU2luZ2Fwb3JlIDE5OTUtOTaEUAQ4AAAA+3RFWHREZXNjcmlwdGlvbgBBIGNvbXBpbGF0aW9uIG9mIGEgc2V0IG9mIGltYWdlcyBjcmVhdGVkIHRvIHRlc3QgdGhlCnZhcmlvdXMgY29sb3ItdHlwZXMgb2YgdGhlIFBORyBmb3JtYXQuIEluY2x1ZGVkIGFyZQpibGFjayZ3aGl0ZSwgY29sb3IsIHBhbGV0dGVkLCB3aXRoIGFscGhhIGNoYW5uZWwsIHdpdGgKdHJhbnNwYXJlbmN5IGZvcm1hdHMuIEFsbCBiaXQtZGVwdGhzIGFsbG93ZWQgYWNjb3JkaW5nCnRvIHRoZSBzcGVjIGFyZSBwcmVzZW50Lk0JDWsAAAA5dEVYdFNvZnR3YXJlAENyZWF0ZWQgb24gYSBOZVhUc3RhdGlvbiBjb2xvciB1c2luZyAicG5tdG9wbmciLmoSZHkAAAAUdEVYdERpc2NsYWltZXIARnJlZXdhcmUuX4AsSgAAAMhJREFUeJxd0cENwjAMBVAfKkAI8AgdoSOwCiNU4sgFMQEbMAJsUEZgA9igRj2VAp/ESVHiHCrnxXXtlGAWJXFrgQ1InvGaiKnxtIBtAvd/zQj8teDfnwnRjT0sFLhm7E9LCucHoNB0jsAoyO8F5JLXHqbtRgs76FC6gK++e3hw510DOYcvB3CPKiQo7CrpeezVg6DX/h7a6efoQPdDvCASCWPUcRaei07bVSOEKTExty6NgRVyEOTwZgMX5DCwgeRnaCilgXT9AB7ZwkX4/4lrAAAAAElFTkSuQmCC",
      "ct1n0g04.png", false, 32, 32, "cf2f1ab4f34d0c70f15f636ef584d53a");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAABlpVFh0VGl0bGUAAABlbgBUaXRsZQBQbmdTdWl0ZdWsxR4AAAA4aVRYdEF1dGhvcgAAAGVuAEF1dGhvcgBXaWxsZW0gdmFuIFNjaGFpayAod2lsbGVtQHNjaGFpay5jb20pRVcgpAAAAEFpVFh0Q29weXJpZ2h0AAAAZW4AQ29weXJpZ2h0AENvcHlyaWdodCBXaWxsZW0gdmFuIFNjaGFpaywgQ2FuYWRhIDIwMTHS6zPBAAABDGlUWHREZXNjcmlwdGlvbgAAAGVuAERlc2NyaXB0aW9uAEEgY29tcGlsYXRpb24gb2YgYSBzZXQgb2YgaW1hZ2VzIGNyZWF0ZWQgdG8gdGVzdCB0aGUgdmFyaW91cyBjb2xvci10eXBlcyBvZiB0aGUgUE5HIGZvcm1hdC4gSW5jbHVkZWQgYXJlIGJsYWNrJndoaXRlLCBjb2xvciwgcGFsZXR0ZWQsIHdpdGggYWxwaGEgY2hhbm5lbCwgd2l0aCB0cmFuc3BhcmVuY3kgZm9ybWF0cy4gQWxsIGJpdC1kZXB0aHMgYWxsb3dlZCBhY2NvcmRpbmcgdG8gdGhlIHNwZWMgYXJlIHByZXNlbnQufjUNRAAAAEdpVFh0U29mdHdhcmUAAABlbgBTb2Z0d2FyZQBDcmVhdGVkIG9uIGEgTmVYVHN0YXRpb24gY29sb3IgdXNpbmcgInBubXRvcG5nIi7EGQUHAAAAJGlUWHREaXNjbGFpbWVyAAAAZW4ARGlzY2xhaW1lcgBGcmVld2FyZS7TvjIJAAAATElEQVQokWP4DwbGxi4uoaFpaeXlDGQJKCkhuB0d5An8/4/gzpxJngDcSRBAlgAIQAxctYo8AYSTwFyyBBDc3bvPnCFPAMGFeo50AQDds/NRVdY0lwAAAABJRU5ErkJggg==",
      "cten0g04.png", false, 32, 32, "ac076245d12023e111ec36acb36dcff1");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAABtpVFh0VGl0bGUAAABmaQBPdHNpa2tvAFBuZ1N1aXRl8x/ISQAAADlpVFh0QXV0aG9yAAAAZmkAVGVraWrDpABXaWxsZW0gdmFuIFNjaGFpayAod2lsbGVtQHNjaGFpay5jb20pTbKY1QAAAEhpVFh0Q29weXJpZ2h0AAAAZmkAVGVraWrDpG5vaWtldWRldABDb3B5cmlnaHQgV2lsbGVtIHZhbiBTY2hhaWssIEthbmFkYSAyMDExGP2/hwAAAOtpVFh0RGVzY3JpcHRpb24AAABmaQBLdXZhdXMAa29rb2VsbWEgam91a29uIGt1dmlhIGx1b3R1IHRlc3RhdGEgZXJpIHbDpHJpLXR5eXBwaXNpw6QgUE5HLW11b2Rvc3NhLiBNdWthbmEgb24gbXVzdGF2YWxrb2luZW4sIHbDpHJpLCBwYWxldHRlZCwgYWxwaGEta2FuYXZhLCBhdm9pbXV1ZGVuIG11b2Rvc3NhLiBLYWlra2kgYml0LXN5dnl5ZGVzc8OkIG11a2FhbiBzYWxsaXR0dWEgc3BlYyBvbiDigIvigItsw6RzbsOkLsc2cVkAAAA/aVRYdFNvZnR3YXJlAAAAZmkAT2hqZWxtaXN0b3QATHVvdHUgTmVYVHN0YXRpb24gdsOkcmnDpCAicG5tdG9wbmciLlFtpV0AAAAtaVRYdERpc2NsYWltZXIAAABmaQBWYXN0dXV2YXBhdXNsYXVzZWtlAEZyZWV3YXJlLvx3Hi8AAABISURBVCiRY/gPBsbGLi6hoWlp5eUMZAkoKSG4HR3kCfz/j+DOnEmeAAqXgTwBBHfVKvIE0LhkCSC4u3efOUOeAILLAAGkCwAA+XLyQRLQxL0AAAAASUVORK5CYII=",
      "ctfn0g04.png", false, 32, 32, "0428cabaa4c89cb12cd5cf0f269a0ff1");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAACBpVFh0VGl0bGUAAABlbADOpM6vz4TOu86/z4IAUG5nU3VpdGUgh0C5AAAARmlUWHRBdXRob3IAAABlbADOo8+FzrPOs8+BzrHPhs6tzrHPggBXaWxsZW0gdmFuIFNjaGFpayAod2lsbGVtQHNjaGFpay5jb20p1io2ZgAAAIlpVFh0Q29weXJpZ2h0AAAAZWwAzqDOvc61z4XOvM6xz4TOuc66zqwgzrTOuc66zrHOuc+OzrzOsc+EzrEAzqDOvc61z4XOvM6xz4TOuc66zqwgzrTOuc66zrHOuc+OzrzOsc+EzrEgU2NoYWlrIHZhbiBXaWxsZW0sIM6azrHOvc6xzrTOrM+CIDIwMTHXI+R2AAAB9WlUWHREZXNjcmlwdGlvbgAAAGVsAM6gzrXPgc65zrPPgc6xz4bOrgDOnM65zrEgz4PPhc67zrvOv86zzq4gzrHPgM+MIM6tzr3OsSDPg8+Nzr3Ov867zr8gzrXOuc66z4zOvc+Jzr0gz4DOv8+FIM60zrfOvM65zr/Phc+BzrPOrs64zrfOus6xzr0gzrPOuc6xIM+EzrcgzrTOv866zrnOvM6uIM+Ez4nOvSDOtM65zrHPhs+Mz4HPic69IM+Hz4HPic68zqzPhM+Jzr0tz4TPjc+Az4nOvSDPhM6/z4UgzrzOv8+Bz4bOriBQTkcuIM6gzrXPgc65zrvOsc68zrLOrM69zr/Ovc+EzrHOuSDOv865IM6xz4PPgM+Bz4zOvM6xz4XPgc61z4IsIM+Hz4HPjs68zrEsIHBhbGV0dGVkLCDOvM61IM6szrvPhs6xIM66zrHOvc6szrvOuSwgzrzOtSDOvM6/z4HPhs6tz4Igz4TOt8+CIM60zrnOsc+GzqzOvc61zrnOsc+CLiDOjM67zr/OuSDOu86vzrPOvy3Oss6szrjOtyDOtc+AzrnPhM+Bzq3PgM61z4TOsc65IM+Dz43OvM+Gz4nOvc6xIM68zrUgz4TOvyBzcGVjIM61zq/Ovc6xzrkgz4DOsc+Bz4zOvc+EzrXPgi6miCkYAAAAiWlUWHRTb2Z0d2FyZQAAAGVsAM6bzr/Os865z4POvM65zrrPjADOlM63zrzOuc6/z4XPgc6zzq7OuM63zrrOtSDPg861IM6tzr3OsSDPh8+Bz47OvM6xIE5lWFRzdGF0aW9uIM+Hz4HOt8+DzrnOvM6/z4DOv865z47Ovc+EzrHPgiAicG5tdG9wbmciLkN4y+sAAABDaVRYdERpc2NsYWltZXIAAABlbADOkc+Azr/PgM6/zq/Ot8+DzrcAzpTPic+BzrXOrM69IM67zr/Os865z4POvM65zrrPjC4snq9sAAAAXUlEQVQokZ3OQREAIQxD0WjBAhawgAUsYKEWsFALWEALFrLTnd3pPcf/DpkAIEuptbXex5gTAkSSf5ppkINma2nQGvl9hLsCkeR7KRIK5CX3vTXIBM7RIAcj7xXgAUU58kEPspNFAAAAAElFTkSuQmCC",
      "ctgn0g04.png", false, 32, 32, "32546f7ef2cce19fcbaa6b8849592c38");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAACZpVFh0VGl0bGUAAABoaQDgpLbgpYDgpLDgpY3gpLfgpJUAUG5nU3VpdGVT/Uu3AAAAPmlUWHRBdXRob3IAAABoaQDgpLLgpYfgpJbgpJUAV2lsbGVtIHZhbiBTY2hhaWsgKHdpbGxlbUBzY2hhaWsuY29tKc9NfecAAABoaVRYdENvcHlyaWdodAAAAGhpAOCkleClieCkquClgOCksOCkvuCkh+CknwDgpJXgpYngpKrgpYDgpLDgpL7gpIfgpJ8gV2lsbGVtIHZhbiBTY2hhaWssIDIwMTEg4KSV4KSo4KS+4KSh4KS+3xTVhQAAAmVpVFh0RGVzY3JpcHRpb24AAABoaQDgpLXgpL/gpLXgpLDgpKMA4KSV4KSw4KSo4KWHIOCkleClhyDgpLLgpL/gpI8gUE5HIOCkquCljeCksOCkvuCksOClguCkqiDgpJXgpYcg4KS14KS/4KSt4KS/4KSo4KWN4KSoIOCksOCkguCklyDgpKrgpY3gpLDgpJXgpL7gpLAg4KSq4KSw4KWA4KSV4KWN4KS34KSjIOCkrOCkqOCkvuCkr+CkviDgpJvgpLXgpL/gpK/gpYvgpIIg4KSV4KS+IOCkj+CklSDgpLjgpYfgpJ8g4KSV4KS+IOCkj+CklSDgpLjgpILgpJXgpLLgpKguIOCktuCkvuCkruCkv+CksiDgpJXgpL7gpLLgpYcg4KSU4KSwIOCkuOCkq+Clh+Ckpiwg4KSw4KSC4KSXLCDgpKrgpYjgpLLgpYfgpJ/gpYfgpKEg4KS54KWI4KSCLCDgpIXgpLLgpY3gpKvgpL4g4KSa4KWI4KSo4KSyIOCkleClhyDgpLjgpL7gpKUg4KSq4KS+4KSw4KSm4KSw4KWN4KS24KS/4KSk4KS+IOCkuOCljeCkteCksOClguCkquCli+CkgiDgpJXgpYcg4KS44KS+4KSlLiDgpLjgpK3gpYAg4KSs4KS/4KSfIOCkl+CkueCksOCkvuCkiCDgpJXgpLLgpY3gpKrgpKjgpL4g4KSV4KWHIOCkheCkqOClgeCkuOCkvuCksCDgpJXgpYAg4KSF4KSo4KWB4KSu4KSk4KS/IOCkpuClgCDgpK7gpYzgpJzgpYLgpKYg4KS54KWI4KSCLvrUkQYAAACRaVRYdFNvZnR3YXJlAAAAaGkA4KS44KWJ4KSr4KWN4KSf4KS14KWH4KSv4KSwAOCkj+CklSBOZVhUc3RhdGlvbiAicG5tdG9wbmcgJ+CkleCkviDgpIngpKrgpK/gpYvgpJcg4KSV4KSwIOCksOCkguCklyDgpKrgpLAg4KSs4KSo4KS+4KSv4KS+IOCkl+Ckr+Ckvi4VxVHXAAAAQmlUWHREaXNjbGFpbWVyAAAAaGkA4KSF4KS44KWN4KS14KWA4KSV4KSw4KSjAOCkq+CljeCksOClgOCkteClh+Ckr+CksC4tT0C7AAAAYElEQVQokWP4/19Q8P9/Y2MXl9DQtLTycgayBJSU/v+HcTs6yBMAARh35kzyBFxc/v8HO4kByGUgTyA09P9/sJMYVq0iTwDJSRBAhgCMCzLwzBnyBGDc3bsZGO7eJUsAAEBI89kMzfvBAAAAAElFTkSuQmCC",
      "cthn0g04.png", false, 32, 32, "c570e7393458556ecef30b71824e0d6e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAACBpVFh0VGl0bGUAAABqYQDjgr/jgqTjg4jjg6sAUG5nU3VpdGUPGlwCAAAAOGlUWHRBdXRob3IAAABqYQDokZfogIUAV2lsbGVtIHZhbiBTY2hhaWsgKHdpbGxlbUBzY2hhaWsuY29tKeXxzKEAAABTaVRYdENvcHlyaWdodAAAAGphAOacrOaWh+OBuADokZfkvZzmqKnjgqbjgqPjg6zjg6Djg7TjgqHjg7Pjgrfjg6PjgqTjgq/jgIHjgqvjg4rjg4AyMDExhF9tvgAAAXdpVFh0RGVzY3JpcHRpb24AAABqYQDmpoLopoEAUE5H5b2i5byP44Gu5qeY44CF44Gq6Imy44Gu56iu6aGe44KS44OG44K544OI44GZ44KL44Gf44KB44Gr5L2c5oiQ44GV44KM44Gf44Kk44Oh44O844K444Gu44K744OD44OI44Gu44Kz44Oz44OR44Kk44Or44CC5ZCr44G+44KM44Gm44GE44KL44Gu44Gv6YCP5piO5bqm44Gu44OV44Kp44O844Oe44OD44OI44Gn44CB44Ki44Or44OV44Kh44OB44Oj44ON44Or44KS5oyB44Gk44CB55m96buS44CB44Kr44Op44O844CB44OR44Os44OD44OI44Gn44GZ44CC44GZ44G544Gm44Gu44OT44OD44OI5rex5bqm44GM5a2Y5Zyo44GX44Gm44GE44KL5LuV5qeY44Gr5b6T44Gj44Gf44GT44Go44GM44Gn44GN44G+44GX44Gf44CCwwUNtAAAAGNpVFh0U29mdHdhcmUAAABqYQDjgr3jg5Xjg4jjgqbjgqfjgqIAInBubXRvcG5nIuOCkuS9v+eUqOOBl+OBpk5lWFRzdGF0aW9u6Imy5LiK44Gr5L2c5oiQ44GV44KM44G+44GZ44CCwoP4MAAAADJpVFh0RGlzY2xhaW1lcgAAAGphAOWFjeiyrOS6i+mghQDjg5Xjg6rjg7zjgqbjgqfjgqLjgIJ28EPmAAAAZUlEQVQokWNgYPgPBMbGLi6hoWlp5eUMZAgICiop/f8P43Z0kCOgpGRs/P8/jDtzJjkCIAP//4cayLBqFTkCaFwGcgSQnMSwe/eZM+QIwLggA8+cuXuXHAEYd/duoKMY3r0jQwAATn/xuQxIlj4AAAAASUVORK5CYII=",
      "ctjn0g04.png", false, 32, 32, "81fd25285e6b46f3996cfd02b1f16071");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAA50RVh0VGl0bGUAUG5nU3VpdGVPVc9MAAAAMXRFWHRBdXRob3IAV2lsbGVtIEEuSi4gdmFuIFNjaGFpawood2lsbGVtQHNjaGFpay5jb20pjsxHHwAAAEF6VFh0Q29weXJpZ2h0AAB4nHPOL6gsykzPKFEIz8zJSc1VKEvMUwhOzkjMzNZRCM7MS08syC9KVTC0tDTVtTQDAIthD6RSWpQSAAAAu3pUWHREZXNjcmlwdGlvbgAAeJwtjrEOwjAMRPd+xU1Mpf/AhFgQv2BcQyLcOEoMVf8eV7BZvnt3dwLbUrOSZyuwBwhdfD/yQk/p4CbkMsMNLt3hSYYPtWzv0EytHX2r4QsiJNyuZzysLeQTLoX1PQdLTYa7Er8Oa8ou4w8cUUnFI3zEmj2BtCYCJypF9PcbvFHpNQIKb//gPuGkinv24yzVUw9Qbd17mK3NuTyHfW2s6VV4b0dt0qX49AUf8lYE8mJ6iAAAAEB6VFh0U29mdHdhcmUAAHiccy5KTSxJTVHIz1NIVPBLjQgpLkksyQTykvNz8osUSosz89IVlAryckvyC/LSlfQApuwRQp5RqK4AAAAdelRYdERpc2NsYWltZXIAAHiccytKTS1PLErVAwARVQNg1K617wAAAMhJREFUeJxd0cENwjAMBVAfKkAI8AgdoSOwCiNU4sgFMQEbMAJsUEZgA9igRj2VAp/ESVHiHCrnxXXtlGAWJXFrgQ1InvGaiKnxtIBtAvd/zQj8teDfnwnRjT0sFLhm7E9LCucHoNB0jsAoyO8F5JLXHqbtRgs76FC6gK++e3hw510DOYcvB3CPKiQo7CrpeezVg6DX/h7a6efoQPdDvCASCWPUcRaei07bVSOEKTExty6NgRVyEOTwZgMX5DCwgeRnaCilgXT9AB7ZwkX4/4lrAAAAAElFTkSuQmCC",
      "ctzn0g04.png", false, 32, 32, "cf2f1ab4f34d0c70f15f636ef584d53a");
  testPngSuiteImage("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",
      "exif2c08.png", false, 32, 32, "7acea6df2b0e2a7613981decc569b2fb");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAABBklEQVR4nIXSL0xCURTH8a/gGHNjBjc3AoFioFhIr7z0molmsxFpVCONSLPRbCYSiWKxWAwWN4ObG5ubcwzk6f3jds71hXPr/eye3/3dy/VkOruZ394tlqv7h8en55fXt/f1x+fXZrv73pflD2NDMDIEQ0NwZQguDcHAEFwYgkILwkoEuRJQFWQi3Jafkgr6IiDmAJWDcxEQkroTVFJ6IsAn9SHUXTgTgX+5kFLdlq4I8DlwZ6g+6IgANwV/W9UYbRHh9DBFdcqpCL8f+1CtcyLC7Sd9BMGxCEj6iIKWCKIg9vEnOEpFWPr1aVZF8j9oJCLLi38/iENDUDcENUNwYAgsgSWwxC/EfcpYUKbOtgAAAABJRU5ErkJggg==",
      "f00n0g08.png", false, 32, 32, "f34b8a71205dc26cd37d58fb19179172");
  testPngSuiteImage("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",
      "f00n2c08.png", false, 32, 32, "d70ea8925988413a9fe9de1633a31033");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAABCElEQVR4nIXSIUvEQRAF8Pd2dnb2BDEIgsFgMVgsJovJZhIMNsFgFC4IpuOSYDi4IlwQDIJBMIjB5Cew+I0MHvff2bC7+ce+meFxki2bmaWUVFWjShQRCSGQJMjbUUfcWFvwOnfEpXXEhXXEmXXEaWoLnpgT5/j2gsepFFfAl/+DR1qIMYAPn8LDNIgpALz5OXigKzEHZmO8+Em5ryuxwCTf4cnvwr04iGyjKR79ttxVJx4w8/fgTnRijnt/MW5HJxaoGsQtceIZVYO4KU68omoQN6IT76gaxPXgxCeqBnFNBvGD/1emMIdB/C5BmcIUClHedCkYQ1tQ2BYMbAuSbUF0BNERREf8AZVRLIMTf6sKAAAAAElFTkSuQmCC",
      "f01n0g08.png", false, 32, 32, "38446c18ff7e0ca3951b7ad5fa5e81c0");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAEY0lEQVR4nK2WP+wlVRmGn/f8/c6cM+fsCDQEIhRYidlCY4GRBBIrsdCQmFCwncZGCk0oLGhIKCgsjDGxojBWkoCxNdBSkKWl2cJCowkNJSzXYu7cndn97Yagk6d4zzfJfZtv7nN0wrCM2ZliV+ev9qqYPnk8L07kTLYzZlfnr/RKt76bR2AJImWS3SHb/+Womy+kkTQiSxYxE+1Asv9xovdvxJEZppFZigi2IxMKwYi7fN95Jtg2v5P111+GXjQKY9IoLFV425Hx5ZDDfeZrCHYne8MX/ek3oU/0SaMymkZlmYWzHRlXHpT9fp5xts/6w+thbvRKnzUao2vMLEPIdmRUvnTOyC5Zb77pW2Oe6V29MwZjaAyWRWB3IX0AnE4/uvfV/dArr/naqI0607paZx70oXkwFgm78Jg+BoB/nX4A5rb5lRlMa/7pr51VlcbUKDO1a+rUQRuqg3mRwxz2Pf2b7fnw9P11+GXQc79wuZIr1siNMss6pVOGpkEZ1EU39Bm7593Tdzx2PwJlf9T1lxWrYiVVUiM18qzcyZ08ZIPfPvn5+rtvnB55Vf8Bfnd6OmB74vEYKJesJ15UmPBVsRIqoREbcVbsxE4aevf658CNU4zYH/Up8KvTUxFbSVu4i0RZgx76oVzBT7gqX/EV3/ANPyt0fCcMhYEfxEXv6TPgx6fHE7aSt3DVpCRM9rxUUMFNqKKKq1LDNTTjOq7LDdxAg3987TbwrdPDGcvbv3zeYZTj0cQzyKBAgQkqVFShQYMZOuowYHD7SYB2epAKyvEovp3IkokiiphEFdWpiuZoYnZ0qTuGu339n0A4ff3BKii7o/hmxUR2mKM4Fc/kqJ7qqJ7maZ7Zqwe6/+LZjwD3yTNawtEAZwmUe5wgvtFJDvNkj3mKVwlMnhrOtEgLzFE9fvHC3wF36yeMqCXf44FopHKciCcGKZA85snrEkeVwBSpkZqokZZomTmdXnobcDd/xjBG1lLsKBA7eyBesnj02rkgBSySIxYpUSUxJWqmJqrRMs1OP38LcO+/yiiMiVG0VDsKZOXiCvHINaInRVIgRSyS1yXOlMyUqcbv3+GqR7f+zKhaZjsKxHbSENc6MRAjaSVhiZyxTDGKMRlv/eXqgpt/Y3TGrGXYUSAXIYh5xntiJEZSOmNGXr+bQilME7VSq2qlNTaBsAmEMbQs99qggLCCD3hPiMRITMS1xs6XJStYoUyUSqlMlZ1A2ATCPDQW2wsEikOEhPe4rcNHYiKsNUbM5/3OBZvIlVKxyk4gbAKhDs3LXUIQLuAc8viA8/iIj4SET8R0voOc97uQJ1LlKBA2gTANylBd9n4QchseF3AeF/ERn84Ew2fiut+FOBErR4GwCQQb5KGyXMwgVi4d2joU8QmXcGl/zyEUwoSvHAXCJhDSIA7ysppB676dO9g65FFEEZfQWmMo49cVPwuEo0DYBMImEEvngq2DrYOAPEQUUYKE0vHOcxAIR4FwEYhf/gtC7nstgnuX5wAAAABJRU5ErkJggg==",
      "f01n2c08.png", false, 32, 32, "a9081889af0c30f206d793919792a0b6");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAABKklEQVR4nIXRL0uDURgF8MP5OBaLIIggCIJBDAODTdgHGCxYh0kwDFYEwwsLA8Ng4U2CYBLBYrEYLIJBEARB9H3uX8Om97nvwr35x+Wc83BwOjyvJtP66ub2/uHx6fnl9e394/PruzHW+RAifxppRESMMdZa66x33nsfQogxRkRKQbApCEpBUAqCUhA0BUHR4mB/d3tzPRM0SnQBAMgEbRJ9oJpM61zQJAEMRURqQOWgTQKYJwVUUtokesfzLoDqQqfEogug2tIuiS6g9qBriz5QqcXoWuJk0eVP0OdiBAyy1ekzMQZ6+V3otJgBR63LMShxDXTat6VP4g7A7HJ8MTpTgiEJ/D/1B0MSK6trG1s7e51DnYNBieXVRRgLgrEgGAuCsSCIgiAK4hfp5Je/v8zr/QAAAABJRU5ErkJggg==",
      "f02n0g08.png", false, 32, 32, "9af1c44cbb489e5385f422d047612dfc");
  testPngSuiteImage("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",
      "f02n2c08.png", false, 32, 32, "14658e750fd12c4777c46a949bb3399c");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAABTElEQVR4nH3SsUtCURTH8e/phERREBENIUWCUBQ0BC4OQcsDB6EhCIoiECEDg6JIzOfaXxBCBUHQIDg0tTgINTQEDUHgH+C/8RrU53nKe3e5HPhw77n3d9R1Mtndvf3Do5Nc/vSseH5xeXV9UyqXbysV1626VU2JIIDQ24ZK3fKihW5KtNANooWuEi00SbTQBNFCl8QXhUb99eX56aF2nx8IjeOLJv2VG5yhi/ii+6d1+DC36IIMhCAcwLvtQ+cJiALUA53qHFaU4DH4Fp3FiDvgrZG1QmfM8/nGdZoEhE7bD0qlPRkSOoUViCdOk4yJe5IR0cIxcU8wIj7ZMXHHxBetdje5L7ZN3DHPFx3aCaAGaRP3uPgi3qH99/sDRZO+KgEBcGznQ8ewYnklubYemCAVrAgmJ4D2yzDRBRFCe0MWKvogVPggTAxAiPgHY2dcQrz+CzkAAAAASUVORK5CYII=",
      "f03n0g08.png", false, 32, 32, "bab714eea8df8ab97f4095442247c9e1");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAE0klEQVR4nH2WXYhUZRjHf8+8Z2fnzNcZv1r7kCyyD7Gy1CiN2kK3kAXDQjJCKk2FCr0QDAp6D3hjflB30lV30k1Xe9FtIMEiXlh2YRBi5UWCre7sfO6cfbo45505M7O7w2GY4Rze3/v/P8/7/I9RcjZXsMWKray2q9fa+x+y69bbRx6zG560GzfZpzfb57babS/al162r4zb13faiTft7km75y279x2771373vv2wAf24CF75Kj95FN77Lg9ccKe/Nx+8aX9ytpTp8yxdcYvRQRQhjKUoARFKEIBCpCHPPiQc9+jkHXfIzACHnhgwEAGMiAgAOboU55W8IOIgARTSmG6jEIKk0thYka2n2F6DPPh84ZANMBfkWKUHaCL6erIp3R0pSytw+zfnqGsVNBA/ZUDDKUozi6lIMnqvpITp0PJClkYUUbAEzzwFAMZIYN5+9UMJaEMFdEAf3WaIZSUklCEglBQ8kIB8oKv+EIORsUxJPFqRPDE6RAzuStDkT7GmiUYRcfIgy8pHcKoMuoYiY6EYSbeyCQmlIWSY4wty4i9WkZHimHGd0vSJHFJy0qABuqv7cQMCf4Kz/8bnr8VnrsZnvszPHs9PPt7eOZqeOZK+PW0Pb2l11qj/a01Ah5m4/ZMA+pQgzmkCrMwi9xVmn40C9+ELZb+fGafbUArdc3DPLRhHiIw67dJA6lDHeaglmLMKHU/+sgWDtrgkF31sR07bB88Yh/+LrwVr35N99WhAQ1oQtMx2g4zD+a+zdJ0D9USHTgGM6o1v3PP/a3CTvktXv0n3VWDeGdpRhtppxgm2EiLAYY4hszCjFL1o5hxWP6OV7+g2+bchgYATWghsY42mNzj0oYm0oKG29EwY9aPzko1Xv2EbqjCHMy555fRYcyjya8mEt9zNZdaijG1otMt7C/hf5fC2w/bVV0Raa8GdBhdJ/NOTiu516t5tx7VCwqsV2/M5u+EbeBqeHuFXTmXtEbXK+nXgek8QAQd11ttaKEttIk20DpaQ2to5jiZk8zqwozfzttcO+wAN8LbozaooXW0jjaggTbRJtpCW9BCDWNZ8Nz5y7rxGA+BvJvUBaEEZQjQSuSX8nasHd4ByvaJ4QCRVIAYVuYQA158+CQ5jjmXL11GHBFlCFQrHT/Q8Cbg263pAJGhADFUfNQgnjvdMSM7zBCKbo4HaKCnrwFZ+9pAgEh/gBhKOfBQk9IRD/hYiq92mql/ZOIZx4h1VPTbaSBj9wyFVFFSdhnyfpJAgzqSYnD5BsDUHzLxQpexsPlc3LIL9gCDQdjnlSGXA0mCrl9Hcu3YxOXrAFNX+fGKXpzW7y/Fq8uNi6pBx1+zKCMuuyGbdQEaM7yUji5jC5d/HZij8vMPUIIKGkT+2DAjDhDDSBaVfkbslZeSMsqOHYyPy8SETE7K3r2yfz+9egRoEPlrF2UYMgYVFpQFkiuCSOlIcvwGDnqjN0zSg1fvaqfp96ZufNWSFwxQQRWFKMWIhHnoOEDbjfzuUIynRGq4R3Wf9HCvYpIXMAWEBUWX1tFyjGUDJKr1MUyvaup0DHjV1bGoV4syqj2GcdvvZ6S96iiR9AGGvRpidGYThulrvZjUZSyvo5myq+6kpBl3fe4NAAYYw14t2lfNVM0HdMz4/wOljpe9l86W/gAAAABJRU5ErkJggg==",
      "f03n2c08.png", false, 32, 32, "eef71a6ef947497fe0c45ea2ba65c5dd");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAAA1ElEQVR4nIXRwW3CMBSA4d/2s81MnaFiBqQOwA0pp26A1EsnqJQtWAtCIebAAdsP2zlEetGn/9mKfG/iJsYYQwjeey/eiXPOWWuNMQYjZ1qPBUySpQmeQi5tgAV6BbAjgB0BrFz7QBW2nCqgCtUH+S/GLw3qwl+1syxMUN+qWsFRgVs2/OhAXTioX5MXZlhg4re5gt0FoKjIPZ+W7P0WzABMHFsrytITrC/x4UMMcWYfs1PIHdUoD7mixFKDUnwGDfSWDKSBkERfSKIvJNEXwkAIA/EAFiZMByGZYIEAAAAASUVORK5CYII=",
      "f04n0g08.png", false, 32, 32, "2bb4337fabd31e8786cbb3aec25315b1");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAADoElEQVR4nK2WzY4bRRSFv1tV/ed225SigSgiCx6AF+AJ2CTvAXv27Fmxy1vwAKyyZ8kqQooQUQQsBjwzjstjuy+Lqm6XPWPLIFpH1q3qdp+qW+ee206pqSvqOqGpH4//262mdn89V2/WVOxRn4jzYQEFOLBgSJdCD1vYwgYMCG7xDJx6t6bkANVlw0jj0uv2NDvYgsEtPlZKoVBfrdO6cpQXz4y7OaRxiydKpdRCpb4JaTkJmoIii0/Oa4otGEU00ribudIIjTIRGvVtwHIO7uK78QxuOpgoE6FVpkKrvgsY/gXsubvutoEWWqUTpspM6NTPA8L/AndXKRU0MIEpdMpcmKv3K9AMgIq8BlRfHs0/iEfZqnt3Rxv3gE6RKXToDOnQuQ/5asbrDwKDXuL8wxgQMOB+X2qNxA00aItMBrIW7XwY8/mFvIkE7wmXH5D7c0mFjkXaQA3NkLMGbf3KonbY+A/6+TvSzKM4Upm7XmqBFPsi1QqpiKxSQ4U2Pnwrt5HgV8JBqYz1cAp/f8ChFin2TyfKAi2REoph+V/r1VvC+bo+mne3qyhlNUi2NbWIA4s65KfPtpHgF8J5x3o445arJACDDoGMQ5N55fNr+4ZVheaoj4dH5qsuBM0lmPSeDfoXKXi/2AHi77ZyB3yqT2u2NbsMfU3fHA4d61IROaw/xUjSscA1YF9/wsKAUQz8BgSuLmkgjlDEN8Z8SHIWq8llbCTQm6vBzFwkWPFUKTNUmiU/DpXScW/Zv8sqVpIljkqLifr5KI9L+XEJT/Sr8+3CcS+HbuuUQpK44x/OXYFnj6l/XxiRYJDPAUdSnXzzJW3NtGJa09Uyq/sX3wPm7XfMm0B7tj9YxybaXn7I8QzGTVRQSbKPRpkkqS0+gqnSrnynmCMwBI5NDybz2JHDZCdRKKUkkTTpwcUMpjBTuuDnpxqCYxfbc4QeGjp56oa8VfLqFV3HTWogMFfmwftHD8mx2aaX74ZvjQ1sYA1rCLCCD7CEBo0eOzQQpgPFDO1Yzb0KewAGx3ZHvofxs+ke7iFAGCzmyNAnENvtJJFpS+j8cU+m72FITw8WduCGfThYZ/quBrITDUQbQutzHTm0pwcBBZPlysI2+xIp2MtqrJAy4xvtrSI0fiwJhyrapzOPmzBDrmympoNazCgfY9WCVeW1QAtc0oz2jPUQaQR2g92ZIcgpzYOqcgcrCIWnTFbzgIPsNw+OyjEP8qUMCwrW/wPZ2mq+jvKj/AAAAABJRU5ErkJggg==",
      "f04n2c08.png", false, 32, 32, "82a6745345706f5c5a662a94fdd0ea29");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAABcUlEQVR4nHVRsU4CQRDd5Y5kj2oXtDi7u1IbNFRWiom1iYk/IMYfMIpfQPwCPf0CobAzAQmxMkaOxMLCgl3srNzdCtaCrLNgMEGdu8zt7cy8eW/GQ3PmT7xXpsj2n2e3C3snYAeb7oxdfDdmkKFEF3IycLEVRwwsH5dcFGDKS2EYBAFCJvvmQMt5xjCUIBvJe5dzmDTaaa+fdlvJcRFAF/fjmDKIKCvE000GrTIG/wxezFgBQEvLYY5ARmCIRurB9xiUOTaIKipZEfnVqzaX0lopebdZ28DZo7Vo2lYrJXkzM+VPvx2w8NbDfECAJjLQWXF/fh4oe9pIBWAC6kfaOt/5DWqt1giUIefh7OGVkGFMjEF2ZN5lzx8rJ9yhaXgUzGz7rJFy4Mp52rqo/CVfKqu00nbyGYB8W8gR+kkMtB2KWzfTTkQpU473oD8lu11NLuv166RW+W9R446QQgzEZLmeqxm+kpGSL48/2x/fzdR/AW8Fs1uE53SkAAAAAElFTkSuQmCC",
      "f99n0g04.png", false, 32, 32, "207c771481d8e0786644d1d3ccdc1253");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAIi4vcVJsAAAARBJREFUeJy11U0KwjAQBeBXrFpFUCjFXcFTNJfwtF4itxBcCCKCpeJfbXUhQTMTE0N0FoEZ6Os3pbTR/Q6tokjvpdT7otD7uGn0wXwOa9EbsIAsswdQMQsYjwMFo1GgYDgMFAwGnoLbTR/0+4GCJAkU9HqBAtcKTkG36ynwXeH/AlcAE9T1jwVxrPdt6ymgr/Ll4hDQACqgAU4BDaDFBNerX8DXgtkMWC7VVAj1eWUCumOnAwB5vlqpExBCSnUygXmF9fq9k9IiOJ9NAgCYTjcbtYBFQFdQzyBNdzsYiglOJ5NgMgHSFNjvhZDyfQUmOB5NAVWl+udlrx8cExwOpoDPxQRl6RfABNttoGCxCBM8AHUVjIYrRN23AAAAAElFTkSuQmCC",
      "g03n0g16.png", false, 32, 32, "33bca103ab06d5288fc5a40d52b46648");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAAIi4vcVJsAAAASlJREFUeJzllk2KwkAQRqsx/kQEERR3LrzEeAiP6yH6FC4EFzKCGGRGZ0bjwojJJI/kw7iyVpVKpV+/LgLtYisOZ/DGu+L6R3E5OANgPi+uL6DfgIuA0YhWggBhBPT7IoAM/qC/1xMBqkG3KwLUGYShCFANOh0RoM6g3RYBqoEMUA1aLRFABrUBVINmUwS8n0EQiAAy+IV+NLhAXTUIyeAH6qoBHhEByID6Xz4DGfCMwXTqzGy5zK4xMzMzf38kg1LAZOJWqzidJKv7bEIGJwA0GkmyXsf5ovnKBgT4N4Px2G02qU1WNzgC4LFZs+HQbbd0Q7sHGXyXAQYDd2OY2W4XJ1vOHxEZfJUBoij7qc8ltyCDAwBq+w/20J+eQaUgg8+6AGRAt+W6DK5anlkjB1vfagAAAABJRU5ErkJggg==",
      "g03n2c08.png", false, 32, 32, "5ab30d7747a459c1051650a1351a4519");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAAIi4vcVJsAAAAB5QTFRFAAAAAP///wD/AMjIra0A3d0A//////8A/93//63/MbogiAAAAGNJREFUeJxjKAcCJSAwBgJBIGBAFmAAAfqoCAWCmUAAV4EsQEcVLkDQAQRwFcgClKlg6DA2YEZS0dDBYcxsgGIGB1wFCKSlJaTBVUAE2MACCBUJDGzMQC1IKtLS4O5AFiBTBQBS03C95h21qwAAAABJRU5ErkJggg==",
      "g03n3p04.png", false, 32, 32, "cab490ee86d478d165f2a516345d0ff0");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAK/INwWK6QAAASJJREFUeJytlDkOwjAQRSciLAEkmjQpUtJwAE6AT+0bcAAaCgoKmhQgZWEJCUVkhZkxGVnGhaX5Up6fvxQHbQtoBQGetcbzbofnsK5xsN3C4KIHMEAcDwOocfh+42C18jRYLh0NKGCxcDR4vXAwn3saRJGjAQXMZp4G06mjAe1AAjADV4DYwWTiaSABxA6cDZ7PPxuMx3huGkcD+i/c74IBBVADChANKIAu0SAMhwHM4PGwG2w2AIeDSZUyzyszsAPW6+PR7ABKaW120aC7wun0PX0/7cyAttx3kKbnc59351sMqsoOSJLLhX9uMShLHHQdxDFAkgBkmVJaK9XXyAyKwmZwvZpZa6GDPLdf4ddiBrcbDkajYQAzyDJPg/3eDUANPik0iSilDmOAAAAAAElFTkSuQmCC",
      "g04n0g16.png", false, 32, 32, "e38a1551172886575b1d91af694ecfde");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAAK/INwWK6QAAATBJREFUeJzVlj1uwkAQRr+VzJ+MRAMFBS0FB+AG8aF9hByAgoKGgoJIKFIifpLIFBiwg1+cT3GTKaz1eHfevJ3GIVN1BMGXNFTnn6rT0ScA5vPq/DPsF3ARMBxSJQgQRsBgYAJcg37fBLgGcWwCXINezwS4BjaADD5gf7drAlwDG+DOoNMxAWRAV9RumwB3Bv/fwAa4Bq2WCSCDE+xHgy/IuwYxGRDANcArOkCeDGwABRkcmwL8xWA2C5IWi3KNRJKUXl9dgyjKF9NpWC6z4iKvnpYXZEAzuwFWq+wxeW/8FmRAgG8zmEzCev3QZFIgkcEeAPdmpfE4bDY/Vhcb1AJGo3BhSNpus7xucmWobgbvdYDdrnw0LTyLQQZvdYDfBhm8NgUgg5emAGRAf8tNGZwBkU1XhkiDotcAAAAASUVORK5CYII=",
      "g04n2c08.png", false, 32, 32, "daa8561d65a6598e69d6ddb060d802b7");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAAK/INwWK6QAAAB5QTFRFAAAAAP///9T/1NQA/wD/ALq6//////8A/5v/m5sAIugsggAAAGhJREFUeJy9zsEJgDAQRNEhsLnbgaQFW7CAXOae07ZgC7Zgt04IhPWq4D8Oj2VxqF1RLQpxQO8fsalTTRGHH8WlipoiDt8ECqsFsZZEq48biaxD9NybkzbEGLILBNGQDYjCff4Rh5fiBou1fg11pxGVAAAAAElFTkSuQmCC",
      "g04n3p04.png", false, 32, 32, "0f81d4307736954402890d7203244bd0");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAANbY1E9YMgAAAQpJREFUeJyt1b0KgzAQB/AI0X4gSMcOzu3SxTdpHvQexdk+QRdBCqK2VewgQXOnCUeaQS8B//nlQA3GURgjCMw5gDm/3825/H7NhctFWAfeQPa9uXA62QOwmAiShCnAAXHsKTgemQLcA1eAU3A4MAWfDy/AKdjvmQJuABHgI+x2ngJXwP8F3ACnIIo8BThgGJgC/DK1rUPwftsFOIAIXAF4OAVhaA9gCLIsz3WtlP68EkHXrQtut7lWCkBfiWAroCiuV10DWAS4y8sm6toqwAHLJq41lAiaBi3I6X4+C5Gmz6dSAMsjEAEO0LuW5XSfHpt/cERQ19tHWBtE8HrxAoigqtCCZAoeDz/BD+1fhGYCQbPgAAAAAElFTkSuQmCC",
      "g05n0g16.png", false, 32, 32, "5f67c34aadb2f3a602fea6d4ad14fa6d");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAANbY1E9YMgAAARVJREFUeJzlljkOwjAQRcdS2EJEQUdDiWiochI4aG5Cj7gADaJA7ItCwSJC8iR/cMdUzvjHL8+xorjcqssZzGSuuj+ubkdnAAwG1f055A24COh2aSUoEI4ukO90RIBqkCQigAwI0G6LANUgjkWAatBqiQDVQAaoBs2mCCCDE+QbDRHwfwYyQDWo10UAGQTbomAGV+iTwRHyCQH20A9mQADVoFaDCSoyIECwU3SA/IdBmrrptLjGxMzMsuflLwajkfvo2OS59Gvwi8Fslg+HruCUeRvQoSi/5ELH3+BLQLnIYOcB6PWcmfX7brHIH49c3iIy2HoAlsu3u7PS4F5ksPEAeBUZrCEfRSKADFaQD2ZAf8uhvkU3ajlNmZwVLFcAAAAASUVORK5CYII=",
      "g05n2c08.png", false, 32, 32, "30cda048c618598b39f96c40141851fa");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAANbY1E9YMgAAAB5QTFRFAAAAAP//zMwA/8z/AK6u/wD/i4sA//////8A/4v/c+IkkgAAAFtJREFUeJxj6ACCUCBwAQJBIGBAFmAAAfqoUAKCmUAAV4EsQEcVaUBgDARwFcgClKkwMHZxYEFWwWDswuKAQwUIlJcXlMNVIAsgqWBgZwFqQVJRXg53B7IAmSoA1Ah4O0rtoFUAAAAASUVORK5CYII=",
      "g05n3p04.png", false, 32, 32, "2be19a2ad1bdba9734899e453a27625b");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAARFwiTtYVgAAAPhJREFUeJzNlbsNhDAMhg1C4SVomYABqFiASchcmYkxEC2Ip4S4AuVEbCAXpbm/SUzx+cMC2TkOUOI4ai2EWte1Wnvbpj7IMngNbkAAafoOwMYEkCSWBnFsaRBFlgY6gNYgDC0NdACtQRAYGqyrGeAPDUwBWgPftzTQAYwN9t3QAP/O02RpgAHEYFneATjEwBSADdwFhX1TVYwxBgDA+dVAzaNBWd7baGdw9gRomqKQ92vIDOb5HoDvnJ8bghj8BuBcIrCBO6HIEeY5QJ4zxjmAEELIHXWgePhDkV3b9jzlapMnmcE4Pr/CXcgMhsEMQAz63tKg6+wMPgLFodTQLHMsAAAAAElFTkSuQmCC",
      "g07n0g16.png", false, 32, 32, "cdd82be241cbfadbceffc98d967dfe30");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAARFwiTtYVgAAAQtJREFUeJzVlkEKgzAQRSciiaK49QS5gCsv4EnqsXImryHdlhZtKdhFK1U6HzM0YDvMIowhz+dXUE3ElyJwxSl+fuDH8Q0AypKfH8F+AlwIKAohAAhDQJ6jk0BJDbJMCJAaiAFSgzQVAqQGYgAyuIYC7GaAAEkiBCAD9IjEgJ8zMEYIkL5F/28AAXcwlxoUUgAyGMF+aHAB890yQAZaCwHAIBqJ2DZm1U2jnotXtZwB114Gda22nVAGgweg66aqUhsAlAECfIbMxN4SuXn9jQE/adcMYBANRGxr/W5rFRFZq7Sez3XzuUsDrmP03Szvt+8X/o74NcrAB+BVKINzKAAyOIUCIAP0MxvK4AEWgFoVP+GhCgAAAABJRU5ErkJggg==",
      "g07n2c08.png", false, 32, 32, "0dcc8b7a828dd05df802b636673ed0ab");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAARFwiTtYVgAAAB5QTFRFAAAAAP//AJycv78A/3b/dnYA/wD///////8A/7//TpdUbAAAAFxJREFUeJxj6ACCNCBQAgJBIGBAFmAAAfqoMAYCFyCAq0AWoKOKUCCYCQRwFcgClKmYMFOJCUUFA1gAuwoQKC8vKFdiYoKogAswMCCrYGBnUkJRUV4OdweyAJkqACOga73pcj3PAAAAAElFTkSuQmCC",
      "g07n3p04.png", false, 32, 32, "8f1a3f91ca328ca507273e80324087e9");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAYagMeiWXwAAAL1JREFUeJztlTEOhCAQRT8bCyw9hlzD0tIDIQfyCF7D0ktYSGLBFoTFmUQC2WRjsv4CZqaYefwQEM6BSAiaa03zcaR5ZS0tSImk+IDiBpy42ndaqOtfEzwe3MGDrwlKTfxHD46jkKBp0g2KPdi2QoI73YNhmKYQGxOe12wP+j7Gxmgd1myCee66sx/G+J0TvCyT/Ajwa2QAAMeUNDHG8XvhBKJtaSEYpxQALItS/vyhSfbPtK7n2dcEz6sMvAHqCJi/5fyWiAAAAABJRU5ErkJggg==",
      "g10n0g16.png", false, 32, 32, "75b64641f0a3c0899ae3b466fcb97c06");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAAYagMeiWXwAAANRJREFUeJztljEKhDAURP8HC+32HpYewdbS+/zNfbyCx7D1FAu7xUK2cA0aM0j4ARWcKhlNJsNTkS2FxQSuCIf9Z9jO3iAgz8P+B9xPIDdDC6IDQOHoAKhUDaBQA8SgKCIDDmtwM3C6GezqsAbJIF+HwRf4sQ0eKOAF/GQMUMB1GCApG7Qtd916D0NERDJPNQ2ahv1IM2/tBpqvad/buuYAdrMY6xn4znR2l6F/inxH1lNNg7JkIqoqHgb7P7hsIGsYjONitWwGk87+HuzrdA1S/VX8ANStTVTe34+eAAAAAElFTkSuQmCC",
      "g10n2c08.png", false, 32, 32, "69926b0e52c1371c81ca49d7cd0cf2b1");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAAYagMeiWXwAAAB5QTFRFAAAAqakAAP///6r/VFQA/wD/AH9///////8A/1X/7g7bWgAAAGNJREFUeJxj6ACCUCBIAwIlIGBAFmAAAfqoEASCmUAAV4EsQEcVLkBgDARwFcgClKkwME5LYENWwWCcxpaApoKNAaICBMrLC8rTGKAq4AJALUgqGNjZgIYiqSgvh7sDWYBMFQBG4oXJmToRDgAAAABJRU5ErkJggg==",
      "g10n3p04.png", false, 32, 32, "832e5401524ab7238a6eccd5d852b8ef");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAA9CQ+FSITwAAATZJREFUeJyllMsNwjAMQN0SQHwixGeNbsSBAToHXQDuHLoCi5A1CgjxhxY4RFGxLRJZ+FDXVvLy6laN3m9AEUW4ThJcbza4VkUBoqAHqO1WBqDGarfznxA02O9lAGYgBTCDw+FPAwqIY6HB8YgbjYbQQApgBqfTnwbnM2koocHlghvNptDgepUBmAEFtFpCg9sNN9ptocH97geUZcCAAgYDXNPXzAweD9zodPwAZvB8+gE0mAEFdLt+ADOgQ+r1bF6tAGYze29M/WtlBlWFG/0+AMBikabuGjCgAK1dzrK6qoMaxBUJrbXWGiDL5vNvgDHOAId6vTDRfQfL5XdljJsEmwF9Jrslz6dTgDy325KkHiSbAW0Mhzav1za7+bscNHCAXxE0GI38AGZAF4zHQgO6YDKRGXwAuz+aGCA4FKQAAAAASUVORK5CYII=",
      "g25n0g16.png", false, 32, 32, "698f892bb4453fdd325ae414dc82b34f");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAABGdBTUEAA9CQ+FSITwAAAUxJREFUeJytlcttAjEQQMeLgSAQ4SPK2I44UAB1oC1gC+BACVQSykB8hIgSSIJzwFq8xE/KSB75MIxn5+lpLGGcxMMI3OQmXn+Ll+0WAOoArt2lAoCw3acCkMGB+uED2hkaHFMByIAAWRav3whABiclAIMMkgHI4Az9jYYSQAYEsFYJIIP3VAAySAYggw/ob7WUADIgQLOpBJDBJ/S320oAGVy0gB+okwEBXl/ggl4FGXxBf6ejBJDBVQugIINv6O92lQAy+A9guTQiMpsFMzYief0DrUGv55OyNPO5C5N4kAG9ugpwz4vCPBWfAwyym0j09Pv+iEhRmMXCififj9jUDWLH0l9gOKss3d+in14tg3ZAgMHAJ6uVm07NPXlMzOvThXdAW6sAIrJeB13h4wlzMiDAcAgXFGRA/aOREkAG1D8eKwFag8lECQCDX4gtYR8yuXeNAAAAAElFTkSuQmCC",
      "g25n2c08.png", false, 32, 32, "a64f63bacd6a0edec500179d538ede01");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAA9CQ+FSITwAAAB5QTFRFAAAAAC0tAP//EBAA/1z//xD//wD/XFwA//////8AUlHX5QAAAGRJREFUeJy9zrENgDAMRFFbSpHWK7ACC7jICqzACrSUXoFtc1EkczVI+eX5FZYHncjQhoQHGa0RFzpQCh4Wih01lIKHf0KaKQtvZQyvcC8pRnHXaqpTzCEqziRCQo0Fyj94+Cg6NXRmxzu0UNgAAAAASUVORK5CYII=",
      "g25n3p04.png", false, 32, 32, "c2b4d9eb0587bc254212b05beb972578");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAYagMeiWXwAAAF5JREFUeJzV0jEKwDAMQ1E5W+9/xtygk8AoezLVKgSj2Y8/OICnuFcTE2OgOoJgHQiZAN2C9kDKBOgW3AZCJkC3oD2QMgG6BbeBkAnQLWgPpExgP28H7E/0GTjPfwAW2EvYX64rn9cAAAAASUVORK5CYII=",
      "oi1n0g16.png", false, 32, 32, "a14e204bbf905586d3763f3cc5dcb2f3");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAACsiDHgAAAABGdBTUEAAYagMeiWXwAAAOVJREFUeJzVlsEKgzAQRKfgQX/Lfrf9rfaWHgYDkoYmZpPMehiGReQ91qCPEEIAPi/gmu9kcnN+GD0nM1/O4vNad7cC6850KHCiM5fz7fJwXdEBYPOygV/o7PICeXSmsMA/dKbkGShD51xsAzXo7DIC9ehMAYG76MypZ6ANnfNJG7BAZx8uYIfOHChgjR4F+MfuDx0AtmfnDfREZ+8m0B+9m8Ao9Chg9x0Yi877jTYwA529WWAeerPAbPQoUH8GNNA5r9yAEjp7sYAeerGAKnoUyJ8BbXTOMxvwgM6eCPhBTwS8oTO/5kL+Xge7xOwAAAAASUVORK5CYII=",
      "oi1n2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAYagMeiWXwAAAEBJREFUeJzV0jEKwDAMQ1E5W+9/xtygk8AoezLVKgSj2Y8/OICnuFcTE2OgOoJgHQiZAN2C9kDKBOgW3AZCJkC3oD2QMjqwwDMAAAAeSURBVAG6BbeBkAnQLWgPpExgP28H7E/0GTjPfwAW2EvYX7J6X30AAAAASUVORK5CYII=",
      "oi2n0g16.png", false, 32, 32, "a14e204bbf905586d3763f3cc5dcb2f3");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAACsiDHgAAAABGdBTUEAAYagMeiWXwAAAIBJREFUeJzVlsEKgzAQRKfgQX/Lfrf9rfaWHgYDkoYmZpPMehiGReQ91qCPEEIAPi/gmu9kcnN+GD0nM1/O4vNad7cC6850KHCiM5fz7fJwXdEBYPOygV/o7PICeXSmsMA/dKbkGShD51xsAzXo7DIC9ehMAYG76MypZ6ANnfNJG7BAZx+ZiKBzAAAAZUlEQVQuYIfOHChgjR4F+MfuDx0AtmfnDfREZ+8m0B+9m8Ao9Chg9x0Yi877jTYwA529WWAeerPAbPQoUH8GNNA5r9yAEjp7sYAeerGAKnoUyJ8BbXTOMxvwgM6eCPhBTwS8oTO/5kL+Xk13nmIAAAAASUVORK5CYII=",
      "oi2n2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAYagMeiWXwAAAB9JREFUeJzV0jEKwDAMQ1E5W+9/xtygk8AoezLVKgSj2Y8/OIdtk98AAAAfSURBVICnuFcTE2OgOoJgHQiZAN2C9kDKBOgW3AZCJkC3oD3Oo8vsAAAAAklEQVSQMsVtZiAAAAAeSURBVAG6BbeBkAnQLWgPpExgP28H7E/0GTjPfwAW2EvYX7J6X30AAAAASUVORK5CYII=",
      "oi4n0g16.png", false, 32, 32, "a14e204bbf905586d3763f3cc5dcb2f3");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAACsiDHgAAAABGdBTUEAAYagMeiWXwAAAGNJREFUeJzVlsEKgzAQRKfgQX/Lfrf9rfaWHgYDkoYmZpPMehiGReQ91qCPEEIAPi/gmu9kcnN+GD0nM1/O4vNad7cC6850KHCiM5fz7fJwXdEBYPOygV/o7PICeXSmsMA/dKbkGShDblRaWAAAAB1JREFU51xsAzXo7DIC9ehMAYG76MypZ6ANnfNJG7BAZx9l6MXmAAAAY0lEQVQuYIfOHChgjR4F+MfuDx0AtmfnDfREZ+8m0B+9m8Ao9Chg9x0Yi877jTYwA529WWAeerPAbPQoUH8GNNA5r9yAEjp7sYAeerGAKnoUyJ8BbXTOMxvwgM6eCPhBTwS8oTO/5kIg4uIpAAAAAklEQVT+XnoXDXoAAAAASUVORK5CYII=",
      "oi4n2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAYagMeiWXwAAAAFJREFUeHbmhOYAAAABSURBVJzRgaKHAAAAAUlEQVTV3oFbswAAAAFJREFU0kDlzhAAAAABSURBVDF55n3SAAAAAUlEQVQKyO2U9gAAAAFJREFUwLNcv1gAAAABSURBVDAO4U1EAAAAAUlEQVQMIY4xwwAAAAFJREFUQ8ftbcIAAAABSURBVFE0VByKAAAAAUlEQVQ5dz314AAAAAFJREFUW9SB9ZQAAAABSURBVO8YjYIBAAAAAUlEQVR/6IIRRQAAAAFJREFUxlo/Gm0AAAABSURBVNynXeMXAAAAAUlEQVSg/u7eAAAAAAFJREFUk0E+vxYAAAABSURBVMCzXL9YAAAAAUlEQVQoHY3VEgAAAAFJREFUe+/v1VwAAAABSURBVDLg7yxoAAAAAUlEQVTV3oFbswAAAAFJREFUKvODtD4AAAABSURBVAQvVbnxAAAAAUlEQVSjZ+ePugAAAAFJREFU2dc3F5gAAAABSURBVI9VP+NZAAAAAUlEQVQ/nl5Q1QAAAAFJREFUOAA6xXYAAAABSURBVIDFgP7IAAAAAUlEQVSnYIpLowAAAAFJREFUuO2CRlYAAAABSURBVFfdN7m/AAAAAUlEQVQTrIY8NgAAAAFJREFUE6yGPDYAAAABSURBVGP8g00KAAAAAUlEQVSg/u7eAAAAAAFJREFUOu40pFoAAAABSURBVIIrjp/kAAAAAUlEQVRgZYocsAAAAAFJREFUHUs+ETEAAAABSURBVAgm4/XaAAAAAUlEQVSZoetWCAAAAAFJREFUACg4fegAAAABSURBVN3QWtOBAAAAAUlEQVSCK46f5AAAAAFJREFU9nzmKsEAAAABSURBVEBe5Dx4AAAAAUlEQVTKU4lWRgAAAAFJREFUBC9VufEAAAABSURBVOiG6ReiAAAAAUlEQVQW3OzIuQAAAAFJREFU3Kdd4xcAAAABSURBVAbBW9jdAAAAAUlEQVRCsOpdVAAAAAFJREFUJvo1+BUAAAABSURBVEBe5Dx4AAAAAUlEQVS3fT1bxwAAAAFJREFUoP7u3gAAAAABSURBVD1wUDH5AAAAAUlEQVSQ2DfurAAAAAFJREFUMuDvLGgAAAABSURBVAFfP01+AAAAAUlEQVS6A4wnegAAAAFJREFUBVhSiWcAAAABSURBVLd9PVvHAAAAAUlEQVSBsofOXgAAAAFJREFUkNg37qwAAAABSURBVAlR5MVMAAAAAUlEQVTQruuvPAAAAAFJREFULW3nIZ0AAAABSURBVGhrUZSCAAAAAUlEQVQPuIdgeQAAAAFJREFUpPmDGhkAAAABSURBVExXUnBTAAAAAUlEQVRgZYocsAAAAAFJREFUP55eUNUAAAABSURBVG/1NQEhAAAAAUlEQVQHtlzoSwAAAAFJREFU7IGE07sAAAABSURBVE/OWyHpAAAAAUlEQVT0kuhL7QAAAAFJREFUGUxT1SgAAAABSURBVDgAOsV2AAAAAUlEQVTPI+OiyQAAAAFJREFUf+iCEUUAAAABSURBVAAoOH3oAAAAAUlEQVQW3OzIuQAAAAFJREFU2KAwJw4AAAABSURBVEvJNuXwAAAAAUlEQVTYoDAnDgAAAAFJREFUX9PsMY0AAAAASUVORK5CYII=",
      "oi9n0g16.png", false, 32, 32, "a14e204bbf905586d3763f3cc5dcb2f3");
  testPngSuiteImage("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",
      "oi9n2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("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QOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDCAASABQ+gAAgAUDhAwgAEgAUPoAAIAFA4QMIABIAFD6AACABQOEDrNu2zb4Gs2n9C98FlZObRxMyAAAAAElFTkSuQmCC",
      "PngSuite.png", false, 256, 256, "183c2504778cb2b6384dbbad46fa2a2a");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAACsiDHgAAAABGdBTUEAAYagMeiWXwAAAohQTFRFAAAAAAAzAABmAACZAADMAAD/ADMAADMzADNmADOZADPMADP/AGYAAGYzAGZmAGaZAGbMAGb/AJkAAJkzAJlmAJmZAJnMAJn/AMwAAMwzAMxmAMyZAMzMAMz/AP8AAP8zAP9mAP+ZAP/MAP//MwAAMwAzMwBmMwCZMwDMMwD/MzMAMzMzMzNmMzOZMzPMMzP/M2YAM2YzM2ZmM2aZM2bMM2b/M5kAM5kzM5lmM5mZM5nMM5n/M8wAM8wzM8xmM8yZM8zMM8z/M/8AM/8zM/9mM/+ZM//MM///ZgAAZgAzZgBmZgCZZgDMZgD/ZjMAZjMzZjNmZjOZZjPMZjP/ZmYAZmYzZmZmZmaZZmbMZmb/ZpkAZpkzZplmZpmZZpnMZpn/ZswAZswzZsxmZsyZZszMZsz/Zv8AZv8zZv9mZv+ZZv/MZv//mQAAmQAzmQBmmQCZmQDMmQD/mTMAmTMzmTNmmTOZmTPMmTP/mWYAmWYzmWZmmWaZmWbMmWb/mZkAmZkzmZlmmZmZmZnMmZn/mcwAmcwzmcxmmcyZmczMmcz/mf8Amf8zmf9mmf+Zmf/Mmf//zAAAzAAzzABmzACZzADMzAD/zDMAzDMzzDNmzDOZzDPMzDP/zGYAzGYzzGZmzGaZzGbMzGb/zJkAzJkzzJlmzJmZzJnMzJn/zMwAzMwzzMxmzMyZzMzMzMz/zP8AzP8zzP9mzP+ZzP/MzP///wAA/wAz/wBm/wCZ/wDM/wD//zMA/zMz/zNm/zOZ/zPM/zP//2YA/2Yz/2Zm/2aZ/2bM/2b//5kA/5kz/5lm/5mZ/5nM/5n//8wA/8wz/8xm/8yZ/8zM/8z///8A//8z//9m//+Z///M////Y7C7UQAAAOVJREFUeJzVlsEKgzAQRKfgQX/Lfrf9rfaWHgYDkoYmZpPMehiGReQ91qCPEEIAPi/gmu9kcnN+GD0nM1/O4vNad7cC6850KHCiM5fz7fJwXdEBYPOygV/o7PICeXSmsMA/dKbkGShD51xsAzXo7DIC9ehMAYG76MypZ6ANnfNJG7BAZx8uYIfOHChgjR4F+MfuDx0AtmfnDfREZ+8m0B+9m8Ao9Chg9x0Yi877jTYwA529WWAeerPAbPQoUH8GNNA5r9yAEjp7sYAeerGAKnoUyJ8BbXTOMxvwgM6eCPhBTwS8oTO/5kL+Xge7xOwAAAAASUVORK5CYII=",
      "pp0n2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABGdBTUEAAYagMeiWXwAAAohQTFRFAAAAAAAzAABmAACZAADMAAD/ADMAADMzADNmADOZADPMADP/AGYAAGYzAGZmAGaZAGbMAGb/AJkAAJkzAJlmAJmZAJnMAJn/AMwAAMwzAMxmAMyZAMzMAMz/AP8AAP8zAP9mAP+ZAP/MAP//MwAAMwAzMwBmMwCZMwDMMwD/MzMAMzMzMzNmMzOZMzPMMzP/M2YAM2YzM2ZmM2aZM2bMM2b/M5kAM5kzM5lmM5mZM5nMM5n/M8wAM8wzM8xmM8yZM8zMM8z/M/8AM/8zM/9mM/+ZM//MM///ZgAAZgAzZgBmZgCZZgDMZgD/ZjMAZjMzZjNmZjOZZjPMZjP/ZmYAZmYzZmZmZmaZZmbMZmb/ZpkAZpkzZplmZpmZZpnMZpn/ZswAZswzZsxmZsyZZszMZsz/Zv8AZv8zZv9mZv+ZZv/MZv//mQAAmQAzmQBmmQCZmQDMmQD/mTMAmTMzmTNmmTOZmTPMmTP/mWYAmWYzmWZmmWaZmWbMmWb/mZkAmZkzmZlmmZmZmZnMmZn/mcwAmcwzmcxmmcyZmczMmcz/mf8Amf8zmf9mmf+Zmf/Mmf//zAAAzAAzzABmzACZzADMzAD/zDMAzDMzzDNmzDOZzDPMzDP/zGYAzGYzzGZmzGaZzGbMzGb/zJkAzJkzzJlmzJmZzJnMzJn/zMwAzMwzzMxmzMyZzMzMzMz/zP8AzP8zzP9mzP+ZzP/MzP///wAA/wAz/wBm/wCZ/wDM/wD//zMA/zMz/zNm/zOZ/zPM/zP//2YA/2Yz/2Zm/2aZ/2bM/2b//5kA/5kz/5lm/5mZ/5nM/5n//8wA/8wz/8xm/8yZ/8zM/8z///8A//8z//9m//+Z///M////Y7C7UQAAAFVJREFUeJzt0DEKwDAMQ1EVPCT3v6BvogzO1KVLQcsfNBgMeuixLcnrlf1x//WzS2pJjgUAAAADyPWrwgMAAABgAMF+VXgAAAAAXIAdS3U3AAAAooADG8P2VRMVDwMAAAAASUVORK5CYII=",
      "pp0n6a08.png", false, 32, 32, "de9a6b2025046b20b3a408990a2b7e71");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAAABGdBTUEAAYagMeiWXwAABRpzUExUc2l4LWN1YmUACAAAAP8AAAAAM/8AAAAAZv8AAAAAmf8AAAAAzP8AAAAA//8AAAAzAP8AAAAzM/8AAAAzZv8AAAAzmf8AAAAzzP8AAAAz//8AAABmAP8AAABmM/8AAABmZv8AAABmmf8AAABmzP8AAABm//8AAACZAP8AAACZM/8AAACZZv8AAACZmf8AAACZzP8AAACZ//8AAADMAP8AAADMM/8AAADMZv8AAADMmf8AAADMzP8AAADM//8AAAD/AP8AAAD/M/8AAAD/Zv8AAAD/mf8AAAD/zP8AAAD///8AADMAAP8AADMAM/8AADMAZv8AADMAmf8AADMAzP8AADMA//8AADMzAP8AADMzM/8AADMzZv8AADMzmf8AADMzzP8AADMz//8AADNmAP8AADNmM/8AADNmZv8AADNmmf8AADNmzP8AADNm//8AADOZAP8AADOZM/8AADOZZv8AADOZmf8AADOZzP8AADOZ//8AADPMAP8AADPMM/8AADPMZv8AADPMmf8AADPMzP8AADPM//8AADP/AP8AADP/M/8AADP/Zv8AADP/mf8AADP/zP8AADP///8AAGYAAP8AAGYAM/8AAGYAZv8AAGYAmf8AAGYAzP8AAGYA//8AAGYzAP8AAGYzM/8AAGYzZv8AAGYzmf8AAGYzzP8AAGYz//8AAGZmAP8AAGZmM/8AAGZmZv8AAGZmmf8AAGZmzP8AAGZm//8AAGaZAP8AAGaZM/8AAGaZZv8AAGaZmf8AAGaZzP8AAGaZ//8AAGbMAP8AAGbMM/8AAGbMZv8AAGbMmf8AAGbMzP8AAGbM//8AAGb/AP8AAGb/M/8AAGb/Zv8AAGb/mf8AAGb/zP8AAGb///8AAJkAAP8AAJkAM/8AAJkAZv8AAJkAmf8AAJkAzP8AAJkA//8AAJkzAP8AAJkzM/8AAJkzZv8AAJkzmf8AAJkzzP8AAJkz//8AAJlmAP8AAJlmM/8AAJlmZv8AAJlmmf8AAJlmzP8AAJlm//8AAJmZAP8AAJmZM/8AAJmZZv8AAJmZmf8AAJmZzP8AAJmZ//8AAJnMAP8AAJnMM/8AAJnMZv8AAJnMmf8AAJnMzP8AAJnM//8AAJn/AP8AAJn/M/8AAJn/Zv8AAJn/mf8AAJn/zP8AAJn///8AAMwAAP8AAMwAM/8AAMwAZv8AAMwAmf8AAMwAzP8AAMwA//8AAMwzAP8AAMwzM/8AAMwzZv8AAMwzmf8AAMwzzP8AAMwz//8AAMxmAP8AAMxmM/8AAMxmZv8AAMxmmf8AAMxmzP8AAMxm//8AAMyZAP8AAMyZM/8AAMyZZv8AAMyZmf8AAMyZzP8AAMyZ//8AAMzMAP8AAMzMM/8AAMzMZv8AAMzMmf8AAMzMzP8AAMzM//8AAMz/AP8AAMz/M/8AAMz/Zv8AAMz/mf8AAMz/zP8AAMz///8AAP8AAP8AAP8AM/8AAP8AZv8AAP8Amf8AAP8AzP8AAP8A//8AAP8zAP8AAP8zM/8AAP8zZv8AAP8zmf8AAP8zzP8AAP8z//8AAP9mAP8AAP9mM/8AAP9mZv8AAP9mmf8AAP9mzP8AAP9m//8AAP+ZAP8AAP+ZM/8AAP+ZZv8AAP+Zmf8AAP+ZzP8AAP+Z//8AAP/MAP8AAP/MM/8AAP/MZv8AAP/Mmf8AAP/MzP8AAP/M//8AAP//AP8AAP//M/8AAP//Zv8AAP//mf8AAP//zP8AAP////8AACL/aC4AAABBSURBVHicY2RgJAAUCMizDAUFjA8IKfj3Hz9geTAcFDDKEZBnZKJ5XAwGBYyP8Mr+/8/4h+ZxMRgUMMrglWVkBABQ5f5xNeLYWQAAAABJRU5ErkJggg==",
      "ps1n0g08.png", false, 32, 32, "f6470f9f6296c5109e2bd730fe203773");
  testPngSuiteImage("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",
      "ps1n2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAAABGdBTUEAAYagMeiWXwAACHpzUExUc2l4LWN1YmUAEAAAAAAAAAD/AAAAAAAAADMA/wAAAAAAAABmAP8AAAAAAAAAmQD/AAAAAAAAAMwA/wAAAAAAAAD/AP8AAAAAADMAAAD/AAAAAAAzADMA/wAAAAAAMwBmAP8AAAAAADMAmQD/AAAAAAAzAMwA/wAAAAAAMwD/AP8AAAAAAGYAAAD/AAAAAABmADMA/wAAAAAAZgBmAP8AAAAAAGYAmQD/AAAAAABmAMwA/wAAAAAAZgD/AP8AAAAAAJkAAAD/AAAAAACZADMA/wAAAAAAmQBmAP8AAAAAAJkAmQD/AAAAAACZAMwA/wAAAAAAmQD/AP8AAAAAAMwAAAD/AAAAAADMADMA/wAAAAAAzABmAP8AAAAAAMwAmQD/AAAAAADMAMwA/wAAAAAAzAD/AP8AAAAAAP8AAAD/AAAAAAD/ADMA/wAAAAAA/wBmAP8AAAAAAP8AmQD/AAAAAAD/AMwA/wAAAAAA/wD/AP8AAAAzAAAAAAD/AAAAMwAAADMA/wAAADMAAABmAP8AAAAzAAAAmQD/AAAAMwAAAMwA/wAAADMAAAD/AP8AAAAzADMAAAD/AAAAMwAzADMA/wAAADMAMwBmAP8AAAAzADMAmQD/AAAAMwAzAMwA/wAAADMAMwD/AP8AAAAzAGYAAAD/AAAAMwBmADMA/wAAADMAZgBmAP8AAAAzAGYAmQD/AAAAMwBmAMwA/wAAADMAZgD/AP8AAAAzAJkAAAD/AAAAMwCZADMA/wAAADMAmQBmAP8AAAAzAJkAmQD/AAAAMwCZAMwA/wAAADMAmQD/AP8AAAAzAMwAAAD/AAAAMwDMADMA/wAAADMAzABmAP8AAAAzAMwAmQD/AAAAMwDMAMwA/wAAADMAzAD/AP8AAAAzAP8AAAD/AAAAMwD/ADMA/wAAADMA/wBmAP8AAAAzAP8AmQD/AAAAMwD/AMwA/wAAADMA/wD/AP8AAABmAAAAAAD/AAAAZgAAADMA/wAAAGYAAABmAP8AAABmAAAAmQD/AAAAZgAAAMwA/wAAAGYAAAD/AP8AAABmADMAAAD/AAAAZgAzADMA/wAAAGYAMwBmAP8AAABmADMAmQD/AAAAZgAzAMwA/wAAAGYAMwD/AP8AAABmAGYAAAD/AAAAZgBmADMA/wAAAGYAZgBmAP8AAABmAGYAmQD/AAAAZgBmAMwA/wAAAGYAZgD/AP8AAABmAJkAAAD/AAAAZgCZADMA/wAAAGYAmQBmAP8AAABmAJkAmQD/AAAAZgCZAMwA/wAAAGYAmQD/AP8AAABmAMwAAAD/AAAAZgDMADMA/wAAAGYAzABmAP8AAABmAMwAmQD/AAAAZgDMAMwA/wAAAGYAzAD/AP8AAABmAP8AAAD/AAAAZgD/ADMA/wAAAGYA/wBmAP8AAABmAP8AmQD/AAAAZgD/AMwA/wAAAGYA/wD/AP8AAACZAAAAAAD/AAAAmQAAADMA/wAAAJkAAABmAP8AAACZAAAAmQD/AAAAmQAAAMwA/wAAAJkAAAD/AP8AAACZADMAAAD/AAAAmQAzADMA/wAAAJkAMwBmAP8AAACZADMAmQD/AAAAmQAzAMwA/wAAAJkAMwD/AP8AAACZAGYAAAD/AAAAmQBmADMA/wAAAJkAZgBmAP8AAACZAGYAmQD/AAAAmQBmAMwA/wAAAJkAZgD/AP8AAACZAJkAAAD/AAAAmQCZADMA/wAAAJkAmQBmAP8AAACZAJkAmQD/AAAAmQCZAMwA/wAAAJkAmQD/AP8AAACZAMwAAAD/AAAAmQDMADMA/wAAAJkAzABmAP8AAACZAMwAmQD/AAAAmQDMAMwA/wAAAJkAzAD/AP8AAACZAP8AAAD/AAAAmQD/ADMA/wAAAJkA/wBmAP8AAACZAP8AmQD/AAAAmQD/AMwA/wAAAJkA/wD/AP8AAADMAAAAAAD/AAAAzAAAADMA/wAAAMwAAABmAP8AAADMAAAAmQD/AAAAzAAAAMwA/wAAAMwAAAD/AP8AAADMADMAAAD/AAAAzAAzADMA/wAAAMwAMwBmAP8AAADMADMAmQD/AAAAzAAzAMwA/wAAAMwAMwD/AP8AAADMAGYAAAD/AAAAzABmADMA/wAAAMwAZgBmAP8AAADMAGYAmQD/AAAAzABmAMwA/wAAAMwAZgD/AP8AAADMAJkAAAD/AAAAzACZADMA/wAAAMwAmQBmAP8AAADMAJkAmQD/AAAAzACZAMwA/wAAAMwAmQD/AP8AAADMAMwAAAD/AAAAzADMADMA/wAAAMwAzABmAP8AAADMAMwAmQD/AAAAzADMAMwA/wAAAMwAzAD/AP8AAADMAP8AAAD/AAAAzAD/ADMA/wAAAMwA/wBmAP8AAADMAP8AmQD/AAAAzAD/AMwA/wAAAMwA/wD/AP8AAAD/AAAAAAD/AAAA/wAAADMA/wAAAP8AAABmAP8AAAD/AAAAmQD/AAAA/wAAAMwA/wAAAP8AAAD/AP8AAAD/ADMAAAD/AAAA/wAzADMA/wAAAP8AMwBmAP8AAAD/ADMAmQD/AAAA/wAzAMwA/wAAAP8AMwD/AP8AAAD/AGYAAAD/AAAA/wBmADMA/wAAAP8AZgBmAP8AAAD/AGYAmQD/AAAA/wBmAMwA/wAAAP8AZgD/AP8AAAD/AJkAAAD/AAAA/wCZADMA/wAAAP8AmQBmAP8AAAD/AJkAmQD/AAAA/wCZAMwA/wAAAP8AmQD/AP8AAAD/AMwAAAD/AAAA/wDMADMA/wAAAP8AzABmAP8AAAD/AMwAmQD/AAAA/wDMAMwA/wAAAP8AzAD/AP8AAAD/AP8AAAD/AAAA/wD/ADMA/wAAAP8A/wBmAP8AAAD/AP8AmQD/AAAA/wD/AMwA/wAAAP8A/wD/AP8AAJbQi4YAAABBSURBVHicY2RgJAAUCMizDAUFjA8IKfj3Hz9geTAcFDDKEZBnZKJ5XAwGBYyP8Mr+/8/4h+ZxMRgUMMrglWVkBABQ5f5xNeLYWQAAAABJRU5ErkJggg==",
      "ps2n0g08.png", false, 32, 32, "f6470f9f6296c5109e2bd730fe203773");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAIAAACsiDHgAAAABGdBTUEAAYagMeiWXwAACHpzUExUc2l4LWN1YmUAEAAAAAAAAAD/AAAAAAAAADMA/wAAAAAAAABmAP8AAAAAAAAAmQD/AAAAAAAAAMwA/wAAAAAAAAD/AP8AAAAAADMAAAD/AAAAAAAzADMA/wAAAAAAMwBmAP8AAAAAADMAmQD/AAAAAAAzAMwA/wAAAAAAMwD/AP8AAAAAAGYAAAD/AAAAAABmADMA/wAAAAAAZgBmAP8AAAAAAGYAmQD/AAAAAABmAMwA/wAAAAAAZgD/AP8AAAAAAJkAAAD/AAAAAACZADMA/wAAAAAAmQBmAP8AAAAAAJkAmQD/AAAAAACZAMwA/wAAAAAAmQD/AP8AAAAAAMwAAAD/AAAAAADMADMA/wAAAAAAzABmAP8AAAAAAMwAmQD/AAAAAADMAMwA/wAAAAAAzAD/AP8AAAAAAP8AAAD/AAAAAAD/ADMA/wAAAAAA/wBmAP8AAAAAAP8AmQD/AAAAAAD/AMwA/wAAAAAA/wD/AP8AAAAzAAAAAAD/AAAAMwAAADMA/wAAADMAAABmAP8AAAAzAAAAmQD/AAAAMwAAAMwA/wAAADMAAAD/AP8AAAAzADMAAAD/AAAAMwAzADMA/wAAADMAMwBmAP8AAAAzADMAmQD/AAAAMwAzAMwA/wAAADMAMwD/AP8AAAAzAGYAAAD/AAAAMwBmADMA/wAAADMAZgBmAP8AAAAzAGYAmQD/AAAAMwBmAMwA/wAAADMAZgD/AP8AAAAzAJkAAAD/AAAAMwCZADMA/wAAADMAmQBmAP8AAAAzAJkAmQD/AAAAMwCZAMwA/wAAADMAmQD/AP8AAAAzAMwAAAD/AAAAMwDMADMA/wAAADMAzABmAP8AAAAzAMwAmQD/AAAAMwDMAMwA/wAAADMAzAD/AP8AAAAzAP8AAAD/AAAAMwD/ADMA/wAAADMA/wBmAP8AAAAzAP8AmQD/AAAAMwD/AMwA/wAAADMA/wD/AP8AAABmAAAAAAD/AAAAZgAAADMA/wAAAGYAAABmAP8AAABmAAAAmQD/AAAAZgAAAMwA/wAAAGYAAAD/AP8AAABmADMAAAD/AAAAZgAzADMA/wAAAGYAMwBmAP8AAABmADMAmQD/AAAAZgAzAMwA/wAAAGYAMwD/AP8AAABmAGYAAAD/AAAAZgBmADMA/wAAAGYAZgBmAP8AAABmAGYAmQD/AAAAZgBmAMwA/wAAAGYAZgD/AP8AAABmAJkAAAD/AAAAZgCZADMA/wAAAGYAmQBmAP8AAABmAJkAmQD/AAAAZgCZAMwA/wAAAGYAmQD/AP8AAABmAMwAAAD/AAAAZgDMADMA/wAAAGYAzABmAP8AAABmAMwAmQD/AAAAZgDMAMwA/wAAAGYAzAD/AP8AAABmAP8AAAD/AAAAZgD/ADMA/wAAAGYA/wBmAP8AAABmAP8AmQD/AAAAZgD/AMwA/wAAAGYA/wD/AP8AAACZAAAAAAD/AAAAmQAAADMA/wAAAJkAAABmAP8AAACZAAAAmQD/AAAAmQAAAMwA/wAAAJkAAAD/AP8AAACZADMAAAD/AAAAmQAzADMA/wAAAJkAMwBmAP8AAACZADMAmQD/AAAAmQAzAMwA/wAAAJkAMwD/AP8AAACZAGYAAAD/AAAAmQBmADMA/wAAAJkAZgBmAP8AAACZAGYAmQD/AAAAmQBmAMwA/wAAAJkAZgD/AP8AAACZAJkAAAD/AAAAmQCZADMA/wAAAJkAmQBmAP8AAACZAJkAmQD/AAAAmQCZAMwA/wAAAJkAmQD/AP8AAACZAMwAAAD/AAAAmQDMADMA/wAAAJkAzABmAP8AAACZAMwAmQD/AAAAmQDMAMwA/wAAAJkAzAD/AP8AAACZAP8AAAD/AAAAmQD/ADMA/wAAAJkA/wBmAP8AAACZAP8AmQD/AAAAmQD/AMwA/wAAAJkA/wD/AP8AAADMAAAAAAD/AAAAzAAAADMA/wAAAMwAAABmAP8AAADMAAAAmQD/AAAAzAAAAMwA/wAAAMwAAAD/AP8AAADMADMAAAD/AAAAzAAzADMA/wAAAMwAMwBmAP8AAADMADMAmQD/AAAAzAAzAMwA/wAAAMwAMwD/AP8AAADMAGYAAAD/AAAAzABmADMA/wAAAMwAZgBmAP8AAADMAGYAmQD/AAAAzABmAMwA/wAAAMwAZgD/AP8AAADMAJkAAAD/AAAAzACZADMA/wAAAMwAmQBmAP8AAADMAJkAmQD/AAAAzACZAMwA/wAAAMwAmQD/AP8AAADMAMwAAAD/AAAAzADMADMA/wAAAMwAzABmAP8AAADMAMwAmQD/AAAAzADMAMwA/wAAAMwAzAD/AP8AAADMAP8AAAD/AAAAzAD/ADMA/wAAAMwA/wBmAP8AAADMAP8AmQD/AAAAzAD/AMwA/wAAAMwA/wD/AP8AAAD/AAAAAAD/AAAA/wAAADMA/wAAAP8AAABmAP8AAAD/AAAAmQD/AAAA/wAAAMwA/wAAAP8AAAD/AP8AAAD/ADMAAAD/AAAA/wAzADMA/wAAAP8AMwBmAP8AAAD/ADMAmQD/AAAA/wAzAMwA/wAAAP8AMwD/AP8AAAD/AGYAAAD/AAAA/wBmADMA/wAAAP8AZgBmAP8AAAD/AGYAmQD/AAAA/wBmAMwA/wAAAP8AZgD/AP8AAAD/AJkAAAD/AAAA/wCZADMA/wAAAP8AmQBmAP8AAAD/AJkAmQD/AAAA/wCZAMwA/wAAAP8AmQD/AP8AAAD/AMwAAAD/AAAA/wDMADMA/wAAAP8AzABmAP8AAAD/AMwAmQD/AAAA/wDMAMwA/wAAAP8AzAD/AP8AAAD/AP8AAAD/AAAA/wD/ADMA/wAAAP8A/wBmAP8AAAD/AP8AmQD/AAAA/wD/AMwA/wAAAP8A/wD/AP8AAJbQi4YAAADlSURBVHic1ZbBCoMwEESn4EF/y363/a32lh4GA5KGJmaTzHoYhkXkPdagjxBCAD4v4JrvZHJzfhg9JzNfzuLzWne3AuvOdChwojOX8+3ycF3RAWDzsoFf6OzyAnl0prDAP3Sm5BkoQ+dcbAM16OwyAvXoTAGBu+jMqWegDZ3zSRuwQGcfLmCHzhwoYI0eBfjH7g8dALZn5w30RGfvJtAfvZvAKPQoYPcdGIvO+402MAOdvVlgHnqzwGz0KFB/BjTQOa/cgBI6e7GAHnqxgCp6FMifAW10zjMb8IDOngj4QU8EvKEzv+ZC/l4Hu8TsAAAAAElFTkSuQmCC",
      "ps2n2c16.png", false, 32, 32, "a3774d09367dd147a3539d2d2f6ca133");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAFS3GZcAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAANQTFRFAAD/injSVwAAAApJREFUeJxjYAAAAAIAAUivpHEAAAAASUVORK5CYII=",
      "s01i3p01.png", false, 1, 1, "c987217b78dd44056a9da58cf06b8c7a");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAANQTFRFAAD/injSVwAAAApJREFUeJxjYAAAAAIAAUivpHEAAAAASUVORK5CYII=",
      "s01n3p01.png", false, 1, 1, "c987217b78dd44056a9da58cf06b8c7a");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAAE/f6/xAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAANQTFRFAP//GVwvJQAAAAtJREFUeJxjYAABAAAGAAH+jGfIAAAAAElFTkSuQmCC",
      "s02i3p01.png", false, 2, 2, "e1b1f768e50f5269db92782b4ad62247");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAANQTFRFAP//GVwvJQAAAAxJREFUeJxjYGBgAAAABAAB9hc4VQAAAABJRU5ErkJggg==",
      "s02n3p01.png", false, 2, 2, "e1b1f768e50f5269db92782b4ad62247");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAMAAAADAQMAAAEb4RdqAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAZQTFRFAP8A/3cAseWlnwAAAAxJREFUeJxjYIADBwAATABB2snmHAAAAABJRU5ErkJggg==",
      "s03i3p01.png", false, 3, 3, "b05c579eb095ddac5d3b30e0329c33f4");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAMAAAADAQMAAABs5if8AAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAZQTFRFAP8A/3cAseWlnwAAAA5JREFUeJxjYGBwYGAAAADGAEE5MQxLAAAAAElFTkSuQmCC",
      "s03n3p01.png", false, 3, 3, "b05c579eb095ddac5d3b30e0329c33f4");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAQAAAAEAQMAAAHkODyrAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAZQTFRF/wB3//8AmvdDuQAAABRJREFUeJxjaGAAwQMMDgwTGD4AABmuBAG53zf2AAAAAElFTkSuQmCC",
      "s04i3p01.png", false, 4, 4, "c268bd54d984c22857d450e233766115");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAQAAAAEAQMAAACTPww9AAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAZQTFRF/wB3//8AmvdDuQAAAA9JREFUeJxj+MAwAQg/AAAMCAMBgre2CgAAAABJRU5ErkJggg==",
      "s04n3p01.png", false, 4, 4, "c268bd54d984c22857d450e233766115");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFAgMAAAGHBv7gAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAlQTFRFAP//dwD//wAAQaSqcwAAABlJREFUeJxjaGBoYFjAACI7gHQAEE9tACIATYMG43AkRkUAAAAASUVORK5CYII=",
      "s05i3p02.png", false, 5, 5, "fceb20e261cb29ebb6349bc6c2265beb");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFAgMAAADwAc52AAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAlQTFRFAP//dwD//wAAQaSqcwAAABRJREFUeJxjWNXAMLWBYSKYXNUAACoHBZCujPRKAAAAAElFTkSuQmCC",
      "s05n3p02.png", false, 5, 5, "fceb20e261cb29ebb6349bc6c2265beb");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAYAAAAGAgMAAAHqpTdNAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAlQTFRFAP8AAHf//wD/o0UOaAAAACJJREFUeJxjaGBoYJgAxA4MLQwrGDwYIhimJjBMSGBYtQAAWccHTMhl7SQAAAAASUVORK5CYII=",
      "s06i3p02.png", false, 6, 6, "b5c9900082b8119515e3b00634a379c5");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAYAAAAGAgMAAACdogfbAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAlQTFRFAP8AAHf//wD/o0UOaAAAABZJREFUeJxjWLWAYWoCwwQwAjJWLQAAOc8GXylw/coAAAAASUVORK5CYII=",
      "s06n3p02.png", false, 6, 6, "b5c9900082b8119515e3b00634a379c5");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAcAAAAHAgMAAAHOO4/WAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAxQTFRF/wB3AP93//8AAAD/G0OznAAAACVJREFUeJxjOMBwgOEBwweGDQyvGf4z/GFIAcI/DFdjGG7MAZIAweMMgVWC+YkAAAAASUVORK5CYII=",
      "s07i3p02.png", false, 7, 7, "cefe38d2a35e41b73b6270a398c283e8");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAcAAAAHAgMAAAC5PL9AAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAxQTFRF/wB3AP93//8AAAD/G0OznAAAABpJREFUeJxj+P+H4WoMw605DDfmgEgg+/8fAHF5CrkeXW0HAAAAAElFTkSuQmCC",
      "s07n3p02.png", false, 7, 7, "cefe38d2a35e41b73b6270a398c283e8");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAgMAAAHOZmaOAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAxQTFRFAP//dwD/d/8A/wAAqrpZHAAAACVJREFUeJxjYAACASB+wGDHoAWk9zDMYVjBoLWCQbeCQf8HUAAAUNcF93DTSq8AAAAASUVORK5CYII=",
      "s08i3p02.png", false, 8, 8, "3f0fc2c825d2fad899359508e7f645e1");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAgMAAAC5YVYYAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAxQTFRFAP//dwD/d/8A/wAAqrpZHAAAABtJREFUeJxjYGBg0FrBoP+DQbcChIAMIJeBAQA9VgU9+UwQEwAAAABJRU5ErkJggg==",
      "s08n3p02.png", false, 8, 8, "3f0fc2c825d2fad899359508e7f645e1");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAkAAAAJAgMAAAHq+N4VAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAxQTFRFAP8AAHf//wD//3cA/1YAZAAAACNJREFUeJxjYEACC4BYC4wYGF4zXAdiBgb7/wwMltEQDGQDAHX/B0YWjJcDAAAAAElFTkSuQmCC",
      "s09i3p02.png", false, 9, 9, "5c55b2480d623eae3a3aaac444eb9542");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAAAkAAAAJAgMAAACd/+6DAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAAAxQTFRFAP8AAHf//wD//3cA/1YAZAAAAB9JREFUeJxjYAAC+/8MDFarGRgso4FYGkKD+CBxIAAAaWUFw2pDfyMAAAAASUVORK5CYII=",
      "s09n3p02.png", false, 9, 9, "5c55b2480d623eae3a3aaac444eb9542");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAAH2U1dRAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAANhJREFUeJx9kL0OgjAURj9FfuJTuBjXhqkkDvBGujo1casOLOyEgZmpM4kk8Fi29FYpMTbNl8O59+Y2AByC48nw5Ehe4Pr25orpfEeQ6LhPNgLgdmpQm2iWsdVxqA3V9lOyWKajTCEwWpDpx8TO6Oz3zMIoHYgtlWDORlWFqqDKgiAk6OBM6XoqgsgBPj0mC4QWcgUHJZW+QD1F56Yighx0ro82Ow5z4tEyDJ6ocfQFMuz8ER1/BaLs4HforcN6hMRF18KlMIyluP4QbCX0qz0hsN6yWjv/iTeEUtKElO3EIwAAAABJRU5ErkJggg==",
      "s32i3p04.png", false, 32, 32, "bbe63d9433641df3fcd2c745fed89a93");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAMAAACBVGfHAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAHxJREFUeJyV0b0NgCAQBeBXAIlxCRt6WrbyNqB3CSsnYTAPTYzvSIhSXMhHcn8A7ch25FiviA40wDEkVAZ4hh2RQXMa6JLmxZaNPwEdBJO0aB9u3NhzraJvBKuCfwNmXQVBW9YQ5AskC1xW2n4ZMDEU2FlCNrOYae+Pt3ACA2HDSOt6Ji4AAAAASUVORK5CYII=",
      "s32n3p04.png", false, 32, 32, "bbe63d9433641df3fcd2c745fed89a93");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACEAAAAhBAMAAAHSze/KAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAPZJREFUeJxdjzFywjAQRT/JMCjEBdzAkxN4RhdIwQHcuKeiplNLqZKWzrUr+jQ+gA6Vv6sVlnkey5K+Vm8NxBvmNMP7DpHzxLmL/HCHG+Cy8xI6l+M0y2GGYBw1lN0kq5gTOaThawlM434SRrT4UVqEsAvCFSNKmjNejpCz3RWTAUs/WsldVOM0Wug/vfISsPcmaWtFxBqrAkqVAesJ+jOkKQ0E/bMYXalhl1bUWRUbykVooPwtPHG5nPkunPG441Fzx8BnOyz0OBEdjF8ciQ7GAfjm9WsX5W+uWqMMK3r0tUZE5qo8m0OtEd48qlq5vtRXm8Td/wMULdZI1p9klQAAAABJRU5ErkJggg==",
      "s33i3p04.png", false, 33, 33, "20708bc9a6ffa8d8ca6e004e1e9aa3ae");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACEAAAAhBAMAAAClyt9cAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAL5JREFUeJxdzy0SwyAQhuGv0+n0V6Q36HCCzHCBih4gBh8VXVeLjIyNi0bV13CAHKrLDi27vAwrEMADpMaS5wN8Sm+EEHAKpQXD0NMu9bAWWytqMU+YZRMMXWxENzhaO1fqsK5rTONXxIPikbvjRfHIPXGleOQaNlWuM1GUa6H/VC46qV26ForEKRLnVB06SaJwiZKUUNn1D/vsEqZNI0mjP3h4SUrR60G3aBOzalcL5TqyTbmMqVzJqV0R5PoCM2LWk+YxJesAAAAASUVORK5CYII=",
      "s33n3p04.png", false, 33, 33, "20708bc9a6ffa8d8ca6e004e1e9aa3ae");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACIAAAAiBAMAAAG/biZnAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAANJJREFUeJx9jr0KgzAURr9q/elbdJGu4mTAQR+pa6eAW+yQxV06ODs5CxX0sWrURHstDcnH4eTe3ABxBz6d5+74b8S7zcck72D7KvMx4XPaHfC4vVCpeP0OS0W1hAg9EQ0imqZhWElEm/OMm28tTdwQQkPzOrVl1pYpWplpcjQ1ME6aulKTawhbXUnI0dRsZG5hyJVHUr9bX5Hp8tl7UbOgXxJFHaL/NhUCYsBwJl0soO9QA5ddSc00vD90/TOgprpQA9rFXWpQMxAzLzIdh/+g/wDxGv/uWt+IKQAAAABJRU5ErkJggg==",
      "s34i3p04.png", false, 34, 34, "0912e0f97224057b298f163739d1365f");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACIAAAAiBAMAAADIaRbxAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAG1JREFUeJyVz7ENgCAQBdBfIIlb2NDbMpYb0LMEFZMwGKcWJv9HwSsu5CX8uwPOOnKNod0dKtbhSHY0EiwkBYHEglk0OW4yPfwXqHhOTraPG234vCcFYykqKwtUeFZS8Sx2NUjqhFz1LVl+vUgHrMXtiDoroU4AAAAASUVORK5CYII=",
      "s34n3p04.png", false, 34, 34, "0912e0f97224057b298f163739d1365f");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACMAAAAjBAMAAAGb8J78AAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAQRJREFUeJxlkD2uglAUhMf4A1GL93ZAWIHJ2YCFC7Cxt7Kmo7WktLWjprJ/DQu4i3pzzuUAF4fwk5k7+SYAzRN96CFyQsPvEIC80ZcIDf04iYZ5HmOeZaQOYzoxDRY05og7MCePDtQ5Al2770woUEahrrPahBaeluWUqiqmMWqBMS2GtEYGHR4XdK2flLVI3OO0AqE/hrjXuRWb3sVIEfHuRLMifxEGbsauFdl/Dk1NvTsthXeDdytUMP3N9MHjcec90x3vF96JXrjx2t5muuJC2cN1xi9lD9cPcCBjQeSGJXEpEhMYdU1hm5E4wlZGTGAHFj9IYTsd8A1MiVujzokXHXH+B9CK7qGbaRQOAAAAAElFTkSuQmCC",
      "s35i3p04.png", false, 35, 35, "b46d9ba87963f526624a6d485ff6465e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACMAAAAjBAMAAADs965qAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAMdJREFUeJxl0SEOg0AQheHXtJSmmPYGhBOQcIEKDoDBo9C42spKLA6Nqq/hAHuoPqZhM7P7E0asmOyXBbbeqpec4Kv6YFkWXBfVjL7v+Ks6VBWOla7ENGIyjSi4vdDlaPklraqBc27dhm9FzWTsPfBkMvYG3JmMvZv4QmNGlTXOvFdo5FFkDCoD4N8YRqPhsSbgsdXyTt7oeak3et5BjIZ3EaPhZVwv76h4kuWdN3JMjIwjImMOa0zEaY3Ocb021tsVrJE+pMMPA+LuR86i5UgAAAAASUVORK5CYII=",
      "s35n3p04.png", false, 35, 35, "b46d9ba87963f526624a6d485ff6465e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACQAAAAkBAMAAAFkKbU9AAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAANlJREFUeJyNkb0KgzAURj/b+tO36CJdxUnBob5Ru3YKuGkHF3dxcHZyDlTQx2piTaJJC4bk43juvUEUiCgIO6/V8d6IVptMSUZx9HhmU0IwJwWe1+aOes7mV9ZzHr6JJfPAzcORbRCMC+Whcq5044bIgQoKXEGhcDn4svoqZRt9mQqyBXWQrpR9lSBHElRf9ZdgLdRVkCSqnaraqnozifXN61G0sT8siaINMGiqhq8rxDjpg7Fv3GUoOPFF72LvoF+/etipav4DtgosYSptELsHdXX2qaZa/jk/GoQXLvsYf8IAAAAASUVORK5CYII=",
      "s36i3p04.png", false, 36, 36, "65e57e33b4763a3b0c3f0fa92295406d");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACQAAAAkBAMAAAATLoWrAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAHdJREFUeJxjYACBwu5llqpHoCQDFiEgxcCCLmTAcARdiIEVXWgBgyq6ENB0DCEsxlsqYDpClSwhBixCbBjGNwDdhe4ILE5F4lBXCBToqEILgEKMqEIMnKoHGNCEgCQWoULCQgYYNjJgsZGBWBvJE8L0EBZvgwMHAABJBMjTkay+AAAAAElFTkSuQmCC",
      "s36n3p04.png", false, 36, 36, "65e57e33b4763a3b0c3f0fa92295406d");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACUAAAAlBAMAAAFAtw2mAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAP5JREFUeJxlkDsSgjAQhv8RHfFR6A0cTsBMLmDhAWjsrajpaFNS0tqlprK3yQFyKDcb8jD5GJLMssu3G2CS0EZDiBYTnY0Bat59DHxuBYG6nihgLBAcmSywm+Sclr9qjkvOKSOIESmxqOPCKNzQOG4Yx/3IDFAICU2TJDAglhUVEzYhYaA/2JFco4tacyEq4YhWGH02brigp0pfG0QQntiQu5S11vUNdzk8dmgx1FaxV1+rTWza19bWS3xTPuj7F70pL7xnvP+Z8aRn90zp8CB4CdxxJXgJXIATiXIvtVJ4C8hb0OVK5ppzyUa1FE5rLb04FN4OuZdG367zplJ6fx0nFJojsT+zAAAAAElFTkSuQmCC",
      "s37i3p04.png", false, 37, 37, "f21eff5c07a755577fea69c01459c65f");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACUAAAAlBAMAAAA3sD0wAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAMVJREFUeJxl0S0Og0AQhuEvoU36J+AGDSfYhAsgegAMHoWuq62sxOJWr6rHcAAO1dkppbMzD9kRmxB4M0D0kp58hUl6I4SAU5A8+r6jI3WoKmRVwmEcMKYGlPSJMnFFS8++lRosyzLH8TfjRnhsajwIj80dBeGxybnV9J4pUPV6+j/TS3e2V3M69ttrUK/RpKmiV6QylcoKLVerXXMnjd4NGrxqjbW212W2F0fbC9vbwPbOF91Lq96t+xXw26+MjUfFHuh8APqFElFWDb0cAAAAAElFTkSuQmCC",
      "s37n3p04.png", false, 37, 37, "f21eff5c07a755577fea69c01459c65f");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACYAAAAmBAMAAAEtFMQLAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAANpJREFUeJylkT8LgkAchp/+WEFfokVapamDhvxGtTYduGmDi7s0NDc1Cwn6sTpF7w7PQEju9/L48N6pCMcCqeZuzeYjOfZT0I6sT1HNYtNkVHcpi5aB2/5xIW/z8TtzKzsDcbCOD5VaEknVY3yw7NrYaoABGucVxmJbmL2zUK0X7zTU6Gl8YWxqupnGlUGsbjYNUzR6ZzSGjFisbjjWbQrtdU2ewi/7JHkGlEOX4zsOwdLZK3z3PNexEjunp17FeYZ995dr/uR24JpvYoIb3euVlyl7x3pCnZd8AfUFRB95/EUWAAAAAElFTkSuQmCC",
      "s38i3p04.png", false, 38, 38, "f6237240a70b5844def0406dc8f63bbd");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACYAAAAmBAMAAABaE/SdAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAGpJREFUeJxjYACBwu5llqpHYCQDNjEgzcCCIWbAcARDjIEVQ2wBgyqGGNAKTDFsdlgqYHGLKrliDNjE2DDtaAC6D8Mt2NyMzBs4MaDL0MUMgGLcaGLAuClgQBcDkmSLYTEPm72DyS3gsAIA8mkrg86sROEAAAAASUVORK5CYII=",
      "s38n3p04.png", false, 38, 38, "f6237240a70b5844def0406dc8f63bbd");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACcAAAAnBAMAAAEJinyQAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAARlJREFUeJxtkDFygzAQRZ/HkwSTFM4NGE7AjC6QIgdw457KtTtaSpVu3VFTuafhADpUVhISCOUxwPC1u/8voHtmM6NUg9ZgDBSimUSbaZRAUWgRjAXlFPmWavdaavypdopKlb6wStM4xTX1PeNQjh4q6gW67qPzMBAL6npTEGA5HcYhFFQ1a8E9FIyU2O20Dy0NSyPqqDzNmqHCzF8uuqwf49ylP06AdYKKE2LGym8eJsQ4OusvR8KEoyJMkCzE/s1ChAnoTYIBx5Tw4nZr5U5oeT547nhwlevtmnDhV3CPlR++BfdYOcOnuGXukih3zxH3nMvOeOOeOh/OmfE0Zc7tuzXfuT9O1nzv7n/lf+b7tQ8uQOpurXn9AQyWNfYM/uLgAAAAAElFTkSuQmCC",
      "s39i3p04.png", false, 39, 39, "ceb3b33633c181e13ecee80b153fb602");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACcAAAAnBAMAAAB+jUwGAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAANVJREFUeJxt0iEOg0AQheFHSJuUGnqDhhOQcIEKDoDBo9A4LBKJxaFX4TEcgEN1ZklDZ2Z/YMQa+DIA3Cga/Bk20QrnHBInWtC2DT2iBkWBuJDlmCfMqgkZvSeTvVHTdatFFY7j2Hn8taOk/Lj6oKf8uOrwovy4Sr3b2p9k1faFvtPa6TBgN+UGftptZLdViv1nL0P2PmSX7ihV7JEXPhj2ttGxYidMV+7mznRlz2OmK/v0YDo0m25o+/kXGjfoDtED9g565dFv7WLlni/tDMeq7AxPli8bpjUVK/+f5gAAAABJRU5ErkJggg==",
      "s39n3p04.png", false, 39, 39, "ceb3b33633c181e13ecee80b153fb602");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACgAAAAoBAMAAAEJ15XIAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAANpJREFUeJytjrEKgzAURa/FWmv7EV2kq2RS6FD/qK5OATd1cHEXB2cn50IL+lnVUBMxr4VCQ97N4ZC8POT+HcexclEQp/3g8GVBnHy4JANgT5kM66zjcx1jIxKLrFfpTFndROLN6aZPmdjgTKLjSUwXyL6gt+MSexCWAei2YVeKjXaBpUQotAoKAWPGTtmu/B1hzViEoPCqEK1EQ2GocGyWNXCfUdYEi0RW7QmJQJfcIiSaALqcltaTuvlJEiP9VZ7GAa21nCYBIUFIHHQJg3huUj3NiGvSHb9pXgoWak5w83t4AAAAAElFTkSuQmCC",
      "s40i3p04.png", false, 40, 40, "140f0d2eb778dad4a1871c17767e2d48");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACgAAAAoBAMAAAB+0KVeAAAABGdBTUEAAYagMeiWXwAAAANzQklUBAQEd/i1owAAACdQTFRFAAAA/wB3AP//AP8AdwD/AHf/d/8A/wD//wAAAP93//8A/3cAAAD/9b8GOwAAAHVJREFUeJzN0LENgDAMRNFDiiwhlqBhiGxFNkifJagyBwWDEagQ/kXoSOHiyVZOp1K1HKnU+Jhi3BBHQCFGjxnRAGVRHms3Xq8LC51/Qurz99iacDg3tDcqpCyHbRLipgBXQk0ed8FHGggpUuCcuOnDYyF3dSfnZ1dwSF0UKQAAAABJRU5ErkJggg==",
      "s40n3p04.png", false, 40, 40, "140f0d2eb778dad4a1871c17767e2d48");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAAJ0Uk5TAA/mLNCpAAAAAmJLR0QAAKqNIzIAAAFISURBVCiRddExT8JAFMBxPoJHWUxcriQuJiaFqpNLWxkdLOVCHJjunSau9GByohwTk8Il+hkcHd0kLrIymLCaOLBq0epdbRRIeNv9pnfvn/temdw6eJktQXJPK7cL8BbRklmsjzNInsJquWRjc/8mhc9B6JZt13aLe6z9rGDEm2W7VvU8d5vzcwUTEXqMcxocMd48VfAqBM8mDI4VvENr2M3eXkMDE1Km4iO7r+BDgxaKkXGnAURv0JZd6uON/FRBDK1eBHIQOAgX9GJzOBO8psA0nIN0UyBdTuS1j228qeELKh0NJ9hCWxoSCCKmwMljtJv+FgJOiLwqGRg1foEyDVbBQv0UIspqRHawgnEKMQBoMNBOdsJHBb0ORvlxBkkERDQtdPh35FiDU5j9ZxgRQf3LxS5DQetL5eaCPiynnFystE2m6+r/AOOSVs9bKk33AAAAAElFTkSuQmCC",
      "tbbn0g04.png", false, 32, 32, "d9b53613bd731e66dcfd5be93186c100");
  testPngSuiteImage("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",
      "tbbn2c16.png", false, 32, 32, "75954a76132c3971509841e973f029cd");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAuJQTFRF////gFZWtbW4qEJCn5+fsSAgixUVnZ2dGxtZm5ubAACEmZmZj6ePl5eXlZWVk5OTKSlWkZGRAACbj4+Pi5WLLi6njY2NgAAAi4uLuQAAiYmJDAzVeHV1h4eHAACyhYWFpQAA3gAAgYGBf39/AACefX19AADJe3t7eXl5NzdWd3d3dXV1c3NzSKlIjgAAAgJkAABiVolWKCh8U4tTiYmPZ2dnZWVlXW1dE+UThiYmby0tRJFEYWFhO507RIlEPZM9AACkAPMAAPEAWVlZV1dXVVVVU1NTNIU0UVFRJJckT09POjpBEBC6sg8PAMcAAMUA/Pz8AMMABASXAMEALXct+vr6AL8AAABoAL0A2tTUEBB7Ca0J+Pj4ALkAALcAnJyh9vb2DKEMALMAALEAEJEQAKsA8vLyAKkAAKcA7u7u7OzsAJcA6urqAABrAI0AAIsAAIkAAIcAMTExGRkqBwdAEhKuCQnu09bTzMzMkwAAoyoqxsbGxMTEzAAA0woKgWtreD4+AwNtAACfCgpWRkZIQUFNc11dUQcHqKio7e3voKCgnp6enJycAAC5mpqasgAAmJiY6wAAlpaWngAAlJSUExMckpKSkJCQjo6OAACRioqKiIiIdqJ2hYiFhoaGhISEeA8PgoKCfoJ+fn5+fHx8enp6SsBKdnZ2dHR0cnJycHBwmAAAbm5uanBqemZmampqhAAARKJES5ZLYWRhYmJiAPQAOJg4XFxcWlpaAOYAAgJdQnhCVlZWAADwLpQuR2hHMTFgANgAUlJSUFBQAM4AIZghFBRtAMgATExM/f39AMYAAACdb2tr6g4OSEhIALwANGY0AgL1U1NgALAAAK4AtwAAAKQA7+/vAKIAj09PlTQ0AJgAAJYAAJIA5+fnAIwA4+PjAIAAkgYGAQFvZFZZAABkTk5rz8/P3d3gAAB7ycnJFhZBISFZV1dZRER4v7+/693dLS1UCgpgAAD/v319AAAAzmH7FgAAAAF0Uk5TAEDm2GYAAAABYktHRPVF0hvbAAACiklEQVQ4jWNgoDJ48CoNj+w9psVmTyyZv3zAKpv5Xsq0rYFNb4P4htVVXyIDUGXTavhWnmmwrJxcKb7Aqr29fcOjdV3PY2CyMa/6luu0WT6arNBfWyupwGa5QHy13pM1Oss5azLBCiqUl2tr35Lsv+p76yarouLEiYq1kuJntIFgfR9YwQv52fPVGX1Zb8poaWnVM9edPVtXxQhkrtp+6D1YQc58pbkzpJQ1UMHyLa6HT9yDuGGR5zVbEX7h+eowsHSpxnqXwyfOOUNdOSvplOOyaXy8U2SXQMHK7UZBUQItC6EKpkVHbLUQnMLLzcktobx4sarWlks+ajPDwwU6oAqmJCbt3DqHX2SjLk93z4zF63e8ld7btKvEgKMcqqDjaOrxrcum6Z5P38fO0rV0h7PoZ7VdxVObNWHBybTvxpWdTiIbj9/e1tPNssL52cW9jd7nXgushAVltXty3hHHTbZ+t+052bvXAA1weNMa1TQzHqYgcnfyw1inFNtT2fZ9nOymb8v2Nh4IUnn5qRqmIGf3lcLEgxmegXfsJ/T12Lz73Mvx+mVuLkcCTEHA/vQ7IcH+d4PvbuLl7tshepHrY7H+Y6FniNhee+3a/sSD+WF5m/h4J7mU7g1vLToml2uCUCB24/IFu+PZ5+9b8/MJ7/Hp1W854HC6uRqhIJTHfbNZ9JXYfGNBfinX0tOfDgTJcTChJKnna8z2JcUVGAoLKrlGcelzzTz2HC1JZs0zv5xUYCwmvNT1Y+NTA6MXDOggoOPo5UJDCbEVbt7FJe86MeSBoHxbyKLZEmsOeRVphWKTZ2C43jV/3mxTj8NdJ7HLA8F7+Xk2h5hwSgPBi+lmFfjkGRgSHuCXxwQADa7/kZ2V28AAAAAASUVORK5CYII=",
      "tbbn3p08.png", false, 32, 32, "d1f6636d81c74f163bfff1405bf406cf");
  testPngSuiteImage("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",
      "tbgn2c16.png", false, 32, 32, "75954a76132c3971509841e973f029cd");
  testPngSuiteImage("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",
      "tbgn3p08.png", false, 32, 32, "d1f6636d81c74f163bfff1405bf406cf");
  testPngSuiteImage("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",
      "tbrn2c08.png", false, 32, 32, "75954a76132c3971509841e973f029cd");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgEAAAAAAGgflrAAAABGdBTUEAAYagMeiWXwAAAAJ0Uk5T///Itd/HAAAAAmJLR0T//xSrMc0AAAS8SURBVEiJY/hPIWCgqQGPH588+fEjWQa8eLF1a11damrUTjf9hB8LBR8JEG3Au3f793d2pqcnTvDZZSaiFaD+Unmr+hy9VwGeE72vbP/zB48Bnz4dOTJ1alFRzrzgSbYLdLP0t2s3q2VqLbc5rnRIglFqq/pLA46ctAULzp//8QPNgO/fT52aNausrLg4bZXHY0NO/SyjbSYXbALsZM1bDAOtZ7tWGFerbdNl1noZ1Z6XV1xcVjZ79pEj797BDThzpr6+rKwUCPzEzc3NmM3m2sSE2qTIBag5zne+6L7dNdAxy07O/IKaWc68UijIypo1C27AnTutrR0dLS3V1ckcLp7u8omvyqLLwaCINeFw2N4gEb9Yb1HfVUk3IaIFBTExQUF798INePWqpaWxsd2zr6+zs76+Ei8oK0tODgkJCPDxefYMbsCPH02FGe5JVypsJtyYPLm/v7m5GgNUAUFlZVZWeDhIs6dnZCRKLHR1ZV4pmdXXPEF20qSpU6dPnzKlvb0GDRQWRnMb3RQpkSjTXeO2p6kJxYBJkzLX5fv2b+zPnThxypTp02fOnD175szu7vr6OiCorS0vT0oKuaR6XbxY4ASPEPd1fek1a1AMmDIl/WMWQ6t4/8YJ8ZMnTy6skqxPnf5r3rw5c/r66uqysqKiwtfrPJOeLpTCc4H9Obe6CvO1aygGLFmSbpoiW3oc6IbCSZNaGPK2JbflGc2dO3/+ggVVVVFRkZF2grIBYod4FaVieUVFCmz//v6NYsC2bWn88empD7tS+ionzKpTL4uLksr7M2fOvHnz55eUREYGfVWYLT2dv8vyioeHlIz+/6IitKR8/HhKXYZNMGf16n6pqkulHaWGkc0FlbNnz507b15eXmSklYxsgLCotrzLEiUuIXdBs7n6aAbcuJEckWHjkZ8T3fsmOr0kqmRWxJv8R7NmgYxITw+fpBwDtP+2+XqxY0KafI5i9sePoxnw6lXi8dSHfsGx9o19SREZnEXXIkILFGbMmDVrzpzERE9X2QBRF8Vz0p/5lHl8eXyVjn/5gmbAnz8JZ5PbwvdHHCxcUcIc9rtwRcjZkhZQdM6aFRVlKSLjzp9hHCS9j1eD10LwUmAwluyc9yhhSsKUUNPMipobgbcLqoLnFzeDktS0aeHh2q8lW7m/OizQ1hY3EpnM49v+HIsBPT3x8ulLA5dlPCr7GvEmb1tQUeHryZOnTu3vDwtTihd14Utxdzd1F5ovxCMkd/QoFgN+/Vqckfw8WTW9KbMnrSLnU+Dt0uqJEydP7uwMDZUxFuIRnGVxVjReJFL++7a0//+xGAACFy7k5qampj3OuJytF7CuVKm/f+LExsbQUEV1fl2eBgEBgdWqnec5///HacD//2/etLWlVaXzZWYGiBcr9Pb291dVhYToN/KJcNdzJxqeeDDh/3+8BoDiY9WqdNP0pX4vi2d3d/f2lpQEB9vaSisLO/lYvNNAV42jWL9yuKg14mD9jY6O7u7c3KAgf39z8/LyX7+wqcVRL7x/v2BBc3NbW0dHenpgYEDAggV//2JXibNm+vfvwIHW1ra2xMSgoO3bcakiULXduzdhQmrqmTP41BCoXL9+ffwYvwqKa2cA4MyW1TM3HhMAAAAASUVORK5CYII=",
      "tbwn0g16.png", false, 32, 32, "56ea136a6e299452015ac02a7837e365");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAt9QTFRF////gFZWtbW4qEJCn5+fsSAgixUVnZ2dGxtZm5ubAACEmZmZj6ePl5eXlZWVk5OTKSlWkZGRAACbj4+Pi5WLLi6njY2NgAAAi4uLuQAAiYmJDAzVeHV1h4eHAACyhYWFpQAA3gAAgYGBf39/AACefX19AADJe3t7eXl5NzdWd3d3dXV1c3NzSKlIjgAAAgJkAABiVolWKCh8U4tTiYmPZ2dnZWVlXW1dE+UThiYmby0tRJFEYWFhO507RIlEPZM9AACkAPMAAPEAWVlZV1dXVVVVU1NTNIU0UVFRJJckT09POjpBEBC6sg8PAMcAAMUA/Pz8AMMABASXAMEALXct+vr6AL8AAABoAL0A2tTUEBB7Ca0J+Pj4ALkAALcAnJyh9vb2DKEMALMAALEAEJEQAKsA8vLyAKkAAKcA7u7u7OzsAJcA6urqAABrAI0AAIsAAIkAAIcAMTExGRkqBwdAEhKuCQnu09bTzMzMkwAAoyoqxsbGxMTEzAAA0woKgWtreD4+AwNtAACfCgpWRkZIQUFNc11dUQcHqKio7e3voKCgnp6enJycAAC5mpqasgAAmJiY6wAAlpaWngAAlJSUExMckpKSkJCQjo6OAACRioqKiIiIdqJ2hYiFhoaGhISEeA8PgoKCfoJ+fn5+fHx8enp6SsBKdnZ2dHR0cnJycHBwmAAAbm5uanBqemZmampqhAAARKJES5ZLYWRhYmJiAPQAOJg4XFxcWlpaAOYAAgJdQnhCVlZWAADwLpQuR2hHMTFgANgAUlJSUFBQAM4AIZghFBRtAMgATExM/f39AMYAAACdb2tr6g4OSEhIALwANGY0AgL1U1NgALAAAK4AtwAAAKQA7+/vAKIAj09PlTQ0AJgAAJYAAJIA5+fnAIwA4+PjAIAAkgYGAQFvZFZZAABkTk5rz8/P3d3gAAB7ycnJFhZBISFZV1dZRER4v7+/693dLS1UCgpgAAD/v319DyW3rQAAAAF0Uk5TAEDm2GYAAAABYktHRACIBR1IAAACiklEQVQ4jWNgoDJ48CoNj+w9psVmTyyZv3zAKpv5Xsq0rYFNb4P4htVVXyIDUGXTavhWnmmwrJxcKb7Aqr29fcOjdV3PY2CyMa/6luu0WT6arNBfWyupwGa5QHy13pM1Oss5azLBCiqUl2tr35Lsv+p76yarouLEiYq1kuJntIFgfR9YwQv52fPVGX1Zb8poaWnVM9edPVtXxQhkrtp+6D1YQc58pbkzpJQ1UMHyLa6HT9yDuGGR5zVbEX7h+eowsHSpxnqXwyfOOUNdOSvplOOyaXy8U2SXQMHK7UZBUQItC6EKpkVHbLUQnMLLzcktobx4sarWlks+ajPDwwU6oAqmJCbt3DqHX2SjLk93z4zF63e8ld7btKvEgKMcqqDjaOrxrcum6Z5P38fO0rV0h7PoZ7VdxVObNWHBybTvxpWdTiIbj9/e1tPNssL52cW9jd7nXgushAVltXty3hHHTbZ+t+052bvXAA1weNMa1TQzHqYgcnfyw1inFNtT2fZ9nOymb8v2Nh4IUnn5qRqmIGf3lcLEgxmegXfsJ/T12Lz73Mvx+mVuLkcCTEHA/vQ7IcH+d4PvbuLl7tshepHrY7H+Y6FniNhee+3a/sSD+WF5m/h4J7mU7g1vLToml2uCUCB24/IFu+PZ5+9b8/MJ7/Hp1W854HC6uRqhIJTHfbNZ9JXYfGNBfinX0tOfDgTJcTChJKnna8z2JcUVGAoLKrlGcelzzTz2HC1JZs0zv5xUYCwmvNT1Y+NTA6MXDOggoOPo5UJDCbEVbt7FJe86MeSBoHxbyKLZEmsOeRVphWKTZ2C43jV/3mxTj8NdJ7HLA8F7+Xk2h5hwSgPBi+lmFfjkGRgSHuCXxwQADa7/kZ2V28AAAAAASUVORK5CYII=",
      "tbwn3p08.png", false, 32, 32, "d1f6636d81c74f163bfff1405bf406cf");
  testPngSuiteImage("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",
      "tbyn3p08.png", false, 32, 32, "d1f6636d81c74f163bfff1405bf406cf");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAgMAAAAOFJJnAAAADFBMVEUAAP8AAP8AAP8AAP+1n0POAAAAA3RSTlMAVaoLuSc5AAAAFElEQVR4XmNkAIJQIB4sjFWDiwEAKxcVYRYzLkEAAAAASUVORK5CYII=",
      "tm3n3p02.png", false, 32, 32, "82e044043a1f2c91533b2fea5e271daa");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAAABGdBTUEAAYagMeiWXwAAAoZJREFUOI1jqCcAGEhU8PjkRzwKXmytS41yS1z4EKuCd/s70xN9zbQ0VDT0AyZe+YOq4NORqUU5wXZ6Bjrq2rbKEtIaBjkLzv+AKfh+alZZcZqnoYGxqY2dhaGNq7G6rnZUXnHZ7CPvwArO1JeVlvqZm5nbhKYEOLl4uDrZWajllAJB1iywgjutHS3VyS7uSWXl5eVFieFBft5+yUBmQUzQXrCCVy2N7X2d9ZWooCw5JMDnGVjBj6aMpIoJk/ubq2GgqqoyKzzAxzMS6ouuzJK+/klTp09pr4GCwmhjEQldtyaogkmZ+f39E6dMnzl7Znd9XV1teVKIqrggD4/+GqiCKemZLf0TJk+uqp8+b05fXVZUuK60EC8Ht8o1qIIl6Sml/f2TmvOS8+bOX1AVFWknK8YrxSti9xuqYFtafGpX34S6sqi8OfPml0QGK0jzW3lIGRTBgvp4SkZwdV9VaWlkwey58/IirWSFtV2UhATnwhTcSM7wyOmNLimJyJ81e256uLK0gLm4EJ/YcZiCV4mpfrGNSRlFEQUzZs1J9JQVVZLh4+FR/gJT8CchOTyisDi8MKRk+sxZUZYy/MYyvLxCgYjozktICM2sCSwILp46fVq4jiSPg7a4CE87QkFPfHpgRllEXlDh5Kn9YUqifO6mQkJCRxEKfi1OTk7PTMsJLJ04uTNURkjQUlREYRtKkruQm5qWkR1Q2j+xMVSRn0dAQPUcWpp805aWnhlQ3NtfFaLPx81t+AAj0f5ZlZ7uX9zdWxJsKy3s8xZbsr9SFFHf0Z0b5G9e/gt7vni/oLmtIz0wYMFfbPkCBP4daG1LDNqOLISe9e5NSD1Tj09B/dfH9fgVYAAA90bMUdlj1V0AAAAASUVORK5CYII=",
      "tp0n0g08.png", false, 32, 32, "38deadbdfb7b0ff5a2b4cad35e866b39");
  testPngSuiteImage("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",
      "tp0n2c08.png", false, 32, 32, "c37c05b6929096c1736f91dccbe93d15");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAt9QTFRFFBRtgFZWtbW4qEJCn5+fsSAgixUVnZ2dGxtZm5ubAACEmZmZj6ePl5eXlZWVk5OTKSlWkZGRAACbj4+Pi5WLLi6njY2NgAAAi4uLuQAAiYmJDAzVeHV1h4eHAACyhYWFpQAA3gAAgYGBf39/AACefX19AADJe3t7eXl5NzdWd3d3dXV1c3NzSKlIjgAAAgJkAABiVolWKCh8U4tTiYmPZ2dnZWVlXW1dE+UThiYmby0tRJFEYWFhO507RIlEPZM9AACkAPMAAPEAWVlZV1dXVVVVU1NTNIU0UVFRJJckT09POjpBEBC6sg8PAMcAAMUA/Pz8AMMABASXAMEALXct+vr6AL8AAABoAL0A2tTUEBB7Ca0J+Pj4ALkAALcAnJyh9vb2DKEMALMAALEAEJEQAKsA8vLyAKkAAKcA7u7u7OzsAJcA6urqAABrAI0AAIsAAIkAAIcAMTExGRkqBwdAEhKuCQnu09bTzMzMkwAAoyoqxsbGxMTEzAAA0woKgWtreD4+AwNtAACfCgpWRkZIQUFNc11dUQcHqKio7e3voKCgnp6enJycAAC5mpqasgAAmJiY6wAAlpaWngAAlJSUExMckpKSkJCQjo6OAACRioqKiIiIdqJ2hYiFhoaGhISEeA8PgoKCfoJ+fn5+fHx8enp6SsBKdnZ2dHR0cnJycHBwmAAAbm5uanBqemZmampqhAAARKJES5ZLYWRhYmJiAPQAOJg4XFxcWlpaAOYAAgJdQnhCVlZWAADwLpQuR2hHMTFgANgAUlJSUFBQAM4AIZgh////AMgATExM/f39AMYAAACdb2tr6g4OSEhIALwANGY0AgL1U1NgALAAAK4AtwAAAKQA7+/vAKIAj09PlTQ0AJgAAJYAAJIA5+fnAIwA4+PjAIAAkgYGAQFvZFZZAABkTk5rz8/P3d3gAAB7ycnJFhZBISFZV1dZRER4v7+/693dLS1UCgpgAAD/v319RGIGqgAAApBJREFUOI1jUCYAGEhU8OBVGh4F95gWmz2xZP7yAauCzPdSpm0NbHobxDesrvoSGYCqIK2Gb+WZBsvKyZXiC6za29s3PFrX9TwGpiDmVd9ynTbLR5MV+mtrJRXYLBeIr9Z7skZnOWdNJlhBhfJybe1bkv1XfW/dZFVUnDhRsVZS/Iw2EKzvAyt4IT97vjqjL+tNGS0trXrmurNn66oYgcxV2w+9ByvIma80d4aUsgYqWL7F9fCJexA3LPK8ZivCLzxfHQaWLtVY73L4xDlnqC9mJZ1yXDaNj3eK7BIoWLndKChKoGUhVMG06IitFoJTeLk5uSWUFy9W1dpyyUdtZni4QAdUwZTEpJ1b5/CLbNTl6e6ZsXj9jrfSe5t2lRhwlEMVdBxNPb512TTd8+n72Fm6lu5wFv2stqt4arNmAFQB074bV3Y6iWw8fntbTzfLCudnF/c2ep97LbASFtTV7sl5Rxw32frdtudk714DNMDhTWtU08x4mILI3ckPY51SbE9l2/dxspu+LdvbeCBI5eWnapiCnN1XChMPZngG3rGf0Ndj8+5zL8frl7m5HAkwBQH70++EBPvfDb67iZe7b4foRa6PxfqPhZ4honvttWv7Ew/mh+Vt4uOd5FK6N7y1iEEu1wShQOzG5Qt2x7PP37fm5xPe49Or33LA4XRzNUJBKI/7ZrPoK7H5xoL8Uq6lpz8dCJLjYEJJcs/XmO1LiiswFBZUco3i0ueayfAcLU1mzTO/nFRgLCa81PVj41MDoxcYiTag4+jlQkMJsRVu3sUl7zqxJfvybSGLZkusOeRVpBWKPV9c75o/b7apx+Guk9jyBQgcey8/z+YQE7IQetZ7Md2sQhmfAuWEB8r4FWAAANxEPMkO1rmYAAAAAElFTkSuQmCC",
      "tp0n3p08.png", false, 32, 32, "985ccf415de9754ff21296de6cf69a38");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAABGdBTUEAAYagMeiWXwAAAt9QTFRF////gFZWtbW4qEJCn5+fsSAgixUVnZ2dGxtZm5ubAACEmZmZj6ePl5eXlZWVk5OTKSlWkZGRAACbj4+Pi5WLLi6njY2NgAAAi4uLuQAAiYmJDAzVeHV1h4eHAACyhYWFpQAA3gAAgYGBf39/AACefX19AADJe3t7eXl5NzdWd3d3dXV1c3NzSKlIjgAAAgJkAABiVolWKCh8U4tTiYmPZ2dnZWVlXW1dE+UThiYmby0tRJFEYWFhO507RIlEPZM9AACkAPMAAPEAWVlZV1dXVVVVU1NTNIU0UVFRJJckT09POjpBEBC6sg8PAMcAAMUA/Pz8AMMABASXAMEALXct+vr6AL8AAABoAL0A2tTUEBB7Ca0J+Pj4ALkAALcAnJyh9vb2DKEMALMAALEAEJEQAKsA8vLyAKkAAKcA7u7u7OzsAJcA6urqAABrAI0AAIsAAIkAAIcAMTExGRkqBwdAEhKuCQnu09bTzMzMkwAAoyoqxsbGxMTEzAAA0woKgWtreD4+AwNtAACfCgpWRkZIQUFNc11dUQcHqKio7e3voKCgnp6enJycAAC5mpqasgAAmJiY6wAAlpaWngAAlJSUExMckpKSkJCQjo6OAACRioqKiIiIdqJ2hYiFhoaGhISEeA8PgoKCfoJ+fn5+fHx8enp6SsBKdnZ2dHR0cnJycHBwmAAAbm5uanBqemZmampqhAAARKJES5ZLYWRhYmJiAPQAOJg4XFxcWlpaAOYAAgJdQnhCVlZWAADwLpQuR2hHMTFgANgAUlJSUFBQAM4AIZghFBRtAMgATExM/f39AMYAAACdb2tr6g4OSEhIALwANGY0AgL1U1NgALAAAK4AtwAAAKQA7+/vAKIAj09PlTQ0AJgAAJYAAJIA5+fnAIwA4+PjAIAAkgYGAQFvZFZZAABkTk5rz8/P3d3gAAB7ycnJFhZBISFZV1dZRER4v7+/693dLS1UCgpgAAD/v319DyW3rQAAAAF0Uk5TAEDm2GYAAAKKSURBVDiNY2CgMnjwKg2P7D2mxWZPLJm/fMAqm/leyrStgU1vg/iG1VVfIgNQZdNq+FaeabCsnFwpvsCqvb19w6N1Xc9jYLIxr/qW67RZPpqs0F9bK6nAZrlAfLXekzU6yzlrMsEKKpSXa2vfkuy/6nvrJqui4sSJirWS4me0gWB9H1jBC/nZ89UZfVlvymhpadUz1509W1fFCGSu2n7oPVhBznyluTOklDVQwfItrodP3IO4YZHnNVsRfuH56jCwdKnGepfDJ845Q105K+mU47JpfLxTZJdAwcrtRkFRAi0LoQqmRUdstRCcwsvNyS2hvHixqtaWSz5qM8PDBTqgCqYkJu3cOodfZKMuT3fPjMXrd7yV3tu0q8SAoxyqoONo6vGty6bpnk/fx87StXSHs+hntV3FU5s1YcHJtO/GlZ1OIhuP397W082ywvnZxb2N3udeC6yEBWW1e3LeEcdNtn637TnZu9cADXB40xrVNDMepiByd/LDWKcU21PZ9n2c7KZvy/Y2HghSefmpGqYgZ/eVwsSDGZ6Bd+wn9PXYvPvcy/H6ZW4uRwJMQcD+9Dshwf53g+9u4uXu2yF6ketjsf5joWeI2F577dr+xIP5YXmb+HgnuZTuDW8tOiaXa4JQIHbj8gW749nn71vz8wnv8enVbzngcLq5GqEglMd9s1n0ldh8Y0F+KdfS058OBMlxMKEkqedrzPYlxRUYCgsquUZx6XPNPPYcLUlmzTO/nFRgLCa81PVj41MDoxcM6CCg4+jlQkMJsRVu3sUl7zox5IGgfFvIotkSaw55FWmFYpNnYLjeNX/ebFOPw10nscsDwXv5eTaHmHBKA8GL6WYV+OQZGBIe4JfHBAANrv+RnZXbwAAAAABJRU5ErkJggg==",
      "tp1n3p08.png", false, 32, 32, "d1f6636d81c74f163bfff1405bf406cf");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAEAAADurUJNAAAABGdBTUEAAYagMeiWXwAAAEFJREFUeJxjZGAkABQIyLMMBQWMDwgp+PcfP2B5MBwUMMoRkGdkonlcDAYFjI/wyv7/z/iH5nExGBQwyuCVZWQEAFDl/nE14thZAAAAAElFTkSuQmCC",
      "xc1n0g08.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAkAAAArGWqiAAAABGdBTUEAAYagMeiWXwAAAEhJREFUeJzt1cEJADAMAkCF7JH9t3ITO0Qr9KH4zuErtA0EO4AKFPgcoO3kfUx4QIECD0qHH8KEBxQo8KB0OCOpQIG7cHejwAGCsfleD0DPSwAAAABJRU5ErkJggg==",
      "xc9n2c08.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0NGg0AAAANSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAEhJREFUeJxjYGAQFFRSMjZ2cQkNTUsrL2cgQwCV29FBjgAqd+ZMcgRQuatWkSOAyt29mxwBVO6ZM+QIoHLv3iVHAJX77h0ZAgAfFO4B6v9B+gAAAABJRU5ErkJggg==",
      "xcrn0g04.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAQAAAABbAUdZAAAABGdBTUEAAYagMeiWXwAAAFtJREFUeJwtzLEJAzAMBdHr0gSySiALejRvkBU8gsGNCmFFB1Hx4IovqurSpIRszqklUwbnUzRXEuIRsiG/SyY9G0JzJSVei9qynm9qyjBpLp0pYW7pbzBl8L8fEIdJL0NTVU0AAAAASUVORK5CYII=",
      "xcsn0g01.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAAIAAADMaKZiAAAABGdBTUEAAYagMeiWXwAAAEhJREFUeJzt1cEJADAMAkCF7JH9t3ITO0Qr9KH4zuErtA0EO4AKFPgcoO3kfUx4QIECD0qHH8KEBxQo8KB0OCOpQIG7cHejwAGCsfleD0DPSwAAAABJRU5ErkJggg==",
      "xd0n2c08.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAwIAAACLyNyyAAAABGdBTUEAAYagMeiWXwAAAEhJREFUeJzt1cEJADAMAkCF7JH9t3ITO0Qr9KH4zuErtA0EO4AKFPgcoO3kfUx4QIECD0qHH8KEBxQo8KB0OCOpQIG7cHejwAGCsfleD0DPSwAAAABJRU5ErkJggg==",
      "xd3n2c08.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgYwIAAAAS+qv/AAAABGdBTUEAAYagMeiWXwAAAEhJREFUeJzt1cEJADAMAkCF7JH9t3ITO0Qr9KH4zuErtA0EO4AKFPgcoO3kfUx4QIECD0qHH8KEBxQo8KB0OCOpQIG7cHejwAGCsfleD0DPSwAAAABJRU5ErkJggg==",
      "xd9n2c08.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAQAAAABbAUdZAAAABGdBTUEAAYagMeiWXwAAAABJRU5ErkJggg==",
      "xdtn0g01.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABDU1VNAAAABGdBTUEAAYagMeiWXwAAAEFJREFUeJxjZGAkABQIyLMMBQWMDwgp+PcfP2B5MBwUMMoRkGdkonlcDAYFjI/wyv7/z/iH5nExGBQwyuCVZWQEAFDl/nE14thZAAAAAElFTkSuQmCC",
      "xhdn0g08.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORwoKGgoAAAAKSUhEUgAAACAAAAAgBAAAAACT4cgpAAAABGdBTUEAAYagMeiWXwAAAEhJREFUeJxjYGAQFFRSMjZ2cQkKTUsrL2cgQwCV29FBjgAqd+ZMcgRQuatWkSOAyt29mxwBVO6ZM+QIoHLv3iVHAJX77h0ZAgAfFO4B6v9B+gAAAABJRU5ErkJggg==",
      "xlfn0g04.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("CVBORw0KGgoAAAANSUhEUgAAACAAAAAgAQAAAABbAUdZAAAABGdBTUEAAYagMeiWXwAAAFtJREFUeJwtzLEJAzAMBdHr0gSySiALejRvkBU8gsGNCmFFB1Hx4IovqurSpIRszqklUwbnUzRXEuIRsiG/SyY9G0JzJSVei9qynm9qyjBpLp0pYW7pbzBl8L8fEIdJL9AvFMkAAAAASUVORK5CYII=",
      "xs1n0g01.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVFORw0KGgoAAAANSUhEUgAAACAAAAAgAQAAAABbAUdZAAAABGdBTUEAAYagMeiWXwAAAFtJREFUeJwtzLEJAzAMBdHr0gSySiALejRvkBU8gsGNCmFFB1Hx4IovqurSpIRszqklUwbnUzRXEuIRsiG/SyY9G0JzJSVei9qynm9qyjBpLp0pYW7pbzBl8L8fEIdJL9AvFMkAAAAASUVORK5CYII=",
      "xs2n0g01.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBOZw0KGgoAAAANSUhEUgAAACAAAAAgAQAAAABbAUdZAAAABGdBTUEAAYagMeiWXwAAAFtJREFUeJwtzLEJAzAMBdHr0gSySiALejRvkBU8gsGNCmFFB1Hx4IovqurSpIRszqklUwbnUzRXEuIRsiG/SyY9G0JzJSVei9qynm9qyjBpLp0pYW7pbzBl8L8fEIdJL9AvFMkAAAAASUVORK5CYII=",
      "xs4n0g01.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KIAoAAAANSUhEUgAAACAAAAAgAQAAAABbAUdZAAAABGdBTUEAAYagMeiWXwAAAFtJREFUeJwtzLEJAzAMBdHr0gSySiALejRvkBU8gsGNCmFFB1Hx4IovqurSpIRszqklUwbnUzRXEuIRsiG/SyY9G0JzJSVei9qynm9qyjBpLp0pYW7pbzBl8L8fEIdJL9AvFMkAAAAASUVORK5CYII=",
      "xs7n0g01.png", true, 32, 32, "d41d8cd98f00b204e9800998ecf8427e");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAMK0lEQVR42gEgDN/zAf//APgAAPgAAPcAAPgAAPgAAPgAAPcAAPgAAPgAAPgAAPgAAPcAAPgAAPgAAPgAAPcAAPgAAPgAAPgAAPcAAPgAAPgAAPgAAPgAAPcAAPgAAPgAAPgAAPcAAPgAAPgAAAQA+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgEAPgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIBAD3AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACAAACQQA+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAgAAAkAAAgEAPgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIAAAJAAAIAAAIBAD4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACAAACQAACAAACAAACAQA9wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAgAAAkAAAgAAAgAAAgAAAkEAPgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIBAD4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACAAACQAACAAACAAACAAACQAACAAACAQA+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgEAPgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIBAD3AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQQA+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgEAPgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIAAAJAAAIAAAIBAD4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQAACAAACAAACAQA9wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkEAPgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIBAD4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQAACAAACAAACAAACQAACAAACAQA+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgEAPcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJBAD4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACQAACAQA+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgEAPgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIBAD4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAQA9wAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkEAPgAAAAAAAAAAAAAAAAAAAAAAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIAAAJAAAIBAD4AAAAAAAAAAAAAAAAAAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQAACAAACAQA+AAAAAAAAAAAAAAAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgAAAgAAAgEAPcAAAAAAAAAAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAIAAAJAAAIAAAIAAAIAAAJBAD4AAAAAAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACQAACAAACAAACAAACAAACQAACAAACAAACAAACQAACAQA+AAAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAgAAAkAAAgAAAgAAAgAAAkAAAgAAAhVk05uHxPwlQAAAABJRU5ErkJggg==",
      "z00n2c08.png", false, 32, 32, "6284c288d49534c897da4e50a9d05002");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAr0lEQVR4XrXR3Q5AMAwF4Epc8P4Py91sIgxb15/TRUSC76Q9U0q0U7m28/5/Zl7Vv/Q+mwsZeJbQgIUoB+Q5Q07RidagCS79nADfwaNHBLx0eAdfHdtBQweuqK2jAro6JIDT/SUPdGfJY92zIpFuDpDqtg4UuqEDna5dkVpXBVh0eQdGXdiBXZesyKUPA7w6HwDQmZIxeq9kmN5cEVL/B4D1Twd4ve4gRL9XFKXngANVk05u39tDGQAAAABJRU5ErkJggg==",
      "z03n2c08.png", false, 32, 32, "6284c288d49534c897da4e50a9d05002");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAp0lEQVR4nLXRSw6AIBAD0JqwwPsfFna4MX4QYT4dVySS19BuraECFSg4D9158ktyLaEi8suhARnICSVQB/agF5x6UEW3HhHw0ukb9Dp3g4FOrGisswJ+dUrATPePvNCdI691T0Ui3Rwg1W0bKHTDBjpdW5FaVwVYdPkGRl24gV2XVOTSlwFefR5A0Ccjc/S/kWn6sCKm/g0g690GfP25QYh+VRSlA/kAVZNObjFSwSwAAAAASUVORK5CYII=",
      "z06n2c08.png", false, 32, 32, "6284c288d49534c897da4e50a9d05002");
  testPngSuiteImage("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAp0lEQVR42rXRSw6AIBAD0JqwwPsfFna4MX4QYT4dVySS19BuraECFSg4D9158ktyLaEi8suhARnICSVQB/agF5x6UEW3HhHw0ukb9Dp3g4FOrGisswJ+dUrATPePvNCdI691T0Ui3Rwg1W0bKHTDBjpdW5FaVwVYdPkGRl24gV2XVOTSlwFefR5A0Ccjc/S/kWn6sCKm/g0g690GfP25QYh+VRSlA/kAVZNObtYRvvUAAAAASUVORK5CYII=",
      "z09n2c08.png", false, 32, 32, "6284c288d49534c897da4e50a9d05002");
}


void testErrorImages() {
  std::cout << "testErrorImages" << std::endl;
  // Image with color type palette but missing PLTE chunk
  testBase64Image("iVBORw0KGgoAAAANSUhEUgAAAQAAAAEAAgMAAAAhHED1AAAAU0lEQVR4Ae3MwQAAAAxFoXnM3/NDvGsBdB8JBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEEQDHGPAW1eyhK0AAAAASUVORK5CYII=", true, 256, 256, "");
}

void doMain() {
  //PNG
  testPngSuite();
  testErrorImages();
  testPNGCodec();
  testPaletteFilterTypesZero();
  testComplexPNG();
  testInspectChunk();
  testPredefinedFilters();
  testFuzzing();
  testEncoderErrors();
  testPaletteToPaletteDecode();
  testPaletteToPaletteDecode2();
  testColorProfile();
  testExif();
  testBkgdChunk();
  testBkgdChunk2();
  testSbitChunk();

  //Colors
#ifndef DISABLE_SLOW
  testFewColors();
#endif // DISABLE_SLOW
  testColorKeyConvert();
  testColorConvert();
  testColorConvert2();
  testPaletteToPaletteConvert();
  testRGBToPaletteConvert();
  test16bitColorEndianness();
  testAutoColorModels();
  testNoAutoConvert();
  testChrmToSrgb();
  testXYZ();
  testICC();
  testICCGray();

  //Zlib
  testCompressZlib();
  testHuffmanCodeLengths();
  testCustomZlibCompress();
  testCustomZlibCompress2();
  testCustomDeflate();
  testCustomZlibDecompress();
  testCustomInflate();
  // TODO: add test for huffman code with exactly 0 and 1 symbols present

  //lodepng_util
  testChunkUtil();
  testGetFilterTypes();

  std::cout << "\ntest successful" << std::endl;
}

int main() {
  try {
    doMain();
  }
  catch(...) {
    std::cout << std::endl;
    std::cout << "caught error!" << std::endl;
    std::cout << "*** TEST FAILED ***" << std::endl;
  }

  return 0;
}
